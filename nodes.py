import logging

import torch
import torch.nn.functional as F

import comfy.utils

from comfy_api.latest import io

# MiniMax H3 的 VAE 空间下采样倍率：latent_h = height // 16, latent_w = width // 16
H3_PIXELS_PER_LATENT = 16

# ref_image autogrow 端口数量上限
MAX_REF_IMAGES = 8


def _decode_to_frames(vae, z):
    """VAE decode -> frames [F, H, W, C]，值域 [0, 1]。兼容不同 VAE 的输出布局。"""
    out = vae.decode(z)
    if out.ndim == 5:
        if out.shape[1] == 3:          # [B, C, T, H, W]，MiniMax H3 底层 NCTHW 格式
            frames = out[0].permute(1, 2, 3, 0)
        else:                          # [B, T, H, W, C]
            frames = out[0]
    elif out.ndim == 4:
        frames = out.movedim(1, -1) if out.shape[1] == 3 else out
    else:
        raise ValueError(f"vae.decode returned unrecognized shape: {tuple(out.shape)}")
    return frames


def _rescale_h3_latent(z, lat_h, lat_w, mode, vae=None, target_w=None, target_h=None):
    """把 [1, C, T, H, W] 的 H3 latent 调整到 (T, lat_h, lat_w)，时间维尽量保持不变。

    vae 已接：decode -> resize 到新像素尺寸 -> encode（从 latent 重建，有损）。
    vae 未接：直接在 latent 空间插值（快）。
    尺寸已匹配时原样返回（幂等）。
    """
    t = z.shape[2]
    if tuple(z.shape[-2:]) == (lat_h, lat_w):
        return z

    if vae is not None and target_w and target_h:
        frames = _decode_to_frames(vae, z)                 # [F, H, W, C] in [0, 1]
        x = frames.movedim(-1, 1)                          # [F, C, H, W]
        x = comfy.utils.common_upscale(x, target_w, target_h, "bicubic", "disabled")
        new_z = vae.encode(x.movedim(1, -1).contiguous())  # -> [1, 24, T', h', w']
        if new_z.shape[2] == t and tuple(new_z.shape[-2:]) == (lat_h, lat_w):
            return new_z
        logging.warning(
            "[HJL] VAE re-encode produced unexpected shape %s (T=%s), expected T=%s (%s, %s); "
            "falling back to latent-space interpolation",
            tuple(new_z.shape), new_z.shape[2], t, lat_h, lat_w,
        )

    zf = z.float()
    if mode == "trilinear":
        out = F.interpolate(zf, size=(t, lat_h, lat_w), mode="trilinear", align_corners=False)
    else:
        out = F.interpolate(zf, size=(t, lat_h, lat_w), mode=mode)
    return out.to(z.dtype)


def _encode_source_image(vae, image, enc_w, enc_h):
    """把原始参考图 [B, H, W, C]（值域 [0,1]）按官方方式 resize + VAE 编码。

    与 T8 官方 resize_image 一致：lanczos、不裁剪。编码尺寸必须是 16 的倍数，
    这样输出 latent 精确等于 (T, enc_h//16, enc_w//16)。
    """
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE [B,H,W,C], got {tuple(image.shape)}")
    x = image[..., :3].movedim(-1, 1)                          # [B, C, H, W]
    x = comfy.utils.common_upscale(x, enc_w, enc_h, "lanczos", "disabled")
    z = vae.encode(x.movedim(1, -1).contiguous())              # [1, 24, T, h, w]
    if z.ndim != 5 or tuple(z.shape[-2:]) != (enc_h // 16, enc_w // 16):
        raise ValueError(
            f"[HJL] VAE encode of source ref image returned unexpected shape {tuple(z.shape)}; "
            f"expected spatial ({enc_h // 16}, {enc_w // 16})"
        )
    return z


def _collect_image_batches(*images):
    """把若干 IMAGE 输入（[N,H,W,C] 或 None）按顺序拆成 [1,H,W,C] 列表。"""
    src = []
    for img in images:
        if img is not None and img.ndim == 4 and img.shape[0] > 0:
            src.extend(img[i:i + 1] for i in range(img.shape[0]))
    return src


def _sorted_autogrow_values(values) -> list:
    """AUTOGROW dict（{'ref_image_0': t, ...}）-> 按尾缀序号排列的非空值列表。"""
    if not values:
        return []

    def sort_key(item):
        try:
            return int(str(item[0]).rsplit("_", 1)[-1])
        except ValueError:
            return 10_000

    return [v for _, v in sorted(dict(values).items(), key=sort_key) if v is not None]


class _EditConditioningWHBase(io.ComfyNode):
    """改 conditioning 的 width / height（针对 MiniMax H3 两阶段放大）。

    H3 的 DiT 不读 conditioning meta 的 width/height，它读 payload 里
    `cond_video_latents = [r["latent"] for r in minimax_refs]`，layout 行数按 ref
    声明的 latent_h / latent_w 计算。latent 放大后如果 ref 张量还是旧尺寸，
    就会报 shape mismatch（810 vs 2444 那个错）。

    本节点做四件事：
    1. 更新 meta 的 width / height（及 target_width / target_height）
    2. 把 minimax_keyframes / minimax_refs 里未重编码的参考 latent 调整到新尺寸
       （插值，或接 vae 时的 decode->encode）
    3. 【最优画质】直接接原始参考图（ref_image autogrow / first_frame / last_frame），
       用 VAE 从原图按新尺寸重新编码并注入 conditioning——绕开 latent 重建损失，
       与官方"高分辨率重新 Conditioning"同质量
    4. 同步 ref 的 latent_h / latent_w / latent_t，保证 layout 与张量一致

    ref_image 为 COMFY_AUTOGROW_V3 端口：接一张自动出现下一个接入点，最多 8 张。

    画质排序：原始图重编码（本节点接图） > latent decode->encode > latent 插值。
    """

    NODE_ID = "H3_EditConditioningWH"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.NODE_ID,
            display_name="H3 Edit Conditioning W/H (HJL)",
            category="conditioning/H3",
            description=(
                "H3 两阶段放大专用：更新 conditioning 宽高，并把参考 latent 调整到新尺寸。"
                "接 vae + 原始首尾帧/参考图时按新分辨率重新 VAE 编码（画质最优）；"
                "否则走 latent 插值。"
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Int.Input("width", default=960, min=64, max=8192, step=8,
                             tooltip="目标像素宽。"),
                io.Int.Input("height", default=544, min=64, max=8192, step=8,
                             tooltip="目标像素高。"),
                io.Combo.Input("rescale_mode",
                               options=["trilinear", "nearest", "area"],
                               default="trilinear",
                               tooltip="未重编码 latent 的插值方式（仅没接 vae 的 ref/keyframe 用到）。"),
                io.Vae.Input("vae", optional=True,
                             tooltip="MiniMax H3 video VAE（VAELoader 输出）。接了 ref_image/first_frame/last_frame 时必接；也可单独用于 latent 的 decode->encode（有损，画质不如直接接原图）。"),
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip="原始参考图，每接一张自动出现下一个接入点（最多 8 张）。按顺序替换 kind=image 的 ref，从原图直接 VAE 编码，画质最好。"),
                        prefix="ref_image_", min=0, max=MAX_REF_IMAGES,
                    ),
                    tooltip="可选。多张原始参考图（自动增减接入点），VAE 按新分辨率重编码。",
                ),
                io.Image.Input("first_frame", optional=True,
                               tooltip="可选。原始首帧图。替换 conditioning 中 resolved_frame_index=0 的 keyframe。"),
                io.Image.Input("last_frame", optional=True,
                               tooltip="可选。原始尾帧图。替换 conditioning 中最后一帧的 keyframe（需要 conditioning 元数据带 minimax_frame_count）。"),
            ],
            outputs=[io.Conditioning.Output()],
        )

    # ---------------- 内部核心 ----------------

    @classmethod
    def _edit_impl(cls, positive, width, height, rescale_mode,
                   vae, ref_slots, legacy_batch, first_frame, last_frame):
        lat_h = max(1, height // H3_PIXELS_PER_LATENT)
        lat_w = max(1, width // H3_PIXELS_PER_LATENT)
        enc_w, enc_h = lat_w * 16, lat_h * 16

        # autogrow / 动态端口的图在前，旧版整批 batch 追加在后
        src_refs = _collect_image_batches(*ref_slots)
        src_refs.extend(_collect_image_batches(legacy_batch))

        want_encode = bool(src_refs or first_frame is not None or last_frame is not None)
        if want_encode and vae is None:
            raise ValueError(
                "[HJL] 接了 ref_image / first_frame / last_frame 但没接 vae；"
                "无法从原始图重新编码。请把 MiniMax H3 video VAE 接到 vae 输入。"
            )

        out = []
        for tensor, meta in positive:
            new_meta = dict(meta)            # 拷贝一份, 别污染上游
            new_meta["width"] = enc_w
            new_meta["height"] = enc_h
            # SDXL 还可能带 target_width / target_height, 一并改了
            for k in ("target_width", "target_height"):
                if k in new_meta:
                    new_meta[k] = enc_w if "width" in k else enc_h

            # ---- 1) 用原始参考图重新编码，替换 kind=image 的 refs（按顺序），多余追加 ----
            if src_refs:
                blocks = list(new_meta.get("minimax_refs") or [])
                img_positions = [i for i, b in enumerate(blocks)
                                 if isinstance(b, dict) and b.get("kind") == "image"]
                n_replace = min(len(src_refs), len(img_positions))
                for j in range(n_replace):
                    blk = dict(blocks[img_positions[j]])
                    blk["latent"] = _encode_source_image(vae, src_refs[j], enc_w, enc_h)
                    blk["latent_h"] = lat_h
                    blk["latent_w"] = lat_w
                    blk["latent_t"] = int(blk["latent"].shape[2])
                    blocks[img_positions[j]] = blk
                for extra in src_refs[n_replace:]:
                    blocks.append({
                        "kind": "image",
                        "latent": _encode_source_image(vae, extra, enc_w, enc_h),
                        "latent_h": lat_h,
                        "latent_w": lat_w,
                        "latent_t": 1,
                    })
                if n_replace < len(img_positions):
                    logging.warning(
                        "[HJL] source ref images(%d) fewer than image refs(%d); "
                        "remaining %d refs keep the latent-rescale path",
                        len(src_refs), len(img_positions), len(img_positions) - n_replace,
                    )
                new_meta["minimax_refs"] = blocks

            # ---- 2) first/last_frame 重编码，替换/注入 keyframes ----
            kfs = list(new_meta.get("minimax_keyframes") or [])
            frame_count = new_meta.get("minimax_frame_count")

            def _apply_anchor(img, want_index, label):
                hit = False
                for i, kf in enumerate(kfs):
                    if isinstance(kf, dict) and kf.get("resolved_frame_index") == want_index:
                        kf = dict(kf)
                        kf["latent"] = _encode_source_image(vae, img, enc_w, enc_h)
                        kfs[i] = kf
                        hit = True
                        break
                if not hit:
                    if want_index == 0:
                        kfs.insert(0, {"resolved_frame_index": 0,
                                       "latent": _encode_source_image(vae, img, enc_w, enc_h)})
                        hit = True
                    elif frame_count is not None:
                        kfs.append({"resolved_frame_index": frame_count - 1,
                                    "latent": _encode_source_image(vae, img, enc_w, enc_h)})
                        hit = True
                    else:
                        logging.warning(
                            "[HJL] %s: no matching keyframe anchor (index=%s) and no "
                            "minimax_frame_count in conditioning metadata; skipped",
                            label, want_index,
                        )
                return hit

            if first_frame is not None:
                _apply_anchor(first_frame[:1], 0, "first_frame")
            if last_frame is not None:
                idx = (frame_count - 1) if frame_count else None
                if idx is None:
                    logging.warning("[HJL] last_frame: no minimax_frame_count in metadata; skipped")
                else:
                    _apply_anchor(last_frame[:1], idx, "last_frame")
            if kfs:
                kfs.sort(key=lambda k: k.get("resolved_frame_index", 0))
                new_meta["minimax_keyframes"] = kfs

            # ---- 3) 其余未被原图替换的 ref/keyframe，走 latent 调整（幂等） ----
            for key in ("minimax_keyframes", "minimax_refs"):
                blocks = new_meta.get(key)
                if not blocks:
                    continue
                new_blocks = []
                for blk in blocks:
                    blk = dict(blk)
                    z = blk.get("latent")
                    if isinstance(z, torch.Tensor) and z.ndim == 5:
                        blk["latent"] = _rescale_h3_latent(
                            z, lat_h, lat_w, rescale_mode,
                            vae=vae, target_w=enc_w, target_h=enc_h,
                        )
                        blk["latent_h"] = lat_h
                        blk["latent_w"] = lat_w
                        if "latent_t" in blk:
                            blk["latent_t"] = int(blk["latent"].shape[2])
                    new_blocks.append(blk)
                new_meta[key] = new_blocks

            out.append((tensor, new_meta))
        return out

    # ---------------- v3 执行入口 ----------------

    @classmethod
    def execute(cls, positive, width, height, rescale_mode,
                vae=None, ref_images=None, first_frame=None, last_frame=None):
        ref_slots = _sorted_autogrow_values(ref_images)
        result = cls._edit_impl(positive, width, height, rescale_mode,
                                vae=vae, ref_slots=ref_slots, legacy_batch=None,
                                first_frame=first_frame, last_frame=last_frame)
        return io.NodeOutput(result)

    # ---------------- 兼容入口（供 H3_TwoPassSampler 内部调用） ----------------

    @classmethod
    def edit(cls, positive, width, height, rescale_mode="trilinear",
             vae=None, ref_images=None, first_frame=None, last_frame=None, **legacy):
        """旧调用约定：ref_image1..8 关键字参数 + 整批 ref_images tensor。

        仍被 h3_two_pass.py 以 H3_EditConditioningWH.edit(...) 形式调用。
        """
        def _ord(name):
            try:
                return int(name.rsplit("ref_image", 1)[-1])
            except ValueError:
                return 10_000

        ref_slots = [v for _, v in sorted(
            ((k, v) for k, v in legacy.items()
             if k.startswith("ref_image") and v is not None),
            key=lambda kv: _ord(kv[0]),
        )]
        return (cls._edit_impl(positive, width, height, rescale_mode,
                               vae=vae, ref_slots=ref_slots, legacy_batch=ref_images,
                               first_frame=first_frame, last_frame=last_frame),)


class EditConditioningWH(_EditConditioningWHBase):
    """H3_EditConditioningWH 注册类。"""


class EditConditioningWHLegacy(_EditConditioningWHBase):
    """旧注册名 EditConditioningWH 的别名类（node_id 需要独立，已保存工作流不断线）。"""

    NODE_ID = "EditConditioningWH"


NODE_CLASS_MAPPINGS = {
    # 新名字
    "H3_EditConditioningWH": EditConditioningWH,
    # 旧名字保留为兼容别名，已保存的工作流不会断线
    "EditConditioningWH": EditConditioningWHLegacy,
}
