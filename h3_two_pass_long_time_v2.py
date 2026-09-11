"""H3_TwoPassSamplerLongTimeV2 —— 时间分块版 H3 双阶段采样（实验性）。

在 H3_TwoPassSampler 的基础上，沿时间轴把长视频切成多段，逐段跑完整的
「阶段1 → latent 放大 → 阶段2 → AV 解码」流程，每段跑完立刻落盘并释放该段帧，
从而把峰值显存压到「单段」水平，而不是随总时长增长。

为什么能省显存
--------------
DiT 把整段视频 latent 当成一个序列做注意力，序列长度 n ∝ 帧数 × 分辨率，
注意力显存是 O(n²)。切成 C 段后每段长度 n/C，单段注意力显存变成 (n/C)²，
且各段串行执行，所以峰值显存 ≈ n²/C，不随总时长增长。

段间连续性
----------
每段生成时把上一段的**最后一帧**作为 first_frame 条件（锚定），并额外多生成
temporal_overlap_frames 帧、丢弃开头的重叠帧，避免相邻段跳变。

内存（RAM）控制
---------------
ComfyUI 的 IMAGE 是 fp32 [0,1]，1.2MP 单帧约 15MB，60 秒约 20GB —— 这才是
长视频 OOM 的第二个瓶颈。本节点把每段帧在累积时转成 uint8（省 4 倍），
并且 save_per_chunk=True 时逐段写盘后立即释放，内存中始终只保留锚定用的尾帧。

注意：帧只累积为 uint8 以降低内存；最终 frames 输出仍转回 fp32 [0,1]，
以兼容 SaveImage / VHS 等下游节点（它们要求 float [0,1]）。
"""

import gc
import glob
import logging
import math
import os

import torch

from comfy_api.latest import InputImpl, io, ui

try:
    from .t8_compat import ensure_t8_hybrid_probe_fix
except ImportError:      # 被当作顶层模块导入时（如离线测试脚本）
    from t8_compat import ensure_t8_hybrid_probe_fix

from .h3_two_pass import (
    ASPECT_RATIOS,
    DEFAULT_UPSCALER,
    AUDIO_MODES,
    DC_SAMPLER_DEFAULT,
    REF_AUDIO_MAX,
    REF_IMAGE_MAX,
    REF_VIDEO_MAX,
    SIZE_MULTIPLE,
    STAGE2_SIGMA_PRESETS,
    TASK_TYPES,
    _dc_sampling_options,
    _dims,
    _node_cls,
    _reserved_vram,
    _run,
    _sorted_autogrow,
    _upscaler_options,
)

MAX_CHUNKS = 200          # 死循环保护
FRAME_DTYPES = ["uint8", "fp32"]


# ------------------------- 帧格式转换 -------------------------

def _to_uint8(frames: torch.Tensor) -> torch.Tensor:
    """fp32 [0,1] [F,H,W,C] → uint8 [0,255]，并搬到 CPU（同时省显存与内存）。"""
    return (frames.detach().float().clamp(0.0, 1.0) * 255.0).round().to(
        dtype=torch.uint8, device="cpu", non_blocking=False
    )


def _to_float32(frames: torch.Tensor) -> torch.Tensor:
    """uint8 [0,255] → fp32 [0,1]（下游 SaveImage / VHS 要求 float [0,1]）。"""
    if frames.dtype == torch.uint8:
        return frames.to(torch.float32) / 255.0
    return frames.detach().float().clamp(0.0, 1.0)


def _last_frame_as_float(frames: torch.Tensor) -> torch.Tensor:
    """取一段帧的最后一帧，转成 conditioning 需要的 fp32 [1,H,W,C] [0,1]。"""
    last = _to_float32(frames[-1:])
    return last.contiguous()


def _concat_frames(parts, dtype: str) -> torch.Tensor:
    """把各段（已按 dtype 存储）拼成完整帧序列，转回 fp32 [0,1] 输出。"""
    if not parts:
        return None
    merged = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
    return _to_float32(merged)


# ------------------------- V2：分段提示词 -------------------------

def _split_prompts(text) -> list:
    """把 prompt 文本按**单独一行的 `---`** 切成若干段，依次对应第 1、2、3... 段视频。

    例：
        一个女人在雪地里行走
        ---
        她停下脚步，抬头看向天空
        ---
        镜头拉远，展现整片雪原

    不含 `---` 时只得到 1 段 → 全段通用（行为与普通 prompt 完全一致）。
    """
    if not text or not str(text).strip():
        return []
    import re
    parts = re.split(r"^\s*-{3,}\s*$", str(text), flags=re.MULTILINE)
    return [p.strip() for p in parts if p and p.strip()]


def _prompt_for_chunk(blocks, idx, fallback) -> str:
    """取第 idx 段（0 起）用的提示词。优先级：分段文本 > 基础 prompt。

    数量不足时沿用已提供的最后一条（例如 3 段但只写了 2 段提示词，
    第 3 段继续用第 2 段），只写 1 段则全段通用。
    """
    if blocks:
        return blocks[min(idx, len(blocks) - 1)]
    return fallback


# ------------------------- V2：latent 续接 -------------------------

def _seed_av_latent(av_latent, prev_tail, frames):
    """把上一段去噪后的视频 latent 尾部，写入新段 latent 的前若干时间位置。

    依据 ComfyUI FLOW 模型采样（comfy/model_sampling.py）：
        x起始 = sigma * noise + (1 - sigma) * latent_image
    因此只要把"上一段末尾的干净 latent"填进 latent_image，并让 sigma 从一个
    较低值起步，采样起点就落在 上一段内容 与 新噪声 之间的插值点上 ——
    运动轨迹、细节、纹理都被继承，而不是从纯随机噪声重新想象。
    """
    if not isinstance(av_latent, dict):
        return av_latent
    samples = av_latent.get("samples")
    if not isinstance(samples, torch.Tensor) or samples.dim() != 5:
        return av_latent
    if not isinstance(prev_tail, torch.Tensor) or prev_tail.dim() != 5:
        return av_latent

    _, _, T, _, _ = samples.shape
    k = int(min(int(frames), T, prev_tail.shape[2]))
    if k <= 0:
        return av_latent

    new = samples.clone()
    ch = min(new.shape[1], prev_tail.shape[1])
    new[:, :ch, :k] = prev_tail[:, :ch, -k:].to(device=new.device, dtype=new.dtype)
    out = dict(av_latent)
    out["samples"] = new
    return out


def _truncate_sigmas(sigmas, strength):
    """按续接强度把 sigma 调度从较低处开始（保留更多上一段内容）。

    strength=0 → 不截断（等同 V1 的纯噪声起点）；
    strength 越大 → 起始 sigma 越低 → 保留的上一段内容越多。
    至少保留 2 个 sigma，否则采样无法进行。
    """
    if sigmas is None or not hasattr(sigmas, "shape"):
        return sigmas
    n = sigmas.shape[0]
    if n < 3 or strength <= 0:
        return sigmas
    sigma_start = float(sigmas[0]) * (1.0 - float(strength))
    below = int((sigmas <= sigma_start).sum().item())
    k = min(below, n - 2)
    return sigmas[k:] if k > 0 else sigmas


# ------------------------- 落盘 -------------------------

def _save_chunk_png(frames: torch.Tensor, out_dir: str, slice_size: int = 8) -> int:
    """把一段帧写成 PNG 序列，返回写出的张数。

    按 slice_size 分片处理，两点考虑：
    1) 已是 uint8 时直接切片取 numpy，绝不回退成 fp32 —— 否则 136 帧 1.2MP
       会凭空产生约 2GB 的 float32 临时张量，正是保存阶段的内存尖峰来源；
    2) 峰值只保留 slice_size 帧，而不是整段的 uint8 副本。
    """
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    n = 0
    total = int(frames.shape[0])
    for start in range(0, total, slice_size):
        slc = frames[start:start + slice_size]
        arr = slc.numpy() if slc.dtype == torch.uint8 else _to_uint8(slc).numpy()
        for k in range(arr.shape[0]):
            Image.fromarray(arr[k]).save(os.path.join(out_dir, f"{n:05d}.png"))
            n += 1
        del slc, arr
    return n


def _combine_pngs_to_mp4(base_dir: str, fps: int = 24, crf: int = 18):
    """把各 chunk 目录里的 PNG 序列按序合成一个 H.264 mp4。

    用 ffconcat 清单把多目录的 PNG 串成一条时间线，全程只经 ffmpeg 流式编码，
    不需要把整段视频帧读进内存 —— 这是长视频结尾不撑 fp32 内存的正解。
    返回输出 mp4 路径，失败返回 None。
    """
    import glob as _glob
    import subprocess

    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        logging.warning("[HJL] 合成 mp4 需要 imageio-ffmpeg: %s", e)
        return None

    pngs = []
    for d in sorted(_glob.glob(os.path.join(base_dir, "chunk_*"))):
        pngs.extend(sorted(_glob.glob(os.path.join(d, "*.png"))))
    if not pngs:
        return None

    list_path = os.path.join(base_dir, "concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for p in pngs:
            f.write("file '%s'\n" % os.path.abspath(p).replace("\\", "/").replace("'", "'\\''"))
            f.write("duration %.6f\n" % (1.0 / fps))

    out_path = os.path.join(base_dir, "combined.mp4")
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
           "-r", str(fps), out_path]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except Exception as e:
        logging.warning("[HJL] ffmpeg 合成失败: %s", e)
        return None
    return out_path


def _output_subdir(prefix: str) -> str:
    import folder_paths

    base = folder_paths.get_output_directory()
    return os.path.join(base, "h3_chunks", prefix.replace("/", "_"))


# ------------------------- 单段双阶段流程 -------------------------

def _generate_one_chunk(*, model, clip, video_vae, audio_vae, prompt,
                        w1, h1, w2, h2, length, task_type, audio_mode,
                        stage2_sigma_steps, stage1_steps, split_step,
                        stage1_sampler, stage1_scheduler, seed,
                        drive_audio, final_audio, first_frame, last_frame,
                        ref_images, ref_videos, ref_video_audios, ref_audios,
                        ref_values, upscaler_model,
                        seed_video_tail=None, continuation_strength=0.0,
                        continuation_frames=8):
    """跑一次完整的双阶段采样 + AV 解码。

    返回 (final_latent, frames, audio, s1_video_tail)，其中 s1_video_tail 是
    本段阶段1 去噪后视频 latent 的尾部，供下一段做 latent 续接（V2）。
    """
    cond_out = _run(
        _node_cls("MiniMaxH3AudioConditioningT8"),
        clip=clip, video_vae=video_vae, audio_vae=audio_vae,
        prompt=prompt, width=w1, height=h1, length=length,
        task_type=task_type, audio_mode=audio_mode,
        audio_denoise_strength=0.35, add_source_as_reference=True,
        prompt_primary_audio_ordinal=1, strict_prompt_tags=True,
        ref_image_size="match", reference_video_policy="official_2_to_15s",
        drive_audio=drive_audio, final_audio=final_audio,
        first_frame=first_frame, last_frame=last_frame,
        ref_images=ref_images if ref_values else None,
        ref_videos=ref_videos or None,
        ref_video_audios=ref_video_audios or None,
        ref_audios=ref_audios or None,
        allow_above_reference_area=True,
    )
    positive, av_latent = cond_out[0], cond_out[1]

    dc_model, dc_sampler, dc_sigmas = _run(
        _node_cls("MiniMaxH3DualClockSamplerT8"),
        model=model, av_latent=av_latent,
        steps=stage1_steps, shift_video=12.0, shift_audio=3.0,
        sampler_name=stage1_sampler, scheduler=stage1_scheduler,
    )
    stage1_sigmas = _run(_node_cls("SplitSigmas"), sigmas=dc_sigmas, step=split_step)[0]
    noise = _run(_node_cls("RandomNoise"), noise_seed=seed)[0]
    guider1 = _run(_node_cls("BasicGuider"), model=dc_model, conditioning=positive)[0]

    # ---- V2：latent 续接 ----
    # 把上一段末尾的干净 latent 写进 latent_image，并让 sigma 从较低处起步，
    # 使采样起点 = σ·噪声 + (1-σ)·上一段内容（FLOW 模型公式）。
    used_sigmas, stage1_latent_in = stage1_sigmas, av_latent
    if seed_video_tail is not None and continuation_strength > 0:
        try:
            used_sigmas = _truncate_sigmas(stage1_sigmas, continuation_strength)
            stage1_latent_in = _seed_av_latent(av_latent, seed_video_tail,
                                               continuation_frames)
        except Exception as e:
            logging.warning("[HJL] latent 续接失败，回退为纯噪声起点: %s", e)
            used_sigmas, stage1_latent_in = stage1_sigmas, av_latent

    s1_out, s1_denoised = _run(
        _node_cls("SamplerCustomAdvanced"),
        noise=noise, guider=guider1, sampler=dc_sampler,
        sigmas=used_sigmas, latent_image=stage1_latent_in,
    )

    video_latent, audio_latent = _run(
        _node_cls("LTXVSeparateAVLatent"), av_latent=s1_denoised,
    )

    # V2：截取本段阶段1 视频 latent 的尾部，供下一段续接（提前取，之后会被 del）
    s1_video_tail = None
    try:
        vs = video_latent["samples"]
        if isinstance(vs, torch.Tensor) and vs.dim() == 5 and continuation_frames > 0:
            k = int(min(int(continuation_frames), vs.shape[2]))
            s1_video_tail = vs[:, :, -k:].detach().clone().cpu()
    except Exception as e:
        logging.warning("[HJL] 截取 latent 尾部失败（续接将降级）: %s", e)
    upscaled_video = _run(
        _node_cls("MinimaxH3LatentUpscaler3D"),
        latent=video_latent, model_name=upscaler_model,
        mode={"mode": "target dimensions", "width": w2, "height": h2},
        align=SIZE_MULTIPLE, keep_proportion=True, device="cuda", precision="fp16",
    )[0]
    stage2_latent_in = _run(
        _node_cls("LTXVConcatAVLatent"),
        video_latent=upscaled_video, audio_latent=audio_latent,
    )[0]

    del dc_model, dc_sampler, dc_sigmas, guider1, s1_out, s1_denoised
    del video_latent, audio_latent, upscaled_video

    edit_kwargs = {}
    for i, img in enumerate(ref_values[:REF_IMAGE_MAX], start=1):
        edit_kwargs[f"ref_image{i}"] = img

    cond2 = _node_cls("H3_EditConditioningWH").edit(
        positive, width=w2, height=h2, rescale_mode="nearest",
        vae=video_vae, first_frame=first_frame, last_frame=last_frame,
        **edit_kwargs,
    )[0]

    guider2 = _run(_node_cls("BasicGuider"), model=model, conditioning=cond2)[0]
    sampler2 = _run(_node_cls("KSamplerSelect"), sampler_name="euler")[0]
    sigmas2 = _run(_node_cls("ManualSigmas"),
                   sigmas=STAGE2_SIGMA_PRESETS[stage2_sigma_steps])[0]

    final_latent = _run(
        _node_cls("SamplerCustomAdvanced"),
        noise=noise, guider=guider2, sampler=sampler2,
        sigmas=sigmas2, latent_image=stage2_latent_in,
    )[0]

    frames, audio = None, None
    try:
        decoded = _run(
            _node_cls("MiniMaxH3AVDecodeT8"),
            av_latent=final_latent, video_vae=video_vae, audio_vae=audio_vae,
        )
        frames, audio = decoded[0], decoded[1]
    except Exception as e:
        logging.warning("[HJL] 分块第 %s 段解码失败: %s", length, e)

    return final_latent, frames, audio, s1_video_tail


# ------------------------- 主节点 -------------------------

class H3_TwoPassSamplerLongTimeV2(io.ComfyNode):
    """MiniMax H3 双阶段采样 + 时间分块（长视频，实验性）。

    与 H3_TwoPassSampler 参数一致，额外增加：
      temporal_chunk_frames   每段帧数（17 的倍数）
      temporal_overlap_frames 段间重叠帧数（17 的倍数，生成后丢弃开头这部分）
      save_per_chunk          每段解完立刻写 PNG 并释放该段帧
      frame_dtype             累积帧用 uint8（省 4 倍内存）还是 fp32
      merge_chunks            是否在最后把所有段拼成一个 frames 输出
                              （长视频建议关掉，否则末尾会占 fp32 全量内存）
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3_TwoPassSamplerLongTimeV2",
            display_name="H3 Two-Pass Long Time V2 (HJL)",
            category="sampling/H3",
            description=(
                "时间分块版双阶段采样：把长视频切成多段串行生成，段间用上一帧锚定 + "
                "重叠帧过渡，每段解完立刻写盘并释放。峰值显存不随总时长增长。"
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="UNETLoader 加载的 MiniMax H3 底模。"),
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default="",
                                tooltip="视频描述 prompt。支持**分段**：用单独一行的 --- 分隔，"
                                        "依次对应第 1/2/3... 段视频（例如 2 段渲染就写 2 段，中间一行 ---）。"
                                        "不含 --- 时全段通用。段数多于提示词时，多余的段沿用最后一条。"),
                io.Combo.Input("aspect_ratio", options=list(ASPECT_RATIOS.keys()),
                               default="16:9"),
                io.Int.Input("length", default=141, min=5, max=3600, step=1,
                             tooltip="**总**帧数（24fps）。会被切成多段生成。"),
                io.Int.Input("temporal_chunk_frames", default=136, min=17, max=3600, step=17,
                             tooltip="单段帧数**上限**，必须 17 的倍数。实际会按总帧数均分（如 294 帧 + 上限136 → 3 段各 98 帧），避免出现极短的尾巴段。建议设成你单次能稳定跑通的长度（6 秒≈136）。"),
                io.Int.Input("temporal_overlap_frames", default=0, min=0, max=272, step=1,
                             tooltip="**额外**丢弃的段首帧数（默认 0）。节点固定会丢弃与上一段尾帧重复的 1 帧；"
                                     "这里填 N 表示再多丢 N 帧。调大会造成时间跳跃、接缝处画面突变，"
                                     "仅当段首几帧质量明显差时才考虑（建议 ≤4）。"),
                io.Boolean.Input("latent_continuation", default=True,
                                 tooltip="【V2】latent 续接：把上一段末尾去噪后的 latent 写入下一段起点，"
                                         "让运动轨迹/细节/纹理得以继承，而不是每段从纯噪声重新想象。"
                                         "（依据 FLOW 采样公式 x = σ·噪声 + (1-σ)·latent）"),
                io.Float.Input("continuation_strength", default=0.5, min=0.0, max=0.95,
                               step=0.05,
                               tooltip="续接强度：越大保留上一段越多、变化越小。0 = 等同 V1（纯噪声起点）。"
                                       "建议 0.4~0.7；过高会让后续段几乎不动。"),
                io.Int.Input("continuation_frames", default=8, min=1, max=64, step=1,
                             tooltip="用上一段末尾多少个 latent 时间帧作为续接种子。"),
                io.Boolean.Input("same_seed_all_chunks", default=False,
                                 tooltip="所有段共用同一个 seed（V2 默认每段 seed+1）。共用可让纹理更一致。"),
                io.Boolean.Input("save_per_chunk", default=True,
                                 tooltip="每段解完立刻写 PNG 到 output/h3_chunks/<prefix>/chunk_XXXX/ 并释放该段帧。"),
                io.Boolean.Input("combine_saved_chunks", default=False,
                                 tooltip="跑完后用 ffmpeg 把所有段的 PNG 合成 output/h3_chunks/<prefix>/combined.mp4（H.264，流式编码不占内存）。长视频推荐开启——配合 merge_chunks=False，结尾不再有 fp32 内存尖峰。需 imageio-ffmpeg 包（已装）。"),
                io.Boolean.Input("merge_chunks", default=True,
                                 tooltip="把所有段拼成一个完整 frames 输出（默认开），可接任意图像/视频节点处理或合成。"
                                         "注意：关闭时 frames 输出**只有最后一段**，完整视频需靠落盘 PNG 或 combined.mp4；"
                                         "超长视频（内存吃紧）可关掉以省下末尾的 fp32 全量内存。"),
                io.Combo.Input("frame_dtype", options=FRAME_DTYPES, default="uint8",
                               tooltip="累积帧的存储格式。uint8 省 4 倍内存且对最终画质无可闻影响（保存视频本来就会量化到 8bit）。"),
                io.String.Input("chunk_filename_prefix", default="h3_chunk",
                                tooltip="落盘子目录名，位于 output/h3_chunks/ 下。"),
                io.Float.Input("stage1_megapixels", default=0.4, min=0.1, max=2.0, step=0.1),
                io.Float.Input("stage2_megapixels", default=1.2, min=0.1, max=8.0, step=0.1),
                io.Combo.Input("task_type", options=TASK_TYPES, default="FL2VA"),
                io.Combo.Input("audio_mode", options=AUDIO_MODES, default="native"),
                io.Combo.Input("stage2_sigma_steps",
                               options=list(STAGE2_SIGMA_PRESETS.keys()), default="3 steps"),
                io.Int.Input("stage1_steps", default=8, min=1, max=100),
                io.Int.Input("split_step", default=4, min=1, max=100),
                io.Combo.Input("stage1_sampler", options=_dc_sampling_options()[0],
                               default=DC_SAMPLER_DEFAULT),
                io.Combo.Input("stage1_scheduler", options=_dc_sampling_options()[1],
                               default="native_flow"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=True),
                io.Audio.Input("drive_audio", optional=True),
                io.Audio.Input("final_audio", optional=True),
                io.Image.Input("first_frame", optional=True,
                               tooltip="首段的首帧；后续段自动用上一段最后一帧锚定。"),
                io.Image.Input("last_frame", optional=True,
                               tooltip="仅最后一段使用（作为整段结尾）。"),
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"),
                        prefix="ref_image_", min=0, max=REF_IMAGE_MAX,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video"),
                        prefix="ref_video_", min=0, max=REF_VIDEO_MAX,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio"),
                        prefix="ref_video_audio_", min=0, max=REF_VIDEO_MAX,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"),
                        prefix="ref_audio_", min=0, max=REF_AUDIO_MAX,
                    ),
                ),
                io.Float.Input("reserved_vram", default=0.5, min=0.0, max=8.0, step=0.1,
                               tooltip="只在整段流程开头执行一次（切不可每段都跑，会导致中途卸载模型、速度劣化 3 倍）。"),
                io.Combo.Input("upscaler_model", options=_upscaler_options(),
                               default=DEFAULT_UPSCALER),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.Image.Output("frames"),
                io.Audio.Output("generated_audio"),
                io.Video.Output("video",
                                tooltip="合成后的完整视频（H.264 mp4）。仅当 combine_saved_chunks=True 且有落盘帧时才有值，可直接接视频预览/保存节点。"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, video_vae, audio_vae, prompt,
                aspect_ratio, length, temporal_chunk_frames, temporal_overlap_frames,
                latent_continuation, continuation_strength, continuation_frames,
                same_seed_all_chunks,
                save_per_chunk, combine_saved_chunks, merge_chunks, frame_dtype,
                chunk_filename_prefix,
                stage1_megapixels, stage2_megapixels,
                task_type, audio_mode, stage2_sigma_steps, stage1_steps, split_step,
                stage1_sampler, stage1_scheduler, seed,
                drive_audio=None, final_audio=None, first_frame=None, last_frame=None,
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None,
                reserved_vram=0.5, upscaler_model=DEFAULT_UPSCALER):

        # 第三方 T8 的 Hybrid 探测在当前 ComfyUI 上必然失败（探测数据用了非法
        # keyframe 索引），这里在内存中打兼容补丁，不改 T8 源文件（详见 t8_compat.py）。
        ensure_t8_hybrid_probe_fix()

        w1, h1 = _dims(aspect_ratio, stage1_megapixels)
        w2, h2 = _dims(aspect_ratio, stage2_megapixels)
        ref_values = _sorted_autogrow(ref_images)
        # prompt 里用单独一行的 --- 分段 → 依次对应第 1/2/3... 段
        prompt_blocks = _split_prompts(prompt)

        chunk = max(17, int(temporal_chunk_frames))
        overlap = max(0, int(temporal_overlap_frames))
        if chunk % 17:
            raise ValueError("temporal_chunk_frames 必须是 17 的倍数")

        num_chunks = max(1, math.ceil(length / chunk))
        # 均分，而不是「每段固定 chunk、零头全丢给最后一段」。
        # 后者会出现 136+136+16 这种尾巴：最后一段太短，且它是唯一拿到
        # 用户 last_frame 的一段，模型只能在不到 1 秒里强行拐到尾帧 → 内容断裂。
        per_chunk = max(1, math.ceil(length / num_chunks))

        # 防呆：save 和 merge 都关 = 除最后一段外全部静默丢失（用户会拿到
        # 一个只有 5 秒的"最后一段"却以为生成了完整视频）。宁可多占内存，
        # 也不能无声丢帧 —— 自动改为全量累积并在 info 里给出醒目提示。
        if not save_per_chunk and not merge_chunks:
            merge_chunks = True
            info_auto_merge = True
        else:
            info_auto_merge = False

        info = [f"total {length}f | 段数 {num_chunks} | 每段约 {per_chunk}f "
                f"(上限 {chunk}) | overlap {overlap}f | "
                f"stage1 {w1}x{h1} | stage2 {w2}x{h2} | dtype {frame_dtype}"]
        # V2 特性播报，方便确认参数是否生效
        if latent_continuation and num_chunks > 1 and continuation_strength > 0:
            info.append(f"latent 续接 ON (strength {continuation_strength}, "
                        f"{continuation_frames} 帧种子)")
        else:
            info.append("latent 续接 OFF（每段纯噪声起点）")
        if same_seed_all_chunks:
            info.append("全段共用 seed")
        if len(prompt_blocks) > 1:
            info.append(f"分段提示词：{len(prompt_blocks)} 段（prompt 内用 --- 分隔）")
        else:
            info.append("分段提示词：无（全段用 prompt）")
        if info_auto_merge:
            info.append("⚠️ save_per_chunk 与 merge_chunks 均为关闭——为避免静默丢帧，"
                        "已自动按 merge 输出完整视频；长视频请改开 save_per_chunk")

        # 多段时任务类型必须交给 T8 自动判定：
        # 中间段只有 first_frame（上一段尾帧锚定）、没有 last_frame，
        # 若沿用用户显式选的 FL2VA（要求首尾帧都接）会直接抛
        # "FL2VA last_frame connection does not match the selected task"；
        # 同理 REF2VA 不允许接 first_frame，也会失败。
        # auto 会按每段实际接线判定（first→i2va、first+last→fl2va、
        # refs+first→hybrid、refs→ref2va），永远合法。
        if num_chunks > 1:
            chunk_task = "auto"
            info.append(f"multi-chunk({num_chunks}): task forced to auto per chunk")
        else:
            chunk_task = task_type
            info.append(f"single-chunk: task {task_type}")

        # VRAM 预留只做一次：每段都跑会触发 unload_all_models，导致模型反复重载、
        # 新旧两代滞留显存，速度劣化 3 倍以上。
        _reserved_vram(reserved_vram, "auto")
        _reserved_vram(reserved_vram, "manual")

        base_dir = _output_subdir(chunk_filename_prefix) if save_per_chunk else None
        if base_dir:
            # 清掉上次运行残留的 chunk 目录 / 合成产物，否则旧 PNG 会被
            # 本次的 combined.mp4 误并入（例如上次 3 段、这次 2 段）。
            import shutil
            for old in glob.glob(os.path.join(base_dir, "chunk_*")):
                shutil.rmtree(old, ignore_errors=True)
            for stale in ("combined.mp4", "concat.txt"):
                try:
                    os.remove(os.path.join(base_dir, stale))
                except OSError:
                    pass
        parts = []                 # merge_chunks=True 时累积全部段
        last_out = None            # merge_chunks=False 时只保留最后一段
        audio_parts = []
        final_latent = None
        anchor = first_frame       # 首段用用户输入的首帧
        prev_latent_tail = None    # V2：上一段阶段1 的视频 latent 尾部
        produced = 0
        idx = 0

        while produced < length and idx < MAX_CHUNKS:
            want = min(per_chunk, length - produced)
            # 段首丢弃帧数：第 2 段起，生成序列的 f0 ≈ 上一段尾帧（锚定帧），
            # 与上一段最后一帧重复，必须丢掉这 1 帧，否则会出现静止重复帧。
            # 注意只丢 1 帧——f1 之后都是真实的新内容，多丢会在时间线上
            # 凭空跳一段（原先丢 overlap=17 帧 → 每次接缝跳 0.7 秒，画面断裂）。
            drop = (1 + overlap) if idx > 0 else 0
            gen_len = want + drop
            is_last = (produced + want) >= length

            # V2：本段用哪条提示词
            chunk_prompt = _prompt_for_chunk(prompt_blocks, idx, prompt)
            # V2：本段 seed（可全段共用）
            chunk_seed = seed if (same_seed_all_chunks or idx == 0) else seed + idx
            # V2：latent 续接（首段没有上一段，恒为 None）
            seed_tail = prev_latent_tail if (latent_continuation and idx > 0) else None

            lat, frames, audio, s1_tail = _generate_one_chunk(
                model=model, clip=clip, video_vae=video_vae, audio_vae=audio_vae,
                prompt=chunk_prompt, w1=w1, h1=h1, w2=w2, h2=h2, length=gen_len,
                task_type=chunk_task, audio_mode=audio_mode,
                stage2_sigma_steps=stage2_sigma_steps, stage1_steps=stage1_steps,
                split_step=split_step, stage1_sampler=stage1_sampler,
                stage1_scheduler=stage1_scheduler,
                seed=chunk_seed,
                seed_video_tail=seed_tail,
                continuation_strength=continuation_strength,
                continuation_frames=continuation_frames,
                drive_audio=drive_audio, final_audio=final_audio,
                first_frame=anchor,
                last_frame=last_frame if is_last else None,
                ref_images=ref_images, ref_videos=ref_videos,
                ref_video_audios=ref_video_audios, ref_audios=ref_audios,
                ref_values=ref_values, upscaler_model=upscaler_model,
            )
            final_latent = lat
            prev_latent_tail = s1_tail   # V2：交给下一段做 latent 续接

            if frames is None:
                info.append(f"chunk {idx}: decode failed, stop")
                break

            # 丢弃开头重叠帧
            keep = frames[drop:] if (drop and frames.shape[0] > drop) else frames
            if keep.shape[0] > want:
                keep = keep[:want]

            stored = _to_uint8(keep) if frame_dtype == "uint8" else \
                keep.detach().float().clamp(0, 1).cpu()

            if save_per_chunk:
                n = _save_chunk_png(stored, os.path.join(base_dir, f"chunk_{idx:04d}"))
                info.append(f"chunk {idx}: {n}f saved")

            # 下一段用本段最后一帧锚定（只需要一帧，转成 fp32 供 conditioning 使用）
            anchor = _last_frame_as_float(stored)

            if merge_chunks:
                parts.append(stored)   # 累积全部，最后拼成一整段
            else:
                last_out = stored      # 只保留最后一段，前面的段自动被回收
            if audio is not None:
                audio_parts.append(audio)

            produced += int(keep.shape[0])
            idx += 1

            # 释放本段的大张量并回收显存缓存（不会卸载已加载的模型）。
            # gc.collect() 尽早回收 CPU 端大帧，给下一段的模型 staging 腾 RAM
            # （32GB 内存机器上，staging 一份 20GB 模型已经贴着上限）。
            del frames, keep, stored
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        info.append(f"chunks {idx} | produced {produced}f")

        # ---------- 输出 frames ----------
        if merge_chunks:
            frames_out = _concat_frames(parts, frame_dtype)
            info.append("merged (fp32 output)")
        else:
            frames_out = _concat_frames([last_out] if last_out is not None else [],
                                        frame_dtype)
            info.append(f"merge off: frames 输出=最后一段，完整 {produced} 帧在磁盘")

        video_out, preview_ui = None, None
        if save_per_chunk:
            info.append(f"saved to {base_dir}")
            if combine_saved_chunks:
                mp4 = _combine_pngs_to_mp4(base_dir)
                if mp4:
                    info.append(f"combined mp4: {mp4}")
                    # 输出成 VIDEO 端口，可直接接视频预览/保存节点
                    try:
                        video_out = InputImpl.VideoFromFile(str(mp4))
                        import folder_paths as _fp
                        sub = os.path.relpath(os.path.dirname(mp4),
                                              _fp.get_output_directory()).replace("\\", "/")
                        preview_ui = ui.PreviewVideo([{
                            "filename": os.path.basename(mp4),
                            "subfolder": "" if sub == "." else sub,
                            "type": "output",
                        }])
                    except Exception as e:
                        logging.warning("[HJL] 生成 VIDEO 输出失败（mp4 仍在磁盘）: %s", e)
                else:
                    info.append("combine failed (见控制台)")

        # ---------- 拼接音频 ----------
        audio_out = None
        if audio_parts:
            try:
                ws = [a["waveform"].cpu() for a in audio_parts]
                audio_out = {
                    "waveform": torch.cat(ws, dim=-1) if len(ws) > 1 else ws[0],
                    "sample_rate": audio_parts[0].get("sample_rate", 24000),
                }
                info.append(f"audio {len(audio_parts)} segs")
            except Exception as e:
                logging.warning("[HJL] 音频拼接失败: %s", e)

        return io.NodeOutput(final_latent, frames_out, audio_out, video_out,
                             " | ".join(info), ui=preview_ui)


NODE_CLASS_MAPPINGS = {
    "H3_TwoPassSamplerLongTimeV2": H3_TwoPassSamplerLongTimeV2,
}
