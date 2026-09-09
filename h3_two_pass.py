"""H3_TwoPassSampler —— 把 MiniMax H3 双阶段采样工作流打包成单节点。

内部完整复刻 0907 H3_OpenVDN_DMD8_FL2VA 工作流（v3 io 节点，ref_images 为 AUTOGROW）：

  阶段1: AudioConditioningT8(w1,h1) → VDNComposer → DualClockSampler(stage1_steps)
         → SplitSigmas(split_step) → SamplerCustomAdvanced
  桥接:  SeparateAVLatent(denoised) → Upscaler3D(w2,h2) + H3_EditConditioningWH(原图重编码)
         → ConcatAVLatent
  阶段2: BasicGuider(同 VDN model) → euler → ManualSigmas(3/4/5 步预设) → SamplerCustomAdvanced

对外暴露少量端口，输出最终 av latent + 已解码的图像帧/音频
（内部调用 T8 的 MiniMaxH3AVDecodeT8）。
所有被复用的节点类在执行期从 ComfyUI 的全局 NODE_CLASS_MAPPINGS 取，避免硬编码路径。
"""

import logging
import math
import sys

import torch

from comfy_api.latest import io

# ------------------------- 常量（与工作流一致） -------------------------

ASPECT_RATIOS = {
    "16:9": (16, 9),
    "9:16": (9, 16),
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "21:9": (21, 9),
}

# 第二阶段 DMD 蒸馏 sigmas 预设（来自工作流 ManualSigmas 节点 34/31/27）
STAGE2_SIGMA_PRESETS = {
    "3 steps": "0.9035, 0.6316, 0.3158, 0.0000",
    "4 steps": "0.9035, 0.8000, 0.6316, 0.3158, 0.0000",
    "5 steps": "0.9231, 0.8780, 0.8000, 0.6316, 0.3158, 0.0000",
}

SIZE_MULTIPLE = 32                     # H3 尺寸对齐（与 ResolutionSelector 的 multiple=32 一致）
DEFAULT_UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"

DC_SAMPLER_DEFAULT = "dual_clock_euler"
DC_SCHEDULER_DEFAULT = "native_flow"


def _dc_sampling_options():
    """从 T8 的 sampling 模块读阶段1 sampler/scheduler 的真实可选列表（含回退）。"""
    samplers, schedulers = [DC_SAMPLER_DEFAULT], [DC_SCHEDULER_DEFAULT, "beta57"]
    try:
        cls = _node_cls("MiniMaxH3DualClockSamplerT8")
        mod = sys.modules.get(cls.__module__)
        s = getattr(mod, "SAMPLER_OPTIONS", None)
        d = getattr(mod, "SCHEDULER_OPTIONS", None)
        if s:
            samplers = list(s)
        if d:
            schedulers = list(d)
    except Exception:
        pass
    return samplers, schedulers

REF_IMAGE_MAX = 8                      # 与 H3_EditConditioningWH 的 ref_image1..8 对齐
REF_VIDEO_MAX = 3                      # T8 官方限制：参考视频最多 3 个
REF_AUDIO_MAX = 3                      # T8 官方限制：独立参考音频最多 3 条

TASK_TYPES = ["auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid"]
AUDIO_MODES = ["lock_source", "remix_source", "reference_only", "native"]


# ------------------------- 内部工具 -------------------------

def _node_cls(name):
    """从 ComfyUI 全局注册表取节点类（延迟 import，避免模块加载顺序问题）。"""
    import nodes as comfy_nodes
    try:
        return comfy_nodes.NODE_CLASS_MAPPINGS[name]
    except KeyError:
        raise RuntimeError(
            f"[HJL] 找不到节点类 '{name}'，请确认对应的自定义节点包已安装并加载成功。"
        )


def _run(node_cls, **kwargs):
    """调用 io.ComfyNode.execute 类方法，返回输出 tuple。"""
    out = node_cls.execute(**kwargs)
    result = getattr(out, "result", None)
    if result is None:
        result = out if isinstance(out, tuple) else (out,)
    return result


def _sorted_autogrow(values) -> list:
    """AUTOGROW dict（如 {'ref_image_0': t, 'ref_image_2': t}）→ 按序号排列的非空值列表。"""
    if not values:
        return []

    def sort_key(item):
        try:
            return int(str(item[0]).rsplit("_", 1)[-1])
        except ValueError:
            return 10_000

    return [v for _, v in sorted(dict(values).items(), key=sort_key) if v is not None]


def _dims(aspect_ratio, megapixels):
    """与核心 ResolutionSelector 相同的算法（multiple=32）。
    16:9 @0.4MP -> 864x480；16:9 @0.5MP -> 960x544；16:9 @1.2MP -> 1504x832。"""
    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    total = megapixels * 1024 * 1024
    scale = math.sqrt(total / (w_ratio * h_ratio))
    w = max(SIZE_MULTIPLE, round(w_ratio * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE)
    h = max(SIZE_MULTIPLE, round(h_ratio * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE)
    return w, h


def _reserved_vram(reserved_gb, mode):
    """复刻工作流里两个 ReservedVRAMSetter 的副作用（前置清显存 + 设置预留）。"""
    try:
        inst = _node_cls("ReservedVRAMSetter")()
        inst.set_vram(reserved=reserved_gb, mode=mode, seed=0,
                      auto_max_reserved=0.0, clean_gpu_before=True, anything=None)
    except Exception as e:  # 清理失败不应中断采样
        logging.warning("[HJL] ReservedVRAM 调用失败（忽略继续）: %s", e)


def _upscaler_options():
    try:
        cls = _node_cls("MinimaxH3LatentUpscaler3D")
        mod = sys.modules.get(cls.__module__)
        opts = list(mod.scan_models())
    except Exception:
        opts = []
    if DEFAULT_UPSCALER not in opts:
        opts.insert(0, DEFAULT_UPSCALER)
    return opts


# ------------------------- 主节点 -------------------------

class H3_TwoPassSampler(io.ComfyNode):
    """MiniMax H3 双阶段一键采样（打包自 0907 OpenVDN DMD8 FL2VA 工作流）。

    阶段1 低分辨率（stage1_megapixels，默认 0.4MP → 16:9 为 864x480），
    DMD 双时钟采样 stage1_steps 步、前 split_step 步用原生 sigmas；
    桥接用官方 3D latent upscaler + 原始首尾帧/参考图 VAE 重编码；
    阶段2 高分辨率（stage2_megapixels，默认 1.2MP → 16:9 为 1504x832），
    按选定步数的蒸馏 sigmas 用 euler 收尾。输出 av latent（音视频合一，可继续外接
    MiniMaxH3AVDecodeT8）+ 已解码的图像帧与音频，省去外接解码节点。

    ref_images / ref_videos / ref_video_audios / ref_audios 均为 AUTOGROW 端口：
    接一张自动出现下一个接入点（图 8 / 视频 3 / 音频 3）。
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3_TwoPassSampler",
            display_name="H3 Two-Pass Sampler (HJL)",
            category="sampling/H3",
            description=(
                "一键复刻 0907 H3 OpenVDN DMD8 双阶段工作流：低分辨率 DMD 双时钟采样 → "
                "3D latent 放大 + 原图 VAE 重编码条件 → 高分辨率蒸馏 sigmas 收尾 → "
                "内置 AV 解码。输出 av latent + 图像帧 + 音频。"
            ),
            inputs=[
                io.Model.Input("model", tooltip="UNETLoader 加载的 MiniMax H3 底模。"),
                io.Clip.Input("clip", tooltip="MiniMax H3 Qwen3-VL CLIP。"),
                io.Vae.Input("video_vae", tooltip="MiniMax H3 video VAE（fp16）。"),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 audio VAE（fp32）。"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default="",
                                tooltip="视频描述 prompt。"),
                io.Combo.Input("aspect_ratio", options=list(ASPECT_RATIOS.keys()),
                               default="16:9"),
                io.Int.Input("length", default=39, min=5, max=3600, step=1,
                             tooltip="视频帧数（24fps）。会自动向上对齐到 17n+5 的 H3 网格，如 39、56、73…"),
                io.Float.Input("stage1_megapixels", default=0.4, min=0.1, max=2.0, step=0.1,
                               tooltip="第一阶段目标像素（MP）。16:9 时 0.4 → 864x480。"),
                io.Float.Input("stage2_megapixels", default=1.2, min=0.1, max=8.0, step=0.1,
                               tooltip="第二阶段目标像素（MP）。16:9 时 1.2 → 1504x832。"),
                io.Combo.Input("task_type", options=TASK_TYPES, default="FL2VA",
                               tooltip="FL2VA = 首尾帧生音视频；auto 交给 T8 自判。"),
                io.Combo.Input("audio_mode", options=AUDIO_MODES, default="native"),
                io.Combo.Input("stage2_sigma_steps", options=list(STAGE2_SIGMA_PRESETS.keys()),
                               default="3 steps",
                               tooltip="第二阶段蒸馏 sigmas 预设（来自工作流 ManualSigmas）。"),
                io.Int.Input("stage1_steps", default=8, min=1, max=100,
                             tooltip="第一阶段双时钟采样总步数。"),
                io.Int.Input("split_step", default=4, min=1, max=100,
                             tooltip="第一阶段 sigmas 分离步数（前 N 步用原生调度）。"),
                io.Combo.Input("stage1_sampler", options=_dc_sampling_options()[0],
                               default=DC_SAMPLER_DEFAULT,
                               tooltip="阶段1 采样器。dual_clock_euler 为 T8 原生双时钟路径；其他选项走 ComfyUI 原生 FLOW_AV 协议。"),
                io.Combo.Input("stage1_scheduler", options=_dc_sampling_options()[1],
                               default=DC_SCHEDULER_DEFAULT,
                               tooltip="阶段1 调度器。native_flow 为 T8 原生 H3 flow 调度；beta57 用 ComfyUI beta 调度（alpha=0.5, beta=0.7）。"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=True),
                io.Audio.Input("drive_audio", optional=True,
                               tooltip="可选。驱动音频（源音频 latent 锁定/参考）。"),
                io.Audio.Input("final_audio", optional=True,
                               tooltip="可选。最终混音音轨，缺省用 drive_audio。"),
                io.Image.Input("first_frame", optional=True,
                               tooltip="可选。首帧图（两个阶段都会用原图重编码）。"),
                io.Image.Input("last_frame", optional=True,
                               tooltip="可选。尾帧图。"),
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image",
                                             tooltip="可选。参考图，每接一张自动出现下一个接入点。"),
                        prefix="ref_image_", min=0, max=REF_IMAGE_MAX,
                    ),
                    tooltip="可选。多张参考图（自动增减接入点），第二阶段会用原图重新 VAE 编码。",
                ),
                io.Autogrow.Input(
                    "ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video",
                                             tooltip="可选。参考视频（24fps 帧序列 batch，至少 5 帧）。"),
                        prefix="ref_video_", min=0, max=REF_VIDEO_MAX,
                    ),
                    tooltip="可选。最多 3 个参考视频（接一个自动出现下一个接入点）。",
                ),
                io.Autogrow.Input(
                    "ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio",
                                             tooltip="可选。对应序号参考视频的音轨（ref_video_audio_0 对应 ref_video_0）。"),
                        prefix="ref_video_audio_", min=0, max=REF_VIDEO_MAX,
                    ),
                    tooltip="可选。为对应序号的 ref_video 提供音轨，不接则该参考视频按无声处理。",
                ),
                io.Autogrow.Input(
                    "ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio",
                                             tooltip="可选。独立参考音频。"),
                        prefix="ref_audio_", min=0, max=REF_AUDIO_MAX,
                    ),
                    tooltip="可选。最多 3 条独立参考音频（接一个自动出现下一个接入点）。",
                ),
                io.Float.Input("reserved_vram", default=0.5, min=0.0, max=8.0, step=0.1,
                               tooltip="ReservedVRAM 预留（GB）。auto+manual 两个 setter 都在本节点开头执行一次（同原工作流时序），运行中不再卸载模型。"),
                io.Combo.Input("upscaler_model", options=_upscaler_options(),
                               default=DEFAULT_UPSCALER),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.Image.Output("frames"),
                io.Audio.Output("generated_audio"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, video_vae, audio_vae, prompt,
                aspect_ratio, length, stage1_megapixels, stage2_megapixels,
                task_type, audio_mode, stage2_sigma_steps, stage1_steps, split_step,
                stage1_sampler, stage1_scheduler, seed,
                drive_audio=None, final_audio=None, first_frame=None, last_frame=None,
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None,
                reserved_vram=0.5, upscaler_model=DEFAULT_UPSCALER):

        w1, h1 = _dims(aspect_ratio, stage1_megapixels)
        w2, h2 = _dims(aspect_ratio, stage2_megapixels)
        ref_values = _sorted_autogrow(ref_images)
        video_values = _sorted_autogrow(ref_videos)
        audio_values = _sorted_autogrow(ref_audios)
        info = [f"stage1 {w1}x{h1} | stage2 {w2}x{h2} | length {length} | seed {seed} | "
                f"refs {len(ref_values)} | ref_videos {len(video_values)} | ref_audios {len(audio_values)}"]

        # ---------- 0) VRAM 预留（严格复刻工作流时序） ----------
        # 原工作流里两个 ReservedVRAMSetter 都在 prompt 最开头执行（ conditioning 之前），
        # 之后全程不再卸载模型，阶段2 直接复用阶段1 已加载的 UNET。
        # 如果在阶段2 前再调一次（内部 unload_all_models），会强制卸载 20GB 模型并重新
        # staging，新旧两代模型滞留导致显存溢出到共享内存，每步慢 3 倍以上——
        # 所以两个 setter 都只在开头执行一次。
        _reserved_vram(reserved_vram, "auto")
        _reserved_vram(reserved_vram, "manual")

        # ---------- 1) 条件编码（第一阶段分辨率） ----------
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

        # ---------- 2) 直接使用 model 输入（不做任何模型加工/加速链） ----------

        # ---------- 3) 阶段1 采样（双时钟 + SplitSigmas） ----------
        dc_model, dc_sampler, dc_sigmas = _run(
            _node_cls("MiniMaxH3DualClockSamplerT8"),
            model=model, av_latent=av_latent,
            steps=stage1_steps, shift_video=12.0, shift_audio=3.0,
            sampler_name=stage1_sampler, scheduler=stage1_scheduler,
        )
        stage1_sigmas = _run(_node_cls("SplitSigmas"), sigmas=dc_sigmas, step=split_step)[0]
        noise = _run(_node_cls("RandomNoise"), noise_seed=seed)[0]
        guider1 = _run(_node_cls("BasicGuider"), model=dc_model, conditioning=positive)[0]

        s1_out, s1_denoised = _run(
            _node_cls("SamplerCustomAdvanced"),
            noise=noise, guider=guider1, sampler=dc_sampler,
            sigmas=stage1_sigmas, latent_image=av_latent,
        )

        # ---------- 4) 桥接：AV 分离 → 3D latent 放大 → 原图重编码条件 → 合并 ----------
        video_latent, audio_latent = _run(
            _node_cls("LTXVSeparateAVLatent"), av_latent=s1_denoised,
        )
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

        # 释放阶段1 独占对象的引用，避免它们滞留到高分辨率阶段2 挤占显存
        # （noise 阶段2 还要用，保留；positive 已被 cond2 取代）
        del dc_model, dc_sampler, dc_sigmas, guider1, s1_out, s1_denoised
        del video_latent, audio_latent, upscaled_video

        edit_kwargs = {}
        for i, img in enumerate(ref_values[:REF_IMAGE_MAX], start=1):
            edit_kwargs[f"ref_image{i}"] = img
        edit_node = _node_cls("H3_EditConditioningWH")
        cond2 = edit_node.edit(
            positive, width=w2, height=h2, rescale_mode="nearest",
            vae=video_vae, first_frame=first_frame, last_frame=last_frame,
            **edit_kwargs,
        )[0]
        info.append("stage2 cond re-encoded from original frames/refs")

        # ---------- 5) 阶段2 采样（euler + 蒸馏 sigmas） ----------
        guider2 = _run(_node_cls("BasicGuider"), model=model, conditioning=cond2)[0]
        sampler2 = _run(_node_cls("KSamplerSelect"), sampler_name="euler")[0]
        sigmas2 = _run(_node_cls("ManualSigmas"),
                       sigmas=STAGE2_SIGMA_PRESETS[stage2_sigma_steps])[0]

        final_latent = _run(
            _node_cls("SamplerCustomAdvanced"),
            noise=noise, guider=guider2, sampler=sampler2,
            sigmas=sigmas2, latent_image=stage2_latent_in,
        )[0]

        info.append(f"stage2 sigmas: {stage2_sigma_steps}")

        # ---------- 6) 内置 AV 解码（复用 T8 官方解码节点） ----------
        frames, generated_audio = None, None
        try:
            decoded = _run(
                _node_cls("MiniMaxH3AVDecodeT8"),
                av_latent=final_latent, video_vae=video_vae, audio_vae=audio_vae,
            )
            frames, generated_audio = decoded[0], decoded[1]
            n_frames = frames.shape[0] if hasattr(frames, "shape") else "?"
            info.append(f"decoded {n_frames} frames @24fps")
        except Exception as e:
            # 解码失败不丢 latent：av_latent 输出仍可外接官方解码节点兜底
            logging.warning("[HJL] 内置 AV 解码失败（latent 仍正常输出）: %s", e)
            info.append("decode failed (use external MiniMaxH3AVDecodeT8)")

        return io.NodeOutput(final_latent, frames, generated_audio, " | ".join(info))


NODE_CLASS_MAPPINGS = {
    "H3_TwoPassSampler": H3_TwoPassSampler,
}
