# ComfyUI-HJL-easyuse

[English](README_EN.md) | 简体中文

面向 **MiniMax H3 音视频生成**的 ComfyUI 自定义节点集。把多节点的双阶段采样工作流打包成一键节点。

## 节点列表

### H3 Two-Pass Sampler (HJL)

一键复刻 MiniMax H3 双阶段采样工作流（低分辨率 DMD 双时钟采样 → 3D latent 放大 + 原图 VAE 重编码条件 → 高分辨率蒸馏 sigmas 收尾 → 内置 AV 解码）。

![H3 Two-Pass Sampler 节点](screenshot-H3-Two-Pass-Sampler.png)

**输入：**

| 端口 | 说明 |
|---|---|
| model / clip / video_vae / audio_vae | H3 底模、Qwen3-VL CLIP、video VAE(fp16)、audio VAE(fp32) |
| prompt / aspect_ratio / length | 提示词、画幅比（16:9 等 6 种）、帧数（自动对齐 17n+5） |
| stage1_megapixels | 第一阶段目标像素（默认 0.4MP，16:9 → 864x480） |
| stage2_megapixels | 第二阶段目标像素（默认 1.2MP，16:9 → 1504x832） |
| task_type / audio_mode | T8 任务类型与音频模式 |
| stage2_sigma_steps | 第二阶段蒸馏 sigmas 预设（3/4/5 steps） |
| stage1_steps / split_step / stage1_sampler / stage1_scheduler | 第一阶段双时钟采样参数 |
| seed | 随机种子 |
| drive_audio / final_audio / first_frame / last_frame | 可选：驱动音频、混音音轨、首尾帧 |
| ref_images (autogrow, 最多 8) | 参考图，第二阶段用原图重新 VAE 编码（画质最优） |
| ref_videos (autogrow, 最多 3) | 参考视频（24fps 帧序列） |
| ref_video_audios (autogrow, 最多 3) | 为对应序号的 ref_video 配音轨 |
| ref_audios (autogrow, 最多 3) | 独立参考音频 |
| reserved_vram / upscaler_model | 显存预留与放大模型选择 |

**输出：**

| 端口 | 说明 |
|---|---|
| av_latent | 音视频合一 latent（仍可外接 MiniMaxH3AVDecodeT8 做后处理） |
| frames | 解码后的图像帧序列（IMAGE） |
| generated_audio | 解码后的音频（AUDIO） |
| info | 执行摘要（分辨率 / 帧数 / 引用数量 / sigmas 预设） |

> **显存时序说明**：节点内部的两个 ReservedVRAM 设置都在开头执行一次（与原工作流真实执行顺序一致），运行期间不卸载模型，阶段2 直接复用阶段1 已加载的 UNET。请勿在外部再对同一 prompt 挂载运行中卸载模型的节点。

### H3 Two-Pass Long Time V2 (HJL)

**长视频**节点：把整段视频按时间**均分**成多段，每段各跑一次完整双阶段采样，再用 **latent 续接**把段与段接上；支持**分段提示词**，让不同段落使用不同描述。

![H3 Two-Pass Long Time V2 节点](screenshot-H3-Two-Pass-Long-Time-V2.png)

单段时行为与 H3 Two-Pass Sampler 一致；超过单段上限（`temporal_chunk_frames`）时自动分段，从而突破单次生成的长度限制。

**核心机制**

| 机制 | 说明 |
|---|---|
| 时间分块 | 段数 = ⌈length ÷ temporal_chunk_frames⌉，再把总帧数**均分**到各段，避免出现极短的尾段（旧版固定切法会出现 136+136+16 这种尾巴，导致最后一段内容接不上） |
| latent 续接 | 取上一段**阶段1 去噪后的视频 latent 尾部**（默认 8 帧），按 `continuation_strength` 截断 sigma 调度后作为下一段起点——继承运动轨迹、细节与纹理，而不是每段从纯噪声重新「想象」 |
| 分段提示词 | `prompt` 内用**单独一行** `---` 分隔，依次对应第 1/2/3… 段；提示词段数少于渲染段数时，多余的段沿用最后一条；不写 `---` 则全段通用 |

**主要输入**（其余端口与 H3 Two-Pass Sampler 相同）

| 端口 | 默认 | 说明 |
|---|---|---|
| prompt | 空 | 视频描述，支持 `---` 分段 |
| length | 141 | **总**帧数（24fps） |
| temporal_chunk_frames | 136 | 单段帧数**上限**，必须 17 的倍数（建议设成你单次能稳定跑通的长度，6 秒 ≈ 136） |
| temporal_overlap_frames | 0 | **额外**丢弃的段首帧数。节点固定会丢弃与上一段尾帧重复的 1 帧，这里填 N 表示再多丢 N 帧，调大会造成时间跳跃 |
| latent_continuation | True | latent 续接开关（关闭则退化为每段纯噪声起点） |
| continuation_strength | 0.5 | 0–1：越大越贴近上一段（更连贯、变化更小）；与分段提示词同用时建议降到 0.3 左右 |
| continuation_frames | 8 | 用上一段末尾几个 latent 时间帧做种子 |
| same_seed_all_chunks | False | 全段共用 seed，纹理更一致 |
| save_per_chunk | True | 每段解完立刻写 PNG 到 `output/h3_chunks/<前缀>/chunk_XXXX/` 并释放该段帧 |
| combine_saved_chunks | False | 跑完用 ffmpeg 把各段 PNG 流式合成 `combined.mp4`（结尾零内存尖峰） |
| merge_chunks | True | 把各段拼成完整 `frames` 输出；**超过约 15 秒建议关闭**，改用上面的磁盘方案 |
| frame_dtype | uint8 | 段内累积精度：uint8 省 4 倍内存（输出仍是 fp32 `[0,1]`，画质无可闻影响） |
| chunk_filename_prefix | h3_chunk | 落盘子目录名 |

**输出**

| 端口 | 说明 |
|---|---|
| av_latent | 最后一段的 AV latent |
| frames | 完整帧序列（`merge_chunks=True`）；否则只含最后一段 |
| generated_audio | 按段拼接好的音轨 |
| **video** | 合成成功时的 VIDEO，可直接接视频预览/保存节点，节点上也会内嵌预览 |
| info | 执行摘要（段数 / 每段帧数 / 续接状态 / 提示词段数 / 落盘路径） |

**时长 ↔ 帧数 ↔ 段数对照**（24fps，`temporal_chunk_frames = 136`）

| 时长 | length | 段数 | 各段帧数 |
|---|---|---|---|
| 5.67s | 136 | 1 | 136 |
| 8s | 192 | 2 | 96 + 96 |
| **11.33s** | **272** | **2** | **136 + 136** ⭐ |
| 12s | 288 | 3 | 96 × 3 |
| **17s** | **408** | **3** | **136 × 3** ⭐ |
| 20s | 480 | 4 | 120 × 4 |
| **22.67s** | **544** | **4** | **136 × 4** ⭐ |
| 30s | 720 | 6 | 120 × 6 |
| **34s** | **816** | **6** | **136 × 6** ⭐ |
| 45.33s | 1088 | 8 | 136 × 8 ⭐ |

规律：`段数 = ⌈length ÷ 136⌉`；`length = N × 136` 时每段恰好等长（5.67 / 11.33 / 17.00 / 22.67 / 28.33 / 34.00 / 39.67 / 45.33 / 51.00 / 56.67 秒），显存与耗时可预测性最好。想固定段数时调 `temporal_chunk_frames`：`chunk ≥ length ÷ 期望段数`，再向上取到 17 的倍数。

**建议的三种取片方式**

| 场景 | 配置 |
|---|---|
| ≤15 秒 | `merge_chunks=True`，直接使用 `frames` / `video` 端口 |
| 更长（推荐） | `merge_chunks=False` + `save_per_chunk=True` + `combine_saved_chunks=True`，全程走磁盘，内存零尖峰 |
| 补救已落盘的 PNG | 独立工具 `combine_h3_chunks.py`（见下） |

> ⚠️ 长视频的瓶颈通常是**系统内存**而非显存：ComfyUI 以 fp32 保存帧（1.2MP 约 15MB/帧），且会按需 staging 模型权重。运行前关闭其他占内存的程序；`merge_chunks=True` 时结尾需要一份完整 fp32 帧序列（11 秒 ≈ 4GB，60 秒 ≈ 21GB）。

### H3 Edit Conditioning W/H (HJL)

H3 两阶段放大专用：更新 conditioning 宽高，并把 `minimax_refs` / `minimax_keyframes` 里的参考 latent 调整到新尺寸。

- 接 `vae` + 原始 ref_image / first_frame / last_frame 时，按新分辨率**从原图重新 VAE 编码**（画质最优，与官方高分辨率重新 Conditioning 同质量）；
- 未接原图的部分走 latent 插值（trilinear / nearest / area）；
- 自动同步 ref 的 latent_h / latent_w / latent_t，彻底解决两阶段放大的 `shape mismatch` 报错。

画质排序：**原图重编码（接图） > latent decode→encode > latent 插值**。

## 第三方兼容补丁：`t8_compat.py`

comfyui-minimax-h3-audio-T8 的 **Hybrid** 兼容性探测在当前 ComfyUI 上会构造一个非法 keyframe 索引（既不是首帧 0、也不是尾帧 `frame_count-1`），导致探测必然抛错，进而禁用 Hybrid 路径——表现为「首帧/尾帧 + 参考图」同时接线时报：

```
RuntimeError: The active MiniMax H3 PackedLayout implementation rejected
the guarded legacy Hybrid compatibility probe.
```

本包**不修改 T8 源文件**，而是在运行时打一个内存补丁（`t8_compat.py` → `ensure_t8_hybrid_probe_fix()`）：遇到非法索引时改写为首帧再交给原函数。T8 的真实生成路径只使用合法索引，因此对生成结果无任何影响；**上游修复后该补丁会自动退化为透明传递**，无需手动移除。包内各采样器节点（H3 Two-Pass Sampler / H3 Two-Pass Long Time V2）在 `execute` 开头都会调用一次（幂等）。

## 独立工具：`combine_h3_chunks.py`

把已落盘的逐段 PNG 序列合成为 MP4，**不需要重跑节点**：

```bash
python combine_h3_chunks.py "D:/ComfyUI_windows_portable_nvidia/ComfyUI/output/h3_chunks/h3_chunk" \
    --fps 24 --crf 18
```

按目录名顺序拼接所有 `chunk_*/` 下的 PNG，输出 `combined.mp4`（H.264 / yuv420p）。依赖 `imageio-ffmpeg`（自带 ffmpeg 二进制，`pip install imageio-ffmpeg`）。

## 依赖节点（必装）

本节点在运行时调用以下节点包，请一并安装：

| 依赖 | 用途 | 许可证 | 链接 |
|---|---|---|---|
| comfyui-minimax-h3-audio-T8 | H3 Conditioning / DualClockSampler / AVDecode 等核心节点 | GPL-3.0-or-later | <https://github.com/T8mars/comfyui-minimax-h3-audio-T8> |
| ComfyUI-ReservedVRAM | 显存预留与清理（ReservedVRAMSetter） | Apache-2.0 | <https://github.com/Windecay/ComfyUI-ReservedVRAM> |
| Comfyui_Minimax_h3_latent_Upscaler | H3 3D latent 放大模型（MinimaxH3LatentUpscaler3D） | 未标注 | <https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler> |
| ComfyUI 内置节点 | RandomNoise / SplitSigmas / BasicGuider / KSamplerSelect / ManualSigmas / SamplerCustomAdvanced / LTXVSeparateAVLatent / LTXVConcatAVLatent | GPL-3.0（ComfyUI 主程序） | ComfyUI 自带 |

> 感谢以上上游项目的作者。本仓库未复制/修改任何上游源码，仅在运行时通过 ComfyUI 节点注册表调用其节点。

## 加速技术（可选）

以下两项**不是本节点的必装依赖**——H3 Two-Pass Sampler 默认走原始 model 直连路径。它们出现在 `workflows/` 的 OpenVDN 加速示例工作流中，按需安装：

### VDN-H3（OpenVDN VideoDeltaNet 混合注意力加速）

<https://github.com/OpenVDN/vdn-minimax-h3>

在 MiniMax H3 上外挂一条帧级线性注意力分支 + 两个小型 LoRA 适配器，实现近无损的推理加速（8 步去噪即可出片）。示例工作流 `Minimax_H3_OpenVDN_DMD8_FL2VA_two_pass_workflow_HJL.json` 通过 T8 包的 `MiniMaxH3VDNModelComposerT8Advanced` 节点加载 HuggingFace 上的 `OpenVDN/vdn-minimax-h3` 权重（`stage_dmd_8nfe` 分支）。

- 代码：Apache-2.0
- **模型权重：单独发布在 HuggingFace，遵循 MiniMax H3 Community License**（不在 Apache 覆盖范围内）

### Sol-Attn（稀疏注意力加速）

Sol-Attn 是 NVIDIA 提出的免训练稀疏注意力方法（[arXiv 2607.24027](https://arxiv.org/abs/2607.24027)），对长序列注意力做块级稀疏化，显著加速视频生成。

示例工作流中的 `SolAttnMiniMax` 节点来自 [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton)（单文件版 `sol_attn_minimax_v2.py`）。注意：

- 该仓库现已标记 **DEPRECATED**——新版 ComfyUI / comfy-kitchen 已内置优化后的 Sparse Attention（含 sol-attn），优先通过 `comfy-kitchen` 的 `sol_attn` CUDA kernel 使用（需 sm_80+ / bf16 / head_dim 128）；
- 依赖 `comfy_kitchen`（安装时需带 `sol_attn` 支持）；
- `tau`（稀疏温度）、`start/end_percent` 用于平衡质量与速度；首次运行需编译 kernel，会偏慢。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hjl668877/ComfyUI-HJL-easyuse.git
```

然后安装依赖节点包并重启 ComfyUI。

所需模型文件（H3 底模 / VAE / CLIP / latent 放大模型）请按 comfyui-minimax-h3-audio-T8 与 upscaler 仓库的说明下载放置。

## Example Workflows

`workflows/` 目录提供三个示例：

| 文件 | 说明 | 额外依赖 |
|---|---|---|
| `Minimax_H3_two_pass_sampler_example_workflow_HJL.json` | 基础双阶段工作流（H3_TwoPassSampler 一键节点版） | 仅必装依赖 |
| `Minimax_H3_Two_Pass_FL2VA_Long_Time_V2_HJL_example.json` | **长视频示例**：H3 Two-Pass Long Time V2 + FL2VA 首尾帧，272 帧（11.33s / 2 段），内置 `---` 分段提示词与「时长-段数」对照便签 | 仅必装依赖 |
| `Minimax_H3_OpenVDN_DMD8_FL2VA_two_pass_workflow_HJL.json` | OpenVDN DMD8 + FL2VA（首尾帧）加速版，含 VDN 模型合成与 Sol-Attn 稀疏注意力 | 需另装 VDN-H3 权重与 Sol-Attn 节点，见上方「加速技术」 |

加载后按 T8 仓库说明放置模型文件即可运行。

基础版工作流整体效果：

![基础示例工作流](screenshot-example-workflow.png)

## License

本节点强依赖 GPL-3.0 的 comfyui-minimax-h3-audio-T8（运行时动态调用），为保持许可证一致，本仓库以 **GPL-3.0-or-later** 发布（见 [LICENSE](LICENSE)）。所依赖的上游节点包归各自作者所有，使用时请遵循其原始许可证（VDN-H3 模型权重遵循 MiniMax H3 Community License）。

## 更新计划

- [x] 示例工作流 JSON（基础版 + OpenVDN 加速版）
- [x] 长视频节点 H3 Two-Pass Long Time V2（时间分块 + latent 续接 + 分段提示词）
- [x] 长视频示例工作流（272 帧 / 2 段 / FL2VA）
- [ ] 更多画幅比与时长组合示例