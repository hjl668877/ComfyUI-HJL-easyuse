# ComfyUI-HJL-easyuse

English | [简体中文](README.md)

A collection of ComfyUI custom nodes for **MiniMax H3 audio-video generation**. Packs the multi-node two-pass sampling workflow into one-click nodes.

## Nodes

### H3 Two-Pass Sampler (HJL)

One-click recreation of the MiniMax H3 two-pass sampling workflow (low-res DMD dual-clock sampling → 3D latent upscale + original-image VAE re-encoded conditioning → high-res distillation-sigmas refinement → built-in AV decoding).

![H3 Two-Pass Sampler node](screenshot-H3-Two-Pass-Sampler.png)

**Inputs:**

| Port | Description |
|---|---|
| model / clip / video_vae / audio_vae | H3 base model, Qwen3-VL CLIP, video VAE (fp16), audio VAE (fp32) |
| prompt / aspect_ratio / length | Prompt, aspect ratio (6 options incl. 16:9), frame count (auto-aligned to 17n+5) |
| stage1_megapixels | Stage-1 target pixels (default 0.4MP; 16:9 → 864x480) |
| stage2_megapixels | Stage-2 target pixels (default 1.2MP; 16:9 → 1504x832) |
| task_type / audio_mode | T8 task type and audio mode |
| stage2_sigma_steps | Stage-2 distillation sigma preset (3/4/5 steps) |
| stage1_steps / split_step / stage1_sampler / stage1_scheduler | Stage-1 dual-clock sampling parameters |
| seed | Random seed |
| drive_audio / final_audio / first_frame / last_frame | Optional: driving audio, mixdown track, first/last frame |
| ref_images (autogrow, up to 8) | Reference images; stage 2 re-encodes them from the originals via VAE (best quality) |
| ref_videos (autogrow, up to 3) | Reference videos (24fps frame batches) |
| ref_video_audios (autogrow, up to 3) | Audio track for the reference video with the same index |
| ref_audios (autogrow, up to 3) | Standalone reference audio |
| reserved_vram / upscaler_model | VRAM reservation and upscaler model selection |

**Outputs:**

| Port | Description |
|---|---|
| av_latent | Combined audio-video latent (can still be fed to MiniMaxH3AVDecodeT8 for post-processing) |
| frames | Decoded image frame sequence (IMAGE) |
| generated_audio | Decoded audio (AUDIO) |
| info | Execution summary (resolution / frame count / reference counts / sigma preset) |

> **VRAM timing note**: both internal ReservedVRAM settings execute once at the very beginning (matching the real execution order of the original workflow). No models are unloaded mid-run — stage 2 reuses the UNET already loaded by stage 1. Do not attach external nodes that unload models mid-run to the same prompt.

### H3 Edit Conditioning W/H (HJL)

Built for H3 two-pass upscaling: updates conditioning width/height and resizes the reference latents inside `minimax_refs` / `minimax_keyframes` to the new resolution.

- With `vae` + original ref_image / first_frame / last_frame connected, **re-encodes from the original images** at the new resolution (best quality, on par with official high-res re-conditioning);
- References without an original image fall back to latent interpolation (trilinear / nearest / area);
- Automatically syncs each reference's latent_h / latent_w / latent_t, completely eliminating the `shape mismatch` error in two-pass upscaling.

Quality ranking: **original-image re-encoding (images connected) > latent decode→encode > latent interpolation**.

## Required Dependencies

These node packages are called at runtime and must be installed alongside:

| Dependency | Purpose | License | Link |
|---|---|---|---|
| comfyui-minimax-h3-audio-T8 | Core nodes: H3 Conditioning / DualClockSampler / AVDecode etc. | GPL-3.0-or-later | <https://github.com/T8mars/comfyui-minimax-h3-audio-T8> |
| ComfyUI-ReservedVRAM | VRAM reservation & cleanup (ReservedVRAMSetter) | Apache-2.0 | <https://github.com/Windecay/ComfyUI-ReservedVRAM> |
| Comfyui_Minimax_h3_latent_Upscaler | H3 3D latent upscaler model (MinimaxH3LatentUpscaler3D) | Unlicensed | <https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler> |
| ComfyUI built-in nodes | RandomNoise / SplitSigmas / BasicGuider / KSamplerSelect / ManualSigmas / SamplerCustomAdvanced / LTXVSeparateAVLatent / LTXVConcatAVLatent | GPL-3.0 (ComfyUI core) | Bundled with ComfyUI |

> Thanks to the authors of the upstream projects. This repository neither copies nor modifies any upstream source code — it only calls their nodes through the ComfyUI node registry at runtime.

## Acceleration Technologies (Optional)

The following two are **not hard dependencies** — H3 Two-Pass Sampler uses the plain model path by default. They appear in the OpenVDN accelerated example workflow under `workflows/`; install as needed:

### VDN-H3 (OpenVDN VideoDeltaNet hybrid-attention acceleration)

<https://github.com/OpenVDN/vdn-minimax-h3>

Attaches a frame-wise linear-attention branch plus two small LoRA adapters to MiniMax H3 for near-lossless inference speedup (generates in as few as 8 denoising steps). The example workflow `Minimax_H3_OpenVDN_DMD8_FL2VA_two_pass_workflow_HJL.json` loads the `OpenVDN/vdn-minimax-h3` weights from HuggingFace (the `stage_dmd_8nfe` branch) via T8's `MiniMaxH3VDNModelComposerT8Advanced` node.

- Code: Apache-2.0
- **Model weights: released separately on HuggingFace under the MiniMax H3 Community License** (not covered by the Apache license)

### Sol-Attn (sparse attention acceleration)

Sol-Attn is a training-free sparse attention method from NVIDIA ([arXiv 2607.24027](https://arxiv.org/abs/2607.24027)) that sparsifies long-sequence attention at the block level, substantially speeding up video generation.

The `SolAttnMiniMax` node in the example workflow comes from [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) (the single-file variant, `sol_attn_minimax_v2.py`). Notes:

- That repository is now marked **DEPRECATED** — recent ComfyUI / comfy-kitchen builds ship an optimized built-in Sparse Attention (including sol-attn); prefer the `sol_attn` CUDA kernels from `comfy-kitchen` (requires sm_80+ / bf16 / head_dim 128);
- Requires `comfy_kitchen` (installed with `sol_attn` support);
- `tau` (sparsity temperature) and `start/end_percent` trade quality against speed; the first run compiles kernels and will be slower.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hjl668877/ComfyUI-HJL-easyuse.git
```

Then install the dependency node packs and restart ComfyUI.

Model files (H3 base model / VAE / CLIP / latent upscaler) should be downloaded and placed according to the instructions in the comfyui-minimax-h3-audio-T8 and upscaler repositories.

## Example Workflows

Two examples are provided in `workflows/`:

| File | Description | Extra requirements |
|---|---|---|
| `Minimax_H3_two_pass_sampler_example_workflow_HJL.json` | Basic two-pass workflow (one-click H3_TwoPassSampler version) | Required dependencies only |
| `Minimax_H3_OpenVDN_DMD8_FL2VA_two_pass_workflow_HJL.json` | OpenVDN DMD8 + FL2VA (first-last frame) accelerated variant with VDN model composition and Sol-Attn sparse attention | Also needs VDN-H3 weights and the Sol-Attn node — see "Acceleration Technologies" above |

Place the model files as described in the T8 repository, then run. Overall layout of the basic workflow:

![Basic example workflow](screenshot-example-workflow.png)

## License

This node pack strongly depends on comfyui-minimax-h3-audio-T8 (GPL-3.0, called dynamically at runtime). To keep the licenses consistent, this repository is released under **GPL-3.0-or-later** (see [LICENSE](LICENSE)). The upstream node packages remain the property of their respective authors — please follow their original licenses (the VDN-H3 model weights are subject to the MiniMax H3 Community License).

## Roadmap

- [x] Example workflow JSONs (basic + OpenVDN accelerated)
- [ ] More aspect-ratio and duration combinations
