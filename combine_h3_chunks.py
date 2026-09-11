"""把 H3_TwoPassSamplerLongTime 逐段落盘的 PNG 序列合成一个 mp4。

用法:
    python combine_h3_chunks.py <h3_chunks子目录> [--fps 24] [--crf 18]

例:
    python combine_h3_chunks.py D:/ComfyUI_windows_portable_nvidia/ComfyUI/output/h3_chunks/h3_chunk

流式经 ffmpeg 编码（H.264/yuv420p），不把整段帧读进内存。
依赖 imageio-ffmpeg（自带 ffmpeg 二进制）。
"""

import argparse
import glob
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir", help="包含 chunk_XXXX 子目录的目录")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--crf", type=int, default=18, help="H.264 质量越小越好，18 接近无损")
    ap.add_argument("--out", default="combined.mp4")
    args = ap.parse_args()

    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("缺少 imageio-ffmpeg：请在 ComfyUI 内置 Python 里 pip install imageio-ffmpeg")

    pngs = []
    for d in sorted(glob.glob(os.path.join(args.base_dir, "chunk_*"))):
        pngs.extend(sorted(glob.glob(os.path.join(d, "*.png"))))
    if not pngs:
        sys.exit(f"在 {args.base_dir} 下没找到 chunk_*/PNG")

    list_path = os.path.join(args.base_dir, "concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for p in pngs:
            f.write("file '%s'\n" % os.path.abspath(p).replace("\\", "/").replace("'", "'\\''"))
            f.write("duration %.6f\n" % (1.0 / args.fps))

    out_path = os.path.join(args.base_dir, args.out)
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf),
           "-r", str(args.fps), out_path]
    print("合成中… 共 %d 帧" % len(pngs))
    subprocess.run(cmd, check=True)
    print("完成:", out_path, "(%.1f MB)" % (os.path.getsize(out_path) / 1048576))


if __name__ == "__main__":
    main()
