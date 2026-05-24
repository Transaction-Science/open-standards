"""WAI source-side neural encoder for `wai.neural.video_bmshj2018`.

Per-frame neural video: each frame is encoded through CompressAI's
bmshj2018-factorized synthesis + zstd-packed int8 latents (the same
payload format as `wai.neural.bmshj2018`), and the per-frame payloads
are bundled into a single WAI envelope with a small frame-table header.

This is browser-decodable — bmshj2018 runs in onnxruntime-web (WebGPU
or WASM). Full DCVC-RT (the 2026 SOTA for inter-frame neural video) is
NOT shipped here: its decode path requires NVIDIA CUDA + custom CUDA
kernels and doesn't run in a browser sink. SPEC.md §5 documents it as
a native-sink-only future capability.

Wire format:
  <IIIIBB>  H, W, n_frames, fps_x_1000, latent_C, latent_S_log2
  n × <I>   per-frame zstd-payload length (L_i)
  n × L_i bytes  zstd-compressed int8 latents shape [C, S, S]

Per-frame format matches `wai.neural.bmshj2018` except the per-image
H,W,C,S header is hoisted to the envelope-level header — every frame
shares the same shape.

Usage:
  python tools/wai_video_bmshj2018_encode.py <input.mp4> -o <output.wai>
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd
from compressai.zoo import models

WAI_MAGIC = b"WAI1"
QUALITY   = 3        # CompressAI quality knob (1..8)


def encode(in_path: str, out_path: str, size: int, fps: float) -> None:
    if size % 16 != 0:
        sys.exit("--size must be a multiple of 16")

    # ffmpeg → raw RGB frames at the target fps + square size.
    print(f"extracting frames via ffmpeg ({size}x{size} @ {fps} fps)…")
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", in_path,
         "-vf", f"scale={size}:{size},fps={fps}",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        check=True, capture_output=True)
    raw_video = np.frombuffer(proc.stdout, dtype=np.uint8)
    bytes_per_frame = size * size * 3
    if len(raw_video) % bytes_per_frame:
        sys.exit(f"ffmpeg output {len(raw_video)} B not a multiple of frame size")
    n_frames = len(raw_video) // bytes_per_frame
    frames = raw_video.reshape(n_frames, size, size, 3)
    print(f"  {n_frames} frames, {len(raw_video):,} B raw RGB")

    print(f"loading bmshj2018-factorized (quality={QUALITY})…")
    net = models["bmshj2018-factorized"](quality=QUALITY, pretrained=True).eval()
    cctx = zstd.ZstdCompressor(level=22)

    per_frame_payloads: list[bytes] = []
    psnrs: list[float] = []
    for i in range(n_frames):
        arr = frames[i].astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            y = net.g_a(x)
            y_q = torch.round(y).clamp(-128, 127).to(torch.int8)
        raw_latents = y_q.squeeze(0).numpy().tobytes()
        per_frame_payloads.append(cctx.compress(raw_latents))
        # PSNR (encoder-side upper bound)
        with torch.no_grad():
            x_hat = net.g_s(y_q.float()).clamp(0, 1)
        psnrs.append(10 * np.log10(1.0 / float(((x_hat - x) ** 2).mean())))
        if (i + 1) % 16 == 0 or i == n_frames - 1:
            print(f"  encoded {i+1}/{n_frames} (mean PSNR {np.mean(psnrs):.2f} dB)")

    C, H_lat, _ = y_q.shape[1:]
    s_log2 = int(np.log2(H_lat))

    # Envelope payload
    header = struct.pack("<IIIIBB", size, size, n_frames, int(round(fps * 1000)),
                         C, s_log2)
    table = b"".join(struct.pack("<I", len(p)) for p in per_frame_payloads)
    body  = b"".join(per_frame_payloads)
    payload = header + table + body

    manifest = {
        "wai": "1.0", "media": "video", "intent": "replicate",
        "model_requirement": {
            "capability": "wai.neural.video_bmshj2018",
            "fallback":   "wai.video.av1",
        },
        "conditioning": {"kind": "video_bmshj2018_frames_zstd"},
        "target": {"H": size, "W": size, "n_frames": int(n_frames),
                   "fps_x_1000": int(round(fps * 1000)),
                   "model_quality": QUALITY},
    }
    mb = json.dumps(manifest, separators=(",", ":")).encode()
    with open(out_path, "wb") as f:
        f.write(WAI_MAGIC)
        f.write(struct.pack("<I", len(mb))); f.write(mb)
        f.write(struct.pack("<I", len(payload))); f.write(payload)

    envelope_size = Path(out_path).stat().st_size
    dur = n_frames / fps
    raw_rgb_total = bytes_per_frame * n_frames
    kbps = envelope_size * 8 / dur / 1000
    print()
    print(f"input video:        {n_frames} frames {size}x{size}@{fps}fps "
          f"= {dur:.2f}s ({raw_rgb_total:,} B raw RGB)")
    print(f"WAI envelope:       {envelope_size:>10,} B  "
          f"({raw_rgb_total/envelope_size:6.1f}x compressed, "
          f"{kbps:6.1f} kbps)")
    print(f"mean encoder PSNR:  {np.mean(psnrs):.2f} dB")
    print(f"model on the wire:           0  (bmshj2018 weights stay at the sink)")
    print(f"capability declared:{manifest['model_requirement']['capability']:>30}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--size", type=int, default=256,
                   help="Square side length; must be a multiple of 16.")
    p.add_argument("--fps",  type=float, default=12.0,
                   help="Target frame rate.")
    a = p.parse_args()
    encode(a.input, a.output, a.size, a.fps)
