"""WAI source-side neural encoder for `wai.neural.bmshj2018`.

bmshj2018-factorized (Ballé-Minnen-Singh-Hwang-Johnston 2018) from
CompressAI's model zoo. Encodes via CompressAI's rANS coder — the
exact byte format upstream uses — and ships the bitstream in the WAI
envelope.

Wire format (v0.4):
  <IIIIBB>  H, W, L, quality, C, S_log2
  L bytes   CompressAI rANS bitstream (one batch element only)

The sink loads `cdfs_q<quality>.json` once per (model, quality), runs
the matching rANS decoder, dequantizes int symbols back to float
latents (adding per-channel medians), and runs g_s through ONNX.

Usage:
  python tools/wai_bmshj2018_encode.py <input.png> -o <output.wai>
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from compressai.zoo import models
from PIL import Image

WAI_MAGIC = b"WAI1"
QUALITY   = 3        # CompressAI quality knob (1..8); must match the shipped CDF JSON


def encode(img_path: str, out_path: str, size: int = 256) -> None:
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    print(f"loading bmshj2018-factorized (quality={QUALITY})…")
    net = models["bmshj2018-factorized"](quality=QUALITY, pretrained=True).eval()
    net.update(force=True)

    with torch.no_grad():
        out = net.compress(x)              # {strings: [[bytes]], shape: latent HW}
        # PSNR sanity (decode via the same model)
        x_hat = net.decompress(out["strings"], out["shape"])["x_hat"].clamp(0, 1)
    psnr = 10 * np.log10(1.0 / float(((x_hat - x) ** 2).mean()))

    rans_bytes = out["strings"][0][0]
    H_lat, W_lat = out["shape"]
    assert H_lat == W_lat, f"non-square latent {H_lat}x{W_lat}"
    s_log2 = int(np.log2(H_lat))
    assert (1 << s_log2) == H_lat
    C = int(net.entropy_bottleneck._quantized_cdf.shape[0])

    header = struct.pack("<IIIIBB", size, size, len(rans_bytes), QUALITY, C, s_log2)
    payload = header + rans_bytes

    manifest = {
        "wai": "1.0", "media": "image", "intent": "replicate",
        "model_requirement": {
            "capability": "wai.neural.bmshj2018",
            "fallback":   "wai.image.jxl",
        },
        "conditioning": {"kind": "bmshj2018_rans"},
        "target": {"H": size, "W": size, "model_quality": QUALITY,
                   "orig_h": orig_h, "orig_w": orig_w,
                   "encoder_psnr_db": round(psnr, 2)},
    }
    mb = json.dumps(manifest, separators=(",", ":")).encode()
    with open(out_path, "wb") as f:
        f.write(WAI_MAGIC)
        f.write(struct.pack("<I", len(mb))); f.write(mb)
        f.write(struct.pack("<I", len(payload))); f.write(payload)

    envelope_size = Path(out_path).stat().st_size
    raw_rgb = size * size * 3
    print(f"input image:        {orig_w}x{orig_h} resized to {size}x{size} "
          f"({raw_rgb:,} B raw RGB)")
    print(f"  latent shape: [C={C}, S={H_lat}]")
    print(f"  rANS bytes:    {len(rans_bytes):,}")
    print(f"  envelope-side reconstruction PSNR: {psnr:.2f} dB")
    print(f"WAI envelope:       {envelope_size:>10,} B  "
          f"({raw_rgb/envelope_size:6.1f}x compressed vs raw RGB)")
    print(f"model on the wire:           0  (bmshj2018 weights stay at the sink)")
    print(f"capability declared:{manifest['model_requirement']['capability']:>30}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--size", type=int, default=256,
                   help="Square side length (must be /16).")
    a = p.parse_args()
    if a.size % 16 != 0:
        sys.exit("--size must be a multiple of 16")
    encode(a.input, a.output, a.size)
