"""Extract per-channel CDF tables + medians from bmshj2018-factorized.

The EntropyBottleneck's prior is *factorized* — every spatial position
in channel `c` shares the same CDF. After `net.update()`, three tensors
parameterize the rANS coder:

  _quantized_cdf   int32 [C, max_cdf_length]   row-major per-channel CDF
  _cdf_length      int32 [C]                   active length per channel
  _offset          int32 [C]                   integer offset per channel
  _get_medians()   float [C, 1, 1]             per-channel median for de-quantization

These four arrays are everything a JS / Rust decoder needs to reproduce
the int32 symbols, then dequantize back to float32 `y_hat` for g_s.

Output is a compact JSON sidecar shipped at
`wai-web/demo/models/bmshj2018/cdfs_q<N>.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from compressai.zoo import models


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading bmshj2018-factorized (quality={args.quality})…")
    net = models["bmshj2018-factorized"](quality=args.quality, pretrained=True).eval()
    net.update(force=True)
    eb = net.entropy_bottleneck

    cdf       = eb._quantized_cdf.cpu().int().tolist()      # list[list[int]]  shape [C, max_len]
    cdf_len   = eb._cdf_length.cpu().int().tolist()         # list[int]        len  C
    offset    = eb._offset.cpu().int().tolist()             # list[int]        len  C
    medians   = eb._get_medians().detach().reshape(-1).cpu().float().tolist()

    C = len(cdf_len)
    blob = {
        "model": "bmshj2018-factorized",
        "quality": args.quality,
        "precision": 16,
        "bypass_precision": 4,
        "channels": C,
        "cdf_length": cdf_len,
        "offset": offset,
        "medians": medians,
        "cdf": cdf,    # large; we trim each row to cdf_length below
    }
    # Save bytes: trim each CDF row to its actual length.
    blob["cdf"] = [row[:cdf_len[i]] for i, row in enumerate(cdf)]

    out.write_text(json.dumps(blob, separators=(",", ":")))
    print(f"  channels={C}  cdf_max_len={max(cdf_len)}")
    print(f"  wrote {out} ({out.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
