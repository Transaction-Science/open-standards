"""Export the bmshj2018-factorized synthesis transform to ONNX.

The full bmshj2018 model has analysis (g_a), synthesis (g_s), and an
entropy bottleneck. For the sink we only need g_s — `int8 latents → RGB`
— since the encoder side does its own quantization and zstd-packs the
latents into the WAI envelope.

ONNX contract for `wai.neural.bmshj2018`:
  input:  latents  float32 [1, C, S, S]   (dequantized to float on the JS side)
  output: image    float32 [1, 3, H, W]   (in [0,1])

Usage:
  python tools/wai_bmshj2018_export_onnx.py \\
    --out wai-web/demo/models/bmshj2018/decoder.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from compressai.zoo import models


class WaiBmshjDecoder(torch.nn.Module):
    """Wraps g_s into the WAI image-neural ONNX contract."""
    def __init__(self, net):
        super().__init__()
        self.g_s = net.g_s

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.g_s(latents).clamp(0.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=int, default=3,
                    help="CompressAI quality knob (1..8). Must match encoder.")
    ap.add_argument("--size", type=int, default=256,
                    help="Square image side length (input/16 = latent side).")
    ap.add_argument("--out",   required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading bmshj2018-factorized (quality={args.quality})…")
    net = models["bmshj2018-factorized"](quality=args.quality, pretrained=True).eval()
    wrap = WaiBmshjDecoder(net).eval()

    # Probe latent shape at the chosen image size.
    with torch.no_grad():
        y = net.g_a(torch.zeros(1, 3, args.size, args.size))
    _, C, S, _ = y.shape
    print(f"  latent shape: [1, C={C}, S={S}, {S}]  →  image {args.size}x{args.size}")
    dummy = torch.zeros(1, C, S, S)

    with torch.no_grad():
        ref = wrap(dummy)
    print(f"reference forward pass OK: {tuple(ref.shape)}")

    print(f"exporting → {out}  (opset {args.opset})…")
    torch.onnx.export(
        wrap, (dummy,), out.as_posix(),
        input_names=["latents"], output_names=["image"],
        dynamic_axes={"latents": {2: "S", 3: "S"},
                      "image":   {2: "H", 3: "W"}},
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
    )
    sz = out.stat().st_size
    print(f"wrote {sz:,} B")

    import onnxruntime as ort, numpy as np
    sess = ort.InferenceSession(out.as_posix(), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"latents": dummy.numpy()})
    diff = float(np.max(np.abs(ort_out[0] - ref.numpy())))
    print(f"onnxruntime sanity: shape={ort_out[0].shape}  "
          f"max |onnx-torch| = {diff:.3e}")
    if diff > 1e-3:
        print("WARNING: numerical mismatch > 1e-3", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
