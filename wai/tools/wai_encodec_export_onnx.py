"""Export the EnCodec-32 kHz decoder to ONNX for the WAI browser runtime.

EnCodec-32kHz has chunk_length_s=None and overlap=None — the whole token
stream is decoded in one pass with no chunking. That makes the ONNX
export a single graph: (audio_codes [1, n_chunks=1, q=4, T_var]) ->
(audio_values [1, 1, samples]).

The exported ONNX runs in the browser via onnxruntime-web (WebGPU when
available, WASM fallback) — bypassing transformers.js since v3.8 does
NOT ship an EncodecModel class (only the feature extractor).

Usage:
  python tools/wai_encodec_export_onnx.py \
    --in  /path/to/encodec-32khz \
    --out wai-web/demo/models/encodec_32khz/decoder.onnx
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import EncodecModel


class EncodecDecoderOnly(torch.nn.Module):
    """Wraps EncodecModel.decode for export.

    Inputs:
      audio_codes:  int64 [1, 1, num_quantizers, T]   (single chunk)
      audio_scales: float [1]                          (1.0 = no scaling)
    Output:
      audio_values: float [1, 1, num_samples]
    """
    def __init__(self, model: EncodecModel):
        super().__init__()
        self.m = model

    def forward(self, audio_codes: torch.Tensor,
                audio_scales: torch.Tensor) -> torch.Tensor:
        out = self.m.decode(audio_codes=audio_codes, audio_scales=audio_scales,
                            padding_mask=None, return_dict=True)
        return out.audio_values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True,
                    help="Path to a local EncodecModel checkpoint dir.")
    ap.add_argument("--out", dest="dst", required=True,
                    help="Output .onnx path.")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset (17 covers everything EnCodec uses).")
    args = ap.parse_args()
    if not Path(args.src).exists():
        sys.exit(f"checkpoint not found: {args.src}")
    out = Path(args.dst); out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading EnCodec from {args.src}…")
    m = EncodecModel.from_pretrained(args.src).eval()
    q = m.config.num_quantizers
    print(f"  num_quantizers={q}  sampling_rate={m.config.sampling_rate}  "
          f"audio_channels={m.config.audio_channels}")
    wrap = EncodecDecoderOnly(m).eval()

    # Dummy inputs with a representative T (160 frames ≈ 5 s @ 50 Hz code rate).
    T = 256
    dummy_codes  = torch.zeros(1, 1, q, T, dtype=torch.long)
    dummy_scales = torch.ones (1,           dtype=torch.float32)

    # Round-trip sanity check before export.
    with torch.no_grad():
        ref = wrap(dummy_codes, dummy_scales)
    print(f"reference forward pass OK: {tuple(ref.shape)}")

    print(f"exporting → {out}  (opset {args.opset})…")
    # Force the legacy TorchScript tracer (dynamo=False). torch 2.10's
    # new torch.export path can't handle the data-dependent control flow
    # in transformers EnCodec (Eq guards on a symbolic length); the
    # legacy tracer just unrolls everything at the dummy shape, which
    # works fine for our fixed-config decoder.
    torch.onnx.export(
        wrap,
        (dummy_codes, dummy_scales),
        out.as_posix(),
        input_names=["audio_codes", "audio_scales"],
        output_names=["audio_values"],
        dynamic_axes={
            "audio_codes":  {3: "T"},            # variable code-frame count
            "audio_values": {2: "samples"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    sz = out.stat().st_size
    print(f"wrote {sz:,} B")

    # Verify with onnxruntime.
    import onnxruntime as ort
    sess = ort.InferenceSession(out.as_posix(), providers=["CPUExecutionProvider"])
    ort_in_names = [i.name for i in sess.get_inputs()]
    print(f"onnxruntime sanity check: inputs={ort_in_names}")
    ort_out = sess.run(None, {
        "audio_codes":  dummy_codes.numpy(),
        "audio_scales": dummy_scales.numpy(),
    })
    print(f"  onnxruntime forward pass OK: {ort_out[0].shape}")
    # Numerical agreement
    import numpy as np
    diff = float(np.max(np.abs(ort_out[0] - ref.numpy())))
    print(f"  max |onnx - torch| = {diff:.3e}")
    if diff > 1e-3:
        print("WARNING: numerical mismatch larger than expected", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
