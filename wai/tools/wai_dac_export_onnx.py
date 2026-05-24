"""Export the DAC 44.1 kHz decoder to ONNX for the WAI browser runtime.

DAC's decode signature differs from EnCodec's: it takes
`audio_codes [B, n_codebooks, T]` directly (no chunking, no audio_scales).
We wrap it so the exported ONNX accepts the same `[1, 1, q, t]` int64 +
`[1]` float32 inputs that the shared WAI audio handler feeds for every
neural audio codec — the `audio_scales` input is accepted and ignored,
and the leading chunks dim is squeezed.

Usage:
  python tools/wai_dac_export_onnx.py \
    --out wai-web/demo/models/dac_44khz/decoder.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import DacModel


class WaiDacDecoder(torch.nn.Module):
    """Adapts DacModel.decode to the WAI neural-audio ONNX contract.

    Inputs:
      audio_codes  int64 [1, 1, q, t]   (chunks dim is unused; kept for ABI parity)
      audio_scales float [1]            (unused; DAC has no per-chunk scaling)
    Output:
      audio_values float [1, 1, n_samples]
    """
    def __init__(self, model: DacModel):
        super().__init__()
        self.m = model

    def forward(self, audio_codes: torch.Tensor,
                audio_scales: torch.Tensor) -> torch.Tensor:
        # Squeeze the WAI "chunks" dim → [B, q, t] which is what DAC wants.
        codes = audio_codes.squeeze(1)
        out = self.m.decode(audio_codes=codes, return_dict=True)
        wav = out.audio_values   # [B, samples] in this transformers version
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)            # → [B, 1, samples]
        # Touch audio_scales so ONNX keeps it in the input signature for ABI
        # parity with EnCodec/WavTokenizer. The expression collapses to 1.0.
        keep_scales = audio_scales.sum() / audio_scales.sum().clamp(min=1e-9)
        return wav * keep_scales


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="descript/dac_44khz",
                    help="HF model id or local path.")
    ap.add_argument("--out", required=True, help="Output .onnx path.")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading DAC from {args.model_id}…")
    m = DacModel.from_pretrained(args.model_id).eval()
    print(f"  n_codebooks={m.config.n_codebooks}  "
          f"sampling_rate={m.config.sampling_rate}  "
          f"hop_length={m.config.hop_length}")
    wrap = WaiDacDecoder(m).eval()

    q = m.config.n_codebooks
    T = 128
    dummy_codes  = torch.zeros(1, 1, q, T, dtype=torch.long)
    dummy_scales = torch.ones (1,           dtype=torch.float32)

    with torch.no_grad():
        ref = wrap(dummy_codes, dummy_scales)
    print(f"reference forward pass OK: {tuple(ref.shape)}")

    print(f"exporting → {out}  (opset {args.opset})…")
    torch.onnx.export(
        wrap,
        (dummy_codes, dummy_scales),
        out.as_posix(),
        input_names=["audio_codes", "audio_scales"],
        output_names=["audio_values"],
        dynamic_axes={
            "audio_codes":  {3: "T"},
            "audio_values": {2: "samples"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,    # legacy tracer; see tools/wai_encodec_export_onnx.py
    )
    sz = out.stat().st_size
    print(f"wrote {sz:,} B")

    import onnxruntime as ort, numpy as np
    sess = ort.InferenceSession(out.as_posix(), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {
        "audio_codes":  dummy_codes.numpy(),
        "audio_scales": dummy_scales.numpy(),
    })
    diff = float(np.max(np.abs(ort_out[0] - ref.numpy())))
    print(f"onnxruntime sanity: shape={ort_out[0].shape}  "
          f"max |onnx-torch| = {diff:.3e}")
    if diff > 1e-3:
        print("WARNING: numerical mismatch > 1e-3", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
