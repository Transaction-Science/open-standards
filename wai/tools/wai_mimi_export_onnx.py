"""Export the Mimi decoder to ONNX for the WAI browser runtime.

Mimi takes `audio_codes [B, n_codebooks, T]` directly (like DAC) and
returns `audio_values [B, 1, samples]`. Same WAI neural-audio ABI as
EnCodec/DAC/WavTokenizer: `[1, 1, q, t]` int64 in + `[1]` float32 in
(scales unused) → `[1, 1, samples]` float out.

Usage:
  python tools/wai_mimi_export_onnx.py \
    --out wai-web/demo/models/mimi/decoder.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import MimiModel


def _patch_packed_sequence_detection() -> None:
    """Mimi's decoder calls into transformers/masking_utils.py to detect
    packed sequences in `position_ids`. That path uses `torch.diff` plus
    a bool->cumsum chain — neither of which the torch.onnx tracer handles
    cleanly. For a single-sequence inference graph (which is what we
    export) the result is always None, so short-circuit it.
    """
    import transformers.masking_utils as mu

    def _const_none(*args, **kwargs):
        return None
    # The real fn is find_packed_sequence_indices (masking_utils.py:708).
    # masking_utils itself calls it via a local name inside other helpers,
    # so we patch the canonical attribute the rest of the module imports.
    mu.find_packed_sequence_indices = _const_none


class WaiMimiDecoder(torch.nn.Module):
    """Adapts MimiModel.decode to the WAI neural-audio ONNX contract."""
    def __init__(self, model: MimiModel):
        super().__init__()
        self.m = model

    def forward(self, audio_codes: torch.Tensor,
                audio_scales: torch.Tensor) -> torch.Tensor:
        codes = audio_codes.squeeze(1)        # [B, 1, q, t] → [B, q, t]
        out = self.m.decode(audio_codes=codes, return_dict=True)
        wav = out.audio_values
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)            # → [B, 1, samples]
        keep_scales = audio_scales.sum() / audio_scales.sum().clamp(min=1e-9)
        return wav * keep_scales


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="kyutai/mimi",
                    help="HF model id or local path.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    _patch_packed_sequence_detection()
    print(f"loading Mimi from {args.model_id}…")
    m = MimiModel.from_pretrained(args.model_id).eval()
    n_q = getattr(m.config, "num_quantizers", None) or \
          (getattr(m.config, "num_semantic_quantizers", 0)
           + getattr(m.config, "num_acoustic_quantizers", 8))
    print(f"  num_quantizers≈{n_q}  sampling_rate={m.config.sampling_rate}  "
          f"frame_rate={m.config.frame_rate}")
    wrap = WaiMimiDecoder(m).eval()

    # Build dummy with the same q the encoder produces (use a runtime probe
    # via a tiny dummy forward pass to discover the true number).
    with torch.no_grad():
        probe = m.encode(torch.zeros(1, 1, 24000), padding_mask=None)
    q = probe.audio_codes.shape[1]
    print(f"  probed q={q}")
    T = 64
    dummy_codes  = torch.zeros(1, 1, q, T, dtype=torch.long)
    dummy_scales = torch.ones (1,           dtype=torch.float32)

    with torch.no_grad():
        ref = wrap(dummy_codes, dummy_scales)
    print(f"reference forward pass OK: {tuple(ref.shape)}")

    print(f"exporting → {out}  (opset {args.opset})…")
    torch.onnx.export(
        wrap, (dummy_codes, dummy_scales), out.as_posix(),
        input_names=["audio_codes", "audio_scales"],
        output_names=["audio_values"],
        dynamic_axes={"audio_codes": {3: "T"}, "audio_values": {2: "samples"}},
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
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
