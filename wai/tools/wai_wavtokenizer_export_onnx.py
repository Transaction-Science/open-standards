"""Export the WavTokenizer-40token decoder to ONNX for the WAI browser.

WavTokenizer has a single-codebook (q=1) VQ at 40 Hz. The model's
`decode` takes `features` (continuous embeddings) — we wire through the
discrete-codes path manually:

    code_idx  →  quantizer.codes_to_embedding (via decode_from_codes)
              →  backbone(emb) → head → audio

Same WAI neural-audio ABI as EnCodec/DAC/Mimi: int64 [1, 1, q, t] codes
+ float [1] scales → float [1, 1, samples].

Usage:
  python tools/wai_wavtokenizer_export_onnx.py \\
    --wavtokenizer-src /tmp/wavtokenizer \\
    --config  /tmp/wavtokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml \\
    --ckpt    /tmp/wavtokenizer_ckpt/large_unify_40.ckpt \\
    --out     wai-web/demo/models/wavtokenizer/decoder.onnx
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch


def _patch_istft_real(model) -> None:
    """ONNX has no complex tensor type. WavTokenizer's ISTFTHead builds a
    complex spectrum (`mag * (cos(p) + 1j*sin(p))`) and passes it through
    `ISTFT.forward` which calls `torch.fft.irfft`. The torch ONNX tracer
    bails with "Unknown number type: complex".

    Replace the head with an equivalent real-arithmetic version: precompute
    the IDFT cos/sin basis matrices, do the irfft as two real matmuls (one
    cosine, one sine), then run the same windowed overlap-add (which is
    already ONNX-friendly via Fold).

    This is a couple of orders of magnitude slower than FFT for very large
    n_fft, but at WavTokenizer's n_fft=2400 / hop=600 it's well under 1 s
    per inference on CPU and equally fast on WebGPU (matmul is the GPU's
    home turf).
    """
    from decoder.heads import ISTFTHead

    head = model.head
    if not isinstance(head, ISTFTHead):
        return
    istft = head.istft
    n_fft, hop, wl = istft.n_fft, istft.hop_length, istft.win_length
    n_freq = n_fft // 2 + 1
    pad = (wl - hop) // 2

    # IDFT basis: x[n] = (1/N) * sum_k W_k * [Re(S_k)*cos(θ_kn) - Im(S_k)*sin(θ_kn)]
    # where θ_kn = 2πkn/N and W_k = 1 for k∈{0, N/2}, 2 otherwise (real-input symmetry).
    n_idx = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)         # [1, N]
    k_idx = torch.arange(n_freq, dtype=torch.float32).unsqueeze(1)        # [n_freq, 1]
    angle = 2 * math.pi * k_idx * n_idx / n_fft                           # [n_freq, N]
    cos_b = torch.cos(angle) / n_fft
    sin_b = torch.sin(angle) / n_fft
    cos_b[1:n_freq - 1, :] *= 2
    sin_b[1:n_freq - 1, :] *= 2

    # Pre-bake an identity weight for ConvTranspose1d-based overlap-add.
    # In-channels=n_fft, out-channels=1, kernel=n_fft, stride=hop. Putting
    # `eye(n_fft)` in the weight makes input channel i contribute to output
    # samples [t*hop + i] which is exactly OLA. We use this because
    # torch.nn.functional.fold's ONNX export (col2im) is broken in
    # multiple opsets.
    ola_w = torch.eye(n_fft, dtype=torch.float32).unsqueeze(1)            # [N, 1, N]

    head.register_buffer("_real_cos_b", cos_b)        # [n_freq, n_fft]
    head.register_buffer("_real_sin_b", sin_b)
    head.register_buffer("_window",     istft.window.float())
    head.register_buffer("_ola_w",      ola_w)
    head._real_pad = pad
    head._real_wl  = wl
    head._real_hop = hop
    head._real_nfft = n_fft

    def _real_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)                                    # [B, 2*n_freq, T]
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=100.0)
        re = mag * torch.cos(p)                                            # [B, n_freq, T]
        im = mag * torch.sin(p)
        # Real IDFT: [n_freq, n_fft]^T @ [B, n_freq, T] → [B, n_fft, T]
        ifft = torch.einsum("kn,bkt->bnt", self._real_cos_b, re) \
             - torch.einsum("kn,bkt->bnt", self._real_sin_b, im)
        ifft = ifft * self._window[None, :, None]
        # ConvTranspose1d-based overlap-add (ONNX-friendly).
        y = torch.nn.functional.conv_transpose1d(
            ifft, self._ola_w, stride=self._real_hop)                     # [B, 1, out]
        y = y.squeeze(1)[:, self._real_pad:-self._real_pad]
        # Window envelope: same OLA against window^2 frames.
        T = ifft.shape[-1]
        win_sq = self._window.square().unsqueeze(0).unsqueeze(-1).expand(1, -1, T)  # [1, N, T]
        env = torch.nn.functional.conv_transpose1d(
            win_sq, self._ola_w, stride=self._real_hop)                   # [1, 1, out]
        env = env.squeeze(1)[:, self._real_pad:-self._real_pad]
        return y / env.clamp(min=1e-11)

    import types
    head.forward = types.MethodType(_real_forward, head)


class WaiWavTokenizerDecoder(torch.nn.Module):
    """Adapts WavTokenizer's three-stage decode to the WAI audio contract.

    Inputs:
      audio_codes  int64 [1, 1, q, t]   (q=1 for the 40-token variant)
      audio_scales float [1]            (no-op, kept for ABI parity)
    Output:
      audio_values float [1, 1, samples]
    """
    def __init__(self, model):
        super().__init__()
        self.feature_extractor = model.feature_extractor
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, audio_codes: torch.Tensor,
                audio_scales: torch.Tensor) -> torch.Tensor:
        # [1, 1, q, t] → [q, 1, t] (WavTokenizer convention)
        codes = audio_codes.squeeze(1).permute(1, 0, 2)
        # decode codes via the residual vector quantizer's codebooks.
        # The feature_extractor exposes the quantizer; its decode() takes
        # the integer codes and returns the continuous features [B, D, T].
        features = self.feature_extractor.encodec.quantizer.decode(codes)
        # Backbone is GRU/transformer producing [B, T, D]; head is iSTFT/iMDCT.
        bandwidth_id = torch.zeros(1, dtype=torch.long)
        x = self.backbone(features, bandwidth_id=bandwidth_id)
        wav = self.head(x)                          # [B, T]
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        keep_scales = audio_scales.sum() / audio_scales.sum().clamp(min=1e-9)
        return wav * keep_scales


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavtokenizer-src", default="/tmp/wavtokenizer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--opset",  type=int, default=17)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, args.wavtokenizer_src)
    from decoder.pretrained import WavTokenizer

    print(f"loading WavTokenizer from {Path(args.ckpt).name}…")
    m = WavTokenizer.from_pretrained0802(args.config, args.ckpt).eval()
    _patch_istft_real(m)
    wrap = WaiWavTokenizerDecoder(m).eval()

    q = 1
    T = 80                                       # 2 s @ 40 Hz frame rate
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
