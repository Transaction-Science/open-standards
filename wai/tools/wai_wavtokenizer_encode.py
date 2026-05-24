"""WAI source-side neural encoder for `wai.neural.wavtokenizer`.

WavTokenizer-large-unify-40token (Jiang et al., ICLR 2025) — single
quantizer, 40 tokens per second of 24 kHz audio, single 4096-entry
codebook. UTMOS at 0.9 kbps reportedly exceeds DAC at 9 kbps.

WavTokenizer is research code, not a transformers integration. We add
the cloned repo to sys.path and use its `decoder.pretrained.WavTokenizer`
class directly. Set --wavtokenizer-src to the cloned repo path (default
/tmp/wavtokenizer from `git clone`).

Wire format (uniform with all wai.neural.<audio> capabilities):
  <III>  q  t  n_samples         (little-endian u32 × 3)
  q*t × u16 codes (row-major, LE)

For WavTokenizer-40token q=1 (single quantizer). The codes are u16 even
though the codebook fits in 12 bits — keeping the WAI wire format
uniform across audio codecs has more value than the 4-bit-per-code save.

Usage:
  python tools/wai_wavtokenizer_encode.py <input.wav> \\
    --wavtokenizer-src /tmp/wavtokenizer \\
    --config  /tmp/wavtokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml \\
    --ckpt    /tmp/wavtokenizer_ckpt/large_unify_40.ckpt \\
    -o <output.wai>
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

WAI_MAGIC = b"WAI1"
SR = 24_000


def encode(wav_path: str, out_path: str, src: str, cfg: str, ckpt: str) -> None:
    sys.path.insert(0, src)
    import torch
    from decoder.pretrained import WavTokenizer
    from encoder.utils import convert_audio

    with wave.open(wav_path, "rb") as w:
        sr_in = w.getframerate()
        raw_bytes = w.getnframes() * 2 * w.getnchannels()

    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", wav_path, "-ac", "1",
         "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    raw = np.frombuffer(pcm, np.float32)

    print(f"loading WavTokenizer from {Path(ckpt).name}…")
    model = WavTokenizer.from_pretrained0802(cfg, ckpt).eval()

    # Reproduce infer.py's pipeline.
    wav = torch.from_numpy(raw).reshape(1, -1)
    wav = convert_audio(wav, SR, SR, 1)
    bandwidth_id = torch.tensor([0])
    with torch.no_grad():
        features, discrete_codes = model.encode_infer(wav, bandwidth_id=bandwidth_id)
    # discrete_codes: [n_q, batch=1, T]. Squeeze to [n_q, T].
    codes = discrete_codes.squeeze(1).numpy().astype(np.uint16)
    q, t = codes.shape

    payload = struct.pack("<III", q, t, len(raw)) + codes.tobytes()

    manifest = {
        "wai": "1.0", "media": "audio", "intent": "replicate",
        "model_requirement": {
            "capability": "wai.neural.wavtokenizer",
            "fallback":   "wai.audio.opus",
        },
        "conditioning": {"kind": "wavtokenizer_tokens"},
        "target": {"sr": SR, "n_samples": int(len(raw)),
                   "codebooks": int(q), "frames": int(t)},
    }
    mb = json.dumps(manifest, separators=(",", ":")).encode()
    with open(out_path, "wb") as f:
        f.write(WAI_MAGIC)
        f.write(struct.pack("<I", len(mb))); f.write(mb)
        f.write(struct.pack("<I", len(payload))); f.write(payload)

    envelope_size = Path(out_path).stat().st_size
    dur = len(raw) / SR
    kbps = envelope_size * 8 / dur / 1000

    print(f"input audio:        {dur:.2f} s mono @ {sr_in} Hz "
          f"({raw_bytes:>10,} B as int16 PCM)")
    print(f"WavTokenizer:       {q} codebook × {t} frames = {q*t} codes")
    print(f"WAI envelope:       {envelope_size:>10,} B  "
          f"({raw_bytes/envelope_size:6.1f}× compressed, "
          f"{kbps:5.1f} kbps)")
    print(f"model on the wire:           0  (WavTokenizer weights stay at the sink)")
    print(f"capability declared:{manifest['model_requirement']['capability']:>30}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--wavtokenizer-src", default="/tmp/wavtokenizer",
                   help="Path to the cloned WavTokenizer GitHub repo.")
    p.add_argument("--config", required=True, help="YAML model config")
    p.add_argument("--ckpt",   required=True, help=".ckpt checkpoint")
    a = p.parse_args()
    encode(a.input, a.output, a.wavtokenizer_src, a.config, a.ckpt)
