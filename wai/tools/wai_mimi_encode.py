"""WAI source-side neural encoder for `wai.neural.mimi`.

Mimi (Kyutai) at 24 kHz, 12.5 Hz frame rate, 8 codebooks of 2048.
Designed for real-time speech LLM use cases. Token rate is among the
lowest of any high-quality neural audio codec (~1.1 kbps), which makes
it the right pick when the wire-byte budget dominates.

Wire format (uniform with all wai.neural.<audio> capabilities):
  <III>  q  t  n_samples         (little-endian u32 × 3)
  q*t × u16 codes (row-major, LE)

Usage:
  python tools/wai_mimi_encode.py <input.wav> -o <output.wai>
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

MODEL_ID = "kyutai/mimi"
WAI_MAGIC = b"WAI1"
SR = 24_000


def encode(wav_path: str, out_path: str) -> None:
    with wave.open(wav_path, "rb") as w:
        sr_in = w.getframerate()
        raw_bytes = w.getnframes() * 2 * w.getnchannels()

    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", wav_path, "-ac", "1",
         "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    raw = np.frombuffer(pcm, np.float32)

    import torch
    from transformers import MimiModel, AutoFeatureExtractor
    model = MimiModel.from_pretrained(MODEL_ID).eval()
    proc  = AutoFeatureExtractor.from_pretrained(MODEL_ID)

    inp = proc(raw_audio=raw, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        enc = model.encode(inp["input_values"], inp.get("padding_mask"))
    # MimiEncoderOutput.audio_codes: [batch=1, n_codebooks, T]
    codes = enc.audio_codes.squeeze(0).numpy().astype(np.uint16)
    q, t = codes.shape

    payload = struct.pack("<III", q, t, len(raw)) + codes.tobytes()

    manifest = {
        "wai": "1.0",
        "media": "audio",
        "intent": "replicate",
        "model_requirement": {
            "capability": "wai.neural.mimi",
            "fallback": "wai.audio.opus",
        },
        "conditioning": {"kind": "mimi_tokens"},
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
    print(f"Mimi tokens:        {q} codebooks × {t} frames = {q*t} codes")
    print(f"WAI envelope:       {envelope_size:>10,} B  "
          f"({raw_bytes/envelope_size:6.1f}× compressed, "
          f"{kbps:5.1f} kbps)")
    print(f"model on the wire:           0  (Mimi weights stay at the sink)")
    print(f"capability declared:{manifest['model_requirement']['capability']:>30}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args()
    encode(a.input, a.output)
