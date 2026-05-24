"""WAI source-side neural encoder for `wai.neural.encodec32`.

This is the first end-to-end demonstration of WAI's actual pitch: the
sink-installed model is the decoder. The encoder runs EnCodec on audio
and ships only the quantized tokens in a WAI envelope. The model
(~236 MB on disk) is NOT included — sinks resolve `wai.neural.encodec32`
against their own local EnCodec install.

Usage:
  python tools/wai_encodec_encode.py <input.wav> -o <output.wai>
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import wave
from pathlib import Path

import numpy as np

# Default location of the EnCodec-32kHz model on this machine. Sinks
# would have their own path / model registry.
ENCODEC_DIR = (
    "/Volumes/macos_4TB_external/neuronexus_ai/model_library/encodec-32khz"
)
WAI_MAGIC = b"WAI1"


def encode(wav_path: str, out_path: str) -> None:
    # 1. Load audio via ffmpeg → mono float32 @ 32 kHz (EnCodec-32 rate)
    import subprocess
    with wave.open(wav_path, "rb") as w:
        sr_in = w.getframerate()
        n_in = w.getnframes()
        raw_audio_bytes = n_in * 2 * w.getnchannels()   # int16 PCM size
    sr = 32_000
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", wav_path, "-ac", "1",
         "-ar", str(sr), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    raw = np.frombuffer(pcm, np.float32)
    import torch

    # 3. Load the EnCodec model FROM LOCAL DISK — this is the
    #    encoder-side install. The bytes never enter the WAI envelope.
    from transformers import AutoProcessor, EncodecModel
    model = EncodecModel.from_pretrained(ENCODEC_DIR).eval()
    proc = AutoProcessor.from_pretrained(ENCODEC_DIR)
    model_disk_size = sum(p.stat().st_size for p in Path(ENCODEC_DIR).rglob("*")
                          if p.is_file())

    # 4. Run encode → quantized integer codes per residual VQ codebook
    inp = proc(raw_audio=raw, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        enc = model.encode(inp["input_values"], inp.get("padding_mask"))
    codes = enc.audio_codes.squeeze(0).squeeze(0).numpy().astype(np.uint16)
    q, t = codes.shape                      # n_codebooks × n_frames

    # 5. Pack payload: <III> q t n_samples | u16 codes (LE) row-major
    payload = struct.pack("<III", q, t, len(raw)) + codes.tobytes()

    # 6. Build WAI envelope
    manifest = {
        "wai": "1.0",
        "media": "audio",
        "intent": "replicate",
        "model_requirement": {
            "capability": "wai.neural.encodec32",
            "fallback": "wai.audio.opus",   # if the sink lacks EnCodec it
                                            # MAY fall through to Opus (a
                                            # separate Opus-encoded payload
                                            # would be needed for v1; the
                                            # capability declaration is the
                                            # forward-looking part)
        },
        "conditioning": {"kind": "encodec_tokens"},
        "target": {"sr": sr, "n_samples": int(len(raw)),
                   "codebooks": int(q), "frames": int(t)},
    }
    mb = json.dumps(manifest, separators=(",", ":")).encode()
    with open(out_path, "wb") as f:
        f.write(WAI_MAGIC)
        f.write(struct.pack("<I", len(mb))); f.write(mb)
        f.write(struct.pack("<I", len(payload))); f.write(payload)

    envelope_size = Path(out_path).stat().st_size
    dur = len(raw) / sr
    kbps = envelope_size * 8 / dur / 1000

    print(f"input audio:        {dur:.2f} s mono @ {sr_in} Hz "
          f"({raw_audio_bytes:>10,} B as int16 PCM)")
    print(f"EnCodec tokens:     {q} codebooks × {t} frames = {q*t} codes")
    print(f"WAI envelope:       {envelope_size:>10,} B  "
          f"({raw_audio_bytes/envelope_size:6.1f}× compressed, "
          f"{kbps:5.1f} kbps)")
    print(f"model on the wire:  {'0':>10} B  "
          f"(EnCodec weights {model_disk_size/1024/1024:.0f} MB stay at the sink)")
    print(f"capability declared:{manifest['model_requirement']['capability']:>30}")
    print(f"fallback declared:  {str(manifest['model_requirement']['fallback']):>30}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args()
    if not Path(ENCODEC_DIR).exists():
        sys.exit(f"EnCodec model not found at {ENCODEC_DIR}; install it locally first")
    encode(a.input, a.output)
