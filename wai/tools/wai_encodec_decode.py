"""WAI sink-side neural decoder for `wai.neural.encodec32`.

This is the receiving half of the demonstration: a sink that has the
EnCodec model installed locally resolves the `wai.neural.encodec32`
capability by loading its own model and reconstructing audio from the
shipped tokens. The WAI envelope did NOT carry any model bytes.

If the sink doesn't have EnCodec installed, the file is inert here
(unless the fallback capability is also wired up — out of scope for
this single-capability demo).

Usage:
  python tools/wai_encodec_decode.py <input.wai> -o <output.wav>
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import wave
from pathlib import Path

import numpy as np

ENCODEC_DIR = (
    "/Volumes/macos_4TB_external/neuronexus_ai/model_library/encodec-32khz"
)


def decode(wai_path: str, out_path: str) -> None:
    raw = Path(wai_path).read_bytes()
    if raw[:4] != b"WAI1":
        sys.exit("not a WAI v1 file (magic mismatch)")
    o = 4
    (ml,) = struct.unpack_from("<I", raw, o); o += 4
    manifest = json.loads(raw[o:o + ml]); o += ml
    (pl,) = struct.unpack_from("<I", raw, o); o += 4
    payload = raw[o:o + pl]

    cap = manifest["model_requirement"]["capability"]
    fb = manifest["model_requirement"].get("fallback")
    kind = manifest["conditioning"]["kind"]

    print(f"WAI envelope:       {len(raw):,} B")
    print(f"  capability:       {cap}")
    print(f"  fallback:         {fb}")
    print(f"  conditioning.kind:{kind}")

    if cap != "wai.neural.encodec32":
        sys.exit(f"this decoder only resolves wai.neural.encodec32; got {cap}")

    # Sink-side capability resolution: do we have EnCodec installed?
    if not Path(ENCODEC_DIR).exists():
        print(f"  resolution:       MISSING — EnCodec model not at {ENCODEC_DIR}")
        if fb is None:
            sys.exit("  → file is INERT at this sink (no fallback declared)")
        sys.exit(f"  → would fall back to {fb} (not implemented in this demo)")

    print(f"  resolution:       ✓ EnCodec installed locally at sink")

    # Decode via locally-installed EnCodec
    import torch
    from transformers import EncodecModel
    model = EncodecModel.from_pretrained(ENCODEC_DIR).eval()
    q, t, n = struct.unpack_from("<III", payload, 0)
    codes = (np.frombuffer(payload[12:], np.uint16).reshape(q, t)
             .astype(np.int64))
    with torch.no_grad():
        audio = model.decode(torch.from_numpy(codes)[None, None], [None])[0]
    samples = audio.squeeze().numpy()[:n]
    sr = manifest["target"]["sr"]

    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((samples.clip(-1, 1) * 32767).astype(np.int16).tobytes())

    dur = len(samples) / sr
    raw_bytes = len(samples) * 2
    print(f"  reconstructed:    {dur:.2f} s mono @ {sr} Hz "
          f"({raw_bytes:,} B as int16 PCM) → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args()
    decode(a.input, a.output)
