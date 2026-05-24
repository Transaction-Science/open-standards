"""WAI audio path — the modality where 'transmit minimum, regenerate at
sink' is the decisive, established 2026 win (neural audio codecs at a
few kbps vs Opus/MP3).

Source: waveform -> ambient neural codec ENCODER -> discrete code tokens
(the wire payload). Sink: same codec's DECODER (a requirement of
existence, never shipped) -> waveform. Tokens are the minimum; the
codec is the shared prior.

Uses the on-disk `encodec-32khz` model. Honest metrics: real bitrate
(kbps) and perceptual fidelity (log-mel L1), compared to libmp3lame /
libopus on the same clip at the same wall-clock.
"""

from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ENCODEC_DIR = "/Volumes/macos_4TB_external/neuronexus_ai/model_library/encodec-32khz"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
AUDIO_KIND = "encodec_tokens_v1"


@lru_cache(maxsize=1)
def _codec():
    from transformers import EncodecModel, AutoProcessor
    m = EncodecModel.from_pretrained(ENCODEC_DIR).to(DEVICE).eval()
    p = AutoProcessor.from_pretrained(ENCODEC_DIR)
    return m, p


@torch.no_grad()
def encode_tokens(wav: np.ndarray, sr: int):
    """wav: float32 mono [-1,1]. Returns (token_bytes, meta)."""
    m, p = _codec()
    msr = p.sampling_rate
    if sr != msr:
        import torchaudio
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav).float(), sr, msr).numpy()
    inp = p(raw_audio=wav, sampling_rate=msr, return_tensors="pt").to(DEVICE)
    enc = m.encode(inp["input_values"], inp.get("padding_mask"))
    codes = enc.audio_codes  # (1, n_q? , n_codebooks, T) or (1,1,Q,T)
    c = codes.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint16)  # (Q,T)
    q, t = c.shape
    n_samples = int(wav.shape[-1])
    blob = struct.pack("<III", q, t, n_samples) + c.tobytes()
    return blob, {"q": q, "t": t, "sr": msr, "n_samples": n_samples}


@torch.no_grad()
def decode_tokens(blob: bytes) -> tuple[np.ndarray, int]:
    m, p = _codec()
    q, t, n = struct.unpack_from("<III", blob, 0)
    c = np.frombuffer(blob[12:], np.uint16).reshape(q, t).astype(np.int64)
    codes = torch.from_numpy(c)[None, None].to(DEVICE)  # (1,1,Q,T)
    audio = m.decode(codes, [None])[0]
    wav = audio.squeeze().cpu().numpy().astype(np.float32)[:n]
    return wav, p.sampling_rate
