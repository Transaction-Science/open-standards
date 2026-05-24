#!/usr/bin/env python3
"""WAI audio: transmit neural-codec tokens, regenerate at the sink.
Honest comparison vs MP3/Opus on the SAME clip, same fidelity ruler."""
import os, subprocess, time, wave
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from wai.audio import encode_tokens, decode_tokens

SRC = "/Users/dcharlot/Movies/The Sonnet Man： Sonnet 18 Come and Be My Sunny Day [C-gG4MBdjvQ].mp4"
SECS = 8


def load_wav_from_video(path, secs):
    out = "/tmp/wai_src.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-t", str(secs),
                    "-ac", "1", "-ar", "32000", "-y", out], check=True)
    with wave.open(out, "rb") as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return (a.astype(np.float32) / 32768.0), sr, os.path.getsize(out)


def log_mel(x, sr):
    # cheap honest spectral fidelity proxy (no extra deps)
    n = 1024
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    frames = [np.abs(np.fft.rfft(x[i:i + n] * np.hanning(n)))
              for i in range(0, len(x) - n, n // 2)]
    S = np.log1p(np.stack(frames) + 1e-6)
    return S


def mel_l1(a, b, sr):
    Sa, Sb = log_mel(a, sr), log_mel(b, sr)
    m = min(len(Sa), len(Sb))
    return float(np.mean(np.abs(Sa[:m] - Sb[:m])))


def codec_size(path, args, secs):
    o = f"/tmp/wai_cmp{os.path.splitext(path)[1]}"
    subprocess.run(["ffmpeg", "-v", "error", "-i", "/tmp/wai_src.wav"]
                   + args + ["-y", o], check=True)
    sz = os.path.getsize(o)
    # decode back for fidelity
    d = "/tmp/wai_cmp_dec.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-i", o, "-ac", "1",
                    "-ar", "32000", "-y", d], check=True)
    with wave.open(d, "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return sz, a.astype(np.float32) / 32768.0


wav, sr, wav_bytes = load_wav_from_video(SRC, SECS)
dur = len(wav) / sr
print(f"source: {dur:.1f}s mono {sr} Hz  (PCM {wav_bytes:,} B = "
      f"{wav_bytes*8/dur/1000:.0f} kbps)")

t0 = time.time()
blob, meta = encode_tokens(wav, sr)
enc_ms = (time.time() - t0) * 1000
t1 = time.time()
rec, rsr = decode_tokens(blob)
dec_ms = (time.time() - t1) * 1000

wai_bytes = len(blob) + 64  # + tiny manifest allowance
wai_kbps = wai_bytes * 8 / dur / 1000
fid = mel_l1(wav[:len(rec)], rec[:len(wav)], sr)
print(f"\nWAI (encodec tokens): {wai_bytes:,} B  = {wai_kbps:5.1f} kbps"
      f"   logmel-L1 {fid:.4f}")
print(f"  sink decode: {dec_ms:6.1f} ms for {dur:.1f}s audio  "
      f"= {dur*1000/dec_ms:5.1f}× real-time  [REAL-TIME ✓]"
      if dec_ms < dur * 1000 else f"  sink decode {dec_ms:.0f} ms (slower than real-time)")

print("\nclassical, same clip, same logmel-L1 ruler:")
for name, args in [("mp3 32k", ["-c:a", "libmp3lame", "-b:a", "32k"]),
                   ("mp3 64k", ["-c:a", "libmp3lame", "-b:a", "64k"]),
                   ("opus 12k", ["-c:a", "libopus", "-b:a", "12k"]),
                   ("opus 24k", ["-c:a", "libopus", "-b:a", "24k"])]:
    try:
        sz, dec = codec_size("/tmp/x" + (".mp3" if "mp3" in name else ".opus"),
                              args, SECS)
        kbps = sz * 8 / dur / 1000
        f = mel_l1(wav[:len(dec)], dec[:len(wav)], sr)
        print(f"  {name:9s}: {sz:7,} B = {kbps:5.1f} kbps   logmel-L1 {f:.4f}")
    except Exception as e:
        print(f"  {name:9s}: ffmpeg failed ({e})")

print(f"\nverdict: WAI {wai_kbps:.1f} kbps @ logmel-L1 {fid:.3f} — compare the "
      f"kbps of the classical codec that matches that fidelity.")
