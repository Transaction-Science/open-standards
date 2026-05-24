"""Source side = UNDERSTAND: source -> compact semantic package(s).

No model-private latent, no entropy-coded VAE blob. Just the
model-agnostic understanding the receiver regenerates from.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .container import Wai
from .model import requirement
from .semantic import PKG_KIND, build_package


def _probe_fps(src: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "0", "-of", "json", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", src],
        capture_output=True, text=True).stdout
    try:
        n, d = json.loads(out)["streams"][0]["r_frame_rate"].split("/")
        return float(n) / float(d)
    except Exception:
        return 30.0


def _frames(src, size, maxf, fps):
    vf = (f"scale={size}:{size}:force_original_aspect_ratio=disable,"
          f"format=rgb24,fps={fps}")
    a = ["ffmpeg", "-v", "error", "-i", src, "-vf", vf]
    if maxf:
        a += ["-frames:v", str(maxf)]
    a += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(a, capture_output=True).stdout
    n = len(raw) // (size * size * 3)
    return np.frombuffer(raw[:n*size*size*3], np.uint8).reshape(n, size, size, 3)


def encode_replicate(src: str, size: int = 512, max_frames: int = 0,
                     out_fps: float = 0.0, tag: str = "",
                     min_clip_sim: float = 0.70) -> Wai:
    p = Path(src)
    vid = p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")
    if vid:
        fps = out_fps or _probe_fps(src)
        fr = _frames(src, size, max_frames, fps)
        imgs = [Image.fromarray(f) for f in fr]
    else:
        imgs = [Image.open(src).convert("RGB").resize((size, size), Image.LANCZOS)]
        fps = 0.0
    pkgs = [build_package(im, tag=tag) for im in imgs]
    manifest = {
        "wai_version": "0.4",
        "model_requirement": requirement(min_clip_sim),
        "intent": {"verb": "replicate", "params": {}},
        "target": {"w": size, "h": size, "channels": 3,
                   "fps": float(fps), "frames": len(pkgs)},
        "conditioning": {"kind": PKG_KIND},
    }
    return Wai(manifest, Wai.pack_frames(pkgs))


def encode_create(prompt: str, size: int = 512, steps: int = 2,
                  seed: int = 0, min_clip_sim: float = 0.0) -> Wai:
    manifest = {
        "wai_version": "0.4",
        "model_requirement": requirement(min_clip_sim),
        "intent": {"verb": "create", "params": {"steps": steps, "seed": seed}},
        "target": {"w": size, "h": size, "channels": 3, "fps": 0.0, "frames": 1},
        "conditioning": {"kind": "text_prompt"},
    }
    return Wai(manifest, Wai.pack_frames([prompt.encode("utf-8")]))


def encode_improve(src_img: str, objective: str, size: int = 512,
                   strength: float = 0.5, steps: int = 3, seed: int = 0,
                   min_clip_sim: float = 0.55) -> Wai:
    im = Image.open(src_img).convert("RGB").resize((size, size), Image.LANCZOS)
    pkg = build_package(im, tag=objective)
    manifest = {
        "wai_version": "0.4",
        "model_requirement": requirement(min_clip_sim),
        "intent": {"verb": "improve",
                   "params": {"objective": objective, "strength": strength,
                              "steps": steps, "seed": seed}},
        "target": {"w": size, "h": size, "channels": 3, "fps": 0.0, "frames": 1},
        "conditioning": {"kind": PKG_KIND},
    }
    return Wai(manifest, Wai.pack_frames([pkg]))
