"""The semantic package — the wire payload of WAI v0.4.

Generative semantic communication (the documented May-2026 paradigm):
do NOT send the content or a model-private latent. Send the *understanding*
— a compact, model-agnostic semantic representation — and let the sink's
own conforming generative model regenerate something semantically
equivalent.

A package is three model-agnostic parts, none of them a private latent:
  • clip   — a CLIP ViT-L/14 image embedding, int8 (768 B). The canonical
             cross-model semantic vector; this is the "understanding".
  • thumb  — a tiny JPEG (global colour / composition).
  • edges  — a 1-bit Canny structure map, PNG (geometry / layout).
Optionally a short text tag.

CLIP is also the honest fidelity metric: semantic equivalence is cosine
similarity in CLIP space, not PSNR. PSNR is the wrong ruler here and is
not used for the conformance gate.
"""

from __future__ import annotations

import io
import struct
from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image

CLIP_DIR = "/Volumes/macos_4TB_external/neuronexus_ai/model_library/openai_clip-vit-large-patch14"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PKG_KIND = "semantic_package_v1"


@lru_cache(maxsize=1)
def _clip():
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained(CLIP_DIR).to(DEVICE).eval()
    p = CLIPProcessor.from_pretrained(CLIP_DIR)
    return m, p


@torch.no_grad()
def clip_embed(img: Image.Image) -> np.ndarray:
    """Canonical CLIP image embedding (projection-dim, unit-norm).
    Computed explicitly via vision pooler + visual_projection so it is
    robust across transformers versions (get_image_features has returned
    token states on some 5.x builds)."""
    m, p = _clip()
    inp = p(images=img, return_tensors="pt").to(DEVICE)
    v = m.vision_model(pixel_values=inp["pixel_values"])
    pooled = v.pooler_output                       # (1, hidden)
    f = m.visual_projection(pooled)[0]             # (proj_dim,)
    f = (f / f.norm()).float().cpu().numpy()
    return f


def clip_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))


def build_package(img: Image.Image, thumb_px: int = 48,
                  tag: str = "") -> bytes:
    """Pack {clip int8 | thumb jpeg | edges png | tag} length-prefixed."""
    emb = clip_embed(img)
    scale = float(np.abs(emb).max()) or 1.0
    clip_q = np.clip(np.round(emb / scale * 127), -127, 127).astype(np.int8)

    th = img.resize((thumb_px, thumb_px), Image.LANCZOS)
    tb = io.BytesIO(); th.save(tb, "JPEG", quality=80)

    g = cv2.cvtColor(np.asarray(img.resize((128, 128))), cv2.COLOR_RGB2GRAY)
    e = (cv2.Canny(g, 80, 180) > 0).astype(np.uint8) * 255
    eb = io.BytesIO()
    Image.fromarray(e).convert("1").save(eb, "PNG", optimize=True)

    tagb = tag.encode("utf-8")
    parts = [struct.pack("<f", scale) + clip_q.tobytes(),
             tb.getvalue(), eb.getvalue(), tagb]
    out = bytearray()
    for p in parts:
        out += struct.pack("<I", len(p)) + p
    return bytes(out)


def unpack_package(blob: bytes):
    o = 0
    parts = []
    for _ in range(4):
        (n,) = struct.unpack_from("<I", blob, o); o += 4
        parts.append(blob[o:o + n]); o += n
    cl, tb, eb, tagb = parts
    scale = struct.unpack_from("<f", cl, 0)[0]
    clip_q = np.frombuffer(cl[4:], np.int8).astype(np.float32) * scale / 127.0
    thumb = Image.open(io.BytesIO(tb)).convert("RGB")
    edges = Image.open(io.BytesIO(eb)).convert("L")
    return {"clip": clip_q, "thumb": thumb, "edges": edges,
            "tag": tagb.decode("utf-8")}
