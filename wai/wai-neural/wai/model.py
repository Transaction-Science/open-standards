"""Capability, not a fixed model.

A .wai declares a *requirement of existence*: "a model satisfying
capability C must exist at the sink." It does NOT pin weights by hash.
Any conforming model serves the file — so each sink uses whatever
conforming model is real-time on *its* hardware (the .mp3/.h264
contract: bitstream fixed, decoder implementation free, decoder never
shipped). Conformance is behavioral, verified at execute time by CLIP
semantic similarity ≥ the file's declared threshold.

This sink advertises one capability, served by an ambient SD-XL-Turbo
generator + on-disk CLIP. A 2026 device would serve the same capability
with its own distilled real-time generative model; the file is
unchanged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Capability profiles this sink can serve (behavioral contract names).
SINK_CAPABILITIES = {"wai.semantic.regen.v1"}

_SDXL_DIR = "/Volumes/macos_4TB_external/neuronexus_ai/model_library/sdxl-turbo"


def requirement(min_clip_sim: float = 0.70) -> dict:
    return {"capability": "wai.semantic.regen.v1",
            "min_clip_sim": float(min_clip_sim)}


def satisfiable(req: dict) -> bool:
    """Requirement of existence: do I have *a* conforming model?"""
    return req.get("capability") in SINK_CAPABILITIES and Path(_SDXL_DIR).exists()


@lru_cache(maxsize=2)
def load_generator(img2img: bool = True):
    from diffusers import (AutoPipelineForImage2Image,
                           AutoPipelineForText2Image)
    cls = AutoPipelineForImage2Image if img2img else AutoPipelineForText2Image
    pipe = cls.from_pretrained(_SDXL_DIR, torch_dtype=torch.float16,
                               variant="fp16", use_safetensors=True).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe
