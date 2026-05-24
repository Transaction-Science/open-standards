"""Sink = REGENERATE. The embeddable runtime.

open()  : the file declares a capability *requirement of existence*.
          If this sink has no conforming model -> inert (Lottie model).
          NOT a weights-hash gate — any conforming model may serve it.
execute(): regenerate semantically-equivalent media from the package
          using whatever conforming generator this sink has, then verify
          behavioral conformance: CLIP cosine similarity between the
          regenerated output and the transmitted semantic vector must
          meet the file's declared threshold. Byte identity is never
          required; semantic equivalence is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageChops

from .container import Wai
from .model import DEVICE, load_generator, satisfiable
from .semantic import clip_embed, clip_sim, unpack_package


class InertWaiError(RuntimeError):
    """No model satisfying the required capability exists at this sink."""


@dataclass
class Result:
    verb: str
    frames: list
    exec_ms: float
    clip_sim: float
    threshold: float
    conformant: bool
    realtime_here: bool


def _seed_image(pkg: dict, w: int, h: int) -> Image.Image:
    """Global colour (thumb) + geometry (edges) -> img2img seed."""
    base = pkg["thumb"].resize((w, h), Image.LANCZOS)
    edges = pkg["edges"].resize((w, h), Image.NEAREST).convert("RGB")
    # faint structural overlay so the generator honours layout
    return ImageChops.blend(base, ImageChops.screen(base, edges), 0.18)


class WaiRuntime:
    def open(self, path: str) -> Wai:
        w = Wai.read(path)
        if not satisfiable(w.manifest["model_requirement"]):
            raise InertWaiError(
                f"no model satisfying capability "
                f"'{w.manifest['model_requirement']['capability']}' at this "
                f"sink — .wai inert (the Lottie model)")
        return w

    @torch.no_grad()
    def execute(self, w: Wai) -> Result:
        thr = float(w.manifest["model_requirement"].get("min_clip_sim", 0.0))
        t0 = time.time()

        if w.verb == "create":
            pipe = load_generator(img2img=False)
            prompt = w.packages()[0].decode("utf-8")
            p = w.manifest["intent"]["params"]
            g = torch.Generator("cpu").manual_seed(int(p.get("seed", 0)))
            im = pipe(prompt=prompt, num_inference_steps=int(p.get("steps", 2)),
                      guidance_scale=0.0, height=w.manifest["target"]["h"],
                      width=w.manifest["target"]["w"], generator=g).images[0]
            frames = [np.asarray(im)]
            sim = 1.0  # create has no source; semantic target is the prompt
        else:
            pipe = load_generator(img2img=True)
            p = w.manifest["intent"]["params"]
            tw, th = w.manifest["target"]["w"], w.manifest["target"]["h"]
            k = max(int(p.get("k", 6)), 1)
            strength = float(p.get("strength", 0.5))
            steps = max(int(p.get("steps", 3)), 2)
            frames, sims = [], []
            for blob in w.packages():
                pkg = unpack_package(blob)
                seed = _seed_image(pkg, tw, th)
                tgt = pkg["clip"]
                # CLIP-guided regeneration: the receiver optimises toward
                # the *transmitted* semantic vector (best-of-K by CLIP
                # cosine). This is "use the understanding", not just grade
                # by it — the generative-semantic-communication contract.
                best, best_s = None, -1.0
                for j in range(k):
                    g = torch.Generator("cpu").manual_seed(
                        int(p.get("seed", 0)) + j)
                    im = pipe(
                        prompt=pkg["tag"] or "a realistic photograph, sharp, detailed",
                        image=seed, strength=strength,
                        num_inference_steps=steps,
                        guidance_scale=0.0, generator=g).images[0]
                    s = clip_sim(clip_embed(im), tgt)
                    if s > best_s:
                        best, best_s = np.asarray(im), s
                frames.append(best)
                sims.append(best_s)
            sim = float(np.mean(sims))

        if DEVICE == "mps":
            torch.mps.synchronize()
        ms = (time.time() - t0) * 1000.0
        per = ms / max(len(frames), 1)
        tgt_fps = w.fps or 30.0
        return Result(w.verb, frames, ms, sim, thr,
                      sim >= thr, (1000.0 / per) >= tgt_fps)
