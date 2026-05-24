#!/usr/bin/env python3
"""Honest operating-curve sweep — not a cherry-picked point.

Generative semantic communication has a real tradeoff: the more
"understanding" you transmit, the less the receiver must hallucinate, so
fidelity rises with package size. This measures that curve directly on
this machine with on-disk models: for several thumbnail richnesses ×
img2img strengths, report package bytes vs CLIP semantic similarity and
whether the 0.70 conformance bar is met. The point is the SHAPE of the
curve and where conformance is reached, reported truthfully — including
that the documented IP-Adapter path (not on disk) would shift the whole
curve left (same fidelity, fewer transmitted bytes).
"""
import os, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from PIL import Image
from wai.semantic import build_package, unpack_package, clip_embed, clip_sim
from wai.model import load_generator
from wai.runtime import _seed_image

IMG = "/Volumes/macos_4TB_external/neuronexus_ai/model_library/sdxl-turbo/output_tile.jpg"
src = Image.open(IMG).convert("RGB").resize((512, 512), Image.LANCZOS)
src_emb = clip_embed(src)
pipe = load_generator(img2img=True)

print(f"{'thumb_px':>8} {'strength':>8} {'pkg_B':>7} {'CLIPsim':>8} "
      f"{'vs-src':>7} {'conformant(>=0.70)':>18}")
for tpx in (48, 96, 160, 224):
    for stg in (0.25, 0.40, 0.55):
        blob = build_package(src, thumb_px=tpx, tag="photograph")
        pkg = unpack_package(blob)
        seed = _seed_image(pkg, 512, 512)
        best = -1.0
        import math
        steps = max(3, math.ceil(2 / stg))  # ensure >=2 effective steps
        for j in range(2):  # K=2, time-bounded
            import torch
            g = torch.Generator("cpu").manual_seed(j)
            im = pipe(prompt="a realistic photograph, sharp, detailed",
                      image=seed, strength=stg, num_inference_steps=steps,
                      guidance_scale=0.0, generator=g).images[0]
            best = max(best, clip_sim(clip_embed(im), pkg["clip"]))
        # also: raw similarity of just the transmitted thumbnail upscaled
        thumb_up = pkg["thumb"].resize((512, 512), Image.LANCZOS)
        vs_src = clip_sim(clip_embed(thumb_up), src_emb)
        print(f"{tpx:>8} {stg:>8.2f} {len(blob):>7,} {best:>8.4f} "
              f"{vs_src:>7.3f} {'YES' if best >= 0.70 else 'no':>18}")
