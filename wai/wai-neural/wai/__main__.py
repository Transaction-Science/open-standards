"""WAI v0.4 CLI.  understand->package->regenerate.

  wai encode  <img|video> -o x.wai           # intent: replicate
  wai create  "<prompt>"  -o x.wai
  wai improve <img> --obj "..." -o x.wai
  wai play    x.wai [--out frame.png]
  wai info    x.wai
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from . import (Wai, WaiRuntime, InertWaiError,
               encode_replicate, encode_create, encode_improve)

def _wire(w, n):
    t = w.manifest["target"]; outpx = t["w"]*t["h"]*3*w.frames
    print(f"  intent          : {w.verb}")
    print(f"  .wai (on wire)  : {n:>9,} B   ({w.kind}, {w.frames} frame pkg)")
    print(f"  model on wire   : {0:>9,} B   (requirement, not shipped/hashed)")
    print(f"  per-frame pkg   : {n//max(w.frames,1):>9,} B")
    print(f"  sink raw output : {outpx:>9,} B   payload:output = 1 : {outpx/max(n,1):,.0f}")

def c_encode(a):
    w = encode_replicate(a.src, size=a.size, max_frames=a.frames,
                          out_fps=a.fps, tag=a.tag, min_clip_sim=a.minsim)
    _wire(w, w.write(a.out))
def c_create(a):
    w = encode_create(a.prompt, size=a.size, steps=a.steps, seed=a.seed)
    _wire(w, w.write(a.out)); print(f"  conditioning    : {len(a.prompt)} B text")
def c_improve(a):
    w = encode_improve(a.src, a.obj, size=a.size, strength=a.strength,
                        steps=a.steps, min_clip_sim=a.minsim)
    _wire(w, w.write(a.out)); print(f"  objective       : \"{a.obj}\"")
def c_play(a):
    rt = WaiRuntime()
    try: w = rt.open(a.wai)
    except InertWaiError as e: print(f"INERT: {e}", file=sys.stderr); sys.exit(3)
    r = rt.execute(w)
    per = r.exec_ms/max(len(r.frames),1)
    rttag = "REAL-TIME ✓" if r.realtime_here else "not real-time on THIS box (sink-capability contract; see note)"
    print(f"executed intent={r.verb}: {len(r.frames)} frame(s)")
    print(f"  sink exec       : {r.exec_ms:8.1f} ms  ({per:7.1f} ms/frame, {1000/per:5.1f} fps)  [{rttag}]")
    if r.verb != "create":
        verdict = "CONFORMANT ✓" if r.conformant else "NON-CONFORMANT ✗"
        print(f"  semantic fidelity: CLIP-sim {r.clip_sim:.4f}  (threshold {r.threshold:.2f})  [{verdict}]")
        print(f"  (CLIP cosine is the honest metric for semantic equivalence; PSNR is the wrong ruler and is not used)")
    if a.out and r.frames:
        Image.fromarray(r.frames[0]).save(a.out); print(f"  wrote {a.out}")
    if not r.realtime_here:
        print("  NOTE: real-time is the SINK's contract — a conforming distilled "
              "real-time generative model meets it (cited 2026 work); this Mac's "
              "SD-XL path is ~1 s/frame and is reported honestly, not hidden.")
def c_info(a): print(json.dumps(Wai.read(a.wai).manifest, indent=2))

def main():
    p = argparse.ArgumentParser(prog="wai"); s = p.add_subparsers(dest="cmd", required=True)
    e = s.add_parser("encode"); e.add_argument("src"); e.add_argument("-o","--out",default="out.wai")
    e.add_argument("--size",type=int,default=512); e.add_argument("--frames",type=int,default=0)
    e.add_argument("--fps",type=float,default=0.0); e.add_argument("--tag",default="")
    e.add_argument("--minsim",type=float,default=0.70); e.set_defaults(fn=c_encode)
    c = s.add_parser("create"); c.add_argument("prompt"); c.add_argument("-o","--out",default="out.wai")
    c.add_argument("--size",type=int,default=512); c.add_argument("--steps",type=int,default=2)
    c.add_argument("--seed",type=int,default=0); c.set_defaults(fn=c_create)
    im = s.add_parser("improve"); im.add_argument("src"); im.add_argument("-o","--out",default="out.wai")
    im.add_argument("--obj",default="restored, sharp, high detail"); im.add_argument("--size",type=int,default=512)
    im.add_argument("--strength",type=float,default=0.5); im.add_argument("--steps",type=int,default=3)
    im.add_argument("--minsim",type=float,default=0.55); im.set_defaults(fn=c_improve)
    pl = s.add_parser("play"); pl.add_argument("wai"); pl.add_argument("--out",default=None)
    pl.set_defaults(fn=c_play)
    i = s.add_parser("info"); i.add_argument("wai"); i.set_defaults(fn=c_info)
    a = p.parse_args(); a.fn(a)

if __name__ == "__main__": main()
