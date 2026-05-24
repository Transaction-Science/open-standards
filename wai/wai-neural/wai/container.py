"""WAI v0.4 — generative semantic communication container.

Paradigm (documented May-2026 SOTA): NOT encode-before-decode. The wire
carries a compact *semantic package* — the understanding of the source —
and the sink's own conforming generative model regenerates something
semantically equivalent. The model is never transported, and it is a
*requirement of existence*, not a fixed artifact: any model satisfying
the declared capability profile may serve the file. Conformance is
behavioral (CLIP semantic similarity ≥ a declared threshold), not a
byte/weights hash.

manifest (v0.4):
{
  "wai_version":"0.4",
  "model_requirement":{"capability":str,"min_clip_sim":float},
  "intent":{"verb":str,"params":{...}},
  "target":{"w":int,"h":int,"channels":3,"fps":float,"frames":int},
  "conditioning":{"kind":"semantic_package_v1"|"text_prompt"}
}
payload: per-frame semantic packages (length-prefixed) or utf-8 prompt.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"WAI4"
WAI_VERSION = "0.4"
VERBS = ("replicate", "create", "improve", "innovate", "understand")


@dataclass
class Wai:
    manifest: dict
    payload: bytes

    @property
    def verb(self) -> str:
        return self.manifest["intent"]["verb"]

    @property
    def kind(self) -> str:
        return self.manifest["conditioning"]["kind"]

    @property
    def frames(self) -> int:
        return int(self.manifest["target"].get("frames", 1))

    @property
    def fps(self) -> float:
        return float(self.manifest["target"].get("fps", 0.0))

    def packages(self) -> list[bytes]:
        """Split the payload into per-frame semantic packages."""
        out, o = [], 0
        for _ in range(self.frames):
            (n,) = struct.unpack_from("<I", self.payload, o); o += 4
            out.append(self.payload[o:o + n]); o += n
        return out

    @staticmethod
    def pack_frames(pkgs: list[bytes]) -> bytes:
        b = bytearray()
        for p in pkgs:
            b += struct.pack("<I", len(p)) + p
        return bytes(b)

    def write(self, path) -> int:
        mb = json.dumps(self.manifest, separators=(",", ":")).encode()
        with open(path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", len(mb))); f.write(mb)
            f.write(struct.pack("<I", len(self.payload))); f.write(self.payload)
        return Path(path).stat().st_size

    @staticmethod
    def read(path) -> "Wai":
        raw = Path(path).read_bytes()
        if raw[:4] != MAGIC:
            raise ValueError("not a WAI v0.4 file")
        o = 4
        (mlen,) = struct.unpack_from("<I", raw, o); o += 4
        manifest = json.loads(raw[o:o + mlen]); o += mlen
        (plen,) = struct.unpack_from("<I", raw, o); o += 4
        return Wai(manifest, raw[o:o + plen])
