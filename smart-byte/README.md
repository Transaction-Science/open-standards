# Smart Byte

A carrier for value. Settlement-agnostic, energy-attributed, no off switch.

The smart byte is not money — it is a **vesicle** (the cellular-biology sense): a content-addressed, signed envelope carrying exactly five things — identity (the blake3 hash of its origin attestation), provenance (the issuer's signed birth certificate), an ownership chain (signature-bound transitions), cargo (whatever value it represents — USD, joules, compute-hours, votes, attestations, claims; the substrate is agnostic), and a two-part joule cost (measured + estimated, cumulative over its life). Consensus is lockstep deterministic simulation with BFT supermajority commit; the substrate scales by federating many bounded clusters, not by growing one. It is TCP/IP for value: an open, application-agnostic carrier that any matching, clearing, or trust-signaling layer can build on.

The substrate is intentionally **ownerless**: no protocol-level token, permissive licensing on the spec text and reference code, conformance test vectors so any implementer can verify canonicity without a trust relationship with the authors. The protocol's identity lives in its conformance vectors and content-addressing, not in any brand. Transaction Science is one steward — it publishes the spec, ships a reference implementation, and operates commercial services around the substrate — not its proprietor.

## Contents

- `spec/treatise_v1_parts_I_II_III_combined.md` — *Smart Byte Substrate: A Treatise* (Parts I–III: the foundation, the substrate architecture, the security engineering; Parts IV–VII previewed)
- `strategic_context.docx` — *The Strategic Context: Why the Substrate Makes Sense in 2026*
- `SECURITY.md` — the working security-engineering summary: it's a digital system on the internet, secured the way digital systems on the internet are secured (conservative cryptography, encrypted transports, reproducible multiply-reviewed builds, standard layer-2 key custody, standard availability engineering, textbook BFT consensus) plus a handful of new compositions of those proven parts. Plus the vulnerability-reporting process.

## Status

Treatise v1.0 (May 2026). The conformance specification (byte-level wire formats) and the reference implementation referenced in the treatise as "v0.3" are to be brought into this repository.

## Related

- EOC (`../eoc/`) — the energy-optimized AI compute substrate; the smart byte's energy attribution depends on an honest joules-per-task substrate.
- Settlement (`settlement.science`) — the flagship regulated financial institution running on the substrate as an issuer and consumer of smart bytes.
