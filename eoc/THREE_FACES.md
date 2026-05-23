# One theorem, three faces

> *You are looking at **face 2 of 3** — the AI-compute face. The other two are linked at the bottom.*

The substrate that Transaction Science publishes appears in three distinct repositories carrying three different kinds of cargo. They look like three different specifications. They are three faces of the same theorem.

## The theorem

A substrate that lets any party perform a cryptographically-authorized operation on a content-addressed data structure, have the operation deterministically replicated across a federated set of independent verifiers, and have the energy cost of the operation honestly reported in microjoules, is sufficient for the operational core of any institutional-middle function that matches, clears, or trust-signals — across any cargo type.

The substrate is built from five commitments, identical across faces:

1. **Content-addressed identity.** Every load-bearing object is identified by the BLAKE3 hash of its canonical serialization. Identity is independently computable by any party; no central authority assigns it.
2. **Signed transitions.** Every state change is authorized by an ed25519 (v1) signature whose preimage explicitly binds the prior state hash, so signatures are scoped to exactly one transition on exactly one object and cannot be replayed. The schema carries a one-byte algorithm identifier so a post-quantum successor can be introduced incrementally without a flag-day migration.
3. **Federated bounded-membership consensus.** Where consensus is required, it runs as lockstep deterministic simulation among a known, signed-membership cluster with Byzantine-fault-tolerant supermajority commit. The substrate scales by federating many bounded clusters, not by growing one. Public verifiability for non-members is provided as an escape hatch — re-derivation when inputs are public, zero-knowledge proofs when they are not.
4. **Energy attribution.** Every operation reports its energy cost in microjoules — a two-part `measured + estimated` record drawn from hardware energy counters and a calibrated estimator. Joule cost is the substrate's only built-in valuation primitive; it is not the market price, but it is the falsifiable cost basis.
5. **Conformance-defined canonicity; commercial layer at the edges.** Canonicity lives in published conformance test vectors against which any implementer can verify, not in any brand or company. The protocol carries no token. The substrate's stewards operate commercial services around the substrate — managed nodes, registries, authoring tools, certification — and never inside it. No steward is acquired by or made exclusive to any single dominant AI lab or capital bloc.

The five commitments are not a style preference; they are the smallest set of architectural decisions that simultaneously delivers cryptographic identity, falsifiable cost reporting, and bounded-membership consensus efficient enough (~1 mJ per transfer) to compete with the institutional middle's rent layer. Removing any commitment breaks at least one of those three properties. Adding any sixth produces a worse design than the five-commitment version composed with application-layer protocols.

## The three faces

The substrate carries opaque cargo; what the cargo means is application-layer concern. Three faces are implemented as independent specifications because three classes of cargo justify productizing separately.

| Face | Repository | Cargo class | What the face implements |
|---|---|---|---|
| **Smart Byte** | `byte.transaction.science` | **Value** — USD, joules, currencies, commodities, attestations, claims, votes, sensor readings, conditional claims, anything-economic | A signed envelope whose payload is application-typed cargo; ownership chain of signature-bound transitions; cluster lockstep at roughly one millijoule per transfer measured |
| **EOC** | `eoc.transaction.science` | **AI compute** — tasks in the state-construct → retrieve → refine → check pipeline; results | A four-stage pipeline operator family with neural generation as a final fallback; energy-attributed work, federated nodes, ownerless registry |
| **Ambient Mobile** (CommunicationOS) | `communicationos-web` | **Signal** — anything else as transport: Smart Byte envelopes, EOC tasks, MLS state, FROST messages, MoQT media tracks, FHIR bundles | Heterogeneous radio paths (Tier 0 ambient wideband, Tier 1 unlicensed, Tier 2 counterparty), per-packet scheduler, AI-codec endpoints |

A Smart Byte transitioning from holder A to holder B passes through Ambient Mobile whenever either party is on a mobile endpoint and reaches the other via radio. An EOC task submitted from a mobile endpoint and answered by a hosted node passes through Ambient Mobile in both directions. The three faces compose by carrying each other as cargo.

## What is identical across faces

- The five commitments above.
- The negative-requirements posture: **no protocol-level token, no infrastructure-ahead-of-users, no single-vendor capture, no carrier-offload-partnership pretense as a primary go-to-market**.
- The stewardship posture: published spec text is permissively licensed (CC-BY-4.0 or equivalent), reference implementations are Apache-2.0, the registry of conformance tests is the source of canonicity, and Transaction Science operates services around the substrate without owning the protocol.
- The honest readiness discipline: every part of each spec is classified Solid / Engineering / Research / Theory, and engineering plans respect the classification when sequencing work.

## What varies across faces

The cargo class. The wire transport (bytes-on-network, energy-attributed-task-graph, radio-paths-with-scheduler). The conformance tests (what a conformant implementation must reproduce). Everything load-bearing in the substrate's architecture stays constant.

## Why the fractal

The institutional middle's three functions — matching, clearing, trust-signaling — appear at every layer of the stack at which value, compute, or signal moves. The maturations of cheap cryptography, ubiquitous deterministic computation, and per-operation energy measurement (the foundational technical case in the *Smart Byte Substrate: A Treatise* §4) apply to all three layers identically because the maturations are not value-specific, compute-specific, or signal-specific. They are commodity-computation properties. Once they are commodity, the substrate composition that uses them is the same architecture wherever it appears.

The three faces are therefore not a marketing taxonomy. They are an engineering prediction: if the substrate composition is correct, the same five commitments produce a viable layer at every cargo class for which the institutional middle currently extracts rent. Additional faces will be carved out as the engineering work matures — an attestation-only face for verifiable credentials, a knowledge-graph face for InformationOS provenance, a media-composition face for Joule Compose's federated basis registry. Each new face is the same theorem with a different cargo class — not a new architecture.

## The three faces in one picture

```
                       The substrate (five commitments)
                                       │
    ┌──────────────────────────────────┼──────────────────────────────────┐
    │                                  │                                  │
Smart Byte                            EOC                          Ambient Mobile
(byte.transaction.science)   (eoc.transaction.science)            (CommunicationOS)
    │                                  │                                  │
   Value                          AI compute                            Signal
    │                                  │                                  │
USD · joules · attestations    state-construct → retrieve →        any cargo over any
claims · votes · sensors           refine → check                 radio path between
                                                                  AI-equipped endpoints

         (each face carries the others as cargo when needed)
```

## Cross-references

- Smart Byte: `byte.transaction.science` / [`../smart-byte/`](../smart-byte/) — *Smart Byte Substrate: A Treatise* in `smart-byte/spec/`.
- EOC: `eoc.transaction.science` / [`../eoc/`](./) — the four-stage pipeline and the wire protocol in [`spec/`](./spec/).
- Ambient Mobile (CommunicationOS): [`../communicationos-web/`](../communicationos-web/) — *Ambient Mobile Technical Specification*.
