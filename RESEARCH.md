# State of the Art — Transaction Science Open Standards

*Dated 2026-05-23*

This document captures the competitive landscape and SOTA features that informed the v0.1.0 design and the planned enhancement roadmap for the three protocols in this repository.

It is dated and a snapshot; future versions will live alongside this one in `docs/research/`. Specific enhancement work is tracked in [GitHub Issues](https://github.com/Transaction-Science/open-standards/issues) labelled by protocol.

The aim is honest accounting of (a) what already exists in each niche, (b) where the v0.1.0 spec or implementation is weak relative to the field, and (c) where the differentiated lane is genuinely open.

---

## OpenPay

### Adjacent open-source projects

| Project | Niche | Licence / Stack | Adoption signal |
|---|---|---|---|
| **Hyperswitch** (Juspay) | Payment orchestrator across 100+ PSPs with routing, vaulting, retries, recovery | Apache-2.0 / Rust | ~40k stars; OpenPay uses it as the card-rail driver |
| **moov-io** | ISO 20022 / FedNow / ACH / wire message libraries | Apache-2.0 / Go | Closest analog to OpenPay's direct-rail thesis; ledger-light |
| **Open Payments** (Interledger Foundation) | Wallet-address-as-URL, GNAP auth, RFC 9421 message signatures | Apache-2.0 / TS+PHP+Java | Solves the addressing/grant problem OpenPay does not; different layer |
| **GNU Taler** (v1.4 Feb 2026) | Privacy-preserving Chaumian e-cash | AGPL/LGPL/GPL split / C+TS | EU/NLnet funded; not card/A2A |
| **Kill Bill** | Subscription billing | Apache-2.0 / Java | Enterprise; not rails-native |
| **BTCPay Server** | Self-hosted crypto acceptance | MIT / .NET | Single-rail |
| **Lago / OpenMeter / UniBee / Flexprice** | Billing / metering | Various | Different layer (delegate collection to PSPs) |

### Commercial baseline

The structural cost OpenPay is positioned against is the per-transaction fee model used by hosted payment SaaS:

- Stripe: 2.9% + $0.30 online; 2.7% + $0.05 in-person; +1.5% international; +1% FX
- Adyen: Interchange++, ~0.60% + €0.10 acquirer markup, €120–€300/mo platform fee
- Square: 2.6% + $0.10 in-person; 2.9% + $0.30 online

On a $10B/yr merchant the Stripe-vs-Adyen delta alone is ~$200M/yr — the wedge for owning the stack.

### What v0.1.0 is honestly missing

Most of these are 30–180 day items, not architectural gaps:

- Webhook delivery SLA, retry policy, replay-protection semantics
- Dispute auto-evidence packager (Stripe Radar / Adyen RevenueProtect equivalent)
- Network token provisioning (Visa VTS / Mastercard MDES) — still PAN-bound
- MCC-aware / least-cost routing
- Sub-merchant onboarding / KYB / Connect-equivalent payouts
- Multi-currency settlement and FX orchestration
- PCI-DSS readiness materials and zero-scope reference deployment
- OpenTelemetry per `Payment<S>` transition

### What v0.1.0 is genuinely ahead on

Five things, where no published open-source stack combines them all:

1. **`Payment<S>` typestate enforced by the Rust type system** — capture-before-auth is a compile error, not a runtime check.
2. **Bi-temporal append-only ledger as first-class** — `as_of(valid, transaction)` time-travel integrated with the rail layer.
3. **One Rust core compiling to iOS, Android, WASM, Linux** — `Money` and `Payment<S>` semantics bit-identical across platforms.
4. **A2A direct via ISO 20022 to FedNow / PIX / SEPA Instant, no PSP intermediary.**
5. **Card + A2A + stablecoin behind one orchestrator with one ledger.**

### Planned enhancements

Tracked in issues [#1](https://github.com/Transaction-Science/open-standards/issues/1), [#2](https://github.com/Transaction-Science/open-standards/issues/2), [#3](https://github.com/Transaction-Science/open-standards/issues/3) — labelled `openpay`.

---

## Smart Byte

### Adjacent conceptual projects

| Project | What it does | Licence | Adoption signal |
|---|---|---|---|
| **KERI + ACDC** (Trust-Over-IP / IETF `draft-ssmith-acdc`) | Self-certifying identifiers with rigorous key-event/rotation history; ACDCs chain like X.509 with property-graph semantics | Apache-2.0 | Production at GLEIF vLEI (verifiable Legal Entity Identifiers), Provenant, Veridian |
| **W3C Verifiable Credentials / DIDs** | Data-model for verifiable claims; transport-agnostic | W3C / open | EU eIDAS 2.0, EUDI Wallet — largest deployed footprint |
| **Interledger (ILP) + STREAM** | Packet-switched value routing across ledgers | Apache-2.0 | Interledger Foundation; Rafiki integration path; 9-year history; niche outside Web Monetization |
| **Bluesky / AT Protocol** | Content-addressed records, DID-anchored repos, working federation | MIT | 30M+ users — strongest live federated CAS+identity |
| **Iroh** (n0) | QUIC-holepunching transport, BLAKE3 verified streaming, blobs/docs | Apache-2.0 / MIT | Used as Holochain's transport; eating IPFS's lunch in Rust P2P |
| **Holochain** | Agent-centric source chains (per-agent hash chains) | CAL-1.0 | Niche; v0.7 dev; small committed community |
| **Nostr** | Signed-event simplicity; relay gossip | Public domain | Millions of pubkeys; weak provenance/revocation |
| **Hypercore / Pears** | Append-only signed feeds, hyperswarm DHT | Apache-2.0 | Mature but specialised |
| **Spritely Goblins** | OCAP-secure distributed objects; v0.15 compiles to WASM | Apache-2.0 | Pre-production; intellectually influential |
| **Veilid** (cDc) | Tor+IPFS-like privacy-preserving overlay | MPL-2.0 | Privacy-first niche |
| **IPFS / libp2p** | Content-addressing (CID), DHT | MIT / Apache | Vast, adoption plateau |

KERI/ACDC is the closest structural cousin. Bluesky / AT Protocol is the largest live federated content-addressed deployment in any niche.

### Prior "universal value carrier" efforts

The Interledger project has explicitly claimed "TCP/IP for value" since 2016. Hyperledger Aries / Indy made similar claims for credential exchange. The W3C Web Payments Working Group effort effectively stalled. These prior efforts illuminate the common failure modes: two-sided liquidity (no value to carry without endpoints, no endpoints without value); regulatory perimeter (money carriers attract MTL/FinCEN/MiCA scrutiny that pure-data protocols escape); incumbent disinterest in interop; crypto-native rails (Lightning, stablecoin L2s) capturing developer mindshare.

### What v0.1.0 is honestly missing

- **A reference implementation.** Spec-only is the single biggest credibility gap relative to every peer in the table above.
- Key rotation / pre-rotation primitives (KERI's strength).
- Revocation registry (VC StatusList 2021 equivalent).
- Privacy primitives (BBS+, SD-JWT, Veilid-style private routing).
- Schema discovery (JSON-LD contexts / ACDC schema SAIDs).
- Formal cross-cluster gateway / dispute format.
- Published conformance test vectors.

### What v0.1.0 is genuinely ahead on

- **Deterministic lockstep BFT for federation** — borrowed from real-time multiplayer game netcode rather than blockchain consensus (PoW/PoS) or single-leader assumption (AT Protocol). 8–32-node clusters with BFT supermajority commit on per-frame state hash. Real architectural distinction.
- **Joule cost as a protocol-level field** rather than telemetry. No peer above encodes energy as first-class. Aligns with the EU AI Act and emerging energy-disclosure direction.
- **Cargo-agnostic by construction.** Money is one cargo type, not the primitive — closer to KERI/ACDC's "money is a cargo" stance than to Interledger's "money is the primitive."
- **Per-byte content-addressed history** maps the Holochain per-agent-chain insight to a value-carrier setting with BFT-finalised federation.

### Differentiated lane

Generic "TCP/IP for value" framing repeats a battle Interledger has been losing for nine years. The genuinely open lane is *signed, energy-metered, BFT-replicated envelopes for AI / agent provenance — money included as one cargo type, not the primitive*. EU AI Act compute-disclosure tailwind, no incumbent.

### Planned enhancements

Tracked in issues [#4](https://github.com/Transaction-Science/open-standards/issues/4), [#5](https://github.com/Transaction-Science/open-standards/issues/5), [#6](https://github.com/Transaction-Science/open-standards/issues/6) — labelled `smart-byte`.

---

## EOC

### Adjacent projects

| Project | What it does | Licence | Adoption signal |
|---|---|---|---|
| **FrugalGPT** (Chen / Zaharia / Zou, arXiv 2305.05176, TMLR 12/2024) | Formal LLM cascade + prompt adaptation + LLM approximation; 98% cost cut matching GPT-4 | Research / open | Canonical academic peer |
| **RouteLLM** (LMSYS, ICLR 2025) | Two-model router on Arena preference data; 4 router architectures; 85% cost cut at 95% MT-Bench | Apache-2.0 / Python | Most-cited open routing framework — mindshare leader |
| **GPTCache** (Zilliz) | Mature semantic cache; LangChain / LlamaIndex integration; 10× cost-cut claim | MIT | Dominant in the embedding-similarity-cache niche |
| **Helicone** | Observability + caching + routing as a gateway | Apache-2.0 + commercial | Real among mid-market AI startups |
| **HuggingFace AI Energy Score v2** (2025) | 166 models × 10 tasks; published joules-per-query | Open / public methodology | Establishes joules as a publishable unit; closest measurement-side ally |
| **GSF SCI-for-AI** (ratified Dec 2025) | Standards-body framework for AI energy accounting | Open / community | ISO/IEC 21031-aligned; the standards opening |
| **CodeCarbon / MLCO2 / ML.ENERGY** | Emissions measurement, not routing | Open | Measurement layer |
| **Anthropic / OpenAI / Google prompt caching** | Vendor primitives; 50–90% baked-in discounts | Commercial | The commercial baseline EOC competes against |
| **Together MoE routing / Portkey / OpenRouter** | Commercial gateway routing | Commercial | Market presence |
| **vLLM / TensorRT-LLM** | Single-model inference optimisation | Open | Orthogonal to cascade |
| **llama.cpp / Ollama / LM Studio** | Local inference engines | Open | The neural fallback stage EOC would invoke |
| **Mythic AI / Lightmatter / Groq** | Hardware-level energy efficiency | Commercial | Different stack |

### What v0.1.0 is honestly missing

- A Rust + WASM reference implementation with measured (not estimated) joule counters via RAPL / NVML / powermetrics.
- Cache-key canonicalisation spec (normalisation, tokenizer-independence, locale).
- Eval harness comparable to RouterBench / RouteLLM's MT-Bench harness.
- Learned router stage (RouteLLM-style matrix-factorisation is table stakes).
- Semantic-similarity threshold calibration methodology.
- Conformance suite and registered MIME / JSON-Schema for the envelope.
- SCI-for-AI mapping in the spec.

### What v0.1.0 is genuinely ahead on

- **Joules as the explicit unit of account in the protocol envelope.** RouteLLM / FrugalGPT optimise dollars; HF Energy Score reports joules but does not route on them. No peer found routes on joules.
- **Four-stage cascade with a graph stage between key-value and neural** — peers are cache→neural or cascade-of-neurals; KG retrieval as a first-class tier is genuinely novel.
- **Browser-runnable / WASM-native posture** — RouteLLM and GPTCache assume server deployment.
- **Federated, no-off-switch substrate stance.** Comparable in spirit to Matrix or ActivityPub for inference. No peer in this niche has taken that stance.

### Differentiated lane

Cascade and caching are being absorbed into vendor APIs (GPT-5's published architecture routes between fast and reasoning models internally; Anthropic / OpenAI / Google ship prompt caching at the API layer with 50–90% discounts already priced in). "Cheaper inference" is a losing axis for an open spec.

The genuinely open lane is *the measurement-and-routing standard for energy-disclosed AI* — joules as a compliance unit, not a cost lever. EU AI Act energy disclosure, GSF SCI-for-AI (Dec 2025 ratification), HF AI Energy Score, ML.ENERGY benchmark together form a measurement stack that EOC can plug into and complete with a routing decision rule.

### Planned enhancements

Tracked in issues [#7](https://github.com/Transaction-Science/open-standards/issues/7), [#8](https://github.com/Transaction-Science/open-standards/issues/8), [#9](https://github.com/Transaction-Science/open-standards/issues/9) — labelled `eoc`.

---

## Cross-cutting strategic notes

Three observations apply across all three protocols.

### Anchor-tenant problem

Open standards do not win on technical merit. They win when a forced anchor tenant carries them past the cold-start: ARPA carried TCP/IP; browsers carried HTTPS; Bitcoin carried Lightning. Each of the three protocols here needs a credible anchor tenant. The Transaction Science platform family — Settlement, PlantOS, TrustOS, TX Science AI — is the natural set, dogfooding the standards into production at the pillars first. This is the most credible adoption path; the second-most-credible is regulatory adoption (next point).

### Regulatory adoption channels

For each of the three, a regulatory or standards-body channel exists where no incumbent owns the position:

- **OpenPay** — EU PSD3, FedNow expansion, sponsor-bank programs, PCI-DSS scope reduction.
- **Smart Byte** — EU AI Act compute-disclosure, energy-attestation requirements, Trust-Over-IP / vLEI interop.
- **EOC** — GSF SCI-for-AI (ratified Dec 2025), EU AI Act energy disclosure, ML.ENERGY benchmark, HF AI Energy Score.

Positioning into regulator-aligned channels rather than "cheaper than X" channels is the structurally winnable game.

### Naming and discoverability

Each of the three names collides meaningfully with existing public-search content. The collisions are factual: OpenPay overlaps Interledger's Open Payments spec and BBVA's Openpay; SmartByte overlaps Rivet Networks / Dell's preinstalled Windows driver; EOC is a heavily-overloaded acronym including an EOLANG compiler and an Emergency Operations Center platform. Treating this as a deliberate decision — rather than letting external surface area accrue under the current names — is tracked in [issue #10](https://github.com/Transaction-Science/open-standards/issues/10).

---

## How this document evolves

This is `RESEARCH.md` v1 — dated **2026-05-23**.

Future SOTA snapshots will land in `docs/research/state-of-the-art-YYYY-MM.md`, with this file replaced by the most recent. Each version cites its sources inline and remains in the repo's history.

Specific enhancement work is tracked in [GitHub Issues](https://github.com/Transaction-Science/open-standards/issues), labelled by protocol (`openpay` / `smart-byte` / `eoc`), type (`spec` / `impl` / `bench` / `standards` / `compliance` / `seo`), priority (`p0`–`p2`), and effort (`1w`–`8w`).
