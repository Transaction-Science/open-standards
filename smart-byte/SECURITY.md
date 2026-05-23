# Security — Smart Byte Substrate

The smart byte substrate is a digital system that runs on the internet. The security concerns it faces are the security concerns every digital system on the internet faces: cryptographic correctness, transport confidentiality, supply-chain integrity, key custody, availability under load, and distributed-systems consistency. There is nothing exotic in that list, and nothing in it is unsolved — each item has a standard, well-understood answer, and this document records the substrate's engineering choices against each, plus the small number of places where the substrate does something the standard toolkit doesn't already cover and the engineering for those.

This is not a "threat model" in the dramatic sense. A digital system doesn't need a catalogue of imagined enemies; it needs to be built correctly, the way digital systems on the internet are built correctly. This is the record of building it that way. The treatise (`spec/treatise_v1_parts_I_II_III_combined.md`, Part III) carries a more exhaustive engagement for the reader who wants it; this file is the working engineering summary.

---

## 1. Cryptographic correctness

The same problem every system that signs and hashes data has, and the same answer.

- **Signatures: Ed25519.** RFC 8032; NIST FIPS 186-5 (2023). The signature scheme the modern internet already runs on — fast, small (32-byte keys, 64-byte signatures), ~128-bit security, and resistant to the implementation pitfalls (nonce reuse) that bit ECDSA. Use an audited library (`ed25519-dalek`), with strict signature encoding per Chalkias et al., *Taming the many EdDSAs* (2020), so signatures are non-malleable. Constant-time by construction; pin compiler versions and run timing-validation tests so an optimizer can't reintroduce a side channel.
- **Hashing: BLAKE3.** Built on BLAKE2's compression function (publicly vetted since 2012); ~128-bit collision / ~256-bit preimage resistance; fast (SIMD, near-linear parallelism). Used for byte identity, the history chain, and state hashes.
- **Serialization: canonical, length-prefixed, deterministic.** Two conformant implementations produce the same bytes for the same input — which is what makes content-addressing work and what makes the conformance test vectors meaningful. No floating point anywhere in state-affecting code (i64 fixed-point only); no wall-clock time; canonical hash-sorted input ordering.
- **Post-quantum: tracked, not panicked.** NIST finalized FIPS 203 (ML-KEM), 204 (ML-DSA), and 205 (SLH-DSA) in August 2024; FALCON and HQC are in ongoing standardization. NIST IR 8547 sets the transition: deprecate quantum-vulnerable algorithms by ~2035, high-risk systems earlier; CISA/NSA published quantum-safe product categories by December 2025; federal TLS-1.3-or-successor adoption is required by January 2030. The substrate's posture matches that guidance exactly: a primitive-version field in the schema, a documented dual-mode migration path (both old and new valid during the window), and a commitment to begin migration well before any credible feasibility — which is what the Mosca framework and the published timelines all say to do. Migrating to a successor primitive is a planned operation, not an emergency. (PQ signatures are larger — FALCON ~1 KB vs Ed25519's 64 bytes — so the per-byte cost structure shifts on migration; that's a known engineering consequence, planned for.)

**The standard discipline applies, fully:** don't roll your own crypto; use proven, audited primitives; fuzz the parsers; pin and reproduce the toolchain; and keep a migration path open because that's what every system that wants to last does.

## 2. Transport confidentiality

The substrate is transport-agnostic — any reliable, ordered, bidirectional channel between peers works. Production deployments run that channel over **TLS 1.3** (or **QUIC**, which is TLS-native), or over an encrypted tunnel (WireGuard, IPsec) — the standard choices, configured at the transport layer, not baked into the protocol. Message contents are signed per-input, so a wire observer can read traffic patterns but cannot forge or alter a message — sign-then-send, the standard arrangement. Browser clients use WebSocket; deeply censored environments can run over Tor or a pluggable transport. None of this is novel; it's the transport stack every internet service uses.

## 3. Supply-chain integrity

This is the security concern that affects *every* open-source project — the XZ Utils backdoor (CVE-2024-3094) made that vivid: a multi-year social-engineering campaign against a trusted-maintainer workflow, a malicious build-system macro, and every conventional defense (SBOMs, dependency audits, code review of the diff) blind to it. The substrate addresses it the way the ecosystem has converged after that incident — there is no smart-byte-specific answer here, only the standard one, applied:

- **Multi-maintainer review** — at least two independent sign-offs; signed commits and signed release tags; single-maintainer commits blocked by branch protection. The XZ workflow's solo-maintainer model is precisely what to avoid.
- **Reproducible builds + provenance** — artifact-matches-source verifiable independently; aim for SLSA Level 3 (hermetic builds, two-party review, signed provenance), which GitHub Actions supports natively via artifact attestations.
- **Maintainer accounts treated as production credentials** — hardware-backed 2FA, scoped per-package tokens, named ownership, rotation on departure.
- **Implementation diversity** — a backdoor in one implementation doesn't propagate to an independently-written one. (Currently the reference implementation is the only one; a second independent implementation is open work, and the project's security is monoculture-bounded until it exists. That's a true statement about every single-implementation system, and the answer is the same: get a second implementation.)
- **Mirrors** — the source on more than one Git host; releases GPG-signed so the authentic version is verifiable regardless of delivery channel.

## 4. Key custody

A user who loses a private key loses the bytes that key controls — the same property cash has (lose the wallet, lose the cash) and the same property every public-key system has. There is no protocol-level "undo" of a transition signed by the rightful key, because there can't be one without a privileged party who can reverse anyone's transactions, which would defeat the point. The standard mitigations all apply, at layer 2, unchanged from how the rest of the industry does them: custodial wallets with rate-limiting and anomaly detection; hardware tokens (Yubikey, Trezor, HSMs for operators); social-recovery and multisig; threshold key management splitting an operator's signing authority across jurisdictions so coercing one party isn't enough. The substrate doesn't reinvent any of this — it composes with the existing toolkit.

## 5. Availability under load

A network with bounded resources can be overloaded — true of every internet service. Standard answers, all of them used here:

- **Rate limiting and abuse detection at the operator's edge** (reverse proxy, firewall, request quotas) — the operator-layer concern it always is.
- **Geographic and jurisdictional distribution** of cluster nodes, so no single failure domain (a datacenter outage, a regional internet failure, a BGP hijack) takes the cluster offline.
- **Multiple submission paths** and gossip-based input replication, so no single operator can quietly censor a transaction.
- **Bounded-cost protocol operations** — verification, gossip propagation, and routing all bound their work explicitly; no operation an adversary can make unboundedly expensive with a small input. (This is also why floating point and unbounded queries are forbidden in state-transition code — a malformed input must not be able to make honest nodes do non-deterministic or unbounded work.)

Note that the substrate's protocol layer deliberately has *no* native token, *no* fee market, and *no* leader election — which removes, by construction, the priority-auction, fee-spike, MEV-ordering, and leader-bribery surfaces that have consumed a great deal of distributed-systems security research. Fewer moving parts, fewer things to attack.

## 6. Distributed-systems consistency

A system replicated across nodes has to agree on state and tolerate some nodes being faulty or hostile — a problem with a forty-year-old textbook answer, used unchanged: **Byzantine fault tolerance with a supermajority commit**. A frame commits when more than two-thirds of a cluster's nodes report the same post-frame state hash, which tolerates up to a third of the cluster being arbitrarily faulty — the standard BFT bound. The state-transition function is deterministic to byte-for-byte equality (no floating point, no time-varying syscalls, no allocator-dependent iteration), so "agreement on the state hash *is* agreement on the state." When a node's state diverges from the supermajority, it halts rather than propagating corruption; persistent divergence is identified through a per-cluster dissent log and handled at the cluster-membership layer. This is ordinary BFT engineering — the same shape as any Byzantine-tolerant system, plus the determinism discipline that game engines and scientific simulations have practiced for decades.

## 7. What the substrate actually invents

Almost nothing here is new — and that's the point; the cryptography, transports, supply-chain practices, key-custody options, availability engineering, and BFT consensus above are all standard, and a system that *needed* novel cryptography to be secure would be a system to distrust. The substrate's genuine inventions are all *new arrangements of well-understood primitives*, which is the safe kind of invention:

- **Per-byte content-addressed history** — Git-style Merkle hash chains, but maintained per *value* rather than per *block*, so a node need only be trusted for a byte's current state and the deep history is independently verifiable by anyone holding the byte.
- **The two-part joule cost** — a `measured` (hardware energy counters) plus `estimated` (deterministic model) energy accounting carried on every byte, cumulative over its life; divergence between the two parts is itself a usable signal.
- **Federation by clusters** — many bounded BFT clusters connected by a gossip overlay, with cross-cluster transfers verified end-to-end against content-addressed history rather than against a cluster's say-so — the scaling pattern the internet itself uses (~100,000 autonomous systems coordinated by a routing protocol), applied to value.

These are compositions, not new primitives. Build the new arrangements carefully; use the proven parts underneath.

## 8. Reporting a vulnerability

Coordinated disclosure: report to `security@byte.transaction.science` (placeholder until the stewarding entity publishes the address); we'll acknowledge, work a fix, and credit the reporter. The conformance test vectors are a standing integrity check — an implementation that produces non-canonical bytes is detectable by any party running the suite, which is part of why the suite is published.

---

*There is no exotic threat here. It is the internet. The substrate is secured the way digital systems on the internet are secured — conservative, audited cryptography; encrypted transports; reproducible, multiply-reviewed builds; the standard key-custody options at layer 2; standard availability engineering; a textbook BFT consensus — plus a handful of new compositions of those proven parts. Treating that as a "threat model" would dramatize the ordinary. This file is the engineering.*
