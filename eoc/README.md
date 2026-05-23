# EOC — Energy-Optimized Compute

A free, always-available, energy-honest AI substrate. AI that costs joules, not tokens.

AI is not free yet. The internet is. EOC closes the gap: a substrate that resolves work at the lowest sufficient cost and refuses to spend unbounded energy chasing benchmark deltas. Every query flows through a four-stage pipeline — **state-construct → retrieve → refine → check** — and a large language model is one of six refine operators, ordered last. The Fourth Truth: refinement before generation. The substrate runs in a browser, runs on commodity CPUs, denominates everything in joules per task against an open evaluation corpus, and cannot be turned off because no one entity runs it.

The substrate is intentionally **ownerless**: no protocol-level token, permissive licensing, conformance test vectors and open eval corpora so any implementation can verify itself without a trust relationship with the authors. Transaction Science is one steward — it publishes the spec, ships a reference implementation, and operates commercial services around the substrate (managed nodes, certification, registry hosting) — not its proprietor.

## Contents

- `spec/eoc1_spec.docx`, `spec/eoc1_v0_1_1_patch.docx`, `spec/eoc1_v0_2.docx`, `spec/eoc1_v0_2_addendum_A.docx` — EOC-1: the substrate architecture (v0.1 tiers → v0.2 four-stage pipeline; Addendum A: non-goals — token economics)
- `spec/eoc2_wire.docx` — EOC-2: the wire protocol (capability exchange, gossip, signed envelopes)
- `spec/eoc3.docx` — EOC-3: artifact distribution (content-addressed fetch for models, state machines, corpora)
- `spec/eoc4_dcy.docx` — EOC-4: DCY (Deterministic Cypher — the query language for the knowledge-graph retrieve operator)
- `spec/eoc5_registry.docx` — EOC-5: the Operator Family Registry (stable identifiers + conformance tests)
- `spec/eoc1_eval001*.docx`, `spec/eval001_worked_instances.docx`, `spec/eval002.docx` — Eval-001 (schema-conformant generation proficiency standard) and Eval-002 (retrieval/provenance)
- `spec/eoc_architecture.docx` — the architecture working draft
- `spec/tokens_concealment.docx` — *Tokens Are the Concealment* (the rhetorical companion to Addendum A)
- `os/THEOREM.md`, `os/Q1_W_DEFINITION.md` — the energy-first OS theorem the substrate is designed to run on, and the resolution of its open question on defining "useful work" in joules-per-work

## Status

EOC-1 at v0.2; EOC-2..5 and Eval-001/002 at working-draft 0.1; OS theorem at v0.2. The reference implementation referenced across the specs as "v0.3" is to be brought into this repository.

## Related

- Smart Byte (`../smart-byte/`) — the value substrate; uses EOC's energy attribution.
- TX Science AI (`ais.transaction.science`) — the AI PaaS; its memoizing cascade is an instance of the EOC four-stage pipeline.
