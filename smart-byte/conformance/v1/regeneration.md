# Regenerating v1 Conformance Vectors

The vectors in this directory are **reference vectors generated alongside the
Rust reference implementation**, expected to be regenerated when
`smart-byte-rs` lands. Until that point, the `expected_*` fields in each
JSON file are placeholder values that are *internally consistent* (the same
envelope produces the same SAID across vectors) but are not yet
cryptographically computed.

This file documents the regeneration procedure so the placeholders can be
replaced with real values in one mechanical pass when the reference
implementation is available.

## Prerequisites

- Rust toolchain pinned by `../../rust-toolchain.toml`.
- `smart-byte-rs` crate (private repo, to be open-sourced alongside spec
  v1.0 freeze).
- `cargo run --bin gen-conformance-vectors` produces the v1 pack.

## Regeneration procedure

```bash
# From the smart-byte-rs checkout
cargo run --release --bin gen-conformance-vectors -- \
    --spec-version v1 \
    --output-dir /path/to/open-standards/smart-byte/conformance/v1
```

The generator does the following, in order:

1. **Canonical CBOR vectors.** For each `input` JSON object, run the
   canonical CBOR encoder and write the hex output to `expected_cbor_hex`.
   The encoder uses RFC 8949 §4.2.1 (Core Deterministic Encoding) plus the
   additional Smart Byte rules: byte-string for byte arrays, no tags,
   sorted map keys.

2. **SAID vectors.** For each `envelope`, substitute a zero-byte placeholder
   for the `said` field, canonical-CBOR-encode, BLAKE3-256 hash, and write
   the result to `expected_said`. The SAID format is the KERI prefix-byte
   scheme: `E` prefix byte (= BLAKE3-256) followed by the base64url
   encoding of the 32-byte hash. The `canonical_cbor_hex` field is also
   populated.

3. **Signature vectors.** For each positive vector, derive the Ed25519
   verifying key from the seed via the standard RFC 8032 procedure, sign
   the `envelope_said` (32 bytes, raw, not the prefix-byte format) under
   the signing key, and write the signature to the `signature` field. For
   negative vectors, the generator does not modify the explicit
   `signature` / `verifying_key` fields; it only verifies that the
   `expected_verify: false` assertion holds when the named verifier runs
   the vector.

4. **Lockstep vectors.** For each vector, the generator runs the reference
   BFT-commit logic on the provided `node_hashes` and confirms that
   `expected_commit` and `expected_state_hash` match. For commits, it also
   confirms the `expected_dissent` log matches what the reference
   dissent-tracking logic produces.

5. **Gossip vectors.** For each scenario, the generator simulates the
   gossip propagation under spec §17 rules and confirms the
   `expected_outcome` / `expected_observations` match the reference
   gossip-engine output.

## Invariant: cross-vector consistency

The generator MUST emit consistent values across vectors. In particular:

- An envelope appearing in `said_vectors.json` and `lockstep_vectors.json`
  has the same SAID in both files.
- A SAID appearing in `signature_vectors.json` as `envelope_said` and in
  `said_vectors.json` as `expected_said` is identical.

The placeholder values in the current pack are internally consistent in
the same way; the regeneration replaces each placeholder family with the
real value computed from the canonical pipeline.

## What is NOT regenerated

- Vector `name` fields (stable across regenerations; implementations pin
  to names, not array positions).
- Vector `description` fields (human prose, not generator output).
- The set of vectors (additions go through `VERSIONS.md` and the
  CI gate).

## Verifying a regeneration

After regeneration, run:

```bash
cargo run --release --bin verify-conformance-vectors -- \
    --pack /path/to/open-standards/smart-byte/conformance/v1
```

The verifier reads each vector, re-runs the canonical pipeline, and
asserts byte-for-byte equality with the file contents. A clean exit
status confirms the pack is self-consistent and matches the reference
implementation's current behaviour.

## Cross-language sanity check

The vectors are the contract; the Rust reference implementation is the
*proximate* generator but not the canonical authority. Before merging a
regeneration, at least one additional implementation (currently planned:
Go and TypeScript) must independently produce the same `expected_*`
values from the same inputs. If implementations disagree, the disagreement
is resolved by appeal to the spec text, not by which implementation
emitted the value first.
