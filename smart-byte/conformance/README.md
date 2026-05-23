# Smart Byte Conformance Vectors

This directory contains the conformance test-vector pack for the Smart Byte
substrate specification. Any third-party implementation, in any language, can
consume these vectors to self-certify against the canonical wire format
*without* trusting the reference implementation.

The vectors are the social contract. The spec text in `../spec/` says what the
substrate is; the vectors here say what the substrate's wire format *looks
like, byte for byte*, on the inputs the spec text describes. An implementation
that passes the vectors is canonical. An implementation that fails the vectors
is not.

The vectors are licensed CC-BY-4.0, consistent with the spec directory. They
contain no `$ref` pointers, no external URLs, and no runtime dependencies.

## What is in the pack

Each spec version has its own subdirectory (`v1/`, `v2/`, ...). Today only
`v1/` exists; it corresponds to the spec text in
`../spec/treatise_v1_parts_I_II_III_combined.md`.

A pack for a given version contains five JSON files plus `regeneration.md`:

| File                             | Tests                                                    |
| -------------------------------- | -------------------------------------------------------- |
| `canonicalisation_vectors.json`  | CBOR canonical-encoding rules (§8 wire format)           |
| `said_vectors.json`              | SAID computation: BLAKE3-256 over canonical CBOR (§8.2)  |
| `signature_vectors.json`         | Ed25519 sign + verify over SAIDs (§8.3, §8.4)            |
| `lockstep_vectors.json`          | BFT supermajority commit logic (§9, §13)                 |
| `gossip_vectors.json`            | Gossip propagation, conflict detection, ordering (§17)   |

`regeneration.md` documents how to regenerate the vectors from the Rust
reference implementation (`smart-byte-rs`, landing alongside spec v1.0).

## How to consume the vectors

Each JSON file is a top-level array of vector objects. Vector objects share
the common shape:

```json
{
  "name": "short-stable-identifier",
  "description": "human-readable description of what is being tested",
  ...test-specific fields...
}
```

The conventional flow for an implementation under test (IUT):

1. Load the JSON file for the relevant spec version.
2. For each vector, run the IUT against the `input` (or equivalent field)
   and compare its output to the `expected_*` field.
3. A vector passes if the IUT's output is byte-for-byte equal to the
   `expected_*` field (or, for negative vectors, the IUT rejects the input
   with the documented `error` class).
4. The IUT is canonical for a spec version iff every vector in that
   version's pack passes.

There is no scoring, no "mostly conforms" status, and no waiver process.
A vector pack is a hard gate.

## Versioning rule

Vectors are versioned **per spec version**, not per implementation, not per
calendar release of the conformance pack.

- `v1/` corresponds to spec v1. Once spec v1 is frozen, `v1/` is append-only
  in scope (additional vectors may be added that test the *same* canonical
  wire format) and never modifies existing vectors' `expected_*` outputs.
- `v2/` will appear when (and only when) the spec moves to a new version
  that introduces an incompatible wire-format change. The compatibility
  matrix lives in `VERSIONS.md`.
- A bug in a vector pack (e.g. an `expected_*` field that disagrees with
  the spec) is fixed by superseding the vector with a new one whose `name`
  carries a suffix (`-fixed`, `-corrected`); the original is retained with
  a `"superseded_by"` field so historical IUTs that pinned to the old
  vector can detect the supersession.

## CI gate

`../../.github/workflows/conformance.yml` is the repository-level CI gate.
Every PR that touches `smart-byte/spec/` or `smart-byte/conformance/` must
pass the gate. The gate validates:

- Every JSON file in `v*/` parses as valid JSON.
- Every spec change is accompanied by a vector-pack update (a PR that
  modifies `smart-byte/spec/` without modifying `smart-byte/conformance/`
  is rejected at the gate).

The gate's purpose is procedural: a spec change without a vector change is
either a clarifying edit (which should declare itself by also touching the
vector pack's `regeneration.md` to record the editorial reason) or a
silent wire-format change (which is prohibited). Either way, both
directories move together.

## What the pack does not cover

The pack is the wire-format gate. It does not cover:

- Application-layer cargo schemas. Cargo is opaque to the substrate; what
  a USD-byte or a joule-byte or a sensor-reading-byte *means* is a
  layer-2 protocol concern. Layer-2 protocols ship their own vectors.
- Performance. The pack tests correctness, not throughput, latency, or
  energy cost. Performance is documented in the reference implementation's
  benchmarks.
- Operational concerns. Cluster topology choice, key custody, hardware
  procurement, regulatory compliance for specific cargo types — all out
  of scope for the substrate's conformance pack.

## License

The vector pack is licensed CC-BY-4.0. See `../../LICENSE` for the
repository-level license terms.
