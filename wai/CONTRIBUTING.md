# Contributing to WAI

WAI has two layers — a spec and a reference implementation. The
contribution paths are different.

## To the spec (`SPEC.md`)

Open for revision. WAI is a v1.0 **draft**; until it freezes at v1.0
final, substantive changes are welcome.

A spec change is in scope if it:

- Names a real-world ambiguity in the existing text (point at the
  section and the case that breaks).
- Adds a missing structural requirement (byte layout, conformance rule,
  dispatch behavior) needed to make a third implementation falsifiable.
- Tightens the capability-dispatch or container semantics.
- Adds a new codec or capability — with a byte-precise wire definition
  and cross-implementation fixtures, not a placeholder.

A spec change is out of scope if it:

- Adds optional fields with no testable conformance impact. WAI leans
  toward fewer well-defined fields.
- Re-litigates the capability-as-requirement model (§1 is intentional:
  the model is named, never transported, never hash-pinned).
- Substitutes an existing primitive for a different one (e.g. another
  entropy coder for rANS) without a concrete failure mode in the
  existing one.

How to propose a change: open a draft in a parallel `SPEC-vX-draft.md`,
quote the section it modifies, and regenerate + run the
cross-implementation conformance suite (below) to show the change does
not regress the two reference implementations.

## To the reference implementations

Code is Apache-2.0. There are two interoperable references:

- **`wai/`** — Python, readability-first. Container, static rANS,
  `entropy_zeroth`, and the image/audio/video zeroth codecs;
  `python -m wai encode|play`. Neural paths (EnCodec, TAESD) live here
  and are out of scope for byte-level conformance (the model is a
  requirement, not part of the standard).
- **`wai-rs/`** — Rust, performance + embeddability. Same wire format;
  `wai` binary with `encode`/`play`.

The two MUST stay byte-interoperable on every lossless path. Before
submitting:

```bash
# Rust: 10 library + 9 cross-Python conformance tests
cd wai-rs
cargo test
cargo build --release --bin wai

# Regenerate the canonical Python reference fixtures and re-check the
# Rust port still agrees byte-for-byte (run from the wai-rs root; the
# sibling `wai/` Python package must be importable)
python tests/gen_cross_fixtures.py
cargo test --test cross_python
```

Both must pass. The cross-Python suite is what catches wire drift
between the two implementations — it is the executable definition of
conformance for v1.0 (`SPEC.md` §9).

## A third implementation

Conformance is **cross-implementation byte-equivalence**, not "passes
its own tests." A new-language implementation is conformant when, given
the fixtures emitted by `wai-rs/tests/gen_cross_fixtures.py`, it
produces byte-identical output for the bit-exact paths (rANS,
`entropy_zeroth`, container) and reconstruction-equivalent output
(identical quantized arrays) for the lossy codecs (image 4:4:4 + 4:2:0,
audio MDCT, video I+P). Validate against the same fixtures the Rust
reference is held to.

A third implementation is in scope when it brings up a language
ecosystem that doesn't have one yet and is validated against those
fixtures. It is *not* in scope when it duplicates an existing reference
for ergonomic reasons; consolidate instead.

## License

By contributing, you agree your contribution is licensed under the same
terms as the corresponding layer:

- Contributions to the spec: published under the same open terms
  (royalty-free; contributors grant patent rights for compliant
  implementations).
- Contributions to either reference implementation: Apache-2.0.

No CLA. The license is the agreement.
