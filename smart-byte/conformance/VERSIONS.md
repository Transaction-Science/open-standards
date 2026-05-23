# Smart Byte Conformance Vector Pack — Version Matrix

This document tracks the correspondence between vector-pack versions and
Smart Byte spec versions.

## Matrix

| Vector pack | Spec version | Spec source                                                   | Status   | Notes                                                                 |
| ----------- | ------------ | ------------------------------------------------------------- | -------- | --------------------------------------------------------------------- |
| `v1/`       | v1.0-draft   | `../spec/treatise_v1_parts_I_II_III_combined.md` (Parts I-III) | DRAFT    | Reference vectors; expected to be regenerated when `smart-byte-rs` lands. |

Future entries are appended below this table when a new spec version is
introduced. Older entries are never removed; an implementation that pins to
a historical spec version must be able to look up that version's vector pack
indefinitely.

## Version compatibility rules

1. **Per-spec-version vector packs.** Each major spec version has exactly one
   vector pack directory (`v1/`, `v2/`, ...). The pack is the canonical
   conformance suite for that spec version.

2. **No backporting of incompatible vectors.** A vector that depends on
   wire-format behaviour introduced in spec v2 is never added to `v1/`.

3. **Append-only inside a frozen version.** Once a spec version is frozen,
   its vector pack is append-only: new vectors may be added that exercise
   *additional* corners of the *same* canonical wire format, but no
   existing vector's `expected_*` outputs may be changed.

4. **Bug fixes via supersession.** A vector with an `expected_*` field that
   disagrees with the spec is fixed by adding a new vector (with a
   distinguishing `name` suffix) and adding a `"superseded_by": "<new-name>"`
   field to the original. Implementations pin to vector names, not array
   positions, so supersession is non-breaking.

5. **Spec-version detection at the wire.** The wire format itself carries
   version information (see spec §8.3, §8.4 — the one-byte algorithm
   identifier is the substrate's wire-version-equivalent for the
   cryptographic primitives). Implementations select the conformance pack
   that matches the wire version they are emitting.

## Cross-references

- Spec text: `../spec/`
- Pack contents: `./v1/`
- Regeneration procedure: `./v1/regeneration.md`
- CI gate: `../../.github/workflows/conformance.yml`
