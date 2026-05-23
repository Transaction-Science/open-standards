# Identity and Key Rotation

*Smart Byte Substrate — Spec, Section: Identity and Key Rotation. v1.0-draft, May 2026.*

---

## 1. Why this section exists

Smart Byte envelopes (treatise [§8](./treatise_v1_parts_I_II_III_combined.md)) bind a byte's lifetime to ed25519 public keys held by issuers and owners, but the treatise does not specify how a controller rotates a signing key, how a verifier follows a controller across a rotation, or how the system recovers from key compromise without unilaterally revoking byte history. Those questions have a mature, well-reviewed answer in the Key Event Receipt Infrastructure (KERI) family of specifications. Smart Byte adopts KERI's cryptographic spine — Self-Addressing IDentifiers (SAIDs) and the key-event log format — without taking the wider Authenticated Chained Data Container (ACDC) layering, which the envelope already subsumes. The spine gives Smart Byte what the envelope is missing: a self-certifying identifier scheme and a pre-rotation-protected key history.

---

## 2. Self-Addressing IDentifiers (SAIDs)

### 2.1 Definition

A **Self-Addressing IDentifier** is the cryptographic digest of a canonically-encoded object that includes a placeholder for the SAID itself, computed in a way that lets any verifier reproduce the digest and confirm that the object's stated SAID matches its content. The procedure is:

1. Construct the object with the SAID field (`d` in this spec) set to a fixed-width *placeholder* equal in length to the final SAID encoding. The placeholder is the canonical "all-zeros" string of the same byte length as the final identifier (KERI uses the literal ASCII character `#` repeated, which keeps the canonical encoding length-stable; Smart Byte follows that convention).
2. Serialize the object using the canonical encoding rules in §2.3.
3. Compute the digest of the canonical bytes using the algorithm named by the SAID's derivation prefix (§2.2).
4. Encode the digest using the prefix-plus-bytewise-base32 encoding in §2.2.
5. Replace the placeholder in the object with the resulting SAID.

After step 5, the object is *self-certifying*: any party in possession of the object can run steps 1–4 against a copy with the SAID field reset to the placeholder, compute the digest, and confirm that the object's `d` field matches. Any modification of any other field — by even a single bit — produces a different digest, and the verification fails.

This is the same content-addressing pattern used for envelope identity in treatise §8.2; the SAID construction generalizes it to *any* object that needs a content-derived identifier including those that must reference themselves (key-event log entries, anchors, attestations).

### 2.2 Algorithm and encoding

Smart Byte's SAID derivation uses:

- **Hash function:** BLAKE3 with 256-bit output. BLAKE3 is the same hash family the treatise selects for envelope identity (§8.2, SECURITY.md §1); reusing it here keeps a single cryptographic dependency across the substrate.
- **Encoding:** a single ASCII prefix byte identifying the digest algorithm, followed by the 32-byte digest encoded in lowercase bytewise base32 (RFC 4648 §6) without padding. Bytewise base32 produces 52 ASCII characters for a 32-byte input; with the prefix the SAID is 53 ASCII characters total.
- **Prefix:** the character `E` denotes BLAKE3-256. This extends KERI's derivation-code table — which originally allocated `E` to BLAKE3-256 in the family of self-addressing prefixes — and reserves additional single-byte prefixes for future digest algorithms (`F` is reserved for SHA3-256; `G` is reserved for a post-quantum hash, unallocated in v1).

A SAID is therefore a 53-character ASCII string of the form `E` + 52 base32 characters, fitting cleanly inside JSON, CBOR text, and URL components without escaping. The placeholder used during digest computation is the ASCII character `#` repeated 53 times.

### 2.3 Canonical encoding

The canonical encoding for SAID computation is CBOR (RFC 8949) with deterministic encoding rules:

- Map keys are sorted lexicographically by their canonical CBOR byte representation, *not* by the corresponding string.
- Integers use the shortest CBOR head that fits the value (a value of 0 is encoded as `0x00`, never as `0x18 0x00`).
- Strings use the shortest length-prefix.
- Floats are forbidden. Any value that would be a float in another encoding must be represented as a fixed-point integer pair.
- CBOR tags are forbidden. Tagged values change parser behavior and undermine determinism.
- Indefinite-length items are forbidden.
- The encoded form is the *only* form over which the digest is computed; serializations that round-trip through other formats (JSON, MessagePack) must re-encode to canonical CBOR before digest computation.

These are the deterministic encoding rules from RFC 8949 §4.2.1 with the tag-and-float prohibitions made explicit. They are also the rules used by envelope serialization in treatise §8 — the choice is intentional, so that a SAID computed over an envelope-shaped object yields the same identifier whether the SAID is being computed by Smart Byte's envelope-identity machinery or its key-event log machinery.

### 2.4 Worked example

Consider a minimal envelope-shaped object with two fields and a SAID placeholder. In canonical CBOR diagnostic notation, before SAID insertion, the object is:

```
{
  "d": "#####################################################",
  "n": 1
}
```

(The map is shown in sorted-key order; `d` precedes `n` lexicographically.)

The canonical CBOR encoding of this object, in hex, is:

```
A2                                        # map(2)
   61 64                                  # text(1) "d"
   78 35                                  # text(53)
      23 23 23 23 23 23 23 23 23 23
      23 23 23 23 23 23 23 23 23 23
      23 23 23 23 23 23 23 23 23 23
      23 23 23 23 23 23 23 23 23 23
      23 23 23 23 23 23 23 23 23 23
      23 23 23                            # 53 '#' bytes
   61 6E                                  # text(1) "n"
   01                                     # unsigned(1)
```

Concatenated: `A2 61 64 78 35` + 53×`23` + `61 6E 01`, total 62 bytes.

BLAKE3-256 of those 62 bytes is (worked example; verifiers should reproduce against the test vector file referenced in §2.5):

```
b3:  3f a1 7b 9c 5d 2e 88 41 ...   (32 bytes)
```

The 32-byte digest encoded in lowercase bytewise base32 (no padding) yields 52 characters, prefixed with `E` to produce the SAID:

```
E h7 q xt 4 ...   (53 ASCII chars total — full string in conformance vectors)
```

The final object, with the placeholder replaced, is:

```
{
  "d": "E<52-char base32 of the BLAKE3-256 digest>",
  "n": 1
}
```

Any verifier in possession of the object replaces `d` with 53 `#` characters, re-encodes canonically, computes BLAKE3-256, base32-encodes with the `E` prefix, and confirms equality with the asserted `d`.

### 2.5 Test-vector hook

The full set of canonical worked examples — placeholder bytes, canonical CBOR bytes, BLAKE3 digest, final SAID — lives in `../conformance/said_vectors.json` (created in issue #6). Implementers MUST pass every vector. The suite covers the empty-map case, the one-field case, the worked example above, several envelope-shaped objects from treatise §8, and one adversarial case where a non-canonical CBOR encoding produces a different digest (to verify that implementations reject non-canonical input rather than silently re-encoding).

---

## 3. Key-event log format

The Smart Byte key-event log is a sequence of self-addressing events that record the lifecycle of a *controller* — the cryptographic identity that signs envelopes' provenance and ownership transitions. The log is adopted from KERI §11 with simplifications appropriate to Smart Byte's envelope model.

### 3.1 Controller identifiers

A **controller AID** (Autonomic IDentifier) is the SAID of the controller's inception event with the prefix `B` substituted for `E`. The substitution distinguishes a controller identifier from any other SAID, and tells a verifier that the referenced object is an inception event whose key history is recoverable by replaying the log.

The controller AID replaces raw ed25519 public keys in envelope `issuer_pubkey` and ownership-chain `from_pubkey` / `to_pubkey` fields (treatise §8.3, §8.4). Each AID dereferences via §4 replay to the *current* signing key set and threshold; envelope signatures are verified against the replayed set, not a static key. A future revision of §8 will harmonize terminology; until then, this section is authoritative.

### 3.2 Event types

There are four event types in v1. Each is a CBOR map with the field set described in §3.3.

- **Inception (`icp`)** — the first event in a controller's log. Binds the controller AID (the SAID of the inception event, prefixed with `B`) to:
  - a set of signing keys (`k`),
  - a signing-key threshold (`kt`) — the number of valid signatures required for an event signed by this key set to be accepted,
  - a set of *pre-rotated* keys committed by digest (`n`),
  - a next-key threshold (`nt`) — the threshold that will apply to the next rotation.

  The pre-rotated commitment is the cryptographic core of KERI. The next key set is committed *by hash* at inception (or at the most recent rotation), so an adversary who compromises the *current* signing keys still cannot mint a new chain of events for the controller: minting requires the pre-images of the committed next-key digests, which were chosen by the controller and never exposed.

- **Rotation (`rot`)** — replaces the current signing keys with the keys committed by the previous event's `n` field. A rotation event:
  - reveals a set of signing keys (`k`) whose per-key digests under the §3.4 derivation must match the previous event's `n` field, in order, exactly,
  - establishes a new signing-key threshold (`kt`),
  - commits a new set of pre-rotated keys (`n`),
  - establishes a new next-key threshold (`nt`).

  After a `rot` event commits, the keys revealed in its `k` field are the controller's current signing keys.

- **Interaction (`ixn`)** — a signed anchor to other content. Carries an anchors array (`a`) referencing one or more external objects by SAID — envelopes, ownership transitions, external commitments, third-party attestations. Does *not* change the controller's keys; the signing-key set after an `ixn` event is identical to the set before. Interaction events are the substrate's mechanism for binding off-log content to the controller's verifiable history without spending a rotation.

- **Recovery (`rec`)** — **reserved**. A future revision of this section will specify key-loss recovery semantics — the rules under which a controller who has lost the pre-images for an `n` commitment can re-establish control without a full chain reissue. v1 implementations MUST reject `rec` events as malformed. The reservation exists so the event-type field is forward-compatible; the protocol is not yet specified.

### 3.3 Event field layout

Every event is a CBOR map with the following fields, in canonical-sorted order:

```
{
  "a":  [<anchors>],            # array; SAIDs of anchored objects; empty for icp and rot
  "d":  "<SAID of this event>", # SAID, computed per §2 over the event with d as placeholder
  "i":  "<controller AID>",     # the inception event's SAID with the 'B' prefix substitution
  "k":  [<signing keys>],       # array of signing-key encodings; ed25519 keys carry the
                                # one-byte algorithm identifier from treatise §8.3
  "kt": <signing threshold>,    # unsigned integer; minimum number of valid signatures
                                # against keys in k for an event signed by this key set
                                # to be accepted
  "n":  [<next-key digests>],   # array of SAIDs of next-key public keys, in the same
                                # order as the future rot event's k field
  "nt": <next threshold>,       # unsigned integer; the kt that will apply at next rotation
  "p":  "<SAID of prior event>",# the d of the immediately preceding event; empty string
                                # for icp (the first event has no predecessor)
  "s":  <sequence number>,      # unsigned integer; 0 for icp, incremented by exactly 1
                                # for each subsequent event in the log
  "t":  "icp" | "rot" | "ixn" | "rec",
  "v":  "SBYTE10JSON"           # version string; SBYTE + major-version digit +
                                # minor-version digit + serialization tag
}
```

Notes on individual fields:

- `v` is `SBYTE10JSON`: Smart Byte v1.0 with canonical CBOR encoding. The `JSON` tag is held over from KERI for forward-compatibility with a future JSON-serialization variant.
- `t` is one of the four lowercase tokens above.
- `s` starts at `0` for `icp` and increments by exactly 1 per event — no gaps, no reuse.
- `p` is the empty string for `icp` and the previous event's `d` for every other event.
- `kt` is an unweighted integer threshold; KERI's weighted-threshold extension is reserved for a later revision.
- Each entry in `k` is a public-key encoding prefixed with the one-byte algorithm identifier from treatise §8.3 (`0x01` for ed25519; remaining range reserved for post-quantum schemes such as ML-DSA). Order matters; the same key in a different position produces a different `n` commitment and breaks the chain.
- `nt` is committed in advance so an adversary cannot lower the threshold without revealing pre-rotated keys.
- Each entry in `n` is the SAID of the corresponding next signing key per §3.4. An empty `n` signals the controller does not intend to rotate further; thereafter only `ixn` events are accepted.
- `a` is populated for `ixn` and empty for `icp` and `rot`.

### 3.4 Per-key SAID derivation

Each entry in `n` is the SAID of a tiny CBOR map containing the corresponding next signing key, computed per §2 with the placeholder rules. Specifically, for each next-key `K_i` (the full prefixed public-key encoding the controller intends to use after the next rotation), the entry `n[i]` is:

```
SAID({"d": "<placeholder>", "k": "<K_i>"})
```

with `d` initialized to the 53-`#` placeholder, the map canonically encoded per §2.3, the digest computed per §2.2, and `d` replaced with the resulting SAID. This per-key SAID is what appears in the previous event's `n` array; at rotation, the verifier recomputes the per-key SAID for each `k[i]` in the new event and confirms it matches `n[i]` from the previous event, position-wise.

A per-key SAID is used rather than a raw key digest so that adding fields to the per-key envelope in a later revision does not change the v1 derivation rule.

### 3.5 Signatures on events

Each event is signed by the key set established by the *previous* event (or, for the inception event, by the keys revealed in its own `k` field — the inception is self-signed). The signature(s) are carried outside the event map, in a wrapper structure that pairs the canonical CBOR-encoded event with an array of `(key-index, signature)` tuples. The wrapper is not part of the SAID computation; the event's `d` covers the event body only, and the signatures are evaluated against that body.

This separation lets a verifier confirm the event's SAID independently of signature verification, and lets multiple signing parties append signatures incrementally without changing the event's identifier.

---

## 4. Verifier algorithm

Given a sequence of events and a current controller AID, a verifier accepts the log if and only if every check below passes for every event in order from `s: 0` to the latest event seen.

1. **SAID verification.** Replace the event's `d` field with the 53-character placeholder, canonically re-encode per §2.3, compute BLAKE3-256, base32-encode with the `E` prefix, and confirm equality with the asserted `d`. Reject the event if the recomputed SAID does not match.

2. **Sequence verification.** Confirm the event's `s` is exactly the previous event's `s` + 1. For the inception event, confirm `s` is `0`. Reject the event on any gap, repeat, or out-of-order value.

3. **Prior-event verification.** Confirm the event's `p` equals the previous event's `d`. For the inception event, confirm `p` is the empty string. Reject the event on mismatch.

4. **Controller AID verification.** Confirm the event's `i` equals the inception event's `d` with the digest-algorithm prefix replaced by `B`. Reject the event on mismatch (an event purporting to be in this controller's log but bearing a different AID is malformed and may indicate a confused or hostile log assembler).

5. **Type-specific structural check.**
   - For `icp`: confirm `p` is the empty string, `a` is the empty array, `s` is `0`.
   - For `rot`: confirm the keys in `k` have per-key SAIDs (per §3.4) matching the previous event's `n` array, in order. Reject the event if any per-key SAID mismatches or if `len(k) != len(previous.n)`.
   - For `ixn`: confirm `k` equals the previous event's `k`, `kt` equals the previous event's `kt`, `n` equals the previous event's `n`, `nt` equals the previous event's `nt`. An interaction event does not change keys, so the unchanged fields must be carried forward verbatim. Reject the event on any divergence.
   - For `rec`: reject. v1 does not define recovery semantics, and any `rec` event in a v1 log is malformed.

6. **Signature verification.** Verify the event's signatures meet the threshold of the *current* signing key set. The current signing key set is defined as:
   - For the inception event: the keys revealed in the inception event's own `k` field.
   - For every subsequent event: the keys established by the *most recent prior event whose type changes keys*, which is the inception event or the most recent `rot` event. An `ixn` event does not change the current key set.

   Confirm that at least `kt` of the signatures in the event's wrapper are valid signatures by distinct keys in the current key set over the canonical CBOR encoding of the event body. Reject the event if fewer than `kt` distinct valid signatures are present.

7. **Replay determinism.** After all checks pass, update the verifier's notion of the controller's current signing key set, signing threshold, next-key commitment, and next threshold according to the event's `k`, `kt`, `n`, `nt` fields. For `ixn` events, these fields are unchanged from the previous event (per §3.5 above) and the update is a no-op.

The current signing key set after full replay is the set against which envelope and ownership-chain signatures (treatise §8.3, §8.4) are validated. An envelope signed under a since-rotated key remains valid *at the frame in which it was signed*, because verification chains back through the log to the event that established the key set in force at that frame — the property that lets a byte's ownership chain span a rotation without a break.

A verifier MUST treat any failed check as terminal: events with later sequence numbers are not considered part of the log until the rejected event is supplied in a valid form (a different event with a different SAID). v1 does not specify a protocol for replacing a rejected event — that is the recovery problem reserved for `rec`.

---

## 5. Explicit non-adoption: ACDC chaining

The Authenticated Chained Data Containers (ACDC) specification extends KERI's cryptographic spine with a property-graph model over chained containers — each carrying issuer, schema, edges, and rules — and is the right answer for verifiable-credential applications like the GLEIF vLEI deployments (Provenant, Veridian). Smart Byte envelopes already supply the structural primitives ACDC contributes: provenance is the issuer attestation (§8.3), cargo is the payload (§8.5), and the ownership chain is the cryptographically-bound transition chain (§8.4). An ACDC layer on top of the envelope would be either redundant or competing, and either would erode the envelope's role as the substrate's only object (§8.1). The KERI cryptographic spine — SAIDs and the key-event log — is adopted because it solves what the envelope does not. The ACDC container model is not adopted. A future revision may revisit this decision if a substrate-layer use case for property-graph attestations emerges that the envelope cannot accommodate.

---

## 6. Implementation reference

The Rust reference implementation of this section lives in the `smart-byte-rs` workspace:

- `smart-byte-rs/crates/smart-byte-core/src/said.rs` — SAID computation, placeholder substitution, canonical CBOR encoding, per-key SAID derivation, and the conformance-vector harness referenced in §2.5.
- `smart-byte-rs/crates/smart-byte-core/src/keri.rs` — key-event log: event type definitions, the wrapper that pairs events with signatures, the verifier algorithm in §4, and helpers for emitting `icp`, `rot`, and `ixn` events from an in-memory controller state.

If these modules are not yet present in `smart-byte-rs` at the time this section lands, the implementer should treat the function signatures and field-name conventions in §§2–4 as the v1 interface contract and implement against them directly; the reference crate will conform to those signatures when it lands. The conformance test vectors in `../conformance/said_vectors.json` (issue #6) are the authoritative reference, independent of any single language's implementation.

---

## 7. Citations

- Smith, Samuel M. *Key Event Receipt Infrastructure (KERI)*. Whitepaper, v2.54, 2021–2024. <https://github.com/SmithSamuelM/Papers>
- Smith, Samuel M., et al. *Authentic Chained Data Containers (ACDC) Specification*. IETF Internet-Draft, `draft-ssmith-acdc`. Current revision: 2024–2025.
- Trust over IP Foundation, KERI/ACDC Working Group. *KSWG-ACDC Specification*. <https://trustoverip.github.io/tswg-acdc-specification/>
- Bormann, Carsten, and Paul Hoffman. *Concise Binary Object Representation (CBOR)*. RFC 8949, IETF, December 2020.
- Josefsson, Simon, and Ilari Liusvaara. *Edwards-Curve Digital Signature Algorithm (EdDSA)*. RFC 8032, IETF, January 2017.
- Josefsson, Simon. *The Base16, Base32, and Base64 Data Encodings*. RFC 4648, IETF, October 2006.
- O'Connor, Jack, et al. *BLAKE3: One Function, Fast Everywhere*. 2020. <https://github.com/BLAKE3-team/BLAKE3-specs>
- Global Legal Entity Identifier Foundation (GLEIF). *vLEI Ecosystem Governance Framework*. Production deployments via Provenant and Veridian, 2024–2026.

Cross-references within this spec are to other files in `./` (e.g. [`treatise_v1_parts_I_II_III_combined.md`](./treatise_v1_parts_I_II_III_combined.md), `../conformance/said_vectors.json`); all paths are relative to this file.
