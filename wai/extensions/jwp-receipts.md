# WAI Extension: JWP Receipt Chain

> Status: Draft. Sub-spec of WAI v1.0. Mirrors the Rust reference impl at
> https://github.com/Transaction-Science/open-standards/tree/main/wai/extensions/jwp-receipts.

Keywords **MUST**, **MUST NOT**, **SHOULD**, **MAY**, **REQUIRED**,
**OPTIONAL** are RFC 2119 / RFC 8174.

---

## 1. Scope and model

This extension defines a JWP-shaped receipt chain that attests
**per-object delivery** of WAI envelope objects through a chain of
relays. A receipt names the bytes a sink received, the publisher that
produced them, the energy spent producing them, and the chain of
relays the bytes traversed — all under a single Ed25519 signature per
group.

In scope:

- A per-object content commitment (BLAKE3-256) over WAI envelope bytes.
- A per-group Merkle root binding every object in a group under one
  signature.
- A per-group Ed25519 signature over canonical group metadata.
- A typed receipt envelope (JSON or equivalent CBOR) with explicit
  parent-receipt linkage for cross-hop preservation.
- Verifier behaviour, relay preservation requirements, and conformance
  test vectors.

Out of scope:

- A transport. This extension does **not** define how receipts move on
  the wire; receipts ride alongside WAI envelopes in the carrier
  protocol (e.g. a MoQT object header field, an HTTP trailer, a
  sidecar file).
- A codec. The receipt chain is opaque to the WAI capability dispatch
  in WAI SPEC.md §4. A receipt attests bytes; it does not interpret
  them.
- An envelope. WAI SPEC.md §2 defines the envelope; this extension
  sits **above** it. A receipt's content hash is computed over the
  full WAI envelope (`WAI1` magic + manifest + payload), not over the
  payload alone.

Relationship to WAI SPEC.md:

- WAI SPEC.md §2 (container) — receipts hash the container bytes
  end-to-end; a sink that mutates the envelope (re-orders manifest
  keys, re-frames the payload) breaks the content hash and MUST be
  rejected by §3 step 4 below.
- WAI SPEC.md §4 (capability dispatch) — receipts attest **what was
  delivered to the dispatcher**, not what the dispatcher produced. A
  sink that successfully verifies a receipt then dispatches the
  envelope per §4 unchanged.

---

## 2. Wire format

### 2.1. Per-object content hash

For each WAI envelope object the publisher emits, the publisher MUST
compute a 32-byte BLAKE3-256 content hash:

```
content_hash = BLAKE3(domain || envelope_bytes)
domain       = "moq-jwp:object\x01"
```

`envelope_bytes` is the full WAI envelope per WAI SPEC.md §2 (the
`WAI1` magic, the manifest-length prefix, the manifest, the payload-
length prefix, the payload — concatenated, with no insertion or
reframing). The domain string is appended as a domain separator to
prevent cross-protocol hash collisions.

### 2.2. Per-group Merkle root

A **group** is an ordered list of one or more objects under a single
publisher key. The publisher MUST build a BLAKE3 binary Merkle tree
over the object content hashes in their group order. Leaves and
internal nodes use distinct domain separators:

```
leaf(content_hash)        = BLAKE3("moq-jwp:merkle-leaf\x01" || content_hash)
node(left, right)         = BLAKE3("moq-jwp:merkle-node\x01" || left || right)
```

When the leaf count at any level is odd, the publisher MUST duplicate
the final leaf to obtain an even pair. The root is the single hash at
the top of the tree. The empty-group root is the 32-byte all-zero
value `0x00..00`.

### 2.3. Per-group Ed25519 signature

The publisher MUST sign the **canonical signing payload** with an
Ed25519 key per RFC 8032 (Pure Ed25519, no prehash). The signing
payload is the big-endian concatenation:

```
signing_payload =
    "moq-jwp:group-sign\x01"          // domain
 || u64_be(track_alias)                // optional carrier-scoped id (0 if unused)
 || u64_be(group_id)                   // monotonic per-track group counter
 || merkle_root[32]                    // §2.2
 || u64_be(leaf_count)                 // number of objects in the group
 || u64_be(total_joules_micro)         // sum of microjoules across the group
 || u32_be(len(group_start)) || group_start_utf8   // RFC 3339 start timestamp
 || u32_be(len(group_end))   || group_end_utf8     // RFC 3339 end timestamp
 || publisher_key[32]                  // Ed25519 verifying key (RFC 8032)
 || u32_be(len(publisher_id)) || publisher_id_utf8 // bound identity string
```

The signature is exactly 64 bytes. Implementations MUST reject any
group receipt whose `signature` length is not 64.

### 2.4. Receipt envelope (typed)

Two receipt kinds are defined: `group` (one per group) and `object`
(one per object). Both are typed JSON; equivalent CBOR is permitted
when the carrier mandates it, provided the field set and semantics
are identical.

#### 2.4.1. `kind: "group"`

Required fields:

| field | type | meaning |
|---|---|---|
| `kind` | string | MUST equal `"group"`. |
| `track_alias` | u64 | Carrier-scoped track identifier; `0` if unused. |
| `group_id` | u64 | Monotonic group counter under `publisher_key`. |
| `root_hash` | 32 B hex | §2.2 Merkle root over object content hashes. |
| `leaf_count` | u64 | Number of objects covered by `root_hash`. |
| `joules_total` | u64 | Sum of microjoules across the group's objects. |
| `group_start` | string | RFC 3339 timestamp of the first object. |
| `group_end` | string | RFC 3339 timestamp of the last object. |
| `signer_pubkey` | 32 B hex | Ed25519 verifying key per RFC 8032. |
| `signer_id` | string | Bound identity string (DID, URN, or display). |
| `sig` | 64 B hex | Ed25519 signature over §2.3 payload. |
| `parent_receipt_hash` | 32 B hex or `null` | BLAKE3 of the previous group receipt's canonical bytes under this publisher key; `null` for the first group. |

#### 2.4.2. `kind: "object"`

Required fields:

| field | type | meaning |
|---|---|---|
| `kind` | string | MUST equal `"object"`. |
| `track_alias` | u64 | Mirrors the parent group receipt. |
| `group_id` | u64 | Mirrors the parent group receipt. |
| `object_id` | u64 | Monotonic from 0 within the group. |
| `content_hash` | 32 B hex | §2.1 BLAKE3 over the WAI envelope bytes. |
| `joules_micro` | u64 | Microjoules consumed producing this object. |
| `origin` | string | Datacenter / origin label. |
| `merkle_proof` | object | Inclusion proof: `{ "siblings": [32 B hex, …], "direction_bits": u64, "leaf_index": u64 }`. `direction_bits` bit `i` is `1` iff the current node at level `i` is the right child. |

### 2.5. Worked example

Manifest JSON (group receipt, formatted for readability — wire form
serializes with no insignificant whitespace and preserves key
insertion order):

```json
{
  "kind": "group",
  "track_alias": 7,
  "group_id": 3,
  "root_hash": "5d2c7b1f9a04e2a16c83b3d774a09f12c44b65a3e8d11f88c2901a47b50d2c61",
  "leaf_count": 2,
  "joules_total": 250,
  "group_start": "2026-04-25T12:00:00Z",
  "group_end":   "2026-04-25T12:00:01Z",
  "signer_pubkey": "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
  "signer_id": "did:joule:publisher-1",
  "sig": "9f3a4d6b1c0e8742f1b6b94e7d1c2a85f5103e4c8b2d61a93f0e4b8f2c70a1d61f6a2b71d4eecbb0d7a2e90c4b1bf2c8a0e4d63b9101fa0e2c4d83b1f70e9a103",
  "parent_receipt_hash": null
}
```

Object receipt for the second object in that group:

```json
{
  "kind": "object",
  "track_alias": 7,
  "group_id": 3,
  "object_id": 1,
  "content_hash": "1c3f4a2e88b0d77192aabd0c66f12345e89c7b41ab2cd35f7e80b912ad6c9012",
  "joules_micro": 150,
  "origin": "joule-cloud-us-west-2",
  "merkle_proof": {
    "siblings": [
      "a07b4d2c91f5e3110687f4b210ff8e1133acdb0741e3920c19acf8b3110d57e2"
    ],
    "direction_bits": 1,
    "leaf_index": 1
  }
}
```

The signature hex shown is illustrative; conformance is checked
against the test vectors in §5.

---

## 3. Verifier algorithm

A conforming verifier MUST execute the following six steps in order
for every group receipt and every object receipt it accepts. Steps
match the algorithm style of WAI SPEC.md §4 (numbered, terminating).

1. **Parse and shape-check.** Read the group receipt. If `kind !=
   "group"`, reject. If `sig` is not 64 bytes, reject. If
   `signer_pubkey` is not a valid 32-byte Ed25519 verifying key
   (RFC 8032), reject. If any required field from §2.4.1 is absent,
   reject.

2. **Verify the Ed25519 group signature.** Construct the canonical
   signing payload per §2.3 from the receipt's fields. Verify `sig`
   against `signer_pubkey` per RFC 8032 Pure Ed25519. If verification
   fails, reject. After this step, the verifier holds a **verified
   root**: the receipt's `root_hash` field, now bound to
   `signer_pubkey` under signature.

3. **Check parent linkage.** If the verifier maintains receipt-chain
   state for `signer_pubkey`, compute `BLAKE3("moq-jwp:receipt-link\x01"
   || canonical_bytes(previous_group_receipt))` and compare to
   `parent_receipt_hash`. If they differ, reject (chain break). If
   no previous receipt is known and `parent_receipt_hash` is `null`,
   accept as the chain head. A non-null `parent_receipt_hash` without
   a corresponding previously-verified receipt is a recoverable
   condition: the verifier MAY request the missing receipt before
   accepting; it MUST NOT silently accept.

4. **Verify the object's content hash.** For each accompanying
   `kind: "object"` receipt: recompute `content_hash` per §2.1 over
   the received WAI envelope bytes and compare to the receipt's
   `content_hash` field. If they differ, reject the object (the
   envelope was mutated in transit).

5. **Verify the Merkle inclusion proof.** Compute the leaf as
   `BLAKE3("moq-jwp:merkle-leaf\x01" || content_hash)`. Walk the
   `merkle_proof.siblings` list from level 0 upward; at each level
   `i`, combine the current value with the sibling using the
   `direction_bits` bit `i` (1 = current is right child, 0 = current
   is left child) and the `moq-jwp:merkle-node\x01` domain. After
   walking all siblings, compare the resulting hash to the verified
   root from step 2. If they differ, reject.

6. **Accept.** The object's bytes are bound to `signer_pubkey` at
   `(track_alias, group_id, object_id)`, with the publisher's
   declared `joules_micro` and `origin`. The verifier MAY now persist
   the group receipt as the new chain head for `signer_pubkey` and
   dispatch the WAI envelope per WAI SPEC.md §4.

A verifier MUST treat any rejection in steps 1–5 as a hard failure
for the object in question. A verifier MUST NOT accept a partial
verification (e.g. valid Merkle proof against an unverified root).

---

## 4. Relay preservation requirements

A **conforming relay** forwards WAI envelopes from one or more
publishers to one or more subscribers. To preserve the receipt
chain across hops, a conforming relay:

### 4.1. MUST

- Forward every group receipt and every object receipt **unaltered**
  to every subscriber that requested receipts for the corresponding
  track. Byte-for-byte equality of receipt bytes between ingress and
  egress is REQUIRED.
- Verify the group receipt's Ed25519 signature per §3 step 2 before
  the first object of the group is fanned out. A relay that cannot
  verify the signature MUST drop the group and emit a structured
  error to upstream operators.
- Forward the `parent_receipt_hash` field as received. A relay that
  has previously forwarded the chain MAY additionally check the link
  per §3 step 3 and refuse to fan out a broken chain.
- Preserve the byte-exact WAI envelope (`WAI1` magic, manifest bytes,
  payload bytes) end-to-end. Any rewriting at the envelope layer
  invalidates §2.1 content hashes.

### 4.2. MAY

- Batch multiple `kind: "object"` receipts into a carrier-level frame
  to amortize per-receipt overhead, provided every receipt's bytes
  are preserved (no field elision, no canonicalization rewrite).
- Emit relay-scoped telemetry alongside the forwarded receipts:
  cumulative `joules_total` across observed groups, per-publisher
  delivery counts, fan-out fan-in ratios. Such telemetry is local to
  the relay and MUST NOT be conflated with publisher-issued
  receipts.
- Reorder objects within a group on egress when the carrier protocol
  permits — Merkle inclusion proofs are position-bound by
  `leaf_index`, so reordering does not invalidate verification at
  the subscriber. The relay MUST NOT renumber `object_id`.

### 4.3. MUST NOT

- Modify the WAI envelope payload bytes in any way (no transcoding,
  no manifest rewrite, no key reordering, no whitespace insertion).
  Envelope mutation is detected at §3 step 4.
- Re-issue the Merkle root or re-compute a new tree over a subset of
  objects. The publisher is the sole authority for the group root.
- Re-sign the group receipt under a relay-owned key. There is one
  signer per group, and it is the publisher named in
  `signer_pubkey`. Relay attestation, when desired, MUST be carried
  as a separate, parallel receipt and MUST NOT replace the publisher
  signature.
- Strip the `parent_receipt_hash` field, even when its value is
  `null`. The field is required by §2.4.1 and its absence is a
  shape-check failure at §3 step 1.
- Truncate the receipt to fewer fields than §2.4 requires, even when
  the carrier framing has size pressure.

---

## 5. Test vectors

Conforming implementations MUST verify against the canonical test
vector set generated by
`joulesperbit/crates/joule-moq-publisher/examples/jwp_receipts_vectors.rs`.
The generator emits five vectors covering the cases below. Each
vector consists of: the WAI envelope bytes for every object in the
group, the canonical group-receipt JSON, the canonical object-receipt
JSON for every object, and a `result.json` declaring the expected
verifier outcome (`accept` or one of the named rejection reasons).

### 5.1. Single-object group (vector `v1_single_object`)

One WAI envelope (`wai.text.zstd`, payload `"hello"` zstd-encoded).
Group receipt with `leaf_count = 1`, `parent_receipt_hash = null`.
Object receipt with `object_id = 0`, `merkle_proof.siblings = []`,
`direction_bits = 0`, `leaf_index = 0`. Expected outcome: `accept`.

### 5.2. Multi-object group with proof verification (vector `v2_multi_object`)

Five WAI envelopes (`wai.audio.opus`, 20 ms frames). Group receipt
with `leaf_count = 5` (the publisher duplicates the trailing leaf
internally when building the tree). Five object receipts, each with
a non-empty `merkle_proof`. Expected outcome: `accept` for each
object receipt.

### 5.3. Cross-hop preservation through 2 relays (vector `v3_cross_hop`)

Three publisher-issued group receipts in chain order G0, G1, G2 with
`parent_receipt_hash` chaining each to its predecessor. Two relay
hops are simulated: ingress bytes → relay A egress bytes → relay B
egress bytes are emitted as three byte-equal copies. The verifier
walks the chain in order; expected outcome: `accept` at every hop,
and the chain head after G2 equals the BLAKE3 of G2's canonical
bytes per §3 step 3.

### 5.4. Invalid signature rejection (vector `v4_bad_sig`)

A well-formed group receipt with a single bit flipped in the `sig`
field. Expected outcome: reject at §3 step 2 with reason
`signature_verify_failed`.

### 5.5. Missing parent_receipt_hash rejection (vector `v5_missing_parent`)

Two group receipts G0, G1. G1 omits the `parent_receipt_hash` field
entirely (rather than setting it to `null` or a hash). Expected
outcome: reject at §3 step 1 with reason `missing_required_field`.

The generator writes vectors to
`joulesperbit/crates/joule-moq-publisher/examples/jwp_receipts_vectors/`
as one directory per case. A conforming verifier MUST accept the
`accept` cases bit-exactly and MUST reject the named cases at the
declared step with the declared reason. The Rust reference verifier
in `inv-moq::receipt::ReceiptVerifier` exercises all five vectors
under `cargo test --lib -p inv-moq`.

---

## 6. Security considerations

### 6.1. Replay

A receipt without sequencing information is replayable: an attacker
who captures group receipt G can re-present it to a downstream
verifier and bind unrelated bytes to the publisher's identity at a
later time. This extension mitigates replay with two mechanisms:

- `parent_receipt_hash` (§2.4.1) links each group receipt to its
  predecessor under the same `signer_pubkey`, so a chain replay
  requires forging every intermediate hash.
- `group_id` is monotonic per publisher key. Verifiers SHOULD track
  the highest `group_id` seen per `signer_pubkey` and reject any
  receipt with a non-increasing `group_id`. This binds the chain
  to forward progress.

### 6.2. Forgery

The signing payload (§2.3) is bound under Ed25519 (RFC 8032). At
classical security levels, Ed25519 provides ~128-bit unforgeability
under chosen-message attack (EUF-CMA). Forging a group receipt is
equivalent to recovering the publisher's private key or breaking
Ed25519. Implementations MUST source signing keys from a CSPRNG and
MUST NOT reuse keys across distinct publisher identities.

### 6.3. Cross-group confusion

The signing payload binds `track_alias`, `group_id`, and
`signer_pubkey` under the same signature. An attacker who attempts
to re-present a group receipt under a different `(track_alias,
group_id)` tuple cannot do so without re-signing — the receipt's
fields are signature-covered, and any field mutation breaks
verification at §3 step 2. The Merkle proof is similarly bound:
`leaf_index` is verifier-checked during the proof walk, so an
object's proof cannot be transplanted to a different position within
the same group.

### 6.4. Envelope mutation in transit

Any mutation of the WAI envelope bytes (manifest rewrite, payload
re-frame, whitespace insertion) changes the §2.1 content hash and is
detected at §3 step 4. Relays that re-canonicalize the manifest
SHOULD be treated as broken: they break content-hash equality even
when the semantic meaning is unchanged. WAI SPEC.md §2 already
requires that the manifest be serialized with no insignificant
whitespace and that encoders preserve key insertion order; receipt
verification depends on relays honouring that requirement.

### 6.5. Quorum and multi-signer receipts

v1.0 of this extension is single-signer: every group receipt is
signed by exactly one Ed25519 key. Multi-signer / quorum receipts
(threshold Ed25519, BLS aggregate signatures) are explicitly out of
scope for v1.0. A future minor revision MAY extend §2.3 with a
multi-signer container; the v1.0 wire format remains a strict subset.

### 6.6. Out of scope

- **Post-quantum group-signing ciphersuites.** Migration to
  hash-based signatures (SLH-DSA), lattice-based signatures
  (ML-DSA), or hybrid Ed25519 + post-quantum receipts is deferred
  to a future major revision of this extension. Implementers
  building for long-lived archives SHOULD plan key-rotation policies
  that allow ciphersuite migration when the upgrade lands.
- **Confidentiality of receipts.** Receipts carry identity,
  joules, and origin labels in cleartext. Carriers that require
  receipt confidentiality (e.g. trade-secret origin labels) MUST
  layer their own encryption above this extension; v1.0 receipts
  are integrity-protected, not confidentiality-protected.
- **Pricing semantics for joules.** `joules_micro` and
  `joules_total` are unit-of-energy fields, not unit-of-account
  fields. Conversion to currency, carbon, or grid-mix CO₂ is
  performed by downstream aggregators and is not normative here.
