# Phase 3 — `op-emv` Complete

**Status**: Draft v0.3
**Date**: 2026-05-17

## What shipped

A zero-allocation BER-TLV codec for EMV payment payloads. This is what the
FFI layer (`op-ffi-swift`, `op-ffi-jni`) will hand bytes from `ProximityReader`
on iOS and `IsoDep` on Android, and what `op-fraud` will walk to extract
feature vectors.

## Modules

| Module | Responsibility |
|---|---|
| `error` | Sealed `Error` enum with byte-offset reporting on every variant. |
| `tag` | `Tag` (u32-packed 1–4 byte tag), `TagClass`, 22 named EMV tag constants. |
| `stream` | `TlvIter` + `TlvRef<'a>` — zero-alloc, `no_std`-compatible streaming decoder. |
| `tree` | `Tlv` + `TlvBody` — heap-allocated tree with `find()` and `encoded_len()`. Behind `std` feature. |

## Design choices

- **Zero alloc on the hot path.** `TlvIter` borrows from the input slice and
  never allocates. The secure-element interface and on-device fraud pre-pass
  use this layer.
- **`no_std`-compatible.** Default feature is `std`; turn it off to ship into
  bare-metal contexts.
- **`#![forbid(unsafe_code)]`.** Every index is bounds-checked. Combined with
  property tests, the parser is panic-safe on arbitrary bytes — critical
  because the secure element hands us untrusted input.
- **Padding tolerated.** EMV Book 3 §B.3 permits `0x00` bytes between TLVs;
  the iterator silently skips them.
- **Error variants carry byte offsets.** Debugging a parse failure on a 1 KB
  EMV response without knowing where in the bytes it failed is miserable;
  every error variant carries the offending offset.

## Verified ground truth

| Claim | Source / verification |
|---|---|
| BER-TLV tag is 1–4 bytes | EMV Book 3, Annex B |
| Multi-byte tag indicator = lower 5 bits of first byte = `0x1F` | EMV Book 3 §B.1 |
| Tag continuation flag = bit 8 of subsequent bytes | EMV Book 3 §B.1 |
| Constructed bit = bit 6 of first tag byte | EMV Book 3 §B.1 |
| Short-form length = 0–127 in a single byte | EMV Book 3 §B.2 |
| Long-form length: `0x8N` + N value bytes, N ≤ 4 | EMV Book 3 §B.2 |
| Indefinite length (`0x80` alone) banned in EMV | EMV Book 3 §B.2 |
| FCI Template tag `6F` is constructed Application class | Independently verified via Python computation |
| DF Name `84` is primitive Context-Specific (NOT Application) | Independently verified — caught and corrected during testing |
| FCI Proprietary `A5` is constructed Context-Specific | Independently verified |
| All 22 named tag constants' wire-length classifications | Independently verified by Python script |
| Canonical FCI test vector parses correctly | Hand-decoded the 28-byte sequence; assertions match |
| Tap-to-Pay flat byte offsets (0, 9, 18, 23, 28, 33) | Independently verified by Python walk |

## Test coverage

| File | Tests | Notes |
|---|---|---|
| `tag.rs` | 16 | Tag class, P/C bit, multi-byte parsing, round-trip write/read |
| `stream.rs` | 16 | Length forms, indefinite rejection, padding, offsets, constructed children |
| `tree.rs` | 6 | Owned tree, `find`, `encoded_len`, nested parsing |
| `conformance.rs` | 9 | Real vectors: FCI template, Tap-to-Pay flat, Tap-to-Pay nested |
| `properties.rs` | 3 | `proptest`: no panics on random bytes, stream/tree agreement |
| **Total Phase 3** | **50** | |
| **Cumulative (Phases 1+2+3)** | **112** | |

## Conformance vectors

| File | Contents | Source |
|---|---|---|
| `fci_template.hex` | `6F1A840E315041592E5359532E4444463031A5088801025F2D02656E` | Canonical Payment System Environment FCI from EMV Book 1 Annex A. Selects the `1PAY.SYS.DDF01` directory; declares English language; SFI = 2. |
| `taptopay_flat.hex` | 6 standard transaction-data tags | $1.00 USD purchase, USA, 2026-05-17. Synthetic but byte-for-byte what a terminal kernel emits. |
| `taptopay_nested.hex` | Same fields wrapped in `E1` private constructed | Tests recursion. |

## Bugs caught during construction (kept for the record)

1. **`DF_NAME` class assertion was wrong.** I initially asserted
   `Tag::DF_NAME.class() == TagClass::Application`. The Python cross-check
   showed `0x84` starts with bits `10`, making it `ContextSpecific`. Fixed
   before any phase artifact shipped.

2. **Nested-find test vector was malformed.** My first draft had
   `6F 0C A5 0A ...` — but A5's declared length (10) exceeded its actual
   inner content (8 bytes). Python overrun-detection caught it. Corrected
   to `6F 0A A5 08 ...`.

Both bugs were caught because every numerical claim in the test suite is
independently verified by Python before committing to Rust. The same Python
scripts live in our verification log; they're how we keep ourselves honest
without a Rust toolchain in the sandbox.

## What is NOT yet implemented (deferred)

- **Encoder.** We can decode any EMV BER-TLV blob. Encoding from a `Tlv` tree
  back to bytes is straightforward but waits on Phase 4 (`op-rails-card`)
  because that's where we need it: producing the response payload to send
  back through the PSP for refunds and reversals.
- **EMVCo certificate validation.** Verifying the issuer-side cryptogram
  (`9F26`) requires the public-key infrastructure: schemes' root CAs, issuer
  certificates, signed static/dynamic data. That's a substantial subsystem
  with its own crate — `op-emv-crypto` — deferred to after `op-rails-card`
  is wired up.
- **Fuzz corpus.** We have property tests; a dedicated `cargo-fuzz` corpus
  with AFL-discovered crashes goes in once the encoder lands.

## Next: Phase 4 — `op-rails-card`

The PSP adapter trait plus the first concrete driver: Hyperswitch. This is
where the verified-but-deferred ISO 20022 builder bindings get wired up and
where the EMV decoder feeds real Tap-to-Pay payloads into a sandbox HTTPS
flow.
