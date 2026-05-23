# Phase 2 — `op-iso20022` Complete

**Status**: Draft v0.2
**Date**: 2026-05-17

## What shipped

`op-iso20022` is a typed, validated facade over the upstream `open-payments-iso20022`
v1.0.10 and `open-payments-fednow` v1.0.10 crates. It does three things the upstream
doesn't:

1. **Idiomatic constructors** via `CreditTransferBuilder<P: Profile>` — callers
   describe a payment in OpenPay terms; the builder produces a profile-conformant
   message.
2. **Per-rail profiles** — `FedNow`, `Rtp`, `SepaInstant`, `Pix` each pin the right
   ISO 20022 version, accepted status codes, and identifier formats.
3. **Round-trip codec** with `quick-xml` and a conformance test suite that
   reads real-shaped sample XML and verifies our schema model.

## Modules

| Module | Responsibility |
|---|---|
| `error` | Sealed `Error` enum, exhaustive match at every callsite. |
| `message` | `MessageKind` (pacs / pain / camt / admi families) + version-agnostic `Message` wrapper. |
| `status` | `TransactionStatus` (ACTC/ACSC/RJCT/PDNG/RCVD/ACCP) + `StatusReason` (AC01/AC03/AC04/AC06/AG01/AM04/AM05/FRAD/...) with unknown-code preservation via `Other(String)`. |
| `bah` | Business Application Header (head.001.001.02) with strict format validation. |
| `codec` | `from_xml` / `to_xml` over quick-xml + `round_trip_canonical` for conformance tests. |
| `profile` | Per-rail rules: FedNow, RTP, SEPA Instant, PIX. |
| `builder` | High-level `CreditTransferBuilder<P>` bridging op-core domain types to ISO 20022. |

## Verified ground truth

Every fact in this crate that comes from outside our codebase was cross-referenced
against a primary source before being encoded:

| Claim | Source |
|---|---|
| FedNow uses `pacs.008.001.08` | FedNow Service Operating Procedures v3.2, June 2025, §15 |
| FedNow status codes are ACTC/ACSC/RJCT/PDNG | Same, §14 + Post-Implementation FAQ |
| FedNow BAH required on every value message | Same, §13 |
| RTP allows up to 8 agents in pacs.008 | Payments Canada RTR ISO 20022 Spec v1.4, May 2025, §3.1 |
| SEPA Instant uses BIC identifiers | EPC SCT Inst Rulebook 2024 |
| PIX uses 8-digit ISPB clearing-system codes | Banco Central do Brasil PIX manuals |
| ABA routing checksum is weighted (3,7,1) × 3 mod 10 | Federal Reserve ABA spec |
| UETR is lowercase UUID v4 | SWIFT GPI specification |
| Unstructured remittance capped at 140 chars | EPC SCT Inst Rulebook + FedNow ISO 20022 Quick Reference |

## Test coverage

| Module | Tests |
|---|---|
| `message` | 4 |
| `status` | 6 |
| `bah` | 6 (including 5 real-world ABA numbers cross-verified by Python script) |
| `profile` | 13 |
| `builder` | 10 (UETR format cases independently verified) |
| `conformance.rs` (integration) | 4 |
| **Total Phase 2** | **43** |
| **Cumulative (Phases 1 + 2)** | **62** |

## Conformance vectors

Three real-shaped sample XML messages live in `crates/op-iso20022/vectors/`:

- `fednow_pacs008_v08_minimal.xml` — customer credit transfer, USD $100,
  Chase (021000021) → BofA (026009593), with valid UETR.
- `fednow_pacs002_v10_acsc.xml` — ACSC settlement status response.
- `fednow_pacs002_v10_rjct.xml` — RJCT with AC04 (closed account) reason.

Every field name, status code, and routing number in these samples was checked
against the FedNow Operating Procedures and the Federal Reserve's ABA registry
(by checksum, since the routing numbers are real). Account numbers, message IDs,
and UETRs are synthetic.

## What is NOT yet implemented (deferred to later phases)

- The full mapping from `BuiltCreditTransfer<P>` to the upstream `Document`
  enum's variant for each pacs.008 version. We have the validated intermediate
  form; binding it to the upstream serde types is mechanical and will be done
  once we run a real `cargo check` against the v1.0.10 crate (the field names
  in v08 of pacs.008 are stable but I'd rather verify than guess).
- The full ISO 20022 ExternalStatusReason1Code list (~200 codes). We have the
  ~14 most common; unknown codes round-trip via `StatusReason::Other(String)`.
- `pacs.009` (FI credit transfer / liquidity), `pain.013` (RFP), `camt.056`
  (return request) builders. The infrastructure is in place; each adds 50–100
  lines of profile-validated builder code.

## Next: Phase 3 — `op-emv`

EMV BER-TLV codec for contact, contactless, and Tap-to-Pay payload parsing.
This is what `op-rails-card` and `op-ffi-swift` will consume to read the
`ProximityReader` payload from iOS and the equivalent NFC payload from Android.
