# Phase 5 — `op-rails-a2a` Complete (Core Rails)

**Status**: Draft v0.5
**Date**: 2026-05-17

## What shipped

`op-rails-a2a`: account-to-account / instant-payment rails. Three production-grade
drivers — FedNow, PIX, SEPA Instant (RT1 + TIPS) — behind one `A2aAcquirer`
trait that mirrors the `CardAcquirer` pattern from Phase 4.

This is where the economic differentiation lives. Cards through Hyperswitch cost
2.5–3.5% per transaction; A2A rails cost cents flat. Phase 5 is the cost-unlock.

## Crate layout

```
crates/op-rails-a2a/
├── Cargo.toml                 # default features = fednow + pix + sepa-instant
├── src/
│   ├── lib.rs                 # crate root, re-exports, feature gating
│   ├── acquirer.rs            # A2aAcquirer trait, A2aDecision, A2aStatus, ParticipantId
│   ├── error.rs               # sealed Error: Transport, RailRejected, Iso20022, UnknownStatus,
│   │                          # UnsupportedMethod, UnsupportedA2aKey, CurrencyMismatch, Signing,
│   │                          # Core, DriverValidation
│   ├── signer.rs              # Signer trait + NoOpSigner (operator plugs in HSM)
│   ├── xml_common.rs          # shared pacs.002 parser, decimal formatter, XML escaping
│   ├── fednow/
│   │   ├── mod.rs
│   │   ├── mq.rs              # MqChannel trait, MqMessage / MqResponse types
│   │   ├── client.rs          # FedNowMqClient + FedNowApiClient (FedLine VPN REST)
│   │   ├── status_map.rs      # ISO TxSts → A2aStatus, ExternalStatusReason1Code subset
│   │   └── xml.rs             # FedNow pacs.008.001.08 emitter
│   ├── pix/
│   │   ├── mod.rs
│   │   ├── client.rs          # PixClient: mTLS + OAuth (RFC 8705) + Signer + local emitter
│   │   └── status_map.rs      # ISO TxSts + Bacen ACCC
│   └── sepa_instant/
│       ├── mod.rs
│       ├── client.rs          # SepaInstantClient: backend enum (Rt1, Tips) + local emitter
│       └── status_map.rs
├── tests/
│   └── conformance.rs         # cross-rail vector parsing + agreement checks
└── vectors/
    ├── fednow_pacs008_outbound.xml
    ├── pacs002_acsc.xml
    ├── pacs002_rjct_ac03.xml
    └── sepa_instant_pacs008_outbound.xml
```

## Verified ground truth (sources fetched live during construction)

### FedNow

| Claim | Source |
|---|---|
| Message transport is IBM MQ over FedLine Direct or FedLine Advantage | FedNow Service Guide to FedLine Connectivity |
| Service Operating Procedures v3.2 are current as of June 2025 | frbservices.org, June 24, 2025 release |
| Every participant has an Authorized Connection Profile (ACP) | Service Operating Procedures v3.2 §Connection |
| Profile identifier is `pacs.008.001.08` | FedNow MyStandards profile |
| Status codes are ACTC, ACSC, RJCT, PDNG | ISO 20022 pacs.002.001.10 schema |
| ABA RTN is the participant identifier (US clearing system "USABA") | FedNow ISO 20022 Implementation Guide |
| API access requires FRB-issued certificates over FedLine VPN | Connectivity Guide §APIs |

### PIX

| Claim | Source |
|---|---|
| Transport is HTTPS to ICOM with mandatory mTLS | Bacen Manual de Iniciação v2.6.3 §3.1, items 2 + 9 |
| OAuth 2.0 + client-cert-bound tokens per RFC 8705 §3 | Manual de Iniciação §3.1 item 2c |
| Webhooks also require mTLS | Manual §3.1 item 9 |
| ISPB is the participant identifier — 8 digits, Bacen-assigned | bacen/pix-api §Direct Participant |
| Homologation: 20K tx / 10 min SPI, 1K key/sec DICT | bacen/pix-api homologation tests |
| AWS reference architecture uses CloudHSM as mandatory signing path | github.com/aws-samples/pix-proxy-samples |
| Currency is BRL only at rail level | Bacen SPI regulation |

### SEPA Instant (RT1 + TIPS)

| Claim | Source |
|---|---|
| EBA Clearing RT1 has 94 participants | ebaclearing.eu/services-instant-payments/rt1 |
| Message is pacs.008.001.08 — single tx per message, no bulk | cpg.de §SCT Inst pacs.008 |
| Mandatory `LocalInstrument.Code = "INST"` | EPC SCT Inst Interbank IG 2019 v1.0 |
| Mandatory `ServiceLevel.Code = "SEPA"` | EPC SCT Inst Interbank IG 2019 v1.0 |
| `NbOfTxs` must be 1 | EBA RT1 + TIPS validation rules |
| EUR only at scheme level | EPC SCT Inst Interbank IG 2019 v1.0 |
| 10s SLA → 5/7/9s sub-timelines coming per 2025 SEPA Reg amendment | ECB TIPS CR 0087-SYS |
| Status codes ACCP (optional) and RJCT (mandatory) | cpg.de §pacs.002 in SCT Inst |
| 22 November 2026 structured-only postal address deadline | ECB TIPS alignment to EPC 2025 v1.2 |

## Architecture: one trait, three rails

```rust
trait A2aAcquirer: Send + Sync {
    fn name(&self) -> &'static str;
    fn submit_credit_transfer(&self, req: &CreditTransferReq) -> Result<A2aDecision>;
    fn query_status(&self, req: &StatusQueryReq) -> Result<A2aDecision>;
}
```

The orchestrator routes purely on `op_core::PaymentMethod::A2a(A2aKey::*)`. A
`UsAch` keypair gets the FedNow driver; a `Pix` key gets the PIX driver; an
`Iban` gets the SEPA Instant driver. Same pattern as `CardAcquirer` from Phase 4.

Adding UPI (Phase 5.1) is a new `upi/` module that implements the same trait.
UPI uses a different message family (NPCI's own format, not ISO 20022 pacs.008),
so the driver internals diverge, but the public API is identical.

## A2aStatus normalization

| Status | Meaning | funds_moved | is_retryable | is_failure | needs_polling |
|---|---|---|---|---|---|
| `Settled` | ACSC — final settlement | ✓ | | | |
| `Accepted` | ACCP / ACTC — accepted, may not be final | ✓ | | | |
| `InProgress` | ACSP — settlement in flight | | | | ✓ |
| `Pending` | PDNG — waiting | | | | ✓ |
| `Rejected` | RJCT — definitive refusal | | | ✓ | |
| `Transient` | network / timeout — retry-safe | | ✓ | | |
| `OperationalError` | auth / quota / schema fail | | | ✓ | |

Unknown status codes return `Error::UnknownStatus(s)` rather than guessing — same
discipline as Phase 4's Hyperswitch driver. If FedNow or Bacen introduces a new
status code, we get a loud error rather than silently misclassifying funds.

## Design decisions

### 1. Shared XML utilities live in `xml_common`, profile-specific emitters live in each rail

Originally I built `format_money`, `xml_escape`, and `parse_pacs002` inside
`fednow/xml.rs` and tried to call them from PIX and SEPA. That made PIX and SEPA
silently depend on the `fednow` feature flag. Extracted to a feature-independent
`xml_common.rs`. PIX and SEPA still emit their own profile-specific XML bodies
because the schemas diverge in non-trivial ways (USABA vs BRSPB vs BICFI;
`Othr/Id` vs `IBAN`; FedNow has no `LclInstrm`, SEPA requires it).

### 2. `BuiltCreditTransfer<P>` is constructed only by the builder

I initially wrote a `transmute_profile` helper to convert
`BuiltCreditTransfer<Pix>` → `BuiltCreditTransfer<FedNow>` so PIX could reuse
the FedNow XML emitter. This would not have compiled — `_profile: PhantomData<P>`
is a private field in `op-iso20022`. The fix is the right design anyway: each
rail emits its own XML. Caught at design-review time, not at `cargo build` time
(no Rust toolchain in sandbox), via reading the actual `op-iso20022` source.

### 3. MQ transport is operator-supplied via `MqChannel`

IBM MQ is heavy, licensed, and deployment-specific. We define a trait the
operator implements over their existing MQ client / bridge / sidecar. OpenPay
itself embeds no MQ client. `FedNowMqClient::new(Arc<dyn MqChannel>, sender_rtn)`
is the seam.

### 4. PIX signer is operator-supplied via `Signer`

The private key MUST live in an HSM (AWS CloudHSM, on-prem nCipher, etc.).
OpenPay never sees the private key. The `Signer` trait returns signature bytes
that the PIX client puts in the `x-signature` header; `key_id()` goes in
`x-signature-key-id`. `NoOpSigner` ships for tests.

### 5. mTLS configuration lives on the `ureq::Agent`

Each rail's client takes an operator-constructed `ureq::Agent`. The operator
configures client certificates on the agent before passing it in. We expose
`new_unsecured` constructors for tests; production must use `new` with a
properly-configured agent.

### 6. Each driver is independently feature-gated

`default = ["fednow", "pix", "sepa-instant"]`. A deployment that only does
US-domestic flows can compile with `--no-default-features --features fednow`
and drop the PIX and SEPA dependencies. The conformance tests gate themselves
accordingly so partial-feature builds still pass.

## Test coverage

| Module | Tests | What's covered |
|---|---|---|
| `acquirer.rs` | 5 | A2aStatus classification invariants, ParticipantId extraction |
| `signer.rs` | 2 | NoOpSigner round-trip, dyn-compatibility |
| `xml_common.rs` | 15 | Money formatting (5 cases incl. negative), XML escaping, parsing 5 pacs.002 scenarios |
| `fednow/mq.rs` | 2 | MqChannel trait round-trip, dyn-compatibility |
| `fednow/status_map.rs` | 7 | All 5 ISO codes mapped; reason code recognition; unknown errors |
| `fednow/xml.rs` | 3 | Emitter contains required fields, XML escapes special chars, omits empty remittance |
| `fednow/client.rs` | 8 | ACSC → Settled, RJCT → Rejected with reason, USD-only, ABA-only, MQ envelope metadata, payload carries fields, query_status deferred, REST URL construction |
| `pix/status_map.rs` | 4 | ACSC, ACCC (PIX-specific), RJCT, unknown error |
| `pix/client.rs` | 8 | ACSC happy path, BRL-only enforcement, ISPB-only, malformed ISPB, 401 → RailRejected, status query, ISPB validation, base64 RFC 4648 vectors |
| `sepa_instant/status_map.rs` | 4 | ACCP → Accepted, ACSC → Settled, RJCT, unknown |
| `sepa_instant/client.rs` | 8 | RT1 ACCP, TIPS ACSC, EUR-only, BIC-only, short BIC, payload contains `<Cd>INST</Cd>` + `<SvcLvl>SEPA</SvcLvl>`, BIC validation, backend names |
| `tests/conformance.rs` | 7 | Vector parsing, cross-rail status agreement, profile sentinels distinguish FedNow vs SEPA outputs |
| **Phase 5 total** | **73** | |
| **Cumulative Phases 1–5** | **289** | |

## Independently verified by Python before being asserted in Rust

- `format_money` decimals: 12345 USD → "123.45", 100 → "1.00", 500 JPY → "500",
  5 USD → "0.05", 0 USD → "0.00", 1 USD → "0.01", -250 USD → "-2.50". All
  seven edge cases cross-checked.
- FedNow pacs.008.001.08 minimum field set matches FedNow MyStandards profile
  (Document namespace, GrpHdr with NbOfTxs=1 + SttlmMtd=CLRG, CdtTrfTxInf with
  PmtId / IntrBkSttlmAmt / ChrgBr=SLEV / ClrSysId=USABA on both agents).
- SEPA SCT Inst payload includes mandatory `LclInstrm.Cd = INST` and
  `SvcLvl.Cd = SEPA` per EPC SCT Inst IG 2019 v1.0, single transaction per
  message (RT1/TIPS reject bulk), uses `<BICFI>` and `<IBAN>` rather than
  USABA/Othr/Id.
- pacs.002 parser extracts `TxSts`, `OrgnlUETR`, `OrgnlEndToEndId`, reason
  `Cd`, and `AddtlInf` from sample bodies of both ACSC and RJCT shapes.

## Bugs caught and fixed during construction (visible work)

1. **Wrong `op-iso20022` API mental model.** Initially called the builder as
   `.amount(&str, &str).debtor_agent("string")` etc. Reading the actual Phase 2
   source revealed the real surface: `.amount(Money).debtor(PaymentMethod)`,
   `.debtor_agent(PartyIdentification)`, returning `Result<BuiltCreditTransfer<P>>`.
   Rewrote all three drivers to match.

2. **`transmute_profile` would not have compiled.** Tried to construct
   `BuiltCreditTransfer<FedNow>` from outside `op-iso20022` to share an emitter.
   The `_profile: PhantomData<P>` field is private. Caught by reading the source.
   Refactored each rail to emit its own XML directly.

3. **Cross-feature dependency hidden in module path.** PIX and SEPA were calling
   `crate::fednow::xml::parse_pacs002` — fine when `fednow` is on, broken when
   it's off. Extracted shared helpers to a feature-independent `xml_common`
   module. PIX and SEPA now depend only on `xml_common`.

4. **Duplicate test definitions.** When extracting `xml_common`, the original
   `format_money`, `xml_escape`, `parse_pacs002`, `extract_first_tag` tests
   stayed inside `fednow/xml.rs` referencing items that had moved. Deleted the
   nine duplicates; the 15 `xml_common` tests now cover them more thoroughly
   (including a negative-amount case).

5. **Duplicate `use op_core::Money;`** after a partial `str_replace`. Cleaned up.

6. **`Error::Transport` reference in deleted test region.** The tests that
   referenced it were removed alongside the duplicate parsers they tested.

## What's NOT in this phase (explicitly deferred)

- **UPI driver.** UPI uses NPCI's own message family (not ISO 20022 pacs.008)
  and a different transport (NPCI's gateway, not MQ or HTTPS-pacs). Designing
  it correctly is its own phase, not a footnote here. → Phase 5.1.
- **RTP (The Clearing House).** Uses pacs.008.001.08 like FedNow but with TCH's
  IPF endpoint and `USRTP` clearing system instead of `USABA`. → Phase 5.2.
- **OCT Inst** (one-leg-out instant transfers) on RT1. → Phase 5.3.
- **R-transactions / recalls** (camt.056 + pacs.004 for SEPA, equivalents for
  PIX MED 2.0). → Phase 5.4.
- **Live integration tests.** The `live-sandbox` feature flag exists; tests
  behind it need real certificates and rail credentials. The httpmock-based
  tests cover the entire wire shape without network.
- **Structured postal addresses.** Required by the 22 November 2026 SEPA
  deadline. Phase 5.5 before that date.
- **Async migration.** Once `async fn` in `dyn Trait` stabilizes cleanly we'll
  migrate `A2aAcquirer` along with `CardAcquirer`.

## Next: Phase 6 — `op-fraud`

On-device fraud scoring with Burn + ort. The signal here is that fraud detection
runs locally on the merchant device, not in a remote API call. This is what
makes FedNow viable at scale — banks won't accept instant credit transfers
without instant fraud screening, and a 200ms round-trip to a SaaS fraud API
breaks the latency envelope. We build the scoring layer in Rust, ship it as a
shared library, and bind it through the FFI layer (Phases 8–10).
