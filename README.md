# Transaction Science Open Standards

This repository houses the four open reference protocols stewarded by [Transaction Science](https://transaction.science). Each is published as a separate top-level directory with its own README, its own licence file, and its own status.

The wire format and the right to fork are public. Transaction Science writes the reference implementation and runs the optional hosted services — the protocols themselves are owned by no one.

## The four standards

### [openpay/](openpay/) — Payment acceptance without the SaaS tax

A reference payment-acceptance stack in Rust. Card, account-to-account (FedNow, PIX, SEPA Instant), and stablecoin rails behind one orchestrator; typestate-enforced payment lifecycle; append-only double-entry ledger with bi-temporal time-travel. One core compiles to iOS, Android, browser, and Linux. PAN never crosses a non-vault boundary.

- **Site:** [openpay.transaction.science](https://openpay.transaction.science) · [spec](https://openpay.transaction.science/spec)
- **Licence:** Apache-2.0 (see [`openpay/LICENSE`](openpay/LICENSE))
- **Status:** v0.1.0 reference. `cargo test --workspace` is 1124 passing, 0 failing; `cargo clippy --workspace --all-targets` is zero warnings. Not certified by a card scheme, not audited by a PCI QSA, not run in regulated production.

### [smart-byte/](smart-byte/) — A carrier for value

A content-addressed, signed envelope that carries any cargo with provenance and energy cost intrinsic, replicated by deterministic lockstep, owned by no one. TCP/IP for value: settlement-agnostic, energy-attributed, with no off switch.

- **Site:** [byte.transaction.science](https://byte.transaction.science)
- **Licence:** CC-BY-4.0 (see [`smart-byte/LICENSE`](smart-byte/LICENSE))
- **Status:** Reference protocol. Treatise + conformance specification published; reference implementation pending.

### [eoc/](eoc/) — AI that costs joules, not tokens

Energy-optimised compute: every query resolves through a four-stage pipeline — cache, key-value, graph, neural — and only invokes a neural model when nothing cheaper can. The substrate runs in a browser, on commodity CPUs, and cannot be turned off because no single entity runs it.

- **Site:** [eoc.transaction.science](https://eoc.transaction.science)
- **Licence:** CC-BY-4.0 (see [`eoc/LICENSE`](eoc/LICENSE))
- **Status:** Reference protocol. Specification suite published across `spec/`; reference implementation pending.

### [wai/](wai/) — Web AI Media Transport & Execution

A container + capability-dispatch standard for media. WAI does not re-implement codecs; it dispatches to SOTA standard libraries (PNG, FLAC, zstd as the mandatory floor; AVIF, JPEG-XL, Opus, AV1, XZ as the recommended modern set) and adds an envelope that lets a neural-shared-prior path coexist with the model-free floor. Two paths open every conforming file: the **neural condition** (sink regenerates from a shared ambient prior) and the **zeroth condition** (registered SOTA codec menu). The capability a file requires is named, not supplied.

- **Site:** [wai.transaction.science](https://wai.transaction.science)
- **Licence:** Apache-2.0 (see [`wai/LICENSE`](wai/LICENSE))
- **Status:** v1.0 draft standard. `wai-rs/` reference implementation (Rust lib + cdylib + staticlib, C FFI) passes 11 capability + envelope round-trip tests. Known consumer: CommunicationOS.

## How this is organised

Each subdirectory is self-contained: it carries its own README, its own licence, and its own contribution guidance. Cross-protocol consistency lives at this level — in this README and in [`CHARTER.md`](CHARTER.md).

```
open-standards/
├── README.md       — this file
├── CHARTER.md      — the stewardship pattern
├── openpay/        — payment-acceptance stack (Apache-2.0, Rust)
├── smart-byte/     — value-carrier substrate (CC-BY-4.0, spec)
├── eoc/            — energy-optimised compute (CC-BY-4.0, spec)
└── wai/            — media transport + capability dispatch (Apache-2.0, Rust ref impl)
```

The three standards do not depend on one another. They share a steward, a unit of accounting (joules), and a doctrine — that the protocol is the public commitment and the operations are the offer.

## Contributing

Contributions are welcome at the subdirectory level. Issues, discussions, and pull requests scope by subdirectory tag. The smallest valuable contributions:

- **OpenPay:** new driver implementations, persistence backends for the domain stores, FFI platform glue, ISO 20022 / EMV test vectors.
- **Smart Byte:** worked examples of cargo types, additional conformance vectors, security-engineering critique.
- **EOC:** evaluation suites, additional `eval*` worked instances, registry entries.
- **WAI:** new capability registrations (neural codec extensions), conformance test vectors per registered capability, language bindings against the `wai-rs` C ABI (Python ctypes, Node N-API, Swift, JNI, Go cgo).

A `CONTRIBUTING.md` per standard captures the specifics where they diverge.

## Contact

- Web: [transaction.science](https://transaction.science)
- Mail: [hello@transaction.science](mailto:hello@transaction.science)
