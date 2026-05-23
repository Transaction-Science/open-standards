# Contributing to OpenPay

OpenPay is a reference payment-acceptance stack. Contributions are welcome — especially the kinds listed below. Before you open a PR, please read this short guide.

## What's in scope

The contributions that move the needle most right now:

- **Driver implementations.** New PSP drivers for `CardAcquirer`, new A2A rail drivers for `A2aAcquirer`, new crypto gateways for `CryptoGateway`. The `op-driver-sdk` conformance harness will tell you whether your driver behaves before you ship it.
- **Backend implementations** of any of the store traits (`LedgerStore`, `WebhookStore`, `ReconciliationStore`, `RefundStore`, `DisputeStore`, `SettlementStore`, `SubscriptionStore`, `IdempotencyStore`) against your persistence layer of choice. The default `Graph*Store` impls persist to a single embedded Minigraf `.graph` file; a Postgres / TigerBeetle / SurrealDB / Spanner backend would be a useful contribution.
- **Real `HttpTransport` and `EvmSigner` impls** that close gaps the reference workspace leaves to operators on purpose (Fireblocks, AWS KMS, GCP KMS, Tessera, Tangem).
- **Test vectors** for ISO 20022 messages, EMV tags, reconciliation scenarios, and FX rounding edge cases.
- **Platform glue** for the FFI bridges — a Swift package, an Android AAR, an npm package wrapping the wasm output.
- **Documentation fixes.** If something in the per-phase docs in `docs/` is wrong or stale, a PR that fixes it is appreciated.

## What's out of scope

- **Adding card-network certification claims** to the README. The architecture admits a certified deployment; the workspace itself isn't one.
- **Adding new opinion-heavy defaults** to the orchestrator or router that aren't strictly necessary. The stack is a *reference* — operators pick policy.
- **Anything that grows the dependency surface significantly** in default-feature builds. The crate boundaries are deliberate; opinionated additions belong behind a feature flag or in a downstream crate.

## Development workflow

1. **Fork + branch.** Branch off `main`.
2. **Match the existing style.** No new opinions about formatting; the codebase already follows `rustfmt` defaults and the lints declared in each crate's `lib.rs` (`clippy::pedantic` in most places). Run `cargo fmt --all` before pushing.
3. **Tests must pass.** `cargo test --workspace` plus any feature-gated tests for the area you touched. CI runs the matrix on every PR.
4. **Clippy must be clean.** `cargo clippy --workspace --all-targets -- -D warnings`. Feature flags you change need the per-feature lint pass too — CI runs both.
5. **Zero new build warnings.** The workspace has a strict zero-warnings policy.

## PR checklist

- [ ] `cargo fmt --all` clean.
- [ ] `cargo test --workspace` green.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [ ] If you touched a feature-gated module: the per-feature `cargo test` and `cargo clippy` are also clean.
- [ ] If you added a new public type, function, or trait: it has a `///` doc comment that explains the contract (not just what the code already says).
- [ ] If you added a domain workflow: there's an end-to-end test that exercises the happy path through HTTP if applicable.
- [ ] No new dependencies in default-feature builds without justification in the PR description.
- [ ] Commit messages are imperative and informative. Squash-merging is fine; please keep the final commit message useful.

## Phase documentation

Each substantial change should add or amend the relevant `docs/NN-name-progress.md` file with:

- **What shipped** (a small table).
- **Honest concerns** carried forward (what we deliberately didn't do, and why).
- A workspace-status snapshot (`cargo test --workspace`, `cargo clippy --workspace --all-targets`).

Look at any of the `docs/2*-*-progress.md` files for the shape.

## Reporting issues

Open a GitHub issue with:

- The crate(s) affected.
- The shortest reproducing snippet you can.
- What you expected vs. what happened.
- The output of `cargo test --workspace 2>&1 | tail -50` if your issue is a test failure.

Security-sensitive issues should go to the email listed on the repository's profile, not the public issue tracker.

## License

By contributing, you agree your contribution is licensed under [Apache-2.0](LICENSE), the same as the rest of the project.
