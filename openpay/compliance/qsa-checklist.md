# QSA Pre-Assessment Checklist

A self-assessment a Qualified Security Assessor (QSA) can run against an OpenPay deployment before the formal Report on Compliance (RoC) engagement. Twelve sections mirror the twelve top-level requirements of PCI-DSS v4.0.1; each item is checkbox-style and points to a specific code, configuration, or policy artifact in this repository.

Mark each item as one of: `[X]` satisfied, `[ ]` not yet satisfied, `[N/A]` not applicable to this deployment.

---

## Requirement 1 — Install and maintain network security controls

- [ ] **1.2.1** Configuration standards for NSCs are documented. *Artifact:* operator-supplied network architecture diagram referencing the CDE / connected / out-of-scope segments described in `compliance/pci-zero-architecture.md`.
- [ ] **1.2.5** Only approved ports / protocols / services are permitted. *Artifact:* the vault host's inbound firewall allows only TCP/443 from the merchant app-server subnet (mTLS) and TCP/443 from the browser-facing CDN (TLS 1.3). All other ports denied.
- [ ] **1.3.1** Inbound traffic to the CDE is restricted to what is necessary. *Artifact:* vault host's NSG / security group / iptables rules; ingress rule list captured in deployment runbook.
- [ ] **1.4.1** NSCs are implemented between trusted and untrusted networks. *Artifact:* CDE network segment is a separate VPC / subnet / namespace with no direct peering to the merchant app-server segment except via the mTLS endpoint.
- [ ] **1.4.4** System components storing CHD are not directly accessible from untrusted networks. *Artifact:* vault host has no public IP; it sits behind an internal load balancer reachable only from app servers and (for the collection iframe) a CDN.

## Requirement 2 — Apply secure configurations to all system components

- [ ] **2.2.1** Configuration standards exist and cover all applicable system components. *Artifact:* the vault host is built from an immutable image containing exactly the OpenPay vault binary (`cargo build --release -p <your-vault> --features op-core/pci-scope,op-vault/<backend>`), the operator's chosen KMS client library, and a minimal init system. No sshd, no shell, no general-purpose package manager in the running image.
- [ ] **2.2.2** Vendor default accounts are removed or changed. *Artifact:* the OpenPay codebase ships no default accounts. The HTTP API uses API-key auth via the `OP_API_KEYS` env var; vault mTLS uses certificates issued from an operator-controlled CA.
- [ ] **2.2.6** System security parameters are configured to prevent misuse. *Artifact:* TLS 1.3 only; TLS 1.2 only as a fallback for acquirer hops that don't speak 1.3 yet; cipher suite list pinned in the `op-server` config (operator-supplied).
- [ ] **2.3.1** Wireless networks are configured securely. *Artifact:* N/A unless the merchant deploys Wi-Fi card-present terminals; in that case, WPA3-Enterprise + cert-based 802.1X.

## Requirement 3 — Protect stored account data

- [ ] **3.2.1** Account data is not stored after authorisation unless there is a documented business need. *Artifact:* `op-ledger` entries reference `psp_payment_id` and masked card metadata only; raw PAN never enters the ledger. Verify via `grep -r "RawPan" crates/op-ledger crates/op-graph` — should match nothing.
- [ ] **3.3.1** SAD is not stored after authorisation. *Artifact:* `op-emv` parses the EMV TLV and immediately forwards it to the acquirer; the parsed blob is not persisted in any store. Verify via inspection of `crates/op-emv/src/`.
- [ ] **3.4.1** PAN is masked when displayed. *Artifact:* `op_vault::CardData::Debug` masks to `first_six` + `last_four`; verified by `tests/test_debug_masks_pan_to_first6_last4` in `crates/op-vault/src/card_data.rs`.
- [ ] **3.5.1** PAN is rendered unreadable wherever it is stored. *Artifact:* `InMemoryVault` uses AES-256-GCM-SIV (RFC 8452) for the reference impl; production vaults use the operator's KMS / HSM backend. See `compliance/hsm-kms-guidance.md`.
- [ ] **3.6.1** Cryptographic keys are protected from disclosure and misuse. *Artifact:* KEK lives in KMS / HSM; DEK exists in vault memory only for the duration of one encrypt or decrypt operation; `aes_gcm_siv::Key` and DEKs derive `Zeroize`.
- [ ] **3.7.1** Key-management policies and procedures are documented. *Artifact:* operator-supplied key-management policy, structured as in `compliance/hsm-kms-guidance.md` §"Key rotation" for each backend.
- [ ] **3.7.4** Cryptographic keys are changed when their cryptoperiod ends. *Artifact:* KMS rotation policy set to ≤365 days for KEKs; per-token DEKs are single-use by construction.

## Requirement 4 — Protect cardholder data with strong cryptography during transmission

- [ ] **4.2.1** PAN is encrypted with strong cryptography during transmission over open, public networks. *Artifact:* `op-rails-card` posts to the acquirer over TLS 1.2+ using `rustls` (default of the `ureq` dependency in `crates/op-rails-card/Cargo.toml`). The collection iframe uses TLS 1.3 from browser to vault.
- [ ] **4.2.1.1** Inventory of trusted keys and certificates is maintained. *Artifact:* operator-supplied cert inventory listing every cert in the TLS chain for vault, app server, acquirer endpoint, KMS endpoint.

## Requirement 5 — Protect all systems and networks from malicious software

- [ ] **5.2.1** An anti-malware solution is deployed on system components commonly affected by malware. *Artifact:* applicability statement. The vault host is an immutable image running a single Rust binary with no shell, no interpreter, and no writable user filesystem; anti-malware on the host is generally not applicable per PCI-DSS v4.0.1's *targeted risk analysis* clause. Document the analysis.
- [ ] **5.3.4** Anti-malware mechanisms are actively running. *Artifact:* same as 5.2.1.
- [ ] **5.4.1** Anti-phishing controls are in place. *Artifact:* DMARC `p=reject` on the operator's email domain; phishing-resistant MFA (FIDO2 / WebAuthn) for all human accounts with vault access.

## Requirement 6 — Develop and maintain secure systems and software

- [ ] **6.2.1** Bespoke and custom software is developed securely. *Artifact:* `#![forbid(unsafe_code)]` or `#![deny(unsafe_code)]` is set in every CDE-relevant crate (`op-vault`, `op-orchestrator`, `op-rails-card`). Verify with `grep -r "forbid(unsafe_code)\|deny(unsafe_code)" crates/`.
- [ ] **6.2.4** Engineering practices prevent or mitigate common software attacks. *Artifact:* `Money` is `i64` minor units (no float-rounding bugs); `Payment<S>` is a typestate (refund-before-capture is a compile error); `Vault` trait does not distinguish "not found" from "auth failed" (no oracle); `op-webhook` HMAC-signs outbound events.
- [ ] **6.3.1** Security vulnerabilities are identified and addressed. *Artifact:* `cargo audit` runs in CI; `cargo clippy --workspace --all-targets -- -D warnings` is green (see `README.md` §Status).
- [ ] **6.3.2** Inventory of bespoke and custom software is maintained. *Artifact:* the workspace `Cargo.toml` `members` list serves as the inventory; each crate's `Cargo.toml` `description` documents its responsibility.
- [ ] **6.3.3** All system components are protected from known vulnerabilities. *Artifact:* monthly `cargo update` cadence; security-advisory subscriptions for `aes-gcm-siv`, `ureq`, `rustls`, `axum`, `tokio`, `k256`.
- [ ] **6.4.1** Public-facing web applications are protected from attacks. *Artifact:* `op-server` ships with API-key auth and token-bucket rate limiting; operators add WAF (CloudFront / Cloudflare / GCP Cloud Armor) at the edge.

## Requirement 7 — Restrict access to system components and cardholder data by business need to know

- [ ] **7.2.1** An access-control model defines roles and access. *Artifact:* operator-supplied role matrix. Recommended baseline: `vault-admin` (KMS + vault host SSH), `app-server-admin` (merchant app server only, no KMS access), `auditor` (read-only access to audit logs, no production system access).
- [ ] **7.2.4** All user accounts are reviewed at least once every six months. *Artifact:* operator-supplied quarterly review record.
- [ ] **7.2.5** All application and system accounts are reviewed periodically. *Artifact:* the vault workload identity (IAM role / service account / AppRole) is reviewed at the same cadence; rotation procedure documented.

## Requirement 8 — Identify users and authenticate access to system components

- [ ] **8.2.1** All access by users and system components is identified. *Artifact:* `op-server` API keys are per-caller (`OP_API_KEYS` is a list); vault mTLS certs are per-workload.
- [ ] **8.3.1** Strong authentication for all access into the CDE. *Artifact:* mTLS to vault from app servers; FIDO2 / WebAuthn for human admin access to the vault host's bastion.
- [ ] **8.3.6** Passwords / passphrases meet minimum complexity and length. *Artifact:* applicability statement — the OpenPay stack does not use passwords on machine-to-machine paths. For human access, operator's IdP enforces ≥12-character passwords + MFA.
- [ ] **8.4.1** MFA is implemented for all non-console access into the CDE. *Artifact:* mTLS counts as MFA when the certificate is hardware-backed (YubiKey, TPM, SEV-SNP attestation); document the binding.

## Requirement 9 — Restrict physical access to cardholder data

- [ ] **9.1.1** Facility entry controls limit and monitor physical access. *Artifact:* cloud-provider's data-centre certification (AWS / GCP / Azure SOC 2 + PCI-DSS attestation) covers this for cloud-hosted vaults. Self-hosted HSM operators provide their own facility evidence.
- [ ] **9.4.1** Media with CHD is protected. *Artifact:* applicability statement — OpenPay does not write CHD to removable media. Encrypted database backups (which contain wrapped DEKs + ciphertext) follow the same KMS encryption discipline.

## Requirement 10 — Log and monitor all access to system components and cardholder data

- [ ] **10.2.1** Audit logs are enabled for all system components. *Artifact:* `tracing` instrumentation is present throughout the codebase (see workspace dependency `tracing = "0.1"` in root `Cargo.toml`); operators wire `tracing-subscriber` with a JSON exporter to ship logs.
- [ ] **10.2.1.1** Individual user access to cardholder data is logged. *Artifact:* the vault emits a `tracing::info!(target: "audit", ...)` event on every `tokenize` / `detokenize` / `delete` / `exists` call with `vault_ref`, `caller_identity`, `outcome`. Required field set documented in `compliance/hsm-kms-guidance.md`.
- [ ] **10.3.1** Audit log files are protected from modification. *Artifact:* logs shipped to a write-once destination (S3 + Object Lock; GCS retention policy; Splunk frozen index).
- [ ] **10.4.1** Audit logs are reviewed at least daily. *Artifact:* SIEM with automated alerting on `audit.outcome = "auth_failed"` rate, `audit.outcome = "detokenize_after_delete"` (any occurrence), KMS access from unexpected source IPs.
- [ ] **10.5.1** Audit log history is retained for at least 12 months. *Artifact:* S3 lifecycle / GCS retention / Splunk index TTL set ≥365 days, with the most recent 90 days hot.

## Requirement 11 — Test security of systems and networks regularly

- [ ] **11.3.1** Internal vulnerability scans are performed at least every three months. *Artifact:* operator-supplied scan reports against the vault host and the merchant app server.
- [ ] **11.4.1** External and internal penetration testing is performed at least annually. *Artifact:* operator-supplied pen-test report covering: vault HTTP API, collection iframe XSS resistance, app-server boundary, KMS access path.
- [ ] **11.5.1** Intrusion-detection / intrusion-prevention techniques are used. *Artifact:* operator-supplied IDS / IPS deployment evidence; for cloud deployments, GuardDuty / Security Command Center / Defender for Cloud at minimum.
- [ ] **11.6.1** Change-and-tamper-detection mechanisms are in place for payment pages. *Artifact:* Subresource Integrity (SRI) hashes on all scripts loaded into the collection-iframe page; CSP header `script-src` restricted to the vault's own origin; nightly diff of the served HTML against a known-good baseline.

## Requirement 12 — Support information security with organisational policies and programmes

- [ ] **12.1.1** An information-security policy is established, published, and disseminated. *Artifact:* operator-supplied policy document. Reference this checklist + the `compliance/` directory as the technical annex.
- [ ] **12.3.1** Targeted risk analyses are performed where Customised Approach is used. *Artifact:* the Req. 5 anti-malware analysis (see §5.2.1 above) is one example; document each one.
- [ ] **12.5.1** An inventory of system components in PCI-DSS scope is maintained. *Artifact:* the per-crate scope table in `compliance/scope-map.md`, plus an operator-supplied inventory of hosts / containers / network appliances.
- [ ] **12.6.1** A security-awareness programme is implemented. *Artifact:* operator-supplied training records for all staff with vault access.
- [ ] **12.8.1** A list of third-party service providers with whom CHD is shared is maintained. *Artifact:* operator-supplied TPSP list; minimum entries: acquirer (Hyperswitch / Stripe / Adyen / direct), KMS provider, hosting provider, CDN.
- [ ] **12.9.1** TPSPs acknowledge in writing their responsibility for PCI-DSS. *Artifact:* operator-supplied responsibility matrix signed with each TPSP.
- [ ] **12.10.1** An incident-response plan is in place and tested. *Artifact:* operator-supplied IR runbook covering: suspected KEK compromise (run forced KEK rotation per `compliance/hsm-kms-guidance.md`), suspected token-vault breach (rotate every `VaultRef` by re-tokenizing under a new vault instance), suspected app-server compromise (revoke API keys, rotate mTLS certs).

---

## Cross-cutting evidence inventory

When the QSA arrives, hand them this binder:

1. **Network diagram** showing CDE / connected / out-of-scope segments and the trust boundaries from `compliance/pci-zero-architecture.md`.
2. **`compliance/scope-map.md`** with the cargo build commands the operator actually runs.
3. **`compliance/hsm-kms-guidance.md`** annotated with which backend the operator chose.
4. **Audit-log samples** from the vault (1 week of `tracing` JSON output) and the KMS (1 week of CloudTrail / Cloud Audit / Vault audit / DSM audit output).
5. **CI pipeline output** showing `cargo audit` clean, `cargo clippy --all-targets -- -D warnings` green, `cargo test --workspace` passing.
6. **Pen-test report** dated within the last 12 months.
7. **Incident-response runbook**.
8. **Key-management policy** with rotation schedule and separation-of-duties matrix.
9. **TPSP responsibility matrix** signed with the acquirer, KMS provider, and hosting provider.
10. **This checklist**, completed.
