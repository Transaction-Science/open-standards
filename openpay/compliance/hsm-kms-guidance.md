# HSM / KMS Integration Guidance

This document gives operator-facing guidance for plugging an HSM or cloud KMS into an OpenPay deployment. The pattern follows the `EvmSigner` trait in `op-rails-crypto` (`crates/op-rails-crypto/src/signer.rs`): the OpenPay code builds an unsigned operation, hands it to an operator-supplied trait, and lets the operator's chosen backend perform the privileged cryptography.

## The `KeyProvider` adapter trait

OpenPay does not ship a `KeyProvider` trait in the workspace because the `Vault` trait already abstracts the entire storage-side concern. Operators implementing a vault against a KMS or HSM should follow the same shape as `EvmSigner`:

```rust
// Pattern, not shipped code — operators add this to their own vault crate.
pub trait KeyProvider: Send + Sync {
    /// Encrypt a fresh DEK (data-encryption key) under the KEK.
    fn wrap_dek(&self, plaintext_dek: &[u8; 32]) -> Result<Vec<u8>>;

    /// Decrypt a previously wrapped DEK.
    fn unwrap_dek(&self, wrapped_dek: &[u8]) -> Result<[u8; 32]>;

    /// Rotate the KEK. Returns the new key version identifier.
    fn rotate_kek(&self) -> Result<KeyVersion>;

    /// Active KEK version, for emitting on audit log entries.
    fn active_kek_version(&self) -> Result<KeyVersion>;
}
```

The operator's vault impl then performs envelope encryption: generate a 32-byte DEK with `getrandom`, encrypt the PAN with AES-256-GCM-SIV under the DEK, call `KeyProvider::wrap_dek` to wrap the DEK under the KEK, store `(wrapped_dek, ciphertext, nonce, kek_version)` keyed by `VaultRef`. On detokenize, the reverse.

The four backends below all fit this shape.

## AWS KMS

### Setup

1. Create a customer-managed CMK in the AWS region hosting the vault. Use a symmetric `SYMMETRIC_DEFAULT` key for envelope encryption; multi-region keys are appropriate if the vault is multi-AZ.
2. Restrict the CMK key policy to the IAM role attached to the vault EC2 / ECS / EKS workload identity. Statement: `kms:Encrypt`, `kms:Decrypt`, `kms:GenerateDataKey`, `kms:DescribeKey`. Deny `kms:ScheduleKeyDeletion` and `kms:DisableKey` to non-break-glass roles.
3. Enable CloudTrail data events on the CMK. Ship the events to a separate AWS account (the *log archive* account in a Control Tower setup) so vault admins cannot tamper with their own audit trail.
4. The operator's `KeyProvider` impl uses the AWS SDK for Rust (`aws-sdk-kms`) and authenticates via IMDSv2 / IRSA / workload identity — never a long-lived access key.

### Latency

- `Encrypt` / `Decrypt` over a 32-byte DEK: typically 5-15 ms intra-region, 95th percentile under 30 ms. AWS does not publish an SLA on KMS latency; budget 50 ms in your detokenize path.
- For high-throughput vaults, use the AWS KMS *Data Key Caching* pattern: cache unwrapped DEKs in memory for up to 60 seconds, bounded by an LRU cache of e.g. 1024 entries. This keeps amortised KMS cost at one call per minute per hot DEK.

### Key rotation

- Enable AWS-managed annual KEK rotation (`kms:EnableKeyRotation`). AWS retains every prior key version internally; `Decrypt` automatically uses the right version. No application changes required.
- For *forced* rotation (suspected compromise), generate a new CMK, re-wrap every DEK by reading each ciphertext, decrypting under the old CMK, re-wrapping under the new CMK, writing back, and finally scheduling the old CMK for deletion (30-day window).

### Audit log requirements

Each `wrap_dek` / `unwrap_dek` call must produce a CloudTrail entry containing the calling principal, the source IP (the vault host), the CMK ARN, the encryption context (operator should pass `{"vault_ref": "<token>", "tenant": "<id>"}` so the audit log identifies *which* PAN was touched), and the timestamp. Ship CloudTrail logs to S3 + CloudWatch Logs; retain for at least one year (PCI-DSS Req. 10.5.1).

## GCP KMS (Cloud KMS)

### Setup

1. Create a key ring in the region hosting the vault. Create a `SOFTWARE` or `HSM` protection-level key inside it. For PCI scope choose `HSM` (FIPS 140-2 Level 3).
2. Bind the vault workload's Google service account to `roles/cloudkms.cryptoKeyEncrypterDecrypter` on the key. Do not grant `cloudkms.cryptoKeys.create` or `cloudkms.cryptoKeys.destroy` to the workload.
3. Enable Cloud Audit Logs (Admin Activity *and* Data Access) on the key ring. Sink to a separate BigQuery dataset or Cloud Storage bucket in a different project.
4. Authenticate the vault workload via Workload Identity Federation (GKE) or the metadata server (GCE) — never a downloaded JSON key file.

### Latency

- `Encrypt` / `Decrypt`: 10-30 ms typical, 100 ms 99th percentile. HSM-protected keys add ~10 ms vs software keys. Budget 100 ms in the detokenize path.
- Cloud KMS does not have a published data-key caching SDK pattern; implement caching identically to the AWS KMS guidance (LRU + 60-second TTL).

### Key rotation

- Set `rotationPeriod = 7776000s` (90 days) on the key. Cloud KMS creates a new primary key version automatically; older versions remain decryptable until explicitly destroyed.
- For forced rotation, create a new key version manually (`gcloud kms keys versions create`), wait for it to become primary, and then run the re-wrap loop.

### Audit log requirements

Cloud Audit Logs record both the call and the encryption context (Cloud KMS calls it *additional authenticated data*; pass the same `{"vault_ref": ..., "tenant": ...}` as for AWS). Retain for one year minimum; PCI-DSS Req. 10.5.1.

## HashiCorp Vault (Transit secret engine)

### Setup

1. Deploy HashiCorp Vault in HA mode (Raft storage backend, 5-node quorum). Enable the *Transit* secret engine at a mount path such as `transit-openpay/`.
2. Create a key with `vault write -f transit-openpay/keys/openpay-cde type=aes256-gcm96 derived=true exportable=false`. `derived=true` requires the caller to supply a per-`VaultRef` context (giving you per-token key derivation for free); `exportable=false` ensures the key never leaves the Vault server.
3. Create a policy granting `update` on `transit-openpay/encrypt/openpay-cde` and `transit-openpay/decrypt/openpay-cde`, and deny everything else.
4. Authenticate the OpenPay vault workload via the AppRole, Kubernetes, or AWS-IAM auth method. Lease TTL: 1 hour, renew before expiry. No root tokens in production.
5. Audit-log to a file device that ships to your SIEM. Enable the `enable_response_wrapping` audit option so token IDs are obfuscated in logs.

### Latency

- `transit/encrypt` / `transit/decrypt` over 32 bytes: 2-8 ms typical against a same-VPC HashiCorp Vault cluster. Lower than cloud KMS because there is no AWS / GCP control-plane hop.
- Budget 25 ms in the detokenize path.

### Key rotation

- `vault write -f transit-openpay/keys/openpay-cde/rotate` creates a new key version. All future encrypts use the new version; decrypts of older ciphertexts continue to work because the ciphertext header carries the version number.
- For forced rotation, also run `vault write transit-openpay/keys/openpay-cde/rewrap` to mass-re-encrypt all data under the new version (HashiCorp Vault has a built-in rewrap API).

### Audit log requirements

The Transit engine logs every encrypt / decrypt with the AppRole identity, source IP, request ID, and the `context` (`derived=true` makes this mandatory). Ship audit logs to a write-once store (S3 + Object Lock, or a dedicated log-archive instance). Retain one year minimum.

## Fortanix DSM (Data Security Manager)

### Setup

1. Provision a Fortanix DSM cluster — SaaS or self-hosted. SaaS is FIPS 140-2 Level 3 certified; self-hosted on Intel SGX is FIPS 140-2 Level 3 (HSM mode).
2. Create an *App* (service identity) for the OpenPay vault workload. Use *App API key* or *certificate authentication*; do not use username + password.
3. Inside the App's group, create an AES-256 *Security Object* with operations `ENCRYPT`, `DECRYPT`, `WRAPKEY`, `UNWRAPKEY` allowed and `EXPORT` denied.
4. Configure quorum approvals if your operational model requires two-person control for KEK rotation or deletion.
5. The Fortanix REST API or its PKCS#11 / KMIP interfaces all map to the `KeyProvider` shape above; choose REST for cross-cloud portability, PKCS#11 for legacy compatibility.

### Latency

- REST API to Fortanix SaaS: 20-50 ms typical (single-region), 100 ms 99th. Self-hosted on-prem on SGX: 5-15 ms.
- Budget 100 ms in the detokenize path for SaaS, 25 ms for self-hosted.

### Key rotation

- DSM supports key versioning and automatic rotation policies. Set `rotation_policy.interval_days = 90` on the Security Object.
- Forced rotation: create a new key version manually, run the operator's re-wrap job. DSM keeps old versions decryptable per the *Key state* lifecycle (`Active → Suspended → Deactivated → Destroyed`).

### Audit log requirements

DSM emits an audit log entry for every Security Object operation, including the calling App, source IP, operation, and Security Object UUID. Forward logs via syslog to your SIEM; DSM supports CEF and JSON formats. Retain one year minimum.

## Cross-provider comparison

| Provider | FIPS level | Typical latency | Caching pattern | Rotation cadence | Auth |
|---|---|---|---|---|---|
| AWS KMS | 140-2 L3 (HSM-backed CMKs) | 5-15 ms | LRU + 60 s TTL | Annual auto, manual on-demand | IAM role |
| GCP KMS | 140-2 L3 (HSM tier) | 10-30 ms | LRU + 60 s TTL | 90-day auto | Workload Identity |
| HashiCorp Vault Transit | 140-2 L1 (default) / L2 (with HSM auto-unseal) | 2-8 ms | not needed (already fast) | Manual, mass-rewrap supported | AppRole / Kubernetes / IAM |
| Fortanix DSM | 140-2 L3 (SGX) | 20-50 ms (SaaS) / 5-15 ms (self-hosted) | LRU + 60 s TTL | 90-day auto, quorum-gated manual | API key / cert |

## Common pitfalls

- **Caching unwrapped DEKs beyond their TTL.** PCI-DSS Req. 3.7.1 expects DEKs to live no longer than necessary. Bound the cache at 60 seconds.
- **Pinning the wrong KEK version into ciphertext metadata.** Always record `kek_version` alongside the wrapped DEK; mass-rewrap operations need it.
- **Logging encryption context that contains the PAN.** The encryption context becomes part of CloudTrail / Cloud Audit / Vault audit logs, which are *not* CDE-scoped storage. Pass only `VaultRef` and `tenant_id`, never PAN-derived material.
- **Using a single KMS key for multiple tenants.** If you operate a multi-tenant vault, derive per-tenant DEKs via the KMS `Derive` operation (AWS KMS supports this via key context; HashiCorp Vault Transit supports it via `derived=true`).
- **Treating the KMS as the CDE boundary.** The KMS holds the KEK, not the PAN. The CDE is wherever the *plaintext PAN* lives, which is the vault process during the brief window between `unwrap_dek` and AES decryption + the subsequent acquirer post.
