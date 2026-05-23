# OpenPay PCI-DSS Readiness Package

This directory is the PCI-DSS v4.0.1 readiness package for an OpenPay deployment. It maps the cardholder-data (CHD) and sensitive-authentication-data (SAD) handling of every crate in this workspace onto the twelve PCI-DSS v4.0.1 requirements, describes a deployment topology where merchant application servers never touch a primary account number (PAN), gives integration guidance for AWS KMS, GCP KMS, HashiCorp Vault, and Fortanix DSM, and provides a checklist a Qualified Security Assessor (QSA) can run against an OpenPay deployment before formal assessment. None of this material has been blessed by the PCI Security Standards Council, a card scheme, or a QSA; it is an engineering self-assessment intended to make a real assessment short.

## Files

```
compliance/
├── README.md                    this file
├── scope-map.md                 per-crate PCI-DSS scope table; how the
│                                 op-core `pci-scope` feature gates raw-PAN
│                                 handling at the code-of-the-stack boundary
├── pci-zero-architecture.md     deployment topology where merchant servers
│                                 never see PAN; references examples/pci-zero
├── hsm-kms-guidance.md          AWS KMS / GCP KMS / HashiCorp Vault /
│                                 Fortanix DSM integration, modelled on the
│                                 existing op-rails-crypto::EvmSigner trait
└── qsa-checklist.md             pre-assessment checklist mapping the twelve
                                 PCI-DSS v4.0.1 requirements to artifacts
                                 in this repository
```

## Companion code

```
examples/pci-zero/               runnable example: vault tokenization →
                                 orchestrator → card rail with no raw PAN
                                 crossing process boundaries on the merchant
                                 application server
```
