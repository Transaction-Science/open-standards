# Charter — Transaction Science Open Standards

This charter describes the stewardship pattern that applies to every standard published in this repository. It is short on purpose. The protocols themselves carry the substance; this document captures the operating principles around them.

## The pattern

Each standard in this repository is published under an open licence — Apache-2.0 for code, CC-BY-4.0 for specification text — and is owned by no one. Transaction Science writes the reference implementation, runs the optional hosted services, and stewards the conformance vectors. The wire format and the right to fork are public.

This is the difference between **the protocol** and **the offer**:

- **The protocol** is what anyone can implement. It is free, in both senses of the word.
- **The offer** is what Transaction Science sells: managed deployments of the reference implementation, driver development against the public traits, persistence backends, custody integrations, certification engagement, and the operational scaffolding around real-world deployments.

The protocol does not tax money movement, message movement, or compute movement. The optional services are the revenue.

## Versioning

Each standard versions independently. Tags follow `openpay-v0.1.0`, `smart-byte-v1.0.0-draft.1`, `eoc-v0.2.0`, etc. — the prefix is the subdirectory name. A change in one standard does not bump the others.

## Conformance

A standard is conformant if it round-trips the published wire vectors. The vectors live under each subdirectory and are signed at release time. Reference implementations are conformant by construction; third-party implementations certify themselves by running the public test suites.

## What this charter does not say

- It does not assign exclusive authority. Anyone may fork, and a fork is no less legitimate for being a fork.
- It does not promise a release cadence. Standards advance when the work is done.
- It does not bind future standards to current ones. A standard added later is bound only by its own conformance vectors and the open-licence requirement.

## Adding a standard

A new standard joins this repository by:

1. Landing in a new top-level subdirectory.
2. Carrying its own README, LICENSE, and (where applicable) CONTRIBUTING.md.
3. Being added to the table in the root [`README.md`](README.md).
4. Inheriting the operating principles in this charter.
