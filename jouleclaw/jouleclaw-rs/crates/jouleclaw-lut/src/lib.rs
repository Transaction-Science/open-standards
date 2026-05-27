//! # jouleclaw-lut
//!
//! Literal lookup-table primitive — the **sub-nanojoule pre-cascade
//! resolver**. The "router IS the table" insight from the OpenIE stack
//! made concrete: a hash-keyed exact-match LUT of explicitly-registered
//! pre-baked answers that NEVER get evicted.
//!
//! ## Doctrine
//!
//! JouleClaw's L0–L4 cascade resolves every operation in cost order:
//!
//! 1. **LUT** (this crate) — registered, never-evicted, ~1 nJ per probe.
//! 2. **L0:Cache** ([`jouleclaw_cascade::l0_cache::L0Cache`]) —
//!    content-addressed cache with eviction.
//! 3. **L1:Lawful** — deterministic primitives (gcd, regex, parse, …).
//! 4. **L2:Embed** — Matryoshka embeddings + intent classifier.
//! 5. **L3:Model** — local SSM / 1-bit / multimodal weights.
//! 6. **L4:Wire** — remote inference, last resort.
//!
//! The LUT sits **beneath** [`L0Cache`] because L0Cache treats every
//! entry as cacheable (insert + evict). The LUT is for queries you have
//! pre-decided the answer for — the cascade must NEVER recompute them.
//!
//! ## Normalisation
//!
//! Inputs are normalised before hashing so `"  GCD 12 8  "` and
//! `"gcd 12 8"` collide. See [`normalize::normalize`]. Normalisation is
//! identical on `register` and `try_lookup`.
//!
//! ## Bulk loading
//!
//! Pre-baked tables ship as TOML or CSV. See [`Lut::load_toml`] and
//! [`Lut::load_csv`].
//!
//! ## Cascade integration
//!
//! [`Lut`] implements [`jouleclaw_cascade::tier::Tier`] with
//! `TierId::L0`, surfacing as the cache-class tier with `joules = 1e-9`
//! and `confidence = 1.0` on hit. Register it as the **first** tier in
//! the cascade so it short-circuits before any other resolver runs.
//!
//! ## Crate invariants
//!
//! - `#![forbid(unsafe_code)]`
//! - No `.unwrap()` outside `#[cfg(test)]`
//! - Flat layout: every `pub mod X;` has a matching `src/X.rs`

#![forbid(unsafe_code)]

pub mod types;
pub mod normalize;
pub mod lut;
pub mod load;
pub mod tier;

pub use types::{LutKey, LutEntry, LutHit, LutError};
pub use normalize::normalize;
pub use lut::Lut;
