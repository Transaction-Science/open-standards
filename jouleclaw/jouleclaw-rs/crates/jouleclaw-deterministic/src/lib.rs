//! JouleClaw deterministic standard library.
//!
//! A drop-in lexicon of L1 primitives that plug into
//! [`jouleclaw_cascade::LawfulRegistry`]. The doctrine made concrete:
//!
//! - **Deterministic-first** — every primitive is pure compute. Same
//!   input, same output, every time. No stochastic models.
//! - **Search-first** — each primitive recognises a *specific* query
//!   syntax. If the syntax doesn't match, the primitive returns `None`
//!   and the registry walks on. The first primitive that handles the
//!   query class wins; there is no fallback into a model.
//! - **LUT-over-compute** — where a static table answers the question
//!   (ISO-3166 codes, ISO-4217 currency symbols, leap-year cycles)
//!   the primitive looks it up rather than computing it.
//! - **Inference-last** — none of these primitives call a model. A
//!   query that one of these primitives can answer never reaches L2,
//!   L3, or L4.
//!
//! ## What ships here
//!
//! Categories (one module each), `register_*()` registries for each
//! plus [`register_all`] that fuses them all:
//!
//! - [`arithmetic`] — gcd, lcm, factorial, fib, primality, factor,
//!   small-expression eval, abs, sign
//! - [`units`] — F/C/K temperature, miles/km, kg/lb, m/ft, L/gal
//! - [`bases`] — dec ↔ hex/bin/oct
//! - [`dates`] — day-of-week, days-between, leap year, ISO weekday,
//!   add-days
//! - [`strings`] — length, case, reverse, word-count, substring tests
//! - [`hashes`] — BLAKE3, SHA-256, MD5
//! - [`lookups`] — ISO-3166-1 alpha-2 country names, ISO-4217 currency
//!   names + symbols
//!
//! ## Query syntax
//!
//! Each primitive's `try_resolve` recognises one or two human-readable
//! query forms. Whitespace is forgiven; the keyword prefix is
//! case-insensitive. If a primitive's numeric argument fails to parse,
//! the primitive returns `None` (try-the-next-primitive) rather than
//! producing a wrong answer.
//!
//! ## Cost class
//!
//! Each primitive declares a microjoule cost in the 50–200 μJ range.
//! These are *declared* costs, accounted against the joule budget on
//! a successful resolution — actual measured energy is carried in the
//! receipt's `tools_touched` entry by the runtime, not here.

#![forbid(unsafe_code)]

use jouleclaw_cascade::LawfulRegistry;
use std::sync::Arc;

pub mod arithmetic;
pub mod bases;
pub mod dates;
pub mod hashes;
pub mod lookups;
pub mod strings;
pub mod units;

/// Build a registry with every primitive this crate ships.
///
/// The registry is walked in insertion order; categories are
/// registered in the order arithmetic → units → bases → dates →
/// strings → hashes → lookups. Within each category primitives are
/// registered in a stable order — see the per-category `register_*`
/// functions.
pub fn register_all() -> LawfulRegistry {
    let mut r = LawfulRegistry::new();
    for p in arithmetic::primitives() {
        r = r.register(p);
    }
    for p in units::primitives() {
        r = r.register(p);
    }
    for p in bases::primitives() {
        r = r.register(p);
    }
    for p in dates::primitives() {
        r = r.register(p);
    }
    for p in strings::primitives() {
        r = r.register(p);
    }
    for p in hashes::primitives() {
        r = r.register(p);
    }
    for p in lookups::primitives() {
        r = r.register(p);
    }
    r
}

fn build(prims: Vec<Arc<dyn jouleclaw_cascade::LawfulPrimitive>>) -> LawfulRegistry {
    let mut r = LawfulRegistry::new();
    for p in prims {
        r = r.register(p);
    }
    r
}

/// Arithmetic-only registry.
pub fn register_arithmetic() -> LawfulRegistry {
    build(arithmetic::primitives())
}

/// Units-only registry.
pub fn register_units() -> LawfulRegistry {
    build(units::primitives())
}

/// Base-conversion-only registry.
pub fn register_bases() -> LawfulRegistry {
    build(bases::primitives())
}

/// Date-arithmetic-only registry.
pub fn register_dates() -> LawfulRegistry {
    build(dates::primitives())
}

/// String-operations-only registry.
pub fn register_strings() -> LawfulRegistry {
    build(strings::primitives())
}

/// Hash-only registry.
pub fn register_hashes() -> LawfulRegistry {
    build(hashes::primitives())
}

/// Static-lookup-only registry.
pub fn register_lookups() -> LawfulRegistry {
    build(lookups::primitives())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_all_has_at_least_thirty_primitives() {
        let r = register_all();
        assert!(
            r.len() >= 30,
            "register_all should ship at least 30 primitives, got {}",
            r.len()
        );
    }

    #[test]
    fn per_category_registries_are_nonempty() {
        assert!(!register_arithmetic().is_empty());
        assert!(!register_units().is_empty());
        assert!(!register_bases().is_empty());
        assert!(!register_dates().is_empty());
        assert!(!register_strings().is_empty());
        assert!(!register_hashes().is_empty());
        assert!(!register_lookups().is_empty());
    }

    #[test]
    fn unknown_query_returns_none() {
        let r = register_all();
        assert!(r.try_resolve("write a haiku about owls").is_none());
    }
}
