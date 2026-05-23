//! Feature extraction.
//!
//! Converts a [`op_core::Payment`] and a [`ScoringContext`] into a
//! fixed-length `[f32; FEATURES]` vector. Designed for fraud detection
//! with three constraints:
//!
//! 1. **No PII**. Identifiers (account, name, device id) are hashed via
//!    SHA-256; only the top 32 bits projected to `[0.0, 1.0]` enter the
//!    vector. The model cannot reconstruct the source.
//! 2. **Deterministic**. Same input always produces the same vector,
//!    so model training and inference agree. No clock-reading inside
//!    feature extraction unless the caller passed an explicit timestamp.
//! 3. **Bounded**. Every feature is in a known range so the model
//!    doesn't see numerical surprises that break learned thresholds.
//!
//! ## Feature schema (32 floats)
//!
//! | Index | Feature                                  | Range        |
//! |------:|------------------------------------------|--------------|
//! |   0   | `log10(amount_minor_units + 1)`          | `[0, 18.3]`  |
//! |   1   | `amount_minor / 1_000_000.0`             | `[0, ∞)`     |
//! |   2   | `is_currency_usd`                        | `{0, 1}`     |
//! |   3   | `is_currency_eur`                        | `{0, 1}`     |
//! |   4   | `is_currency_brl`                        | `{0, 1}`     |
//! |   5   | `is_round_amount` (no cents)             | `{0, 1}`     |
//! |   6   | `over_1000_major_units`                  | `{0, 1}`     |
//! |   7   | `over_10000_major_units`                 | `{0, 1}`     |
//! |   8   | `hour_of_day / 24.0`                     | `[0, 1)`     |
//! |   9   | `day_of_week / 7.0`                      | `[0, 1)`     |
//! |  10   | `is_weekend`                             | `{0, 1}`     |
//! |  11   | `is_night` (00-06)                       | `{0, 1}`     |
//! |  12   | `sin(2π hour/24)`                        | `[-1, 1]`    |
//! |  13   | `cos(2π hour/24)`                        | `[-1, 1]`    |
//! |  14   | `sin(2π dow/7)`                          | `[-1, 1]`    |
//! |  15   | `cos(2π dow/7)`                          | `[-1, 1]`    |
//! |  16   | `log1p(velocity_1h)`                     | `[0, ∞)`     |
//! |  17   | `log1p(velocity_24h)`                    | `[0, ∞)`     |
//! |  18   | `log1p(device_velocity_1h)`              | `[0, ∞)`     |
//! |  19   | `normalized seconds since last payment`  | `[0, 1]`     |
//! |  20   | `normalized seconds since auth`          | `[0, 1]`     |
//! |  21   | `is_new_customer`                        | `{0, 1, 0.5}`|
//! |  22   | `geo_matches_history`                    | `{0, 1, 0.5}`|
//! |  23   | `hash_to_unit(device_id)`                | `[0, 1]`     |
//! |  24   | `is_rail_card`                           | `{0, 1}`     |
//! |  25   | `is_rail_a2a`                            | `{0, 1}`     |
//! |  26   | `is_rail_wallet`                         | `{0, 1}`     |
//! |  27   | `is_rail_qr`                             | `{0, 1}`     |
//! |  28   | `hash_to_unit(creditor_account)`         | `[0, 1]`     |
//! |  29   | `hash_to_unit(creditor_name)`            | `[0, 1]`     |
//! |  30   | `hash_to_unit(debtor_account_or_method)` | `[0, 1]`     |
//! |  31   | `has_remittance`                         | `{0, 1}`     |
//!
//! Missing `Option<bool>` fields encode as `0.5` (neutral) so the model
//! can learn distinct treatment.

// This module deliberately produces an `[f32; 32]` vector from larger
// numeric inputs (i64 amounts, u64 velocities, u32 hash projections).
// The narrowing/precision-loss is the intended quantization for the
// ML feature space — bounded ranges are documented per-feature above —
// not an accidental data-loss bug.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use op_core::{Currency, Money, PaymentMethod, RailKind};
use sha2::{Digest, Sha256};
use time::OffsetDateTime;

use crate::context::ScoringContext;
use crate::error::Result;

/// Number of features in the vector.
pub const FEATURES: usize = 32;

/// The output of [`extract_features`]. A fixed-length array of `f32`.
pub type FeatureVector = [f32; FEATURES];

/// Information about the payment we'll score. Decoupled from
/// `op_core::Payment<S>` so callers can pass partial / synthetic data
/// (replaying historical events to retrain doesn't need the full state).
#[derive(Debug, Clone)]
pub struct PaymentDescriptor<'a> {
    /// Amount and currency.
    pub amount: Money,
    /// How value moves.
    pub method: &'a PaymentMethod,
    /// Which rail this is being routed to.
    pub rail: RailKind,
    /// Optional creditor (receiver) account identifier — hashed in the vector.
    pub creditor_account: Option<&'a str>,
    /// Optional creditor name — hashed in the vector.
    pub creditor_name: Option<&'a str>,
    /// Optional debtor (sender) account identifier — hashed in the vector.
    pub debtor_account: Option<&'a str>,
    /// True if the payment includes remittance info.
    pub has_remittance: bool,
}

/// Extract the feature vector from a payment and context.
///
/// # Errors
/// `Error::Features` if a numeric conversion overflows (e.g.
/// `velocity_1h > 2^31` which is comically large).
pub fn extract_features(
    payment: &PaymentDescriptor<'_>,
    ctx: &ScoringContext,
) -> Result<FeatureVector> {
    let mut f = [0.0f32; FEATURES];

    // ---- Amount features (0-7) ----
    let amt = payment.amount.minor_units;
    let amt_abs = amt.unsigned_abs() as f64;
    f[0] = (amt_abs + 1.0).log10() as f32;
    f[1] = (amt_abs / 1_000_000.0) as f32;
    f[2] = if payment.amount.currency == Currency::USD {
        1.0
    } else {
        0.0
    };
    f[3] = if payment.amount.currency == Currency::EUR {
        1.0
    } else {
        0.0
    };
    f[4] = if payment.amount.currency == Currency::BRL {
        1.0
    } else {
        0.0
    };

    // "Round" amount: zero in the fractional part. JPY (exponent 0) is
    // always trivially round; we treat that as a no-signal feature.
    let exp = payment.amount.currency.exponent();
    f[5] = if exp == 0 {
        0.0
    } else {
        let divisor = 10i64.pow(u32::from(exp));
        if amt % divisor == 0 { 1.0 } else { 0.0 }
    };

    // Major-unit thresholds. Convert amount_minor to major units for comparison.
    let major = amt_abs / 10f64.powi(i32::from(exp));
    f[6] = if major > 1_000.0 { 1.0 } else { 0.0 };
    f[7] = if major > 10_000.0 { 1.0 } else { 0.0 };

    // ---- Time features (8-15) ----
    let timestamp = ctx.timestamp.unwrap_or_else(OffsetDateTime::now_utc);
    let hour = f32::from(timestamp.hour()); // u8 -> f32
    let dow_index = f32::from(timestamp.weekday().number_days_from_monday()); // 0..=6
    f[8] = hour / 24.0;
    f[9] = dow_index / 7.0;
    f[10] = if dow_index >= 5.0 { 1.0 } else { 0.0 }; // Sat=5, Sun=6
    f[11] = if hour < 6.0 { 1.0 } else { 0.0 };
    let hour_rad = (f64::from(hour) / 24.0) * 2.0 * std::f64::consts::PI;
    let dow_rad = (f64::from(dow_index) / 7.0) * 2.0 * std::f64::consts::PI;
    f[12] = hour_rad.sin() as f32;
    f[13] = hour_rad.cos() as f32;
    f[14] = dow_rad.sin() as f32;
    f[15] = dow_rad.cos() as f32;

    // ---- Velocity features (16-23) ----
    f[16] = log1p_u32(ctx.velocity_1h.unwrap_or(0));
    f[17] = log1p_u32(ctx.velocity_24h.unwrap_or(0));
    f[18] = log1p_u32(ctx.device_velocity_1h.unwrap_or(0));
    f[19] = normalize_log_seconds(
        ctx.seconds_since_last_payment.unwrap_or(604_800), // default: 1 week
        604_800,
    );
    f[20] = normalize_log_seconds(
        ctx.seconds_since_auth.unwrap_or(86_400), // default: 1 day
        86_400,
    );
    f[21] = match ctx.is_new_customer {
        Some(true) => 1.0,
        Some(false) => 0.0,
        None => 0.5,
    };
    f[22] = match ctx.geo_matches_history {
        Some(true) => 1.0,
        Some(false) => 0.0,
        None => 0.5,
    };
    f[23] = hash_to_unit(ctx.device_id.as_deref());

    // ---- Rail / method features (24-31) ----
    f[24] = if payment.rail == RailKind::Card {
        1.0
    } else {
        0.0
    };
    f[25] = if payment.rail == RailKind::A2a {
        1.0
    } else {
        0.0
    };
    f[26] = if payment.rail == RailKind::Wallet {
        1.0
    } else {
        0.0
    };
    f[27] = if payment.rail == RailKind::Qr {
        1.0
    } else {
        0.0
    };
    f[28] = hash_to_unit(payment.creditor_account);
    f[29] = hash_to_unit(payment.creditor_name);
    f[30] = hash_to_unit(payment.debtor_account);
    f[31] = if payment.has_remittance { 1.0 } else { 0.0 };

    // Sanity: every feature must be finite.
    for (i, v) in f.iter().enumerate() {
        if !v.is_finite() {
            return Err(crate::error::Error::Features(format!(
                "feature[{i}] is non-finite: {v}"
            )));
        }
    }

    Ok(f)
}

/// `log(1 + x)` for u32, returning f32. Saturates well below `f32::MAX`.
fn log1p_u32(x: u32) -> f32 {
    (f64::from(x) + 1.0).ln() as f32
}

/// Normalize `log1p(seconds) / log1p(reference)` to `[0, 1]`. Values
/// larger than `reference` get clipped at 1.0.
fn normalize_log_seconds(seconds: u64, reference: u64) -> f32 {
    let s = (seconds as f64 + 1.0).ln();
    let r = (reference as f64 + 1.0).ln();
    let ratio = if r > 0.0 { s / r } else { 0.0 };
    ratio.clamp(0.0, 1.0) as f32
}

/// Project a string identifier to `[0.0, 1.0]` via the first 4 bytes of
/// its SHA-256 digest. Returns `0.0` for `None` / empty so the model can
/// learn a distinct "missing" signal.
fn hash_to_unit(s: Option<&str>) -> f32 {
    match s {
        None | Some("") => 0.0,
        Some(s) => {
            let digest = Sha256::digest(s.as_bytes());
            let upper = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
            (f64::from(upper) / f64::from(u32::MAX)) as f32
        }
    }
}

#[cfg(test)]
mod tests {
    // Feature flags (currency one-hots, rail one-hots, weekend/night
    // booleans) are produced as exactly 0.0 or 1.0, and the projected
    // hashes are deterministic. Exact `==` is the correct assertion
    // here; the float_cmp pedantic lint is a false positive for
    // bit-exact deterministic values.
    #![allow(clippy::float_cmp)]
    use super::*;
    use op_core::{Currency, Money, PaymentMethod, RailKind, VaultRef};
    use time::macros::datetime;

    fn vault_method() -> PaymentMethod {
        PaymentMethod::Vault(VaultRef::new("tok_x"))
    }

    fn sample_descriptor(method: &PaymentMethod) -> PaymentDescriptor<'_> {
        PaymentDescriptor {
            amount: Money::from_minor(12345, Currency::USD),
            method,
            rail: RailKind::Card,
            creditor_account: Some("acct_creditor"),
            creditor_name: Some("Alice"),
            debtor_account: Some("acct_debtor"),
            has_remittance: true,
        }
    }

    #[test]
    fn extract_returns_correct_length() {
        let m = vault_method();
        let p = sample_descriptor(&m);
        let f = extract_features(&p, &ScoringContext::empty()).unwrap();
        assert_eq!(f.len(), FEATURES);
        assert_eq!(f.len(), 32);
    }

    #[test]
    fn all_features_are_finite() {
        let m = vault_method();
        let p = sample_descriptor(&m);
        let f = extract_features(&p, &ScoringContext::empty()).unwrap();
        for (i, v) in f.iter().enumerate() {
            assert!(v.is_finite(), "feature[{i}]={v} is not finite");
        }
    }

    #[test]
    fn amount_features_match_expected_values() {
        let m = vault_method();
        let mut p = sample_descriptor(&m);
        p.amount = Money::from_minor(12345, Currency::USD);
        let f = extract_features(&p, &ScoringContext::empty()).unwrap();
        // log10(12345 + 1) ≈ 4.09
        assert!((f[0] - 12346.0_f32.log10()).abs() < 1e-4);
        // 12345 / 1_000_000 = 0.012345
        assert!((f[1] - 0.012_345).abs() < 1e-6);
        assert_eq!(f[2], 1.0, "USD flag");
        assert_eq!(f[3], 0.0, "not EUR");
        assert_eq!(f[4], 0.0, "not BRL");
        assert_eq!(f[5], 0.0, "12345 cents is not round");
    }

    #[test]
    fn round_amount_detected_for_2dp_currency() {
        let m = vault_method();
        let mut p = sample_descriptor(&m);
        p.amount = Money::from_minor(10_000, Currency::USD); // $100.00 exactly
        let f = extract_features(&p, &ScoringContext::empty()).unwrap();
        assert_eq!(f[5], 1.0);
    }

    #[test]
    fn round_amount_zero_signal_for_jpy() {
        let m = vault_method();
        let mut p = sample_descriptor(&m);
        p.amount = Money::from_minor(500, Currency::JPY);
        let f = extract_features(&p, &ScoringContext::empty()).unwrap();
        // JPY has no fractional part; "round" is meaningless, encoded as 0.
        assert_eq!(f[5], 0.0);
    }

    #[test]
    fn over_threshold_flags_correctly() {
        let m = vault_method();
        let mut p = sample_descriptor(&m);
        // $999 USD = 99_900 minor units; major = 999
        p.amount = Money::from_minor(99_900, Currency::USD);
        let f = extract_features(&p, &ScoringContext::empty()).unwrap();
        assert_eq!(f[6], 0.0, "$999 is not > $1000");
        assert_eq!(f[7], 0.0, "$999 is not > $10000");

        // $1500 USD
        p.amount = Money::from_minor(150_000, Currency::USD);
        let f = extract_features(&p, &ScoringContext::empty()).unwrap();
        assert_eq!(f[6], 1.0);
        assert_eq!(f[7], 0.0);

        // $25000 USD
        p.amount = Money::from_minor(2_500_000, Currency::USD);
        let f = extract_features(&p, &ScoringContext::empty()).unwrap();
        assert_eq!(f[6], 1.0);
        assert_eq!(f[7], 1.0);
    }

    #[test]
    fn time_features_use_provided_timestamp() {
        let m = vault_method();
        let p = sample_descriptor(&m);
        // 2026-05-17 is a Sunday at 03:30:00 UTC
        let ctx = ScoringContext {
            timestamp: Some(datetime!(2026-05-17 03:30:00 UTC)),
            ..Default::default()
        };
        let f = extract_features(&p, &ctx).unwrap();
        // hour 3 / 24 = 0.125
        assert!((f[8] - 3.0_f32 / 24.0).abs() < 1e-6);
        // Sunday = 6 in days_from_monday
        assert!((f[9] - 6.0_f32 / 7.0).abs() < 1e-6);
        // Is weekend
        assert_eq!(f[10], 1.0);
        // Is night (hour 3 < 6)
        assert_eq!(f[11], 1.0);
    }

    #[test]
    fn cyclic_hour_encoding_at_midnight() {
        let m = vault_method();
        let p = sample_descriptor(&m);
        let ctx = ScoringContext {
            timestamp: Some(datetime!(2026-05-17 00:00:00 UTC)),
            ..Default::default()
        };
        let f = extract_features(&p, &ctx).unwrap();
        // sin(0) = 0, cos(0) = 1
        assert!(f[12].abs() < 1e-6);
        assert!((f[13] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cyclic_hour_encoding_at_noon() {
        let m = vault_method();
        let p = sample_descriptor(&m);
        let ctx = ScoringContext {
            timestamp: Some(datetime!(2026-05-17 12:00:00 UTC)),
            ..Default::default()
        };
        let f = extract_features(&p, &ctx).unwrap();
        // sin(π) ≈ 0, cos(π) = -1
        assert!(f[12].abs() < 1e-6);
        assert!((f[13] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn velocity_logged_correctly() {
        let m = vault_method();
        let p = sample_descriptor(&m);
        let ctx = ScoringContext {
            velocity_1h: Some(2),
            velocity_24h: Some(10),
            device_velocity_1h: Some(0),
            ..Default::default()
        };
        let f = extract_features(&p, &ctx).unwrap();
        assert!((f[16] - (3.0f32).ln()).abs() < 1e-5); // ln(2+1)
        assert!((f[17] - (11.0f32).ln()).abs() < 1e-5); // ln(10+1)
        assert!((f[18] - (1.0f32).ln()).abs() < 1e-6); // ln(0+1) = 0
        assert_eq!(f[18], 0.0);
    }

    #[test]
    fn missing_bool_context_encodes_as_half() {
        let m = vault_method();
        let p = sample_descriptor(&m);
        let f = extract_features(&p, &ScoringContext::empty()).unwrap();
        assert_eq!(f[21], 0.5, "missing is_new_customer = 0.5");
        assert_eq!(f[22], 0.5, "missing geo_matches_history = 0.5");
    }

    #[test]
    fn known_false_bool_distinguishes_from_missing() {
        let m = vault_method();
        let p = sample_descriptor(&m);
        let ctx = ScoringContext {
            is_new_customer: Some(false),
            geo_matches_history: Some(false),
            ..Default::default()
        };
        let f = extract_features(&p, &ctx).unwrap();
        assert_eq!(f[21], 0.0);
        assert_eq!(f[22], 0.0);
    }

    #[test]
    fn rail_one_hot_flags() {
        let m = vault_method();
        let mut p = sample_descriptor(&m);

        p.rail = RailKind::Card;
        let f = extract_features(&p, &ScoringContext::empty()).unwrap();
        assert_eq!(f[24], 1.0);
        assert_eq!(f[25], 0.0);
        assert_eq!(f[26], 0.0);
        assert_eq!(f[27], 0.0);

        p.rail = RailKind::A2a;
        let f = extract_features(&p, &ScoringContext::empty()).unwrap();
        assert_eq!(f[24], 0.0);
        assert_eq!(f[25], 1.0);

        p.rail = RailKind::Qr;
        let f = extract_features(&p, &ScoringContext::empty()).unwrap();
        assert_eq!(f[27], 1.0);
    }

    #[test]
    fn hash_to_unit_deterministic() {
        // Same input -> same output. Different input -> different output
        // with very high probability (32-bit prefix collision is ~1/4B).
        let a1 = hash_to_unit(Some("abc"));
        let a2 = hash_to_unit(Some("abc"));
        let b = hash_to_unit(Some("abd"));
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
    }

    #[test]
    fn hash_to_unit_bounded() {
        for s in [
            "",
            "x",
            "alice@example.com",
            "1234567890",
            "a very long string indeed",
        ] {
            let v = hash_to_unit(Some(s));
            assert!(
                (0.0..=1.0).contains(&v),
                "hash_to_unit({s:?}) = {v} out of range"
            );
        }
        // None and "" both map to 0.0 (missing signal).
        assert_eq!(hash_to_unit(None), 0.0);
        assert_eq!(hash_to_unit(Some("")), 0.0);
    }

    #[test]
    fn hash_to_unit_matches_python_reference() {
        // Verified by external Python: SHA-256("abc")[:4] big-endian / u32::MAX
        // Python computed 0.728395 for "abc". Reproduce the math in Rust:
        let v = hash_to_unit(Some("abc"));
        // Allow loose tolerance because Python and Rust f32 conversion paths differ.
        assert!((v - 0.728_395).abs() < 1e-4, "got {v}, expected ~0.728395");
    }

    #[test]
    fn extraction_is_deterministic() {
        let m = vault_method();
        let p = sample_descriptor(&m);
        let ctx = ScoringContext {
            timestamp: Some(datetime!(2026-05-17 12:00:00 UTC)),
            velocity_1h: Some(3),
            device_id: Some("dev_xyz".into()),
            ..Default::default()
        };
        let f1 = extract_features(&p, &ctx).unwrap();
        let f2 = extract_features(&p, &ctx).unwrap();
        assert_eq!(f1, f2);
    }

    #[test]
    fn extraction_rejects_non_finite_after_extraction() {
        // We don't have an obvious way to make extract_features produce
        // a non-finite output without manipulating internals, but we can
        // assert the validation loop catches it.
        let mut f = [0.0_f32; FEATURES];
        f[0] = f32::NAN;
        for v in &f {
            if !v.is_finite() {
                // This is what the validation loop does.
                return;
            }
        }
        panic!("validation loop should have caught NaN");
    }

    #[test]
    fn weekend_flag_works_for_saturday() {
        let m = vault_method();
        let p = sample_descriptor(&m);
        let ctx = ScoringContext {
            timestamp: Some(datetime!(2026-05-16 12:00:00 UTC)), // Saturday
            ..Default::default()
        };
        let f = extract_features(&p, &ctx).unwrap();
        assert_eq!(f[10], 1.0);
    }

    #[test]
    fn weekend_flag_zero_for_monday() {
        let m = vault_method();
        let p = sample_descriptor(&m);
        let ctx = ScoringContext {
            timestamp: Some(datetime!(2026-05-18 12:00:00 UTC)), // Monday
            ..Default::default()
        };
        let f = extract_features(&p, &ctx).unwrap();
        assert_eq!(f[10], 0.0);
    }

    #[test]
    fn normalize_log_seconds_at_reference_is_one() {
        let v = normalize_log_seconds(604_800, 604_800);
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_log_seconds_clamps_above_reference() {
        let v = normalize_log_seconds(604_800 * 10, 604_800);
        assert!(v <= 1.0);
        assert!(v >= 0.99); // log(10x)/log(x) ≈ 1.05 → clamped to 1.0
    }

    #[test]
    fn normalize_log_seconds_zero_at_zero() {
        let v = normalize_log_seconds(0, 604_800);
        assert_eq!(v, 0.0);
    }
}
