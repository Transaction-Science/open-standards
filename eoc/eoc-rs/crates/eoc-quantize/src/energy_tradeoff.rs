//! Joules-per-token estimator across quantization schemes.
//!
//! The estimator is *first-order*: it scales a reference fp32 cost by
//! the bits-per-weight ratio plus a small fixed overhead for
//! dequantization. It is intentionally simple so the cascade and meter
//! can route on a stable, easy-to-explain number.
//!
//! Sources / sanity checks:
//!
//! * fp32 → fp16 yields ~2x energy reduction in memory-bound regimes
//!   (Horowitz "Computing's energy problem", ISSCC 2014).
//! * int8 vs fp32 ~4x reduction (operand width + cheaper MAC).
//! * int4 ~6–8x reduction (memory-bound, with dequant overhead).
//! * int2 / ternary up to ~16x in custom silicon, ~8–10x on GPUs.
//!
//! The constants here are order-of-magnitude; users should override
//! `EnergyModel::overrides` for measured silicon.

use crate::scheme::Numeric;

/// Per-scheme energy model. Values are *relative to fp32*.
#[derive(Debug, Clone)]
pub struct EnergyModel {
    /// Reference fp32 cost per token, in micro-joules.
    pub fp32_microjoules_per_token: u64,
    /// Per-numeric multiplier overriding the default ratio.
    pub overrides: Option<fn(Numeric) -> f32>,
}

impl Default for EnergyModel {
    fn default() -> Self {
        Self {
            fp32_microjoules_per_token: 50_000_000, // 50 J / token — placeholder
            overrides: None,
        }
    }
}

impl EnergyModel {
    /// Construct with a custom fp32 baseline.
    pub fn with_fp32_baseline(microjoules_per_token: u64) -> Self {
        Self {
            fp32_microjoules_per_token: microjoules_per_token,
            overrides: None,
        }
    }

    /// Multiplier for a given numeric kind, defaulting to the standard
    /// table below.
    pub fn multiplier(&self, n: Numeric) -> f32 {
        if let Some(f) = self.overrides {
            return f(n);
        }
        default_multiplier(n)
    }

    /// Estimate micro-joules per token for a given numeric kind.
    pub fn microjoules_per_token(&self, n: Numeric) -> u64 {
        let m = self.multiplier(n).max(0.0);
        (self.fp32_microjoules_per_token as f64 * m as f64).round() as u64
    }
}

/// Default per-numeric multipliers relative to fp32 (1.0 = fp32 cost).
pub fn default_multiplier(n: Numeric) -> f32 {
    match n {
        Numeric::Fp32 => 1.00,
        Numeric::Fp16 | Numeric::Bf16 => 0.50,
        Numeric::Fp8E4m3 | Numeric::Fp8E5m2 => 0.30,
        Numeric::Int8 => 0.25,
        Numeric::Int4 | Numeric::Nf4 => 0.14,
        Numeric::Int2 => 0.08,
    }
}

/// Bits-per-weight + relative energy summary — handy for a table.
#[derive(Debug, Clone)]
pub struct TradeoffRow {
    /// Numeric kind.
    pub numeric: Numeric,
    /// Bits per weight on disk.
    pub bits_per_weight: u32,
    /// Energy multiplier relative to fp32.
    pub energy_multiplier: f32,
    /// Estimated micro-joules per token under the given model.
    pub microjoules_per_token: u64,
}

/// Produce a full tradeoff table for the standard catalogue.
pub fn tradeoff_table(model: &EnergyModel) -> Vec<TradeoffRow> {
    let all = [
        Numeric::Fp32,
        Numeric::Fp16,
        Numeric::Bf16,
        Numeric::Fp8E4m3,
        Numeric::Fp8E5m2,
        Numeric::Int8,
        Numeric::Int4,
        Numeric::Nf4,
        Numeric::Int2,
    ];
    all.iter()
        .map(|&n| TradeoffRow {
            numeric: n,
            bits_per_weight: n.bits_per_weight(),
            energy_multiplier: model.multiplier(n),
            microjoules_per_token: model.microjoules_per_token(n),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_precision_is_cheaper() {
        let m = EnergyModel::default();
        assert!(m.microjoules_per_token(Numeric::Int8) < m.microjoules_per_token(Numeric::Fp16));
        assert!(m.microjoules_per_token(Numeric::Int4) < m.microjoules_per_token(Numeric::Int8));
        assert!(m.microjoules_per_token(Numeric::Int2) < m.microjoules_per_token(Numeric::Int4));
    }

    #[test]
    fn table_has_full_catalogue() {
        let t = tradeoff_table(&EnergyModel::default());
        assert_eq!(t.len(), 9);
    }
}
