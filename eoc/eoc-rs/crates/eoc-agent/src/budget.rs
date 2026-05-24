//! Per-step budgets — tokens, joules, tool calls.
//!
//! Every loop in this crate consults the same [`Budget`](crate::agent::Budget)
//! to decide whether to take the next step. Budgets are *charge-as-you-go*
//! and saturating: a single oversized step exhausts the budget rather
//! than wrapping around to zero.

use serde::{Deserialize, Serialize};

/// Token budget. Tracks prompt + completion tokens against a cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenBudget {
    /// Maximum tokens the loop may consume.
    pub max_tokens: u64,
    /// Tokens consumed so far.
    pub used: u64,
}

impl TokenBudget {
    /// Construct a fresh token budget.
    pub fn new(max_tokens: u64) -> Self {
        Self { max_tokens, used: 0 }
    }

    /// An effectively-unlimited budget.
    pub fn unlimited() -> Self {
        Self { max_tokens: u64::MAX, used: 0 }
    }

    /// Charge `n` tokens (saturating). Returns `true` iff the budget is
    /// **still** within the cap after the charge.
    pub fn charge(&mut self, n: u64) -> bool {
        self.used = self.used.saturating_add(n);
        self.used <= self.max_tokens
    }

    /// Remaining headroom (saturating subtract).
    pub fn remaining(&self) -> u64 {
        self.max_tokens.saturating_sub(self.used)
    }

    /// Has the budget been exhausted?
    pub fn exhausted(&self) -> bool {
        self.used >= self.max_tokens
    }
}

/// Joule budget. Tracks energy in micro-joules against a cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct JouleBudget {
    /// Maximum micro-joules the loop may consume.
    pub max_microjoules: u64,
    /// Micro-joules consumed so far.
    pub used_microjoules: u64,
}

impl JouleBudget {
    /// Construct a fresh joule budget. `joules` is a real-valued joule cap.
    pub fn from_joules(joules: f64) -> Self {
        let max = (joules.max(0.0) * 1_000_000.0) as u64;
        Self { max_microjoules: max, used_microjoules: 0 }
    }

    /// Construct directly from micro-joules.
    pub fn from_microjoules(max_microjoules: u64) -> Self {
        Self { max_microjoules, used_microjoules: 0 }
    }

    /// An effectively-unlimited budget.
    pub fn unlimited() -> Self {
        Self {
            max_microjoules: u64::MAX,
            used_microjoules: 0,
        }
    }

    /// Charge `microjoules` (saturating). Returns `true` iff the budget
    /// is **still** within the cap after the charge.
    pub fn charge(&mut self, microjoules: u64) -> bool {
        self.used_microjoules = self.used_microjoules.saturating_add(microjoules);
        self.used_microjoules <= self.max_microjoules
    }

    /// Remaining headroom in micro-joules.
    pub fn remaining_microjoules(&self) -> u64 {
        self.max_microjoules.saturating_sub(self.used_microjoules)
    }

    /// Remaining headroom as joules.
    pub fn remaining_joules(&self) -> f64 {
        (self.remaining_microjoules() as f64) / 1_000_000.0
    }

    /// Has the budget been exhausted?
    pub fn exhausted(&self) -> bool {
        self.used_microjoules >= self.max_microjoules
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_budget_saturates() {
        let mut b = TokenBudget::new(10);
        assert!(b.charge(4));
        assert!(b.charge(6));
        assert!(b.exhausted());
        // Charging past the cap stays saturated, doesn't wrap.
        assert!(!b.charge(u64::MAX));
        assert_eq!(b.remaining(), 0);
    }

    #[test]
    fn joule_budget_from_joules() {
        let b = JouleBudget::from_joules(0.001);
        assert_eq!(b.max_microjoules, 1_000);
    }

    #[test]
    fn joule_budget_negative_clamped() {
        let b = JouleBudget::from_joules(-7.0);
        assert_eq!(b.max_microjoules, 0);
    }
}
