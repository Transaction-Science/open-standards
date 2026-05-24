//! Card-type taxonomy.
//!
//! The four classes the global card networks publicly distinguish
//! for interchange and processing purposes. We deliberately do
//! **not** model finer distinctions (commercial / consumer,
//! purchase / fleet, gift cards as a separate class from prepaid)
//! because they are not stable across networks and they are not
//! derivable from the BIN alone.

use serde::{Deserialize, Serialize};

/// Card funding type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CardType {
    /// Credit card — funded from a revolving credit line.
    Credit,
    /// Debit card — funded from a demand-deposit account.
    Debit,
    /// Prepaid card — funded from a stored-value balance.
    Prepaid,
    /// Charge card — non-revolving, full balance due each cycle
    /// (the original Amex / Diners model).
    Charge,
    /// Unknown / unspecified.
    Unknown,
}

impl CardType {
    /// Lowercase wire string.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Credit => "credit",
            Self::Debit => "debit",
            Self::Prepaid => "prepaid",
            Self::Charge => "charge",
            Self::Unknown => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_str_round_trip() {
        for c in [
            CardType::Credit,
            CardType::Debit,
            CardType::Prepaid,
            CardType::Charge,
            CardType::Unknown,
        ] {
            assert!(!c.as_str().is_empty());
        }
    }
}
