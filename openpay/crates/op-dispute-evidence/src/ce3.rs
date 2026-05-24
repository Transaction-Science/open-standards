//! Visa Compelling Evidence 3.0 (CE3.0) qualifier.
//!
//! CE3.0 (effective April 2023) gives merchants a way to deflect
//! Visa fraud chargebacks (reason **10.4**) when they can prove the
//! disputing cardholder has *transacted with them before*, free of
//! disputes, on at least **two** historical orders that share
//! linking data with the disputed transaction.
//!
//! Qualifying criteria (the published rules summarized — operators
//! should still cross-reference the Visa Core Rules at filing):
//!
//! 1. The disputed transaction's reason code is Visa **10.4**.
//! 2. At least **two** prior transactions from the same merchant,
//!    same cardholder, with **none** of them disputed.
//! 3. Each prior transaction is **120-365 days old** at the time of
//!    the disputed transaction (the CE3.0 lookback window).
//! 4. At least two of the linking-data points (IP address, device
//!    ID, shipping address, account login ID) match between the
//!    historical transactions and the disputed transaction.
//!
//! When all four hold, the chargeback is **ineligible** for the
//! 10.4 dispute reason — the issuer must withdraw or re-file under
//! a different code.

use serde::{Deserialize, Serialize};
use time::{Duration, OffsetDateTime};

use crate::error::{Error, Result};
use crate::network::VisaReasonCode;
use crate::reason_codes::ReasonCode;

/// CE3.0 lookback floor: a qualifying transaction must be at least
/// this old (relative to the disputed transaction) to count.
pub const CE3_MIN_AGE: Duration = Duration::days(120);
/// CE3.0 lookback ceiling: a qualifying transaction must be no
/// older than this.
pub const CE3_MAX_AGE: Duration = Duration::days(365);
/// Minimum count of qualifying historical transactions.
pub const CE3_MIN_QUALIFIERS: usize = 2;
/// Minimum count of linking-data-point matches.
pub const CE3_MIN_LINKING_MATCHES: usize = 2;

/// A single historical transaction the merchant proposes as a
/// CE3.0 qualifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualifyingTransaction {
    /// Operator-side stable id (for diagnostics + reproducibility).
    pub id: String,
    /// When the transaction was authorized.
    pub authorized_at: OffsetDateTime,
    /// IP address used at checkout, if recorded.
    pub ip: Option<String>,
    /// Device fingerprint hash, if recorded.
    pub device_id: Option<String>,
    /// Shipping-address hash / canonical form.
    pub shipping_address: Option<String>,
    /// Account-login id (merchant-side customer id).
    pub account_login: Option<String>,
    /// True iff this transaction itself was ever disputed.
    pub was_disputed: bool,
}

/// Linking-data fingerprint for either the disputed transaction or
/// a qualifier.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinkingData {
    /// IP address used at checkout.
    pub ip: Option<String>,
    /// Device fingerprint hash.
    pub device_id: Option<String>,
    /// Shipping address (canonical form).
    pub shipping_address: Option<String>,
    /// Account-login id.
    pub account_login: Option<String>,
}

impl LinkingData {
    /// Count of fields that match `other` (only when both sides
    /// have a non-None value).
    #[must_use]
    pub fn match_count(&self, other: &LinkingData) -> usize {
        let mut n = 0;
        if matches(&self.ip, &other.ip) {
            n += 1;
        }
        if matches(&self.device_id, &other.device_id) {
            n += 1;
        }
        if matches(&self.shipping_address, &other.shipping_address) {
            n += 1;
        }
        if matches(&self.account_login, &other.account_login) {
            n += 1;
        }
        n
    }
}

fn matches(a: &Option<String>, b: &Option<String>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

impl From<&QualifyingTransaction> for LinkingData {
    fn from(q: &QualifyingTransaction) -> Self {
        Self {
            ip: q.ip.clone(),
            device_id: q.device_id.clone(),
            shipping_address: q.shipping_address.clone(),
            account_login: q.account_login.clone(),
        }
    }
}

/// Result of running the CE3.0 qualifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ce3Eligibility {
    /// True when the chargeback is CE3.0-ineligible (good for the
    /// merchant — the issuer must withdraw).
    pub eligible: bool,
    /// Subset of the supplied qualifiers that satisfied all CE3.0
    /// constraints.
    pub matched_qualifiers: Vec<String>,
    /// Best linking-match count we found.
    pub best_linking_matches: usize,
    /// Human-readable note explaining the verdict.
    pub note: &'static str,
}

/// Stateless evaluator for CE3.0.
#[derive(Debug, Clone, Copy)]
pub struct Ce3Qualifier;

impl Ce3Qualifier {
    /// Evaluate the disputed transaction + qualifier set.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Ce3Ineligible`] if the inputs are
    /// structurally unsuitable for CE3.0 (wrong reason code, etc.).
    /// A successful return is *not* a guarantee of eligibility;
    /// check [`Ce3Eligibility::eligible`].
    pub fn evaluate(
        reason: ReasonCode,
        disputed_at: OffsetDateTime,
        disputed_link: &LinkingData,
        qualifiers: &[QualifyingTransaction],
    ) -> Result<Ce3Eligibility> {
        // CE3.0 is Visa 10.4 only.
        let ReasonCode::Visa(VisaReasonCode::F1040) = reason else {
            return Err(Error::Ce3Ineligible("reason code is not Visa 10.4"));
        };

        let mut matched: Vec<String> = Vec::new();
        let mut best_links: usize = 0;

        for q in qualifiers {
            if q.was_disputed {
                continue;
            }
            let age = disputed_at - q.authorized_at;
            if age < CE3_MIN_AGE || age > CE3_MAX_AGE {
                continue;
            }
            let qlink = LinkingData::from(q);
            let m = disputed_link.match_count(&qlink);
            if m > best_links {
                best_links = m;
            }
            if m >= CE3_MIN_LINKING_MATCHES {
                matched.push(q.id.clone());
            }
        }

        let eligible = matched.len() >= CE3_MIN_QUALIFIERS;
        Ok(Ce3Eligibility {
            eligible,
            matched_qualifiers: matched,
            best_linking_matches: best_links,
            note: if eligible {
                "CE3.0 ineligible chargeback: 2+ qualifying history matches"
            } else {
                "insufficient qualifying history or linking matches"
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::VisaReasonCode;

    #[test]
    fn non_10_4_rejected_outright() {
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).expect("ok");
        let link = LinkingData::default();
        let err = Ce3Qualifier::evaluate(
            ReasonCode::Visa(VisaReasonCode::F1030),
            now,
            &link,
            &[],
        )
        .expect_err("non-10.4");
        assert!(matches!(err, Error::Ce3Ineligible(_)));
    }
}
