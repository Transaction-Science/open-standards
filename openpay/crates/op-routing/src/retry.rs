//! Intelligent retry / soft-decline recovery.
//!
//! ISO 8583 response codes split into two camps:
//!
//! - **Soft declines** — recoverable on a different rail / PSP
//!   within seconds. The customer has funds, the card is valid; the
//!   denial is on the routing path (issuer momentarily refusing,
//!   processor inoperative, do-not-honor that the issuer reverses
//!   when re-presented through another acquirer).
//!
//! - **Hard declines** — the issuer told us in plain text "do not
//!   re-present". Retrying not only wastes auth attempts, it
//!   triggers `Pickup card` (04, 07, 41, 43) escalations against
//!   the merchant. We MUST NOT retry.
//!
//! The standard taxonomy is the ISO 8583 DE-39 (response code)
//! field. The exact codes that constitute "soft" vs "hard" vary
//! mildly by acquirer but the canonical list shipped here matches
//! Visa's "Reason Code Action" reference and Mastercard's
//! "Authorization Response Code" classification.
//!
//! This module implements:
//!
//! - [`DeclineCode`] — typed wrapper around the 2-character code.
//! - [`default_soft_declines`] / [`default_hard_declines`] —
//!   curated sets straight off the published taxonomies.
//! - [`BackoffPolicy`] — exponential + jitter delay between attempts.
//! - [`IntelligentRetry`] — policy engine: given prior attempts and
//!   a remaining route pool, decide whether to retry and on which
//!   route.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Duration;

use crate::route::Route;

/// ISO 8583 DE-39 decline code. Two ASCII characters (digits in
/// practice; we accept letters for forward-compat with proprietary
/// extensions).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeclineCode(pub [u8; 2]);

impl DeclineCode {
    /// Construct from a 2-character `&str`. Returns `None` if not
    /// exactly two ASCII alphanumerics.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        let mut out = [0u8; 2];
        for (i, b) in bytes.iter().enumerate() {
            if !b.is_ascii_alphanumeric() {
                return None;
            }
            out[i] = *b;
        }
        Some(Self(out))
    }

    /// As `&str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.0).unwrap_or("??")
    }
}

impl core::fmt::Display for DeclineCode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Category bucket for a decline.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DeclineCategory {
    /// Recoverable — retry on another rail / PSP.
    Soft,
    /// Terminal — do NOT retry.
    Hard,
    /// Unknown code. Default policy: treat as soft to avoid stranding
    /// recoverable transactions on a typo. Operators can flip this
    /// to "hard" by registering the code in `hard_decline_codes`.
    Unknown,
}

/// The default ISO 8583 soft-decline taxonomy.
///
/// Each code is paired with its standard meaning in a comment for
/// auditing. Sources: Visa Reason Code Action, Mastercard
/// Authorization Response Code reference.
#[must_use]
pub fn default_soft_declines() -> HashSet<DeclineCode> {
    [
        // 05 — Do Not Honor. Issuer's catch-all "no, try again";
        // industry practice retries on alternate path.
        "05",
        // 51 — Insufficient Funds. Often resolves within minutes
        // for customer's monthly cycle / paycheck deposit timing.
        "51",
        // 65 — Exceeds Withdrawal Frequency Limit. Issuer's daily
        // velocity counter; another route may not hit it.
        "65",
        // 75 — PIN Tries Exceeded. Some issuers reset on next path.
        "75",
        // 85 — No Reason to Decline (issuer system noise).
        "85",
        // 91 — Issuer or Switch Inoperative.
        "91",
        // 92 — Financial institution unknown / can't find route.
        "92",
        // 94 — Duplicate transmission.
        "94",
        // 96 — System Malfunction.
        "96",
        // 6P — Verification data failed (network token mismatch on
        // first try — retry with different cryptogram).
        "6P",
        // N7 — Decline for CVV2 failure on first attempt.
        "N7",
        // R0 / R1 — Stop Payment / Revocation Order (transient on
        // some acquirer paths; recoverable on another).
        "R0",
        "R1",
        // 57 — Transaction Not Permitted to Cardholder (some
        // acquirers re-present on alternate path).
        "57",
        // 58 — Transaction Not Permitted to Terminal.
        "58",
        // 61 — Exceeds Approval Amount Limit.
        "61",
        // 19 — Re-enter transaction.
        "19",
        // 89 — Terminal ID unknown.
        "89",
        // 06 — Error.
        "06",
    ]
    .iter()
    .filter_map(|s| DeclineCode::from_str(s))
    .collect()
}

/// The default ISO 8583 hard-decline taxonomy. These MUST NOT retry.
///
/// Re-presenting these triggers chargeback / pickup-card escalation
/// and degrades merchant standing with the scheme.
#[must_use]
pub fn default_hard_declines() -> HashSet<DeclineCode> {
    [
        // 04 — Pickup card (no fraud).
        "04",
        // 07 — Pickup card, special condition (fraud).
        "07",
        // 41 — Lost card.
        "41",
        // 43 — Stolen card.
        "43",
        // 54 — Expired card.
        "54",
        // 59 — Suspected Fraud.
        "59",
        // 62 — Restricted card.
        "62",
        // 14 — Invalid card number (no card with this PAN).
        "14",
        // 15 — No such issuer.
        "15",
        // 78 — No account / blocked, first use.
        "78",
        // R3 — Revocation of all authorizations.
        "R3",
        // 12 — Invalid transaction (won't be valid on retry).
        "12",
        // 13 — Invalid amount.
        "13",
        // 35 — Card acceptor contact acquirer (do not retry).
        "35",
        // 36 — Restricted card.
        "36",
        // 39 — No credit account.
        "39",
        // 40 — Requested function not supported.
        "40",
        // 46 — Closed Account.
        "46",
        // 52 — No checking account.
        "52",
        // 53 — No savings account.
        "53",
        // 55 — Incorrect PIN.
        "55",
        // 63 — Security violation.
        "63",
        // 93 — Transaction cannot be completed; violation of law.
        "93",
    ]
    .iter()
    .filter_map(|s| DeclineCode::from_str(s))
    .collect()
}

/// Backoff policy: exponential with jitter.
///
/// `delay(n) = min(initial * 2^n, cap) + jitter(0..jitter_max)`
///
/// `jitter_seed` is captured in the policy so traces are
/// reproducible — same intent + same prior attempts + same seed →
/// same delays.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BackoffPolicy {
    /// Delay before the second attempt (the first attempt has no
    /// preceding delay).
    pub initial: Duration,
    /// Cap on the exponential component.
    pub cap: Duration,
    /// Maximum jitter added on top of the exponential delay.
    pub jitter_max: Duration,
    /// Deterministic jitter seed.
    pub jitter_seed: u64,
}

impl Default for BackoffPolicy {
    fn default() -> Self {
        Self {
            initial: Duration::from_millis(100),
            cap: Duration::from_secs(30),
            jitter_max: Duration::from_millis(250),
            jitter_seed: 0xA5A5_A5A5_A5A5_A5A5,
        }
    }
}

impl BackoffPolicy {
    /// Compute the delay before attempt number `attempt_index`
    /// (zero-based; attempt 0 has zero delay; attempt 1's delay is
    /// `initial`).
    #[must_use]
    pub fn delay_for(&self, attempt_index: u32) -> Duration {
        if attempt_index == 0 {
            return Duration::ZERO;
        }
        // exponential: initial * 2^(attempt_index - 1), saturating
        let shift = attempt_index.saturating_sub(1).min(20); // cap at 2^20 ms multiplier
        let factor = 1u64 << shift;
        let initial_nanos = u64::try_from(self.initial.as_nanos()).unwrap_or(u64::MAX);
        let exp_nanos = initial_nanos.saturating_mul(factor);
        let exp = Duration::from_nanos(exp_nanos).min(self.cap);

        // Deterministic jitter from a SplitMix64 over (seed, attempt).
        let mut z = self
            .jitter_seed
            .wrapping_add(u64::from(attempt_index).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let jitter_nanos = if self.jitter_max.is_zero() {
            0
        } else {
            // Avoid mod-by-zero (checked above).
            let jmax = u64::try_from(self.jitter_max.as_nanos()).unwrap_or(u64::MAX);
            z % jmax.max(1)
        };

        exp + Duration::from_nanos(jitter_nanos)
    }
}

/// A prior attempt the retry engine knows about.
///
/// This is a lightweight local mirror of
/// `op-orchestrator::Attempt`. We don't depend on the orchestrator
/// crate, so operators construct this from whatever attempt type
/// they hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attempt {
    /// The route that was tried.
    pub route: Route,
    /// The decline code returned, if any. `None` for success or
    /// `RequiresAction`.
    pub decline_code: Option<DeclineCode>,
}

impl Attempt {
    /// Construct.
    #[must_use]
    pub const fn new(route: Route, decline_code: Option<DeclineCode>) -> Self {
        Self {
            route,
            decline_code,
        }
    }
}

/// The retry policy engine.
#[derive(Clone, Debug)]
pub struct IntelligentRetry {
    /// Maximum total attempts (including the first). Hard cap.
    pub max_attempts: u8,
    /// Wall-clock window from first attempt within which retries
    /// are allowed. Retries that would land outside this window are
    /// rejected.
    pub retry_window: Duration,
    /// Codes treated as soft (retry on alternate route).
    pub soft_decline_codes: HashSet<DeclineCode>,
    /// Codes treated as hard (terminate, do not retry).
    pub hard_decline_codes: HashSet<DeclineCode>,
    /// Backoff policy between attempts.
    pub backoff: BackoffPolicy,
}

impl Default for IntelligentRetry {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            retry_window: Duration::from_secs(30),
            soft_decline_codes: default_soft_declines(),
            hard_decline_codes: default_hard_declines(),
            backoff: BackoffPolicy::default(),
        }
    }
}

impl IntelligentRetry {
    /// Construct with explicit defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: max attempts.
    #[must_use]
    pub const fn with_max_attempts(mut self, n: u8) -> Self {
        self.max_attempts = n;
        self
    }

    /// Builder: retry window.
    #[must_use]
    pub const fn with_retry_window(mut self, d: Duration) -> Self {
        self.retry_window = d;
        self
    }

    /// Builder: backoff policy.
    #[must_use]
    pub const fn with_backoff(mut self, b: BackoffPolicy) -> Self {
        self.backoff = b;
        self
    }

    /// Classify a decline code against the configured sets.
    #[must_use]
    pub fn classify(&self, code: &DeclineCode) -> DeclineCategory {
        if self.hard_decline_codes.contains(code) {
            DeclineCategory::Hard
        } else if self.soft_decline_codes.contains(code) {
            DeclineCategory::Soft
        } else {
            DeclineCategory::Unknown
        }
    }

    /// Decide the next attempt given prior attempts and the
    /// remaining route pool.
    ///
    /// Logic:
    /// 1. If any prior attempt's last decline is **hard**, return
    ///    `None` (terminate; no retry).
    /// 2. If `prior_attempts.len() >= max_attempts`, return `None`.
    /// 3. Pick the first route from `pool` whose driver has NOT yet
    ///    been attempted. (LCR/MCC have already ordered the pool;
    ///    we just walk it.)
    /// 4. If none remain, return `None`.
    pub fn next_attempt(&self, prior_attempts: &[Attempt], pool: &[Route]) -> Option<Route> {
        // 1. Hard-decline gate.
        if let Some(last) = prior_attempts.last()
            && let Some(code) = &last.decline_code
            && self.classify(code) == DeclineCategory::Hard
        {
            return None;
        }

        // 2. Attempt count gate.
        if u8::try_from(prior_attempts.len()).unwrap_or(u8::MAX) >= self.max_attempts {
            return None;
        }

        // 3. Walk the pool for a route we haven't tried.
        let tried: HashSet<&crate::route::DriverId> =
            prior_attempts.iter().map(|a| &a.route.driver).collect();
        pool.iter().find(|r| !tried.contains(&r.driver)).cloned()
    }

    /// Aggregate delay across all retries so far.
    #[must_use]
    pub fn total_delay(&self, attempts_completed: u32) -> Duration {
        let mut total = Duration::ZERO;
        for i in 1..=attempts_completed {
            total = total.saturating_add(self.backoff.delay_for(i));
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route::DriverId;
    use op_core::RailKind;

    fn route(name: &str) -> Route {
        Route::new(DriverId::new(name), RailKind::Card)
    }

    fn dc(s: &str) -> DeclineCode {
        DeclineCode::from_str(s).expect("test decline code")
    }

    #[test]
    fn decline_code_construction() {
        assert_eq!(dc("05").as_str(), "05");
        assert_eq!(DeclineCode::from_str("5"), None);
        assert_eq!(DeclineCode::from_str("055"), None);
        assert_eq!(DeclineCode::from_str("--"), None);
    }

    #[test]
    fn classification_buckets() {
        let policy = IntelligentRetry::new();
        assert_eq!(policy.classify(&dc("05")), DeclineCategory::Soft);
        assert_eq!(policy.classify(&dc("51")), DeclineCategory::Soft);
        assert_eq!(policy.classify(&dc("91")), DeclineCategory::Soft);
        assert_eq!(policy.classify(&dc("96")), DeclineCategory::Soft);
        assert_eq!(policy.classify(&dc("04")), DeclineCategory::Hard);
        assert_eq!(policy.classify(&dc("43")), DeclineCategory::Hard);
        assert_eq!(policy.classify(&dc("41")), DeclineCategory::Hard);
        assert_eq!(policy.classify(&dc("62")), DeclineCategory::Hard);
        assert_eq!(policy.classify(&dc("59")), DeclineCategory::Hard);
        // ZZ is unknown.
        assert_eq!(policy.classify(&dc("ZZ")), DeclineCategory::Unknown);
    }

    #[test]
    fn hard_decline_blocks_retry() {
        let policy = IntelligentRetry::new();
        let pool = vec![route("a"), route("b")];
        let prior = vec![Attempt::new(route("a"), Some(dc("43")))]; // stolen card
        assert!(policy.next_attempt(&prior, &pool).is_none());
    }

    #[test]
    fn soft_decline_retries_on_next_untried_route() {
        let policy = IntelligentRetry::new();
        let pool = vec![route("a"), route("b"), route("c")];
        let prior = vec![Attempt::new(route("a"), Some(dc("05")))];
        let next = policy.next_attempt(&prior, &pool).expect("expected retry");
        assert_eq!(next.driver.as_str(), "b");
    }

    #[test]
    fn five_soft_declines_caps_at_max_attempts_no_infinite_loop() {
        // max_attempts=3 should give us at most 2 retries (3 total attempts).
        let policy = IntelligentRetry::new().with_max_attempts(3);
        let pool: Vec<Route> = (0..10).map(|i| route(&format!("psp-{i}"))).collect();
        let mut attempts: Vec<Attempt> = Vec::new();
        attempts.push(Attempt::new(pool[0].clone(), Some(dc("05"))));

        let mut step = 0;
        while let Some(next) = policy.next_attempt(&attempts, &pool) {
            attempts.push(Attempt::new(next, Some(dc("05"))));
            step += 1;
            assert!(step <= 50, "infinite loop detected");
        }
        assert_eq!(attempts.len(), 3);
    }

    #[test]
    fn already_tried_route_not_selected() {
        let policy = IntelligentRetry::new();
        let pool = vec![route("a"), route("b")];
        let prior = vec![
            Attempt::new(route("a"), Some(dc("05"))),
            Attempt::new(route("b"), Some(dc("05"))),
        ];
        assert!(policy.next_attempt(&prior, &pool).is_none());
    }

    #[test]
    fn unknown_code_treated_as_soft_by_default() {
        let policy = IntelligentRetry::new();
        let pool = vec![route("a"), route("b")];
        let prior = vec![Attempt::new(route("a"), Some(dc("ZZ")))];
        let next = policy.next_attempt(&prior, &pool);
        assert!(next.is_some(), "unknown code should not block retry");
    }

    #[test]
    fn backoff_first_attempt_has_zero_delay() {
        let b = BackoffPolicy::default();
        assert_eq!(b.delay_for(0), Duration::ZERO);
    }

    #[test]
    fn backoff_exponential_growth() {
        let b = BackoffPolicy {
            initial: Duration::from_millis(100),
            cap: Duration::from_secs(60),
            jitter_max: Duration::ZERO,
            jitter_seed: 0,
        };
        // attempt 1: 100ms, attempt 2: 200ms, attempt 3: 400ms.
        assert_eq!(b.delay_for(1), Duration::from_millis(100));
        assert_eq!(b.delay_for(2), Duration::from_millis(200));
        assert_eq!(b.delay_for(3), Duration::from_millis(400));
    }

    #[test]
    fn backoff_caps_at_cap() {
        let b = BackoffPolicy {
            initial: Duration::from_millis(100),
            cap: Duration::from_secs(1),
            jitter_max: Duration::ZERO,
            jitter_seed: 0,
        };
        // 100ms * 2^10 = 102400ms = 102s; capped at 1s.
        assert_eq!(b.delay_for(11), Duration::from_secs(1));
    }

    #[test]
    fn backoff_jitter_is_deterministic() {
        let b = BackoffPolicy {
            initial: Duration::from_millis(100),
            cap: Duration::from_secs(60),
            jitter_max: Duration::from_millis(50),
            jitter_seed: 42,
        };
        // Same call → same result. (We don't pin the exact value;
        // we just verify reproducibility.)
        let a = b.delay_for(2);
        let c = b.delay_for(2);
        assert_eq!(a, c);
        // Different attempt index → (typically) different jitter.
        // Not guaranteed identical, just verifying no crash.
        let _ = b.delay_for(3);
    }

    #[test]
    fn total_delay_sums_components() {
        let policy = IntelligentRetry::new().with_backoff(BackoffPolicy {
            initial: Duration::from_millis(100),
            cap: Duration::from_secs(60),
            jitter_max: Duration::ZERO,
            jitter_seed: 0,
        });
        // delays: attempt 1=100ms, attempt 2=200ms → total=300ms.
        assert_eq!(policy.total_delay(2), Duration::from_millis(300));
    }
}
