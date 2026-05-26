//! Safety policies for body-tier dispatch.
//!
//! Body tiers touch the world. The cascade routing layer above them
//! is happy to dispatch any query that matches a body-tier
//! coordinate — but the *act* of committing should be gated by
//! explicit policy. This module is that policy layer.
//!
//! Three orthogonal controls:
//!
//!   * **Rate limit.** No more than N commits per window. Sliding
//!     window of timestamps; old ones drop off.
//!
//!   * **Joule budget.** Total committed action cost per window
//!     can't exceed a configured ceiling. Useful for capping the
//!     downstream resource consumption a body tier can trigger
//!     (network bandwidth, downstream service charges, electricity).
//!
//!   * **Dry-run requirement.** Commit must be preceded by a
//!     successful dry-run of the same plan. Forces a two-step
//!     human-in-the-loop or test-before-prod discipline.
//!
//! Policies compose: a `SafetyPolicy` can have all three set, any
//! subset, or none. `SafetyPolicy::permissive()` is the default
//! (no limits); `SafetyPolicy::strict()` enables all checks at
//! conservative defaults.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// A safety policy. Each field is `None` if that check is disabled,
/// `Some(value)` if enabled.
#[derive(Debug, Clone)]
pub struct SafetyPolicy {
    /// Maximum commits per `rate_window`.
    pub max_commits_per_window: Option<u32>,
    pub rate_window: Duration,

    /// Maximum total joules of action cost per `budget_window`.
    pub max_joules_per_window: Option<f64>,
    pub budget_window: Duration,

    /// Whether commits must be preceded by a successful dry-run
    /// of the same plan within `dry_run_window`.
    pub require_dry_run: bool,
    pub dry_run_window: Duration,
}

impl SafetyPolicy {
    /// No limits. The default for tests and trusted contexts.
    pub fn permissive() -> Self {
        Self {
            max_commits_per_window: None,
            rate_window: Duration::from_secs(60),
            max_joules_per_window: None,
            budget_window: Duration::from_secs(60 * 60),
            require_dry_run: false,
            dry_run_window: Duration::from_secs(60),
        }
    }

    /// Strict defaults: 10 commits/min, 1 J/hour, dry-run required.
    /// Sensible starting point for production deployments.
    pub fn strict() -> Self {
        Self {
            max_commits_per_window: Some(10),
            rate_window: Duration::from_secs(60),
            max_joules_per_window: Some(1.0),
            budget_window: Duration::from_secs(60 * 60),
            require_dry_run: true,
            dry_run_window: Duration::from_secs(60 * 5),
        }
    }

    pub fn with_rate_limit(mut self, max: u32, window: Duration) -> Self {
        self.max_commits_per_window = Some(max);
        self.rate_window = window;
        self
    }

    pub fn with_joule_budget(mut self, max: f64, window: Duration) -> Self {
        self.max_joules_per_window = Some(max);
        self.budget_window = window;
        self
    }

    pub fn with_dry_run_required(mut self, window: Duration) -> Self {
        self.require_dry_run = true;
        self.dry_run_window = window;
        self
    }
}

impl Default for SafetyPolicy {
    fn default() -> Self { Self::permissive() }
}

/// The accumulator that tracks recent commits and dry-runs. Bound to
/// a single `BodyDispatch`. Uses real `Instant`s so the test suite
/// can verify time-windowed behavior. Production deployments can
/// replace with a monotonic clock injection if the timing model needs
/// to be deterministic (the policy struct is the only thing that
/// needs to change).
#[derive(Debug)]
pub struct SafetyState {
    commit_times: VecDeque<Instant>,
    budget_entries: VecDeque<(Instant, f64)>,
    /// Plans (by hash) that have had a successful dry-run.
    dry_run_hashes: VecDeque<(Instant, u64)>,
    pub denied_count: u64,
}

impl SafetyState {
    pub fn new() -> Self {
        Self {
            commit_times: VecDeque::new(),
            budget_entries: VecDeque::new(),
            dry_run_hashes: VecDeque::new(),
            denied_count: 0,
        }
    }

    /// Record that a dry-run succeeded on a plan with the given hash.
    pub fn record_dry_run(&mut self, plan_hash: u64) {
        self.dry_run_hashes.push_back((Instant::now(), plan_hash));
    }

    /// Check whether a commit would be allowed under the policy.
    /// Returns `Ok(())` if allowed, `Err(reason)` if denied.
    pub fn check_commit(
        &mut self,
        policy: &SafetyPolicy,
        plan_hash: u64,
        plan_joules: f64,
    ) -> Result<(), DenyReason> {
        let now = Instant::now();
        self.prune(now, policy);

        if let Some(max) = policy.max_commits_per_window {
            if self.commit_times.len() as u32 >= max {
                self.denied_count += 1;
                return Err(DenyReason::RateLimit {
                    recent_commits: self.commit_times.len() as u32,
                    limit: max,
                });
            }
        }

        if let Some(max_j) = policy.max_joules_per_window {
            let current: f64 = self.budget_entries.iter().map(|(_, j)| j).sum();
            if current + plan_joules > max_j {
                self.denied_count += 1;
                return Err(DenyReason::BudgetExceeded {
                    current_joules: current,
                    plan_joules,
                    limit: max_j,
                });
            }
        }

        if policy.require_dry_run {
            let cutoff = now - policy.dry_run_window;
            let seen = self.dry_run_hashes.iter()
                .any(|(t, h)| *h == plan_hash && *t >= cutoff);
            if !seen {
                self.denied_count += 1;
                return Err(DenyReason::DryRunRequired);
            }
        }

        Ok(())
    }

    /// Record a successful commit. Must be called by `BodyDispatch`
    /// after `check_commit` returns OK and the body tier actually
    /// performed the action.
    pub fn record_commit(&mut self, plan_joules: f64) {
        let now = Instant::now();
        self.commit_times.push_back(now);
        self.budget_entries.push_back((now, plan_joules));
    }

    /// Drop entries outside their respective windows.
    fn prune(&mut self, now: Instant, policy: &SafetyPolicy) {
        let rate_cutoff = now.checked_sub(policy.rate_window).unwrap_or(now);
        while let Some(&t) = self.commit_times.front() {
            if t < rate_cutoff { self.commit_times.pop_front(); } else { break; }
        }

        let budget_cutoff = now.checked_sub(policy.budget_window).unwrap_or(now);
        while let Some(&(t, _)) = self.budget_entries.front() {
            if t < budget_cutoff { self.budget_entries.pop_front(); } else { break; }
        }

        let dr_cutoff = now.checked_sub(policy.dry_run_window).unwrap_or(now);
        while let Some(&(t, _)) = self.dry_run_hashes.front() {
            if t < dr_cutoff { self.dry_run_hashes.pop_front(); } else { break; }
        }
    }

    pub fn commits_in_window(&self) -> usize { self.commit_times.len() }
    pub fn joules_in_window(&self) -> f64 {
        self.budget_entries.iter().map(|(_, j)| j).sum()
    }
}

impl Default for SafetyState {
    fn default() -> Self { Self::new() }
}

/// Why a commit was denied.
#[derive(Debug, Clone, PartialEq)]
pub enum DenyReason {
    RateLimit { recent_commits: u32, limit: u32 },
    BudgetExceeded { current_joules: f64, plan_joules: f64, limit: f64 },
    DryRunRequired,
}

impl std::fmt::Display for DenyReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RateLimit { recent_commits, limit } =>
                write!(f, "rate limit: {} recent commits, limit {}",
                    recent_commits, limit),
            Self::BudgetExceeded { current_joules, plan_joules, limit } =>
                write!(f, "budget exceeded: current {:.3e} J + plan {:.3e} J > limit {:.3e} J",
                    current_joules, plan_joules, limit),
            Self::DryRunRequired =>
                write!(f, "policy requires successful dry-run before commit"),
        }
    }
}

/// Hash a `Plan` to a stable u64. Used by the dry-run gate to match
/// commits against prior dry-runs.
pub fn hash_plan(plan: &super::body::Plan) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    plan.description.hash(&mut h);
    plan.payload.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Plan;

    fn plan(desc: &str, joules: f64) -> Plan {
        Plan {
            description: desc.to_string(),
            payload: desc.as_bytes().to_vec(),
            action_joules: joules,
            reversible: false,
        }
    }

    #[test]
    fn permissive_allows_everything() {
        let policy = SafetyPolicy::permissive();
        let mut state = SafetyState::new();
        let p = plan("write x", 1e-6);
        for _ in 0..100 {
            assert!(state.check_commit(&policy, hash_plan(&p), p.action_joules).is_ok());
            state.record_commit(p.action_joules);
        }
        assert_eq!(state.denied_count, 0);
    }

    #[test]
    fn rate_limit_blocks_after_max() {
        let policy = SafetyPolicy::permissive()
            .with_rate_limit(3, Duration::from_secs(10));
        let mut state = SafetyState::new();
        let p = plan("x", 1e-6);

        // 3 commits OK.
        for _ in 0..3 {
            assert!(state.check_commit(&policy, hash_plan(&p), p.action_joules).is_ok());
            state.record_commit(p.action_joules);
        }
        // 4th denied.
        let r = state.check_commit(&policy, hash_plan(&p), p.action_joules);
        assert!(matches!(r, Err(DenyReason::RateLimit { .. })));
    }

    #[test]
    fn budget_blocks_when_exceeded() {
        let policy = SafetyPolicy::permissive()
            .with_joule_budget(1e-5, Duration::from_secs(60));
        let mut state = SafetyState::new();

        // 5 small commits (5e-6 J total) — OK.
        let small = plan("small", 1e-6);
        for _ in 0..5 {
            assert!(state.check_commit(&policy,
                hash_plan(&small), small.action_joules).is_ok());
            state.record_commit(small.action_joules);
        }
        // Now a big plan that would push over the limit.
        let big = plan("big", 1e-5);  // alone exceeds remaining 5e-6
        let r = state.check_commit(&policy, hash_plan(&big), big.action_joules);
        assert!(matches!(r, Err(DenyReason::BudgetExceeded { .. })),
            "expected BudgetExceeded, got {:?}", r);
    }

    #[test]
    fn dry_run_gate_requires_prior_dry_run() {
        let policy = SafetyPolicy::permissive()
            .with_dry_run_required(Duration::from_secs(60));
        let mut state = SafetyState::new();
        let p = plan("hello", 1e-6);

        // Direct commit denied — no dry-run on record.
        let r1 = state.check_commit(&policy, hash_plan(&p), p.action_joules);
        assert!(matches!(r1, Err(DenyReason::DryRunRequired)));

        // Record a dry-run for this plan hash.
        state.record_dry_run(hash_plan(&p));

        // Now commit allowed.
        let r2 = state.check_commit(&policy, hash_plan(&p), p.action_joules);
        assert!(r2.is_ok());
    }

    #[test]
    fn dry_run_for_one_plan_does_not_authorize_another() {
        let policy = SafetyPolicy::permissive()
            .with_dry_run_required(Duration::from_secs(60));
        let mut state = SafetyState::new();
        let p1 = plan("plan-A", 1e-6);
        let p2 = plan("plan-B", 1e-6);

        state.record_dry_run(hash_plan(&p1));

        // Committing plan-A allowed.
        assert!(state.check_commit(&policy, hash_plan(&p1), p1.action_joules).is_ok());
        // Committing plan-B denied.
        let r = state.check_commit(&policy, hash_plan(&p2), p2.action_joules);
        assert!(matches!(r, Err(DenyReason::DryRunRequired)));
    }

    #[test]
    fn strict_policy_blocks_first_commit_without_dry_run() {
        let policy = SafetyPolicy::strict();
        let mut state = SafetyState::new();
        let p = plan("x", 1e-6);
        let r = state.check_commit(&policy, hash_plan(&p), p.action_joules);
        // strict requires dry-run, so this fails.
        assert!(matches!(r, Err(DenyReason::DryRunRequired)));
    }

    #[test]
    fn rate_limit_releases_after_window_passes() {
        // Use a very short window so we can actually wait through it.
        let policy = SafetyPolicy::permissive()
            .with_rate_limit(2, Duration::from_millis(50));
        let mut state = SafetyState::new();
        let p = plan("x", 1e-6);

        // Fill the window.
        for _ in 0..2 {
            state.check_commit(&policy, hash_plan(&p), p.action_joules).unwrap();
            state.record_commit(p.action_joules);
        }
        // Wait past the window.
        std::thread::sleep(Duration::from_millis(60));
        // Now allowed again.
        assert!(state.check_commit(&policy, hash_plan(&p), p.action_joules).is_ok());
    }

    #[test]
    fn denied_count_increments_per_denial() {
        let policy = SafetyPolicy::permissive()
            .with_rate_limit(0, Duration::from_secs(60));   // deny all
        let mut state = SafetyState::new();
        let p = plan("x", 1e-6);
        for _ in 0..5 {
            let _ = state.check_commit(&policy, hash_plan(&p), p.action_joules);
        }
        assert_eq!(state.denied_count, 5);
    }
}
