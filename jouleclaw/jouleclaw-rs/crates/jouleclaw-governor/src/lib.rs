//! L10 — governor (meta-cognitive control plane).
//!
//! The top-level safety valve. Where L8 *tunes* the cascade toward
//! efficiency, L10 *enforces* hard limits: a rolling joule budget over a
//! time window, per-tenant quotas, and a manual kill switch. It is the
//! one component allowed to say "no, this query does not run" before any
//! tier is touched.
//!
//! The model is deliberately simple accounting — no probabilistic
//! admission control, no clever shaping. A query is admitted iff its
//! estimated joules fit within both the global window budget and the
//! tenant's remaining quota, and the kill switch is disengaged. This is
//! the part of the system that must be obviously correct under audit, so
//! it stays boring on purpose.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A tenant identifier (customer, project, API key, …).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(pub String);

impl TenantId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

/// Per-tenant joule budget and running spend.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TenantQuota {
    /// Joules this tenant may spend within the current window.
    pub joules_budget: f64,
    /// Joules spent so far this window.
    pub joules_spent: f64,
}

impl TenantQuota {
    pub fn new(joules_budget: f64) -> Self {
        Self {
            joules_budget,
            joules_spent: 0.0,
        }
    }

    /// Remaining joules for this tenant (never negative).
    pub fn remaining(&self) -> f64 {
        (self.joules_budget - self.joules_spent).max(0.0)
    }
}

/// Why a query was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectReason {
    /// The kill switch is engaged; nothing runs.
    KillSwitch,
    /// The global window budget would be exceeded.
    GlobalBudgetExhausted,
    /// The named tenant's quota would be exceeded.
    TenantQuotaExhausted,
    /// The query named a tenant with no registered quota.
    UnknownTenant,
}

/// The governor's verdict on a query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdmitDecision {
    /// Cleared to run.
    Admit,
    /// Blocked, with cause.
    Reject(RejectReason),
}

impl AdmitDecision {
    pub fn is_admit(&self) -> bool {
        matches!(self, AdmitDecision::Admit)
    }
}

/// The L10 governor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Governor {
    /// Total joules allowed within `window_secs`.
    pub global_joules_budget: f64,
    /// Window length in seconds.
    pub window_secs: u64,
    /// Joules spent in the current window.
    pub spent: f64,
    /// Unix-seconds timestamp the current window opened at.
    pub window_start_secs: u64,
    /// Per-tenant quotas. A query with a tenant not in this map is
    /// rejected with [`RejectReason::UnknownTenant`] *only when a tenant
    /// is supplied*; untenanted queries are governed by the global
    /// budget alone.
    pub tenant_quotas: HashMap<TenantId, TenantQuota>,
    /// Manual override. When true, every query is rejected.
    pub kill_switch: bool,
}

impl Governor {
    /// New governor with a global window budget. `window_start_secs`
    /// seeds the first window (pass your clock's current time).
    pub fn new(global_joules_budget: f64, window_secs: u64, window_start_secs: u64) -> Self {
        Self {
            global_joules_budget,
            window_secs,
            spent: 0.0,
            window_start_secs,
            tenant_quotas: HashMap::new(),
            kill_switch: false,
        }
    }

    /// Register or replace a tenant's quota.
    pub fn set_tenant_quota(&mut self, tenant: TenantId, quota: TenantQuota) {
        self.tenant_quotas.insert(tenant, quota);
    }

    /// Joules remaining in the global window.
    pub fn global_remaining(&self) -> f64 {
        (self.global_joules_budget - self.spent).max(0.0)
    }

    /// Roll the window forward if `now` is past its end. Resets the
    /// global `spent` counter and every tenant's `joules_spent`.
    pub fn reset_window(&mut self, now: u64) {
        if now.saturating_sub(self.window_start_secs) >= self.window_secs {
            self.spent = 0.0;
            self.window_start_secs = now;
            for q in self.tenant_quotas.values_mut() {
                q.joules_spent = 0.0;
            }
        }
    }

    /// Decide whether a query estimated at `estimated_joules` may run.
    /// Pass `tenant = None` for untenanted (global-only) governance.
    pub fn admit(
        &self,
        estimated_joules: f64,
        tenant: Option<&TenantId>,
    ) -> AdmitDecision {
        if self.kill_switch {
            return AdmitDecision::Reject(RejectReason::KillSwitch);
        }
        if self.spent + estimated_joules > self.global_joules_budget {
            return AdmitDecision::Reject(RejectReason::GlobalBudgetExhausted);
        }
        if let Some(t) = tenant {
            match self.tenant_quotas.get(t) {
                None => return AdmitDecision::Reject(RejectReason::UnknownTenant),
                Some(q) => {
                    if q.joules_spent + estimated_joules > q.joules_budget {
                        return AdmitDecision::Reject(RejectReason::TenantQuotaExhausted);
                    }
                }
            }
        }
        AdmitDecision::Admit
    }

    /// Debit actual spend after a query runs. Updates the global counter
    /// and, if a tenant is given and known, that tenant's counter.
    pub fn record(&mut self, actual_joules: f64, tenant: Option<&TenantId>) {
        self.spent += actual_joules;
        if let Some(t) = tenant {
            if let Some(q) = self.tenant_quotas.get_mut(t) {
                q.joules_spent += actual_joules;
            }
        }
    }

    /// Engage the kill switch — all subsequent `admit` calls reject.
    pub fn engage_kill_switch(&mut self) {
        self.kill_switch = true;
    }

    /// Release the kill switch.
    pub fn disengage_kill_switch(&mut self) {
        self.kill_switch = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gov() -> Governor {
        Governor::new(100.0, 60, 1_000)
    }

    #[test]
    fn admits_within_budget() {
        let g = gov();
        assert_eq!(g.admit(10.0, None), AdmitDecision::Admit);
    }

    #[test]
    fn rejects_over_global_budget() {
        let g = gov();
        assert_eq!(
            g.admit(101.0, None),
            AdmitDecision::Reject(RejectReason::GlobalBudgetExhausted)
        );
    }

    #[test]
    fn record_debits_global() {
        let mut g = gov();
        g.record(40.0, None);
        assert!((g.global_remaining() - 60.0).abs() < 1e-9);
        // Now a 70 J query won't fit.
        assert_eq!(
            g.admit(70.0, None),
            AdmitDecision::Reject(RejectReason::GlobalBudgetExhausted)
        );
    }

    #[test]
    fn unknown_tenant_rejected() {
        let g = gov();
        let t = TenantId::new("acme");
        assert_eq!(
            g.admit(1.0, Some(&t)),
            AdmitDecision::Reject(RejectReason::UnknownTenant)
        );
    }

    #[test]
    fn tenant_quota_enforced() {
        let mut g = gov();
        let t = TenantId::new("acme");
        g.set_tenant_quota(t.clone(), TenantQuota::new(5.0));
        assert_eq!(g.admit(3.0, Some(&t)), AdmitDecision::Admit);
        assert_eq!(
            g.admit(6.0, Some(&t)),
            AdmitDecision::Reject(RejectReason::TenantQuotaExhausted)
        );
    }

    #[test]
    fn tenant_isolation() {
        let mut g = gov();
        let a = TenantId::new("a");
        let b = TenantId::new("b");
        g.set_tenant_quota(a.clone(), TenantQuota::new(5.0));
        g.set_tenant_quota(b.clone(), TenantQuota::new(5.0));
        g.record(5.0, Some(&a));
        // a is exhausted, b is untouched.
        assert_eq!(
            g.admit(1.0, Some(&a)),
            AdmitDecision::Reject(RejectReason::TenantQuotaExhausted)
        );
        assert_eq!(g.admit(4.0, Some(&b)), AdmitDecision::Admit);
    }

    #[test]
    fn window_reset_clears_counters() {
        let mut g = gov();
        let t = TenantId::new("acme");
        g.set_tenant_quota(t.clone(), TenantQuota::new(5.0));
        g.record(50.0, Some(&t));
        g.reset_window(1_000 + 60); // window elapsed
        assert_eq!(g.spent, 0.0);
        assert_eq!(g.tenant_quotas.get(&t).unwrap().joules_spent, 0.0);
        assert_eq!(g.window_start_secs, 1_060);
    }

    #[test]
    fn window_does_not_reset_early() {
        let mut g = gov();
        g.record(50.0, None);
        g.reset_window(1_000 + 30); // still within window
        assert!((g.spent - 50.0).abs() < 1e-9);
    }

    #[test]
    fn kill_switch_rejects_everything() {
        let mut g = gov();
        g.engage_kill_switch();
        assert_eq!(
            g.admit(0.0, None),
            AdmitDecision::Reject(RejectReason::KillSwitch)
        );
        g.disengage_kill_switch();
        assert_eq!(g.admit(0.0, None), AdmitDecision::Admit);
    }

    #[test]
    fn kill_switch_takes_precedence_over_budget() {
        let mut g = gov();
        g.engage_kill_switch();
        // Even a free query is rejected, and by KillSwitch not budget.
        assert_eq!(
            g.admit(0.0, None),
            AdmitDecision::Reject(RejectReason::KillSwitch)
        );
    }

    #[test]
    fn tenant_quota_remaining() {
        let mut q = TenantQuota::new(10.0);
        q.joules_spent = 7.0;
        assert!((q.remaining() - 3.0).abs() < 1e-9);
        q.joules_spent = 15.0;
        assert_eq!(q.remaining(), 0.0); // never negative
    }

    #[test]
    fn untenanted_query_ignores_tenant_map() {
        let mut g = gov();
        g.set_tenant_quota(TenantId::new("acme"), TenantQuota::new(0.0));
        // No tenant supplied → global budget only, acme's zero quota
        // does not block.
        assert_eq!(g.admit(1.0, None), AdmitDecision::Admit);
    }

    #[test]
    fn decision_is_admit_helper() {
        assert!(AdmitDecision::Admit.is_admit());
        assert!(!AdmitDecision::Reject(RejectReason::KillSwitch).is_admit());
    }
}
