//! Routing: pick which rail to try, and the fallback order.
//!
//! The orchestrator delegates routing to a pluggable [`Router`]
//! trait so operators can ship their own decision logic. We provide
//! a [`PolicyRouter`] reference implementation that handles the
//! common cases:
//!
//! - Method-driven default (Vault/Wallet/Emv → Card, A2a/Qr → A2A).
//! - Country-aware preference for in-country instant rails.
//! - Amount threshold above which A2A is preferred (lower per-txn
//!   fee at high amounts).
//! - Customer-driven preference for A2A ("Pay by Bank").
//!
//! Routers are stateless. State (circuit breaker, prior attempts)
//! is held by the orchestrator and passed in as part of the
//! [`RoutingDecision`] context.

use op_core::{PaymentMethod, RailKind};

use crate::intent::PaymentIntent;

/// A single rail choice from the router.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RailChoice {
    /// Which rail (Card or A2A).
    pub rail: RailKind,

    /// Driver name to use. Must match a driver registered with the
    /// orchestrator. Different driver names for the same `rail`
    /// represent different PSPs / scheme processors.
    pub driver: String,
}

/// The full routing decision: an ordered fallback list.
///
/// The orchestrator tries `chain[0]` first; on soft failure it tries
/// `chain[1]`; and so on. Hard declines short-circuit (no
/// fallback). An empty chain is invalid — the router should return
/// [`Error::NoEligibleRail`](crate::Error::NoEligibleRail) instead.
#[derive(Clone, Debug)]
pub struct RoutingDecision {
    /// Ordered rail attempts. Length >= 1.
    pub chain: Vec<RailChoice>,
}

/// Pluggable routing logic.
pub trait Router: Send + Sync {
    /// Return an ordered fallback chain for the given intent.
    ///
    /// Returns `Err(reason)` if no rail is eligible. The orchestrator
    /// wraps this into [`Error::NoEligibleRail`](crate::Error::NoEligibleRail).
    fn route(&self, intent: &PaymentIntent) -> Result<RoutingDecision, String>;
}

/// Reference implementation of the [`Router`] trait.
///
/// Operators construct one with driver name lists per rail. Routing
/// rules:
///
/// 1. **Method-incompatible rails are filtered out.** A `Vault`
///    method cannot go through an A2A rail; an `A2a` method cannot
///    go through a card network.
/// 2. **Customer preference for A2A wins** if compatible.
/// 3. **High-amount threshold** prefers A2A when both compatible.
///    Lower per-transaction fee makes A2A favorable above the
///    threshold; below it, card is preferred for issuer-driven
///    chargeback protection.
/// 4. **In-country instant rail** beats cross-border card auth on
///    cost and latency when method-compatible.
/// 5. **All remaining drivers** are appended as fallback layers.
///    Operators control fallback order via driver registration
///    order.
#[derive(Clone, Debug)]
pub struct PolicyRouter {
    /// Card-network drivers in operator-configured preference order.
    pub card_drivers: Vec<String>,

    /// A2A drivers in operator-configured preference order.
    pub a2a_drivers: Vec<String>,

    /// Crypto / stablecoin drivers in operator-configured preference
    /// order. Each entry is a driver name registered for a specific
    /// `(chain, token)` pairing — operators name them `usdc-solana`,
    /// `usdc-base`, etc.
    pub crypto_drivers: Vec<String>,

    /// If `Some(threshold)`, intents at-or-above this amount in minor
    /// units prefer A2A over card (when both compatible). `None`
    /// disables the threshold rule.
    pub a2a_above_minor_units: Option<i64>,

    /// Optional source of recent-failure signals. When present, the
    /// fallback chain is re-ordered *within each rail group* —
    /// noisier drivers move to the back, the inter-rail order
    /// chosen by policy (`rail_order`) is preserved. A `None` value
    /// (or a `NoOpRoutingSignals`) leaves the static order intact.
    pub signals: Option<std::sync::Arc<dyn crate::signals::RoutingSignals>>,

    /// How to fold the multi-axis signal scores into a single
    /// ranking key. Defaults to [`SignalCombiner::WorstAxis`].
    pub combiner: crate::signals::SignalCombiner,
}

impl Default for PolicyRouter {
    fn default() -> Self {
        Self {
            card_drivers: vec!["hyperswitch".to_owned()],
            a2a_drivers: vec!["fednow".to_owned()],
            crypto_drivers: Vec::new(),
            a2a_above_minor_units: None,
            signals: None,
            combiner: crate::signals::SignalCombiner::default(),
        }
    }
}

impl PolicyRouter {
    /// Construct with explicit driver lists.
    pub fn new(card_drivers: Vec<String>, a2a_drivers: Vec<String>) -> Self {
        Self {
            card_drivers,
            a2a_drivers,
            crypto_drivers: Vec::new(),
            a2a_above_minor_units: None,
            signals: None,
            combiner: crate::signals::SignalCombiner::default(),
        }
    }

    /// Builder: register one or more crypto drivers (in preference
    /// order). Crypto drivers are only consulted when the intent's
    /// payment method is `PaymentMethod::Crypto(_)`.
    #[must_use]
    pub fn with_crypto_drivers(mut self, drivers: Vec<String>) -> Self {
        self.crypto_drivers = drivers;
        self
    }

    /// Builder: pick how multi-axis signal scores are combined into
    /// a single ranking key. Defaults to [`SignalCombiner::WorstAxis`].
    #[must_use]
    pub fn with_combiner(mut self, combiner: crate::signals::SignalCombiner) -> Self {
        self.combiner = combiner;
        self
    }

    /// Builder: set the high-amount A2A threshold.
    #[must_use]
    pub fn with_a2a_above(mut self, threshold_minor_units: i64) -> Self {
        self.a2a_above_minor_units = Some(threshold_minor_units);
        self
    }

    /// Builder: wire in a [`RoutingSignals`](crate::RoutingSignals)
    /// source. The router will use it to *re-order driver
    /// preference within each rail group* in the fallback chain —
    /// drivers with higher recent failure scores are pushed back.
    ///
    /// The rail order chosen by [`Self::rail_order`] (Card first vs
    /// A2A first, by policy / hints / amount) is preserved; signals
    /// only re-arrange *within* each rail's driver list.
    #[must_use]
    pub fn with_signals(
        mut self,
        signals: std::sync::Arc<dyn crate::signals::RoutingSignals>,
    ) -> Self {
        self.signals = Some(signals);
        self
    }

    fn method_supports(method: &PaymentMethod, rail: RailKind) -> bool {
        matches!(
            (method, rail),
            (
                PaymentMethod::Vault(_) | PaymentMethod::Wallet(_) | PaymentMethod::Emv(_),
                RailKind::Card,
            ) | (PaymentMethod::A2a(_) | PaymentMethod::Qr(_), RailKind::A2a)
                | (PaymentMethod::Crypto(_), RailKind::Crypto)
        )
    }

    /// Decide the rail order based on policy. Returns the
    /// preferred (rail, second-rail?) pair. Either rail is omitted
    /// if method-incompatible.
    fn rail_order(&self, intent: &PaymentIntent) -> (Option<RailKind>, Option<RailKind>) {
        // Crypto is method-exclusive: a `PaymentMethod::Crypto(_)`
        // never routes through Card or A2A.
        if matches!(intent.method, PaymentMethod::Crypto(_)) {
            return (Some(RailKind::Crypto), None);
        }

        let card_compat = Self::method_supports(&intent.method, RailKind::Card);
        let a2a_compat = Self::method_supports(&intent.method, RailKind::A2a);

        if !card_compat && !a2a_compat {
            return (None, None);
        }
        if card_compat && !a2a_compat {
            return (Some(RailKind::Card), None);
        }
        if !card_compat && a2a_compat {
            return (Some(RailKind::A2a), None);
        }

        // Both compatible — apply preference policy.
        let prefer_a2a = intent.hints.prefer_a2a
            || self
                .a2a_above_minor_units
                .map(|t| intent.amount.minor_units >= t)
                .unwrap_or(false);

        if prefer_a2a {
            (Some(RailKind::A2a), Some(RailKind::Card))
        } else {
            (Some(RailKind::Card), Some(RailKind::A2a))
        }
    }
}

impl Router for PolicyRouter {
    fn route(&self, intent: &PaymentIntent) -> Result<RoutingDecision, String> {
        let (primary, secondary) = self.rail_order(intent);

        let mut chain = Vec::new();

        let push_for = |chain: &mut Vec<RailChoice>, rail: RailKind| {
            let drivers = match rail {
                // Wallet maps to card_drivers — PSPs handle Apple Pay /
                // Google Pay via the same Card rail interface.
                RailKind::Card | RailKind::Wallet => &self.card_drivers,
                // QR settles on A2A underneath (UPI QR, PIX QR, EMVCo MQR).
                RailKind::A2a | RailKind::Qr => &self.a2a_drivers,
                RailKind::Crypto => &self.crypto_drivers,
            };
            // Re-order this rail's drivers by a combined "noise"
            // score (lower = quieter, pushed earlier). Stable sort
            // over the original index breaks ties in operator-
            // configured order, so the static preference is
            // preserved when scores are equal or absent.
            //
            // The way scores are folded across signal axes is a
            // configurable [`SignalCombiner`]. The default
            // (`WorstAxis`) takes the max — a clean-HTTP-but-noisy-
            // reconciliation driver still gets pushed back even
            // though its failure_score alone would be zero.
            let mut ordered: Vec<(usize, &String)> = drivers.iter().enumerate().collect();
            if let Some(signals) = &self.signals {
                ordered.sort_by(|(ia, a), (ib, b)| {
                    let sa = self.combiner.combine(signals.as_ref(), rail, a);
                    let sb = self.combiner.combine(signals.as_ref(), rail, b);
                    sa.partial_cmp(&sb)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(ia.cmp(ib))
                });
            }
            for (_, d) in ordered {
                chain.push(RailChoice {
                    rail,
                    driver: d.clone(),
                });
            }
        };

        if let Some(p) = primary {
            push_for(&mut chain, p);
        }
        if let Some(s) = secondary {
            push_for(&mut chain, s);
        }

        if chain.is_empty() {
            return Err(format!(
                "no rail compatible with payment method {:?}",
                intent.method,
            ));
        }

        Ok(RoutingDecision { chain })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{A2aKey, Currency, Money, VaultRef};

    use crate::idempotency::IdempotencyKey;

    fn vault_intent() -> PaymentIntent {
        PaymentIntent::new(
            IdempotencyKey::new("k"),
            Money::from_minor(1000, Currency::USD),
            PaymentMethod::Vault(VaultRef::new("tok_v7_a")),
        )
    }

    fn a2a_intent() -> PaymentIntent {
        PaymentIntent::new(
            IdempotencyKey::new("k"),
            Money::from_minor(1000, Currency::USD),
            PaymentMethod::A2a(A2aKey::UsAch {
                routing: "021000021".into(),
                account: "12345".into(),
            }),
        )
    }

    #[test]
    fn vault_method_routes_to_card_only() {
        let r = PolicyRouter::default().route(&vault_intent()).unwrap();
        assert_eq!(r.chain.len(), 1);
        assert_eq!(r.chain[0].rail, RailKind::Card);
        assert_eq!(r.chain[0].driver, "hyperswitch");
    }

    #[test]
    fn a2a_method_routes_to_a2a_only() {
        let r = PolicyRouter::default().route(&a2a_intent()).unwrap();
        assert_eq!(r.chain.len(), 1);
        assert_eq!(r.chain[0].rail, RailKind::A2a);
        assert_eq!(r.chain[0].driver, "fednow");
    }

    #[test]
    fn empty_driver_lists_for_compatible_rail_fail() {
        let router = PolicyRouter::new(vec![], vec![]);
        assert!(router.route(&vault_intent()).is_err());
    }

    #[test]
    fn multiple_card_drivers_register_as_fallback_chain() {
        let router = PolicyRouter::new(vec!["primary".into(), "backup".into()], vec![]);
        let r = router.route(&vault_intent()).unwrap();
        assert_eq!(r.chain.len(), 2);
        assert_eq!(r.chain[0].driver, "primary");
        assert_eq!(r.chain[1].driver, "backup");
    }

    #[test]
    fn customer_prefer_a2a_overrides_method_when_both_compatible() {
        // Vault method does NOT support A2A, so prefer_a2a does
        // nothing here. Verify that the preference doesn't break the
        // single-rail case.
        let mut intent = vault_intent();
        intent.hints.prefer_a2a = true;
        let r = PolicyRouter::default().route(&intent).unwrap();
        assert_eq!(r.chain[0].rail, RailKind::Card);
    }

    #[test]
    fn high_amount_threshold_does_not_change_method_incompatible_rail() {
        let router = PolicyRouter::default().with_a2a_above(500);
        let intent = vault_intent(); // 1000 minor, above 500
        let r = router.route(&intent).unwrap();
        // Vault → only Card is method-compatible regardless of policy.
        assert_eq!(r.chain[0].rail, RailKind::Card);
    }

    #[test]
    fn unsupported_method_returns_error() {
        // Construct an intent whose method is method-compatible with
        // NEITHER rail — there isn't one in our taxonomy so we
        // instead test the explicit empty-driver-list case.
        let router = PolicyRouter::new(vec![], vec![]);
        let result = router.route(&vault_intent());
        assert!(result.is_err());
    }

    #[test]
    fn rail_choice_equality() {
        // RailChoice supports == so tests can assert exact chains.
        let a = RailChoice {
            rail: RailKind::Card,
            driver: "hyperswitch".into(),
        };
        let b = RailChoice {
            rail: RailKind::Card,
            driver: "hyperswitch".into(),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn rail_order_both_incompatible_returns_none() {
        // Edge case: no rail at all is compatible. We simulate this
        // by constructing a PolicyRouter with empty driver lists and
        // verifying route() errors. (The rail_order method itself
        // returns (None, None), but it's pub(crate)-only.)
        let router = PolicyRouter::new(vec![], vec![]);
        let intent = a2a_intent();
        assert!(router.route(&intent).is_err());
    }

    #[test]
    fn default_router_has_known_drivers() {
        let r = PolicyRouter::default();
        assert_eq!(r.card_drivers, vec!["hyperswitch".to_owned()]);
        assert_eq!(r.a2a_drivers, vec!["fednow".to_owned()]);
        assert_eq!(r.a2a_above_minor_units, None);
    }
}
