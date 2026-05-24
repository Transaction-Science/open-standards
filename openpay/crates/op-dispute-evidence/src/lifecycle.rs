//! Dispute lifecycle state machine.
//!
//! Every card network has its own stage names (Visa "pre-arbitration",
//! Mastercard "second presentment", Amex "final dispute decision",
//! etc.). The operator-facing lifecycle collapses them onto one
//! ordered ladder so a single switch can drive UX, scheduling, and
//! webhook emission across all five networks.
//!
//! ```text
//!   Retrieval ──► FirstChargeback ──► Representment ──► PreArbitration ──► Arbitration ──► Final{Won|Lost}
//!                       │
//!                       └──► Final{Accepted}  (merchant accepts loss)
//! ```
//!
//! Per the OpenPay deterministic-contract doctrine, the state
//! machine is a *pure function over events* — no I/O, no hidden
//! storage, no inference. Operators feed it events; it returns the
//! next phase or an [`Error::IllegalTransition`].

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::error::{Error, Result};

/// A phase in the unified dispute lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    /// Pre-dispute: issuer asked the acquirer for the transaction
    /// receipt. Often a soft-touch step before a formal chargeback.
    /// Visa calls this "retrieval request" (still rare under VCR);
    /// Mastercom emits it as message 6305.
    Retrieval,
    /// The dispute is open and funds have been provisionally pulled
    /// from the merchant.
    FirstChargeback,
    /// Merchant has filed evidence to win back the funds. Visa:
    /// "Dispute Response". Mastercard: "Second Presentment".
    Representment,
    /// Issuer escalated after representment. Visa: "Pre-Arbitration".
    /// Mastercard: "Arbitration Chargeback" pre-filing.
    PreArbitration,
    /// Filed with the network for binding arbitration.
    Arbitration,
    /// Terminal: merchant accepted the loss without representment.
    FinalAccepted,
    /// Terminal: merchant won (chargeback reversed, funds restored).
    FinalWon,
    /// Terminal: merchant lost.
    FinalLost,
}

impl Phase {
    /// True if no further transitions are allowed.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::FinalAccepted | Self::FinalWon | Self::FinalLost
        )
    }

    /// Returns the legal next phases from `self`.
    ///
    /// Operators can use this to render a UI of available actions
    /// without re-implementing the transition table.
    #[must_use]
    pub fn legal_next(self) -> &'static [Phase] {
        match self {
            Self::Retrieval => &[Phase::FirstChargeback, Phase::FinalAccepted],
            Self::FirstChargeback => &[
                Phase::Representment,
                Phase::FinalAccepted,
                Phase::FinalLost,
            ],
            Self::Representment => &[
                Phase::PreArbitration,
                Phase::FinalWon,
                Phase::FinalLost,
            ],
            Self::PreArbitration => &[Phase::Arbitration, Phase::FinalWon, Phase::FinalLost],
            Self::Arbitration => &[Phase::FinalWon, Phase::FinalLost],
            Self::FinalAccepted | Self::FinalWon | Self::FinalLost => &[],
        }
    }
}

/// A single event applied to the lifecycle.
///
/// Events are *intent*; the [`LifecycleMachine`] decides whether
/// they're permitted and what the resulting phase is.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleEvent {
    /// Issuer filed a retrieval request.
    RetrievalReceived {
        /// Wall-clock time the event was observed.
        at: OffsetDateTime,
    },
    /// First chargeback received from the issuer.
    ChargebackReceived {
        /// Wall-clock time the event was observed.
        at: OffsetDateTime,
    },
    /// Merchant elected to accept the loss without representing.
    Accepted {
        /// Wall-clock time the event was observed.
        at: OffsetDateTime,
    },
    /// Merchant filed representment with a packaged evidence
    /// bundle.
    RepresentmentFiled {
        /// Wall-clock time the event was observed.
        at: OffsetDateTime,
    },
    /// Issuer escalated to pre-arbitration after the representment.
    PreArbReceived {
        /// Wall-clock time the event was observed.
        at: OffsetDateTime,
    },
    /// Case escalated to network arbitration.
    ArbitrationFiled {
        /// Wall-clock time the event was observed.
        at: OffsetDateTime,
    },
    /// Network ruled in the merchant's favor.
    Won {
        /// Wall-clock time the event was observed.
        at: OffsetDateTime,
    },
    /// Network ruled against the merchant.
    Lost {
        /// Wall-clock time the event was observed.
        at: OffsetDateTime,
    },
}

impl LifecycleEvent {
    /// Wall-clock time this event was observed at.
    #[must_use]
    pub fn at(&self) -> OffsetDateTime {
        match self {
            Self::RetrievalReceived { at }
            | Self::ChargebackReceived { at }
            | Self::Accepted { at }
            | Self::RepresentmentFiled { at }
            | Self::PreArbReceived { at }
            | Self::ArbitrationFiled { at }
            | Self::Won { at }
            | Self::Lost { at } => *at,
        }
    }
}

/// Pure state machine driving the dispute lifecycle.
///
/// Construct one with [`LifecycleMachine::new`], feed it events via
/// [`LifecycleMachine::apply`], and read the current phase /
/// transition log back at any time. No internal mutability is hidden
/// from the caller — `&mut self` on apply is the only mutation
/// surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleMachine {
    phase: Phase,
    history: Vec<(Phase, LifecycleEvent)>,
}

impl LifecycleMachine {
    /// Start a fresh machine in the [`Phase::Retrieval`] phase.
    ///
    /// Most disputes skip retrieval and start at
    /// [`Phase::FirstChargeback`] in practice; use
    /// [`LifecycleMachine::starting_at`] to begin there.
    #[must_use]
    pub fn new() -> Self {
        Self {
            phase: Phase::Retrieval,
            history: Vec::new(),
        }
    }

    /// Start the machine at an arbitrary non-terminal phase.
    ///
    /// Useful when bootstrapping from an existing PSP-side dispute
    /// that's already past first-chargeback.
    #[must_use]
    pub fn starting_at(phase: Phase) -> Self {
        Self {
            phase,
            history: Vec::new(),
        }
    }

    /// Current phase.
    #[must_use]
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Chronological list of `(prev_phase, event)` transitions.
    #[must_use]
    pub fn history(&self) -> &[(Phase, LifecycleEvent)] {
        &self.history
    }

    /// Apply an event and return the resulting phase.
    ///
    /// # Errors
    ///
    /// Returns [`Error::IllegalTransition`] when the event would
    /// move out of a terminal phase or into a phase not listed in
    /// [`Phase::legal_next`].
    pub fn apply(&mut self, event: LifecycleEvent) -> Result<Phase> {
        let next = self.derive_next(&event)?;
        if !self.phase.legal_next().contains(&next) {
            return Err(Error::IllegalTransition {
                from: self.phase,
                to: next,
            });
        }
        let prev = self.phase;
        self.phase = next;
        self.history.push((prev, event));
        Ok(self.phase)
    }

    fn derive_next(&self, event: &LifecycleEvent) -> Result<Phase> {
        // Terminal phases reject every event.
        if self.phase.is_terminal() {
            // Pretend the event would have targeted FinalLost so the
            // returned error names a concrete target; the contract
            // is "no transitions from terminal".
            return Err(Error::IllegalTransition {
                from: self.phase,
                to: Phase::FinalLost,
            });
        }
        Ok(match event {
            LifecycleEvent::RetrievalReceived { .. } => Phase::Retrieval,
            LifecycleEvent::ChargebackReceived { .. } => Phase::FirstChargeback,
            LifecycleEvent::Accepted { .. } => Phase::FinalAccepted,
            LifecycleEvent::RepresentmentFiled { .. } => Phase::Representment,
            LifecycleEvent::PreArbReceived { .. } => Phase::PreArbitration,
            LifecycleEvent::ArbitrationFiled { .. } => Phase::Arbitration,
            LifecycleEvent::Won { .. } => Phase::FinalWon,
            LifecycleEvent::Lost { .. } => Phase::FinalLost,
        })
    }
}

impl Default for LifecycleMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: i64) -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(s).expect("valid unix time")
    }

    #[test]
    fn happy_path_retrieval_to_won() {
        let mut m = LifecycleMachine::new();
        assert_eq!(m.phase(), Phase::Retrieval);
        m.apply(LifecycleEvent::ChargebackReceived { at: t(1) })
            .expect("retrieval -> first chargeback");
        m.apply(LifecycleEvent::RepresentmentFiled { at: t(2) })
            .expect("first chargeback -> representment");
        m.apply(LifecycleEvent::Won { at: t(3) })
            .expect("representment -> won");
        assert_eq!(m.phase(), Phase::FinalWon);
        assert!(m.phase().is_terminal());
    }

    #[test]
    fn terminal_rejects_further_events() {
        let mut m = LifecycleMachine::starting_at(Phase::Representment);
        m.apply(LifecycleEvent::Lost { at: t(1) }).expect("lose");
        let err = m
            .apply(LifecycleEvent::Won { at: t(2) })
            .expect_err("must reject");
        match err {
            Error::IllegalTransition { from, .. } => assert_eq!(from, Phase::FinalLost),
            other => panic!("unexpected: {other:?}"),
        }
    }
}
