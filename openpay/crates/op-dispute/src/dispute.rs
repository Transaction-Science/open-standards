//! [`Dispute`] domain type, its [`Status`] state machine, and
//! [`DisputeId`].

use op_core::Money;
use op_ledger::TransactionId;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::evidence::EvidenceRef;
use crate::reason::DisputeReason;

/// Opaque id for a dispute.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DisputeId(pub Uuid);

impl DisputeId {
    /// Mint a fresh id (UUID v7 — time-sortable).
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// Wrap an existing UUID.
    #[must_use]
    pub fn from_uuid(u: Uuid) -> Self {
        Self(u)
    }

    /// The wrapped UUID.
    #[must_use]
    pub fn as_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for DisputeId {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Display for DisputeId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

/// Where a dispute is in its lifecycle.
///
/// The canonical card-network workflow is:
///   `Inquiry` (retrieval request, optional) → `Chargeback` (funds
///   pulled) → either `Representment` (merchant defends, evidence
///   submitted) → `Won` / `Lost` / `Accepted`, or directly `Accepted`
///   / `Lost` (merchant skips representment). A2A rails compress
///   this — see crate docs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    /// Issuer requested information about a transaction — funds
    /// not yet pulled. Optional stage; many disputes skip it.
    Inquiry,
    /// Funds clawed back from the merchant; merchant has a fixed
    /// window to decide whether to defend.
    Chargeback,
    /// Merchant submitted evidence and is contesting. Awaiting
    /// the issuer's response.
    Representment,
    /// Merchant won — funds restored, no chargeback fee (depending
    /// on the network's policy).
    Won,
    /// Merchant lost — funds stay clawed back, chargeback fee
    /// applies. Operator typically posts a reversing ledger
    /// transaction.
    Lost,
    /// Merchant chose not to defend; funds stay clawed back without
    /// going through representment. Identical bookkeeping to `Lost`;
    /// distinct status for reporting.
    Accepted,
}

impl Status {
    /// True for `Won`, `Lost`, `Accepted`.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Won | Self::Lost | Self::Accepted)
    }

    /// Short string code for filtering / reporting.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::Inquiry => "inquiry",
            Self::Chargeback => "chargeback",
            Self::Representment => "representment",
            Self::Won => "won",
            Self::Lost => "lost",
            Self::Accepted => "accepted",
        }
    }
}

/// One dispute record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dispute {
    /// Stable id.
    pub id: DisputeId,
    /// The ledger transaction being disputed.
    pub original_tx_id: TransactionId,
    /// Operator-supplied idempotency token. Conventionally reused
    /// as the `external_id` on the reversing ledger transaction
    /// when the dispute is `Lost` or `Accepted`.
    pub external_id: Option<String>,
    /// Disputed amount. Equal to the original tx for full
    /// chargebacks; smaller for partial disputes (rare but legal
    /// on some networks).
    pub amount: Money,
    /// Normalized reason.
    pub reason: DisputeReason,
    /// Raw network reason code (e.g. `"10.4"` for Visa fraud
    /// reason 10.4). Operator-supplied; we just store it.
    pub network_reason_code: Option<String>,
    /// Current lifecycle position.
    pub status: Status,
    /// When the dispute was opened (unix epoch seconds).
    pub opened_at_unix_secs: u64,
    /// Deadline by which the merchant must act, if known. Past this
    /// timestamp, declining to represent = automatic loss.
    pub due_by_unix_secs: Option<u64>,
    /// Evidence the merchant has attached, in attachment order.
    pub evidence: Vec<EvidenceRef>,
    /// Free-form operator metadata.
    pub metadata: Vec<(String, String)>,
}

impl Dispute {
    /// Construct a dispute in the [`Status::Chargeback`] state —
    /// the most common entry point, since most disputes the
    /// merchant learns about have already had funds pulled.
    ///
    /// Use [`Self::with_status`] to override (e.g. `Inquiry` for a
    /// retrieval request before the chargeback).
    ///
    /// # Errors
    /// [`Error::Invalid`] if `amount` is negative.
    pub fn new(
        original_tx_id: TransactionId,
        amount: Money,
        reason: DisputeReason,
        opened_at_unix_secs: u64,
    ) -> Result<Self> {
        if amount.minor_units < 0 {
            return Err(Error::Invalid(format!(
                "dispute amount must be non-negative, got {}",
                amount.minor_units
            )));
        }
        Ok(Self {
            id: DisputeId::new(),
            original_tx_id,
            external_id: None,
            amount,
            reason,
            network_reason_code: None,
            status: Status::Chargeback,
            opened_at_unix_secs,
            due_by_unix_secs: None,
            evidence: Vec::new(),
            metadata: Vec::new(),
        })
    }

    /// Builder: override the initial status.
    #[must_use]
    pub fn with_status(mut self, status: Status) -> Self {
        self.status = status;
        self
    }

    /// Builder: attach an external id for idempotency.
    #[must_use]
    pub fn with_external_id(mut self, id: impl Into<String>) -> Self {
        self.external_id = Some(id.into());
        self
    }

    /// Builder: attach the raw network reason code.
    #[must_use]
    pub fn with_network_reason_code(mut self, code: impl Into<String>) -> Self {
        self.network_reason_code = Some(code.into());
        self
    }

    /// Builder: set the response deadline.
    #[must_use]
    pub fn with_due_by(mut self, due_by_unix_secs: u64) -> Self {
        self.due_by_unix_secs = Some(due_by_unix_secs);
        self
    }

    /// Builder: attach a metadata key/value pair.
    #[must_use]
    pub fn with_metadata(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.metadata.push((k.into(), v.into()));
        self
    }

    /// Append a piece of evidence to the dispute's record.
    ///
    /// # Errors
    /// [`Error::InvalidTransition`] if the dispute is already
    /// terminal — attaching evidence after resolution is a category
    /// error.
    pub fn attach_evidence(&mut self, ev: EvidenceRef) -> Result<()> {
        if self.status.is_terminal() {
            return Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: self.status.code().to_owned(),
            });
        }
        self.evidence.push(ev);
        Ok(())
    }

    /// Move to [`Status::Chargeback`] from [`Status::Inquiry`] —
    /// the issuer escalated.
    ///
    /// # Errors
    /// [`Error::InvalidTransition`] if not in `Inquiry`.
    pub fn escalate_to_chargeback(&mut self) -> Result<()> {
        match self.status {
            Status::Inquiry => {
                self.status = Status::Chargeback;
                Ok(())
            }
            ref other => Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: other.code().to_owned(),
            }),
        }
    }

    /// Move to [`Status::Representment`] from [`Status::Chargeback`].
    /// Merchant has decided to defend and (typically) submitted
    /// evidence.
    ///
    /// # Errors
    /// [`Error::InvalidTransition`] if not in `Chargeback`.
    pub fn represent(&mut self) -> Result<()> {
        match self.status {
            Status::Chargeback => {
                self.status = Status::Representment;
                Ok(())
            }
            ref other => Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: other.code().to_owned(),
            }),
        }
    }

    /// Terminal transition to [`Status::Won`].
    ///
    /// # Errors
    /// [`Error::InvalidTransition`] if already terminal.
    pub fn resolve_won(&mut self) -> Result<()> {
        if self.status.is_terminal() {
            return Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: self.status.code().to_owned(),
            });
        }
        self.status = Status::Won;
        Ok(())
    }

    /// Terminal transition to [`Status::Lost`].
    ///
    /// # Errors
    /// [`Error::InvalidTransition`] if already terminal.
    pub fn resolve_lost(&mut self) -> Result<()> {
        if self.status.is_terminal() {
            return Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: self.status.code().to_owned(),
            });
        }
        self.status = Status::Lost;
        Ok(())
    }

    /// Terminal transition to [`Status::Accepted`].
    ///
    /// # Errors
    /// [`Error::InvalidTransition`] if already terminal.
    pub fn accept(&mut self) -> Result<()> {
        if self.status.is_terminal() {
            return Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: self.status.code().to_owned(),
            });
        }
        self.status = Status::Accepted;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money};

    fn fresh() -> Dispute {
        Dispute::new(
            TransactionId::new(),
            Money::from_minor(2_500, Currency::USD),
            DisputeReason::Fraudulent,
            1_000,
        )
        .unwrap()
    }

    #[test]
    fn negative_amount_rejected() {
        let err = Dispute::new(
            TransactionId::new(),
            Money::from_minor(-1, Currency::USD),
            DisputeReason::Fraudulent,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::Invalid(_)));
    }

    #[test]
    fn happy_path_represent_and_win() {
        let mut d = fresh();
        assert_eq!(d.status.code(), "chargeback");
        d.attach_evidence(EvidenceRef::new("receipt", "s3://b/k", 1_001))
            .unwrap();
        d.represent().unwrap();
        assert_eq!(d.status.code(), "representment");
        d.resolve_won().unwrap();
        assert!(d.status.is_terminal());
    }

    #[test]
    fn accept_short_circuits_directly() {
        let mut d = fresh();
        d.accept().unwrap();
        assert_eq!(d.status.code(), "accepted");
    }

    #[test]
    fn evidence_cant_be_attached_after_resolution() {
        let mut d = fresh();
        d.resolve_lost().unwrap();
        let err = d
            .attach_evidence(EvidenceRef::new("late", "s3://b/late", 9_000))
            .unwrap_err();
        assert!(matches!(err, Error::InvalidTransition { .. }));
    }

    #[test]
    fn inquiry_escalates_to_chargeback() {
        let mut d = fresh().with_status(Status::Inquiry);
        d.escalate_to_chargeback().unwrap();
        assert_eq!(d.status.code(), "chargeback");
        // Can't re-escalate from chargeback.
        assert!(d.escalate_to_chargeback().is_err());
    }

    #[test]
    fn represent_only_from_chargeback() {
        let mut d = fresh().with_status(Status::Inquiry);
        assert!(d.represent().is_err());
        d.escalate_to_chargeback().unwrap();
        d.represent().unwrap();
    }

    #[test]
    fn terminal_states_reject_further_resolutions() {
        let mut d = fresh();
        d.resolve_lost().unwrap();
        assert!(d.resolve_won().is_err());
        assert!(d.accept().is_err());
    }
}
