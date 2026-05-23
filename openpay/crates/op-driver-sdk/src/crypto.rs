//! Deterministic mock crypto gateway.
//!
//! [`DeterministicCryptoGateway`] is to [`CryptoGateway`] what the
//! card / A2A mocks in this crate are to their traits: a
//! programmable, side-effect-free implementation operators use
//! for tests and bring-up without touching a real chain.
//!
//! Same control-axis pattern: default status, per-idempotency-key
//! overrides, amount-threshold rules, transport-error mode,
//! request history.

use std::sync::Mutex;

use op_core::{CryptoAddress, Money};
use op_rails_crypto::gateway::{CryptoDecision, CryptoStatus, CryptoTransferReq, StatusQueryReq};
use op_rails_crypto::{CryptoGateway, Error, Result, StableToken, TokenRef};
use serde::{Deserialize, Serialize};

/// Programmable crypto gateway mock.
pub struct DeterministicCryptoGateway {
    name: &'static str,
    token: TokenRef,
    policy: Mutex<Policy>,
    history: Mutex<History>,
}

#[derive(Default)]
struct Policy {
    default_status: Option<CryptoStatus>,
    key_overrides: Vec<(String, CryptoStatus, Option<String>)>,
    amount_rules: Vec<AmountRule>,
    transport_error: Option<String>,
    next_tx_seq: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct AmountRule {
    op: Comparator,
    threshold_minor: i64,
    currency: String,
    status: CryptoStatus,
    reason: Option<String>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum Comparator {
    Ge,
    Gt,
    Le,
    Lt,
    Eq,
}

#[derive(Default)]
struct History {
    transfers: Vec<CryptoTransferReq>,
    queries: Vec<StatusQueryReq>,
}

impl DeterministicCryptoGateway {
    /// Fresh gateway servicing USDC on Base. Defaults to
    /// [`CryptoStatus::Finalized`] on every input.
    #[must_use]
    pub fn new() -> Self {
        Self::for_token(StableToken::UsdcBase)
    }

    /// Fresh gateway servicing a specific token.
    #[must_use]
    pub fn for_token(token: StableToken) -> Self {
        Self {
            name: "deterministic-crypto",
            token: token.token_ref(),
            policy: Mutex::default(),
            history: Mutex::default(),
        }
    }

    /// Builder: rename the gateway.
    #[must_use]
    pub fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }

    /// Builder: set the default status.
    #[must_use]
    pub fn with_default_status(self, status: CryptoStatus) -> Self {
        self.policy.lock().expect("poisoned").default_status = Some(status);
        self
    }

    /// Builder: force a specific status for an idempotency key.
    #[must_use]
    pub fn with_key_override(
        self,
        idempotency_key: impl Into<String>,
        status: CryptoStatus,
        reason: Option<String>,
    ) -> Self {
        self.policy.lock().expect("poisoned").key_overrides.push((
            idempotency_key.into(),
            status,
            reason,
        ));
        self
    }

    /// Builder: amount-threshold rule (`>=`).
    #[must_use]
    pub fn with_amount_ge(
        self,
        threshold: Money,
        status: CryptoStatus,
        reason: Option<String>,
    ) -> Self {
        self.policy
            .lock()
            .expect("poisoned")
            .amount_rules
            .push(AmountRule {
                op: Comparator::Ge,
                threshold_minor: threshold.minor_units,
                currency: threshold.currency.code().to_owned(),
                status,
                reason,
            });
        self
    }

    /// Builder: amount-threshold rule (`<`).
    #[must_use]
    pub fn with_amount_lt(
        self,
        threshold: Money,
        status: CryptoStatus,
        reason: Option<String>,
    ) -> Self {
        self.policy
            .lock()
            .expect("poisoned")
            .amount_rules
            .push(AmountRule {
                op: Comparator::Lt,
                threshold_minor: threshold.minor_units,
                currency: threshold.currency.code().to_owned(),
                status,
                reason,
            });
        self
    }

    /// Builder: every call returns `Err(Error::Transport(_))`.
    #[must_use]
    pub fn with_transport_error(self, message: impl Into<String>) -> Self {
        self.policy.lock().expect("poisoned").transport_error = Some(message.into());
        self
    }

    /// Inspect every transfer request the gateway has seen.
    #[must_use]
    pub fn transfer_history(&self) -> Vec<CryptoTransferReq> {
        self.history.lock().expect("poisoned").transfers.clone()
    }

    /// Inspect every status query seen.
    #[must_use]
    pub fn query_history(&self) -> Vec<StatusQueryReq> {
        self.history.lock().expect("poisoned").queries.clone()
    }

    fn next_tx_hash(&self) -> String {
        let mut p = self.policy.lock().expect("poisoned");
        p.next_tx_seq = p.next_tx_seq.saturating_add(1);
        // Synthetic hash — looks like an EVM hash (0x + 64 hex chars)
        // but the suffix is the sequence, padded.
        format!("0x{:064x}", p.next_tx_seq)
    }

    fn resolve_status(&self, req: &CryptoTransferReq) -> (CryptoStatus, Option<String>) {
        let p = self.policy.lock().expect("poisoned");
        for (k, status, reason) in &p.key_overrides {
            if k == &req.idempotency_key {
                return (*status, reason.clone());
            }
        }
        for rule in &p.amount_rules {
            if rule.currency != req.amount.currency.code() {
                continue;
            }
            let m = req.amount.minor_units;
            let matched = match rule.op {
                Comparator::Ge => m >= rule.threshold_minor,
                Comparator::Gt => m > rule.threshold_minor,
                Comparator::Le => m <= rule.threshold_minor,
                Comparator::Lt => m < rule.threshold_minor,
                Comparator::Eq => m == rule.threshold_minor,
            };
            if matched {
                return (rule.status, rule.reason.clone());
            }
        }
        (p.default_status.unwrap_or(CryptoStatus::Finalized), None)
    }
}

impl Default for DeterministicCryptoGateway {
    fn default() -> Self {
        Self::new()
    }
}

impl CryptoGateway for DeterministicCryptoGateway {
    fn name(&self) -> &'static str {
        self.name
    }

    fn chain(&self) -> &str {
        &self.token.chain
    }

    fn token(&self) -> &TokenRef {
        &self.token
    }

    fn supports(&self, to: &CryptoAddress) -> bool {
        to.chain == self.token.chain
    }

    fn submit_transfer(&self, req: &CryptoTransferReq) -> Result<CryptoDecision> {
        self.history
            .lock()
            .expect("poisoned")
            .transfers
            .push(req.clone());
        if let Some(msg) = self
            .policy
            .lock()
            .expect("poisoned")
            .transport_error
            .clone()
        {
            return Err(Error::Transport(msg));
        }
        if req.to.chain != self.token.chain {
            return Err(Error::UnsupportedChain(req.to.chain.clone()));
        }
        if req.token.contract != self.token.contract {
            return Err(Error::UnsupportedToken(req.token.symbol.clone()));
        }
        let (status, reason) = self.resolve_status(req);
        let confirmations = match status {
            CryptoStatus::Finalized => 12,
            CryptoStatus::Confirming => 3,
            _ => 0,
        };
        Ok(CryptoDecision {
            status,
            tx_hash: Some(self.next_tx_hash()),
            confirmations,
            settled_amount: matches!(status, CryptoStatus::Finalized).then_some(req.amount),
            raw_status: Some(format!("{status:?}").to_lowercase()),
            reason,
        })
    }

    fn query_status(&self, req: &StatusQueryReq) -> Result<CryptoDecision> {
        self.history
            .lock()
            .expect("poisoned")
            .queries
            .push(req.clone());
        // Conservative default: return Finalized so polling loops
        // terminate. Operators can override per-key via the policy
        // (matching by idempotency_key when provided).
        let (status, reason) = if let Some(k) = &req.idempotency_key {
            let p = self.policy.lock().expect("poisoned");
            p.key_overrides
                .iter()
                .find(|(kk, _, _)| kk == k)
                .map_or((CryptoStatus::Finalized, None), |(_, s, r)| (*s, r.clone()))
        } else {
            (CryptoStatus::Finalized, None)
        };
        Ok(CryptoDecision {
            status,
            tx_hash: Some(req.tx_hash.clone()),
            confirmations: if matches!(status, CryptoStatus::Finalized) {
                12
            } else {
                0
            },
            settled_amount: None,
            raw_status: Some(format!("{status:?}").to_lowercase()),
            reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money};

    fn transfer(key: &str, amount_minor: i64) -> CryptoTransferReq {
        let token = StableToken::UsdcBase.token_ref();
        CryptoTransferReq {
            token: token.clone(),
            to: CryptoAddress::new("base", "0xabc"),
            amount: Money::from_minor(amount_minor, Currency::USD),
            idempotency_key: key.into(),
            memo: None,
        }
    }

    #[test]
    fn default_finalizes() {
        let gw = DeterministicCryptoGateway::new();
        let d = gw.submit_transfer(&transfer("k1", 1000)).unwrap();
        assert_eq!(d.status, CryptoStatus::Finalized);
        assert!(d.tx_hash.unwrap().starts_with("0x"));
        assert_eq!(d.confirmations, 12);
    }

    #[test]
    fn key_override_rejects() {
        let gw = DeterministicCryptoGateway::new().with_key_override(
            "k-bad",
            CryptoStatus::Rejected,
            Some("revert".into()),
        );
        let d = gw.submit_transfer(&transfer("k-bad", 100)).unwrap();
        assert_eq!(d.status, CryptoStatus::Rejected);
        assert_eq!(d.reason.as_deref(), Some("revert"));
        assert_eq!(d.confirmations, 0);
    }

    #[test]
    fn amount_ge_triggers_confirming() {
        let gw = DeterministicCryptoGateway::new().with_amount_ge(
            Money::from_minor(1_000_000, Currency::USD),
            CryptoStatus::Confirming,
            None,
        );
        let big = gw.submit_transfer(&transfer("k-big", 2_000_000)).unwrap();
        assert_eq!(big.status, CryptoStatus::Confirming);
        let small = gw.submit_transfer(&transfer("k-small", 100)).unwrap();
        assert_eq!(small.status, CryptoStatus::Finalized);
    }

    #[test]
    fn transport_error_short_circuits() {
        let gw = DeterministicCryptoGateway::new().with_transport_error("rpc down");
        let err = gw.submit_transfer(&transfer("k1", 100)).unwrap_err();
        assert!(matches!(err, Error::Transport(_)));
    }

    #[test]
    fn rejects_cross_chain_send() {
        let gw = DeterministicCryptoGateway::for_token(StableToken::UsdcBase);
        let mut req = transfer("k1", 100);
        req.to = CryptoAddress::new("solana", "abc");
        let err = gw.submit_transfer(&req).unwrap_err();
        assert!(matches!(err, Error::UnsupportedChain(c) if c == "solana"));
    }

    #[test]
    fn tx_hashes_unique_per_call() {
        let gw = DeterministicCryptoGateway::new();
        let d1 = gw.submit_transfer(&transfer("k1", 1)).unwrap();
        let d2 = gw.submit_transfer(&transfer("k2", 1)).unwrap();
        assert_ne!(d1.tx_hash, d2.tx_hash);
    }

    #[test]
    fn history_captures_transfers() {
        let gw = DeterministicCryptoGateway::new();
        gw.submit_transfer(&transfer("k1", 1)).unwrap();
        gw.submit_transfer(&transfer("k2", 2)).unwrap();
        assert_eq!(gw.transfer_history().len(), 2);
    }

    #[test]
    fn for_solana_advertises_solana_chain() {
        let gw = DeterministicCryptoGateway::for_token(StableToken::UsdcSolana);
        assert_eq!(gw.chain(), "solana");
        assert_eq!(gw.token().symbol, "USDC");
    }
}
