//! [`WebhookEventSource`] — reconcile against settlement webhooks.

use op_webhook::WebhookEvent;

use crate::error::{Error, Result};
use crate::source::ReconciliationSource;
use crate::sources::currency_from_code;
use crate::statement::{LineDirection, StatementLine};

/// The `event_type` this source treats as reconciliation input.
/// Webhooks of any other type are silently ignored — an operator's
/// stream is full of unrelated events.
pub const SETTLEMENT_EVENT_TYPE: &str = "psp.settlement.confirmed";

/// The reference JSON shape a `psp.settlement.confirmed` payload must
/// have. This is **policy, not protocol**: it's the schema `OpenPay`
/// ships as the reference. An operator whose PSP emits a different
/// shape implements their own [`ReconciliationSource`] — the engine
/// is unaffected. Documented here so that contract is explicit.
#[derive(serde::Deserialize)]
struct SettlementPayload {
    /// PSP-assigned settlement id (becomes `StatementLine::source_id`).
    source_id: String,
    /// The order / end-to-end reference, if the PSP echoes one.
    #[serde(default)]
    external_id: Option<String>,
    /// Amount in minor units (already integer — no float games).
    amount_minor: i64,
    /// ISO 4217 alphabetic code.
    currency: String,
    /// `"credit"` (funds in) or `"debit"` (funds out / fee /
    /// chargeback). Anything else is a malformed payload.
    direction: String,
    /// When the PSP says it settled. Falls back to the webhook's own
    /// `created_at_unix_secs` when the payload omits it.
    #[serde(default)]
    posted_at_unix_secs: Option<u64>,
}

/// Reconciliation source over a borrowed slice of webhook events.
///
/// Holds a reference, not an owned copy — operators typically already
/// have the events in hand from `op-webhook`'s store.
pub struct WebhookEventSource<'a> {
    events: &'a [WebhookEvent],
}

impl<'a> WebhookEventSource<'a> {
    /// Wrap a slice of webhook events. Only those whose `event_type`
    /// equals [`SETTLEMENT_EVENT_TYPE`] become statement lines.
    #[must_use]
    pub fn new(events: &'a [WebhookEvent]) -> Self {
        Self { events }
    }
}

impl ReconciliationSource for WebhookEventSource<'_> {
    fn iter_lines(&self) -> Box<dyn Iterator<Item = Result<StatementLine>> + '_> {
        Box::new(
            self.events
                .iter()
                .filter(|e| e.event_type == SETTLEMENT_EVENT_TYPE)
                .map(|e| {
                    let p: SettlementPayload =
                        serde_json::from_slice(&e.payload).map_err(|err| {
                            Error::UnrecognizedWebhook(format!("event {}: {err}", e.id))
                        })?;

                    let currency = currency_from_code(&p.currency)?;
                    let amount = op_core::Money {
                        minor_units: p.amount_minor.abs(),
                        currency,
                    };
                    let dir = match p.direction.as_str() {
                        "credit" => LineDirection::Credit,
                        "debit" => LineDirection::Debit,
                        other => {
                            return Err(Error::UnrecognizedWebhook(format!(
                                "event {}: direction {other:?} not credit/debit",
                                e.id
                            )));
                        }
                    };
                    let posted = p.posted_at_unix_secs.unwrap_or(e.created_at_unix_secs);

                    let mut line = StatementLine::new(p.source_id, amount, dir, posted);
                    if let Some(eid) = p.external_id {
                        line = line.with_external_id(eid);
                    }
                    Ok(line)
                }),
        )
    }
}
