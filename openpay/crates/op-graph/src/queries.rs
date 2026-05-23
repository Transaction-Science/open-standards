//! Opinionated read-side graph queries.
//!
//! These are the four traversals that appear so often in payment-
//! adjacent code that we lift them out of the trait surface and
//! into typed functions. They all take a `GraphHandle` so they
//! work against either a `GraphLedgerStore` or a `GraphWebhookStore`
//! (or both — the schema is one graph).

use op_ledger::{AccountId, Direction, TransactionId};
use op_webhook::{DeliveryAttemptId, WebhookEventId};

use crate::error::{Error, Result};
use crate::graph::{GraphHandle, etypes, vtypes};

/// One side of a transaction's entry list, materialized for
/// inspection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccountTouch {
    /// The account.
    pub account_id: AccountId,
    /// Debit or credit.
    pub direction: Direction,
    /// Amount in minor units. We return this raw rather than
    /// constructing a `Money` so the function doesn't have to
    /// re-resolve currency — operators read the currency from
    /// `currency_code` separately if they need it.
    pub amount_minor: i64,
    /// ISO 4217 currency code on the entry.
    pub currency_code: String,
}

/// List every account touched by the given transaction, with
/// direction and amount. One hop in the graph (out-edges of the tx
/// vertex).
///
/// Returns an empty `Vec` if the tx has no entries (which should
/// never happen for a well-formed posted transaction but the graph
/// itself doesn't enforce that invariant — we surface it as empty
/// rather than as an error so callers can detect it explicitly).
pub fn accounts_touched_by_transaction(
    handle: &GraphHandle,
    tx_id: TransactionId,
) -> Result<Vec<AccountTouch>> {
    if !handle.vertex_exists(tx_id.as_uuid())? {
        return Err(Error::VertexNotFound {
            vertex_type: vtypes::LEDGER_TX.into(),
            id: tx_id.to_string(),
        });
    }
    let mut out: Vec<AccountTouch> = Vec::new();
    for (etype, dir) in [
        (etypes::LEDGER_DEBIT, Direction::Debit),
        (etypes::LEDGER_CREDIT, Direction::Credit),
    ] {
        let edges = handle.out_edges(tx_id.as_uuid(), etype)?;
        for edge in edges {
            let props = handle.get_edge_properties(&edge)?;
            let amount_minor = props
                .get("amount_minor")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| Error::Invariant("entry edge missing amount_minor".to_string()))?;
            let currency_code = props
                .get("currency_code")
                .and_then(|v| v.as_str())
                .map(|s| s.to_owned())
                .ok_or_else(|| Error::Invariant("entry edge missing currency_code".to_string()))?;
            out.push(AccountTouch {
                account_id: AccountId::from_uuid(edge.to),
                direction: dir,
                amount_minor,
                currency_code,
            });
        }
    }
    Ok(out)
}

/// List every transaction that has at least one entry against the
/// given account. The returned order is unspecified (graph
/// in-edges are not time-ordered).
pub fn transactions_touching_account(
    handle: &GraphHandle,
    account_id: AccountId,
) -> Result<Vec<TransactionId>> {
    if !handle.vertex_exists(account_id.as_uuid())? {
        return Err(Error::VertexNotFound {
            vertex_type: vtypes::LEDGER_ACCOUNT.into(),
            id: account_id.to_string(),
        });
    }
    let mut tx_set: std::collections::HashSet<TransactionId> = std::collections::HashSet::new();
    for etype in [etypes::LEDGER_DEBIT, etypes::LEDGER_CREDIT] {
        let edges = handle.in_edges(account_id.as_uuid(), etype)?;
        for edge in edges {
            tx_set.insert(TransactionId::from_uuid(edge.from));
        }
    }
    Ok(tx_set.into_iter().collect())
}

/// Walk the `ledger_reverses` chain starting at `tx_id`. Returns
/// the linear chain ordered from the **original** to **newest**.
///
/// The chain is computed by repeatedly following the
/// `ledger_reverses` *inbound* edges (newer transactions point at
/// older ones) starting from `tx_id`, and also walking *outbound*
/// edges (older transactions point at the newer reversal of them).
///
/// In normal use a transaction is reversed at most once, so the
/// chain length grows by one per correction. We guard against
/// cycles with a visited set.
pub fn reversal_chain(handle: &GraphHandle, tx_id: TransactionId) -> Result<Vec<TransactionId>> {
    if !handle.vertex_exists(tx_id.as_uuid())? {
        return Err(Error::VertexNotFound {
            vertex_type: vtypes::LEDGER_TX.into(),
            id: tx_id.to_string(),
        });
    }
    use std::collections::HashSet;
    let mut visited: HashSet<TransactionId> = HashSet::new();
    // Walk backwards: this tx → older tx it reverses → ...
    let mut backwards: Vec<TransactionId> = Vec::new();
    let mut cursor = tx_id;
    loop {
        if !visited.insert(cursor) {
            break; // cycle
        }
        backwards.push(cursor);
        let edges = handle.out_edges(cursor.as_uuid(), etypes::LEDGER_REVERSES)?;
        match edges.into_iter().next() {
            Some(e) => cursor = TransactionId::from_uuid(e.to),
            None => break,
        }
    }
    backwards.reverse(); // now original → ... → tx_id
    // Walk forwards from tx_id: newer txs that point back at this
    // one (in-edges of ledger_reverses).
    let mut forwards_cursor = tx_id;
    loop {
        let edges = handle.in_edges(forwards_cursor.as_uuid(), etypes::LEDGER_REVERSES)?;
        let next = edges.into_iter().next();
        match next {
            Some(e) => {
                let nid = TransactionId::from_uuid(e.from);
                if !visited.insert(nid) {
                    break;
                }
                backwards.push(nid);
                forwards_cursor = nid;
            }
            None => break,
        }
    }
    Ok(backwards)
}

/// List every delivery attempt for the given event across every
/// endpoint. One hop in the graph (out-edges of the event vertex).
pub fn attempts_for_event(
    handle: &GraphHandle,
    event_id: WebhookEventId,
) -> Result<Vec<DeliveryAttemptId>> {
    if !handle.vertex_exists(event_id.as_uuid())? {
        return Err(Error::VertexNotFound {
            vertex_type: vtypes::WEBHOOK_EVENT.into(),
            id: event_id.to_string(),
        });
    }
    let edges = handle.out_edges(event_id.as_uuid(), etypes::WEBHOOK_DELIVERS)?;
    Ok(edges
        .into_iter()
        .map(|e| DeliveryAttemptId::from_uuid(e.to))
        .collect())
}

// ============================================================
// Fraud-graph queries
// ============================================================

/// Account-account pairs linked by a reversal chain. Two accounts
/// are "linked via chargeback" if there's a transaction `tx_a`
/// debiting / crediting one of them that has been reversed by a
/// later transaction `tx_b` debiting / crediting the other.
///
/// Implementation: walks every `ledger_reverses` edge; for each,
/// collects the set of accounts touched by the original tx and
/// the reversal tx, and emits all cross pairs.
///
/// **Cost:** O(reversals × max_entries). Fine for the reference
/// impl; production deployments with many reversals page their
/// own joins through a secondary index.
pub fn accounts_linked_via_chargeback(handle: &GraphHandle) -> Result<Vec<(AccountId, AccountId)>> {
    use std::collections::HashSet;
    let txs = handle.vertices_of_type(vtypes::LEDGER_TX)?;
    let mut pairs: Vec<(AccountId, AccountId)> = Vec::new();
    let mut seen: HashSet<(AccountId, AccountId)> = HashSet::new();
    for tx in txs {
        let rev_edges = handle.out_edges(tx.id, etypes::LEDGER_REVERSES)?;
        for edge in rev_edges {
            let from_accts =
                accounts_touched_by_transaction(handle, TransactionId::from_uuid(edge.from))?;
            let to_accts =
                accounts_touched_by_transaction(handle, TransactionId::from_uuid(edge.to))?;
            for a in &from_accts {
                for b in &to_accts {
                    if a.account_id == b.account_id {
                        continue;
                    }
                    // Canonicalize the pair so (X, Y) and (Y, X)
                    // collapse to one entry — the relationship is
                    // symmetric for "linked via."
                    let (lo, hi) = if a.account_id.as_uuid() < b.account_id.as_uuid() {
                        (a.account_id, b.account_id)
                    } else {
                        (b.account_id, a.account_id)
                    };
                    if seen.insert((lo, hi)) {
                        pairs.push((lo, hi));
                    }
                }
            }
        }
    }
    Ok(pairs)
}

/// Endpoints whose decoded webhook signing secrets share a prefix
/// of at least `prefix_bytes` bytes. A weak but useful proxy for
/// "two endpoints belonging to the same operator setup," which in
/// fraud-investigation contexts is often the signal you want to
/// surface for review (e.g. several merchant accounts all sharing
/// a webhook secret prefix because they're behind the same
/// orchestrator).
///
/// Returns canonicalized pairs `(lo, hi)` — each pair appears once.
pub fn endpoints_sharing_secret_prefix(
    handle: &GraphHandle,
    prefix_bytes: usize,
) -> Result<Vec<(op_webhook::EndpointId, op_webhook::EndpointId)>> {
    use op_webhook::EndpointId;
    let endpoints = handle.vertices_of_type(vtypes::WEBHOOK_ENDPOINT)?;
    let mut secrets: Vec<(EndpointId, Vec<u8>)> = Vec::with_capacity(endpoints.len());
    for v in endpoints {
        let props = handle.get_vertex_properties(v.id)?;
        let Some(serde_json::Value::String(b64)) = props.get("secret_b64") else {
            continue;
        };
        // We reuse the simple base64 codec convention. We don't
        // depend on op-webhook's private helper, so we tolerate
        // either the standard alphabet or a misencoding by failing
        // soft (skip the endpoint).
        let Ok(decoded) = simple_base64_decode(b64) else {
            continue;
        };
        secrets.push((EndpointId(v.id), decoded));
    }
    let mut pairs: Vec<(EndpointId, EndpointId)> = Vec::new();
    for i in 0..secrets.len() {
        for j in (i + 1)..secrets.len() {
            let a = &secrets[i].1;
            let b = &secrets[j].1;
            if a.len() < prefix_bytes || b.len() < prefix_bytes {
                continue;
            }
            if a[..prefix_bytes] == b[..prefix_bytes] {
                let (id_a, id_b) = (secrets[i].0, secrets[j].0);
                let (lo, hi) = if id_a.0 < id_b.0 {
                    (id_a, id_b)
                } else {
                    (id_b, id_a)
                };
                pairs.push((lo, hi));
            }
        }
    }
    Ok(pairs)
}

/// Delivery attempts whose endpoints resolve to the same IP. The
/// graph itself doesn't record IPs — that's an operator
/// observability concern outside `op-graph`'s schema — so this
/// query takes the resolution as input: a closure `endpoint_id →
/// Option<IpAddr-equivalent String>`.
///
/// Returns canonicalized attempt pairs. A single attempt against
/// the same IP doesn't appear; pairs require two distinct attempts.
pub fn attempts_with_shared_ip<F>(
    handle: &GraphHandle,
    ip_of: F,
) -> Result<Vec<(DeliveryAttemptId, DeliveryAttemptId)>>
where
    F: Fn(op_webhook::EndpointId) -> Option<String>,
{
    use op_webhook::EndpointId;
    use std::collections::HashMap;
    let attempts = handle.vertices_of_type(vtypes::WEBHOOK_ATTEMPT)?;
    // For each attempt, find its endpoint via the `webhook_to`
    // outbound edge, then look up the IP.
    let mut buckets: HashMap<String, Vec<DeliveryAttemptId>> = HashMap::new();
    for v in attempts {
        let aid = DeliveryAttemptId::from_uuid(v.id);
        let to_edges = handle.out_edges(v.id, etypes::WEBHOOK_TO)?;
        for e in to_edges {
            let endpoint = EndpointId(e.to);
            if let Some(ip) = ip_of(endpoint) {
                buckets.entry(ip).or_default().push(aid);
            }
        }
    }
    let mut pairs: Vec<(DeliveryAttemptId, DeliveryAttemptId)> = Vec::new();
    for ids in buckets.values() {
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let (a, b) = (ids[i], ids[j]);
                let (lo, hi) = if a.0 < b.0 { (a, b) } else { (b, a) };
                pairs.push((lo, hi));
            }
        }
    }
    Ok(pairs)
}

/// Minimal base64 decoder mirroring the encoder
/// `op-webhook` uses internally. Standard alphabet (+/), no
/// line breaks, padding required.
fn simple_base64_decode(s: &str) -> std::result::Result<Vec<u8>, ()> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = s.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err(());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let lookup = |c: u8| -> std::result::Result<u8, ()> {
        TABLE
            .iter()
            .position(|&t| t == c)
            .map(|p| p as u8)
            .ok_or(())
    };
    for chunk in bytes.chunks(4) {
        let pad = chunk.iter().filter(|&&b| b == b'=').count();
        let c0 = lookup(chunk[0])?;
        let c1 = lookup(chunk[1])?;
        let c2 = if chunk[2] == b'=' {
            0
        } else {
            lookup(chunk[2])?
        };
        let c3 = if chunk[3] == b'=' {
            0
        } else {
            lookup(chunk[3])?
        };
        out.push((c0 << 2) | (c1 >> 4));
        if pad < 2 {
            out.push(((c1 & 0x0F) << 4) | (c2 >> 2));
        }
        if pad == 0 {
            out.push(((c2 & 0x03) << 6) | c3);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money};
    use op_ledger::{Account, AccountClass, Entry, Ledger, LedgerStore, Transaction};
    use op_webhook::{DeliveryAttempt, Endpoint, WebhookEvent, WebhookStore};

    use crate::ledger_store::GraphLedgerStore;
    use crate::webhook_store::GraphWebhookStore;

    fn ledger_setup() -> (
        GraphLedgerStore,
        op_ledger::LedgerId,
        op_ledger::AccountId,
        op_ledger::AccountId,
    ) {
        let store = GraphLedgerStore::new_in_memory();
        let l = Ledger::new("L").unwrap();
        let lid = l.id;
        store.create_ledger(l).unwrap();
        let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);
        let rev = Account::new(lid, "rev", AccountClass::Revenue, Currency::USD);
        let cid = cash.id;
        let rid = rev.id;
        store.create_account(cash).unwrap();
        store.create_account(rev).unwrap();
        (store, lid, cid, rid)
    }

    #[test]
    fn accounts_touched_lists_both_sides() {
        let (store, lid, cash, rev) = ledger_setup();
        let t = Transaction::new_pending(
            lid,
            0,
            vec![
                Entry::debit(cash, Money::from_minor(123, Currency::USD)),
                Entry::credit(rev, Money::from_minor(123, Currency::USD)),
            ],
        )
        .unwrap();
        let tid = store.post_transaction(t).unwrap();
        let touches = accounts_touched_by_transaction(store.handle(), tid).unwrap();
        assert_eq!(touches.len(), 2);
        // One debit and one credit.
        let debits: Vec<_> = touches
            .iter()
            .filter(|t| t.direction == Direction::Debit)
            .collect();
        let credits: Vec<_> = touches
            .iter()
            .filter(|t| t.direction == Direction::Credit)
            .collect();
        assert_eq!(debits.len(), 1);
        assert_eq!(credits.len(), 1);
        assert_eq!(debits[0].account_id, cash);
        assert_eq!(credits[0].account_id, rev);
        assert_eq!(debits[0].amount_minor, 123);
        assert_eq!(debits[0].currency_code, "USD");
    }

    #[test]
    fn accounts_touched_unknown_tx_errors() {
        let store = GraphLedgerStore::new_in_memory();
        let r = accounts_touched_by_transaction(store.handle(), TransactionId::new());
        assert!(matches!(r, Err(Error::VertexNotFound { .. })));
    }

    #[test]
    fn transactions_touching_account_lists_unique_txs() {
        let (store, lid, cash, rev) = ledger_setup();
        // Post two transactions, both touching cash.
        for amt in [100i64, 200] {
            let t = Transaction::new_pending(
                lid,
                0,
                vec![
                    Entry::debit(cash, Money::from_minor(amt, Currency::USD)),
                    Entry::credit(rev, Money::from_minor(amt, Currency::USD)),
                ],
            )
            .unwrap();
            store.post_transaction(t).unwrap();
        }
        let txs = transactions_touching_account(store.handle(), cash).unwrap();
        assert_eq!(txs.len(), 2);
    }

    #[test]
    fn transactions_touching_unknown_account_errors() {
        let store = GraphLedgerStore::new_in_memory();
        let r = transactions_touching_account(store.handle(), AccountId::new());
        assert!(matches!(r, Err(Error::VertexNotFound { .. })));
    }

    #[test]
    fn reversal_chain_single_node_is_self() {
        let (store, lid, cash, rev) = ledger_setup();
        let t = Transaction::new_pending(
            lid,
            0,
            vec![
                Entry::debit(cash, Money::from_minor(10, Currency::USD)),
                Entry::credit(rev, Money::from_minor(10, Currency::USD)),
            ],
        )
        .unwrap();
        let tid = store.post_transaction(t).unwrap();
        let chain = reversal_chain(store.handle(), tid).unwrap();
        assert_eq!(chain, vec![tid]);
    }

    #[test]
    fn reversal_chain_links_via_graph_edge() {
        let (store, lid, cash, rev) = ledger_setup();
        let t1 = Transaction::new_pending(
            lid,
            0,
            vec![
                Entry::debit(cash, Money::from_minor(50, Currency::USD)),
                Entry::credit(rev, Money::from_minor(50, Currency::USD)),
            ],
        )
        .unwrap();
        let t1_id = store.post_transaction(t1).unwrap();
        let t2 = Transaction::new_pending(
            lid,
            0,
            vec![
                // Reversal: credit cash, debit rev.
                Entry::credit(cash, Money::from_minor(50, Currency::USD)),
                Entry::debit(rev, Money::from_minor(50, Currency::USD)),
            ],
        )
        .unwrap();
        let t2_id = store.post_transaction(t2).unwrap();
        // Wire the ledger_reverses edge manually (t2 reverses t1).
        store
            .handle()
            .create_edge(t2_id.as_uuid(), etypes::LEDGER_REVERSES, t1_id.as_uuid())
            .unwrap();
        // Walk from either end; both should yield [t1, t2].
        let chain = reversal_chain(store.handle(), t1_id).unwrap();
        assert_eq!(chain, vec![t1_id, t2_id]);
        let chain = reversal_chain(store.handle(), t2_id).unwrap();
        assert_eq!(chain, vec![t1_id, t2_id]);
    }

    #[test]
    fn reversal_chain_unknown_tx_errors() {
        let store = GraphLedgerStore::new_in_memory();
        let r = reversal_chain(store.handle(), TransactionId::new());
        assert!(matches!(r, Err(Error::VertexNotFound { .. })));
    }

    #[test]
    fn attempts_for_event_lists_all() {
        let store = GraphWebhookStore::new_in_memory();
        let endpoint =
            Endpoint::new("https://x.example/h", b"s".to_vec(), vec!["*".to_string()]).unwrap();
        let eid = endpoint.id;
        store.put_endpoint(endpoint).unwrap();
        let event = WebhookEvent::new("any", b"".to_vec(), 0);
        let evid = event.id;
        store.put_event(event).unwrap();
        let mut ids = Vec::new();
        for n in 0..3 {
            let a = DeliveryAttempt::new_pending(evid, eid, n, 0);
            ids.push(a.id);
            store.put_attempt(a).unwrap();
        }
        let list = attempts_for_event(store.handle(), evid).unwrap();
        assert_eq!(list.len(), 3);
        for id in &ids {
            assert!(list.contains(id));
        }
    }

    #[test]
    fn attempts_for_unknown_event_errors() {
        let store = GraphWebhookStore::new_in_memory();
        let r = attempts_for_event(store.handle(), WebhookEventId::new());
        assert!(matches!(r, Err(Error::VertexNotFound { .. })));
    }

    #[test]
    fn empty_account_has_no_transactions() {
        let (store, _lid, cash, _rev) = ledger_setup();
        let txs = transactions_touching_account(store.handle(), cash).unwrap();
        assert!(txs.is_empty());
    }

    #[test]
    fn empty_event_has_no_attempts() {
        let store = GraphWebhookStore::new_in_memory();
        let event = WebhookEvent::new("orphan", b"".to_vec(), 0);
        let evid = event.id;
        store.put_event(event).unwrap();
        let attempts = attempts_for_event(store.handle(), evid).unwrap();
        assert!(attempts.is_empty());
    }
}
