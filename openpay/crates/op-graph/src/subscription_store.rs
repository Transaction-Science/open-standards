//! [`GraphSubscriptionStore`] — Minigraf-backed
//! [`SubscriptionStore`].
//!
//! Same shape as the other graph-backed stores: one
//! `subscription` vertex per subscription, indexed properties
//! (`external_id`, `customer_ref`, `status_code`,
//! `current_period_end`), full state JSON in `state`. No edges
//! yet — subscriptions don't reference the ledger directly; the
//! charge that runs against a due subscription becomes an
//! ordinary `ledger_tx` that joins back via `customer_ref` /
//! external metadata.

use op_subscriptions::{
    Error as SubError, Result as SubResult, Subscription, SubscriptionId, SubscriptionStore,
};
use serde_json::Value as Json;

use crate::graph::{GraphHandle, vtypes};

/// Graph-backed subscription store.
pub struct GraphSubscriptionStore {
    handle: GraphHandle,
}

impl GraphSubscriptionStore {
    /// Construct on a fresh in-memory graph.
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::with_handle(GraphHandle::new_in_memory())
    }

    /// Construct on a shared handle.
    #[must_use]
    pub fn with_handle(handle: GraphHandle) -> Self {
        Self { handle }
    }

    /// Borrow the underlying handle.
    #[must_use]
    pub fn handle(&self) -> &GraphHandle {
        &self.handle
    }

    /// Diagnostic: number of subscription vertices.
    pub fn len(&self) -> usize {
        self.handle
            .vertices_of_type(vtypes::SUBSCRIPTION)
            .map_or(0, |v| v.len())
    }

    /// True iff no subscriptions stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn load(&self, id: SubscriptionId) -> SubResult<Option<Subscription>> {
        if !self.handle.vertex_exists(id.as_uuid()).map_err(g2s)? {
            return Ok(None);
        }
        let props = self
            .handle
            .get_vertex_properties(id.as_uuid())
            .map_err(g2s)?;
        let state = props
            .get("state")
            .ok_or_else(|| SubError::Invalid("subscription vertex missing `state`".into()))?;
        let s: Subscription = serde_json::from_value(state.clone())
            .map_err(|e| SubError::Invalid(format!("subscription decode: {e}")))?;
        Ok(Some(s))
    }

    fn persist(&self, s: &Subscription) -> SubResult<()> {
        let state =
            serde_json::to_value(s).map_err(|e| SubError::Invalid(format!("encode: {e}")))?;
        let id = s.id.as_uuid();
        self.handle
            .set_vertex_property(id, "state", state)
            .map_err(g2s)?;
        self.handle
            .set_vertex_property(id, "status_code", Json::String(s.status.code().to_owned()))
            .map_err(g2s)?;
        self.handle
            .set_vertex_property(id, "customer_ref", Json::String(s.customer_ref.clone()))
            .map_err(g2s)?;
        self.handle
            .set_vertex_property(
                id,
                "current_period_end",
                Json::Number(s.current_period_end_unix_secs.into()),
            )
            .map_err(g2s)?;
        if let Some(ext) = &s.external_id {
            self.handle
                .set_vertex_property(id, "external_id", Json::String(ext.clone()))
                .map_err(g2s)?;
        }
        Ok(())
    }

    fn lookup_by_external_id(&self, external_id: &str) -> SubResult<Option<SubscriptionId>> {
        let vertices = self
            .handle
            .vertices_of_type(vtypes::SUBSCRIPTION)
            .map_err(g2s)?;
        for v in vertices {
            let props = self.handle.get_vertex_properties(v.id).map_err(g2s)?;
            if let Some(Json::String(ext)) = props.get("external_id")
                && ext == external_id
            {
                return Ok(Some(SubscriptionId::from_uuid(v.id)));
            }
        }
        Ok(None)
    }
}

fn g2s(e: crate::Error) -> SubError {
    SubError::Invalid(format!("graph backend: {e}"))
}

fn bodies_equivalent(a: &Subscription, b: &Subscription) -> bool {
    a.customer_ref == b.customer_ref
        && a.plan.id == b.plan.id
        && a.plan.amount == b.plan.amount
        && a.plan.interval == b.plan.interval
        && a.plan.interval_count == b.plan.interval_count
        && a.external_id == b.external_id
}

impl SubscriptionStore for GraphSubscriptionStore {
    fn create_subscription(&self, s: Subscription) -> SubResult<SubscriptionId> {
        if let Some(ext) = &s.external_id
            && let Some(existing_id) = self.lookup_by_external_id(ext)?
        {
            let existing = self
                .load(existing_id)?
                .ok_or_else(|| SubError::Invalid("indexed subscription vanished".into()))?;
            if bodies_equivalent(&existing, &s) {
                return Ok(existing_id);
            }
            return Err(SubError::IdempotencyMismatch(ext.clone()));
        }
        let id = s.id;
        self.handle
            .create_vertex(vtypes::SUBSCRIPTION, id.as_uuid())
            .map_err(g2s)?;
        self.persist(&s)?;
        Ok(id)
    }

    fn get_subscription(&self, id: SubscriptionId) -> SubResult<Subscription> {
        self.load(id)?
            .ok_or_else(|| SubError::NotFound(id.to_string()))
    }

    fn find_by_external_id(&self, external_id: &str) -> SubResult<Option<Subscription>> {
        let Some(id) = self.lookup_by_external_id(external_id)? else {
            return Ok(None);
        };
        self.load(id)
    }

    fn list_for_customer(&self, customer_ref: &str) -> SubResult<Vec<Subscription>> {
        let vertices = self
            .handle
            .vertices_of_type(vtypes::SUBSCRIPTION)
            .map_err(g2s)?;
        let mut out = Vec::new();
        for v in vertices {
            let props = self.handle.get_vertex_properties(v.id).map_err(g2s)?;
            if matches!(props.get("customer_ref"), Some(Json::String(c)) if c == customer_ref)
                && let Some(state) = props.get("state")
            {
                let s: Subscription = serde_json::from_value(state.clone())
                    .map_err(|e| SubError::Invalid(format!("decode: {e}")))?;
                out.push(s);
            }
        }
        Ok(out)
    }

    fn list_due_at(&self, as_of_unix_secs: u64) -> SubResult<Vec<Subscription>> {
        let vertices = self
            .handle
            .vertices_of_type(vtypes::SUBSCRIPTION)
            .map_err(g2s)?;
        let mut out = Vec::new();
        for v in vertices {
            let props = self.handle.get_vertex_properties(v.id).map_err(g2s)?;
            // Skip terminal.
            if matches!(props.get("status_code"), Some(Json::String(s)) if s == "canceled") {
                continue;
            }
            let end = match props.get("current_period_end") {
                Some(Json::Number(n)) => n.as_u64().unwrap_or(u64::MAX),
                _ => continue,
            };
            if end <= as_of_unix_secs
                && let Some(state) = props.get("state")
            {
                let s: Subscription = serde_json::from_value(state.clone())
                    .map_err(|e| SubError::Invalid(format!("decode: {e}")))?;
                out.push(s);
            }
        }
        Ok(out)
    }

    fn update<F>(&self, id: SubscriptionId, f: F) -> SubResult<Subscription>
    where
        F: FnOnce(&mut Subscription) -> SubResult<()>,
    {
        let mut staged = self.get_subscription(id)?;
        f(&mut staged)?;
        self.persist(&staged)?;
        Ok(staged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money, PaymentMethod, VaultRef};
    use op_subscriptions::plan::{Interval, Plan};

    fn sub(customer: &str, ext: Option<&str>) -> Subscription {
        let plan = Plan::new(
            "p",
            Money::from_minor(1000, Currency::USD),
            Interval::Month,
            1,
        )
        .unwrap();
        let mut s = Subscription::new(
            customer,
            plan,
            PaymentMethod::Vault(VaultRef::new("tok")),
            1_700_000_000,
        )
        .unwrap();
        if let Some(e) = ext {
            s = s.with_external_id(e);
        }
        s
    }

    #[test]
    fn round_trip() {
        let store = GraphSubscriptionStore::new_in_memory();
        let s = sub("c-1", None);
        let id = s.id;
        store.create_subscription(s.clone()).unwrap();
        let got = store.get_subscription(id).unwrap();
        assert_eq!(got.id, id);
        assert_eq!(got.customer_ref, "c-1");
    }

    #[test]
    fn idempotency_same_body() {
        let store = GraphSubscriptionStore::new_in_memory();
        let s = sub("c-1", Some("ext"));
        let id1 = store.create_subscription(s.clone()).unwrap();
        let id2 = store.create_subscription(s).unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn list_for_customer() {
        let store = GraphSubscriptionStore::new_in_memory();
        store.create_subscription(sub("c-1", Some("a"))).unwrap();
        store.create_subscription(sub("c-2", Some("b"))).unwrap();
        assert_eq!(store.list_for_customer("c-1").unwrap().len(), 1);
        assert_eq!(store.list_for_customer("c-2").unwrap().len(), 1);
    }

    #[test]
    fn list_due_at_filters_by_period_end() {
        let store = GraphSubscriptionStore::new_in_memory();
        let s = sub("c-1", None);
        let end = s.current_period_end_unix_secs;
        store.create_subscription(s).unwrap();
        assert!(store.list_due_at(end - 1).unwrap().is_empty());
        assert_eq!(store.list_due_at(end + 1).unwrap().len(), 1);
    }
}
