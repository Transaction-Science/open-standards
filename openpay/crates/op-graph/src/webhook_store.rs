//! [`GraphWebhookStore`]: a graph-backed implementation of
//! [`op_webhook::WebhookStore`].
//!
//! ## Storage layout
//!
//! - Each [`Endpoint`] is a `webhook_endpoint` vertex with
//!   properties `url`, `secret_b64` (base64-encoded bytes),
//!   `event_filters_csv` (comma-separated; literal filters only —
//!   we don't permit commas in filter strings), `status`,
//!   `consecutive_failures`, `metadata` (JSON object).
//! - Each [`WebhookEvent`] is a `webhook_event` vertex with
//!   `event_type`, `payload_b64`, `created_at_unix_secs`.
//! - Each [`DeliveryAttempt`] is a `webhook_attempt` vertex with
//!   `attempt_number`, `status`, `http_status` (optional),
//!   `response_body_excerpt` (optional), `started_at_unix_secs`,
//!   `completed_at_unix_secs` (optional),
//!   `next_attempt_at_unix_secs` (optional), `error` (optional);
//!   plus a `webhook_delivers` *inbound* edge from its event and a
//!   `webhook_to` *outbound* edge to its endpoint.
//!
//! ## Why base64 for binary fields?
//!
//! IndraDB properties are `serde_json::Value`s. JSON can't carry
//! raw bytes; we must encode. Base64 is the convention; the
//! payload size cost (~33%) is acceptable for a reference impl. A
//! production deployment with a custom datastore can swap the
//! codec out by re-implementing the trait against the same graph.

use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::Value as Json;

use op_webhook::{
    DeliveryAttempt, DeliveryAttemptId, DeliveryStatus, Endpoint, EndpointId, EndpointStatus,
    Error as WebhookError, Result as WebhookResult, WebhookEvent, WebhookEventId, WebhookStore,
};

use crate::error::{Error, Result};
use crate::graph::{GraphHandle, etypes, vtypes};

// ============================================================
// GraphWebhookStore
// ============================================================

/// Graph-backed implementation of [`op_webhook::WebhookStore`].
///
/// Clone is cheap (`GraphHandle` is `Arc`-backed).
#[derive(Clone)]
pub struct GraphWebhookStore {
    handle: GraphHandle,
    /// Per-event-type endpoint id index. Lazily seeded; not
    /// authoritative.
    endpoint_index: std::sync::Arc<Mutex<HashMap<String, Vec<EndpointId>>>>,
}

impl GraphWebhookStore {
    /// Build a fresh store with its own in-memory graph.
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::with_handle(GraphHandle::new_in_memory())
    }

    /// Build over an existing handle (so multiple stores share one
    /// graph).
    #[must_use]
    pub fn with_handle(handle: GraphHandle) -> Self {
        Self {
            handle,
            endpoint_index: std::sync::Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Access the underlying graph handle.
    #[must_use]
    pub fn handle(&self) -> &GraphHandle {
        &self.handle
    }

    // --------------------------------------------------------
    // Endpoints
    // --------------------------------------------------------

    fn write_endpoint_props(&self, ep: &Endpoint) -> Result<()> {
        let uid = ep.id.0;
        self.handle
            .set_vertex_property(uid, "url", Json::String(ep.url.clone()))?;
        self.handle.set_vertex_property(
            uid,
            "secret_b64",
            Json::String(b64::encode(&ep.secret)),
        )?;
        let filters_csv = encode_filters(&ep.event_filters);
        self.handle
            .set_vertex_property(uid, "event_filters_csv", Json::String(filters_csv))?;
        self.handle.set_vertex_property(
            uid,
            "status",
            Json::String(endpoint_status_str(ep.status).to_owned()),
        )?;
        self.handle.set_vertex_property(
            uid,
            "consecutive_failures",
            Json::Number(serde_json::Number::from(ep.consecutive_failures)),
        )?;
        let mut md = serde_json::Map::new();
        for (k, v) in &ep.metadata {
            md.insert(k.clone(), Json::String(v.clone()));
        }
        self.handle
            .set_vertex_property(uid, "metadata", Json::Object(md))?;
        Ok(())
    }

    fn read_endpoint(&self, id: EndpointId) -> Result<Endpoint> {
        let _ = self
            .handle
            .get_typed_vertex(id.0, vtypes::WEBHOOK_ENDPOINT)?;
        let props = self.handle.get_vertex_properties(id.0)?;
        let url = json_string(&props, "url")?;
        let secret_b64 = json_string(&props, "secret_b64")?;
        let secret = b64::decode(&secret_b64)
            .map_err(|()| Error::Invariant(format!("bad base64 secret on endpoint {id}")))?;
        let filters_csv = json_string(&props, "event_filters_csv")?;
        let event_filters = decode_filters(&filters_csv);
        let status = parse_endpoint_status(&json_string(&props, "status")?)?;
        let consecutive_failures = json_u64(&props, "consecutive_failures")? as u32;
        let metadata = json_opt_object_strs(&props, "metadata");
        Ok(Endpoint {
            id,
            url,
            secret,
            event_filters,
            status,
            consecutive_failures,
            metadata,
        })
    }

    // --------------------------------------------------------
    // Events
    // --------------------------------------------------------

    fn write_event_props(&self, event: &WebhookEvent) -> Result<()> {
        let uid = event.id.as_uuid();
        self.handle.set_vertex_property(
            uid,
            "event_type",
            Json::String(event.event_type.clone()),
        )?;
        self.handle.set_vertex_property(
            uid,
            "payload_b64",
            Json::String(b64::encode(&event.payload)),
        )?;
        self.handle.set_vertex_property(
            uid,
            "created_at_unix_secs",
            Json::Number(serde_json::Number::from(event.created_at_unix_secs)),
        )?;
        Ok(())
    }

    fn read_event(&self, id: WebhookEventId) -> Result<WebhookEvent> {
        let _ = self
            .handle
            .get_typed_vertex(id.as_uuid(), vtypes::WEBHOOK_EVENT)?;
        let props = self.handle.get_vertex_properties(id.as_uuid())?;
        let event_type = json_string(&props, "event_type")?;
        let payload_b64 = json_string(&props, "payload_b64")?;
        let payload = b64::decode(&payload_b64)
            .map_err(|()| Error::Invariant(format!("bad base64 payload on event {id}")))?;
        let created_at_unix_secs = json_u64(&props, "created_at_unix_secs")?;
        Ok(WebhookEvent {
            id,
            event_type,
            payload,
            created_at_unix_secs,
        })
    }

    // --------------------------------------------------------
    // Attempts
    // --------------------------------------------------------

    fn write_attempt_props(&self, a: &DeliveryAttempt) -> Result<()> {
        let uid = a.id.0;
        self.handle.set_vertex_property(
            uid,
            "attempt_number",
            Json::Number(serde_json::Number::from(a.attempt_number)),
        )?;
        self.handle.set_vertex_property(
            uid,
            "status",
            Json::String(delivery_status_str(a.status).to_owned()),
        )?;
        self.handle.set_vertex_property(
            uid,
            "http_status",
            match a.http_status {
                Some(s) => Json::Number(serde_json::Number::from(s)),
                None => Json::Null,
            },
        )?;
        self.handle.set_vertex_property(
            uid,
            "response_body_excerpt",
            match &a.response_body_excerpt {
                Some(s) => Json::String(s.clone()),
                None => Json::Null,
            },
        )?;
        self.handle.set_vertex_property(
            uid,
            "started_at_unix_secs",
            Json::Number(serde_json::Number::from(a.started_at_unix_secs)),
        )?;
        self.handle.set_vertex_property(
            uid,
            "completed_at_unix_secs",
            match a.completed_at_unix_secs {
                Some(t) => Json::Number(serde_json::Number::from(t)),
                None => Json::Null,
            },
        )?;
        self.handle.set_vertex_property(
            uid,
            "next_attempt_at_unix_secs",
            match a.next_attempt_at_unix_secs {
                Some(t) => Json::Number(serde_json::Number::from(t)),
                None => Json::Null,
            },
        )?;
        self.handle.set_vertex_property(
            uid,
            "error",
            match &a.error {
                Some(e) => Json::String(e.clone()),
                None => Json::Null,
            },
        )?;
        Ok(())
    }

    fn read_attempt(&self, id: DeliveryAttemptId) -> Result<DeliveryAttempt> {
        let _ = self
            .handle
            .get_typed_vertex(id.0, vtypes::WEBHOOK_ATTEMPT)?;
        let props = self.handle.get_vertex_properties(id.0)?;
        let attempt_number = json_u64(&props, "attempt_number")? as u32;
        let status = parse_delivery_status(&json_string(&props, "status")?)?;
        let http_status = props
            .get("http_status")
            .and_then(|v| v.as_u64())
            .map(|n| n as u16);
        let response_body_excerpt = json_opt_string(&props, "response_body_excerpt");
        let started_at_unix_secs = json_u64(&props, "started_at_unix_secs")?;
        let completed_at_unix_secs = props.get("completed_at_unix_secs").and_then(|v| v.as_u64());
        let next_attempt_at_unix_secs = props
            .get("next_attempt_at_unix_secs")
            .and_then(|v| v.as_u64());
        let error = json_opt_string(&props, "error");
        // Resolve event_id via the (one) inbound webhook_delivers edge.
        let inbound = self.handle.in_edges(id.0, etypes::WEBHOOK_DELIVERS)?;
        let event_uuid = inbound.into_iter().next().map(|e| e.from).ok_or_else(|| {
            Error::Invariant(format!("attempt {id} has no inbound webhook_delivers edge"))
        })?;
        // Resolve endpoint_id via the (one) outbound webhook_to edge.
        let outbound = self.handle.out_edges(id.0, etypes::WEBHOOK_TO)?;
        let endpoint_uuid = outbound.into_iter().next().map(|e| e.to).ok_or_else(|| {
            Error::Invariant(format!("attempt {id} has no outbound webhook_to edge"))
        })?;
        Ok(DeliveryAttempt {
            id,
            event_id: WebhookEventId::from_uuid(event_uuid),
            endpoint_id: EndpointId::from_uuid(endpoint_uuid),
            attempt_number,
            status,
            http_status,
            response_body_excerpt,
            started_at_unix_secs,
            completed_at_unix_secs,
            next_attempt_at_unix_secs,
            error,
        })
    }

    /// Was this attempt vertex already created? Allow `put_attempt`
    /// to overwrite-update without re-creating edges.
    fn attempt_exists(&self, id: DeliveryAttemptId) -> Result<bool> {
        self.handle.vertex_exists(id.0)
    }
}

// ============================================================
// WebhookStore impl
// ============================================================

impl WebhookStore for GraphWebhookStore {
    fn put_endpoint(&self, endpoint: Endpoint) -> WebhookResult<EndpointId> {
        let id = endpoint.id;
        // Idempotent create — if the vertex already exists, just
        // overwrite properties.
        if !self
            .handle
            .vertex_exists(id.0)
            .map_err(WebhookError::from)?
        {
            self.handle
                .create_vertex(vtypes::WEBHOOK_ENDPOINT, id.0)
                .map_err(WebhookError::from)?;
        }
        self.write_endpoint_props(&endpoint)
            .map_err(WebhookError::from)?;
        // Update the per-event-type index.
        let mut idx = self.endpoint_index.lock().expect("poisoned");
        for filter in &endpoint.event_filters {
            idx.entry(filter.clone()).or_default().push(id);
        }
        Ok(id)
    }

    fn get_endpoint(&self, id: EndpointId) -> WebhookResult<Endpoint> {
        self.read_endpoint(id).map_err(|e| match e {
            Error::VertexNotFound { .. } => WebhookError::EndpointNotFound(id.to_string()),
            other => WebhookError::from(other),
        })
    }

    fn list_active_endpoints_for(&self, event_type: &str) -> WebhookResult<Vec<Endpoint>> {
        // Build the candidate set: indexed for the exact event_type
        // and for the wildcard "*".
        let mut candidates: Vec<EndpointId> = Vec::new();
        {
            let idx = self.endpoint_index.lock().expect("poisoned");
            if let Some(v) = idx.get(event_type) {
                candidates.extend(v.iter().copied());
            }
            if event_type != "*"
                && let Some(v) = idx.get("*")
            {
                candidates.extend(v.iter().copied());
            }
        }
        // De-dupe (an endpoint could match multiple filters).
        candidates.sort_by_key(|id| id.0);
        candidates.dedup();
        let mut out: Vec<Endpoint> = Vec::with_capacity(candidates.len());
        for cid in candidates {
            // Read & filter to Active.
            let ep = self.read_endpoint(cid).map_err(WebhookError::from)?;
            if ep.status == EndpointStatus::Active && ep.matches(event_type) {
                out.push(ep);
            }
        }
        Ok(out)
    }

    fn set_endpoint_status(&self, id: EndpointId, status: EndpointStatus) -> WebhookResult<()> {
        // Ensure exists.
        let _ = self.read_endpoint(id).map_err(|e| match e {
            Error::VertexNotFound { .. } => WebhookError::EndpointNotFound(id.to_string()),
            other => WebhookError::from(other),
        })?;
        self.handle
            .set_vertex_property(
                id.0,
                "status",
                Json::String(endpoint_status_str(status).to_owned()),
            )
            .map_err(WebhookError::from)?;
        Ok(())
    }

    fn set_endpoint_consecutive_failures(
        &self,
        id: EndpointId,
        failures: u32,
    ) -> WebhookResult<()> {
        let _ = self.read_endpoint(id).map_err(|e| match e {
            Error::VertexNotFound { .. } => WebhookError::EndpointNotFound(id.to_string()),
            other => WebhookError::from(other),
        })?;
        self.handle
            .set_vertex_property(
                id.0,
                "consecutive_failures",
                Json::Number(serde_json::Number::from(failures)),
            )
            .map_err(WebhookError::from)?;
        Ok(())
    }

    fn put_event(&self, event: WebhookEvent) -> WebhookResult<WebhookEventId> {
        let id = event.id;
        if !self
            .handle
            .vertex_exists(id.as_uuid())
            .map_err(WebhookError::from)?
        {
            self.handle
                .create_vertex(vtypes::WEBHOOK_EVENT, id.as_uuid())
                .map_err(WebhookError::from)?;
        }
        self.write_event_props(&event).map_err(WebhookError::from)?;
        Ok(id)
    }

    fn get_event(&self, id: WebhookEventId) -> WebhookResult<WebhookEvent> {
        self.read_event(id).map_err(|e| match e {
            Error::VertexNotFound { .. } => WebhookError::EventNotFound(id.to_string()),
            other => WebhookError::from(other),
        })
    }

    fn put_attempt(&self, attempt: DeliveryAttempt) -> WebhookResult<DeliveryAttemptId> {
        let id = attempt.id;
        let exists = self.attempt_exists(id).map_err(WebhookError::from)?;
        if !exists {
            // Verify event + endpoint vertices exist BEFORE creating
            // the attempt vertex (to avoid orphan-vertex leaks on
            // misuse).
            if !self
                .handle
                .vertex_exists(attempt.event_id.as_uuid())
                .map_err(WebhookError::from)?
            {
                return Err(WebhookError::EventNotFound(attempt.event_id.to_string()));
            }
            if !self
                .handle
                .vertex_exists(attempt.endpoint_id.0)
                .map_err(WebhookError::from)?
            {
                return Err(WebhookError::EndpointNotFound(
                    attempt.endpoint_id.to_string(),
                ));
            }
            // Now create the attempt vertex + edges.
            self.handle
                .create_vertex(vtypes::WEBHOOK_ATTEMPT, id.0)
                .map_err(WebhookError::from)?;
            // event --delivers--> attempt
            self.handle
                .create_edge(attempt.event_id.as_uuid(), etypes::WEBHOOK_DELIVERS, id.0)
                .map_err(WebhookError::from)?;
            // attempt --to--> endpoint
            self.handle
                .create_edge(id.0, etypes::WEBHOOK_TO, attempt.endpoint_id.0)
                .map_err(WebhookError::from)?;
        }
        self.write_attempt_props(&attempt)
            .map_err(WebhookError::from)?;
        Ok(id)
    }

    fn get_attempt(&self, id: DeliveryAttemptId) -> WebhookResult<DeliveryAttempt> {
        self.read_attempt(id).map_err(|e| match e {
            Error::VertexNotFound { .. } => WebhookError::AttemptNotFound(id.to_string()),
            other => WebhookError::from(other),
        })
    }

    fn list_attempts(
        &self,
        event_id: WebhookEventId,
        endpoint_id: EndpointId,
    ) -> WebhookResult<Vec<DeliveryAttempt>> {
        // Out-edges of event_id (webhook_delivers): yields candidate
        // attempt vertex ids. Filter by endpoint via the read.
        let edges = self
            .handle
            .out_edges(event_id.as_uuid(), etypes::WEBHOOK_DELIVERS)
            .map_err(WebhookError::from)?;
        let mut out: Vec<DeliveryAttempt> = Vec::new();
        for e in edges {
            let aid = DeliveryAttemptId::from_uuid(e.to);
            let attempt = self.read_attempt(aid).map_err(WebhookError::from)?;
            if attempt.endpoint_id == endpoint_id {
                out.push(attempt);
            }
        }
        out.sort_by_key(|a| a.attempt_number);
        Ok(out)
    }

    fn list_due_retries(&self, now_unix_secs: u64) -> WebhookResult<Vec<DeliveryAttempt>> {
        let vertices = self
            .handle
            .vertices_of_type(vtypes::WEBHOOK_ATTEMPT)
            .map_err(WebhookError::from)?;
        let mut out: Vec<DeliveryAttempt> = Vec::new();
        for v in vertices {
            let aid = DeliveryAttemptId::from_uuid(v.id);
            let attempt = self.read_attempt(aid).map_err(WebhookError::from)?;
            if attempt.status == DeliveryStatus::RetryScheduled
                && attempt
                    .next_attempt_at_unix_secs
                    .map(|t| t <= now_unix_secs)
                    .unwrap_or(false)
            {
                out.push(attempt);
            }
        }
        Ok(out)
    }
}

// ============================================================
// Codec helpers
// ============================================================

mod b64 {
    //! Minimal base64 codec for our internal payload/secret
    //! encoding. Standard alphabet, no line breaks, no padding
    //! omission. We could pull in the `base64` crate but a sixty-
    //! line implementation removes one supply-chain audit.

    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub fn encode(bytes: &[u8]) -> String {
        let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
        let chunks = bytes.chunks(3);
        for chunk in chunks {
            let b0 = chunk[0];
            let b1 = if chunk.len() > 1 { chunk[1] } else { 0 };
            let b2 = if chunk.len() > 2 { chunk[2] } else { 0 };
            let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | (b2 as u32);
            out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
            out.push(if chunk.len() > 1 {
                ALPHABET[((n >> 6) & 0x3f) as usize] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                ALPHABET[(n & 0x3f) as usize] as char
            } else {
                '='
            });
        }
        out
    }

    pub fn decode(s: &str) -> core::result::Result<Vec<u8>, ()> {
        if !s.len().is_multiple_of(4) {
            return Err(());
        }
        let bytes = s.as_bytes();
        let mut out = Vec::with_capacity(s.len() / 4 * 3);
        let mut i = 0;
        while i < bytes.len() {
            let c0 = val(bytes[i])?;
            let c1 = val(bytes[i + 1])?;
            let c2 = bytes[i + 2];
            let c3 = bytes[i + 3];
            let n = ((c0 as u32) << 18) | ((c1 as u32) << 12);
            out.push((n >> 16) as u8);
            if c2 != b'=' {
                let cv = val(c2)?;
                let n = n | ((cv as u32) << 6);
                out.push((n >> 8) as u8);
                if c3 != b'=' {
                    let dv = val(c3)?;
                    let n = n | (dv as u32);
                    out.push(n as u8);
                }
            }
            i += 4;
        }
        Ok(out)
    }

    fn val(c: u8) -> core::result::Result<u8, ()> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(()),
        }
    }
}

fn json_string(props: &serde_json::Map<String, Json>, key: &str) -> Result<String> {
    props
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned())
        .ok_or_else(|| Error::PropertyTypeMismatch {
            vertex_id: "?".into(),
            property: key.into(),
            expected_type: "string".into(),
        })
}

fn json_opt_string(props: &serde_json::Map<String, Json>, key: &str) -> Option<String> {
    props.get(key).and_then(|v| match v {
        Json::String(s) => Some(s.clone()),
        _ => None,
    })
}

fn json_u64(props: &serde_json::Map<String, Json>, key: &str) -> Result<u64> {
    props
        .get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| Error::PropertyTypeMismatch {
            vertex_id: "?".into(),
            property: key.into(),
            expected_type: "u64".into(),
        })
}

fn json_opt_object_strs(props: &serde_json::Map<String, Json>, key: &str) -> Vec<(String, String)> {
    props
        .get(key)
        .and_then(|v| v.as_object())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
                .collect()
        })
        .unwrap_or_default()
}

fn endpoint_status_str(s: EndpointStatus) -> &'static str {
    match s {
        EndpointStatus::Active => "active",
        EndpointStatus::Disabled => "disabled",
        EndpointStatus::AutoDisabled => "auto_disabled",
    }
}

fn parse_endpoint_status(s: &str) -> Result<EndpointStatus> {
    Ok(match s {
        "active" => EndpointStatus::Active,
        "disabled" => EndpointStatus::Disabled,
        "auto_disabled" => EndpointStatus::AutoDisabled,
        other => {
            return Err(Error::Invariant(format!(
                "unknown endpoint status: {other}"
            )));
        }
    })
}

fn delivery_status_str(s: DeliveryStatus) -> &'static str {
    match s {
        DeliveryStatus::Pending => "pending",
        DeliveryStatus::InFlight => "in_flight",
        DeliveryStatus::Succeeded => "succeeded",
        DeliveryStatus::RetryScheduled => "retry_scheduled",
        DeliveryStatus::Failed => "failed",
    }
}

fn parse_delivery_status(s: &str) -> Result<DeliveryStatus> {
    Ok(match s {
        "pending" => DeliveryStatus::Pending,
        "in_flight" => DeliveryStatus::InFlight,
        "succeeded" => DeliveryStatus::Succeeded,
        "retry_scheduled" => DeliveryStatus::RetryScheduled,
        "failed" => DeliveryStatus::Failed,
        other => {
            return Err(Error::Invariant(format!(
                "unknown delivery status: {other}"
            )));
        }
    })
}

fn encode_filters(filters: &[String]) -> String {
    // We use a comma separator. If a filter contains a comma we
    // escape it as the literal sequence `\,` — collision-resistant
    // enough for a reference impl.
    filters
        .iter()
        .map(|f| f.replace(',', "\\,"))
        .collect::<Vec<_>>()
        .join(",")
}

fn decode_filters(s: &str) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    // Split on un-escaped commas. Hand-rolled, single pass.
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut escape = false;
    for c in s.chars() {
        if escape {
            current.push(c);
            escape = false;
        } else if c == '\\' {
            escape = true;
        } else if c == ',' {
            out.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    out.push(current);
    out
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ep() -> Endpoint {
        Endpoint::new(
            "https://merchant.example/h",
            b"whsec_test_secret".to_vec(),
            vec!["payment.authorized".to_string(), "*".to_string()],
        )
        .unwrap()
    }

    #[test]
    fn endpoint_round_trips() {
        let store = GraphWebhookStore::new_in_memory();
        let e = ep();
        let id = e.id;
        store.put_endpoint(e.clone()).unwrap();
        let recovered = store.get_endpoint(id).unwrap();
        assert_eq!(recovered, e);
    }

    #[test]
    fn endpoint_status_set_persists() {
        let store = GraphWebhookStore::new_in_memory();
        let e = ep();
        let id = e.id;
        store.put_endpoint(e).unwrap();
        store
            .set_endpoint_status(id, EndpointStatus::AutoDisabled)
            .unwrap();
        let r = store.get_endpoint(id).unwrap();
        assert_eq!(r.status, EndpointStatus::AutoDisabled);
    }

    #[test]
    fn endpoint_consecutive_failures_set_persists() {
        let store = GraphWebhookStore::new_in_memory();
        let e = ep();
        let id = e.id;
        store.put_endpoint(e).unwrap();
        store.set_endpoint_consecutive_failures(id, 7).unwrap();
        assert_eq!(store.get_endpoint(id).unwrap().consecutive_failures, 7);
    }

    #[test]
    fn list_active_filters_by_status_and_match() {
        let store = GraphWebhookStore::new_in_memory();
        // e1: active, matches "payment.authorized" explicitly.
        let mut e1 = Endpoint::new(
            "https://a.example/h",
            b"s".to_vec(),
            vec!["payment.authorized".to_string()],
        )
        .unwrap();
        let e1_id = e1.id;
        e1.status = EndpointStatus::Active;
        store.put_endpoint(e1).unwrap();
        // e2: active, wildcard.
        let e2 =
            Endpoint::new("https://b.example/h", b"s".to_vec(), vec!["*".to_string()]).unwrap();
        let e2_id = e2.id;
        store.put_endpoint(e2).unwrap();
        // e3: disabled, wildcard — should be filtered out.
        let mut e3 =
            Endpoint::new("https://c.example/h", b"s".to_vec(), vec!["*".to_string()]).unwrap();
        e3.status = EndpointStatus::Disabled;
        let e3_id = e3.id;
        store.put_endpoint(e3).unwrap();

        let got = store
            .list_active_endpoints_for("payment.authorized")
            .unwrap();
        let ids: Vec<EndpointId> = got.iter().map(|e| e.id).collect();
        assert!(ids.contains(&e1_id));
        assert!(ids.contains(&e2_id));
        assert!(!ids.contains(&e3_id));

        let got = store.list_active_endpoints_for("other.kind").unwrap();
        let ids: Vec<EndpointId> = got.iter().map(|e| e.id).collect();
        assert!(!ids.contains(&e1_id));
        assert!(ids.contains(&e2_id));
    }

    #[test]
    fn event_round_trips() {
        let store = GraphWebhookStore::new_in_memory();
        let ev = WebhookEvent::new("ledger.tx.posted", b"{\"x\":1}".to_vec(), 1_700_000_000);
        let id = ev.id;
        store.put_event(ev.clone()).unwrap();
        assert_eq!(store.get_event(id).unwrap(), ev);
    }

    #[test]
    fn attempt_round_trips_with_edges() {
        let store = GraphWebhookStore::new_in_memory();
        let e = ep();
        let eid = e.id;
        store.put_endpoint(e).unwrap();
        let ev = WebhookEvent::new("x", b"".to_vec(), 0);
        let evid = ev.id;
        store.put_event(ev).unwrap();
        let attempt = DeliveryAttempt::new_pending(evid, eid, 0, 100);
        let aid = attempt.id;
        store.put_attempt(attempt.clone()).unwrap();
        let recovered = store.get_attempt(aid).unwrap();
        assert_eq!(recovered.event_id, evid);
        assert_eq!(recovered.endpoint_id, eid);
        assert_eq!(recovered.attempt_number, 0);
        assert_eq!(recovered.status, DeliveryStatus::Pending);
    }

    #[test]
    fn attempt_against_missing_endpoint_fails() {
        let store = GraphWebhookStore::new_in_memory();
        let ev = WebhookEvent::new("x", b"".to_vec(), 0);
        let evid = ev.id;
        store.put_event(ev).unwrap();
        let phantom_endpoint = EndpointId::new();
        let attempt = DeliveryAttempt::new_pending(evid, phantom_endpoint, 0, 0);
        let r = store.put_attempt(attempt);
        assert!(matches!(r, Err(WebhookError::EndpointNotFound(_))));
    }

    #[test]
    fn attempt_against_missing_event_fails() {
        let store = GraphWebhookStore::new_in_memory();
        let e = ep();
        let eid = e.id;
        store.put_endpoint(e).unwrap();
        let phantom_event = WebhookEventId::new();
        let attempt = DeliveryAttempt::new_pending(phantom_event, eid, 0, 0);
        let r = store.put_attempt(attempt);
        assert!(matches!(r, Err(WebhookError::EventNotFound(_))));
    }

    #[test]
    fn put_attempt_twice_updates_properties() {
        let store = GraphWebhookStore::new_in_memory();
        let e = ep();
        let eid = e.id;
        store.put_endpoint(e).unwrap();
        let ev = WebhookEvent::new("x", b"".to_vec(), 0);
        let evid = ev.id;
        store.put_event(ev).unwrap();

        let mut a = DeliveryAttempt::new_pending(evid, eid, 0, 100);
        let aid = a.id;
        store.put_attempt(a.clone()).unwrap();
        // Update status.
        a.status = DeliveryStatus::Succeeded;
        a.http_status = Some(200);
        a.completed_at_unix_secs = Some(200);
        store.put_attempt(a).unwrap();
        let recovered = store.get_attempt(aid).unwrap();
        assert_eq!(recovered.status, DeliveryStatus::Succeeded);
        assert_eq!(recovered.http_status, Some(200));
        assert_eq!(recovered.completed_at_unix_secs, Some(200));
    }

    #[test]
    fn list_attempts_returns_ordered_by_number() {
        let store = GraphWebhookStore::new_in_memory();
        let e = ep();
        let eid = e.id;
        store.put_endpoint(e).unwrap();
        let ev = WebhookEvent::new("x", b"".to_vec(), 0);
        let evid = ev.id;
        store.put_event(ev).unwrap();
        // Insert out of order.
        for n in [2u32, 0, 1] {
            let a = DeliveryAttempt::new_pending(evid, eid, n, n as u64 * 10);
            store.put_attempt(a).unwrap();
        }
        let list = store.list_attempts(evid, eid).unwrap();
        let numbers: Vec<u32> = list.iter().map(|a| a.attempt_number).collect();
        assert_eq!(numbers, vec![0, 1, 2]);
    }

    #[test]
    fn list_due_retries_filters_correctly() {
        let store = GraphWebhookStore::new_in_memory();
        let e = ep();
        let eid = e.id;
        store.put_endpoint(e).unwrap();
        let ev = WebhookEvent::new("x", b"".to_vec(), 0);
        let evid = ev.id;
        store.put_event(ev).unwrap();
        // a0: RetryScheduled at t=100 (due).
        let mut a0 = DeliveryAttempt::new_pending(evid, eid, 0, 0);
        a0.status = DeliveryStatus::RetryScheduled;
        a0.next_attempt_at_unix_secs = Some(100);
        let a0_id = a0.id;
        store.put_attempt(a0).unwrap();
        // a1: RetryScheduled at t=500 (not yet due).
        let mut a1 = DeliveryAttempt::new_pending(evid, eid, 1, 0);
        a1.status = DeliveryStatus::RetryScheduled;
        a1.next_attempt_at_unix_secs = Some(500);
        store.put_attempt(a1).unwrap();
        // a2: Failed at t=50 (terminal — not due).
        let mut a2 = DeliveryAttempt::new_pending(evid, eid, 2, 0);
        a2.status = DeliveryStatus::Failed;
        a2.next_attempt_at_unix_secs = Some(50);
        store.put_attempt(a2).unwrap();

        let due = store.list_due_retries(200).unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].id, a0_id);
    }

    #[test]
    fn unknown_endpoint_status_set_errors() {
        let store = GraphWebhookStore::new_in_memory();
        let r = store.set_endpoint_status(EndpointId::new(), EndpointStatus::Disabled);
        assert!(matches!(r, Err(WebhookError::EndpointNotFound(_))));
    }

    #[test]
    fn unknown_endpoint_get_errors() {
        let store = GraphWebhookStore::new_in_memory();
        let r = store.get_endpoint(EndpointId::new());
        assert!(matches!(r, Err(WebhookError::EndpointNotFound(_))));
    }

    #[test]
    fn unknown_event_get_errors() {
        let store = GraphWebhookStore::new_in_memory();
        let r = store.get_event(WebhookEventId::new());
        assert!(matches!(r, Err(WebhookError::EventNotFound(_))));
    }

    #[test]
    fn unknown_attempt_get_errors() {
        let store = GraphWebhookStore::new_in_memory();
        let r = store.get_attempt(DeliveryAttemptId::new());
        assert!(matches!(r, Err(WebhookError::AttemptNotFound(_))));
    }

    #[test]
    fn base64_round_trip() {
        for input in [
            b"".to_vec(),
            b"f".to_vec(),
            b"fo".to_vec(),
            b"foo".to_vec(),
            b"foob".to_vec(),
            b"fooba".to_vec(),
            b"foobar".to_vec(),
            (0u8..=255).collect::<Vec<u8>>(),
        ] {
            let s = b64::encode(&input);
            let d = b64::decode(&s).unwrap();
            assert_eq!(
                d,
                input,
                "b64 round-trip failed for input of len {}",
                input.len()
            );
        }
    }

    #[test]
    fn base64_known_vectors() {
        // RFC 4648 test vectors.
        assert_eq!(b64::encode(b""), "");
        assert_eq!(b64::encode(b"f"), "Zg==");
        assert_eq!(b64::encode(b"fo"), "Zm8=");
        assert_eq!(b64::encode(b"foo"), "Zm9v");
        assert_eq!(b64::encode(b"foob"), "Zm9vYg==");
        assert_eq!(b64::encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(b64::encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn filter_csv_codec_round_trip() {
        let inputs = vec![
            vec![],
            vec!["payment.authorized".to_string()],
            vec!["a".to_string(), "b".to_string(), "*".to_string()],
            vec!["with,comma".to_string(), "plain".to_string()],
        ];
        for input in inputs {
            let s = encode_filters(&input);
            let recovered = decode_filters(&s);
            assert_eq!(recovered, input);
        }
    }

    #[test]
    fn endpoint_status_codec_round_trip() {
        for s in [
            EndpointStatus::Active,
            EndpointStatus::Disabled,
            EndpointStatus::AutoDisabled,
        ] {
            assert_eq!(parse_endpoint_status(endpoint_status_str(s)).unwrap(), s);
        }
    }

    #[test]
    fn delivery_status_codec_round_trip() {
        for s in [
            DeliveryStatus::Pending,
            DeliveryStatus::InFlight,
            DeliveryStatus::Succeeded,
            DeliveryStatus::RetryScheduled,
            DeliveryStatus::Failed,
        ] {
            assert_eq!(parse_delivery_status(delivery_status_str(s)).unwrap(), s);
        }
    }
}
