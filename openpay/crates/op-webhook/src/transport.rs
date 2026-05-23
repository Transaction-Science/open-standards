//! HTTP transport seam.
//!
//! The crate ships no real HTTP client. Operators inject `reqwest`,
//! `hyper`, `ureq`, or anything else through the [`HttpTransport`]
//! trait. This keeps the dependency footprint minimal and lets
//! operators pick whatever fits their async/sync story.
//!
//! [`MockTransport`] is the in-test stand-in: programmable per-call
//! responses, request capture for assertions.

use std::sync::Mutex;

use crate::error::{Error, Result};

/// HTTP request emitted by the dispatcher.
///
/// Headers include the `OpenPay-Signature` value and any
/// content-type the operator configures globally (we default to
/// `application/json` but accept anything).
#[derive(Clone, Debug)]
pub struct HttpRequest {
    /// Destination URL.
    pub url: String,
    /// Header list. `(name, value)` pairs.
    pub headers: Vec<(String, String)>,
    /// Request body.
    pub body: Vec<u8>,
    /// Maximum time to wait for a response. The transport SHOULD
    /// honor this; if not, the dispatcher's external watchdog (an
    /// operator-side concern) is the backstop.
    pub timeout_secs: u32,
}

/// HTTP response.
#[derive(Clone, Debug)]
pub struct HttpResponse {
    /// Status code.
    pub status: u16,
    /// Response body. Truncated by the transport at its discretion.
    pub body: Vec<u8>,
}

/// HTTP transport trait. Sync.
pub trait HttpTransport: Send + Sync {
    /// Send `request` and synchronously return a response.
    ///
    /// # Errors
    /// [`Error::Transport`] on network / DNS / TLS / timeout
    /// failures. Non-2xx HTTP responses are NOT errors at this
    /// layer — they're returned in `HttpResponse.status` and
    /// classified by the dispatcher.
    fn send(&self, request: &HttpRequest) -> Result<HttpResponse>;
}

// ============================================================
// MockTransport
// ============================================================

/// Programmable in-process HTTP transport for tests.
///
/// Construct, then `push_response(...)` one or more times to queue
/// the responses successive calls to `send()` will return. After
/// the queue empties, subsequent calls return a default 200.
///
/// `take_captured()` returns the list of [`HttpRequest`]s captured
/// in order.
pub struct MockTransport {
    queued: Mutex<Vec<MockResponse>>,
    captured: Mutex<Vec<HttpRequest>>,
}

/// What a `MockTransport` returns for a single call.
#[derive(Clone, Debug)]
pub enum MockResponse {
    /// HTTP 2xx-5xx response.
    Response(HttpResponse),
    /// Transport-level failure.
    TransportError(String),
}

impl Default for MockTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl MockTransport {
    /// Construct.
    #[must_use]
    pub fn new() -> Self {
        Self {
            queued: Mutex::new(Vec::new()),
            captured: Mutex::new(Vec::new()),
        }
    }

    /// Queue a response. Calls are answered in queued order.
    pub fn push_response(&self, r: MockResponse) {
        self.queued.lock().expect("poisoned").push(r);
    }

    /// Queue a simple 2xx success.
    pub fn push_ok(&self) {
        self.push_response(MockResponse::Response(HttpResponse {
            status: 200,
            body: b"ok".to_vec(),
        }));
    }

    /// Queue a 5xx server error.
    pub fn push_5xx(&self, status: u16) {
        self.push_response(MockResponse::Response(HttpResponse {
            status,
            body: format!("error {status}").into_bytes(),
        }));
    }

    /// Queue a transport failure (timeout, DNS error, etc.).
    pub fn push_transport_err(&self, msg: impl Into<String>) {
        self.push_response(MockResponse::TransportError(msg.into()));
    }

    /// Return and clear the captured request log.
    pub fn take_captured(&self) -> Vec<HttpRequest> {
        let mut g = self.captured.lock().expect("poisoned");
        std::mem::take(&mut *g)
    }

    /// Number of captured requests so far.
    pub fn captured_count(&self) -> usize {
        self.captured.lock().expect("poisoned").len()
    }
}

impl HttpTransport for MockTransport {
    fn send(&self, request: &HttpRequest) -> Result<HttpResponse> {
        self.captured
            .lock()
            .expect("poisoned")
            .push(request.clone());

        // Pop next queued response. If queue is empty, default to 200.
        let next = {
            let mut q = self.queued.lock().expect("poisoned");
            if q.is_empty() {
                None
            } else {
                Some(q.remove(0))
            }
        };
        match next {
            None => Ok(HttpResponse {
                status: 200,
                body: b"ok".to_vec(),
            }),
            Some(MockResponse::Response(r)) => Ok(r),
            Some(MockResponse::TransportError(msg)) => Err(Error::Transport(msg)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> HttpRequest {
        HttpRequest {
            url: "https://merchant/h".into(),
            headers: vec![],
            body: b"body".to_vec(),
            timeout_secs: 10,
        }
    }

    #[test]
    fn default_returns_200_when_queue_empty() {
        let t = MockTransport::new();
        let r = t.send(&req()).unwrap();
        assert_eq!(r.status, 200);
    }

    #[test]
    fn queued_responses_returned_in_order() {
        let t = MockTransport::new();
        t.push_5xx(503);
        t.push_5xx(502);
        t.push_ok();
        assert_eq!(t.send(&req()).unwrap().status, 503);
        assert_eq!(t.send(&req()).unwrap().status, 502);
        assert_eq!(t.send(&req()).unwrap().status, 200);
    }

    #[test]
    fn transport_error_is_returned_as_err() {
        let t = MockTransport::new();
        t.push_transport_err("connection refused");
        let r = t.send(&req());
        assert!(matches!(r, Err(Error::Transport(_))));
    }

    #[test]
    fn requests_are_captured_in_order() {
        let t = MockTransport::new();
        let mut r1 = req();
        r1.body = b"first".to_vec();
        let mut r2 = req();
        r2.body = b"second".to_vec();
        let _ = t.send(&r1);
        let _ = t.send(&r2);
        let captured = t.take_captured();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].body, b"first");
        assert_eq!(captured[1].body, b"second");
    }

    #[test]
    fn take_captured_clears_log() {
        let t = MockTransport::new();
        let _ = t.send(&req());
        assert_eq!(t.captured_count(), 1);
        let _ = t.take_captured();
        assert_eq!(t.captured_count(), 0);
    }
}
