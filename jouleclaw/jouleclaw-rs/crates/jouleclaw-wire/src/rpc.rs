//! `RpcTier` — a cascade tier backed by a remote Joule node.
//!
//! This is the integration that makes federation operational. The tier
//! takes a `Transport` (any closure that ships bytes round-trip) and,
//! on dispatch, encodes a wire request, runs the transport, decodes
//! the response, and surfaces the answer.
//!
//! The receiving side mirrors this with `serve_request` — given a wire
//! request and a `HistoryLayer`, look up the key and respond with a
//! Response or NotFound Error.
//!
//! Cost model: a Transport call has bandwidth + latency cost; the
//! caller bills it. R9 records the *charged* joules from the response
//! and trusts the remote node's accounting. A future trust layer would
//! verify and dispute.

use crate::*;

/// A transport: send bytes, receive bytes. Abstract over any actual
/// network. The closure receives the encoded request and must produce
/// the encoded response (or an empty Vec on transport failure).
pub trait Transport: Send {
    fn round_trip(&mut self, request: &[u8]) -> Result<Vec<u8>, String>;
}

/// A `Tier` that fetches answers from a remote Joule node.
pub struct RpcTier {
    /// Transport for shipping bytes. Owned by the tier.
    transport: Box<dyn Transport>,
    /// Joule cost added on top of whatever the remote node charges.
    /// Accounts for transport + decode work the local node must do.
    pub local_overhead_joules: f64,
    /// Cost per byte of the request payload. Models network bandwidth.
    pub per_byte_joules: f64,
    /// What tier label to report for answers served from this RPC.
    /// Conceptually they're L0 hits (the remote node already paid for
    /// the heavy work), but receivers can report it as the actual
    /// origin tier from the response.
    pub report_as: Option<TierId>,
    /// Monotonic request_id counter.
    next_id: u64,
}

impl RpcTier {
    pub fn new(transport: Box<dyn Transport>) -> Self {
        Self {
            transport,
            local_overhead_joules: 1e-6,
            per_byte_joules: 1e-9,
            report_as: None,
            next_id: 1,
        }
    }
}

impl Tier for RpcTier {
    fn id(&self) -> TierId {
        // RpcTier reports as L0 by default — semantically, it's a
        // remote cache. If the remote answer originated from a higher
        // tier, the trace captures that in the response metadata.
        self.report_as.unwrap_or(TierId::L0)
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        let len = match &q.input {
            QueryInput::Text(s) => s.len(),
            QueryInput::Structured(b) | QueryInput::Binary(b) => b.len(),
            QueryInput::Image(b) | QueryInput::Audio(b) => b.len(),
            QueryInput::Multimodal { text, images, audio } => {
                text.len()
                    + images.iter().map(|v| v.len()).sum::<usize>()
                    + audio.iter().map(|v| v.len()).sum::<usize>()
            }
        };
        // Upper bound: full request bytes + response decode + remote
        // node's worst-case quote. For R9 we use a conservative ~1 mJ
        // baseline as the upper bound; real systems would track
        // observed quote distribution and quote-then-fetch.
        let bytes = 120 + len;   // request envelope + key + per-byte
        let local = self.local_overhead_joules + self.per_byte_joules * bytes as f64;
        Some(TierEstimate {
            joules: local + 1e-3,    // conservative remote cost bound
            latency: std::time::Duration::from_millis(50),
            confidence_floor: 0.5,   // remote answer is "trust but verify"
        })
    }

    fn try_answer(&mut self, q: &Query, _b: f64) -> Result<Answer, AnswerError> {
        let request_id = self.next_id;
        self.next_id += 1;

        let query_key = key_for(q);
        let max_joules = q.budget.remaining(0.0);

        let request = WireMessage::Request(WireRequest {
            query_key, max_joules, request_id,
        });
        let request_bytes = encode(&request);
        let local_cost = self.local_overhead_joules
            + self.per_byte_joules * request_bytes.len() as f64;

        let response_bytes = match self.transport.round_trip(&request_bytes) {
            Ok(b) => b,
            Err(e) => {
                // Transport failed — refuse, charging only local cost.
                let mut trace = ExecutionTrace::default();
                trace.attempts.push(TraceEntry {
                    tier: self.id(),
                    outcome: TraceOutcome::Refused(
                        RefusalReason::TierSpecific(format!("transport: {}", e))),
                    joules: local_cost,
                });
                return Ok(Answer {
                    output: AnswerOutput::Refused(
                        RefusalReason::TierSpecific(format!("transport: {}", e))),
                    tier_used: self.id(),
                    joules_spent: local_cost,
                    confidence: 0.0,
                    trace,
                    verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
                });
            }
        };

        match decode(&response_bytes) {
            Ok(WireMessage::Response(r)) => {
                let total_cost = local_cost + r.joules_charged;
                // Verify the request_id matches; reject otherwise.
                if r.request_id != request_id {
                    return Ok(Answer {
                        output: AnswerOutput::Refused(
                            RefusalReason::TierSpecific(format!(
                                "wire: request_id mismatch (sent {}, got {})",
                                request_id, r.request_id))),
                        tier_used: self.id(),
                        joules_spent: local_cost,
                        confidence: 0.0,
                        trace: ExecutionTrace::default(),
                        verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
                    });
                }
                let mut trace = ExecutionTrace::default();
                trace.attempts.push(TraceEntry {
                    tier: r.origin_tier,
                    outcome: TraceOutcome::Hit,
                    joules: r.joules_charged,
                });
                Ok(Answer {
                    output: r.output,
                    tier_used: self.id(),
                    joules_spent: total_cost,
                    confidence: r.confidence,
                    trace,
                    verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
                })
            }
            Ok(WireMessage::Error(e)) => {
                Ok(Answer {
                    output: AnswerOutput::Refused(
                        RefusalReason::TierSpecific(format!("wire error: {}", e.message))),
                    tier_used: self.id(),
                    joules_spent: local_cost,
                    confidence: 0.0,
                    trace: ExecutionTrace::default(),
                    verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
                })
            }
            Ok(_other) => Ok(Answer {
                output: AnswerOutput::Refused(
                    RefusalReason::TierSpecific("wire: unexpected response kind".into())),
                tier_used: self.id(),
                joules_spent: local_cost,
                confidence: 0.0,
                trace: ExecutionTrace::default(),
                verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
            }),
            Err(e) => Ok(Answer {
                output: AnswerOutput::Refused(
                    RefusalReason::TierSpecific(format!("wire decode: {}", e))),
                tier_used: self.id(),
                joules_spent: local_cost,
                confidence: 0.0,
                trace: ExecutionTrace::default(),
                verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
            }),
        }
    }
}

// ============================================================
// Server-side helper
// ============================================================

/// Serve a `WireRequest` against a `HistoryLayer`. Returns the bytes
/// of a `WireResponse` (on hit) or `WireError::NotFound` (on miss).
///
/// This is the receiving-side counterpart to `RpcTier`. A real server
/// wraps this in TCP/QUIC; the bytes-in/bytes-out shape is enough
/// for federation tests.
pub fn serve_request(
    request_bytes: &[u8],
    history: &mut dyn HistoryLayer,
) -> Vec<u8> {
    let msg = match decode(request_bytes) {
        Ok(WireMessage::Request(r)) => r,
        Ok(_) => {
            return encode(&WireMessage::Error(WireError {
                request_id: 0,
                code: ErrorCode::Malformed,
                message: "expected Request".into(),
            }));
        }
        Err(e) => {
            return encode(&WireMessage::Error(WireError {
                request_id: 0,
                code: ErrorCode::Malformed,
                message: format!("{}", e),
            }));
        }
    };

    match history.lookup_exact(&msg.query_key) {
        Ok(Some(ha)) => {
            // Cost charged: the history layer's lookup cost. The
            // receiver knows what it cost the server to satisfy this.
            let joules_charged = 1e-7;   // representative L0 hit cost
            encode(&WireMessage::Response(WireResponse {
                query_key: msg.query_key,
                request_id: msg.request_id,
                joules_charged,
                confidence: ha.confidence,
                origin_tier: ha.originating_tier,
                expiry_secs: 0,
                output: ha.output,
            }))
        }
        Ok(None) => {
            encode(&WireMessage::Error(WireError {
                request_id: msg.request_id,
                code: ErrorCode::NotFound,
                message: "key not in history".into(),
            }))
        }
        Err(e) => {
            encode(&WireMessage::Error(WireError {
                request_id: msg.request_id,
                code: ErrorCode::Other,
                message: format!("{}", e),
            }))
        }
    }
}
