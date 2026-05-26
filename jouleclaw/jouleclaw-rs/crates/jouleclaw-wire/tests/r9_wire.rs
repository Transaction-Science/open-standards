//! R9 tests — wire protocol round-trips, error handling, and the
//! federation end-to-end.

use jouleclaw_cascade::*;
use jouleclaw_history::InMemoryHistory;
use jouleclaw_wire::*;

// ============================================================
// Round-trip
// ============================================================

#[test]
fn round_trip_request() {
    let msg = WireMessage::Request(WireRequest {
        query_key: [0xab; 32],
        max_joules: 1e-3,
        request_id: 42,
    });
    let bytes = encode(&msg);
    let decoded = decode(&bytes).unwrap();
    match decoded {
        WireMessage::Request(r) => {
            assert_eq!(r.query_key, [0xab; 32]);
            assert_eq!(r.max_joules, 1e-3);
            assert_eq!(r.request_id, 42);
        }
        other => panic!("expected Request, got {:?}", other),
    }
}

#[test]
fn round_trip_quote() {
    let msg = WireMessage::Quote(WireQuote {
        query_key: [0x33; 32],
        request_id: 17,
        joules_charged: 5e-7,
        confidence: 0.95,
        origin_tier: TierId::L1(L1Primitive::Execute),
        expiry_secs: 1_700_000_000,
    });
    let bytes = encode(&msg);
    let decoded = decode(&bytes).unwrap();
    match decoded {
        WireMessage::Quote(q) => {
            assert_eq!(q.query_key, [0x33; 32]);
            assert_eq!(q.request_id, 17);
            assert_eq!(q.joules_charged, 5e-7);
            assert!((q.confidence - 0.95).abs() < 1e-6);
            assert_eq!(q.origin_tier, TierId::L1(L1Primitive::Execute));
            assert_eq!(q.expiry_secs, 1_700_000_000);
        }
        other => panic!("expected Quote, got {:?}", other),
    }
}

#[test]
fn round_trip_response_text() {
    let msg = WireMessage::Response(WireResponse {
        query_key: [0x77; 32],
        request_id: 100,
        joules_charged: 1e-7,
        confidence: 0.9,
        origin_tier: TierId::L4(L4ModelId(0)),
        expiry_secs: 0,
        output: AnswerOutput::Text("Paris".into()),
    });
    let bytes = encode(&msg);
    let decoded = decode(&bytes).unwrap();
    match decoded {
        WireMessage::Response(r) => {
            assert_eq!(r.output, AnswerOutput::Text("Paris".into()));
            assert_eq!(r.origin_tier, TierId::L4(L4ModelId(0)));
        }
        other => panic!("expected Response, got {:?}", other),
    }
}

#[test]
fn round_trip_response_structured() {
    let msg = WireMessage::Response(WireResponse {
        query_key: [0; 32],
        request_id: 1,
        joules_charged: 1e-6,
        confidence: 1.0,
        origin_tier: TierId::L1(L1Primitive::Regex),
        expiry_secs: 0,
        output: AnswerOutput::Structured(vec![1, 2, 3, 4, 5]),
    });
    let bytes = encode(&msg);
    let decoded = decode(&bytes).unwrap();
    if let WireMessage::Response(r) = decoded {
        assert_eq!(r.output, AnswerOutput::Structured(vec![1, 2, 3, 4, 5]));
    } else {
        panic!("wrong kind");
    }
}

#[test]
fn round_trip_response_refused_inapplicable() {
    let msg = WireMessage::Response(WireResponse {
        query_key: [0; 32],
        request_id: 1,
        joules_charged: 0.0,
        confidence: 0.0,
        origin_tier: TierId::L0,
        expiry_secs: 0,
        output: AnswerOutput::Refused(RefusalReason::Inapplicable),
    });
    let bytes = encode(&msg);
    let decoded = decode(&bytes).unwrap();
    if let WireMessage::Response(r) = decoded {
        assert_eq!(r.output, AnswerOutput::Refused(RefusalReason::Inapplicable));
    }
}

#[test]
fn round_trip_response_refused_tier_specific() {
    let msg = WireMessage::Response(WireResponse {
        query_key: [0; 32],
        request_id: 1,
        joules_charged: 0.0,
        confidence: 0.0,
        origin_tier: TierId::L0,
        expiry_secs: 0,
        output: AnswerOutput::Refused(RefusalReason::TierSpecific("nope".into())),
    });
    let bytes = encode(&msg);
    let decoded = decode(&bytes).unwrap();
    if let WireMessage::Response(r) = decoded {
        assert_eq!(r.output,
            AnswerOutput::Refused(RefusalReason::TierSpecific("nope".into())));
    }
}

#[test]
fn round_trip_error() {
    let msg = WireMessage::Error(WireError {
        request_id: 999,
        code: ErrorCode::BudgetExceeded,
        message: "remote node ran out of budget".into(),
    });
    let bytes = encode(&msg);
    let decoded = decode(&bytes).unwrap();
    match decoded {
        WireMessage::Error(e) => {
            assert_eq!(e.request_id, 999);
            assert_eq!(e.code, ErrorCode::BudgetExceeded);
            assert_eq!(e.message, "remote node ran out of budget");
        }
        other => panic!("expected Error, got {:?}", other),
    }
}

// ============================================================
// Error handling
// ============================================================

#[test]
fn bad_magic_detected() {
    let bad = b"NOTJOULE000000000000000000000000000000";
    let result = decode(bad);
    assert!(matches!(result, Err(DecodeError::BadMagic)));
}

#[test]
fn truncated_input_returns_clean_error() {
    let msg = WireMessage::Request(WireRequest {
        query_key: [0; 32], max_joules: 1.0, request_id: 1,
    });
    let bytes = encode(&msg);
    // Truncate to half.
    let truncated = &bytes[..bytes.len() / 2];
    let result = decode(truncated);
    assert!(matches!(result, Err(DecodeError::Truncated { .. })));
}

#[test]
fn unknown_kind_returns_clean_error() {
    // Build a syntactically valid envelope with bogus kind 99.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.push(99);                          // unknown kind
    bytes.extend_from_slice(&0u64.to_be_bytes());  // payload_len 0
    bytes.extend_from_slice(&[0u8; SIGNATURE_LEN]);
    let result = decode(&bytes);
    assert!(matches!(result, Err(DecodeError::UnknownKind(99))));
}

// ============================================================
// Server-side: serve_request against a history layer
// ============================================================

fn text(s: &str) -> Query {
    Query {
        input: QueryInput::Text(s.to_string()),
        budget: JouleBudget::standard(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    }
}

fn answer(text: &str, tier: TierId, joules: f64, conf: f32) -> Answer {
    Answer {
        output: AnswerOutput::Text(text.to_string()),
        tier_used: tier,
        joules_spent: joules,
        confidence: conf,
        trace: ExecutionTrace::default(),
        verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
    }
}

#[test]
fn serve_request_returns_cached_answer() {
    let mut history = InMemoryHistory::new();
    let q = text("capital of France?");
    let a = answer("Paris", TierId::L4(L4ModelId(0)), 0.5, 0.9);
    history.record(&q, &a).unwrap();

    let request = WireMessage::Request(WireRequest {
        query_key: key_for(&q),
        max_joules: 1.0,
        request_id: 7,
    });
    let request_bytes = encode(&request);
    let response_bytes = serve_request(&request_bytes, &mut history);
    let response = decode(&response_bytes).unwrap();
    match response {
        WireMessage::Response(r) => {
            assert_eq!(r.request_id, 7);
            assert_eq!(r.output, AnswerOutput::Text("Paris".into()));
            assert_eq!(r.origin_tier, TierId::L4(L4ModelId(0)));
            assert!((r.confidence - 0.9).abs() < 1e-6);
        }
        other => panic!("expected Response, got {:?}", other),
    }
}

#[test]
fn serve_request_returns_not_found_for_missing_key() {
    let mut history = InMemoryHistory::new();
    let request = WireMessage::Request(WireRequest {
        query_key: [0xff; 32],
        max_joules: 1.0,
        request_id: 99,
    });
    let request_bytes = encode(&request);
    let response_bytes = serve_request(&request_bytes, &mut history);
    let response = decode(&response_bytes).unwrap();
    match response {
        WireMessage::Error(e) => {
            assert_eq!(e.code, ErrorCode::NotFound);
            assert_eq!(e.request_id, 99);
        }
        other => panic!("expected NotFound Error, got {:?}", other),
    }
}

// ============================================================
// Federation end-to-end
// ============================================================

/// A transport that wraps a server's history layer in a closure.
/// Each `round_trip` calls `serve_request` against the server's
/// history. Models two nodes sharing a backing store.
struct InProcessTransport {
    server_history: std::sync::Arc<std::sync::Mutex<InMemoryHistory>>,
    /// Count of how many round-trips happened — observability for tests.
    pub call_count: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl Transport for InProcessTransport {
    fn round_trip(&mut self, request: &[u8]) -> Result<Vec<u8>, String> {
        self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut h = self.server_history.lock().unwrap();
        Ok(serve_request(request, &mut *h))
    }
}

/// A mock L4 tier — used to seed the server's history.
struct MockL4;
impl Tier for MockL4 {
    fn id(&self) -> TierId { TierId::L4(L4ModelId(0)) }
    fn estimate_cost(&self, _q: &Query) -> Option<TierEstimate> {
        Some(TierEstimate {
            joules: 0.5,
            latency: std::time::Duration::from_millis(100),
            confidence_floor: 0.9,
        })
    }
    fn try_answer(&mut self, _q: &Query, _b: f64) -> Result<Answer, AnswerError> {
        Ok(Answer {
            output: AnswerOutput::Text("Paris".into()),
            tier_used: self.id(),
            joules_spent: 0.5,
            confidence: 0.9,
            trace: ExecutionTrace::default(),
            verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
        })
    }
}

#[test]
fn federation_two_runtimes_share_answers_via_wire() {
    // Server side: a Runtime that has answered the query via L4.
    let server_history = std::sync::Arc::new(std::sync::Mutex::new(InMemoryHistory::new()));
    {
        let mut server_cascade = Cascade::new();
        server_cascade.register(Box::new(MockL4));
        let history_clone: Box<dyn HistoryLayer> = Box::new(server_history.lock().unwrap().clone_for_test());
        // We can't pass an Arc<Mutex> to the runtime directly; for the
        // test we seed the server history manually by running a query
        // through a server Runtime and then mirroring the answer.
        let mut rt = Runtime::new_with_history(server_cascade, history_clone);
        let q = text("capital of France?");
        let a = rt.answer(q.clone()).unwrap();
        // Manually record into the shared history.
        server_history.lock().unwrap().record(&q, &a).unwrap();
    }

    // Client side: a Runtime with an RpcTier that reaches into the
    // server's history.
    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let transport = InProcessTransport {
        server_history: server_history.clone(),
        call_count: call_count.clone(),
    };
    let rpc = RpcTier::new(Box::new(transport));

    let mut client_cascade = Cascade::new();
    client_cascade.register(Box::new(rpc));
    client_cascade.register(Box::new(MockL4));   // local fallback
    let mut client_rt = Runtime::new(client_cascade);

    let a = client_rt.answer(text("capital of France?")).unwrap();

    // The RpcTier should have answered the query at low cost,
    // skipping the local L4.
    assert_eq!(a.tier_used, TierId::L0,
        "RpcTier should report as L0; got {:?}", a.tier_used);
    assert_eq!(a.output, AnswerOutput::Text("Paris".into()));
    // Cost: local RPC overhead (~1 µJ) + remote charged (~100 nJ).
    // Should be far less than 500 mJ (L4 cost).
    assert!(a.joules_spent < 1e-3,
        "federated cache hit should be <<L4 cost; got {:.3e}", a.joules_spent);
    assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1,
        "exactly one RPC round-trip should have occurred");
}

// Helper: clone the in-memory history for testing.
trait CloneForTest {
    fn clone_for_test(&self) -> Self;
}

impl CloneForTest for InMemoryHistory {
    fn clone_for_test(&self) -> Self {
        // Simple clone — entries and stats are not visible directly,
        // so we just create a fresh one (the test re-seeds anyway).
        InMemoryHistory::new()
    }
}

// ============================================================
// Demo
// ============================================================

#[test]
fn r9_wire_demo() {
    println!("\n=== R9: federated cascade via metered wire protocol ===\n");

    // Same setup as the federation test but with verbose printing.
    let server_history = std::sync::Arc::new(std::sync::Mutex::new(InMemoryHistory::new()));
    {
        let q = text("capital of France?");
        let a = answer("Paris", TierId::L4(L4ModelId(0)), 0.5, 0.9);
        server_history.lock().unwrap().record(&q, &a).unwrap();
        println!("Server seeded with one answer (cost 500 mJ at L4):");
        println!("  query: \"capital of France?\" → \"Paris\"");
    }

    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let transport = InProcessTransport {
        server_history: server_history.clone(),
        call_count: call_count.clone(),
    };
    let rpc = RpcTier::new(Box::new(transport));

    let mut client_cascade = Cascade::new();
    client_cascade.register(Box::new(rpc));
    client_cascade.register(Box::new(MockL4));
    let mut client_rt = Runtime::new(client_cascade);

    println!("\nClient queries: \"capital of France?\"");
    let a = client_rt.answer(text("capital of France?")).unwrap();

    println!("  tier_used:   {:?}", a.tier_used);
    println!("  answer:      {:?}", a.output);
    println!("  client cost: {:.3e} J  (vs L4 cost 5.000e-1 J)", a.joules_spent);
    println!("  speedup:     ~{:.0}× cheaper than local L4",
        0.5 / a.joules_spent.max(1e-12));
    println!("  rpc calls:   {}", call_count.load(std::sync::atomic::Ordering::SeqCst));

    // Wire bytes for reference.
    let request = encode(&WireMessage::Request(WireRequest {
        query_key: key_for(&text("capital of France?")),
        max_joules: 1.0, request_id: 1,
    }));
    println!("\nA request envelope is {} bytes on the wire", request.len());
    println!("  16 header (magic+version+kind+payload_len)");
    println!("  48 payload (32 key + 8 max_joules + 8 request_id)");
    println!("  64 signature placeholder (reserved for crypto)");
    println!();
    println!("This is the seam where federation works:");
    println!("• Server pays L4 cost once; bills clients per-RPC.");
    println!("• Client pays local RPC overhead + the server's quote.");
    println!("• Every message carries its origin tier, confidence,");
    println!("  joules charged, and expiry — receivers know what");
    println!("  they're buying.");
}
