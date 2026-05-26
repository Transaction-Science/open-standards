//! Tests for the history layer: both backends, durability across
//! "restarts", and integration with the cascade runtime.

use jouleclaw_cascade::*;
use jouleclaw_history::*;

fn text(s: &str) -> Query {
    Query {
        input: QueryInput::Text(s.to_string()),
        budget: JouleBudget::standard(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    }
}

fn answer(text: &str, tier: TierId, joules: f64, confidence: f32) -> Answer {
    Answer {
        output: AnswerOutput::Text(text.to_string()),
        tier_used: tier,
        joules_spent: joules,
        confidence,
        trace: ExecutionTrace::default(),
        verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
    }
}

// ============================================================
// InMemoryHistory tests
// ============================================================

#[test]
fn in_memory_empty_lookup_misses() {
    let mut h = InMemoryHistory::new();
    let key = key_for(&text("hello"));
    assert!(h.lookup_exact(&key).unwrap().is_none());
    assert_eq!(h.stats().misses, 1);
    assert_eq!(h.stats().hits, 0);
}

#[test]
fn in_memory_record_then_lookup_hits() {
    let mut h = InMemoryHistory::new();
    let q = text("what is 2+2?");
    let a = answer("4", TierId::L1(L1Primitive::Execute), 1e-7, 1.0);
    let key = h.record(&q, &a).unwrap();
    let got = h.lookup_exact(&key).unwrap().unwrap();
    assert_eq!(got.output, AnswerOutput::Text("4".to_string()));
    assert_eq!(got.originating_tier, TierId::L1(L1Primitive::Execute));
    assert_eq!(got.confidence, 1.0);
    assert_eq!(h.stats().hits, 1);
    assert_eq!(h.stats().writes, 1);
}

#[test]
fn in_memory_stats_track_writes_and_hits() {
    let mut h = InMemoryHistory::new();
    for i in 0..5 {
        let q = text(&format!("q{}", i));
        let a = answer(&format!("a{}", i), TierId::L4(L4ModelId(0)), 0.5, 0.9);
        h.record(&q, &a).unwrap();
    }
    assert_eq!(h.stats().entry_count, 5);
    assert_eq!(h.stats().writes, 5);
    assert!((h.stats().joules_recorded - 2.5).abs() < 1e-9);
}

#[test]
fn in_memory_idempotent_record() {
    let mut h = InMemoryHistory::new();
    let q = text("x");
    let a = answer("y", TierId::L4(L4ModelId(0)), 0.5, 0.9);
    h.record(&q, &a).unwrap();
    h.record(&q, &a).unwrap();
    h.record(&q, &a).unwrap();
    assert_eq!(h.stats().writes, 1);
    assert_eq!(h.stats().entry_count, 1);
}

#[test]
fn in_memory_semantic_lookup_unsupported_in_r3() {
    let mut h = InMemoryHistory::new();
    let r = h.lookup_semantic(&[0.1, 0.2, 0.3], 5, 0.7);
    assert!(matches!(r, Err(HistoryError::Unsupported(_))));
}

// ============================================================
// DiskHistory tests
// ============================================================

/// Helper: create a unique temp path and arrange cleanup.
struct TempPath {
    path: std::path::PathBuf,
}

impl TempPath {
    fn new(label: &str) -> Self {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!(
            "jouleclaw-history-test-{}-{}-{}.log", label, pid, nanos));
        Self { path }
    }
    fn p(&self) -> &std::path::Path { &self.path }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[test]
fn disk_fresh_file_has_header() {
    let t = TempPath::new("fresh");
    let h = DiskHistory::open(t.p()).unwrap();
    assert!(h.is_empty());
    // File should now exist with 16-byte header.
    let meta = std::fs::metadata(t.p()).unwrap();
    assert_eq!(meta.len(), 16);
}

#[test]
fn disk_record_persists_across_reopen() {
    let t = TempPath::new("persist");
    {
        let mut h = DiskHistory::open(t.p()).unwrap();
        let q = text("what is 2+2?");
        let a = answer("4", TierId::L1(L1Primitive::Execute), 1e-7, 1.0);
        h.record(&q, &a).unwrap();
        h.flush().unwrap();
        assert_eq!(h.len(), 1);
    }
    // Reopen the same file — entry must still be there.
    {
        let mut h = DiskHistory::open(t.p()).unwrap();
        assert_eq!(h.len(), 1);
        let key = key_for(&text("what is 2+2?"));
        let got = h.lookup_exact(&key).unwrap().unwrap();
        assert_eq!(got.output, AnswerOutput::Text("4".to_string()));
        assert_eq!(got.originating_tier, TierId::L1(L1Primitive::Execute));
        assert_eq!(got.confidence, 1.0);
    }
}

#[test]
fn disk_many_records_survive_reopen() {
    let t = TempPath::new("many");
    let count = 50;
    {
        let mut h = DiskHistory::open(t.p()).unwrap();
        for i in 0..count {
            let q = text(&format!("query-{}", i));
            let a = answer(&format!("answer-{}", i),
                TierId::L4(L4ModelId(0)), 0.5, 0.95);
            h.record(&q, &a).unwrap();
        }
        h.flush().unwrap();
        assert_eq!(h.len(), count);
    }
    let mut h = DiskHistory::open(t.p()).unwrap();
    assert_eq!(h.len(), count);
    for i in 0..count {
        let key = key_for(&text(&format!("query-{}", i)));
        let got = h.lookup_exact(&key).unwrap().unwrap();
        assert_eq!(got.output, AnswerOutput::Text(format!("answer-{}", i)));
    }
}

#[test]
fn disk_corrupt_header_errors_cleanly() {
    let t = TempPath::new("corrupt");
    // Write garbage instead of a valid header.
    std::fs::write(t.p(), b"NOTJOULE00000000extra").unwrap();
    let r = DiskHistory::open(t.p());
    assert!(matches!(r, Err(HistoryError::Corrupt(_))));
}

#[test]
fn disk_all_output_kinds_round_trip() {
    let t = TempPath::new("kinds");
    let test_cases = vec![
        (text("text-out"),
         Answer {
             output: AnswerOutput::Text("hello".into()),
             tier_used: TierId::L1(L1Primitive::Execute),
             joules_spent: 1e-7, confidence: 1.0,
             trace: ExecutionTrace::default(),
             verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
         }),
        (text("structured-out"),
         Answer {
             output: AnswerOutput::Structured(b"{\"k\":1}".to_vec()),
             tier_used: TierId::L2(L2ModelId(7)),
             joules_spent: 1e-3, confidence: 0.8,
             trace: ExecutionTrace::default(),
             verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
         }),
        (text("refused-inapplicable"),
         Answer {
             output: AnswerOutput::Refused(RefusalReason::Inapplicable),
             tier_used: TierId::L1(L1Primitive::Regex),
             joules_spent: 1e-8, confidence: 0.0,
             trace: ExecutionTrace::default(),
             verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
         }),
        (text("refused-tier-specific"),
         Answer {
             output: AnswerOutput::Refused(
                 RefusalReason::TierSpecific("nope".into())),
             tier_used: TierId::L3(L3ModelId(42)),
             joules_spent: 1e-2, confidence: 0.0,
             trace: ExecutionTrace::default(),
             verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
         }),
    ];
    {
        let mut h = DiskHistory::open(t.p()).unwrap();
        for (q, a) in &test_cases {
            h.record(q, a).unwrap();
        }
        h.flush().unwrap();
    }
    let mut h = DiskHistory::open(t.p()).unwrap();
    for (q, a) in &test_cases {
        let key = key_for(q);
        let got = h.lookup_exact(&key).unwrap().unwrap();
        assert_eq!(got.output, a.output);
        assert_eq!(got.originating_tier, a.tier_used);
        assert!((got.joules_spent - a.joules_spent).abs() < 1e-12);
        assert!((got.confidence - a.confidence).abs() < 1e-6);
    }
}

// ============================================================
// Runtime integration
// ============================================================

/// MockL4 for the runtime tests below.
struct MockL4 {
    cost: f64,
    text: String,
}

impl Tier for MockL4 {
    fn id(&self) -> TierId { TierId::L4(L4ModelId(0)) }
    fn estimate_cost(&self, _q: &Query) -> Option<TierEstimate> {
        Some(TierEstimate {
            joules: self.cost,
            latency: std::time::Duration::from_millis(100),
            confidence_floor: 0.9,
        })
    }
    fn try_answer(&mut self, _q: &Query, _b: f64) -> Result<Answer, AnswerError> {
        Ok(Answer {
            output: AnswerOutput::Text(self.text.clone()),
            tier_used: self.id(),
            joules_spent: self.cost,
            confidence: 0.9,
            trace: ExecutionTrace::default(),
            verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
        })
    }
}

/// Durability: a query answered by L4 in one runtime is L0-cached on
/// disk; a SECOND runtime instance pointed at the same file hits L0
/// for the same query.
#[test]
fn runtime_history_survives_restart() {
    let t = TempPath::new("runtime");

    let q = text("capital of France?");

    // First "process": L4 answers, history records.
    {
        let mut cascade = Cascade::new();
        cascade.register(Box::new(MockL4 {
            cost: 0.5, text: "Paris".into(),
        }));
        let history = Box::new(DiskHistory::open(t.p()).unwrap());
        let mut rt = Runtime::new_with_history(cascade, history);
        let a = rt.answer(q.clone()).unwrap();
        assert_eq!(a.tier_used, TierId::L4(L4ModelId(0)));
        assert_eq!(a.output, AnswerOutput::Text("Paris".to_string()));
    }

    // Second "process": fresh runtime, same disk path. Same query
    // should hit L0 (via the history layer).
    {
        let mut cascade = Cascade::new();
        cascade.register(Box::new(MockL4 {
            cost: 0.5, text: "Paris".into(),
        }));
        let history = Box::new(DiskHistory::open(t.p()).unwrap());
        let mut rt = Runtime::new_with_history(cascade, history);
        let a = rt.answer(q).unwrap();
        assert_eq!(a.tier_used, TierId::L0,
            "history should make second process hit L0; got {:?}", a.tier_used);
        assert_eq!(a.output, AnswerOutput::Text("Paris".to_string()));
        // L0 hit cost is ~nJ; the L4 dispatch in the first process
        // cost 500 mJ.
        assert!(a.joules_spent < 1e-6,
            "L0 hit should be cheap; got {:.3e} J", a.joules_spent);
    }
}

#[test]
fn in_memory_history_runtime_integration() {
    let mut cascade = Cascade::new();
    cascade.register(Box::new(MockL4 {
        cost: 0.5, text: "Paris".into(),
    }));
    let history = Box::new(InMemoryHistory::new());
    let mut rt = Runtime::new_with_history(cascade, history);

    let q = text("capital of France?");
    let a1 = rt.answer(q.clone()).unwrap();
    assert_eq!(a1.tier_used, TierId::L4(L4ModelId(0)));

    let a2 = rt.answer(q).unwrap();
    assert_eq!(a2.tier_used, TierId::L0);
    assert!(a2.joules_spent * 1e4 < a1.joules_spent,
        "L0 hit should be 10000+× cheaper");
}

// ============================================================
// Demo
// ============================================================

#[test]
fn history_durability_demo() {
    println!("\n=== R3: history layer survives runtime restart ===\n");
    let t = TempPath::new("demo");

    println!("--- Process 1 ---");
    {
        let mut cascade = Cascade::new();
        cascade.register(Box::new(MockL4 {
            cost: 0.5, text: "Paris".into(),
        }));
        let history = Box::new(DiskHistory::open(t.p()).unwrap());
        let mut rt = Runtime::new_with_history(cascade, history);
        let a = rt.answer(text("capital of France?")).unwrap();
        println!("  Answer: {:?}", a.output);
        println!("  Tier:   {:?}", a.tier_used);
        println!("  Cost:   {:.3e} J", a.joules_spent);
        println!("  Disk file: {:?}", t.p().file_name().unwrap());
    }
    println!("[process exits; disk file remains]");

    println!("\n--- Process 2 (fresh runtime, same disk path) ---");
    {
        let mut cascade = Cascade::new();
        cascade.register(Box::new(MockL4 {
            cost: 0.5, text: "Paris".into(),
        }));
        let history = Box::new(DiskHistory::open(t.p()).unwrap());
        let mut rt = Runtime::new_with_history(cascade, history);
        let a = rt.answer(text("capital of France?")).unwrap();
        println!("  Answer: {:?}", a.output);
        println!("  Tier:   {:?}  <-- cache hit from disk!", a.tier_used);
        println!("  Cost:   {:.3e} J", a.joules_spent);
    }

    let meta = std::fs::metadata(t.p()).unwrap();
    println!("\n  Disk file size: {} bytes", meta.len());
    println!("  The cache survived. Same answer, ~10^7× cheaper.");
}
