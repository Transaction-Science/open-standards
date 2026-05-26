//! R8 tests — tiered memory transitions.
//!
//! Proves:
//!   - Hot tier respects capacity; excess items demote to Warm
//!   - Warm tier respects capacity; excess items demote to Cold
//!   - Lookups walk Hot → Warm → Cold
//!   - Cold/Warm hits promote back to Hot
//!   - Transition joule costs are recorded and bounded

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

fn answer(text: &str, tier: TierId, joules: f64) -> Answer {
    Answer {
        output: AnswerOutput::Text(text.to_string()),
        tier_used: tier,
        joules_spent: joules,
        confidence: 0.9,
        trace: ExecutionTrace::default(),
        verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
    }
}

struct TempPath { path: std::path::PathBuf }
impl TempPath {
    fn new(label: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        Self {
            path: std::env::temp_dir().join(format!(
                "joule-r8-test-{}-{}-{}.log",
                label, std::process::id(), nanos)),
        }
    }
    fn p(&self) -> &std::path::Path { &self.path }
}
impl Drop for TempPath {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.path); }
}

// ============================================================
// WarmCache unit tests
// ============================================================

#[test]
fn warm_cache_respects_capacity() {
    let mut warm = WarmCache::new(3);
    for i in 0..3 {
        let q = text(&format!("query-{}", i));
        let a = answer(&format!("answer-{}", i), TierId::L4(L4ModelId(0)), 0.5);
        warm.record(&q, &a).unwrap();
    }
    assert_eq!(warm.len(), 3);

    // Insert a 4th item — should evict the LRU (query-0).
    let q = text("query-3");
    let a = answer("answer-3", TierId::L4(L4ModelId(0)), 0.5);
    warm.record(&q, &a).unwrap();
    assert_eq!(warm.len(), 3, "capacity should hold at 3");

    // query-0 should be gone.
    let key_0 = key_for(&text("query-0"));
    assert!(warm.lookup_exact(&key_0).unwrap().is_none());

    // query-3 should be there.
    let key_3 = key_for(&text("query-3"));
    assert!(warm.lookup_exact(&key_3).unwrap().is_some());
}

#[test]
fn warm_cache_touches_on_lookup() {
    let mut warm = WarmCache::new(3);
    for i in 0..3 {
        let q = text(&format!("q-{}", i));
        let a = answer(&format!("a-{}", i), TierId::L4(L4ModelId(0)), 0.5);
        warm.record(&q, &a).unwrap();
    }

    // Touch q-0 — should move it to MRU.
    let key_0 = key_for(&text("q-0"));
    warm.lookup_exact(&key_0).unwrap();

    // Insert q-3 — q-1 should be evicted now (not q-0).
    let q = text("q-3");
    let a = answer("a-3", TierId::L4(L4ModelId(0)), 0.5);
    warm.record(&q, &a).unwrap();

    assert!(warm.lookup_exact(&key_0).unwrap().is_some(), "q-0 should survive (touched)");
    let key_1 = key_for(&text("q-1"));
    assert!(warm.lookup_exact(&key_1).unwrap().is_none(), "q-1 should have been evicted");
}

// ============================================================
// TieredMemory integration
// ============================================================

#[test]
fn tiered_memory_fills_hot_first() {
    let t = TempPath::new("hot_first");
    let mut m = TieredMemory::open(2, 4, t.p()).unwrap();

    // Insert two items — both should land in Hot.
    for i in 0..2 {
        let q = text(&format!("q-{}", i));
        let a = answer(&format!("a-{}", i), TierId::L4(L4ModelId(0)), 0.5);
        m.record(&q, &a).unwrap();
    }
    let (hot, warm, _) = m.tier_sizes();
    assert_eq!(hot, 2);
    assert_eq!(warm, 0);
}

#[test]
fn tiered_memory_demotes_overflow_from_hot_to_warm() {
    let t = TempPath::new("demote_hw");
    let mut m = TieredMemory::open(2, 4, t.p()).unwrap();

    // Insert 4 items into a 2-capacity Hot. Two should fall through
    // to Warm.
    for i in 0..4 {
        let q = text(&format!("q-{}", i));
        let a = answer(&format!("a-{}", i), TierId::L4(L4ModelId(0)), 0.5);
        m.record(&q, &a).unwrap();
        // Sleep 1ms-ish to ensure distinct timestamps for LRU.

    }
    let (hot, warm, cold) = m.tier_sizes();
    assert_eq!(hot, 2, "Hot should hold 2 items");
    assert_eq!(warm, 2, "Warm should hold the 2 demoted items");
    assert_eq!(cold, 4, "Cold (durable) should hold all 4");
}

#[test]
fn tiered_memory_lookup_walks_hot_warm_cold() {
    let t = TempPath::new("walk");
    let mut m = TieredMemory::open(2, 2, t.p()).unwrap();

    // Insert 6 items into a 2/2/inf tiered memory. Final state:
    //   Hot: q-4, q-5 (latest 2)
    //   Warm: q-2, q-3 (next 2 most recent)
    //   Cold: all 6
    for i in 0..6 {
        let q = text(&format!("q-{}", i));
        let a = answer(&format!("a-{}", i), TierId::L4(L4ModelId(0)), 0.5);
        m.record(&q, &a).unwrap();

    }

    // Lookup q-5 — should hit Hot.
    m.lookup_exact(&key_for(&text("q-5"))).unwrap();
    assert_eq!(m.last_hit_tier, Some(MemoryTier::Hot));

    // Lookup q-0 — should hit Cold (it's old, fell out of Hot and Warm).
    m.lookup_exact(&key_for(&text("q-0"))).unwrap();
    assert_eq!(m.last_hit_tier, Some(MemoryTier::Cold),
        "q-0 should be in Cold only; got {:?}", m.last_hit_tier);
}

#[test]
fn tiered_memory_promotes_on_cold_hit() {
    let t = TempPath::new("promote");
    let mut m = TieredMemory::open(2, 2, t.p()).unwrap();

    // Fill hierarchy beyond Hot+Warm capacity.
    for i in 0..6 {
        let q = text(&format!("q-{}", i));
        let a = answer(&format!("a-{}", i), TierId::L4(L4ModelId(0)), 0.5);
        m.record(&q, &a).unwrap();

    }

    // Lookup q-0 — Cold hit.
    let k0 = key_for(&text("q-0"));
    let r1 = m.lookup_exact(&k0).unwrap();
    assert!(r1.is_some());
    assert_eq!(m.last_hit_tier, Some(MemoryTier::Cold));

    // Lookup q-0 again — should now be in Hot (promoted on first access).
    let r2 = m.lookup_exact(&k0).unwrap();
    assert!(r2.is_some());
    assert_eq!(m.last_hit_tier, Some(MemoryTier::Hot),
        "after promotion q-0 should be Hot; got {:?}", m.last_hit_tier);
}

#[test]
fn tiered_memory_records_transition_joules() {
    let t = TempPath::new("joules");
    let mut m = TieredMemory::open(1, 1, t.p()).unwrap();

    // Fill enough to force transitions.
    for i in 0..3 {
        let q = text(&format!("q-{}", i));
        let a = answer(&format!("a-{}", i), TierId::L4(L4ModelId(0)), 0.5);
        m.record(&q, &a).unwrap();

    }

    // Lookup the oldest — Cold hit, promotion to Hot.
    let _ = m.lookup_exact(&key_for(&text("q-0"))).unwrap();

    // total_transition_joules should be nonzero and small.
    assert!(m.total_transition_joules > 0.0);
    assert!(m.total_transition_joules < 1e-3,
        "transition cost should be small; got {:.3e}", m.total_transition_joules);
}

#[test]
fn tiered_memory_miss_returns_none() {
    let t = TempPath::new("miss");
    let mut m = TieredMemory::open(2, 2, t.p()).unwrap();
    let result = m.lookup_exact(&key_for(&text("never recorded"))).unwrap();
    assert!(result.is_none());
    assert_eq!(m.last_hit_tier, None);
}

// ============================================================
// Runtime integration: tiered memory plugged into the cascade
// ============================================================

struct MockL4 { cost: f64, response: String }
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
            output: AnswerOutput::Text(self.response.clone()),
            tier_used: self.id(),
            joules_spent: self.cost,
            confidence: 0.9,
            trace: ExecutionTrace::default(),
            verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
        })
    }
}

#[test]
fn runtime_with_tiered_memory_hits_l0_on_repeat_query() {
    let t = TempPath::new("runtime");
    let memory = Box::new(TieredMemory::open(2, 2, t.p()).unwrap());

    let mut cascade = Cascade::new();
    cascade.register(Box::new(MockL4 {
        cost: 0.5, response: "Paris".into(),
    }));
    let mut rt = Runtime::new_with_history(cascade, memory);

    let q = text("capital of France?");
    let a1 = rt.answer(q.clone()).unwrap();
    assert_eq!(a1.tier_used, TierId::L4(L4ModelId(0)));

    let a2 = rt.answer(q).unwrap();
    assert_eq!(a2.tier_used, TierId::L0,
        "second query should hit L0 (via tiered memory)");
    assert!(a2.joules_spent < 1e-5,
        "L0 hit should be cheap; got {:.3e}", a2.joules_spent);
}

#[test]
fn runtime_query_falls_through_tiers_on_repeat() {
    // Spam more queries than fit in Hot+Warm. Old ones must be
    // recoverable from Cold via the L0 path.
    let t = TempPath::new("falls_through");
    let memory = Box::new(TieredMemory::open(2, 2, t.p()).unwrap());

    let mut cascade = Cascade::new();
    cascade.register(Box::new(MockL4 {
        cost: 0.5, response: "answer".into(),
    }));
    let mut rt = Runtime::new_with_history(cascade, memory);

    // Send 10 distinct queries; only 4 can fit in Hot+Warm.
    for i in 0..10 {
        rt.answer(text(&format!("query-{}", i))).unwrap();
    }
    // The first query should still be retrievable via Cold.
    let a = rt.answer(text("query-0")).unwrap();
    assert_eq!(a.tier_used, TierId::L0,
        "old query should still hit L0 via Cold tier; got {:?}", a.tier_used);
    assert_eq!(a.output, AnswerOutput::Text("answer".to_string()));
}

// ============================================================
// Demo
// ============================================================

#[test]
fn r8_tiered_memory_demo() {
    println!("\n=== R8: tiered memory (Hot/Warm/Cold) with joule-priced transitions ===\n");

    let t = TempPath::new("demo");
    let mut m = TieredMemory::open(2, 2, t.p()).unwrap();

    println!("Memory hierarchy: Hot=2, Warm=2, Cold=∞ (disk)\n");

    // Insert 6 items, observing the hierarchy.
    for i in 0..6 {
        let q = text(&format!("query-{}", i));
        let a = answer(&format!("answer-{}", i), TierId::L4(L4ModelId(0)), 0.5);
        m.record(&q, &a).unwrap();

    }
    let (hot, warm, cold) = m.tier_sizes();
    println!("After 6 records:");
    println!("  Hot:  {} items  (most recent)", hot);
    println!("  Warm: {} items  (recently demoted)", warm);
    println!("  Cold: {} items  (durable on disk)", cold);
    println!("  Total transition joules so far: {:.3e} J", m.total_transition_joules);

    println!("\nLookups walk Hot → Warm → Cold and promote on hit:\n");
    for i in [5, 3, 0] {
        let key = key_for(&text(&format!("query-{}", i)));
        let _ = m.lookup_exact(&key).unwrap();
        println!("  query-{}: hit {:?}", i, m.last_hit_tier.unwrap());
    }
    println!();
    println!("After looking up query-0 (Cold hit):");
    let (hot, warm, cold) = m.tier_sizes();
    println!("  Hot:  {} items   (query-0 promoted here)", hot);
    println!("  Warm: {} items", warm);
    println!("  Cold: {} items", cold);
    println!("  Total transition joules: {:.3e} J", m.total_transition_joules);

    println!("\nKey point: hot data costs ~6 nJ; cold data costs ~1 µJ;");
    println!("but a single promotion lifts a hot-accessed item back to Hot,");
    println!("so repeated access converges to Hot cost.");
}
