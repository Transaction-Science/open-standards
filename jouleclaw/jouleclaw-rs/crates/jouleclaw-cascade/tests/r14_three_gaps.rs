//! R14 integration — the three previously-empty cells filled.
//!
//! Demonstrates:
//!   * V=Delayed: an answer's verification resolves over time
//!   * E=Active: a synthesizer that initiates work without being asked
//!   * I=Bodies: a synthesizer that touches the world irreversibly
//!
//! These three axis values were unoccupied in the R10 cascade map.
//! After R14, Joule has at least one synthesizer in each.

use jouleclaw_cascade::*;
use jouleclaw_cascade::active::PeriodicTrigger;
use jouleclaw_cascade::coord::*;

// ============================================================
// V=Delayed
// ============================================================

#[test]
fn delayed_verification_lifecycle() {
    // A query "issued, awaiting outcome" — the answer is Pending,
    // the caller later resolves it.
    let mut ledger = VerificationLedger::new();
    let coord = Coord::new(
        Zone::Z2_3, Entity::Reactive, Thermo::L2_Landauer,
        Interface::Tokens, Verify::Delayed, Encoding::Facts,
    );
    let token = ledger.issue(
        TierId::L4(L4ModelId(0)), Some(coord.clone()),
        0.5, 0.5,
    );
    assert_eq!(ledger.pending_count(), 1);
    assert_eq!(ledger.issued, 1);

    // Time passes. The outcome resolves.
    let resolved = ledger.resolve(
        token,
        &VerificationOutcome::Success { actual_joules: 0.62 },
    ).unwrap();

    assert_eq!(resolved.initial_estimate, 0.5);
    assert_eq!(ledger.pending_count(), 0);
    assert_eq!(ledger.succeeded, 1);
}

#[test]
fn pending_answer_can_be_resolved_via_runtime_ledger() {
    // Token issued via the verification module; outcome lands later.
    let mut ledger = VerificationLedger::new();
    let coord = Coord::new(
        Zone::Z3, Entity::Reactive, Thermo::L2_Max,
        Interface::Tokens, Verify::Delayed, Encoding::Facts,
    );
    let token = ledger.issue(
        TierId::L4(L4ModelId(1)), Some(coord),
        0.4, 0.4,
    );

    // Hand the token off to "the caller" who reports back later.
    let outcome = VerificationOutcome::Success { actual_joules: 0.45 };
    let resolved = ledger.resolve(token, &outcome).unwrap();
    // The runtime can now feed (estimate, actual) into calibration.
    assert!((resolved.initial_estimate - 0.4).abs() < 1e-12);
}

// ============================================================
// E=Active
// ============================================================

fn active_coord() -> Coord {
    Coord::new(
        Zone::Z1, Entity::Active, Thermo::L1_Measure,
        Interface::Signals, Verify::Delayed, Encoding::None,
    )
}

#[test]
fn active_tier_initiates_emissions() {
    let mut reg = ActiveRegistry::new();
    let scripted = Box::new(PeriodicTrigger::new(
        TierId::L1(L1Primitive::Execute),
        active_coord(),
        "health check",
        2,    // emit every 2 ticks
    ));
    reg.register(scripted);

    // Tick 1: counter 1, idle. Tick 2: counter 2 ≥ 2, emit + reset.
    // Tick 3: counter 1, idle.
    let t1 = reg.tick_all();
    let t2 = reg.tick_all();
    let t3 = reg.tick_all();

    let total_emissions = t1.len() + t2.len() + t3.len();
    assert!(total_emissions >= 1,
        "active tier should emit at least once over 3 ticks");
    assert_eq!(t1.len(), 0, "tick 1 should be idle");
    assert_eq!(t2.len(), 1, "tick 2 should emit");
    assert_eq!(t3.len(), 0, "tick 3 should be idle (counter reset)");
}

#[test]
fn active_registry_reports_e_active_coord() {
    let mut reg = ActiveRegistry::new();
    reg.register(Box::new(PeriodicTrigger::new(
        TierId::L0, active_coord(), "x", 1,
    )));
    let coords = reg.tier_coords();
    assert_eq!(coords.len(), 1);
    assert_eq!(coords[0].1.entity, Entity::Active,
        "active registry must report E=Active");
}

// ============================================================
// I=Bodies
// ============================================================

fn body_coord() -> Coord {
    Coord::new(
        Zone::Z1, Entity::Reactive, Thermo::L1_Measure,
        Interface::Bodies, Verify::Delayed, Encoding::None,
    )
}

fn tmpdir() -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "joule-r14-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn body_tier_plan_does_not_touch_world() {
    let dir = tmpdir();
    let writer = Box::new(FileWriter::new(&dir,
        TierId::L1(L1Primitive::Execute), body_coord()));
    let mut dispatch = BodyDispatch::new(writer);

    let q = Query {
        input: QueryInput::Text("write payload to side-effect.txt".to_string()),
        budget: JouleBudget::standard(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    };

    let (token, answer) = dispatch.plan_for(&q).unwrap();

    // Plan returned, answer is Pending.
    assert_eq!(answer.verification, VerificationStatus::Pending(token));

    // World untouched.
    let path = dir.join("side-effect.txt");
    assert!(!path.exists(),
        "plan() must not write to disk; file exists at {:?}", path);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn body_tier_commit_touches_world_and_resolves_verification() {
    let dir = tmpdir();
    let writer = Box::new(FileWriter::new(&dir,
        TierId::L1(L1Primitive::Execute), body_coord()));
    let mut dispatch = BodyDispatch::new(writer);

    let q = Query {
        input: QueryInput::Text("write hello-world to greeting.txt".to_string()),
        budget: JouleBudget::standard(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    };

    let (token, _answer) = dispatch.plan_for(&q).unwrap();
    let actual = dispatch.commit_plan(token).unwrap();

    let path = dir.join("greeting.txt");
    assert!(path.exists());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello-world");
    assert!(actual > 0.0);
    assert_eq!(dispatch.commits_succeeded, 1);
    assert_eq!(dispatch.ledger().succeeded, 1);
    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================
// Coverage analysis — three new cells filled
// ============================================================

#[test]
fn r14_fills_three_previously_empty_cells() {
    // Collect coords from all R14-created cells.
    let delayed_coord = Coord::new(
        Zone::Z3, Entity::Reactive, Thermo::L2_Landauer,
        Interface::Tokens, Verify::Delayed, Encoding::Facts,
    );
    let active = active_coord();
    let body = body_coord();

    // Before R14, none of these cells were in the prebuilt list.
    let prebuilt_coords = [
        prebuilt::l0_cache(),
        prebuilt::l1_execute(),
        prebuilt::l1_regex(),
        prebuilt::l1_template(),
        prebuilt::l2_embedder(),
        prebuilt::l2_classifier(),
        prebuilt::l3_small_model(),
        prebuilt::l4_frontier_model(),
        prebuilt::rpc_tier(),
    ];

    // Confirm: no prebuilt has V=Delayed, E=Active, or I=Bodies.
    for c in &prebuilt_coords {
        assert_ne!(c.verify, Verify::Delayed);
        assert_ne!(c.entity, Entity::Active);
        assert_ne!(c.interface, Interface::Bodies);
    }

    // Confirm: R14 cells DO have these.
    assert_eq!(delayed_coord.verify, Verify::Delayed);
    assert_eq!(active.entity, Entity::Active);
    assert_eq!(body.interface, Interface::Bodies);

    // The three new cells have distinct IDs.
    let mut ids = std::collections::HashSet::new();
    ids.insert(delayed_coord.cell_id());
    ids.insert(active.cell_id());
    ids.insert(body.cell_id());
    assert_eq!(ids.len(), 3);
}

#[test]
fn r14_demo() {
    println!("\n=== R14: filling the three empty cells ===\n");

    // 1. V=Delayed.
    let mut ledger = VerificationLedger::new();
    let token = ledger.issue(
        TierId::L4(L4ModelId(0)),
        Some(Coord::new(
            Zone::Z3, Entity::Reactive, Thermo::L2_Max,
            Interface::Tokens, Verify::Delayed, Encoding::Facts,
        )),
        0.5, 0.5,
    );
    println!("V=Delayed:");
    println!("  issued token {:?} for L4 dispatch", token);
    println!("  pending: {}", ledger.pending_count());

    let _ = ledger.resolve(
        token,
        &VerificationOutcome::Success { actual_joules: 0.7 },
    );
    println!("  resolved: actual = 0.7 J vs estimated 0.5 J");
    println!("  ledger: {} succeeded, {} pending", ledger.succeeded, ledger.pending_count());
    println!();

    // 2. E=Active.
    println!("E=Active:");
    let mut reg = ActiveRegistry::new();
    reg.register(Box::new(PeriodicTrigger::new(
        TierId::L1(L1Primitive::Execute),
        active_coord(),
        "scheduled health check", 2,
    )));
    let mut emissions = 0;
    for tick in 1..=5 {
        let out = reg.tick_all();
        emissions += out.len();
        println!("  tick {}: {} emission(s)", tick, out.len());
    }
    println!("  total emissions across 5 ticks: {}", emissions);
    println!();

    // 3. I=Bodies.
    println!("I=Bodies:");
    let dir = tmpdir();
    let writer = Box::new(FileWriter::new(&dir,
        TierId::L1(L1Primitive::Execute), body_coord()));
    let mut dispatch = BodyDispatch::new(writer);
    let q = Query {
        input: QueryInput::Text(
            "write demonstration-of-bodies to demo.txt".into()),
        budget: JouleBudget::standard(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    };
    let (token, answer) = dispatch.plan_for(&q).unwrap();
    println!("  planned: {}", match &answer.output {
        AnswerOutput::Text(s) => s.clone(),
        _ => "(non-text)".to_string(),
    });
    println!("  verification: {:?} (token issued)", answer.verification);
    let file_exists_before = dir.join("demo.txt").exists();
    println!("  file exists before commit: {}", file_exists_before);

    let actual = dispatch.commit_plan(token).unwrap();
    let file_exists_after = dir.join("demo.txt").exists();
    println!("  committed; actual cost = {:.3e} J", actual);
    println!("  file exists after commit: {}", file_exists_after);

    let _ = std::fs::remove_dir_all(&dir);

    println!("\n— Three previously-empty cells now occupied:");
    println!("  V=Delayed:  L4 dispatches whose outcome lands over time");
    println!("  E=Active:   tiers that initiate work without being asked");
    println!("  I=Bodies:   tiers that irreversibly touch the world");
    println!();
    println!("Joule's coverage of the 8,000-cell Synthesis space:");
    println!("  pre-R14:  10 cells (cell 3756 added in R13c)");
    println!("  post-R14: 13+ cells. The active/body/delayed axes are");
    println!("            now reachable from the type system.");
}
