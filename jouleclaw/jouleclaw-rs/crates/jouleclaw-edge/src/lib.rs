//! # jouleclaw-edge
//!
//! End-to-end edge demo for JouleClaw — builds a [`Runtime`] with every
//! portable JouleClaw tier registered and runs a corpus of representative
//! queries through it. The output is the runtime's trace: which tier was
//! tried, what each tier reported, and where the cascade landed.
//!
//! This crate intentionally ships **no** lawful primitives. Consumers
//! populate the L1 [`LawfulRegistry`] at startup with their own
//! deterministic lexicon (pattern-lang's `pattern-core` is the largest
//! known consumer; OpenIE's `joule-l1` provides ~688 verified
//! primitives). When the registry is empty, L1 refuses with
//! `Inapplicable` and the cascade walks to L2.
//!
//! Ported from pattern-lang's `joule-edge` crate. The OpenIE-IP-bound
//! tier registrations (`joule_l1::LawfulTier`, `lean_bridge::LeanProofTier`,
//! `synth::SynthesisTier`, `joule_l1::PlannerTier`, etc.) were dropped
//! during the port — they re-enter at the consumer site through the
//! [`LawfulRegistry`] shim and any consumer-side tiers the consumer
//! cares to register.

use std::fmt::Write;
use std::sync::Arc;

use jouleclaw_cascade::*;
use jouleclaw_ebm::EbmTier;
use jouleclaw_liquid::{synthetic_lm, LiquidTier, LmConfig};
use jouleclaw_lmm::LmmTier;
use jouleclaw_mrl::{IdentityEmbedder, MatryoshkaEmbedder, MrlTier};
use jouleclaw_prism::{synthetic_model, ModelConfig, PrismTier};

/// Run the full edge demo. Returns the rendered trace as a string so
/// callers can print it, write it to a file, or assert against it.
///
/// Wrapped in a 32 MiB thread because some tier constructors accrete
/// large vec! literals while building default vocabulary; needs more
/// than the default 2 MiB test thread stack.
pub fn run_demo() -> String {
    let handle = std::thread::Builder::new()
        .name("jouleclaw-edge-demo".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(run_demo_inner)
        .expect("spawn demo thread");
    handle.join().expect("demo thread panicked")
}

/// Build the demo Runtime with all portable JouleClaw tiers registered
/// and an empty [`LawfulRegistry`] (consumers plug their own primitives
/// in).
pub fn build_runtime() -> Runtime {
    build_runtime_with(LawfulRegistry::new())
}

/// Build the demo Runtime with a consumer-supplied lawful registry.
/// Pattern-lang and OpenIE consumers wire their lexicons through here;
/// the registry can be empty for runtimes that ship no lawful library.
pub fn build_runtime_with(lawful: LawfulRegistry) -> Runtime {
    let mut cascade = Cascade::new();

    // L1 — lawful primitives via the JouleClaw thin shim. Consumers plug
    // their own primitives in through `LawfulRegistry`; the shim wraps
    // it as a `Tier` so the cascade walker treats it like any other tier.
    cascade.register(Box::new(LawfulRegistryTier::new(lawful)));

    // L3 — energy-based constraint satisfaction (SAT / sudoku / n-queens /
    // graph colouring). Deterministic backtracking, not a model.
    cascade.register(Box::new(EbmTier::new()));

    // L2 — Matryoshka embeddings + nearest-neighbour against a tiny
    // in-memory demo corpus.
    let mrl_embedder = MatryoshkaEmbedder::with_powers_of_two(IdentityEmbedder::new(64));
    let mut mrl_tier = MrlTier::new(1, mrl_embedder).with_top_k(3);
    for doc in [
        "JouleClaw is the energy-optimised AI runtime open standard",
        "the L0:Cache tier returns prior answers for the cost of a hash lookup",
        "the L1:Lawful tier answers deterministic queries in nanojoules",
        "ternary weights replace floating-point multiplies with conditional adds",
        "closed-form continuous-time cells skip the ODE solver",
        "Matryoshka nested embeddings let one model serve many dims",
    ] {
        mrl_tier.add_doc(doc).expect("seed corpus");
    }
    cascade.register(Box::new(mrl_tier));

    // L3 — Liquid CfC recurrent LM, the SSM-class tier. O(L) state vs
    // transformer O(L²) attention. Loaded with synthetic weights for the
    // demo; production deployments swap in trained checkpoints.
    let liquid_lm =
        synthetic_lm(LmConfig::tiny_byte(), 0x119D1D).expect("synthetic liquid lm");
    cascade.register(Box::new(
        LiquidTier::from_lm(2, liquid_lm).with_max_new_tokens(8),
    ));

    // L3 — Prism ternary transformer (BitNet-class).
    let prism_decoder =
        synthetic_model(ModelConfig::tiny_byte(), 0x5EED1).expect("synthetic ternary model");
    cascade.register(Box::new(
        PrismTier::from_decoder(3, prism_decoder).with_max_new_tokens(8),
    ));

    // L3 — LMM multimodal on Prism's TernaryDecoder backbone.
    let lmm_decoder =
        synthetic_model(ModelConfig::tiny_byte(), 0x1AAA5).expect("synthetic lmm decoder");
    cascade.register(Box::new(
        LmmTier::from_decoder(4, lmm_decoder).with_max_new_tokens(8),
    ));

    Runtime::new_without_l0(cascade)
}

/// A [`Tier`] implementation that delegates to a [`LawfulRegistry`].
/// This is the JouleClaw shim that replaces pattern-lang's
/// `joule_l1::LawfulTier`. Returns refusal when the registry is empty
/// or no primitive recognises the query.
pub struct LawfulRegistryTier {
    registry: LawfulRegistry,
    id: TierId,
}

impl LawfulRegistryTier {
    /// Wrap a registry as a cascade tier. The tier reports
    /// `TierId::L1(L1Primitive::Lawful)`.
    pub fn new(registry: LawfulRegistry) -> Self {
        Self {
            registry,
            id: TierId::L1(L1Primitive::Lawful),
        }
    }
}

impl Tier for LawfulRegistryTier {
    fn id(&self) -> TierId {
        self.id
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        // Only Text queries can hit a lawful primitive; multimodal /
        // image / audio inputs skip straight past.
        match &q.input {
            QueryInput::Text(_) => Some(TierEstimate {
                joules: 1e-9, // nanojoule class
                latency: std::time::Duration::from_micros(10),
                confidence_floor: if self.registry.is_empty() { 0.0 } else { 0.8 },
            }),
            _ => None,
        }
    }

    fn try_answer(&mut self, q: &Query, _budget_remaining: f64) -> Result<Answer, AnswerError> {
        let text = match &q.input {
            QueryInput::Text(s) => s,
            _ => {
                return Ok(Answer {
                    output: AnswerOutput::Refused(RefusalReason::Inapplicable),
                    tier_used: self.id,
                    joules_spent: 0.0,
                    confidence: 0.0,
                    trace: ExecutionTrace::default(),
                    verification: crate::VerificationStatus::Resolved,
                });
            }
        };
        if self.registry.is_empty() {
            return Ok(Answer {
                output: AnswerOutput::Refused(RefusalReason::TierSpecific(
                    "lawful registry is empty — consumer did not register any primitives".into(),
                )),
                tier_used: self.id,
                joules_spent: 0.0,
                confidence: 0.0,
                trace: ExecutionTrace::default(),
                verification: crate::VerificationStatus::Resolved,
            });
        }
        match self.registry.try_resolve(text) {
            Some(hit) => Ok(Answer {
                output: AnswerOutput::Text(hit.answer),
                tier_used: self.id,
                joules_spent: (hit.declared_cost_uj as f64) / 1e6,
                confidence: 1.0,
                trace: ExecutionTrace::default(),
                verification: crate::VerificationStatus::Resolved,
            }),
            None => Ok(Answer {
                output: AnswerOutput::Refused(RefusalReason::TierSpecific(format!(
                    "no lawful primitive recognised the query ({} primitives in registry)",
                    self.registry.len()
                ))),
                tier_used: self.id,
                joules_spent: 0.0,
                confidence: 0.0,
                trace: ExecutionTrace::default(),
                verification: crate::VerificationStatus::Resolved,
            }),
        }
    }
}

fn run_demo_inner() -> String {
    let mut buf = String::new();
    writeln!(buf, "JouleClaw edge demo").unwrap();
    writeln!(buf, "===================").unwrap();
    writeln!(buf).unwrap();
    writeln!(buf, "Building the cascade:").unwrap();
    writeln!(buf, "  L1::Lawful   - LawfulRegistry (empty by default — consumer plugs in)")
        .unwrap();
    writeln!(buf, "  L3::Ebm      - constraint satisfaction (SAT / sudoku / n-queens)")
        .unwrap();
    writeln!(buf, "  L2::MRL      - Matryoshka embeddings + nearest-neighbor").unwrap();
    writeln!(buf, "  L3::Liquid   - CfC recurrent LM (SSM-class)").unwrap();
    writeln!(buf, "  L3::Prism    - ternary transformer (BitNet-class)").unwrap();
    writeln!(buf, "  L3::LMM      - vision/text/audio multimodal").unwrap();
    writeln!(buf).unwrap();

    let mut rt = build_runtime();

    run_one(&mut buf, &mut rt, "compute gcd 12 8", text_query("compute gcd 12 8"));
    run_one(
        &mut buf, &mut rt, "what's the weather today?",
        text_query("what's the weather today?"),
    );
    run_one(
        &mut buf, &mut rt, "describe this image",
        multimodal_query("describe this image", vec![vec![0u8; 1024]]),
    );
    run_one(&mut buf, &mut rt, "[opaque image bytes]", image_query(vec![0u8; 4096]));
    run_one(&mut buf, &mut rt, "nqueens 8", text_query("nqueens 8"));
    run_one(
        &mut buf, &mut rt, "sat 1 2 ; -1 3 ; -2 -3",
        text_query("sat 1 2 ; -1 3 ; -2 -3"),
    );

    buf
}

fn run_one(buf: &mut String, rt: &mut Runtime, label: &str, q: Query) {
    writeln!(buf, "[{label}]").unwrap();
    match rt.answer(q) {
        Ok(ans) => {
            for entry in &ans.trace.attempts {
                writeln!(
                    buf, "    {:<28}  {}  ({:.3e} J)",
                    format!("{:?}", entry.tier), fmt_outcome(&entry.outcome), entry.joules
                ).unwrap();
            }
            writeln!(
                buf, "    >> RESULT: {}  (total {:.3e} J)",
                fmt_output(&ans.output), ans.joules_spent
            ).unwrap();
        }
        Err(AnswerError::NoTierSatisfied { refusals }) => {
            for (tid, reason) in &refusals {
                writeln!(
                    buf, "    {:<28}  Refused: {}",
                    format!("{:?}", tid), fmt_reason(reason)
                ).unwrap();
            }
            writeln!(buf, "    >> RESULT: no tier satisfied").unwrap();
        }
        Err(e) => {
            writeln!(buf, "    >> RUNTIME ERROR: {e:?}").unwrap();
        }
    }
    writeln!(buf).unwrap();
}

fn fmt_outcome(o: &TraceOutcome) -> String {
    match o {
        TraceOutcome::Hit => "Hit".into(),
        TraceOutcome::Refused(r) => format!("Refused: {}", fmt_reason(r)),
        TraceOutcome::SkippedBudget => "Skipped (budget)".into(),
        TraceOutcome::SkippedQuality => "Skipped (quality)".into(),
        TraceOutcome::SkippedInapplicable => "Skipped (inapplicable)".into(),
    }
}

fn fmt_reason(r: &RefusalReason) -> String {
    match r {
        RefusalReason::Inapplicable => "inapplicable".into(),
        RefusalReason::LowConfidence(c) => format!("low confidence ({c})"),
        RefusalReason::TierSpecific(m) => truncate(m, 160),
    }
}

fn fmt_output(out: &AnswerOutput) -> String {
    match out {
        AnswerOutput::Text(s) => format!("Text({s:?})"),
        AnswerOutput::Structured(b) => format!("Structured({} bytes)", b.len()),
        AnswerOutput::Refused(r) => format!("Refused: {}", fmt_reason(r)),
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() } else {
        let head: String = s.chars().take(max - 1).collect();
        format!("{head}...")
    }
}

fn text_query(s: &str) -> Query {
    Query {
        input: QueryInput::Text(s.to_string()),
        budget: JouleBudget::expensive(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    }
}

fn multimodal_query(text: &str, images: Vec<Vec<u8>>) -> Query {
    Query {
        input: QueryInput::Multimodal { text: text.to_string(), images, audio: vec![] },
        budget: JouleBudget::expensive(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    }
}

fn image_query(bytes: Vec<u8>) -> Query {
    Query {
        input: QueryInput::Image(bytes),
        budget: JouleBudget::expensive(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy lawful primitive — recognises `gcd <a> <b>` syntax.
    struct GcdPrimitive;
    impl LawfulPrimitive for GcdPrimitive {
        fn id(&self) -> &str { "lawful:demo:gcd" }
        fn try_resolve(&self, q: &str) -> Option<String> {
            let parts: Vec<&str> = q.trim().split_whitespace().collect();
            if parts.len() != 3 || (parts[0] != "gcd" && parts[0] != "compute") {
                return None;
            }
            let rest = if parts[0] == "compute" { &parts[1..] } else { &parts[1..] };
            let (a, b): (u64, u64) = match (rest.first()?.parse(), rest.get(1)?.parse()) {
                (Ok(a), Ok(b)) => (a, b),
                _ => return None,
            };
            let (mut x, mut y) = (a, b);
            while y != 0 { let t = y; y = x % y; x = t; }
            Some(x.to_string())
        }
    }

    #[test]
    fn empty_registry_refuses_gracefully() {
        let mut rt = build_runtime();
        let ans = rt.answer(text_query("gcd 12 8"));
        // With an empty registry, L1 refuses; the model tiers will pick
        // it up. Either way, the run does not panic.
        let _ = ans;
    }

    #[test]
    fn populated_registry_answers_at_l1() {
        let reg = LawfulRegistry::new().register(Arc::new(GcdPrimitive));
        let mut rt = build_runtime_with(reg);
        let ans = rt.answer(text_query("gcd 12 8")).expect("answer");
        // L1 should win because it's the cheapest tier with a hit.
        if let AnswerOutput::Text(t) = ans.output {
            assert_eq!(t, "4");
        } else {
            panic!("expected Text answer, got {:?}", ans.output);
        }
    }

    #[test]
    fn demo_runs_to_completion() {
        let trace = run_demo();
        assert!(trace.contains("JouleClaw edge demo"));
        assert!(trace.contains("L1::Lawful"));
        assert!(trace.contains("L3::LMM"));
    }
}
