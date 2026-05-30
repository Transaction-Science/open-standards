//! Skill-compilation — the generalizing cousin of yang→yin promotion.
//!
//! [`jouleclaw-promote`] caches an *exact* verified answer: the same input
//! bytes resolve deterministically forever. A **skill** goes further — it
//! captures the *shape* of a resolution as a typed, parameterized,
//! deterministic procedure, so an entire **class** of queries resolves
//! without the model. This is the Hermes "skill from experience" idea
//! made deterministic: the model figures a task out once; the skill
//! resolves every future instance of that task at lookup energy.
//!
//! A [`Skill`] is three things:
//! 1. a [`Template`] that recognizes the query class (`"greet {name}"`),
//! 2. a [`jouleclaw_program::Signature`] that types its inputs/output,
//! 3. a deterministic [`Procedure`] that computes the output from the
//!    bound template holes.
//!
//! A [`SkillTier`] registered at the front of the cascade resolves any
//! matching query by binding holes and running the procedure — no model,
//! no per-instance cost. The energy/determinism win over every agentic
//! framework: recurring *task shapes* converge to deterministic procedures
//! instead of re-invoking inference per instance.
//!
//! ## Scope (v1, honest)
//!
//! This crate ships the **mechanism**: typed parameterized deterministic
//! skills as a cascade tier + registry, and a [`SkillInducer`] trait as
//! the compile-from-experience extension point. Parameterized skills are
//! authored via [`Skill::new`]; the trivial [`ConstantSkillInducer`]
//! compiles an exact-match skill from one resolution (degenerate — it
//! equals promotion). Automatic *generalizing* induction from execution
//! traces (Hermes/GEPA: read a trace, infer the template + procedure,
//! propose the skill) is the named extension point — implement
//! [`SkillInducer`] — not a claim this crate already does it.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, Query, QueryInput, RefusalReason, TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;
use jouleclaw_program::Signature;
use regex::Regex;

/// Energy charged for a skill resolution: a template match plus a
/// string substitution — tens of nanojoules. Compare with the
/// megajoule-class model call it replaces, per query instance.
pub const SKILL_JOULES: f64 = 20e-9;

/// Errors building or compiling skills.
#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    #[error("invalid template: {0}")]
    BadTemplate(String),
    #[error("template regex failed: {0}")]
    Regex(String),
}

// ─────────────────────────────────────────────────────────────────────
// Template
// ─────────────────────────────────────────────────────────────────────

/// A query-class matcher: literal text with `{named}` holes. Compiles to
/// an anchored regex with one non-greedy capture per hole, so
/// `"convert {n} celsius"` binds `n` from `"convert 20 celsius"`.
#[derive(Debug, Clone)]
pub struct Template {
    regex: Regex,
    holes: Vec<String>,
    pattern: String,
}

impl Template {
    /// Parse a `{hole}` template. Hole names must be non-empty
    /// alphanumeric/underscore and unique.
    pub fn parse(pattern: &str) -> Result<Self, SkillError> {
        let mut rx = String::from("^");
        let mut holes: Vec<String> = Vec::new();
        let mut lit = String::new();
        let mut chars = pattern.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '{' {
                rx.push_str(&regex::escape(&lit));
                lit.clear();
                let mut name = String::new();
                let mut closed = false;
                for n in chars.by_ref() {
                    if n == '}' {
                        closed = true;
                        break;
                    }
                    name.push(n);
                }
                if !closed {
                    return Err(SkillError::BadTemplate(format!(
                        "unclosed hole in `{pattern}`"
                    )));
                }
                if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    return Err(SkillError::BadTemplate(format!(
                        "invalid hole name `{name}` in `{pattern}`"
                    )));
                }
                if holes.contains(&name) {
                    return Err(SkillError::BadTemplate(format!(
                        "duplicate hole `{name}` in `{pattern}`"
                    )));
                }
                rx.push_str(&format!("(?P<{name}>.+?)"));
                holes.push(name);
            } else {
                lit.push(c);
            }
        }
        rx.push_str(&regex::escape(&lit));
        rx.push('$');

        let regex = Regex::new(&rx).map_err(|e| SkillError::Regex(e.to_string()))?;
        Ok(Self {
            regex,
            holes,
            pattern: pattern.to_string(),
        })
    }

    /// A template that matches `text` exactly, with no holes (the
    /// constant / exact-match case — used by [`ConstantSkillInducer`]).
    pub fn literal(text: &str) -> Self {
        let rx = format!("^{}$", regex::escape(text.trim()));
        // A literal pattern is always valid regex.
        let regex = Regex::new(&rx).unwrap_or_else(|_| Regex::new("^$").expect("trivial regex"));
        Self {
            regex,
            holes: Vec::new(),
            pattern: text.to_string(),
        }
    }

    /// Try to match `text`; on success, bind each hole to its captured
    /// (trimmed) value.
    pub fn match_text(&self, text: &str) -> Option<HashMap<String, String>> {
        let caps = self.regex.captures(text.trim())?;
        let mut bindings = HashMap::with_capacity(self.holes.len());
        for h in &self.holes {
            bindings.insert(h.clone(), caps.name(h)?.as_str().trim().to_string());
        }
        Some(bindings)
    }

    pub fn holes(&self) -> &[String] {
        &self.holes
    }

    pub fn pattern(&self) -> &str {
        &self.pattern
    }
}

// ─────────────────────────────────────────────────────────────────────
// Procedure
// ─────────────────────────────────────────────────────────────────────

/// A deterministic computation from bound holes to an output string.
///
/// v1 ships string-level transforms. Compute-backed procedures (binding
/// holes into a deterministic tool — e.g. arithmetic, unit conversion via
/// `jouleclaw-tools`) are the obvious extension; they slot in as a new
/// variant without changing the [`Skill`] / [`SkillTier`] surface.
#[derive(Debug, Clone)]
pub enum Procedure {
    /// Always return this fixed output (the exact-match / cache case).
    Constant(String),
    /// Return the named hole's bound value verbatim.
    Passthrough { hole: String },
    /// A format string with `{hole}` placeholders substituted from the
    /// bindings. Unknown placeholders resolve to empty.
    Format(String),
}

impl Procedure {
    /// Run the procedure against the bound holes. Returns `None` only if a
    /// `Passthrough` references a hole that was not bound.
    pub fn run(&self, bindings: &HashMap<String, String>) -> Option<String> {
        match self {
            Procedure::Constant(s) => Some(s.clone()),
            Procedure::Passthrough { hole } => bindings.get(hole).cloned(),
            Procedure::Format(fmt) => Some(substitute(fmt, bindings)),
        }
    }
}

/// Replace `{key}` occurrences in `fmt` with `bindings[key]`; unknown
/// keys collapse to empty. Single-pass, no nested expansion.
fn substitute(fmt: &str, bindings: &HashMap<String, String>) -> String {
    let mut out = String::with_capacity(fmt.len());
    let mut chars = fmt.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' {
            let mut key = String::new();
            let mut closed = false;
            for n in chars.by_ref() {
                if n == '}' {
                    closed = true;
                    break;
                }
                key.push(n);
            }
            if closed {
                if let Some(v) = bindings.get(&key) {
                    out.push_str(v);
                }
                // unknown key → empty
            } else {
                out.push('{');
                out.push_str(&key);
            }
        } else {
            out.push(c);
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────
// Skill
// ─────────────────────────────────────────────────────────────────────

/// A compiled skill: recognize a query class, type it, resolve it
/// deterministically.
pub struct Skill {
    /// Stable name (for traces / dedup).
    pub name: String,
    /// Typed input/output contract (`jouleclaw-program` signature).
    pub signature: Signature,
    /// Query-class matcher.
    pub template: Template,
    /// Deterministic output computation.
    pub procedure: Procedure,
    /// The model tier whose verified resolution this skill was compiled
    /// from (provenance).
    pub origin_tier: TierId,
}

impl Skill {
    pub fn new(
        name: impl Into<String>,
        signature: Signature,
        template: Template,
        procedure: Procedure,
        origin_tier: TierId,
    ) -> Self {
        Self {
            name: name.into(),
            signature,
            template,
            procedure,
            origin_tier,
        }
    }

    /// Resolve `query_text` if it matches this skill's template.
    pub fn resolve(&self, query_text: &str) -> Option<String> {
        let bindings = self.template.match_text(query_text)?;
        self.procedure.run(&bindings)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Store
// ─────────────────────────────────────────────────────────────────────

/// The result of resolving a query against the skill registry.
#[derive(Debug, Clone)]
pub struct SkillHit {
    pub answer: String,
    pub skill_name: String,
}

/// A registry of compiled skills.
pub trait SkillStore: Send {
    /// Register a skill (later registrations match after earlier ones).
    fn register(&mut self, skill: Skill);
    /// Resolve a query against the registered skills, first match wins;
    /// counts the reuse (one model invocation avoided).
    fn resolve(&mut self, query_text: &str) -> Option<SkillHit>;
    /// Number of skills registered.
    fn len(&self) -> usize;
    /// Total model invocations avoided across all skills (sum of reuses).
    fn invocations_avoided(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// In-memory skill registry.
#[derive(Default)]
pub struct InMemorySkillStore {
    skills: Vec<Skill>,
    uses: u64,
}

impl InMemorySkillStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SkillStore for InMemorySkillStore {
    fn register(&mut self, skill: Skill) {
        self.skills.push(skill);
    }

    fn resolve(&mut self, query_text: &str) -> Option<SkillHit> {
        for skill in &self.skills {
            if let Some(answer) = skill.resolve(query_text) {
                self.uses += 1;
                return Some(SkillHit {
                    answer,
                    skill_name: skill.name.clone(),
                });
            }
        }
        None
    }

    fn len(&self) -> usize {
        self.skills.len()
    }

    fn invocations_avoided(&self) -> u64 {
        self.uses
    }
}

/// Shared skill store, held by the [`SkillTier`] (and any compiler).
pub type SharedSkills<S> = Arc<Mutex<S>>;

/// A fresh shared in-memory skill store.
pub fn shared_in_memory() -> SharedSkills<InMemorySkillStore> {
    Arc::new(Mutex::new(InMemorySkillStore::new()))
}

// ─────────────────────────────────────────────────────────────────────
// Tier
// ─────────────────────────────────────────────────────────────────────

/// A deterministic cascade tier that resolves any query matching a
/// registered skill — at lookup energy, never touching the model.
pub struct SkillTier<S: SkillStore> {
    store: SharedSkills<S>,
}

impl<S: SkillStore> SkillTier<S> {
    pub fn new(store: SharedSkills<S>) -> Self {
        Self { store }
    }
}

impl<S: SkillStore + 'static> Tier for SkillTier<S> {
    fn id(&self) -> TierId {
        // Skills are deterministic compiled procedures — the tool/compute class.
        TierId::L0_5ToolCompute
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        match &q.input {
            QueryInput::Text(_) | QueryInput::Multimodal { .. } => Some(TierEstimate {
                joules: SKILL_JOULES,
                latency: Duration::from_micros(1),
                confidence_floor: 0.99,
            }),
            _ => None,
        }
    }

    fn try_answer(&mut self, q: &Query, _budget_remaining: f64) -> Result<Answer, AnswerError> {
        let text = match &q.input {
            QueryInput::Text(t) => t.as_str(),
            QueryInput::Multimodal { text, .. } => text.as_str(),
            _ => return Ok(refused()),
        };
        let hit = self
            .store
            .lock()
            .map_err(|e| AnswerError::TierFailed {
                tier: TierId::L0_5ToolCompute,
                cause: format!("skill store lock poisoned: {e}"),
            })?
            .resolve(text);
        match hit {
            Some(h) => Ok(Answer {
                output: AnswerOutput::Text(h.answer),
                tier_used: TierId::L0_5ToolCompute,
                joules_spent: SKILL_JOULES,
                confidence: 1.0,
                trace: ExecutionTrace::default(),
                verification: VerificationStatus::Resolved,
            }),
            None => Ok(refused()),
        }
    }
}

fn refused() -> Answer {
    Answer {
        output: AnswerOutput::Refused(RefusalReason::Inapplicable),
        tier_used: TierId::L0_5ToolCompute,
        joules_spent: SKILL_JOULES,
        confidence: 0.0,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Induction (compile-from-experience extension point)
// ─────────────────────────────────────────────────────────────────────

/// Compiles a [`Skill`] from a verified resolution. The general,
/// *generalizing* inducer (Hermes/GEPA: read an execution trace, infer
/// the template + procedure that reproduces the result over a class) is
/// a hard synthesis problem and is intentionally left as this trait —
/// implement it with your own synthesizer. The crate ships only the
/// trivial reference below.
pub trait SkillInducer: Send + Sync {
    fn induce(&self, query_text: &str, answer_text: &str, origin: TierId) -> Option<Skill>;
}

/// Trivial reference inducer: compiles an **exact-match constant** skill
/// (the query as a literal template, the answer as a constant). This is
/// degenerate — it generalizes to nothing and equals what
/// `jouleclaw-promote` already does — but it shows the [`SkillInducer`]
/// contract end-to-end. Real value comes from parameterized skills
/// authored via [`Skill::new`] or a generalizing inducer.
pub struct ConstantSkillInducer;

impl SkillInducer for ConstantSkillInducer {
    fn induce(&self, query_text: &str, answer_text: &str, origin: TierId) -> Option<Skill> {
        if query_text.trim().is_empty() {
            return None;
        }
        let signature = Signature::new(
            "constant",
            "exact-match constant skill",
            vec![jouleclaw_program::Field::text("query", "the exact query")],
            vec![jouleclaw_program::Field::text("answer", "the constant answer")],
        );
        Some(Skill::new(
            format!("const:{:.32}", query_text.trim()),
            signature,
            Template::literal(query_text),
            Procedure::Constant(answer_text.to_string()),
            origin,
        ))
    }
}

/// A **generalizing** inducer, guided by a library of query-class templates.
///
/// Where [`ConstantSkillInducer`] freezes one exact answer (and so is
/// shadowed by an L0 cache for repeats), `PatternInducer` compiles a
/// *parameterized* skill from a single resolution, so the whole class
/// resolves without the model — including instances never seen before.
///
/// ## How it generalizes from one example
///
/// The caller supplies the query *shapes* worth compiling (e.g.
/// `"greet {name}"`, `"convert {n} celsius"`). When a resolved query matches
/// a shape, the inducer binds the holes and then **rewrites the answer**,
/// replacing each bound hole value with its `{hole}` placeholder, to
/// synthesize a [`Procedure::Format`]. For `"greet Alice" → "Hello, Alice!"`
/// under shape `"greet {name}"`, it emits `Format("Hello, {name}!")`, which
/// resolves `"greet Bob" → "Hello, Bob!"`.
///
/// ## Honest scope and safety
///
/// This is sound exactly when the answer is a literal function of the holes
/// (greetings, labels, formatted lookups). It is *conservative*: if any
/// hole's bound value does not appear verbatim in the answer, the inducer
/// declines that shape (returns nothing for it) rather than emit a skill
/// that would drop the parameter and answer constantly across the class.
/// Because a wrong generalization would poison the deterministic surface,
/// induce only from **verified** resolutions (the agent gates on
/// confidence; pair with a verifier for high-stakes use). This is not a
/// general program synthesizer — it is template-guided substitution
/// induction, which is the honest, useful middle between exact-match and
/// full trace synthesis.
pub struct PatternInducer {
    shapes: Vec<Template>,
}

impl PatternInducer {
    /// Build an inducer over a library of query-class templates, tried in
    /// order (first faithful match wins).
    pub fn new(shapes: Vec<Template>) -> Self {
        Self { shapes }
    }

    /// Convenience: parse `{hole}` patterns into templates, skipping any
    /// that fail to parse.
    pub fn from_patterns<I, P>(patterns: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<str>,
    {
        let shapes = patterns
            .into_iter()
            .filter_map(|p| Template::parse(p.as_ref()).ok())
            .collect();
        Self { shapes }
    }
}

impl SkillInducer for PatternInducer {
    fn induce(&self, query_text: &str, answer_text: &str, origin: TierId) -> Option<Skill> {
        for shape in &self.shapes {
            let Some(bindings) = shape.match_text(query_text) else {
                continue;
            };
            // Rewrite the answer into a format string: each bound hole value
            // becomes its `{hole}` placeholder. Decline (try the next shape)
            // unless every hole's value is present, so the procedure
            // faithfully parameterizes the class.
            let mut fmt = answer_text.to_string();
            let mut faithful = true;
            for hole in shape.holes() {
                let val = &bindings[hole];
                if val.is_empty() || !fmt.contains(val.as_str()) {
                    faithful = false;
                    break;
                }
                fmt = fmt.replace(val.as_str(), &format!("{{{hole}}}"));
            }
            if !faithful {
                continue;
            }
            let inputs: Vec<_> = shape
                .holes()
                .iter()
                .map(|h| jouleclaw_program::Field::text(h.clone(), "bound template hole"))
                .collect();
            let signature = Signature::new(
                "pattern",
                "parameterized skill induced from one resolution",
                inputs,
                vec![jouleclaw_program::Field::text(
                    "output",
                    "the formatted output",
                )],
            );
            return Some(Skill::new(
                format!("pattern:{}", shape.pattern()),
                signature,
                shape.clone(),
                Procedure::Format(fmt),
                origin,
            ));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::types::{
        ContextRef, JouleBudget, L3ModelId, QualityFloor,
    };
    use jouleclaw_program::Field;

    fn sig() -> Signature {
        Signature::new(
            "greet",
            "greet a person by name",
            vec![Field::text("name", "person's name")],
            vec![Field::text("greeting", "the greeting")],
        )
    }

    fn q(text: &str) -> Query {
        Query {
            input: QueryInput::Text(text.to_string()),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn template_binds_a_single_hole() {
        let t = Template::parse("greet {name}").unwrap();
        let b = t.match_text("greet Alice").unwrap();
        assert_eq!(b.get("name").unwrap(), "Alice");
        assert!(t.match_text("farewell Alice").is_none());
    }

    #[test]
    fn template_binds_multiple_holes() {
        let t = Template::parse("{a} plus {b}").unwrap();
        let bnd = t.match_text("2 plus 3").unwrap();
        assert_eq!(bnd.get("a").unwrap(), "2");
        assert_eq!(bnd.get("b").unwrap(), "3");
    }

    #[test]
    fn template_rejects_bad_holes() {
        assert!(Template::parse("greet {na me}").is_err()); // space in hole
        assert!(Template::parse("greet {name").is_err()); // unclosed
        assert!(Template::parse("{x} and {x}").is_err()); // duplicate
    }

    #[test]
    fn format_procedure_substitutes() {
        let p = Procedure::Format("Hello, {name}!".into());
        let mut b = HashMap::new();
        b.insert("name".to_string(), "Alice".to_string());
        assert_eq!(p.run(&b).unwrap(), "Hello, Alice!");
    }

    #[test]
    fn passthrough_and_constant() {
        let mut b = HashMap::new();
        b.insert("x".to_string(), "val".to_string());
        assert_eq!(Procedure::Passthrough { hole: "x".into() }.run(&b).unwrap(), "val");
        assert_eq!(Procedure::Constant("fixed".into()).run(&b).unwrap(), "fixed");
        assert!(Procedure::Passthrough { hole: "missing".into() }.run(&b).is_none());
    }

    #[test]
    fn skill_generalizes_over_a_class() {
        // ONE skill resolves MANY inputs — the whole point.
        let skill = Skill::new(
            "greet",
            sig(),
            Template::parse("greet {name}").unwrap(),
            Procedure::Format("Hello, {name}!".into()),
            TierId::L3(L3ModelId(0)),
        );
        assert_eq!(skill.resolve("greet Alice").unwrap(), "Hello, Alice!");
        assert_eq!(skill.resolve("greet Bob").unwrap(), "Hello, Bob!");
        assert_eq!(skill.resolve("greet the whole team").unwrap(), "Hello, the whole team!");
        assert!(skill.resolve("dismiss Alice").is_none());
    }

    #[test]
    fn store_resolves_and_counts() {
        let mut store = InMemorySkillStore::new();
        store.register(Skill::new(
            "greet",
            sig(),
            Template::parse("greet {name}").unwrap(),
            Procedure::Format("Hello, {name}!".into()),
            TierId::L3(L3ModelId(0)),
        ));
        assert_eq!(store.resolve("greet Alice").unwrap().answer, "Hello, Alice!");
        assert_eq!(store.resolve("greet Bob").unwrap().answer, "Hello, Bob!");
        assert!(store.resolve("unrelated").is_none());
        assert_eq!(store.invocations_avoided(), 2);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn skill_tier_serves_class_at_lookup_energy() {
        let store = shared_in_memory();
        store.lock().unwrap().register(Skill::new(
            "greet",
            sig(),
            Template::parse("greet {name}").unwrap(),
            Procedure::Format("Hello, {name}!".into()),
            TierId::L3(L3ModelId(0)),
        ));
        let mut tier = SkillTier::new(store);
        let ans = tier.try_answer(&q("greet Carol"), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Text(t) => assert_eq!(t, "Hello, Carol!"),
            other => panic!("expected skill answer, got {other:?}"),
        }
        assert_eq!(ans.tier_used, TierId::L0_5ToolCompute);
        assert!(ans.joules_spent < 1e-6); // nanojoules
        // A query the skill doesn't recognize is refused (cascade continues).
        let miss = tier.try_answer(&q("compute pi"), 1.0).unwrap();
        assert!(matches!(miss.output, AnswerOutput::Refused(_)));
    }

    #[test]
    fn skill_tier_via_cascade_runtime() {
        use jouleclaw_cascade::tier::{Cascade, Runtime};
        let store = shared_in_memory();
        store.lock().unwrap().register(Skill::new(
            "echo_topic",
            sig(),
            Template::parse("define {term}").unwrap(),
            Procedure::Format("{term} is defined deterministically".into()),
            TierId::L3(L3ModelId(0)),
        ));
        let mut cascade = Cascade::new();
        cascade.register(Box::new(SkillTier::new(store)));
        let mut rt = Runtime::new_without_l0(cascade);
        let ans = rt.answer(q("define entropy")).unwrap();
        match ans.output {
            AnswerOutput::Text(t) => assert_eq!(t, "entropy is defined deterministically"),
            other => panic!("expected skill answer, got {other:?}"),
        }
        assert_eq!(ans.tier_used, TierId::L0_5ToolCompute);
    }

    #[test]
    fn constant_inducer_compiles_exact_match() {
        let inducer = ConstantSkillInducer;
        let skill = inducer
            .induce("what is the meaning of life", "42", TierId::L3(L3ModelId(0)))
            .unwrap();
        assert_eq!(skill.resolve("what is the meaning of life").unwrap(), "42");
        // Exact-match only — does not generalize.
        assert!(skill.resolve("what is the meaning of death").is_none());
        assert!(inducer.induce("   ", "x", TierId::L3(L3ModelId(0))).is_none());
    }

    #[test]
    fn pattern_inducer_generalizes_from_one_example() {
        // From ONE resolution, compile a skill that resolves the whole class.
        let inducer = PatternInducer::from_patterns(["greet {name}"]);
        let skill = inducer
            .induce("greet Alice", "Hello, Alice!", TierId::L3(L3ModelId(0)))
            .expect("matches the shape and generalizes faithfully");
        // Instances never seen during induction now resolve deterministically.
        assert_eq!(skill.resolve("greet Bob").unwrap(), "Hello, Bob!");
        assert_eq!(skill.resolve("greet the whole team").unwrap(), "Hello, the whole team!");
        assert!(skill.resolve("dismiss Alice").is_none());
    }

    #[test]
    fn pattern_inducer_handles_multiple_holes() {
        let inducer = PatternInducer::from_patterns(["{a} plus {b}"]);
        let skill = inducer
            .induce("2 plus 3", "2 + 3 = 5", TierId::L3(L3ModelId(0)))
            .unwrap();
        // Both holes parameterize; the literal "= 5" is class-constant here
        // (single-example limit), but the holes track.
        assert_eq!(skill.resolve("7 plus 8").unwrap(), "7 + 8 = 5");
    }

    #[test]
    fn pattern_inducer_declines_when_hole_value_absent_from_answer() {
        // The answer does not contain the bound value "Alice", so a Format
        // skill would drop the parameter — the inducer conservatively
        // declines rather than emit a constant-answer skill for the class.
        let inducer = PatternInducer::from_patterns(["greet {name}"]);
        assert!(inducer
            .induce("greet Alice", "Greetings, friend!", TierId::L3(L3ModelId(0)))
            .is_none());
    }

    #[test]
    fn pattern_inducer_returns_none_when_no_shape_matches() {
        let inducer = PatternInducer::from_patterns(["convert {n} celsius"]);
        assert!(inducer
            .induce("what is the capital of france", "Paris", TierId::L3(L3ModelId(0)))
            .is_none());
    }
}
