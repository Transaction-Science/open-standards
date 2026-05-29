//! L6 agent driver — the deferred loop, closed.
//!
//! [`jouleclaw_agent::AgentTier`] decomposes a multi-step query, dispatches
//! each sub-query through "the cascade", and composes the parts. It talks
//! to the cascade through the [`jouleclaw_agent::AgentCascade`] trait
//! precisely so it need not hold a `Runtime` (which would contain it — a
//! cycle). This module supplies the missing consumer adapter: a cascade
//! shim wired to *this stack's* resolver [`Runtime`](crate::JouleClawStack::runtime),
//! plus the piece that makes the agent pay the model tax only once.
//!
//! ## Compile once, resolve forever
//!
//! Every time the agent resolves a sub-query through the **statistical
//! compartment** (a Model/Wire-class tier), [`LearningCascade`] hands the
//! `(query, answer)` to a [`SkillInducer`] and registers the induced
//! [`Skill`] in the stack's shared skill store. The front
//! [`jouleclaw_skill::SkillTier`] then resolves the next matching query at
//! lookup energy — the model is never invoked again for that task shape.
//! A multi-step agent stops "re-paying the model tax every run" and trends
//! to L0/L1 energy with use. This is the cascade's inference-last doctrine
//! applied *over time*.
//!
//! ## Honest scope
//!
//! The default inducer ([`jouleclaw_skill::ConstantSkillInducer`]) compiles
//! an **exact-match** skill — so a *repeated* sub-query resolves free, but a
//! new instance of the same shape does not (that case is what
//! [`jouleclaw_promote`] already covers). A *generalizing* inducer — one
//! that reads the resolution and emits a parameterized [`Skill`] so a whole
//! class resolves free — is supplied by the caller via
//! [`JouleClawStack::answer_agentic_with`]. The crate ships the loop and the
//! extension point, not a synthesizer.

use jouleclaw_agent::{AgentCascade, AgentTier};
use jouleclaw_cascade::tier::{Runtime, Tier};
use jouleclaw_cascade::types::{Answer, AnswerError, AnswerOutput, JouleClass, Query, QueryInput};
use jouleclaw_promote::PromotionStore;
use jouleclaw_skill::{
    ConstantSkillInducer, InMemorySkillStore, SharedSkills, SkillInducer, SkillStore,
};

use crate::JouleClawStack;

/// Text of a query, if it carries any.
fn query_text(q: &Query) -> Option<&str> {
    match &q.input {
        QueryInput::Text(t) => Some(t.as_str()),
        QueryInput::Multimodal { text, .. } => Some(text.as_str()),
        _ => None,
    }
}

/// An [`AgentCascade`] that dispatches sub-queries through a live stack
/// [`Runtime`] and induces a skill from every model-class resolution.
///
/// The borrowed `runtime` does **not** contain the agent — the agent is a
/// transient driver built around it — so dispatching back in does not
/// re-enter the agent.
pub struct LearningCascade<'a> {
    runtime: &'a mut Runtime,
    skills: SharedSkills<InMemorySkillStore>,
    inducer: &'a dyn SkillInducer,
    /// A model answer must clear this confidence bar before it is compiled
    /// into a skill — compiling a low-confidence guess would poison the
    /// deterministic surface.
    min_confidence: f32,
}

impl<'a> LearningCascade<'a> {
    /// Wire a learning cascade to `runtime`, registering induced skills in
    /// `skills` (the same store the front `SkillTier` reads).
    pub fn new(
        runtime: &'a mut Runtime,
        skills: SharedSkills<InMemorySkillStore>,
        inducer: &'a dyn SkillInducer,
        min_confidence: f32,
    ) -> Self {
        Self {
            runtime,
            skills,
            inducer,
            min_confidence,
        }
    }
}

impl AgentCascade for LearningCascade<'_> {
    fn dispatch(&mut self, q: &Query) -> Result<Answer, AnswerError> {
        let ans = self.runtime.answer(q.clone())?;

        // Compile-once: induce only from a Model/Wire-class resolution that
        // cleared the confidence bar and produced text. A deterministic-tier
        // answer (a promotion/skill/tool hit) needs no skill — that work is
        // already at the floor.
        let is_compartment = matches!(
            ans.tier_used.joule_class(),
            JouleClass::Model | JouleClass::Wire
        );
        if is_compartment && ans.confidence >= self.min_confidence {
            if let (Some(qt), AnswerOutput::Text(answer_text)) = (query_text(q), &ans.output) {
                if let Some(skill) = self.inducer.induce(qt, answer_text, ans.tier_used) {
                    if let Ok(mut store) = self.skills.lock() {
                        store.register(skill);
                    }
                }
            }
        }
        Ok(ans)
    }
}

impl<S: PromotionStore + 'static> JouleClawStack<S> {
    /// Answer a query **agentically**: decompose it into sub-queries,
    /// resolve each through the resolver cascade, compose the parts — and
    /// compile a skill from every model-class sub-resolution so the next
    /// occurrence of that sub-task resolves at the front `SkillTier` for
    /// free. Uses the exact-match [`ConstantSkillInducer`]; see
    /// [`answer_agentic_with`](Self::answer_agentic_with) for a generalizing
    /// inducer.
    ///
    /// The induction confidence bar is [`promote_confidence`](Self::promote_confidence).
    pub fn answer_agentic(&mut self, query: Query) -> Result<Answer, AnswerError> {
        self.answer_agentic_with(query, &ConstantSkillInducer)
    }

    /// Like [`answer_agentic`](Self::answer_agentic) but with a
    /// caller-supplied [`SkillInducer`]. A generalizing inducer — one that
    /// emits a parameterized [`jouleclaw_skill::Skill`] — turns a single
    /// model resolution into free resolution of the whole query *class*,
    /// not just the exact repeat.
    pub fn answer_agentic_with(
        &mut self,
        query: Query,
        inducer: &dyn SkillInducer,
    ) -> Result<Answer, AnswerError> {
        let skills = self.skill_store.clone();
        let min_conf = self.promote_confidence;
        let budget = query.budget.hard_limit;
        let shim = LearningCascade::new(&mut self.runtime, skills, inducer, min_conf);
        let mut agent = AgentTier::new(shim);
        agent.try_answer(&query, budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::types::{
        ContextRef, JouleBudget, QualityFloor, TierId,
    };
    use jouleclaw_program::{Field, Signature};
    use jouleclaw_skill::{Procedure, Skill, Template};

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
    fn model_subresolutions_compile_to_reusable_skills() {
        // promote_confidence 0.0 → any model resolution is eligible to
        // compile, decoupling the test from the reference backend's
        // confidence value.
        let mut stack = JouleClawStack::with_defaults().with_promote_confidence(0.0);
        assert_eq!(stack.skill_store.lock().unwrap().len(), 0);

        // One agentic run: two clauses → two model sub-resolutions → two
        // skills compiled (the compile-once half of the doctrine).
        let _ = stack.answer_agentic(q("alpha and beta")).unwrap();
        assert_eq!(
            stack.skill_store.lock().unwrap().len(),
            2,
            "one skill induced per model-class sub-resolution"
        );

        // The compiled skills resolve their class deterministically. We
        // check the skill layer directly: the runtime's built-in L0 cache
        // would *also* shadow an exact repeat (a correct optimization), so
        // going straight to the store isolates the skill's contribution —
        // the resolve-forever half. (Exact-match skills overlap the cache;
        // the cache-distinct win is generalization — see the next test.)
        let mut store = stack.skill_store.lock().unwrap();
        assert!(store.resolve("alpha").is_some());
        assert!(store.resolve("beta").is_some());
        assert_eq!(store.invocations_avoided(), 2);
    }

    /// A generalizing inducer: from any "echo X" resolution it compiles a
    /// parameterized `echo {x}` skill, so a *new* instance resolves free.
    struct EchoSkillInducer;
    impl SkillInducer for EchoSkillInducer {
        fn induce(&self, query_text: &str, _answer: &str, origin: TierId) -> Option<Skill> {
            let arg = query_text.trim().strip_prefix("echo ")?;
            if arg.is_empty() {
                return None;
            }
            Some(Skill::new(
                "echo",
                Signature::new(
                    "echo",
                    "echo the argument",
                    vec![Field::text("x", "the argument")],
                    vec![Field::text("out", "the echoed argument")],
                ),
                Template::parse("echo {x}").ok()?,
                Procedure::Passthrough { hole: "x".into() },
                origin,
            ))
        }
    }

    #[test]
    fn generalizing_inducer_resolves_an_unseen_instance_for_free() {
        let mut stack = JouleClawStack::with_defaults().with_promote_confidence(0.0);

        // Solve one instance agentically; the generalizing inducer compiles
        // a parameterized skill from it.
        let _ = stack
            .answer_agentic_with(q("echo foo"), &EchoSkillInducer)
            .unwrap();
        assert_eq!(stack.skill_store.lock().unwrap().len(), 1);

        // A *different* instance of the same shape now resolves at the front
        // SkillTier — the model never sees it. Drive the runtime directly so
        // we can read the tier that closed it.
        let before = stack.skill_store.lock().unwrap().invocations_avoided();
        let ans = stack.runtime.answer(q("echo bar")).unwrap();
        match ans.output {
            AnswerOutput::Text(t) => assert_eq!(t, "bar"),
            other => panic!("expected the skill to echo `bar`, got {other:?}"),
        }
        assert_eq!(ans.tier_used, TierId::L0_5ToolCompute);
        assert!(ans.joules_spent < 1e-6, "resolved at lookup energy");
        assert_eq!(
            stack.skill_store.lock().unwrap().invocations_avoided() - before,
            1
        );
    }

    #[test]
    fn deterministic_subresolution_induces_no_skill() {
        // Pre-register a skill so the sub-query resolves deterministically
        // on the very first run; nothing new should be compiled from it.
        let mut stack = JouleClawStack::with_defaults().with_promote_confidence(0.0);
        stack.register_skill(Skill::new(
            "known",
            Signature::new(
                "known",
                "known",
                vec![Field::text("q", "q")],
                vec![Field::text("a", "a")],
            ),
            Template::literal("gamma"),
            Procedure::Constant("g".into()),
            TierId::L0_5ToolCompute,
        ));
        let start = stack.skill_store.lock().unwrap().len();
        let _ = stack.answer_agentic(q("gamma and delta")).unwrap();
        // "gamma" hit the pre-registered skill (no induction); "delta" hit
        // the model (one induction) → exactly one new skill.
        assert_eq!(stack.skill_store.lock().unwrap().len(), start + 1);
    }
}
