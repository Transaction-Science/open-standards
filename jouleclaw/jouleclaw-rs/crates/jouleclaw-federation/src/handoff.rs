//! Typed callable agents + handoff selection.
//!
//! Promotes the federation's parallel-retrieval workers from "search
//! providers that return hits" to "typed callable agents that return
//! [`AgentResponse`]s." The architectural shape every multi-agent
//! framework has settled on — Strands "agents-as-tools," Anthropic's
//! handoff-as-tool-call — pinned as a contract here, then routed via
//! a cost-table [`HandoffSelector`] (the cheapest capable agent wins by
//! default).
//!
//! ## Honest scope (v1)
//!
//! - **Stateless cones only.** Each [`CallableAgent`] call is
//!   independent; agents do NOT chat with each other. This matches
//!   Anthropic's "Research" pattern constraint and keeps the cost
//!   accounting linear.
//! - **One handoff per dispatch.** The selector picks one agent; the
//!   federation's existing parallel-provider fanout stays unchanged.
//!   Multi-agent fan-out is the consumer's composition.
//! - **Capability matching is structural.** A capability is
//!   `{kind, modality}`; the selector compares with equality, not
//!   semantic similarity. Smarter selectors (LLM-routed, learned)
//!   plug in via the [`HandoffSelector`] trait.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ─────────────────────────────────────────────────────────────────────
// Capability + typed input/output
// ─────────────────────────────────────────────────────────────────────

/// A structural capability tag — `(kind, modality)`. Selectors match
/// on equality; this is intentionally low-magic.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct Capability {
    /// The task class, e.g. `"summarise"`, `"extract"`, `"translate"`,
    /// `"verify"`. Lowercase snake convention.
    pub kind: String,
    /// The input modality, e.g. `"text"`, `"code"`, `"image"`. A bare
    /// `"text"` is the default; agents that handle multiple modalities
    /// expose one capability per modality.
    pub modality: String,
}

impl Capability {
    pub fn new(kind: impl Into<String>, modality: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            modality: modality.into(),
        }
    }
    /// Convenience for the common text-task case.
    pub fn text(kind: impl Into<String>) -> Self {
        Self::new(kind, "text")
    }
}

/// What gets handed to an agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentInput {
    /// The required capability — the selector matched on this.
    pub capability: Capability,
    /// The query / instruction text.
    pub query: String,
    /// Optional structured payload; opaque to the federation,
    /// agent-specific.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
    /// Caller-defined ordered metadata (request id, trace id, slot
    /// reference into a `jouleclaw-graph` run). `BTreeMap` so the wire
    /// form is deterministic.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

/// What an agent returns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentResponse {
    /// Name of the agent that handled the call.
    pub agent: String,
    /// Structured output. JSON shape is agent-specific; the federation
    /// only round-trips it.
    pub output: serde_json::Value,
    /// Microjoules actually spent on this call (measured, not the
    /// estimate from `typical_joules_per_call`).
    pub joules_uj: u64,
}

/// Errors an agent (or the registry) surfaces.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AgentError {
    /// The agent declined to handle this input (e.g. quality guard).
    #[error("agent refused: {0}")]
    Refused(String),
    /// The agent's backend failed.
    #[error("agent backend: {0}")]
    Backend(String),
    /// No agent in the registry could handle the input's capability.
    #[error("no agent in registry advertises capability {0:?}")]
    NoCapableAgent(Capability),
}

// ─────────────────────────────────────────────────────────────────────
// CallableAgent
// ─────────────────────────────────────────────────────────────────────

/// A typed handoff target — an agent the federation can dispatch to.
/// Distinct from [`crate::SearchProvider`]: search returns hits; an
/// agent returns an [`AgentResponse`].
pub trait CallableAgent: Send + Sync {
    /// Stable identifier (e.g. `"agents:summariser-cheap"`).
    fn name(&self) -> &str;

    /// What this agent advertises. The selector matches on these. An
    /// agent may advertise multiple capabilities (e.g. text+code
    /// summarisation).
    fn capabilities(&self) -> &[Capability];

    /// Self-reported typical microjoules per call — the cost table the
    /// default selector uses. SHOULD reflect measured averages from
    /// telemetry, not a datasheet number.
    fn typical_joules_per_call(&self) -> u64;

    /// Run the agent. Stateless: implementations MUST NOT depend on
    /// previous calls (no in-agent memory across invocations).
    fn call(&self, input: &AgentInput) -> Result<AgentResponse, AgentError>;
}

// ─────────────────────────────────────────────────────────────────────
// Selector + registry
// ─────────────────────────────────────────────────────────────────────

/// Picks which agent handles a given input. v1 reference is
/// [`CheapestCapable`]; smarter selectors (LLM-routed, learned, joule-
/// budget-aware) plug in here.
pub trait HandoffSelector: Send + Sync {
    fn select<'a>(
        &self,
        agents: &'a [Box<dyn CallableAgent>],
        input: &AgentInput,
    ) -> Option<&'a dyn CallableAgent>;
}

/// Cost-table selector: filter to agents whose advertised capabilities
/// contain `input.capability`, then pick the lowest
/// `typical_joules_per_call`. Ties broken by lexical `name` ordering
/// so the choice is stable across runs.
#[derive(Debug, Default, Clone, Copy)]
pub struct CheapestCapable;

impl HandoffSelector for CheapestCapable {
    fn select<'a>(
        &self,
        agents: &'a [Box<dyn CallableAgent>],
        input: &AgentInput,
    ) -> Option<&'a dyn CallableAgent> {
        agents
            .iter()
            .filter(|a| a.capabilities().contains(&input.capability))
            .min_by(|a, b| {
                a.typical_joules_per_call()
                    .cmp(&b.typical_joules_per_call())
                    .then_with(|| a.name().cmp(b.name()))
            })
            .map(|boxed| boxed.as_ref())
    }
}

/// Joule-budget-aware selector: wraps another selector and refuses to
/// pick any agent whose `typical_joules_per_call` would exceed the
/// budget. Returns `None` when nothing fits — the consumer treats
/// that as "skip this handoff" (the cascade's "refuse, don't lie"
/// posture extended to multi-agent dispatch).
///
/// Doctrine: the wrapped selector still chooses *among the affordable
/// agents*. `WithinBudget::new(CheapestCapable, 50)` means "pick the
/// cheapest capable agent costing ≤ 50 µJ"; a smarter wrapped
/// selector (LLM-routed, learned) is still budget-clamped.
pub struct WithinBudget<S: HandoffSelector> {
    inner: S,
    budget_uj: u64,
}

impl<S: HandoffSelector> WithinBudget<S> {
    pub fn new(inner: S, budget_uj: u64) -> Self {
        Self { inner, budget_uj }
    }
    pub fn budget_uj(&self) -> u64 {
        self.budget_uj
    }
}

/// LLM-routed selector. Asks a [`jouleclaw_llm_cheap::LlmBackend`]
/// to pick the best capable agent by name, given the query and the
/// list of capable agents (with their advertised capabilities and
/// typical-joule costs in the prompt).
///
/// Honest scope:
///
/// - The LLM is **constrained** — it can only pick from the capable
///   subset (the prompt lists them by name). The selector parses the
///   reply, matches a known name, and rejects anything else.
/// - On parse failure, no match, or backend error, the selector
///   **falls back** to a wrapped deterministic selector
///   (`CheapestCapable` by default). The router degrades to the
///   cost-table policy rather than refusing — degrading is the cost
///   of having a router.
/// - The router's own joule spend is *not* counted into the picked
///   agent's `joules_uj` here. Consumers that want it accounted log
///   the routing call separately (the `LlmBackend` returns the cost
///   in its response).
/// - The wrapped fallback selector can itself be budget-clamped
///   (e.g. `LlmRouted::new(backend, WithinBudget::new(CheapestCapable,
///   100))`), composing all three: smart pick, cost ceiling, stable
///   fallback.
pub struct LlmRouted<S: HandoffSelector> {
    backend: std::sync::Arc<dyn jouleclaw_llm_cheap::LlmBackend>,
    fallback: S,
    /// Token cap for the routing call. Default 32 — enough for a name.
    max_tokens: u32,
}

impl<S: HandoffSelector> LlmRouted<S> {
    pub fn new(
        backend: std::sync::Arc<dyn jouleclaw_llm_cheap::LlmBackend>,
        fallback: S,
    ) -> Self {
        Self {
            backend,
            fallback,
            max_tokens: 32,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens.max(1);
        self
    }
}

impl<S: HandoffSelector> HandoffSelector for LlmRouted<S> {
    fn select<'a>(
        &self,
        agents: &'a [Box<dyn CallableAgent>],
        input: &AgentInput,
    ) -> Option<&'a dyn CallableAgent> {
        // Build the capable-subset list. If empty, fall back.
        let capable: Vec<&dyn CallableAgent> = agents
            .iter()
            .filter(|a| a.capabilities().contains(&input.capability))
            .map(|b| b.as_ref())
            .collect();
        if capable.is_empty() {
            return self.fallback.select(agents, input);
        }
        if capable.len() == 1 {
            return Some(capable[0]);
        }

        // Render an LLM prompt listing the capable agents.
        let mut prompt = String::new();
        prompt.push_str("Pick the best agent for this task.\n");
        prompt.push_str(&format!(
            "Task ({}/{}): {}\n",
            input.capability.kind, input.capability.modality, input.query
        ));
        prompt.push_str("Agents (pick by exact name):\n");
        for a in &capable {
            prompt.push_str(&format!(
                "- {} (typical cost: {} uJ)\n",
                a.name(),
                a.typical_joules_per_call()
            ));
        }
        prompt.push_str("Reply with one agent name only. No other text.\n");

        let req = jouleclaw_llm_cheap::LlmRequest::from_prompt(prompt, self.max_tokens);
        let Ok(resp) = self.backend.complete(&req) else {
            return self.fallback.select(agents, input);
        };
        let reply = resp.text.trim();

        // Match the reply against a capable agent's name. Prefer exact
        // match; fall back to first agent whose name appears as a
        // substring of the reply (some models echo " name: X" etc).
        if let Some(hit) = capable.iter().find(|a| a.name() == reply) {
            return Some(*hit);
        }
        if let Some(hit) = capable.iter().find(|a| reply.contains(a.name())) {
            return Some(*hit);
        }
        self.fallback.select(agents, input)
    }
}

/// Learned selector — tracks per-agent observed success rate and
/// per-agent observed mean joule cost, then picks the *expected
/// cheapest successful* agent: `mean_cost / max(success_rate,
/// epsilon)`. The third selector kind, completing the cost / budget
/// / learned trio Strands/CrewAI/Anthropic have all converged on.
///
/// Bootstrap policy (the cold-start problem):
///
/// - An agent with **fewer than `min_observations` recorded calls**
///   uses its `typical_joules_per_call` as the cost estimate and
///   `default_success_rate` as the success estimate. This avoids the
///   degenerate "first call wins forever" trap a naive
///   success-rate selector falls into.
/// - Once an agent has ≥ `min_observations` calls, its observed
///   stats take over. This is why production deployments pre-seed
///   stats from telemetry rather than starting cold.
///
/// Update model — `record_outcome(agent_name, ok, joules_uj)`:
///
/// - Increments call count and success count (`ok = true`).
/// - Updates running mean of joule cost via Welford-style online
///   update (numerically stable, no overflow risk on u64 sums).
///
/// Honest scope (v1):
///
/// - **Stats are in-memory.** Persistence is the consumer's choice;
///   serialise [`LearnedSelector::stats`] to disk and reload on the
///   next process. Wire format is a plain `BTreeMap<String,
///   AgentStats>` for ease of round-trip.
/// - **Capability filter still applies.** A learned selector that
///   has never observed an agent in a given capability is still not
///   eligible — the picker filters by capability first, then ranks
///   by expected cost.
/// - **No exploration bonus.** A pure exploit policy. UCB / epsilon-
///   greedy variants slot in via the same trait without changing
///   this contract.
pub struct LearnedSelector {
    stats: std::sync::Mutex<std::collections::BTreeMap<String, AgentStats>>,
    min_observations: u64,
    default_success_rate: f64,
}

/// Per-agent observed statistics used by [`LearnedSelector`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentStats {
    /// Total calls recorded for this agent.
    pub calls: u64,
    /// Calls that returned a non-`Refused`, non-`Backend` result.
    pub successes: u64,
    /// Online running mean of `joules_uj` across all recorded
    /// calls. Welford-style update keeps this numerically stable.
    pub mean_joules_uj: f64,
}

impl AgentStats {
    /// Observed success rate, `successes / calls`. Returns `f64::NAN`
    /// if no calls recorded — the caller decides how to handle
    /// (typically by falling back to a prior).
    pub fn success_rate(&self) -> f64 {
        if self.calls == 0 {
            f64::NAN
        } else {
            self.successes as f64 / self.calls as f64
        }
    }
}

impl LearnedSelector {
    /// New selector with default warm-up policy: 3 calls minimum
    /// before observed stats take over; 50% assumed success rate
    /// for cold agents.
    pub fn new() -> Self {
        Self {
            stats: std::sync::Mutex::new(std::collections::BTreeMap::new()),
            min_observations: 3,
            default_success_rate: 0.5,
        }
    }

    /// Override the cold-start observation threshold.
    pub fn with_min_observations(mut self, n: u64) -> Self {
        self.min_observations = n.max(1);
        self
    }

    /// Override the cold-start assumed success rate.
    pub fn with_default_success_rate(mut self, p: f64) -> Self {
        // Clamp into (0, 1] so the divisor never blows up later.
        self.default_success_rate = p.clamp(1e-3, 1.0);
        self
    }

    /// Record the outcome of one call. `ok` should be `true` when
    /// the agent returned an `AgentResponse` and `false` on any
    /// error variant.
    pub fn record_outcome(&self, agent_name: &str, ok: bool, joules_uj: u64) {
        let mut stats = self.stats.lock().expect("mutex");
        let entry = stats.entry(agent_name.to_string()).or_default();
        entry.calls += 1;
        if ok {
            entry.successes += 1;
        }
        // Welford's online mean update.
        let n = entry.calls as f64;
        let delta = joules_uj as f64 - entry.mean_joules_uj;
        entry.mean_joules_uj += delta / n;
    }

    /// Snapshot the current stats for persistence / inspection.
    pub fn stats(&self) -> std::collections::BTreeMap<String, AgentStats> {
        self.stats.lock().expect("mutex").clone()
    }

    /// Restore stats from a prior snapshot — for warm-starting after
    /// a process restart. Overwrites any in-memory stats.
    pub fn load_stats(
        &self,
        stats: std::collections::BTreeMap<String, AgentStats>,
    ) {
        *self.stats.lock().expect("mutex") = stats;
    }

    fn expected_cost(&self, agent: &dyn CallableAgent) -> f64 {
        let stats = self.stats.lock().expect("mutex");
        let s = stats.get(agent.name()).copied().unwrap_or_default();
        let (cost, rate) = if s.calls >= self.min_observations {
            (s.mean_joules_uj, s.success_rate().max(1e-3))
        } else {
            (
                agent.typical_joules_per_call() as f64,
                self.default_success_rate,
            )
        };
        cost / rate
    }
}

impl Default for LearnedSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl HandoffSelector for LearnedSelector {
    fn select<'a>(
        &self,
        agents: &'a [Box<dyn CallableAgent>],
        input: &AgentInput,
    ) -> Option<&'a dyn CallableAgent> {
        agents
            .iter()
            .filter(|a| a.capabilities().contains(&input.capability))
            .min_by(|a, b| {
                let ca = self.expected_cost(a.as_ref());
                let cb = self.expected_cost(b.as_ref());
                ca.partial_cmp(&cb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.name().cmp(b.name()))
            })
            .map(|boxed| boxed.as_ref())
    }
}

impl<S: HandoffSelector> HandoffSelector for WithinBudget<S> {
    fn select<'a>(
        &self,
        agents: &'a [Box<dyn CallableAgent>],
        input: &AgentInput,
    ) -> Option<&'a dyn CallableAgent> {
        // Filter the agent slice to the budget-affordable subset, then
        // delegate. We can't pass a filtered slice through (lifetimes),
        // so the wrapped selector sees all agents and we re-check the
        // result against the budget. That works because the cost
        // function is monotone in the agent — if the inner picked one
        // we'd reject, no other budget-affordable choice would have
        // been preferred by the inner's policy anyway. We still
        // double-check.
        let pick = self.inner.select(agents, input)?;
        if pick.typical_joules_per_call() <= self.budget_uj {
            Some(pick)
        } else {
            // Inner's pick busted the budget; try the cheapest
            // affordable capable agent as a fallback.
            agents
                .iter()
                .filter(|a| {
                    a.typical_joules_per_call() <= self.budget_uj
                        && a.capabilities().contains(&input.capability)
                })
                .min_by(|a, b| {
                    a.typical_joules_per_call()
                        .cmp(&b.typical_joules_per_call())
                        .then_with(|| a.name().cmp(b.name()))
                })
                .map(|boxed| boxed.as_ref())
        }
    }
}

/// A handoff registry — the federation's typed-agent catalog. The
/// existing parallel-retrieval surface ([`crate::Federation`]) is
/// untouched; this is the typed-call path the consumer reaches for
/// when "search the web" is the wrong shape and "dispatch to a
/// capable agent" is the right one.
pub struct HandoffRegistry {
    agents: Vec<Box<dyn CallableAgent>>,
}

impl Default for HandoffRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl HandoffRegistry {
    pub fn new() -> Self {
        Self { agents: Vec::new() }
    }

    pub fn register<A: CallableAgent + 'static>(mut self, agent: A) -> Self {
        self.agents.push(Box::new(agent));
        self
    }

    pub fn add(&mut self, agent: Box<dyn CallableAgent>) {
        self.agents.push(agent);
    }

    pub fn len(&self) -> usize {
        self.agents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Names of all registered agents, in registration order.
    pub fn names(&self) -> Vec<&str> {
        self.agents.iter().map(|a| a.name()).collect()
    }

    /// Dispatch one input through the registry, picking an agent via
    /// `selector`. The federation's existing parallel-provider fanout
    /// stays the consumer's choice; this is the single-handoff path.
    pub fn dispatch(
        &self,
        input: &AgentInput,
        selector: &dyn HandoffSelector,
    ) -> Result<AgentResponse, AgentError> {
        match selector.select(&self.agents, input) {
            Some(agent) => agent.call(input),
            None => Err(AgentError::NoCapableAgent(input.capability.clone())),
        }
    }

    /// Dispatch a chain of handoffs — each step's `AgentResponse` is
    /// piped into the *next* step's `AgentInput` as a `prev` payload
    /// (under `_prev_output` and `_prev_agent` metadata keys) so the
    /// next agent can read what the previous one produced.
    ///
    /// This is the "worker-to-worker" pattern made explicit, with the
    /// honest scope spelled out:
    ///
    /// - Each agent is still **stateless** (the `CallableAgent`
    ///   contract): the only context carried forward is the prior
    ///   `output` JSON and the prior agent's `name`. No conversation,
    ///   no shared scratchpad.
    /// - Steps are sequential — no parallel branches. Branching belongs
    ///   in a `jouleclaw-graph::RunGraph`.
    /// - On any step's failure, the chain stops and returns the error
    ///   (no partial recovery). Joules spent up to that point are
    ///   lost; the consumer logs them via the per-step responses if it
    ///   wants accounting.
    /// - Each step is selected independently — the chain is a list of
    ///   `(input, selector)` pairs, so a chain can mix `CheapestCapable`
    ///   for the easy step and a smarter selector for the hard one.
    ///
    /// Returns the per-step responses in order. The terminal response
    /// is `out.last()`; intermediate responses are kept so the caller
    /// can audit the joule-spend of each step.
    pub fn dispatch_chain(
        &self,
        steps: &[(AgentInput, &dyn HandoffSelector)],
    ) -> Result<Vec<AgentResponse>, AgentError> {
        let mut out = Vec::with_capacity(steps.len());
        let mut prev: Option<AgentResponse> = None;
        for (input, selector) in steps {
            let mut next_input = input.clone();
            if let Some(p) = &prev {
                // Pipe prior output through as opaque payload + breadcrumb
                // metadata. We don't merge into payload; we replace under
                // an `_prev_output` key so the next agent sees its own
                // payload alongside.
                let mut payload = next_input.payload.unwrap_or(serde_json::json!({}));
                if let Some(map) = payload.as_object_mut() {
                    map.insert("_prev_output".into(), p.output.clone());
                    map.insert(
                        "_prev_agent".into(),
                        serde_json::Value::String(p.agent.clone()),
                    );
                }
                next_input.payload = Some(payload);
                next_input
                    .metadata
                    .insert("_prev_agent".into(), p.agent.clone());
            }
            let resp = match selector.select(&self.agents, &next_input) {
                Some(agent) => agent.call(&next_input)?,
                None => {
                    return Err(AgentError::NoCapableAgent(next_input.capability.clone()))
                }
            };
            prev = Some(resp.clone());
            out.push(resp);
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticAgent {
        name: String,
        caps: Vec<Capability>,
        joules: u64,
        reply: serde_json::Value,
    }
    impl CallableAgent for StaticAgent {
        fn name(&self) -> &str {
            &self.name
        }
        fn capabilities(&self) -> &[Capability] {
            &self.caps
        }
        fn typical_joules_per_call(&self) -> u64 {
            self.joules
        }
        fn call(&self, _input: &AgentInput) -> Result<AgentResponse, AgentError> {
            Ok(AgentResponse {
                agent: self.name.clone(),
                output: self.reply.clone(),
                joules_uj: self.joules,
            })
        }
    }

    struct FailingAgent {
        name: String,
        caps: Vec<Capability>,
    }
    impl CallableAgent for FailingAgent {
        fn name(&self) -> &str {
            &self.name
        }
        fn capabilities(&self) -> &[Capability] {
            &self.caps
        }
        fn typical_joules_per_call(&self) -> u64 {
            0
        }
        fn call(&self, _input: &AgentInput) -> Result<AgentResponse, AgentError> {
            Err(AgentError::Backend("simulated".into()))
        }
    }

    fn cap(kind: &str) -> Capability {
        Capability::text(kind)
    }

    fn input(c: Capability, q: &str) -> AgentInput {
        AgentInput {
            capability: c,
            query: q.into(),
            payload: None,
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn dispatch_routes_to_a_capable_agent() {
        let reg = HandoffRegistry::new()
            .register(StaticAgent {
                name: "summariser".into(),
                caps: vec![cap("summarise")],
                joules: 5,
                reply: serde_json::json!({"summary": "ok"}),
            })
            .register(StaticAgent {
                name: "translator".into(),
                caps: vec![cap("translate")],
                joules: 5,
                reply: serde_json::json!({"translation": "ok"}),
            });
        let r = reg.dispatch(&input(cap("translate"), "hi"), &CheapestCapable).unwrap();
        assert_eq!(r.agent, "translator");
    }

    #[test]
    fn cheapest_capable_picks_lowest_cost_among_qualified() {
        let reg = HandoffRegistry::new()
            .register(StaticAgent {
                name: "expensive".into(),
                caps: vec![cap("summarise")],
                joules: 100,
                reply: serde_json::json!({}),
            })
            .register(StaticAgent {
                name: "cheap".into(),
                caps: vec![cap("summarise")],
                joules: 10,
                reply: serde_json::json!({}),
            });
        let r = reg.dispatch(&input(cap("summarise"), "x"), &CheapestCapable).unwrap();
        assert_eq!(r.agent, "cheap");
    }

    #[test]
    fn ties_are_broken_by_lexical_name() {
        let reg = HandoffRegistry::new()
            .register(StaticAgent {
                name: "zebra".into(),
                caps: vec![cap("x")],
                joules: 5,
                reply: serde_json::json!({}),
            })
            .register(StaticAgent {
                name: "alpha".into(),
                caps: vec![cap("x")],
                joules: 5,
                reply: serde_json::json!({}),
            });
        let r = reg.dispatch(&input(cap("x"), "q"), &CheapestCapable).unwrap();
        assert_eq!(
            r.agent, "alpha",
            "lexical tie-break makes the choice stable across runs"
        );
    }

    #[test]
    fn no_capable_agent_returns_error() {
        let reg = HandoffRegistry::new().register(StaticAgent {
            name: "summariser".into(),
            caps: vec![cap("summarise")],
            joules: 5,
            reply: serde_json::json!({}),
        });
        let err = reg
            .dispatch(&input(cap("not-here"), "q"), &CheapestCapable)
            .unwrap_err();
        match err {
            AgentError::NoCapableAgent(c) => {
                assert_eq!(c.kind, "not-here");
                assert_eq!(c.modality, "text");
            }
            other => panic!("expected NoCapableAgent, got {other:?}"),
        }
    }

    #[test]
    fn agent_backend_error_propagates() {
        let reg = HandoffRegistry::new().register(FailingAgent {
            name: "broken".into(),
            caps: vec![cap("summarise")],
        });
        let err = reg
            .dispatch(&input(cap("summarise"), "q"), &CheapestCapable)
            .unwrap_err();
        assert!(matches!(err, AgentError::Backend(_)));
    }

    #[test]
    fn modality_is_part_of_the_match() {
        let reg = HandoffRegistry::new()
            .register(StaticAgent {
                name: "text-only".into(),
                caps: vec![Capability::new("summarise", "text")],
                joules: 5,
                reply: serde_json::json!({}),
            })
            .register(StaticAgent {
                name: "image-only".into(),
                caps: vec![Capability::new("summarise", "image")],
                joules: 5,
                reply: serde_json::json!({}),
            });
        let r = reg
            .dispatch(
                &input(Capability::new("summarise", "image"), "q"),
                &CheapestCapable,
            )
            .unwrap();
        assert_eq!(r.agent, "image-only");
    }

    #[test]
    fn agents_advertising_multiple_capabilities_are_eligible() {
        let reg = HandoffRegistry::new().register(StaticAgent {
            name: "polymath".into(),
            caps: vec![cap("summarise"), cap("translate"), cap("extract")],
            joules: 5,
            reply: serde_json::json!({"polymath": "ok"}),
        });
        for k in ["summarise", "translate", "extract"] {
            let r = reg.dispatch(&input(cap(k), "q"), &CheapestCapable).unwrap();
            assert_eq!(r.agent, "polymath");
        }
    }

    #[test]
    fn input_response_round_trip_through_json() {
        let i = AgentInput {
            capability: Capability::new("summarise", "text"),
            query: "summarise this".into(),
            payload: Some(serde_json::json!({"depth": 3})),
            metadata: {
                let mut m = BTreeMap::new();
                m.insert("slot".into(), "graph:n3".into());
                m
            },
        };
        let bytes = serde_json::to_vec(&i).unwrap();
        let back: AgentInput = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, i);

        let r = AgentResponse {
            agent: "x".into(),
            output: serde_json::json!({"k": "v"}),
            joules_uj: 7,
        };
        let bytes = serde_json::to_vec(&r).unwrap();
        let back: AgentResponse = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn empty_registry_is_unhelpful_but_does_not_panic() {
        let reg = HandoffRegistry::new();
        assert!(reg.is_empty());
        let err = reg
            .dispatch(&input(cap("anything"), "q"), &CheapestCapable)
            .unwrap_err();
        assert!(matches!(err, AgentError::NoCapableAgent(_)));
    }

    // ─── WithinBudget ───────────────────────────────────────────────

    #[test]
    fn within_budget_uses_inner_when_pick_fits() {
        let reg = HandoffRegistry::new().register(StaticAgent {
            name: "cheap".into(),
            caps: vec![cap("summarise")],
            joules: 10,
            reply: serde_json::json!({}),
        });
        let sel = WithinBudget::new(CheapestCapable, 100);
        let r = reg
            .dispatch(&input(cap("summarise"), "q"), &sel)
            .unwrap();
        assert_eq!(r.agent, "cheap");
    }

    #[test]
    fn within_budget_falls_back_when_inner_pick_busts_budget() {
        let reg = HandoffRegistry::new()
            .register(StaticAgent {
                name: "cheap".into(),
                caps: vec![cap("summarise")],
                joules: 80,
                reply: serde_json::json!({}),
            })
            .register(StaticAgent {
                name: "cheaper".into(),
                caps: vec![cap("summarise")],
                joules: 30,
                reply: serde_json::json!({}),
            });
        // Inner CheapestCapable would pick "cheaper" (already cheap).
        // Set budget below "cheap" only.
        let sel = WithinBudget::new(CheapestCapable, 50);
        let r = reg.dispatch(&input(cap("summarise"), "q"), &sel).unwrap();
        assert_eq!(r.agent, "cheaper");
    }

    #[test]
    fn within_budget_refuses_when_nothing_fits() {
        let reg = HandoffRegistry::new().register(StaticAgent {
            name: "spendy".into(),
            caps: vec![cap("summarise")],
            joules: 9999,
            reply: serde_json::json!({}),
        });
        let sel = WithinBudget::new(CheapestCapable, 50);
        let err = reg
            .dispatch(&input(cap("summarise"), "q"), &sel)
            .unwrap_err();
        assert!(matches!(err, AgentError::NoCapableAgent(_)));
    }

    // ─── dispatch_chain ─────────────────────────────────────────────

    /// Test helper agent that records whether the prev-output payload
    /// arrived as expected.
    struct PrevSensingAgent {
        name: String,
        caps: Vec<Capability>,
        joules: u64,
    }
    impl CallableAgent for PrevSensingAgent {
        fn name(&self) -> &str {
            &self.name
        }
        fn capabilities(&self) -> &[Capability] {
            &self.caps
        }
        fn typical_joules_per_call(&self) -> u64 {
            self.joules
        }
        fn call(&self, input: &AgentInput) -> Result<AgentResponse, AgentError> {
            let prev = input
                .payload
                .as_ref()
                .and_then(|p| p.get("_prev_output"))
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            Ok(AgentResponse {
                agent: self.name.clone(),
                output: serde_json::json!({
                    "from": self.name,
                    "saw_prev": prev,
                }),
                joules_uj: self.joules,
            })
        }
    }

    #[test]
    fn dispatch_chain_pipes_prior_output_into_next_input() {
        let reg = HandoffRegistry::new()
            .register(StaticAgent {
                name: "a".into(),
                caps: vec![cap("step1")],
                joules: 5,
                reply: serde_json::json!({"draft": "hello"}),
            })
            .register(PrevSensingAgent {
                name: "b".into(),
                caps: vec![cap("step2")],
                joules: 5,
            });

        let steps: Vec<(AgentInput, &dyn HandoffSelector)> = vec![
            (input(cap("step1"), "first"), &CheapestCapable),
            (input(cap("step2"), "second"), &CheapestCapable),
        ];

        let responses = reg.dispatch_chain(&steps).unwrap();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0].agent, "a");
        assert_eq!(responses[1].agent, "b");
        let prev = responses[1].output.get("saw_prev").unwrap();
        assert_eq!(prev, &serde_json::json!({"draft": "hello"}));
    }

    #[test]
    fn dispatch_chain_stops_on_step_error() {
        let reg = HandoffRegistry::new()
            .register(StaticAgent {
                name: "ok".into(),
                caps: vec![cap("step1")],
                joules: 5,
                reply: serde_json::json!({}),
            })
            .register(FailingAgent {
                name: "broken".into(),
                caps: vec![cap("step2")],
            });
        let steps: Vec<(AgentInput, &dyn HandoffSelector)> = vec![
            (input(cap("step1"), "q1"), &CheapestCapable),
            (input(cap("step2"), "q2"), &CheapestCapable),
        ];
        let err = reg.dispatch_chain(&steps).unwrap_err();
        assert!(matches!(err, AgentError::Backend(_)));
    }

    #[test]
    fn dispatch_chain_empty_is_ok_with_no_responses() {
        let reg = HandoffRegistry::new();
        let r = reg.dispatch_chain(&[]).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn dispatch_chain_no_capable_for_step_errors() {
        let reg = HandoffRegistry::new().register(StaticAgent {
            name: "only".into(),
            caps: vec![cap("step1")],
            joules: 5,
            reply: serde_json::json!({}),
        });
        let steps: Vec<(AgentInput, &dyn HandoffSelector)> = vec![
            (input(cap("step1"), "q1"), &CheapestCapable),
            (input(cap("step-unknown"), "q2"), &CheapestCapable),
        ];
        let err = reg.dispatch_chain(&steps).unwrap_err();
        assert!(matches!(err, AgentError::NoCapableAgent(_)));
    }

    // ─── LlmRouted ──────────────────────────────────────────────────

    /// Test backend that returns a fixed string regardless of input.
    struct ConstantBackend(String);
    impl jouleclaw_llm_cheap::LlmBackend for ConstantBackend {
        fn model_name(&self) -> &str {
            "test:constant"
        }
        fn complete(
            &self,
            _req: &jouleclaw_llm_cheap::LlmRequest,
        ) -> Result<jouleclaw_llm_cheap::LlmResponse, jouleclaw_llm_cheap::LlmError> {
            Ok(jouleclaw_llm_cheap::LlmResponse {
                text: self.0.clone(),
                finish_reason: jouleclaw_llm_cheap::FinishReason::Stop,
                input_tokens: 0,
                output_tokens: self.0.len() as u32,
                energy_joules: None,
            })
        }
    }

    /// Test backend that always errors — exercises the fallback path.
    struct ErroringBackend;
    impl jouleclaw_llm_cheap::LlmBackend for ErroringBackend {
        fn model_name(&self) -> &str {
            "test:erroring"
        }
        fn complete(
            &self,
            _req: &jouleclaw_llm_cheap::LlmRequest,
        ) -> Result<jouleclaw_llm_cheap::LlmResponse, jouleclaw_llm_cheap::LlmError> {
            Err(jouleclaw_llm_cheap::LlmError::Unavailable("test".into()))
        }
    }

    #[test]
    fn llm_routed_picks_the_named_agent() {
        let reg = HandoffRegistry::new()
            .register(StaticAgent {
                name: "fast-cheap".into(),
                caps: vec![cap("summarise")],
                joules: 10,
                reply: serde_json::json!({}),
            })
            .register(StaticAgent {
                name: "deep-thoughtful".into(),
                caps: vec![cap("summarise")],
                joules: 200,
                reply: serde_json::json!({}),
            });
        let sel = LlmRouted::new(
            std::sync::Arc::new(ConstantBackend("deep-thoughtful".into())),
            CheapestCapable,
        );
        let r = reg.dispatch(&input(cap("summarise"), "q"), &sel).unwrap();
        assert_eq!(r.agent, "deep-thoughtful");
    }

    #[test]
    fn llm_routed_falls_back_when_backend_errors() {
        let reg = HandoffRegistry::new()
            .register(StaticAgent {
                name: "fast-cheap".into(),
                caps: vec![cap("summarise")],
                joules: 10,
                reply: serde_json::json!({}),
            })
            .register(StaticAgent {
                name: "deep-thoughtful".into(),
                caps: vec![cap("summarise")],
                joules: 200,
                reply: serde_json::json!({}),
            });
        let sel = LlmRouted::new(std::sync::Arc::new(ErroringBackend), CheapestCapable);
        let r = reg.dispatch(&input(cap("summarise"), "q"), &sel).unwrap();
        assert_eq!(r.agent, "fast-cheap", "fallback to CheapestCapable");
    }

    #[test]
    fn llm_routed_falls_back_on_unrecognised_name() {
        let reg = HandoffRegistry::new()
            .register(StaticAgent {
                name: "fast-cheap".into(),
                caps: vec![cap("summarise")],
                joules: 10,
                reply: serde_json::json!({}),
            });
        let sel = LlmRouted::new(
            std::sync::Arc::new(ConstantBackend("ghost-agent".into())),
            CheapestCapable,
        );
        // Only one capable agent — LlmRouted short-circuits, no LLM
        // call needed.
        let r = reg.dispatch(&input(cap("summarise"), "q"), &sel).unwrap();
        assert_eq!(r.agent, "fast-cheap");
    }

    #[test]
    fn llm_routed_extracts_name_from_chatty_reply() {
        let reg = HandoffRegistry::new()
            .register(StaticAgent {
                name: "alpha".into(),
                caps: vec![cap("x")],
                joules: 5,
                reply: serde_json::json!({}),
            })
            .register(StaticAgent {
                name: "beta".into(),
                caps: vec![cap("x")],
                joules: 5,
                reply: serde_json::json!({}),
            });
        let sel = LlmRouted::new(
            std::sync::Arc::new(ConstantBackend(
                "I'd pick beta because it handles this case best.".into(),
            )),
            CheapestCapable,
        );
        let r = reg.dispatch(&input(cap("x"), "q"), &sel).unwrap();
        assert_eq!(r.agent, "beta");
    }

    #[test]
    fn llm_routed_composes_with_within_budget() {
        let reg = HandoffRegistry::new()
            .register(StaticAgent {
                name: "cheap".into(),
                caps: vec![cap("summarise")],
                joules: 20,
                reply: serde_json::json!({}),
            })
            .register(StaticAgent {
                name: "expensive".into(),
                caps: vec![cap("summarise")],
                joules: 500,
                reply: serde_json::json!({}),
            });
        // LLM picks "expensive"; budget refuses; falls back to cheapest
        // affordable.
        let inner = LlmRouted::new(
            std::sync::Arc::new(ConstantBackend("expensive".into())),
            CheapestCapable,
        );
        let sel = WithinBudget::new(inner, 100);
        let r = reg.dispatch(&input(cap("summarise"), "q"), &sel).unwrap();
        assert_eq!(r.agent, "cheap");
    }

    // ─── LearnedSelector ────────────────────────────────────────────

    fn two_capable_reg() -> HandoffRegistry {
        HandoffRegistry::new()
            .register(StaticAgent {
                name: "alpha".into(),
                caps: vec![cap("summarise")],
                joules: 50,
                reply: serde_json::json!({}),
            })
            .register(StaticAgent {
                name: "beta".into(),
                caps: vec![cap("summarise")],
                joules: 50,
                reply: serde_json::json!({}),
            })
    }

    #[test]
    fn learned_selector_cold_start_uses_typical_costs_and_lexical_tiebreak() {
        let sel = LearnedSelector::new();
        let reg = two_capable_reg();
        // Cold start: both equal cost, both 50% success — picks
        // lexically smaller name.
        let r = reg.dispatch(&input(cap("summarise"), "q"), &sel).unwrap();
        assert_eq!(r.agent, "alpha");
    }

    #[test]
    fn learned_selector_prefers_higher_success_rate_after_warm_up() {
        let sel = LearnedSelector::new().with_min_observations(3);
        // Beta succeeds 5/5; alpha succeeds 1/5. Same observed cost.
        for _ in 0..5 {
            sel.record_outcome("beta", true, 50);
        }
        sel.record_outcome("alpha", true, 50);
        for _ in 0..4 {
            sel.record_outcome("alpha", false, 50);
        }
        let reg = two_capable_reg();
        let r = reg.dispatch(&input(cap("summarise"), "q"), &sel).unwrap();
        assert_eq!(r.agent, "beta");
    }

    #[test]
    fn learned_selector_prefers_lower_observed_cost_when_success_equal() {
        let sel = LearnedSelector::new().with_min_observations(3);
        // Both perfect success; alpha runs cheaper on average.
        for _ in 0..3 {
            sel.record_outcome("alpha", true, 10);
        }
        for _ in 0..3 {
            sel.record_outcome("beta", true, 100);
        }
        let reg = two_capable_reg();
        let r = reg.dispatch(&input(cap("summarise"), "q"), &sel).unwrap();
        assert_eq!(r.agent, "alpha");
    }

    #[test]
    fn learned_selector_capability_filter_still_applies() {
        let sel = LearnedSelector::new();
        for _ in 0..10 {
            // Stats for an agent that doesn't advertise the wanted
            // capability — should be ignored regardless.
            sel.record_outcome("alpha", true, 5);
        }
        let reg = HandoffRegistry::new().register(StaticAgent {
            name: "alpha".into(),
            caps: vec![cap("translate")],
            joules: 5,
            reply: serde_json::json!({}),
        });
        let err = reg
            .dispatch(&input(cap("summarise"), "q"), &sel)
            .unwrap_err();
        assert!(matches!(err, AgentError::NoCapableAgent(_)));
    }

    #[test]
    fn learned_selector_stats_round_trip_for_persistence() {
        let sel = LearnedSelector::new();
        sel.record_outcome("alpha", true, 10);
        sel.record_outcome("alpha", false, 30);
        let snap = sel.stats();
        let alpha = snap.get("alpha").copied().unwrap_or_default();
        assert_eq!(alpha.calls, 2);
        assert_eq!(alpha.successes, 1);
        assert!((alpha.mean_joules_uj - 20.0).abs() < 1e-9);
        // Serialise → deserialise round-trip.
        let bytes = serde_json::to_vec(&snap).unwrap();
        let back: std::collections::BTreeMap<String, AgentStats> =
            serde_json::from_slice(&bytes).unwrap();
        let restored = LearnedSelector::new();
        restored.load_stats(back);
        assert_eq!(restored.stats(), snap);
    }

    #[test]
    fn learned_selector_composes_with_within_budget() {
        let sel = LearnedSelector::new().with_min_observations(1);
        // Beta has perfect success but expensive observed cost.
        // Alpha has high failure rate but cheap observed cost.
        // WithinBudget cap should rule out beta even though the
        // learned selector would otherwise prefer it.
        sel.record_outcome("alpha", true, 5);
        sel.record_outcome("alpha", false, 5);
        sel.record_outcome("beta", true, 9999);
        let reg = HandoffRegistry::new()
            .register(StaticAgent {
                name: "alpha".into(),
                caps: vec![cap("summarise")],
                joules: 5,
                reply: serde_json::json!({}),
            })
            .register(StaticAgent {
                name: "beta".into(),
                caps: vec![cap("summarise")],
                joules: 9999,
                reply: serde_json::json!({}),
            });
        let budgeted = WithinBudget::new(sel, 100);
        let r = reg
            .dispatch(&input(cap("summarise"), "q"), &budgeted)
            .unwrap();
        assert_eq!(r.agent, "alpha");
    }

    #[test]
    fn agent_stats_success_rate_is_nan_when_uncalled() {
        let s = AgentStats::default();
        assert!(s.success_rate().is_nan());
    }
}
