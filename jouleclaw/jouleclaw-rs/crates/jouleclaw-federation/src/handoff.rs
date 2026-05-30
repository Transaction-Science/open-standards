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
}
