//! Session tree — an energy-attributed, branchable log of resolutions.
//!
//! Pi lets you fork a session at any point, navigate the tree, and
//! continue from any branch with full history preserved. JouleClaw
//! ingests that and adds the energy dimension: every [`Turn`] records the
//! joules its resolution spent, so a [`Session`]'s cost is the sum of its
//! turns and **each fork branch carries its own energy total**. You can
//! see what an exploration path cost, not just what it concluded.
//!
//! A [`Session`] is an append-only list of [`Turn`]s plus a parent
//! pointer. [`SessionTree::fork`] copies a session's prefix up to a chosen
//! turn into a new session (parent = the original); the two then diverge
//! without affecting each other. The tree records lineage and branches.
//!
//! This is orchestration state, not a cascade tier. Record each runtime
//! resolution with [`SessionTree::record_answer`]; fork to branch.

#![forbid(unsafe_code)]

use std::collections::HashMap;

use jouleclaw_cascade::types::{Answer, AnswerOutput, Query, QueryInput, TierId};
use serde::{Deserialize, Serialize};

/// Opaque session identifier (monotonic within a tree).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SessionId(pub u64);

/// Index of a turn within its session (0-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TurnId(pub usize);

/// One resolution in a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    /// The query text (best-effort; binary inputs render as a placeholder).
    pub query: String,
    /// The answer text (refusals render as a placeholder).
    pub answer: String,
    /// Wire tag of the tier that resolved it (e.g. `"L0.1"`, `"L3"`).
    pub tier: String,
    /// Joules this resolution spent.
    pub joules: f64,
    /// Confidence in `[0, 1]`.
    pub confidence: f32,
}

/// An append-only session: a list of turns plus its fork lineage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: SessionId,
    /// The session this one was forked from, if any.
    pub parent: Option<SessionId>,
    /// The turn index in the parent at which this fork diverged.
    pub forked_at: Option<TurnId>,
    /// The turns, in order.
    pub turns: Vec<Turn>,
}

impl Session {
    /// Total joules across all turns in this session (its prefix +
    /// everything appended after the fork).
    pub fn total_joules(&self) -> f64 {
        self.turns.iter().map(|t| t.joules).sum()
    }

    pub fn len(&self) -> usize {
        self.turns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }
}

/// Errors operating on a session tree.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("unknown session {0:?}")]
    UnknownSession(SessionId),
    #[error("turn {turn:?} out of range for session {session:?} (len {len})")]
    TurnOutOfRange {
        session: SessionId,
        turn: TurnId,
        len: usize,
    },
    #[error("serialize session tree: {0}")]
    Serialize(String),
}

/// A tree of sessions related by forking.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SessionTree {
    sessions: HashMap<SessionId, Session>,
    next_id: u64,
}

impl SessionTree {
    pub fn new() -> Self {
        Self::default()
    }

    fn alloc_id(&mut self) -> SessionId {
        let id = SessionId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Open a fresh root session.
    pub fn new_session(&mut self) -> SessionId {
        let id = self.alloc_id();
        self.sessions.insert(
            id,
            Session {
                id,
                parent: None,
                forked_at: None,
                turns: Vec::new(),
            },
        );
        id
    }

    /// Append a turn to a session; returns its [`TurnId`].
    pub fn record(
        &mut self,
        session: SessionId,
        turn: Turn,
    ) -> Result<TurnId, SessionError> {
        let s = self
            .sessions
            .get_mut(&session)
            .ok_or(SessionError::UnknownSession(session))?;
        let idx = TurnId(s.turns.len());
        s.turns.push(turn);
        Ok(idx)
    }

    /// Append a turn built from a cascade [`Query`] + [`Answer`].
    pub fn record_answer(
        &mut self,
        session: SessionId,
        q: &Query,
        a: &Answer,
    ) -> Result<TurnId, SessionError> {
        self.record(session, turn_from(q, a))
    }

    /// Fork `session` at turn `at` (inclusive): create a new session whose
    /// turns are a copy of `session`'s `[0..=at]`, with `parent = session`
    /// and `forked_at = at`. The new session can then diverge freely.
    pub fn fork(
        &mut self,
        session: SessionId,
        at: TurnId,
    ) -> Result<SessionId, SessionError> {
        let prefix = {
            let s = self
                .sessions
                .get(&session)
                .ok_or(SessionError::UnknownSession(session))?;
            if at.0 >= s.turns.len() {
                return Err(SessionError::TurnOutOfRange {
                    session,
                    turn: at,
                    len: s.turns.len(),
                });
            }
            s.turns[..=at.0].to_vec()
        };
        let id = self.alloc_id();
        self.sessions.insert(
            id,
            Session {
                id,
                parent: Some(session),
                forked_at: Some(at),
                turns: prefix,
            },
        );
        Ok(id)
    }

    /// Fork at the latest turn (the head). Errors if the session is empty.
    pub fn fork_head(&mut self, session: SessionId) -> Result<SessionId, SessionError> {
        let len = self
            .sessions
            .get(&session)
            .ok_or(SessionError::UnknownSession(session))?
            .turns
            .len();
        if len == 0 {
            return Err(SessionError::TurnOutOfRange {
                session,
                turn: TurnId(0),
                len: 0,
            });
        }
        self.fork(session, TurnId(len - 1))
    }

    pub fn session(&self, id: SessionId) -> Option<&Session> {
        self.sessions.get(&id)
    }

    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// Direct children (sessions forked from `id`), sorted by id.
    pub fn children(&self, id: SessionId) -> Vec<SessionId> {
        let mut kids: Vec<SessionId> = self
            .sessions
            .values()
            .filter(|s| s.parent == Some(id))
            .map(|s| s.id)
            .collect();
        kids.sort();
        kids
    }

    /// Lineage from the root down to `id` (inclusive).
    pub fn lineage(&self, id: SessionId) -> Vec<SessionId> {
        let mut chain = Vec::new();
        let mut cur = Some(id);
        while let Some(c) = cur {
            chain.push(c);
            cur = self.sessions.get(&c).and_then(|s| s.parent);
        }
        chain.reverse();
        chain
    }

    /// Total joules of a session (its own turns).
    pub fn total_joules(&self, id: SessionId) -> Option<f64> {
        self.sessions.get(&id).map(|s| s.total_joules())
    }

    /// Serialize the whole tree (for durable session storage).
    pub fn to_json(&self) -> Result<String, SessionError> {
        serde_json::to_string(self).map_err(|e| SessionError::Serialize(e.to_string()))
    }

    /// Reload a tree from JSON.
    pub fn from_json(s: &str) -> Result<Self, SessionError> {
        serde_json::from_str(s).map_err(|e| SessionError::Serialize(e.to_string()))
    }
}

fn query_text(q: &Query) -> String {
    match &q.input {
        QueryInput::Text(t) => t.clone(),
        QueryInput::Multimodal { text, .. } => text.clone(),
        QueryInput::Structured(b) => {
            std::str::from_utf8(b).map(|s| s.to_string()).unwrap_or_else(|_| "<structured>".into())
        }
        QueryInput::Binary(_) => "<binary>".into(),
        QueryInput::Image(_) => "<image>".into(),
        QueryInput::Audio(_) => "<audio>".into(),
    }
}

fn answer_text(a: &Answer) -> String {
    match &a.output {
        AnswerOutput::Text(t) => t.clone(),
        AnswerOutput::Structured(b) => String::from_utf8_lossy(b).into_owned(),
        AnswerOutput::Refused(_) => "<refused>".into(),
    }
}

fn turn_from(q: &Query, a: &Answer) -> Turn {
    Turn {
        query: query_text(q),
        answer: answer_text(a),
        tier: tier_tag(a.tier_used).to_string(),
        joules: a.joules_spent,
        confidence: a.confidence,
    }
}

fn tier_tag(t: TierId) -> &'static str {
    t.wire_tag()
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::types::{
        ContextRef, ExecutionTrace, JouleBudget, L3ModelId, QualityFloor, TierId,
    };
    use jouleclaw_cascade::verification::VerificationStatus;

    fn q(text: &str) -> Query {
        Query {
            input: QueryInput::Text(text.to_string()),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    fn ans(text: &str, tier: TierId, joules: f64) -> Answer {
        Answer {
            output: AnswerOutput::Text(text.to_string()),
            tier_used: tier,
            joules_spent: joules,
            confidence: 0.9,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        }
    }

    #[test]
    fn records_turns_and_sums_energy() {
        let mut tree = SessionTree::new();
        let s = tree.new_session();
        tree.record_answer(s, &q("one"), &ans("a1", TierId::L0, 1.0)).unwrap();
        tree.record_answer(s, &q("two"), &ans("a2", TierId::L3(L3ModelId(0)), 2.0)).unwrap();
        let sess = tree.session(s).unwrap();
        assert_eq!(sess.len(), 2);
        assert!((sess.total_joules() - 3.0).abs() < 1e-9);
        assert_eq!(sess.turns[1].tier, "L3");
        assert_eq!(sess.turns[0].query, "one");
    }

    #[test]
    fn fork_copies_prefix_and_diverges() {
        let mut tree = SessionTree::new();
        let s = tree.new_session();
        tree.record_answer(s, &q("t0"), &ans("a0", TierId::L0, 1.0)).unwrap();
        tree.record_answer(s, &q("t1"), &ans("a1", TierId::L0, 1.0)).unwrap();
        tree.record_answer(s, &q("t2"), &ans("a2", TierId::L0, 1.0)).unwrap();

        // Fork at turn 1 → new session has turns [t0, t1].
        let f = tree.fork(s, TurnId(1)).unwrap();
        assert_eq!(tree.session(f).unwrap().len(), 2);
        assert_eq!(tree.session(f).unwrap().parent, Some(s));
        assert_eq!(tree.session(f).unwrap().forked_at, Some(TurnId(1)));

        // Diverge: append a different turn to the fork.
        tree.record_answer(f, &q("alt"), &ans("alt-ans", TierId::L0, 5.0)).unwrap();
        // Original is unaffected (still 3 turns), fork has 3 (prefix 2 + 1).
        assert_eq!(tree.session(s).unwrap().len(), 3);
        assert_eq!(tree.session(f).unwrap().len(), 3);
        assert_eq!(tree.session(f).unwrap().turns[2].query, "alt");
        // Each branch carries its own energy total.
        assert!((tree.total_joules(s).unwrap() - 3.0).abs() < 1e-9);
        assert!((tree.total_joules(f).unwrap() - 7.0).abs() < 1e-9); // 1+1+5
    }

    #[test]
    fn fork_head_and_out_of_range() {
        let mut tree = SessionTree::new();
        let s = tree.new_session();
        assert!(tree.fork_head(s).is_err()); // empty
        tree.record_answer(s, &q("t0"), &ans("a0", TierId::L0, 1.0)).unwrap();
        let f = tree.fork_head(s).unwrap();
        assert_eq!(tree.session(f).unwrap().len(), 1);
        assert!(matches!(
            tree.fork(s, TurnId(99)),
            Err(SessionError::TurnOutOfRange { .. })
        ));
    }

    #[test]
    fn lineage_and_children() {
        let mut tree = SessionTree::new();
        let root = tree.new_session();
        tree.record_answer(root, &q("t0"), &ans("a0", TierId::L0, 1.0)).unwrap();
        let a = tree.fork_head(root).unwrap();
        tree.record_answer(a, &q("ta"), &ans("aa", TierId::L0, 1.0)).unwrap();
        let b = tree.fork_head(a).unwrap();
        // lineage(b) = root → a → b
        assert_eq!(tree.lineage(b), vec![root, a, b]);
        // children(root) = [a]; children(a) = [b]
        assert_eq!(tree.children(root), vec![a]);
        assert_eq!(tree.children(a), vec![b]);
    }

    #[test]
    fn tree_round_trips_through_json() {
        let mut tree = SessionTree::new();
        let s = tree.new_session();
        tree.record_answer(s, &q("hi"), &ans("yo", TierId::L0_1FactLut, 5e-9)).unwrap();
        let json = tree.to_json().unwrap();
        let reloaded = SessionTree::from_json(&json).unwrap();
        assert_eq!(reloaded.len(), 1);
        assert_eq!(reloaded.session(s).unwrap().turns[0].answer, "yo");
        assert_eq!(reloaded.session(s).unwrap().turns[0].tier, "L0.1");
    }

    #[test]
    fn unknown_session_errors() {
        let mut tree = SessionTree::new();
        assert!(matches!(
            tree.record(SessionId(99), Turn {
                query: "x".into(), answer: "y".into(), tier: "L0".into(), joules: 0.0, confidence: 0.0
            }),
            Err(SessionError::UnknownSession(_))
        ));
    }
}
