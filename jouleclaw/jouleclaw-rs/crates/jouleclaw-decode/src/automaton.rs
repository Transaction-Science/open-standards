//! Byte-level NFA with epsilon transitions.
//!
//! All grammar surfaces (regex, JSON-schema, CFG) lower to this form.
//! Simulation is the classic subset construction kept implicit: we
//! maintain the set of active states and propagate epsilon closure on
//! every transition.

use std::collections::BTreeSet;

/// A 256-bit byte class. `set[b]` is true when byte `b` is in the class.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ByteClass {
    bits: [u64; 4],
}

impl ByteClass {
    /// Empty class — matches no bytes.
    pub fn empty() -> Self {
        Self { bits: [0; 4] }
    }

    /// Class containing every byte 0..=255.
    pub fn full() -> Self {
        Self { bits: [!0u64; 4] }
    }

    /// Class containing a single byte.
    pub fn single(b: u8) -> Self {
        let mut s = Self::empty();
        s.add(b);
        s
    }

    /// Class spanning an inclusive byte range.
    pub fn range(lo: u8, hi: u8) -> Self {
        let mut s = Self::empty();
        for b in lo..=hi {
            s.add(b);
        }
        s
    }

    /// Insert a byte.
    pub fn add(&mut self, b: u8) {
        let w = (b as usize) / 64;
        let bit = (b as usize) % 64;
        self.bits[w] |= 1u64 << bit;
    }

    /// True when the class contains the byte.
    pub fn contains(&self, b: u8) -> bool {
        let w = (b as usize) / 64;
        let bit = (b as usize) % 64;
        (self.bits[w] >> bit) & 1 == 1
    }

    /// Union with another class (in place).
    pub fn union_with(&mut self, other: &ByteClass) {
        for i in 0..4 {
            self.bits[i] |= other.bits[i];
        }
    }

    /// Complement.
    pub fn complement(&self) -> ByteClass {
        Self {
            bits: [!self.bits[0], !self.bits[1], !self.bits[2], !self.bits[3]],
        }
    }
}

/// A transition out of a state.
#[derive(Clone, Debug)]
pub enum Edge {
    /// Move on any byte in the class.
    Class(ByteClass, StateId),
    /// Move without consuming input.
    Epsilon(StateId),
}

/// Index into [`Nfa::states`].
pub type StateId = u32;

/// Non-deterministic finite automaton over bytes with epsilon edges.
#[derive(Clone, Debug)]
pub struct Nfa {
    pub states: Vec<Vec<Edge>>,
    pub start: StateId,
    pub accepts: BTreeSet<StateId>,
}

impl Nfa {
    /// Build an empty NFA with no states.
    pub fn new() -> Self {
        Self {
            states: vec![],
            start: 0,
            accepts: BTreeSet::new(),
        }
    }

    /// Add a fresh state and return its id.
    pub fn add_state(&mut self) -> StateId {
        let id = self.states.len() as u32;
        self.states.push(vec![]);
        id
    }

    /// Add an edge.
    pub fn add_edge(&mut self, from: StateId, edge: Edge) {
        self.states[from as usize].push(edge);
    }

    /// Epsilon-closure of a set of states.
    pub fn epsilon_closure(&self, set: &BTreeSet<StateId>) -> BTreeSet<StateId> {
        let mut out = set.clone();
        let mut stack: Vec<StateId> = set.iter().copied().collect();
        while let Some(s) = stack.pop() {
            for edge in &self.states[s as usize] {
                if let Edge::Epsilon(t) = edge
                    && out.insert(*t)
                {
                    stack.push(*t);
                }
            }
        }
        out
    }

    /// Step every state in `set` on input byte `b`.
    pub fn step(&self, set: &BTreeSet<StateId>, b: u8) -> BTreeSet<StateId> {
        let mut next = BTreeSet::new();
        for s in set {
            for edge in &self.states[*s as usize] {
                if let Edge::Class(class, t) = edge
                    && class.contains(b)
                {
                    next.insert(*t);
                }
            }
        }
        self.epsilon_closure(&next)
    }

    /// True when any state in the set is accepting.
    pub fn any_accept(&self, set: &BTreeSet<StateId>) -> bool {
        set.iter().any(|s| self.accepts.contains(s))
    }

    /// Step a whole byte-string from a starting state set.
    /// Returns `None` when the run hits a dead end.
    pub fn run_bytes(&self, start: &BTreeSet<StateId>, bytes: &[u8]) -> Option<BTreeSet<StateId>> {
        let mut cur = start.clone();
        for b in bytes {
            cur = self.step(&cur, *b);
            if cur.is_empty() {
                return None;
            }
        }
        Some(cur)
    }

    /// The set of states reachable at NFA start, after epsilon-closure.
    pub fn start_set(&self) -> BTreeSet<StateId> {
        let mut s = BTreeSet::new();
        s.insert(self.start);
        self.epsilon_closure(&s)
    }
}

impl Default for Nfa {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_class_basic() {
        let mut c = ByteClass::empty();
        c.add(b'a');
        c.add(b'z');
        assert!(c.contains(b'a'));
        assert!(c.contains(b'z'));
        assert!(!c.contains(b'b'));
    }

    #[test]
    fn byte_class_range() {
        let c = ByteClass::range(b'0', b'9');
        for b in b'0'..=b'9' {
            assert!(c.contains(b));
        }
        assert!(!c.contains(b'/'));
        assert!(!c.contains(b':'));
    }

    #[test]
    fn nfa_single_byte_match() {
        // NFA matching only `a`.
        let mut nfa = Nfa::new();
        let s0 = nfa.add_state();
        let s1 = nfa.add_state();
        nfa.start = s0;
        nfa.accepts.insert(s1);
        nfa.add_edge(s0, Edge::Class(ByteClass::single(b'a'), s1));
        let start = nfa.start_set();
        let after = nfa.run_bytes(&start, b"a").unwrap();
        assert!(nfa.any_accept(&after));
        assert!(nfa.run_bytes(&start, b"b").is_none());
    }

    #[test]
    fn nfa_epsilon_closure() {
        // s0 -eps-> s1 -a-> s2 (accept).
        let mut nfa = Nfa::new();
        let s0 = nfa.add_state();
        let s1 = nfa.add_state();
        let s2 = nfa.add_state();
        nfa.start = s0;
        nfa.accepts.insert(s2);
        nfa.add_edge(s0, Edge::Epsilon(s1));
        nfa.add_edge(s1, Edge::Class(ByteClass::single(b'a'), s2));
        let start = nfa.start_set();
        assert!(start.contains(&s0));
        assert!(start.contains(&s1));
        let after = nfa.run_bytes(&start, b"a").unwrap();
        assert!(nfa.any_accept(&after));
    }
}
