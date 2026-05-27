//! Context-free grammar surface with a byte-level Earley recogniser.
//!
//! Terminals are byte classes. Non-terminals are named by `String`.
//! The start non-terminal is conventionally `"start"` unless the
//! caller picks otherwise.
//!
//! The recogniser is a textbook Earley: scan / predict / complete on
//! an `EarleySet` per input position. We carry the Earley state inside
//! the decoder so that incremental token-stepping is O(grammar) per
//! input byte.

use std::collections::{BTreeMap, BTreeSet};

use crate::automaton::ByteClass;
use crate::error::DecodeError;

/// A symbol inside a CFG production.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CfgSymbol {
    /// A literal byte class (terminal).
    Term(ByteClass),
    /// A reference to another non-terminal by name.
    NonTerm(String),
}

impl CfgSymbol {
    /// Convenience: terminal matching exactly one byte.
    pub fn byte(b: u8) -> Self {
        CfgSymbol::Term(ByteClass::single(b))
    }
    /// Convenience: terminal matching a byte range.
    pub fn range(lo: u8, hi: u8) -> Self {
        CfgSymbol::Term(ByteClass::range(lo, hi))
    }
    /// Convenience: a non-terminal reference.
    pub fn nt(name: impl Into<String>) -> Self {
        CfgSymbol::NonTerm(name.into())
    }
}

/// A single production `lhs -> rhs`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CfgRule {
    pub lhs: String,
    pub rhs: Vec<CfgSymbol>,
}

impl CfgRule {
    /// Shortcut constructor.
    pub fn new(lhs: impl Into<String>, rhs: Vec<CfgSymbol>) -> Self {
        Self {
            lhs: lhs.into(),
            rhs,
        }
    }
}

/// Compiled CFG: rules grouped by lhs and indexed by integer id.
#[derive(Clone, Debug)]
pub struct CompiledCfg {
    pub rules: Vec<CfgRule>,
    pub by_lhs: BTreeMap<String, Vec<usize>>,
    pub start: String,
}

impl CompiledCfg {
    /// Construct from a slice of rules. The first rule's lhs is the start.
    pub fn from_rules(rules: &[CfgRule]) -> Result<Self, DecodeError> {
        if rules.is_empty() {
            return Err(DecodeError::Cfg("no rules supplied".into()));
        }
        let start = rules[0].lhs.clone();
        let mut by_lhs: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, r) in rules.iter().enumerate() {
            by_lhs.entry(r.lhs.clone()).or_default().push(i);
        }
        // Check all non-terminal references resolve.
        for r in rules {
            for sym in &r.rhs {
                if let CfgSymbol::NonTerm(name) = sym
                    && !by_lhs.contains_key(name)
                {
                    return Err(DecodeError::Cfg(format!(
                        "non-terminal `{name}` referenced but never defined"
                    )));
                }
            }
        }
        Ok(Self {
            rules: rules.to_vec(),
            by_lhs,
            start,
        })
    }

    /// Set the start non-terminal explicitly.
    pub fn with_start(mut self, start: impl Into<String>) -> Result<Self, DecodeError> {
        let s = start.into();
        if !self.by_lhs.contains_key(&s) {
            return Err(DecodeError::Cfg(format!(
                "start non-terminal `{s}` has no productions"
            )));
        }
        self.start = s;
        Ok(self)
    }
}

/// One Earley item: `(rule_id, dot, origin)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Item {
    pub rule: u32,
    pub dot: u32,
    pub origin: u32,
}

/// One position's worth of Earley items.
#[derive(Clone, Debug, Default)]
pub struct EarleySet {
    pub items: BTreeSet<Item>,
}

/// Earley state — one set per input position.
#[derive(Clone, Debug)]
pub struct EarleyState {
    pub sets: Vec<EarleySet>,
}

impl EarleyState {
    /// Start with the seed set at position 0.
    pub fn new(cfg: &CompiledCfg) -> Self {
        let mut set0 = EarleySet::default();
        if let Some(rule_ids) = cfg.by_lhs.get(&cfg.start) {
            for &rid in rule_ids {
                set0.items.insert(Item {
                    rule: rid as u32,
                    dot: 0,
                    origin: 0,
                });
            }
        }
        let mut state = Self { sets: vec![set0] };
        Self::close(cfg, &mut state, 0);
        state
    }

    /// Predict + complete on set `i` until fixpoint.
    fn close(cfg: &CompiledCfg, state: &mut EarleyState, i: usize) {
        loop {
            let snapshot: Vec<Item> = state.sets[i].items.iter().copied().collect();
            let mut changed = false;
            for item in snapshot {
                let rule = &cfg.rules[item.rule as usize];
                if (item.dot as usize) < rule.rhs.len() {
                    // Predict.
                    if let CfgSymbol::NonTerm(name) = &rule.rhs[item.dot as usize]
                        && let Some(rule_ids) = cfg.by_lhs.get(name)
                    {
                        for &rid in rule_ids {
                            let new = Item {
                                rule: rid as u32,
                                dot: 0,
                                origin: i as u32,
                            };
                            if state.sets[i].items.insert(new) {
                                changed = true;
                            }
                        }
                    }
                } else {
                    // Complete: for every item in set[origin] whose dot is on this lhs,
                    // advance the dot and add to set[i].
                    let lhs = &rule.lhs;
                    let origin = item.origin as usize;
                    let parents: Vec<Item> = state.sets[origin].items.iter().copied().collect();
                    for parent in parents {
                        let prule = &cfg.rules[parent.rule as usize];
                        if (parent.dot as usize) < prule.rhs.len()
                            && let CfgSymbol::NonTerm(pname) = &prule.rhs[parent.dot as usize]
                            && pname == lhs
                        {
                            let new = Item {
                                rule: parent.rule,
                                dot: parent.dot + 1,
                                origin: parent.origin,
                            };
                            if state.sets[i].items.insert(new) {
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    /// Scan one byte. Returns a new state with an extra set, or `None` on dead end.
    pub fn step(&self, cfg: &CompiledCfg, b: u8) -> Option<Self> {
        let i = self.sets.len() - 1;
        let mut next_set = EarleySet::default();
        for item in &self.sets[i].items {
            let rule = &cfg.rules[item.rule as usize];
            if (item.dot as usize) < rule.rhs.len()
                && let CfgSymbol::Term(class) = &rule.rhs[item.dot as usize]
                && class.contains(b)
            {
                next_set.items.insert(Item {
                    rule: item.rule,
                    dot: item.dot + 1,
                    origin: item.origin,
                });
            }
        }
        if next_set.items.is_empty() {
            return None;
        }
        let mut new_state = self.clone();
        new_state.sets.push(next_set);
        Self::close(cfg, &mut new_state, i + 1);
        Some(new_state)
    }

    /// True when some completed item spans the entire input under the start NT.
    pub fn is_accept(&self, cfg: &CompiledCfg) -> bool {
        let i = self.sets.len() - 1;
        for item in &self.sets[i].items {
            let rule = &cfg.rules[item.rule as usize];
            if rule.lhs == cfg.start
                && item.dot as usize == rule.rhs.len()
                && item.origin == 0
            {
                return true;
            }
        }
        false
    }

    /// True when no further input could possibly lead to accept (i.e. the
    /// current set has no items with the dot before a symbol).
    pub fn is_dead(&self) -> bool {
        let i = self.sets.len() - 1;
        self.sets[i].items.is_empty()
    }

    /// All terminal byte classes that would be acceptable as the next byte.
    pub fn permitted_next(&self, cfg: &CompiledCfg) -> ByteClass {
        let i = self.sets.len() - 1;
        let mut acc = ByteClass::empty();
        for item in &self.sets[i].items {
            let rule = &cfg.rules[item.rule as usize];
            if (item.dot as usize) < rule.rhs.len()
                && let CfgSymbol::Term(class) = &rule.rhs[item.dot as usize]
            {
                acc.union_with(class);
            }
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paren_balanced_grammar() {
        // S -> ( S ) | ε
        let rules = vec![
            CfgRule::new(
                "S",
                vec![CfgSymbol::byte(b'('), CfgSymbol::nt("S"), CfgSymbol::byte(b')')],
            ),
            CfgRule::new("S", vec![]),
        ];
        let cfg = CompiledCfg::from_rules(&rules).unwrap();
        let mut state = EarleyState::new(&cfg);
        assert!(state.is_accept(&cfg)); // empty is OK
        for b in b"(())" {
            state = state.step(&cfg, *b).expect("should advance");
        }
        assert!(state.is_accept(&cfg));
    }

    #[test]
    fn paren_imbalanced_rejected() {
        let rules = vec![
            CfgRule::new(
                "S",
                vec![CfgSymbol::byte(b'('), CfgSymbol::nt("S"), CfgSymbol::byte(b')')],
            ),
            CfgRule::new("S", vec![]),
        ];
        let cfg = CompiledCfg::from_rules(&rules).unwrap();
        let mut state = EarleyState::new(&cfg);
        for b in b"((" {
            state = state.step(&cfg, *b).expect("advance");
        }
        // Not yet accepting: still need closing parens.
        assert!(!state.is_accept(&cfg));
        // A `)` should advance one, still not accepting.
        state = state.step(&cfg, b')').unwrap();
        assert!(!state.is_accept(&cfg));
        // No further input allowed for `(((`.
        let mut s2 = EarleyState::new(&cfg);
        for b in b"((" {
            s2 = s2.step(&cfg, *b).unwrap();
        }
        // After "((", attempt `x` — dead.
        assert!(s2.step(&cfg, b'x').is_none());
    }

    #[test]
    fn permitted_next_basic() {
        let rules = vec![CfgRule::new(
            "S",
            vec![CfgSymbol::byte(b'a'), CfgSymbol::byte(b'b')],
        )];
        let cfg = CompiledCfg::from_rules(&rules).unwrap();
        let state = EarleyState::new(&cfg);
        let next = state.permitted_next(&cfg);
        assert!(next.contains(b'a'));
        assert!(!next.contains(b'b'));
    }

    #[test]
    fn undefined_nonterm_rejected() {
        let rules = vec![CfgRule::new("S", vec![CfgSymbol::nt("Missing")])];
        let err = CompiledCfg::from_rules(&rules).unwrap_err();
        match err {
            DecodeError::Cfg(_) => {}
            _ => panic!("expected Cfg error"),
        }
    }
}
