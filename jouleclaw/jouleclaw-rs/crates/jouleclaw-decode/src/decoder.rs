//! Stateful decoder.
//!
//! Tracks the current parse position inside the grammar, exposes the
//! per-step `TokenMask`, and advances the state when the caller commits
//! a token id.
//!
//! ## Per-step mask projection
//!
//! For an NFA grammar, a token `t` (a byte string `bytes(t)`) is allowed
//! iff `run_bytes(active_state, bytes(t))` does not end in a dead set.
//!
//! For a CFG grammar, a token `t` is allowed iff replaying `bytes(t)`
//! through the Earley engine never produces an empty set.
//!
//! Acceptance: an NFA grammar accepts when the current state set
//! intersects the accept set. A CFG accepts when an Earley item
//! covering the full input under the start symbol is present.

use std::collections::BTreeSet;

use crate::automaton::StateId;
use crate::cfg::EarleyState;
use crate::error::DecodeError;
use crate::grammar::{Compiled, Grammar};
use crate::mask::TokenMask;

/// Stateful grammar-constrained decoder.
#[derive(Debug)]
pub struct Decoder {
    grammar: Grammar,
    vocab: Vec<Vec<u8>>,
    state: DecoderState,
    cached_mask: Option<TokenMask>,
}

#[derive(Clone, Debug)]
enum DecoderState {
    Nfa { active: BTreeSet<StateId> },
    Cfg { earley: EarleyState },
}

impl Decoder {
    /// Build a decoder from a compiled grammar and a vocabulary.
    pub fn new(grammar: Grammar, vocab: Vec<Vec<u8>>) -> Result<Self, DecodeError> {
        if vocab.is_empty() {
            return Err(DecodeError::EmptyVocabulary);
        }
        let state = match grammar.compiled() {
            Compiled::Nfa(n) => DecoderState::Nfa {
                active: n.nfa.start_set(),
            },
            Compiled::Cfg(cfg) => DecoderState::Cfg {
                earley: EarleyState::new(cfg),
            },
        };
        let mut dec = Self {
            grammar,
            vocab,
            state,
            cached_mask: None,
        };
        dec.recompute_mask();
        Ok(dec)
    }

    /// True when the current state accepts the input so far.
    pub fn is_accept(&self) -> bool {
        match (&self.grammar.compiled(), &self.state) {
            (Compiled::Nfa(n), DecoderState::Nfa { active }) => n.nfa.any_accept(active),
            (Compiled::Cfg(cfg), DecoderState::Cfg { earley }) => earley.is_accept(cfg),
            _ => false,
        }
    }

    /// True when no further input could lead to accept.
    pub fn is_dead(&self) -> bool {
        match (&self.grammar.compiled(), &self.state) {
            (Compiled::Nfa(_), DecoderState::Nfa { active }) => active.is_empty(),
            (Compiled::Cfg(_), DecoderState::Cfg { earley }) => earley.is_dead(),
            _ => true,
        }
    }

    /// Borrow the current token mask. Recomputed lazily after `step`.
    pub fn current_mask(&self) -> &TokenMask {
        self.cached_mask
            .as_ref()
            .expect("mask should be cached by Decoder::new / Decoder::step")
    }

    /// Vocabulary size.
    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    /// Commit a token id: advance the state machine.
    pub fn step(&mut self, token_id: u32) -> Result<(), DecodeError> {
        if (token_id as usize) >= self.vocab.len() {
            return Err(DecodeError::TokenOutOfRange(token_id, self.vocab.len()));
        }
        if !self.current_mask().allowed(token_id) {
            return Err(DecodeError::TokenNotAllowed(token_id));
        }
        let bytes = self.vocab[token_id as usize].clone();
        match (&self.grammar.compiled(), &mut self.state) {
            (Compiled::Nfa(n), DecoderState::Nfa { active }) => {
                match n.nfa.run_bytes(active, &bytes) {
                    Some(next) => {
                        *active = next;
                    }
                    None => return Err(DecodeError::TokenNotAllowed(token_id)),
                }
            }
            (Compiled::Cfg(cfg), DecoderState::Cfg { earley }) => {
                let mut cur = earley.clone();
                for b in &bytes {
                    cur = match cur.step(cfg, *b) {
                        Some(s) => s,
                        None => return Err(DecodeError::TokenNotAllowed(token_id)),
                    };
                }
                *earley = cur;
            }
            _ => return Err(DecodeError::TokenNotAllowed(token_id)),
        }
        self.recompute_mask();
        Ok(())
    }

    /// Recompute the per-token allowed mask.
    fn recompute_mask(&mut self) {
        let mut mask = TokenMask::new(self.vocab.len());
        match (&self.grammar.compiled(), &self.state) {
            (Compiled::Nfa(n), DecoderState::Nfa { active }) => {
                if active.is_empty() {
                    self.cached_mask = Some(mask);
                    return;
                }
                for (i, tok) in self.vocab.iter().enumerate() {
                    if tok.is_empty() {
                        // Empty token is allowed when current state isn't dead.
                        mask.allow(i as u32);
                        continue;
                    }
                    if n.nfa.run_bytes(active, tok).is_some() {
                        mask.allow(i as u32);
                    }
                }
            }
            (Compiled::Cfg(cfg), DecoderState::Cfg { earley }) => {
                if earley.is_dead() {
                    self.cached_mask = Some(mask);
                    return;
                }
                for (i, tok) in self.vocab.iter().enumerate() {
                    if tok.is_empty() {
                        mask.allow(i as u32);
                        continue;
                    }
                    let mut ok = true;
                    let mut cur = earley.clone();
                    for b in tok {
                        match cur.step(cfg, *b) {
                            Some(next) => cur = next,
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        mask.allow(i as u32);
                    }
                }
            }
            _ => {}
        }
        self.cached_mask = Some(mask);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{CfgRule, CfgSymbol};
    use serde_json::json;

    fn vocab(strs: &[&str]) -> Vec<Vec<u8>> {
        strs.iter().map(|s| s.as_bytes().to_vec()).collect()
    }

    #[test]
    fn regex_mask_filters_disallowed_tokens() {
        let g = Grammar::from_regex(r"[0-9]+").unwrap();
        let v = vocab(&["0", "1", "9", "a", "12", "5x"]);
        let dec = Decoder::new(g, v).unwrap();
        let m = dec.current_mask();
        assert!(m.allowed(0));
        assert!(m.allowed(1));
        assert!(m.allowed(2));
        assert!(!m.allowed(3));
        assert!(m.allowed(4));
        assert!(!m.allowed(5));
    }

    #[test]
    fn step_advances_state() {
        let g = Grammar::from_regex(r"hello").unwrap();
        let v = vocab(&["h", "e", "l", "lo", "x", "hello"]);
        let mut dec = Decoder::new(g, v).unwrap();
        dec.step(0).unwrap(); // "h"
        dec.step(1).unwrap(); // "e"
        dec.step(2).unwrap(); // "l"
        dec.step(2).unwrap(); // "l"
        // After "hell" the only next token is "o" (id 1 -> 'e' not allowed,
        // we picked "lo" earlier so we need just "o"; but we don't have it,
        // we only have "lo"). So `lo` would have produced "hellolo" — wrong.
        // Sanity: at this point mask should disallow id 5 ("hello") since
        // we've already consumed "hell".
        let m = dec.current_mask();
        assert!(!m.allowed(5));
    }

    #[test]
    fn is_accept_after_full_string() {
        let g = Grammar::from_regex(r"hello").unwrap();
        let v = vocab(&["hello", "h"]);
        let mut dec = Decoder::new(g, v).unwrap();
        assert!(!dec.is_accept());
        dec.step(0).unwrap();
        assert!(dec.is_accept());
    }

    #[test]
    fn stepping_disallowed_token_errs() {
        let g = Grammar::from_regex(r"\d+").unwrap();
        let v = vocab(&["1", "a"]);
        let mut dec = Decoder::new(g, v).unwrap();
        let err = dec.step(1).unwrap_err();
        match err {
            DecodeError::TokenNotAllowed(1) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn out_of_range_token_errs() {
        let g = Grammar::from_regex(r".").unwrap();
        let v = vocab(&["x"]);
        let mut dec = Decoder::new(g, v).unwrap();
        let err = dec.step(99).unwrap_err();
        match err {
            DecodeError::TokenOutOfRange(99, 1) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn cfg_paren_decoder() {
        // S -> ( S ) | ε
        let rules = vec![
            CfgRule::new(
                "S",
                vec![
                    CfgSymbol::byte(b'('),
                    CfgSymbol::nt("S"),
                    CfgSymbol::byte(b')'),
                ],
            ),
            CfgRule::new("S", vec![]),
        ];
        let g = Grammar::from_cfg(&rules).unwrap();
        let v = vocab(&["(", ")", "()", "(()", "x"]);
        let mut dec = Decoder::new(g, v).unwrap();
        // Initially can open or close (since S→ε means we're at accept,
        // but we can't *close* before opening) — let's verify.
        let m = dec.current_mask();
        assert!(m.allowed(0)); // "("
        assert!(!m.allowed(1)); // ")"
        assert!(m.allowed(2)); // "()"
        assert!(!m.allowed(4)); // "x"
        assert!(dec.is_accept()); // empty is OK
        dec.step(0).unwrap(); // "("
        assert!(!dec.is_accept());
        dec.step(1).unwrap(); // ")"
        assert!(dec.is_accept());
    }

    #[test]
    fn json_schema_decoder_integer() {
        let g = Grammar::from_json_schema(&json!({"type":"integer"})).unwrap();
        let v = vocab(&["-", "0", "1", "2", "a", "-7"]);
        let dec = Decoder::new(g, v).unwrap();
        let m = dec.current_mask();
        assert!(m.allowed(0)); // '-'
        assert!(m.allowed(1)); // '0'
        assert!(m.allowed(2)); // '1'
        assert!(!m.allowed(4)); // 'a'
        assert!(m.allowed(5)); // "-7"
    }

    #[test]
    fn empty_vocab_errs() {
        let g = Grammar::from_regex(".").unwrap();
        let err = Decoder::new(g, vec![]).unwrap_err();
        match err {
            DecodeError::EmptyVocabulary => {}
            other => panic!("unexpected: {other:?}"),
        }
    }
}
