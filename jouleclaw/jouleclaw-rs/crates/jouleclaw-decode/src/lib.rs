//! # jouleclaw-decode
//!
//! Grammar-constrained decoding for JouleClaw. Compiles a grammar
//! (regex, CFG, or a JSON-schema subset) into a byte-level automaton,
//! then projects per-token vocabulary masks that forbid any token whose
//! byte sequence would lead the automaton off-grammar.
//!
//! ## Where this fits in JouleClaw
//!
//! The cascade prefers L0/L1/L2 deterministic resolution. When L3/L4
//! model inference does fire, the answer must conform to a typed
//! schema so upstream tools can parse it. This crate turns "ask the
//! model nicely" into "the model cannot emit a token that violates
//! the grammar."
//!
//! ## Usage
//!
//! ```
//! use jouleclaw_decode::{Grammar, Decoder};
//!
//! let grammar = Grammar::from_regex(r"[0-9]+").unwrap();
//! let vocab: Vec<Vec<u8>> = vec![
//!     b"1".to_vec(),
//!     b"2".to_vec(),
//!     b"a".to_vec(),
//!     b"42".to_vec(),
//! ];
//! let mut dec = Decoder::new(grammar, vocab).unwrap();
//! let mask = dec.current_mask();
//! assert!(mask.allowed(0));    // "1"
//! assert!(mask.allowed(1));    // "2"
//! assert!(!mask.allowed(2));   // "a"  — not a digit
//! assert!(mask.allowed(3));    // "42"
//! ```

#![forbid(unsafe_code)]
#![deny(missing_debug_implementations)]

pub mod automaton;
pub mod cfg;
pub mod decoder;
pub mod error;
pub mod grammar;
pub mod json_schema;
pub mod mask;
pub mod regex_compile;

pub use crate::cfg::{CfgRule, CfgSymbol};
pub use crate::decoder::Decoder;
pub use crate::error::DecodeError;
pub use crate::grammar::{Compiled, Grammar};
pub use crate::mask::TokenMask;
