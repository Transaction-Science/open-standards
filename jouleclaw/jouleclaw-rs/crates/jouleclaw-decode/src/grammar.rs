//! The `Grammar` and `Compiled` surfaces.
//!
//! `Grammar` is the constructor entry point. `Compiled` is the
//! state-machine form actually fed into the [`Decoder`]: either an NFA
//! over bytes (for regex / JSON-schema) or a CFG with an Earley state
//! tracker.

use serde_json::Value;

use crate::automaton::Nfa;
use crate::cfg::{CfgRule, CompiledCfg};
use crate::error::DecodeError;
use crate::json_schema;
use crate::regex_compile;

/// A compiled grammar form.
#[derive(Clone, Debug)]
pub enum Compiled {
    /// Byte-level NFA from regex or JSON-schema.
    Nfa(NfaCompiled),
    /// CFG with Earley scanning.
    Cfg(CompiledCfg),
}

/// NFA wrapped together with metadata.
#[derive(Clone, Debug)]
pub struct NfaCompiled {
    pub nfa: Nfa,
    pub source: String,
}

/// A grammar surface. Construct with one of the `from_*` methods, then
/// feed to a [`Decoder`].
#[derive(Clone, Debug)]
pub struct Grammar {
    compiled: Compiled,
}

impl Grammar {
    /// Compile a regex pattern (subset of the `regex` crate syntax —
    /// see [`regex_compile`] for the supported constructs).
    pub fn from_regex(pattern: &str) -> Result<Self, DecodeError> {
        let nfa = regex_compile::compile(pattern)?;
        Ok(Self {
            compiled: Compiled::Nfa(NfaCompiled {
                nfa,
                source: pattern.to_string(),
            }),
        })
    }

    /// Compile a JSON-schema subset. See [`json_schema`] for the
    /// supported constructs and v0.1 compromises.
    pub fn from_json_schema(schema: &Value) -> Result<Self, DecodeError> {
        let re = json_schema::schema_to_regex(schema)?;
        let nfa = regex_compile::compile(&re)?;
        Ok(Self {
            compiled: Compiled::Nfa(NfaCompiled { nfa, source: re }),
        })
    }

    /// Compile an explicit context-free grammar.
    pub fn from_cfg(rules: &[CfgRule]) -> Result<Self, DecodeError> {
        let cfg = CompiledCfg::from_rules(rules)?;
        Ok(Self {
            compiled: Compiled::Cfg(cfg),
        })
    }

    /// Pick the start non-terminal for a CFG grammar.
    pub fn with_start(self, start: impl Into<String>) -> Result<Self, DecodeError> {
        match self.compiled {
            Compiled::Cfg(cfg) => Ok(Self {
                compiled: Compiled::Cfg(cfg.with_start(start)?),
            }),
            Compiled::Nfa(_) => Err(DecodeError::Cfg(
                "with_start is only meaningful for CFG grammars".into(),
            )),
        }
    }

    /// Borrow the compiled form.
    pub fn compiled(&self) -> &Compiled {
        &self.compiled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::CfgSymbol;
    use serde_json::json;

    #[test]
    fn regex_constructor() {
        Grammar::from_regex(r"[0-9]+").expect("good pattern");
    }

    #[test]
    fn bad_regex_errors() {
        let err = Grammar::from_regex("[unterminated").unwrap_err();
        match err {
            DecodeError::RegexParse { .. } => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn json_schema_constructor() {
        Grammar::from_json_schema(&json!({"type":"integer"})).expect("good schema");
    }

    #[test]
    fn cfg_constructor() {
        let rules = vec![crate::cfg::CfgRule::new(
            "S",
            vec![CfgSymbol::byte(b'a')],
        )];
        Grammar::from_cfg(&rules).expect("good cfg");
    }
}
