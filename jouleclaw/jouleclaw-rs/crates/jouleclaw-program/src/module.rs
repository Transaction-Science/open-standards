//! Module trait and the three built-in modules: [`Predict`],
//! [`ChainOfThought`], and [`ProgramOfThought`].
//!
//! A module declares a [`Signature`] and a kind. The compiler reads both to
//! lower the module into one or more [`Dispatch`](crate::compiler::Dispatch)
//! entries; the runner reads the kind to know which backend pattern to use.
//!
//! ## Backends and code runners
//!
//! The runner doesn't know how to call a model. It calls a [`Backend`] for
//! each model-tier dispatch and a [`CodeRunner`] for each
//! `ProgramOfThought` code-execution step. Both are caller-supplied; this
//! crate ships no sandbox.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::grammar::GrammarHandle;
use crate::record::Record;
use crate::signature::{Field, Signature};

/// One module in a program.
pub trait Module {
    /// The module's signature (read by the compiler).
    fn signature(&self) -> &Signature;

    /// Which built-in module pattern this is. The compiler uses this to
    /// produce the right [`Dispatch`](crate::compiler::Dispatch) shape.
    fn kind(&self) -> ModuleKind;

    /// Render this module's prompt for model-tier dispatches.
    ///
    /// The default implementation produces the canonical DSPy-style prompt:
    /// instruction + numbered inputs + numbered outputs. Backends are free to
    /// ignore it.
    fn render_prompt(&self) -> String {
        render_default_prompt(self.signature(), self.kind())
    }

    /// The *effective* output signature for this module — i.e. the one the
    /// compiler hands the backend.
    ///
    /// For `Predict` and `ProgramOfThought` this is the declared signature.
    /// For `ChainOfThought` an unobserved `reasoning` field is inserted
    /// before the declared outputs.
    fn effective_signature(&self) -> Signature {
        let mut sig = self.signature().clone();
        if matches!(self.kind(), ModuleKind::ChainOfThought) {
            let reasoning = Field::text(
                "reasoning",
                "step-by-step reasoning toward the final answer (unobserved)",
            );
            // Insert at the front of outputs so the model writes it before
            // committing to the final fields.
            sig.outputs.insert(0, reasoning);
        }
        sig
    }
}

/// Kinds of built-in module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModuleKind {
    Predict,
    ChainOfThought,
    ProgramOfThought,
}

fn render_default_prompt(sig: &Signature, kind: ModuleKind) -> String {
    let mut out = String::new();
    out.push_str(&sig.instruction);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push('\n');
    out.push_str("Given the inputs, produce the outputs.\n\n");

    out.push_str("Inputs:\n");
    for (i, f) in sig.inputs.iter().enumerate() {
        out.push_str(&format!(
            "  {}. {} ({}): {}\n",
            i + 1,
            f.name,
            f.ty.label(),
            f.description
        ));
    }
    out.push('\n');

    out.push_str("Outputs:\n");
    if matches!(kind, ModuleKind::ChainOfThought) {
        out.push_str(
            "  1. reasoning (text): step-by-step reasoning toward the final answer (unobserved)\n",
        );
        for (i, f) in sig.outputs.iter().enumerate() {
            out.push_str(&format!(
                "  {}. {} ({}): {}\n",
                i + 2,
                f.name,
                f.ty.label(),
                f.description
            ));
        }
    } else {
        for (i, f) in sig.outputs.iter().enumerate() {
            out.push_str(&format!(
                "  {}. {} ({}): {}\n",
                i + 1,
                f.name,
                f.ty.label(),
                f.description
            ));
        }
        if matches!(kind, ModuleKind::ProgramOfThought) {
            out.push_str(
                "  +. code (text): a self-contained code snippet whose stdout produces the \
                 declared outputs in JSON form\n",
            );
        }
    }
    out
}

/// Single typed model call. Equivalent to `dspy.Predict`.
#[derive(Debug, Clone)]
pub struct Predict {
    pub signature: Signature,
}

impl Predict {
    pub fn new(signature: Signature) -> Self {
        Self { signature }
    }
}

impl Module for Predict {
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn kind(&self) -> ModuleKind {
        ModuleKind::Predict
    }
}

/// Adds an unobserved `reasoning` field before the declared outputs.
/// Equivalent to `dspy.ChainOfThought`.
#[derive(Debug, Clone)]
pub struct ChainOfThought {
    pub signature: Signature,
}

impl ChainOfThought {
    pub fn new(signature: Signature) -> Self {
        Self { signature }
    }
}

impl Module for ChainOfThought {
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn kind(&self) -> ModuleKind {
        ModuleKind::ChainOfThought
    }
}

/// Outputs include a `code` field that is executed by a caller-supplied
/// [`CodeRunner`]. Equivalent to `dspy.ProgramOfThought`.
#[derive(Debug, Clone)]
pub struct ProgramOfThought {
    pub signature: Signature,
}

impl ProgramOfThought {
    pub fn new(signature: Signature) -> Self {
        Self { signature }
    }
}

impl Module for ProgramOfThought {
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn kind(&self) -> ModuleKind {
        ModuleKind::ProgramOfThought
    }
}

/// One model-tier call produced by a compiled dispatch.
///
/// The runner builds this from a [`Dispatch`](crate::compiler::Dispatch) plus
/// the current [`Record`] and hands it to a [`Backend`].
#[derive(Debug, Clone)]
pub struct BackendCall<'a> {
    pub module_name: &'a str,
    pub module_kind: ModuleKind,
    pub signature: &'a Signature,
    pub effective_signature: &'a Signature,
    pub prompt: &'a str,
    pub grammar: &'a GrammarHandle,
    pub inputs: &'a Record,
}

/// Caller-supplied backend that performs model-tier dispatches.
///
/// Backends own the choice of tier (cache / lawful / embed / model / wire) and
/// the choice of model when they fall through to L3+. v0.1 keeps this trait
/// minimal — caller hands back a record matching the effective output
/// signature, or an error string.
pub trait Backend {
    fn call(&self, call: BackendCall<'_>) -> core::result::Result<BackendResponse, String>;
}

/// What a [`Backend`] returns for a single [`BackendCall`].
#[derive(Debug, Clone, Default)]
pub struct BackendResponse {
    /// The fields the model produced, keyed by output-field name.
    pub outputs: Record,
    /// Optional raw JSON the model emitted, kept for telemetry.
    pub raw: Option<JsonValue>,
}

/// Caller-supplied executor for `ProgramOfThought` code fields.
///
/// JouleClaw does not ship a sandbox. This trait is the integration seam: the
/// caller wires in whatever execution surface they trust (a WASM runtime, a
/// subprocess, a pure interpreter, …) and returns the resulting output
/// record. Type-checking against the declared signature happens after the
/// runner gets the record back.
pub trait CodeRunner {
    fn run(&self, code: &str, expected_outputs: &[Field]) -> core::result::Result<Record, String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qa_sig() -> Signature {
        Signature::new(
            "qa",
            "Answer the question.",
            vec![Field::text("question", "the question")],
            vec![Field::text("answer", "the answer")],
        )
    }

    #[test]
    fn predict_effective_sig_equals_declared() {
        let m = Predict::new(qa_sig());
        assert_eq!(m.effective_signature(), qa_sig());
        assert_eq!(m.kind(), ModuleKind::Predict);
    }

    #[test]
    fn cot_effective_sig_inserts_reasoning_first() {
        let m = ChainOfThought::new(qa_sig());
        let eff = m.effective_signature();
        assert_eq!(eff.outputs[0].name, "reasoning");
        assert_eq!(eff.outputs[1].name, "answer");
    }

    #[test]
    fn pot_effective_sig_equals_declared() {
        let m = ProgramOfThought::new(qa_sig());
        assert_eq!(m.effective_signature(), qa_sig());
        assert_eq!(m.kind(), ModuleKind::ProgramOfThought);
    }

    #[test]
    fn rendered_prompt_includes_instruction_and_fields() {
        let m = Predict::new(qa_sig());
        let p = m.render_prompt();
        assert!(p.contains("Answer the question."));
        assert!(p.contains("question (text)"));
        assert!(p.contains("answer (text)"));
        assert!(p.contains("Inputs:"));
        assert!(p.contains("Outputs:"));
    }

    #[test]
    fn cot_prompt_mentions_reasoning() {
        let m = ChainOfThought::new(qa_sig());
        let p = m.render_prompt();
        assert!(p.contains("reasoning (text)"));
    }

    #[test]
    fn pot_prompt_mentions_code() {
        let m = ProgramOfThought::new(qa_sig());
        let p = m.render_prompt();
        assert!(p.contains("code (text)"));
    }
}
