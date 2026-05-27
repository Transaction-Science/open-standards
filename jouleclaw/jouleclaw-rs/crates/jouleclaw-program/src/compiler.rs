//! Lower a [`Program`] into a flat sequence of [`Dispatch`]es.
//!
//! For each module in topological order, the compiler emits one dispatch
//! containing:
//!
//! * the module's effective signature (with reasoning prepended for
//!   ChainOfThought),
//! * the rendered prompt,
//! * a JSON Schema grammar surface,
//! * the argument bindings — for each input field, where the value comes from
//!   (an upstream module's output, or the program's externally-supplied
//!   input record).
//!
//! The runner takes a [`Compiled`] plan and walks it: model-tier dispatches
//! call the [`Backend`](crate::module::Backend); `ProgramOfThought`
//! dispatches additionally feed the resulting `code` field through a
//! [`CodeRunner`](crate::module::CodeRunner).

use std::collections::HashMap;

use serde_json::Value as JsonValue;

use crate::error::{Error, Result};
use crate::grammar::GrammarHandle;
use crate::module::ModuleKind;
use crate::program::{Port, Program};
use crate::signature::Signature;

/// How a single input field gets bound at runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgBinding {
    /// From an upstream module's output, by `(module, field)`.
    FromOutput(Port),
    /// From the externally-supplied program input record, by field name.
    FromProgramInput(String),
}

/// One dispatch in the compiled plan.
#[derive(Debug, Clone)]
pub struct Dispatch {
    /// Module name (matches the original `Program`).
    pub module_name: String,
    /// What kind of module this is — picked up by the runner.
    pub kind: ModuleKind,
    /// The declared signature, exactly as written by the caller.
    pub signature: Signature,
    /// The effective signature (with reasoning prepended for ChainOfThought).
    pub effective_signature: Signature,
    /// Rendered prompt for model-tier backends.
    pub prompt: String,
    /// Grammar handle (JSON Schema by default).
    pub grammar: GrammarHandle,
    /// For each input field on the *declared* signature: where to fetch it.
    pub args: HashMap<String, ArgBinding>,
}

/// A compiled program: the dispatch list plus the set of input field names
/// the caller must supply.
#[derive(Debug, Clone)]
pub struct Compiled {
    pub dispatches: Vec<Dispatch>,
    /// Program-level input fields (the set of `FromProgramInput` keys
    /// referenced by any dispatch).
    pub program_inputs: Vec<String>,
    /// Program-level output fields (the final module's declared outputs).
    pub program_outputs: Vec<String>,
}

impl Compiled {
    /// Render the full grammar surface as a single JSON Schema.
    ///
    /// The schema is an `object` keyed by `module_name`, each value being the
    /// effective-signature object schema. Useful for one-shot inspection,
    /// telemetry, and external grammar-constrained decoders.
    pub fn compile_to_jsonschema(&self) -> JsonValue {
        let mut props = serde_json::Map::new();
        for d in &self.dispatches {
            props.insert(d.module_name.clone(), signature_object_schema(&d.effective_signature));
        }
        serde_json::json!({
            "type": "object",
            "properties": props,
            "additionalProperties": false,
        })
    }
}

/// Compiler for [`Program`] -> [`Compiled`].
#[derive(Debug, Default, Clone, Copy)]
pub struct Compiler;

impl Compiler {
    pub fn new() -> Self {
        Self
    }

    /// Lower a program into its dispatch plan.
    pub fn compile(self, program: &Program) -> Result<Compiled> {
        program.validate()?;
        let order = program.topo_order()?;

        // Build a quick `module_name -> &NamedModule` index.
        let mut by_name = HashMap::new();
        for m in &program.modules {
            by_name.insert(m.name.clone(), m);
        }

        // For each `to.module.to.field`, the originating port.
        let mut incoming: HashMap<(String, String), Port> = HashMap::new();
        for e in &program.flow {
            incoming.insert(
                (e.to.module.clone(), e.to.field.clone()),
                e.from.clone(),
            );
        }

        let mut dispatches = Vec::new();
        let mut program_input_names = Vec::new();
        let mut seen_program_inputs = std::collections::HashSet::new();

        for name in &order {
            let nm = by_name.get(name).ok_or(Error::UnknownModule {
                module: name.clone(),
            })?;
            let module = &nm.module;
            let declared = module.signature().clone();
            let effective = module.effective_signature();
            let prompt = module.render_prompt();
            let schema = signature_object_schema(&effective);
            let grammar = GrammarHandle::new(schema, format!("{}::{}", name, declared.name));

            let mut args = HashMap::new();
            for f in &declared.inputs {
                let key = (name.clone(), f.name.clone());
                if let Some(from) = incoming.get(&key) {
                    args.insert(f.name.clone(), ArgBinding::FromOutput(from.clone()));
                } else {
                    // Program-level input. We use the input field name
                    // directly — callers supplying e.g. two modules that both
                    // accept `question` will share that input.
                    args.insert(
                        f.name.clone(),
                        ArgBinding::FromProgramInput(f.name.clone()),
                    );
                    if seen_program_inputs.insert(f.name.clone()) {
                        program_input_names.push(f.name.clone());
                    }
                }
            }

            dispatches.push(Dispatch {
                module_name: name.clone(),
                kind: module.kind(),
                signature: declared,
                effective_signature: effective,
                prompt,
                grammar,
                args,
            });
        }

        // Program outputs = declared outputs of the last topo-ordered
        // module. Multi-output programs should reach for a different shape;
        // v0.1 keeps it simple.
        let program_outputs = dispatches
            .last()
            .map(|d| d.signature.outputs.iter().map(|f| f.name.clone()).collect())
            .unwrap_or_default();

        Ok(Compiled {
            dispatches,
            program_inputs: program_input_names,
            program_outputs,
        })
    }
}

fn signature_object_schema(sig: &Signature) -> JsonValue {
    let mut props = serde_json::Map::new();
    let mut required = Vec::new();
    for f in &sig.outputs {
        props.insert(f.name.clone(), {
            let mut s = f.ty.to_jsonschema();
            if let Some(obj) = s.as_object_mut() {
                obj.insert("description".into(), JsonValue::String(f.description.clone()));
            }
            s
        });
        required.push(JsonValue::String(f.name.clone()));
    }
    serde_json::json!({
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::{ChainOfThought, Predict};
    use crate::program::{Edge, NamedModule};
    use crate::signature::Field;
    use std::rc::Rc;

    fn qa(name: &str) -> NamedModule {
        let sig = Signature::new(
            name,
            "answer the question",
            vec![Field::text("question", "q")],
            vec![Field::text("answer", "a")],
        );
        NamedModule::new(name, Rc::new(Predict::new(sig)))
    }

    #[test]
    fn compile_single_predict_yields_one_dispatch() {
        let p = Program::single(qa("m"));
        let c = Compiler::new().compile(&p).expect("compile");
        assert_eq!(c.dispatches.len(), 1);
        assert_eq!(c.dispatches[0].module_name, "m");
        assert_eq!(c.dispatches[0].kind, ModuleKind::Predict);
    }

    #[test]
    fn unbound_inputs_become_program_inputs() {
        let p = Program::single(qa("m"));
        let c = Compiler::new().compile(&p).expect("compile");
        assert_eq!(c.program_inputs, vec!["question".to_string()]);
        match c.dispatches[0].args.get("question") {
            Some(ArgBinding::FromProgramInput(s)) => assert_eq!(s, "question"),
            other => panic!("expected FromProgramInput, got {:?}", other),
        }
    }

    #[test]
    fn edge_produces_from_output_binding() {
        let a = qa("a");
        let b_sig = Signature::new(
            "b",
            "do the next thing",
            vec![Field::text("seed", "seed")],
            vec![Field::text("final", "f")],
        );
        let b = NamedModule::new("b", Rc::new(Predict::new(b_sig)));
        let p = Program::new(
            vec![a, b],
            vec![Edge::new(Port::new("a", "answer"), Port::new("b", "seed"))],
        );
        let c = Compiler::new().compile(&p).expect("compile");
        assert_eq!(c.dispatches.len(), 2);
        let b_dispatch = &c.dispatches[1];
        match b_dispatch.args.get("seed") {
            Some(ArgBinding::FromOutput(port)) => {
                assert_eq!(port.module, "a");
                assert_eq!(port.field, "answer");
            }
            other => panic!("expected FromOutput, got {:?}", other),
        }
    }

    #[test]
    fn chain_of_thought_effective_signature_includes_reasoning() {
        let sig = Signature::new(
            "cot",
            "answer",
            vec![Field::text("q", "")],
            vec![Field::text("a", "")],
        );
        let p = Program::single(NamedModule::new(
            "cot",
            Rc::new(ChainOfThought::new(sig)),
        ));
        let c = Compiler::new().compile(&p).expect("compile");
        let names: Vec<&str> = c.dispatches[0]
            .effective_signature
            .outputs
            .iter()
            .map(|f| f.name.as_str())
            .collect();
        assert_eq!(names, vec!["reasoning", "a"]);
    }

    #[test]
    fn jsonschema_surface_keys_are_module_names() {
        let p = Program::single(qa("m"));
        let c = Compiler::new().compile(&p).expect("compile");
        let s = c.compile_to_jsonschema();
        assert_eq!(s["type"], "object");
        assert!(s["properties"]["m"].is_object());
    }

    #[test]
    fn jsonschema_for_each_dispatch_requires_outputs() {
        let p = Program::single(qa("m"));
        let c = Compiler::new().compile(&p).expect("compile");
        let s = c.compile_to_jsonschema();
        assert_eq!(s["properties"]["m"]["required"], serde_json::json!(["answer"]));
    }

    #[test]
    fn shared_input_name_collapses_in_program_inputs() {
        // Two unconnected modules both reading `question` — should only
        // surface a single `question` program-input.
        let a = qa("a");
        let b = qa("b");
        let p = Program::new(vec![a, b], vec![]);
        let c = Compiler::new().compile(&p).expect("compile");
        assert_eq!(c.program_inputs, vec!["question".to_string()]);
    }

    #[test]
    fn program_outputs_come_from_last_topo_module() {
        let a = qa("a");
        let b_sig = Signature::new(
            "b",
            "next",
            vec![Field::text("seed", "")],
            vec![Field::text("final_answer", "")],
        );
        let b = NamedModule::new("b", Rc::new(Predict::new(b_sig)));
        let p = Program::new(
            vec![a, b],
            vec![Edge::new(Port::new("a", "answer"), Port::new("b", "seed"))],
        );
        let c = Compiler::new().compile(&p).expect("compile");
        assert_eq!(c.program_outputs, vec!["final_answer".to_string()]);
    }
}
