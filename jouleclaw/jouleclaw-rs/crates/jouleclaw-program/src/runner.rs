//! Execute a [`Compiled`] plan against a caller-supplied [`Backend`].
//!
//! The runner walks the dispatch list in compile order. For each dispatch
//! it:
//!
//! 1. Materialises the input record by resolving each [`ArgBinding`].
//! 2. Calls the backend with the rendered prompt and grammar.
//! 3. Type-checks the backend's outputs against the effective signature.
//! 4. For `ProgramOfThought` modules, additionally feeds the `code` output
//!    through the supplied [`CodeRunner`] and *replaces* the module's
//!    outputs with the code runner's result.
//! 5. Merges the result back into the global record under the module's
//!    declared output names.
//!
//! On the final dispatch, the runner returns a [`Record`] containing only the
//! program-level outputs.

use crate::compiler::{ArgBinding, Compiled};
use crate::error::{Error, Result};
use crate::module::{Backend, BackendCall, CodeRunner, ModuleKind};
use crate::record::{Record, Value};
use crate::signature::Field;

/// Executor for [`Compiled`] plans.
pub struct Runner<'a, B: Backend, C: CodeRunner> {
    backend: &'a B,
    code_runner: Option<&'a C>,
}

impl<'a, B: Backend, C: CodeRunner> Runner<'a, B, C> {
    /// Build a runner with backend and code runner. Pass `None` for the code
    /// runner if you don't plan to execute `ProgramOfThought` modules.
    pub fn new(backend: &'a B, code_runner: Option<&'a C>) -> Self {
        Self {
            backend,
            code_runner,
        }
    }

    /// Execute a compiled plan, returning the final program output record.
    pub fn run(&self, compiled: &Compiled, inputs: Record) -> Result<Record> {
        // Validate program inputs.
        for name in &compiled.program_inputs {
            if inputs.get(name).is_none() {
                return Err(Error::MissingInput {
                    field: name.clone(),
                });
            }
        }

        // The global record accumulates everything: program inputs at the
        // start, plus each module's outputs under `module_name::field_name`
        // (for graph lookup) and the raw output field name (for downstream
        // modules that read by field name).
        let mut globals = Record::new();
        for (k, v) in inputs.values.iter() {
            globals.insert(k.clone(), v.clone());
        }

        for d in &compiled.dispatches {
            // Resolve inputs.
            let mut call_inputs = Record::new();
            for f in &d.signature.inputs {
                let binding =
                    d.args.get(&f.name).ok_or_else(|| Error::UnboundInput {
                        module: d.module_name.clone(),
                        field: f.name.clone(),
                    })?;
                let v = match binding {
                    ArgBinding::FromProgramInput(name) => {
                        inputs.get(name).cloned().ok_or_else(|| Error::MissingInput {
                            field: name.clone(),
                        })?
                    }
                    ArgBinding::FromOutput(port) => {
                        let key = scoped_key(&port.module, &port.field);
                        globals.get(&key).cloned().ok_or_else(|| Error::MissingOutput {
                            module: port.module.clone(),
                            field: port.field.clone(),
                        })?
                    }
                };
                v.check(&f.ty, &d.module_name, &f.name)?;
                call_inputs.insert(f.name.clone(), v);
            }

            // Dispatch.
            let response = self
                .backend
                .call(BackendCall {
                    module_name: &d.module_name,
                    module_kind: d.kind,
                    signature: &d.signature,
                    effective_signature: &d.effective_signature,
                    prompt: &d.prompt,
                    grammar: &d.grammar,
                    inputs: &call_inputs,
                })
                .map_err(|e| Error::Backend {
                    module: d.module_name.clone(),
                    message: e,
                })?;

            // For ProgramOfThought, hand the code through the code runner
            // and replace outputs with whatever the code runner produced.
            let module_outputs = if matches!(d.kind, ModuleKind::ProgramOfThought) {
                let code_val = response.outputs.get("code").cloned().ok_or_else(|| {
                    Error::MissingOutput {
                        module: d.module_name.clone(),
                        field: "code".into(),
                    }
                })?;
                let code = match code_val {
                    Value::Text(s) => s,
                    other => {
                        return Err(Error::TypeMismatch {
                            module: d.module_name.clone(),
                            field: "code".into(),
                            expected: "text".into(),
                            actual: other.label().to_string(),
                        });
                    }
                };
                let runner = self.code_runner.ok_or_else(|| Error::CodeRunner {
                    module: d.module_name.clone(),
                    message: "no CodeRunner supplied for ProgramOfThought module".into(),
                })?;
                let expected: Vec<Field> = d.signature.outputs.clone();
                let produced = runner.run(&code, &expected).map_err(|e| Error::CodeRunner {
                    module: d.module_name.clone(),
                    message: e,
                })?;
                produced
            } else {
                response.outputs
            };

            // Type-check + commit. For ChainOfThought we tolerate the
            // reasoning field being present (effective_signature drives the
            // check) but only declared outputs are visible to downstream
            // edges.
            for f in &d.effective_signature.outputs {
                if let Some(v) = module_outputs.get(&f.name) {
                    v.check(&f.ty, &d.module_name, &f.name)?;
                } else if matches!(d.kind, ModuleKind::ProgramOfThought) {
                    // PoT skips backend outputs entirely; ChainOfThought
                    // *requires* reasoning from the backend. Predict
                    // requires declared outputs. Branch on kind below.
                }
            }

            // Declared-output presence check (skip for PoT-the-code-field
            // already-consumed special case where outputs come from runner).
            for f in &d.signature.outputs {
                let v = module_outputs.get(&f.name).ok_or_else(|| Error::MissingOutput {
                    module: d.module_name.clone(),
                    field: f.name.clone(),
                })?;
                v.check(&f.ty, &d.module_name, &f.name)?;
            }

            // Stash module outputs into the global record under two keys: a
            // scoped key for edge resolution, and the raw output name for
            // convenience.
            for f in &d.effective_signature.outputs {
                if let Some(v) = module_outputs.get(&f.name) {
                    globals.insert(scoped_key(&d.module_name, &f.name), v.clone());
                    globals.insert(f.name.clone(), v.clone());
                }
            }
        }

        // Project program-level outputs out of the globals.
        let mut out = Record::new();
        for name in &compiled.program_outputs {
            if let Some(v) = globals.get(name) {
                out.insert(name.clone(), v.clone());
            }
        }
        Ok(out)
    }
}

fn scoped_key(module: &str, field: &str) -> String {
    format!("{}::{}", module, field)
}

// ---------- MockBackend & MockCodeRunner used by tests below ----------

/// Trivial backend that echoes synthesised outputs back. Useful for tests
/// and for callers writing their own integration adapters.
///
/// For each output field on the effective signature, it produces a
/// type-appropriate placeholder:
///
/// * `text` -> `"<module_name>::<field>:<input_summary>"`
/// * `bool` -> `false`
/// * `int` -> `0`
/// * `float` -> `0.0`
/// * `list<T>` -> empty list
/// * `json(_)` -> `null`
/// * `one_of` -> first variant
///
/// For `ProgramOfThought`, the `code` field is emitted as the string
/// `"<code>"` and the [`MockCodeRunner`] interprets it.
pub struct MockBackend;

impl Backend for MockBackend {
    fn call(
        &self,
        call: BackendCall<'_>,
    ) -> core::result::Result<crate::module::BackendResponse, String> {
        let mut record = Record::new();
        for f in &call.effective_signature.outputs {
            let v = synth_value(&f.ty);
            let labelled = match (&v, &f.ty) {
                (Value::Text(_), _) => Value::Text(format!(
                    "{}::{}::{}",
                    call.module_name,
                    f.name,
                    summarise(call.inputs)
                )),
                _ => v,
            };
            record.insert(f.name.clone(), labelled);
        }
        // PoT must emit a `code` field.
        if matches!(call.module_kind, ModuleKind::ProgramOfThought) {
            record.insert("code", Value::Text("<code>".into()));
        }
        Ok(crate::module::BackendResponse {
            outputs: record,
            raw: None,
        })
    }
}

/// Trivial code runner that pretends to execute the literal string `"<code>"`
/// and emits zeros / empty strings for each declared output.
pub struct MockCodeRunner;

impl CodeRunner for MockCodeRunner {
    fn run(
        &self,
        _code: &str,
        expected_outputs: &[Field],
    ) -> core::result::Result<Record, String> {
        let mut r = Record::new();
        for f in expected_outputs {
            r.insert(f.name.clone(), synth_value(&f.ty));
        }
        Ok(r)
    }
}

fn synth_value(ty: &crate::signature::FieldType) -> Value {
    use crate::signature::FieldType::*;
    match ty {
        Text => Value::Text(String::new()),
        Bool => Value::Bool(false),
        Int => Value::Int(0),
        Float => Value::Float(0.0),
        List(_) => Value::List(Vec::new()),
        Json(_) => Value::Json(serde_json::Value::Null),
        OneOf(v) => v
            .first()
            .map(|s| Value::Enum(s.clone()))
            .unwrap_or(Value::Enum(String::new())),
    }
}

fn summarise(r: &Record) -> String {
    let mut keys: Vec<&String> = r.values.keys().collect();
    keys.sort();
    keys.into_iter()
        .map(|k| format!("{}={}", k, r.values[k].label()))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::Compiler;
    use crate::module::{ChainOfThought, Predict, ProgramOfThought};
    use crate::program::{Edge, NamedModule, Port, Program};
    use crate::signature::{Field, Signature};
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
    fn run_single_predict_returns_program_outputs() {
        let p = Program::single(qa("m"));
        let c = Compiler::new().compile(&p).expect("compile");
        let backend = MockBackend;
        let coder = MockCodeRunner;
        let r = Runner::<MockBackend, MockCodeRunner>::new(&backend, Some(&coder));
        let mut inputs = Record::new();
        inputs.insert("question", Value::text("what?"));
        let out = r.run(&c, inputs).expect("run");
        assert!(out.get("answer").is_some());
    }

    #[test]
    fn run_missing_program_input_errors() {
        let p = Program::single(qa("m"));
        let c = Compiler::new().compile(&p).expect("compile");
        let backend = MockBackend;
        let coder = MockCodeRunner;
        let r = Runner::<MockBackend, MockCodeRunner>::new(&backend, Some(&coder));
        let err = r.run(&c, Record::new()).unwrap_err();
        assert!(matches!(err, Error::MissingInput { .. }));
    }

    #[test]
    fn run_pipeline_threads_outputs_between_modules() {
        let a = qa("a");
        let b_sig = Signature::new(
            "b",
            "next",
            vec![Field::text("seed", "")],
            vec![Field::text("final", "")],
        );
        let b = NamedModule::new("b", Rc::new(Predict::new(b_sig)));
        let p = Program::new(
            vec![a, b],
            vec![Edge::new(Port::new("a", "answer"), Port::new("b", "seed"))],
        );
        let c = Compiler::new().compile(&p).expect("compile");
        let backend = MockBackend;
        let coder = MockCodeRunner;
        let r = Runner::<MockBackend, MockCodeRunner>::new(&backend, Some(&coder));
        let mut inputs = Record::new();
        inputs.insert("question", Value::text("?"));
        let out = r.run(&c, inputs).expect("run");
        // MockBackend renders text outputs containing "<module>::<field>::<input-summary>"
        // — for `b`, that summary should mention `seed=text`.
        match out.get("final") {
            Some(Value::Text(s)) => assert!(s.contains("seed=text"), "got {:?}", s),
            other => panic!("expected text, got {:?}", other),
        }
    }

    #[test]
    fn cot_reasoning_visible_in_globals_but_not_program_output() {
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
        let backend = MockBackend;
        let coder = MockCodeRunner;
        let r = Runner::<MockBackend, MockCodeRunner>::new(&backend, Some(&coder));
        let mut inputs = Record::new();
        inputs.insert("q", Value::text("hi"));
        let out = r.run(&c, inputs).expect("run");
        // Reasoning is not a declared output, so it's not in the program output.
        assert!(out.get("reasoning").is_none());
        assert!(out.get("a").is_some());
    }

    #[test]
    fn pot_uses_code_runner_and_replaces_outputs() {
        let sig = Signature::new(
            "pot",
            "compute",
            vec![Field::text("q", "")],
            vec![Field::int("answer", "")],
        );
        let p = Program::single(NamedModule::new(
            "pot",
            Rc::new(ProgramOfThought::new(sig)),
        ));
        let c = Compiler::new().compile(&p).expect("compile");
        let backend = MockBackend;
        let coder = MockCodeRunner;
        let r = Runner::<MockBackend, MockCodeRunner>::new(&backend, Some(&coder));
        let mut inputs = Record::new();
        inputs.insert("q", Value::text("2 + 2"));
        let out = r.run(&c, inputs).expect("run");
        // MockCodeRunner emits Int(0).
        assert_eq!(out.get("answer"), Some(&Value::Int(0)));
    }

    /// Failing backend — used to confirm errors surface as `Error::Backend`.
    struct FailingBackend;
    impl Backend for FailingBackend {
        fn call(
            &self,
            _call: BackendCall<'_>,
        ) -> core::result::Result<crate::module::BackendResponse, String> {
            Err("network is on fire".into())
        }
    }

    #[test]
    fn backend_error_surfaces_as_typed_error() {
        let p = Program::single(qa("m"));
        let c = Compiler::new().compile(&p).expect("compile");
        let backend = FailingBackend;
        let coder = MockCodeRunner;
        let r = Runner::<FailingBackend, MockCodeRunner>::new(&backend, Some(&coder));
        let mut inputs = Record::new();
        inputs.insert("question", Value::text("?"));
        let err = r.run(&c, inputs).unwrap_err();
        assert!(matches!(err, Error::Backend { .. }));
    }

    /// Backend that returns the wrong type for the declared output.
    struct WrongTypeBackend;
    impl Backend for WrongTypeBackend {
        fn call(
            &self,
            _call: BackendCall<'_>,
        ) -> core::result::Result<crate::module::BackendResponse, String> {
            let mut r = Record::new();
            r.insert("answer", Value::Int(42)); // declared as text
            Ok(crate::module::BackendResponse {
                outputs: r,
                raw: None,
            })
        }
    }

    #[test]
    fn type_mismatch_in_backend_output_errors() {
        let p = Program::single(qa("m"));
        let c = Compiler::new().compile(&p).expect("compile");
        let backend = WrongTypeBackend;
        let coder = MockCodeRunner;
        let r = Runner::<WrongTypeBackend, MockCodeRunner>::new(&backend, Some(&coder));
        let mut inputs = Record::new();
        inputs.insert("question", Value::text("?"));
        let err = r.run(&c, inputs).unwrap_err();
        assert!(matches!(err, Error::TypeMismatch { .. }));
    }

    #[test]
    fn pot_without_code_runner_errors() {
        let sig = Signature::new(
            "pot",
            "compute",
            vec![Field::text("q", "")],
            vec![Field::int("answer", "")],
        );
        let p = Program::single(NamedModule::new(
            "pot",
            Rc::new(ProgramOfThought::new(sig)),
        ));
        let c = Compiler::new().compile(&p).expect("compile");
        let backend = MockBackend;
        let r = Runner::<MockBackend, MockCodeRunner>::new(&backend, None);
        let mut inputs = Record::new();
        inputs.insert("q", Value::text("?"));
        let err = r.run(&c, inputs).unwrap_err();
        assert!(matches!(err, Error::CodeRunner { .. }));
    }
}
