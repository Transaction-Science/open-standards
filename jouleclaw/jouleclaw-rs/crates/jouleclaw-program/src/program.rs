//! Composition of modules with explicit data flow.
//!
//! A [`Program`] is a directed graph: nodes are [`NamedModule`]s, edges wire
//! one module's output field into another module's input field. Inputs the
//! program receives from the caller are bound by name through
//! [`Runner`](crate::runner::Runner) — they aren't represented as graph nodes.
//!
//! Validation lives here because it can be done without a compiler:
//!
//! * unique module names,
//! * known module + field on every edge endpoint,
//! * matching types on each edge,
//! * acyclic graph (so the compiler can topo-sort).

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::error::{Error, Result};
use crate::module::Module;
use crate::signature::FieldType;

/// A named module in a program. `Rc` so the same module description can
/// participate in multiple programs without cloning the whole signature.
#[derive(Clone)]
pub struct NamedModule {
    pub name: String,
    pub module: Rc<dyn Module>,
}

impl NamedModule {
    pub fn new(name: impl Into<String>, module: Rc<dyn Module>) -> Self {
        Self {
            name: name.into(),
            module,
        }
    }
}

impl core::fmt::Debug for NamedModule {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NamedModule")
            .field("name", &self.name)
            .field("kind", &self.module.kind())
            .field("signature_name", &self.module.signature().name)
            .finish()
    }
}

/// One endpoint of a data-flow edge: `(module_name, field_name)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Port {
    pub module: String,
    pub field: String,
}

impl Port {
    pub fn new(module: impl Into<String>, field: impl Into<String>) -> Self {
        Self {
            module: module.into(),
            field: field.into(),
        }
    }
}

/// Directed edge from one module's output to another module's input.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Edge {
    pub from: Port,
    pub to: Port,
}

impl Edge {
    pub fn new(from: Port, to: Port) -> Self {
        Self { from, to }
    }
}

/// A composition of named modules with explicit data flow.
#[derive(Clone, Debug)]
pub struct Program {
    pub modules: Vec<NamedModule>,
    pub flow: Vec<Edge>,
}

impl Program {
    pub fn new(modules: Vec<NamedModule>, flow: Vec<Edge>) -> Self {
        Self { modules, flow }
    }

    /// Convenience builder for a single-module program (no edges).
    pub fn single(module: NamedModule) -> Self {
        Self {
            modules: vec![module],
            flow: Vec::new(),
        }
    }

    /// Look up a module by name.
    pub fn module(&self, name: &str) -> Option<&NamedModule> {
        self.modules.iter().find(|m| m.name == name)
    }

    /// Validate the program in isolation.
    ///
    /// Returns the same kinds of errors the compiler would raise on its own
    /// pre-check pass; the compiler calls this internally so callers don't
    /// have to.
    pub fn validate(&self) -> Result<()> {
        if self.modules.is_empty() {
            return Err(Error::EmptyProgram);
        }

        // Unique module names.
        let mut seen = HashSet::new();
        for m in &self.modules {
            if !seen.insert(m.name.as_str()) {
                return Err(Error::DuplicateModule {
                    name: m.name.clone(),
                });
            }
            // Validate signature in isolation while we're here.
            m.module.signature().validate()?;
        }

        // Edges reference known modules + fields, and types match.
        for e in &self.flow {
            let from = self.modules.iter().find(|m| m.name == e.from.module).ok_or(
                Error::UnknownModule {
                    module: e.from.module.clone(),
                },
            )?;
            let to = self.modules.iter().find(|m| m.name == e.to.module).ok_or(
                Error::UnknownModule {
                    module: e.to.module.clone(),
                },
            )?;
            let from_field = from
                .module
                .effective_signature()
                .output(&e.from.field)
                .cloned()
                .ok_or(Error::UnknownField {
                    module: e.from.module.clone(),
                    field: e.from.field.clone(),
                })?;
            let to_field = to
                .module
                .signature()
                .input(&e.to.field)
                .cloned()
                .ok_or(Error::UnknownField {
                    module: e.to.module.clone(),
                    field: e.to.field.clone(),
                })?;
            if !types_compatible(&from_field.ty, &to_field.ty) {
                return Err(Error::EdgeTypeMismatch {
                    from_module: e.from.module.clone(),
                    from_field: e.from.field.clone(),
                    from_ty: from_field.ty.label(),
                    to_module: e.to.module.clone(),
                    to_field: e.to.field.clone(),
                    to_ty: to_field.ty.label(),
                });
            }
        }

        // Cycle check.
        self.topo_order()?;
        Ok(())
    }

    /// Return module names in dependency order.
    ///
    /// Pure data-dependency Kahn's algorithm over the edge set. If a cycle
    /// exists, returns the first module that couldn't be placed.
    pub fn topo_order(&self) -> Result<Vec<String>> {
        let mut indeg: HashMap<String, usize> = HashMap::new();
        let mut forward: HashMap<String, Vec<String>> = HashMap::new();
        for m in &self.modules {
            indeg.entry(m.name.clone()).or_insert(0);
            forward.entry(m.name.clone()).or_default();
        }
        for e in &self.flow {
            if e.from.module == e.to.module {
                return Err(Error::ProgramCycle {
                    module: e.from.module.clone(),
                });
            }
            *indeg.entry(e.to.module.clone()).or_insert(0) += 1;
            forward
                .entry(e.from.module.clone())
                .or_default()
                .push(e.to.module.clone());
        }

        // Stable order: walk modules in declaration order so the output is
        // deterministic for callers that care.
        let mut order = Vec::new();
        let mut queue: Vec<String> = self
            .modules
            .iter()
            .filter(|m| indeg.get(&m.name).copied().unwrap_or(0) == 0)
            .map(|m| m.name.clone())
            .collect();
        while let Some(name) = queue.pop() {
            order.push(name.clone());
            if let Some(succs) = forward.get(&name) {
                let succs = succs.clone();
                for s in succs {
                    if let Some(d) = indeg.get_mut(&s) {
                        *d -= 1;
                        if *d == 0 {
                            queue.push(s);
                        }
                    }
                }
            }
        }
        if order.len() != self.modules.len() {
            // Find a module still with positive in-degree.
            let bad = self
                .modules
                .iter()
                .find(|m| indeg.get(&m.name).copied().unwrap_or(0) > 0)
                .map(|m| m.name.clone())
                .unwrap_or_else(|| self.modules[0].name.clone());
            return Err(Error::ProgramCycle { module: bad });
        }
        Ok(order)
    }
}

/// Types are compatible if they're equal, with a couple of widenings:
///
/// * `Int` flows into `Float`,
/// * `Enum` and `Text` are interchangeable (the runner will re-check enum
///   variants against the receiving signature on apply).
pub(crate) fn types_compatible(from: &FieldType, to: &FieldType) -> bool {
    if from == to {
        return true;
    }
    match (from, to) {
        (FieldType::Int, FieldType::Float) => true,
        (FieldType::OneOf(_), FieldType::Text) => true,
        (FieldType::Text, FieldType::OneOf(_)) => true,
        (FieldType::List(a), FieldType::List(b)) => types_compatible(a, b),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::{Predict, ProgramOfThought};
    use crate::signature::{Field, Signature};
    use std::rc::Rc;

    fn predict(name: &str, in_f: &str, in_t: FieldType, out_f: &str, out_t: FieldType) -> NamedModule {
        let sig = Signature::new(
            name,
            "do a thing",
            vec![Field {
                name: in_f.into(),
                ty: in_t,
                description: "in".into(),
            }],
            vec![Field {
                name: out_f.into(),
                ty: out_t,
                description: "out".into(),
            }],
        );
        NamedModule::new(name, Rc::new(Predict::new(sig)))
    }

    #[test]
    fn empty_program_fails() {
        let p = Program::new(Vec::new(), Vec::new());
        assert!(matches!(p.validate(), Err(Error::EmptyProgram)));
    }

    #[test]
    fn duplicate_module_names_fail() {
        let m1 = predict("a", "x", FieldType::Text, "y", FieldType::Text);
        let m2 = predict("a", "x", FieldType::Text, "y", FieldType::Text);
        let p = Program::new(vec![m1, m2], vec![]);
        assert!(matches!(p.validate(), Err(Error::DuplicateModule { .. })));
    }

    #[test]
    fn edge_to_unknown_module_fails() {
        let m = predict("a", "x", FieldType::Text, "y", FieldType::Text);
        let edge = Edge::new(Port::new("a", "y"), Port::new("nope", "x"));
        let p = Program::new(vec![m], vec![edge]);
        assert!(matches!(p.validate(), Err(Error::UnknownModule { .. })));
    }

    #[test]
    fn edge_to_unknown_field_fails() {
        let a = predict("a", "x", FieldType::Text, "y", FieldType::Text);
        let b = predict("b", "x", FieldType::Text, "y", FieldType::Text);
        let edge = Edge::new(Port::new("a", "y"), Port::new("b", "nope"));
        let p = Program::new(vec![a, b], vec![edge]);
        assert!(matches!(p.validate(), Err(Error::UnknownField { .. })));
    }

    #[test]
    fn edge_type_mismatch_fails() {
        let a = predict("a", "x", FieldType::Text, "y", FieldType::Text);
        let b = predict("b", "x", FieldType::Int, "y", FieldType::Text);
        let edge = Edge::new(Port::new("a", "y"), Port::new("b", "x"));
        let p = Program::new(vec![a, b], vec![edge]);
        assert!(matches!(p.validate(), Err(Error::EdgeTypeMismatch { .. })));
    }

    #[test]
    fn cycle_is_rejected() {
        let a = predict("a", "x", FieldType::Text, "y", FieldType::Text);
        let b = predict("b", "x", FieldType::Text, "y", FieldType::Text);
        let p = Program::new(
            vec![a, b],
            vec![
                Edge::new(Port::new("a", "y"), Port::new("b", "x")),
                Edge::new(Port::new("b", "y"), Port::new("a", "x")),
            ],
        );
        assert!(matches!(p.validate(), Err(Error::ProgramCycle { .. })));
    }

    #[test]
    fn topo_order_respects_edges() {
        let a = predict("a", "x", FieldType::Text, "y", FieldType::Text);
        let b = predict("b", "x", FieldType::Text, "y", FieldType::Text);
        let c = predict("c", "x", FieldType::Text, "y", FieldType::Text);
        let p = Program::new(
            vec![c, b, a],
            vec![
                Edge::new(Port::new("a", "y"), Port::new("b", "x")),
                Edge::new(Port::new("b", "y"), Port::new("c", "x")),
            ],
        );
        let order = p.topo_order().expect("valid topo");
        let ia = order.iter().position(|n| n == "a").unwrap();
        let ib = order.iter().position(|n| n == "b").unwrap();
        let ic = order.iter().position(|n| n == "c").unwrap();
        assert!(ia < ib);
        assert!(ib < ic);
    }

    #[test]
    fn types_compatible_widens_int_to_float() {
        assert!(types_compatible(&FieldType::Int, &FieldType::Float));
        assert!(!types_compatible(&FieldType::Float, &FieldType::Int));
    }

    #[test]
    fn types_compatible_lists_match_recursively() {
        let a = FieldType::List(Box::new(FieldType::Int));
        let b = FieldType::List(Box::new(FieldType::Float));
        assert!(types_compatible(&a, &b));
    }

    #[test]
    fn cot_outputs_visible_to_edges() {
        // ChainOfThought injects a reasoning output. Make sure programs can
        // route that field if they want to.
        use crate::module::ChainOfThought;
        let cot_sig = Signature::new(
            "cot",
            "answer the question",
            vec![Field::text("question", "q")],
            vec![Field::text("answer", "a")],
        );
        let cot = NamedModule::new("cot", Rc::new(ChainOfThought::new(cot_sig)));
        let downstream =
            predict("d", "trace", FieldType::Text, "out", FieldType::Text);
        let p = Program::new(
            vec![cot, downstream],
            vec![Edge::new(
                Port::new("cot", "reasoning"),
                Port::new("d", "trace"),
            )],
        );
        assert!(p.validate().is_ok(), "{:?}", p.validate());
    }

    #[test]
    fn pot_does_not_implicitly_route_code() {
        // ProgramOfThought emits a `code` field at runtime via the runner —
        // not via the signature output list. The compiler / program should
        // not see `code` as an output field for graph wiring.
        let pot_sig = Signature::new(
            "pot",
            "compute the answer",
            vec![Field::text("question", "q")],
            vec![Field::int("answer", "a")],
        );
        let m = NamedModule::new("pot", Rc::new(ProgramOfThought::new(pot_sig)));
        let next = predict(
            "next",
            "code",
            FieldType::Text,
            "out",
            FieldType::Text,
        );
        let bad_edge = Edge::new(Port::new("pot", "code"), Port::new("next", "code"));
        let p = Program::new(vec![m, next], vec![bad_edge]);
        assert!(matches!(p.validate(), Err(Error::UnknownField { .. })));
    }
}
