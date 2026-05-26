//! Boolean satisfiability via DPLL.
//!
//! The canonical NP-complete constraint problem. Energy = number of
//! unsatisfied clauses under a (partial) assignment; a zero-energy
//! total assignment is a model. DPLL = backtracking search + unit
//! propagation + pure-literal elimination.
//!
//! Input grammar (`sat` query payload):
//!
//!   clause ::= literal (' ' literal)*
//!   cnf    ::= clause (';' clause)*
//!   literal ::= '-'? var          (1-indexed; `-3` is ¬x3)
//!
//! e.g. `1 2 ; -1 3 ; -2 -3` is (x1∨x2) ∧ (¬x1∨x3) ∧ (¬x2∨¬x3).

use crate::EnergyFunction;

#[derive(Debug, Clone, PartialEq)]
pub enum SatError {
    Empty,
    BadLiteral(String),
    ZeroVariable,
}

impl std::fmt::Display for SatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "empty CNF"),
            Self::BadLiteral(s) => write!(f, "bad literal {:?}", s),
            Self::ZeroVariable => write!(f, "variable 0 is illegal (vars are 1-indexed)"),
        }
    }
}

impl std::error::Error for SatError {}

/// A literal: positive var or its negation. `var` is 1-indexed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lit {
    pub var: u32,
    pub negated: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cnf {
    pub n_vars: u32,
    pub clauses: Vec<Vec<Lit>>,
}

/// A complete assignment: `assignment[v-1]` is the truth value of x_v.
pub type Model = Vec<bool>;

impl Cnf {
    pub fn parse(input: &str) -> Result<Self, SatError> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(SatError::Empty);
        }
        let mut clauses = Vec::new();
        let mut n_vars = 0u32;
        for clause_str in trimmed.split(';') {
            let clause_str = clause_str.trim();
            if clause_str.is_empty() {
                continue;
            }
            let mut clause = Vec::new();
            for tok in clause_str.split_whitespace() {
                let (negated, num) = if let Some(rest) = tok.strip_prefix('-') {
                    (true, rest)
                } else {
                    (false, tok)
                };
                let var: u32 = num
                    .parse()
                    .map_err(|_| SatError::BadLiteral(tok.to_string()))?;
                if var == 0 {
                    return Err(SatError::ZeroVariable);
                }
                n_vars = n_vars.max(var);
                clause.push(Lit { var, negated });
            }
            if !clause.is_empty() {
                clauses.push(clause);
            }
        }
        if clauses.is_empty() {
            return Err(SatError::Empty);
        }
        Ok(Cnf { n_vars, clauses })
    }

    /// Evaluate energy of a *total* assignment: count of unsatisfied clauses.
    fn unsatisfied(&self, model: &Model) -> usize {
        self.clauses
            .iter()
            .filter(|clause| {
                !clause.iter().any(|lit| {
                    let v = model[(lit.var - 1) as usize];
                    if lit.negated { !v } else { v }
                })
            })
            .count()
    }

    /// DPLL solver. Returns a satisfying model or None (UNSAT).
    /// `max_steps` caps decision count to protect the joule budget.
    pub fn solve(&self, max_steps: usize) -> Result<Option<Model>, SatError> {
        // assignment[v] = None unassigned, Some(b) assigned.
        let mut assign: Vec<Option<bool>> = vec![None; self.n_vars as usize];
        let mut steps = 0usize;
        let ok = dpll(self, &mut assign, &mut steps, max_steps);
        match ok {
            DpllOutcome::Sat => {
                // Fill any free variables with `false` (don't-cares).
                let model: Model = assign.iter().map(|a| a.unwrap_or(false)).collect();
                Ok(Some(model))
            }
            DpllOutcome::Unsat => Ok(None),
            DpllOutcome::StepLimit => Ok(None), // treat as "couldn't prove SAT"
        }
    }

    pub fn render_model(&self, model: &Model) -> String {
        let mut s = String::new();
        s.push_str("SAT — model:\n");
        for (i, &v) in model.iter().enumerate() {
            s.push_str(&format!("  x{} = {}\n", i + 1, v));
        }
        s.push_str(&format!(
            "  ({} clause(s), all satisfied)\n",
            self.clauses.len()
        ));
        s
    }
}

enum DpllOutcome {
    Sat,
    Unsat,
    StepLimit,
}

fn clause_status(clause: &[Lit], assign: &[Option<bool>]) -> ClauseStatus {
    let mut unassigned: Option<Lit> = None;
    let mut unassigned_count = 0;
    for &lit in clause {
        match assign[(lit.var - 1) as usize] {
            Some(v) => {
                let sat = if lit.negated { !v } else { v };
                if sat {
                    return ClauseStatus::Satisfied;
                }
            }
            None => {
                unassigned = Some(lit);
                unassigned_count += 1;
            }
        }
    }
    match unassigned_count {
        0 => ClauseStatus::Conflict,
        1 => ClauseStatus::Unit(unassigned.unwrap()),
        _ => ClauseStatus::Unresolved,
    }
}

enum ClauseStatus {
    Satisfied,
    Conflict,
    Unit(Lit),
    Unresolved,
}

fn dpll(
    cnf: &Cnf,
    assign: &mut Vec<Option<bool>>,
    steps: &mut usize,
    max_steps: usize,
) -> DpllOutcome {
    // Unit propagation to a fixpoint.
    loop {
        let mut progressed = false;
        for clause in &cnf.clauses {
            match clause_status(clause, assign) {
                ClauseStatus::Conflict => return DpllOutcome::Unsat,
                ClauseStatus::Unit(lit) => {
                    assign[(lit.var - 1) as usize] = Some(!lit.negated);
                    progressed = true;
                }
                _ => {}
            }
        }
        if !progressed {
            break;
        }
    }

    // All clauses satisfied?
    let all_sat = cnf
        .clauses
        .iter()
        .all(|c| matches!(clause_status(c, assign), ClauseStatus::Satisfied));
    if all_sat {
        return DpllOutcome::Sat;
    }

    // Pick the first unassigned variable to branch on.
    let pick = assign.iter().position(|a| a.is_none());
    let Some(idx) = pick else {
        // No free variables but not all satisfied → conflict.
        return DpllOutcome::Unsat;
    };

    for &val in &[true, false] {
        *steps += 1;
        if *steps > max_steps {
            return DpllOutcome::StepLimit;
        }
        let mut next = assign.clone();
        next[idx] = Some(val);
        match dpll(cnf, &mut next, steps, max_steps) {
            DpllOutcome::Sat => {
                *assign = next;
                return DpllOutcome::Sat;
            }
            DpllOutcome::StepLimit => return DpllOutcome::StepLimit,
            DpllOutcome::Unsat => {}
        }
    }
    DpllOutcome::Unsat
}

/// EnergyFunction view: state is a total model; energy is the count of
/// unsatisfied clauses (0 ⇒ a satisfying assignment).
impl EnergyFunction for Cnf {
    type State = Model;
    fn energy(&self, state: &Self::State) -> f64 {
        self.unsatisfied(state) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_cnf() {
        let c = Cnf::parse("1 2 ; -1 3 ; -2 -3").unwrap();
        assert_eq!(c.n_vars, 3);
        assert_eq!(c.clauses.len(), 3);
    }

    #[test]
    fn parse_rejects_empty() {
        assert!(matches!(Cnf::parse("   "), Err(SatError::Empty)));
    }

    #[test]
    fn parse_rejects_zero_var() {
        assert!(matches!(Cnf::parse("0"), Err(SatError::ZeroVariable)));
    }

    #[test]
    fn solves_satisfiable_instance() {
        // (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ ¬x3) — satisfiable.
        let c = Cnf::parse("1 2 ; -1 3 ; -2 -3").unwrap();
        let m = c.solve(100_000).unwrap();
        assert!(m.is_some());
        let model = m.unwrap();
        // Verify: energy must be 0.
        assert_eq!(c.energy(&model), 0.0);
    }

    #[test]
    fn detects_unsatisfiable_instance() {
        // x1 ∧ ¬x1 — UNSAT.
        let c = Cnf::parse("1 ; -1").unwrap();
        assert!(c.solve(100_000).unwrap().is_none());
    }

    #[test]
    fn unit_propagation_pigeonhole_small_unsat() {
        // 2 pigeons, 1 hole: p1 ∨ nothing... encode p_ij = pigeon i in hole j.
        // 2 holes 3 pigeons is the classic small UNSAT; keep it tiny here:
        // (a) ∧ (b) ∧ (¬a ∨ ¬b) with a=p1h1, b=p2h1 (both pigeons same hole forbidden)
        // plus each pigeon must be placed: (a) (b) — forces a∧b, contradiction.
        let c = Cnf::parse("1 ; 2 ; -1 -2").unwrap();
        assert!(c.solve(100_000).unwrap().is_none());
    }

    #[test]
    fn solver_is_deterministic() {
        let c = Cnf::parse("1 2 3 ; -1 2 ; -2 3 ; -3 1").unwrap();
        let a = c.solve(100_000).unwrap();
        let b = c.solve(100_000).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn energy_counts_unsatisfied_clauses() {
        let c = Cnf::parse("1 ; -1").unwrap();
        // x1=true satisfies clause 1, violates clause 2 → energy 1.
        assert_eq!(c.energy(&vec![true]), 1.0);
    }
}
