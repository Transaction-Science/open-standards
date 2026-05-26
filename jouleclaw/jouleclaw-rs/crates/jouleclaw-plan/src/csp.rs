//! Hand-rolled CSP solver for the Plan pillar (spec §4.2).
//!
//! Python's `python-constraint` library has no Rust drop-in. The
//! planner's search space is small enough (tens of sub-queries × a
//! handful of stores) that backtracking with constraint propagation
//! is fast and stays predictable. This module is generic over
//! variable / value types so it's testable in isolation; the planner
//! instantiates it for `(sub_id, store_id)` assignments.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum CspError {
    Unsatisfiable,
    Timeout,
}

impl std::fmt::Display for CspError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsatisfiable => write!(f, "no valid assignment exists"),
            Self::Timeout => write!(f, "solver timed out"),
        }
    }
}

impl std::error::Error for CspError {}

/// A complete (or partial during search) variable→value assignment.
pub type Assignment<V, T> = BTreeMap<V, T>;

/// Per-variable score (lower = better). Used to rank otherwise-equal
/// solutions; the solver returns the one with minimum total score.
pub type Score = f64;

/// User-provided unary cost evaluator for a single (variable, value)
/// pair. Returning `f64::INFINITY` excludes the assignment outright.
pub trait UnaryCost<V, T>: Fn(&V, &T) -> Score {}
impl<V, T, F: Fn(&V, &T) -> Score> UnaryCost<V, T> for F {}

/// User-provided constraint over the entire (partial or full)
/// assignment. Return `true` iff the assignment is still feasible.
/// Called incrementally as variables get assigned, so it should be
/// fast for partial assignments.
pub trait Constraint<V, T>: Fn(&Assignment<V, T>) -> bool {}
impl<V, T, F: Fn(&Assignment<V, T>) -> bool> Constraint<V, T> for F {}

/// Backtracking CSP solver. Variables are assigned in the order
/// given to [`CspSolver::variables`]; for each variable, values are
/// tried in ascending unary-cost order. Returns the
/// minimum-total-cost feasible assignment, or [`CspError`].
pub struct CspSolver<V, T, C, U>
where
    V: Clone + Ord,
    T: Clone + Ord,
    C: Constraint<V, T>,
    U: UnaryCost<V, T>,
{
    variables: Vec<V>,
    domains: BTreeMap<V, Vec<T>>,
    constraint: C,
    unary_cost: U,
    timeout: Duration,
}

impl<V, T, C, U> CspSolver<V, T, C, U>
where
    V: Clone + Ord,
    T: Clone + Ord,
    C: Constraint<V, T>,
    U: UnaryCost<V, T>,
{
    pub fn new(constraint: C, unary_cost: U) -> Self {
        Self {
            variables: Vec::new(),
            domains: BTreeMap::new(),
            constraint,
            unary_cost,
            timeout: Duration::from_millis(200),
        }
    }

    pub fn variable(mut self, var: V, domain: Vec<T>) -> Self {
        self.variables.push(var.clone());
        self.domains.insert(var, domain);
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn solve(&self) -> Result<(Assignment<V, T>, Score), CspError> {
        let start = Instant::now();
        let mut best: Option<(Assignment<V, T>, Score)> = None;
        let mut current = Assignment::<V, T>::new();
        self.backtrack(0, &mut current, 0.0, &mut best, start)?;
        best.ok_or(CspError::Unsatisfiable)
    }

    fn backtrack(
        &self,
        idx: usize,
        current: &mut Assignment<V, T>,
        score_so_far: Score,
        best: &mut Option<(Assignment<V, T>, Score)>,
        started_at: Instant,
    ) -> Result<(), CspError> {
        if started_at.elapsed() > self.timeout {
            return Err(CspError::Timeout);
        }
        if idx == self.variables.len() {
            if (self.constraint)(current) {
                if best
                    .as_ref()
                    .map(|(_, s)| score_so_far < *s)
                    .unwrap_or(true)
                {
                    *best = Some((current.clone(), score_so_far));
                }
            }
            return Ok(());
        }

        let var = self.variables[idx].clone();
        let mut domain = self
            .domains
            .get(&var)
            .cloned()
            .unwrap_or_default();
        let unary = &self.unary_cost;
        domain.sort_by(|a, b| {
            unary(&var, a)
                .partial_cmp(&unary(&var, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for value in domain {
            let cost = (self.unary_cost)(&var, &value);
            if !cost.is_finite() {
                continue;
            }
            let projected = score_so_far + cost;
            if let Some((_, best_score)) = best {
                if projected >= *best_score {
                    continue;
                }
            }
            current.insert(var.clone(), value);
            if (self.constraint)(current) {
                self.backtrack(idx + 1, current, projected, best, started_at)?;
            }
            current.remove(&var);
        }
        Ok(())
    }
}

/// Pretty-print an assignment ordered by variable. Useful for
/// failure diagnostics.
pub fn render_assignment<V, T>(assignment: &Assignment<V, T>) -> String
where
    V: std::fmt::Debug + Ord,
    T: std::fmt::Debug,
{
    let mut out = String::new();
    let keys: BTreeSet<_> = assignment.keys().collect();
    for k in keys {
        let v = assignment.get(k).unwrap();
        out.push_str(&format!("  {k:?} => {v:?}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_trivial_two_variable_problem() {
        // Variables x, y in {1,2,3}. Constraint x + y == 4. Cost x+y.
        let solver = CspSolver::new(
            |a: &Assignment<&'static str, i32>| {
                let x = a.get("x").copied();
                let y = a.get("y").copied();
                match (x, y) {
                    (Some(xv), Some(yv)) => xv + yv == 4,
                    _ => true,
                }
            },
            |_v, val: &i32| *val as f64,
        )
        .variable("x", vec![1, 2, 3])
        .variable("y", vec![1, 2, 3]);

        let (assignment, score) = solver.solve().unwrap();
        let xv = assignment.get("x").copied().unwrap();
        let yv = assignment.get("y").copied().unwrap();
        assert_eq!(xv + yv, 4);
        assert_eq!(score, (xv + yv) as f64);
    }

    #[test]
    fn returns_unsatisfiable_when_no_solution() {
        let solver = CspSolver::new(
            |a: &Assignment<&'static str, i32>| {
                let x = a.get("x").copied();
                let y = a.get("y").copied();
                match (x, y) {
                    (Some(xv), Some(yv)) => xv + yv == 99,
                    _ => true,
                }
            },
            |_v, _val: &i32| 0.0,
        )
        .variable("x", vec![1, 2, 3])
        .variable("y", vec![1, 2, 3]);

        assert!(matches!(solver.solve(), Err(CspError::Unsatisfiable)));
    }

    #[test]
    fn excludes_inf_cost_values() {
        // Domain has bogus value 99; unary cost INF should exclude it.
        let solver = CspSolver::new(
            |_a: &Assignment<&'static str, i32>| true,
            |_v, val: &i32| if *val == 99 { f64::INFINITY } else { *val as f64 },
        )
        .variable("x", vec![1, 2, 99]);

        let (a, _) = solver.solve().unwrap();
        assert_eq!(a.get("x").copied().unwrap(), 1);
    }

    #[test]
    fn picks_minimum_total_cost() {
        // x in {1,2,3}, y in {1,2,3}. Constraint x != y. Pick lowest sum.
        let solver = CspSolver::new(
            |a: &Assignment<&'static str, i32>| {
                let x = a.get("x").copied();
                let y = a.get("y").copied();
                match (x, y) {
                    (Some(xv), Some(yv)) => xv != yv,
                    _ => true,
                }
            },
            |_v, val: &i32| *val as f64,
        )
        .variable("x", vec![3, 2, 1])
        .variable("y", vec![3, 2, 1]);

        let (a, score) = solver.solve().unwrap();
        let xv = a.get("x").copied().unwrap();
        let yv = a.get("y").copied().unwrap();
        assert_ne!(xv, yv);
        assert_eq!(score, (xv + yv) as f64);
        // Lowest sum with x != y is (1, 2) or (2, 1) = 3.
        assert_eq!(xv + yv, 3);
    }
}
