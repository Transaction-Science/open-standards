//! N-Queens: place N queens on an N×N board so none attack another.
//!
//! Energy = number of attacking pairs; a zero-energy placement is a
//! solution. Solved by column-by-column backtracking (one queen per
//! column, search rows).

use crate::EnergyFunction;

#[derive(Debug, Clone, PartialEq)]
pub enum NQueensError {
    BadN(String),
    TooLarge { n: usize, max: usize },
}

impl std::fmt::Display for NQueensError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadN(s) => write!(f, "n-queens: cannot parse N from {:?}", s),
            Self::TooLarge { n, max } => {
                write!(f, "n-queens: N={} exceeds cap {}", n, max)
            }
        }
    }
}

impl std::error::Error for NQueensError {}

#[derive(Debug, Clone)]
pub struct NQueens {
    pub n: usize,
}

/// A placement: `rows[col]` = the row of the queen in that column.
pub type Placement = Vec<usize>;

impl NQueens {
    pub fn parse(input: &str) -> Result<Self, NQueensError> {
        let n: usize = input
            .trim()
            .parse()
            .map_err(|_| NQueensError::BadN(input.to_string()))?;
        if n > 64 {
            return Err(NQueensError::TooLarge { n, max: 64 });
        }
        Ok(NQueens { n })
    }

    /// True if placing a queen at (row, col) doesn't attack any queen
    /// already placed in columns `0..col`.
    fn safe(rows: &[usize], col: usize, row: usize) -> bool {
        for (c, &r) in rows.iter().enumerate().take(col) {
            if r == row {
                return false; // same row
            }
            let dc = col - c;
            let dr = r.abs_diff(row);
            if dc == dr {
                return false; // same diagonal
            }
        }
        true
    }

    /// Solve via backtracking. Returns one solution or None
    /// (no solution exists for n=2,3).
    pub fn solve(&self, max_steps: usize) -> Option<Placement> {
        if self.n == 0 {
            return Some(vec![]);
        }
        let mut rows = vec![usize::MAX; self.n];
        let mut steps = 0usize;
        if self.place(&mut rows, 0, &mut steps, max_steps) {
            Some(rows)
        } else {
            None
        }
    }

    fn place(
        &self,
        rows: &mut [usize],
        col: usize,
        steps: &mut usize,
        max_steps: usize,
    ) -> bool {
        if col == self.n {
            return true;
        }
        for row in 0..self.n {
            *steps += 1;
            if *steps > max_steps {
                return false;
            }
            if Self::safe(rows, col, row) {
                rows[col] = row;
                if self.place(rows, col + 1, steps, max_steps) {
                    return true;
                }
                rows[col] = usize::MAX;
            }
        }
        false
    }

    pub fn render(&self, placement: &Placement) -> String {
        let mut s = format!("{}-queens solution:\n", self.n);
        for r in 0..self.n {
            for c in 0..self.n {
                s.push(if placement[c] == r { 'Q' } else { '.' });
                s.push(' ');
            }
            s.push('\n');
        }
        let cols: Vec<String> = placement.iter().map(|r| (r + 1).to_string()).collect();
        s.push_str(&format!("  rows per column: [{}]\n", cols.join(", ")));
        s
    }
}

/// Energy = attacking pairs (0 ⇒ valid solution).
impl EnergyFunction for NQueens {
    type State = Placement;
    fn energy(&self, state: &Self::State) -> f64 {
        let mut attacks = 0usize;
        for i in 0..state.len() {
            for j in (i + 1)..state.len() {
                if state[i] == state[j] || (j - i) == state[i].abs_diff(state[j]) {
                    attacks += 1;
                }
            }
        }
        attacks as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_n() {
        assert_eq!(NQueens::parse("8").unwrap().n, 8);
        assert!(matches!(NQueens::parse("x"), Err(NQueensError::BadN(_))));
        assert!(matches!(
            NQueens::parse("100"),
            Err(NQueensError::TooLarge { .. })
        ));
    }

    #[test]
    fn no_solution_for_n2_and_n3() {
        assert!(NQueens { n: 2 }.solve(10_000).is_none());
        assert!(NQueens { n: 3 }.solve(10_000).is_none());
    }

    #[test]
    fn solves_n4() {
        let q = NQueens { n: 4 };
        let sol = q.solve(10_000).unwrap();
        assert_eq!(sol.len(), 4);
        assert_eq!(q.energy(&sol), 0.0);
    }

    #[test]
    fn solves_n8() {
        let q = NQueens { n: 8 };
        let sol = q.solve(1_000_000).unwrap();
        assert_eq!(q.energy(&sol), 0.0);
    }

    #[test]
    fn solves_n16_within_budget() {
        let q = NQueens { n: 16 };
        let sol = q.solve(5_000_000).unwrap();
        assert_eq!(q.energy(&sol), 0.0);
    }

    #[test]
    fn solution_is_deterministic() {
        let q = NQueens { n: 8 };
        assert_eq!(q.solve(1_000_000), q.solve(1_000_000));
    }

    #[test]
    fn energy_nonzero_for_attacking_placement() {
        let q = NQueens { n: 4 };
        // All queens on row 0 → many attacks.
        assert!(q.energy(&vec![0, 0, 0, 0]) > 0.0);
    }
}
