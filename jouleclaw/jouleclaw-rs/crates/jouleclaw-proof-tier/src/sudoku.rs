//! Sudoku ⇄ CNF translation.
//!
//! We encode a Sudoku grid (4×4 or 9×9) as a Boolean satisfiability
//! problem so the bundled [`DpllSolver`][crate::DpllSolver] can solve
//! it. The encoding is the textbook one (one variable per
//! (row, col, digit) triple, plus exactly-one constraints over rows,
//! columns, and boxes). It is intentionally not the most efficient
//! encoding in the literature — that would obscure the point. A
//! caller who wants 9×9 solved at native speed plugs in a stronger
//! [`Solver`][crate::Solver] (e.g. an AC-3 + MRV implementation); the
//! tier surface does not change.

use crate::solver::SolverError;

/// Which grid size the encoder is working with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SudokuSize {
    /// 4×4 Sudoku: 16 cells, 16 vars × 4 digits = 64 variables.
    N4,
    /// 9×9 Sudoku: 81 cells, 81 × 9 = 729 variables.
    N9,
}

impl SudokuSize {
    /// Edge length (4 or 9).
    pub fn n(self) -> usize {
        match self {
            SudokuSize::N4 => 4,
            SudokuSize::N9 => 9,
        }
    }
    /// Box edge length (2 or 3).
    pub fn box_edge(self) -> usize {
        match self {
            SudokuSize::N4 => 2,
            SudokuSize::N9 => 3,
        }
    }
}

/// Encode a Sudoku grid string as CNF.
///
/// The grid is read row-major. `.`, `0`, and `_` mean empty. Any other
/// character must be a digit in the valid range (1–4 or 1–9).
///
/// Returns `(clauses, n_vars)` where `n_vars = n * n * n`. Variables
/// are numbered from 1 (DIMACS-style) and computed as
/// `var(r, c, d) = r * n * n + c * n + (d - 1) + 1`.
pub fn sudoku_to_cnf(
    grid: &str,
    size: SudokuSize,
) -> Result<(Vec<Vec<i32>>, usize), SolverError> {
    let n = size.n();
    let cells: Vec<char> = grid.chars().filter(|c| !c.is_whitespace()).collect();
    if cells.len() != n * n {
        return Err(SolverError::Malformed(format!(
            "sudoku grid must be {} chars (got {})",
            n * n,
            cells.len()
        )));
    }

    let var = |r: usize, c: usize, d: usize| -> i32 {
        (r * n * n + c * n + (d - 1) + 1) as i32
    };

    let mut clauses: Vec<Vec<i32>> = Vec::new();

    // 1. Each cell takes at least one digit.
    for r in 0..n {
        for c in 0..n {
            let mut cl = Vec::with_capacity(n);
            for d in 1..=n {
                cl.push(var(r, c, d));
            }
            clauses.push(cl);
        }
    }

    // 2. Each cell takes at most one digit (pairwise exclusion).
    for r in 0..n {
        for c in 0..n {
            for d1 in 1..=n {
                for d2 in (d1 + 1)..=n {
                    clauses.push(vec![-var(r, c, d1), -var(r, c, d2)]);
                }
            }
        }
    }

    // 3. Each digit appears at most once in each row.
    for r in 0..n {
        for d in 1..=n {
            for c1 in 0..n {
                for c2 in (c1 + 1)..n {
                    clauses.push(vec![-var(r, c1, d), -var(r, c2, d)]);
                }
            }
        }
    }

    // 4. Each digit appears at most once in each column.
    for c in 0..n {
        for d in 1..=n {
            for r1 in 0..n {
                for r2 in (r1 + 1)..n {
                    clauses.push(vec![-var(r1, c, d), -var(r2, c, d)]);
                }
            }
        }
    }

    // 5. Each digit appears at most once in each box.
    let bx = size.box_edge();
    for br in 0..bx {
        for bc in 0..bx {
            for d in 1..=n {
                let mut cells_in_box: Vec<(usize, usize)> = Vec::new();
                for r in 0..bx {
                    for c in 0..bx {
                        cells_in_box.push((br * bx + r, bc * bx + c));
                    }
                }
                for i in 0..cells_in_box.len() {
                    for j in (i + 1)..cells_in_box.len() {
                        let (r1, c1) = cells_in_box[i];
                        let (r2, c2) = cells_in_box[j];
                        clauses.push(vec![-var(r1, c1, d), -var(r2, c2, d)]);
                    }
                }
            }
        }
    }

    // 6. Unit clauses for clues already filled in the input.
    for (idx, &ch) in cells.iter().enumerate() {
        let r = idx / n;
        let c = idx % n;
        if ch == '.' || ch == '0' || ch == '_' {
            continue;
        }
        let d = ch.to_digit(10).ok_or_else(|| {
            SolverError::Malformed(format!("non-digit '{}' in sudoku grid", ch))
        })? as usize;
        if d < 1 || d > n {
            return Err(SolverError::Malformed(format!(
                "digit {} out of range for {}x{} sudoku",
                d, n, n
            )));
        }
        clauses.push(vec![var(r, c, d)]);
    }

    Ok((clauses, n * n * n))
}

/// Decode a SAT model back into a compact Sudoku grid string.
///
/// `model[i]` is the truth of variable `i + 1`. The decoder scans
/// every cell and emits the digit whose variable is true; if no
/// variable is true for some cell (which would indicate a solver bug)
/// it emits `.` for that cell rather than panicking, so the receipt
/// still shows the solver's actual claim rather than disappearing
/// behind an unwrap.
pub fn decode_sudoku_assignment(
    model: &[bool],
    size: SudokuSize,
) -> Result<String, SolverError> {
    let n = size.n();
    if model.len() < n * n * n {
        return Err(SolverError::Malformed(format!(
            "model too short: {} < {} expected",
            model.len(),
            n * n * n
        )));
    }
    let mut out = String::with_capacity(n * n);
    for r in 0..n {
        for c in 0..n {
            let mut emitted: Option<usize> = None;
            for d in 1..=n {
                let v = r * n * n + c * n + (d - 1);
                if model[v] {
                    emitted = Some(d);
                    break;
                }
            }
            match emitted {
                Some(d) => {
                    // Safe: d is in 1..=9, so as u32 fits in a char.
                    let ch = char::from_digit(d as u32, 10).unwrap_or('?');
                    out.push(ch);
                }
                None => out.push('.'),
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_length() {
        let err = sudoku_to_cnf("...", SudokuSize::N4).unwrap_err();
        assert!(matches!(err, SolverError::Malformed(_)));
    }

    #[test]
    fn empty_4x4_grid_encodes() {
        let (clauses, n_vars) = sudoku_to_cnf(&".".repeat(16), SudokuSize::N4)
            .expect("encode empty 4x4");
        assert_eq!(n_vars, 64);
        // 16 at-least-one clauses, plus at-most-one (pairwise) clauses.
        assert!(clauses.len() > 100);
        // No unit clauses (no clues).
        assert!(clauses.iter().all(|cl| cl.len() != 1));
    }

    #[test]
    fn clues_become_unit_clauses() {
        let grid = "1...".to_string() + &".".repeat(12);
        let (clauses, _) = sudoku_to_cnf(&grid, SudokuSize::N4).expect("encode");
        let units: Vec<_> = clauses.iter().filter(|cl| cl.len() == 1).collect();
        assert_eq!(units.len(), 1);
    }

    #[test]
    fn non_digit_rejected() {
        let bad = "x".to_string() + &".".repeat(15);
        let err = sudoku_to_cnf(&bad, SudokuSize::N4).unwrap_err();
        assert!(matches!(err, SolverError::Malformed(_)));
    }

    #[test]
    fn out_of_range_digit_rejected() {
        // 5 is out of range for a 4×4.
        let bad = "5".to_string() + &".".repeat(15);
        let err = sudoku_to_cnf(&bad, SudokuSize::N4).unwrap_err();
        assert!(matches!(err, SolverError::Malformed(_)));
    }

    #[test]
    fn decoder_round_trips_a_known_grid() {
        // Construct a model that says var(r, c, d) where d = (r+c) % 4 + 1.
        // (Not a valid sudoku — we're just exercising the decoder.)
        let size = SudokuSize::N4;
        let n = size.n();
        let mut model = vec![false; n * n * n];
        for r in 0..n {
            for c in 0..n {
                let d = (r + c) % n + 1;
                let v = r * n * n + c * n + (d - 1);
                model[v] = true;
            }
        }
        let s = decode_sudoku_assignment(&model, size).expect("decode");
        assert_eq!(s.len(), n * n);
        assert!(s.chars().all(|ch| ch.is_ascii_digit()));
    }
}
