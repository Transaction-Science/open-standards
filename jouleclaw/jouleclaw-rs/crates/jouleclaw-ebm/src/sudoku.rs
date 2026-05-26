//! `Sudoku` — 9×9 Sudoku as a concrete `EnergyFunction`.
//!
//! State: an 81-cell board, row-major, 0 = empty, 1-9 = filled.
//! Energy: count of cells that violate at least one of the three
//! Sudoku constraints (row uniqueness, column uniqueness, 3×3 box
//! uniqueness). Zero energy = a valid completed Sudoku.

use std::fmt;

use crate::EnergyFunction;

#[derive(Debug, Clone, PartialEq)]
pub enum SudokuError {
    BadLength { expected: usize, got: usize },
    BadCharacter { ch: char, pos: usize },
    DigitOutOfRange { val: u8, pos: usize },
}

impl fmt::Display for SudokuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadLength { expected, got } => {
                write!(f, "sudoku input must have exactly {} cells, got {}", expected, got)
            }
            Self::BadCharacter { ch, pos } => {
                write!(f, "sudoku char {:?} at position {} not a digit or '.'", ch, pos)
            }
            Self::DigitOutOfRange { val, pos } => {
                write!(f, "sudoku cell at {} has value {} (must be 0-9)", pos, val)
            }
        }
    }
}

impl std::error::Error for SudokuError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Sudoku {
    /// 81 cells, row-major. 0 = empty, 1-9 = filled.
    pub cells: [u8; 81],
}

impl Sudoku {
    /// Parse a Sudoku from a string. Each non-whitespace character is
    /// either a digit `1`..`9` (filled), `0` or `.` (empty). Whitespace
    /// is ignored, so multi-line boards work.
    pub fn parse(input: &str) -> Result<Self, SudokuError> {
        let mut cells = [0u8; 81];
        let mut i = 0usize;
        for (pos, ch) in input.chars().enumerate() {
            if ch.is_whitespace() {
                continue;
            }
            let v = match ch {
                '.' | '0' => 0,
                '1'..='9' => (ch as u8) - b'0',
                _ => return Err(SudokuError::BadCharacter { ch, pos }),
            };
            if i >= 81 {
                return Err(SudokuError::BadLength { expected: 81, got: i + 1 });
            }
            cells[i] = v;
            i += 1;
        }
        if i != 81 {
            return Err(SudokuError::BadLength { expected: 81, got: i });
        }
        Ok(Self { cells })
    }

    #[inline]
    pub fn at(&self, row: usize, col: usize) -> u8 {
        self.cells[row * 9 + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: u8) {
        self.cells[row * 9 + col] = val;
    }

    /// Energy-gradient cell selection: the empty cell with the *fewest*
    /// legal candidates (minimum-remaining-values). Assigning the most
    /// constrained variable is the steepest descent on the
    /// constraint-violation energy — it prunes the most search space per
    /// decision. This is the deterministic, hand-derived form of the
    /// variable-ordering policy a learned energy model (Konna) would
    /// induce. Returns None if the board is full; returns a cell with
    /// 0 candidates (a dead end) immediately so the caller backtracks.
    pub fn most_constrained_empty(&self) -> Option<(usize, usize)> {
        let mut best: Option<((usize, usize), u8)> = None;
        for r in 0..9 {
            for c in 0..9 {
                if self.at(r, c) != 0 {
                    continue;
                }
                let mut n = 0u8;
                for v in 1u8..=9 {
                    if self.is_legal(r, c, v) {
                        n += 1;
                    }
                }
                if n == 0 {
                    return Some((r, c)); // dead end — fail fast
                }
                match best {
                    Some((_, bn)) if bn <= n => {}
                    _ => best = Some(((r, c), n)),
                }
            }
        }
        best.map(|(rc, _)| rc)
    }

    /// Find the next empty cell, scanning row-major. Returns (row, col)
    /// or None if the board is full.
    pub fn next_empty(&self) -> Option<(usize, usize)> {
        for r in 0..9 {
            for c in 0..9 {
                if self.at(r, c) == 0 {
                    return Some((r, c));
                }
            }
        }
        None
    }

    /// True if placing `val` at (row, col) wouldn't violate any
    /// row/column/box uniqueness constraint.
    pub fn is_legal(&self, row: usize, col: usize, val: u8) -> bool {
        if val == 0 || val > 9 {
            return false;
        }
        // Row.
        for c in 0..9 {
            if c != col && self.at(row, c) == val {
                return false;
            }
        }
        // Column.
        for r in 0..9 {
            if r != row && self.at(r, col) == val {
                return false;
            }
        }
        // 3×3 box.
        let br = (row / 3) * 3;
        let bc = (col / 3) * 3;
        for r in br..br + 3 {
            for c in bc..bc + 3 {
                if (r, c) != (row, col) && self.at(r, c) == val {
                    return false;
                }
            }
        }
        true
    }

    /// True if every cell is filled and no constraint is violated.
    pub fn is_solved(&self) -> bool {
        self.cells.iter().all(|&v| (1..=9).contains(&v)) && self.energy_count() == 0
    }

    /// Count the number of cells whose value duplicates another in its
    /// row, column, or box. The energy function uses this.
    fn energy_count(&self) -> usize {
        let mut violations = 0usize;
        for r in 0..9 {
            for c in 0..9 {
                let v = self.at(r, c);
                if v == 0 {
                    // An empty cell is a violation too — the board isn't done.
                    violations += 1;
                    continue;
                }
                if !self.is_legal(r, c, v) {
                    violations += 1;
                }
            }
        }
        violations
    }

    /// Pretty-print the board with row/box separators.
    pub fn render(&self) -> String {
        let mut s = String::with_capacity(128);
        for r in 0..9 {
            if r % 3 == 0 && r > 0 {
                s.push_str("------+-------+------\n");
            }
            for c in 0..9 {
                if c % 3 == 0 && c > 0 {
                    s.push_str("| ");
                }
                let v = self.at(r, c);
                if v == 0 {
                    s.push('.');
                } else {
                    s.push((b'0' + v) as char);
                }
                s.push(' ');
            }
            s.push('\n');
        }
        s
    }

    /// Compact one-line form (81 chars, '.' for empty).
    pub fn render_compact(&self) -> String {
        let mut s = String::with_capacity(81);
        for &v in &self.cells {
            if v == 0 {
                s.push('.');
            } else {
                s.push((b'0' + v) as char);
            }
        }
        s
    }
}

impl EnergyFunction for Sudoku {
    type State = Sudoku;

    /// Energy = number of constraint violations + empty cells.
    /// A solved board has energy 0.
    fn energy(&self, state: &Self::State) -> f64 {
        state.energy_count() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A well-known easy Sudoku puzzle.
    const EASY: &str =
        "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";

    const EASY_SOLVED: &str =
        "534678912672195348198342567859761423426853791713924856961537284287419635345286179";

    #[test]
    fn parse_easy_puzzle() {
        let s = Sudoku::parse(EASY).unwrap();
        assert_eq!(s.at(0, 0), 5);
        assert_eq!(s.at(0, 1), 3);
        assert_eq!(s.at(0, 2), 0);
        assert_eq!(s.render_compact(), EASY);
    }

    #[test]
    fn parse_handles_whitespace() {
        let board = "5 3 . . 7 . . . .
                     6 . . 1 9 5 . . .
                     . 9 8 . . . . 6 .
                     8 . . . 6 . . . 3
                     4 . . 8 . 3 . . 1
                     7 . . . 2 . . . 6
                     . 6 . . . . 2 8 .
                     . . . 4 1 9 . . 5
                     . . . . 8 . . 7 9";
        let s = Sudoku::parse(board).unwrap();
        assert_eq!(s.render_compact(), EASY);
    }

    #[test]
    fn parse_rejects_wrong_length() {
        let bad = "12345";
        assert!(matches!(Sudoku::parse(bad), Err(SudokuError::BadLength { .. })));
    }

    #[test]
    fn parse_rejects_bad_character() {
        let mut bad = EASY.to_string();
        bad.replace_range(0..1, "X");
        assert!(matches!(Sudoku::parse(&bad), Err(SudokuError::BadCharacter { .. })));
    }

    #[test]
    fn is_legal_detects_row_conflict() {
        let s = Sudoku::parse(EASY).unwrap();
        // (0,0) has 5; placing another 5 in row 0 should be illegal.
        assert!(!s.is_legal(0, 2, 5));
        // But placing it where no conflict exists should be legal.
        assert!(s.is_legal(0, 2, 1));
    }

    #[test]
    fn is_legal_detects_box_conflict() {
        let s = Sudoku::parse(EASY).unwrap();
        // (0,0)=5, (1,0)=6, (2,1)=9, (2,2)=8 — top-left 3x3 has 5,3,6,9,8.
        // Try placing 5 at (2,2): same 3×3 box → illegal.
        assert!(!s.is_legal(2, 2, 5));
    }

    #[test]
    fn solved_puzzle_is_solved() {
        let s = Sudoku::parse(EASY_SOLVED).unwrap();
        assert!(s.is_solved());
    }

    #[test]
    fn unsolved_puzzle_is_not_solved() {
        let s = Sudoku::parse(EASY).unwrap();
        assert!(!s.is_solved());
    }

    #[test]
    fn energy_zero_on_solved_board() {
        let s = Sudoku::parse(EASY_SOLVED).unwrap();
        let e = s.energy(&s);
        assert_eq!(e, 0.0);
    }

    #[test]
    fn energy_positive_on_unsolved_board() {
        let s = Sudoku::parse(EASY).unwrap();
        let e = s.energy(&s);
        // 81 cells, many empty → energy ≈ count of empty cells.
        assert!(e > 0.0);
    }

    #[test]
    fn render_round_trips() {
        let s = Sudoku::parse(EASY_SOLVED).unwrap();
        let again = Sudoku::parse(&s.render_compact()).unwrap();
        assert_eq!(s, again);
    }
}
