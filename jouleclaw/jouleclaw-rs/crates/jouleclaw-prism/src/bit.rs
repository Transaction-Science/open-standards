//! 1-bit-weight matrix: each weight ∈ {-1, +1}, packed 8 per byte,
//! with a per-row f32 scale.
//!
//! Encoding: each bit `b` of a byte represents one weight, low bit first.
//! `0 → -1`, `1 → +1`. Pure 1 bit per weight.
//!
//! For a logical M×N matrix the storage is M rows of `ceil(N/8)` packed
//! bytes, plus M f32 scales. Total: `M * (ceil(N/8) + 4)` ≈ 1.5 bpw for
//! large N.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum BitError {
    DimensionMismatch { expected: usize, got: usize },
    EmptyMatrix,
}

impl fmt::Display for BitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::EmptyMatrix => write!(f, "matrix has zero rows or columns"),
        }
    }
}

impl std::error::Error for BitError {}

#[derive(Debug, Clone)]
pub struct BitMatrix {
    rows: usize,
    cols: usize,
    /// Per-row f32 scale.
    pub scales: Vec<f32>,
    /// Packed sign bits, row-major.
    pub data: Vec<u8>,
}

#[inline]
pub fn row_bytes(cols: usize) -> usize {
    (cols + 7) / 8
}

impl BitMatrix {
    /// Quantize from row-major f32 weights via sign-only encoding.
    /// Each row's scale is the mean absolute value of the row.
    pub fn from_f32(rows: usize, cols: usize, w: &[f32]) -> Result<Self, BitError> {
        if rows == 0 || cols == 0 {
            return Err(BitError::EmptyMatrix);
        }
        if w.len() != rows * cols {
            return Err(BitError::DimensionMismatch {
                expected: rows * cols,
                got: w.len(),
            });
        }
        let rb = row_bytes(cols);
        let mut scales = Vec::with_capacity(rows);
        let mut data = vec![0u8; rows * rb];
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let mean_abs = row.iter().map(|v| v.abs()).sum::<f32>() / cols as f32;
            scales.push(mean_abs.max(f32::MIN_POSITIVE));
            let row_offset = r * rb;
            for c in 0..cols {
                // Sign-only: positive bit → +1, otherwise -1.
                // Treat exact zeros as +1 to keep the scheme strictly 1-bit
                // (no third state). Real Bonsai training uses ternary then
                // collapses by post-training conditions.
                let bit = if row[c] >= 0.0 { 1u8 } else { 0u8 };
                let byte_idx = c / 8;
                let slot = c % 8;
                data[row_offset + byte_idx] |= bit << slot;
            }
        }
        Ok(Self { rows, cols, scales, data })
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }

    /// Extract the bit at (row, col): returns -1 or +1 (i8).
    #[inline]
    pub fn sign(&self, row: usize, col: usize) -> i8 {
        let rb = row_bytes(self.cols);
        let byte = self.data[row * rb + col / 8];
        let slot = col % 8;
        if ((byte >> slot) & 1) == 1 { 1 } else { -1 }
    }

    #[inline]
    pub fn at(&self, row: usize, col: usize) -> f32 {
        self.sign(row, col) as f32 * self.scales[row]
    }

    pub fn to_f32(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.rows * self.cols);
        for r in 0..self.rows {
            let s = self.scales[r];
            for c in 0..self.cols {
                out.push(self.sign(r, c) as f32 * s);
            }
        }
        out
    }

    /// y = W * x. Each inner loop step is a conditional add or subtract;
    /// one float multiply per output row for the scale.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) -> Result<(), BitError> {
        if x.len() != self.cols {
            return Err(BitError::DimensionMismatch {
                expected: self.cols,
                got: x.len(),
            });
        }
        if y.len() != self.rows {
            return Err(BitError::DimensionMismatch {
                expected: self.rows,
                got: y.len(),
            });
        }
        let rb = row_bytes(self.cols);
        for r in 0..self.rows {
            let row_offset = r * rb;
            let mut acc: f32 = 0.0;
            let mut c = 0;
            for byte_idx in 0..rb {
                let byte = self.data[row_offset + byte_idx];
                for slot in 0..8 {
                    if c >= self.cols { break; }
                    if ((byte >> slot) & 1) == 1 {
                        acc += x[c];
                    } else {
                        acc -= x[c];
                    }
                    c += 1;
                }
            }
            y[r] = acc * self.scales[r];
        }
        Ok(())
    }

    /// Cost model: every bit contributes one add/subtract (no skip), plus
    /// one scale multiply per output row, plus the dispatch floor.
    pub fn matvec_joules(&self) -> f64 {
        const ENERGY_PER_OP_PJ: f64 = 1.0;
        const DISPATCH_FLOOR_NJ: f64 = 10.0;
        let ops = (self.rows * self.cols + self.rows) as f64;
        ops * ENERGY_PER_OP_PJ * 1e-12 + DISPATCH_FLOOR_NJ * 1e-9
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ternary::matvec_f32;

    #[test]
    fn roundtrip_preserves_sign() {
        let w = vec![1.0_f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.0, -0.0];
        let m = BitMatrix::from_f32(1, 8, &w).unwrap();
        // All non-negative inputs map to +1; negatives map to -1.
        let q = m.to_f32();
        assert!(q[0] > 0.0);
        assert!(q[1] < 0.0);
        assert!(q[2] > 0.0);
        assert!(q[3] < 0.0);
    }

    #[test]
    fn matvec_matches_f32_when_weights_are_signed() {
        let rows = 4;
        let cols = 16;
        // Weights in {-1, +1} so dequant is exact.
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let m = BitMatrix::from_f32(rows, cols, &w).unwrap();
        let dequant = m.to_f32();

        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let mut y_bit = vec![0.0_f32; rows];
        let mut y_f32 = vec![0.0_f32; rows];
        m.matvec(&x, &mut y_bit).unwrap();
        matvec_f32(rows, cols, &dequant, &x, &mut y_f32);

        for r in 0..rows {
            assert!(
                (y_bit[r] - y_f32[r]).abs() < 1e-5,
                "row {}: bit={} f32={}", r, y_bit[r], y_f32[r]
            );
        }
    }

    #[test]
    fn dimension_errors_surface() {
        let m = BitMatrix::from_f32(2, 8, &vec![1.0_f32; 16]).unwrap();
        let x = vec![0.0_f32; 7];
        let mut y = vec![0.0_f32; 2];
        assert!(matches!(
            m.matvec(&x, &mut y),
            Err(BitError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn joules_scale_with_size() {
        let small = BitMatrix::from_f32(4, 8, &vec![1.0_f32; 32]).unwrap();
        let large = BitMatrix::from_f32(64, 128, &vec![1.0_f32; 8192]).unwrap();
        assert!(large.matvec_joules() > small.matvec_joules());
        // Both still sub-microjoule for these tiny sizes.
        assert!(large.matvec_joules() < 1e-6);
    }
}
