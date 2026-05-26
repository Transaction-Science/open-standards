//! Ternary-weight matrix: each weight ∈ {-1, 0, +1}, packed 4 per byte
//! (2 bits per trit), with a per-row f32 scale.
//!
//! Encoding (one byte, low bits first):
//!
//!   bit  7 6 | 5 4 | 3 2 | 1 0
//!   trit  3  |  2  |  1  |  0
//!
//! Trit value: `00 → 0`, `01 → +1`, `10 → -1`, `11 → reserved (treated as 0)`.
//!
//! For a logical M×N matrix the storage is M rows of `ceil(N/4)` packed bytes,
//! plus M f32 scales. Total bytes: `M * (ceil(N/4) + 4)` ≈ 2.25 bits per weight
//! plus a row scale that amortises to ≈ 0 bpw for large N.

use std::fmt;

/// Encoded ternary value packed into 2 bits.
const TRIT_ZERO: u8 = 0b00;
const TRIT_POS1: u8 = 0b01;
const TRIT_NEG1: u8 = 0b10;

#[derive(Debug, Clone, PartialEq)]
pub enum TernaryError {
    DimensionMismatch { expected: usize, got: usize },
    EmptyMatrix,
}

impl fmt::Display for TernaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::EmptyMatrix => write!(f, "matrix has zero rows or columns"),
        }
    }
}

impl std::error::Error for TernaryError {}

#[derive(Debug, Clone)]
pub struct TernaryMatrix {
    rows: usize,
    cols: usize,
    /// Per-row f32 scale. Length = `rows`.
    pub scales: Vec<f32>,
    /// Packed trits, row-major. Length = `rows * row_bytes(cols)`.
    pub data: Vec<u8>,
}

/// Bytes per packed row for `cols` trits at 4 trits/byte.
#[inline]
pub fn row_bytes(cols: usize) -> usize {
    (cols + 3) / 4
}

impl TernaryMatrix {
    /// Construct from raw f32 weights, row-major. For each row the scale is
    /// chosen as the row's mean absolute value, and each weight is rounded
    /// to the nearest of {−scale, 0, +scale}. This is the simplest
    /// post-training quantizer; quantization-aware-trained weights
    /// generally produce better results.
    pub fn from_f32(rows: usize, cols: usize, w: &[f32]) -> Result<Self, TernaryError> {
        if rows == 0 || cols == 0 {
            return Err(TernaryError::EmptyMatrix);
        }
        if w.len() != rows * cols {
            return Err(TernaryError::DimensionMismatch {
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
            // Decision threshold: weights below half the scale round to 0.
            let thresh = mean_abs * 0.5;
            scales.push(mean_abs.max(f32::MIN_POSITIVE));
            let row_offset = r * rb;
            for c in 0..cols {
                let v = row[c];
                let trit = if v > thresh {
                    TRIT_POS1
                } else if v < -thresh {
                    TRIT_NEG1
                } else {
                    TRIT_ZERO
                };
                let byte_idx = c / 4;
                let slot = c % 4;
                data[row_offset + byte_idx] |= trit << (slot * 2);
            }
        }
        Ok(Self { rows, cols, scales, data })
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }

    /// Extract the trit at (row, col): returns -1, 0, or +1 (i8).
    #[inline]
    pub fn trit(&self, row: usize, col: usize) -> i8 {
        let rb = row_bytes(self.cols);
        let byte = self.data[row * rb + col / 4];
        let slot = col % 4;
        match (byte >> (slot * 2)) & 0b11 {
            TRIT_POS1 => 1,
            TRIT_NEG1 => -1,
            _ => 0,
        }
    }

    /// Reconstruct an f32 weight at (row, col).
    #[inline]
    pub fn at(&self, row: usize, col: usize) -> f32 {
        self.trit(row, col) as f32 * self.scales[row]
    }

    /// Dequantize the full matrix to f32 (row-major). Useful for testing.
    pub fn to_f32(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.rows * self.cols);
        for r in 0..self.rows {
            let s = self.scales[r];
            for c in 0..self.cols {
                out.push(self.trit(r, c) as f32 * s);
            }
        }
        out
    }

    /// y = W * x, where `x.len() == cols` and `y.len() == rows`. The inner
    /// loop is a conditional add/subtract (no float multiply), then a single
    /// f32 multiply per output row for the scale.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) -> Result<(), TernaryError> {
        if x.len() != self.cols {
            return Err(TernaryError::DimensionMismatch {
                expected: self.cols,
                got: x.len(),
            });
        }
        if y.len() != self.rows {
            return Err(TernaryError::DimensionMismatch {
                expected: self.rows,
                got: y.len(),
            });
        }
        let rb = row_bytes(self.cols);
        for r in 0..self.rows {
            let row_offset = r * rb;
            let mut acc: f32 = 0.0;
            let mut c = 0;
            // Unrolled inner loop: 4 trits per byte.
            for byte_idx in 0..rb {
                let byte = self.data[row_offset + byte_idx];
                for slot in 0..4 {
                    if c >= self.cols { break; }
                    match (byte >> (slot * 2)) & 0b11 {
                        TRIT_POS1 => acc += x[c],
                        TRIT_NEG1 => acc -= x[c],
                        _ => {} // zero or reserved: skip
                    }
                    c += 1;
                }
            }
            y[r] = acc * self.scales[r];
        }
        Ok(())
    }

    /// Estimated joules for a matvec on this matrix.
    /// Cost model: per nonzero trit ≈ 1 pJ (one float add), per row ≈ 1 pJ
    /// (the scale multiply), plus a 10 nJ dispatch floor. This is a
    /// first-order static estimate; R29+ will calibrate against measurement.
    pub fn matvec_joules(&self) -> f64 {
        const ENERGY_PER_OP_PJ: f64 = 1.0;
        const DISPATCH_FLOOR_NJ: f64 = 10.0;
        let nonzero = self.count_nonzero() as f64;
        let scale_ops = self.rows as f64;
        (nonzero + scale_ops) * ENERGY_PER_OP_PJ * 1e-12
            + DISPATCH_FLOOR_NJ * 1e-9
    }

    /// Number of nonzero trits.
    pub fn count_nonzero(&self) -> usize {
        let rb = row_bytes(self.cols);
        let mut n = 0usize;
        for r in 0..self.rows {
            let row_offset = r * rb;
            let mut c = 0;
            for byte_idx in 0..rb {
                let byte = self.data[row_offset + byte_idx];
                for slot in 0..4 {
                    if c >= self.cols { break; }
                    let t = (byte >> (slot * 2)) & 0b11;
                    if t == TRIT_POS1 || t == TRIT_NEG1 {
                        n += 1;
                    }
                    c += 1;
                }
            }
        }
        n
    }
}

/// Reference matvec on a row-major f32 matrix. Used to verify
/// `TernaryMatrix::matvec` against a known-correct path.
pub fn matvec_f32(rows: usize, cols: usize, w: &[f32], x: &[f32], y: &mut [f32]) {
    for r in 0..rows {
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += w[r * cols + c] * x[c];
        }
        y[r] = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_preserves_sign_pattern() {
        // 1×8 row with mixed signs and zeros.
        let w = vec![1.0_f32, -1.0, 0.0, 0.5, -0.5, 0.0, 0.9, -0.9];
        let m = TernaryMatrix::from_f32(1, 8, &w).unwrap();
        // Signs preserved for non-near-zero entries; sub-threshold rounds to zero.
        let q = m.to_f32();
        assert!(q[0] > 0.0, "1.0 → positive trit");
        assert!(q[1] < 0.0, "-1.0 → negative trit");
        assert_eq!(q[2], 0.0, "0.0 → zero trit");
        assert!(q[6] > 0.0, "0.9 → positive trit");
        assert!(q[7] < 0.0, "-0.9 → negative trit");
    }

    #[test]
    fn matvec_matches_f32_when_weights_are_ternary() {
        // If all weights are already in {-s, 0, +s} for some s, the
        // ternary matvec must agree exactly with the f32 matvec.
        let rows = 4;
        let cols = 12;
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| match i % 3 { 0 => 1.0, 1 => -1.0, _ => 0.0 })
            .collect();
        let m = TernaryMatrix::from_f32(rows, cols, &w).unwrap();
        let dequant = m.to_f32();

        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1).collect();
        let mut y_ternary = vec![0.0_f32; rows];
        let mut y_f32 = vec![0.0_f32; rows];
        m.matvec(&x, &mut y_ternary).unwrap();
        matvec_f32(rows, cols, &dequant, &x, &mut y_f32);

        for r in 0..rows {
            assert!(
                (y_ternary[r] - y_f32[r]).abs() < 1e-5,
                "row {}: ternary={} f32={}", r, y_ternary[r], y_f32[r]
            );
        }
    }

    #[test]
    fn matvec_dimension_errors_surface() {
        let m = TernaryMatrix::from_f32(2, 4, &[0.0; 8]).unwrap();
        let x = vec![0.0_f32; 3];
        let mut y = vec![0.0_f32; 2];
        assert!(matches!(
            m.matvec(&x, &mut y),
            Err(TernaryError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn count_nonzero_is_accurate() {
        let w = vec![1.0_f32, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0];
        let m = TernaryMatrix::from_f32(1, 8, &w).unwrap();
        assert_eq!(m.count_nonzero(), 5);
    }

    #[test]
    fn joule_estimate_scales_with_nonzero_count() {
        // All-zero weights → only dispatch floor.
        let m_zero = TernaryMatrix::from_f32(2, 4, &vec![0.0_f32; 8]).unwrap();
        let j_zero = m_zero.matvec_joules();

        // All-nonzero weights → much higher.
        let m_full = TernaryMatrix::from_f32(2, 4, &vec![1.0_f32; 8]).unwrap();
        let j_full = m_full.matvec_joules();

        assert!(j_full > j_zero);
        // Sanity: both well under a microjoule for tiny matrices.
        assert!(j_full < 1e-6);
    }
}
