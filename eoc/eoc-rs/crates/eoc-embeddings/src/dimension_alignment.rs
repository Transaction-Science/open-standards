//! Dimension alignment between heterogeneous embedders.
//!
//! When the EOC KV stage uses one embedder for stored vectors and a
//! different one for incoming queries, dimensions don't match. Two
//! alignment strategies are provided here:
//!
//! * **Matryoshka-style truncation** — used by OpenAI v3,
//!   nomic-embed-text-v1.5, and mxbai-embed-large-v1, which produce
//!   embeddings whose leading components carry the most signal. Truncating
//!   to a smaller dimension is lossy but information-preserving.
//! * **Zero-padding** — for models without Matryoshka structure, padding
//!   to a larger dimension is the safe fallback. Cosine similarity is
//!   *not* preserved across dimensions when models differ; document this
//!   to callers.
//!
//! Cross-vendor similarity is generally unreliable even after dimension
//! alignment — different training corpora produce different geometries.
//! [`requires_alignment`] is a structural check, not a semantic one.

use crate::embedder::Embedder;

/// Project `vector` into `target_dim`.
///
/// * `target_dim < vector.len()` → Matryoshka-style truncation (drop tail).
/// * `target_dim > vector.len()` → zero-padding (append zeros).
/// * `target_dim == vector.len()` → returned as a copy.
///
/// Truncation preserves cosine similarity *only* for models that emit
/// Matryoshka embeddings; for arbitrary models it is lossy. Zero-padding
/// preserves cosine similarity exactly (the padded dot product and norms
/// are unchanged) but does not magically make vectors from different
/// embedders comparable.
pub fn project(vector: &[f32], target_dim: usize) -> Vec<f32> {
    if target_dim == vector.len() {
        return vector.to_vec();
    }
    if target_dim < vector.len() {
        return vector[..target_dim].to_vec();
    }
    let mut out = Vec::with_capacity(target_dim);
    out.extend_from_slice(vector);
    out.resize(target_dim, 0.0);
    out
}

/// Whether two embedders produce vectors that need alignment before
/// similarity comparison.
///
/// Returns `true` if the dimensions differ. (Even matched dimensions can
/// be semantically incomparable across vendors — this is a structural
/// check only.)
pub fn requires_alignment(a: &dyn Embedder, b: &dyn Embedder) -> bool {
    a.dimensions() != b.dimensions()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    #[test]
    fn project_identity() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(project(&v, 3), v);
    }

    #[test]
    fn project_truncates() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(project(&v, 2), vec![1.0, 2.0]);
    }

    #[test]
    fn project_zero_pads() {
        let v = vec![1.0_f32, 2.0];
        assert_eq!(project(&v, 4), vec![1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn matryoshka_truncation_preserves_relative_ordering() {
        // Synthetic Matryoshka-ish: leading dims carry most norm.
        let a = vec![0.9_f32, 0.4, 0.1, 0.05];
        let b = vec![0.85_f32, 0.42, 0.11, 0.06];
        let c = vec![-0.9_f32, 0.4, 0.1, 0.05];

        let ab_full = cosine(&a, &b);
        let ac_full = cosine(&a, &c);

        let a_t = project(&a, 2);
        let b_t = project(&b, 2);
        let c_t = project(&c, 2);
        let ab_t = cosine(&a_t, &b_t);
        let ac_t = cosine(&a_t, &c_t);

        // a/b are similar in both; a/c are dissimilar in both.
        assert!(ab_full > ac_full);
        assert!(ab_t > ac_t);
        assert!(ab_t > 0.9, "truncation should retain high a/b similarity");
    }

    #[test]
    fn zero_pad_preserves_cosine_against_self_padded() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0_f32, 2.5, 2.5];
        let c_full = cosine(&a, &b);
        let a_p = project(&a, 6);
        let b_p = project(&b, 6);
        let c_p = cosine(&a_p, &b_p);
        assert!((c_full - c_p).abs() < 1e-6);
    }
}
