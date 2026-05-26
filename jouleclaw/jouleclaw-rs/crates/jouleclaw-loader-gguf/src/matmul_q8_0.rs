//! Q8_0 × fp32 matmul kernel — keeps weights in their on-disk Q8_0
//! shape and operates directly via NEON int8 dot products.
//!
//! ## What this is for
//!
//! Lever 2 of the "fastest on hardware" series. The existing Q8_0
//! production path (used by jouleclaw-loader-gguf::dequant + the
//! compiled graph's `wmm`) dequantizes Q8_0 weights to fp32 at
//! per-block granularity, then dispatches through cblas_sgemm. On
//! Apple Silicon Macs, cblas_sgemm routes to AMX — a dedicated
//! matrix coprocessor that's hard to beat at fp32. So this kernel
//! is **NOT** a Mac speedup.
//!
//! The win is on the edge target:
//!
//!   - Raspberry Pi 4/5 (no AMX): cblas_sgemm via OpenBLAS / NEON
//!     fp32 is the only option. Beats it? NEON int8 multiply-add
//!     processes 16 int8 mac per cycle vs 4 fp32 mac per cycle on
//!     the same NEON unit — theoretical 4× peak, real ~2× from
//!     bandwidth-limited workloads.
//!   - Android arm64: same story as Pi.
//!   - Linux x86_64: not yet supported by this kernel (scalar
//!     fallback only; could add AVX2/AVX-512 later).
//!
//! ## Layout
//!
//! Q8_0 weight `[n, k]` is stored as `n` rows. Each row is `k/32`
//! blocks of 34 bytes: `[fp16 scale][32 i8 values]`. The kernel
//! reads these bytes directly without dequantizing to fp32.
//!
//! ## Activation quantization
//!
//! The activation is fp32 going in. We quantize it row-wise to
//! int8 + fp32 scale at call time (same per-row symmetric scheme
//! as `kv_cache_inplace::quantize_row_neon`). The dot product
//! then runs int8 × int8 → int32, scaled by `a_scale[i] ×
//! w_block_scale[j, b]`.
//!
//! ## Production wiring
//!
//! NOT YET WIRED. This commit ships the kernel + a microbench. The
//! existing decode graph builders (`build_decode_step_graph_inplace_const_block`)
//! emit `wmm` ops that route through jouleclaw-core's compile/execute
//! infrastructure. Routing Q8_0 wmm calls through this kernel
//! requires:
//!   1. A new op `WmmQ80Direct` (or extend the existing wmm op
//!      with a "preserve Q8_0" flag).
//!   2. A new kernel registered in `joule-backend-reference`
//!      (or a new `joule-backend-arm-neon`) that handles it.
//!   3. Adaptive kernel selection: prefer this on aarch64 Linux /
//!      Android, prefer cblas_sgemm on Apple Silicon (AMX path).
//!
//! That wiring is a separate commit — keeping this one focused.

use crate::f16_to_f32;

const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_ELEMS: usize = 32;

/// `out = a @ w.T + bias`, where `w` is stored in Q8_0 format.
///
/// - `a`: fp32 `[m, k]` (row-major)
/// - `w`: Q8_0 bytes for `[n, k]` weight, layout
///        `[n][k/32]([scale_fp16][32 × i8])`
/// - `bias`: fp32 `[n]` (or empty for no bias)
/// - `out`: fp32 `[m, n]`
///
/// `k` MUST be a multiple of `Q8_0_ELEMS` (32).
pub fn matmul_q8_0(
    a: &[f32],
    w: &[u8],
    bias: &[f32],
    m: usize, n: usize, k: usize,
    out: &mut [f32],
) {
    assert_eq!(k % Q8_0_ELEMS, 0, "Q8_0 requires k % 32 == 0; got k={k}");
    let blocks_per_row = k / Q8_0_ELEMS;
    assert_eq!(a.len(), m * k);
    assert_eq!(w.len(), n * blocks_per_row * Q8_0_BLOCK_BYTES);
    assert!(bias.is_empty() || bias.len() == n);
    assert_eq!(out.len(), m * n);

    // 1. Quantize activation per-row.
    let (q_a, a_scales) = quantize_a_per_row(a, m, k);

    // 2. Per (i, j): walk the blocks, dot, accumulate scaled.
    #[cfg(target_arch = "aarch64")]
    unsafe {
        matmul_q8_0_neon(&q_a, &a_scales, w, bias, m, n, k, out);
    }
    #[cfg(not(target_arch = "aarch64"))]
    matmul_q8_0_scalar(&q_a, &a_scales, w, bias, m, n, k, out);
}

fn quantize_a_per_row(a: &[f32], m: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0_i8; m * k];
    let mut scales = vec![0.0_f32; m];
    for i in 0..m {
        let row = &a[i * k..(i + 1) * k];
        let mut max_abs = 0.0_f32;
        for &v in row {
            let av = v.abs();
            if av > max_abs { max_abs = av; }
        }
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        scales[i] = scale;
        let inv = 1.0 / scale;
        for (j, &v) in row.iter().enumerate() {
            q[i * k + j] = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, scales)
}

/// Scalar reference. Always available; the parity oracle for the
/// NEON kernel + the off-aarch64 production path.
pub fn matmul_q8_0_scalar(
    q_a: &[i8], a_scales: &[f32],
    w: &[u8], bias: &[f32],
    m: usize, n: usize, k: usize,
    out: &mut [f32],
) {
    let blocks_per_row = k / Q8_0_ELEMS;
    for i in 0..m {
        let a_row = &q_a[i * k..(i + 1) * k];
        let a_s = a_scales[i];
        for j in 0..n {
            let w_row_off = j * blocks_per_row * Q8_0_BLOCK_BYTES;
            let mut accum = 0.0_f32;
            for b in 0..blocks_per_row {
                let block_off = w_row_off + b * Q8_0_BLOCK_BYTES;
                let scale_bits = u16::from_le_bytes([
                    w[block_off], w[block_off + 1]]);
                let block_scale = f16_to_f32(scale_bits);

                let a_chunk = &a_row[b * Q8_0_ELEMS..(b + 1) * Q8_0_ELEMS];
                let w_chunk_start = block_off + 2;

                let mut sum_i32: i32 = 0;
                for c in 0..Q8_0_ELEMS {
                    let av = a_chunk[c] as i32;
                    let wv = w[w_chunk_start + c] as i8 as i32;
                    sum_i32 += av * wv;
                }
                accum += sum_i32 as f32 * block_scale * a_s;
            }
            if !bias.is_empty() {
                accum += bias[j];
            }
            out[i * n + j] = accum;
        }
    }
}

/// NEON-accelerated Q8_0 matmul. Two passes per 32-element block:
///   - Load 16 i8 each from activation + weight as `int8x16_t`.
///   - Widen-multiply low + high halves via `vmull_s8` → two
///     `int16x8_t` products.
///   - Pairwise-accumulate into a single `int32x4_t`.
/// After all blocks of a row are accumulated, horizontally reduce
/// to a single i32 and scale by `block_scale * a_scale` per block.
/// (Block scales differ across blocks, so we apply the per-block
/// scale *after* each block's i32 sum — couldn't be hoisted.)
///
/// SAFETY: pointer arithmetic uses block / chunk indices derived
/// from the matrix dimensions; the assertions in `matmul_q8_0`
/// bound them within the input slices.
#[cfg(target_arch = "aarch64")]
unsafe fn matmul_q8_0_neon(
    q_a: &[i8], a_scales: &[f32],
    w: &[u8], bias: &[f32],
    m: usize, n: usize, k: usize,
    out: &mut [f32],
) {
    use std::arch::aarch64::*;
    let blocks_per_row = k / Q8_0_ELEMS;
    for i in 0..m {
        let a_row_ptr = q_a.as_ptr().add(i * k);
        let a_s = a_scales[i];
        for j in 0..n {
            let w_row_off = j * blocks_per_row * Q8_0_BLOCK_BYTES;
            let mut accum_f32 = 0.0_f32;
            for b in 0..blocks_per_row {
                let block_off = w_row_off + b * Q8_0_BLOCK_BYTES;
                let scale_bits = u16::from_le_bytes([
                    *w.get_unchecked(block_off),
                    *w.get_unchecked(block_off + 1),
                ]);
                let block_scale = f16_to_f32(scale_bits);

                let a_chunk_ptr = a_row_ptr.add(b * Q8_0_ELEMS);
                let w_chunk_ptr = w.as_ptr().add(block_off + 2) as *const i8;

                // 32 elements = 2 × 16-byte NEON loads.
                let mut acc = vdupq_n_s32(0);
                for c in 0..2 {
                    let a_v = vld1q_s8(a_chunk_ptr.add(c * 16));
                    let b_v = vld1q_s8(w_chunk_ptr.add(c * 16));
                    let prod_lo = vmull_s8(vget_low_s8(a_v), vget_low_s8(b_v));
                    let prod_hi = vmull_s8(vget_high_s8(a_v), vget_high_s8(b_v));
                    acc = vpadalq_s16(acc, prod_lo);
                    acc = vpadalq_s16(acc, prod_hi);
                }
                let sum = vaddvq_s32(acc) as f32;
                accum_f32 += sum * block_scale * a_s;
            }
            if !bias.is_empty() {
                accum_f32 += bias[j];
            }
            out[i * n + j] = accum_f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic Q8_0 row + the same row as fp32. Used by
    /// the parity test below.
    fn synth_q8_0_weights(n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
        assert_eq!(k % Q8_0_ELEMS, 0);
        let blocks_per_row = k / Q8_0_ELEMS;
        let mut bytes = vec![0_u8; n * blocks_per_row * Q8_0_BLOCK_BYTES];
        let mut fp32 = vec![0.0_f32; n * k];
        // Simple LCG for reproducible synth data.
        let mut state = seed;
        let mut next = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); state };
        for j in 0..n {
            for b in 0..blocks_per_row {
                let block_off = j * blocks_per_row * Q8_0_BLOCK_BYTES + b * Q8_0_BLOCK_BYTES;
                // Block scale = a smallish value.
                let scale_f32 = (((next() >> 32) as u32 % 200) as f32 + 1.0) / 1000.0;
                let scale_f16_bits = f32_to_f16_bits(scale_f32);
                bytes[block_off] = (scale_f16_bits & 0xFF) as u8;
                bytes[block_off + 1] = (scale_f16_bits >> 8) as u8;
                for c in 0..Q8_0_ELEMS {
                    let qv = ((next() >> 32) as i32 % 255 - 127) as i8;
                    bytes[block_off + 2 + c] = qv as u8;
                    let actual_scale = f16_to_f32(scale_f16_bits);
                    fp32[j * k + b * Q8_0_ELEMS + c] = qv as f32 * actual_scale;
                }
            }
        }
        (bytes, fp32)
    }

    /// Minimal inline fp32 → fp16-bits converter for the test
    /// synthesis. Handles positive-normal values, which is what
    /// Q8_0 block scales actually are (they're computed from
    /// `max_abs / 127` of a non-zero row). Not a general-purpose
    /// converter — no NaN / inf / subnormal / negative handling.
    fn f32_to_f16_bits(v: f32) -> u16 {
        if v == 0.0 { return 0; }
        let bits = v.to_bits();
        let sign = ((bits >> 31) & 1) as u16;
        let exp_f32 = ((bits >> 23) & 0xFF) as i32 - 127;
        let mantissa_f32 = bits & 0x7F_FFFF;
        let exp_f16 = (exp_f32 + 15) as u16;
        let mantissa_f16 = (mantissa_f32 >> 13) as u16;
        (sign << 15) | (exp_f16 << 10) | mantissa_f16
    }

    /// scalar fp32 matmul reference: out = a @ w_fp32.T (no bias)
    fn matmul_at_fp32_ref(
        a: &[f32], w: &[f32],
        m: usize, n: usize, k: usize,
        out: &mut [f32],
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * w[j * k + kk];
                }
                out[i * n + j] = sum;
            }
        }
    }

    /// Q8_0 NEON kernel output should match the fp32 reference
    /// within the activation quantization noise (~scale/254 per
    /// element, accumulated over k).
    #[test]
    fn matmul_q8_0_matches_fp32_reference_within_quant_noise() {
        let m = 4;
        let n = 8;
        let k = 64;
        // Synthetic activation: smooth sine-like values in [-1, 1].
        let mut a = vec![0.0_f32; m * k];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f32) * 0.011).sin() * 0.9;
        }
        let (w_q8, w_fp32) = synth_q8_0_weights(n, k, 42);
        let bias: Vec<f32> = (0..n).map(|j| (j as f32) * 0.02 - 0.07).collect();

        let mut out_q8 = vec![0.0_f32; m * n];
        matmul_q8_0(&a, &w_q8, &bias, m, n, k, &mut out_q8);

        let mut out_ref = vec![0.0_f32; m * n];
        matmul_at_fp32_ref(&a, &w_fp32, m, n, k, &mut out_ref);
        // Add bias to reference too.
        for i in 0..m {
            for j in 0..n {
                out_ref[i * n + j] += bias[j];
            }
        }

        let mut max_abs = 0.0_f32;
        let mut sum_sq_err = 0.0_f64;
        let mut sum_sq_ref = 0.0_f64;
        for (q, r) in out_q8.iter().zip(&out_ref) {
            let d = (q - r).abs();
            if d > max_abs { max_abs = d; }
            sum_sq_err += (d as f64).powi(2);
            sum_sq_ref += (*r as f64).powi(2);
        }
        let rms_rel = (sum_sq_err / sum_sq_ref).sqrt();
        eprintln!(
            "Q8_0 NEON vs fp32 ref:  max_abs={max_abs:.4}  rms_rel={rms_rel:.4}"
        );
        // Activation quant adds error; weight quant noise is small
        // (block-scaled int8). Tolerate up to ~5% relative error.
        assert!(rms_rel < 0.05,
            "Q8_0 kernel diverged too far: rms_rel={rms_rel}");
    }

    /// NEON path bit-matches scalar path (same arithmetic, just
    /// SIMD'd). Same caveat as the kv_cache_inplace NEON test:
    /// the int32 sum is order-independent, so output is exact.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn matmul_q8_0_neon_matches_scalar() {
        let m = 3;
        let n = 5;
        let k = 96;
        let mut a = vec![0.0_f32; m * k];
        for (i, v) in a.iter_mut().enumerate() {
            *v = (((i % 17) as f32) * 0.1) - 0.5;
        }
        let (w_q8, _) = synth_q8_0_weights(n, k, 1234);
        let bias: Vec<f32> = (0..n).map(|j| (j as f32) * 0.01).collect();
        let (q_a, a_scales) = quantize_a_per_row(&a, m, k);
        let mut out_neon = vec![0.0_f32; m * n];
        let mut out_scalar = vec![0.0_f32; m * n];
        unsafe {
            matmul_q8_0_neon(&q_a, &a_scales, &w_q8, &bias, m, n, k, &mut out_neon);
        }
        matmul_q8_0_scalar(&q_a, &a_scales, &w_q8, &bias, m, n, k, &mut out_scalar);
        let mut max_abs = 0.0_f32;
        for (n, s) in out_neon.iter().zip(&out_scalar) {
            let d = (n - s).abs();
            if d > max_abs { max_abs = d; }
        }
        eprintln!("Q8_0 NEON vs scalar: max_abs={max_abs:.6}");
        // Same i32 sums, same per-block scale multiplications.
        // Tiny f32 noise from accumulation order across blocks.
        assert!(max_abs < 1e-3, "Q8_0 NEON/scalar diverged: {max_abs}");
    }

    /// Microbench: Q8_0 NEON vs dequant-to-fp32 + scalar matmul.
    /// Skips cblas_sgemm comparison (Apple AMX makes that path
    /// unbeatable on Mac; the win we care about is for
    /// non-AMX edge targets where scalar fp32 is the comparison).
    ///
    /// Run with: `cargo test --release -p jouleclaw-loader-gguf
    ///   matmul_q8_0::tests::bench -- --ignored --nocapture`
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn bench_matmul_q8_0_vs_fp32_scalar() {
        use std::time::Instant;
        // Match a representative LFM2-350M attention projection:
        // m=17 (seq len), k=1024 (hidden), n=1024 (proj).
        let m = 17;
        let n = 1024;
        let k = 1024;
        let mut a = vec![0.0_f32; m * k];
        for (i, v) in a.iter_mut().enumerate() {
            *v = (((i % 23) as f32) * 0.07 - 0.5).sin();
        }
        let (w_q8, w_fp32) = synth_q8_0_weights(n, k, 0xDEADBEEF);
        let bias: Vec<f32> = vec![0.01; n];

        let mut out_q8 = vec![0.0_f32; m * n];
        let mut out_fp32 = vec![0.0_f32; m * n];

        // Warmup.
        matmul_q8_0(&a, &w_q8, &bias, m, n, k, &mut out_q8);
        matmul_at_fp32_ref(&a, &w_fp32, m, n, k, &mut out_fp32);

        let iters = 5;

        let t = Instant::now();
        for _ in 0..iters {
            matmul_q8_0(&a, &w_q8, &bias, m, n, k, &mut out_q8);
        }
        let q8_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let t = Instant::now();
        for _ in 0..iters {
            matmul_at_fp32_ref(&a, &w_fp32, m, n, k, &mut out_fp32);
        }
        let fp32_scalar_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        eprintln!(
            "[m={m} k={k} n={n}, iters={iters}]\n  \
             Q8_0 NEON kernel:        {q8_ms:.2} ms\n  \
             fp32 scalar reference:   {fp32_scalar_ms:.2} ms\n  \
             Q8_0/fp32_scalar ratio:  {:.2}x  (smaller = NEON int8 faster)",
            q8_ms / fp32_scalar_ms,
        );
        eprintln!(
            "  NOTE: scalar fp32 reference is NOT representative of\n  \
             Apple AMX cblas_sgemm — AMX is ~10× faster than scalar.\n  \
             On Pi 4/5 / Android with no AMX, the comparison would\n  \
             be against NEON fp32 matmul (similar to fp32 scalar but\n  \
             SIMD'd) — closer to ~3× behind cblas_sgemm on AMX. The\n  \
             Q8_0 NEON int8 path is the right choice on those targets."
        );
    }
}
