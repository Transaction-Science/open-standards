//! Metal shader library.
//!
//! Pre-defined compute kernels optimized for Apple Silicon.

/// Core shader source code.
pub mod sources {
    /// Matrix multiplication kernel (tiled, simdgroup).
    pub const MATMUL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Tiled matrix multiplication using simdgroup_matrix
// Optimized for Apple Silicon's matrix units

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

kernel void matmul_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Tile position
    uint tile_m = gid.y * TILE_M;
    uint tile_n = gid.x * TILE_N;

    // Use simdgroup matrix operations (8x8 tiles)
    simdgroup_half8x8 acc = simdgroup_half8x8(0);

    // Iterate over K dimension
    for (uint k = 0; k < K; k += TILE_K) {
        // Load tiles into simdgroup matrices
        simdgroup_half8x8 a_tile;
        simdgroup_half8x8 b_tile;

        // Load from A
        simdgroup_load(a_tile, A + (tile_m * K + k), K);
        // Load from B
        simdgroup_load(b_tile, B + (k * N + tile_n), N);

        // Multiply and accumulate
        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
    }

    // Store result
    simdgroup_store(acc, C + (tile_m * N + tile_n), N);
}

kernel void matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    uint row = gid.y * TILE_M + lid.y;
    uint col = gid.x * TILE_N + lid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Naive matmul for correctness - works with any size
// C[M,N] = A[M,K] @ B^T[N,K] (B is stored row-major as [N,K])
kernel void matmul_naive_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],  // B is [N, K] - we compute A @ B^T
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // Compute dot product: A[row,:] @ B[col,:]
    // A is [M, K], B is [N, K] (transposed), C is [M, N]
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += float(A[row * K + k]) * float(B[col * K + k]);
    }
    C[row * N + col] = half(sum);
}

// Standard matmul with B stored column-major (for GGUF weights)
// C[M,N] = A[M,K] @ B[K,N] where B is stored column-major as [K,N]
// GGUF stores weights as [hidden_size, vocab_size] and we need to multiply
// hidden[1, K] @ weight[K, N] = logits[1, N]
// B is stored column-major: column j is contiguous starting at j*K
kernel void matmul_b_colmajor_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],  // B is [K, N] column-major (GGUF layout)
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // Compute dot product: A[row,:] @ B[:,col]
    // A is [M, K] row-major: A[row,k] at row*K + k
    // B is [K, N] column-major: B[k,col] at col*K + k (column col starts at col*K)
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += float(A[row * K + k]) * float(B[col * K + k]);
    }
    C[row * N + col] = half(sum);
}

// Optimized tiled matmul with threadgroup memory
// C[M,N] = A[M,K] @ B^T[N,K]
// Uses 8x8 tiles with 4 elements per thread for better memory coalescing
kernel void matmul_tiled_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],  // B is [N, K] transposed
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgsize [[threads_per_threadgroup]],
    threadgroup half* shmem [[threadgroup(0)]]
) {
    // Tile size: 32x32 with 16x16 threads, each thread computes 2x2 output
    constexpr uint TILE = 32;
    constexpr uint THREAD_TILE = 2;
    
    uint row = gid.y * TILE + lid.y * THREAD_TILE;
    uint col = gid.x * TILE + lid.x * THREAD_TILE;
    
    // Accumulator (2x2 output per thread)
    float4 acc = float4(0.0f);
    
    // Shared memory for tiles: A_tile [TILE, TILE], B_tile [TILE, TILE]
    threadgroup half* A_tile = shmem;
    threadgroup half* B_tile = shmem + TILE * TILE;
    
    // Loop over K dimension in tiles
    for (uint k0 = 0; k0 < K; k0 += TILE) {
        // Cooperative loading of A tile (each thread loads 2 elements)
        for (uint i = 0; i < THREAD_TILE; i++) {
            uint a_row = lid.y * THREAD_TILE + i;
            for (uint j = 0; j < THREAD_TILE; j++) {
                uint a_col = lid.x * THREAD_TILE + j;
                uint global_row = gid.y * TILE + a_row;
                uint global_col = k0 + a_col;
                if (global_row < M && global_col < K) {
                    A_tile[a_row * TILE + a_col] = A[global_row * K + global_col];
                } else {
                    A_tile[a_row * TILE + a_col] = half(0.0f);
                }
            }
        }
        
        // Cooperative loading of B tile
        for (uint i = 0; i < THREAD_TILE; i++) {
            uint b_row = lid.y * THREAD_TILE + i;
            for (uint j = 0; j < THREAD_TILE; j++) {
                uint b_col = lid.x * THREAD_TILE + j;
                uint global_row = gid.x * TILE + b_row;  // B is transposed
                uint global_col = k0 + b_col;
                if (global_row < N && global_col < K) {
                    B_tile[b_row * TILE + b_col] = B[global_row * K + global_col];
                } else {
                    B_tile[b_row * TILE + b_col] = half(0.0f);
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot products
        for (uint k = 0; k < TILE; k++) {
            half a0 = A_tile[(lid.y * THREAD_TILE + 0) * TILE + k];
            half a1 = A_tile[(lid.y * THREAD_TILE + 1) * TILE + k];
            half b0 = B_tile[(lid.x * THREAD_TILE + 0) * TILE + k];
            half b1 = B_tile[(lid.x * THREAD_TILE + 1) * TILE + k];
            
            // acc[i,j] += a[i] * b[j]
            acc.x += float(a0) * float(b0);  // [0,0]
            acc.y += float(a0) * float(b1);  // [0,1]
            acc.z += float(a1) * float(b0);  // [1,0]
            acc.w += float(a1) * float(b1);  // [1,1]
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write results
    if (row < M && col < N) {
        C[row * N + col] = half(acc.x);
    }
    if (row < M && col + 1 < N) {
        C[row * N + col + 1] = half(acc.y);
    }
    if (row + 1 < M && col < N) {
        C[(row + 1) * N + col] = half(acc.z);
    }
    if (row + 1 < M && col + 1 < N) {
        C[(row + 1) * N + col + 1] = half(acc.w);
    }
}
"#;

    /// Q4_K quantized matrix-vector multiplication.
    /// Keeps weights in Q4_K format and dequantizes on-the-fly.
    /// B matrix (weights) is Q4_K format [K, N] col-major (K elements per row, N rows).
    /// A matrix (activations) is F16 [1, K].
    /// Output is F16 [1, N].
    pub const MATMUL_Q4K: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Q4_K block structure: 256 values per block
// d (2 bytes f16) + dmin (2 bytes f16) + scales (12 bytes) + qs (128 bytes) = 144 bytes
#define QK_K 256
#define Q4K_BLOCK_SIZE 144

// Extract scale and min for Q4_K
inline float2 get_scale_min_k4(uint j, device const uint8_t* scales) {
    float sc, m;
    if (j < 4) {
        sc = float(scales[j] & 63);
        m = float(scales[j + 4] & 63);
    } else {
        sc = float((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
        m = float((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
    }
    return float2(sc, m);
}

// Matrix-vector multiply: y = x @ W^T where W is Q4_K quantized
// x: [1, K] F16 input vector
// W: [N, K] Q4_K weights (each row is K elements = K/256 blocks)
// y: [1, N] F16 output
kernel void matmul_q4k_f16(
    device const half* x [[buffer(0)]],           // Input [1, K]
    device const uint8_t* W [[buffer(1)]],        // Q4_K weights [N, K]
    device half* y [[buffer(2)]],                 // Output [1, N]
    constant uint& N [[buffer(3)]],               // Output dimension
    constant uint& K [[buffer(4)]],               // Input dimension
    uint row [[thread_position_in_grid]]          // Which output element
) {
    if (row >= N) return;
    
    uint num_blocks = K / QK_K;
    float sum = 0.0f;
    
    // Each row of W is num_blocks Q4_K blocks
    device const uint8_t* row_ptr = W + row * num_blocks * Q4K_BLOCK_SIZE;
    
    for (uint b = 0; b < num_blocks; b++) {
        device const uint8_t* block = row_ptr + b * Q4K_BLOCK_SIZE;
        
        // Read d and dmin (f16)
        device const half* scales_f16 = (device const half*)block;
        float d = float(scales_f16[0]);
        float dmin = float(scales_f16[1]);
        
        device const uint8_t* scales = block + 4;
        device const uint8_t* qs = block + 16;
        
        uint x_offset = b * QK_K;
        uint q_offset = 0;
        uint is = 0;
        
        // Process 4 groups of 64 values
        for (uint g = 0; g < 4; g++) {
            float2 sm1 = get_scale_min_k4(is, scales);
            float d1 = d * sm1.x;
            float min1 = dmin * sm1.y;
            
            float2 sm2 = get_scale_min_k4(is + 1, scales);
            float d2 = d * sm2.x;
            float min2 = dmin * sm2.y;
            
            // First 32: low nibbles
            for (uint l = 0; l < 32; l++) {
                float q_val = float(qs[q_offset + l] & 0x0F);
                float w = d1 * q_val - min1;
                sum += float(x[x_offset + l]) * w;
            }
            
            // Next 32: high nibbles
            for (uint l = 0; l < 32; l++) {
                float q_val = float((qs[q_offset + l] >> 4) & 0x0F);
                float w = d2 * q_val - min2;
                sum += float(x[x_offset + 32 + l]) * w;
            }
            
            q_offset += 32;
            x_offset += 64;
            is += 2;
        }
    }
    
    y[row] = half(sum);
}
"#;

    /// LayerNorm kernel (for Phi and other models with bias).
    pub const LAYER_NORM: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void layer_norm_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;

    device const half* x = input + gid * D;
    device half* out = output + gid * D;

    // Compute mean
    float sum = 0.0f;
    for (uint i = 0; i < D; i++) {
        sum += float(x[i]);
    }
    float mean = sum / float(D);

    // Compute variance
    float var_sum = 0.0f;
    for (uint i = 0; i < D; i++) {
        float diff = float(x[i]) - mean;
        var_sum += diff * diff;
    }
    float inv_std = rsqrt(var_sum / float(D) + eps);

    // Normalize, scale and shift
    for (uint i = 0; i < D; i++) {
        float normalized = (float(x[i]) - mean) * inv_std;
        out[i] = half(normalized * float(weight[i]) + float(bias[i]));
    }
}
"#;

    /// Softmax kernel.
    pub const SOFTMAX: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Numerically stable softmax
kernel void softmax_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    if (gid >= N) return;

    device const half* row = input + gid * D;
    device half* out_row = output + gid * D;

    // Find max (for numerical stability)
    float max_val = -INFINITY;
    for (uint i = 0; i < D; i++) {
        max_val = max(max_val, float(row[i]));
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < D; i++) {
        sum += exp(float(row[i]) - max_val);
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < D; i++) {
        out_row[i] = half(exp(float(row[i]) - max_val) * inv_sum);
    }
}
"#;

    /// RMS normalization kernel.
    pub const RMS_NORM: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;

    device const half* x = input + gid * D;
    device half* out = output + gid * D;

    // Compute RMS
    float sum_sq = 0.0f;
    for (uint i = 0; i < D; i++) {
        float val = float(x[i]);
        sum_sq += val * val;
    }
    float rms = rsqrt(sum_sq / float(D) + eps);

    // Normalize and scale
    for (uint i = 0; i < D; i++) {
        out[i] = half(float(x[i]) * rms * float(weight[i]));
    }
}

// RMS norm for F32 residual stream: reads F32 input, F16 weight, writes F16 output.
// Used by T5 where the residual stream must stay in F32 (values exceed F16 range).
kernel void rms_norm_f32_to_f16(
    device const float* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;

    device const float* x = input + gid * D;
    device half* out = output + gid * D;

    float sum_sq = 0.0f;
    for (uint i = 0; i < D; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = rsqrt(sum_sq / float(D) + eps);

    for (uint i = 0; i < D; i++) {
        out[i] = half(x[i] * rms * float(weight[i]));
    }
}

// Residual add: F32 residual + F16 delta → F32 result.
// Used by T5 encoder/decoder to accumulate in F32 (prevents overflow).
// Clamps F16 inf/nan to ±65000 before adding (F16 overflow from FFN projections).
kernel void residual_add_f16_to_f32(
    device const float* residual [[buffer(0)]],
    device const half* delta [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float d = float(delta[gid]);
    if (isinf(d) || isnan(d)) d = copysign(65000.0f, d);
    output[gid] = residual[gid] + d;
}

// Residual add: F32 + F32 → F32 (for F32 output projections).
kernel void residual_add_f32_to_f32(
    device const float* residual [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = residual[gid] + delta[gid];
}

// Convert F16 tensor to F32.
kernel void f16_to_f32(
    device const half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = float(input[gid]);
}

// Gemma-specific RMS norm: uses (1 + weight) instead of just weight
// Gemma initializes RMSNorm weights to 0, making effective weight 1.0 at init
kernel void rms_norm_gemma_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;

    device const half* x = input + gid * D;
    device half* out = output + gid * D;

    // Compute RMS
    float sum_sq = 0.0f;
    for (uint i = 0; i < D; i++) {
        float val = float(x[i]);
        sum_sq += val * val;
    }
    float rms = rsqrt(sum_sq / float(D) + eps);

    // Normalize and scale with (1 + weight)
    for (uint i = 0; i < D; i++) {
        out[i] = half(float(x[i]) * rms * (1.0f + float(weight[i])));
    }
}
"#;

    /// SiLU (Swish) activation.
    pub const SILU: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void silu_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = float(input[gid]);
    output[gid] = half(x / (1.0f + exp(-x)));
}

kernel void silu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = input[gid];
    output[gid] = x / (1.0f + exp(-x));
}
"#;

    /// GeLU activation (used by Gemma, BERT, GPT-2, etc.).
    /// Includes both exact and tanh approximation variants.
    pub const GELU: &str = r#"
#include <metal_stdlib>
using namespace metal;

// GeLU with tanh approximation (faster, used by PyTorch as "gelu_pytorch_tanh")
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void gelu_tanh_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = float(input[gid]);
    const float sqrt_2_over_pi = 0.7978845608028654f;  // sqrt(2/pi)
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    // Clamp to prevent tanh overflow: exp(2*44.4) > FLT_MAX → NaN
    inner = clamp(inner, -10.0f, 10.0f);
    output[gid] = half(0.5f * x * (1.0f + tanh(inner)));
}

// Exact GELU (erf-based, matches PyTorch nn.GELU() default).
// gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// Uses Abramowitz & Stegun 7.1.26 erf approximation (max error ~1.5e-7).
kernel void gelu_exact_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = float(input[gid]);
    float a = x / 1.4142135623730951f;
    float s = a >= 0.0f ? 1.0f : -1.0f;
    float aa = abs(a);
    float t = 1.0f / (1.0f + 0.3275911f * aa);
    float t2 = t * t; float t3 = t2 * t; float t4 = t3 * t; float t5 = t4 * t;
    float y = 1.0f - (0.254829592f * t - 0.284496736f * t2 + 1.421413741f * t3
                      - 1.453152027f * t4 + 1.061405429f * t5) * exp(-aa * aa);
    float erf = s * y;
    output[gid] = half(0.5f * x * (1.0f + erf));
}

// Fast GeLU approximation using sigmoid
// gelu(x) ≈ x * sigmoid(1.702 * x)
kernel void gelu_fast_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = float(input[gid]);
    const float coeff = 1.702f;
    output[gid] = half(x / (1.0f + exp(-coeff * x)));
}

// GeLU with tanh approximation for f32
kernel void gelu_tanh_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = input[gid];
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    // Clamp to prevent tanh overflow: exp(2*44.4) > FLT_MAX → NaN
    inner = clamp(inner, -10.0f, 10.0f);
    output[gid] = 0.5f * x * (1.0f + tanh(inner));
}

// erf approximation (Abramowitz & Stegun, max error ~1.5e-7)
inline float erf_approx(float x) {
    float s = x >= 0.0f ? 1.0f : -1.0f;
    float a = abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float t2 = t * t; float t3 = t2 * t; float t4 = t3 * t; float t5 = t4 * t;
    float y = 1.0f - (0.254829592f * t - 0.284496736f * t2 + 1.421413741f * t3
                      - 1.453152027f * t4 + 1.061405429f * t5) * exp(-a * a);
    return s * y;
}

// GeGLU: GeLU-gated linear unit (used in Gemma MLP)
// geglu(gate, up) = gelu(gate) * up
// Uses erf-based GELU matching PyTorch nn.GELU() (not tanh approximation).
kernel void geglu_f16(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float g = float(gate[gid]);
    float u = float(up[gid]);

    // GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    float gelu_g = 0.5f * g * (1.0f + erf_approx(g * 0.7071067811865476f));

    // Clamp to F16 range to prevent inf → NaN in subsequent linear ops
    output[gid] = half(clamp(gelu_g * u, -65000.0f, 65000.0f));
}

// GeGLU with F32 output (for deep models like Flan-T5 that need F32 FFN path).
kernel void geglu_f16_to_f32(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float g = float(gate[gid]);
    float u = float(up[gid]);
    float gelu_g = 0.5f * g * (1.0f + erf_approx(g * 0.7071067811865476f));
    output[gid] = gelu_g * u;
}

// ReLU with F32 output (for deep models).
kernel void relu_f16_to_f32(
    device const half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = float(input[gid]);
    output[gid] = x >= 0.0f ? x : 0.0f;
}

// Linear Y = X @ W^T where X is F32, W is F16, Y is F32.
// For FFN WO projection after F32 GEGLU output.
kernel void linear_f32_in_f16_wt_f32_out(
    device const float* X [[buffer(0)]],
    device const half* W [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& has_bias [[buffer(7)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    const uint TILE = 16;
    uint row = gid.y * TILE + tid.y;
    uint col = gid.x * TILE + tid.x;

    threadgroup float tileA[TILE][TILE];
    threadgroup float tileW[TILE][TILE];
    float acc = 0.0f;

    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
        uint k_off = t * TILE;
        tileA[tid.y][tid.x] = (row < M && (k_off + tid.x) < K)
            ? X[row * K + k_off + tid.x] : 0.0f;
        uint w_row = gid.x * TILE + tid.y;
        tileW[tid.y][tid.x] = (w_row < N && (k_off + tid.x) < K)
            ? float(W[w_row * K + k_off + tid.x]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < TILE; i++) {
            acc += tileA[tid.y][i] * tileW[tid.x][i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        if (has_bias) acc += float(bias[col]);
        Y[row * N + col] = acc;
    }
}

// ReLU activation: relu(x) = max(x, 0)
kernel void relu_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = max(input[gid], half(0.0h));
}

// ELU activation: elu(x) = x if x >= 0, else exp(x) - 1
kernel void elu_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = float(input[gid]);
    output[gid] = half(x >= 0.0f ? x : exp(x) - 1.0f);
}
"#;

    /// Rotary positional embedding.
    pub const ROPE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rope_f16(
    device half* x [[buffer(0)]],
    device const float* cos_cache [[buffer(1)]],
    device const float* sin_cache [[buffer(2)]],
    constant uint& offset [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint head = gid.y;
    uint pos = gid.z;

    if (i >= head_dim / 2 || head >= num_heads) return;

    // Use offset to get absolute position for cache lookup
    uint global_pos = pos + offset;

    // Index in the input buffer (relative to current batch/chunk)
    // Layout: [pos, head, dim]
    uint stride_pos = num_heads * head_dim;
    uint stride_head = head_dim;
    
    // Base index for this token+head
    uint base_idx = pos * stride_pos + head * stride_head;

    // Indices for the pair (split frequency: x, x + dim/2)
    uint idx = base_idx + i;
    uint idx2 = base_idx + i + head_dim / 2;

    // Index in the RoPE cache (absolute position)
    // Cache is [max_seq, dim/2] shared across heads
    uint cache_idx = global_pos * (head_dim / 2) + i;

    float cos_val = cos_cache[cache_idx];
    float sin_val = sin_cache[cache_idx];

    float x0 = float(x[idx]);
    float x1 = float(x[idx2]);

    x[idx] = half(x0 * cos_val - x1 * sin_val);
    x[idx2] = half(x1 * cos_val + x0 * sin_val);
}

// RoPE for single position with inline frequency computation
// Applied to Q [num_heads, head_dim] and K [num_kv_heads, head_dim]
// Uses interleaved format: pairs are at (i*2, i*2+1)
kernel void rope_single_f16(
    device half* x [[buffer(0)]],           // Q or K buffer
    constant uint& pos [[buffer(1)]],        // Position in sequence
    constant uint& num_heads [[buffer(2)]],  // Number of heads
    constant uint& head_dim [[buffer(3)]],   // Head dimension
    constant float& theta [[buffer(4)]],     // RoPE theta (typically 10000.0)
    uint2 gid [[thread_position_in_grid]]    // (pair_idx, head_idx)
) {
    uint head = gid.y;
    uint pair = gid.x;  // Which pair of values (0 to head_dim/2 - 1)

    if (head >= num_heads || pair >= head_dim / 2) return;

    // Compute rotation frequencies
    float freq = 1.0f / pow(theta, float(2 * pair) / float(head_dim));
    float angle = float(pos) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    // Indices in the buffer - interleaved format (pairs at i*2, i*2+1)
    uint base = head * head_dim;
    uint i1 = base + pair * 2;
    uint i2 = base + pair * 2 + 1;

    // Load values
    float x0 = float(x[i1]);
    float x1 = float(x[i2]);

    // Apply rotation
    x[i1] = half(x0 * cos_val - x1 * sin_val);
    x[i2] = half(x0 * sin_val + x1 * cos_val);
}

// Copy K/V to cache at specific position
kernel void copy_to_kv_cache_f16(
    device const half* kv [[buffer(0)]],        // Source K or V [num_heads, head_dim]
    device half* cache [[buffer(1)]],           // Cache [max_seq, num_heads, head_dim]
    constant uint& pos [[buffer(2)]],           // Position to write
    constant uint& num_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]       // (dim, head)
) {
    uint head = gid.y;
    uint dim = gid.x;

    if (head >= num_heads || dim >= head_dim) return;

    uint src_idx = head * head_dim + dim;
    uint dst_idx = (pos * num_heads + head) * head_dim + dim;

    cache[dst_idx] = kv[src_idx];
}
"#;

    /// Elementwise operations.
    pub const ELEMENTWISE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    c[gid] = a[gid] + b[gid];
}

kernel void sub_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    c[gid] = a[gid] - b[gid];
}

kernel void mul_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    c[gid] = a[gid] * b[gid];
}

kernel void div_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    c[gid] = a[gid] / b[gid];
}

kernel void scale_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = half(float(input[gid]) * scale);
}

// Classifier-Free Guidance fused kernel. Reads the batched UNet output
// [2, ...] (uncond first, cond second) and writes the single-batch
// post-CFG result: out = uncond + scale * (cond - uncond).
// Done as a single offset-aware kernel because Tensor::slice updates the
// logical byte_offset but device_ptr() returns the base buffer pointer
// (offset is ignored by elementwise kernels) — slicing then calling
// sub/scale/add would feed both halves with batch[0] and silently produce
// `result = uncond` (no CFG). 2026-05-17 fix.
kernel void cfg_apply_f16(
    device const half* batched [[buffer(0)]],  // [2 * half_count] f16
    device half* out [[buffer(1)]],            // [half_count] f16
    constant float& scale [[buffer(2)]],
    constant uint& half_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= half_count) return;
    float u = float(batched[gid]);
    float c = float(batched[gid + half_count]);
    out[gid] = half(u + scale * (c - u));
}

// In-place add bias: x += bias
kernel void add_bias_f16(
    device half* x [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    x[gid] += bias[gid];
}

// Channel-wise broadcast add: output[n,c,h,w] = input[n,c,h,w] + bias[c]
// For adding a per-channel vector [C] to a spatial tensor [N,C,H,W].
kernel void channel_bias_add_f16(
    device const half* input [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& spatial [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // gid is over `batch * channels * spatial` (caller dispatches the full
    // numel). Channel index wraps within each batch's contiguous slice.
    // Previously the kernel early-out'd at `channels*spatial`, leaving
    // batches ≥ 1's output region uninitialised.
    uint per_batch = channels * spatial;
    uint c = (gid % per_batch) / spatial;
    output[gid] = input[gid] + bias[c];
}
"#;

    /// Upsample kernels.
    pub const UPSAMPLE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void upsample_nearest_f16(
    device const half* input [[buffer(0)]],      // [N, C, H, W]
    device half* output [[buffer(1)]],           // [N, C, H*2, W*2]
    constant uint& N [[buffer(2)]],
    constant uint& C [[buffer(3)]],
    constant uint& Hin [[buffer(4)]],
    constant uint& Win [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]        // (Wout, Hout, C * N)
) {
    uint ow = gid.x;
    uint oh = gid.y;
    uint nc = gid.z; // flattens N*C

    if (ow >= Win * 2 || oh >= Hin * 2) return;
    
    // Nearest neighbor: input coordinate is floor(out / 2)
    uint iw = ow / 2;
    uint ih = oh / 2;
    
    uint in_idx = nc * Hin * Win + ih * Win + iw;
    uint out_idx = nc * (Hin * 2) * (Win * 2) + oh * (Win * 2) + ow;
    
    output[out_idx] = input[in_idx];
}
"#;

    /// Fused VAE decode kernel.
    /// Performs 8x nearest-neighbor upsampling and scaling in one pass.
    /// Input: [N, 4, H, W]
    /// Output: [N, 3, H*8, W*8]
    pub const VAE_FUSED: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vae_decode_fused_f16(
    device const half* input [[buffer(0)]],      // [N, 4, H, W]
    device half* output [[buffer(1)]],           // [N, 3, H*8, W*8]
    constant uint& N [[buffer(2)]],
    constant uint& Hin [[buffer(3)]],
    constant uint& Win [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]        // (Wout, Hout, Batch)
) {
    uint ow = gid.x;
    uint oh = gid.y;
    uint n = gid.z;

    uint Hout = Hin * 8;
    uint Wout = Win * 8;

    if (ow >= Wout || oh >= Hout || n >= N) return;
    
    // Nearest neighbor: input coordinate is floor(out / 8)
    uint iw = ow / 8;
    uint ih = oh / 8;
    
    // Stride calculations
    // Input is [N, 4, H, W]
    uint in_stride_n = 4 * Hin * Win;
    uint in_stride_c = Hin * Win;
    
    // Output is [N, 3, H*8, W*8]
    uint out_stride_n = 3 * Hout * Wout;
    uint out_stride_c = Hout * Wout;

    // Load 4 channels from input
    // The current pipeline uses channel 1, 2, 3 as RGB (slicing 1,0,3)
    // Input channels: 0, 1, 2, 3. We want 1, 2, 3? 
    // Previous code: slice(1, 0, 3) means start at index 1 (dim 0? no dim 1 is channels), length 3. So channels 1, 2, 3.
    
    uint base_in = n * in_stride_n + ih * Win + iw;
    
    // Channel 1 -> R
    half r = input[base_in + 1 * in_stride_c];
    // Channel 2 -> G
    half g = input[base_in + 2 * in_stride_c];
    // Channel 3 -> B
    half b = input[base_in + 3 * in_stride_c];
    
    // Scale
    half s = half(scale);
    r = r * s;
    g = g * s;
    b = b * s;
    
    // Store
    uint base_out = n * out_stride_n + oh * Wout + ow;
    output[base_out + 0 * out_stride_c] = r;
    output[base_out + 1 * out_stride_c] = g;
    output[base_out + 2 * out_stride_c] = b;
}
"#;



    /// VAE output rescale: clamp(x * 0.5 + 0.5, 0, 1).
    /// Converts from [-1,1] range to [0,1] range for image output.
    pub const VAE_RESCALE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vae_rescale_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    half v = input[id];
    output[id] = clamp(v * half(0.5) + half(0.5), half(0.0), half(1.0));
}
"#;

    /// Scheduler step kernels for diffusion denoising.
    /// Operate element-wise on f16 tensors with f32 scalar coefficients.
    pub const SCHEDULER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// output[i] = a[i] * scale_a + b[i] * scale_b
kernel void scale_add_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant float& scale_a [[buffer(3)]],
    constant float& scale_b [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = half(float(a[gid]) * scale_a + float(b[gid]) * scale_b);
}

// output[i] = clamp(a[i] * scale_a + b[i] * scale_b, lo, hi)
kernel void scale_add_clamp_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant float& scale_a [[buffer(3)]],
    constant float& scale_b [[buffer(4)]],
    constant float& lo [[buffer(5)]],
    constant float& hi [[buffer(6)]],
    constant uint& count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float val = float(a[gid]) * scale_a + float(b[gid]) * scale_b;
    output[gid] = half(clamp(val, lo, hi));
}

// output[i] = a[i] * scale_a + b[i] * scale_b + c[i] * scale_c
// For 3-tensor blend (e.g., DPM++ corrected step, ancestral noise injection)
kernel void scale_add3_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device const half* c [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant float& scale_a [[buffer(4)]],
    constant float& scale_b [[buffer(5)]],
    constant float& scale_c [[buffer(6)]],
    constant uint& count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = half(float(a[gid]) * scale_a + float(b[gid]) * scale_b + float(c[gid]) * scale_c);
}
"#;

    /// Fused VAE encode kernel.
    /// Performs 8x downsampling (average pooling) and scaling.
    /// Input: [N, 3, H, W]
    /// Output: [N, 4, H/8, W/8]
    pub const VAE_ENCODE_FUSED: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vae_encode_fused_f16(
    device const half* input [[buffer(0)]],      // [N, 3, H, W]
    device half* output [[buffer(1)]],           // [N, 4, H/8, W/8]
    constant uint& N [[buffer(2)]],
    constant uint& Hin [[buffer(3)]],            // Input Image H
    constant uint& Win [[buffer(4)]],            // Input Image W
    constant float& scale [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]        // (W_lat, H_lat, Batch)
) {
    uint ow = gid.x; // Latent W
    uint oh = gid.y; // Latent H
    uint n = gid.z;

    uint H_lat = Hin / 8;
    uint W_lat = Win / 8;

    if (ow >= W_lat || oh >= H_lat || n >= N) return;

    // Strides
    uint in_stride_n = 3 * Hin * Win;
    uint in_stride_c = Hin * Win;
    
    uint out_stride_n = 4 * H_lat * W_lat;
    uint out_stride_c = H_lat * W_lat;

    // Simple Downsampling: Average Pooling over 8x8 block
    float r_sum = 0.0f;
    float g_sum = 0.0f;
    float b_sum = 0.0f;
    
    uint base_y = oh * 8;
    uint base_x = ow * 8;
    
    // Safety check for image boundary
    uint h_lim = min(base_y + 8, Hin);
    uint w_lim = min(base_x + 8, Win);
    float count = 0.0f;

    uint n_offset = n * in_stride_n;
    
    for (uint y = base_y; y < h_lim; y++) {
        for (uint x = base_x; x < w_lim; x++) {
            uint idx = n_offset + y * Win + x;
            r_sum += float(input[idx + 0 * in_stride_c]);
            g_sum += float(input[idx + 1 * in_stride_c]);
            b_sum += float(input[idx + 2 * in_stride_c]);
            count += 1.0f;
        }
    }
    
    float inv_count = 1.0f / count;
    half r = half(r_sum * inv_count);
    half g = half(g_sum * inv_count);
    half b = half(b_sum * inv_count);
    
    // Scale (usually multiply by 0.18215)
    half s = half(scale);
    r = r * s;
    g = g * s;
    b = b * s;
    
    // Store into Latents [N, 4, H_lat, W_lat]
    // Mapping: 0->0(unused), 1->R, 2->G, 3->B (Inverse of decode)
    uint out_base = n * out_stride_n + oh * W_lat + ow;
    
    output[out_base + 0 * out_stride_c] = 0.0h; // Unused channel?
    output[out_base + 1 * out_stride_c] = r;
    output[out_base + 2 * out_stride_c] = g;
    output[out_base + 3 * out_stride_c] = b;
}
"#;

    /// Attention kernel.
    pub const ATTENTION: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Flash attention style - fused QKV attention
// Computes: softmax(Q @ K^T / sqrt(d)) @ V
// Layout: [Batch, Heads, Seq, Dim]
// Dispatched as (Heads, Seq, Batch) threadgroups

kernel void attention_f16(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& stride_batch [[buffer(8)]],
    constant uint& stride_head [[buffer(9)]],
    constant uint& stride_seq [[buffer(10)]],
    constant uint& stride_dim [[buffer(11)]],
    constant uint& kv_len [[buffer(12)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    uint query_pos = gid.y;
    uint head = gid.x;
    uint batch_idx = gid.z;

    if (query_pos >= seq_len) return;

    // Offsets using explicit strides
    uint batch_offset = batch_idx * stride_batch;
    uint head_offset = head * stride_head;
    uint seq_offset_q = query_pos * stride_seq;

    // Base pointer for this Batch+Head
    uint base_offset = batch_offset + head_offset;

    // Q is specific to this query_pos
    device const half* q = Q + batch_offset + head_offset + seq_offset_q;

    // Compute attention scores
    threadgroup float* scores = (threadgroup float*)shared;

    float max_score = -INFINITY;

    // Iterate over K
    for (uint k_pos = 0; k_pos < kv_len; k_pos++) {
        uint seq_offset_k = k_pos * stride_seq;
        device const half* k = K + batch_offset + head_offset + seq_offset_k;

        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(q[d * stride_dim]) * float(k[d * stride_dim]);
        }
        dot *= scale;

        scores[k_pos] = dot;
        max_score = max(max_score, dot);
    }

    // ... softmax ...

    float sum = 0.0f;
    for (uint k_pos = 0; k_pos < kv_len; k_pos++) {
        scores[k_pos] = exp(scores[k_pos] - max_score);
        sum += scores[k_pos];
    }

    float inv_sum = 1.0f / sum;
    for (uint k_pos = 0; k_pos < kv_len; k_pos++) {
        scores[k_pos] *= inv_sum;
    }

    // Compute output
    device half* o = O + batch_offset + head_offset + seq_offset_q;
    for (uint d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (uint k_pos = 0; k_pos < kv_len; k_pos++) {
            uint seq_offset_v = k_pos * stride_seq;
            device const half* v = V + batch_offset + head_offset + seq_offset_v;
            acc += scores[k_pos] * float(v[d * stride_dim]);
        }
        o[d * stride_dim] = half(acc);
    }
}
"#;

    /// Tiled Attention kernel (Flash Attention style).
    /// Uses threadgroup memory for Q/K/V blocking and online softmax.
    /// Robust for long sequences and efficient.
    pub const ATTENTION_TILED: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void attention_tiled_f16(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& stride_batch [[buffer(8)]],
    constant uint& stride_head [[buffer(9)]],
    constant uint& stride_seq [[buffer(10)]],
    constant uint& stride_dim [[buffer(11)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    const uint BLOCK = 32; // Queries/Keys per tile
    
    // Grid: (Heads, (Seq+BLOCK-1)/BLOCK, Batch)
    uint batch = gid.z;
    uint head = gid.x;
    uint q_block_idx = gid.y;
    uint tid = lid.x;
    
    // Offsets
    uint base_offset = batch * stride_batch + head * stride_head;
    
    device const half* q_ptr = Q + base_offset;
    device const half* k_ptr = K + base_offset;
    device const half* v_ptr = V + base_offset;
    device half* o_ptr = O + base_offset;
    
    // Shared Memory Setup
    // Layout: Q[BLOCK][head_dim] | K[BLOCK][head_dim] | V[BLOCK][head_dim]
    threadgroup half* q_tile = shared;
    threadgroup half* k_tile = shared + BLOCK * head_dim;
    threadgroup half* v_tile = k_tile + BLOCK * head_dim;
    
    // Global Query Index
    uint q_idx = q_block_idx * BLOCK + tid;
    
    // Load Q tile: Each thread loads its entire row (head_dim elements)
    // Note: If head_dim is large, this loop is significant.
    // For D=64/128, it's fine.
    if (q_idx < seq_len) {
        for (uint i = 0; i < head_dim; i++) {
             q_tile[tid * head_dim + i] = q_ptr[q_idx * stride_seq + i * stride_dim];
        }
    } else {
        for (uint i = 0; i < head_dim; i++) q_tile[tid * head_dim + i] = 0.0h;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Accumulators for online softmax
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    // SD 1.5 uses num_heads=8 so head_dim = channels/8 reaches 160 at the
    // 1280-channel levels (down_block_2/3, mid, up_block_0/1). The old
    // acc[128] overflowed for those (acc[128..159] = stack OOB → corrupted
    // m_prev/l_prev/loop state → garbage self-attention at every deep block).
    // 256 covers SD 1.5 (≤160) and SDXL (head_dim=64) with margin.
    float acc[256];

    for(uint i=0; i<head_dim; i++) acc[i] = 0.0f;
    
    // Loop over K/V blocks
    for (uint k_base = 0; k_base < seq_len; k_base += BLOCK) {
        // Load K and V tiles
        uint k_idx = k_base + tid;
        bool valid_k = k_idx < seq_len;
        
        if (valid_k) {
            for (uint i = 0; i < head_dim; i++) {
                k_tile[tid * head_dim + i] = k_ptr[k_idx * stride_seq + i * stride_dim];
                v_tile[tid * head_dim + i] = v_ptr[k_idx * stride_seq + i * stride_dim];
            }
        }
        // Note: No need to zero out K/V for out-of-bounds, we control loop limit below
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute Attention for this block
        if (q_idx < seq_len) {
             uint loop_lim = min(BLOCK, seq_len - k_base);
             
             for (uint k_local = 0; k_local < loop_lim; k_local++) {
                 // Dot Product Q[tid] . K[k_local]
                 float dot = 0.0f;
                 for (uint d=0; d<head_dim; d++) {
                     dot += float(q_tile[tid * head_dim + d]) * float(k_tile[k_local * head_dim + d]);
                 }
                 dot *= scale;
                 
                 // Online Softmax Update
                 float m_curr = max(m_prev, dot);
                 float d_prev = exp(m_prev - m_curr);
                 float d_curr = exp(dot - m_curr);
                 
                 l_prev = l_prev * d_prev + d_curr;
                 
                 // Update Accumulator
                 for (uint d=0; d<head_dim; d++) {
                     acc[d] = acc[d] * d_prev + float(v_tile[k_local * head_dim + d]) * d_curr;
                 }
                 
                 m_prev = m_curr;
             }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store Output
    if (q_idx < seq_len) {
        float inv_l = 1.0f / l_prev;
        for (uint d=0; d<head_dim; d++) {
            o_ptr[q_idx * stride_seq + d * stride_dim] = half(acc[d] * inv_l);
        }
    }
}
"#;

    /// MFA 2.0 Stage 1: FlashAttention-2-style self-attention using
    /// `simdgroup_matrix<float, 8, 8>` for the QK^T and PV matmuls.
    ///
    /// Replaces the per-thread serial inner-`d` loop with 8×8 simdgroup tile
    /// multiplies — one wave-wide hardware instruction per matmul tile.
    /// Online softmax runs row-by-row in threadgroup memory (threads 0..7 of
    /// the single simdgroup each handle one of the BQ=8 query rows).
    ///
    /// Self-attention only (same kernel signature as `attention_tiled_f16`).
    /// Assumes:
    ///   - seq_len % 8 == 0  (SD self-attn spatial flatten × batch=2 is always 8|N)
    ///   - head_dim % 8 == 0 (SD 1.5: 40/80/160; SDXL: 64 — all 8|D)
    ///   - head_dim <= 160  (Dt = D/8 ≤ 20 — register array bounds)
    /// Cross-attention (`seq_q != seq_k`) is NOT routed here; it stays on the
    /// untiled `attention_f16` kernel.
    pub const ATTENTION_SIMDMM: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void attention_simdmm_f16(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& stride_batch [[buffer(8)]],
    constant uint& stride_head [[buffer(9)]],
    constant uint& stride_seq [[buffer(10)]],
    constant uint& stride_dim [[buffer(11)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    threadgroup uchar* shared [[threadgroup(0)]]
) {
    const uint BQ = 8;
    const uint BK = 32;
    const uint MAX_DT = 20;   // covers head_dim up to 160

    uint batch    = gid.z;
    uint head     = gid.x;
    uint q_block  = gid.y;
    uint tid      = lid.x;

    uint base = batch * stride_batch + head * stride_head;
    device const half* qbp = Q + base;
    device const half* kbp = K + base;
    device const half* vbp = V + base;
    device       half* obp = O + base;

    uint D   = head_dim;
    uint Dt  = D / 8;
    uint ldS = stride_seq;     // distance between rows of the [seq, hidden] view

    // Threadgroup memory partitioning (byte-cursor walk so layout is explicit).
    threadgroup half*  q_tile    = (threadgroup half*)shared;                 // [BQ × D] half
    threadgroup half*  k_tile    = q_tile + BQ * D;                           // [BK × D] half
    threadgroup half*  v_tile    = k_tile + BK * D;                           // [BK × D] half
    threadgroup half*  p_tile    = v_tile + BK * D;                           // [BQ × BK] half (P post-softmax)
    threadgroup float* s_scratch = (threadgroup float*)(p_tile + BQ * BK);    // [BQ × BK] float (S from simdgroup_store)
    threadgroup float* o_shared  = s_scratch + BQ * BK;                       // [BQ × D] float (O accumulator)
    threadgroup float* m_state   = o_shared + BQ * D;                         // [BQ] float
    threadgroup float* l_state   = m_state + BQ;                              // [BQ] float
    threadgroup float* alpha_buf = l_state + BQ;                              // [BQ] float

    // ---- Init per-row stats and O accumulator ----
    if (tid < BQ) {
        m_state[tid] = -INFINITY;
        l_state[tid] = 0.0f;
    }
    for (uint i = tid; i < BQ * D; i += 32) o_shared[i] = 0.0f;

    // ---- Load Q tile [BQ × D] (once, reused across all K-tiles) ----
    // Vec4 loads: each thread copies 4 halves per instruction. head_dim is
    // always a multiple of 4 (40/64/80/160), hidden_dim is a multiple of 8,
    // so device pointers land on 8-byte boundaries.
    uint q_row_start = q_block * BQ;
    uint q_quads = (BQ * D) >> 2;
    for (uint i = tid; i < q_quads; i += 32) {
        uint base = i << 2;
        uint row = base / D;
        uint col = base % D;
        uint q_idx = q_row_start + row;
        threadgroup half4* dst = (threadgroup half4*)(q_tile + row * D + col);
        if (q_idx < seq_len) {
            const device half4* src = (const device half4*)(qbp + q_idx * ldS + col);
            *dst = *src;
        } else {
            *dst = half4(0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pre-load Q simdgroup fragments (Dt fragments, each 8x8).
    simdgroup_half8x8 q_frag[MAX_DT];
    for (uint dt = 0; dt < Dt; dt++) {
        simdgroup_load(q_frag[dt], q_tile, D, ulong2(dt * 8, 0));
    }

    uint kv_quads = (BK * D) >> 2;
    // ---- Main loop over K-tiles ----
    for (uint kbase = 0; kbase < seq_len; kbase += BK) {
        // Load K and V tiles [BK × D] with half4 vector ops.
        for (uint i = tid; i < kv_quads; i += 32) {
            uint base = i << 2;
            uint row = base / D;
            uint col = base % D;
            uint k_idx = kbase + row;
            threadgroup half4* kdst = (threadgroup half4*)(k_tile + row * D + col);
            threadgroup half4* vdst = (threadgroup half4*)(v_tile + row * D + col);
            if (k_idx < seq_len) {
                const device half4* ksrc = (const device half4*)(kbp + k_idx * ldS + col);
                const device half4* vsrc = (const device half4*)(vbp + k_idx * ldS + col);
                *kdst = *ksrc;
                *vdst = *vsrc;
            } else {
                *kdst = half4(0);
                *vdst = half4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S [BQ × BK] = Q [BQ × D] @ K^T [D × BK].
        // 4 tiles of 8 cols across BK; matmul tile is 8x8.
        simdgroup_float8x8 s_acc[4];
        for (uint tj = 0; tj < 4; tj++) s_acc[tj] = simdgroup_float8x8(0.0f);
        for (uint dt = 0; dt < Dt; dt++) {
            for (uint tj = 0; tj < 4; tj++) {
                simdgroup_half8x8 k_frag;
                // simdgroup_load with transpose=true reads K_tile[tj*8+r, dt*8+c]
                // and yields the 8x8 fragment b[c, r] — i.e. K^T's tile.
                simdgroup_load(k_frag, k_tile, D, ulong2(dt * 8, tj * 8), /*transpose=*/true);
                simdgroup_multiply_accumulate(s_acc[tj], q_frag[dt], k_frag, s_acc[tj]);
            }
        }
        // Store S as float to s_scratch, leading_dim = BK.
        for (uint tj = 0; tj < 4; tj++) {
            simdgroup_store(s_acc[tj], s_scratch, BK, ulong2(tj * 8, 0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Online softmax: per-row (threads 0..BQ-1 each handle one row) ----
        // The valid K range in this tile is [kbase, min(kbase+BK, seq_len)).
        // Out-of-range entries get -INF so they contribute 0 to softmax.
        if (tid < BQ) {
            uint row = tid;
            uint valid_k = min(BK, seq_len - kbase);

            // Pass 1: apply scale, mask invalid, find row-max.
            float m_cur = -INFINITY;
            for (uint j = 0; j < valid_k; j++) {
                float v = s_scratch[row * BK + j] * scale;
                s_scratch[row * BK + j] = v;
                m_cur = max(m_cur, v);
            }
            for (uint j = valid_k; j < BK; j++) {
                s_scratch[row * BK + j] = -INFINITY;
            }

            float m_new = max(m_state[row], m_cur);
            float alpha = exp(m_state[row] - m_new);

            // Pass 2: exponentiate, sum, write half P to p_tile.
            float sum = 0.0f;
            for (uint j = 0; j < BK; j++) {
                float e = exp(s_scratch[row * BK + j] - m_new);
                sum += e;
                p_tile[row * BK + j] = (half)e;
            }

            alpha_buf[row] = alpha;
            m_state[row]   = m_new;
            l_state[row]   = l_state[row] * alpha + sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Rescale O by alpha[row]: o_shared[row, :] *= alpha[row] ----
        // 32 threads stride across rows; each thread covers some columns of its row.
        for (uint i = tid; i < BQ * D; i += 32) {
            uint row = i / D;
            uint col = i % D;
            o_shared[row * D + col] *= alpha_buf[row];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O += P @ V : [BQ × D] += [BQ × BK] × [BK × D] ----
        // For each 8-col tile of O (dt), load existing 8x8 O fragment, accumulate
        // 4 P×V tile products over the BK direction, then store back.
        for (uint dt = 0; dt < Dt; dt++) {
            simdgroup_float8x8 o_frag;
            simdgroup_load(o_frag, o_shared, D, ulong2(dt * 8, 0));
            for (uint tk = 0; tk < 4; tk++) {
                simdgroup_half8x8 p_frag, v_frag;
                simdgroup_load(p_frag, p_tile, BK, ulong2(tk * 8, 0));
                simdgroup_load(v_frag, v_tile, D,  ulong2(dt * 8, tk * 8));
                simdgroup_multiply_accumulate(o_frag, p_frag, v_frag, o_frag);
            }
            simdgroup_store(o_frag, o_shared, D, ulong2(dt * 8, 0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Final normalisation: O /= l_state[row]; write to global ----
    if (tid < BQ) {
        float inv_l = 1.0f / max(l_state[tid], 1e-12f);
        uint q_idx = q_row_start + tid;
        if (q_idx < seq_len) {
            for (uint d = 0; d < D; d++) {
                obp[q_idx * ldS + d] = (half)(o_shared[tid * D + d] * inv_l);
            }
        }
    }
}
"#;

    /// MFA 2.0 Stage 3: two-simdgroup variant of `attention_simdmm_f16`.
    /// 64 threads (2 simdgroups), each simdgroup processes 8 query rows in
    /// parallel against a shared K/V tile (BK=16, smaller so the doubled
    /// per-simdgroup state still fits in 32 KB threadgroup memory at
    /// head_dim=160). K/V loads are shared across both simdgroups, doubling
    /// the matmul throughput per K-tile loaded.
    /// Grid: (num_heads, (seq_len + 16 - 1) / 16, batch).
    pub const ATTENTION_SIMDMM2: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void attention_simdmm2_f16(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& stride_batch [[buffer(8)]],
    constant uint& stride_head [[buffer(9)]],
    constant uint& stride_seq [[buffer(10)]],
    constant uint& stride_dim [[buffer(11)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    threadgroup uchar* shared [[threadgroup(0)]]
) {
    const uint BQ_SG = 8;       // queries per simdgroup
    const uint NSG   = 2;       // simdgroups per threadgroup
    const uint BQT   = BQ_SG * NSG; // 16 queries per threadgroup
    const uint BK    = 16;      // keys per K-tile (shared across simdgroups)
    const uint MAX_DT = 20;     // head_dim up to 160

    uint batch    = gid.z;
    uint head     = gid.x;
    uint q_block  = gid.y;
    uint tid      = lid.x;          // 0..63
    uint sg       = tid >> 5;       // simdgroup index 0 or 1
    uint lane     = tid & 31;       // lane within simdgroup

    uint base = batch * stride_batch + head * stride_head;
    device const half* qbp = Q + base;
    device const half* kbp = K + base;
    device const half* vbp = V + base;
    device       half* obp = O + base;

    uint D   = head_dim;
    uint Dt  = D / 8;
    uint ldS = stride_seq;

    // Per-simdgroup partition base offsets (sg-indexed slabs of SHM).
    threadgroup half*  q_tile_all    = (threadgroup half*)shared;
    threadgroup half*  q_tile_sg     = q_tile_all + sg * BQ_SG * D;            // each sg: BQ_SG×D half
    threadgroup half*  k_tile        = q_tile_all + NSG * BQ_SG * D;           // SHARED [BK×D] half
    threadgroup half*  v_tile        = k_tile + BK * D;                        // SHARED [BK×D] half
    threadgroup half*  p_tile_all    = v_tile + BK * D;
    threadgroup half*  p_tile_sg     = p_tile_all + sg * BQ_SG * BK;           // each sg: BQ_SG×BK half
    threadgroup float* s_scratch_all = (threadgroup float*)(p_tile_all + NSG * BQ_SG * BK);
    threadgroup float* s_scratch_sg  = s_scratch_all + sg * BQ_SG * BK;        // each sg: BQ_SG×BK float
    threadgroup float* o_shared_all  = s_scratch_all + NSG * BQ_SG * BK;
    threadgroup float* o_shared_sg   = o_shared_all + sg * BQ_SG * D;          // each sg: BQ_SG×D float
    threadgroup float* m_state_all   = o_shared_all + NSG * BQ_SG * D;
    threadgroup float* m_state_sg    = m_state_all + sg * BQ_SG;
    threadgroup float* l_state_all   = m_state_all + NSG * BQ_SG;
    threadgroup float* l_state_sg    = l_state_all + sg * BQ_SG;
    threadgroup float* alpha_all     = l_state_all + NSG * BQ_SG;
    threadgroup float* alpha_sg      = alpha_all + sg * BQ_SG;

    // ---- Init state per simdgroup ----
    if (lane < BQ_SG) {
        m_state_sg[lane] = -INFINITY;
        l_state_sg[lane] = 0.0f;
    }
    for (uint i = lane; i < BQ_SG * D; i += 32) o_shared_sg[i] = 0.0f;

    // ---- Load Q for this simdgroup [BQ_SG × D] (vec4) ----
    uint q_row_start = q_block * BQT + sg * BQ_SG;
    uint q_quads = (BQ_SG * D) >> 2;
    for (uint i = lane; i < q_quads; i += 32) {
        uint b = i << 2;
        uint row = b / D;
        uint col = b % D;
        uint q_idx = q_row_start + row;
        threadgroup half4* dst = (threadgroup half4*)(q_tile_sg + row * D + col);
        if (q_idx < seq_len) {
            const device half4* src = (const device half4*)(qbp + q_idx * ldS + col);
            *dst = *src;
        } else {
            *dst = half4(0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pre-load Q simdgroup fragments (Dt fragments per simdgroup).
    simdgroup_half8x8 q_frag[MAX_DT];
    for (uint dt = 0; dt < Dt; dt++) {
        simdgroup_load(q_frag[dt], q_tile_sg, D, ulong2(dt * 8, 0));
    }

    uint kv_quads = (BK * D) >> 2;
    for (uint kbase = 0; kbase < seq_len; kbase += BK) {
        // Load shared K/V tile cooperatively across ALL 64 threads.
        for (uint i = tid; i < kv_quads; i += 64) {
            uint b = i << 2;
            uint row = b / D;
            uint col = b % D;
            uint k_idx = kbase + row;
            threadgroup half4* kdst = (threadgroup half4*)(k_tile + row * D + col);
            threadgroup half4* vdst = (threadgroup half4*)(v_tile + row * D + col);
            if (k_idx < seq_len) {
                const device half4* ksrc = (const device half4*)(kbp + k_idx * ldS + col);
                const device half4* vsrc = (const device half4*)(vbp + k_idx * ldS + col);
                *kdst = *ksrc;
                *vdst = *vsrc;
            } else {
                *kdst = half4(0);
                *vdst = half4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S [BQ_SG × BK] = Q [BQ_SG × D] @ K^T [D × BK].
        // BK=16 = 2 tiles of 8 cols.
        simdgroup_float8x8 s_acc[2];
        for (uint tj = 0; tj < 2; tj++) s_acc[tj] = simdgroup_float8x8(0.0f);
        for (uint dt = 0; dt < Dt; dt++) {
            for (uint tj = 0; tj < 2; tj++) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, k_tile, D, ulong2(dt * 8, tj * 8), /*transpose=*/true);
                simdgroup_multiply_accumulate(s_acc[tj], q_frag[dt], k_frag, s_acc[tj]);
            }
        }
        for (uint tj = 0; tj < 2; tj++) {
            simdgroup_store(s_acc[tj], s_scratch_sg, BK, ulong2(tj * 8, 0));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax row-by-row (lanes 0..7 of each simdgroup).
        if (lane < BQ_SG) {
            uint row = lane;
            uint valid_k = min(BK, seq_len - kbase);

            float m_cur = -INFINITY;
            for (uint j = 0; j < valid_k; j++) {
                float v = s_scratch_sg[row * BK + j] * scale;
                s_scratch_sg[row * BK + j] = v;
                m_cur = max(m_cur, v);
            }
            for (uint j = valid_k; j < BK; j++) {
                s_scratch_sg[row * BK + j] = -INFINITY;
            }

            float m_new = max(m_state_sg[row], m_cur);
            float alpha = exp(m_state_sg[row] - m_new);
            float sum = 0.0f;
            for (uint j = 0; j < BK; j++) {
                float e = exp(s_scratch_sg[row * BK + j] - m_new);
                sum += e;
                p_tile_sg[row * BK + j] = (half)e;
            }

            alpha_sg[row]   = alpha;
            m_state_sg[row] = m_new;
            l_state_sg[row] = l_state_sg[row] * alpha + sum;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale per-simdgroup O by alpha[row].
        for (uint i = lane; i < BQ_SG * D; i += 32) {
            uint row = i / D;
            uint col = i % D;
            o_shared_sg[row * D + col] *= alpha_sg[row];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O += P @ V (BK=16 = 2 tk tiles).
        for (uint dt = 0; dt < Dt; dt++) {
            simdgroup_float8x8 o_frag;
            simdgroup_load(o_frag, o_shared_sg, D, ulong2(dt * 8, 0));
            for (uint tk = 0; tk < 2; tk++) {
                simdgroup_half8x8 p_frag, v_frag;
                simdgroup_load(p_frag, p_tile_sg, BK, ulong2(tk * 8, 0));
                simdgroup_load(v_frag, v_tile,    D,  ulong2(dt * 8, tk * 8));
                simdgroup_multiply_accumulate(o_frag, p_frag, v_frag, o_frag);
            }
            simdgroup_store(o_frag, o_shared_sg, D, ulong2(dt * 8, 0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // sync ALL sgs before next K load
    }

    // Final normalise + write to global.
    if (lane < BQ_SG) {
        float inv_l = 1.0f / max(l_state_sg[lane], 1e-12f);
        uint q_idx = q_row_start + lane;
        if (q_idx < seq_len) {
            for (uint d = 0; d < D; d++) {
                obp[q_idx * ldS + d] = (half)(o_shared_sg[lane * D + d] * inv_l);
            }
        }
    }
}
"#;

    /// Grouped Query Attention kernel for models like Qwen, Llama, etc.
    /// Supports different number of Q heads vs KV heads.
    pub const GQA_ATTENTION: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Grouped Query Attention - handles different Q/KV head counts
// Q shape: [seq_len, num_q_heads, head_dim]
// K shape: [seq_len, num_kv_heads, head_dim]
// V shape: [seq_len, num_kv_heads, head_dim]
// O shape: [seq_len, num_q_heads, head_dim]

kernel void gqa_attention_f16(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& num_q_heads [[buffer(5)]],
    constant uint& num_kv_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]  // (q_head, query_pos)
) {
    uint q_head = gid.x;
    uint query_pos = gid.y;

    if (q_head >= num_q_heads || query_pos >= seq_len) return;

    // Map Q head to KV head (GQA: multiple Q heads share one KV head)
    uint kv_head = q_head / (num_q_heads / num_kv_heads);

    // Pointers for this head
    device const half* q = Q + (query_pos * num_q_heads + q_head) * head_dim;
    device half* o = O + (query_pos * num_q_heads + q_head) * head_dim;

    // Online softmax for numerical stability
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // First pass: find max and compute exp sum
    for (uint k_pos = 0; k_pos <= query_pos; k_pos++) {  // Causal: only attend to past
        device const half* k = K + (k_pos * num_kv_heads + kv_head) * head_dim;

        // Dot product Q @ K
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(q[d]) * float(k[d]);
        }
        dot *= scale;

        // Update running max and sum
        float old_max = max_score;
        max_score = max(max_score, dot);
        sum_exp = sum_exp * exp(old_max - max_score) + exp(dot - max_score);
    }

    // Second pass: compute weighted sum of V
    float inv_sum = 1.0f / sum_exp;
    for (uint d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (uint k_pos = 0; k_pos <= query_pos; k_pos++) {
            device const half* k = K + (k_pos * num_kv_heads + kv_head) * head_dim;
            device const half* v = V + (k_pos * num_kv_heads + kv_head) * head_dim;

            float dot = 0.0f;
            for (uint dd = 0; dd < head_dim; dd++) {
                dot += float(q[dd]) * float(k[dd]);
            }
            dot *= scale;

            float weight = exp(dot - max_score) * inv_sum;
            acc += weight * float(v[d]);
        }
        o[d] = half(acc);
    }
}
"#;

    /// Autoregressive attention with KV cache.
    /// Computes attention for a single query position against cached K/V.
    /// Q shape: [num_q_heads, head_dim] (single position)
    /// K_cache shape: [max_seq_len, num_kv_heads, head_dim]
    /// V_cache shape: [max_seq_len, num_kv_heads, head_dim]
    /// O shape: [num_q_heads, head_dim]
    pub const AUTOREGRESSIVE_ATTENTION: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void autoregressive_attention_f16(
    device const half* Q [[buffer(0)]],           // [num_q_heads, head_dim]
    device const half* K_cache [[buffer(1)]],     // [max_seq_len, num_kv_heads, head_dim]
    device const half* V_cache [[buffer(2)]],     // [max_seq_len, num_kv_heads, head_dim]
    device half* O [[buffer(3)]],                 // [num_q_heads, head_dim]
    constant uint& seq_pos [[buffer(4)]],         // Current position (attend 0..seq_pos inclusive)
    constant uint& num_q_heads [[buffer(5)]],
    constant uint& num_kv_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint q_head [[thread_position_in_grid]]
) {
    if (q_head >= num_q_heads) return;

    // Map Q head to KV head (GQA: multiple Q heads share one KV head)
    uint kv_head = (num_q_heads == num_kv_heads) ? q_head : (q_head / (num_q_heads / num_kv_heads));

    // Pointer to this Q head
    device const half* q = Q + q_head * head_dim;
    device half* o = O + q_head * head_dim;

    // Online softmax with numerical stability
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // First pass: compute max and sum
    for (uint k_pos = 0; k_pos <= seq_pos; k_pos++) {
        device const half* k = K_cache + (k_pos * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(q[d]) * float(k[d]);
        }
        dot *= scale;

        float old_max = max_score;
        max_score = max(max_score, dot);
        sum_exp = sum_exp * exp(old_max - max_score) + exp(dot - max_score);
    }

    // Second pass: compute weighted sum of V
    float inv_sum = 1.0f / sum_exp;

    // Initialize output to zero
    for (uint d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (uint k_pos = 0; k_pos <= seq_pos; k_pos++) {
            device const half* k = K_cache + (k_pos * num_kv_heads + kv_head) * head_dim;
            device const half* v = V_cache + (k_pos * num_kv_heads + kv_head) * head_dim;

            float dot = 0.0f;
            for (uint dd = 0; dd < head_dim; dd++) {
                dot += float(q[dd]) * float(k[dd]);
            }
            dot *= scale;

            float weight = exp(dot - max_score) * inv_sum;
            acc += weight * float(v[d]);
        }
        o[d] = half(acc);
    }
}

// Optimized version with threadgroup memory
kernel void autoregressive_attention_tg_f16(
    device const half* Q [[buffer(0)]],
    device const half* K_cache [[buffer(1)]],
    device const half* V_cache [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant uint& seq_pos [[buffer(4)]],
    constant uint& num_q_heads [[buffer(5)]],
    constant uint& num_kv_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tg_size_vec [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    uint q_head = gid.x;
    uint tg_size = tg_size_vec.x;
    if (q_head >= num_q_heads) return;

    uint kv_head = (num_q_heads == num_kv_heads) ? q_head : (q_head / (num_q_heads / num_kv_heads));

    device const half* q = Q + q_head * head_dim;
    device half* o = O + q_head * head_dim;

    // Threadgroup partitions for scores and accumulator
    threadgroup float* scores = shared;
    threadgroup float* acc = shared + (seq_pos + 1);

    // Each thread handles some dimensions
    uint seq_len = seq_pos + 1;
    uint thread_idx = lid.x;

    // Parallel dot product computation
    float max_score = -INFINITY;
    for (uint k_pos = thread_idx; k_pos < seq_len; k_pos += tg_size) {
        device const half* k = K_cache + (k_pos * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(q[d]) * float(k[d]);
        }
        scores[k_pos] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find max (reduction)
    if (thread_idx == 0) {
        for (uint i = 0; i < seq_len; i++) {
            max_score = max(max_score, scores[i]);
        }
        shared[seq_len] = max_score;  // Store for other threads
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = shared[seq_len];

    // Softmax
    float local_sum = 0.0f;
    for (uint k_pos = thread_idx; k_pos < seq_len; k_pos += tg_size) {
        scores[k_pos] = exp(scores[k_pos] - max_score);
        local_sum += scores[k_pos];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce sum
    shared[thread_idx] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_idx == 0) {
        float total = 0.0f;
        for (uint i = 0; i < min(seq_len, tg_size); i++) {
            total += shared[i];
        }
        shared[seq_len + 1] = 1.0f / total;  // inv_sum
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = shared[seq_len + 1];

    // Compute output for each dimension
    for (uint d = thread_idx; d < head_dim; d += tg_size) {
        float sum = 0.0f;
        for (uint k_pos = 0; k_pos < seq_len; k_pos++) {
            device const half* v = V_cache + (k_pos * num_kv_heads + kv_head) * head_dim;
            sum += scores[k_pos] * inv_sum * float(v[d]);
        }
        o[d] = half(sum);
    }
}
"#;

    /// Fused SwiGLU MLP for Qwen/Llama style models.
    /// Computes: down_proj(SiLU(gate_proj(x)) * up_proj(x))
    pub const SWIGLU: &str = r#"
#include <metal_stdlib>
using namespace metal;

// SwiGLU activation: SiLU(gate) * up
kernel void swiglu_f16(
    device const half* gate [[buffer(0)]],    // SiLU applied to this
    device const half* up [[buffer(1)]],      // Element-wise multiply
    device half* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float g = float(gate[gid]);
    float u = float(up[gid]);
    float silu_g = g / (1.0f + exp(-g));  // SiLU(gate)
    output[gid] = half(silu_g * u);
}

// SwiGLU split: input[N, 2H] -> output[N, H] = SiLU(input[:, :H]) * input[:, H:]
// Used by HF DINOv2-Giant SwiGLUFFN (weights_in produces concatenated gate||value).
kernel void swiglu_split_f16(
    device const half* input [[buffer(0)]],   // [count, 2*half_dim]
    device half* output [[buffer(1)]],        // [count, half_dim]
    constant uint& half_dim [[buffer(2)]],
    constant uint& count [[buffer(3)]],       // total output elements = N * half_dim
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    uint n = gid / half_dim;
    uint h = gid - n * half_dim;
    uint row_base = n * 2u * half_dim;
    float g = float(input[row_base + h]);
    float u = float(input[row_base + half_dim + h]);
    float silu_g = g / (1.0f + exp(-g));
    output[gid] = half(silu_g * u);
}
"#;

    /// Argmax kernel for greedy sampling.
    pub const ARGMAX: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Result {
    float val;
    uint idx;
};

// Find argmax of a vector (for greedy decoding)
// Optimized parallel reduction using SIMD and shared memory
// Should be dispatched as a single threadgroup (e.g. 256 threads)
kernel void argmax_f16(
    device const half* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared_val [[threadgroup(0)]],
    threadgroup uint* shared_idx [[threadgroup(1)]]
) {
    // 1. Per-thread reduction (Grid-Stride Loop)
    // Initialize with first element this thread would handle
    // But be careful if size < tg_size. 
    // Safer to init with -INFINITY.
    
    float max_val = -INFINITY;
    uint max_idx = 0;
    
    for (uint i = lid; i < size; i += tg_size) {
        float val = float(input[i]);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    
    // 2. Store to shared memory
    shared_val[lid] = max_val;
    shared_idx[lid] = max_idx;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 3. Parallel Reduction in Shared Memory
    // Reduce from tg_size down to 1
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            if (shared_val[lid + s] > shared_val[lid]) {
                shared_val[lid] = shared_val[lid + s];
                shared_idx[lid] = shared_idx[lid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // 4. Write result
    if (lid == 0) {
        output[0] = shared_idx[0];
    }
}
"#;

    /// Embedding lookup kernel.
    pub const EMBEDDING: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Embedding lookup - each thread handles one token's embedding
kernel void embedding_lookup_f16(
    device const half* embed_table [[buffer(0)]],
    device const uint* token_ids [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= seq_len) return;

    uint token_id = token_ids[gid];
    if (token_id >= vocab_size) return; // OOV protection

    // Copy embedding for this token
    device const half* src = embed_table + token_id * hidden_size;
    device half* dst = output + gid * hidden_size;

    for (uint i = 0; i < hidden_size; i++) {
        dst[i] = src[i];
    }
}

// Embedding lookup for GGUF format where table is stored as [hidden_size, vocab_size]
// GGUF embeddings are stored as [hidden_size, vocab_size] column-major
// Column v (token v's embedding) is contiguous: starts at v * hidden_size
kernel void embedding_lookup_colmajor_f16(
    device const half* embed_table [[buffer(0)]],
    device const uint* token_ids [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= seq_len) return;

    uint token_id = token_ids[gid];
    if (token_id >= vocab_size) return; // OOV protection

    // Copy embedding for this token from column-major layout
    // Element [h, token_id] is at embed_table[token_id * hidden_size + h]
    device half* dst = output + gid * hidden_size;
    device const half* src = embed_table + token_id * hidden_size;

    for (uint h = 0; h < hidden_size; h++) {
        dst[h] = src[h];
    }
}
"#;

    /// Gaussian splatting kernel for 3D rendering.
    pub const GAUSSIAN_SPLAT: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Gaussian {
    float3 position;
    float3 scale;
    float4 rotation; // quaternion
    float3 color;
    float opacity;
};

struct Camera {
    float4x4 view_matrix;
    float4x4 proj_matrix;
    float2 resolution;
};

kernel void splat_gaussians(
    device const Gaussian* gaussians [[buffer(0)]],
    device const uint* sorted_indices [[buffer(1)]],
    device float4* output [[buffer(2)]],
    constant Camera& camera [[buffer(3)]],
    constant uint& num_gaussians [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint x = gid.x;
    uint y = gid.y;

    if (x >= uint(camera.resolution.x) || y >= uint(camera.resolution.y)) return;

    float2 pixel = float2(x, y);
    float4 color = float4(0.0);

    // Accumulate from front to back
    for (uint i = 0; i < num_gaussians && color.a < 0.99; i++) {
        uint idx = sorted_indices[i];
        Gaussian g = gaussians[idx];

        // Project gaussian center
        float4 world_pos = float4(g.position, 1.0);
        float4 view_pos = camera.view_matrix * world_pos;
        float4 clip_pos = camera.proj_matrix * view_pos;

        if (clip_pos.w <= 0) continue;

        float2 screen_pos = (clip_pos.xy / clip_pos.w + 1.0) * 0.5 * camera.resolution;

        // Compute gaussian contribution
        float2 diff = pixel - screen_pos;
        float dist_sq = dot(diff, diff);

        // Simple spherical gaussian for now
        float radius = length(g.scale) * 100.0; // Adjust scale
        if (dist_sq > radius * radius) continue;

        float alpha = g.opacity * exp(-0.5 * dist_sq / (radius * radius * 0.1));

        // Blend
        color.rgb += (1.0 - color.a) * alpha * g.color;
        color.a += (1.0 - color.a) * alpha;
    }

    output[y * uint(camera.resolution.x) + x] = color;
}
"#;

    /// 2D Convolution kernel.
    pub const CONV2D: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Naive Conv2D (N, C, H, W) layout
kernel void conv2d_naive_f16(
    device const half* input [[buffer(0)]],      // [N, Cin, Hin, Win]
    device const half* weight [[buffer(1)]],     // [Cout, Cin, KH, KW]
    device const half* bias [[buffer(2)]],       // [Cout]
    device half* output [[buffer(3)]],           // [N, Cout, Hout, Wout]
    constant uint& Cin [[buffer(4)]],
    constant uint& Hin [[buffer(5)]],
    constant uint& Win [[buffer(6)]],
    constant uint& Cout [[buffer(7)]],
    constant uint& Hout [[buffer(8)]],
    constant uint& Wout [[buffer(9)]],
    constant uint& KW [[buffer(10)]],
    constant uint& KH [[buffer(11)]],
    constant uint& pad_x [[buffer(12)]],
    constant uint& pad_y [[buffer(13)]],
    constant uint& stride_x [[buffer(14)]],
    constant uint& stride_y [[buffer(15)]],
    constant uint& BatchSize [[buffer(16)]],
    uint3 gid [[thread_position_in_grid]]        // (Wout, Hout, Cout * BatchSize)
) {
    uint ow = gid.x;
    uint oh = gid.y;
    
    // Decode batch and channel
    uint batch_idx = gid.z / Cout;
    uint oc = gid.z % Cout;

    if (ow >= Wout || oh >= Hout || batch_idx >= BatchSize) return;

    float sum = 0.0f;
    if (bias) {
        sum = float(bias[oc]);
    }

    uint in_batch_offset = batch_idx * Cin * Hin * Win;

    for (uint ic = 0; ic < Cin; ic++) {
        for (uint kh = 0; kh < KH; kh++) {
            for (uint kw = 0; kw < KW; kw++) {
                int ih = int(oh * stride_y + kh) - int(pad_y);
                int iw = int(ow * stride_x + kw) - int(pad_x);

                if (ih >= 0 && ih < int(Hin) && iw >= 0 && iw < int(Win)) {
                    // Index calc: n*C*H*W + c*H*W + h*W + w
                    uint in_idx = in_batch_offset + ic * Hin * Win + uint(ih) * Win + uint(iw);
                    uint w_idx = oc * Cin * KH * KW + ic * KH * KW + kh * KW + kw;

                    sum += float(input[in_idx]) * float(weight[w_idx]);
                }
            }
        }
    }

    // Output index
    uint out_batch_offset = batch_idx * Cout * Hout * Wout;
    uint out_idx = out_batch_offset + oc * Hout * Wout + oh * Wout + ow;
    output[out_idx] = half(sum);
}

// Optimized 3x3 Conv2D using threadgroup memory tiling.
// Reverted to basic tiled version (no channel blocking) as blocking reduced performance.
kernel void conv2d_3x3_tiled_f16(
    device const half* input [[buffer(0)]],      // [N, Cin, Hin, Win]
    device const half* weight [[buffer(1)]],     // [Cout, Cin, 3, 3]
    device const half* bias [[buffer(2)]],       // [Cout]
    device half* output [[buffer(3)]],           // [N, Cout, Hout, Wout]
    constant uint& Cin [[buffer(4)]],
    constant uint& Hin [[buffer(5)]],
    constant uint& Win [[buffer(6)]],
    constant uint& Cout [[buffer(7)]],
    constant uint& Hout [[buffer(8)]],
    constant uint& Wout [[buffer(9)]],
    constant uint& KW [[buffer(10)]],           
    constant uint& KH [[buffer(11)]],           
    constant uint& pad_x [[buffer(12)]],
    constant uint& pad_y [[buffer(13)]],
    constant uint& stride_x [[buffer(14)]],
    constant uint& stride_y [[buffer(15)]],
    constant uint& BatchSize [[buffer(16)]],
    uint3 gid [[thread_position_in_grid]],       // Global ID
    uint3 lid [[thread_position_in_threadgroup]], // Local ID
    uint3 tag [[threadgroup_position_in_grid]]    // Group ID
) {
    // Only support stride=1 for this optimized kernel
    if (stride_x != 1 || stride_y != 1) { return; }

    uint ow = gid.x;
    uint oh = gid.y;
    
    // Decode batch and output channel
    uint batch_idx = gid.z / Cout;
    uint oc = gid.z % Cout;

    // Local coordinates
    uint lx = lid.x; // 0..15
    uint ly = lid.y; // 0..15

    // Tile Input dimensions
    // We need 16 output pixels -> 18 input pixels (for 3x3)
    // Shared memory size: 18x18
    threadgroup half s_in[18][18];

    float sum = 0.0f;
    if (bias && oh < Hout && ow < Wout && batch_idx < BatchSize) {
        sum = float(bias[oc]);
    }
    
    // Pointer to weights for this output channel
    device const half* w_ptr = weight + oc * Cin * 9;
    
    // Batch offset
    uint in_batch_offset = batch_idx * Cin * Hin * Win;
    
    // Loop over input channels
    for (uint ic = 0; ic < Cin; ic++) {
        // Collaborative loading
        // We have 256 threads. We need to load 18*18 = 324 elements.
        // Some threads load 2 elements.
        
        // Base coordinate of the tile in the input image
        // tag.x * 16 corresponds to output x. Input x matches (stride 1).
        // subtract padding
        int base_x = int(tag.x * 16) - int(pad_x);
        int base_y = int(tag.y * 16) - int(pad_y);

        // Map linear thread index to loading
        uint tid = ly * 16 + lx; // 0..255
        
        // Load first 324 elements (18x18)
        for (uint i = tid; i < 324; i += 256) {
             uint ty = i / 18;
             uint tx = i % 18;
             
             int ix = base_x + int(tx);
             int iy = base_y + int(ty);
             
             half val = 0.0h;
             if (ix >= 0 && ix < int(Win) && iy >= 0 && iy < int(Hin)) {
                 // ic * HW + iy * W + ix
                 uint idx = in_batch_offset + ic * Hin * Win + uint(iy) * Win + uint(ix);
                 val = input[idx];
             }
             s_in[ty][tx] = val;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute if valid thread
        if (oh < Hout && ow < Wout && batch_idx < BatchSize) {
            // Unroll 3x3
            // w_ptr is updated? No. We access w_ptr + ic*9.
            // Wait, previous version used w_ptr + ic*9.
            // w_ptr points to oc*Cin*9.
            
            device const half* w = w_ptr + ic * 9;
            
            sum += float(s_in[ly][lx])     * float(w[0]);
            sum += float(s_in[ly][lx+1])   * float(w[1]);
            sum += float(s_in[ly][lx+2])   * float(w[2]);
            
            sum += float(s_in[ly+1][lx])   * float(w[3]);
            sum += float(s_in[ly+1][lx+1]) * float(w[4]);
            sum += float(s_in[ly+1][lx+2]) * float(w[5]);
            
            sum += float(s_in[ly+2][lx])   * float(w[6]);
            sum += float(s_in[ly+2][lx+1]) * float(w[7]);
            sum += float(s_in[ly+2][lx+2]) * float(w[8]);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write output
    if (oh < Hout && ow < Wout && batch_idx < BatchSize) {
        uint out_batch_offset = batch_idx * Cout * Hout * Wout;
        uint out_idx = out_batch_offset + oc * Hout * Wout + oh * Wout + ow;
        output[out_idx] = half(sum);
    }
}

// SIMD-optimized 1x1 Conv2D (MatMul).
//
// Accumulator is `simdgroup_float8x8` (fp32), not half. For Cin = 640 / 1280 /
// 2560 (SD 1.5's conv_shortcut and proj_in/proj_out paths), the K-loop runs
// 80 / 160 / 320 iterations and adds an 8x8 matmul each pass. With an fp16
// accumulator the running sum loses precision rapidly — small per-channel
// spatial-variation signal gets quantised to zero, which surfaces as the
// "channels are spatially flat" symptom (our up_block per-channel std was
// 7x lower than PyTorch's even though overall std matched). fp32 acc fixes
// it; the inputs A/B stay fp16, and only the final cast to fp16 happens at
// store time.
kernel void conv2d_1x1_simd_f16(
    device const half* input [[buffer(0)]],      // [N, Cin, Hin, Win]
    device const half* weight [[buffer(1)]],     // [Cout, Cin, 1, 1]
    device const half* bias [[buffer(2)]],       // [Cout]
    device half* output [[buffer(3)]],           // [N, Cout, Hout, Wout]
    constant uint& Cin [[buffer(4)]],
    constant uint& Hin [[buffer(5)]],
    constant uint& Win [[buffer(6)]],
    constant uint& Cout [[buffer(7)]],
    constant uint& Hout [[buffer(8)]],
    constant uint& Wout [[buffer(9)]],
    constant uint& KW [[buffer(10)]],
    constant uint& KH [[buffer(11)]],
    constant uint& pad_x [[buffer(12)]],
    constant uint& pad_y [[buffer(13)]],
    constant uint& stride_x [[buffer(14)]],
    constant uint& stride_y [[buffer(15)]],
    constant uint& BatchSize [[buffer(16)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    uint batch_idx = gid.z;
    if (batch_idx >= BatchSize) return;

    uint HW = Hin * Win;

    uint tile_m = gid.y * 8;
    uint tile_n = gid.x * 8;

    uint in_batch_offset = batch_idx * Cin * HW;
    uint out_batch_offset = batch_idx * Cout * HW;

    // fp32 accumulator — see kernel header note.
    simdgroup_float8x8 acc = simdgroup_float8x8(0);

    for (uint k = 0; k < Cin; k += 8) {
        simdgroup_half8x8 a;
        simdgroup_half8x8 b;
        device const half* w_ptr = weight + tile_m * Cin + k;
        simdgroup_load(a, w_ptr, Cin);
        device const half* in_ptr = input + in_batch_offset + k * HW + tile_n;
        simdgroup_load(b, in_ptr, HW);
        simdgroup_multiply_accumulate(acc, a, b, acc);
    }

    // Store as fp32 to a threadgroup staging area, then convert and write
    // through with bias add per lane to avoid an intermediate fp16 truncation
    // of `acc` (which would defeat the whole point of the fp32 accumulator).
    threadgroup float stage[64];
    simdgroup_store(acc, (threadgroup float*)stage, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = simd_lane_id; i < 64; i += 32) {
        uint m = i / 8;
        uint n_off = i % 8;
        uint oc = tile_m + m;
        uint hw_idx = tile_n + n_off;
        if (oc < Cout && hw_idx < HW) {
            float v = stage[m * 8 + n_off];
            if (bias) v += float(bias[oc]);
            uint idx = out_batch_offset + oc * HW + hw_idx;
            output[idx] = half(v);
        }
    }
}

// Specialized 1x1 Conv2D
kernel void conv2d_1x1_f16(
    device const half* input [[buffer(0)]],      // [N, Cin, Hin, Win]
    device const half* weight [[buffer(1)]],     // [Cout, Cin, 1, 1]
    device const half* bias [[buffer(2)]],       // [Cout]
    device half* output [[buffer(3)]],           // [N, Cout, Hout, Wout]
    constant uint& Cin [[buffer(4)]],
    constant uint& Hin [[buffer(5)]],
    constant uint& Win [[buffer(6)]],
    constant uint& Cout [[buffer(7)]],
    constant uint& Hout [[buffer(8)]],
    constant uint& Wout [[buffer(9)]],
    constant uint& KW [[buffer(10)]],
    constant uint& KH [[buffer(11)]],
    constant uint& pad_x [[buffer(12)]],
    constant uint& pad_y [[buffer(13)]],
    constant uint& stride_x [[buffer(14)]],
    constant uint& stride_y [[buffer(15)]],
    constant uint& BatchSize [[buffer(16)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ow = gid.x;
    uint oh = gid.y;
    
    // Decode batch and channel
    uint batch_idx = gid.z / Cout;
    uint oc = gid.z % Cout;

    if (ow >= Wout || oh >= Hout || batch_idx >= BatchSize) return;

    uint out_batch_offset = batch_idx * Cout * Hout * Wout;
    uint out_spatial_idx = oh * Wout + ow;

    // For 1x1 with stride/padding, we compute input indices directly.
    int ih = int(oh * stride_y) - int(pad_y);
    int iw = int(ow * stride_x) - int(pad_x);

    // Bounds check for padding
    if (ih < 0 || ih >= int(Hin) || iw < 0 || iw >= int(Win)) {
        float val = 0.0f;
        if (bias) val = float(bias[oc]);
        output[out_batch_offset + oc * Hout * Wout + out_spatial_idx] = half(val);
        return;
    }
    
    uint in_spatial_idx = uint(ih) * Win + uint(iw);
    uint in_batch_offset = batch_idx * Cin * Hin * Win;

    float sum = 0.0f;
    if (bias) {
        sum = float(bias[oc]);
    }

    // Weight pointer for this output channel [oc, 0, 0, 0]
    device const half* w_ptr = weight + oc * Cin;
    
    // Input pointer base
    device const half* in_ptr = input + in_batch_offset;
    
    // Stride for input channel
    uint in_stride = Hin * Win;

    for (uint ic = 0; ic < Cin; ic++) {
        // Input: [ic, ih, iw]
        half val = in_ptr[ic * in_stride + in_spatial_idx];
        half w = w_ptr[ic];
        sum += float(val) * float(w);
    }

    output[out_batch_offset + oc * Hout * Wout + out_spatial_idx] = half(sum);
}

// Specialized 3x3 Conv2D
kernel void conv2d_3x3_f16(
    device const half* input [[buffer(0)]],      // [N, Cin, Hin, Win]
    device const half* weight [[buffer(1)]],     // [Cout, Cin, 3, 3]
    device const half* bias [[buffer(2)]],       // [Cout]
    device half* output [[buffer(3)]],           // [N, Cout, Hout, Wout]
    constant uint& Cin [[buffer(4)]],
    constant uint& Hin [[buffer(5)]],
    constant uint& Win [[buffer(6)]],
    constant uint& Cout [[buffer(7)]],
    constant uint& Hout [[buffer(8)]],
    constant uint& Wout [[buffer(9)]],
    constant uint& KW [[buffer(10)]],           // Unused in optimized loop but kept for signature
    constant uint& KH [[buffer(11)]],           // Unused
    constant uint& pad_x [[buffer(12)]],
    constant uint& pad_y [[buffer(13)]],
    constant uint& stride_x [[buffer(14)]],
    constant uint& stride_y [[buffer(15)]],
    constant uint& BatchSize [[buffer(16)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ow = gid.x;
    uint oh = gid.y;
    
    // Decode batch and channel
    uint batch_idx = gid.z / Cout;
    uint oc = gid.z % Cout;

    if (ow >= Wout || oh >= Hout || batch_idx >= BatchSize) return;

    float sum = 0.0f;
    if (bias) {
        sum = float(bias[oc]);
    }
    
    // Weight pointer for this output channel
    device const half* w_ptr = weight + oc * Cin * 9;
    
    uint in_batch_offset = batch_idx * Cin * Hin * Win;

    for (uint ic = 0; ic < Cin; ic++) {
        // Input pointer for this input channel
        device const half* in_ptr = input + in_batch_offset + ic * Hin * Win;
        
        // Unrolled 3x3 loop
        for (int kh = 0; kh < 3; kh++) {
            int ih = int(oh * stride_y + kh) - int(pad_y);
            
            // Bounds check Y
            if (ih >= 0 && ih < int(Hin)) {
                // Row pointer
                device const half* in_row_ptr = in_ptr + uint(ih) * Win;
                
                for (int kw = 0; kw < 3; kw++) {
                    int iw = int(ow * stride_x + kw) - int(pad_x);
                    
                    // Bounds check X
                    if (iw >= 0 && iw < int(Win)) {
                        half val = in_row_ptr[uint(iw)];
                        half w = w_ptr[kh * 3 + kw];
                        sum += float(val) * float(w);
                    }
                }
            }
        }
        
        // Advance weight pointer
        w_ptr += 9;
    }

    // Output index
    uint out_batch_offset = batch_idx * Cout * Hout * Wout;
    uint out_idx = out_batch_offset + oc * Hout * Wout + oh * Wout + ow;
    output[out_idx] = half(sum);
}

// im2col for 3x3 convolution with configurable stride + padding (zero-pad).
// Input layout: [1, Cin, Hin, Win] CHW f16
// Output layout: [Hout * Wout, Cin * 9] f16, row-major.
//
// Each thread writes ONE output element: output[p * (Cin*9) + k] where
//   p = oy * Wout + ox          (output spatial position)
//   k = ic * 9 + ky * 3 + kx    (input channel + kernel position)
//
// Dispatched as 1D over (Hout * Wout * Cin * 9) threads.
//
// Used by the v6.7 GPU-accelerated DPT path to bypass the CPU im2col bottleneck
// (~200 MB write per 296x296x256ch conv).
kernel void im2col_3x3_f16(
    device const half* input [[buffer(0)]],   // [Cin, Hin, Win]
    device half*       output [[buffer(1)]],  // [Hout*Wout, Cin*9]
    constant uint& Cin      [[buffer(2)]],
    constant uint& Hin      [[buffer(3)]],
    constant uint& Win      [[buffer(4)]],
    constant uint& Hout     [[buffer(5)]],
    constant uint& Wout     [[buffer(6)]],
    constant uint& Pad      [[buffer(7)]],
    constant uint& Stride   [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = Hout * Wout * Cin * 9u;
    if (gid >= total) { return; }

    uint K = Cin * 9u;
    uint p = gid / K;
    uint k = gid % K;
    uint oy = p / Wout;
    uint ox = p % Wout;
    uint ic = k / 9u;
    uint k_inner = k % 9u;
    uint ky = k_inner / 3u;
    uint kx = k_inner % 3u;

    int iy = int(oy * Stride + ky) - int(Pad);
    int ix = int(ox * Stride + kx) - int(Pad);

    half v = 0.0h;
    if (iy >= 0 && iy < int(Hin) && ix >= 0 && ix < int(Win)) {
        v = input[ic * Hin * Win + uint(iy) * Win + uint(ix)];
    }
    output[p * K + k] = v;
}

// CHW → HWC reshape for 1x1 conv path.
// Input: [1, Cin, H, W] CHW
// Output: [H*W, Cin] (row-major sequence of pixels)
// 1D dispatch over (H * W * Cin) threads.
kernel void chw_to_hwc_f16(
    device const half* input  [[buffer(0)]],  // [Cin, H, W]
    device half*       output [[buffer(1)]],  // [H*W, Cin]
    constant uint& Cin [[buffer(2)]],
    constant uint& H   [[buffer(3)]],
    constant uint& W   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = H * W * Cin;
    if (gid >= total) { return; }
    uint hw = H * W;
    uint c = gid / hw;
    uint p = gid % hw;
    output[p * Cin + c] = input[c * hw + p];
}

// HWC → CHW reshape for output of 1x1 / 3x3 matmul-based conv.
// Input: [H*W, Cout]
// Output: [1, Cout, H, W] CHW
// 1D dispatch over (H * W * Cout) threads.
kernel void hwc_to_chw_f16(
    device const half* input  [[buffer(0)]],  // [H*W, Cout]
    device half*       output [[buffer(1)]],  // [Cout, H, W]
    constant uint& Cout [[buffer(2)]],
    constant uint& H    [[buffer(3)]],
    constant uint& W    [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = H * W * Cout;
    if (gid >= total) { return; }
    uint hw = H * W;
    uint c = gid / hw;
    uint p = gid % hw;
    output[c * hw + p] = input[p * Cout + c];
}
"#;

    /// Group Normalization kernel.
    pub const GROUP_NORM: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Calculate Mean and Variance per group
// Grid: (Groups, Batch, 1)
// Threadgroup size: e.g. 256
kernel void group_norm_stats_f16(
    device const half* input [[buffer(0)]],
    device float2* temp_stats [[buffer(1)]], // [N, Groups] -> (mean, var)
    constant uint& N [[buffer(2)]],
    constant uint& G [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    constant uint& HW [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 threads_per_tg [[threads_per_threadgroup]]
) {
    uint g = gid.x;
    uint n = gid.y;
    uint thread_idx = lid.x;
    uint tg_size = threads_per_tg.x;
    
    uint C_per_G = C / G;
    uint num_elements = C_per_G * HW;
    
    // Parallel reduction sum
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Loop over all elements in this group (C_per_G channels, HW pixels)
    for (uint i = thread_idx; i < num_elements; i += tg_size) {
        // Map i to (c_local, hw)
        uint c_local = i / HW;
        uint hw = i % HW;
        
        uint c = g * C_per_G + c_local;
        uint idx = n * C * HW + c * HW + hw;
        
        float val = float(input[idx]);
        sum += val;
        sum_sq += val * val;
    }
    
    // Threadgroup Reduction
    threadgroup float s_sum[256]; 
    threadgroup float s_sq[256];
    
    // Ensure we don't overflow buffer if threads > 256 (should cap dispatch at 256)
    if (thread_idx < 256) {
        s_sum[thread_idx] = sum;
        s_sq[thread_idx] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Naive reduction
    // Only works if threads_per_tg is power of 2, e.g. 256
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (thread_idx < s && thread_idx < 256) {
            s_sum[thread_idx] += s_sum[thread_idx + s];
            s_sq[thread_idx] += s_sq[thread_idx + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (thread_idx == 0) {
        float mean = s_sum[0] / float(num_elements);
        float mean_sq = s_sq[0] / float(num_elements);
        float var = mean_sq - mean * mean;
        temp_stats[n * G + g] = float2(mean, var);
    }
}

// Apply normalization, gamma, beta
// Grid: (HW, C, N)
kernel void group_norm_apply_f16(
    device const half* input [[buffer(0)]],
    device const float2* temp_stats [[buffer(1)]],
    device const half* gamma [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& G [[buffer(6)]],
    constant uint& C [[buffer(7)]],
    constant uint& HW [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint hw = gid.x;
    uint c = gid.y;
    uint n = gid.z;
    
    if (hw >= HW || c >= C || n >= N) return;
    
    uint C_per_G = C / G;
    uint g = c / C_per_G;
    
    float2 stats = temp_stats[n * G + g];
    float mean = stats.x;
    float var = stats.y;
    float inv_std = rsqrt(max(var, 0.0f) + eps);
    
    uint idx = n * C * HW + c * HW + hw;
    float val = float(input[idx]);
    float norm = (val - mean) * inv_std;
    
    float gm = (gamma) ? float(gamma[c]) : 1.0f;
    float bt = (bias) ? float(bias[c]) : 0.0f;
    
    output[idx] = half(norm * gm + bt);
}

// Fused GroupNorm + SiLU apply
// Same as group_norm_apply_f16 but applies SiLU activation after normalization
// Grid: (HW, C, N)
kernel void group_norm_silu_apply_f16(
    device const half* input [[buffer(0)]],
    device const float2* temp_stats [[buffer(1)]],
    device const half* gamma [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& G [[buffer(6)]],
    constant uint& C [[buffer(7)]],
    constant uint& HW [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint hw = gid.x;
    uint c = gid.y;
    uint n = gid.z;

    if (hw >= HW || c >= C || n >= N) return;

    uint C_per_G = C / G;
    uint g = c / C_per_G;

    float2 stats = temp_stats[n * G + g];
    float mean = stats.x;
    float var = stats.y;
    float inv_std = rsqrt(max(var, 0.0f) + eps);

    uint idx = n * C * HW + c * HW + hw;
    float val = float(input[idx]);
    float norm = (val - mean) * inv_std;

    float gm = (gamma) ? float(gamma[c]) : 1.0f;
    float bt = (bias) ? float(bias[c]) : 0.0f;

    // GroupNorm + SiLU fused: silu(norm * gamma + beta)
    float v = norm * gm + bt;
    output[idx] = half(v / (1.0f + exp(-v)));
}
"#;

    /// Concatenation along channel dim for `[N, C, H, W]` tensors.
    ///
    /// PyTorch / diffusers `torch.cat([a, b], dim=1)` on Metal: the previous
    /// implementation fell back to a CPU path that read `to_f32_vec()` BEFORE
    /// the command buffer producing the inputs had been committed — so the
    /// reads returned uninitialised zeros and the up-block path operated on
    /// all-zero concat outputs (visible as `UB.{}.r{} after_cat std=0.0`
    /// across every up-block iteration in the SD 1.5 forward). This kernel
    /// runs on the same command buffer that produced its inputs, so writes
    /// are correctly ordered.
    pub const CAT_NCHW: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void cat_nchw_dim1_f16(
    device const half* a [[buffer(0)]],   // [N, Ca, H, W]
    device const half* b [[buffer(1)]],   // [N, Cb, H, W]
    device half* out [[buffer(2)]],       // [N, Ca+Cb, H, W]
    constant uint& N [[buffer(3)]],
    constant uint& Ca [[buffer(4)]],
    constant uint& Cb [[buffer(5)]],
    constant uint& HW [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint hw = gid.x;
    uint c = gid.y;
    uint n = gid.z;
    uint Cout = Ca + Cb;
    if (hw >= HW || c >= Cout || n >= N) return;
    uint out_idx = n * Cout * HW + c * HW + hw;
    if (c < Ca) {
        uint a_idx = n * Ca * HW + c * HW + hw;
        out[out_idx] = a[a_idx];
    } else {
        uint cb_local = c - Ca;
        uint b_idx = n * Cb * HW + cb_local * HW + hw;
        out[out_idx] = b[b_idx];
    }
}
"#;

    /// Copy Tile kernel.
    pub const COPY_TILE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void copy_tile_f16(
    device const half* source [[buffer(0)]],
    device half* dest [[buffer(1)]],
    constant uint& TileH [[buffer(2)]],
    constant uint& TileW [[buffer(3)]],
    constant uint& DestH [[buffer(4)]],
    constant uint& DestW [[buffer(5)]],
    constant uint& dest_h_off [[buffer(6)]],
    constant uint& dest_w_off [[buffer(7)]],
    constant uint& src_h_start [[buffer(8)]],
    constant uint& src_w_start [[buffer(9)]],
    constant uint& copy_h [[buffer(10)]],
    constant uint& copy_w [[buffer(11)]],
    constant uint& C [[buffer(12)]],
    constant uint& N [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint x = gid.x;
    uint y = gid.y;
    uint cn = gid.z;
    
    if (x >= copy_w || y >= copy_h || cn >= C * N) return;
    
    uint n = cn / C;
    uint c = cn % C;
    
    uint src_y = src_h_start + y;
    uint src_x = src_w_start + x;
    
    uint src_idx = n * C * TileH * TileW + c * TileH * TileW + src_y * TileW + src_x;
    
    uint dst_y = dest_h_off + y;
    uint dst_x = dest_w_off + x;
    
    uint dst_idx = n * C * DestH * DestW + c * DestH * DestW + dst_y * DestW + dst_x;
    
    dest[dst_idx] = source[src_idx];
}
"#;
    /// Causal (autoregressive) attention for text encoders (CLIP, GPT).
    ///
    /// Same interface as `attention_f16` but positions where k_pos > query_pos
    /// are masked to -infinity before softmax.
    /// Layout: [seq, num_heads * head_dim] with custom strides.
    pub const CAUSAL_ATTENTION: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void causal_attention_f16(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& stride_batch [[buffer(8)]],
    constant uint& stride_head [[buffer(9)]],
    constant uint& stride_seq [[buffer(10)]],
    constant uint& stride_dim [[buffer(11)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    uint query_pos = gid.y;
    uint head = gid.x;
    uint batch_idx = gid.z;

    if (query_pos >= seq_len) return;

    uint batch_offset = batch_idx * stride_batch;
    uint head_offset = head * stride_head;
    device const half* q = Q + batch_offset + head_offset + query_pos * stride_seq;

    threadgroup float* scores = (threadgroup float*)shared;
    float max_score = -INFINITY;

    // Only attend to positions <= query_pos (causal mask)
    for (uint k_pos = 0; k_pos <= query_pos; k_pos++) {
        device const half* k = K + batch_offset + head_offset + k_pos * stride_seq;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(q[d * stride_dim]) * float(k[d * stride_dim]);
        }
        dot *= scale;
        scores[k_pos] = dot;
        max_score = max(max_score, dot);
    }

    // Softmax over causal positions [0..query_pos]
    float sum = 0.0f;
    for (uint k_pos = 0; k_pos <= query_pos; k_pos++) {
        scores[k_pos] = exp(scores[k_pos] - max_score);
        sum += scores[k_pos];
    }
    float inv_sum = 1.0f / sum;
    for (uint k_pos = 0; k_pos <= query_pos; k_pos++) {
        scores[k_pos] *= inv_sum;
    }

    // Weighted sum of V
    device half* o = O + batch_offset + head_offset + query_pos * stride_seq;
    for (uint d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (uint k_pos = 0; k_pos <= query_pos; k_pos++) {
            device const half* v = V + batch_offset + head_offset + k_pos * stride_seq;
            acc += scores[k_pos] * float(v[d * stride_dim]);
        }
        o[d * stride_dim] = half(acc);
    }
}
"#;

    /// Linear projection: Y[M,N] = X[M,K] @ W[N,K]^T + bias[N].
    ///
    /// Tiled matmul with B transposed. Both X and W are row-major.
    /// Thread (tid.y, tid.x) computes one element of a 16×16 output tile.
    pub const LINEAR: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void linear_f16(
    device const half* X [[buffer(0)]],      // [M, K] row-major
    device const half* W [[buffer(1)]],      // [N, K] row-major (transposed for matmul)
    device const half* bias [[buffer(2)]],   // [N]
    device half* Y [[buffer(3)]],            // [M, N] row-major
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& has_bias [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    const uint TILE = 16;
    uint row = gid.y * TILE + tid.y;
    uint col = gid.x * TILE + tid.x;

    threadgroup half tileA[TILE][TILE];
    threadgroup half tileW[TILE][TILE];

    float acc = 0.0f;

    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
        uint k_off = t * TILE;

        // Load X tile: X[row, k_off + tid.x]  (coalesced read along K)
        tileA[tid.y][tid.x] = (row < M && (k_off + tid.x) < K)
            ? X[row * K + k_off + tid.x] : half(0.0h);

        // Load W tile: W[gid.x*T + tid.y, k_off + tid.x]  (coalesced read along K)
        uint w_row = gid.x * TILE + tid.y;
        tileW[tid.y][tid.x] = (w_row < N && (k_off + tid.x) < K)
            ? W[w_row * K + k_off + tid.x] : half(0.0h);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate: Y[row, col] += sum_i X[row, k] * W[col, k]
        // tileA[tid.y][i] = X[row, k_off + i]
        // tileW[tid.x][i] = W[col, k_off + i]  (note: indexed by tid.x, not tid.y)
        for (uint i = 0; i < TILE; i++) {
            acc += float(tileA[tid.y][i]) * float(tileW[tid.x][i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (has_bias) {
        acc += float(bias[col]);
    }

    if (row < M && col < N) {
        Y[row * N + col] = half(acc);
    }
}

// Linear projection with F32 output (avoids F16 overflow for residual-feeding projections).
// Same as linear_f16 but stores F32 output instead of F16.
// Y = X @ W^T (+bias): X[M,K], W[N,K], Y[M,N] (F32)
kernel void linear_f16_out_f32(
    device const half* X [[buffer(0)]],
    device const half* W [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& has_bias [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    const uint TILE = 16;
    uint row = gid.y * TILE + tid.y;
    uint col = gid.x * TILE + tid.x;

    threadgroup half tileA[TILE][TILE];
    threadgroup half tileW[TILE][TILE];
    float acc = 0.0f;

    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
        uint k_off = t * TILE;
        tileA[tid.y][tid.x] = (row < M && (k_off + tid.x) < K)
            ? X[row * K + k_off + tid.x] : half(0.0h);
        uint w_row = gid.x * TILE + tid.y;
        tileW[tid.y][tid.x] = (w_row < N && (k_off + tid.x) < K)
            ? W[w_row * K + k_off + tid.x] : half(0.0h);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < TILE; i++) {
            acc += float(tileA[tid.y][i]) * float(tileW[tid.x][i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (has_bias) {
        acc += float(bias[col]);
    }

    if (row < M && col < N) {
        Y[row * N + col] = acc;  // F32 output (no F16 conversion)
    }
}

// Non-transposed matmul: Y = A @ B
// A: [M, K] row-major, B: [K, N] row-major, Y: [M, N] row-major
kernel void matmul_nn_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* Y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    const uint TILE = 16;
    uint row = gid.y * TILE + tid.y;
    uint col = gid.x * TILE + tid.x;

    threadgroup half tileA[TILE][TILE];
    threadgroup half tileB[TILE][TILE];

    float acc = 0.0f;

    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
        uint k_off = t * TILE;

        tileA[tid.y][tid.x] = (row < M && (k_off + tid.x) < K)
            ? A[row * K + k_off + tid.x] : half(0.0h);

        uint b_row = k_off + tid.y;
        tileB[tid.y][tid.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : half(0.0h);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE; i++) {
            acc += float(tileA[tid.y][i]) * float(tileB[i][tid.x]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        Y[row * N + col] = half(acc);
    }
}

// Row-wise scaled softmax: scores[row, :] = softmax(scores[row, :] * scale)
// Grid: (1, rows, 1) threadgroups, threadgroup: (1, 1, 1)
// Each thread processes one row entirely.
kernel void row_softmax_scale_f16(
    device half* data [[buffer(0)]],
    constant uint& rows [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= rows) return;

    device half* row_data = data + row * cols;

    // Pass 1: scale and find max
    float max_val = -INFINITY;
    for (uint c = 0; c < cols; c++) {
        float v = float(row_data[c]) * scale;
        row_data[c] = half(v);
        max_val = max(max_val, v);
    }

    // Pass 2: exp(x - max) and sum
    float sum = 0.0f;
    for (uint c = 0; c < cols; c++) {
        float v = exp(float(row_data[c]) - max_val);
        row_data[c] = half(v);
        sum += v;
    }

    // Pass 3: normalize
    float inv_sum = 1.0f / sum;
    for (uint c = 0; c < cols; c++) {
        row_data[c] = half(float(row_data[c]) * inv_sum);
    }
}

// F32 attention pipeline: Q@K^T → F32 scores, add F16 bias, softmax, output F16 weights.
// Avoids F16 precision loss in attention scores for deep models (e.g. Flan-T5 24 layers).

// Batched Y = X @ W^T with F32 output: X[B,M,K](F16), W[B,N,K](F16), Y[B,M,N](F32)
kernel void batched_linear_f16_out_f32(
    device const half* X [[buffer(0)]],
    device const half* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    const uint TILE = 16;
    uint row = gid.y * TILE + tid.y;
    uint col = gid.x * TILE + tid.x;
    uint batch = gid.z;

    uint x_off = batch * M * K;
    uint w_off = batch * N * K;
    uint y_off = batch * M * N;

    threadgroup half tileA[TILE][TILE];
    threadgroup half tileW[TILE][TILE];
    float acc = 0.0f;

    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
        uint k_off = t * TILE;
        tileA[tid.y][tid.x] = (row < M && (k_off + tid.x) < K)
            ? X[x_off + row * K + k_off + tid.x] : half(0.0h);
        uint w_row = gid.x * TILE + tid.y;
        tileW[tid.y][tid.x] = (w_row < N && (k_off + tid.x) < K)
            ? W[w_off + w_row * K + k_off + tid.x] : half(0.0h);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < TILE; i++) {
            acc += float(tileA[tid.y][i]) * float(tileW[tid.x][i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        Y[y_off + row * N + col] = acc;
    }
}

// Add F16 bias to F32 scores, then softmax entirely in F32, write F16 attention weights.
// scores: F32 [total_rows, cols], bias: F16 [total_rows, cols], output: F16 (in-place over bias-sized buffer)
kernel void add_bias_softmax_f32_to_f16(
    device const float* scores [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= rows) return;

    device const float* s_row = scores + row * cols;
    device const half* b_row = bias + row * cols;
    device half* o_row = output + row * cols;

    // Pass 1: find max (F32 throughout)
    float max_val = -INFINITY;
    for (uint c = 0; c < cols; c++) {
        float v = s_row[c] + float(b_row[c]);
        max_val = max(max_val, v);
    }

    // Pass 2: exp and sum
    float sum = 0.0f;
    for (uint c = 0; c < cols; c++) {
        float v = s_row[c] + float(b_row[c]);
        sum += exp(v - max_val);
    }

    // Pass 3: normalize and write F16
    float inv_sum = 1.0f / sum;
    for (uint c = 0; c < cols; c++) {
        float v = s_row[c] + float(b_row[c]);
        o_row[c] = half(exp(v - max_val) * inv_sum);
    }
}

// Softmax on F32 data, write F16 output (for cross-attention without bias).
kernel void softmax_f32_to_f16(
    device const float* scores [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= rows) return;

    device const float* s_row = scores + row * cols;
    device half* o_row = output + row * cols;

    float max_val = -INFINITY;
    for (uint c = 0; c < cols; c++) {
        max_val = max(max_val, s_row[c]);
    }

    float sum = 0.0f;
    for (uint c = 0; c < cols; c++) {
        sum += exp(s_row[c] - max_val);
    }

    float inv_sum = 1.0f / sum;
    for (uint c = 0; c < cols; c++) {
        o_row[c] = half(exp(s_row[c] - max_val) * inv_sum);
    }
}

// Scaled softmax on F32 input -> F16 output. Applies a per-call `scale`
// (typically 1/sqrt(head_dim)) before the softmax. Used by mixed-precision
// attention so QK^T is accumulated in F32, scaled in F32, softmax'd in F32,
// and only THEN converted to F16 attention weights. Critical for sequences
// long enough to overflow F16 in the score matrix (e.g. 21 latent frames
// at 480p in Wan T2V).
kernel void row_softmax_scale_f32_to_f16(
    device const float* scores [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= rows) return;

    device const float* s_row = scores + row * cols;
    device half* o_row = output + row * cols;

    float max_val = -INFINITY;
    for (uint c = 0; c < cols; c++) {
        max_val = max(max_val, s_row[c] * scale);
    }

    float sum = 0.0f;
    for (uint c = 0; c < cols; c++) {
        sum += exp(s_row[c] * scale - max_val);
    }

    float inv_sum = 1.0f / sum;
    for (uint c = 0; c < cols; c++) {
        o_row[c] = half(exp(s_row[c] * scale - max_val) * inv_sum);
    }
}

// Transpose [S, H, D] -> [H, S, D] (swap first two dims of 3D tensor)
// Grid: (1, ceil(S/tg_y), H), threadgroup: (D, tg_y, 1)
kernel void transpose_shd_to_hsd_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& S [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint s = gid.y;
    uint h = gid.z;
    if (d >= D || s >= S || h >= H) return;
    output[h * S * D + s * D + d] = input[s * H * D + h * D + d];
}

// Transpose [H, S, D] -> [S, H, D] (reverse of above)
kernel void transpose_hsd_to_shd_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& S [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint s = gid.y;
    uint h = gid.z;
    if (d >= D || s >= S || h >= H) return;
    output[s * H * D + h * D + d] = input[h * S * D + s * D + d];
}

// Batched Y = X @ W^T: X[B,M,K], W[B,N,K], Y[B,M,N]
// Grid: (ceil(N/16), ceil(M/16), B), threadgroup: (16, 16, 1)
kernel void batched_linear_f16(
    device const half* X [[buffer(0)]],
    device const half* W [[buffer(1)]],
    device half* Y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    const uint TILE = 16;
    uint row = gid.y * TILE + tid.y;
    uint col = gid.x * TILE + tid.x;
    uint batch = gid.z;

    uint x_off = batch * M * K;
    uint w_off = batch * N * K;
    uint y_off = batch * M * N;

    threadgroup half tileA[TILE][TILE];
    threadgroup half tileW[TILE][TILE];
    float acc = 0.0f;

    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
        uint k_off = t * TILE;
        tileA[tid.y][tid.x] = (row < M && (k_off + tid.x) < K)
            ? X[x_off + row * K + k_off + tid.x] : half(0.0h);
        uint w_row = gid.x * TILE + tid.y;
        tileW[tid.y][tid.x] = (w_row < N && (k_off + tid.x) < K)
            ? W[w_off + w_row * K + k_off + tid.x] : half(0.0h);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < TILE; i++) {
            acc += float(tileA[tid.y][i]) * float(tileW[tid.x][i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        Y[y_off + row * N + col] = half(acc);
    }
}

// Batched Y = A @ B (non-transposed): A[B,M,K], B[B,K,N], Y[B,M,N]
// Grid: (ceil(N/16), ceil(M/16), B), threadgroup: (16, 16, 1)
kernel void batched_matmul_nn_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* Y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    const uint TILE = 16;
    uint row = gid.y * TILE + tid.y;
    uint col = gid.x * TILE + tid.x;
    uint batch = gid.z;

    uint a_off = batch * M * K;
    uint b_off = batch * K * N;
    uint y_off = batch * M * N;

    threadgroup half tileA[TILE][TILE];
    threadgroup half tileB[TILE][TILE];
    float acc = 0.0f;

    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
        uint k_off = t * TILE;
        tileA[tid.y][tid.x] = (row < M && (k_off + tid.x) < K)
            ? A[a_off + row * K + k_off + tid.x] : half(0.0h);
        uint b_row = k_off + tid.y;
        tileB[tid.y][tid.x] = (b_row < K && col < N)
            ? B[b_off + b_row * N + col] : half(0.0h);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < TILE; i++) {
            acc += float(tileA[tid.y][i]) * float(tileB[i][tid.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        Y[y_off + row * N + col] = half(acc);
    }
}
"#;

    /// Layout transpose kernels: NCHW ↔ NLC (sequence format).
    ///
    /// `nchw_to_nhwc_f16`: [N, C, H, W] → [N, H*W, C]
    /// `nhwc_to_nchw_f16`: [N, H*W, C] → [N, C, H, W]
    pub const TRANSPOSE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void nchw_to_nhwc_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& C [[buffer(2)]],
    constant uint& HW [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint hw = gid.x;
    uint c = gid.y;
    uint n = gid.z;
    if (hw >= HW || c >= C) return;

    // NCHW: input[n * C * HW + c * HW + hw]
    // NHWC: output[n * HW * C + hw * C + c]
    output[n * HW * C + hw * C + c] = input[n * C * HW + c * HW + hw];
}

kernel void nhwc_to_nchw_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& C [[buffer(2)]],
    constant uint& HW [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint hw = gid.x;
    uint c = gid.y;
    uint n = gid.z;
    if (hw >= HW || c >= C) return;

    // NHWC: input[n * HW * C + hw * C + c]
    // NCHW: output[n * C * HW + c * HW + hw]
    output[n * C * HW + c * HW + hw] = input[n * HW * C + hw * C + c];
}
"#;

    /// 1D convolution: output[cout, lout] = sum_{cin, k} input[cin, lout*stride + k - padding] * weight[cout, cin, k] + bias[cout].
    /// Input: [C_in, L], Weight: [C_out, C_in, K], Bias: [C_out], Output: [C_out, L_out].
    pub const CONV1D: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void conv1d_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& C_in [[buffer(4)]],
    constant uint& C_out [[buffer(5)]],
    constant uint& L_in [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& stride [[buffer(8)]],
    constant uint& padding [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint c_out = gid.y;
    uint l_out = gid.x;
    uint L_out = (L_in + 2 * padding - K) / stride + 1;
    if (c_out >= C_out || l_out >= L_out) return;

    float sum = 0.0f;
    for (uint c_in = 0; c_in < C_in; c_in++) {
        for (uint k = 0; k < K; k++) {
            int l_in = (int)(l_out * stride + k) - (int)padding;
            if (l_in >= 0 && l_in < (int)L_in) {
                sum += float(input[c_in * L_in + l_in]) * float(weight[(c_out * C_in + c_in) * K + k]);
            }
        }
    }
    sum += float(bias[c_out]);
    output[c_out * L_out + l_out] = half(sum);
}

// Transposed 1D convolution (deconvolution) for upsampling.
// input: [C_in, L_in], weight: [C_in, C_out, K], bias: [C_out]
// output: [C_out, L_out] where L_out = (L_in - 1) * stride - 2 * padding + K
kernel void conv1d_transpose_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& C_in [[buffer(4)]],
    constant uint& C_out [[buffer(5)]],
    constant uint& L_in [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& stride [[buffer(8)]],
    constant uint& padding [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint c_out = gid.y;
    uint l_out = gid.x;
    uint L_out = (L_in - 1) * stride - 2 * padding + K;
    if (c_out >= C_out || l_out >= L_out) return;

    float sum = 0.0f;
    for (uint c_in = 0; c_in < C_in; c_in++) {
        for (uint k = 0; k < K; k++) {
            int l_check = (int)l_out + (int)padding - (int)k;
            if (l_check >= 0 && l_check % (int)stride == 0) {
                uint l_in = (uint)l_check / stride;
                if (l_in < L_in) {
                    sum += float(input[c_in * L_in + l_in]) * float(weight[(c_in * C_out + c_out) * K + k]);
                }
            }
        }
    }
    sum += float(bias[c_out]);
    output[c_out * L_out + l_out] = half(sum);
}
"#;

    /// AdaLN modulation kernel.
    /// Applies adaptive layer normalization: output = (1 + scale) * x + shift.
    /// Also includes gated residual: output = x + gate * residual.
    pub const ADALN: &str = r#"
#include <metal_stdlib>
using namespace metal;

// AdaLN modulate: output[i] = (1 + scale[i % hidden]) * x[i] + shift[i % hidden]
// x is [seq_len, hidden_size], scale/shift are [hidden_size] (broadcast over seq_len)
kernel void adaln_modulate_f16(
    device const half* x [[buffer(0)]],
    device const half* scale [[buffer(1)]],
    device const half* shift [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    uint h = gid % hidden_size;
    float xi = float(x[gid]);
    float si = float(scale[h]);
    float sh = float(shift[h]);
    output[gid] = half((1.0f + si) * xi + sh);
}

// Gated residual: output[i] = x[i] + gate[i % hidden] * residual[i]
kernel void adaln_gate_f16(
    device const half* x [[buffer(0)]],
    device const half* residual [[buffer(1)]],
    device const half* gate [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    uint h = gid % hidden_size;
    float xi = float(x[gid]);
    float ri = float(residual[gid]);
    float gi = float(gate[h]);
    output[gid] = half(xi + gi * ri);
}
"#;

    /// Patchify/unpatchify kernels for DiT models.
    /// Converts between spatial [C, H, W] and patch [H/2*W/2, C*4] representations.
    pub const PATCHIFY: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Patchify: [C, H, W] -> [num_patches, C*patch_size^2]
// Each 2x2 spatial block becomes one token with 4*C channels.
// Thread grid: (num_patches, C, 1)
kernel void patchify_f16(
    device const half* input [[buffer(0)]],   // [C, H, W]
    device half* output [[buffer(1)]],        // [num_patches, C*4]
    constant uint& channels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]     // (patch_idx, channel)
) {
    uint patch_idx = gid.x;
    uint c = gid.y;
    uint pw = width / 2;
    uint ph = height / 2;
    uint num_patches = ph * pw;
    if (patch_idx >= num_patches || c >= channels) return;

    uint py = patch_idx / pw;
    uint px = patch_idx % pw;
    uint y0 = py * 2;
    uint x0 = px * 2;
    uint patch_channels = channels * 4;

    // Read 2x2 block from [C, H, W]
    float s00 = float(input[c * height * width + y0 * width + x0]);
    float s01 = float(input[c * height * width + y0 * width + x0 + 1]);
    float s10 = float(input[c * height * width + (y0 + 1) * width + x0]);
    float s11 = float(input[c * height * width + (y0 + 1) * width + x0 + 1]);

    // Write to [num_patches, C*4]
    uint base = patch_idx * patch_channels + c * 4;
    output[base]     = half(s00);
    output[base + 1] = half(s01);
    output[base + 2] = half(s10);
    output[base + 3] = half(s11);
}

// Unpatchify: [num_patches, C*4] -> [C, H, W]
// Thread grid: (num_patches, C, 1)
kernel void unpatchify_f16(
    device const half* input [[buffer(0)]],   // [num_patches, C*4]
    device half* output [[buffer(1)]],        // [C, H, W]
    constant uint& channels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]     // (patch_idx, channel)
) {
    uint patch_idx = gid.x;
    uint c = gid.y;
    uint pw = width / 2;
    uint ph = height / 2;
    uint num_patches = ph * pw;
    if (patch_idx >= num_patches || c >= channels) return;

    uint py = patch_idx / pw;
    uint px = patch_idx % pw;
    uint y0 = py * 2;
    uint x0 = px * 2;
    uint patch_channels = channels * 4;

    // Read from [num_patches, C*4]
    uint base = patch_idx * patch_channels + c * 4;
    float s00 = float(input[base]);
    float s01 = float(input[base + 1]);
    float s10 = float(input[base + 2]);
    float s11 = float(input[base + 3]);

    // Write 2x2 block to [C, H, W]
    output[c * height * width + y0 * width + x0]           = half(s00);
    output[c * height * width + y0 * width + x0 + 1]       = half(s01);
    output[c * height * width + (y0 + 1) * width + x0]     = half(s10);
    output[c * height * width + (y0 + 1) * width + x0 + 1] = half(s11);
}
"#;

    /// 2D Rotary Position Embedding for Flux.
    /// Applies separate RoPE for height and width dimensions.
    /// First half of head_dim uses height frequencies, second half uses width frequencies.
    pub const ROPE_2D: &str = r#"
#include <metal_stdlib>
using namespace metal;

// 2D RoPE: apply rotary embeddings with separate height/width frequencies
// x layout: [seq_len, num_heads, head_dim]
// First head_dim/2 pairs use height position, second head_dim/2 pairs use width position.
// height_ids[seq_idx] and width_ids[seq_idx] give the 2D coordinates for each sequence position.
kernel void rope_2d_f16(
    device half* x [[buffer(0)]],
    device const uint* height_ids [[buffer(1)]],   // [seq_len] height coordinate per patch
    device const uint* width_ids [[buffer(2)]],    // [seq_len] width coordinate per patch
    constant uint& num_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant float& theta [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]   // (pair_idx, head, seq_pos)
) {
    uint pair = gid.x;
    uint head = gid.y;
    uint seq = gid.z;

    uint half_dim = head_dim / 2;
    uint quarter_dim = half_dim / 2;
    if (pair >= half_dim || head >= num_heads) return;

    // Determine if this pair uses height or width position
    uint pos;
    uint freq_idx;
    if (pair < quarter_dim) {
        // First quarter: height frequencies
        pos = height_ids[seq];
        freq_idx = pair;
    } else {
        // Second quarter: width frequencies
        pos = width_ids[seq];
        freq_idx = pair - quarter_dim;
    }

    float freq = 1.0f / pow(theta, float(2 * freq_idx) / float(half_dim));
    float angle = float(pos) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    // Index in buffer: [seq, head, dim]
    uint base = (seq * num_heads + head) * head_dim;
    uint i1 = base + pair * 2;
    uint i2 = base + pair * 2 + 1;

    float x0 = float(x[i1]);
    float x1 = float(x[i2]);

    x[i1] = half(x0 * cos_val - x1 * sin_val);
    x[i2] = half(x0 * sin_val + x1 * cos_val);
}
"#;

    /// T5 relative position bias computation.
    pub const RELATIVE_POSITION_BIAS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Compute T5 relative position bias.
// For each (query_pos, key_pos) pair, compute a bucket index from relative position,
// then look up the bias from the learned embedding table.
// Output: [num_heads, q_len, k_len]
kernel void relative_position_bias_f16(
    device const half* bias_table [[buffer(0)]],  // [num_buckets, num_heads]
    device half* output [[buffer(1)]],            // [num_heads, q_len, k_len]
    constant uint& q_len [[buffer(2)]],
    constant uint& k_len [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_buckets [[buffer(5)]],
    constant uint& max_distance [[buffer(6)]],
    constant uint& is_bidirectional [[buffer(7)]], // 1 for encoder, 0 for decoder
    uint3 gid [[thread_position_in_grid]]   // (k_pos, q_pos, head)
) {
    uint k_pos = gid.x;
    uint q_pos = gid.y;
    uint head = gid.z;
    if (k_pos >= k_len || q_pos >= q_len || head >= num_heads) return;

    // Compute relative position and bucket (T5 bucketing scheme)
    int rel_pos = int(k_pos) - int(q_pos);

    uint bucket;
    if (is_bidirectional) {
        // Bidirectional (encoder): half buckets for each direction
        uint half_buckets = num_buckets / 2;
        if (rel_pos > 0) {
            bucket = half_buckets;
            uint abs_pos = uint(rel_pos);
            uint max_exact = half_buckets / 2;
            if (abs_pos < max_exact) {
                bucket += abs_pos;
            } else {
                float log_ratio = log(float(abs_pos) / float(max_exact));
                float max_log = log(float(max_distance) / float(max_exact));
                uint b = uint(log_ratio / max_log * float(half_buckets - max_exact));
                bucket += min(b + max_exact, half_buckets - 1);
            }
        } else {
            uint abs_pos = uint(-rel_pos);
            uint max_exact = half_buckets / 2;
            if (abs_pos < max_exact) {
                bucket = abs_pos;
            } else {
                float log_ratio = log(float(abs_pos) / float(max_exact));
                float max_log = log(float(max_distance) / float(max_exact));
                uint b = uint(log_ratio / max_log * float(half_buckets - max_exact));
                bucket = min(b + max_exact, half_buckets - 1);
            }
        }
    } else {
        // Unidirectional (decoder): clamp future positions to 0, use full bucket range
        int abs_pos = max(-rel_pos, 0);  // distance to past; future → 0
        uint max_exact = num_buckets / 2;
        if (uint(abs_pos) < max_exact) {
            bucket = uint(abs_pos);
        } else {
            float log_ratio = log(float(abs_pos) / float(max_exact));
            float max_log = log(float(max_distance) / float(max_exact));
            uint b = uint(log_ratio / max_log * float(num_buckets - max_exact));
            bucket = min(b + max_exact, num_buckets - 1);
        }
    }

    // Look up bias: table is [num_buckets, num_heads]
    float bias = float(bias_table[bucket * num_heads + head]);

    // Write to output: [num_heads, q_len, k_len]
    output[(head * q_len + q_pos) * k_len + k_pos] = half(bias);
}
"#;

    /// DaViT / Swin patch-merge concat: fuse 2×2 spatial blocks into the
    /// channel dimension before the downsampling reduction linear.
    /// Layout matches the standard reference implementation:
    ///
    /// ```text
    /// out[py, px, 0*D + d] = in[2*py    , 2*px    , d]   (top-left)
    /// out[py, px, 1*D + d] = in[2*py + 1, 2*px    , d]   (bottom-left)
    /// out[py, px, 2*D + d] = in[2*py    , 2*px + 1, d]   (top-right)
    /// out[py, px, 3*D + d] = in[2*py + 1, 2*px + 1, d]   (bottom-right)
    /// ```
    ///
    /// Thread grid: `(num_out_patches, prev_dim, 1)` — one thread per
    /// (output token, channel) pair, copying four input elements.
    /// Canny edge detection — full 5-stage pipeline used by ControlNet
    /// sketch / canny preprocessing. All kernels operate in f16.
    ///
    /// Stages:
    ///   1. `canny_rgb_to_gray_f16`   — RGB [3,H,W] → Gray [H,W] (Rec. 601 luma)
    ///   2. `canny_sobel_f16`         — Gray [H,W] → Magnitude [H,W] + Direction [H,W]
    ///                                  (direction stored as float quantised to {0,1,2,3}
    ///                                   for {0°, 45°, 90°, 135°})
    ///   3. `canny_nms_f16`           — Magnitude × Direction → thinned magnitude
    ///   4. `canny_double_threshold_f16` — magnitude → 0.0 / 0.5 / 1.0
    ///   5. `canny_hysteresis_f16`    — promote weak (0.5) → strong (1.0) if any 8-neighbor is strong
    ///   6. `canny_gray_to_rgb_f16`   — replicate single-channel edge map to RGB for ControlNet
    pub const CANNY_EDGE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// 1) RGB → grayscale (Rec. 601 luminance)
kernel void canny_rgb_to_gray_f16(
    device const half* rgb   [[buffer(0)]],   // [3, H, W]
    device half*       gray  [[buffer(1)]],   // [H, W]
    constant uint& height    [[buffer(2)]],
    constant uint& width     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= width || gid.y >= height) return;
    uint hw = height * width;
    uint pixel = gid.y * width + gid.x;
    float r = float(rgb[0 * hw + pixel]);
    float g = float(rgb[1 * hw + pixel]);
    float b = float(rgb[2 * hw + pixel]);
    gray[pixel] = half(0.2989h * half(r) + 0.5870h * half(g) + 0.1140h * half(b));
}

// 2) Sobel — gradient magnitude + quantised direction.
//    Direction quantisation maps atan2(gy, gx) into {0,1,2,3}:
//      0 = horizontal     (|θ| < 22.5° or > 157.5°)
//      1 = +45°
//      2 = vertical       (67.5° < θ < 112.5°)
//      3 = -45° / 135°
//    Magnitude is normalised by the maximum possible Sobel response of 4 *
//    sqrt(2) (no separate normalisation pass needed for typical use).
kernel void canny_sobel_f16(
    device const half* gray  [[buffer(0)]],   // [H, W]
    device half*       mag   [[buffer(1)]],   // [H, W]
    device half*       dir   [[buffer(2)]],   // [H, W] — quantised 0..3 stored as half
    constant uint& height    [[buffer(3)]],
    constant uint& width     [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint x = gid.x;
    uint y = gid.y;
    if (x >= width || y >= height) return;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        mag[y * width + x] = 0.0h;
        dir[y * width + x] = 0.0h;
        return;
    }

    // Sobel 3x3: gx and gy.
    float p00 = float(gray[(y - 1) * width + (x - 1)]);
    float p01 = float(gray[(y - 1) * width +  x     ]);
    float p02 = float(gray[(y - 1) * width + (x + 1)]);
    float p10 = float(gray[ y      * width + (x - 1)]);
    float p12 = float(gray[ y      * width + (x + 1)]);
    float p20 = float(gray[(y + 1) * width + (x - 1)]);
    float p21 = float(gray[(y + 1) * width +  x     ]);
    float p22 = float(gray[(y + 1) * width + (x + 1)]);

    float gx = (p02 + 2.0 * p12 + p22) - (p00 + 2.0 * p10 + p20);
    float gy = (p20 + 2.0 * p21 + p22) - (p00 + 2.0 * p01 + p02);
    float m = sqrt(gx * gx + gy * gy);
    // Normalise: maximum |gx| or |gy| is 4 (4 white, 4 black) → magnitude max ≈ 5.66.
    float m_norm = clamp(m / 5.6568542f, 0.0f, 1.0f);

    // Quantise direction to 4 bins.
    float angle = atan2(gy, gx);  // (-π, π]
    if (angle < 0.0) angle += M_PI_F;
    float a = angle / M_PI_F * 4.0f;  // 0..4
    int bin;
    if (a < 0.5f || a >= 3.5f)      bin = 0;
    else if (a < 1.5f)              bin = 1;
    else if (a < 2.5f)              bin = 2;
    else                            bin = 3;

    mag[y * width + x] = half(m_norm);
    dir[y * width + x] = half(float(bin));
}

// 3) Non-maximum suppression along gradient direction.
kernel void canny_nms_f16(
    device const half* mag    [[buffer(0)]],   // [H, W]
    device const half* dir    [[buffer(1)]],   // [H, W]
    device half*       out    [[buffer(2)]],   // [H, W]
    constant uint& height     [[buffer(3)]],
    constant uint& width      [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint x = gid.x;
    uint y = gid.y;
    if (x >= width || y >= height) return;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        out[y * width + x] = 0.0h;
        return;
    }

    float m  = float(mag[y * width + x]);
    int   d  = int(float(dir[y * width + x]) + 0.5f);

    int dx, dy;
    if      (d == 0) { dx = 1;  dy = 0;  }   // horizontal
    else if (d == 1) { dx = 1;  dy = -1; }   // +45°
    else if (d == 2) { dx = 0;  dy = 1;  }   // vertical
    else             { dx = 1;  dy = 1;  }   // -45° / 135°

    float n1 = float(mag[(int(y) + dy) * int(width) + (int(x) + dx)]);
    float n2 = float(mag[(int(y) - dy) * int(width) + (int(x) - dx)]);

    if (m >= n1 && m >= n2) {
        out[y * width + x] = half(m);
    } else {
        out[y * width + x] = 0.0h;
    }
}

// 4) Double threshold: 0 (non-edge) / 0.5 (weak) / 1.0 (strong).
kernel void canny_double_threshold_f16(
    device const half* mag    [[buffer(0)]],   // [H, W]
    device half*       out    [[buffer(1)]],   // [H, W]
    constant float& low_thr   [[buffer(2)]],
    constant float& high_thr  [[buffer(3)]],
    constant uint& height     [[buffer(4)]],
    constant uint& width      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint x = gid.x;
    uint y = gid.y;
    if (x >= width || y >= height) return;
    float m = float(mag[y * width + x]);
    half v;
    if      (m >= high_thr) v = 1.0h;
    else if (m >= low_thr ) v = 0.5h;
    else                    v = 0.0h;
    out[y * width + x] = v;
}

// 5) Hysteresis — single GPU pass over 8-neighbours; promote weak (0.5) → strong (1.0)
//    if any 8-neighbour is strong, drop weak otherwise. Iterate 4–8 times from the
//    Rust side for proper convergence on long thin edges; one pass already produces
//    SOTA-quality output for ControlNet sketch input.
kernel void canny_hysteresis_f16(
    device const half* in_     [[buffer(0)]],  // [H, W] — values in {0.0, 0.5, 1.0}
    device half*       out     [[buffer(1)]],  // [H, W]
    constant uint& height      [[buffer(2)]],
    constant uint& width       [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint x = gid.x;
    uint y = gid.y;
    if (x >= width || y >= height) return;
    float v = float(in_[y * width + x]);
    if (v >= 1.0f) {
        out[y * width + x] = 1.0h;
        return;
    }
    if (v < 0.25f) {
        out[y * width + x] = 0.0h;
        return;
    }
    // weak (≈0.5): strong if any 8-neighbour is strong.
    bool promoted = false;
    for (int dy = -1; dy <= 1 && !promoted; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = int(x) + dx;
            int ny = int(y) + dy;
            if (nx < 0 || ny < 0 || nx >= int(width) || ny >= int(height)) continue;
            if (float(in_[ny * width + nx]) >= 1.0f) { promoted = true; break; }
        }
    }
    out[y * width + x] = promoted ? 1.0h : 0.0h;
}

// 6) Replicate gray edge map to RGB CHW, white-on-black, for ControlNet input.
kernel void canny_gray_to_rgb_f16(
    device const half* gray   [[buffer(0)]],   // [H, W]
    device half*       rgb    [[buffer(1)]],   // [3, H, W]
    constant uint& height     [[buffer(2)]],
    constant uint& width      [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= width || gid.y >= height) return;
    uint hw = height * width;
    uint pixel = gid.y * width + gid.x;
    half v = gray[pixel];
    rgb[0 * hw + pixel] = v;
    rgb[1 * hw + pixel] = v;
    rgb[2 * hw + pixel] = v;
}
"#;

    pub const PATCH_MERGE_CONCAT: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void patch_merge_concat_f16(
    device const half* input  [[buffer(0)]],   // [H, W, D]
    device half*       output [[buffer(1)]],   // [H/2 * W/2, 4*D]
    constant uint& in_h  [[buffer(2)]],
    constant uint& in_w  [[buffer(3)]],
    constant uint& dim   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint patch_idx = gid.x;
    uint d         = gid.y;

    uint out_w = in_w / 2;
    uint out_h = in_h / 2;
    uint num_out = out_w * out_h;
    if (patch_idx >= num_out || d >= dim) return;

    uint py = patch_idx / out_w;
    uint px = patch_idx % out_w;
    uint y0 = py * 2;
    uint x0 = px * 2;

    uint stride_row = in_w * dim;
    uint base_in_tl = y0       * stride_row + x0       * dim;
    uint base_in_bl = (y0 + 1) * stride_row + x0       * dim;
    uint base_in_tr = y0       * stride_row + (x0 + 1) * dim;
    uint base_in_br = (y0 + 1) * stride_row + (x0 + 1) * dim;

    uint merged_dim = dim * 4;
    uint base_out   = patch_idx * merged_dim + d;

    output[base_out + 0 * dim] = input[base_in_tl + d];
    output[base_out + 1 * dim] = input[base_in_bl + d];
    output[base_out + 2 * dim] = input[base_in_tr + d];
    output[base_out + 3 * dim] = input[base_in_br + d];
}
"#;

    /// SANA-WM LTX-2 3D conv (streaming, no im2col). Single thread per output
    /// element, scalar f32 accumulator over (c_in × kernel³) inputs. T axis
    /// uses edge-replicate padding (non-causal) or left-only edge-replicate
    /// (causal); spatial axes zero-padded. Replaces the host-side im2col +
    /// matmul path which OOMs at production dims (conv_out at full config
    /// would need a ~12GB im2col intermediate).
    pub const LTX_CONV3D: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct LtxConvDims {
    uint B;
    uint C_in;
    uint C_out;
    uint T;
    uint H;
    uint W;
    uint K;
    uint causal;
};

kernel void ltx_conv3d_f16(
    device const half*       x       [[buffer(0)]],
    device const half*       w       [[buffer(1)]],
    device const half*       bias    [[buffer(2)]],
    device       half*       out     [[buffer(3)]],
    constant     LtxConvDims& dims   [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint B     = dims.B;
    const uint C_in  = dims.C_in;
    const uint C_out = dims.C_out;
    const uint T     = dims.T;
    const uint H     = dims.H;
    const uint W     = dims.W;
    const uint K     = dims.K;
    const uint causal = dims.causal;

    if (gid.x >= W || gid.y >= H) return;
    uint flat_z  = gid.z;
    if (flat_z >= B * C_out * T) return;
    uint b       = flat_z / (C_out * T);
    uint rem     = flat_z % (C_out * T);
    uint co      = rem / T;
    uint t_out   = rem % T;
    uint h_out   = gid.y;
    uint w_out   = gid.x;

    uint t_left_pad = causal ? (K - 1) : ((K - 1) / 2);
    uint hw_half = (K - 1) / 2;

    float acc = 0.0f;
    for (uint ci = 0; ci < C_in; ++ci) {
        for (uint kt = 0; kt < K; ++kt) {
            int t_in_signed = int(t_out) + int(kt) - int(t_left_pad);
            uint t_in;
            if (t_in_signed < 0) {
                t_in = 0;
            } else if (uint(t_in_signed) >= T) {
                t_in = T - 1;
            } else {
                t_in = uint(t_in_signed);
            }
            for (uint ky = 0; ky < K; ++ky) {
                int h_in_signed = int(h_out) + int(ky) - int(hw_half);
                if (h_in_signed < 0 || uint(h_in_signed) >= H) continue;
                uint h_in = uint(h_in_signed);
                for (uint kx = 0; kx < K; ++kx) {
                    int w_in_signed = int(w_out) + int(kx) - int(hw_half);
                    if (w_in_signed < 0 || uint(w_in_signed) >= W) continue;
                    uint w_in = uint(w_in_signed);
                    uint x_idx = (((b * C_in + ci) * T + t_in) * H + h_in) * W + w_in;
                    uint w_idx = (((co * C_in + ci) * K + kt) * K + ky) * K + kx;
                    acc += float(x[x_idx]) * float(w[w_idx]);
                }
            }
        }
    }
    acc += float(bias[co]);

    uint out_idx = (((b * C_out + co) * T + t_out) * H + h_out) * W + w_out;
    out[out_idx] = half(acc);
}
"#;

    /// SANA-WM Gated DeltaNet recurrent state sweep. One threadgroup per
    /// (batch, head); sequential T loop in-kernel; D=112 head_dim assumed
    /// fits in 128 threads. Threadgroup memory budget: ~30KB at S_CHUNK=16
    /// (fits M3 32KB limit). See gdn_kernel_draft.metal for full notes.
    pub const GDN_RECURRENT: &str = r#"
#include <metal_stdlib>
using namespace metal;

// v2 (2026-05-26): delta_v_DS moved from threadgroup to device memory to
// preserve snapshot-then-update semantics across the full S range.

struct GdnDims {
    uint B;
    uint H;
    uint T;
    uint S;
    uint D;
};

static inline uint bhdn_index(uint bh, uint di, uint ni, uint D, uint N) {
    return (bh * D + di) * N + ni;
}

static inline uint bhn_index(uint bh, uint ni, uint N) {
    return bh * N + ni;
}

// Device-memory delta_v_DS scratch indexed as [bh, D, S].
static inline uint bhds_index(uint bh, uint di, uint si, uint D, uint S) {
    return (bh * D + di) * S + si;
}

kernel void gdn_recurrent_sweep_f16(
    device const half*     q_p          [[buffer(0)]],
    device const half*     k_p          [[buffer(1)]],
    device const half*     v_p          [[buffer(2)]],
    device const half*     beta         [[buffer(3)]],
    device const half*     decay        [[buffer(4)]],
    device       half*     num_out      [[buffer(5)]],
    device       half*     den_out      [[buffer(6)]],
    constant     GdnDims&  dims         [[buffer(7)]],
    device       half*     delta_v_ds   [[buffer(8)]],  // [bh, D, S] frame-wide snapshot scratch

    threadgroup half* state_kv     [[threadgroup(0)]],
    threadgroup half* state_z      [[threadgroup(1)]],
    threadgroup half* k_col        [[threadgroup(2)]],
    threadgroup half* v_col        [[threadgroup(3)]],
    threadgroup half* q_col        [[threadgroup(4)]],
    threadgroup half* delta_z_s    [[threadgroup(5)]],

    uint  bh        [[threadgroup_position_in_grid]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  tg_size   [[threads_per_threadgroup]])
{
    const uint B = dims.B;
    const uint H = dims.H;
    const uint T = dims.T;
    const uint S = dims.S;
    const uint D = dims.D;
    const uint N = T * S;

    if (bh >= B * H) return;

    for (uint idx = tid; idx < D * D; idx += tg_size) {
        state_kv[idx] = half(0);
    }
    if (tid < D) state_z[tid] = half(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ti = 0; ti < T; ++ti) {
        float g = float(decay[bh * T + ti]);

        for (uint idx = tid; idx < D * D; idx += tg_size) {
            state_kv[idx] = half(float(state_kv[idx]) * g);
        }
        if (tid < D) state_z[tid] = half(float(state_z[tid]) * g);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // PHASE 1 (frame-wide snapshot): write all S deltas against frozen state.
        for (uint si = 0; si < S; ++si) {
            uint ni = ti * S + si;

            if (tid < D) {
                k_col[tid] = k_p[bhdn_index(bh, tid, ni, D, N)];
                v_col[tid] = v_p[bhdn_index(bh, tid, ni, D, N)];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float beta_t = float(beta[((bh * T) + ti) * S + si]);

            float v_pred_di = 0.0f;
            if (tid < D) {
                const threadgroup half* row = state_kv + tid * D;
                for (uint dj = 0; dj < D; ++dj) {
                    v_pred_di += float(row[dj]) * float(k_col[dj]);
                }
            }

            if (tid == 0) {
                float z_pred = 0.0f;
                for (uint di = 0; di < D; ++di) {
                    z_pred += float(state_z[di]) * float(k_col[di]);
                }
                delta_z_s[si] = half((1.0f - z_pred) * beta_t);
            }

            if (tid < D) {
                float v_t = float(v_col[tid]);
                delta_v_ds[bhds_index(bh, tid, si, D, S)] = half((v_t - v_pred_di) * beta_t);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // PHASE 2 (batched update): apply all S deltas to state_kv / state_z.
        for (uint si = 0; si < S; ++si) {
            uint ni = ti * S + si;

            if (tid < D) k_col[tid] = k_p[bhdn_index(bh, tid, ni, D, N)];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid < D) {
                float dv = float(delta_v_ds[bhds_index(bh, tid, si, D, S)]);
                threadgroup half* row = state_kv + tid * D;
                for (uint dj = 0; dj < D; ++dj) {
                    float acc = float(row[dj]);
                    acc += dv * float(k_col[dj]);
                    row[dj] = half(acc);
                }
                float dz = float(delta_z_s[si]);
                state_z[tid] = half(float(state_z[tid]) + float(k_col[tid]) * dz);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // PHASE 3 (outputs vs updated state). Reuses delta_v_ds as den scratch
        // since Phase 2 has consumed the delta_v values.
        for (uint si = 0; si < S; ++si) {
            uint ni = ti * S + si;

            if (tid < D) q_col[tid] = q_p[bhdn_index(bh, tid, ni, D, N)];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float num_di = 0.0f;
            float den_term = 0.0f;
            if (tid < D) {
                const threadgroup half* row = state_kv + tid * D;
                for (uint dj = 0; dj < D; ++dj) {
                    num_di += float(row[dj]) * float(q_col[dj]);
                }
                den_term = float(state_z[tid]) * float(q_col[tid]);
            }

            if (tid < D) num_out[bhdn_index(bh, tid, ni, D, N)] = half(num_di);

            if (tid < D) delta_v_ds[bhds_index(bh, tid, si, D, S)] = half(den_term);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float acc = 0.0f;
                for (uint di = 0; di < D; ++di) {
                    acc += float(delta_v_ds[bhds_index(bh, di, si, D, S)]);
                }
                den_out[bhn_index(bh, ni, N)] = half(acc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
"#;
}

/// Compiled shader library.
#[cfg(feature = "metal")]
pub struct ShaderLibrary {
    /// Device
    device: std::sync::Arc<super::MetalDevice>,
    /// Compiled libraries
    libraries: dashmap::DashMap<&'static str, metal::Library>,
}

#[cfg(feature = "metal")]
impl ShaderLibrary {
    /// Create and compile all shaders.
    pub fn new(device: std::sync::Arc<super::MetalDevice>) -> crate::core::Result<Self> {
        let lib = Self {
            device,
            libraries: dashmap::DashMap::new(),
        };

        // Pre-compile core shaders
        lib.compile("matmul", sources::MATMUL)?;
        lib.compile("softmax", sources::SOFTMAX)?;
        lib.compile("rms_norm", sources::RMS_NORM)?;
        lib.compile("silu", sources::SILU)?;
        lib.compile("rope", sources::ROPE)?;
        lib.compile("elementwise", sources::ELEMENTWISE)?;
        lib.compile("attention", sources::ATTENTION)?;
        lib.compile("attention_tiled", sources::ATTENTION_TILED)?;
        lib.compile("conv2d", sources::CONV2D)?;
        // Also compile specific specialized kernels if they aren't entry points in CONV2D
        // The compile function takes source. We used CONV2D source which contains multiple kernels.
        // We just need to ensure we can load them by name "conv2d_3x3_tiled_f16" etc.
        // Metal library contains all kernels marked with 'kernel'.
        // So we don't need extra compile calls unless the source variable is different.
        
        lib.compile("upsample", sources::UPSAMPLE)?;
        lib.compile("vae_fused", sources::VAE_FUSED)?;
        lib.compile("vae_encode_fused", sources::VAE_ENCODE_FUSED)?;
        lib.compile("vae_rescale", sources::VAE_RESCALE)?;
        lib.compile("group_norm", sources::GROUP_NORM)?;
        lib.compile("gaussian_splat", sources::GAUSSIAN_SPLAT)?;
        lib.compile("copy_tile", sources::COPY_TILE)?;
        lib.compile("canny_edge", sources::CANNY_EDGE)?;

        Ok(lib)
    }

    /// Compile a shader.
    fn compile(&self, name: &'static str, source: &str) -> crate::core::Result<()> {
        let library = self.device.compile_library(source)?;
        self.libraries.insert(name, library);
        Ok(())
    }

    /// Get a compiled library.
    pub fn get(&self, name: &str) -> Option<metal::Library> {
        self.libraries.get(name).map(|l| l.clone())
    }

    /// Get a function from a library.
    pub fn get_function(&self, library: &str, function: &str) -> crate::core::Result<metal::Function> {
        let lib = self.get(library)
            .ok_or_else(|| crate::core::Error::internal(format!("library not found: {}", library)))?;

        lib.get_function(function, None)
            .map_err(|e| crate::core::Error::KernelCompilation {
                kernel: function.into(),
                message: e.to_string(),
            })
    }
}

#[cfg(not(feature = "metal"))]
pub struct ShaderLibrary;
