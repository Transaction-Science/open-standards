// SANA-WM Gated DeltaNet recurrent state sweep kernel.
// Draft from agent run 2026-05-26.
//
// Integration plan:
//   1. Add this as `pub const GDN_RECURRENT: &str = r#"..."#` in shader.rs::sources
//   2. Compile via SanaWmKernels: `gdn_recurrent: compute.compile_pipeline("gdn_recurrent", sources::GDN_RECURRENT, "gdn_recurrent_sweep_f16")`
//   3. Write dispatcher `gdn_recurrent_sweep_on(&self, cb, q_p, k_p, v_p, beta, decay, batch, num_heads, t, s, d) -> (num_tensor, den_tensor)`
//   4. Threadgroup memory allocation: caller must `setThreadgroupMemoryLength:atIndex:` for buffers 0-6 (state_kv=D*D*2 bytes, state_z=D*2, k_col=D*2, v_col=D*2, q_col=D*2, delta_v_ds=D*S_CHUNK*2, delta_z_s=S_CHUNK*2)
//   5. Threadgroup grid: (B * H, 1, 1); threads per group: 128 (must be >= D)
//   6. Compile-time S_CHUNK=16 default fits M3 32KB threadgroup limit at D=112
//
// Verification: cos against cpu::gdn_forward_with_linear at multiple (T, S, D)
// configurations; expected ULP drift only (float accumulators inside f16 state).

#include <metal_stdlib>
using namespace metal;

#ifndef MAX_D
#define MAX_D 112
#endif

#ifndef S_CHUNK
#define S_CHUNK 16
#endif

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

kernel void gdn_recurrent_sweep_f16(
    device const half*     q_p     [[buffer(0)]],
    device const half*     k_p     [[buffer(1)]],
    device const half*     v_p     [[buffer(2)]],
    device const half*     beta    [[buffer(3)]],
    device const half*     decay   [[buffer(4)]],
    device       half*     num_out [[buffer(5)]],
    device       half*     den_out [[buffer(6)]],
    constant     GdnDims&  dims    [[buffer(7)]],

    threadgroup half* state_kv     [[threadgroup(0)]], // [D, D]
    threadgroup half* state_z      [[threadgroup(1)]], // [D]
    threadgroup half* k_col        [[threadgroup(2)]], // [D]
    threadgroup half* v_col        [[threadgroup(3)]], // [D]
    threadgroup half* q_col        [[threadgroup(4)]], // [D]
    threadgroup half* delta_v_ds   [[threadgroup(5)]], // [D, S_CHUNK]
    threadgroup half* delta_z_s    [[threadgroup(6)]], // [S_CHUNK]

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

    if (bh >= B * H) {
        return;
    }

    // Zero state.
    for (uint idx = tid; idx < D * D; idx += tg_size) {
        state_kv[idx] = half(0);
    }
    if (tid < D) {
        state_z[tid] = half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ti = 0; ti < T; ++ti) {
        float g = float(decay[bh * T + ti]);

        for (uint idx = tid; idx < D * D; idx += tg_size) {
            state_kv[idx] = half(float(state_kv[idx]) * g);
        }
        if (tid < D) {
            state_z[tid] = half(float(state_z[tid]) * g);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s_base = 0; s_base < S; s_base += S_CHUNK) {
            uint s_lo = s_base;
            uint s_hi = min(s_base + (uint)S_CHUNK, S);

            // Phase 1: snapshot pass (compute deltas against post-decay state).
            for (uint si = s_lo; si < s_hi; ++si) {
                uint ni = ti * S + si;
                uint sj = si - s_lo;

                if (tid < D) {
                    k_col[tid] = k_p[bhdn_index(bh, tid, ni, D, N)];
                    v_col[tid] = v_p[bhdn_index(bh, tid, ni, D, N)];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float beta_t = float(beta[((bh * T) + ti) * S + si]);

                float v_pred_di = 0.0f;
                float z_term_di = 0.0f;
                if (tid < D) {
                    const threadgroup half* row = state_kv + tid * D;
                    for (uint dj = 0; dj < D; ++dj) {
                        v_pred_di += float(row[dj]) * float(k_col[dj]);
                    }
                    z_term_di = float(state_z[tid]) * float(k_col[tid]);
                }

                if (tid < D) {
                    delta_v_ds[tid] = half(z_term_di);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Tree reduce delta_v_ds[0 .. D] -> delta_v_ds[0] (z_pred).
                for (uint stride = 1; stride < D; stride <<= 1) {
                    if (tid < D && (tid % (stride << 1)) == 0) {
                        uint partner = tid + stride;
                        float a = float(delta_v_ds[tid]);
                        float b = (partner < D) ? float(delta_v_ds[partner]) : 0.0f;
                        delta_v_ds[tid] = half(a + b);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                float z_pred = float(delta_v_ds[0]);

                if (tid < D) {
                    float v_t = float(v_col[tid]);
                    delta_v_ds[tid * S_CHUNK + sj] = half((v_t - v_pred_di) * beta_t);
                }
                if (tid == 0) {
                    delta_z_s[sj] = half((1.0f - z_pred) * beta_t);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Phase 2: batched state update.
            for (uint si = s_lo; si < s_hi; ++si) {
                uint ni = ti * S + si;
                uint sj = si - s_lo;

                if (tid < D) {
                    k_col[tid] = k_p[bhdn_index(bh, tid, ni, D, N)];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tid < D) {
                    float dv = float(delta_v_ds[tid * S_CHUNK + sj]);
                    threadgroup half* row = state_kv + tid * D;
                    for (uint dj = 0; dj < D; ++dj) {
                        float acc = float(row[dj]);
                        acc += dv * float(k_col[dj]);
                        row[dj] = half(acc);
                    }
                    float dz = float(delta_z_s[sj]);
                    state_z[tid] = half(float(state_z[tid]) + float(k_col[tid]) * dz);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Phase 3: outputs against updated state.
            for (uint si = s_lo; si < s_hi; ++si) {
                uint ni = ti * S + si;

                if (tid < D) {
                    q_col[tid] = q_p[bhdn_index(bh, tid, ni, D, N)];
                }
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

                if (tid < D) {
                    num_out[bhdn_index(bh, tid, ni, D, N)] = half(num_di);
                }

                uint sj = si - s_lo;
                if (tid < D) {
                    delta_v_ds[tid * S_CHUNK + sj] = half(den_term);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tid == 0) {
                    float acc = 0.0f;
                    for (uint di = 0; di < D; ++di) {
                        acc += float(delta_v_ds[di * S_CHUNK + sj]);
                    }
                    den_out[bhn_index(bh, ni, N)] = half(acc);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}
