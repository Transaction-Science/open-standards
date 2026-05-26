//! GGML-based compute graph for Metal inference.
//!
//! Ports the llama.cpp Metal backend pattern to Rust:
//! - Build a static compute graph with pre-allocated buffers
//! - Encode all ops onto a single Metal command buffer
//! - Commit once, wait once
//! - Uses ggml Metal shader kernels (ggml-metal.metal) for bit-identical precision
//!
//! The ggml shader files are in `src/hal/metal/ggml/`:
//! - `ggml-metal.metal` — 9.8K lines, 103 kernels
//! - `ggml-common.h` — quantization type definitions
//! - `ggml-metal-impl.h` — kernel argument structs
//!
//! The Rust side defines matching `#[repr(C)]` argument structs and
//! encodes kernel dispatches using the `metal` crate.

/// Kernel argument structs matching ggml-metal-impl.h.
/// Must be `#[repr(C)]` with exact field order/types for Metal buffer binding.
pub mod kargs {
    /// Arguments for norm/rms_norm kernel (matches ggml_metal_kargs_norm exactly).
    /// Supports fused norm+mul+add with up to 3 chained ops.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct Norm {
        pub ne00: i32,
        pub ne00_t: i32, // ne00/4 if divisible, else ne00
        pub nb1: u64,
        pub nb2: u64,
        pub nb3: u64,
        pub eps: f32,
        pub nef1: [i32; 3], // ne01 for each fused op (norm, mul, add)
        pub nef2: [i32; 3], // ne02
        pub nef3: [i32; 3], // ne03
        pub nbf1: [u64; 3], // nb01
        pub nbf2: [u64; 3], // nb02
        pub nbf3: [u64; 3], // nb03
    }

    /// Arguments for binary ops (add, mul, sub, div).
    /// Must exactly match ggml_metal_kargs_bin from ggml-metal-impl.h.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct Bin {
        // src0 shape (dst uses same indexing)
        pub ne00: i32,
        pub ne01: i32,
        pub ne02: i32,
        pub ne03: i32,
        pub nb00: u64,
        pub nb01: u64,
        pub nb02: u64,
        pub nb03: u64,
        // src1 shape (broadcast via modular indexing)
        pub ne10: i32,
        pub ne11: i32,
        pub ne12: i32,
        pub ne13: i32,
        pub nb10: u64,
        pub nb11: u64,
        pub nb12: u64,
        pub nb13: u64,
        // dst shape
        pub ne0: i32,
        pub ne1: i32,
        pub ne2: i32,
        pub ne3: i32,
        pub nb0: u64,
        pub nb1: u64,
        pub nb2: u64,
        pub nb3: u64,
        // extra
        pub offs: u64,
        pub o1: [u64; 8],
    }

    /// Arguments for matrix-vector multiply (mul_mv).
    /// Must exactly match ggml_metal_kargs_mul_mv from ggml-metal-impl.h.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct MulMv {
        pub ne00: i32,
        pub ne01: i32,
        pub ne02: i32,
        pub nb00: u64,
        pub nb01: u64,
        pub nb02: u64,
        pub nb03: u64,
        pub ne10: i32,
        pub ne11: i32,
        pub ne12: i32,
        pub nb10: u64,
        pub nb11: u64,
        pub nb12: u64,
        pub nb13: u64,
        pub ne0: i32,
        pub ne1: i32,
        pub nr0: i32,
        pub r2: i16,
        pub r3: i16,
    }

    /// Arguments for matrix-vector multiply with expert ID (mul_mv_id).
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct MulMvId {
        pub nei0: i32,  // n_expert_used
        pub nei1: i32,  // n_tokens
        pub nbi1: u64,
        pub ne00: i32,
        pub ne01: i32,
        pub ne02: i32,
        pub nb00: u64,
        pub nb01: u64,
        pub nb02: u64,
        pub ne10: i32,
        pub ne11: i32,
        pub ne12: i32,
        pub ne13: i32,
        pub nb10: u64,
        pub nb11: u64,
        pub nb12: u64,
        pub ne0: i32,
        pub ne1: i32,
        pub nb1: u64,
        pub nr0: i32,
    }

    /// Arguments for SSM scan (Mamba-2). Exact match of ggml_metal_kargs_ssm_scan.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct SsmScan {
        pub d_state: i64,
        pub d_inner: i64,      // head_dim (NOT full d_inner)
        pub n_head: i64,
        pub n_group: i64,
        pub n_seq_tokens: i64,
        pub n_seqs: i64,
        pub s_off: u64,        // state output offset in dst
        pub nb00: u64,         // src0 (state) strides
        pub nb01: u64,
        pub nb02: u64,
        pub nb03: u64,
        pub nb10: u64,         // src1 (x) strides
        pub nb11: u64,
        pub nb12: u64,
        pub ns12: u64,         // nb12/nb10
        pub nb13: u64,
        pub nb20: u64,         // src2 (dt) strides
        pub nb21: u64,
        pub ns21: u64,         // nb21/nb20
        pub nb22: u64,
        pub ne30: i64,         // src3 (A) shape
        pub nb31: u64,
        pub nb41: u64,         // src4 (B) strides
        pub nb42: u64,
        pub ns42: u64,         // nb42/nb40
        pub nb43: u64,
        pub nb51: u64,         // src5 (C) strides
        pub nb52: u64,
        pub ns52: u64,         // nb52/nb50
        pub nb53: u64,
        pub nb0: u64,          // dst stride
    }

    /// Arguments for SSM conv1d. Exact match of ggml_metal_kargs_ssm_conv.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct SsmConv {
        pub ne00: i64,       // src0 (conv_x window) inner dim = d_conv-1+n_t
        pub ne01: i64,       // conv_dim (channels)
        pub ne02: i64,       // n_seqs
        pub nb00: u64,
        pub nb01: u64,
        pub nb02: u64,
        pub ne10: i64,       // src1 (conv weight) inner dim = d_conv
        pub ne11: i64,       // conv_dim
        pub nb10: u64,
        pub nb11: u64,
        pub ne0: i64,        // dst inner dim = conv_dim
        pub ne1: i64,        // n_seq_tokens
        pub ne2: i64,        // n_seqs
        pub nb0: u64,
        pub nb1: u64,
        pub nb2: u64,
    }

    /// Arguments for RoPE (rotary position embedding).
    /// Arguments for RoPE kernel. EXACT match of ggml_metal_kargs_rope.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Rope {
        pub ne00: i32,
        pub ne01: i32,
        pub ne02: i32,
        pub ne03: i32,
        pub nb00: u64,
        pub nb01: u64,
        pub nb02: u64,
        pub nb03: u64,
        pub ne0: i32,
        pub ne1: i32,
        pub ne2: i32,
        pub ne3: i32,
        pub nb0: u64,
        pub nb1: u64,
        pub nb2: u64,
        pub nb3: u64,
        pub n_past: i32,
        pub n_dims: i32,
        pub n_ctx_orig: i32,
        pub freq_base: f32,
        pub freq_scale: f32,
        pub ext_factor: f32,
        pub attn_factor: f32,
        pub beta_fast: f32,
        pub beta_slow: f32,
        pub sect_0: i32,
        pub sect_1: i32,
        pub sect_2: i32,
        pub sect_3: i32,
        pub has_freq_factors: bool,
    }

    /// Arguments for F32→F16 copy kernel. Matches ggml_metal_kargs_cpy.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct Cpy {
        pub nk0: i64,
        pub ne00: i64, pub ne01: i64, pub ne02: i64, pub ne03: i64,
        pub nb00: u64, pub nb01: u64, pub nb02: u64, pub nb03: u64,
        pub ne0: i64, pub ne1: i64, pub ne2: i64, pub ne3: i64,
        pub nb0: u64, pub nb1: u64, pub nb2: u64, pub nb3: u64,
    }

    /// Arguments for flash attention. Matches ggml_metal_kargs_flash_attn_ext.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct FlashAttnExt {
        pub ne01: i32,       // num_heads (Q rows per batch)
        pub ne02: i32,       // Q batch dim 2
        pub ne03: i32,       // Q batch dim 3
        pub nb01: u64,       // Q row stride (head_dim * 4 for F32)
        pub nb02: u64,
        pub nb03: u64,
        pub ne11: i32,       // seq_len (K rows = cache length)
        pub ne_12_2: i32,    // num_kv_heads
        pub ne_12_3: i32,    // 1
        pub ns10: i32,       // K element stride (1 for contiguous)
        pub nb11: u64,       // K row stride (head_dim * 2 for F16)
        pub nb12: u64,       // K kv_head stride
        pub nb13: u64,
        pub ns20: i32,       // V element stride
        pub nb21: u64,       // V row stride
        pub nb22: u64,
        pub nb23: u64,
        pub ne31: i32,       // mask dim1
        pub ne32: i32,       // mask dim2
        pub ne33: i32,       // mask dim3
        pub nb31: u64,
        pub nb32: u64,
        pub nb33: u64,
        pub ne1: i32,        // dst dim1
        pub ne2: i32,        // dst dim2
        pub ne3: i32,        // dst dim3
        pub scale: f32,      // 1/sqrt(head_dim)
        pub max_bias: f32,   // 0 for no ALiBi
        pub m0: f32,
        pub m1: f32,
        pub n_head_log2: i32,
        pub logit_softcap: f32,
    }

    /// Arguments for matrix-matrix multiply (prefill).
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct MulMm {
        pub ne00: i32,
        pub ne02: i32,
        pub nb01: u64,
        pub nb02: u64,
        pub nb03: u64,
        pub ne11: i32,
        pub nb10: u64,
        pub nb11: u64,
        pub nb12: u64,
        pub nb13: u64,
        pub ne20: i32,
        pub ne21: i32,
        pub ne0: i32,
        pub ne1: i32,
        pub r2: u32,
        pub r3: u32,
    }
}

/// GGML Metal shader library — compiled once, provides pipeline states for all kernels.
pub struct GgmlShaderLibrary {
    /// Compiled Metal library from ggml-metal.metal
    #[cfg(feature = "metal")]
    library: metal::Library,
    /// Cached pipeline states by kernel name
    #[cfg(feature = "metal")]
    pipelines: std::collections::HashMap<String, metal::ComputePipelineState>,
}

#[cfg(feature = "metal")]
impl GgmlShaderLibrary {
    /// Compile the ggml Metal shader and build pipeline states for Nemotron kernels.
    pub fn new(device: &metal::DeviceRef) -> Result<Self> {
        // Load the pre-processed combined shader (headers inlined)
        let shader_source = include_str!("../hal/metal/ggml/ggml-metal-combined.metal");

        eprintln!("[ggml] Compiling Metal shader ({} chars)...", shader_source.len());
        let t0 = std::time::Instant::now();

        let options = metal::CompileOptions::new();
        options.set_fast_math_enabled(true);
        options.set_language_version(metal::MTLLanguageVersion::V3_1);

        let library = match device.new_library_with_source(shader_source, &options) {
            Ok(lib) => {
                eprintln!("[ggml] Shader compiled in {:.1}s", t0.elapsed().as_secs_f64());
                lib
            }
            Err(e) => {
                eprintln!("[ggml] Shader compilation FAILED: {}", e);
                return Err(crate::core::Error::KernelCompilation {
                    kernel: "ggml-metal".into(),
                    message: format!("Failed to compile ggml shader: {}", e),
                });
            }
        };

        // Build pipeline states for the kernels we need
        let kernel_names = [
            "kernel_mul_mv_q4_K_f32",
            "kernel_mul_mv_q8_0_f32",
            "kernel_mul_mv_f16_f32",
            "kernel_mul_mv_f32_f32",
            "kernel_mul_mv_id_q8_0_f32",
            "kernel_mul_mv_id_f16_f32",
            "kernel_mul_mv_id_q4_K_f32",
            "kernel_ssm_conv_f32_f32",
            "kernel_ssm_scan_f32",
            "kernel_rms_norm_f32",
            "kernel_rms_norm_mul_f32",
            "kernel_rope_neox_f16",
            "kernel_get_rows_q5_0",
            "kernel_get_rows_q4_K",
            "kernel_argsort_f32_i32_desc",
        ];

        let mut pipelines = std::collections::HashMap::new();

        // Helper: create pipeline with function constants
        // Function constant value: (index, type, value_bytes)
        enum FcVal { Short(i16), Bool(bool), Int(i32) }

        let make_pipeline = |name: &str, constants: &[(u64, FcVal)]| -> Option<metal::ComputePipelineState> {
            let cv = metal::FunctionConstantValues::new();
            for (idx, val) in constants {
                match val {
                    FcVal::Short(v) => cv.set_constant_value_at_index(
                        v as *const i16 as *const std::ffi::c_void,
                        metal::MTLDataType::Short, *idx,
                    ),
                    FcVal::Bool(v) => cv.set_constant_value_at_index(
                        v as *const bool as *const std::ffi::c_void,
                        metal::MTLDataType::Bool, *idx,
                    ),
                    FcVal::Int(v) => cv.set_constant_value_at_index(
                        v as *const i32 as *const std::ffi::c_void,
                        metal::MTLDataType::Int, *idx,
                    ),
                }
            }
            match library.get_function(name, Some(cv)) {
                Ok(func) => {
                    match device.new_compute_pipeline_state_with_function(&func) {
                        Ok(pso) => Some(pso),
                        Err(e) => { eprintln!("[ggml] PSO failed for {}: {}", name, e); None }
                    }
                }
                Err(e) => { eprintln!("[ggml] Kernel {} not found: {}", name, e); None }
            }
        };

        // FC indices from ggml-metal-impl.h
        const FC_MUL_MV: u64 = 600;

        // Matmul kernels with function constants (nsg, nxpsg at FC_MUL_MV+0, FC_MUL_MV+1)
        // Q4_K: nsg=2, nxpsg=0 (not used for Q4_K)
        if let Some(p) = make_pipeline("kernel_mul_mv_q4_K_f32", &[(FC_MUL_MV, FcVal::Short(2)), (FC_MUL_MV+1, FcVal::Short(0))]) {
            pipelines.insert("kernel_mul_mv_q4_K_f32".into(), p);
        }
        // Q8_0: nsg=4
        if let Some(p) = make_pipeline("kernel_mul_mv_q8_0_f32", &[(FC_MUL_MV, FcVal::Short(4)), (FC_MUL_MV+1, FcVal::Short(0))]) {
            pipelines.insert("kernel_mul_mv_q8_0_f32".into(), p);
        }
        // F16: nsg=4, nxpsg=2 (for ne00=2688; nxpsg>0 required, 32/nxpsg = nypsg)
        if let Some(p) = make_pipeline("kernel_mul_mv_f16_f32", &[(FC_MUL_MV, FcVal::Short(4)), (FC_MUL_MV+1, FcVal::Short(2))]) {
            pipelines.insert("kernel_mul_mv_f16_f32".into(), p);
        }
        // F32: nsg=4, nxpsg=2
        if let Some(p) = make_pipeline("kernel_mul_mv_f32_f32", &[(FC_MUL_MV, FcVal::Short(4)), (FC_MUL_MV+1, FcVal::Short(2))]) {
            pipelines.insert("kernel_mul_mv_f32_f32".into(), p);
        }
        // Quantized matmul kernels matching llama.cpp's N_SG_* defaults.
        // Q5_0: nsg=2; Q5_K: nsg=2; Q6_K: nsg=2; Q4_0: nsg=2 (nxpsg=0 for Q-quants, unused).
        if let Some(p) = make_pipeline("kernel_mul_mv_q5_0_f32", &[(FC_MUL_MV, FcVal::Short(2)), (FC_MUL_MV+1, FcVal::Short(0))]) {
            pipelines.insert("kernel_mul_mv_q5_0_f32".into(), p);
        }
        if let Some(p) = make_pipeline("kernel_mul_mv_q5_K_f32", &[(FC_MUL_MV, FcVal::Short(2)), (FC_MUL_MV+1, FcVal::Short(0))]) {
            pipelines.insert("kernel_mul_mv_q5_K_f32".into(), p);
        }
        if let Some(p) = make_pipeline("kernel_mul_mv_q6_K_f32", &[(FC_MUL_MV, FcVal::Short(2)), (FC_MUL_MV+1, FcVal::Short(0))]) {
            pipelines.insert("kernel_mul_mv_q6_K_f32".into(), p);
        }
        if let Some(p) = make_pipeline("kernel_mul_mv_q4_0_f32", &[(FC_MUL_MV, FcVal::Short(2)), (FC_MUL_MV+1, FcVal::Short(0))]) {
            pipelines.insert("kernel_mul_mv_q4_0_f32".into(), p);
        }

        // mul_mv_id (MoE): same FC constants
        if let Some(p) = make_pipeline("kernel_mul_mv_id_q8_0_f32", &[(FC_MUL_MV, FcVal::Short(4)), (FC_MUL_MV+1, FcVal::Short(0))]) {
            pipelines.insert("kernel_mul_mv_id_q8_0_f32".into(), p);
        }
        if let Some(p) = make_pipeline("kernel_mul_mv_id_f16_f32", &[(FC_MUL_MV, FcVal::Short(4)), (FC_MUL_MV+1, FcVal::Short(0))]) {
            pipelines.insert("kernel_mul_mv_id_f16_f32".into(), p);
        }
        if let Some(p) = make_pipeline("kernel_mul_mv_id_q4_K_f32", &[(FC_MUL_MV, FcVal::Short(2)), (FC_MUL_MV+1, FcVal::Short(0))]) {
            pipelines.insert("kernel_mul_mv_id_q4_K_f32".into(), p);
        }
        // mul_mv_id quant variants for MoE expert matmuls
        if let Some(p) = make_pipeline("kernel_mul_mv_id_q5_0_f32", &[(FC_MUL_MV, FcVal::Short(2)), (FC_MUL_MV+1, FcVal::Short(0))]) {
            pipelines.insert("kernel_mul_mv_id_q5_0_f32".into(), p);
        }
        if let Some(p) = make_pipeline("kernel_mul_mv_id_q5_K_f32", &[(FC_MUL_MV, FcVal::Short(2)), (FC_MUL_MV+1, FcVal::Short(0))]) {
            pipelines.insert("kernel_mul_mv_id_q5_K_f32".into(), p);
        }
        if let Some(p) = make_pipeline("kernel_mul_mv_id_q6_K_f32", &[(FC_MUL_MV, FcVal::Short(2)), (FC_MUL_MV+1, FcVal::Short(0))]) {
            pipelines.insert("kernel_mul_mv_id_q6_K_f32".into(), p);
        }

        // SSM kernels (no function constants for basic version)
        for name in &["kernel_ssm_conv_f32_f32", "kernel_ssm_scan_f32"] {
            if let Some(p) = make_pipeline(name, &[]) {
                pipelines.insert(name.to_string(), p);
            }
        }

        // RMS norm (no FC for basic version)
        for name in &["kernel_rms_norm_f32", "kernel_rms_norm_mul_f32"] {
            if let Some(p) = make_pipeline(name, &[]) {
                pipelines.insert(name.to_string(), p);
            }
        }

        // RoPE (needs FC_ROPE for is_imrope)
        const FC_ROPE: u64 = 800;
        if let Some(p) = make_pipeline("kernel_rope_neox_f16", &[(FC_ROPE, FcVal::Bool(false))]) { // is_imrope=false
            pipelines.insert("kernel_rope_neox_f16".into(), p);
        }

        // Get rows (no FC)
        for name in &["kernel_get_rows_q5_0", "kernel_get_rows_q4_K"] {
            if let Some(p) = make_pipeline(name, &[]) {
                pipelines.insert(name.to_string(), p);
            }
        }

        // Argsort (no FC)
        if let Some(p) = make_pipeline("kernel_argsort_f32_i32_desc", &[]) {
            pipelines.insert("kernel_argsort_f32_i32_desc".into(), p);
        }

        // F32→F16 copy (for KV cache write)
        if let Some(p) = make_pipeline("kernel_cpy_f32_f16", &[]) {
            pipelines.insert("kernel_cpy_f32_f16".into(), p);
        }
        // F32→F32 copy (for SSM state persistence within single CB)
        if let Some(p) = make_pipeline("kernel_cpy_f32_f32", &[]) {
            pipelines.insert("kernel_cpy_f32_f32".into(), p);
        }

        // Flash attention for head_dim=128 (F16 KV cache)
        // KV cache layout: [pos, kv_heads, head_dim] contiguous F16
        // ns10/ns20 = stride in elements between consecutive KV positions
        //           = kv_heads * head_dim = 2 * 128 = 256 (from nb11/nb10)
        // has_mask=true: causal mask zeroes out positions beyond seq_len
        // has_kvpad=false: we handle partial chunks via mask instead of pad kernel
        const FC_FA: u64 = 300; // FC_FLASH_ATTN_EXT
        let ns_kv = 2 * 128; // num_kv_heads * head_dim = position stride in elements
        if let Some(p) = make_pipeline("kernel_flash_attn_ext_f16_dk128_dv128", &[
            (FC_FA + 0, FcVal::Bool(true)),       // has_mask (causal mask)
            (FC_FA + 1, FcVal::Bool(false)),       // has_sinks
            (FC_FA + 2, FcVal::Bool(false)),       // has_bias
            (FC_FA + 3, FcVal::Bool(false)),       // has_scap
            (FC_FA + 4, FcVal::Bool(false)),       // has_kvpad (handled by mask)
            (FC_FA + 10, FcVal::Bool(false)),      // bc_mask
            (FC_FA + 20, FcVal::Int(ns_kv as i32)), // ns10 (K position stride in elements)
            (FC_FA + 21, FcVal::Int(ns_kv as i32)), // ns20 (V position stride in elements)
            (FC_FA + 22, FcVal::Int(4)),           // nsg (simdgroups per threadgroup)
        ]) {
            pipelines.insert("flash_attn_f16_dk128".into(), p);
        }

        // Unary ops via function constants (FC_UNARY+0 = op, FC_UNARY+1 = has_count)
        const FC_UNARY: u64 = 1200;
        let unary_ops = [
            ("silu", 106i16),
            ("relu", 101),
            ("sqr",  13),
            ("sigmoid", 102),
            ("exp", 114),
            ("neg", 108),
            ("scale", 10),  // dst = scale * src + bias (uses args.scale, args.bias)
        ];
        for (name, op_id) in &unary_ops {
            if let Some(p) = make_pipeline("kernel_unary_f32_f32", &[(FC_UNARY, FcVal::Short(*op_id)), (FC_UNARY+1, FcVal::Bool(false))]) {
                pipelines.insert(format!("unary_{}", name), p);
            }
        }

        // Binary ops via function constants (FC_BIN+0 = op, FC_BIN+1 = format, FC_BIN+2 = row_broadcast)
        const FC_BIN: u64 = 1300;
        // FC_OP: 0=add, 1=sub, 2=mul, 3=div. FC_F: 1=tensor src1. FC_RB: false.
        let bin_ops = [
            ("add", 0i16),
            ("mul", 2),
        ];
        for (name, op_id) in &bin_ops {
            if let Some(p) = make_pipeline("kernel_bin_fuse_f32_f32_f32", &[(FC_BIN, FcVal::Short(*op_id)), (FC_BIN+1, FcVal::Short(1)), (FC_BIN+2, FcVal::Bool(false))]) {
                pipelines.insert(format!("bin_{}", name), p);
            }
        }

        // Custom MoE kernels (GPU-resident routing, no CB flush)
        if let Some(p) = make_pipeline("kernel_moe_weights_compute", &[]) {
            pipelines.insert("moe_weights_compute".into(), p);
        }
        if let Some(p) = make_pipeline("kernel_moe_weighted_sum", &[]) {
            pipelines.insert("moe_weighted_sum".into(), p);
        }
        if let Some(p) = make_pipeline("kernel_ssm_conv_state_update", &[]) {
            pipelines.insert("ssm_conv_state_update".into(), p);
        }
        if let Some(p) = make_pipeline("kernel_embed_lookup", &[]) {
            pipelines.insert("embed_lookup".into(), p);
        }
        if let Some(p) = make_pipeline("kernel_mask_unlock", &[]) {
            pipelines.insert("mask_unlock".into(), p);
        }

        eprintln!("[ggml] {} pipeline states created", pipelines.len());

        Ok(Self { library, pipelines })
    }

    /// Get a pipeline state by kernel name.
    pub fn get_pipeline(&self, name: &str) -> Option<&metal::ComputePipelineState> {
        self.pipelines.get(name)
    }
}

/// Pre-allocated Metal buffers for decode (m=1). Zero allocations during generation.
/// All sizes derived from model config at init time.
#[cfg(feature = "metal")]
pub struct DecodeBuffers {
    /// Hidden state ping-pong: [hidden_size] F32 (ggml uses F32 intermediates)
    pub hidden_a: metal::Buffer,
    pub hidden_b: metal::Buffer,
    /// Normed output: [hidden_size] F32
    pub normed: metal::Buffer,
    /// SSM in_proj output: [in_proj_dim] F32
    pub ssm_xz: metal::Buffer,
    /// SSM conv output: [conv_dim] F32
    pub ssm_conv_out: metal::Buffer,
    /// SSM scan output (y before gate): [d_inner] F32
    pub ssm_scan_out: metal::Buffer,
    /// SSM gated output: [d_inner] F32
    pub ssm_gated: metal::Buffer,
    /// Gate logits: [num_experts] F32
    pub gate_logits: metal::Buffer,
    /// Expert IDs from argsort: [num_experts] I32 (top-k indices in first k slots)
    pub expert_ids: metal::Buffer,
    /// Expert routing weights: [k] F32
    pub expert_weights: metal::Buffer,
    /// Argsort tmp buffer: [num_experts] I32
    pub argsort_tmp: metal::Buffer,
    /// MoE expert up output: [k * intermediate] F32
    pub expert_up: metal::Buffer,
    /// MoE expert down output: [k * hidden] F32
    pub expert_down: metal::Buffer,
    /// MoE reduced output: [hidden_size] F32
    pub moe_out: metal::Buffer,
    /// Shared expert intermediate: [shared_intermediate] F32
    pub shared_inter: metal::Buffer,
    /// Attention Q: [num_heads * head_dim] F32
    pub attn_q: metal::Buffer,
    /// Attention K: [num_kv_heads * head_dim] F32
    pub attn_k: metal::Buffer,
    /// Attention V: [num_kv_heads * head_dim] F32
    pub attn_v: metal::Buffer,
    /// Attention output: [num_heads * head_dim] F32
    pub attn_out: metal::Buffer,
    /// Logits: [vocab_size] F32
    pub logits: metal::Buffer,
    /// Pre-computed A = -exp(A_log) per SSM layer: [n_heads] F32
    /// Reused across layers (recomputed per layer during encode)
    pub ssm_a_buf: metal::Buffer,
    /// Per-MoE-layer routing weights from previous token: [k_experts] F32
    pub moe_weights: Vec<metal::Buffer>,
    /// Per-MoE-layer gate logit snapshots: [num_experts] F32 (saved per layer for weight update)
    pub moe_gate_snapshots: Vec<metal::Buffer>,
    /// Per-MoE-layer expert ID snapshots: [num_experts] I32
    pub moe_ids_snapshots: Vec<metal::Buffer>,
    /// Per-SSM-layer conv state: [d_conv, conv_dim] F32 sliding window
    pub ssm_conv_states: Vec<metal::Buffer>,
    /// Per-SSM-layer xBC snapshot: [conv_dim] F32, captured after in_proj for conv state update
    pub ssm_xbc_snapshots: Vec<metal::Buffer>,
    /// Per-SSM-layer scan state: [d_state, head_dim, n_head] F32
    pub ssm_scan_states: Vec<metal::Buffer>,
    /// Scan output + state: must hold [d_inner + d_state*head_dim*n_head] F32
    pub ssm_scan_full: metal::Buffer,
    /// KV cache: per-attention-layer K cache [max_seq_len, num_kv_heads, head_dim] F16
    pub kv_k_cache: Vec<metal::Buffer>,
    /// KV cache: per-attention-layer V cache [max_seq_len, num_kv_heads, head_dim] F16
    pub kv_v_cache: Vec<metal::Buffer>,
    /// Temp F16 buffer for F32→F16 conversion: [num_kv_heads * head_dim] F16
    pub attn_kv_f16: metal::Buffer,
    /// Causal mask for flash attention: [max_seq_len] F16
    /// mask[i] = 0 for i < seq_len (attend), -INF for i >= seq_len (ignore)
    pub attn_mask: metal::Buffer,
    /// Block descriptor for flash attention: [max_seq_len/64 + 1] bytes
    /// All 1s = "partial" (always apply mask). Avoids running blk precompute kernel.
    pub attn_blk: metal::Buffer,
    /// Per-token token ID: [1] u32. Written CPU-side before encode_decode_step, read by GPU embed lookup.
    pub token_id_buf: metal::Buffer,
}

#[cfg(feature = "metal")]
impl DecodeBuffers {
    /// Allocate all decode buffers from model config.
    pub fn new(device: &metal::DeviceRef, config: &crate::inference::model::ModelConfig) -> Self {
        let f32_buf = |size: usize| -> metal::Buffer {
            let buf = device.new_buffer((size * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
            // Zero-initialize — critical for SSM state buffers (garbage causes NaN in scan)
            unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, size * 4); }
            buf
        };

        let hidden = config.hidden_size;
        let d_inner = config.ssm_inner_size;
        let d_state = config.ssm_state_size;
        let n_groups = config.ssm_group_count;
        let n_heads_ssm = config.ssm_time_step_rank;
        let d_conv = config.ssm_conv_kernel;
        let conv_dim = d_inner + 2 * n_groups * d_state;
        let in_proj_dim = 2 * d_inner + 2 * n_groups * d_state + n_heads_ssm;
        let intermediate = config.intermediate_size;
        let k = config.num_active_experts;
        let shared_inter = 3712; // Nemotron shared expert intermediate
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim();
        let vocab = config.vocab_size.unwrap_or(131072);

        let total_bytes = (hidden * 2 + hidden + in_proj_dim + conv_dim + d_inner * 2 +
            config.num_experts + k * intermediate + k * hidden + hidden + shared_inter +
            num_heads * head_dim + num_kv_heads * head_dim * 2 + num_heads * head_dim + vocab) * 4;
        eprintln!("[ggml] Decode buffers: {:.1} KB", total_bytes as f64 / 1024.0);

        Self {
            hidden_a: f32_buf(hidden),
            hidden_b: f32_buf(hidden),
            normed: f32_buf(hidden),
            ssm_xz: f32_buf(in_proj_dim),
            ssm_conv_out: f32_buf(conv_dim),
            ssm_scan_out: f32_buf(d_inner),
            ssm_gated: f32_buf(d_inner),
            gate_logits: f32_buf(config.num_experts),
            expert_ids: f32_buf(config.num_experts), // I32 reuses f32 buf (same size)
            expert_weights: f32_buf(k),
            argsort_tmp: f32_buf(config.num_experts),
            expert_up: f32_buf(k * intermediate),
            expert_down: f32_buf(k * hidden),
            moe_out: f32_buf(hidden),
            shared_inter: f32_buf(shared_inter),
            attn_q: f32_buf(num_heads * head_dim),
            attn_k: f32_buf(num_kv_heads * head_dim),
            attn_v: f32_buf(num_kv_heads * head_dim),
            attn_out: f32_buf(num_heads * head_dim),
            logits: f32_buf(vocab),
            ssm_a_buf: f32_buf(n_heads_ssm),
            moe_gate_snapshots: {
                let num_moe = 23;
                let ne = config.num_experts.max(1);
                (0..num_moe).map(|_| f32_buf(ne)).collect()
            },
            moe_ids_snapshots: {
                let num_moe = 23;
                let ne = config.num_experts.max(1);
                (0..num_moe).map(|_| f32_buf(ne)).collect() // I32 fits in f32 buf
            },
            moe_weights: {
                let num_moe = 23; // Nemotron: 23 MoE layers
                let k_act = config.num_active_experts.max(1);
                let uniform_w = 2.5 / k_act as f32;
                (0..num_moe).map(|_| {
                    let buf = f32_buf(k_act);
                    unsafe {
                        let p = buf.contents() as *mut f32;
                        for i in 0..k_act { *p.add(i) = uniform_w; }
                    }
                    buf
                }).collect()
            },
            // Per-SSM-layer persistent state
            ssm_conv_states: {
                let num_ssm = 23; // Nemotron: 23 SSM layers
                let conv_dim = d_inner + 2 * n_groups * d_state;
                let d_conv_val = config.ssm_conv_kernel.max(4);
                (0..num_ssm).map(|_| f32_buf(d_conv_val * conv_dim)).collect()
            },
            ssm_xbc_snapshots: {
                let num_ssm = 23;
                let conv_dim = d_inner + 2 * n_groups * d_state;
                (0..num_ssm).map(|_| f32_buf(conv_dim)).collect()
            },
            ssm_scan_states: {
                let num_ssm = 23;
                let head_dim_ssm = if n_heads_ssm > 0 { d_inner / n_heads_ssm } else { 64 };
                (0..num_ssm).map(|_| f32_buf(d_state * head_dim_ssm * n_heads_ssm)).collect()
            },
            ssm_scan_full: {
                let head_dim_ssm = if n_heads_ssm > 0 { d_inner / n_heads_ssm } else { 64 };
                f32_buf(d_inner + d_state * head_dim_ssm * n_heads_ssm)
            },
            // KV cache: allocate per attention layer
            // Nemotron has 6 attention layers. max_seq_len=4096 default.
            // Zero-fill to prevent NaN from garbage data in unused positions.
            kv_k_cache: {
                let max_seq = 4096usize;
                let num_attn_layers = 6; // Nemotron: 6 attention layers
                let kv_row_bytes = (num_kv_heads * head_dim * 2) as u64; // F16
                let total = max_seq as u64 * kv_row_bytes;
                (0..num_attn_layers).map(|_| {
                    let buf = device.new_buffer(total, metal::MTLResourceOptions::StorageModeShared);
                    unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, total as usize); }
                    buf
                }).collect()
            },
            kv_v_cache: {
                let max_seq = 4096usize;
                let num_attn_layers = 6;
                let kv_row_bytes = (num_kv_heads * head_dim * 2) as u64;
                let total = max_seq as u64 * kv_row_bytes;
                (0..num_attn_layers).map(|_| {
                    let buf = device.new_buffer(total, metal::MTLResourceOptions::StorageModeShared);
                    unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, total as usize); }
                    buf
                }).collect()
            },
            attn_kv_f16: device.new_buffer((num_kv_heads * head_dim * 2) as u64, metal::MTLResourceOptions::StorageModeShared),
            // Causal mask: [max_seq_len] F16, pre-filled with -INF (0xFC00 = -inf in F16)
            // Each decode step sets mask[position] = 0 to "unlock" that position.
            attn_mask: {
                let max_seq = 4096usize;
                let buf = device.new_buffer((max_seq * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
                unsafe {
                    let p = buf.contents() as *mut u16;
                    for i in 0..max_seq { *p.add(i) = 0xFC00u16; } // F16 -inf
                }
                buf
            },
            // Block descriptors: all 1 = "partial" (always apply mask)
            attn_blk: {
                let max_seq = 4096usize;
                let nblk = (max_seq + 63) / 64 + 1; // ceil(max_seq/C) + safety
                let buf = device.new_buffer(nblk as u64, metal::MTLResourceOptions::StorageModeShared);
                unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 1, nblk); }
                buf
            },
            token_id_buf: {
                let buf = device.new_buffer(4u64, metal::MTLResourceOptions::StorageModeShared);
                unsafe { *(buf.contents() as *mut u32) = 0; }
                buf
            },
        }
    }
}

/// A single pre-computed GPU dispatch command.
/// All data needed to encode one `dispatchThreadgroups` call.
#[cfg(feature = "metal")]
pub struct CachedDispatch {
    /// Index into GgmlShaderLibrary.pipelines (by name hash)
    pub pso: metal::ComputePipelineState,
    /// Kernel arguments (raw bytes, set via set_bytes at index 0)
    pub kargs: Vec<u8>,
    /// Buffer bindings: (index, buffer_ref_key, offset)
    /// buffer_ref_key indexes into a flat buffer table
    pub buffers: Vec<(u64, usize, u64)>,  // (arg_index, buf_table_idx, offset)
    /// Grid dimensions
    pub grid: metal::MTLSize,
    /// Threads per threadgroup
    pub threads: metal::MTLSize,
    /// Threadgroup memory size (0 = none)
    pub smem: u64,
}

/// Pre-compiled dispatch list for zero-overhead token encoding.
/// Built once from model weights at init. Replayed per token.
#[cfg(feature = "metal")]
pub struct CachedGraph {
    /// Flat dispatch list (in execution order)
    pub dispatches: Vec<CachedDispatch>,
    /// Buffer table: all Metal buffers referenced by dispatches
    pub buffer_table: Vec<metal::Buffer>,
    /// Special buffer indices that change per token
    pub hidden_a_idx: usize,
    pub normed_idx: usize,
    pub logits_idx: usize,
}

/// Layer type for dispatch.
#[derive(Clone, Copy, PartialEq)]
pub enum LayerType { Ssm, Moe, Attention }

/// Layer info resolved at init time.
pub struct LayerInfo {
    pub layer_type: LayerType,
    pub idx: usize,
}

/// Resolve layer types from model weights.
pub fn resolve_layer_types(model: &crate::inference::model::Model, num_layers: usize) -> Vec<LayerInfo> {
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let prefix = format!("model.layers.{}", i);
        let is_ssm = model.get_weight(&format!("{}.ssm_in.weight", prefix)).is_some();
        let is_attn = model.get_weight(&format!("{}.self_attn.q_proj.weight", prefix)).is_some();
        let lt = if is_ssm { LayerType::Ssm } else if is_attn { LayerType::Attention } else { LayerType::Moe };
        layers.push(LayerInfo { layer_type: lt, idx: i });
    }
    layers
}

/// GGML compute graph for Nemotron decode.
pub struct GgmlGraph {
    /// Shader library with compiled kernels
    #[cfg(feature = "metal")]
    pub shader: Option<GgmlShaderLibrary>,
    /// Pre-allocated decode buffers
    #[cfg(feature = "metal")]
    pub buffers: Option<DecodeBuffers>,
    /// Cached dispatch list (built on first encode, replayed on subsequent)
    #[cfg(feature = "metal")]
    pub cached_graph: Option<CachedGraph>,
}

impl GgmlGraph {
    /// Create a new graph.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "metal")]
            shader: None,
            #[cfg(feature = "metal")]
            buffers: None,
            #[cfg(feature = "metal")]
            cached_graph: None,
        }
    }

    /// Initialize shader + buffers.
    #[cfg(feature = "metal")]
    pub fn init(&mut self, device: &metal::DeviceRef, config: &crate::inference::model::ModelConfig) -> Result<()> {
        if self.shader.is_none() {
            self.shader = Some(GgmlShaderLibrary::new(device)?);
        }
        if self.buffers.is_none() {
            self.buffers = Some(DecodeBuffers::new(device, config));
        }
        Ok(())
    }

    /// Check if initialized.
    pub fn is_ready(&self) -> bool {
        #[cfg(feature = "metal")]
        { self.shader.is_some() && self.buffers.is_some() }
        #[cfg(not(feature = "metal"))]
        { false }
    }
}

use crate::core::Result;
use crate::tensor::DType;

/// Fast-path: replay a pre-built dispatch list onto an encoder.
/// This replaces the 91ms encode_decode_step with ~1-2ms of buffer binding.
#[cfg(feature = "metal")]
pub fn encode_from_cache(
    cache: &CachedGraph,
    encoder: &metal::ComputeCommandEncoderRef,
) {
    for d in &cache.dispatches {
        encoder.set_compute_pipeline_state(&d.pso);
        encoder.set_bytes(0, d.kargs.len() as u64, d.kargs.as_ptr() as *const _);
        for &(idx, buf_key, offset) in &d.buffers {
            encoder.set_buffer(idx, Some(&cache.buffer_table[buf_key]), offset);
        }
        if d.smem > 0 {
            encoder.set_threadgroup_memory_length(0, d.smem);
        }
        encoder.dispatch_thread_groups(d.grid, d.threads);
    }
}

/// Helper: dispatch RMS norm using ggml kernel.
/// For Nemotron decode (m=1): normalizes a [hidden_size] F32 vector.
#[cfg(feature = "metal")]
fn dispatch_rms_norm(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    src: &metal::Buffer,       // input [hidden_size] F32
    norm_w: &metal::Buffer,    // norm weight [hidden_size] F32
    dst: &metal::Buffer,       // output [hidden_size] F32
    hidden_size: usize,
    eps: f32,
) {
    // Use fused rms_norm_mul (norm + elementwise multiply by weight)
    // kernel_rms_norm_mul_f32 uses T=float, so ne00_t = ne00 (not ne00/4 which is for float4)
    let ne00 = hidden_size as i32;
    let ne00_t = ne00;
    let nb1 = (hidden_size * 4) as u64; // F32 stride

    let args = kargs::Norm {
        ne00,
        ne00_t,
        nb1,
        nb2: nb1,
        nb3: nb1,
        eps,
        nef1: [1, 1, 0],  // [norm_ne01, mul_ne01, unused]
        nef2: [1, 1, 0],
        nef3: [1, 1, 0],
        nbf1: [nb1, nb1, 0],
        nbf2: [nb1, nb1, 0],
        nbf3: [nb1, nb1, 0],
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_bytes(0, std::mem::size_of::<kargs::Norm>() as u64,
        &args as *const kargs::Norm as *const _);
    encoder.set_buffer(1, Some(src), 0);
    encoder.set_buffer(2, Some(norm_w), 0); // fused mul weight
    encoder.set_buffer(3, Some(norm_w), 0); // fused add (unused, same buffer)
    encoder.set_buffer(4, Some(dst), 0);

    // Threadgroup memory: 32 * sizeof(float) = 128 bytes (for simd reduction)
    encoder.set_threadgroup_memory_length(0, 128);

    // nth = next power of 2 >= ne00_t, capped at 1024
    let nth = (ne00_t as u64).next_power_of_two().min(1024) as u64;

    // Grid: (1, 1, 1) for m=1 decode
    encoder.dispatch_thread_groups(
        metal::MTLSize::new(1, 1, 1),
        metal::MTLSize::new(nth, 1, 1),
    );
}

/// Map a LazyTensor's GGUF block format to (pipeline_name, nb00, block_size, nr0, nsg).
/// This selects the right Metal matmul kernel for any supported weight format
/// without needing the caller to know the on-disk encoding.
#[cfg(feature = "metal")]
#[inline]
fn matmul_kernel_for(t: &crate::hal::metal::LazyTensor) -> (&'static str, u64, usize, i32, i32) {
    use crate::inference::formats::GgmlType;
    let gt = t.ggml_type();
    let nb00 = gt.type_size() as u64;
    let bs = gt.block_size();
    match gt {
        GgmlType::F16  => ("kernel_mul_mv_f16_f32", 2,  1, 2, 4),
        GgmlType::F32  => ("kernel_mul_mv_f32_f32", 4,  1, 2, 4),
        GgmlType::Q8_0 => ("kernel_mul_mv_q8_0_f32", nb00, bs, 2, 4),
        GgmlType::Q5_0 => ("kernel_mul_mv_q5_0_f32", nb00, bs, 2, 2),
        GgmlType::Q4K  => ("kernel_mul_mv_q4_K_f32", nb00, bs, 2, 2),
        GgmlType::Q5K  => ("kernel_mul_mv_q5_K_f32", nb00, bs, 2, 2),
        GgmlType::Q6K  => ("kernel_mul_mv_q6_K_f32", nb00, bs, 2, 2),
        // Fallthrough — anything else is treated as F16 (the loader dequantizes
        // unsupported quant formats before reaching here).
        _ => ("kernel_mul_mv_f16_f32", 2, 1, 2, 4),
    }
}

/// Same picker for the mul_mv_id (MoE expert) variant.
#[cfg(feature = "metal")]
#[inline]
fn matmul_id_kernel_for(t: &crate::hal::metal::LazyTensor) -> (&'static str, u64, usize, i32, i32) {
    use crate::inference::formats::GgmlType;
    let gt = t.ggml_type();
    let nb00 = gt.type_size() as u64;
    let bs = gt.block_size();
    match gt {
        GgmlType::F16  => ("kernel_mul_mv_id_f16_f32", 2,  1, 2, 4),
        GgmlType::Q8_0 => ("kernel_mul_mv_id_q8_0_f32", nb00, bs, 2, 4),
        GgmlType::Q5_0 => ("kernel_mul_mv_id_q5_0_f32", nb00, bs, 2, 2),
        GgmlType::Q4K  => ("kernel_mul_mv_id_q4_K_f32", nb00, bs, 2, 2),
        GgmlType::Q5K  => ("kernel_mul_mv_id_q5_K_f32", nb00, bs, 2, 2),
        GgmlType::Q6K  => ("kernel_mul_mv_id_q6_K_f32", nb00, bs, 2, 2),
        _ => ("kernel_mul_mv_id_f16_f32", 2, 1, 2, 4),
    }
}

/// Helper: dispatch matrix-vector multiply using ggml kernel.
/// For decode (m=1): output[N] = input[K] × weight[N,K]^T
///
/// `nb00`: ggml type_size for the weight type (34 for Q8_0, 2 for F16, 4 for F32, 144/256*k for Q4K)
/// `nb01`: bytes per weight row = K * nb00 / block_size
/// `nr0`: rows per simdgroup (from ggml: 2 for Q4_K/Q8_0, 2 for F16/F32)
/// `nsg`: simdgroups per threadgroup (from ggml: 2 for Q4_K, 4 for Q8_0/F16/F32)
#[cfg(feature = "metal")]
fn dispatch_mul_mv(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    weight: &metal::Buffer,    // [N, K] quantized or F16/F32
    input: &metal::Buffer,     // [K] F32
    dst: &metal::Buffer,       // [N] F32
    n: usize,                  // output dim (ne01 = weight rows)
    k: usize,                  // input dim (ne00 = weight cols)
    nb00: u64,                 // weight type_size (bytes per block)
    block_size: usize,         // elements per block (32 for Q8_0, 256 for Q4K, 1 for F16/F32)
    nr0: i32,                  // rows per simdgroup
    nsg: i32,                  // simdgroups per threadgroup
) {
    let nb01 = (k as u64 * nb00) / block_size as u64; // bytes per weight row

    let args = kargs::MulMv {
        ne00: k as i32,
        ne01: n as i32,
        ne02: 1,
        nb00,
        nb01,
        nb02: nb01 * n as u64,
        nb03: nb01 * n as u64,
        ne10: k as i32,
        ne11: 1,
        ne12: 1,
        nb10: 4,                // F32 input: 4 bytes per element
        nb11: (k * 4) as u64,
        nb12: (k * 4) as u64,
        nb13: (k * 4) as u64,
        ne0: n as i32,          // output dim
        ne1: 1,
        nr0,
        r2: 1,
        r3: 1,
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_bytes(0, std::mem::size_of::<kargs::MulMv>() as u64,
        &args as *const kargs::MulMv as *const _);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(input), 0);
    encoder.set_buffer(3, Some(dst), 0);

    // Threadgroup memory: 32 * sizeof(float) * nr0 (for Q8_0/Q4K reduction)
    let smem = (32 * 4 * nr0 as u64).max(128);
    encoder.set_threadgroup_memory_length(0, smem);

    // Grid: ((ne01+nr0-1)/nr0, 1, 1) for Q8_0/Q4K
    // Threads: (32, nsg, 1)
    let n_tg = ((n as u64) + nr0 as u64 - 1) / nr0 as u64;
    encoder.dispatch_thread_groups(
        metal::MTLSize::new(n_tg, 1, 1),
        metal::MTLSize::new(32, nsg as u64, 1),
    );
}

/// Helper: dispatch mul_mv_id — matrix-vector with expert ID indirection.
/// For MoE decode: runs k expert matmuls in parallel using expert IDs from GPU buffer.
/// Matches llama.cpp's kernel_mul_mv_id dispatch pattern.
#[cfg(feature = "metal")]
fn dispatch_mul_mv_id(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    fused_weight: &metal::Buffer, // [K, N, num_all_experts] F16 (dequanted)
    input: &metal::Buffer,        // input — see input_per_expert
    dst: &metal::Buffer,          // [k_active * N] F32 output
    ids: &metal::Buffer,          // [k_active] I32 expert IDs
    n: usize,                     // output dim per expert (ne01 = weight rows)
    k: usize,                     // input dim (ne00 = weight cols)
    k_active: usize,              // number of active experts
    num_experts: usize,           // total experts in fused weight
    input_per_expert: bool,       // true: input is [k, k_active] (DOWN), false: [k] broadcast (UP)
    nb00: u64,                    // weight type_size
    block_size: usize,            // elements per block
    nr0: i32,
    nsg: i32,
) {
    let nb01 = (k as u64 * nb00) / block_size as u64;

    // For UP projection: input is broadcast, ne11=1
    // For DOWN projection: each expert has its own input column, ne11=k_active
    let ne11 = if input_per_expert { k_active as i32 } else { 1i32 };

    let args = kargs::MulMvId {
        nei0: k_active as i32, // n_expert_used
        nei1: 1,               // n_tokens
        nbi1: (k_active * 4) as u64, // stride for ids
        ne00: k as i32,
        ne01: n as i32,
        ne02: num_experts as i32, // total experts in fused weight
        nb00,
        nb01,
        nb02: nb01 * n as u64,   // stride to next expert
        ne10: k as i32,
        ne11,
        ne12: 1,
        ne13: 1,
        nb10: 4,
        nb11: (k * 4) as u64,
        nb12: (k * 4) as u64,
        ne0: n as i32,
        ne1: 1,
        nb1: (n * 4) as u64,
        nr0,
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_bytes(0, std::mem::size_of::<kargs::MulMvId>() as u64, &args as *const _ as *const _);
    encoder.set_buffer(1, Some(fused_weight), 0);
    encoder.set_buffer(2, Some(input), 0);
    encoder.set_buffer(3, Some(dst), 0);
    encoder.set_buffer(4, Some(ids), 0);

    let smem = (32 * 4 * nr0 as u64).max(128);
    encoder.set_threadgroup_memory_length(0, smem);

    // Grid z-dim = k_active * 1 (one token, k experts)
    let n_tg = ((n as u64) + nr0 as u64 - 1) / nr0 as u64;
    encoder.dispatch_thread_groups(
        metal::MTLSize::new(n_tg, 1, k_active as u64),
        metal::MTLSize::new(32, nsg as u64, 1),
    );
}

/// Helper: dispatch argsort for top-k expert selection.
/// Sorts gate logits descending, writes sorted indices to dst.
#[cfg(feature = "metal")]
fn dispatch_argsort(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    src: &metal::Buffer,    // [num_experts] F32 gate logits
    dst: &metal::Buffer,    // [num_experts] I32 sorted indices
    num_experts: usize,
) {
    #[repr(C)]
    struct ArgsortArgs {
        ne00: i32, ne01: i32, ne02: i32, ne03: i32,
        nb00: u64, nb01: u64, nb02: u64, nb03: u64,
        ne0: i32, ne1: i32, ne2: i32, ne3: i32,
        top_k: i32,
    }
    let nb = (num_experts * 4) as u64;
    let args = ArgsortArgs {
        ne00: num_experts as i32, ne01: 1, ne02: 1, ne03: 1,
        nb00: 4, nb01: nb, nb02: nb, nb03: nb,
        ne0: num_experts as i32, ne1: 1, ne2: 1, ne3: 1,
        top_k: num_experts as i32,
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_bytes(0, std::mem::size_of::<ArgsortArgs>() as u64, &args as *const _ as *const _);
    encoder.set_buffer(1, Some(src), 0);
    encoder.set_buffer(2, Some(dst), 0);

    // Bitonic sort: nth = next power of 2 >= num_experts
    let nth = (num_experts as u64).next_power_of_two().min(1024);
    let smem = ((nth * 4 + 15) / 16) * 16; // padded to 16
    encoder.set_threadgroup_memory_length(0, smem);

    encoder.dispatch_thread_groups(
        metal::MTLSize::new(1, 1, 1),
        metal::MTLSize::new(nth, 1, 1),
    );
}

/// Helper: dispatch binary op (add/mul) on F32 tensors.
/// For decode: element-wise op on [size] F32 vectors.
/// Uses the ggml bin kernel's non-row-broadcast path (FC_RB=false).
/// Grid: (ne01, ne02, ne03) threadgroups. Threads loop over ne0 elements per row.
#[cfg(feature = "metal")]
fn dispatch_bin_op(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    src0: &metal::Buffer,
    src1: &metal::Buffer,
    dst: &metal::Buffer,
    size: usize,
) {
    let n = size as i32;
    let nb = (size * 4) as u64;
    let args = kargs::Bin {
        // src0: [size] contiguous F32
        ne00: n, ne01: 1, ne02: 1, ne03: 1,
        nb00: 4, nb01: nb, nb02: nb, nb03: nb,
        // src1: [size] contiguous F32
        ne10: n, ne11: 1, ne12: 1, ne13: 1,
        nb10: 4, nb11: nb, nb12: nb, nb13: nb,
        // dst: [size] contiguous F32
        ne0: n, ne1: 1, ne2: 1, ne3: 1,
        nb0: 4, nb1: nb, nb2: nb, nb3: nb,
        offs: 0,
        o1: [0; 8],
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_bytes(0, std::mem::size_of::<kargs::Bin>() as u64, &args as *const kargs::Bin as *const _);
    encoder.set_buffer(1, Some(src0), 0);
    encoder.set_buffer(2, Some(src1), 0);
    encoder.set_buffer(3, Some(dst), 0);

    // Grid: (ne01=1, ne02=1, ne03=1) for 1D — one threadgroup
    // Threads per group: nth handles ne0 elements via grid-strided loop
    if size == 0 { return; }
    let nth = 256u64.min(size as u64);
    encoder.dispatch_thread_groups(
        metal::MTLSize::new(1, 1, 1),
        metal::MTLSize::new(nth, 1, 1),
    );
}

/// Helper: dispatch unary op (silu/relu/sqr/exp/neg) on F32 tensor.
/// Matches ggml_metal_kargs_unary exactly.
#[cfg(feature = "metal")]
fn dispatch_unary(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    src: &metal::Buffer,
    dst: &metal::Buffer,
    size: usize,
) {
    #[repr(C)]
    #[derive(Default)]
    struct UnaryArgs {
        // src0 shape
        ne00: i32, ne01: i32, ne02: i32, ne03: i32,
        nb00: u64, nb01: u64, nb02: u64, nb03: u64,
        // dst shape
        ne0: i32, ne1: i32, ne2: i32, ne3: i32,
        nb0: u64, nb1: u64, nb2: u64, nb3: u64,
        // extra params (for leaky_relu, scale, clamp, etc.)
        slope: f32, scale: f32, bias: f32, val: f32, min: f32, max: f32,
    }
    let n = size as i32;
    let nb = (size * 4) as u64;
    let args = UnaryArgs {
        ne00: n, ne01: 1, ne02: 1, ne03: 1,
        nb00: 4, nb01: nb, nb02: nb, nb03: nb,
        ne0: n, ne1: 1, ne2: 1, ne3: 1,
        nb0: 4, nb1: nb, nb2: nb, nb3: nb,
        ..Default::default()
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_bytes(0, std::mem::size_of::<UnaryArgs>() as u64, &args as *const UnaryArgs as *const _);
    encoder.set_buffer(1, Some(src), 0);
    encoder.set_buffer(2, Some(dst), 0);

    // FC_CNT=false path: grid=(ceil(ne0/nth/ne01), 1, 1), threads=(nth, 1, 1)
    // For 1D: ne01=1, so grid.x = ceil(ne0/nth)
    if size == 0 { return; }
    let nth = 256u64.min(size as u64);
    let n_tg = (size as u64 + nth - 1) / nth;
    encoder.dispatch_thread_groups(
        metal::MTLSize::new(n_tg, 1, 1),
        metal::MTLSize::new(nth, 1, 1),
    );
}

/// Encode a single decode step for Nemotron onto one Metal command buffer.
/// This is the core function that achieves zero-flush inference.
///
/// Replaces the imperative `forward_layer()` loop with a pre-planned graph:
/// - All 52 layers encoded onto one CB
/// - All intermediate buffers pre-allocated (DecodeBuffers)
/// - All kernels from ggml shader (exact llama.cpp precision)
/// - Single `commit()` + `wait_until_completed()` at the end
///
/// Implementation status: scaffolding in place, kernel dispatch TODO.
/// Encode a complete decode step onto a single Metal command buffer.
///
/// Currently dispatches: 39 ops across 52 layers including matmuls, norms,
/// activations, residual adds, argsort, mul_mv_id for MoE, SSM conv/scan.
///
/// Still TODO: RoPE (needs position buffer), flash attention (needs KV cache),
/// routing weight application (needs sigmoid + bias on GPU).
///
/// When fully wired, this function replaces the entire imperative forward_layer loop.
#[cfg(feature = "metal")]
pub fn encode_decode_step(
    _graph: &GgmlGraph,
    _model: &crate::inference::model::Model,
    _input_token: u32,
    _position: usize,
    _cb: &metal::CommandBufferRef,
    kv_cache_k: &[Option<&metal::Buffer>],
    kv_cache_v: &[Option<&metal::Buffer>],
    ssm_conv_states: &[Option<&metal::Buffer>],
    ssm_scan_states: &[Option<&metal::Buffer>],
    position_buf: &metal::Buffer,
    rope_cos: &metal::BufferRef,
    rope_sin: &metal::BufferRef,
    command_queue: &metal::CommandQueueRef,
) -> Result<()> {
    // The full 52-layer encode sequence will be implemented incrementally.
    // Architecture for each layer type:
    //
    // SSM layer:
    //   rms_norm(hidden → normed)
    //   mul_mv(normed × ssm_in_w → xz)         [F16 weight, F32 output]
    //   ssm_conv(xz → conv_out)                  [F32 in/out, uses persistent state]
    //   ssm_scan(conv_out → scan_out)             [F32, updates persistent state]
    //   mul_mv(scan_out × ssm_out_w → layer_out) [Q4K weight, F32 output]
    //   add(hidden + layer_out → hidden)          [F32 residual]
    //
    // MoE layer:
    //   rms_norm(hidden → normed)
    //   mul_mv(normed × gate_w → gate_logits)    [F32 weight, F32 output]
    //   argsort(gate_logits → expert_ids)         [top-k selection]
    //   mul_mv_id(normed × up_w[ids] → expert_up)   [F16 weight, F32 output, k experts]
    //   sqr(relu(expert_up))                      [ReLU² activation]
    //   mul_mv_id(expert_up × down_w[ids] → expert_down) [Q8_0 weight, F32 output]
    //   reduce(expert_down × weights → moe_out)   [weighted sum]
    //   mul_mv(normed × shared_up → shared_inter) [shared expert]
    //   sqr(relu(shared_inter))
    //   mul_mv(shared_inter × shared_down → shared_out)
    //   add(moe_out + shared_out → layer_out)
    //   add(hidden + layer_out → hidden)
    //
    // Attention layer:
    //   rms_norm(hidden → normed)
    //   mul_mv(normed × q_w → q)
    //   mul_mv(normed × k_w → k)
    //   mul_mv(normed × v_w → v)
    //   rope(q), rope(k)
    //   flash_attn(q, k_cache, v_cache → attn_out)
    //   mul_mv(attn_out × o_w → layer_out)
    //   add(hidden + layer_out → hidden)
    //
    // All dispatches encoded onto `cb` — zero flushes between layers.
    // Buffer ping-pong: hidden_a ↔ hidden_b alternate as input/output.

    let t_encode_start = std::time::Instant::now();
    let shader = _graph.shader.as_ref().ok_or_else(|| crate::core::Error::internal("ggml shader not compiled"))?;
    let bufs = _graph.buffers.as_ref().ok_or_else(|| crate::core::Error::internal("ggml decode buffers not allocated"))?;
    let config = _model.config();
    let hidden = config.hidden_size;

    // Cache layer types (stable across tokens)
    static LAYERS_CACHE: std::sync::OnceLock<Vec<LayerInfo>> = std::sync::OnceLock::new();
    let layers = LAYERS_CACHE.get_or_init(|| resolve_layer_types(_model, config.num_layers));

    // Cache weight buffer pointers (stable across tokens — weights don't move)
    static WEIGHT_CACHE: std::sync::OnceLock<std::collections::HashMap<String, usize>> = std::sync::OnceLock::new();
    let _wc = WEIGHT_CACHE.get_or_init(|| std::collections::HashMap::new());

    let wb = |name: &str| -> Option<&metal::Buffer> {
        _model.get_weight(name).map(|w| w.buffer())
    };

    // Lookup the LazyTensor (not just the buffer) so we can read its ggml_type
    // and pick the right matmul kernel.
    let wt = |name: &str| -> Option<&crate::hal::metal::LazyTensor> {
        _model.get_weight(name)
    };

    // Pre-convert norm weights from F16→F32 (cached across calls)
    static NORM_CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, metal::Buffer>>> = std::sync::OnceLock::new();
    let norm_cache = NORM_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let norm_f32 = |name: &str| -> metal::Buffer {
        let mut cache = norm_cache.lock().unwrap();
        if let Some(buf) = cache.get(name) { return buf.clone(); }
        let tensor = _model.get_weight(name).expect(&format!("norm weight: {}", name));
        let numel = tensor.shape().numel();
        let device = bufs.hidden_a.device();
        let f32_buf = device.new_buffer((numel * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        unsafe {
            let dst = f32_buf.contents() as *mut f32;
            if tensor.dtype() == DType::F16 {
                let src = tensor.buffer().contents() as *const half::f16;
                for i in 0..numel { *dst.add(i) = (*src.add(i)).to_f32(); }
            } else {
                let src = tensor.buffer().contents() as *const f32;
                std::ptr::copy_nonoverlapping(src, dst, numel);
            }
        }
        cache.insert(name.to_string(), f32_buf.clone());
        f32_buf
    };

    // Get pipeline state helper
    let pso = |name: &str| -> &metal::ComputePipelineState {
        shader.get_pipeline(name).expect(&format!("missing pipeline: {}", name))
    };

    // Quant-aware matmul: dispatches the correct kernel for the weight's
    // GGUF block format (Q5_0, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32). Reads the
    // raw weight buffer (which may be a zero-copy mmap of quant blocks).
    let mm_w = |encoder: &metal::ComputeCommandEncoderRef,
                wname: &str,
                input: &metal::Buffer,
                dst: &metal::Buffer,
                n: usize,
                k: usize| {
        if let Some(t) = wt(wname) {
            let (kname, nb00, bs, nr0, nsg) = matmul_kernel_for(t);
            dispatch_mul_mv(encoder, pso(kname), t.buffer(), input, dst, n, k, nb00, bs, nr0, nsg);
        }
    };

    // Quant-aware matmul_id (MoE expert matmul).
    let mm_id_w = |encoder: &metal::ComputeCommandEncoderRef,
                   wname: &str,
                   input: &metal::Buffer,
                   dst: &metal::Buffer,
                   ids: &metal::Buffer,
                   n: usize,
                   k: usize,
                   k_active: usize,
                   num_experts: usize,
                   input_per_expert: bool| {
        if let Some(t) = wt(wname) {
            let (kname, nb00, bs, nr0, nsg) = matmul_id_kernel_for(t);
            dispatch_mul_mv_id(encoder, pso(kname), t.buffer(), input, dst, ids,
                n, k, k_active, num_experts, input_per_expert, nb00, bs, nr0, nsg);
        }
    };

    // Write per-token scalars to GPU buffers (~30ns CPU work).
    unsafe {
        *(bufs.token_id_buf.contents() as *mut u32) = _input_token;
    }

    let queue = command_queue;
    let mut current_cb = _cb.to_owned();
    let mut encoder = current_cb.new_compute_command_encoder();

    // Embedding lookup on GPU: hidden_a[i] = F32(embed[token_id][i])
    if let Some(embed_w) = _model.get_weight("model.embed_tokens.weight") {
        #[repr(C)]
        struct EmbedArgs { hidden: i32 }
        let args = EmbedArgs { hidden: hidden as i32 };
        encoder.set_compute_pipeline_state(pso("embed_lookup"));
        encoder.set_bytes(0, std::mem::size_of::<EmbedArgs>() as u64, &args as *const _ as *const _);
        encoder.set_buffer(1, Some(&bufs.token_id_buf), 0);
        encoder.set_buffer(2, Some(embed_w.buffer()), 0);
        encoder.set_buffer(3, Some(&bufs.hidden_a), 0);
        let nth = 256u64;
        let n_tg = (hidden as u64 + nth - 1) / nth;
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(n_tg, 1, 1),
            metal::MTLSize::new(nth, 1, 1),
        );
    }

    // For each layer, encode ops onto the current command encoder
    let mut ssm_layer_idx = 0usize;
    let mut attn_layer_idx = 0usize;
    let skip_layers = std::env::var("SKIP_LAYERS").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
    let max_layers = std::env::var("MAX_LAYERS").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(usize::MAX);
    // Pre-build weight names to avoid per-token string allocations (cached on first call).
    struct LayerNames {
        prefix: String,
        norm: String,
        ssm_in: String, ssm_out: String, ssm_conv1d_w: String, ssm_conv1d_b: String,
        ssm_dt_bias: String, ssm_a: String, ssm_d: String, ssm_norm: String,
        gate: String, experts_up: String, experts_down: String,
        shared_up: String, shared_down: String, bias: String,
        q_proj: String, k_proj: String, v_proj: String, o_proj: String,
    }
    static LAYER_NAMES: std::sync::OnceLock<Vec<LayerNames>> = std::sync::OnceLock::new();
    let layer_names = LAYER_NAMES.get_or_init(|| {
        layers.iter().map(|l| {
            let p = format!("model.layers.{}", l.idx);
            LayerNames {
                norm: format!("{}.input_layernorm.weight", p),
                ssm_in: format!("{}.ssm_in.weight", p),
                ssm_out: format!("{}.ssm_out.weight", p),
                ssm_conv1d_w: format!("{}.ssm_conv1d.weight", p),
                ssm_conv1d_b: format!("{}.ssm_conv1d.bias", p),
                ssm_dt_bias: format!("{}.ssm_dt.bias", p),
                ssm_a: format!("{}.ssm_a", p),
                ssm_d: format!("{}.ssm_d", p),
                ssm_norm: format!("{}.ssm_norm.weight", p),
                gate: format!("{}.mlp.gate.weight", p),
                experts_up: format!("{}.mlp.experts_up.weight", p),
                experts_down: format!("{}.mlp.experts_down.weight", p),
                shared_up: format!("{}.mlp.shared_experts.up_proj.weight", p),
                shared_down: format!("{}.mlp.shared_experts.down_proj.weight", p),
                bias: format!("{}.exp_probs_b.bias", p),
                q_proj: format!("{}.self_attn.q_proj.weight", p),
                k_proj: format!("{}.self_attn.k_proj.weight", p),
                v_proj: format!("{}.self_attn.v_proj.weight", p),
                o_proj: format!("{}.self_attn.o_proj.weight", p),
                prefix: p,
            }
        }).collect()
    });

    for (layer_i, layer) in layers.iter().enumerate() {
        if layer_i < skip_layers || layer_i >= max_layers { continue; }

        let ln = &layer_names[layer_i];
        let norm_w_ref = wb(&ln.norm).unwrap();

        // RMS norm: hidden → normed (fused with F32 weight multiply)
        {
            let norm_w = norm_f32(&ln.norm);
            dispatch_rms_norm(&encoder, pso("kernel_rms_norm_mul_f32"),
                &bufs.hidden_a, &norm_w, &bufs.normed, hidden, config.rms_norm_eps);
        }

        match layer.layer_type {
            LayerType::Ssm => {
                let d_inner = config.ssm_inner_size;
                let conv_dim = d_inner + 2 * config.ssm_group_count * config.ssm_state_size;
                let in_proj_dim = 2 * d_inner + 2 * config.ssm_group_count * config.ssm_state_size + config.ssm_time_step_rank;

                // ── Step 1a: in_proj matmul [hidden → in_proj_dim] (quant-aware) ──
                mm_w(&encoder, &ln.ssm_in,
                    &bufs.normed, &bufs.ssm_xz, in_proj_dim, hidden);

                // ── Update conv state BEFORE the conv kernel runs, on GPU ──
                // The conv kernel reads a [d_conv, conv_dim] sliding window. We shift the
                // per-layer stored state left by 1 and insert the current token's xBC at
                // the end. xBC is at ssm_xz[d_inner..d_inner+conv_dim].
                // Single GPU dispatch — no CB flush.
                if ssm_layer_idx < bufs.ssm_conv_states.len() {
                    #[repr(C)]
                    struct ConvUpdArgs { conv_dim: i32, d_conv: i32 }
                    let cargs = ConvUpdArgs {
                        conv_dim: conv_dim as i32,
                        d_conv: config.ssm_conv_kernel as i32,
                    };
                    encoder.set_compute_pipeline_state(pso("ssm_conv_state_update"));
                    encoder.set_bytes(0, std::mem::size_of::<ConvUpdArgs>() as u64,
                        &cargs as *const _ as *const _);
                    encoder.set_buffer(1, Some(&bufs.ssm_conv_states[ssm_layer_idx]), 0);
                    // xBC starts at d_inner floats into ssm_xz
                    encoder.set_buffer(2, Some(&bufs.ssm_xz), (d_inner * 4) as u64);
                    let nth = 256u64.min(conv_dim as u64);
                    let n_tg = (conv_dim as u64 + nth - 1) / nth;
                    encoder.dispatch_thread_groups(
                        metal::MTLSize::new(n_tg, 1, 1),
                        metal::MTLSize::new(nth, 1, 1),
                    );
                }

                // ── Step 1a2: Snapshot xBC for conv state update (legacy, not strictly needed now) ──
                // Kept for debug purposes
                if let Some(cpy_pso) = shader.get_pipeline("kernel_cpy_f32_f32") {
                    if ssm_layer_idx < bufs.ssm_xbc_snapshots.len() {
                        let cpy_args = kargs::Cpy {
                            nk0: conv_dim as i64,
                            ne00: conv_dim as i64, ne01: 1, ne02: 1, ne03: 1,
                            nb00: 4, nb01: (conv_dim * 4) as u64,
                            nb02: (conv_dim * 4) as u64, nb03: (conv_dim * 4) as u64,
                            ne0: conv_dim as i64, ne1: 1, ne2: 1, ne3: 1,
                            nb0: 4, nb1: (conv_dim * 4) as u64,
                            nb2: (conv_dim * 4) as u64, nb3: (conv_dim * 4) as u64,
                        };
                        encoder.set_compute_pipeline_state(cpy_pso);
                        encoder.set_bytes(0, std::mem::size_of::<kargs::Cpy>() as u64, &cpy_args as *const _ as *const _);
                        encoder.set_buffer(1, Some(&bufs.ssm_xz), (d_inner * 4) as u64); // src: xBC at offset d_inner
                        encoder.set_buffer(2, Some(&bufs.ssm_xbc_snapshots[ssm_layer_idx]), 0); // dst
                        let nth = 256u64.min(conv_dim as u64);
                        let n_tg = (conv_dim as u64 + nth - 1) / nth;
                        encoder.dispatch_thread_groups(
                            metal::MTLSize::new(n_tg, 1, 1),
                            metal::MTLSize::new(nth, 1, 1),
                        );
                    }
                }

                // (all SSM layers run)

                // ── Step 1b: ssm_conv1d ──
                if let Some(conv_pso) = shader.get_pipeline("kernel_ssm_conv_f32_f32") {
                    let d_conv = config.ssm_conv_kernel;
                    let ncs = d_conv; // d_conv-1 past + 1 new = d_conv for decode
                    let args = kargs::SsmConv {
                        ne00: ncs as i64,              // conv window = d_conv
                        ne01: conv_dim as i64,         // channels
                        ne02: 1,                       // n_seqs
                        nb00: 4, nb01: (ncs * 4) as u64, nb02: (conv_dim * ncs * 4) as u64,
                        ne10: d_conv as i64,           // kernel size
                        ne11: conv_dim as i64,
                        nb10: 4, nb11: (d_conv * 4) as u64,
                        ne0: conv_dim as i64, ne1: 1, ne2: 1,
                        nb0: 4, nb1: (conv_dim * 4) as u64, nb2: (conv_dim * 4) as u64,
                    };
                    encoder.set_compute_pipeline_state(conv_pso);
                    encoder.set_bytes(0, std::mem::size_of::<kargs::SsmConv>() as u64, &args as *const _ as *const _);
                    // src0 = persistent conv state [d_conv, conv_dim] from DecodeBuffers
                    // The conv state is updated between tokens by update_ssm_conv_state()
                    if ssm_layer_idx < bufs.ssm_conv_states.len() {
                        encoder.set_buffer(1, Some(&bufs.ssm_conv_states[ssm_layer_idx]), 0);
                    } else {
                        encoder.set_buffer(1, Some(&bufs.ssm_xz), (d_inner * 4) as u64);
                    }
                    // Conv weight must be F32 (kernel reads as float*), but load_gguf_f16 stores as F16
                    let conv_w_name = &ln.ssm_conv1d_w;
                    let conv_w_f32 = norm_f32(&conv_w_name); // reuse F16→F32 converter
                    encoder.set_buffer(2, Some(&conv_w_f32), 0);
                    encoder.set_buffer(3, Some(&bufs.ssm_conv_out), 0);
                    encoder.dispatch_thread_groups(
                        metal::MTLSize::new(conv_dim as u64, 1, 1),
                        metal::MTLSize::new(1, 1, 1),
                    );
                }

                // ── Step 2: Add conv1d bias to conv output ──
                // Bias is F16 (load_gguf_f16), bin_add reads F32 — convert
                let conv_bias_name = &ln.ssm_conv1d_b;
                if _model.get_weight(&conv_bias_name).is_some() {
                    let conv_bias_f32 = norm_f32(&conv_bias_name);
                    dispatch_bin_op(&encoder, pso("bin_add"), &bufs.ssm_conv_out, &conv_bias_f32, &bufs.ssm_conv_out, conv_dim);
                }

                // ── Step 3: SiLU on ENTIRE conv output (x, B, C all get SiLU) ──
                // llama.cpp applies silu to the full xBC tensor before extracting x, B, C
                dispatch_unary(&encoder, pso("unary_silu"), &bufs.ssm_conv_out, &bufs.ssm_conv_out, conv_dim);

                // ── Step 4: Prepare A parameter ──
                // In Nemotron GGUF, ssm_a is stored as -exp(A_log) (already negative).
                // The scan kernel computes dA = exp(dtsp * A), which is < 1 when A < 0. ✓
                {
                    let n_head = config.ssm_time_step_rank;
                    if let Some(a_tensor) = _model.get_weight(&ln.ssm_a) {
                        let a_buf = a_tensor.buffer();
                        let a_buf_bytes = a_buf.length() as usize;
                        let expected_f32 = n_head * 4;
                        let expected_f16 = n_head * 2;
                        unsafe {
                            let dst = bufs.ssm_a_buf.contents() as *mut f32;
                            if a_buf_bytes == expected_f16 || a_buf_bytes < expected_f32 {
                                // Data is F16 — convert to F32 (already -exp(A_log))
                                let src = a_buf.contents() as *const half::f16;
                                for i in 0..n_head { *dst.add(i) = (*src.add(i)).to_f32(); }
                            } else {
                                // Data is F32 — copy directly (already -exp(A_log))
                                let src = a_buf.contents() as *const f32;
                                std::ptr::copy_nonoverlapping(src, dst, n_head);
                            }
                            // Debug: check A values for first SSM layer
                            static A_DBG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                            if !A_DBG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                                eprintln!("[ggml] A[0..4] = [{:.6},{:.6},{:.6},{:.6}] (pre-computed, should be negative)",
                                    *dst, *dst.add(1), *dst.add(2), *dst.add(3));
                            }
                        }
                    }
                }

                // ── Step 5: Add dt_bias to dt_raw (in ssm_xz at dt offset) ──
                // The scan kernel applies softplus(dt) internally, but expects dt + bias
                {
                    let n_head = config.ssm_time_step_rank;
                    let n_group = config.ssm_group_count;
                    let d_state = config.ssm_state_size;
                    let dt_byte_offset = ((2 * d_inner + 2 * n_group * d_state) * 4) as u64;
                    // dt_bias is F16 (load_gguf_f16), kernel reads as F32 — convert
                    let dt_bias_name = &ln.ssm_dt_bias;
                    if _model.get_weight(&dt_bias_name).is_some() {
                        let dt_bias_f32 = norm_f32(&dt_bias_name);
                        let nb = (n_head * 4) as u64;
                        let args = kargs::Bin {
                            ne00: n_head as i32, ne01: 1, ne02: 1, ne03: 1,
                            nb00: 4, nb01: nb, nb02: nb, nb03: nb,
                            ne10: n_head as i32, ne11: 1, ne12: 1, ne13: 1,
                            nb10: 4, nb11: nb, nb12: nb, nb13: nb,
                            ne0: n_head as i32, ne1: 1, ne2: 1, ne3: 1,
                            nb0: 4, nb1: nb, nb2: nb, nb3: nb,
                            offs: 0, o1: [0; 8],
                        };
                        encoder.set_compute_pipeline_state(pso("bin_add"));
                        encoder.set_bytes(0, std::mem::size_of::<kargs::Bin>() as u64, &args as *const _ as *const _);
                        encoder.set_buffer(1, Some(&bufs.ssm_xz), dt_byte_offset);
                        encoder.set_buffer(2, Some(&dt_bias_f32), 0);
                        encoder.set_buffer(3, Some(&bufs.ssm_xz), dt_byte_offset);
                        let nth = 64u64.min(n_head as u64);
                        encoder.dispatch_thread_groups(
                            metal::MTLSize::new(1, 1, 1),
                            metal::MTLSize::new(nth, 1, 1),
                        );
                    }
                }

                // ── Step 6: SSM scan ──
                if let Some(scan_pso) = shader.get_pipeline("kernel_ssm_scan_f32") {
                    let n_head = config.ssm_time_step_rank; // 64
                    let head_dim_ssm = if n_head > 0 { d_inner / n_head } else { 0 }; // 64
                    let d_state = config.ssm_state_size; // 128
                    let n_group = config.ssm_group_count; // 8

                    // src0 = state: [d_state, head_dim, n_head, n_seqs]
                    let s_nb00 = 4u64; // F32
                    let s_nb01 = d_state as u64 * 4;
                    let s_nb02 = head_dim_ssm as u64 * s_nb01;
                    let s_nb03 = n_head as u64 * s_nb02;

                    // src1 = x: [head_dim, n_head, n_seq_tokens, n_seqs] contiguous F32
                    let x_nb10 = 4u64;                            // stride for head_dim (innermost) = sizeof(f32)
                    let x_nb11 = (head_dim_ssm * 4) as u64;      // stride for n_head = head_dim * 4
                    let x_nb12 = (d_inner * 4) as u64;            // stride for n_seq_tokens = n_head * head_dim * 4
                    let x_nb13 = x_nb12;                          // stride for n_seqs (single seq)

                    // src2 = dt: [n_head, n_seq_tokens, n_seqs]
                    let dt_nb20 = 4u64;
                    let dt_nb21 = n_head as u64 * 4;
                    let dt_nb22 = dt_nb21;

                    // src3 = A: [1, n_head] — pre-computed -exp(A_log) in ssm_a_buf
                    let a_ne30 = 1i64;

                    // src4 = B: [d_state, n_group, n_seq_tokens, n_seqs]
                    let b_nb40 = 4u64;
                    let b_nb41 = d_state as u64 * 4;
                    let b_nb42 = n_group as u64 * b_nb41;
                    let b_nb43 = b_nb42;

                    // src5 = C: same layout as B
                    let c_nb51 = b_nb41;
                    let c_nb52 = b_nb42;
                    let c_nb53 = b_nb43;

                    let x_elements = head_dim_ssm * n_head;

                    let args = kargs::SsmScan {
                        d_state: d_state as i64,
                        d_inner: head_dim_ssm as i64, // ggml d_inner = head_dim in scan
                        n_head: n_head as i64,
                        n_group: n_group as i64,
                        n_seq_tokens: 1,
                        n_seqs: 1,
                        s_off: (x_elements * 4) as u64,
                        nb00: s_nb00, nb01: s_nb01, nb02: s_nb02, nb03: s_nb03,
                        nb10: x_nb10, nb11: x_nb11, nb12: x_nb12,
                        ns12: x_nb12 / x_nb10,
                        nb13: x_nb13,
                        nb20: dt_nb20, nb21: dt_nb21,
                        ns21: dt_nb21 / dt_nb20,
                        nb22: dt_nb22,
                        ne30: a_ne30,
                        nb31: (a_ne30 as u64) * 4, // stride for A head dim = ne30 * sizeof(float)
                        nb41: b_nb41, nb42: b_nb42,
                        ns42: b_nb42 / b_nb40,
                        nb43: b_nb43,
                        nb51: c_nb51, nb52: c_nb52,
                        ns52: c_nb52 / b_nb40,
                        nb53: c_nb53,
                        nb0: (head_dim_ssm as u64) * 4,
                    };
                    encoder.set_compute_pipeline_state(scan_pso);
                    encoder.set_bytes(0, std::mem::size_of::<kargs::SsmScan>() as u64, &args as *const _ as *const _);

                    // Buffer bindings:
                    // src0 = persistent SSM state, src1 = x, src2 = dt(+bias), src3 = A(-exp),
                    // src4 = B, src5 = C, src6 = ids, dst = scan_full (y + updated state)
                    if ssm_layer_idx < bufs.ssm_scan_states.len() {
                        encoder.set_buffer(1, Some(&bufs.ssm_scan_states[ssm_layer_idx]), 0);
                    }
                    encoder.set_buffer(2, Some(&bufs.ssm_conv_out), 0);     // src1 = x (SiLU'd)
                    encoder.set_buffer(3, Some(&bufs.ssm_xz), (2 * d_inner * 4 + 2 * n_group * d_state * 4) as u64); // src2 = dt+bias
                    encoder.set_buffer(4, Some(&bufs.ssm_a_buf), 0); // src3 = A (F16→F32 converted)
                    encoder.set_buffer(5, Some(&bufs.ssm_conv_out), (d_inner * 4) as u64); // src4 = B
                    let c_offset = (d_inner + n_group * d_state) * 4;
                    encoder.set_buffer(6, Some(&bufs.ssm_conv_out), c_offset as u64); // src5 = C
                    // src6 = ids (sequence index, NOT position). Must be 0 for single-sequence decode.
                    // position_buf holds the token position which is WRONG here — it would cause
                    // out-of-bounds reads into scan_states (ids[0]*nb03 = pos * state_size_bytes).
                    static ZERO_IDS: std::sync::OnceLock<metal::Buffer> = std::sync::OnceLock::new();
                    let zero_ids = ZERO_IDS.get_or_init(|| {
                        let buf = bufs.hidden_a.device().new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
                        unsafe { *(buf.contents() as *mut i32) = 0; }
                        buf
                    });
                    encoder.set_buffer(7, Some(zero_ids), 0);           // src6 = ids (always 0)

                    // CPU scan fallback (CPU_SCAN env var) - slower but useful for debugging
                    if std::env::var("CPU_SCAN").is_ok() {
                        // Flush to read conv_out, dt, etc.
                        encoder.end_encoding();
                        current_cb.commit();
                        current_cb.wait_until_completed();

                        if ssm_layer_idx == 0 && _position < 2 {
                            eprintln!("[CPU-SCAN] Running CPU scan for SSM layer {}, pos={}", ssm_layer_idx, _position);
                        }
                        unsafe {
                            let conv_out_ptr = bufs.ssm_conv_out.contents() as *const f32;
                            let dt_offset = 2 * d_inner + 2 * n_group * d_state;
                            let dt_ptr = (bufs.ssm_xz.contents() as *const f32).add(dt_offset);
                            let a_ptr = bufs.ssm_a_buf.contents() as *const f32;
                            let b_base = conv_out_ptr.add(d_inner);
                            let c_base = conv_out_ptr.add(d_inner + n_group * d_state);
                            let s0_ptr = bufs.ssm_scan_states[0].contents() as *const f32;
                            let dst_ptr = bufs.ssm_scan_full.contents() as *mut f32;

                            let n_g = n_group;

                            // For each head and dim position
                            for ir in 0..n_head {
                                let g = ir / (n_head / n_g);
                                let a_val = *a_ptr.add(ir); // A for this head (ne30=1, so A[0] for all dims)

                                for i1 in 0..head_dim_ssm {
                                    // x for this (head, dim)
                                    let x_val = *conv_out_ptr.add(i1 + ir * head_dim_ssm);
                                    // dt for this head
                                    let dt_val = *dt_ptr.add(ir);
                                    let dtsp = if dt_val <= 20.0 { (1.0 + dt_val.exp()).ln() } else { dt_val };
                                    let x_dt = x_val * dtsp;
                                    let da = (dtsp * a_val).exp();

                                    let mut y_val = 0.0f32;
                                    for i0 in 0..d_state {
                                        let s_idx = i0 + i1 * d_state + ir * head_dim_ssm * d_state;
                                        let s_old = *s0_ptr.add(s_idx);
                                        let b_val = *b_base.add(i0 + g * d_state);
                                        let c_val = *c_base.add(i0 + g * d_state);
                                        let s_new = s_old * da + b_val * x_dt;
                                        y_val += s_new * c_val;

                                        // Write updated state to scan_full at s_off
                                        let s_off = d_inner; // y occupies [0..d_inner], state at [d_inner..]
                                        *dst_ptr.add(s_off + s_idx) = s_new;
                                    }

                                    // Write y
                                    let y_idx = i1 + ir * head_dim_ssm;
                                    *dst_ptr.add(y_idx) = y_val;
                                }
                            }

                            if ssm_layer_idx == 0 && _position < 2 {
                                let y_l2: f32 = (0..d_inner).map(|i| (*dst_ptr.add(i)).powi(2)).sum::<f32>().sqrt();
                                eprintln!("[CPU-SCAN] y L2={:.6} y[0..4]=[{:.6},{:.6},{:.6},{:.6}]",
                                    y_l2, *dst_ptr, *dst_ptr.add(1), *dst_ptr.add(2), *dst_ptr.add(3));
                            }
                        }

                        // Skip the GPU scan dispatch (we already wrote results)
                        current_cb = queue.new_command_buffer().to_owned();
                        encoder = current_cb.new_compute_command_encoder();
                        // Jump past the GPU scan dispatch
                    } else {
                    encoder.set_buffer(8, Some(&bufs.ssm_scan_full), 0);    // dst = y + state

                    let n_simdgroups = (d_state as u64 + 31) / 32;
                    let smem = (n_simdgroups * 32 + n_simdgroups * 2) * 4;
                    encoder.set_threadgroup_memory_length(0, smem);

                    encoder.dispatch_thread_groups(
                        metal::MTLSize::new(head_dim_ssm as u64, n_head as u64, 1),
                        metal::MTLSize::new(d_state as u64, 1, 1),
                    );
                    }
                }


                // ── Step 6b: Copy updated scan state to persistent buffer ──
                // The scan kernel wrote updated state to scan_full[s_off..].
                // Copy it to scan_states[ssm_layer_idx] for the next token.
                if let Some(cpy_pso) = shader.get_pipeline("kernel_cpy_f32_f32") {
                    let n_head = config.ssm_time_step_rank;
                    let head_dim_ssm = if n_head > 0 { d_inner / n_head } else { 0 };
                    let d_state = config.ssm_state_size;
                    let state_elements = d_state * head_dim_ssm * n_head;
                    let s_off_bytes = (d_inner * 4) as u64;

                    if ssm_layer_idx < bufs.ssm_scan_states.len() && state_elements > 0 {
                        let cpy_args = kargs::Cpy {
                            nk0: state_elements as i64,
                            ne00: state_elements as i64, ne01: 1, ne02: 1, ne03: 1,
                            nb00: 4, nb01: (state_elements * 4) as u64,
                            nb02: (state_elements * 4) as u64, nb03: (state_elements * 4) as u64,
                            ne0: state_elements as i64, ne1: 1, ne2: 1, ne3: 1,
                            nb0: 4, nb1: (state_elements * 4) as u64,
                            nb2: (state_elements * 4) as u64, nb3: (state_elements * 4) as u64,
                        };
                        encoder.set_compute_pipeline_state(cpy_pso);
                        encoder.set_bytes(0, std::mem::size_of::<kargs::Cpy>() as u64, &cpy_args as *const _ as *const _);
                        encoder.set_buffer(1, Some(&bufs.ssm_scan_full), s_off_bytes); // src: state in scan_full
                        encoder.set_buffer(2, Some(&bufs.ssm_scan_states[ssm_layer_idx]), 0); // dst: persistent state
                        let nth = 256u64.min(state_elements as u64);
                        let n_tg = (state_elements as u64 + nth - 1) / nth;
                        encoder.dispatch_thread_groups(
                            metal::MTLSize::new(n_tg, 1, 1),
                            metal::MTLSize::new(nth, 1, 1),
                        );
                    }
                }

                // ── Step 7: D skip connection — y += D * x ──
                // D[n_heads] broadcasts to [head_dim, n_heads] = [d_inner]
                // Uses 2D bin_mul: D[h] * x[h*head_dim + d] for all (h, d)
                {
                    let n_head = config.ssm_time_step_rank;
                    let head_dim_ssm = d_inner / n_head;
                    let d_name = &ln.ssm_d;
                    if _model.get_weight(&d_name).is_some() {
                        let d_f32 = norm_f32(&d_name); // F16→F32
                        let hd4 = (head_dim_ssm * 4) as u64;
                        // 2D broadcast multiply: D[1, n_heads] × x[head_dim, n_heads]
                        let args = kargs::Bin {
                            // src0 (x from conv_out, SiLU'd): [head_dim, n_heads]
                            ne00: head_dim_ssm as i32, ne01: n_head as i32, ne02: 1, ne03: 1,
                            nb00: 4, nb01: hd4, nb02: (d_inner * 4) as u64, nb03: (d_inner * 4) as u64,
                            // src1 (D weights): [1, n_heads] — broadcasts across head_dim
                            ne10: 1, ne11: n_head as i32, ne12: 1, ne13: 1,
                            nb10: 4, nb11: 4, nb12: (n_head * 4) as u64, nb13: (n_head * 4) as u64,
                            // dst (ssm_gated as temp): [head_dim, n_heads]
                            ne0: head_dim_ssm as i32, ne1: n_head as i32, ne2: 1, ne3: 1,
                            nb0: 4, nb1: hd4, nb2: (d_inner * 4) as u64, nb3: (d_inner * 4) as u64,
                            offs: 0, o1: [0; 8],
                        };
                        encoder.set_compute_pipeline_state(pso("bin_mul"));
                        encoder.set_bytes(0, std::mem::size_of::<kargs::Bin>() as u64, &args as *const _ as *const _);
                        encoder.set_buffer(1, Some(&bufs.ssm_conv_out), 0); // src0 = x
                        encoder.set_buffer(2, Some(&d_f32), 0);             // src1 = D (F32)
                        encoder.set_buffer(3, Some(&bufs.ssm_gated), 0);   // dst = D*x (temp)
                        let nth = 256u64;
                        encoder.dispatch_thread_groups(
                            metal::MTLSize::new(n_head as u64, 1, 1), // ne01 threadgroups
                            metal::MTLSize::new(nth.min(head_dim_ssm as u64), 1, 1),
                        );
                        // Add D*x to scan output: y += D*x
                        dispatch_bin_op(&encoder, pso("bin_add"), &bufs.ssm_scan_full, &bufs.ssm_gated, &bufs.ssm_scan_full, d_inner);
                    }
                }

                // ── Step 8: SwiGLU gate — y = silu(z) * y (BEFORE norm, matching llama.cpp) ──
                // z is the first d_inner elements of ssm_xz (output gate from in_proj)
                dispatch_unary(&encoder, pso("unary_silu"), &bufs.ssm_xz, &bufs.ssm_gated, d_inner);
                dispatch_bin_op(&encoder, pso("bin_mul"), &bufs.ssm_scan_full, &bufs.ssm_gated, &bufs.ssm_scan_full, d_inner);

                // ── Step 9: Group RMS norm on gated output ──
                // 8 groups of d_inner/n_groups elements each, with per-element norm weight
                {
                    let n_groups = config.ssm_group_count; // 8
                    let group_size = d_inner / n_groups;   // 512
                    let ssm_norm_name = &ln.ssm_norm;
                    if _model.get_weight(&ssm_norm_name).is_some() {
                        let norm_w_buf = norm_f32(&ssm_norm_name);
                        let norm_w = &norm_w_buf;
                        let ne00 = group_size as i32;
                        let ne00_t = ne00; // T=float, not float4
                        let nb1 = (group_size * 4) as u64;
                        let args = kargs::Norm {
                            ne00,
                            ne00_t,
                            nb1,
                            nb2: (d_inner * 4) as u64,
                            nb3: (d_inner * 4) as u64,
                            eps: config.rms_norm_eps,
                            nef1: [n_groups as i32, n_groups as i32, 0],
                            nef2: [1, 1, 0],
                            nef3: [1, 1, 0],
                            nbf1: [nb1, nb1, 0],
                            nbf2: [(d_inner * 4) as u64, (d_inner * 4) as u64, 0],
                            nbf3: [(d_inner * 4) as u64, (d_inner * 4) as u64, 0],
                        };
                        encoder.set_compute_pipeline_state(pso("kernel_rms_norm_mul_f32"));
                        encoder.set_bytes(0, std::mem::size_of::<kargs::Norm>() as u64, &args as *const _ as *const _);
                        encoder.set_buffer(1, Some(&bufs.ssm_scan_full), 0); // src
                        encoder.set_buffer(2, Some(norm_w), 0);              // mul weight
                        encoder.set_buffer(3, Some(norm_w), 0);              // add weight (unused)
                        encoder.set_buffer(4, Some(&bufs.ssm_scan_full), 0); // dst (in-place)
                        let nth = (ne00_t as u64).next_power_of_two().min(1024);
                        encoder.set_threadgroup_memory_length(0, 128);
                        // One threadgroup per group
                        encoder.dispatch_thread_groups(
                            metal::MTLSize::new(n_groups as u64, 1, 1),
                            metal::MTLSize::new(nth, 1, 1),
                        );
                    }
                }


                // ── Step 10: Output projection — [d_inner → hidden] (quant-aware) ──
                mm_w(&encoder, &ln.ssm_out,
                    &bufs.ssm_scan_full, &bufs.hidden_b, hidden, d_inner);

                // ── Step 11: Residual add — hidden_a += layer_output ──
                dispatch_bin_op(&encoder, pso("bin_add"), &bufs.hidden_a, &bufs.hidden_b, &bufs.hidden_a, hidden);

                ssm_layer_idx += 1;
            }
            LayerType::Moe => {
                let intermediate = config.intermediate_size; // 1856
                let k_experts = config.num_active_experts;   // 6

                // Gate matmul: [hidden → num_experts] (quant-aware; orig F32, kept as F16 by loader)
                mm_w(&encoder, &ln.gate,
                    &bufs.normed, &bufs.gate_logits, config.num_experts, hidden);

                // ── MoE routing (GPU-resident, no CB flush) ──
                // 1. sigmoid(logits) → unbiased probs
                // 2. save unbiased probs to argsort_tmp via cpy
                // 3. add bias to gate_logits → biased probs
                // 4. argsort_desc(biased) → expert_ids (top-k in first k positions)
                // 5. kernel_moe_weights_compute: read ids + unbiased → normalized scaled weights
                //    (weights stored in bufs.expert_weights, k_active floats)
                // All steps stay on a single CB — no CPU round-trip.

                // Step 1: sigmoid(gate_logits) → gate_logits (in-place)
                dispatch_unary(&encoder, pso("unary_sigmoid"), &bufs.gate_logits, &bufs.gate_logits, config.num_experts);

                // Step 2: Save unbiased probs to argsort_tmp
                if let Some(cpy_pso) = shader.get_pipeline("kernel_cpy_f32_f32") {
                    let ne = config.num_experts;
                    let cpy_args = kargs::Cpy {
                        nk0: ne as i64,
                        ne00: ne as i64, ne01: 1, ne02: 1, ne03: 1,
                        nb00: 4, nb01: (ne * 4) as u64, nb02: (ne * 4) as u64, nb03: (ne * 4) as u64,
                        ne0: ne as i64, ne1: 1, ne2: 1, ne3: 1,
                        nb0: 4, nb1: (ne * 4) as u64, nb2: (ne * 4) as u64, nb3: (ne * 4) as u64,
                    };
                    encoder.set_compute_pipeline_state(cpy_pso);
                    encoder.set_bytes(0, std::mem::size_of::<kargs::Cpy>() as u64, &cpy_args as *const _ as *const _);
                    encoder.set_buffer(1, Some(&bufs.gate_logits), 0);
                    encoder.set_buffer(2, Some(&bufs.argsort_tmp), 0);
                    let nth = 256u64.min(ne as u64);
                    encoder.dispatch_thread_groups(
                        metal::MTLSize::new((ne as u64 + nth - 1) / nth, 1, 1),
                        metal::MTLSize::new(nth, 1, 1),
                    );
                }

                // Step 3: Add bias to gate_logits for SELECTION (biased probs)
                let moe_bias_name = &ln.bias;
                if _model.get_weight(&moe_bias_name).is_some() {
                    let moe_bias_f32 = norm_f32(&moe_bias_name);
                    dispatch_bin_op(&encoder, pso("bin_add"), &bufs.gate_logits, &moe_bias_f32, &bufs.gate_logits, config.num_experts);
                }

                // Step 4: Argsort desc (expert_ids[0..k] = top-k indices by biased prob)
                dispatch_argsort(&encoder, pso("kernel_argsort_f32_i32_desc"),
                    &bufs.gate_logits, &bufs.expert_ids, config.num_experts);

                // Step 5: GPU-side weight compute. Reads unbiased probs via indirect lookup,
                // writes normalized+scaled weights to bufs.expert_weights (k_experts floats).
                {
                    #[repr(C)]
                    struct MoeWArgs { k_active: i32, num_experts: i32, scale: f32 }
                    let wargs = MoeWArgs {
                        k_active: k_experts as i32,
                        num_experts: config.num_experts as i32,
                        scale: 2.5f32,
                    };
                    encoder.set_compute_pipeline_state(pso("moe_weights_compute"));
                    encoder.set_bytes(0, std::mem::size_of::<MoeWArgs>() as u64, &wargs as *const _ as *const _);
                    encoder.set_buffer(1, Some(&bufs.expert_ids), 0);
                    encoder.set_buffer(2, Some(&bufs.argsort_tmp), 0); // unbiased probs
                    encoder.set_buffer(3, Some(&bufs.expert_weights), 0);
                    // threadgroup memory for tree reduction (k_experts floats)
                    encoder.set_threadgroup_memory_length(0, (k_experts * 4) as u64);
                    encoder.dispatch_thread_groups(
                        metal::MTLSize::new(1, 1, 1),
                        metal::MTLSize::new(k_experts as u64, 1, 1),
                    );
                }

                // 4. Expert matmuls with mul_mv_id (quant-aware: Q5_0 up, Q8_0 down)
                let num_experts_total = config.num_experts; // 128 total experts in fused weight
                // UP projection: input is single token (broadcast), ne11=1
                mm_id_w(&encoder, &ln.experts_up,
                    &bufs.normed, &bufs.expert_up, &bufs.expert_ids,
                    intermediate, hidden, k_experts, num_experts_total, false);
                dispatch_unary(&encoder, pso("unary_relu"), &bufs.expert_up, &bufs.expert_up, k_experts * intermediate);
                dispatch_unary(&encoder, pso("unary_sqr"), &bufs.expert_up, &bufs.expert_up, k_experts * intermediate);
                // DOWN projection: each expert has its own intermediate input column, ne11=k_active
                mm_id_w(&encoder, &ln.experts_down,
                    &bufs.expert_up, &bufs.expert_down, &bufs.expert_ids,
                    hidden, intermediate, k_experts, num_experts_total, true);
                // 5. Fused weighted sum: moe_out[h] = Σ weights[e] * expert_down[e*hidden+h]
                // Single dispatch replaces k_experts * 2 dispatches of scale+add.
                {
                    #[repr(C)]
                    struct MoeSumArgs { k_active: i32, hidden: i32 }
                    let sargs = MoeSumArgs { k_active: k_experts as i32, hidden: hidden as i32 };
                    encoder.set_compute_pipeline_state(pso("moe_weighted_sum"));
                    encoder.set_bytes(0, std::mem::size_of::<MoeSumArgs>() as u64, &sargs as *const _ as *const _);
                    encoder.set_buffer(1, Some(&bufs.expert_weights), 0);
                    encoder.set_buffer(2, Some(&bufs.expert_down), 0);
                    encoder.set_buffer(3, Some(&bufs.moe_out), 0);
                    let nth = 256u64;
                    let n_tg = (hidden as u64 + nth - 1) / nth;
                    encoder.dispatch_thread_groups(
                        metal::MTLSize::new(n_tg, 1, 1),
                        metal::MTLSize::new(nth, 1, 1),
                    );
                }

                // Shared expert: up + relu² + down (quant-aware)
                let skip_shexp = std::env::var("SKIP_SHEXP").is_ok();
                if !skip_shexp {
                    mm_w(&encoder, &ln.shared_up,
                        &bufs.normed, &bufs.shared_inter, 3712, hidden);
                    // ReLU² on shared_inter
                    dispatch_unary(&encoder, pso("unary_relu"), &bufs.shared_inter, &bufs.shared_inter, 3712);
                    dispatch_unary(&encoder, pso("unary_sqr"), &bufs.shared_inter, &bufs.shared_inter, 3712);

                    mm_w(&encoder, &ln.shared_down,
                        &bufs.shared_inter, &bufs.hidden_b, hidden, 3712);

                    // Add: moe_out + shared_out → hidden_b (combined MoE output)
                    dispatch_bin_op(&encoder, pso("bin_add"), &bufs.moe_out, &bufs.hidden_b, &bufs.hidden_b, hidden);
                } else {
                    // Without shared expert, moe_out IS the combined output
                    // Copy moe_out to hidden_b for the residual add below
                    if let Some(cpy_pso) = shader.get_pipeline("kernel_cpy_f32_f32") {
                        let cpy_args = kargs::Cpy {
                            nk0: hidden as i64,
                            ne00: hidden as i64, ne01: 1, ne02: 1, ne03: 1,
                            nb00: 4, nb01: (hidden * 4) as u64, nb02: (hidden * 4) as u64, nb03: (hidden * 4) as u64,
                            ne0: hidden as i64, ne1: 1, ne2: 1, ne3: 1,
                            nb0: 4, nb1: (hidden * 4) as u64, nb2: (hidden * 4) as u64, nb3: (hidden * 4) as u64,
                        };
                        encoder.set_compute_pipeline_state(cpy_pso);
                        encoder.set_bytes(0, std::mem::size_of::<kargs::Cpy>() as u64, &cpy_args as *const _ as *const _);
                        encoder.set_buffer(1, Some(&bufs.moe_out), 0);
                        encoder.set_buffer(2, Some(&bufs.hidden_b), 0);
                        let nth = 256u64.min(hidden as u64);
                        let n_tg = (hidden as u64 + nth - 1) / nth;
                        encoder.dispatch_thread_groups(
                            metal::MTLSize::new(n_tg, 1, 1),
                            metal::MTLSize::new(nth, 1, 1),
                        );
                    }
                }

                // Residual: hidden_a + hidden_b → hidden_a
                dispatch_bin_op(&encoder, pso("bin_add"), &bufs.hidden_a, &bufs.hidden_b, &bufs.hidden_a, hidden);
            }
            LayerType::Attention => {
                if std::env::var("SKIP_ATTN").is_ok() {
                    attn_layer_idx += 1;
                    continue;
                }
                let num_heads = config.num_heads;
                let num_kv_heads = config.num_kv_heads;
                let head_dim = config.head_dim();
                let q_dim = num_heads * head_dim;
                let kv_dim = num_kv_heads * head_dim;

                // Q/K/V projections (quant-aware: Q5_0 for Q/V, Q8_0 for K)
                mm_w(&encoder, &ln.q_proj,
                    &bufs.normed, &bufs.attn_q, q_dim, hidden);
                mm_w(&encoder, &ln.k_proj,
                    &bufs.normed, &bufs.attn_k, kv_dim, hidden);
                mm_w(&encoder, &ln.v_proj,
                    &bufs.normed, &bufs.attn_v, kv_dim, hidden);

                // Nemotron-H does NOT use RoPE (rope_type = -1 / LLAMA_ROPE_TYPE_NONE per llama.cpp).
                // Confirmed with llama.cpp reference: no RoPE is applied to Nemotron attention.
                if false { let _ = shader.get_pipeline("kernel_rope_neox_f16"); }
                if false && let Some(rope_pso) = shader.get_pipeline("kernel_rope_neox_f16") {
                    let rope_dim = if config.rope_dim > 0 { config.rope_dim as i32 } else { 84 };
                    let args = kargs::Rope {
                        ne00: head_dim as i32, ne01: num_heads as i32, ne02: 1, ne03: 1,
                        nb00: 4, nb01: (head_dim * 4) as u64, nb02: (q_dim * 4) as u64, nb03: (q_dim * 4) as u64,
                        ne0: head_dim as i32, ne1: num_heads as i32, ne2: 1, ne3: 1,
                        nb0: 4, nb1: (head_dim * 4) as u64, nb2: (q_dim * 4) as u64, nb3: (q_dim * 4) as u64,
                        n_past: 0,
                        n_dims: rope_dim,
                        n_ctx_orig: 1048576, // max context
                        freq_base: config.rope_theta,
                        freq_scale: 1.0,
                        ext_factor: 0.0, attn_factor: 1.0,
                        beta_fast: 32.0, beta_slow: 1.0,
                        sect_0: 0, sect_1: 0, sect_2: 0, sect_3: 0,
                        has_freq_factors: false,
                    };
                    // RoPE kernel signature: kargs, src0 (input), src1 (positions), src2 (freq_factors opt), dst
                    // src2 is a frequency factor array (1 per half-dim). We pass attn_kv_f16 as dummy.
                    // The kernel checks `args.src2 ? freq_factors[ic] : 1.0f` — but args.src2 is a flag.
                    // Set it to 0 in kargs so kernel uses 1.0 freq_factor.
                    encoder.set_compute_pipeline_state(rope_pso);
                    encoder.set_bytes(0, std::mem::size_of::<kargs::Rope>() as u64, &args as *const _ as *const _);
                    encoder.set_buffer(1, Some(&bufs.attn_q), 0);      // src0 = Q input
                    encoder.set_buffer(2, Some(position_buf), 0);       // src1 = positions
                    encoder.set_buffer(3, Some(&bufs.attn_kv_f16), 0);  // src2 = freq_factors (dummy)
                    encoder.set_buffer(4, Some(&bufs.attn_q), 0);       // dst = Q output (in-place)
                    encoder.dispatch_thread_groups(
                        metal::MTLSize::new((rope_dim as u64 / 2 + 31) / 32, num_heads as u64, 1),
                        metal::MTLSize::new(32, 1, 1),
                    );
                    // RoPE on K (same kernel, different head count)
                    let k_args = kargs::Rope {
                        ne01: num_kv_heads as i32,
                        nb01: (head_dim * 4) as u64,
                        nb02: (kv_dim * 4) as u64,
                        nb03: (kv_dim * 4) as u64,
                        ne1: num_kv_heads as i32,
                        nb1: (head_dim * 4) as u64,
                        nb2: (kv_dim * 4) as u64,
                        nb3: (kv_dim * 4) as u64,
                        ..args
                    };
                    encoder.set_bytes(0, std::mem::size_of::<kargs::Rope>() as u64, &k_args as *const _ as *const _);
                    encoder.set_buffer(1, Some(&bufs.attn_k), 0);       // src0 = K input
                    encoder.set_buffer(4, Some(&bufs.attn_k), 0);       // dst = K output (in-place)
                    encoder.dispatch_thread_groups(
                        metal::MTLSize::new((rope_dim as u64 / 2 + 31) / 32, num_kv_heads as u64, 1),
                        metal::MTLSize::new(32, 1, 1),
                    );
                }

                // ── KV cache write: F32→F16 copy K,V to cache at position ──
                // Use cpy_f32_f16 compute kernel (stays on same encoder, no blit needed)
                {
                    let attn_idx = layers.iter().take(layer.idx + 1)
                        .filter(|l| l.layer_type == LayerType::Attention).count() - 1;
                    let kv_dim = num_kv_heads * head_dim;
                    let kv_row_f16 = (kv_dim * 2) as u64; // F16 bytes per row
                    let dst_offset = _position as u64 * kv_row_f16;

                    if let Some(cpy_pso) = shader.get_pipeline("kernel_cpy_f32_f16") {
                        let cpy_args = kargs::Cpy {
                            nk0: kv_dim as i64,
                            ne00: kv_dim as i64, ne01: 1, ne02: 1, ne03: 1,
                            nb00: 4, nb01: (kv_dim * 4) as u64,
                            nb02: (kv_dim * 4) as u64, nb03: (kv_dim * 4) as u64,
                            ne0: kv_dim as i64, ne1: 1, ne2: 1, ne3: 1,
                            nb0: 2, nb1: kv_row_f16, nb2: kv_row_f16, nb3: kv_row_f16,
                        };

                        // Copy K (F32→F16) into kv_k_cache[attn_idx] at position offset
                        if attn_idx < bufs.kv_k_cache.len() {
                            encoder.set_compute_pipeline_state(cpy_pso);
                            encoder.set_bytes(0, std::mem::size_of::<kargs::Cpy>() as u64, &cpy_args as *const _ as *const _);
                            encoder.set_buffer(1, Some(&bufs.attn_k), 0);
                            encoder.set_buffer(2, Some(&bufs.kv_k_cache[attn_idx]), dst_offset);
                            let nth = 256u64.min(kv_dim as u64);
                            let n_tg = (kv_dim as u64 + nth - 1) / nth;
                            encoder.dispatch_thread_groups(
                                metal::MTLSize::new(n_tg, 1, 1),
                                metal::MTLSize::new(nth, 1, 1),
                            );
                        }

                        // Copy V (F32→F16) into kv_v_cache[attn_idx] at position offset
                        if attn_idx < bufs.kv_v_cache.len() {
                            encoder.set_compute_pipeline_state(cpy_pso);
                            encoder.set_bytes(0, std::mem::size_of::<kargs::Cpy>() as u64, &cpy_args as *const _ as *const _);
                            encoder.set_buffer(1, Some(&bufs.attn_v), 0);
                            encoder.set_buffer(2, Some(&bufs.kv_v_cache[attn_idx]), dst_offset);
                            let nth = 256u64.min(kv_dim as u64);
                            let n_tg = (kv_dim as u64 + nth - 1) / nth;
                            encoder.dispatch_thread_groups(
                                metal::MTLSize::new(n_tg, 1, 1),
                                metal::MTLSize::new(nth, 1, 1),
                            );
                        }
                    }

                    // ── Flash attention: Q × K_cache → softmax → × V_cache → attn_out ──
                    // Update causal mask on GPU: unlock current position. Only the first
                    // attention layer needs this; subsequent layers see the same mask.
                    if attn_idx == 0 {
                        #[repr(C)]
                        struct MaskArgs { pos: i32 }
                        let margs = MaskArgs { pos: _position as i32 };
                        encoder.set_compute_pipeline_state(pso("mask_unlock"));
                        encoder.set_bytes(0, std::mem::size_of::<MaskArgs>() as u64,
                            &margs as *const _ as *const _);
                        encoder.set_buffer(1, Some(&bufs.attn_mask), 0);
                        encoder.dispatch_thread_groups(
                            metal::MTLSize::new(1, 1, 1),
                            metal::MTLSize::new(1, 1, 1),
                        );
                    }

                    let seq_len = _position + 1; // attend to all positions up to and including current
                    if let Some(fa_pso) = shader.get_pipeline("flash_attn_f16_dk128") {
                        if attn_idx < bufs.kv_k_cache.len() && attn_idx < bufs.kv_v_cache.len() {
                            let scale = 1.0f32 / (head_dim as f32).sqrt();
                            // KV cache layout: [pos, kv_heads, head_dim] contiguous F16
                            // Strides in bytes:
                            let kv_pos_stride = (num_kv_heads * head_dim * 2) as u64;  // nb11: to next position
                            let kv_head_stride = (head_dim * 2) as u64;                 // nb12: to next kv_head
                            // ns10/ns20 = kv_pos_stride / 2 = num_kv_heads * head_dim (compiled as FC)
                            let ns_kv = (num_kv_heads * head_dim) as i32;

                            // Q layout: [head_dim * num_heads] F32 = [head_dim, 1_token, num_heads]
                            // nb01 = head_dim * 4 (stride between tokens within a head)
                            // nb02 = head_dim * 4 (stride between heads, since n_tokens=1)
                            let q_head_stride = (head_dim * 4) as u64;

                            // Mask: [max_seq_len] F16, mask[i]=0 for i<seq_len, -INF beyond
                            // nb31 = max_seq_len * 2 (stride between query tokens in mask)
                            let max_seq = 4096u64;
                            let mask_bytes = max_seq * 2;

                            let fa_args = kargs::FlashAttnExt {
                                ne01: 1,                           // n_tokens (decode = 1)
                                ne02: num_heads as i32,            // n_heads_q
                                ne03: 1,
                                nb01: q_head_stride,               // Q token stride (F32)
                                nb02: q_head_stride,               // Q head stride (= token stride when n_tokens=1)
                                nb03: (q_dim * 4) as u64,          // Q batch stride
                                ne11: seq_len as i32,              // KV sequence length
                                ne_12_2: num_kv_heads as i32,      // n_kv_heads
                                ne_12_3: 1,
                                ns10: ns_kv,                       // K position stride in elements (matches FC)
                                nb11: kv_pos_stride,               // K position stride in bytes
                                nb12: kv_head_stride,              // K kv_head stride in bytes
                                nb13: kv_pos_stride * max_seq,     // K batch stride (unused, ne_12_3=1)
                                ns20: ns_kv,                       // V position stride in elements (matches FC)
                                nb21: kv_pos_stride,               // V position stride in bytes
                                nb22: kv_head_stride,              // V kv_head stride in bytes
                                nb23: kv_pos_stride * max_seq,     // V batch stride
                                ne31: 1,                           // mask n_tokens dimension
                                ne32: 1,                           // mask shared across heads
                                ne33: 1,
                                nb31: mask_bytes,                  // mask stride per query token
                                nb32: 0,                           // mask head stride (shared)
                                nb33: 0,
                                ne1: num_heads as i32,             // dst heads dimension
                                ne2: 1,                            // dst n_tokens
                                ne3: 1,
                                scale,
                                max_bias: 0.0, m0: 0.0, m1: 0.0,
                                n_head_log2: 0,
                                logit_softcap: 0.0,
                            };
                            encoder.set_compute_pipeline_state(fa_pso);
                            encoder.set_bytes(0, std::mem::size_of::<kargs::FlashAttnExt>() as u64, &fa_args as *const _ as *const _);
                            encoder.set_buffer(1, Some(&bufs.attn_q), 0);              // Q [num_heads * head_dim] F32
                            encoder.set_buffer(2, Some(&bufs.kv_k_cache[attn_idx]), 0); // K cache
                            encoder.set_buffer(3, Some(&bufs.kv_v_cache[attn_idx]), 0); // V cache
                            encoder.set_buffer(4, Some(&bufs.attn_mask), 0);           // causal mask [max_seq] F16
                            encoder.set_buffer(5, Some(&bufs.attn_kv_f16), 0);         // sinks (dummy, unused)
                            encoder.set_buffer(6, Some(&bufs.attn_kv_f16), 0);         // pad (dummy, has_kvpad=false)
                            encoder.set_buffer(7, Some(&bufs.attn_blk), 0);            // blk descriptors (all 1s)
                            encoder.set_buffer(8, Some(&bufs.attn_out), 0);            // dst [num_heads * head_dim] F32

                            // Dispatch: grid = ((ne01+Q-1)/Q, ne02, ne03) = (1, num_heads, 1)
                            let nsg = 4u64;

                            // Shared memory: Q*(DK+2*PV+2*SH) + NSG*4*16*KV, in bytes (half)
                            let q_per_tg = 8u64;
                            let dk = 128u64;
                            let dv = 128u64;
                            let pv = ((dv + 63) / 64) * 64; // PAD2(DV, 64) = 128
                            let c = 64u64;
                            let sh = 2 * c;
                            let t = dk + 2 * pv;
                            let ts = 2 * sh;
                            let shmem = (q_per_tg * (t + ts) + nsg * 4 * 16 * 8) * 2;
                            encoder.set_threadgroup_memory_length(0, shmem);

                            encoder.dispatch_thread_groups(
                                metal::MTLSize::new(1, num_heads as u64, 1),
                                metal::MTLSize::new(32, nsg, 1),
                            );
                        }
                    }
                }

                // O projection: [q_dim → hidden] (quant-aware, Q5_K native)
                mm_w(&encoder, &ln.o_proj,
                    &bufs.attn_out, &bufs.hidden_b, hidden, q_dim);

                // Residual: hidden_a + hidden_b → hidden_a
                dispatch_bin_op(&encoder, pso("bin_add"), &bufs.hidden_a, &bufs.hidden_b, &bufs.hidden_a, hidden);

                attn_layer_idx += 1;
            }
        }
    }

    // Final RMS norm (fused with F32 weight multiply)
    {
        let final_norm_w = norm_f32("model.norm.weight");
        dispatch_rms_norm(&encoder, pso("kernel_rms_norm_mul_f32"),
            &bufs.hidden_a, &final_norm_w, &bufs.normed, hidden, config.rms_norm_eps);
    }

    // LM head matmul (quant-aware: Q8_0 native for Nemotron's output.weight).
    {
        let vocab = config.vocab_size.unwrap_or(131072);
        mm_w(&encoder, "lm_head.weight", &bufs.normed, &bufs.logits, vocab, hidden);
    }

    encoder.end_encoding();

    // Split timing for diagnostics
    let t_encoded = std::time::Instant::now();
    current_cb.commit();
    current_cb.wait_until_completed();
    let t_done = std::time::Instant::now();

    static CPU_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static GPU_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    CPU_NS.fetch_add((t_encoded - t_encode_start).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    GPU_NS.fetch_add((t_done - t_encoded).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n % 50 == 0 {
        eprintln!("[ggml split] {}tok avg: cpu_encode={:.2}ms gpu_wait={:.2}ms",
            n,
            CPU_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / n as f64 / 1e6,
            GPU_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / n as f64 / 1e6);
    }
    Ok(())
}

/// Update persistent SSM states after a token's CB completes.
/// Must be called after `cb.wait_until_completed()`.
///
/// 1. Copies updated scan state from `scan_full[s_off..]` to `scan_states[i]`
/// 2. Updates conv sliding windows: shifts left, inserts new xBC from `ssm_xz`
#[cfg(feature = "metal")]
pub fn update_ssm_states(
    _graph: &GgmlGraph,
    _config: &crate::inference::model::ModelConfig,
) {
    // All state updates (MoE weights, SSM conv window, SSM scan state) now happen
    // on GPU within the single command buffer. This function is retained as a no-op
    // for API compatibility with the caller in llm.rs.
}
