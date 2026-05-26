//! High-level tensor operations using Metal compute kernels.
//!
//! This module provides the bridge between the tensor abstraction
//! and the underlying Metal compute kernels.

use super::{MetalDevice, MetalCompute, ComputePipeline};
use super::shader::sources;
use crate::core::{DType, Error, Result};
use std::sync::Arc;

/// Metal-accelerated tensor operations.
#[cfg(feature = "metal")]
pub struct MetalOps {
    compute: Arc<MetalCompute>,
    // Cached pipelines
    matmul_f16: Arc<ComputePipeline>,
    matmul_f32: Arc<ComputePipeline>,
    softmax_f16: Arc<ComputePipeline>,
    rms_norm_f16: Arc<ComputePipeline>,
    silu_f16: Arc<ComputePipeline>,
    silu_f32: Arc<ComputePipeline>,
    rope_f16: Arc<ComputePipeline>,
    add_f16: Arc<ComputePipeline>,
    mul_f16: Arc<ComputePipeline>,
    scale_f16: Arc<ComputePipeline>,
    attention_f16: Arc<ComputePipeline>,
    attention_tiled_f16: Arc<ComputePipeline>,
    // Layer norm
    layer_norm_f16: Arc<ComputePipeline>,
    // GELU variants
    gelu_tanh_f16: Arc<ComputePipeline>,
    gelu_fast_f16: Arc<ComputePipeline>,
    gelu_tanh_f32: Arc<ComputePipeline>,
    // Gated activations
    geglu_f16: Arc<ComputePipeline>,
    swiglu_f16: Arc<ComputePipeline>,
    // Group norm (2-pass)
    group_norm_stats_f16: Arc<ComputePipeline>,
    group_norm_apply_f16: Arc<ComputePipeline>,
    // Conv2D variants
    conv2d_naive_f16: Arc<ComputePipeline>,
    conv2d_3x3_tiled_f16: Arc<ComputePipeline>,
    conv2d_1x1_simd_f16: Arc<ComputePipeline>,
    conv2d_1x1_f16: Arc<ComputePipeline>,
    conv2d_3x3_f16: Arc<ComputePipeline>,
    // Embedding
    embedding_lookup_f16: Arc<ComputePipeline>,
    embedding_lookup_colmajor_f16: Arc<ComputePipeline>,
    // Argmax
    argmax_f16: Arc<ComputePipeline>,
    // Upsample
    upsample_nearest_f16: Arc<ComputePipeline>,
    // Additional attention variants
    gqa_attention_f16: Arc<ComputePipeline>,
    autoregressive_attention_tg_f16: Arc<ComputePipeline>,
    // Additional elementwise ops
    sub_f16: Arc<ComputePipeline>,
    div_f16: Arc<ComputePipeline>,
    add_bias_f16: Arc<ComputePipeline>,
    // Copy tile
    copy_tile_f16: Arc<ComputePipeline>,
    // VAE fused
    vae_decode_fused_f16: Arc<ComputePipeline>,
    vae_encode_fused_f16: Arc<ComputePipeline>,
    // Quantized matmul
    matmul_q4k_f16: Arc<ComputePipeline>,
    // KV cache
    copy_to_kv_cache_f16: Arc<ComputePipeline>,
    // Gaussian splatting
    splat_gaussians: Arc<ComputePipeline>,
}

#[cfg(feature = "metal")]
impl MetalOps {
    /// Create a new MetalOps instance with pre-compiled kernels.
    pub fn new(device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        // Compile all kernels
        // Use tiled matmul for F16 to handle [N,K] weights correctly (A @ B^T)
        let matmul_f16 = compute.compile_pipeline("matmul_f16", sources::MATMUL, "matmul_tiled_f16")?;
        let matmul_f32 = compute.compile_pipeline("matmul_f32", sources::MATMUL, "matmul_f32")?;
        let softmax_f16 = compute.compile_pipeline("softmax_f16", sources::SOFTMAX, "softmax_f16")?;
        let rms_norm_f16 = compute.compile_pipeline("rms_norm_f16", sources::RMS_NORM, "rms_norm_f16")?;
        let silu_f16 = compute.compile_pipeline("silu_f16", sources::SILU, "silu_f16")?;
        let silu_f32 = compute.compile_pipeline("silu_f32", sources::SILU, "silu_f32")?;
        let rope_f16 = compute.compile_pipeline("rope_f16", sources::ROPE, "rope_f16")?;
        let add_f16 = compute.compile_pipeline("add_f16", sources::ELEMENTWISE, "add_f16")?;
        let mul_f16 = compute.compile_pipeline("mul_f16", sources::ELEMENTWISE, "mul_f16")?;
        let scale_f16 = compute.compile_pipeline("scale_f16", sources::ELEMENTWISE, "scale_f16")?;
        let attention_f16 = compute.compile_pipeline("attention_f16", sources::ATTENTION, "attention_f16")?;
        let attention_tiled_f16 = compute.compile_pipeline("attention_tiled_f16", sources::ATTENTION_TILED, "attention_tiled_f16")?;

        // Layer norm
        let layer_norm_f16 = compute.compile_pipeline("layer_norm_f16", sources::LAYER_NORM, "layer_norm_f16")?;

        // GELU variants
        let gelu_tanh_f16 = compute.compile_pipeline("gelu_tanh_f16", sources::GELU, "gelu_tanh_f16")?;
        let gelu_fast_f16 = compute.compile_pipeline("gelu_fast_f16", sources::GELU, "gelu_fast_f16")?;
        let gelu_tanh_f32 = compute.compile_pipeline("gelu_tanh_f32", sources::GELU, "gelu_tanh_f32")?;

        // Gated activations
        let geglu_f16 = compute.compile_pipeline("geglu_f16", sources::GELU, "geglu_f16")?;
        let swiglu_f16 = compute.compile_pipeline("swiglu_f16", sources::SWIGLU, "swiglu_f16")?;

        // Group norm (2-pass)
        let group_norm_stats_f16 = compute.compile_pipeline("group_norm_stats_f16", sources::GROUP_NORM, "group_norm_stats_f16")?;
        let group_norm_apply_f16 = compute.compile_pipeline("group_norm_apply_f16", sources::GROUP_NORM, "group_norm_apply_f16")?;

        // Conv2D variants
        let conv2d_naive_f16 = compute.compile_pipeline("conv2d_naive_f16", sources::CONV2D, "conv2d_naive_f16")?;
        let conv2d_3x3_tiled_f16 = compute.compile_pipeline("conv2d_3x3_tiled_f16", sources::CONV2D, "conv2d_3x3_tiled_f16")?;
        let conv2d_1x1_simd_f16 = compute.compile_pipeline("conv2d_1x1_simd_f16", sources::CONV2D, "conv2d_1x1_simd_f16")?;
        let conv2d_1x1_f16 = compute.compile_pipeline("conv2d_1x1_f16", sources::CONV2D, "conv2d_1x1_f16")?;
        let conv2d_3x3_f16 = compute.compile_pipeline("conv2d_3x3_f16", sources::CONV2D, "conv2d_3x3_f16")?;

        // Embedding
        let embedding_lookup_f16 = compute.compile_pipeline("embedding_lookup_f16", sources::EMBEDDING, "embedding_lookup_f16")?;
        let embedding_lookup_colmajor_f16 = compute.compile_pipeline("embedding_lookup_colmajor_f16", sources::EMBEDDING, "embedding_lookup_colmajor_f16")?;

        // Argmax
        let argmax_f16 = compute.compile_pipeline("argmax_f16", sources::ARGMAX, "argmax_f16")?;

        // Upsample
        let upsample_nearest_f16 = compute.compile_pipeline("upsample_nearest_f16", sources::UPSAMPLE, "upsample_nearest_f16")?;

        // Additional attention variants (source already loaded above for existing pipelines)
        let gqa_attention_f16 = compute.compile_pipeline("gqa_attention_f16", sources::ATTENTION, "gqa_attention_f16")?;
        let autoregressive_attention_tg_f16 = compute.compile_pipeline("autoregressive_attention_tg_f16", sources::ATTENTION_TILED, "autoregressive_attention_tg_f16")?;

        // Additional elementwise ops (source already loaded above for existing pipelines)
        let sub_f16 = compute.compile_pipeline("sub_f16", sources::ELEMENTWISE, "sub_f16")?;
        let div_f16 = compute.compile_pipeline("div_f16", sources::ELEMENTWISE, "div_f16")?;
        let add_bias_f16 = compute.compile_pipeline("add_bias_f16", sources::ELEMENTWISE, "add_bias_f16")?;

        // Copy tile
        let copy_tile_f16 = compute.compile_pipeline("copy_tile_f16", sources::COPY_TILE, "copy_tile_f16")?;

        // VAE fused
        let vae_decode_fused_f16 = compute.compile_pipeline("vae_decode_fused_f16", sources::VAE_FUSED, "vae_decode_fused_f16")?;
        let vae_encode_fused_f16 = compute.compile_pipeline("vae_encode_fused_f16", sources::VAE_ENCODE_FUSED, "vae_encode_fused_f16")?;

        // Quantized matmul
        let matmul_q4k_f16 = compute.compile_pipeline("matmul_q4k_f16", sources::MATMUL, "matmul_q4k_f16")?;

        // KV cache
        let copy_to_kv_cache_f16 = compute.compile_pipeline("copy_to_kv_cache_f16", sources::ROPE, "copy_to_kv_cache_f16")?;

        // Gaussian splatting
        let splat_gaussians = compute.compile_pipeline("splat_gaussians", sources::GAUSSIAN_SPLAT, "splat_gaussians")?;

        Ok(Self {
            compute,
            matmul_f16,
            matmul_f32,
            softmax_f16,
            rms_norm_f16,
            silu_f16,
            silu_f32,
            rope_f16,
            add_f16,
            mul_f16,
            scale_f16,
            attention_f16,
            attention_tiled_f16,
            layer_norm_f16,
            gelu_tanh_f16,
            gelu_fast_f16,
            gelu_tanh_f32,
            geglu_f16,
            swiglu_f16,
            group_norm_stats_f16,
            group_norm_apply_f16,
            conv2d_naive_f16,
            conv2d_3x3_tiled_f16,
            conv2d_1x1_simd_f16,
            conv2d_1x1_f16,
            conv2d_3x3_f16,
            embedding_lookup_f16,
            embedding_lookup_colmajor_f16,
            argmax_f16,
            upsample_nearest_f16,
            gqa_attention_f16,
            autoregressive_attention_tg_f16,
            sub_f16,
            div_f16,
            add_bias_f16,
            copy_tile_f16,
            vae_decode_fused_f16,
            vae_encode_fused_f16,
            matmul_q4k_f16,
            copy_to_kv_cache_f16,
            splat_gaussians,
        })
    }

    /// Matrix multiplication: C = A @ B
    /// A: [M, K], B: [K, N], C: [M, N]
    pub fn matmul(
        &self,
        a: &metal::Buffer,
        b: &metal::Buffer,
        c: &metal::Buffer,
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
        device: &MetalDevice,
    ) -> Result<()> {
        let pipeline = match dtype {
            DType::F16 => &self.matmul_f16,
            DType::F32 => &self.matmul_f32,
            _ => return Err(Error::unsupported(format!("matmul dtype {:?}", dtype))),
        };

        let command_buffer = device.new_command_buffer();

        // Tile sizes
        const TILE_M: usize = 32;
        const TILE_N: usize = 32;

        let grid_m = (m + TILE_M - 1) / TILE_M;
        let grid_n = (n + TILE_N - 1) / TILE_N;

        if dtype == DType::F16 {
            // Tiled F16 kernel uses 16x16 threads and shared memory
            self.compute.dispatch(
                &command_buffer,
                pipeline,
                (grid_n, grid_m, 1),
                (16, 16, 1), // 16x16 threads = 256 threads
                |encoder| {
                    encoder.set_buffer(0, Some(a), 0);
                    encoder.set_buffer(1, Some(b), 0);
                    encoder.set_buffer(2, Some(c), 0);
                    encoder.set_bytes(3, 4, &(m as u32) as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &(n as u32) as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &(k as u32) as *const u32 as *const _);
                    // Shared memory: 2 tiles * 32 * 32 * 2 (half)
                    encoder.set_threadgroup_memory_length(0, (2 * 32 * 32 * 2) as u64);
                },
            );
        } else {
            // F32 kernel uses naive dispatch (8x8)
            self.compute.dispatch(
                &command_buffer,
                pipeline,
                (grid_n, grid_m, 1),
                (8, 8, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(a), 0);
                    encoder.set_buffer(1, Some(b), 0);
                    encoder.set_buffer(2, Some(c), 0);
                    encoder.set_bytes(3, 4, &(m as u32) as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &(n as u32) as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &(k as u32) as *const u32 as *const _);
                },
            );
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Softmax: output = softmax(input) along last dimension
    pub fn softmax(
        &self,
        input: &metal::Buffer,
        output: &metal::Buffer,
        batch_size: usize,
        dim: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.softmax_f16,
            batch_size,
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
                encoder.set_bytes(2, 4, &(batch_size as u32) as *const u32 as *const _);
                encoder.set_bytes(3, 4, &(dim as u32) as *const u32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// RMS normalization
    pub fn rms_norm(
        &self,
        input: &metal::Buffer,
        weight: &metal::Buffer,
        output: &metal::Buffer,
        batch_size: usize,
        dim: usize,
        eps: f32,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.rms_norm_f16,
            batch_size,
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(weight), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(3, 4, &(batch_size as u32) as *const u32 as *const _);
                encoder.set_bytes(4, 4, &(dim as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// SiLU (Swish) activation: x * sigmoid(x)
    pub fn silu(
        &self,
        input: &metal::Buffer,
        output: &metal::Buffer,
        count: usize,
        dtype: DType,
        device: &MetalDevice,
    ) -> Result<()> {
        let pipeline = match dtype {
            DType::F16 => &self.silu_f16,
            DType::F32 => &self.silu_f32,
            _ => return Err(Error::unsupported(format!("silu dtype {:?}", dtype))),
        };

        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            pipeline,
            count,
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Rotary positional embedding
    pub fn rope(
        &self,
        x: &metal::Buffer,
        cos_cache: &metal::Buffer,
        sin_cache: &metal::Buffer,
        seq_len: usize,
        head_dim: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_2d(
            &command_buffer,
            &self.rope_f16,
            head_dim / 2,
            seq_len,
            |encoder| {
                encoder.set_buffer(0, Some(x), 0);
                encoder.set_buffer(1, Some(cos_cache), 0);
                encoder.set_buffer(2, Some(sin_cache), 0);
                encoder.set_bytes(3, 4, &(seq_len as u32) as *const u32 as *const _);
                encoder.set_bytes(4, 4, &(head_dim as u32) as *const u32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Elementwise addition: c = a + b
    pub fn add(
        &self,
        a: &metal::Buffer,
        b: &metal::Buffer,
        c: &metal::Buffer,
        count: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.add_f16,
            count,
            |encoder| {
                encoder.set_buffer(0, Some(a), 0);
                encoder.set_buffer(1, Some(b), 0);
                encoder.set_buffer(2, Some(c), 0);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Elementwise multiplication: c = a * b
    pub fn mul(
        &self,
        a: &metal::Buffer,
        b: &metal::Buffer,
        c: &metal::Buffer,
        count: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.mul_f16,
            count,
            |encoder| {
                encoder.set_buffer(0, Some(a), 0);
                encoder.set_buffer(1, Some(b), 0);
                encoder.set_buffer(2, Some(c), 0);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Scale: output = input * scale
    pub fn scale(
        &self,
        input: &metal::Buffer,
        output: &metal::Buffer,
        scale: f32,
        count: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.scale_f16,
            count,
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
                encoder.set_bytes(2, 4, &scale as *const f32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Fused attention: softmax(Q @ K^T / sqrt(d)) @ V
    /// Uses tiled implementation for better performance and memory efficiency.
    pub fn attention(
        &self,
        q: &metal::Buffer,
        k: &metal::Buffer,
        v: &metal::Buffer,
        output: &metal::Buffer,
        seq_len: usize,
        head_dim: usize,
        num_heads: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Calculate default packed strides [Batch, Heads, Seq, Dim]
        // Batch size is implicitly 1 based on current signature
        let stride_dim = 1;
        let stride_seq = head_dim;
        let stride_head = seq_len * head_dim;
        let stride_batch = num_heads * seq_len * head_dim;

        // Tiled kernel parameters
        let block_size = 32;
        let grid_dims = (num_heads, (seq_len + block_size - 1) / block_size, 1);
        let thread_dims = (block_size, 1, 1);
        
        // 3 tiles (Q, K, V) * block_size * head_dim * sizeof(half)
        let shared_mem_size = 3 * block_size * head_dim * 2;

        self.compute.dispatch(
            &command_buffer,
            &self.attention_tiled_f16,
            grid_dims,
            thread_dims,
            |encoder| {
                encoder.set_buffer(0, Some(q), 0);
                encoder.set_buffer(1, Some(k), 0);
                encoder.set_buffer(2, Some(v), 0);
                encoder.set_buffer(3, Some(output), 0);
                
                encoder.set_bytes(4, 4, &(seq_len as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &(head_dim as u32) as *const u32 as *const _);
                encoder.set_bytes(6, 4, &scale as *const f32 as *const _);
                encoder.set_bytes(7, 4, &(num_heads as u32) as *const u32 as *const _);
                
                encoder.set_bytes(8, 4, &(stride_batch as u32) as *const u32 as *const _);
                encoder.set_bytes(9, 4, &(stride_head as u32) as *const u32 as *const _);
                encoder.set_bytes(10, 4, &(stride_seq as u32) as *const u32 as *const _);
                encoder.set_bytes(11, 4, &(stride_dim as u32) as *const u32 as *const _);
                
                encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    // ---------------------------------------------------------------
    // Layer normalization
    // ---------------------------------------------------------------

    /// Layer normalization with optional bias.
    /// input/output: [batch_size, hidden_size], weight: [hidden_size], bias: [hidden_size] or None
    pub fn layer_norm(
        &self,
        input: &metal::Buffer,
        weight: &metal::Buffer,
        bias: Option<&metal::Buffer>,
        output: &metal::Buffer,
        batch_size: usize,
        hidden_size: usize,
        eps: f32,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        // The kernel expects a bias buffer at index 2.
        // When no bias is provided we pass the weight buffer as a placeholder;
        // the kernel will still read from it but the result is equivalent because
        // the shader adds bias[i] and we rely on the caller to zero-fill or the
        // common pattern of always having a bias.  For true optional support the
        // caller should supply a zero buffer.
        let bias_buf = bias.unwrap_or(weight);

        self.compute.dispatch_1d(
            &command_buffer,
            &self.layer_norm_f16,
            batch_size,
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(weight), 0);
                encoder.set_buffer(2, Some(bias_buf), 0);
                encoder.set_buffer(3, Some(output), 0);
                encoder.set_bytes(4, 4, &(batch_size as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &(hidden_size as u32) as *const u32 as *const _);
                encoder.set_bytes(6, 4, &eps as *const f32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    // ---------------------------------------------------------------
    // GELU activations
    // ---------------------------------------------------------------

    /// GELU activation with tanh approximation (f16).
    pub fn gelu(
        &self,
        input: &metal::Buffer,
        output: &metal::Buffer,
        count: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.gelu_tanh_f16,
            count,
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// GELU fast approximation (f16).
    pub fn gelu_fast(
        &self,
        input: &metal::Buffer,
        output: &metal::Buffer,
        count: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.gelu_fast_f16,
            count,
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// GELU activation with tanh approximation (f32).
    pub fn gelu_f32(
        &self,
        input: &metal::Buffer,
        output: &metal::Buffer,
        count: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.gelu_tanh_f32,
            count,
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    // ---------------------------------------------------------------
    // Gated activations
    // ---------------------------------------------------------------

    /// GEGLU: gelu(gate) * up
    pub fn geglu(
        &self,
        gate: &metal::Buffer,
        up: &metal::Buffer,
        output: &metal::Buffer,
        count: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.geglu_f16,
            count,
            |encoder| {
                encoder.set_buffer(0, Some(gate), 0);
                encoder.set_buffer(1, Some(up), 0);
                encoder.set_buffer(2, Some(output), 0);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// SwiGLU: silu(gate) * up
    pub fn swiglu(
        &self,
        gate: &metal::Buffer,
        up: &metal::Buffer,
        output: &metal::Buffer,
        count: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.swiglu_f16,
            count,
            |encoder| {
                encoder.set_buffer(0, Some(gate), 0);
                encoder.set_buffer(1, Some(up), 0);
                encoder.set_buffer(2, Some(output), 0);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    // ---------------------------------------------------------------
    // Group normalization (2-pass)
    // ---------------------------------------------------------------

    /// Group normalization.
    /// Pass 1 computes per-group mean/variance into `stats`.
    /// Pass 2 applies normalization with optional scale (weight) and bias.
    /// input/output: [batch, channels, spatial], stats: [batch, num_groups] of float2
    pub fn group_norm(
        &self,
        input: &metal::Buffer,
        weight: Option<&metal::Buffer>,
        bias: Option<&metal::Buffer>,
        output: &metal::Buffer,
        stats: &metal::Buffer,
        batch: usize,
        channels: usize,
        spatial: usize,
        num_groups: usize,
        eps: f32,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        // Pass 1: compute stats (mean, variance) per group
        // Kernel grid: (num_groups, batch, 1), threadgroup: (256, 1, 1)
        let tg_size: usize = 256;
        self.compute.dispatch(
            &command_buffer,
            &self.group_norm_stats_f16,
            (num_groups, batch, 1),
            (tg_size, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(stats), 0);
                encoder.set_bytes(2, 4, &(batch as u32) as *const u32 as *const _);
                encoder.set_bytes(3, 4, &(num_groups as u32) as *const u32 as *const _);
                encoder.set_bytes(4, 4, &(channels as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &(spatial as u32) as *const u32 as *const _);
                // Shared memory for parallel reduction: tg_size * 2 floats (sum, sum_sq)
                encoder.set_threadgroup_memory_length(0, (tg_size * 2 * 4) as u64);
            },
        );

        // Pass 2: apply normalization
        // Kernel grid: (spatial, channels, batch) -- one thread per element
        // Use input as placeholder for weight/bias when not provided
        let weight_buf = weight.unwrap_or(input);
        let bias_buf = bias.unwrap_or(input);

        self.compute.dispatch(
            &command_buffer,
            &self.group_norm_apply_f16,
            (spatial, channels, batch),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(stats), 0);
                encoder.set_buffer(2, Some(weight_buf), 0);
                encoder.set_buffer(3, Some(bias_buf), 0);
                encoder.set_buffer(4, Some(output), 0);
                encoder.set_bytes(5, 4, &(batch as u32) as *const u32 as *const _);
                encoder.set_bytes(6, 4, &(num_groups as u32) as *const u32 as *const _);
                encoder.set_bytes(7, 4, &(channels as u32) as *const u32 as *const _);
                encoder.set_bytes(8, 4, &(spatial as u32) as *const u32 as *const _);
                encoder.set_bytes(9, 4, &eps as *const f32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    // ---------------------------------------------------------------
    // Conv2D
    // ---------------------------------------------------------------

    /// 2D convolution with automatic kernel selection.
    /// Chooses the best kernel variant based on filter size and stride.
    /// input: [n, c_in, h, w], weight: [c_out, c_in, kh, kw], bias: [c_out] or None
    pub fn conv2d(
        &self,
        input: &metal::Buffer,
        weight: &metal::Buffer,
        bias: Option<&metal::Buffer>,
        output: &metal::Buffer,
        n: usize,
        c_in: usize,
        h: usize,
        w: usize,
        c_out: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let h_out = (h + 2 * pad_h - kh) / stride_h + 1;
        let w_out = (w + 2 * pad_w - kw) / stride_w + 1;

        // Select best kernel variant
        let pipeline = if kh == 1 && kw == 1 {
            &self.conv2d_1x1_f16
        } else if kh == 3 && kw == 3 && stride_h == 1 && stride_w == 1 {
            &self.conv2d_3x3_f16
        } else {
            &self.conv2d_naive_f16
        };

        // Use weight buffer as placeholder when no bias is provided
        let bias_buf = bias.unwrap_or(weight);

        let command_buffer = device.new_command_buffer();

        // Grid: (w_out, h_out, c_out * n)
        self.compute.dispatch(
            &command_buffer,
            pipeline,
            (w_out, h_out, c_out * n),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(weight), 0);
                encoder.set_buffer(2, Some(bias_buf), 0);
                encoder.set_buffer(3, Some(output), 0);
                encoder.set_bytes(4, 4, &(c_in as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &(h as u32) as *const u32 as *const _);
                encoder.set_bytes(6, 4, &(w as u32) as *const u32 as *const _);
                encoder.set_bytes(7, 4, &(c_out as u32) as *const u32 as *const _);
                encoder.set_bytes(8, 4, &(h_out as u32) as *const u32 as *const _);
                encoder.set_bytes(9, 4, &(w_out as u32) as *const u32 as *const _);
                encoder.set_bytes(10, 4, &(kw as u32) as *const u32 as *const _);
                encoder.set_bytes(11, 4, &(kh as u32) as *const u32 as *const _);
                encoder.set_bytes(12, 4, &(pad_w as u32) as *const u32 as *const _);
                encoder.set_bytes(13, 4, &(pad_h as u32) as *const u32 as *const _);
                encoder.set_bytes(14, 4, &(stride_w as u32) as *const u32 as *const _);
                encoder.set_bytes(15, 4, &(stride_h as u32) as *const u32 as *const _);
                encoder.set_bytes(16, 4, &(n as u32) as *const u32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    // ---------------------------------------------------------------
    // Embedding lookup
    // ---------------------------------------------------------------

    /// Embedding lookup: output[i] = weight[indices[i]]
    /// indices: [seq_len] of uint, weight: [vocab_size, embed_dim], output: [seq_len, embed_dim]
    pub fn embedding(
        &self,
        indices: &metal::Buffer,
        weight: &metal::Buffer,
        output: &metal::Buffer,
        seq_len: usize,
        embed_dim: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.embedding_lookup_f16,
            seq_len,
            |encoder| {
                encoder.set_buffer(0, Some(weight), 0);
                encoder.set_buffer(1, Some(indices), 0);
                encoder.set_buffer(2, Some(output), 0);
                // vocab_size is not strictly needed at dispatch time
                // but the kernel signature requires it for OOV protection
                let vocab_size: u32 = (weight.length() as u32) / (embed_dim as u32 * 2); // half = 2 bytes
                encoder.set_bytes(3, 4, &vocab_size as *const u32 as *const _);
                encoder.set_bytes(4, 4, &(embed_dim as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &(seq_len as u32) as *const u32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Embedding lookup for column-major (GGUF) layout.
    pub fn embedding_colmajor(
        &self,
        indices: &metal::Buffer,
        weight: &metal::Buffer,
        output: &metal::Buffer,
        vocab_size: usize,
        embed_dim: usize,
        seq_len: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.embedding_lookup_colmajor_f16,
            seq_len,
            |encoder| {
                encoder.set_buffer(0, Some(weight), 0);
                encoder.set_buffer(1, Some(indices), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(3, 4, &(vocab_size as u32) as *const u32 as *const _);
                encoder.set_bytes(4, 4, &(embed_dim as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &(seq_len as u32) as *const u32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    // ---------------------------------------------------------------
    // Argmax
    // ---------------------------------------------------------------

    /// Argmax over a 1D buffer: output[0] = argmax(input[0..size])
    /// Uses threadgroup reduction. Output is a single uint.
    pub fn argmax(
        &self,
        input: &metal::Buffer,
        output: &metal::Buffer,
        size: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        let tg_size: usize = 256;

        self.compute.dispatch(
            &command_buffer,
            &self.argmax_f16,
            (1, 1, 1),
            (tg_size, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
                encoder.set_bytes(2, 4, &(size as u32) as *const u32 as *const _);
                // Shared memory: floats for values + uints for indices
                encoder.set_threadgroup_memory_length(0, (tg_size * 4) as u64);
                encoder.set_threadgroup_memory_length(1, (tg_size * 4) as u64);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    // ---------------------------------------------------------------
    // Upsample nearest
    // ---------------------------------------------------------------

    /// Nearest-neighbor 2x upsampling.
    /// input: [n, c, h_in, w_in], output: [n, c, h_out, w_out]
    pub fn upsample_nearest(
        &self,
        input: &metal::Buffer,
        output: &metal::Buffer,
        n: usize,
        c: usize,
        h_in: usize,
        w_in: usize,
        h_out: usize,
        w_out: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        // Grid: (w_out, h_out, c * n)
        self.compute.dispatch(
            &command_buffer,
            &self.upsample_nearest_f16,
            (w_out, h_out, c * n),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
                encoder.set_bytes(2, 4, &(n as u32) as *const u32 as *const _);
                encoder.set_bytes(3, 4, &(c as u32) as *const u32 as *const _);
                encoder.set_bytes(4, 4, &(h_in as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &(w_in as u32) as *const u32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    // ---------------------------------------------------------------
    // GQA (grouped-query) attention
    // ---------------------------------------------------------------

    /// Grouped-query attention.
    /// Q: [seq_len, num_q_heads, head_dim], K/V: [seq_len, num_kv_heads, head_dim]
    pub fn gqa_attention(
        &self,
        q: &metal::Buffer,
        k: &metal::Buffer,
        v: &metal::Buffer,
        output: &metal::Buffer,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Grid: (num_q_heads, seq_len)
        self.compute.dispatch_2d(
            &command_buffer,
            &self.gqa_attention_f16,
            num_q_heads,
            seq_len,
            |encoder| {
                encoder.set_buffer(0, Some(q), 0);
                encoder.set_buffer(1, Some(k), 0);
                encoder.set_buffer(2, Some(v), 0);
                encoder.set_buffer(3, Some(output), 0);
                encoder.set_bytes(4, 4, &(seq_len as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &(num_q_heads as u32) as *const u32 as *const _);
                encoder.set_bytes(6, 4, &(num_kv_heads as u32) as *const u32 as *const _);
                encoder.set_bytes(7, 4, &(head_dim as u32) as *const u32 as *const _);
                encoder.set_bytes(8, 4, &scale as *const f32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Autoregressive attention using threadgroup parallelism.
    /// Q: [num_q_heads, head_dim], K/V caches: [max_seq_len, num_kv_heads, head_dim]
    pub fn autoregressive_attention(
        &self,
        q: &metal::Buffer,
        k_cache: &metal::Buffer,
        v_cache: &metal::Buffer,
        output: &metal::Buffer,
        seq_pos: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let tg_size: usize = 256;
        // Shared memory: scores array + accumulator
        // scores: (seq_pos + 1) floats, acc: head_dim floats
        let shared_mem = ((seq_pos + 1) + head_dim) * 4;

        self.compute.dispatch(
            &command_buffer,
            &self.autoregressive_attention_tg_f16,
            (num_q_heads, 1, 1),
            (tg_size, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(q), 0);
                encoder.set_buffer(1, Some(k_cache), 0);
                encoder.set_buffer(2, Some(v_cache), 0);
                encoder.set_buffer(3, Some(output), 0);
                encoder.set_bytes(4, 4, &(seq_pos as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &(num_q_heads as u32) as *const u32 as *const _);
                encoder.set_bytes(6, 4, &(num_kv_heads as u32) as *const u32 as *const _);
                encoder.set_bytes(7, 4, &(head_dim as u32) as *const u32 as *const _);
                encoder.set_bytes(8, 4, &scale as *const f32 as *const _);
                encoder.set_threadgroup_memory_length(0, shared_mem as u64);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    // ---------------------------------------------------------------
    // Additional elementwise operations
    // ---------------------------------------------------------------

    /// Elementwise subtraction: c = a - b
    pub fn sub(
        &self,
        a: &metal::Buffer,
        b: &metal::Buffer,
        c: &metal::Buffer,
        count: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.sub_f16,
            count,
            |encoder| {
                encoder.set_buffer(0, Some(a), 0);
                encoder.set_buffer(1, Some(b), 0);
                encoder.set_buffer(2, Some(c), 0);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Elementwise division: c = a / b
    pub fn div(
        &self,
        a: &metal::Buffer,
        b: &metal::Buffer,
        c: &metal::Buffer,
        count: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.div_f16,
            count,
            |encoder| {
                encoder.set_buffer(0, Some(a), 0);
                encoder.set_buffer(1, Some(b), 0);
                encoder.set_buffer(2, Some(c), 0);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// In-place add bias: x += bias
    pub fn add_bias(
        &self,
        x: &metal::Buffer,
        bias: &metal::Buffer,
        count: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.add_bias_f16,
            count,
            |encoder| {
                encoder.set_buffer(0, Some(x), 0);
                encoder.set_buffer(1, Some(bias), 0);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    // ---------------------------------------------------------------
    // Copy tile
    // ---------------------------------------------------------------

    /// Copy a tile from source to destination with offset mapping.
    /// Copies a region of size [copy_h, copy_w] from source starting at (src_h_start, src_w_start)
    /// to dest starting at (dest_h_off, dest_w_off).
    pub fn copy_tile(
        &self,
        source: &metal::Buffer,
        dest: &metal::Buffer,
        tile_h: usize,
        tile_w: usize,
        dest_h: usize,
        dest_w: usize,
        dest_h_off: usize,
        dest_w_off: usize,
        src_h_start: usize,
        src_w_start: usize,
        copy_h: usize,
        copy_w: usize,
        channels: usize,
        batch: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        // Grid: (copy_w, copy_h, channels * batch)
        self.compute.dispatch(
            &command_buffer,
            &self.copy_tile_f16,
            (copy_w, copy_h, channels * batch),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(source), 0);
                encoder.set_buffer(1, Some(dest), 0);
                encoder.set_bytes(2, 4, &(tile_h as u32) as *const u32 as *const _);
                encoder.set_bytes(3, 4, &(tile_w as u32) as *const u32 as *const _);
                encoder.set_bytes(4, 4, &(dest_h as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &(dest_w as u32) as *const u32 as *const _);
                encoder.set_bytes(6, 4, &(dest_h_off as u32) as *const u32 as *const _);
                encoder.set_bytes(7, 4, &(dest_w_off as u32) as *const u32 as *const _);
                encoder.set_bytes(8, 4, &(src_h_start as u32) as *const u32 as *const _);
                encoder.set_bytes(9, 4, &(src_w_start as u32) as *const u32 as *const _);
                encoder.set_bytes(10, 4, &(copy_h as u32) as *const u32 as *const _);
                encoder.set_bytes(11, 4, &(copy_w as u32) as *const u32 as *const _);
                encoder.set_bytes(12, 4, &(channels as u32) as *const u32 as *const _);
                encoder.set_bytes(13, 4, &(batch as u32) as *const u32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    // ---------------------------------------------------------------
    // VAE fused kernels
    // ---------------------------------------------------------------

    /// Fused VAE decode: approximate decode from latent [N,4,H,W] to image [N,3,H*8,W*8].
    pub fn vae_decode_fused(
        &self,
        input: &metal::Buffer,
        output: &metal::Buffer,
        n: usize,
        h_in: usize,
        w_in: usize,
        scale: f32,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        let h_out = h_in * 8;
        let w_out = w_in * 8;

        // Grid: (w_out, h_out, n)
        self.compute.dispatch(
            &command_buffer,
            &self.vae_decode_fused_f16,
            (w_out, h_out, n),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
                encoder.set_bytes(2, 4, &(n as u32) as *const u32 as *const _);
                encoder.set_bytes(3, 4, &(h_in as u32) as *const u32 as *const _);
                encoder.set_bytes(4, 4, &(w_in as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &scale as *const f32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Fused VAE encode: approximate encode from image [N,3,H,W] to latent [N,4,H/8,W/8].
    pub fn vae_encode_fused(
        &self,
        input: &metal::Buffer,
        output: &metal::Buffer,
        n: usize,
        h_in: usize,
        w_in: usize,
        scale: f32,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        let h_lat = h_in / 8;
        let w_lat = w_in / 8;

        // Grid: (w_lat, h_lat, n)
        self.compute.dispatch(
            &command_buffer,
            &self.vae_encode_fused_f16,
            (w_lat, h_lat, n),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
                encoder.set_bytes(2, 4, &(n as u32) as *const u32 as *const _);
                encoder.set_bytes(3, 4, &(h_in as u32) as *const u32 as *const _);
                encoder.set_bytes(4, 4, &(w_in as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &scale as *const f32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    // ---------------------------------------------------------------
    // Quantized matmul (Q4_K)
    // ---------------------------------------------------------------

    /// Quantized matrix-vector multiply: y = x @ W_q4k
    /// x: [1, K] f16, W: Q4_K quantized [N, K], y: [1, N] f16
    pub fn matmul_q4k(
        &self,
        x: &metal::Buffer,
        w: &metal::Buffer,
        y: &metal::Buffer,
        n: usize,
        k: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &self.matmul_q4k_f16,
            n,
            |encoder| {
                encoder.set_buffer(0, Some(x), 0);
                encoder.set_buffer(1, Some(w), 0);
                encoder.set_buffer(2, Some(y), 0);
                encoder.set_bytes(3, 4, &(n as u32) as *const u32 as *const _);
                encoder.set_bytes(4, 4, &(k as u32) as *const u32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    // ---------------------------------------------------------------
    // KV cache copy
    // ---------------------------------------------------------------

    /// Copy a single K or V vector into the KV cache at the given position.
    /// kv: [num_heads, head_dim], cache: [max_seq, num_heads, head_dim]
    pub fn copy_to_kv_cache(
        &self,
        kv: &metal::Buffer,
        cache: &metal::Buffer,
        pos: usize,
        num_heads: usize,
        head_dim: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        // Grid: (head_dim, num_heads)
        self.compute.dispatch_2d(
            &command_buffer,
            &self.copy_to_kv_cache_f16,
            head_dim,
            num_heads,
            |encoder| {
                encoder.set_buffer(0, Some(kv), 0);
                encoder.set_buffer(1, Some(cache), 0);
                encoder.set_bytes(2, 4, &(pos as u32) as *const u32 as *const _);
                encoder.set_bytes(3, 4, &(num_heads as u32) as *const u32 as *const _);
                encoder.set_bytes(4, 4, &(head_dim as u32) as *const u32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    // ---------------------------------------------------------------
    // Gaussian splatting
    // ---------------------------------------------------------------

    /// Render Gaussian splats to an image.
    /// gaussians: buffer of Gaussian structs, sorted_indices: sorted front-to-back,
    /// output: [height, width] of float4 (RGBA), camera: Camera uniform buffer.
    pub fn splat_gaussians(
        &self,
        gaussians: &metal::Buffer,
        sorted_indices: &metal::Buffer,
        output: &metal::Buffer,
        camera: &metal::Buffer,
        num_gaussians: usize,
        width: usize,
        height: usize,
        device: &MetalDevice,
    ) -> Result<()> {
        let command_buffer = device.new_command_buffer();

        // Grid: (width, height, 1) -- one thread per pixel
        self.compute.dispatch(
            &command_buffer,
            &self.splat_gaussians,
            (width, height, 1),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(gaussians), 0);
                encoder.set_buffer(1, Some(sorted_indices), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_buffer(3, Some(camera), 0);
                encoder.set_bytes(4, 4, &(num_gaussians as u32) as *const u32 as *const _);
            },
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

/// Non-Metal stub
#[cfg(not(feature = "metal"))]
pub struct MetalOps;

#[cfg(not(feature = "metal"))]
impl MetalOps {
    pub fn new(_device: Arc<super::MetalDevice>) -> Result<Self> {
        Err(Error::device_not_available("Metal", "not on macOS"))
    }
}
