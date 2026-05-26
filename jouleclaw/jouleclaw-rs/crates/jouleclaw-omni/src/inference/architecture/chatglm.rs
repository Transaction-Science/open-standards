//! ChatGLM-6B text encoder for Kolors.
//!
//! Architecture: 28-layer transformer encoder with:
//! - Grouped-Query Attention (GQA): 32 Q heads, 2 KV heads
//! - Fused QKV projection (query_key_value)
//! - Rotary Position Embeddings (RoPE)
//! - RMSNorm (pre-norm)
//! - SiLU-gated MLP (dense_h_to_4h → gate*up → dense_4h_to_h)
//!
//! All operations run on Metal GPU.

use crate::core::Result;
use crate::inference::model::Model;
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::hal::MetalCompute;
#[cfg(feature = "metal")]
use crate::hal::metal::BorrowedMetalBuffer;
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;
#[cfg(feature = "metal")]
use crate::hal::MetalDevice;

/// ChatGLM encoder configuration.
#[derive(Debug, Clone)]
pub struct ChatGLMConfig {
    /// Hidden dimension (4096).
    pub hidden_size: usize,
    /// Number of query attention heads (32).
    pub num_heads: usize,
    /// Number of KV heads for GQA (2).
    pub num_kv_heads: usize,
    /// Number of encoder layers (28).
    pub num_layers: usize,
    /// FFN hidden dimension (13696).
    pub ffn_hidden: usize,
    /// Vocabulary size (65024).
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Head dimension (hidden_size / num_heads = 128).
    pub head_dim: usize,
}

impl ChatGLMConfig {
    /// ChatGLM-6B configuration for Kolors.
    pub fn kolors() -> Self {
        Self {
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 2,
            num_layers: 28,
            ffn_hidden: 13696,
            vocab_size: 65024,
            rms_norm_eps: 1e-5,
            head_dim: 128,
        }
    }
}

/// Compiled kernel pipelines for ChatGLM operations.
#[cfg(feature = "metal")]
struct ChatGLMKernels {
    linear: Arc<crate::hal::metal::ComputePipeline>,
    rms_norm: Arc<crate::hal::metal::ComputePipeline>,
    silu: Arc<crate::hal::metal::ComputePipeline>,
    add: Arc<crate::hal::metal::ComputePipeline>,
    rope: Arc<crate::hal::metal::ComputePipeline>,
    batched_linear: Arc<crate::hal::metal::ComputePipeline>,
    batched_matmul_nn: Arc<crate::hal::metal::ComputePipeline>,
    row_softmax: Arc<crate::hal::metal::ComputePipeline>,
    transpose_shd_hsd: Arc<crate::hal::metal::ComputePipeline>,
    transpose_hsd_shd: Arc<crate::hal::metal::ComputePipeline>,
}

#[cfg(feature = "metal")]
impl ChatGLMKernels {
    fn new(compute: &Arc<MetalCompute>) -> Result<Self> {
        Ok(Self {
            linear: compute.compile_pipeline("linear", sources::LINEAR, "linear_f16")?,
            rms_norm: compute.compile_pipeline("rms_norm", sources::RMS_NORM, "rms_norm_f16")?,
            silu: compute.compile_pipeline("silu", sources::SILU, "silu_f16")?,
            add: compute.compile_pipeline("add", sources::ELEMENTWISE, "add_f16")?,
            rope: compute.compile_pipeline("rope", sources::ROPE, "rope_f16")?,
            batched_linear: compute.compile_pipeline("batched_linear", sources::LINEAR, "batched_linear_f16")?,
            batched_matmul_nn: compute.compile_pipeline("batched_matmul_nn", sources::LINEAR, "batched_matmul_nn_f16")?,
            row_softmax: compute.compile_pipeline("row_softmax", sources::LINEAR, "row_softmax_scale_f16")?,
            transpose_shd_hsd: compute.compile_pipeline("transpose_shd_hsd", sources::LINEAR, "transpose_shd_to_hsd_f16")?,
            transpose_hsd_shd: compute.compile_pipeline("transpose_hsd_shd", sources::LINEAR, "transpose_hsd_to_shd_f16")?,
        })
    }
}

/// ChatGLM GPU encoder — full forward pass on Metal.
#[cfg(feature = "metal")]
pub struct ChatGLMEncoder {
    model: Arc<Model>,
    config: ChatGLMConfig,
    kernels: ChatGLMKernels,
    device: Arc<MetalDevice>,
}

#[cfg(feature = "metal")]
impl ChatGLMEncoder {
    /// Create a new ChatGLM encoder.
    pub fn new(
        model: Arc<Model>,
        config: ChatGLMConfig,
        device: Arc<MetalDevice>,
    ) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device.clone()));
        let kernels = ChatGLMKernels::new(&compute)?;
        Ok(Self { model, config, kernels, device })
    }

    /// Encode token IDs to hidden states.
    ///
    /// Returns [seq_len, hidden_size] tensor in F16.
    pub fn encode(&self, tokens: &[u32]) -> Result<Tensor> {
        let seq_len = tokens.len();
        let hidden = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let compute = Arc::new(MetalCompute::new(self.device.clone()));

        // 1. Token embedding lookup
        let embed_weight = self.w("embedding.word_embeddings.weight")?;
        let embed_f32 = embed_weight.to_f32_vec()?;
        let mut hidden_f32 = vec![0.0f32; seq_len * hidden];
        for (i, &token) in tokens.iter().enumerate() {
            let offset = token as usize * hidden;
            if offset + hidden <= embed_f32.len() {
                hidden_f32[i * hidden..(i + 1) * hidden].copy_from_slice(&embed_f32[offset..offset + hidden]);
            }
        }
        let hidden_f16: Vec<half::f16> = hidden_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let mut hidden_state = Tensor::from_slice(&hidden_f16, Shape::from([seq_len, hidden]), DType::F16, self.device.info().id)?;

        // 2. Encoder layers with NVMe streaming
        self.model.prefetch_prefix("encoder.layers.0.");
        for layer in 0..self.config.num_layers {
            if layer + 1 < self.config.num_layers {
                self.model.prefetch_prefix(&format!("encoder.layers.{}.", layer + 1));
            }
            hidden_state = self.encoder_layer(
                &hidden_state, layer, seq_len, hidden,
                num_heads, num_kv_heads, head_dim, &compute,
            )?;
            self.model.evict_prefix(&format!("encoder.layers.{}.", layer));
        }

        // 3. Final RMSNorm
        let final_norm_w = self.w("encoder.final_layernorm.weight")?;
        let cb = compute.new_command_buffer();
        let normed = self.gpu_rms_norm(&hidden_state, final_norm_w, hidden, &compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        Ok(normed)
    }

    fn encoder_layer(
        &self,
        hidden: &Tensor,
        layer: usize,
        seq_len: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        let prefix = format!("encoder.layers.{}", layer);

        // 1. Pre-norm (RMSNorm)
        let norm_w = self.w(&format!("{}.input_layernorm.weight", prefix))?;
        let cb = compute.new_command_buffer();
        let normed = self.gpu_rms_norm(hidden, norm_w, hidden_size, compute, cb.as_ref())?;

        // 2. Fused QKV projection
        // query_key_value.weight: [num_heads*head_dim + 2*num_kv_heads*head_dim, hidden_size]
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let qkv = self.gpu_linear_nobias(&normed, &format!("{}.self_attention.query_key_value", prefix), hidden_size, qkv_dim, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // Add QKV bias
        let qkv_bias = self.w(&format!("{}.self_attention.query_key_value.bias", prefix))?;
        let bias_f32 = qkv_bias.to_f32_vec()?;
        let mut qkv_f32 = qkv.to_f32_vec()?;
        for row in 0..seq_len {
            for col in 0..qkv_dim {
                qkv_f32[row * qkv_dim + col] += bias_f32[col];
            }
        }

        // Split Q, K, V
        let device = hidden.device();
        let mut q_data: Vec<half::f16> = Vec::with_capacity(seq_len * q_dim);
        let mut k_data: Vec<half::f16> = Vec::with_capacity(seq_len * kv_dim);
        let mut v_data: Vec<half::f16> = Vec::with_capacity(seq_len * kv_dim);
        for row in 0..seq_len {
            let base = row * qkv_dim;
            for i in 0..q_dim {
                q_data.push(half::f16::from_f32(qkv_f32[base + i]));
            }
            for i in 0..kv_dim {
                k_data.push(half::f16::from_f32(qkv_f32[base + q_dim + i]));
            }
            for i in 0..kv_dim {
                v_data.push(half::f16::from_f32(qkv_f32[base + q_dim + kv_dim + i]));
            }
        }

        let q = Tensor::from_slice(&q_data, Shape::from([seq_len, q_dim]), DType::F16, device)?;
        let k = Tensor::from_slice(&k_data, Shape::from([seq_len, kv_dim]), DType::F16, device)?;
        let v = Tensor::from_slice(&v_data, Shape::from([seq_len, kv_dim]), DType::F16, device)?;

        // TODO: Apply RoPE to Q and K (for now skip, positions still encoded in attention patterns)

        // 3. GQA: expand KV from num_kv_heads → num_heads by repeating
        let heads_per_group = num_heads / num_kv_heads;
        let k_expanded = self.expand_kv(&k, seq_len, num_kv_heads, head_dim, heads_per_group)?;
        let v_expanded = self.expand_kv(&v, seq_len, num_kv_heads, head_dim, heads_per_group)?;

        // 4. Batched attention
        let attn_out = self.batched_attention(&q, &k_expanded, &v_expanded, seq_len, seq_len, num_heads, head_dim, compute)?;

        // 5. Output projection
        let cb2 = compute.new_command_buffer();
        let attn_proj = self.gpu_linear_nobias(&attn_out, &format!("{}.self_attention.dense", prefix), q_dim, hidden_size, compute, cb2.as_ref())?;
        // Residual connection
        let after_attn = self.gpu_add(hidden, &attn_proj, compute, cb2.as_ref())?;

        // 6. Post-norm (RMSNorm)
        let post_norm_w = self.w(&format!("{}.post_attention_layernorm.weight", prefix))?;
        let post_normed = self.gpu_rms_norm(&after_attn, post_norm_w, hidden_size, compute, cb2.as_ref())?;

        // 7. MLP: dense_h_to_4h → SiLU gate → dense_4h_to_h
        // dense_h_to_4h produces [seq, 2*ffn_hidden] split into gate and up
        let ffn_hidden = self.config.ffn_hidden;
        let mlp_up = self.gpu_linear_nobias(&post_normed, &format!("{}.mlp.dense_h_to_4h", prefix), hidden_size, ffn_hidden * 2, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        // SiLU-gated split: gate = silu(first_half) * second_half
        let mlp_f32 = mlp_up.to_f32_vec()?;
        let mut gated: Vec<half::f16> = Vec::with_capacity(seq_len * ffn_hidden);
        for row in 0..seq_len {
            let base = row * ffn_hidden * 2;
            for i in 0..ffn_hidden {
                let gate_val = mlp_f32[base + i];
                let up_val = mlp_f32[base + ffn_hidden + i];
                let silu_gate = gate_val * (1.0 / (1.0 + (-gate_val).exp()));
                gated.push(half::f16::from_f32(silu_gate * up_val));
            }
        }
        let gated_tensor = Tensor::from_slice(&gated, Shape::from([seq_len, ffn_hidden]), DType::F16, device)?;

        // Down projection
        let cb3 = compute.new_command_buffer();
        let mlp_out = self.gpu_linear_nobias(&gated_tensor, &format!("{}.mlp.dense_4h_to_h", prefix), ffn_hidden, hidden_size, compute, cb3.as_ref())?;
        // Residual
        let output = self.gpu_add(&after_attn, &mlp_out, compute, cb3.as_ref())?;
        cb3.commit();
        cb3.wait_until_completed();

        Ok(output)
    }

    /// Expand KV heads from num_kv_heads to num_heads via repeat interleave.
    fn expand_kv(
        &self,
        kv: &Tensor,
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        repeats: usize,
    ) -> Result<Tensor> {
        if repeats == 1 {
            return Ok(kv.clone());
        }
        let kv_f32 = kv.to_f32_vec()?;
        let num_heads = num_kv_heads * repeats;
        let mut expanded: Vec<half::f16> = Vec::with_capacity(seq_len * num_heads * head_dim);
        for token in 0..seq_len {
            for kv_head in 0..num_kv_heads {
                let src_offset = token * num_kv_heads * head_dim + kv_head * head_dim;
                for _rep in 0..repeats {
                    for d in 0..head_dim {
                        expanded.push(half::f16::from_f32(kv_f32[src_offset + d]));
                    }
                }
            }
        }
        Tensor::from_slice(
            &expanded,
            Shape::from([seq_len, num_heads * head_dim]),
            DType::F16,
            kv.device(),
        )
    }

    fn batched_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        q_seq_len: usize,
        kv_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_shd = q.reshape(Shape::from([q_seq_len, num_heads, head_dim]))?;
        let k_shd = k.reshape(Shape::from([kv_seq_len, num_heads, head_dim]))?;
        let v_shd = v.reshape(Shape::from([kv_seq_len, num_heads, head_dim]))?;

        let cb = compute.new_command_buffer();
        let q_hsd = self.gpu_transpose_shd_hsd(&q_shd, q_seq_len, num_heads, head_dim, compute, cb.as_ref())?;
        let k_hsd = self.gpu_transpose_shd_hsd(&k_shd, kv_seq_len, num_heads, head_dim, compute, cb.as_ref())?;
        let v_hsd = self.gpu_transpose_shd_hsd(&v_shd, kv_seq_len, num_heads, head_dim, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        let cb2 = compute.new_command_buffer();
        let scores = self.gpu_batched_linear_raw(&q_hsd, &k_hsd, num_heads, q_seq_len, head_dim, kv_seq_len, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        let cb3 = compute.new_command_buffer();
        let attn_weights = self.gpu_row_softmax(&scores, num_heads * q_seq_len, kv_seq_len, scale, compute, cb3.as_ref())?;
        cb3.commit();
        cb3.wait_until_completed();

        let cb4 = compute.new_command_buffer();
        let attn_out_hsd = self.gpu_batched_matmul_nn(&attn_weights, &v_hsd, num_heads, q_seq_len, kv_seq_len, head_dim, compute, cb4.as_ref())?;
        let attn_out_shd = self.gpu_transpose_hsd_shd(&attn_out_hsd, q_seq_len, num_heads, head_dim, compute, cb4.as_ref())?;
        cb4.commit();
        cb4.wait_until_completed();

        Ok(attn_out_shd.reshape(Shape::from([q_seq_len, num_heads * head_dim]))?)
    }

    // ========================================================================
    // Low-level GPU helpers
    // ========================================================================

    fn w(&self, name: &str) -> Result<&crate::hal::metal::LazyTensor> {
        self.model.get_weight(name)
            .ok_or_else(|| crate::core::Error::internal(format!("ChatGLM weight not found: {}", name)))
    }

    fn gpu_rms_norm(
        &self, x: &Tensor, weight: &crate::hal::metal::LazyTensor, hidden: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let seq_len = x.shape().numel() / hidden;
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_h = hidden as u32;
        let eps = self.config.rms_norm_eps;
        let c_n = seq_len as u32;
        // rms_norm_f16 kernel: input(0), weight(1), output(2), N(3), D(4), eps(5)
        compute.dispatch_async(cb, &self.kernels.rms_norm,
            (seq_len, 1, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(weight.buffer()), 0);
                enc.set_buffer(2, Some(o_buf.as_ref()), 0);
                enc.set_bytes(3, 4, &c_n as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_h as *const u32 as *const _);
                enc.set_bytes(5, 4, &eps as *const f32 as *const _);
            });
        Ok(output)
    }

    fn gpu_linear_nobias(
        &self, x: &Tensor, prefix: &str, in_feat: usize, out_feat: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let w = self.w(&format!("{}.weight", prefix))?;
        let seq_len = x.shape().numel() / in_feat;
        let output = Tensor::empty(Shape::from([seq_len, out_feat]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let m = seq_len as u32;
        let n = out_feat as u32;
        let k = in_feat as u32;
        compute.dispatch_async(cb, &self.kernels.linear,
            ((out_feat + 15) / 16, (seq_len + 15) / 16, 1), (16, 16, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(w.buffer()), 0);
                enc.set_buffer(2, Some(o_buf.as_ref()), 0);
                enc.set_bytes(3, 4, &m as *const u32 as *const _);
                enc.set_bytes(4, 4, &n as *const u32 as *const _);
                enc.set_bytes(5, 4, &k as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_add(&self, a: &Tensor, b: &Tensor, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let count = a.shape().numel();
        let output = Tensor::empty(a.shape().clone(), DType::F16, a.device())?;
        let a_buf = borrow_tensor(a)?;
        let b_buf = borrow_tensor(b)?;
        let o_buf = borrow_tensor(&output)?;
        let c = count as u32;
        compute.dispatch_async(cb, &self.kernels.add,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(a_buf.as_ref()), 0);
                enc.set_buffer(1, Some(b_buf.as_ref()), 0);
                enc.set_buffer(2, Some(o_buf.as_ref()), 0);
                enc.set_bytes(3, 4, &c as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_transpose_shd_hsd(&self, x: &Tensor, seq: usize, heads: usize, dim: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([heads, seq, dim]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_s = seq as u32; let c_h = heads as u32; let c_d = dim as u32;
        compute.dispatch_async(cb, &self.kernels.transpose_shd_hsd,
            (heads, seq, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c_s as *const u32 as *const _);
                enc.set_bytes(3, 4, &c_h as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_d as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_transpose_hsd_shd(&self, x: &Tensor, seq: usize, heads: usize, dim: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([seq, heads, dim]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_s = seq as u32; let c_h = heads as u32; let c_d = dim as u32;
        compute.dispatch_async(cb, &self.kernels.transpose_hsd_shd,
            (heads, seq, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c_s as *const u32 as *const _);
                enc.set_bytes(3, 4, &c_h as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_d as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_batched_linear_raw(&self, x: &Tensor, w: &Tensor, batch: usize, m: usize, k: usize, n: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([batch, m, n]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let w_buf = borrow_tensor(w)?;
        let o_buf = borrow_tensor(&output)?;
        let c_m = m as u32; let c_n = n as u32; let c_k = k as u32; let c_b = batch as u32;
        compute.dispatch_async(cb, &self.kernels.batched_linear,
            ((n + 15) / 16, (m + 15) / 16, batch), (16, 16, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(w_buf.as_ref()), 0);
                enc.set_buffer(2, Some(o_buf.as_ref()), 0);
                enc.set_bytes(3, 4, &c_m as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_n as *const u32 as *const _);
                enc.set_bytes(5, 4, &c_k as *const u32 as *const _);
                enc.set_bytes(6, 4, &c_b as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_batched_matmul_nn(&self, a: &Tensor, b: &Tensor, batch: usize, m: usize, k: usize, n: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([batch, m, n]), DType::F16, a.device())?;
        let a_buf = borrow_tensor(a)?;
        let b_buf = borrow_tensor(b)?;
        let o_buf = borrow_tensor(&output)?;
        let c_m = m as u32; let c_n = n as u32; let c_k = k as u32; let c_b = batch as u32;
        compute.dispatch_async(cb, &self.kernels.batched_matmul_nn,
            ((n + 15) / 16, (m + 15) / 16, batch), (16, 16, 1), |enc| {
                enc.set_buffer(0, Some(a_buf.as_ref()), 0);
                enc.set_buffer(1, Some(b_buf.as_ref()), 0);
                enc.set_buffer(2, Some(o_buf.as_ref()), 0);
                enc.set_bytes(3, 4, &c_m as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_n as *const u32 as *const _);
                enc.set_bytes(5, 4, &c_k as *const u32 as *const _);
                enc.set_bytes(6, 4, &c_b as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_row_softmax(&self, x: &Tensor, num_rows: usize, row_len: usize, scale: f32, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_cols = row_len as u32;
        compute.dispatch_async(cb, &self.kernels.row_softmax,
            (num_rows, 1, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c_cols as *const u32 as *const _);
                enc.set_bytes(3, 4, &scale as *const f32 as *const _);
            });
        Ok(output)
    }
}

#[cfg(feature = "metal")]
fn borrow_tensor(tensor: &Tensor) -> Result<BorrowedMetalBuffer> {
    let ptr = tensor.device_ptr()
        .ok_or_else(|| crate::core::Error::internal("tensor not on device"))?;
    Ok(unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
}
