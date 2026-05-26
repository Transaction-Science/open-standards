//! T5/UMT5 text encoder for DiT-based diffusion models.
//!
//! Supports both standard T5 (rel pos bias in block 0 only) and UMT5
//! (scalable_attention: rel pos bias in every block). All operations
//! on Metal GPU.

use crate::inference::model::Model;
use crate::core::Result;
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::hal::{MetalCompute, MetalDevice};
#[cfg(feature = "metal")]
use crate::hal::metal::{BorrowedMetalBuffer, ComputePipeline, LazyTensor};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;

/// T5 encoder configuration.
#[derive(Debug, Clone)]
pub struct T5Config {
    /// Model dimension.
    pub d_model: usize,
    /// FFN inner dimension.
    pub d_ff: usize,
    /// Dimension per attention head (d_kv).
    pub d_kv: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of encoder layers.
    pub num_layers: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Number of relative position bias buckets.
    pub num_buckets: usize,
    /// Maximum distance for relative position bias.
    pub max_distance: usize,
    /// Whether each block has its own relative position bias (UMT5) or only block 0 (T5).
    pub scalable_attention: bool,
    /// Layer norm epsilon.
    pub layer_norm_epsilon: f32,
    /// Whether FFN uses gated activation (wi_0/wi_1 + GEGLU) vs plain (wi + ReLU).
    /// T5-base/large: false (plain ReLU). UMT5/T5-XXL/Flan-T5: true (gated GEGLU).
    pub is_gated_ffn: bool,
}

impl T5Config {
    /// UMT5 config for AuraFlow (d_model=2048, 24 layers).
    pub fn umt5_auraflow() -> Self {
        Self {
            d_model: 2048,
            d_ff: 5120,
            d_kv: 64,
            num_heads: 32,
            num_layers: 24,
            vocab_size: 32128,
            num_buckets: 32,
            max_distance: 128,
            scalable_attention: true,
            layer_norm_epsilon: 1e-6,
            is_gated_ffn: true,
        }
    }

    /// T5-XXL config for Flux (d_model=4096, 24 layers).
    pub fn t5_xxl() -> Self {
        Self {
            d_model: 4096,
            d_ff: 10240,
            d_kv: 64,
            num_heads: 64,
            num_layers: 24,
            vocab_size: 32128,
            num_buckets: 32,
            max_distance: 128,
            scalable_attention: false,
            layer_norm_epsilon: 1e-6,
            is_gated_ffn: true,
        }
    }

    /// UMT5-XXL config for Wan2.1 (d_model=4096, 24 layers, scalable_attention).
    pub fn umt5_xxl() -> Self {
        Self {
            d_model: 4096,
            d_ff: 10240,
            d_kv: 64,
            num_heads: 64,
            num_layers: 24,
            vocab_size: 256300,
            num_buckets: 32,
            max_distance: 128,
            scalable_attention: true,
            layer_norm_epsilon: 1e-6,
            is_gated_ffn: true,
        }
    }

    /// T5-v1.1-large config for AudioGen (d_model=1024, 24 layers, gated-GELU FFN).
    pub fn t5_v1_1_large() -> Self {
        Self {
            d_model: 1024,
            d_ff: 2816,
            d_kv: 64,
            num_heads: 16,
            num_layers: 24,
            vocab_size: 32128,
            num_buckets: 32,
            max_distance: 128,
            scalable_attention: false,
            layer_norm_epsilon: 1e-6,
            is_gated_ffn: true,
        }
    }

    /// T5-v1.1-base config for MAGNet (d_model=768, 12 layers, gated-GELU FFN).
    pub fn t5_v1_1_base() -> Self {
        Self {
            d_model: 768,
            d_ff: 2048,
            d_kv: 64,
            num_heads: 12,
            num_layers: 12,
            vocab_size: 32128,
            num_buckets: 32,
            max_distance: 128,
            scalable_attention: false,
            layer_norm_epsilon: 1e-6,
            is_gated_ffn: true,
        }
    }

    /// T5-base config for MusicGen (d_model=768, 12 layers, plain ReLU FFN).
    pub fn t5_base() -> Self {
        Self {
            d_model: 768,
            d_ff: 3072,
            d_kv: 64,
            num_heads: 12,
            num_layers: 12,
            vocab_size: 32128,
            num_buckets: 32,
            max_distance: 128,
            scalable_attention: false,
            layer_norm_epsilon: 1e-6,
            is_gated_ffn: false,
        }
    }
}

/// T5 text encoder running on Metal GPU.
#[cfg(feature = "metal")]
pub struct T5Encoder {
    pub(crate) model: Arc<Model>,
    pub(crate) config: T5Config,
    pub(crate) compute: Arc<MetalCompute>,
    pub(crate) kernels: T5Kernels,
    prefix: String,
}

#[cfg(feature = "metal")]
pub(crate) struct T5Kernels {
    pub(crate) rms_norm: Arc<ComputePipeline>,
    pub(crate) rms_norm_f32: Arc<ComputePipeline>,
    pub(crate) linear: Arc<ComputePipeline>,
    pub(crate) linear_out_f32: Arc<ComputePipeline>,
    pub(crate) add: Arc<ComputePipeline>,
    pub(crate) residual_add: Arc<ComputePipeline>,
    pub(crate) residual_add_f32: Arc<ComputePipeline>,
    pub(crate) residual_add_f32_f32: Arc<ComputePipeline>,
    pub(crate) f16_to_f32: Arc<ComputePipeline>,
    pub(crate) geglu: Arc<ComputePipeline>,
    pub(crate) relu: Arc<ComputePipeline>,
    pub(crate) embedding: Arc<ComputePipeline>,
    pub(crate) rel_pos_bias: Arc<ComputePipeline>,
    pub(crate) batched_linear: Arc<ComputePipeline>,
    pub(crate) batched_matmul_nn: Arc<ComputePipeline>,
    pub(crate) row_softmax: Arc<ComputePipeline>,
    pub(crate) transpose_shd_hsd: Arc<ComputePipeline>,
    pub(crate) transpose_hsd_shd: Arc<ComputePipeline>,
    pub(crate) batched_linear_f32_out: Arc<ComputePipeline>,
    pub(crate) add_bias_softmax_f32: Arc<ComputePipeline>,
    pub(crate) softmax_f32: Arc<ComputePipeline>,
    pub(crate) geglu_f32: Arc<ComputePipeline>,
    pub(crate) relu_f32: Arc<ComputePipeline>,
    pub(crate) linear_f32_wt_f32: Arc<ComputePipeline>,
}

#[cfg(feature = "metal")]
impl T5Kernels {
    /// Compile all T5 kernels on the given compute device.
    pub(crate) fn compile(compute: &MetalCompute) -> Result<Self> {
        Ok(Self {
            rms_norm: compute.compile_pipeline("rms_norm", sources::RMS_NORM, "rms_norm_f16")?,
            rms_norm_f32: compute.compile_pipeline("rms_norm_f32", sources::RMS_NORM, "rms_norm_f32_to_f16")?,
            linear: compute.compile_pipeline("linear", sources::LINEAR, "linear_f16")?,
            linear_out_f32: compute.compile_pipeline("linear_out_f32", sources::LINEAR, "linear_f16_out_f32")?,
            add: compute.compile_pipeline("add", sources::ELEMENTWISE, "add_f16")?,
            residual_add: compute.compile_pipeline("residual_add", sources::SCHEDULER, "scale_add_clamp_f16")?,
            residual_add_f32: compute.compile_pipeline("residual_add_f32", sources::RMS_NORM, "residual_add_f16_to_f32")?,
            residual_add_f32_f32: compute.compile_pipeline("residual_add_f32_f32", sources::RMS_NORM, "residual_add_f32_to_f32")?,
            f16_to_f32: compute.compile_pipeline("f16_to_f32", sources::RMS_NORM, "f16_to_f32")?,
            geglu: compute.compile_pipeline("geglu", sources::GELU, "geglu_f16")?,
            relu: compute.compile_pipeline("relu", sources::GELU, "relu_f16")?,
            embedding: compute.compile_pipeline("embedding", sources::EMBEDDING, "embedding_lookup_f16")?,
            rel_pos_bias: compute.compile_pipeline("rel_pos_bias", sources::RELATIVE_POSITION_BIAS, "relative_position_bias_f16")?,
            batched_linear: compute.compile_pipeline("batched_linear", sources::LINEAR, "batched_linear_f16")?,
            batched_matmul_nn: compute.compile_pipeline("batched_matmul_nn", sources::LINEAR, "batched_matmul_nn_f16")?,
            row_softmax: compute.compile_pipeline("row_softmax_scale", sources::LINEAR, "row_softmax_scale_f16")?,
            transpose_shd_hsd: compute.compile_pipeline("transpose_shd_hsd", sources::LINEAR, "transpose_shd_to_hsd_f16")?,
            transpose_hsd_shd: compute.compile_pipeline("transpose_hsd_shd", sources::LINEAR, "transpose_hsd_to_shd_f16")?,
            batched_linear_f32_out: compute.compile_pipeline("batched_linear_f32_out", sources::LINEAR, "batched_linear_f16_out_f32")?,
            add_bias_softmax_f32: compute.compile_pipeline("add_bias_softmax_f32", sources::LINEAR, "add_bias_softmax_f32_to_f16")?,
            softmax_f32: compute.compile_pipeline("softmax_f32", sources::LINEAR, "softmax_f32_to_f16")?,
            geglu_f32: compute.compile_pipeline("geglu_f32", sources::GELU, "geglu_f16_to_f32")?,
            relu_f32: compute.compile_pipeline("relu_f32", sources::GELU, "relu_f16_to_f32")?,
            linear_f32_wt_f32: compute.compile_pipeline("linear_f32_wt_f32", sources::GELU, "linear_f32_in_f16_wt_f32_out")?,
        })
    }
}

#[cfg(feature = "metal")]
impl T5Encoder {
    /// Create a new T5 encoder.
    pub fn new(model: Arc<Model>, config: T5Config, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        let kernels = T5Kernels {
            rms_norm: compute.compile_pipeline("rms_norm", sources::RMS_NORM, "rms_norm_f16")?,
            rms_norm_f32: compute.compile_pipeline("rms_norm_f32", sources::RMS_NORM, "rms_norm_f32_to_f16")?,
            linear: compute.compile_pipeline("linear", sources::LINEAR, "linear_f16")?,
            linear_out_f32: compute.compile_pipeline("linear_out_f32", sources::LINEAR, "linear_f16_out_f32")?,
            add: compute.compile_pipeline("add", sources::ELEMENTWISE, "add_f16")?,
            residual_add: compute.compile_pipeline("residual_add", sources::SCHEDULER, "scale_add_clamp_f16")?,
            residual_add_f32: compute.compile_pipeline("residual_add_f32", sources::RMS_NORM, "residual_add_f16_to_f32")?,
            residual_add_f32_f32: compute.compile_pipeline("residual_add_f32_f32", sources::RMS_NORM, "residual_add_f32_to_f32")?,
            f16_to_f32: compute.compile_pipeline("f16_to_f32", sources::RMS_NORM, "f16_to_f32")?,
            geglu: compute.compile_pipeline("geglu", sources::GELU, "geglu_f16")?,
            relu: compute.compile_pipeline("relu", sources::GELU, "relu_f16")?,
            embedding: compute.compile_pipeline("embedding", sources::EMBEDDING, "embedding_lookup_f16")?,
            rel_pos_bias: compute.compile_pipeline("rel_pos_bias", sources::RELATIVE_POSITION_BIAS, "relative_position_bias_f16")?,
            batched_linear: compute.compile_pipeline("batched_linear", sources::LINEAR, "batched_linear_f16")?,
            batched_matmul_nn: compute.compile_pipeline("batched_matmul_nn", sources::LINEAR, "batched_matmul_nn_f16")?,
            row_softmax: compute.compile_pipeline("row_softmax_scale", sources::LINEAR, "row_softmax_scale_f16")?,
            transpose_shd_hsd: compute.compile_pipeline("transpose_shd_hsd", sources::LINEAR, "transpose_shd_to_hsd_f16")?,
            transpose_hsd_shd: compute.compile_pipeline("transpose_hsd_shd", sources::LINEAR, "transpose_hsd_to_shd_f16")?,
            batched_linear_f32_out: compute.compile_pipeline("batched_linear_f32_out", sources::LINEAR, "batched_linear_f16_out_f32")?,
            add_bias_softmax_f32: compute.compile_pipeline("add_bias_softmax_f32", sources::LINEAR, "add_bias_softmax_f32_to_f16")?,
            softmax_f32: compute.compile_pipeline("softmax_f32", sources::LINEAR, "softmax_f32_to_f16")?,
            geglu_f32: compute.compile_pipeline("geglu_f32", sources::GELU, "geglu_f16_to_f32")?,
            relu_f32: compute.compile_pipeline("relu_f32", sources::GELU, "relu_f16_to_f32")?,
            linear_f32_wt_f32: compute.compile_pipeline("linear_f32_wt_f32", sources::GELU, "linear_f32_in_f16_wt_f32_out")?,
        };

        Ok(Self { model, config, compute, kernels, prefix: String::new() })
    }

    /// Set a weight name prefix (e.g., "text_encoder." for MusicGen).
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.to_string();
        self
    }

    /// Encode token IDs to hidden states on Metal GPU.
    /// Returns [seq_len, d_model] in F16.
    pub fn encode(&self, token_ids: &[u32]) -> Result<Tensor> {
        let config = &self.config;
        let seq_len = token_ids.len();
        let device_id = self.compute.device().info().id;

        // 1. Token embedding lookup → F16
        let embed_w = self.w("shared.weight")
            .or_else(|_| self.w("encoder.embed_tokens.weight"))?;
        let embed_f16 = self.gpu_embedding(embed_w, token_ids, config.d_model, device_id)?;

        // Convert residual stream to F32 (T5 values exceed F16 max in deeper layers)
        let mut h = self.gpu_f16_to_f32(&embed_f16)?;

        // 2. Encoder blocks
        let mut block0_bias: Option<Tensor> = None;

        for layer in 0..config.num_layers {
            let prefix = format!("encoder.block.{}.layer", layer);

            // Pre-attention RMSNorm: F32 input → F16 output (for linear projections)
            let ln1_w = self.w(&format!("{}.0.layer_norm.weight", prefix))?;
            let normed = self.gpu_rms_norm_f32(&h, ln1_w, config.d_model, config.layer_norm_epsilon)?;

            // Self-attention Q/K/V projections (all F16)
            let q_w = self.w(&format!("{}.0.SelfAttention.q.weight", prefix))?;
            let k_w = self.w(&format!("{}.0.SelfAttention.k.weight", prefix))?;
            let v_w = self.w(&format!("{}.0.SelfAttention.v.weight", prefix))?;
            let o_w = self.w(&format!("{}.0.SelfAttention.o.weight", prefix))?;

            let q = self.gpu_linear_lazy(&normed, q_w, seq_len, config.d_model, config.d_model)?;
            let k = self.gpu_linear_lazy(&normed, k_w, seq_len, config.d_model, config.d_model)?;
            let v = self.gpu_linear_lazy(&normed, v_w, seq_len, config.d_model, config.d_model)?;

            // Relative position bias
            let bias = if config.scalable_attention || layer == 0 {
                let bias_key = format!("{}.0.SelfAttention.relative_attention_bias.weight", prefix);
                let bias_w = self.w(&bias_key)?;
                let b = self.gpu_relative_position_bias(bias_w, seq_len, seq_len, true)?;
                if layer == 0 && !config.scalable_attention {
                    block0_bias = Some(b.clone());
                }
                b
            } else {
                block0_bias.clone().unwrap()
            };

            // Attention: Q@K^T in F32, bias + softmax in F32, S@V in F16
            let attn_out = self.gpu_attention_f32(&q, &k, &v, &bias, seq_len, config.num_heads, config.d_kv)?;

            // Output projection → F32 directly + residual accumulation (F32)
            let projected = self.gpu_linear_lazy_f32(&attn_out, o_w, seq_len, config.d_model, config.d_model)?;
            h = self.gpu_residual_add_f32_f32(&h, &projected)?;

            // Pre-FFN RMSNorm: F32 input → F16 output
            let ln2_w = self.w(&format!("{}.1.layer_norm.weight", prefix))?;
            let normed2 = self.gpu_rms_norm_f32(&h, ln2_w, config.d_model, config.layer_norm_epsilon)?;

            // FFN: GEGLU/ReLU → F32 → WO (F32 in, F16 weight, F32 out)
            let ffn_out = if config.is_gated_ffn {
                let wi0_w = self.w(&format!("{}.1.DenseReluDense.wi_0.weight", prefix))?;
                let wi1_w = self.w(&format!("{}.1.DenseReluDense.wi_1.weight", prefix))?;
                let wo_w = self.w(&format!("{}.1.DenseReluDense.wo.weight", prefix))?;
                let gate = self.gpu_linear_lazy(&normed2, wi0_w, seq_len, config.d_model, config.d_ff)?;
                let up = self.gpu_linear_lazy(&normed2, wi1_w, seq_len, config.d_model, config.d_ff)?;
                let gated = self.gpu_geglu_f32(&gate, &up, seq_len * config.d_ff)?;
                self.gpu_linear_f32_in_f32_out(&gated, wo_w, seq_len, config.d_ff, config.d_model)?
            } else {
                let wi_w = self.w(&format!("{}.1.DenseReluDense.wi.weight", prefix))?;
                let wo_w = self.w(&format!("{}.1.DenseReluDense.wo.weight", prefix))?;
                let hidden = self.gpu_linear_lazy(&normed2, wi_w, seq_len, config.d_model, config.d_ff)?;
                let activated = self.gpu_relu_f32(&hidden, seq_len * config.d_ff)?;
                self.gpu_linear_f32_in_f32_out(&activated, wo_w, seq_len, config.d_ff, config.d_model)?
            };
            h = self.gpu_residual_add_f32_f32(&h, &ffn_out)?;
        }

        // 3. Final RMSNorm: F32 → F16 output
        let final_ln_w = self.w("encoder.final_layer_norm.weight")?;
        let h_f16 = self.gpu_rms_norm_f32(&h, final_ln_w, config.d_model, config.layer_norm_epsilon)?;

        Ok(h_f16)
    }

    // === Weight access ===

    pub(crate) fn w(&self, name: &str) -> Result<&LazyTensor> {
        let full_name = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}{}", self.prefix, name)
        };
        self.model.get_weight(&full_name)
            .ok_or_else(|| crate::core::Error::internal(format!("T5 weight not found: {}", full_name)))
    }

    // === GPU operations ===

    pub(crate) fn gpu_embedding(&self, weight: &LazyTensor, token_ids: &[u32], dim: usize, device_id: crate::hal::DeviceId) -> Result<Tensor> {
        let seq_len = token_ids.len();
        let vocab_size = self.config.vocab_size;
        let output = Tensor::empty(Shape::from([seq_len, dim]), DType::F16, device_id)?;
        let o_buf = borrow_tensor(&output)?;

        // Create GPU buffer for token IDs
        let device = self.compute.device().raw();
        let id_buf = device.new_buffer_with_data(
            token_ids.as_ptr() as *const _,
            (seq_len * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let cb = self.compute.new_command_buffer();
        let c_vocab = vocab_size as u32;
        let c_dim = dim as u32;
        let c_seq = seq_len as u32;

        // embedding_lookup_f16: embed_table(0), token_ids(1), output(2), vocab_size(3), hidden_size(4), seq_len(5)
        self.compute.dispatch_async(cb.as_ref(), &self.kernels.embedding,
            (seq_len, 1, 1), (1, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(weight.buffer()), 0);
                encoder.set_buffer(1, Some(&id_buf), 0);
                encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &c_vocab as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_dim as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_seq as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    pub(crate) fn gpu_rms_norm(&self, input: &Tensor, weight: &LazyTensor, dim: usize, eps: f32) -> Result<Tensor> {
        let num_rows = input.shape().numel() / dim;
        let output = Tensor::empty(input.shape().clone(), DType::F16, input.device())?;

        let cb = self.compute.new_command_buffer();
        let in_buf = borrow_tensor(input)?;
        let o_buf = borrow_tensor(&output)?;
        let c_n = num_rows as u32;
        let c_dim = dim as u32;

        // rms_norm_f16 kernel: input(0), weight(1), output(2), N(3), D(4), eps(5)
        self.compute.dispatch_async(cb.as_ref(), &self.kernels.rms_norm,
            (num_rows, 1, 1), (1, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(weight.buffer()), 0);
                encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_dim as *const u32 as *const _);
                encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// Linear Y = X @ W^T. Weight is LazyTensor [N, K], input is Tensor [M, K].
    pub(crate) fn gpu_linear_lazy(&self, input: &Tensor, weight: &LazyTensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([m, n]), DType::F16, input.device())?;

        let cb = self.compute.new_command_buffer();
        let x_buf = borrow_tensor(input)?;
        let o_buf = borrow_tensor(&output)?;
        let c_m = m as u32;
        let c_n = n as u32;
        let c_k = k as u32;
        let has_bias: u32 = 0;

        // linear_f16 kernel: X(0), W(1), bias(2), Y(3), M(4), N(5), K(6), has_bias(7)
        // TILE=16 requires (16, 16, 1) threadgroups
        self.compute.dispatch_async(cb.as_ref(), &self.kernels.linear,
            ((n + 15) / 16, (m + 15) / 16, 1), (16, 16, 1), |encoder| {
                encoder.set_buffer(0, Some(x_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(weight.buffer()), 0);
                encoder.set_buffer(2, Some(x_buf.as_ref()), 0); // dummy bias (has_bias=0)
                encoder.set_buffer(3, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(4, 4, &c_m as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(6, 4, &c_k as *const u32 as *const _);
                encoder.set_bytes(7, 4, &has_bias as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    pub(crate) fn gpu_geglu(&self, gate: &Tensor, up: &Tensor, count: usize) -> Result<Tensor> {
        let output = Tensor::empty(gate.shape().clone(), DType::F16, gate.device())?;

        let cb = self.compute.new_command_buffer();
        let g_buf = borrow_tensor(gate)?;
        let u_buf = borrow_tensor(up)?;
        let o_buf = borrow_tensor(&output)?;

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.geglu,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(g_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(u_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    pub(crate) fn gpu_relu(&self, input: &Tensor, count: usize) -> Result<Tensor> {
        let output = Tensor::empty(input.shape().clone(), DType::F16, input.device())?;

        let cb = self.compute.new_command_buffer();
        let i_buf = borrow_tensor(input)?;
        let o_buf = borrow_tensor(&output)?;

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.relu,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(i_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(o_buf.as_ref()), 0);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// GEGLU with F32 output (for deep models). Reads F16 gate/up, outputs F32.
    pub(crate) fn gpu_geglu_f32(&self, gate: &Tensor, up: &Tensor, count: usize) -> Result<Tensor> {
        let output = Tensor::empty(gate.shape().clone(), DType::F32, gate.device())?;

        let cb = self.compute.new_command_buffer();
        let g_buf = borrow_tensor(gate)?;
        let u_buf = borrow_tensor(up)?;
        let o_buf = borrow_tensor(&output)?;

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.geglu_f32,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(g_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(u_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// ReLU with F32 output (for deep models). Reads F16 input, outputs F32.
    pub(crate) fn gpu_relu_f32(&self, input: &Tensor, count: usize) -> Result<Tensor> {
        let output = Tensor::empty(input.shape().clone(), DType::F32, input.device())?;

        let cb = self.compute.new_command_buffer();
        let i_buf = borrow_tensor(input)?;
        let o_buf = borrow_tensor(&output)?;

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.relu_f32,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(i_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(o_buf.as_ref()), 0);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// Linear Y = X @ W^T with F32 input (X), F16 weight (W), F32 output (Y).
    /// For WO projection after F32 GEGLU/ReLU output.
    pub(crate) fn gpu_linear_f32_in_f32_out(&self, input: &Tensor, weight: &LazyTensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([m, n]), DType::F32, input.device())?;

        let cb = self.compute.new_command_buffer();
        let x_buf = borrow_tensor(input)?;
        let o_buf = borrow_tensor(&output)?;
        let c_m = m as u32;
        let c_n = n as u32;
        let c_k = k as u32;
        let has_bias: u32 = 0;

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.linear_f32_wt_f32,
            ((n + 15) / 16, (m + 15) / 16, 1), (16, 16, 1), |encoder| {
                encoder.set_buffer(0, Some(x_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(weight.buffer()), 0);
                encoder.set_buffer(2, Some(x_buf.as_ref()), 0); // dummy bias (has_bias=0)
                encoder.set_buffer(3, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(4, 4, &c_m as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(6, 4, &c_k as *const u32 as *const _);
                encoder.set_bytes(7, 4, &has_bias as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    pub(crate) fn gpu_add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let count = a.shape().numel();
        let output = Tensor::empty(a.shape().clone(), DType::F16, a.device())?;

        let cb = self.compute.new_command_buffer();
        let a_buf = borrow_tensor(a)?;
        let b_buf = borrow_tensor(b)?;
        let o_buf = borrow_tensor(&output)?;

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.add,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(a_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(b_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// Clamped residual add: output = clamp(a + b, -65000, 65000).
    /// Prevents F16 overflow to inf which causes NaN in subsequent RMSNorm.
    pub(crate) fn gpu_residual_add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let count = a.shape().numel();
        let output = Tensor::empty(a.shape().clone(), DType::F16, a.device())?;

        let cb = self.compute.new_command_buffer();
        let a_buf = borrow_tensor(a)?;
        let b_buf = borrow_tensor(b)?;
        let o_buf = borrow_tensor(&output)?;
        let scale_a = 1.0_f32;
        let scale_b = 1.0_f32;
        let lo = -65000.0_f32;
        let hi = 65000.0_f32;
        let c_count = count as u32;

        // scale_add_clamp_f16: a(0), b(1), output(2), scale_a(3), scale_b(4), lo(5), hi(6), count(7)
        self.compute.dispatch_async(cb.as_ref(), &self.kernels.residual_add,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(a_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(b_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &scale_a as *const f32 as *const _);
                encoder.set_bytes(4, 4, &scale_b as *const f32 as *const _);
                encoder.set_bytes(5, 4, &lo as *const f32 as *const _);
                encoder.set_bytes(6, 4, &hi as *const f32 as *const _);
                encoder.set_bytes(7, 4, &c_count as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// RMS norm from F32 residual → F16 output (for linear projections).
    pub(crate) fn gpu_rms_norm_f32(&self, input: &Tensor, weight: &LazyTensor, dim: usize, eps: f32) -> Result<Tensor> {
        let num_rows = input.shape().numel() / dim;
        let device_id = input.device();
        let output = Tensor::empty(Shape::from([num_rows, dim]), DType::F16, device_id)?;

        let cb = self.compute.new_command_buffer();
        let in_buf = borrow_tensor(input)?;
        let o_buf = borrow_tensor(&output)?;
        let c_n = num_rows as u32;
        let c_dim = dim as u32;

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.rms_norm_f32,
            (num_rows, 1, 1), (1, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(weight.buffer()), 0);
                encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_dim as *const u32 as *const _);
                encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// Residual add: F32 residual + F16 delta → F32 output (clamps F16 inf/nan).
    pub(crate) fn gpu_residual_add_f32(&self, residual: &Tensor, delta: &Tensor) -> Result<Tensor> {
        let count = residual.shape().numel();
        let output = Tensor::empty(residual.shape().clone(), DType::F32, residual.device())?;

        let cb = self.compute.new_command_buffer();
        let r_buf = borrow_tensor(residual)?;
        let d_buf = borrow_tensor(delta)?;
        let o_buf = borrow_tensor(&output)?;
        let c_count = count as u32;

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.residual_add_f32,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(r_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(d_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &c_count as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// Residual add: F32 + F32 → F32 (no clamping needed).
    pub(crate) fn gpu_residual_add_f32_f32(&self, residual: &Tensor, delta: &Tensor) -> Result<Tensor> {
        let count = residual.shape().numel();
        let output = Tensor::empty(residual.shape().clone(), DType::F32, residual.device())?;

        let cb = self.compute.new_command_buffer();
        let r_buf = borrow_tensor(residual)?;
        let d_buf = borrow_tensor(delta)?;
        let o_buf = borrow_tensor(&output)?;
        let c_count = count as u32;

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.residual_add_f32_f32,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(r_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(d_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &c_count as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// Linear Y = X @ W^T with F32 output. Reads F16 input/weight, accumulates F32, outputs F32.
    pub(crate) fn gpu_linear_lazy_f32(&self, input: &Tensor, weight: &LazyTensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([m, n]), DType::F32, input.device())?;

        let cb = self.compute.new_command_buffer();
        let x_buf = borrow_tensor(input)?;
        let o_buf = borrow_tensor(&output)?;
        let c_m = m as u32;
        let c_n = n as u32;
        let c_k = k as u32;
        let has_bias: u32 = 0;

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.linear_out_f32,
            ((n + 15) / 16, (m + 15) / 16, 1), (16, 16, 1), |encoder| {
                encoder.set_buffer(0, Some(x_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(weight.buffer()), 0);
                encoder.set_buffer(2, Some(x_buf.as_ref()), 0); // dummy bias (has_bias=0)
                encoder.set_buffer(3, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(4, 4, &c_m as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(6, 4, &c_k as *const u32 as *const _);
                encoder.set_bytes(7, 4, &has_bias as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// Convert F16 tensor to F32.
    pub(crate) fn gpu_f16_to_f32(&self, input: &Tensor) -> Result<Tensor> {
        let count = input.shape().numel();
        let output = Tensor::empty(input.shape().clone(), DType::F32, input.device())?;

        let cb = self.compute.new_command_buffer();
        let in_buf = borrow_tensor(input)?;
        let o_buf = borrow_tensor(&output)?;
        let c_count = count as u32;

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.f16_to_f32,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(2, 4, &c_count as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    pub(crate) fn gpu_relative_position_bias(&self, bias_table: &LazyTensor, q_len: usize, k_len: usize, bidirectional: bool) -> Result<Tensor> {
        let config = &self.config;
        let device_id = self.compute.device().info().id;
        let output = Tensor::empty(
            Shape::from([config.num_heads, q_len, k_len]),
            DType::F16, device_id,
        )?;

        let cb = self.compute.new_command_buffer();
        let o_buf = borrow_tensor(&output)?;
        let c_q = q_len as u32;
        let c_k = k_len as u32;
        let c_heads = config.num_heads as u32;
        let c_buckets = config.num_buckets as u32;
        let c_max_dist = config.max_distance as u32;
        let c_bidir = if bidirectional { 1u32 } else { 0u32 };

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.rel_pos_bias,
            ((k_len + 7) / 8, (q_len + 7) / 8, config.num_heads),
            (8, 8, 1), |encoder| {
                encoder.set_buffer(0, Some(bias_table.buffer()), 0);
                encoder.set_buffer(1, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(2, 4, &c_q as *const u32 as *const _);
                encoder.set_bytes(3, 4, &c_k as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_heads as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_buckets as *const u32 as *const _);
                encoder.set_bytes(6, 4, &c_max_dist as *const u32 as *const _);
                encoder.set_bytes(7, 4, &c_bidir as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    pub(crate) fn gpu_attention(
        &self, q: &Tensor, k: &Tensor, v: &Tensor, bias: &Tensor,
        seq_len: usize, num_heads: usize, d_kv: usize,
    ) -> Result<Tensor> {
        // Transpose [seq_len, num_heads*d_kv] → [num_heads, seq_len, d_kv]
        let q_t = self.gpu_transpose_shd_to_hsd(q, seq_len, num_heads, d_kv)?;
        let k_t = self.gpu_transpose_shd_to_hsd(k, seq_len, num_heads, d_kv)?;
        let v_t = self.gpu_transpose_shd_to_hsd(v, seq_len, num_heads, d_kv)?;

        // Q@K^T: [num_heads, seq, d_kv] × [num_heads, d_kv, seq] → [num_heads, seq, seq]
        let scores = self.gpu_batched_qk(&q_t, &k_t, num_heads, seq_len, d_kv)?;

        // T5 uses UNSCALED attention (no 1/sqrt(d_kv)): Q@K^T + bias → softmax
        let scale = 1.0_f32;
        let scaled = self.gpu_add_scale_softmax(&scores, bias, scale, num_heads, seq_len)?;

        // S@V: [num_heads, seq, seq] × [num_heads, seq, d_kv] → [num_heads, seq, d_kv]
        let attn_out = self.gpu_batched_sv(&scaled, &v_t, num_heads, seq_len, d_kv)?;

        // Transpose back: [num_heads, seq, d_kv] → [seq, num_heads*d_kv]
        self.gpu_transpose_hsd_to_shd(&attn_out, seq_len, num_heads, d_kv)
    }

    /// F32 attention: Q@K^T in F32, add bias + softmax in F32, S@V in F16.
    /// Prevents precision loss compounding over deep models (24+ layers).
    pub(crate) fn gpu_attention_f32(
        &self, q: &Tensor, k: &Tensor, v: &Tensor, bias: &Tensor,
        seq_len: usize, num_heads: usize, d_kv: usize,
    ) -> Result<Tensor> {
        // Transpose [seq_len, num_heads*d_kv] → [num_heads, seq_len, d_kv]
        let q_t = self.gpu_transpose_shd_to_hsd(q, seq_len, num_heads, d_kv)?;
        let k_t = self.gpu_transpose_shd_to_hsd(k, seq_len, num_heads, d_kv)?;
        let v_t = self.gpu_transpose_shd_to_hsd(v, seq_len, num_heads, d_kv)?;

        // Q@K^T → F32 scores: [num_heads, seq, d_kv] × [num_heads, d_kv, seq] → [num_heads, seq, seq]
        let scores = self.gpu_batched_qk_f32(&q_t, &k_t, num_heads, seq_len, d_kv)?;

        // Add F16 bias + softmax entirely in F32 → F16 weights
        let weights = self.gpu_add_bias_softmax_f32(&scores, bias, num_heads, seq_len, seq_len)?;

        // S@V: [num_heads, seq, seq] × [num_heads, seq, d_kv] → [num_heads, seq, d_kv]
        let attn_out = self.gpu_batched_sv(&weights, &v_t, num_heads, seq_len, d_kv)?;

        // Transpose back: [num_heads, seq, d_kv] → [seq, num_heads*d_kv]
        self.gpu_transpose_hsd_to_shd(&attn_out, seq_len, num_heads, d_kv)
    }

    /// F32 cross-attention: Q@K^T in F32, softmax in F32 (no bias), S@V in F16.
    pub(crate) fn gpu_cross_attention_f32(
        &self, q: &Tensor, k: &Tensor, v: &Tensor,
        q_len: usize, kv_len: usize, num_heads: usize, d_kv: usize,
    ) -> Result<Tensor> {
        // Transpose to [num_heads, seq, d_kv]
        let q_t = self.gpu_transpose_shd_to_hsd(q, q_len, num_heads, d_kv)?;
        let k_t = self.gpu_transpose_shd_to_hsd(k, kv_len, num_heads, d_kv)?;
        let v_t = self.gpu_transpose_shd_to_hsd(v, kv_len, num_heads, d_kv)?;

        // Q@K^T → F32 scores: [num_heads, q_len, kv_len]
        let scores = self.gpu_batched_cross_qk_f32(&q_t, &k_t, num_heads, q_len, kv_len, d_kv)?;

        // Softmax in F32 → F16 weights (no bias)
        let weights = self.gpu_softmax_f32(&scores, num_heads, q_len, kv_len)?;

        // S@V: [num_heads, q_len, kv_len] × [num_heads, kv_len, d_kv] → [num_heads, q_len, d_kv]
        let attn_out = self.gpu_batched_cross_sv(&weights, &v_t, num_heads, q_len, kv_len, d_kv)?;

        // Transpose back: [num_heads, q_len, d_kv] → [q_len, num_heads*d_kv]
        self.gpu_transpose_hsd_to_shd(&attn_out, q_len, num_heads, d_kv)
    }

    /// Q@K^T with F32 output (same-length Q and K).
    fn gpu_batched_qk_f32(&self, q: &Tensor, k: &Tensor, batch: usize, seq_len: usize, d_kv: usize) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([batch, seq_len, seq_len]), DType::F32, q.device())?;
        let cb = self.compute.new_command_buffer();
        let q_buf = borrow_tensor(q)?;
        let k_buf = borrow_tensor(k)?;
        let o_buf = borrow_tensor(&output)?;
        let (c_m, c_n, c_k) = (seq_len as u32, seq_len as u32, d_kv as u32);

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.batched_linear_f32_out,
            ((seq_len + 15) / 16, (seq_len + 15) / 16, batch), (16, 16, 1), |encoder| {
                encoder.set_buffer(0, Some(q_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(k_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &c_m as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_k as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// Q@K^T with F32 output (cross-attention: different Q and K lengths).
    pub(crate) fn gpu_batched_cross_qk_f32(
        &self, q: &Tensor, k: &Tensor,
        batch: usize, q_len: usize, kv_len: usize, d_kv: usize,
    ) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([batch, q_len, kv_len]), DType::F32, q.device())?;
        let cb = self.compute.new_command_buffer();
        let q_buf = borrow_tensor(q)?;
        let k_buf = borrow_tensor(k)?;
        let o_buf = borrow_tensor(&output)?;
        let (c_m, c_n, c_k) = (q_len as u32, kv_len as u32, d_kv as u32);

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.batched_linear_f32_out,
            ((kv_len + 15) / 16, (q_len + 15) / 16, batch), (16, 16, 1), |encoder| {
                encoder.set_buffer(0, Some(q_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(k_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &c_m as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_k as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// Add F16 bias to F32 scores, softmax in F32, output F16 weights.
    fn gpu_add_bias_softmax_f32(&self, scores: &Tensor, bias: &Tensor, num_heads: usize, q_len: usize, kv_len: usize) -> Result<Tensor> {
        let total_rows = num_heads * q_len;
        let output = Tensor::empty(Shape::from([num_heads, q_len, kv_len]), DType::F16, scores.device())?;

        let cb = self.compute.new_command_buffer();
        let s_buf = borrow_tensor(scores)?;
        let b_buf = borrow_tensor(bias)?;
        let o_buf = borrow_tensor(&output)?;
        let c_rows = total_rows as u32;
        let c_cols = kv_len as u32;

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.add_bias_softmax_f32,
            (total_rows, 1, 1), (1, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(s_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(b_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &c_rows as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_cols as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// Softmax on F32 scores → F16 output (no bias, for cross-attention).
    pub(crate) fn gpu_softmax_f32(&self, scores: &Tensor, num_heads: usize, q_len: usize, kv_len: usize) -> Result<Tensor> {
        let total_rows = num_heads * q_len;
        let output = Tensor::empty(Shape::from([num_heads, q_len, kv_len]), DType::F16, scores.device())?;

        let cb = self.compute.new_command_buffer();
        let s_buf = borrow_tensor(scores)?;
        let o_buf = borrow_tensor(&output)?;
        let c_rows = total_rows as u32;
        let c_cols = kv_len as u32;

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.softmax_f32,
            (total_rows, 1, 1), (1, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(s_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(2, 4, &c_rows as *const u32 as *const _);
                encoder.set_bytes(3, 4, &c_cols as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    pub(crate) fn gpu_transpose_shd_to_hsd(&self, input: &Tensor, s: usize, h: usize, d: usize) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([h, s, d]), DType::F16, input.device())?;
        let cb = self.compute.new_command_buffer();
        let i_buf = borrow_tensor(input)?;
        let o_buf = borrow_tensor(&output)?;
        let (c_s, c_h, c_d) = (s as u32, h as u32, d as u32);

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.transpose_shd_hsd,
            ((d + 15) / 16, (s + 15) / 16, h), (16, 16, 1), |encoder| {
                encoder.set_buffer(0, Some(i_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(2, 4, &c_s as *const u32 as *const _);
                encoder.set_bytes(3, 4, &c_h as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_d as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    pub(crate) fn gpu_transpose_hsd_to_shd(&self, input: &Tensor, s: usize, h: usize, d: usize) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([s, h * d]), DType::F16, input.device())?;
        let cb = self.compute.new_command_buffer();
        let i_buf = borrow_tensor(input)?;
        let o_buf = borrow_tensor(&output)?;
        let (c_s, c_h, c_d) = (s as u32, h as u32, d as u32);

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.transpose_hsd_shd,
            ((d + 15) / 16, (s + 15) / 16, h), (16, 16, 1), |encoder| {
                encoder.set_buffer(0, Some(i_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(2, 4, &c_s as *const u32 as *const _);
                encoder.set_bytes(3, 4, &c_h as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_d as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    pub(crate) fn gpu_batched_qk(&self, q: &Tensor, k: &Tensor, batch: usize, seq_len: usize, d_kv: usize) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([batch, seq_len, seq_len]), DType::F16, q.device())?;
        let cb = self.compute.new_command_buffer();
        let q_buf = borrow_tensor(q)?;
        let k_buf = borrow_tensor(k)?;
        let o_buf = borrow_tensor(&output)?;
        let (c_m, c_n, c_k, c_batch) = (seq_len as u32, seq_len as u32, d_kv as u32, batch as u32);

        // batched_linear_f16: TILE=16, grid: (ceil(N/16), ceil(M/16), B)
        self.compute.dispatch_async(cb.as_ref(), &self.kernels.batched_linear,
            ((seq_len + 15) / 16, (seq_len + 15) / 16, batch), (16, 16, 1), |encoder| {
                encoder.set_buffer(0, Some(q_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(k_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &c_m as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_k as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    pub(crate) fn gpu_add_scale_softmax(&self, scores: &Tensor, bias: &Tensor, scale: f32, num_heads: usize, seq_len: usize) -> Result<Tensor> {
        // Step 1: Add bias to scores (element-wise)
        let combined = self.gpu_add(scores, bias)?;

        // Step 2: In-place scaled softmax on combined
        // row_softmax_scale_f16: data(0), rows(1), cols(2), scale(3)
        let cb = self.compute.new_command_buffer();
        let c_buf = borrow_tensor(&combined)?;
        let c_rows = (num_heads * seq_len) as u32;
        let c_cols = seq_len as u32;

        self.compute.dispatch_async(cb.as_ref(), &self.kernels.row_softmax,
            (num_heads * seq_len, 1, 1), (1, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(c_buf.as_ref()), 0);
                encoder.set_bytes(1, 4, &c_rows as *const u32 as *const _);
                encoder.set_bytes(2, 4, &c_cols as *const u32 as *const _);
                encoder.set_bytes(3, 4, &scale as *const f32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(combined)
    }

    pub(crate) fn gpu_batched_sv(&self, scores: &Tensor, v: &Tensor, batch: usize, seq_len: usize, d_kv: usize) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([batch, seq_len, d_kv]), DType::F16, scores.device())?;
        let cb = self.compute.new_command_buffer();
        let s_buf = borrow_tensor(scores)?;
        let v_buf = borrow_tensor(v)?;
        let o_buf = borrow_tensor(&output)?;
        let (c_m, c_n, c_k, c_batch) = (seq_len as u32, d_kv as u32, seq_len as u32, batch as u32);

        // batched_matmul_nn_f16: TILE=16, grid: (ceil(N/16), ceil(M/16), B)
        self.compute.dispatch_async(cb.as_ref(), &self.kernels.batched_matmul_nn,
            ((d_kv + 15) / 16, (seq_len + 15) / 16, batch), (16, 16, 1), |encoder| {
                encoder.set_buffer(0, Some(s_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(v_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &c_m as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_k as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// S@V for cross-attention: S=[batch, q_len, kv_len], V=[batch, kv_len, d_kv] → [batch, q_len, d_kv].
    /// Unlike gpu_batched_sv, correctly handles different q_len and kv_len (K dimension = kv_len).
    pub(crate) fn gpu_batched_cross_sv(
        &self, scores: &Tensor, v: &Tensor,
        batch: usize, q_len: usize, kv_len: usize, d_kv: usize,
    ) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([batch, q_len, d_kv]), DType::F16, scores.device())?;
        let cb = self.compute.new_command_buffer();
        let s_buf = borrow_tensor(scores)?;
        let v_buf = borrow_tensor(v)?;
        let o_buf = borrow_tensor(&output)?;
        let (c_m, c_n, c_k) = (q_len as u32, d_kv as u32, kv_len as u32);

        // batched_matmul_nn_f16: Y = A @ B, A=[M,K], B=[K,N], Y=[M,N]
        self.compute.dispatch_async(cb.as_ref(), &self.kernels.batched_matmul_nn,
            ((d_kv + 15) / 16, (q_len + 15) / 16, batch), (16, 16, 1), |encoder| {
                encoder.set_buffer(0, Some(s_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(v_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &c_m as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_k as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }
}

#[cfg(feature = "metal")]
pub(crate) fn borrow_tensor(tensor: &Tensor) -> Result<BorrowedMetalBuffer> {
    let ptr = tensor.device_ptr()
        .ok_or_else(|| crate::core::Error::internal("tensor not on device"))?;
    Ok(unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
}
