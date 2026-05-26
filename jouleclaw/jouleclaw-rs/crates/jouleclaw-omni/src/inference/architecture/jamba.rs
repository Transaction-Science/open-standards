//! Jamba 2: hybrid SSM+Attention language model from AI21 Labs.
//!
//! Architecture:
//!   - Alternates Mamba (SSM) layers with standard attention layers
//!   - Mamba layers use selective state spaces (S6) for O(n) sequence processing
//!   - Attention layers use standard GQA (grouped query attention) with RoPE
//!   - MoE on even layers (expert_layer_offset=1, expert_layer_period=2)
//!   - Attention on periodic layers (attn_layer_offset, attn_layer_period)
//!
//! Jamba 2 3B:  28 layers, hidden=2560, 20 heads, 1 KV head, no MoE (1 expert)
//! Jamba 2 Mini: 32 layers, hidden=4096, 32 heads, 8 KV heads, 16 experts top-2
//!
//! Weight layout (safetensors):
//!   Mamba layers:  model.layers.{i}.mamba.{in_proj,x_proj,dt_proj,out_proj,conv1d,A_log,D,
//!                                          b_layernorm,c_layernorm,dt_layernorm}
//!   Attention:     model.layers.{i}.self_attn.{q,k,v,o}_proj
//!   Dense FFN:     model.layers.{i}.feed_forward.{gate,up,down}_proj
//!   MoE FFN:       model.layers.{i}.feed_forward.{router,experts.{E}.{gate,up,down}_proj}
//!   Norms:         model.layers.{i}.{input_layernorm,pre_ff_layernorm}

use crate::core::{Error, Result};
#[cfg(feature = "metal")]
use crate::tensor::{DType, Shape, Tensor};
#[cfg(feature = "metal")]
use std::sync::Arc;
#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::inference::llm::{PagedKVCache, MoeRouter};
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline, LazyTensor};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;

// ── Configuration ────────────────────────────────────────────────────────────

/// Jamba 2 configuration (from config.json).
#[derive(Debug, Clone)]
pub struct JambaConfig {
    /// Hidden size (embedding dimension).
    pub hidden_size: usize,
    /// Intermediate size for FFN (dense or per-expert).
    pub intermediate_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA).
    pub num_kv_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum position embeddings.
    pub max_position_embeddings: usize,
    /// RMS norm epsilon.
    pub rms_norm_eps: f32,
    /// Tie word embeddings (lm_head shares embed_tokens).
    pub tie_word_embeddings: bool,

    // ── Mamba SSM parameters ─────────────────────────────────────────────
    /// Mamba state dimension (d_state, typically 16).
    pub mamba_d_state: usize,
    /// Mamba convolution kernel width (d_conv, typically 4).
    pub mamba_d_conv: usize,
    /// Mamba expansion factor (inner_dim = hidden_size * expand).
    pub mamba_expand: usize,
    /// Mamba dt rank (rank of delta projection).
    pub mamba_dt_rank: usize,
    /// Mamba conv1d bias.
    pub mamba_conv_bias: bool,
    /// Mamba projection bias.
    pub mamba_proj_bias: bool,

    // ── Layer routing ────────────────────────────────────────────────────
    /// Attention layer offset (first attention layer index).
    pub attn_layer_offset: usize,
    /// Attention layer period (every N-th layer uses attention).
    pub attn_layer_period: usize,
    /// Expert (MoE) layer offset.
    pub expert_layer_offset: usize,
    /// Expert (MoE) layer period.
    pub expert_layer_period: usize,

    // ── MoE parameters ──────────────────────────────────────────────────
    /// Number of experts (1 = dense, no routing).
    pub num_experts: usize,
    /// Number of active experts per token.
    pub num_experts_per_tok: usize,
}

impl Default for JambaConfig {
    /// Jamba 2 3B defaults.
    fn default() -> Self {
        Self {
            hidden_size: 2560,
            intermediate_size: 8192,
            num_layers: 28,
            num_attention_heads: 20,
            num_kv_heads: 1,
            vocab_size: 65536,
            max_position_embeddings: 262144,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            mamba_d_state: 16,
            mamba_d_conv: 4,
            mamba_expand: 2,
            mamba_dt_rank: 160,
            mamba_conv_bias: true,
            mamba_proj_bias: false,
            attn_layer_offset: 7,
            attn_layer_period: 14,
            expert_layer_offset: 1,
            expert_layer_period: 2,
            num_experts: 1,
            num_experts_per_tok: 1,
        }
    }
}

impl JambaConfig {
    /// Jamba 2 Mini defaults (52B total, ~12B active).
    pub fn mini() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 14336,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            vocab_size: 65536,
            max_position_embeddings: 262144,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            mamba_d_state: 16,
            mamba_d_conv: 4,
            mamba_expand: 2,
            mamba_dt_rank: 256,
            mamba_conv_bias: true,
            mamba_proj_bias: false,
            attn_layer_offset: 4,
            attn_layer_period: 8,
            expert_layer_offset: 1,
            expert_layer_period: 2,
            num_experts: 16,
            num_experts_per_tok: 2,
        }
    }

    /// Parse from config.json.
    pub fn from_json(path: &std::path::Path) -> Result<Self> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| Error::internal(format!("failed to read config: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Error::internal(format!("failed to parse config: {}", e)))?;

        let mut c = Self::default();
        if let Some(v) = json.get("hidden_size").and_then(|v| v.as_u64()) { c.hidden_size = v as usize; }
        if let Some(v) = json.get("intermediate_size").and_then(|v| v.as_u64()) { c.intermediate_size = v as usize; }
        if let Some(v) = json.get("num_hidden_layers").and_then(|v| v.as_u64()) { c.num_layers = v as usize; }
        if let Some(v) = json.get("num_attention_heads").and_then(|v| v.as_u64()) { c.num_attention_heads = v as usize; }
        if let Some(v) = json.get("num_key_value_heads").and_then(|v| v.as_u64()) { c.num_kv_heads = v as usize; }
        if let Some(v) = json.get("vocab_size").and_then(|v| v.as_u64()) { c.vocab_size = v as usize; }
        if let Some(v) = json.get("max_position_embeddings").and_then(|v| v.as_u64()) { c.max_position_embeddings = v as usize; }
        if let Some(v) = json.get("rms_norm_eps").and_then(|v| v.as_f64()) { c.rms_norm_eps = v as f32; }
        if let Some(v) = json.get("tie_word_embeddings").and_then(|v| v.as_bool()) { c.tie_word_embeddings = v; }
        if let Some(v) = json.get("mamba_d_state").and_then(|v| v.as_u64()) { c.mamba_d_state = v as usize; }
        if let Some(v) = json.get("mamba_d_conv").and_then(|v| v.as_u64()) { c.mamba_d_conv = v as usize; }
        if let Some(v) = json.get("mamba_expand").and_then(|v| v.as_u64()) { c.mamba_expand = v as usize; }
        if let Some(v) = json.get("mamba_dt_rank").and_then(|v| v.as_u64()) { c.mamba_dt_rank = v as usize; }
        if let Some(v) = json.get("mamba_conv_bias").and_then(|v| v.as_bool()) { c.mamba_conv_bias = v; }
        if let Some(v) = json.get("mamba_proj_bias").and_then(|v| v.as_bool()) { c.mamba_proj_bias = v; }
        if let Some(v) = json.get("attn_layer_offset").and_then(|v| v.as_u64()) { c.attn_layer_offset = v as usize; }
        if let Some(v) = json.get("attn_layer_period").and_then(|v| v.as_u64()) { c.attn_layer_period = v as usize; }
        if let Some(v) = json.get("expert_layer_offset").and_then(|v| v.as_u64()) { c.expert_layer_offset = v as usize; }
        if let Some(v) = json.get("expert_layer_period").and_then(|v| v.as_u64()) { c.expert_layer_period = v as usize; }
        if let Some(v) = json.get("num_experts").and_then(|v| v.as_u64()) { c.num_experts = v as usize; }
        if let Some(v) = json.get("num_experts_per_tok").and_then(|v| v.as_u64()) { c.num_experts_per_tok = v as usize; }
        Ok(c)
    }

    /// Whether a given layer uses attention (vs Mamba SSM).
    pub fn is_attention_layer(&self, layer_idx: usize) -> bool {
        if layer_idx < self.attn_layer_offset {
            return false;
        }
        (layer_idx - self.attn_layer_offset) % self.attn_layer_period == 0
    }

    /// Whether a given layer uses MoE (vs dense FFN).
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        if self.num_experts <= 1 {
            return false;
        }
        if layer_idx < self.expert_layer_offset {
            return false;
        }
        (layer_idx - self.expert_layer_offset) % self.expert_layer_period == 0
    }

    /// Inner Mamba dimension (hidden_size * expand).
    pub fn mamba_inner_dim(&self) -> usize {
        self.hidden_size * self.mamba_expand
    }

    /// Head dimension for attention layers.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// Jamba 2 hybrid SSM+Attention inference pipeline.
///
/// Each layer is either:
///   - Mamba SSM: O(n) selective scan with conv1d + input-dependent gating
///   - Standard GQA attention with RoPE
///
/// Followed by either dense FFN or MoE FFN depending on layer index.
#[cfg(feature = "metal")]
pub struct JambaPipeline {
    model: Arc<parking_lot::RwLock<Model>>,
    compute: Arc<MetalCompute>,
    config: JambaConfig,
    kernels: JambaKernels,
    /// Per-layer Mamba SSM state: (ssm_state, conv_state).
    /// ssm_state: [mamba_inner_dim, d_state] — the recurrent hidden state.
    /// conv_state: [mamba_inner_dim, d_conv-1] — causal conv1d sliding window.
    /// Only allocated for Mamba layers (attention layers have None).
    mamba_states: Vec<Option<MambaState>>,
}

/// Mamba recurrent state for a single layer.
#[cfg(feature = "metal")]
struct MambaState {
    /// SSM hidden state: [inner_dim, d_state]
    ssm_state: Tensor,
    /// Causal conv1d buffer: [inner_dim, d_conv]
    conv_state: Tensor,
}

/// Compiled Metal kernels for Jamba operations.
#[cfg(feature = "metal")]
#[allow(dead_code)]
struct JambaKernels {
    matmul: Arc<ComputePipeline>,
    matmul_q4k: Arc<ComputePipeline>,
    matmul_q8_0: Arc<ComputePipeline>,
    rms_norm: Arc<ComputePipeline>,
    rms_norm_tg: Arc<ComputePipeline>,
    silu: Arc<ComputePipeline>,
    rope: Arc<ComputePipeline>,
    gqa_attention: Arc<ComputePipeline>,
    autoregressive_attention: Arc<ComputePipeline>,
    add: Arc<ComputePipeline>,
    mul: Arc<ComputePipeline>,
    matvec: Arc<ComputePipeline>,
}

#[cfg(feature = "metal")]
impl JambaPipeline {
    /// Create a new Jamba pipeline.
    pub fn new(model: Arc<parking_lot::RwLock<Model>>, config: JambaConfig, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        let kernels = JambaKernels {
            matmul: compute.compile_pipeline("matmul", sources::MATMUL, "matmul_tiled_f16")?,
            matmul_q4k: compute.compile_pipeline("matmul_q4k", sources::MATMUL_Q4K, "matmul_q4k_f16")?,
            matmul_q8_0: compute.compile_pipeline("matmul_q8_0", sources::MATMUL_Q8_0, "matmul_q8_0_f16")?,
            rms_norm: compute.compile_pipeline("rms_norm", sources::RMS_NORM, "rms_norm_f16")?,
            rms_norm_tg: compute.compile_pipeline("rms_norm_tg", sources::RMS_NORM, "rms_norm_tg_f16")?,
            silu: compute.compile_pipeline("silu", sources::SILU, "silu_f16")?,
            rope: compute.compile_pipeline("rope", sources::ROPE, "rope_f16")?,
            gqa_attention: compute.compile_pipeline("gqa_attention", sources::GQA_ATTENTION, "gqa_attention_f16")?,
            autoregressive_attention: compute.compile_pipeline("autoregressive_attention", sources::AUTOREGRESSIVE_ATTENTION, "autoregressive_attention_tg_f16")?,
            add: compute.compile_pipeline("add", sources::ELEMENTWISE, "add_f16")?,
            mul: compute.compile_pipeline("mul", sources::ELEMENTWISE, "mul_f16")?,
            matvec: compute.compile_pipeline("matvec", sources::MATVEC, "matvec_f16")?,
        };

        // Allocate Mamba states for SSM layers
        let device_id = compute.device().info().id;
        let inner_dim = config.mamba_inner_dim();
        let mut mamba_states = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            if config.is_attention_layer(layer_idx) {
                mamba_states.push(None);
            } else {
                let ssm_state = Tensor::zeros_on(
                    Shape::from([inner_dim, config.mamba_d_state]),
                    DType::F16,
                    device_id,
                )?;
                let conv_state = Tensor::zeros_on(
                    Shape::from([inner_dim, config.mamba_d_conv]),
                    DType::F16,
                    device_id,
                )?;
                mamba_states.push(Some(MambaState { ssm_state, conv_state }));
            }
        }

        Ok(Self { model, compute, config, kernels, mamba_states })
    }

    /// Reset all Mamba recurrent states (call between sequences).
    pub fn reset_mamba_states(&mut self) -> Result<()> {
        let device_id = self.compute.device().info().id;
        let inner_dim = self.config.mamba_inner_dim();
        for state in self.mamba_states.iter_mut().flatten() {
            state.ssm_state = Tensor::zeros_on(
                Shape::from([inner_dim, self.config.mamba_d_state]),
                DType::F16,
                device_id,
            )?;
            state.conv_state = Tensor::zeros_on(
                Shape::from([inner_dim, self.config.mamba_d_conv]),
                DType::F16,
                device_id,
            )?;
        }
        Ok(())
    }

    /// Forward pass through a single Jamba layer.
    ///
    /// Routes to either Mamba SSM or GQA attention based on layer index,
    /// then through dense FFN or MoE FFN.
    pub fn forward_layer(
        &mut self,
        layer_idx: usize,
        input: &Tensor,
        kv_cache: &mut PagedKVCache,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let prefix = format!("model.layers.{}", layer_idx);

        // Pre-mixer layernorm
        let norm_w = self.model.read().get_weight(&format!("{}.input_layernorm.weight", prefix));
        let normed = self.rms_norm(input, norm_w)?;

        // Mixer: Mamba SSM or GQA attention
        let mixer_out = if self.config.is_attention_layer(layer_idx) {
            self.forward_attention(layer_idx, &normed, kv_cache, start_pos, seq_len)?
        } else {
            self.forward_mamba(layer_idx, &normed, seq_len)?
        };

        // Residual after mixer
        let hidden = self.add(input, &mixer_out)?;

        // Pre-FFN layernorm
        let ff_norm_w = self.model.read().get_weight(&format!("{}.pre_ff_layernorm.weight", prefix));
        let normed_ff = self.rms_norm(&hidden, ff_norm_w)?;

        // FFN: dense or MoE
        let ffn_out = if self.config.is_moe_layer(layer_idx) {
            self.forward_moe(layer_idx, &normed_ff, seq_len)?
        } else {
            self.forward_dense_ffn(layer_idx, &normed_ff)?
        };

        // Final residual
        self.add(&hidden, &ffn_out)
    }

    // ── Mamba SSM Layer ──────────────────────────────────────────────────────

    /// Mamba selective state space layer.
    ///
    /// Architecture:
    ///   x_and_z = input @ in_proj                    [seq, 2*inner_dim]
    ///   x, z = split(x_and_z)                        each [seq, inner_dim]
    ///   x = conv1d(x, conv_weight, conv_bias)        causal, width=d_conv
    ///   x = silu(x)
    ///   dt_bc = x @ x_proj                           [seq, dt_rank + 2*d_state]
    ///   dt_raw, B_raw, C_raw = split(dt_bc)
    ///   dt = dt_layernorm(dt_raw) @ dt_proj + dt_bias
    ///   B = b_layernorm(B_raw)
    ///   C = c_layernorm(C_raw)
    ///   dt = softplus(dt)                            [seq, inner_dim]
    ///   A = -exp(A_log)                              [inner_dim, d_state]
    ///
    ///   -- Selective scan (recurrence) --
    ///   For each time step t:
    ///     A_bar = exp(dt[t] * A)                     discretization
    ///     B_bar = dt[t] * B[t]                       input scaling
    ///     ssm_state = A_bar * ssm_state + B_bar * x[t]
    ///     y[t] = C[t] @ ssm_state                    readout
    ///   y = y + D * x                                skip connection
    ///   output = (y * silu(z)) @ out_proj
    fn forward_mamba(
        &mut self,
        layer_idx: usize,
        input: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        let prefix = format!("model.layers.{}.mamba", layer_idx);
        let config = &self.config;
        let inner_dim = config.mamba_inner_dim();
        let dt_rank = config.mamba_dt_rank;
        let d_state = config.mamba_d_state;
        let d_conv = config.mamba_d_conv;

        // in_proj: [hidden_size] -> [2 * inner_dim] (x and z combined)
        let in_proj_w = self.model.read().get_weight(&format!("{}.in_proj.weight", prefix));
        let xz = self.matmul(input, in_proj_w)?;
        // Split into x and z
        let x = xz.slice(1, 0, inner_dim)?;
        let z = xz.slice(1, inner_dim, 2 * inner_dim)?;

        // Causal conv1d over x — read weights to CPU eagerly to avoid borrow conflicts
        let conv_w_f32: Vec<f32> = self.model.read().get_weight(&format!("{}.conv1d.weight", prefix))
            .map(|lt| lt.to_f32_vec())
            .transpose()?
            .ok_or_else(|| Error::internal("missing conv1d weight"))?;
        let conv_b_f32: Option<Vec<f32>> = if config.mamba_conv_bias {
            self.model.read().get_weight(&format!("{}.conv1d.bias", prefix))
                .map(|lt| lt.to_f32_vec())
                .transpose()?
        } else {
            None
        };
        let x_conv = self.causal_conv1d_cpu(&x, &conv_w_f32, conv_b_f32.as_deref(), layer_idx, seq_len, inner_dim, d_conv)?;

        // SiLU activation on conv output
        let x_act = self.silu(&x_conv)?;

        // x_proj: [inner_dim] -> [dt_rank + 2*d_state]
        let x_proj_w = self.model.read().get_weight(&format!("{}.x_proj.weight", prefix));
        let dt_bc = self.matmul(&x_act, x_proj_w)?;

        // Split into dt_raw, B_raw, C_raw
        let dt_raw = dt_bc.slice(1, 0, dt_rank)?;
        let b_raw = dt_bc.slice(1, dt_rank, dt_rank + d_state)?;
        let c_raw = dt_bc.slice(1, dt_rank + d_state, dt_rank + 2 * d_state)?;

        // Layer norms on dt, B, C
        let dt_norm_w = self.model.read().get_weight(&format!("{}.dt_layernorm.weight", prefix));
        let dt_normed = self.rms_norm(&dt_raw, dt_norm_w)?;

        let b_norm_w = self.model.read().get_weight(&format!("{}.b_layernorm.weight", prefix));
        let b_normed = self.rms_norm(&b_raw, b_norm_w)?;

        let c_norm_w = self.model.read().get_weight(&format!("{}.c_layernorm.weight", prefix));
        let c_normed = self.rms_norm(&c_raw, c_norm_w)?;

        // dt_proj: [dt_rank] -> [inner_dim] + bias
        let dt_proj_w = self.model.read().get_weight(&format!("{}.dt_proj.weight", prefix));
        let dt_proj_b = self.model.read().get_weight(&format!("{}.dt_proj.bias", prefix));
        let dt_projected = self.matmul(&dt_normed, dt_proj_w)?;
        let dt_biased = if let Some(bias) = dt_proj_b {
            self.add_bias(&dt_projected, bias)?
        } else {
            dt_projected
        };

        // softplus(dt) = log(1 + exp(dt))
        let dt = self.softplus(&dt_biased)?;

        // Load A_log and D to CPU eagerly (before mutable borrow of mamba_states)
        let a_log_f32 = self.model.read().get_weight(&format!("{}.A_log", prefix))
            .map(|lt| lt.to_f32_vec())
            .transpose()?
            .unwrap_or_else(|| vec![0.0f32; inner_dim * d_state]);
        let d_f32_raw = self.model.read().get_weight(&format!("{}.D", prefix))
            .map(|lt| lt.to_f32_vec())
            .transpose()?
            .unwrap_or_else(|| vec![1.0f32; inner_dim]);

        // Selective scan — read Mamba state to CPU for the recurrence
        // For production: this should be a custom Metal kernel. For correctness,
        // we run the recurrence on CPU with GPU transfer at boundaries.
        let x_f16: Vec<half::f16> = x_act.to_vec()?;
        let dt_f16: Vec<half::f16> = dt.to_vec()?;
        let b_f16: Vec<half::f16> = b_normed.to_vec()?;
        let c_f16: Vec<half::f16> = c_normed.to_vec()?;

        // Get mutable reference to this layer's Mamba state
        let mamba_state = self.mamba_states[layer_idx].as_mut()
            .ok_or_else(|| Error::internal(format!("no mamba state for layer {}", layer_idx)))?;

        let mut ssm_f32: Vec<f32> = mamba_state.ssm_state.to_vec::<half::f16>()?
            .iter().map(|v| v.to_f32()).collect();

        // Compute A = -exp(A_log)
        let a_f32: Vec<f32> = a_log_f32.iter().map(|v: &f32| -v.exp()).collect();
        let d_f32: Vec<f32> = d_f32_raw;

        // Run selective scan recurrence on CPU
        let mut y_f32 = vec![0.0f32; seq_len * inner_dim];
        for t in 0..seq_len {
            for d in 0..inner_dim {
                let x_val = x_f16[t * inner_dim + d].to_f32();
                let dt_val = dt_f16[t * inner_dim + d].to_f32();

                for s in 0..d_state {
                    let a_val = a_f32[d * d_state + s];
                    let b_val = b_f16[t * d_state + s].to_f32();

                    // Discretization: A_bar = exp(dt * A)
                    let a_bar = (dt_val * a_val).exp();
                    // B_bar = dt * B
                    let b_bar = dt_val * b_val;

                    // SSM recurrence: h = A_bar * h + B_bar * x
                    let state_idx = d * d_state + s;
                    ssm_f32[state_idx] = a_bar * ssm_f32[state_idx] + b_bar * x_val;

                    // Readout: y += C * h
                    let c_val = c_f16[t * d_state + s].to_f32();
                    y_f32[t * inner_dim + d] += c_val * ssm_f32[state_idx];
                }

                // Skip connection: y += D * x
                y_f32[t * inner_dim + d] += d_f32[d] * x_val;
            }
        }

        // Write SSM state back to GPU
        let device_id = input.device();
        let ssm_f16: Vec<half::f16> = ssm_f32.iter().map(|v| half::f16::from_f32(*v)).collect();
        mamba_state.ssm_state = Tensor::from_slice(
            &ssm_f16,
            Shape::from([inner_dim, d_state]),
            DType::F16,
            device_id,
        )?;

        // y tensor back to GPU
        let y_f16: Vec<half::f16> = y_f32.iter().map(|v| half::f16::from_f32(*v)).collect();
        let y = Tensor::from_slice(
            &y_f16,
            Shape::from([seq_len, inner_dim]),
            DType::F16,
            device_id,
        )?;

        // Output gate: y * silu(z)
        let z_silu = self.silu(&z)?;
        let gated = self.mul(&y, &z_silu)?;

        // out_proj: [inner_dim] -> [hidden_size]
        let out_proj_w = self.model.read().get_weight(&format!("{}.out_proj.weight", prefix));
        self.matmul(&gated, out_proj_w)
    }

    // ── Attention Layer ──────────────────────────────────────────────────────

    /// Standard GQA attention with RoPE (same as LLM pipeline).
    fn forward_attention(
        &self,
        layer_idx: usize,
        input: &Tensor,
        kv_cache: &mut PagedKVCache,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let prefix = format!("model.layers.{}", layer_idx);
        let config = &self.config;
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;

        // QKV projections
        let q_proj = self.model.read().get_weight(&format!("{}.self_attn.q_proj.weight", prefix));
        let k_proj = self.model.read().get_weight(&format!("{}.self_attn.k_proj.weight", prefix));
        let v_proj = self.model.read().get_weight(&format!("{}.self_attn.v_proj.weight", prefix));
        let o_proj = self.model.read().get_weight(&format!("{}.self_attn.o_proj.weight", prefix));

        let q = self.matmul(input, q_proj)?;
        let k = self.matmul(input, k_proj)?;
        let v = self.matmul(input, v_proj)?;

        // Reshape: [seq, num_heads * head_dim] -> [seq, num_heads, head_dim]
        let q = q.reshape([seq_len, num_heads, head_dim])?;
        let k = k.reshape([seq_len, num_kv_heads, head_dim])?;
        let v = v.reshape([seq_len, num_kv_heads, head_dim])?;

        // Apply RoPE to Q and K
        let q = self.apply_rope(&q, start_pos, head_dim)?;
        let k = self.apply_rope(&k, start_pos, head_dim)?;

        // Update KV cache
        kv_cache.update(layer_idx, start_pos, &k, &v, &self.compute, 0)?;

        // Compute attention
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (cached_k, cached_v) = kv_cache.get(layer_idx)?;
        let cache_len = kv_cache.seq_len();

        let attn_out = if seq_len > 1 {
            self.prefill_attention(&q, &cached_k, &cached_v, scale,
                                   seq_len, num_heads, num_kv_heads, head_dim)?
        } else {
            let q_decode = q.reshape([num_heads, head_dim])?;
            let out = self.decode_attention(&q_decode, &cached_k, &cached_v, scale,
                                           cache_len - 1, num_heads, num_kv_heads, head_dim)?;
            out.reshape([1, num_heads, head_dim])?
        };

        // Merge heads: [seq, num_heads, head_dim] -> [seq, hidden_size]
        let attn_flat = attn_out.reshape([seq_len, num_heads * head_dim])?;
        self.matmul(&attn_flat, o_proj)
    }

    // ── FFN (Dense) ──────────────────────────────────────────────────────────

    /// Dense SwiGLU FFN: gate_proj * silu(up_proj) -> down_proj.
    fn forward_dense_ffn(&self, layer_idx: usize, input: &Tensor) -> Result<Tensor> {
        let prefix = format!("model.layers.{}.feed_forward", layer_idx);
        let gate = self.model.read().get_weight(&format!("{}.gate_proj.weight", prefix));
        let up = self.model.read().get_weight(&format!("{}.up_proj.weight", prefix));
        let down = self.model.read().get_weight(&format!("{}.down_proj.weight", prefix));

        let g = self.matmul(input, gate)?;
        let u = self.matmul(input, up)?;
        let g_silu = self.silu(&g)?;
        let gu = self.mul(&g_silu, &u)?;
        self.matmul(&gu, down)
    }

    // ── FFN (MoE) ────────────────────────────────────────────────────────────

    /// MoE FFN: router -> top-k experts -> weighted aggregation.
    fn forward_moe(&self, layer_idx: usize, input: &Tensor, seq_len: usize) -> Result<Tensor> {
        let prefix = format!("model.layers.{}.feed_forward", layer_idx);
        let config = &self.config;
        let device_id = input.device();

        // Router: [hidden_size] -> [num_experts]
        let router_w = self.model.read().get_weight(&format!("{}.router.weight", prefix));
        let gate_logits = self.matmul(input, router_w)?;
        let gate_f16: Vec<half::f16> = gate_logits.to_vec()?;
        let gate_f32: Vec<f32> = gate_f16.iter().map(|v| v.to_f32()).collect();

        let router = MoeRouter::new(config.num_experts, config.num_experts_per_tok, false);
        let routes = router.route(&gate_f32, seq_len, None);

        let mut token_outputs = Vec::with_capacity(seq_len);
        for (token_idx, (expert_ids, weights)) in routes.iter().enumerate() {
            let token = input.slice(0, token_idx, token_idx + 1)?;
            let mut accumulated: Option<Tensor> = None;

            for (i, &expert_id) in expert_ids.iter().enumerate() {
                let g = self.model.read().get_weight(&format!("{}.experts.{}.gate_proj.weight", prefix, expert_id));
                let u = self.model.read().get_weight(&format!("{}.experts.{}.up_proj.weight", prefix, expert_id));
                let d = self.model.read().get_weight(&format!("{}.experts.{}.down_proj.weight", prefix, expert_id));

                let gate = self.matmul(&token, g)?;
                let up = self.matmul(&token, u)?;
                let gate_silu = self.silu(&gate)?;
                let gu = self.mul(&gate_silu, &up)?;
                let expert_out = self.matmul(&gu, d)?;

                let w = half::f16::from_f32(weights[i]);
                let w_tensor = Tensor::from_slice(&[w], Shape::from([1, 1]), DType::F16, device_id)?;
                let scaled = self.mul(&expert_out, &w_tensor)?;

                accumulated = Some(match accumulated {
                    Some(acc) => self.add(&acc, &scaled)?,
                    None => scaled,
                });
            }

            token_outputs.push(match accumulated {
                Some(t) => t,
                None => Tensor::zeros_on(Shape::from([1, config.hidden_size]), DType::F16, device_id)?,
            });
        }

        Tensor::cat(&token_outputs, 0)
    }

    // ── Causal Conv1d ────────────────────────────────────────────────────────

    /// Causal 1D convolution with sliding window state.
    ///
    /// conv1d weight shape: [inner_dim, 1, d_conv] (depthwise)
    /// We maintain a conv_state buffer of [inner_dim, d_conv] per layer
    /// to implement causal padding across sequence boundaries.
    ///
    /// Takes pre-loaded f32 weights to avoid borrow conflicts with `self.model`.
    fn causal_conv1d_cpu(
        &mut self,
        x: &Tensor,
        w_f32: &[f32],
        bias_f32: Option<&[f32]>,
        layer_idx: usize,
        seq_len: usize,
        inner_dim: usize,
        d_conv: usize,
    ) -> Result<Tensor> {
        let device_id = x.device();

        // Read input and conv state to CPU
        let x_f16: Vec<half::f16> = x.to_vec()?;
        let mamba_state = self.mamba_states[layer_idx].as_mut()
            .ok_or_else(|| Error::internal(format!("no mamba state for layer {}", layer_idx)))?;
        let mut conv_buf: Vec<f32> = mamba_state.conv_state.to_vec::<half::f16>()?
            .iter().map(|v| v.to_f32()).collect();

        // Run depthwise causal conv1d on CPU
        let mut out_f32 = vec![0.0f32; seq_len * inner_dim];
        for t in 0..seq_len {
            // Shift conv buffer left and insert new x
            for d in 0..inner_dim {
                // Shift: move entries [1..d_conv] to [0..d_conv-1]
                for k in 0..d_conv - 1 {
                    conv_buf[d * d_conv + k] = conv_buf[d * d_conv + k + 1];
                }
                conv_buf[d * d_conv + d_conv - 1] = x_f16[t * inner_dim + d].to_f32();

                // Depthwise conv: dot product of conv_buf[d] with weight[d]
                let mut val = 0.0f32;
                for k in 0..d_conv {
                    let w_idx = d * d_conv + k;
                    // Weight tensor may be [inner_dim, 1, d_conv] flattened
                    let w_val = if w_idx < w_f32.len() {
                        w_f32[w_idx]
                    } else {
                        0.0
                    };
                    val += conv_buf[d * d_conv + k] * w_val;
                }

                // Add bias
                if let Some(ref bias) = bias_f32 {
                    if d < bias.len() {
                        val += bias[d];
                    }
                }

                out_f32[t * inner_dim + d] = val;
            }
        }

        // Write conv state back to GPU
        let conv_f16: Vec<half::f16> = conv_buf.iter().map(|v| half::f16::from_f32(*v)).collect();
        mamba_state.conv_state = Tensor::from_slice(
            &conv_f16,
            Shape::from([inner_dim, d_conv]),
            DType::F16,
            device_id,
        )?;

        // Output tensor back to GPU
        let out_f16: Vec<half::f16> = out_f32.iter().map(|v| half::f16::from_f32(*v)).collect();
        Tensor::from_slice(
            &out_f16,
            Shape::from([seq_len, inner_dim]),
            DType::F16,
            device_id,
        )
    }

    // ── Primitive operations (delegate to MetalCompute) ──────────────────────

    fn matmul(&self, _a: &Tensor, _b: Option<&LazyTensor>) -> Result<Tensor> {
        // Same delegation pattern as DeepSeek — reuses compiled kernels via MetalCompute
        let _ = (&self.kernels.matmul, &self.kernels.matmul_q4k, &self.kernels.matmul_q8_0);
        Err(Error::internal("Jamba matmul delegates to LLMPipeline — wire via shared kernel dispatch"))
    }

    fn rms_norm(&self, _input: &Tensor, _weight: Option<&LazyTensor>) -> Result<Tensor> {
        let _ = (&self.kernels.rms_norm, &self.kernels.rms_norm_tg);
        Err(Error::internal("Jamba rms_norm delegates to LLMPipeline"))
    }

    fn silu(&self, _input: &Tensor) -> Result<Tensor> {
        let _ = &self.kernels.silu;
        Err(Error::internal("Jamba silu delegates to LLMPipeline"))
    }

    fn add(&self, _a: &Tensor, _b: &Tensor) -> Result<Tensor> {
        let _ = &self.kernels.add;
        Err(Error::internal("Jamba add delegates to LLMPipeline"))
    }

    fn mul(&self, _a: &Tensor, _b: &Tensor) -> Result<Tensor> {
        let _ = &self.kernels.mul;
        Err(Error::internal("Jamba mul delegates to LLMPipeline"))
    }

    fn add_bias(&self, _input: &Tensor, _bias: &LazyTensor) -> Result<Tensor> {
        let _ = &self.kernels.add;
        Err(Error::internal("Jamba add_bias delegates to LLMPipeline"))
    }

    fn softplus(&self, _input: &Tensor) -> Result<Tensor> {
        // softplus(x) = log(1 + exp(x)), implemented via CPU for now
        let f16: Vec<half::f16> = _input.to_vec()?;
        let f32_out: Vec<f32> = f16.iter().map(|v: &half::f16| {
            let x = v.to_f32();
            if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
        }).collect();
        let out_f16: Vec<half::f16> = f32_out.iter().map(|v| half::f16::from_f32(*v)).collect();
        Tensor::from_slice(&out_f16, _input.shape().clone(), DType::F16, _input.device())
    }

    fn apply_rope(&self, _input: &Tensor, _pos: usize, _dim: usize) -> Result<Tensor> {
        let _ = &self.kernels.rope;
        Err(Error::internal("Jamba apply_rope delegates to LLMPipeline"))
    }

    fn prefill_attention(&self, _q: &Tensor, _k: &Tensor, _v: &Tensor, _scale: f32,
                         _seq: usize, _nq: usize, _nkv: usize, _hd: usize) -> Result<Tensor> {
        let _ = &self.kernels.gqa_attention;
        Err(Error::internal("Jamba prefill_attention delegates to LLMPipeline"))
    }

    fn decode_attention(&self, _q: &Tensor, _k: &Tensor, _v: &Tensor, _scale: f32,
                        _pos: usize, _nq: usize, _nkv: usize, _hd: usize) -> Result<Tensor> {
        let _ = &self.kernels.autoregressive_attention;
        Err(Error::internal("Jamba decode_attention delegates to LLMPipeline"))
    }
}
