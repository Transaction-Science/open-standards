//! Mamba 2: Pure State Space Model with Structured State Space Duality (SSD).
//!
//! Reference: Gu & Dao, "Transformers are SSMs: Generalized Models and Efficient
//! Algorithms Through Structured State Space Duality" (2024).
//!
//! Architecture (NO attention layers):
//!   Token IDs → embedding lookup [vocab_size, d_model]
//!   → N × Mamba2Layer:
//!       input_layernorm (RMS norm)
//!       in_proj: [d_model] → [z || x || B || C || dt]
//!         z: [d_inner]          — output gate
//!         x: [d_inner]          — SSM input (goes through conv1d)
//!         B: [ngroups * d_state] — input-dependent state matrix
//!         C: [ngroups * d_state] — input-dependent readout matrix
//!         dt: [nheads]           — input-dependent discretization step
//!       conv1d on [x, B, C] (depthwise, kernel=d_conv, causal)
//!       SiLU activation on x
//!       Multi-head SSD selective scan:
//!         A = -exp(A_log)            [nheads]
//!         dt = softplus(dt + dt_bias) [nheads]
//!         Per head h (headdim channels each):
//!           state_h[t] = exp(dt[t]*A_h) * state_h[t-1] + dt[t]*B[t]*x_h[t]
//!           y_h[t] = C[t] · state_h[t] + D_h * x_h[t]
//!       group_norm on y (ngroups groups)
//!       y = y * silu(z)       — output gate
//!       out_proj: [d_inner] → [d_model]
//!       + residual
//!   → final RMS norm
//!   → lm_head: [d_model] → [vocab_size]
//!
//! Weight layout (safetensors, state-spaces/mamba2-1.3b and mamba2-2.7b):
//!   backbone.embedding.weight:          [vocab_size, d_model]
//!   backbone.layers.{i}.mixer.in_proj.weight:   [d_inner*2 + ngroups*d_state*2 + nheads, d_model]
//!   backbone.layers.{i}.mixer.conv1d.weight:    [d_inner + ngroups*d_state*2, 1, d_conv]
//!   backbone.layers.{i}.mixer.conv1d.bias:      [d_inner + ngroups*d_state*2]
//!   backbone.layers.{i}.mixer.dt_bias:          [nheads]
//!   backbone.layers.{i}.mixer.A_log:            [nheads]
//!   backbone.layers.{i}.mixer.D:                [nheads]
//!   backbone.layers.{i}.mixer.norm.weight:      [d_inner]
//!   backbone.layers.{i}.mixer.out_proj.weight:  [d_model, d_inner]
//!   backbone.layers.{i}.norm.weight:            [d_model]
//!   backbone.norm_f.weight:                     [d_model]
//!   lm_head.weight:                             [vocab_size, d_model]

use crate::core::{Error, Result};
#[cfg(feature = "metal")]
use crate::tensor::{DType, Shape, Tensor};
#[cfg(feature = "metal")]
use std::sync::Arc;
#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline, LazyTensor, BorrowedMetalBuffer};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;
#[cfg(feature = "metal")]
use crate::hal::metal::lazy_load::QuantType;

// ── Configuration ────────────────────────────────────────────────────────────

/// Mamba 2 configuration derived from config.json + weight tensor shapes.
#[derive(Debug, Clone)]
pub struct Mamba2Config {
    /// Model embedding dimension.
    pub d_model: usize,
    /// Number of Mamba 2 layers.
    pub n_layer: usize,
    /// SSM state dimension (derived: ngroups * d_state_per_group).
    /// For standard Mamba 2 checkpoints: ngroups=1, d_state=128.
    pub d_state: usize,
    /// Convolution kernel width (typically 4).
    pub d_conv: usize,
    /// Inner dimension = d_model * expand.
    pub d_inner: usize,
    /// Expansion factor (typically 2).
    pub expand: usize,
    /// Dimension per SSM head = d_inner / nheads.
    pub headdim: usize,
    /// Number of SSM heads = d_inner / headdim.
    pub nheads: usize,
    /// Number of groups for B/C projections (typically 1).
    pub ngroups: usize,
    /// Vocabulary size (padded to pad_vocab_size_multiple).
    pub vocab_size: usize,
    /// RMS norm epsilon.
    pub rms_norm_eps: f32,
    /// Whether lm_head shares embedding weights.
    pub tie_embeddings: bool,
    /// Whether to use residual in fp32 (original paper default).
    pub residual_in_fp32: bool,
}

impl Default for Mamba2Config {
    /// Mamba 2 1.3B defaults (state-spaces/mamba2-1.3b).
    fn default() -> Self {
        Self {
            d_model: 2048,
            n_layer: 48,
            d_state: 128,
            d_conv: 4,
            d_inner: 4096,   // 2048 * 2
            expand: 2,
            headdim: 64,
            nheads: 64,      // 4096 / 64
            ngroups: 1,
            vocab_size: 50288, // 50277 padded to multiple of 16
            rms_norm_eps: 1e-5,
            tie_embeddings: true,
            residual_in_fp32: true,
        }
    }
}

impl Mamba2Config {
    /// Mamba 2 2.7B configuration (state-spaces/mamba2-2.7b).
    pub fn mamba2_2_7b() -> Self {
        Self {
            d_model: 2560,
            n_layer: 64,
            d_state: 128,
            d_conv: 4,
            d_inner: 5120,   // 2560 * 2
            expand: 2,
            headdim: 64,
            nheads: 80,      // 5120 / 64
            ngroups: 1,
            vocab_size: 50288,
            rms_norm_eps: 1e-5,
            tie_embeddings: true,
            residual_in_fp32: true,
        }
    }

    /// Parse configuration from config.json, inferring SSM hyperparameters
    /// from weight tensor shapes when not explicitly provided.
    pub fn from_json(path: &std::path::Path) -> Result<Self> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| Error::internal(format!("failed to read config: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Error::internal(format!("failed to parse config: {}", e)))?;

        let mut c = Self::default();
        if let Some(v) = json.get("d_model").and_then(|v| v.as_u64()) { c.d_model = v as usize; }
        if let Some(v) = json.get("n_layer").and_then(|v| v.as_u64()) { c.n_layer = v as usize; }
        if let Some(v) = json.get("vocab_size").and_then(|v| v.as_u64()) {
            // Pad to multiple of 16 (pad_vocab_size_multiple)
            let pad = json.get("pad_vocab_size_multiple").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
            let raw = v as usize;
            c.vocab_size = ((raw + pad - 1) / pad) * pad;
        }
        if let Some(v) = json.get("rms_norm").and_then(|v| v.as_bool()) {
            if !v { tracing::warn!("Mamba 2 config has rms_norm=false, using LayerNorm not supported"); }
        }
        if let Some(v) = json.get("residual_in_fp32").and_then(|v| v.as_bool()) { c.residual_in_fp32 = v; }
        if let Some(v) = json.get("tie_embeddings").and_then(|v| v.as_bool()) { c.tie_embeddings = v; }

        // SSM config may be nested under ssm_cfg
        let ssm = json.get("ssm_cfg").unwrap_or(&json);
        if let Some(v) = ssm.get("d_state").and_then(|v| v.as_u64()) { c.d_state = v as usize; }
        if let Some(v) = ssm.get("d_conv").and_then(|v| v.as_u64()) { c.d_conv = v as usize; }
        if let Some(v) = ssm.get("expand").and_then(|v| v.as_u64()) { c.expand = v as usize; }
        if let Some(v) = ssm.get("headdim").and_then(|v| v.as_u64()) { c.headdim = v as usize; }
        if let Some(v) = ssm.get("ngroups").and_then(|v| v.as_u64()) { c.ngroups = v as usize; }

        // Derive d_inner and nheads
        c.d_inner = c.d_model * c.expand;
        c.nheads = c.d_inner / c.headdim;

        Ok(c)
    }

    /// Infer hyperparameters from actual weight tensor shapes.
    /// Call after loading the model to validate/override config.json values.
    #[cfg(feature = "metal")]
    pub fn infer_from_weights(&mut self, model: &Model) {
        // Use layer 0 weight shapes as reference
        if let Some(in_proj) = model.get_weight("backbone.layers.0.mixer.in_proj.weight") {
            let in_proj_dim = in_proj.shape().dim(0).unwrap_or(0);
            // in_proj_dim = 2*d_inner + 2*ngroups*d_state + nheads
            // We know d_inner from d_model * expand
            let remainder = in_proj_dim.saturating_sub(2 * self.d_inner);
            // remainder = 2*ngroups*d_state + nheads
            // nheads = d_inner / headdim
            let nheads = self.d_inner / self.headdim;
            let bc_dim = remainder.saturating_sub(nheads);
            // bc_dim = 2 * ngroups * d_state
            if bc_dim > 0 && self.ngroups > 0 {
                self.d_state = bc_dim / (2 * self.ngroups);
            }
            self.nheads = nheads;
            tracing::info!(
                d_model = self.d_model,
                d_inner = self.d_inner,
                nheads = self.nheads,
                d_state = self.d_state,
                ngroups = self.ngroups,
                in_proj_dim = in_proj_dim,
                "Mamba 2 architecture inferred from weights"
            );
        }

        // Infer d_conv from conv1d weight shape [conv_dim, 1, d_conv]
        if let Some(conv_w) = model.get_weight("backbone.layers.0.mixer.conv1d.weight") {
            if let Some(d_conv) = conv_w.shape().dim(2) {
                self.d_conv = d_conv;
            }
        }
    }

    /// Dimension of the conv1d input: d_inner + 2 * ngroups * d_state.
    /// This covers x, B, and C channels that go through causal conv.
    pub fn conv_dim(&self) -> usize {
        self.d_inner + 2 * self.ngroups * self.d_state
    }
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// Mamba 2 inference pipeline — pure SSM, no attention.
///
/// All N layers are identical SSD (Structured State Space Duality) blocks.
/// O(1) memory per token during autoregressive generation (no KV cache).
/// State per layer: SSM hidden [nheads, headdim, d_state] + conv buffer [conv_dim, d_conv].
#[cfg(feature = "metal")]
pub struct Mamba2Pipeline {
    model: Arc<parking_lot::RwLock<Model>>,
    compute: Arc<MetalCompute>,
    config: Mamba2Config,
    kernels: Mamba2Kernels,
    /// Per-layer recurrent state.
    states: Vec<Mamba2State>,
}

/// Per-layer Mamba 2 recurrent state.
#[cfg(feature = "metal")]
struct Mamba2State {
    /// SSM hidden state per head: [nheads, headdim, d_state]
    /// Each head maintains independent state of size [headdim, d_state].
    ssm_state: Vec<f32>,
    /// Causal conv1d sliding window: [conv_dim, d_conv]
    conv_state: Vec<f32>,
}

/// Compiled Metal kernels for Mamba 2.
#[cfg(feature = "metal")]
#[allow(dead_code)]
struct Mamba2Kernels {
    matmul: Arc<ComputePipeline>,
    matmul_q4k: Arc<ComputePipeline>,
    matmul_q8_0: Arc<ComputePipeline>,
    matvec: Arc<ComputePipeline>,
    rms_norm: Arc<ComputePipeline>,
    silu: Arc<ComputePipeline>,
    embedding: Arc<ComputePipeline>,
    add: Arc<ComputePipeline>,
    mul: Arc<ComputePipeline>,
}

#[cfg(feature = "metal")]
impl Mamba2Pipeline {
    /// Create a new Mamba 2 pipeline.
    pub fn new(model: Arc<parking_lot::RwLock<Model>>, mut config: Mamba2Config, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        // Infer architecture hyperparams from actual weight shapes
        config.infer_from_weights(&model);

        let kernels = Mamba2Kernels {
            matmul: compute.compile_pipeline("matmul", sources::MATMUL, "matmul_tiled_f16")?,
            matmul_q4k: compute.compile_pipeline("matmul_q4k", sources::MATMUL_Q4K, "matmul_q4k_f16")?,
            matmul_q8_0: compute.compile_pipeline("matmul_q8_0", sources::MATMUL_Q8_0, "matmul_q8_0_f16")?,
            matvec: compute.compile_pipeline("matvec", sources::MATVEC, "matvec_f16")?,
            rms_norm: compute.compile_pipeline("rms_norm", sources::RMS_NORM, "rms_norm_f16")?,
            silu: compute.compile_pipeline("silu", sources::SILU, "silu_f16")?,
            embedding: compute.compile_pipeline("embedding", sources::EMBEDDING, "embedding_lookup_f16")?,
            add: compute.compile_pipeline("add", sources::ELEMENTWISE, "add_f16")?,
            mul: compute.compile_pipeline("mul", sources::ELEMENTWISE, "mul_f16")?,
        };

        // Allocate per-layer state on CPU (selective scan runs on CPU)
        let nheads = config.nheads;
        let headdim = config.headdim;
        let d_state = config.d_state;
        let conv_dim = config.conv_dim();
        let d_conv = config.d_conv;

        let mut states = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            states.push(Mamba2State {
                ssm_state: vec![0.0f32; nheads * headdim * d_state],
                conv_state: vec![0.0f32; conv_dim * d_conv],
            });
        }

        tracing::info!(
            d_model = config.d_model,
            n_layer = config.n_layer,
            d_inner = config.d_inner,
            nheads = config.nheads,
            headdim = config.headdim,
            d_state = config.d_state,
            ngroups = config.ngroups,
            vocab_size = config.vocab_size,
            "Mamba 2 pipeline created"
        );

        Ok(Self { model, compute, config, kernels, states })
    }

    /// Reset all recurrent states (call between sequences).
    pub fn reset_states(&mut self) {
        let nheads = self.config.nheads;
        let headdim = self.config.headdim;
        let d_state = self.config.d_state;
        let conv_dim = self.config.conv_dim();
        let d_conv = self.config.d_conv;
        for state in &mut self.states {
            state.ssm_state.fill(0.0);
            state.ssm_state.resize(nheads * headdim * d_state, 0.0);
            state.conv_state.fill(0.0);
            state.conv_state.resize(conv_dim * d_conv, 0.0);
        }
    }

    /// Generate text autoregressively from a prompt token sequence.
    ///
    /// Returns generated token IDs (excluding the prompt).
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        self.reset_states();

        // Prefill: process all prompt tokens
        let mut last_logits = None;
        for &token_id in prompt_ids {
            last_logits = Some(self.forward_token(token_id)?);
        }

        let mut logits = match last_logits {
            Some(l) => l,
            None => return Err(Error::internal("empty prompt")),
        };

        let mut generated = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let next_token = sample_token(&logits, self.config.vocab_size, temperature);
            if let Some(eos) = eos_token_id {
                if next_token == eos {
                    break;
                }
            }
            generated.push(next_token);
            logits = self.forward_token(next_token)?;
        }

        Ok(generated)
    }

    /// Forward pass for a single token through all layers.
    /// Returns logits [vocab_size] as f32 on CPU.
    fn forward_token(&mut self, token_id: u32) -> Result<Vec<f32>> {
        let d_model = self.config.d_model;
        let n_layer = self.config.n_layer;
        let rms_eps = self.config.rms_norm_eps;
        let tie_embeddings = self.config.tie_embeddings;

        // 1. Embedding lookup → [d_model] as f32
        let mut hidden = self.embedding_lookup(token_id)?;

        // 2. Forward through all layers
        for layer_idx in 0..n_layer {
            hidden = self.forward_layer(layer_idx, &hidden)?;
        }

        // 3. Final RMS norm
        let norm_w = self.load_weight_f32("backbone.norm_f.weight", d_model)?;
        rms_norm_inplace(&mut hidden, &norm_w, rms_eps);

        // 4. LM head projection → [vocab_size]
        let logits = if tie_embeddings {
            self.lm_head_tied(&hidden)?
        } else {
            self.lm_head_separate(&hidden)?
        };

        Ok(logits)
    }

    /// Forward one layer: norm → SSD mixer → residual.
    fn forward_layer(&mut self, layer_idx: usize, input: &[f32]) -> Result<Vec<f32>> {
        let config = &self.config;
        let d_model = config.d_model;

        // Pre-mixer RMS norm
        let norm_w = self.load_weight_f32(
            &format!("backbone.layers.{}.norm.weight", layer_idx),
            d_model,
        )?;
        let mut normed = input.to_vec();
        rms_norm_inplace(&mut normed, &norm_w, config.rms_norm_eps);

        // SSD mixer
        let mixer_out = self.forward_ssd_mixer(layer_idx, &normed)?;

        // Residual connection
        let mut output = input.to_vec();
        for i in 0..d_model {
            output[i] += mixer_out[i];
        }

        Ok(output)
    }

    /// Mamba 2 SSD mixer for a single token.
    ///
    /// in_proj → split(z, x, B, C, dt) → conv1d(x,B,C) → SiLU(x) →
    /// multi-head selective scan → group_norm → gated output → out_proj
    fn forward_ssd_mixer(&mut self, layer_idx: usize, input: &[f32]) -> Result<Vec<f32>> {
        let config = &self.config;
        let d_model = config.d_model;
        let d_inner = config.d_inner;
        let nheads = config.nheads;
        let headdim = config.headdim;
        let d_state = config.d_state;
        let ngroups = config.ngroups;
        let d_conv = config.d_conv;
        let conv_dim = config.conv_dim(); // d_inner + 2*ngroups*d_state

        let prefix = format!("backbone.layers.{}.mixer", layer_idx);

        // ── in_proj: [d_model] → [in_proj_dim] ──────────────────────────────
        // in_proj_dim = d_inner + d_inner + ngroups*d_state + ngroups*d_state + nheads
        //             = 2*d_inner + 2*ngroups*d_state + nheads
        let in_proj = self.matmul_cpu(
            input,
            &format!("{}.in_proj.weight", prefix),
            d_model,
        )?;

        // Split in_proj output into components
        let mut offset = 0;
        let z = &in_proj[offset..offset + d_inner];
        offset += d_inner;
        let x_raw = &in_proj[offset..offset + d_inner];
        offset += d_inner;
        let b_raw = &in_proj[offset..offset + ngroups * d_state];
        offset += ngroups * d_state;
        let c_raw = &in_proj[offset..offset + ngroups * d_state];
        offset += ngroups * d_state;
        let dt_raw = &in_proj[offset..offset + nheads];

        // ── Causal conv1d on [x, B, C] combined ─────────────────────────────
        // Assemble conv input: [x (d_inner), B (ngroups*d_state), C (ngroups*d_state)]
        let mut conv_input = Vec::with_capacity(conv_dim);
        conv_input.extend_from_slice(x_raw);
        conv_input.extend_from_slice(b_raw);
        conv_input.extend_from_slice(c_raw);

        let conv_output = self.causal_conv1d(
            layer_idx,
            &conv_input,
            &format!("{}.conv1d.weight", prefix),
            &format!("{}.conv1d.bias", prefix),
            conv_dim,
            d_conv,
        )?;

        // Split conv output back into x, B, C
        let x_conv = &conv_output[0..d_inner];
        let b_conv = &conv_output[d_inner..d_inner + ngroups * d_state];
        let c_conv = &conv_output[d_inner + ngroups * d_state..conv_dim];

        // SiLU activation on x
        let mut x = vec![0.0f32; d_inner];
        for i in 0..d_inner {
            x[i] = silu(x_conv[i]);
        }

        // ── Load A_log, dt_bias, D ──────────────────────────────────────────
        let a_log = self.load_weight_f32(&format!("{}.A_log", prefix), nheads)?;
        let dt_bias = self.load_weight_f32(&format!("{}.dt_bias", prefix), nheads)?;
        let d_param = self.load_weight_f32_any_dtype(&format!("{}.D", prefix), nheads)?;

        // ── Multi-head SSD selective scan ────────────────────────────────────
        // dt = softplus(dt_raw + dt_bias)
        let mut dt = vec![0.0f32; nheads];
        for h in 0..nheads {
            dt[h] = softplus(dt_raw[h] + dt_bias[h]);
        }

        // A = -exp(A_log)
        let mut a = vec![0.0f32; nheads];
        for h in 0..nheads {
            a[h] = -a_log[h].exp();
        }

        // Selective scan per head
        let state = &mut self.states[layer_idx];
        let mut y = vec![0.0f32; d_inner];

        // B and C are shared across heads within a group.
        // nheads_per_group = nheads / ngroups
        let nheads_per_group = nheads / ngroups;

        for h in 0..nheads {
            let group = h / nheads_per_group;
            let a_bar = (dt[h] * a[h]).exp(); // discretized transition
            let dt_h = dt[h];

            for d in 0..headdim {
                let x_val = x[h * headdim + d];

                let mut y_val = 0.0f32;
                for s in 0..d_state {
                    let b_val = b_conv[group * d_state + s];
                    let c_val = c_conv[group * d_state + s];

                    // SSM recurrence: h_new = A_bar * h_old + dt * B * x
                    let state_idx = (h * headdim + d) * d_state + s;
                    state.ssm_state[state_idx] =
                        a_bar * state.ssm_state[state_idx] + dt_h * b_val * x_val;

                    // Readout: y += C * h
                    y_val += c_val * state.ssm_state[state_idx];
                }

                // Skip connection: y += D * x
                y[h * headdim + d] = y_val + d_param[h] * x_val;
            }
        }

        // ── Group norm on y ─────────────────────────────────────────────────
        let norm_w = self.load_weight_f32(&format!("{}.norm.weight", prefix), d_inner)?;
        // RMS norm per group (ngroups groups of d_inner/ngroups elements)
        let group_size = d_inner / ngroups;
        for g in 0..ngroups {
            let start = g * group_size;
            let end = start + group_size;
            let slice = &mut y[start..end];
            // Compute RMS
            let rms = (slice.iter().map(|v| v * v).sum::<f32>() / group_size as f32 + self.config.rms_norm_eps).sqrt();
            let inv_rms = 1.0 / rms;
            for i in 0..group_size {
                slice[i] = slice[i] * inv_rms * norm_w[start + i];
            }
        }

        // ── Output gate: y * silu(z) ────────────────────────────────────────
        let mut gated = vec![0.0f32; d_inner];
        for i in 0..d_inner {
            gated[i] = y[i] * silu(z[i]);
        }

        // ── out_proj: [d_inner] → [d_model] ────────────────────────────────
        // out_proj weight shape: [d_model, d_inner] — output = gated @ out_proj^T
        let output = self.matmul_cpu_transposed(
            &gated,
            &format!("{}.out_proj.weight", prefix),
            d_inner,
            d_model,
        )?;

        Ok(output)
    }

    // ── Causal Conv1d ────────────────────────────────────────────────────────

    /// Depthwise causal conv1d with sliding window state.
    /// Weight shape: [conv_dim, 1, d_conv] (depthwise per channel).
    fn causal_conv1d(
        &mut self,
        layer_idx: usize,
        input: &[f32],
        weight_key: &str,
        bias_key: &str,
        conv_dim: usize,
        d_conv: usize,
    ) -> Result<Vec<f32>> {
        // Load weights
        let w_f32 = self.load_weight_f32(weight_key, conv_dim * d_conv)?;
        let bias_f32 = match self.load_weight_f32(bias_key, conv_dim) {
            Ok(b) => b,
            Err(_) => vec![0.0f32; conv_dim],
        };

        let state = &mut self.states[layer_idx];

        // Shift conv buffer left and insert new input
        for d in 0..conv_dim {
            for k in 0..d_conv - 1 {
                state.conv_state[d * d_conv + k] = state.conv_state[d * d_conv + k + 1];
            }
            state.conv_state[d * d_conv + d_conv - 1] = input[d];
        }

        // Depthwise convolution: dot product per channel
        let mut output = vec![0.0f32; conv_dim];
        for d in 0..conv_dim {
            let mut val = bias_f32[d];
            for k in 0..d_conv {
                val += state.conv_state[d * d_conv + k] * w_f32[d * d_conv + k];
            }
            output[d] = val;
        }

        Ok(output)
    }

    // ── Embedding ────────────────────────────────────────────────────────────

    /// Look up a single token embedding on GPU, return as f32 on CPU.
    fn embedding_lookup(&self, token_id: u32) -> Result<Vec<f32>> {
        let embed_w = self.model.read().get_weight("backbone.embedding.weight")
            .ok_or_else(|| Error::internal("backbone.embedding.weight not found"))?;
        let d_model = self.config.d_model;

        // Read the single row from the embedding table
        // embed_w shape: [vocab_size, d_model], dtype F16
        // Row offset: token_id * d_model * 2 bytes
        let all_f32 = embed_w.to_f32_vec()?;
        let start = token_id as usize * d_model;
        if start + d_model > all_f32.len() {
            return Err(Error::internal(format!(
                "token_id {} out of range (embedding has {} rows)",
                token_id,
                all_f32.len() / d_model
            )));
        }
        Ok(all_f32[start..start + d_model].to_vec())
    }

    // ── LM Head ──────────────────────────────────────────────────────────────

    /// Tied LM head: reuse embedding weight for final projection.
    fn lm_head_tied(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let embed_w = self.model.read().get_weight("backbone.embedding.weight")
            .ok_or_else(|| Error::internal("backbone.embedding.weight not found"))?;
        let all_f32 = embed_w.to_f32_vec()?;
        let d_model = self.config.d_model;
        let vocab_size = self.config.vocab_size;

        // hidden [d_model] @ embed_w^T [d_model, vocab_size] → [vocab_size]
        let mut logits = vec![0.0f32; vocab_size];
        for v in 0..vocab_size {
            let mut sum = 0.0f32;
            let row_start = v * d_model;
            for d in 0..d_model {
                sum += hidden[d] * all_f32[row_start + d];
            }
            logits[v] = sum;
        }
        Ok(logits)
    }

    /// Separate LM head weight.
    fn lm_head_separate(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let lm_head_w = self.model.read().get_weight("lm_head.weight")
            .ok_or_else(|| Error::internal("lm_head.weight not found"))?;
        let all_f32 = lm_head_w.to_f32_vec()?;
        let d_model = self.config.d_model;
        let vocab_size = self.config.vocab_size;

        let mut logits = vec![0.0f32; vocab_size];
        for v in 0..vocab_size {
            let mut sum = 0.0f32;
            let row_start = v * d_model;
            for d in 0..d_model {
                sum += hidden[d] * all_f32[row_start + d];
            }
            logits[v] = sum;
        }
        Ok(logits)
    }

    // ── Weight loading helpers ───────────────────────────────────────────────

    /// Load a weight tensor as f32 vector.
    fn load_weight_f32(&self, name: &str, expected_len: usize) -> Result<Vec<f32>> {
        let lt = self.model.read().get_weight(name)
            .ok_or_else(|| Error::internal(format!("weight not found: {}", name)))?;
        let data = lt.to_f32_vec()?;
        if data.len() < expected_len {
            return Err(Error::internal(format!(
                "weight {} has {} elements, expected at least {}",
                name, data.len(), expected_len
            )));
        }
        Ok(data)
    }

    /// Load a weight tensor as f32, handling both F16 and F32 source dtypes.
    /// Some weights (e.g., D in 2.7B) are stored as F32.
    fn load_weight_f32_any_dtype(&self, name: &str, expected_len: usize) -> Result<Vec<f32>> {
        self.load_weight_f32(name, expected_len)
    }

    /// Matrix-vector multiply on CPU: input[in_dim] @ weight^T → output[out_dim].
    /// Weight shape: [out_dim, in_dim] (row-major, transposed multiply).
    fn matmul_cpu(
        &self,
        input: &[f32],
        weight_name: &str,
        in_dim: usize,
    ) -> Result<Vec<f32>> {
        let lt = self.model.read().get_weight(weight_name)
            .ok_or_else(|| Error::internal(format!("weight not found: {}", weight_name)))?;
        let out_dim = lt.shape().dim(0).unwrap_or(0);
        let w_f32 = lt.to_f32_vec()?;

        let mut output = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            let mut sum = 0.0f32;
            let row_start = o * in_dim;
            for i in 0..in_dim {
                sum += input[i] * w_f32[row_start + i];
            }
            output[o] = sum;
        }
        Ok(output)
    }

    /// Matrix-vector multiply for out_proj which has shape [d_model, d_inner].
    /// input[d_inner] @ weight^T → output[d_model].
    fn matmul_cpu_transposed(
        &self,
        input: &[f32],
        weight_name: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        let lt = self.model.read().get_weight(weight_name)
            .ok_or_else(|| Error::internal(format!("weight not found: {}", weight_name)))?;
        let w_f32 = lt.to_f32_vec()?;

        let mut output = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            let mut sum = 0.0f32;
            let row_start = o * in_dim;
            for i in 0..in_dim {
                sum += input[i] * w_f32[row_start + i];
            }
            output[o] = sum;
        }
        Ok(output)
    }

    /// GPU-accelerated matmul for prefill (batched tokens).
    /// input: [seq_len, in_features], weight: LazyTensor [out_features, in_features]
    /// Returns: [seq_len, out_features]
    #[allow(dead_code)]
    fn matmul_gpu(&self, input: &Tensor, weight: &LazyTensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let weight_shape = weight.shape();
        let m = input_shape.dim(0).unwrap_or(1);
        let k = input_shape.dim(1).unwrap_or(1);
        let n = weight_shape.dim(0).unwrap_or(1);

        let qt = weight.quant_type();
        if qt != QuantType::None && m == 1 {
            return self.matmul_quantized_gpu(input, weight, k, n, qt);
        }

        if qt == QuantType::None {
            return self.matmul_f16_gpu(input, weight.buffer(), m, k, n);
        }

        // Quantized prefill: dequantize on CPU, then tiled matmul
        let numel = n * k;
        let device = self.compute.device().raw();
        let f16_buf = device.new_buffer(
            (numel * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            let src = std::slice::from_raw_parts(
                weight.buffer().contents() as *const u8,
                weight.buffer().length() as usize,
            );
            let dst = std::slice::from_raw_parts_mut(
                f16_buf.contents() as *mut u16,
                numel,
            );
            match qt {
                QuantType::Q4K => crate::hal::metal::lazy_load::dequantize_q4k_public(src, dst, numel)?,
                QuantType::Q8_0 => crate::hal::metal::lazy_load::dequantize_q8_0_public(src, dst, numel)?,
                QuantType::None => unreachable!(),
            }
        }
        self.matmul_f16_gpu(input, &f16_buf, m, k, n)
    }

    /// F16 tiled matmul on GPU.
    #[allow(dead_code)]
    fn matmul_f16_gpu(&self, input: &Tensor, weight_buffer: &metal::Buffer, m: usize, k: usize, n: usize) -> Result<Tensor> {
        let device = self.compute.device().raw();
        let device_id = self.compute.device().info().id;
        let output_buffer = device.new_buffer((m * n * 2) as u64, metal::MTLResourceOptions::StorageModeShared);

        let cb = self.compute.new_command_buffer();
        if m == 1 {
            self.compute.dispatch(&cb, &self.kernels.matvec, (n, 1, 1), (32, 1, 1), |encoder| {
                if let Some(ptr) = input.device_ptr() {
                    let buf = unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) };
                    encoder.set_buffer(0, Some(buf.as_ref()), 0);
                }
                encoder.set_buffer(1, Some(weight_buffer), 0);
                encoder.set_buffer(2, Some(&output_buffer), 0);
                encoder.set_bytes(3, 4, &(k as u32) as *const u32 as *const _);
                encoder.set_bytes(4, 4, &(n as u32) as *const u32 as *const _);
            });
        } else {
            let tile_size = 32;
            let tg_size = (16, 16, 1);
            let grid_size = ((n + tile_size - 1) / tile_size, (m + tile_size - 1) / tile_size, 1);
            self.compute.dispatch(&cb, &self.kernels.matmul, grid_size, tg_size, |encoder| {
                if let Some(ptr) = input.device_ptr() {
                    let buf = unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) };
                    encoder.set_buffer(0, Some(buf.as_ref()), 0);
                }
                encoder.set_buffer(1, Some(weight_buffer), 0);
                encoder.set_buffer(2, Some(&output_buffer), 0);
                let m_u32 = m as u32;
                let n_u32 = n as u32;
                let k_u32 = k as u32;
                encoder.set_bytes(3, 4, &m_u32 as *const u32 as *const _);
                encoder.set_bytes(4, 4, &n_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &k_u32 as *const u32 as *const _);
                encoder.set_threadgroup_memory_length(0, 4096);
            });
        }
        cb.commit();
        cb.wait_until_completed();

        Ok(Tensor::from_metal_buffer(output_buffer, Shape::from([m, n]), DType::F16, device_id))
    }

    /// Quantized matvec on GPU (m=1).
    #[allow(dead_code)]
    fn matmul_quantized_gpu(&self, input: &Tensor, weight: &LazyTensor, k: usize, n: usize, qt: QuantType) -> Result<Tensor> {
        let device = self.compute.device().raw();
        let device_id = self.compute.device().info().id;
        let output_buffer = device.new_buffer((n * 2) as u64, metal::MTLResourceOptions::StorageModeShared);

        let kernel = match qt {
            QuantType::Q4K => &self.kernels.matmul_q4k,
            QuantType::Q8_0 => &self.kernels.matmul_q8_0,
            QuantType::None => unreachable!(),
        };

        let cb = self.compute.new_command_buffer();
        self.compute.dispatch(&cb, kernel, (n, 1, 1), (32, 1, 1), |encoder| {
            if let Some(ptr) = input.device_ptr() {
                let buf = unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) };
                encoder.set_buffer(0, Some(buf.as_ref()), 0);
            }
            encoder.set_buffer(1, Some(weight.buffer()), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_bytes(3, 4, &(k as u32) as *const u32 as *const _);
            encoder.set_bytes(4, 4, &(n as u32) as *const u32 as *const _);
        });
        cb.commit();
        cb.wait_until_completed();

        Ok(Tensor::from_metal_buffer(output_buffer, Shape::from([1, n]), DType::F16, device_id))
    }
}

// ── CPU helper functions ─────────────────────────────────────────────────────

/// SiLU (Swish) activation: x * sigmoid(x).
#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softplus: log(1 + exp(x)), with numerical stability.
#[inline(always)]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// In-place RMS normalization: x = (x / rms) * weight.
fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..n {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

/// Sample a token from logits using temperature-scaled sampling.
fn sample_token(logits: &[f32], vocab_size: usize, temperature: f32) -> u32 {
    let logits = &logits[..vocab_size];
    if temperature <= 0.0 || temperature < 1e-6 {
        // Greedy: argmax
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0u32;
        for (i, &v) in logits.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i as u32;
            }
        }
        return max_idx;
    }

    // Temperature-scaled softmax + sampling
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&v| ((v - max_logit) / temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let inv_sum = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }

    // Random sampling (simple LCG for determinism in tests; production should use thread_rng)
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let r = (seed as f32) / (u32::MAX as f32);

    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i as u32;
        }
    }
    (vocab_size - 1) as u32
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((silu(1.0) - 0.7310586).abs() < 1e-4);
        assert!((silu(-1.0) - (-0.2689414)).abs() < 1e-4);
    }

    #[test]
    fn test_softplus() {
        assert!((softplus(0.0) - 0.6931472).abs() < 1e-4);
        assert!((softplus(1.0) - 1.3132616).abs() < 1e-4);
        assert!((softplus(25.0) - 25.0).abs() < 1e-4); // saturated
        assert!((softplus(-25.0) - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        rms_norm_inplace(&mut x, &w, 1e-5);
        // rms = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (7.5f32 + 1e-5).sqrt();
        assert!((x[0] - 1.0 / rms).abs() < 1e-4);
        assert!((x[2] - 3.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_config_default() {
        let c = Mamba2Config::default();
        assert_eq!(c.d_model, 2048);
        assert_eq!(c.n_layer, 48);
        assert_eq!(c.d_inner, 4096);
        assert_eq!(c.nheads, 64);
        assert_eq!(c.headdim, 64);
        assert_eq!(c.d_state, 128);
        assert_eq!(c.conv_dim(), 4096 + 2 * 1 * 128); // 4352
    }

    #[test]
    fn test_config_2_7b() {
        let c = Mamba2Config::mamba2_2_7b();
        assert_eq!(c.d_model, 2560);
        assert_eq!(c.n_layer, 64);
        assert_eq!(c.d_inner, 5120);
        assert_eq!(c.nheads, 80);
        assert_eq!(c.conv_dim(), 5120 + 256); // 5376
    }

    #[test]
    fn test_sample_greedy() {
        let logits = vec![0.1, 0.5, 0.2, 0.9, 0.3];
        let token = sample_token(&logits, 5, 0.0);
        assert_eq!(token, 3); // argmax
    }
}
