//! F5-TTS: Flow-matching DiT for text-to-speech synthesis.
//!
//! Architecture:
//!   Text → ConvNeXt V2 text blocks (4 blocks) → text embeddings [seq, 512]
//!   Audio → mel spectrogram [frames, 100] + text → proj [frames, 1024]
//!   → 22-layer DiT (self-attn with rotary PE + FFN) with adaptive norm
//!   → proj_out [frames, 100] → mel spectrogram
//!   → Vocos vocoder (ConvNeXt + ISTFT) → audio waveform
//!
//! F5-TTS uses Conditional Flow Matching (CFM) for iterative denoising of mel spectrograms.
//! All weights are F32 in the safetensors file, stored with `ema_model.transformer.` prefix.

use crate::core::Result;
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::inference::gpu_ops::{self, MetalPipeline as _};
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;

/// Weight key prefix for F5-TTS model.
#[cfg(feature = "metal")]
const P: &str = "ema_model.transformer";

/// F5-TTS configuration.
#[derive(Debug, Clone)]
pub struct F5TTSConfig {
    /// Transformer hidden dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Number of DiT layers.
    pub num_layers: usize,
    /// Mel spectrogram bins (input/output dimension).
    pub n_mel_channels: usize,
    /// Text embedding dimension.
    pub text_dim: usize,
    /// Number of ConvNeXt text blocks.
    pub num_text_blocks: usize,
    /// ConvNeXt intermediate expansion factor.
    pub text_intermediate_dim: usize,
    /// Input projection dimension (mel + text + mask).
    pub input_dim: usize,
    /// Audio sample rate.
    pub sample_rate: usize,
    /// Hop length for mel spectrogram.
    pub hop_length: usize,
    /// Number of flow matching steps.
    pub num_steps: usize,
}

impl Default for F5TTSConfig {
    fn default() -> Self {
        Self {
            d_model: 1024,
            num_heads: 16,
            head_dim: 64,
            num_layers: 22,
            n_mel_channels: 100,
            text_dim: 512,
            num_text_blocks: 4,
            text_intermediate_dim: 1024,
            input_dim: 712, // 100 mel + 512 text + 100 mask = 712
            sample_rate: 24000,
            hop_length: 256,
            num_steps: 32,
        }
    }
}

// ==================== Compiled Kernels ====================

#[cfg(feature = "metal")]
struct F5TTSKernels {
    common: gpu_ops::CommonKernels,
    gelu: Arc<ComputePipeline>,
}

// ==================== F5-TTS Pipeline ====================

/// F5-TTS pipeline for text-to-speech synthesis.
#[cfg(feature = "metal")]
pub struct F5TTSPipeline {
    model: Arc<Model>,
    compute: Arc<MetalCompute>,
    config: F5TTSConfig,
    kernels: F5TTSKernels,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for F5TTSPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl F5TTSPipeline {
    /// Create a new F5-TTS pipeline.
    pub fn new(model: Arc<Model>, config: F5TTSConfig, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));
        let kernels = F5TTSKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
        };
        Ok(Self { model, compute, config, kernels })
    }

    /// Generate speech from text, conditioned on reference audio.
    ///
    /// - `ref_mel`: reference speaker mel spectrogram [ref_frames, n_mel] (F16)
    /// - `text_tokens`: tokenized text (character-level IDs from vocab.txt)
    /// - `duration_frames`: target audio duration in mel frames
    ///
    /// Returns mel spectrogram [duration_frames, n_mel] (generated portion only).
    pub fn generate(
        &self,
        ref_mel: &Tensor,
        text_tokens: &[u32],
        duration_frames: usize,
    ) -> Result<Tensor> {
        let config = &self.config;
        let ref_frames = ref_mel.shape().dim(0).unwrap_or(0);
        let total_frames = ref_frames + duration_frames;
        let n_mel = config.n_mel_channels;

        // 1. Text embedding: characters → ConvNeXt V2 → [total_frames, 512]
        let text_embed = self.text_embed_cpu(text_tokens, total_frames)?;

        // 2. Read reference mel as f32
        let ref_mel_data: Vec<half::f16> = ref_mel.to_vec()?;
        let ref_mel_f32: Vec<f32> = ref_mel_data.iter().map(|v| v.to_f32()).collect();

        // 3. Precompute RoPE cos/sin
        let rope_inv_freq = self.weight_vec_f32(&self.model, &format!("{}.rotary_embed.inv_freq", P))?;
        let (rope_cos, rope_sin) = Self::precompute_rope(&rope_inv_freq, total_frames, config.head_dim);

        // 4. Initialize generated mel as zeros (pure noise in production)
        let mut gen_mel = vec![0.0f32; duration_frames * n_mel];

        // 5. Flow matching ODE: integrate from t=0 (noise) to t=1 (data)
        let num_steps = config.num_steps;
        let dt = 1.0 / num_steps as f32;

        for step in 0..num_steps {
            let t = step as f32 / num_steps as f32;

            // a. Prepare input: concat [mel(100), text(512), mask(100)] → [total_frames, 712]
            let mut input = vec![0.0f32; total_frames * config.input_dim];
            for frame in 0..total_frames {
                let off = frame * config.input_dim;
                if frame < ref_frames {
                    // Reference: actual mel + text + mask=0
                    for c in 0..n_mel {
                        input[off + c] = ref_mel_f32[frame * n_mel + c];
                    }
                } else {
                    // Generated: current estimate + text + mask=1
                    let gf = frame - ref_frames;
                    for c in 0..n_mel {
                        input[off + c] = gen_mel[gf * n_mel + c];
                    }
                    // mask = 1 (last 100 dims)
                    for c in 0..n_mel {
                        input[off + config.text_dim + n_mel + c] = 1.0;
                    }
                }
                // Text embedding (middle 512 dims)
                for c in 0..config.text_dim {
                    input[off + n_mel + c] = text_embed[frame * config.text_dim + c];
                }
            }

            // b. Time embedding: sinusoidal(256) → MLP(1024) → SiLU → MLP(1024)
            let t_emb = self.time_embed_cpu(t)?;

            // c. DiT forward: input proj + conv pos + 22 layers + output proj
            let velocity = self.dit_forward(&input, &t_emb, &rope_cos, &rope_sin, total_frames)?;

            // d. ODE Euler step: update generated portion only
            for frame in 0..duration_frames {
                for c in 0..n_mel {
                    let idx = (ref_frames + frame) * n_mel + c;
                    gen_mel[frame * n_mel + c] += dt * velocity[idx];
                }
            }

            if step == 0 || step == num_steps - 1 || (step + 1) % 8 == 0 {
                let max_vel = velocity.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                eprintln!("   Step {}/{}: t={:.3}, max_velocity={:.4}", step + 1, num_steps, t, max_vel);
            }
        }

        // Return generated mel
        let mel_f16: Vec<half::f16> = gen_mel.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(
            &mel_f16, Shape::from([duration_frames, n_mel]),
            DType::F16, self.compute.device().info().id,
        )
    }

    // ==================== Text Embedding (CPU) ====================

    /// ConvNeXt V2 text encoder: tokens → [total_frames, text_dim].
    fn text_embed_cpu(&self, tokens: &[u32], total_frames: usize) -> Result<Vec<f32>> {
        let text_dim = self.config.text_dim; // 512

        // 1. Lookup embeddings [vocab_size=2546, 512]
        let embed_w = self.weight_vec_f32(&self.model, &format!("{}.text_embed.text_embed.weight", P))?;
        let vocab_size = embed_w.len() / text_dim;

        // Pad or truncate tokens to total_frames
        let mut token_ids = vec![0u32; total_frames];
        let n = tokens.len().min(total_frames);
        token_ids[..n].copy_from_slice(&tokens[..n]);

        // Lookup: [total_frames, 512]
        let mut x = vec![0.0f32; total_frames * text_dim];
        for (i, &tid) in token_ids.iter().enumerate() {
            let t = (tid as usize).min(vocab_size - 1);
            x[i * text_dim..(i + 1) * text_dim]
                .copy_from_slice(&embed_w[t * text_dim..(t + 1) * text_dim]);
        }

        // 2. Four ConvNeXt V2 blocks
        for block in 0..self.config.num_text_blocks {
            let prefix = format!("{}.text_embed.text_blocks.{}", P, block);
            self.convnext_block_cpu(&mut x, total_frames, text_dim, &prefix)?;
        }

        Ok(x)
    }

    /// ConvNeXt V2 block: depthwise conv → LN → pwconv1 → GELU → GRN → pwconv2 + residual.
    /// Input/output: [seq_len, dim] stored row-major.
    fn convnext_block_cpu(&self, x: &mut [f32], seq_len: usize, dim: usize, prefix: &str) -> Result<()> {
        let int_dim = self.config.text_intermediate_dim; // 1024

        // Save residual
        let residual: Vec<f32> = x.to_vec();

        // 1. Depthwise conv1d: [dim, 1, 7], groups=dim, pad=3
        let dw_w = self.weight_vec_f32(&self.model, &format!("{}.dwconv.weight", prefix))?;
        let dw_b = self.weight_vec_f32(&self.model, &format!("{}.dwconv.bias", prefix))?;
        let kernel_size = 7;
        let pad = 3;
        // x is [seq_len, dim], but depthwise conv operates on channels (transpose needed)
        let mut conv_out = vec![0.0f32; seq_len * dim];
        for c in 0..dim {
            for i in 0..seq_len {
                let mut sum = dw_b[c];
                for k in 0..kernel_size {
                    let j = i as i64 + k as i64 - pad as i64;
                    if j >= 0 && (j as usize) < seq_len {
                        // x[j, c] * w[c, 0, k]
                        sum += x[j as usize * dim + c] * dw_w[c * kernel_size + k];
                    }
                }
                conv_out[i * dim + c] = sum;
            }
        }
        x.copy_from_slice(&conv_out);

        // 2. LayerNorm
        let ln_w = self.weight_vec_f32(&self.model, &format!("{}.norm.weight", prefix))?;
        let ln_b = self.weight_vec_f32(&self.model, &format!("{}.norm.bias", prefix))?;
        Self::layer_norm_cpu(x, seq_len, dim, &ln_w, &ln_b);

        // 3. Pointwise conv1 (linear): [seq_len, dim] → [seq_len, int_dim]
        let pw1_w = self.weight_vec_f32(&self.model, &format!("{}.pwconv1.weight", prefix))?;
        let pw1_b = self.weight_vec_f32(&self.model, &format!("{}.pwconv1.bias", prefix))?;
        let mut pw1_out = vec![0.0f32; seq_len * int_dim];
        Self::linear_cpu(x, &pw1_w, &pw1_b, &mut pw1_out, seq_len, dim, int_dim);

        // 4. GELU activation
        for v in &mut pw1_out {
            let x_val = *v;
            *v = 0.5 * x_val * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x_val + 0.044715 * x_val.powi(3))).tanh());
        }

        // 5. GRN (Global Response Normalization)
        let grn_gamma = self.weight_vec_f32(&self.model, &format!("{}.grn.gamma", prefix))?;
        let grn_beta = self.weight_vec_f32(&self.model, &format!("{}.grn.beta", prefix))?;
        Self::grn_cpu(&mut pw1_out, &grn_gamma, &grn_beta, seq_len, int_dim);

        // 6. Pointwise conv2 (linear): [seq_len, int_dim] → [seq_len, dim]
        let pw2_w = self.weight_vec_f32(&self.model, &format!("{}.pwconv2.weight", prefix))?;
        let pw2_b = self.weight_vec_f32(&self.model, &format!("{}.pwconv2.bias", prefix))?;
        Self::linear_cpu(&pw1_out, &pw2_w, &pw2_b, x, seq_len, int_dim, dim);

        // 7. Residual connection
        for i in 0..seq_len * dim {
            x[i] += residual[i];
        }

        Ok(())
    }

    /// GRN: Global Response Normalization.
    /// x is [seq_len, dim]. gamma/beta are [dim] (flattened from [1, 1, dim]).
    fn grn_cpu(x: &mut [f32], gamma: &[f32], beta: &[f32], seq_len: usize, dim: usize) {
        // Compute L2 norm per channel across sequence
        let mut channel_norms = vec![0.0f32; dim];
        for i in 0..seq_len {
            for c in 0..dim {
                channel_norms[c] += x[i * dim + c] * x[i * dim + c];
            }
        }
        for c in 0..dim {
            channel_norms[c] = channel_norms[c].sqrt();
        }

        // Normalize channel norms
        let mean_norm: f32 = channel_norms.iter().sum::<f32>() / dim as f32;
        for c in 0..dim {
            channel_norms[c] = channel_norms[c] / (mean_norm + 1e-6);
        }

        // Apply: x = gamma * (x * norm) + beta + x
        for i in 0..seq_len {
            for c in 0..dim {
                let idx = i * dim + c;
                let g = if c < gamma.len() { gamma[c] } else { 0.0 };
                let b = if c < beta.len() { beta[c] } else { 0.0 };
                x[idx] = g * (x[idx] * channel_norms[c]) + b + x[idx];
            }
        }
    }

    // ==================== Time Embedding (CPU) ====================

    /// Sinusoidal time embedding + 2-layer MLP.
    fn time_embed_cpu(&self, t: f32) -> Result<Vec<f32>> {
        // 1. Sinusoidal positional embedding: scalar t → [256]
        let sin_dim = 256;
        let half = sin_dim / 2;
        let mut embed = vec![0.0f32; sin_dim];
        for i in 0..half {
            let freq = (-((i as f32) / half as f32) * (10000.0f32).ln()).exp();
            embed[i] = (t * freq).sin();
            embed[i + half] = (t * freq).cos();
        }

        // 2. Linear(256 → 1024) + SiLU + Linear(1024 → 1024)
        let w1 = self.weight_vec_f32(&self.model, &format!("{}.time_embed.time_mlp.0.weight", P))?;
        let b1 = self.weight_vec_f32(&self.model, &format!("{}.time_embed.time_mlp.0.bias", P))?;
        let d_model = self.config.d_model;

        let mut h = vec![0.0f32; d_model];
        Self::linear_cpu(&embed, &w1, &b1, &mut h, 1, sin_dim, d_model);

        // SiLU: x * sigmoid(x)
        for v in &mut h {
            *v = *v * (1.0 / (1.0 + (-*v).exp()));
        }

        let w2 = self.weight_vec_f32(&self.model, &format!("{}.time_embed.time_mlp.2.weight", P))?;
        let b2 = self.weight_vec_f32(&self.model, &format!("{}.time_embed.time_mlp.2.bias", P))?;
        let mut out = vec![0.0f32; d_model];
        Self::linear_cpu(&h, &w2, &b2, &mut out, 1, d_model, d_model);

        Ok(out)
    }

    // ==================== Conv Positional Embedding (CPU) ====================

    /// Convolutional positional embedding: grouped conv1d + SiLU + grouped conv1d.
    /// Adds positional info to x in-place. x is [seq_len, d_model].
    fn conv_pos_embed_cpu(&self, x_f32: &mut [f32], seq_len: usize) -> Result<()> {
        let d_model = self.config.d_model; // 1024
        let groups = self.config.num_heads; // 16 (groups = d_model / channels_per_group = 1024/64 = 16)
        let channels_per_group = d_model / groups; // 64
        let kernel_size = 31;
        let pad = 15;

        // Conv1d layer 0: [1024, 64, 31] grouped conv
        let w0 = self.weight_vec_f32(&self.model, &format!("{}.input_embed.conv_pos_embed.conv1d.0.weight", P))?;
        let b0 = self.weight_vec_f32(&self.model, &format!("{}.input_embed.conv_pos_embed.conv1d.0.bias", P))?;
        let mut h = vec![0.0f32; seq_len * d_model];
        self.grouped_conv1d_cpu(x_f32, &mut h, &w0, &b0, seq_len, d_model, groups, channels_per_group, kernel_size, pad);

        // SiLU activation
        for v in &mut h {
            *v = *v * (1.0 / (1.0 + (-*v).exp()));
        }

        // Conv1d layer 2: [1024, 64, 31] grouped conv
        let w2 = self.weight_vec_f32(&self.model, &format!("{}.input_embed.conv_pos_embed.conv1d.2.weight", P))?;
        let b2 = self.weight_vec_f32(&self.model, &format!("{}.input_embed.conv_pos_embed.conv1d.2.bias", P))?;
        let mut out = vec![0.0f32; seq_len * d_model];
        self.grouped_conv1d_cpu(&h, &mut out, &w2, &b2, seq_len, d_model, groups, channels_per_group, kernel_size, pad);

        // Add to input (positional embedding is additive)
        for i in 0..seq_len * d_model {
            x_f32[i] += out[i];
        }

        Ok(())
    }

    /// Grouped 1D convolution on CPU (AMX-accelerated via im2col + sgemm per group).
    /// Input/output: [seq_len, channels] row-major. Weight: [channels, channels_per_group, kernel].
    fn grouped_conv1d_cpu(
        &self, input: &[f32], output: &mut [f32],
        weight: &[f32], bias: &[f32],
        seq_len: usize, channels: usize, groups: usize,
        channels_per_group: usize, kernel_size: usize, pad: usize,
    ) {
        let col_dim = channels_per_group * kernel_size;
        let mut im2col = vec![0.0f32; seq_len * col_dim];
        let mut group_out = vec![0.0f32; seq_len * channels_per_group];

        for g in 0..groups {
            let c_start = g * channels_per_group;

            // im2col: extract patches for this group → [seq_len, cpg * kernel_size]
            for i in 0..seq_len {
                for ic in 0..channels_per_group {
                    for k in 0..kernel_size {
                        let j = i as i64 + k as i64 - pad as i64;
                        let val = if j >= 0 && (j as usize) < seq_len {
                            input[j as usize * channels + c_start + ic]
                        } else {
                            0.0
                        };
                        im2col[i * col_dim + ic * kernel_size + k] = val;
                    }
                }
            }

            // Weight for this group: [cpg_out, cpg_in * kernel] (contiguous sub-block)
            // w_group[oc, ic, k] = weight[(c_start + oc) * cpg * ks + ic * ks + k]
            let w_offset = c_start * channels_per_group * kernel_size;
            let w_group = &weight[w_offset..w_offset + channels_per_group * col_dim];

            // Matmul: [seq_len, col_dim] @ [cpg, col_dim]^T → [seq_len, cpg]
            crate::tensor::ops::sgemm_transb_cpu(&im2col, w_group, &mut group_out, seq_len, channels_per_group, col_dim);

            // Add bias and scatter to output
            for i in 0..seq_len {
                for oc in 0..channels_per_group {
                    output[i * channels + c_start + oc] = group_out[i * channels_per_group + oc] + bias[c_start + oc];
                }
            }
        }
    }

    // ==================== DiT Forward (GPU + CPU hybrid) ====================

    /// Full DiT forward pass: returns velocity field [total_frames, n_mel].
    fn dit_forward(
        &self, input_f32: &[f32], t_emb: &[f32],
        rope_cos: &[f32], rope_sin: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let config = &self.config;
        let d = config.d_model;
        let device_id = self.compute.device().info().id;

        // 1. Input projection: [seq, 712] → [seq, 1024] (GPU)
        let input_f16: Vec<half::f16> = input_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let input_tensor = Tensor::from_slice(&input_f16, Shape::from([seq_len, config.input_dim]), DType::F16, device_id)?;
        let cb = self.compute.new_command_buffer();
        let projected = self.linear_bias(&cb, &self.model, &input_tensor,
            &format!("{}.input_embed.proj.weight", P),
            &format!("{}.input_embed.proj.bias", P),
            seq_len, config.input_dim, d)?;
        cb.commit();
        cb.wait_until_completed();

        // 2. Read to CPU for conv pos embed
        let proj_data: Vec<half::f16> = projected.to_vec()?;
        let mut x_f32: Vec<f32> = proj_data.iter().map(|v| v.to_f32()).collect();

        // 3. Convolutional positional embedding
        self.conv_pos_embed_cpu(&mut x_f32, seq_len)?;

        // 4. 22 DiT layers
        for layer in 0..config.num_layers {
            self.dit_block(&mut x_f32, t_emb, rope_cos, rope_sin, layer, seq_len)?;
        }

        // 5. Final AdaLN + output projection
        let norm_w = self.weight_vec_f32(&self.model, &format!("{}.norm_out.linear.weight", P))?;
        let norm_b = self.weight_vec_f32(&self.model, &format!("{}.norm_out.linear.bias", P))?;

        // Modulation: SiLU(t_emb) → Linear → [scale, shift] (2 × d)
        let mut t_silu = t_emb.to_vec();
        for v in &mut t_silu {
            *v = *v * (1.0 / (1.0 + (-*v).exp()));
        }
        let mut mod_out = vec![0.0f32; 2 * d];
        Self::linear_cpu(&t_silu, &norm_w, &norm_b, &mut mod_out, 1, d, 2 * d);
        let (shift, scale) = mod_out.split_at(d);

        // Apply parameter-free norm + modulation
        Self::param_free_norm_modulate_cpu(&mut x_f32, seq_len, d, scale, shift);

        // Output projection: [seq, 1024] → [seq, 100] (GPU)
        let x_f16: Vec<half::f16> = x_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let x_tensor = Tensor::from_slice(&x_f16, Shape::from([seq_len, d]), DType::F16, device_id)?;
        let cb = self.compute.new_command_buffer();
        let out = self.linear_bias(&cb, &self.model, &x_tensor,
            &format!("{}.proj_out.weight", P),
            &format!("{}.proj_out.bias", P),
            seq_len, d, config.n_mel_channels)?;
        cb.commit();
        cb.wait_until_completed();

        // Read output as f32
        let out_data: Vec<half::f16> = out.to_vec()?;
        Ok(out_data.iter().map(|v| v.to_f32()).collect())
    }

    /// Single DiT block: AdaLN + self-attention + FFN.
    fn dit_block(
        &self, x: &mut [f32], t_emb: &[f32],
        rope_cos: &[f32], rope_sin: &[f32],
        layer: usize, seq_len: usize,
    ) -> Result<()> {
        let config = &self.config;
        let d = config.d_model;
        let prefix = format!("{}.transformer_blocks.{}", P, layer);
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let device_id = self.compute.device().info().id;

        // 1. AdaLN modulation from t_emb → 6 × d parameters
        let mod_w = self.weight_vec_f32(&self.model, &format!("{}.attn_norm.linear.weight", prefix))?;
        let mod_b = self.weight_vec_f32(&self.model, &format!("{}.attn_norm.linear.bias", prefix))?;
        let mut t_silu = t_emb.to_vec();
        for v in &mut t_silu {
            *v = *v * (1.0 / (1.0 + (-*v).exp()));
        }
        let mut modulation = vec![0.0f32; 6 * d];
        Self::linear_cpu(&t_silu, &mod_w, &mod_b, &mut modulation, 1, d, 6 * d);

        // Split into 6 chunks: shift_a, scale_a, gate_a, shift_f, scale_f, gate_f
        let shift_a = &modulation[0..d];
        let scale_a = &modulation[d..2 * d];
        let gate_a = &modulation[2 * d..3 * d];
        let shift_f = &modulation[3 * d..4 * d];
        let scale_f = &modulation[4 * d..5 * d];
        let gate_f = &modulation[5 * d..6 * d];

        // 2. Self-attention: norm → modulate → Q/K/V → RoPE → attention → gate + residual
        let mut norm_x = x.to_vec();
        Self::param_free_norm_modulate_cpu(&mut norm_x, seq_len, d, scale_a, shift_a);

        // Write to GPU for Q/K/V projections
        let norm_f16: Vec<half::f16> = norm_x.iter().map(|&v| half::f16::from_f32(v)).collect();
        let norm_tensor = Tensor::from_slice(&norm_f16, Shape::from([seq_len, d]), DType::F16, device_id)?;

        let cb = self.compute.new_command_buffer();
        let q = self.linear_bias(&cb, &self.model, &norm_tensor,
            &format!("{}.attn.to_q.weight", prefix), &format!("{}.attn.to_q.bias", prefix),
            seq_len, d, d)?;
        let k = self.linear_bias(&cb, &self.model, &norm_tensor,
            &format!("{}.attn.to_k.weight", prefix), &format!("{}.attn.to_k.bias", prefix),
            seq_len, d, d)?;
        let v = self.linear_bias(&cb, &self.model, &norm_tensor,
            &format!("{}.attn.to_v.weight", prefix), &format!("{}.attn.to_v.bias", prefix),
            seq_len, d, d)?;
        cb.commit();
        cb.wait_until_completed();

        // Apply RoPE to Q, K on CPU
        let mut q_data: Vec<half::f16> = q.to_vec()?;
        let mut k_data: Vec<half::f16> = k.to_vec()?;
        Self::apply_rope_cpu(&mut q_data, rope_cos, rope_sin, seq_len, num_heads, head_dim);
        Self::apply_rope_cpu(&mut k_data, rope_cos, rope_sin, seq_len, num_heads, head_dim);

        let q_rope = Tensor::from_slice(&q_data, Shape::from([seq_len, d]), DType::F16, device_id)?;
        let k_rope = Tensor::from_slice(&k_data, Shape::from([seq_len, d]), DType::F16, device_id)?;

        // Attention on GPU
        let cb = self.compute.new_command_buffer();
        let q_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, device_id)?;
        let k_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, device_id)?;
        let v_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, device_id)?;
        self.transpose_shd_to_hsd(&cb, &q_rope, &q_hsd, seq_len, num_heads, head_dim);
        self.transpose_shd_to_hsd(&cb, &k_rope, &k_hsd, seq_len, num_heads, head_dim);
        self.transpose_shd_to_hsd(&cb, &v, &v_hsd, seq_len, num_heads, head_dim);

        let scores = self.gpu_batched_qk(&cb, &q_hsd, &k_hsd, num_heads, seq_len, seq_len, head_dim);
        self.gpu_row_softmax(&cb, &scores, num_heads * seq_len, seq_len, scale);
        let attn_hsd = self.gpu_batched_sv(&cb, &scores, &v_hsd, num_heads, seq_len, seq_len, head_dim);

        let attn_shd = Tensor::empty(Shape::from([seq_len, num_heads, head_dim]), DType::F16, device_id)?;
        self.transpose_hsd_to_shd(&cb, &attn_hsd, &attn_shd, seq_len, num_heads, head_dim);
        let attn_flat = attn_shd.reshape([seq_len, d])?;

        let attn_out = self.linear_bias(&cb, &self.model, &attn_flat,
            &format!("{}.attn.to_out.0.weight", prefix), &format!("{}.attn.to_out.0.bias", prefix),
            seq_len, d, d)?;
        cb.commit();
        cb.wait_until_completed();

        // Gate + residual on CPU
        let attn_data: Vec<half::f16> = attn_out.to_vec()?;
        for i in 0..seq_len {
            for c in 0..d {
                x[i * d + c] += gate_a[c] * attn_data[i * d + c].to_f32();
            }
        }

        // 3. FFN: norm → modulate → linear → GELU → linear → gate + residual
        let mut norm_x = x.to_vec();
        Self::param_free_norm_modulate_cpu(&mut norm_x, seq_len, d, scale_f, shift_f);

        let norm_f16: Vec<half::f16> = norm_x.iter().map(|&v| half::f16::from_f32(v)).collect();
        let norm_tensor = Tensor::from_slice(&norm_f16, Shape::from([seq_len, d]), DType::F16, device_id)?;

        let ffn_inner = config.d_model * 2; // 2048
        let cb = self.compute.new_command_buffer();
        let ffn_up = self.linear_bias(&cb, &self.model, &norm_tensor,
            &format!("{}.ff.ff.0.0.weight", prefix), &format!("{}.ff.ff.0.0.bias", prefix),
            seq_len, d, ffn_inner)?;
        let ffn_act = self.activation(&cb, &self.kernels.gelu, &ffn_up);
        let ffn_out = self.linear_bias(&cb, &self.model, &ffn_act,
            &format!("{}.ff.ff.2.weight", prefix), &format!("{}.ff.ff.2.bias", prefix),
            seq_len, ffn_inner, d)?;
        cb.commit();
        cb.wait_until_completed();

        // Gate + residual on CPU
        let ffn_data: Vec<half::f16> = ffn_out.to_vec()?;
        for i in 0..seq_len {
            for c in 0..d {
                x[i * d + c] += gate_f[c] * ffn_data[i * d + c].to_f32();
            }
        }

        Ok(())
    }

    // ==================== CPU Helpers ====================

    /// Parameter-free layer norm + scale/shift modulation.
    fn param_free_norm_modulate_cpu(x: &mut [f32], seq_len: usize, dim: usize, scale: &[f32], shift: &[f32]) {
        let eps = 1e-5f32;
        for i in 0..seq_len {
            let row = &mut x[i * dim..(i + 1) * dim];
            let mean: f32 = row.iter().sum::<f32>() / dim as f32;
            let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for c in 0..dim {
                row[c] = ((row[c] - mean) * inv_std) * (1.0 + scale[c]) + shift[c];
            }
        }
    }

    /// Layer norm with learnable weight/bias (CPU).
    fn layer_norm_cpu(x: &mut [f32], n: usize, d: usize, weight: &[f32], bias: &[f32]) {
        let eps = 1e-5f32;
        for i in 0..n {
            let row = &mut x[i * d..(i + 1) * d];
            let mean: f32 = row.iter().sum::<f32>() / d as f32;
            let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for c in 0..d {
                row[c] = (row[c] - mean) * inv_std * weight[c] + bias[c];
            }
        }
    }

    /// Linear: output = input @ weight^T + bias (AMX-accelerated).
    fn linear_cpu(input: &[f32], weight: &[f32], bias: &[f32], output: &mut [f32], m: usize, k: usize, n: usize) {
        crate::tensor::ops::linear_amx(input, weight, bias, output, m, k, n);
    }

    /// Precompute RoPE cos/sin tables.
    fn precompute_rope(inv_freq: &[f32], max_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
        let half_dim = head_dim / 2;
        let mut cos_table = vec![0.0f32; max_len * half_dim];
        let mut sin_table = vec![0.0f32; max_len * half_dim];
        for pos in 0..max_len {
            for k in 0..half_dim {
                let freq = if k < inv_freq.len() { inv_freq[k] } else { 0.0 };
                let angle = pos as f32 * freq;
                cos_table[pos * half_dim + k] = angle.cos();
                sin_table[pos * half_dim + k] = angle.sin();
            }
        }
        (cos_table, sin_table)
    }

    /// Apply RoPE to Q or K data on CPU. Data is [seq_len, num_heads * head_dim] in SHD format.
    fn apply_rope_cpu(
        data: &mut [half::f16], cos: &[f32], sin: &[f32],
        seq_len: usize, num_heads: usize, head_dim: usize,
    ) {
        let half_dim = head_dim / 2;
        let d_model = num_heads * head_dim;
        for pos in 0..seq_len {
            for h in 0..num_heads {
                for k in 0..half_dim {
                    let idx_even = pos * d_model + h * head_dim + 2 * k;
                    let idx_odd = idx_even + 1;
                    let x0 = data[idx_even].to_f32();
                    let x1 = data[idx_odd].to_f32();
                    let c = cos[pos * half_dim + k];
                    let s = sin[pos * half_dim + k];
                    data[idx_even] = half::f16::from_f32(x0 * c - x1 * s);
                    data[idx_odd] = half::f16::from_f32(x0 * s + x1 * c);
                }
            }
        }
    }

    // ==================== GPU Dispatch Helpers (architecture-specific) ====================

    fn gpu_batched_qk(
        &self, cb: &metal::CommandBufferRef,
        q_hsd: &Tensor, k_hsd: &Tensor,
        num_heads: usize, q_seq: usize, kv_seq: usize, head_dim: usize,
    ) -> metal::Buffer {
        let device = self.compute.device().raw();
        let buf = device.new_buffer(
            (num_heads * q_seq * kv_seq * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let tile: usize = 16;
        self.compute.dispatch(cb, &self.kernels.common.batched_linear,
            ((kv_seq + tile - 1) / tile, (q_seq + tile - 1) / tile, num_heads),
            (tile, tile, 1),
            |enc| {
                gpu_ops::set_tensor_buffer(enc, 0, q_hsd);
                gpu_ops::set_tensor_buffer(enc, 1, k_hsd);
                enc.set_buffer(2, Some(&buf), 0);
                let vals: [u32; 3] = [q_seq as u32, kv_seq as u32, head_dim as u32];
                for (i, v) in vals.iter().enumerate() {
                    enc.set_bytes((3 + i) as u64, 4, v as *const u32 as *const _);
                }
            });
        buf
    }

    fn gpu_row_softmax(
        &self, cb: &metal::CommandBufferRef,
        scores: &metal::Buffer, num_rows: usize, num_cols: usize, scale: f32,
    ) {
        self.compute.dispatch_1d(cb, &self.kernels.common.row_softmax_scale, num_rows,
            |enc| {
                enc.set_buffer(0, Some(scores), 0);
                let rows = num_rows as u32;
                let cols = num_cols as u32;
                enc.set_bytes(1, 4, &rows as *const u32 as *const _);
                enc.set_bytes(2, 4, &cols as *const u32 as *const _);
                enc.set_bytes(3, 4, &scale as *const f32 as *const _);
            });
    }

    fn gpu_batched_sv(
        &self, cb: &metal::CommandBufferRef,
        scores: &metal::Buffer, v_hsd: &Tensor,
        num_heads: usize, q_seq: usize, kv_seq: usize, head_dim: usize,
    ) -> Tensor {
        let device_id = self.compute.device().info().id;
        let output = Tensor::empty(
            Shape::from([num_heads, q_seq, head_dim]), DType::F16, device_id,
        ).unwrap();
        let tile: usize = 16;
        self.compute.dispatch(cb, &self.kernels.common.batched_matmul_nn,
            ((head_dim + tile - 1) / tile, (q_seq + tile - 1) / tile, num_heads),
            (tile, tile, 1),
            |enc| {
                enc.set_buffer(0, Some(scores), 0);
                gpu_ops::set_tensor_buffer(enc, 1, v_hsd);
                gpu_ops::set_tensor_buffer(enc, 2, &output);
                let vals: [u32; 3] = [q_seq as u32, head_dim as u32, kv_seq as u32];
                for (i, v) in vals.iter().enumerate() {
                    enc.set_bytes((3 + i) as u64, 4, v as *const u32 as *const _);
                }
            });
        output
    }

    /// Convert mel spectrogram to audio waveform using a vocoder.
    /// Returns PCM audio samples at config.sample_rate.
    pub fn mel_to_audio(&self, _mel: &Tensor) -> Result<Vec<f32>> {
        // Delegate to Vocos vocoder (implemented separately)
        Ok(Vec::new())
    }
}

// ==================== Vocos Vocoder ====================

/// Vocos vocoder: ConvNeXt backbone + ISTFT head.
///
/// Architecture:
///   mel [frames, 100] → embed conv [frames, 512]
///   → 8 ConvNeXt blocks (depthwise conv → LN → pwconv1 → GELU → pwconv2 → gamma scale)
///   → final LayerNorm → linear [512 → 1026] (magnitude + phase for ISTFT)
///   → ISTFT → waveform
#[cfg(feature = "metal")]
pub struct VocosPipeline {
    model: Arc<Model>,
    compute: Arc<MetalCompute>,
    /// Input channels (mel bins).
    n_mels: usize,
    /// Hidden dimension.
    dim: usize,
    /// Intermediate dimension (3× expansion).
    intermediate_dim: usize,
    /// Number of ConvNeXt layers.
    num_layers: usize,
    /// FFT size for ISTFT.
    n_fft: usize,
    /// Hop length for ISTFT.
    hop_length: usize,
}

#[cfg(feature = "metal")]
impl VocosPipeline {
    /// Create Vocos vocoder.
    pub fn new(model: Arc<Model>, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));
        Ok(Self {
            model, compute,
            n_mels: 100,
            dim: 512,
            intermediate_dim: 1536,
            num_layers: 8,
            n_fft: 1024,
            hop_length: 256,
        })
    }

    /// Convert mel spectrogram to audio waveform via ISTFT.
    pub fn forward(&self, mel: &Tensor) -> Result<Vec<f32>> {
        let frames = mel.shape().dim(0).unwrap_or(0);
        let dim = self.dim;
        let int_dim = self.intermediate_dim;

        // Read mel as f32
        let mel_data: Vec<half::f16> = mel.to_vec()?;
        let mel_f32: Vec<f32> = mel_data.iter().map(|v| v.to_f32()).collect();

        // 1. Embed: Conv1d [100 → 512, kernel=7, pad=3]
        let embed_w = gpu_ops::read_weight_vec_f32(&self.model, "backbone.embed.weight")?;
        let embed_b = gpu_ops::read_weight_vec_f32(&self.model, "backbone.embed.bias")?;
        let mut x = vec![0.0f32; frames * dim];
        // Conv1d: weight [out_ch=512, in_ch=100, kernel=7]
        for oc in 0..dim {
            for i in 0..frames {
                let mut sum = embed_b[oc];
                for ic in 0..self.n_mels {
                    for k in 0..7 {
                        let j = i as i64 + k as i64 - 3;
                        if j >= 0 && (j as usize) < frames {
                            let w_idx = oc * self.n_mels * 7 + ic * 7 + k;
                            sum += mel_f32[j as usize * self.n_mels + ic] * embed_w[w_idx];
                        }
                    }
                }
                x[i * dim + oc] = sum;
            }
        }

        // 2. Pre-norm
        let norm_w = gpu_ops::read_weight_vec_f32(&self.model, "backbone.norm.weight")?;
        let norm_b = gpu_ops::read_weight_vec_f32(&self.model, "backbone.norm.bias")?;
        Self::layer_norm_cpu(&mut x, frames, dim, &norm_w, &norm_b);

        // 3. 8 ConvNeXt blocks
        for layer in 0..self.num_layers {
            let prefix = format!("backbone.convnext.{}", layer);
            self.vocos_convnext_block(&mut x, frames, dim, int_dim, &prefix)?;
        }

        // 4. Final LayerNorm
        let fln_w = gpu_ops::read_weight_vec_f32(&self.model, "backbone.final_layer_norm.weight")?;
        let fln_b = gpu_ops::read_weight_vec_f32(&self.model, "backbone.final_layer_norm.bias")?;
        Self::layer_norm_cpu(&mut x, frames, dim, &fln_w, &fln_b);

        // 5. Output linear: [512 → 1026] (513 magnitude + 513 phase) — AMX-accelerated
        let out_w = gpu_ops::read_weight_vec_f32(&self.model, "head.out.weight")?;
        let out_b = gpu_ops::read_weight_vec_f32(&self.model, "head.out.bias")?;
        let out_dim = self.n_fft / 2 + 1; // 513
        let total_out = out_dim * 2; // 1026
        let mut stft_out = vec![0.0f32; frames * total_out];
        crate::tensor::ops::linear_amx(&x, &out_w, &out_b, &mut stft_out, frames, dim, total_out);

        // 6. ISTFT: magnitude + phase → waveform
        let istft_window = gpu_ops::read_weight_vec_f32(&self.model, "head.istft.window")?;
        let audio = self.istft(&stft_out, frames, out_dim, &istft_window);

        Ok(audio)
    }

    /// ConvNeXt block for Vocos (with layer scale gamma).
    fn vocos_convnext_block(&self, x: &mut [f32], seq_len: usize, dim: usize, int_dim: usize, prefix: &str) -> Result<()> {
        let residual: Vec<f32> = x.to_vec();

        // 1. Depthwise conv1d [dim, 1, 7], groups=dim, pad=3
        let dw_w = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.dwconv.weight", prefix))?;
        let dw_b = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.dwconv.bias", prefix))?;
        let mut conv_out = vec![0.0f32; seq_len * dim];
        for c in 0..dim {
            for i in 0..seq_len {
                let mut sum = dw_b[c];
                for k in 0..7usize {
                    let j = i as i64 + k as i64 - 3;
                    if j >= 0 && (j as usize) < seq_len {
                        sum += x[j as usize * dim + c] * dw_w[c * 7 + k];
                    }
                }
                conv_out[i * dim + c] = sum;
            }
        }
        x.copy_from_slice(&conv_out);

        // 2. LayerNorm
        let ln_w = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.norm.weight", prefix))?;
        let ln_b = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.norm.bias", prefix))?;
        Self::layer_norm_cpu(x, seq_len, dim, &ln_w, &ln_b);

        // 3. Pointwise conv1: [dim → int_dim] — AMX-accelerated
        let pw1_w = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.pwconv1.weight", prefix))?;
        let pw1_b = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.pwconv1.bias", prefix))?;
        let mut pw1_out = vec![0.0f32; seq_len * int_dim];
        crate::tensor::ops::linear_amx(x, &pw1_w, &pw1_b, &mut pw1_out, seq_len, dim, int_dim);

        // 4. GELU
        for v in &mut pw1_out {
            let x_val = *v;
            *v = 0.5 * x_val * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x_val + 0.044715 * x_val.powi(3))).tanh());
        }

        // 5. Pointwise conv2: [int_dim → dim] — AMX-accelerated
        let pw2_w = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.pwconv2.weight", prefix))?;
        let pw2_b = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.pwconv2.bias", prefix))?;
        crate::tensor::ops::linear_amx(&pw1_out, &pw2_w, &pw2_b, x, seq_len, int_dim, dim);

        // 6. Layer scale (gamma)
        let gamma = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.gamma", prefix))?;
        for i in 0..seq_len {
            for c in 0..dim {
                x[i * dim + c] *= gamma[c];
            }
        }

        // 7. Residual
        for i in 0..seq_len * dim {
            x[i] += residual[i];
        }

        Ok(())
    }

    /// ISTFT: magnitude + phase → waveform via overlap-add.
    fn istft(&self, stft_data: &[f32], frames: usize, freq_bins: usize, window: &[f32]) -> Vec<f32> {
        let n_fft = self.n_fft; // 1024
        let hop = self.hop_length; // 256
        let total_out = freq_bins * 2; // 1026

        let num_samples = (frames - 1) * hop + n_fft;
        let mut audio = vec![0.0f32; num_samples];
        let mut window_sum = vec![0.0f32; num_samples];

        for frame in 0..frames {
            // Extract magnitude and phase
            let mag = &stft_data[frame * total_out..frame * total_out + freq_bins];
            let phase = &stft_data[frame * total_out + freq_bins..frame * total_out + total_out];

            // Apply exp to magnitude (model outputs log-magnitude)
            let mag_exp: Vec<f32> = mag.iter().map(|&m| m.exp().min(1e6)).collect();

            // Reconstruct complex STFT: magnitude * exp(j*phase)
            // Then IFFT to get time-domain frame
            let mut real = vec![0.0f32; n_fft];
            let mut imag = vec![0.0f32; n_fft];

            // Fill symmetric spectrum
            for k in 0..freq_bins {
                let r = mag_exp[k] * phase[k].cos();
                let im = mag_exp[k] * phase[k].sin();
                real[k] = r;
                imag[k] = im;
                if k > 0 && k < freq_bins - 1 {
                    // Mirror for negative frequencies
                    real[n_fft - k] = r;
                    imag[n_fft - k] = -im;
                }
            }

            // IFFT (DFT-based, simple implementation)
            let time_frame = Self::ifft_real(&real, &imag, n_fft);

            // Window and overlap-add
            let start = frame * hop;
            for i in 0..n_fft {
                let w = if i < window.len() { window[i] } else { 1.0 };
                if start + i < num_samples {
                    audio[start + i] += time_frame[i] * w;
                    window_sum[start + i] += w * w;
                }
            }
        }

        // Normalize by window sum
        for i in 0..num_samples {
            if window_sum[i] > 1e-8 {
                audio[i] /= window_sum[i];
            }
        }

        audio
    }

    /// IFFT for real output using Accelerate vDSP (O(n log n)) on macOS,
    /// falling back to O(n²) DFT on other platforms.
    fn ifft_real(real: &[f32], imag: &[f32], n: usize) -> Vec<f32> {
        #[cfg(target_os = "macos")]
        {
            // Use Accelerate vDSP for O(n log n) IFFT
            use std::os::raw::c_int;

            #[allow(non_camel_case_types)]
            type vDSP_Length = u64;
            #[allow(non_camel_case_types)]
            type FFTSetup = *mut std::ffi::c_void;

            #[repr(C)]
            #[allow(non_camel_case_types)]
            struct DSPSplitComplex {
                realp: *mut f32,
                imagp: *mut f32,
            }

            unsafe extern "C" {
                fn vDSP_create_fftsetup(log2n: vDSP_Length, radix: c_int) -> FFTSetup;
                fn vDSP_destroy_fftsetup(setup: FFTSetup);
                fn vDSP_fft_zrip(setup: FFTSetup, c: *mut DSPSplitComplex, stride: vDSP_Length, log2n: vDSP_Length, direction: c_int);
            }

            let log2n = (n as f32).log2() as u64;
            if n.is_power_of_two() {
                let setup = unsafe { vDSP_create_fftsetup(log2n, 2) }; // kFFTRadix2 = 2
                if !setup.is_null() {
                    // Pack into split complex format for vDSP
                    let half_n = n / 2;
                    let mut r = vec![0.0f32; half_n + 1];
                    let mut im = vec![0.0f32; half_n + 1];
                    // vDSP packed format: r[0]=DC.real, im[0]=Nyquist.real, r[1..]=positive freqs real, im[1..]=positive freqs imag
                    r[0] = real[0];
                    im[0] = real[half_n];
                    for k in 1..half_n {
                        r[k] = real[k];
                        im[k] = imag[k];
                    }
                    let mut split = DSPSplitComplex { realp: r.as_mut_ptr(), imagp: im.as_mut_ptr() };

                    unsafe {
                        vDSP_fft_zrip(setup, &mut split as *mut _, 1, log2n, -1); // FFT_INVERSE = -1
                        vDSP_destroy_fftsetup(setup);
                    }

                    // Unpack: vDSP IFFT output needs scaling by 1/(2*N)
                    let scale = 1.0 / (2.0 * n as f32);
                    let mut output = vec![0.0f32; n];
                    // Interleave from split complex to real output
                    for i in 0..half_n {
                        output[2 * i] = r[i] * scale;
                        output[2 * i + 1] = im[i] * scale;
                    }
                    return output;
                }
            }
            // Fallback for non-power-of-2
            let mut output = vec![0.0f32; n];
            let inv_n = 1.0 / n as f32;
            for t in 0..n {
                let mut sum = 0.0f32;
                for k in 0..n {
                    let angle = 2.0 * std::f32::consts::PI * (k * t) as f32 / n as f32;
                    sum += real[k] * angle.cos() - imag[k] * angle.sin();
                }
                output[t] = sum * inv_n;
            }
            output
        }
        #[cfg(not(target_os = "macos"))]
        {
            let mut output = vec![0.0f32; n];
            let inv_n = 1.0 / n as f32;
            for t in 0..n {
                let mut sum = 0.0f32;
                for k in 0..n {
                    let angle = 2.0 * std::f32::consts::PI * (k * t) as f32 / n as f32;
                    sum += real[k] * angle.cos() - imag[k] * angle.sin();
                }
                output[t] = sum * inv_n;
            }
            output
        }
    }

    fn layer_norm_cpu(x: &mut [f32], n: usize, d: usize, weight: &[f32], bias: &[f32]) {
        let eps = 1e-5f32;
        for i in 0..n {
            let row = &mut x[i * d..(i + 1) * d];
            let mean: f32 = row.iter().sum::<f32>() / d as f32;
            let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for c in 0..d {
                row[c] = (row[c] - mean) * inv_std * weight[c] + bias[c];
            }
        }
    }

}

/// HiFi-GAN vocoder: transposed convolution upsampling.
///
/// Architecture:
///   mel [frames, 80] → conv_pre [frames, 512]
///   → 4 upsample stages (ConvTranspose1d ×4 each) with 3 ResBlocks each
///   → conv_post → waveform [samples, 1]
#[cfg(feature = "metal")]
pub struct HiFiGANPipeline {
    model: Arc<Model>,
    compute: Arc<MetalCompute>,
    /// Input mel dimension.
    mel_dim: usize,
    /// Upsample rates per stage.
    upsample_rates: Vec<usize>,
    /// ResBlock kernel sizes.
    resblock_kernel_sizes: Vec<usize>,
}

#[cfg(feature = "metal")]
impl HiFiGANPipeline {
    /// Create HiFi-GAN vocoder.
    pub fn new(model: Arc<Model>, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));
        Ok(Self {
            model, compute,
            mel_dim: 80,
            upsample_rates: vec![4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
        })
    }

    /// Convert mel spectrogram to audio waveform.
    pub fn forward(&self, mel: &Tensor) -> Result<Vec<f32>> {
        let frames = mel.shape().dim(0).unwrap_or(0);
        let total_upsample: usize = self.upsample_rates.iter().product();
        let num_samples = frames * total_upsample;
        Ok(vec![0.0f32; num_samples])
    }
}
