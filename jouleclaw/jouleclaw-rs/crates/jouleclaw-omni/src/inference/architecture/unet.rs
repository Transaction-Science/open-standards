//! UNet architecture for Stable Diffusion XL.
//!
//! Implements the 2D UNet backbone used in SDXL/SD1.5.

use crate::core::Result;
use crate::tensor::{Tensor, Shape, DType};
use crate::inference::model::Model;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

#[cfg(feature = "metal")]
use crate::hal::metal::MetalCompute;
#[cfg(feature = "metal")]
use crate::hal::metal::BorrowedMetalBuffer;
#[cfg(feature = "metal")]
use metal::CommandBufferRef;

/// SD_DIAG=1 snapshot table: collect (label, tensor) clones during the UNet
/// forward and dump them AFTER the caller's command buffer is committed.
/// Reading device buffers while their producer CB is still queued returns
/// uninitialised memory (the v53 hazard) — so we never read in flight; we
/// keep the Tensor (Arc-backed) alive and replay the stats at the end.
#[cfg(feature = "metal")]
thread_local! {
    static DIAG_SNAPSHOTS: std::cell::RefCell<Vec<(String, Tensor)>> =
        std::cell::RefCell::new(Vec::new());
    static DIAG_CURRENT_STEP: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

/// Tell the diag layer which denoise step we are on, so an optional
/// `SD_DIAG_STEP=<n>` env can scope dumps to a single step.
#[cfg(feature = "metal")]
pub fn diag_set_step(step: usize) {
    DIAG_CURRENT_STEP.with(|c| c.set(step));
}

#[cfg(feature = "metal")]
#[inline]
fn diag_should_dump() -> bool {
    if std::env::var("SD_DIAG").ok().as_deref() != Some("1") { return false; }
    if let Ok(want) = std::env::var("SD_DIAG_STEP") {
        if let Ok(want) = want.parse::<usize>() {
            let cur = DIAG_CURRENT_STEP.with(|c| c.get());
            return cur == want;
        }
    }
    true
}

/// Stash a tensor clone keyed by `label` for post-commit stat replay.
#[cfg(feature = "metal")]
#[inline]
fn diag_snapshot(label: &str, t: &Tensor) {
    if !diag_should_dump() { return; }
    DIAG_SNAPSHOTS.with(|c| c.borrow_mut().push((label.to_string(), t.clone())));
}

/// Drain the snapshot table and log stats. Must be called only after the
/// command buffer that produced these tensors has been committed and waited.
#[cfg(feature = "metal")]
pub fn diag_drain_and_log() {
    if !diag_should_dump() { return; }
    let dump_per_channel = std::env::var("SD_DIAG_PERCH").ok().as_deref() == Some("1");
    // When SD_DUMP_DIR is set, also write the block-boundary tensors to f32
    // files so a PT side-by-side can compute cos at each block. Only the
    // labels in this list dump (matches PT's hook points exactly).
    let dump_dir = std::env::var("SD_DUMP_DIR").ok();
    let dump_labels: &[&str] = &[
        "02 after_conv_in",
        "03 after_down_block_0", "03 after_down_block_1",
        "03 after_down_block_2", "03 after_down_block_3",
        "04 mid_resnet_0", "06 mid_resnet_1",
        "07 after_up_block_0", "07 after_up_block_1",
        "07 after_up_block_2", "07 after_up_block_3",
        "09 after_conv_out (final)",
    ];
    DIAG_SNAPSHOTS.with(|c| {
        let snaps = std::mem::take(&mut *c.borrow_mut());
        for (label, t) in snaps {
            if let Ok(v) = t.to_f32_vec() {
                if v.is_empty() { continue; }
                if let Some(ref dir) = dump_dir {
                    if dump_labels.iter().any(|p| label.trim() == *p) {
                        let safe: String = label.chars()
                            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                            .collect();
                        let mut bytes = Vec::with_capacity(v.len() * 4);
                        for f in &v { bytes.extend_from_slice(&f.to_le_bytes()); }
                        let _ = std::fs::write(format!("{}/blk_{}.f32", dir, safe), &bytes);
                    }
                }
                let mut mn = f32::INFINITY;
                let mut mx = f32::NEG_INFINITY;
                let mut s = 0.0f64;
                for &x in &v { if x < mn { mn = x; } if x > mx { mx = x; } s += x as f64; }
                let mean = (s / v.len() as f64) as f32;
                let mut var = 0.0f64;
                for &x in &v { let d = (x as f64) - (mean as f64); var += d * d; }
                let std = (var / v.len() as f64).sqrt() as f32;
                tracing::info!(
                    "[diag-unet] {:<30} shape={:?} mean={:+.4} std={:.4} min={:+.4} max={:+.4}",
                    label, t.shape(), mean, std, mn, mx,
                );
                // Per-channel stats for [N, C, H, W] tensors when SD_DIAG_PERCH=1.
                // Critical for diagnosing "right magnitude, wrong pattern" issues:
                // a channel-permutation bug between our GN/SiLU output and what
                // the trained conv_out expects would show as matching pc_std
                // distribution but in a different ORDER.
                if dump_per_channel && t.shape().dims().len() == 4 {
                    let dims = t.shape().dims();
                    let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
                    let hw = h * w;
                    let mut pc_stds = Vec::with_capacity(c);
                    let mut pc_means = Vec::with_capacity(c);
                    for ci in 0..c {
                        let mut s = 0.0f64;
                        let mut cnt = 0usize;
                        for ni in 0..n {
                            let base = ni * c * hw + ci * hw;
                            for k in 0..hw {
                                s += v[base + k] as f64;
                                cnt += 1;
                            }
                        }
                        let m = (s / cnt as f64) as f32;
                        let mut ss = 0.0f64;
                        for ni in 0..n {
                            let base = ni * c * hw + ci * hw;
                            for k in 0..hw {
                                let d = (v[base + k] as f64) - m as f64;
                                ss += d * d;
                            }
                        }
                        pc_means.push(m);
                        pc_stds.push((ss / cnt as f64).sqrt() as f32);
                    }
                    let pc_std_mean: f32 = pc_stds.iter().sum::<f32>() / pc_stds.len() as f32;
                    let pc_std_min: f32 = pc_stds.iter().cloned().fold(f32::INFINITY, f32::min);
                    let pc_std_max: f32 = pc_stds.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let first_10_stds: Vec<String> = pc_stds.iter().take(10).map(|v| format!("{:.4}", v)).collect();
                    let first_10_means: Vec<String> = pc_means.iter().take(10).map(|v| format!("{:+.4}", v)).collect();
                    tracing::info!(
                        "[diag-unet] {:<30} per-channel: pc_std[mean={:.4} min={:.4} max={:.4}] first10_std=[{}] first10_mean=[{}]",
                        label, pc_std_mean, pc_std_min, pc_std_max,
                        first_10_stds.join(","),
                        first_10_means.join(","),
                    );
                }
            }
        }
    });
}

/// Non-metal stub so callers can unconditionally set the diag step.
#[cfg(not(feature = "metal"))]
pub fn diag_set_step(_step: usize) {}

/// UNet 2D Condition Model.
pub struct UNet2DConditionModel {
    /// Lazy-loaded weights
    model: Arc<Model>,
    /// Sample size (e.g., 64)
    sample_size: usize,
    /// In channels (e.g., 4)
    in_channels: usize,
    /// Out channels (e.g., 4)
    out_channels: usize,
    /// Down blocks
    down_blocks: Vec<DownBlockType>,
    /// Up blocks (reversed)
    up_blocks: Vec<UpBlockType>,
    /// Block out channels
    block_out_channels: Vec<usize>,
    /// Layers per block
    layers_per_block: usize,
    /// Attention head dim
    attention_head_dim: usize,
    /// Cache for dummy weights (to avoid re-generating on every step)
    dummy_cache: Mutex<HashMap<String, Tensor>>,
}

#[derive(Debug, Clone, PartialEq)]
enum DownBlockType {
    DownBlock2D,
    CrossAttnDownBlock2D,
}

#[derive(Debug, Clone, PartialEq)]
enum UpBlockType {
    UpBlock2D,
    CrossAttnUpBlock2D,
}

impl UNet2DConditionModel {
    /// Create a new UNet model wrapper. Detects SD 1.5 vs SDXL by probing
    /// for `down_blocks.3.*` weights (SD 1.5 has 4 down blocks; SDXL has 3).
    pub fn new(model: Arc<Model>) -> Self {
        #[cfg(feature = "metal")]
        let is_sd15 = model
            .get_weight("down_blocks.3.resnets.0.norm1.weight")
            .is_some()
            || model
                .get_weight("down_blocks.3.resnets.0.conv1.weight")
                .is_some();
        #[cfg(not(feature = "metal"))]
        let is_sd15 = false;

        if is_sd15 {
            // SD 1.5: 4 down blocks, [CrossAttn, CrossAttn, CrossAttn, Down]
            //         4 up blocks, [Up, CrossAttn, CrossAttn, CrossAttn]
            //         block_out_channels [320, 640, 1280, 1280]
            //         sample_size 64 (512×512 latent → 64×64).
            Self {
                model,
                sample_size: 64,
                in_channels: 4,
                out_channels: 4,
                down_blocks: vec![
                    DownBlockType::CrossAttnDownBlock2D,
                    DownBlockType::CrossAttnDownBlock2D,
                    DownBlockType::CrossAttnDownBlock2D,
                    DownBlockType::DownBlock2D,
                ],
                up_blocks: vec![
                    UpBlockType::UpBlock2D,
                    UpBlockType::CrossAttnUpBlock2D,
                    UpBlockType::CrossAttnUpBlock2D,
                    UpBlockType::CrossAttnUpBlock2D,
                ],
                block_out_channels: vec![320, 640, 1280, 1280],
                layers_per_block: 2,
                // SD 1.5 uses head_dim=8 (so num_heads varies per level:
                // 40 for 320ch, 80 for 640ch, 160 for 1280ch). The diffusers
                // config.json key is named "attention_head_dim" and the value
                // there literally is the per-head dim, NOT num_heads.
                // SDXL uses head_dim=64 with num_heads=10/20 per level.
                // Using 64 for SD 1.5 would compute attention with 5/10/20
                // heads at the wrong head_dim, scaling softmax by 1/√64
                // instead of 1/√8 — the dominant cause of network-wide
                // noise_pred magnitude damping (std ≈0.1 vs trained ≈1).
                attention_head_dim: 8,
                dummy_cache: Mutex::new(HashMap::new()),
            }
        } else {
            // SDXL default: 3 down blocks, [Down, CrossAttn, CrossAttn]
            //               3 up blocks, [CrossAttn, CrossAttn, Up]
            //               block_out_channels [320, 640, 1280]
            Self {
                model,
                sample_size: 128, // SDXL is 1024x1024 → 128x128 latents
                in_channels: 4,
                out_channels: 4,
                down_blocks: vec![
                    DownBlockType::DownBlock2D,
                    DownBlockType::CrossAttnDownBlock2D,
                    DownBlockType::CrossAttnDownBlock2D,
                ],
                up_blocks: vec![
                    UpBlockType::CrossAttnUpBlock2D,
                    UpBlockType::CrossAttnUpBlock2D,
                    UpBlockType::UpBlock2D,
                ],
                block_out_channels: vec![320, 640, 1280],
                layers_per_block: 2,
                attention_head_dim: 64,
                dummy_cache: Mutex::new(HashMap::new()),
            }
        }
    }

    /// Forward pass.
    ///
    /// `controlnet_residuals`, if `Some`, is a slice produced by
    /// `ControlNet::get_conditioning` — 12 down-block residuals followed by
    /// 1 mid-block residual (13 total) at SD-1.5 default shapes. They are
    /// added into the corresponding skip connections / mid output. Pass
    /// `None` for vanilla generation.
    #[cfg(feature = "metal")]
    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: f32,
        _encoder_hidden_states: &Tensor,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        self.forward_with_residuals(sample, timestep, _encoder_hidden_states, None, compute, command_buffer)
    }

    /// Number of downsampling blocks (4 for SD 1.5 / SDXL).
    pub fn down_block_count(&self) -> usize { self.down_blocks.len() }

    /// Number of resnet layers per downsampling block (2 for SD 1.5 / SDXL).
    pub fn layers_per_block_count(&self) -> usize { self.layers_per_block }

    /// Whether down block `i` carries cross-attention (true for SD 1.5
    /// blocks 0..2, false for block 3 which is a plain DownBlock2D).
    pub fn down_block_has_cross_attn(&self, i: usize) -> bool {
        matches!(self.down_blocks.get(i), Some(DownBlockType::CrossAttnDownBlock2D))
    }

    /// Encoder-only forward: runs `conv_in` + downsampling blocks + mid
    /// block, returning the per-layer skip-connection residuals plus the
    /// mid-block output. Used by ControlNet's encoder copy to produce
    /// residuals at the same shapes the U-Net's up blocks consume.
    ///
    /// Returns `(down_residuals, mid_output)`. The residuals are in the
    /// order they're pushed during downsampling — same order the up-block
    /// loop pops them in `forward_with_residuals`.
    #[cfg(feature = "metal")]
    pub fn encode(
        &self,
        sample: &Tensor,
        timestep: f32,
        encoder_hidden_states: &Tensor,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<(Vec<Tensor>, Tensor)> {
        let t_emb = self.get_timestep_embedding(timestep, 320, compute, command_buffer)?;
        let sample = self.conv2d(
            sample,
            "conv_in.weight",
            Some("conv_in.bias"),
            compute,
            command_buffer,
            1, 1,
        )?;
        // SD 1.5 reference pushes the conv_in output as the very first
        // skip residual (consumed by the deepest up block's last resnet).
        let mut down_block_res_samples = vec![sample.clone()];
        let mut hidden_states = sample.clone();
        for (i, down_block_type) in self.down_blocks.iter().enumerate() {
            let is_final_block = i == self.down_blocks.len() - 1;
            for j in 0..self.layers_per_block {
                let prefix = format!("down_blocks.{}.resnets.{}", i, j);
                hidden_states = self.resnet_block(&hidden_states, &t_emb, &prefix, compute, command_buffer)?;
                if matches!(down_block_type, DownBlockType::CrossAttnDownBlock2D) {
                    let attn_prefix = format!("down_blocks.{}.attentions.{}", i, j);
                    hidden_states = self.transformer_block(&hidden_states, encoder_hidden_states, &attn_prefix, compute, command_buffer)?;
                }
                down_block_res_samples.push(hidden_states.clone());
            }
            if !is_final_block {
                let prefix = format!("down_blocks.{}.downsamplers.0", i);
                hidden_states = self.downsample(&hidden_states, &prefix, compute, command_buffer)?;
                down_block_res_samples.push(hidden_states.clone());
            }
        }
        let mid_prefix = "mid_block";
        hidden_states = self.resnet_block(&hidden_states, &t_emb, &format!("{}.resnets.0", mid_prefix), compute, command_buffer)?;
        hidden_states = self.transformer_block(&hidden_states, encoder_hidden_states, &format!("{}.attentions.0", mid_prefix), compute, command_buffer)?;
        hidden_states = self.resnet_block(&hidden_states, &t_emb, &format!("{}.resnets.1", mid_prefix), compute, command_buffer)?;
        Ok((down_block_res_samples, hidden_states))
    }

    /// Forward pass with optional ControlNet residual injection.
    #[cfg(feature = "metal")]
    pub fn forward_with_residuals(
        &self,
        sample: &Tensor,
        timestep: f32,
        _encoder_hidden_states: &Tensor,
        controlnet_residuals: Option<&[Tensor]>,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        diag_snapshot("00 input_latent", sample);

        // 0. Pre-process timestep embedding: sinusoidal(320) → MLP → [1, 1280]
        let t_emb = self.get_timestep_embedding(timestep, 320, compute, command_buffer)?;
        diag_snapshot("01 t_emb", &t_emb);

        // 1. Initial convolution
        let sample = self.conv2d(
            sample,
            "conv_in.weight",
            Some("conv_in.bias"),
            compute,
            command_buffer,
            1, 1
        )?;
        diag_snapshot("02 after_conv_in", &sample);

        // 2. Downsampling
        // SD 1.5 reference pushes conv_in output as residual 0; align with it
        // so 12 down + 1 mid skip count matches HF ControlNet zero-conv count.
        let initial_save = match controlnet_residuals.and_then(|r| r.get(0)) {
            Some(r) => self.add(&sample, r, compute, command_buffer)?,
            None => sample.clone(),
        };
        let mut down_block_res_samples = vec![initial_save];
        let mut residual_idx: usize = 1;
        let mut hidden_states = sample.clone();

        for (i, down_block_type) in self.down_blocks.iter().enumerate() {
            let is_final_block = i == self.down_blocks.len() - 1;
            // let current_dim = self.block_out_channels[i];
            // let prev_dim = if i == 0 { self.block_out_channels[0] } else { self.block_out_channels[i-1] };

            // Iterate layers in block
            for j in 0..self.layers_per_block {
                let prefix = format!("down_blocks.{}.resnets.{}", i, j);
                hidden_states = self.resnet_block(&hidden_states, &t_emb, &prefix, compute, command_buffer)?;
                diag_snapshot(&format!("DB.{}.r{} after_resnet", i, j), &hidden_states);

                if matches!(down_block_type, DownBlockType::CrossAttnDownBlock2D) {
                    let attn_prefix = format!("down_blocks.{}.attentions.{}", i, j);
                    hidden_states = self.transformer_block(&hidden_states, _encoder_hidden_states, &attn_prefix, compute, command_buffer)?;
                    diag_snapshot(&format!("DB.{}.a{} after_xform", i, j), &hidden_states);
                }

                // ControlNet skip-connection residual injection: bias the
                // saved skip activation by the matching residual before the
                // up block consumes it. Residuals are pre-scaled by the
                // caller, so this is a plain add.
                let to_save = match controlnet_residuals.and_then(|r| r.get(residual_idx)) {
                    Some(r) => self.add(&hidden_states, r, compute, command_buffer)?,
                    None => hidden_states.clone(),
                };
                residual_idx += 1;
                down_block_res_samples.push(to_save);
            }

            if !is_final_block {
                let prefix = format!("down_blocks.{}.downsamplers.0", i);
                // HF diffusers reference: push the POST-downsample tensor,
                // at block N+1's resolution. The up block at that resolution
                // consumes this skip with shapes matching its native input.
                hidden_states = self.downsample(&hidden_states, &prefix, compute, command_buffer)?;
                diag_snapshot(&format!("DB.{} after_downsample", i), &hidden_states);
                let to_save = match controlnet_residuals.and_then(|r| r.get(residual_idx)) {
                    Some(r) => self.add(&hidden_states, r, compute, command_buffer)?,
                    None => hidden_states.clone(),
                };
                residual_idx += 1;
                down_block_res_samples.push(to_save);
            }
            diag_snapshot(&format!("03 after_down_block_{}", i), &hidden_states);
        }

        // 3. Mid block
        let mid_prefix = "mid_block";
        hidden_states = self.resnet_block(&hidden_states, &t_emb, &format!("{}.resnets.0", mid_prefix), compute, command_buffer)?;
        diag_snapshot("04 mid_resnet_0", &hidden_states);
        hidden_states = self.transformer_block(&hidden_states, _encoder_hidden_states, &format!("{}.attentions.0", mid_prefix), compute, command_buffer)?;
        diag_snapshot("05 mid_attn", &hidden_states);
        hidden_states = self.resnet_block(&hidden_states, &t_emb, &format!("{}.resnets.1", mid_prefix), compute, command_buffer)?;
        diag_snapshot("06 mid_resnet_1", &hidden_states);

        // ControlNet mid-block residual injection.
        if let Some(residuals) = controlnet_residuals {
            if let Some(mid_r) = residuals.get(residual_idx) {
                hidden_states = self.add(&hidden_states, mid_r, compute, command_buffer)?;
            }
        }

        // 4. Upsampling
        for (i, up_block_type) in self.up_blocks.iter().enumerate() {
            let is_final_block = i == self.up_blocks.len() - 1;
            
            // All up blocks consume `layers_per_block + 1` residuals — the
            // deepest one's extra residual is the conv_in passthrough that
            // we push before the down loop. Matches HF diffusers reference
            // and the 12-slot HF ControlNet zero-conv layout.
            let num_layers = self.layers_per_block + 1;
            
            for j in 0..num_layers {
                diag_snapshot(&format!("UB.{}.r{} before_pop hidden", i, j), &hidden_states);
                let res_hidden_states = down_block_res_samples.pop().ok_or(crate::core::Error::internal("ResNet stack mismatch"))?;
                diag_snapshot(&format!("UB.{}.r{} skip_popped", i, j), &res_hidden_states);

                // Concat along channel dim on GPU (same command buffer as
                // producers — see CAT_NCHW kernel note for the v53-style bug
                // the CPU-fallback used to hit).
                hidden_states = self.gpu_cat_dim1(&hidden_states, &res_hidden_states, compute, command_buffer)?;
                diag_snapshot(&format!("UB.{}.r{} after_cat", i, j), &hidden_states);

                let prefix = format!("up_blocks.{}.resnets.{}", i, j);
                hidden_states = self.resnet_block(&hidden_states, &t_emb, &prefix, compute, command_buffer)?;
                diag_snapshot(&format!("UB.{}.r{} after_resnet", i, j), &hidden_states);

                if matches!(up_block_type, UpBlockType::CrossAttnUpBlock2D) {
                    let attn_prefix = format!("up_blocks.{}.attentions.{}", i, j);
                    hidden_states = self.transformer_block(&hidden_states, _encoder_hidden_states, &attn_prefix, compute, command_buffer)?;
                    diag_snapshot(&format!("UB.{}.a{} after_xform", i, j), &hidden_states);
                }
            }

            if !is_final_block {
                let prefix = format!("up_blocks.{}.upsamplers.0", i);
                hidden_states = self.upsample(&hidden_states, &prefix, compute, command_buffer)?;
                diag_snapshot(&format!("UB.{} after_upsample", i), &hidden_states);
            }
            diag_snapshot(&format!("07 after_up_block_{}", i), &hidden_states);
        }

        // 5. Final convolution. Split GN and SiLU so we can compare the
        // GN-only output directly against the HF diffusers reference dump
        // (PyTorch's `unet.conv_norm_out` hook captures GN BEFORE SiLU).
        // PyTorch reference at step 0: GN std=2.54 → final std=1.0; ours
        // with the fused kernel: GN+SiLU std=0.13 → final std=0.05 (20×
        // damping). Splitting localises which kernel diverges.
        let pre_silu = self.group_norm(&hidden_states, "conv_norm_out", compute, command_buffer)?;
        diag_snapshot("08a after_conv_norm_out_GN_only", &pre_silu);
        hidden_states = self.silu(&pre_silu, compute, command_buffer)?;
        diag_snapshot("08b after_conv_norm_out_silu", &hidden_states);
        hidden_states = self.conv2d(&hidden_states, "conv_out.weight", Some("conv_out.bias"), compute, command_buffer, 1, 1)?;
        diag_snapshot("09 after_conv_out (final)", &hidden_states);

        Ok(hidden_states)
    }

    #[cfg(feature = "metal")]
    pub fn resnet_block(
        &self,
        input: &Tensor,
        temb: &Tensor,
        prefix: &str,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        diag_snapshot(&format!("R[{}] 0 input", prefix), input);
        let act1 = self.group_norm_silu(input, &format!("{}.norm1", prefix), compute, command_buffer)?;
        diag_snapshot(&format!("R[{}] 1 norm1_silu", prefix), &act1);
        let conv1 = self.conv2d(&act1, &format!("{}.conv1.weight", prefix), Some(&format!("{}.conv1.bias", prefix)), compute, command_buffer, 1, 1)?;
        diag_snapshot(&format!("R[{}] 2 conv1", prefix), &conv1);

        // Time embedding injection: SiLU → project 1280 → block_channels → broadcast add
        // Entirely on Metal GPU (no CPU roundtrip)
        let has_time_proj = self.model.get_weight(&format!("{}.time_emb_proj.weight", prefix)).is_some();
        let conv1 = if has_time_proj {
            let temb_silu = self.silu(temb, compute, command_buffer)?;
            let inner_dim = temb.shape().numel();
            let (_, block_ch, h, w) = conv1.shape().dims4().unwrap_or((1, 320, 1, 1));
            let temb_proj = self.gpu_linear(
                &temb_silu, &format!("{}.time_emb_proj", prefix),
                1, inner_dim, block_ch, true, compute, command_buffer,
            )?;
            diag_snapshot(&format!("R[{}] 3 temb_proj", prefix), &temb_proj);
            let withtemb = self.channel_bias_add(&conv1, &temb_proj, block_ch, h * w, compute, command_buffer)?;
            diag_snapshot(&format!("R[{}] 4 conv1+temb", prefix), &withtemb);
            withtemb
        } else {
            conv1
        };

        let act2 = self.group_norm_silu(&conv1, &format!("{}.norm2", prefix), compute, command_buffer)?;
        diag_snapshot(&format!("R[{}] 5 norm2_silu", prefix), &act2);
        let conv2 = self.conv2d(&act2, &format!("{}.conv2.weight", prefix), Some(&format!("{}.conv2.bias", prefix)), compute, command_buffer, 1, 1)?;
        diag_snapshot(&format!("R[{}] 6 conv2", prefix), &conv2);

        // Residual connection: use conv_shortcut if dimensions change
        let has_shortcut = self.model.get_weight(&format!("{}.conv_shortcut.weight", prefix)).is_some();
        let residual = if has_shortcut {
            self.conv2d(input, &format!("{}.conv_shortcut.weight", prefix), Some(&format!("{}.conv_shortcut.bias", prefix)), compute, command_buffer, 1, 0)?
        } else {
            input.clone()
        };
        diag_snapshot(&format!("R[{}] 7 shortcut", prefix), &residual);

        let out = self.add(&residual, &conv2, compute, command_buffer)?;
        diag_snapshot(&format!("R[{}] 8 out", prefix), &out);
        Ok(out)
    }

    /// Transformer2DModel forward pass.
    ///
    /// Supports two weight layouts:
    /// - **SDXL/HF**: `{prefix}.norm`, `{prefix}.proj_in`, `{prefix}.transformer_blocks.{i}.*`, `{prefix}.proj_out`
    /// - **Flat/dummy**: `{prefix}.norm1`, `{prefix}.attn1.*`, etc. (single block, no proj)
    #[cfg(feature = "metal")]
    pub fn transformer_block(
        &self,
        input: &Tensor,
        context: &Tensor,
        prefix: &str,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        // Detect SDXL layout: has transformer_blocks.0
        let has_transformer_blocks = self.model.get_weight(
            &format!("{}.transformer_blocks.0.attn1.to_q.weight", prefix)
        ).is_some();

        if has_transformer_blocks {
            // SDXL Transformer2DModel: norm -> proj_in -> N x BasicTransformerBlock -> proj_out
            let normed = self.group_norm(input, &format!("{}.norm", prefix), compute, command_buffer)?;
            let mut hidden = self.conv2d(
                &normed,
                &format!("{}.proj_in.weight", prefix),
                Some(&format!("{}.proj_in.bias", prefix)),
                compute, command_buffer, 1, 0,
            )?;

            // Count transformer blocks by probing weights
            let mut num_blocks = 0;
            while self.model.get_weight(
                &format!("{}.transformer_blocks.{}.attn1.to_q.weight", prefix, num_blocks)
            ).is_some() {
                num_blocks += 1;
            }

            for b in 0..num_blocks {
                let bp = format!("{}.transformer_blocks.{}", prefix, b);
                hidden = self.basic_transformer_block(&hidden, context, &bp, compute, command_buffer)?;
            }

            // proj_out
            hidden = self.conv2d(
                &hidden,
                &format!("{}.proj_out.weight", prefix),
                Some(&format!("{}.proj_out.bias", prefix)),
                compute, command_buffer, 1, 0,
            )?;

            // Residual with original input
            self.add(input, &hidden, compute, command_buffer)
        } else {
            // Flat layout (dummy models / simple SD1.5)
            if self.model.info().name == "dummy-model" && self.model.get_weight(&format!("{}.attn1.to_q.weight", prefix)).is_none() {
                return Err(crate::core::Error::internal("Transformer2DModel weights not loaded"));
            }
            self.basic_transformer_block(input, context, prefix, compute, command_buffer)
        }
    }

    /// BasicTransformerBlock — 100% Metal GPU.
    ///
    /// self-attn → cross-attn → GEGLU FF, each with LayerNorm + residual.
    /// Input/output: [B, C, H, W] (NCHW) Metal tensors.
    /// Internally transposes to [B*seq, hidden] and dispatches all compute on GPU.
    #[cfg(feature = "metal")]
    fn basic_transformer_block(
        &self,
        input: &Tensor,
        context: &Tensor,
        prefix: &str,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        let (b, c, h, w) = input.shape().dims4().unwrap_or((1, 1, 1, 1));
        let seq_len = h * w;
        let hidden_dim = c;
        // The `attention_head_dim` field semantically holds NUM_HEADS, matching
        // the diffusers `UNet2DConditionModel` config field of the same name —
        // which is misleadingly named: when `num_attention_heads` is None
        // (SD 1.5 and SDXL both omit it), diffusers does
        //   `num_attention_heads = attention_head_dim`
        // and computes `head_dim = inner_dim / num_attention_heads` per level.
        // So for SD 1.5 with `attention_head_dim: 8`:
        //   num_heads = 8 (fixed across all levels)
        //   head_dim  = 40, 80, 160 (varies per level 320/640/1280-ch)
        // Previously we did `num_heads = hidden_dim / attention_head_dim`,
        // which is the dual interpretation and gives 40/80/160 heads of dim 8
        // — softmax scale `1/√8` instead of `1/√40` and the head reshape
        // slices the trained Q/K/V weights into the wrong head pattern.
        let num_heads = self.attention_head_dim.max(1);
        let head_dim = (hidden_dim / num_heads).max(1);

        // ---- Transpose input: NCHW → [B*seq, hidden] on GPU ----
        let hidden_buf = self.gpu_nchw_to_nhwc(input, b, c, seq_len, compute, command_buffer)?;

        // ---- Reshape context: CLIP outputs [B, seq, hidden] in row-major
        // memory order, which is *already* `[B*seq, hidden]` once flattened.
        // No transpose needed; the previous code path tried to interpret
        // the 3D tensor through `dims4()`, which returned `None` and fell
        // back to (1,1,1,1) — that silently shrank the cross-attention
        // context to a single mostly-zero element, making text conditioning
        // a no-op and starving the U-Net of the residual contribution it
        // was trained to expect (the dominant network-wide damping bug).
        let ctx_shape = context.shape();
        let (b_ctx, seq_ctx, dim_ctx) = if ctx_shape.dims().len() == 3 {
            (
                ctx_shape.dim(0).unwrap_or(1),
                ctx_shape.dim(1).unwrap_or(1),
                ctx_shape.dim(2).unwrap_or(1),
            )
        } else if let Some((b, c, h, w)) = ctx_shape.dims4() {
            (b, h * w, c)
        } else {
            (1, 1, 1)
        };
        let ctx_buf = context.reshape([b_ctx * seq_ctx, dim_ctx])?;

        // ============================================================
        // Self-attention: LayerNorm → Q,K,V proj → attention → out proj → residual
        // ============================================================
        diag_snapshot(&format!("T[{}] 00 hidden_in", prefix), &hidden_buf);
        let normed = self.gpu_layer_norm(&hidden_buf, prefix, "norm1", b * seq_len, hidden_dim, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 01 norm1", prefix), &normed);

        let q = self.gpu_linear(&normed, &format!("{}.attn1.to_q", prefix), b * seq_len, hidden_dim, hidden_dim, false, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 02 self_q", prefix), &q);
        let k = self.gpu_linear(&normed, &format!("{}.attn1.to_k", prefix), b * seq_len, hidden_dim, hidden_dim, false, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 03 self_k", prefix), &k);
        let v = self.gpu_linear(&normed, &format!("{}.attn1.to_v", prefix), b * seq_len, hidden_dim, hidden_dim, false, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 04 self_v", prefix), &v);

        // Self-attention: Q,K,V all have same seq_len — use tiled (Flash) attention
        let self_attn_out = self.gpu_attention_tiled(&q, &k, &v, b, seq_len, num_heads, head_dim, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 05 self_attn_out", prefix), &self_attn_out);

        let self_attn = self.gpu_linear(&self_attn_out, &format!("{}.attn1.to_out.0", prefix), b * seq_len, hidden_dim, hidden_dim, true, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 06 self_attn_proj", prefix), &self_attn);

        // Residual: hidden = hidden + self_attn
        let hidden_buf = self.gpu_add_inplace(&hidden_buf, &self_attn, b * seq_len * hidden_dim, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 07 after_self_res", prefix), &hidden_buf);

        // ============================================================
        // Cross-attention: LayerNorm → Q(img),K,V(text) → attention → out proj → residual
        // ============================================================
        let normed2 = self.gpu_layer_norm(&hidden_buf, prefix, "norm2", b * seq_len, hidden_dim, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 08 norm2", prefix), &normed2);

        let cq = self.gpu_linear(&normed2, &format!("{}.attn2.to_q", prefix), b * seq_len, hidden_dim, hidden_dim, false, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 09 cross_q", prefix), &cq);
        let ck = self.gpu_linear(&ctx_buf, &format!("{}.attn2.to_k", prefix), b_ctx * seq_ctx, dim_ctx, hidden_dim, false, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 10 cross_k", prefix), &ck);
        let cv = self.gpu_linear(&ctx_buf, &format!("{}.attn2.to_v", prefix), b_ctx * seq_ctx, dim_ctx, hidden_dim, false, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 11 cross_v", prefix), &cv);

        // Cross-attention: Q from image (seq_len), K/V from text (seq_ctx) — use strided attention
        let cross_attn_out = self.gpu_attention_cross(&cq, &ck, &cv, b, seq_len, seq_ctx, num_heads, head_dim, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 12 cross_attn_out", prefix), &cross_attn_out);

        let cross_attn = self.gpu_linear(&cross_attn_out, &format!("{}.attn2.to_out.0", prefix), b * seq_len, hidden_dim, hidden_dim, true, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 13 cross_attn_proj", prefix), &cross_attn);

        let hidden_buf = self.gpu_add_inplace(&hidden_buf, &cross_attn, b * seq_len * hidden_dim, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 14 after_cross_res", prefix), &hidden_buf);

        // ============================================================
        // GEGLU Feed-Forward: LayerNorm → linear(→2x) → GEGLU → linear → residual
        // ============================================================
        let normed3 = self.gpu_layer_norm(&hidden_buf, prefix, "norm3", b * seq_len, hidden_dim, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 15 norm3", prefix), &normed3);

        // Determine inner_dim from ff.net.0.proj.bias shape
        let ff1_bias = self.model.get_weight(&format!("{}.ff.net.0.proj.bias", prefix))
            .ok_or_else(|| crate::core::Error::internal(format!("Missing {}.ff.net.0.proj.bias", prefix)))?;
        let doubled_dim = ff1_bias.shape().numel();
        let inner_dim = doubled_dim / 2;

        // Linear → [seq, 2*inner_dim] — the 640→5120 large projection.
        let ff1 = self.gpu_linear(&normed3, &format!("{}.ff.net.0.proj", prefix), b * seq_len, hidden_dim, doubled_dim, true, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 16 ff1_proj", prefix), &ff1);

        // GEGLU: split ff1 into [seq, inner_dim] × 2, apply gelu(gate) * x
        let geglu_out = self.gpu_geglu_split(&ff1, b * seq_len, inner_dim, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 17 geglu_out", prefix), &geglu_out);

        // Linear → [seq, hidden_dim]
        let ff_out = self.gpu_linear(&geglu_out, &format!("{}.ff.net.2", prefix), b * seq_len, inner_dim, hidden_dim, true, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 18 ff_out", prefix), &ff_out);

        let hidden_buf = self.gpu_add_inplace(&hidden_buf, &ff_out, b * seq_len * hidden_dim, compute, command_buffer)?;
        diag_snapshot(&format!("T[{}] 19 after_ff_res", prefix), &hidden_buf);

        // ---- Transpose back: [B*seq, hidden] → NCHW ----
        self.gpu_nhwc_to_nchw(&hidden_buf, input, b, c, seq_len, compute, command_buffer)
    }

    // ================================================================
    // Metal GPU helper methods for BasicTransformerBlock
    // ================================================================

    /// Transpose NCHW → [N*HW, C] (NLC flattened) on GPU.
    #[cfg(feature = "metal")]
    fn gpu_nchw_to_nhwc(
        &self, input: &Tensor, n: usize, c: usize, hw: usize,
        compute: &Arc<MetalCompute>, command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([n * hw, c]), DType::F16, input.device())?;
        let in_ptr = input.device_ptr().ok_or(crate::core::Error::internal("nchw input not on device"))?;
        let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("nchw output not on device"))?;
        let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
        let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };

        let pipeline = compute.compile_pipeline("nchw_to_nhwc_f16", crate::hal::metal::shader::sources::TRANSPOSE, "nchw_to_nhwc_f16")?;

        let tg_x = hw.min(256);
        let grid = ((hw + tg_x - 1) / tg_x, c, n);
        let threadgroup = (tg_x, 1, 1);

        compute.dispatch_async(command_buffer, &pipeline, grid, threadgroup, |encoder| {
            encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
            encoder.set_buffer(1, Some(out_buf.as_ref()), 0);
            encoder.set_bytes(2, 4, &(c as u32) as *const u32 as *const _);
            encoder.set_bytes(3, 4, &(hw as u32) as *const u32 as *const _);
        });

        Ok(output)
    }

    /// Transpose [N*HW, C] → NCHW on GPU. Uses `reference` tensor for output shape.
    #[cfg(feature = "metal")]
    fn gpu_nhwc_to_nchw(
        &self, input: &Tensor, reference: &Tensor, n: usize, c: usize, hw: usize,
        compute: &Arc<MetalCompute>, command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(reference.shape().clone(), DType::F16, reference.device())?;
        let in_ptr = input.device_ptr().ok_or(crate::core::Error::internal("nhwc input not on device"))?;
        let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("nhwc output not on device"))?;
        let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
        let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };

        let pipeline = compute.compile_pipeline("nhwc_to_nchw_f16", crate::hal::metal::shader::sources::TRANSPOSE, "nhwc_to_nchw_f16")?;

        let tg_x = hw.min(256);
        let grid = ((hw + tg_x - 1) / tg_x, c, n);
        let threadgroup = (tg_x, 1, 1);

        compute.dispatch_async(command_buffer, &pipeline, grid, threadgroup, |encoder| {
            encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
            encoder.set_buffer(1, Some(out_buf.as_ref()), 0);
            encoder.set_bytes(2, 4, &(c as u32) as *const u32 as *const _);
            encoder.set_bytes(3, 4, &(hw as u32) as *const u32 as *const _);
        });

        Ok(output)
    }

    /// GPU LayerNorm: reads weight/bias from model, dispatches layer_norm_f16 kernel.
    #[cfg(feature = "metal")]
    fn gpu_layer_norm(
        &self, input: &Tensor, prefix: &str, norm_name: &str,
        batch_size: usize, hidden_dim: usize,
        compute: &Arc<MetalCompute>, command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(input.shape().clone(), DType::F16, input.device())?;

        let w_name = format!("{}.{}.weight", prefix, norm_name);
        let b_name = format!("{}.{}.bias", prefix, norm_name);
        let weight = self.model.get_weight(&w_name)
            .ok_or_else(|| crate::core::Error::internal(format!("Missing {}", w_name)))?;
        let bias = self.model.get_weight(&b_name)
            .ok_or_else(|| crate::core::Error::internal(format!("Missing {}", b_name)))?;

        let in_ptr = input.device_ptr().ok_or(crate::core::Error::internal("ln input not on device"))?;
        let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("ln output not on device"))?;
        let w_ptr = weight.device_ptr().ok_or(crate::core::Error::internal("ln weight not on device"))?;
        let b_ptr = bias.device_ptr().ok_or(crate::core::Error::internal("ln bias not on device"))?;

        let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
        let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };
        let w_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(w_ptr) };
        let b_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(b_ptr) };

        let pipeline = compute.compile_pipeline("layer_norm_f16", crate::hal::metal::shader::sources::LAYER_NORM, "layer_norm_f16")?;
        let eps: f32 = 1e-5;

        let grid = ((batch_size + 255) / 256, 1, 1);
        let threadgroup = (256.min(batch_size), 1, 1);

        compute.dispatch_async(command_buffer, &pipeline, grid, threadgroup, |encoder| {
            encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
            encoder.set_buffer(1, Some(w_buf.as_ref()), 0);
            encoder.set_buffer(2, Some(b_buf.as_ref()), 0);
            encoder.set_buffer(3, Some(out_buf.as_ref()), 0);
            encoder.set_bytes(4, 4, &(batch_size as u32) as *const u32 as *const _);
            encoder.set_bytes(5, 4, &(hidden_dim as u32) as *const u32 as *const _);
            encoder.set_bytes(6, 4, &eps as *const f32 as *const _);
        });

        Ok(output)
    }

    /// GPU linear projection: Y = X @ W^T + bias.
    /// Loads weight (and optional bias) from model by `proj_prefix.weight` / `proj_prefix.bias`.
    #[cfg(feature = "metal")]
    fn gpu_linear(
        &self, input: &Tensor, proj_prefix: &str,
        m: usize, k: usize, n: usize, with_bias: bool,
        compute: &Arc<MetalCompute>, command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([m, n]), DType::F16, input.device())?;

        let w_name = format!("{}.weight", proj_prefix);
        let weight = self.model.get_weight(&w_name)
            .ok_or_else(|| crate::core::Error::internal(format!("Missing {}", w_name)))?;

        let in_ptr = input.device_ptr().ok_or(crate::core::Error::internal("linear input not on device"))?;
        let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("linear output not on device"))?;
        let w_ptr = weight.device_ptr().ok_or(crate::core::Error::internal("linear weight not on device"))?;

        let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
        let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };
        let w_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(w_ptr) };

        // Bias: use weight buffer as placeholder when no bias (kernel checks has_bias flag)
        let bias_buf = if with_bias {
            let b_name = format!("{}.bias", proj_prefix);
            let bias = self.model.get_weight(&b_name)
                .ok_or_else(|| crate::core::Error::internal(format!("Missing {}", b_name)))?;
            let b_ptr = bias.device_ptr().ok_or(crate::core::Error::internal("linear bias not on device"))?;
            unsafe { BorrowedMetalBuffer::from_device_ptr(b_ptr) }
        } else {
            unsafe { BorrowedMetalBuffer::from_device_ptr(w_ptr) } // placeholder
        };

        let pipeline = compute.compile_pipeline("linear_f16", crate::hal::metal::shader::sources::LINEAR, "linear_f16")?;
        let has_bias: u32 = if with_bias { 1 } else { 0 };

        let tile = 16usize;
        let grid = ((n + tile - 1) / tile, (m + tile - 1) / tile, 1);
        let threadgroup = (tile, tile, 1);

        compute.dispatch_async(command_buffer, &pipeline, grid, threadgroup, |encoder| {
            encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
            encoder.set_buffer(1, Some(w_buf.as_ref()), 0);
            encoder.set_buffer(2, Some(bias_buf.as_ref()), 0);
            encoder.set_buffer(3, Some(out_buf.as_ref()), 0);
            encoder.set_bytes(4, 4, &(m as u32) as *const u32 as *const _);
            encoder.set_bytes(5, 4, &(n as u32) as *const u32 as *const _);
            encoder.set_bytes(6, 4, &(k as u32) as *const u32 as *const _);
            encoder.set_bytes(7, 4, &has_bias as *const u32 as *const _);
        });

        Ok(output)
    }

    /// GPU tiled attention (Flash Attention style) for self-attention (same Q/KV length).
    ///
    /// Q, K, V: [seq_len, hidden_dim] in interleaved head layout.
    /// Uses custom strides so the tiled kernel works on [seq, num_heads*head_dim] directly.
    #[cfg(feature = "metal")]
    fn gpu_attention_tiled(
        &self, q: &Tensor, k: &Tensor, v: &Tensor,
        batch: usize, seq_len: usize, num_heads: usize, head_dim: usize,
        compute: &Arc<MetalCompute>, command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        let hidden_dim = num_heads * head_dim;
        let output = Tensor::empty(Shape::from([batch * seq_len, hidden_dim]), DType::F16, q.device())?;

        let q_ptr = q.device_ptr().ok_or(crate::core::Error::internal("attn Q not on device"))?;
        let k_ptr = k.device_ptr().ok_or(crate::core::Error::internal("attn K not on device"))?;
        let v_ptr = v.device_ptr().ok_or(crate::core::Error::internal("attn V not on device"))?;
        let o_ptr = output.device_ptr().ok_or(crate::core::Error::internal("attn O not on device"))?;

        let q_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(q_ptr) };
        let k_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(k_ptr) };
        let v_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(v_ptr) };
        let o_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(o_ptr) };

        // MFA 2.0 dispatch tiers:
        //   default (no env)        : Stage 2 — simdgroup_matrix + vec4 loads (BQ=8 per threadgroup)
        //   MFA_V3=1                : Stage 3 — 2-simdgroup variant (BQT=16 per threadgroup, BK=16)
        //   MFA_V2=0                : Legacy per-thread serial kernel (rollback path)
        let use_mfa_v3 = std::env::var("MFA_V3").ok().as_deref() == Some("1");
        let use_mfa_v2 = std::env::var("MFA_V2").ok().as_deref() != Some("0");
        let pipeline = if use_mfa_v3 {
            compute.compile_pipeline(
                "attention_simdmm2_f16",
                crate::hal::metal::shader::sources::ATTENTION_SIMDMM2,
                "attention_simdmm2_f16",
            )?
        } else if use_mfa_v2 {
            compute.compile_pipeline(
                "attention_simdmm_f16",
                crate::hal::metal::shader::sources::ATTENTION_SIMDMM,
                "attention_simdmm_f16",
            )?
        } else {
            compute.compile_pipeline(
                "attention_tiled_f16",
                crate::hal::metal::shader::sources::ATTENTION_TILED,
                "attention_tiled_f16",
            )?
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        // Custom strides for interleaved [seq, hidden] layout:
        let stride_dim: u32 = 1;
        let stride_head: u32 = head_dim as u32;
        let stride_seq: u32 = hidden_dim as u32;
        let stride_batch: u32 = (seq_len * hidden_dim) as u32;

        // BLOCK = queries per threadgroup. v3 (2 simdgroups) does 16; v2 (1
        // simdgroup) does 8; baseline does 32 (1 query per thread).
        let block_size: usize = if use_mfa_v3 { 16 } else if use_mfa_v2 { 8 } else { 32 };
        // Grid Z = batch so the kernel's `gid.z` indexes each CFG branch's
        // (uncond, cond) sub-sequence independently. Was hardcoded to 1,
        // which left batch≥2's output region uninitialised and produced
        // garbage for everything past row `seq_len` in the next gpu_linear.
        let grid = (num_heads, (seq_len + block_size - 1) / block_size, batch);
        // Threadgroup = 64 (2 simdgroups) for v3, 32 for v2/baseline.
        let threadgroup = (if use_mfa_v3 { 64usize } else { 32usize }, 1, 1);
        // Shared memory budget:
        //   v3 (2 sg)    : 2×(BQ_SG×D×2) + BK×D×2 + BK×D×2 + 2×(BQ_SG×BK×2)
        //                 + 2×(BQ_SG×BK×4) + 2×(BQ_SG×D×4) + 3×2×BQ_SG×4
        //   v2 (1 sg)    : BQ×D×2 + 2×BK×D×2 + BQ×BK×2 + BQ×BK×4 + BQ×D×4 + 3×BQ×4
        //   baseline     : 3 × BLOCK × head_dim × 2B
        let shared_mem_size: u64 = if use_mfa_v3 {
            let bq_sg = 8u64;
            let nsg = 2u64;
            let bk = 16u64;
            let d = head_dim as u64;
            nsg * bq_sg * d * 2          // q_tile (per sg)
            + bk * d * 2                 // k_tile (shared)
            + bk * d * 2                 // v_tile (shared)
            + nsg * bq_sg * bk * 2       // p_tile (per sg, half)
            + nsg * bq_sg * bk * 4       // s_scratch (per sg, float)
            + nsg * bq_sg * d * 4        // o_shared (per sg, float)
            + 3 * nsg * bq_sg * 4        // m/l/alpha (per sg)
        } else if use_mfa_v2 {
            let bq = 8u64;
            let bk = 32u64;
            let d = head_dim as u64;
            bq * d * 2 + bk * d * 2 + bk * d * 2 + bq * bk * 2 + bq * bk * 4 + bq * d * 4 + 3 * bq * 4
        } else {
            (3 * block_size * head_dim * 2) as u64
        };

        compute.dispatch_async(command_buffer, &pipeline, grid, threadgroup, |encoder| {
            encoder.set_buffer(0, Some(q_buf.as_ref()), 0);
            encoder.set_buffer(1, Some(k_buf.as_ref()), 0);
            encoder.set_buffer(2, Some(v_buf.as_ref()), 0);
            encoder.set_buffer(3, Some(o_buf.as_ref()), 0);
            encoder.set_bytes(4, 4, &(seq_len as u32) as *const u32 as *const _);
            encoder.set_bytes(5, 4, &(head_dim as u32) as *const u32 as *const _);
            encoder.set_bytes(6, 4, &scale as *const f32 as *const _);
            encoder.set_bytes(7, 4, &(num_heads as u32) as *const u32 as *const _);
            encoder.set_bytes(8, 4, &stride_batch as *const u32 as *const _);
            encoder.set_bytes(9, 4, &stride_head as *const u32 as *const _);
            encoder.set_bytes(10, 4, &stride_seq as *const u32 as *const _);
            encoder.set_bytes(11, 4, &stride_dim as *const u32 as *const _);
            encoder.set_threadgroup_memory_length(0, shared_mem_size);
        });

        Ok(output)
    }

    /// GPU cross-attention for different Q and K/V lengths.
    ///
    /// Q: [seq_q, hidden], K/V: [seq_k, hidden] in interleaved head layout.
    /// Uses the strided attention kernel that supports separate kv_len.
    #[cfg(feature = "metal")]
    fn gpu_attention_cross(
        &self, q: &Tensor, k: &Tensor, v: &Tensor,
        batch: usize, seq_q: usize, seq_k: usize, num_heads: usize, head_dim: usize,
        compute: &Arc<MetalCompute>, command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        let hidden_dim = num_heads * head_dim;
        let output = Tensor::empty(Shape::from([batch * seq_q, hidden_dim]), DType::F16, q.device())?;

        let q_ptr = q.device_ptr().ok_or(crate::core::Error::internal("xattn Q not on device"))?;
        let k_ptr = k.device_ptr().ok_or(crate::core::Error::internal("xattn K not on device"))?;
        let v_ptr = v.device_ptr().ok_or(crate::core::Error::internal("xattn V not on device"))?;
        let o_ptr = output.device_ptr().ok_or(crate::core::Error::internal("xattn O not on device"))?;

        let q_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(q_ptr) };
        let k_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(k_ptr) };
        let v_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(v_ptr) };
        let o_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(o_ptr) };

        let pipeline = compute.compile_pipeline("attention_f16", crate::hal::metal::shader::sources::ATTENTION, "attention_f16")?;

        let scale = 1.0 / (head_dim as f32).sqrt();
        // Custom strides for interleaved [seq, hidden] layout. Q lives in a
        // [batch*seq_q, hidden] buffer; K/V live in [batch*seq_k, hidden].
        // The kernel uses a single `stride_batch`, so dispatching with grid
        // Z=batch would mis-stride either Q or K/V (whichever isn't
        // matched). Solve by dispatching once per batch with the Q/K/V/O
        // pointers offset to that batch's start; each per-batch dispatch
        // then runs the kernel as if batch=1.
        let stride_dim: u32 = 1;
        let stride_head: u32 = head_dim as u32;
        let stride_seq: u32 = hidden_dim as u32;
        let stride_batch: u32 = 0; // single-batch dispatch — unused
        let threadgroup = (1, 1, 1);
        let shared_mem_size = (seq_k * 4) as u64;
        let dtype_bytes = 2usize; // f16
        let q_step_bytes = (seq_q * hidden_dim * dtype_bytes) as u64;
        let kv_step_bytes = (seq_k * hidden_dim * dtype_bytes) as u64;

        for bi in 0..batch {
            let q_off = bi as u64 * q_step_bytes;
            let kv_off = bi as u64 * kv_step_bytes;
            let o_off = bi as u64 * q_step_bytes;
            let grid = (num_heads, seq_q, 1);

            compute.dispatch_async(command_buffer, &pipeline, grid, threadgroup, |encoder| {
                encoder.set_buffer(0, Some(q_buf.as_ref()), q_off);
                encoder.set_buffer(1, Some(k_buf.as_ref()), kv_off);
                encoder.set_buffer(2, Some(v_buf.as_ref()), kv_off);
                encoder.set_buffer(3, Some(o_buf.as_ref()), o_off);
                encoder.set_bytes(4, 4, &(seq_q as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &(head_dim as u32) as *const u32 as *const _);
                encoder.set_bytes(6, 4, &scale as *const f32 as *const _);
                encoder.set_bytes(7, 4, &(num_heads as u32) as *const u32 as *const _);
                encoder.set_bytes(8, 4, &stride_batch as *const u32 as *const _);
                encoder.set_bytes(9, 4, &stride_head as *const u32 as *const _);
                encoder.set_bytes(10, 4, &stride_seq as *const u32 as *const _);
                encoder.set_bytes(11, 4, &stride_dim as *const u32 as *const _);
                encoder.set_bytes(12, 4, &(seq_k as u32) as *const u32 as *const _);
                encoder.set_threadgroup_memory_length(0, shared_mem_size);
            });
        }

        Ok(output)
    }

    /// GPU GEGLU: split [seq, 2*inner_dim] into x and gate, compute gelu(gate) * x.
    ///
    /// Input: [seq, 2*inner_dim], Output: [seq, inner_dim].
    /// The first half is x, the second half is the gate.
    #[cfg(feature = "metal")]
    fn gpu_geglu_split(
        &self, input: &Tensor, seq_len: usize, inner_dim: usize,
        compute: &Arc<MetalCompute>, command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([seq_len, inner_dim]), DType::F16, input.device())?;

        let in_ptr = input.device_ptr().ok_or(crate::core::Error::internal("geglu input not on device"))?;
        let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("geglu output not on device"))?;
        let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
        let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };

        // The geglu_f16 kernel takes separate gate and up buffers.
        // Our input is [seq, 2*inner_dim] with first half = x, second half = gate.
        // We can point into the same buffer with offsets.
        let doubled_dim = inner_dim * 2;
        // x starts at offset 0, gate starts at offset inner_dim per row.
        // But geglu_f16 expects flat buffers (element-wise gelu(gate[i]) * up[i]).
        // Our data is interleaved per row, so we need a kernel that handles the split.
        // Write a small split-GEGLU kernel inline.
        let geglu_split_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void geglu_split_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant uint& inner_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= seq_len * inner_dim) return;

    uint s = gid / inner_dim;
    uint i = gid % inner_dim;
    uint doubled = inner_dim * 2;

    float x = float(input[s * doubled + i]);
    float gate = float(input[s * doubled + inner_dim + i]);

    // GELU tanh approximation. inner = 0.798 * (gate + 0.045 * gate^3).
    // For gate ≈ 16 (which occurs in some transformer blocks at t=999),
    // gate^3 ≈ 4400 → inner ≈ 170. Metal's `tanh` computes via the
    // expansion `(e^x - e^-x) / (e^x + e^-x)`; `exp(170)` overflows fp32
    // (limit ≈ exp(88)), giving `inf / inf = NaN`. Clamping `inner` to a
    // value well past saturation (|inner| ≥ ~20 gives `tanh ≈ ±1` to fp32
    // precision) avoids the overflow without changing the math.
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float g3 = gate * gate * gate;
    float inner = sqrt_2_over_pi * (gate + coeff * g3);
    inner = clamp(inner, -20.0f, 20.0f);
    float gelu_gate = 0.5f * gate * (1.0f + tanh(inner));

    output[gid] = half(x * gelu_gate);
}
"#;

        let pipeline = compute.compile_pipeline("geglu_split_f16", geglu_split_source, "geglu_split_f16")?;
        let total = seq_len * inner_dim;
        let grid = ((total + 255) / 256, 1, 1);
        let threadgroup = (256, 1, 1);

        compute.dispatch_async(command_buffer, &pipeline, grid, threadgroup, |encoder| {
            encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
            encoder.set_buffer(1, Some(out_buf.as_ref()), 0);
            encoder.set_bytes(2, 4, &(seq_len as u32) as *const u32 as *const _);
            encoder.set_bytes(3, 4, &(inner_dim as u32) as *const u32 as *const _);
        });

        Ok(output)
    }

    /// GPU elementwise add, returning a new tensor: output = a + b.
    #[cfg(feature = "metal")]
    /// GPU concatenation along channel dim for `[N, C, H, W]` tensors. Runs
    /// on the caller's command buffer so writes are correctly ordered with
    /// the producer GPU dispatches. `Tensor::cat` previously fell back to a
    /// CPU host-roundtrip path that read producer buffers before commit and
    /// silently returned zeros.
    #[cfg(feature = "metal")]
    fn gpu_cat_dim1(
        &self, a: &Tensor, b: &Tensor,
        compute: &Arc<MetalCompute>, command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        let (an, ca, ah, aw) = a.shape().dims4()
            .ok_or_else(|| crate::core::Error::internal("gpu_cat_dim1 a not 4D"))?;
        let (bn, cb_, bh, bw) = b.shape().dims4()
            .ok_or_else(|| crate::core::Error::internal("gpu_cat_dim1 b not 4D"))?;
        if an != bn || ah != bh || aw != bw {
            return Err(crate::core::Error::internal(format!(
                "gpu_cat_dim1 shape mismatch: a=[{},{},{},{}] b=[{},{},{},{}]",
                an, ca, ah, aw, bn, cb_, bh, bw,
            )));
        }
        let n = an;
        let hw = ah * aw;
        let cout = ca + cb_;
        let output = Tensor::empty(Shape::from([n, cout, ah, aw]), DType::F16, a.device())?;

        let a_ptr = a.device_ptr().ok_or(crate::core::Error::internal("cat a not on device"))?;
        let b_ptr = b.device_ptr().ok_or(crate::core::Error::internal("cat b not on device"))?;
        let o_ptr = output.device_ptr().ok_or(crate::core::Error::internal("cat out not on device"))?;
        let a_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(a_ptr) };
        let b_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(b_ptr) };
        let o_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(o_ptr) };

        let pipeline = compute.compile_pipeline(
            "cat_nchw_dim1_f16",
            crate::hal::metal::shader::sources::CAT_NCHW,
            "cat_nchw_dim1_f16",
        )?;

        let tg_x = hw.min(64);
        let grid = ((hw + tg_x - 1) / tg_x, cout, n);
        let threadgroup = (tg_x, 1, 1);
        let c_n = n as u32;
        let c_ca = ca as u32;
        let c_cb = cb_ as u32;
        let c_hw = hw as u32;

        compute.dispatch_async(command_buffer, &pipeline, grid, threadgroup, |encoder| {
            encoder.set_buffer(0, Some(a_buf.as_ref()), 0);
            encoder.set_buffer(1, Some(b_buf.as_ref()), 0);
            encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
            encoder.set_bytes(3, 4, &c_n as *const u32 as *const _);
            encoder.set_bytes(4, 4, &c_ca as *const u32 as *const _);
            encoder.set_bytes(5, 4, &c_cb as *const u32 as *const _);
            encoder.set_bytes(6, 4, &c_hw as *const u32 as *const _);
        });

        Ok(output)
    }

    #[cfg(feature = "metal")]
    fn gpu_add_inplace(
        &self, a: &Tensor, b: &Tensor, numel: usize,
        compute: &Arc<MetalCompute>, command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(a.shape().clone(), DType::F16, a.device())?;
        let a_ptr = a.device_ptr().ok_or(crate::core::Error::internal("add a not on device"))?;
        let b_ptr = b.device_ptr().ok_or(crate::core::Error::internal("add b not on device"))?;
        let o_ptr = output.device_ptr().ok_or(crate::core::Error::internal("add out not on device"))?;

        let a_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(a_ptr) };
        let b_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(b_ptr) };
        let o_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(o_ptr) };

        let pipeline = compute.compile_pipeline("add_f16", crate::hal::metal::shader::sources::ELEMENTWISE, "add_f16")?;
        let grid = ((numel + 255) / 256, 1, 1);
        let threadgroup = (256, 1, 1);

        compute.dispatch_async(command_buffer, &pipeline, grid, threadgroup, |encoder| {
            encoder.set_buffer(0, Some(a_buf.as_ref()), 0);
            encoder.set_buffer(1, Some(b_buf.as_ref()), 0);
            encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
        });

        Ok(output)
    }

    /// Attention block: projects input to Q, context to K/V, computes scaled dot-product attention.
    #[cfg(feature = "metal")]
    fn attention_block(
        &self,
        input: &Tensor,
        context: &Tensor,
        prefix: &str,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        // Project Q from input, K/V from context
        let q = self.conv2d(input, &format!("{}.to_q.weight", prefix), None, compute, command_buffer, 1, 0)?;
        let k = self.conv2d(context, &format!("{}.to_k.weight", prefix), None, compute, command_buffer, 1, 0)?;
        let v = self.conv2d(context, &format!("{}.to_v.weight", prefix), None, compute, command_buffer, 1, 0)?;

        // Scaled dot-product attention (Q @ K^T / sqrt(d_k)) @ V
        // For now, return the value projection as a simplified path
        // Full implementation would use a dedicated attention Metal kernel
        let out = self.conv2d(&v, &format!("{}.to_out.0.weight", prefix), Some(&format!("{}.to_out.0.bias", prefix)), compute, command_buffer, 1, 0)?;
        Ok(out)
    }

    #[cfg(feature = "metal")]
    pub fn downsample(
        &self,
        input: &Tensor,
        prefix: &str,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        // Conv2d with stride 2
        self.conv2d(input, &format!("{}.conv.weight", prefix), Some(&format!("{}.conv.bias", prefix)), compute, command_buffer, 2, 1)
    }

    #[cfg(feature = "metal")]
    fn upsample(
        &self,
        input: &Tensor,
        prefix: &str,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        // HF Upsample2D for SD 1.5 is nearest-neighbor 2x FOLLOWED BY a
        // learned 3x3 conv (`<prefix>.conv.weight/bias`). Skipping the conv
        // produces tiled copies whose feature vectors are uncorrelated with
        // the trained conv's mixed-neighbor output — up_blocks 0/1/2 land
        // at cos≈0 vs PT; up_block_3 is fine because it has no upsample.
        let (n, c, h, w) = input.shape().dims4().unwrap_or((1, input.shape().dim(1).unwrap_or(1), input.shape().dim(2).unwrap_or(32), input.shape().dim(3).unwrap_or(32)));
        
        let new_h = h * 2;
        let new_w = w * 2;
        
        // Output tensor
        let output = Tensor::empty(Shape::from([n, c, new_h, new_w]), DType::F16, input.device())?;
        
        let input_ptr = input.device_ptr().ok_or(crate::core::Error::internal("Input tensor not on device"))?;
        let output_ptr = output.device_ptr().ok_or(crate::core::Error::internal("Output tensor not on device"))?;

        // SAFETY: input/output tensors own the underlying Metal buffers and outlive
        // the BorrowedMetalBuffer wrappers and the GPU command buffer submission.
        let input_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(input_ptr) };
        let output_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(output_ptr) };

        let pipeline = compute.compile_pipeline("upsample_nearest", crate::hal::metal::shader::sources::UPSAMPLE, "upsample_nearest_f16")?;

        let threadgroup_size = (8, 8, 1);
        let grid_size = (
            (new_w + threadgroup_size.0 - 1) / threadgroup_size.0,
            (new_h + threadgroup_size.1 - 1) / threadgroup_size.1,
            n * c
        );

        compute.dispatch_async(
            command_buffer,
            &pipeline,
            grid_size,
            threadgroup_size,
            |encoder| {
                encoder.set_buffer(0, Some(input_buffer.as_ref()), 0);
                encoder.set_buffer(1, Some(output_buffer.as_ref()), 0);

                let c_n = n as u32;
                let c_c = c as u32;
                let c_hin = h as u32;
                let c_win = w as u32;

                encoder.set_bytes(2, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(3, 4, &c_c as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_hin as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_win as *const u32 as *const _);
            }
        );

        // Apply the learned 3x3 conv that HF Upsample2D applies after nearest.
        let out_conv = self.conv2d(
            &output,
            &format!("{}.conv.weight", prefix),
            Some(&format!("{}.conv.bias", prefix)),
            compute,
            command_buffer,
            1, // stride
            1, // padding (3x3 same)
        )?;
        Ok(out_conv)
    }

    #[cfg(feature = "metal")]
    fn group_norm(
        &self,
        input: &Tensor,
        name: &str,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        // GroupNorm(32, x)
        let output = Tensor::empty(input.shape().clone(), DType::F16, input.device())?;
        
        let input_ptr = input.device_ptr().ok_or(crate::core::Error::internal("Input tensor not on device"))?;
        let output_ptr = output.device_ptr().ok_or(crate::core::Error::internal("Output tensor not on device"))?;

        // Retrieve weights and bias
        let weight_opt = self.model.get_weight(&format!("{}.weight", name));
        let bias_opt = self.model.get_weight(&format!("{}.bias", name));

        // Dimensions
        let (n, c, h, w) = input.shape().dims4().unwrap_or((1, input.shape().dim(1).unwrap_or(1), input.shape().dim(2).unwrap_or(32), input.shape().dim(3).unwrap_or(32)));
        
        // Handle dummy weights
        let (dummy_weight, dummy_bias) = if self.model.info().name == "dummy-model" && weight_opt.is_none() {
            let w_name = format!("{}.weight", name);
            let b_name = format!("{}.bias", name);

            let (cached_w, cached_b) = {
                let cache = self.dummy_cache.lock().map_err(|_| crate::core::Error::internal("dummy_cache mutex poisoned"))?;
                (cache.get(&w_name).cloned(), cache.get(&b_name).cloned())
            };

            let w = if let Some(w) = cached_w {
                w
            } else {
                let w = Tensor::ones_on(Shape::from([c]), DType::F16, compute.device().info().id)?;
                self.dummy_cache.lock().map_err(|_| crate::core::Error::internal("dummy_cache mutex poisoned"))?.insert(w_name, w.clone());
                w
            };

            let b = if let Some(b) = cached_b {
                b
            } else {
                let b = Tensor::zeros_on(Shape::from([c]), DType::F16, compute.device().info().id)?;
                self.dummy_cache.lock().map_err(|_| crate::core::Error::internal("dummy_cache mutex poisoned"))?.insert(b_name, b.clone());
                b
            };

            (Some(w), Some(b))
        } else {
            (None, None)
        };

        // SAFETY: input/output tensors own the underlying Metal buffers and outlive
        // the BorrowedMetalBuffer wrappers and the GPU command buffer submission.
        let input_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(input_ptr) };
        let output_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(output_ptr) };

        let weight_buffer = if let Some(w) = weight_opt {
             if let Some(ptr) = w.device_ptr() {
                Some(unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
            } else { None }
        } else if let Some(w) = &dummy_weight {
             if let Some(ptr) = w.device_ptr() {
                Some(unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
            } else { None }
        } else { None };

        let bias_buffer = if let Some(b) = bias_opt {
             if let Some(ptr) = b.device_ptr() {
                Some(unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
            } else { None }
        } else if let Some(b) = &dummy_bias {
             if let Some(ptr) = b.device_ptr() {
                Some(unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
            } else { None }
        } else { None };

        let groups = 32;
        let hw = h * w;
        let eps: f32 = 1e-5;

        // Allocate temp stats buffer: [N, Groups] of float2 (8 bytes)
        let stats_size = (n * groups * 8) as u64;
        let stats_buffer = compute.device().create_buffer(stats_size as usize, crate::hal::metal::ResourceOptions::activations())?;
        
        let stats_pipeline = compute.compile_pipeline("group_norm_stats", crate::hal::metal::shader::sources::GROUP_NORM, "group_norm_stats_f16")?;
        let apply_pipeline = compute.compile_pipeline("group_norm_apply", crate::hal::metal::shader::sources::GROUP_NORM, "group_norm_apply_f16")?;
        
        // 1. Stats Kernel
        // Grid: (Groups, N, 1) threadgroups
        compute.dispatch_async(
            command_buffer,
            &stats_pipeline,
            (groups, n, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input_buffer.as_ref()), 0);
                encoder.set_buffer(1, Some(&stats_buffer), 0);

                let c_n = n as u32;
                let c_g = groups as u32;
                let c_c = c as u32;
                let c_hw = hw as u32;

                encoder.set_bytes(2, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(3, 4, &c_g as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_c as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_hw as *const u32 as *const _);
            }
        );

        // 2. Apply Kernel
        // Grid: (ceil(HW/256), C, N) threadgroups
        let hw_groups = (hw + 255) / 256;
        compute.dispatch_async(
            command_buffer,
            &apply_pipeline,
            (hw_groups, c, n),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input_buffer.as_ref()), 0);
                encoder.set_buffer(1, Some(&stats_buffer), 0);

                if let Some(w) = &weight_buffer {
                    encoder.set_buffer(2, Some(w.as_ref()), 0);
                } else {
                    encoder.set_buffer(2, None, 0);
                }
                if let Some(b) = &bias_buffer {
                    encoder.set_buffer(3, Some(b.as_ref()), 0);
                } else {
                    encoder.set_buffer(3, None, 0);
                }
                encoder.set_buffer(4, Some(output_buffer.as_ref()), 0);
                
                let c_n = n as u32;
                let c_g = groups as u32;
                let c_c = c as u32;
                let c_hw = hw as u32;
                
                encoder.set_bytes(5, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(6, 4, &c_g as *const u32 as *const _);
                encoder.set_bytes(7, 4, &c_c as *const u32 as *const _);
                encoder.set_bytes(8, 4, &c_hw as *const u32 as *const _);
                encoder.set_bytes(9, 4, &eps as *const f32 as *const _);
            }
        );
        
        Ok(output)
    }

    /// Fused GroupNorm + SiLU: single kernel dispatch instead of two.
    /// Saves ~96 dispatches per SDXL generation (2 per ResNet block × 12 blocks × 4 steps).
    #[cfg(feature = "metal")]
    pub fn group_norm_silu(
        &self,
        input: &Tensor,
        name: &str,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(input.shape().clone(), DType::F16, input.device())?;

        let input_ptr = input.device_ptr().ok_or(crate::core::Error::internal("Input tensor not on device"))?;
        let output_ptr = output.device_ptr().ok_or(crate::core::Error::internal("Output tensor not on device"))?;

        let weight_opt = self.model.get_weight(&format!("{}.weight", name));
        let bias_opt = self.model.get_weight(&format!("{}.bias", name));

        let (n, c, h, w) = input.shape().dims4().unwrap_or((1, input.shape().dim(1).unwrap_or(1), input.shape().dim(2).unwrap_or(32), input.shape().dim(3).unwrap_or(32)));

        let (dummy_weight, dummy_bias) = if self.model.info().name == "dummy-model" && weight_opt.is_none() {
            let w_name = format!("{}.weight", name);
            let b_name = format!("{}.bias", name);

            let (cached_w, cached_b) = {
                let cache = self.dummy_cache.lock().map_err(|_| crate::core::Error::internal("dummy_cache mutex poisoned"))?;
                (cache.get(&w_name).cloned(), cache.get(&b_name).cloned())
            };

            let w = if let Some(w) = cached_w {
                w
            } else {
                let w = Tensor::ones_on(Shape::from([c]), DType::F16, compute.device().info().id)?;
                self.dummy_cache.lock().map_err(|_| crate::core::Error::internal("dummy_cache mutex poisoned"))?.insert(w_name, w.clone());
                w
            };

            let b = if let Some(b) = cached_b {
                b
            } else {
                let b = Tensor::zeros_on(Shape::from([c]), DType::F16, compute.device().info().id)?;
                self.dummy_cache.lock().map_err(|_| crate::core::Error::internal("dummy_cache mutex poisoned"))?.insert(b_name, b.clone());
                b
            };

            (Some(w), Some(b))
        } else {
            (None, None)
        };

        let input_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(input_ptr) };
        let output_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(output_ptr) };

        let weight_buffer = if let Some(w) = weight_opt {
             if let Some(ptr) = w.device_ptr() {
                Some(unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
            } else { None }
        } else if let Some(w) = &dummy_weight {
             if let Some(ptr) = w.device_ptr() {
                Some(unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
            } else { None }
        } else { None };

        let bias_buffer = if let Some(b) = bias_opt {
             if let Some(ptr) = b.device_ptr() {
                Some(unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
            } else { None }
        } else if let Some(b) = &dummy_bias {
             if let Some(ptr) = b.device_ptr() {
                Some(unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
            } else { None }
        } else { None };

        let groups = 32;
        let hw = h * w;
        let eps: f32 = 1e-5;

        // Allocate temp stats buffer: [N, Groups] of float2 (8 bytes)
        let stats_size = (n * groups * 8) as u64;
        let stats_buffer = compute.device().create_buffer(stats_size as usize, crate::hal::metal::ResourceOptions::activations())?;

        let stats_pipeline = compute.compile_pipeline("group_norm_stats", crate::hal::metal::shader::sources::GROUP_NORM, "group_norm_stats_f16")?;
        let apply_pipeline = compute.compile_pipeline("group_norm_silu_apply", crate::hal::metal::shader::sources::GROUP_NORM, "group_norm_silu_apply_f16")?;

        // 1. Stats Kernel (same as group_norm)
        compute.dispatch_async(
            command_buffer,
            &stats_pipeline,
            (groups, n, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input_buffer.as_ref()), 0);
                encoder.set_buffer(1, Some(&stats_buffer), 0);

                let c_n = n as u32;
                let c_g = groups as u32;
                let c_c = c as u32;
                let c_hw = hw as u32;

                encoder.set_bytes(2, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(3, 4, &c_g as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_c as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_hw as *const u32 as *const _);
            }
        );

        // 2. Fused Apply + SiLU Kernel
        let hw_groups = (hw + 255) / 256;
        compute.dispatch_async(
            command_buffer,
            &apply_pipeline,
            (hw_groups, c, n),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input_buffer.as_ref()), 0);
                encoder.set_buffer(1, Some(&stats_buffer), 0);

                if let Some(w) = &weight_buffer {
                    encoder.set_buffer(2, Some(w.as_ref()), 0);
                } else {
                    encoder.set_buffer(2, None, 0);
                }
                if let Some(b) = &bias_buffer {
                    encoder.set_buffer(3, Some(b.as_ref()), 0);
                } else {
                    encoder.set_buffer(3, None, 0);
                }
                encoder.set_buffer(4, Some(output_buffer.as_ref()), 0);

                let c_n = n as u32;
                let c_g = groups as u32;
                let c_c = c as u32;
                let c_hw = hw as u32;

                encoder.set_bytes(5, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(6, 4, &c_g as *const u32 as *const _);
                encoder.set_bytes(7, 4, &c_c as *const u32 as *const _);
                encoder.set_bytes(8, 4, &c_hw as *const u32 as *const _);
                encoder.set_bytes(9, 4, &eps as *const f32 as *const _);
            }
        );

        Ok(output)
    }

    #[cfg(feature = "metal")]
    pub fn silu(
        &self,
        input: &Tensor,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(input.shape().clone(), DType::F16, input.device())?;
        
        let input_ptr = input.device_ptr().ok_or(crate::core::Error::internal("Input tensor not on device"))?;
        let output_ptr = output.device_ptr().ok_or(crate::core::Error::internal("Output tensor not on device"))?;

        // SAFETY: input/output tensors own the underlying Metal buffers and outlive
        // the BorrowedMetalBuffer wrappers and the GPU command buffer submission.
        let input_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(input_ptr) };
        let output_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(output_ptr) };

        let pipeline = compute.compile_pipeline("silu", crate::hal::metal::shader::sources::SILU, "silu_f16")?;

        let numel = input.shape().numel();
        // Use grid dispatch helper
        let grid_size = ((numel + 255) / 256, 1, 1);
        let threadgroup_size = (256, 1, 1);

        compute.dispatch_async(
            command_buffer,
            &pipeline,
            grid_size,
            threadgroup_size,
            |encoder| {
                encoder.set_buffer(0, Some(input_buffer.as_ref()), 0);
                encoder.set_buffer(1, Some(output_buffer.as_ref()), 0);
            }
        );
        
        Ok(output)
    }

    #[cfg(feature = "metal")]
    pub fn add(
        &self,
        a: &Tensor,
        b: &Tensor,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        // Simple elementwise add
        let output = Tensor::empty(a.shape().clone(), DType::F16, a.device())?;
        
        let a_ptr = a.device_ptr().ok_or(crate::core::Error::internal("A tensor not on device"))?;
        let b_ptr = b.device_ptr().ok_or(crate::core::Error::internal("B tensor not on device"))?;
        let output_ptr = output.device_ptr().ok_or(crate::core::Error::internal("Output tensor not on device"))?;

        // SAFETY: input/output tensors own the underlying Metal buffers and outlive
        // the BorrowedMetalBuffer wrappers and the GPU command buffer submission.
        let a_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(a_ptr) };
        let b_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(b_ptr) };
        let output_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(output_ptr) };

        let pipeline = compute.compile_pipeline("add_f16", crate::hal::metal::shader::sources::ELEMENTWISE, "add_f16")?;

        let numel = a.shape().numel();
        let grid_size = ((numel + 255) / 256, 1, 1);
        let threadgroup_size = (256, 1, 1);

        compute.dispatch_async(
            command_buffer,
            &pipeline,
            grid_size,
            threadgroup_size,
            |encoder| {
                encoder.set_buffer(0, Some(a_buffer.as_ref()), 0);
                encoder.set_buffer(1, Some(b_buffer.as_ref()), 0);
                encoder.set_buffer(2, Some(output_buffer.as_ref()), 0);
            }
        );
        
        Ok(output)
    }

    /// Channel-wise broadcast add: output[n,c,h,w] = input[n,c,h,w] + bias[c].
    /// Adds a [C] vector to every spatial position of a [N,C,H,W] tensor.
    #[cfg(feature = "metal")]
    fn channel_bias_add(
        &self,
        input: &Tensor,
        bias: &Tensor,
        channels: usize,
        spatial: usize,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(input.shape().clone(), DType::F16, input.device())?;

        let in_ptr = input.device_ptr().ok_or(crate::core::Error::internal("input not on device"))?;
        let bias_ptr = bias.device_ptr().ok_or(crate::core::Error::internal("bias not on device"))?;
        let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("output not on device"))?;

        let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
        let bias_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(bias_ptr) };
        let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };

        let pipeline = compute.compile_pipeline("channel_bias_add_f16", crate::hal::metal::shader::sources::ELEMENTWISE, "channel_bias_add_f16")?;

        // Need to dispatch over `batch * channels * spatial`, not just
        // `channels * spatial`. The old code under-dispatched for batch ≥ 2
        // (CFG): the second batch's output region was left UNINITIALISED,
        // because `Tensor::empty` doesn't zero-fill. The kernel modulo logic
        // (`c = gid / spatial`) wraps within one batch's slice and applies
        // bias correctly, since each batch occupies a contiguous
        // `channels * spatial` chunk.
        let n = input.shape().dim(0).unwrap_or(1).max(1);
        let numel = n * channels * spatial;
        let grid_size = ((numel + 255) / 256, 1, 1);
        let threadgroup_size = (256, 1, 1);

        let c_channels = channels as u32;
        let c_spatial = spatial as u32;

        compute.dispatch_async(
            command_buffer,
            &pipeline,
            grid_size,
            threadgroup_size,
            |encoder| {
                encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(bias_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(out_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &c_channels as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_spatial as *const u32 as *const _);
            }
        );

        Ok(output)
    }

    #[cfg(feature = "metal")]
    pub fn conv2d(
        &self,
        input: &Tensor,
        weight_name: &str,
        bias_name: Option<&str>,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
        stride: usize,
        padding: usize,
    ) -> Result<Tensor> {
        // Get dimensions
        let (_n, cin, hin, win) = input.shape().dims4().unwrap_or((1, input.shape().dim(1).unwrap_or(1), input.shape().dim(2).unwrap_or(32), input.shape().dim(3).unwrap_or(32)));

        let weight_opt = self.model.get_weight(weight_name);
        
        // Handle dummy weights or loaded weights
        let (_dummy_weight, weight_ptr, cout, kh, kw) = if self.model.info().name == "dummy-model" && weight_opt.is_none() {
             
             // Check cache
             let cached = {
                 let cache = self.dummy_cache.lock().map_err(|_| crate::core::Error::internal("dummy_cache mutex poisoned"))?;
                 cache.get(weight_name).cloned()
             };

             if let Some(w) = cached {
                 let ptr = w.device_ptr().ok_or(crate::core::Error::internal("Cached weight not on device"))?;
                 let (co, _, k_h, k_w) = w.shape().dims4().unwrap_or((w.shape().dim(0).unwrap_or(1), 1, 3, 3));
                 (Some(w), ptr, co, k_h, k_w)
             } else {
                 let cout = if weight_name.contains("conv_in") {
                     320
                 } else if weight_name.contains("conv_out") {
                     4
                 } else if weight_name.contains("to_q") || weight_name.contains("to_k") || weight_name.contains("to_v") {
                     cin // Attention projections usually preserve dim or split heads
                 } else {
                     cin // Preserve by default for blocks
                 };

                 // Kernel size heuristic
                 // Actually, SD uses 3x3 for almost everything except shortcuts.
                 let k = if stride > 1 || padding > 0 { 3 } else { 1 };
                 // Better heuristic: use padding as hint. If padding=1, likely 3x3.
                 let k = if padding == 1 { 3 } else { 1 };

                 // Create dummy tensor
                 // Use zeros to avoid NaN/denormal slowdowns and faster init
                 let w = Tensor::zeros_on(Shape::from([cout, cin, k, k]), DType::F16, compute.device().info().id)?;

                 // Insert into cache
                 {
                     let mut cache = self.dummy_cache.lock().map_err(|_| crate::core::Error::internal("dummy_cache mutex poisoned"))?;
                     cache.insert(weight_name.to_string(), w.clone());
                 }

                 let ptr = w.device_ptr().ok_or(crate::core::Error::internal("Dummy weight not on device"))?;

                 (Some(w), ptr, cout, k, k)
             }
        } else {
             let w = weight_opt.ok_or_else(|| crate::core::Error::ModelLoad {
                model: "unet".into(),
                message: format!("weight {} not found", weight_name),
                #[cfg(feature = "std")]
                source: None,
            })?;
            let ptr = w.device_ptr().ok_or(crate::core::Error::internal("Weight not on device"))?;
            let (co, _, k_h, k_w) = w.shape().dims4().unwrap_or((w.shape().dim(0).unwrap_or(1), 1, 3, 3));
            (None, ptr, co, k_h, k_w)
        };
            
        let bias_opt = if let Some(name) = bias_name {
            self.model.get_weight(name)
        } else {
            None
        };
        
        let (_dummy_bias, bias_ptr) = if self.model.info().name == "dummy-model" && bias_name.is_some() && bias_opt.is_none() {
             let name = bias_name.unwrap();

             // Check cache
             let cached = {
                 let cache = self.dummy_cache.lock().map_err(|_| crate::core::Error::internal("dummy_cache mutex poisoned"))?;
                 cache.get(name).cloned()
             };

             if let Some(b) = cached {
                 let ptr = b.device_ptr().ok_or(crate::core::Error::internal("Cached bias not on device"))?;
                 (Some(b), Some(ptr))
             } else {
                 let b = Tensor::zeros_on(Shape::from([cout]), DType::F16, compute.device().info().id)?;

                 // Insert into cache
                 {
                     let mut cache = self.dummy_cache.lock().map_err(|_| crate::core::Error::internal("dummy_cache mutex poisoned"))?;
                     cache.insert(name.to_string(), b.clone());
                 }

                 let ptr = b.device_ptr().ok_or(crate::core::Error::internal("Dummy bias not on device"))?;
                 (Some(b), Some(ptr))
             }
        } else if let Some(b) = bias_opt {
             let ptr = b.device_ptr().ok_or(crate::core::Error::internal("Bias not on device"))?;
             (None, Some(ptr))
        } else {
             // No bias requested. Metal kernels' `if (bias)` null check is
             // unreliable; passing a real zero buffer is safer + matches
             // the bias-add semantics (sum += 0 = no-op). Cache by a
             // synthetic key so the per-conv allocation is amortised.
             let synthetic = format!("__zero_bias_{}", cout);
             let cached = {
                 let cache = self.dummy_cache.lock().map_err(|_| crate::core::Error::internal("dummy_cache mutex poisoned"))?;
                 cache.get(&synthetic).cloned()
             };
             if let Some(b) = cached {
                 let ptr = b.device_ptr().ok_or(crate::core::Error::internal("Cached zero bias not on device"))?;
                 (Some(b), Some(ptr))
             } else {
                 let b = Tensor::zeros_on(Shape::from([cout]), DType::F16, compute.device().info().id)?;
                 {
                     let mut cache = self.dummy_cache.lock().map_err(|_| crate::core::Error::internal("dummy_cache mutex poisoned"))?;
                     cache.insert(synthetic, b.clone());
                 }
                 let ptr = b.device_ptr().ok_or(crate::core::Error::internal("Zero bias not on device"))?;
                 (Some(b), Some(ptr))
             }
        };
        
        // Use actual batch size
        let n = input.shape().dim(0).unwrap_or(1);

        let hout = (hin + 2 * padding - kh) / stride + 1;
        let wout = (win + 2 * padding - kw) / stride + 1;

        // Allocate output
        let output_shape = Shape::from([n, cout, hout, wout]);
        let output = Tensor::empty(output_shape, DType::F16, input.device())?;

        // Retrieve buffers
        let input_ptr = input.device_ptr().ok_or(crate::core::Error::internal("Input tensor not on device"))?;
        let output_ptr = output.device_ptr().ok_or(crate::core::Error::internal("Output tensor not on device"))?;
        
        // SAFETY: input/weight/output tensors own the underlying Metal buffers and outlive
        // the BorrowedMetalBuffer wrappers and the GPU command buffer submission.
        let input_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(input_ptr) };
        let weight_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(weight_ptr) };
        let output_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(output_ptr) };
        let bias_buffer = if let Some(ptr) = bias_ptr {
            Some(unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
        } else {
            None
        };

        // Determine kernel
        let hw = hout * wout;
        let use_simd_1x1 = cin % 32 == 0 && cout % 32 == 0 && hw % 32 == 0;
        
        // SD_DIAG=1 escape hatch: force conv2d_naive_f16 for ALL 3×3 calls
        // so we can isolate whether the tiled kernel is producing wrong
        // output magnitudes vs the naive (obviously-correct) one. Tested
        // against PyTorch reference: with the tiled kernel, our `conv_out`
        // applies ~17× less amplification than HF diffusers does on the
        // same weights + same input distribution.
        let force_naive = std::env::var("SD_FORCE_NAIVE_CONV").ok().as_deref() == Some("1");
        let kernel_name = if kh == 1 && kw == 1 {
            if use_simd_1x1 {
                "conv2d_1x1_simd_f16"
            } else {
                "conv2d_1x1_f16"
            }
        } else if !force_naive && kh == 3 && kw == 3 && stride == 1 && padding == 1 {
            "conv2d_3x3_tiled_f16"
        } else {
            "conv2d_naive_f16"
        };
        
        let pipeline = compute.compile_pipeline(kernel_name, crate::hal::metal::shader::sources::CONV2D, kernel_name)?;

        // Dispatch
        let (threadgroup_size, grid_size) = if kernel_name == "conv2d_3x3_tiled_f16" {
            let tg = (16, 16, 1);
            let grid = (
                (wout + 15) / 16,
                (hout + 15) / 16,
                cout * n
            );
            (tg, grid)
        } else if kernel_name == "conv2d_1x1_simd_f16" {
            let tg = (32, 1, 1); // 32 threads (1 simdgroup)
            // Grid: (HW / 8, Cout / 8, Batch)
            // Note: HW is mapped to tile_n (gid.x), Cout to tile_m (gid.y)
            let grid = (
                hw / 8,
                cout / 8,
                n
            );
            (tg, grid)
        } else {
            // Naive or 1x1 basic
            let tg = (8, 8, 1);
            let grid = (
                (wout + tg.0 - 1) / tg.0,
                (hout + tg.1 - 1) / tg.1,
                cout * n
            );
            (tg, grid)
        };

        compute.dispatch_async(
            command_buffer,
            &pipeline,
            grid_size,
            threadgroup_size,
            |encoder| {
                encoder.set_buffer(0, Some(input_buffer.as_ref()), 0);
                encoder.set_buffer(1, Some(weight_buffer.as_ref()), 0);
                if let Some(b) = &bias_buffer {
                    encoder.set_buffer(2, Some(b.as_ref()), 0);
                } else {
                     encoder.set_buffer(2, None, 0);
                }
                encoder.set_buffer(3, Some(output_buffer.as_ref()), 0);
                
                let c_cin = cin as u32;
                let c_hin = hin as u32;
                let c_win = win as u32;
                let c_cout = cout as u32;
                let c_hout = hout as u32;
                let c_wout = wout as u32;
                let c_kw = kw as u32;
                let c_kh = kh as u32;
                let c_pad_x = padding as u32;
                let c_pad_y = padding as u32;
                let c_stride_x = stride as u32;
                let c_stride_y = stride as u32;

                encoder.set_bytes(4, 4, &c_cin as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_hin as *const u32 as *const _);
                encoder.set_bytes(6, 4, &c_win as *const u32 as *const _);
                encoder.set_bytes(7, 4, &c_cout as *const u32 as *const _);
                encoder.set_bytes(8, 4, &c_hout as *const u32 as *const _);
                encoder.set_bytes(9, 4, &c_wout as *const u32 as *const _);
                encoder.set_bytes(10, 4, &c_kw as *const u32 as *const _);
                encoder.set_bytes(11, 4, &c_kh as *const u32 as *const _);
                
                encoder.set_bytes(12, 4, &c_pad_x as *const u32 as *const _);
                encoder.set_bytes(13, 4, &c_pad_y as *const u32 as *const _);
                encoder.set_bytes(14, 4, &c_stride_x as *const u32 as *const _);
                encoder.set_bytes(15, 4, &c_stride_y as *const u32 as *const _);
                
                let c_n = n as u32;
                encoder.set_bytes(16, 4, &c_n as *const u32 as *const _);
            }
        );
        
        Ok(output)
    }
    
    /// Sinusoidal timestep embedding → MLP(linear_1 → SiLU → linear_2).
    ///
    /// Sinusoidal encoding computed on CPU (tiny: 320 values), then MLP runs
    /// entirely on Metal GPU via two linear layers with SiLU activation.
    /// Returns [1, 1280] f16 tensor on device.
    #[cfg(feature = "metal")]
    pub fn get_timestep_embedding(
        &self,
        timestep: f32,
        dim: usize,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<Tensor> {
        let half_dim = dim / 2;
        // HF SD 1.5 `Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)`:
        //   exponent = -ln(10000) * arange(half) / (half - downscale_freq_shift)
        // downscale_freq_shift=0 → divide by `half_dim`, NOT `half_dim - 1`.
        // The off-by-one made the sinusoidal frequency basis wrong at every
        // timestep except t≈1 (verified cos vs PT: t=951 0.636→1.0000,
        // t=451 0.898→1.0000 after this fix). The trained time-MLP expects
        // the exact basis, so a wrong basis degraded the noise prediction
        // increasingly as the loop moved away from the initial timestep —
        // eps std collapsed (PT stays ~1.0; ours dropped 0.99→0.18 by step
        // 19), so the sample never converged.
        let log_base = (10000.0f32).ln() / (half_dim as f32);

        // Sinusoidal embedding [dim] — tiny (320 values), keep on CPU
        let mut emb = vec![0.0f32; dim];
        for i in 0..half_dim {
            let freq = (-(i as f32) * log_base).exp();
            let val = timestep * freq;
            emb[i] = val.cos();
            emb[i + half_dim] = val.sin();
        }

        // Upload to GPU as [1, dim] for gpu_linear
        let emb_tensor = Self::f32_to_tensor(&emb, Shape::from([1, dim]), compute.device().info().id)?;

        // MLP on GPU: linear_1(dim→1280) → SiLU → linear_2(1280→1280)
        let hidden_dim = self.model.get_weight("time_embedding.linear_1.bias")
            .map(|t| t.shape().numel()).unwrap_or(1280);
        let hidden = self.gpu_linear(&emb_tensor, "time_embedding.linear_1", 1, dim, hidden_dim, true, compute, command_buffer)?;
        let hidden = self.silu(&hidden, compute, command_buffer)?;
        self.gpu_linear(&hidden, "time_embedding.linear_2", 1, hidden_dim, hidden_dim, true, compute, command_buffer)
    }

    #[cfg(not(feature = "metal"))]
    pub fn get_timestep_embedding(&self, _timestep: f32, dim: usize) -> Result<Tensor> {
        Ok(Tensor::zeros(Shape::from([1, dim]), DType::F32)?)
    }

    /// Read a model weight as f32 CPU vec.
    #[cfg(feature = "metal")]
    fn get_weight_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.model.get_weight(name)
            .ok_or_else(|| crate::core::Error::internal(format!("UNet weight not found: {}", name)))?
            .to_f32_vec()
    }

    /// Read a tensor from GPU as f32 vec.
    #[cfg(feature = "metal")]
    fn tensor_to_f32(tensor: &Tensor) -> Result<Vec<f32>> {
        let f16_data: Vec<half::f16> = tensor.to_vec()?;
        Ok(f16_data.iter().map(|v| v.to_f32()).collect())
    }

    /// Create an f16 tensor on device from f32 data.
    #[cfg(feature = "metal")]
    fn f32_to_tensor(data: &[f32], shape: Shape, device: crate::hal::DeviceId) -> Result<Tensor> {
        let f16_data: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&f16_data, shape, DType::F16, device)
    }
}

