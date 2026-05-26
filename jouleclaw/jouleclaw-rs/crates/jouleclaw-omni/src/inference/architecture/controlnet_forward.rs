//! SD 1.5 ControlNet encoder forward.
//!
//! ControlNet (Lvmin Zhang et al., 2023) is a copy of the U-Net's encoder
//! (`down_blocks` + `mid_block`) plus:
//!
//!   1. A small **conditioning embedder** (8-layer conv stack) that maps
//!      a `[3, 512, 512]` control image → `[320, 64, 64]` (matching the
//!      U-Net's `conv_in` output).
//!   2. **Zero-convolutions** at every encoder skip-output (12 down + 1
//!      mid). These are 1x1 convs initialised at zero (so the trained
//!      ControlNet starts as a no-op) that project the encoder activations
//!      into the matching residual shape.
//!
//! Forward pass:
//! ```text
//!   sample = conv_in(latents)                                 # base U-Net's conv_in (shared weights)
//!   cond   = controlnet_cond_embedding(control_image)          # 8-layer conv stack
//!   x      = sample + cond
//!   (downs, mid) = encode(x, timestep, prompt_embeds)         # base U-Net encoder
//!   residuals = [zero_conv_i(downs[i]) for i] + [zero_conv_mid(mid)]
//!   return residuals  # 13 tensors at SD 1.5: 12 down + 1 mid
//! ```
//!
//! The ControlNet weights file (`lllyasviel/control_v11p_sd15_canny`,
//! `_scribble`, etc.) contains:
//!   - The U-Net encoder copy: `down_blocks.{0..3}.*`, `mid_block.*`,
//!     `conv_in.*`, `time_embedding.*`, `class_embedding.*` (often unused
//!     for SD 1.5).
//!   - `controlnet_cond_embedding.{conv_in, blocks.{0..5}, conv_out}.*`
//!   - `controlnet_down_blocks.{0..11}.*` (12 zero-convs, 1x1 with bias)
//!   - `controlnet_mid_block.*` (1 zero-conv)

#[cfg(feature = "metal")]
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::core::Result;
#[cfg(feature = "metal")]
use crate::hal::metal::MetalCompute;
#[cfg(feature = "metal")]
use metal::CommandBufferRef;
#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::tensor::Tensor;

#[cfg(feature = "metal")]
use super::unet::UNet2DConditionModel;

/// Compiled ControlNet pipeline. Built once per `ControlNet` struct on
/// first forward; subsequent calls reuse it.
#[cfg(feature = "metal")]
pub struct ControlNetRuntime {
    /// UNet wrapper of the ControlNet weights — re-used for the encoder
    /// portion (`encode()`) and the conv2d/silu/group_norm helpers.
    unet: UNet2DConditionModel,
}

#[cfg(feature = "metal")]
impl ControlNetRuntime {
    pub fn new(model: Arc<Model>) -> Self {
        Self { unet: UNet2DConditionModel::new(model) }
    }

    /// Run the ControlNet forward pass. Returns 13 tensors (12 down + 1
    /// mid) ready to be summed into the U-Net's skip connections.
    ///
    /// `sample`: same `[B, 4, h_lat, w_lat]` latent input the base U-Net
    ///   sees this timestep (caller has already done CFG batch-doubling).
    /// `control_image`: `[B, 3, H, W]` f16 in 0..1, where (H, W) is the
    ///   FULL image resolution (typically 8 * h_lat = 512 for SD 1.5).
    /// Output residuals are produced at batch=B; caller handles their
    /// addition to the matching U-Net skips.
    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: f32,
        encoder_hidden_states: &Tensor,
        control_image: &Tensor,
        compute: &Arc<MetalCompute>,
        command_buffer: &CommandBufferRef,
    ) -> Result<Vec<Tensor>> {
        // 1. Pre-process the latent through the ControlNet's conv_in. The
        //    base U-Net's encode() also runs conv_in, but we want to add
        //    the cond embedding BEFORE the down blocks, so we run our own
        //    pre-conv_in here and pass the merged tensor through encode_no_conv_in.
        //    (Encode reuses conv_in internally, so we can't easily skip it.
        //    Pragmatic compromise: feed encode the latents — its conv_in
        //    pass produces a tensor we then add the cond embed to and run
        //    via encode again? That double-runs conv_in.)
        //
        //    The cleanest path here is to inline the encoder forward and
        //    add the cond embed after our own conv_in:
        let cond = self.cond_embedding(control_image, compute, command_buffer)?;

        // Run conv_in + add cond + run down/mid (mirrors UNet2DConditionModel::encode
        // but with the cond add hooked in right after conv_in).
        let t_emb = self.unet.get_timestep_embedding(timestep, 320, compute, command_buffer)?;
        let mut x = self.unet.conv2d(
            sample,
            "conv_in.weight",
            Some("conv_in.bias"),
            compute, command_buffer, 1, 1,
        )?;
        x = self.unet.add(&x, &cond, compute, command_buffer)?;

        // SD 1.5 reference: residual 0 is the merged tensor (conv_in + cond)
        // BEFORE any down block runs. Matches the U-Net's reference push.
        let mut down_residuals: Vec<Tensor> = vec![x.clone()];
        // SD 1.5 has 4 down blocks: 3 cross-attn + 1 plain DownBlock. Their
        // arch is held by `self.unet.down_blocks` already (UNet2DConditionModel::new
        // populates it from the model weights).
        let down_block_count = self.unet.down_block_count();
        for i in 0..down_block_count {
            let is_final = i == down_block_count - 1;
            let has_cross_attn = self.unet.down_block_has_cross_attn(i);
            for j in 0..self.unet.layers_per_block_count() {
                let prefix = format!("down_blocks.{}.resnets.{}", i, j);
                x = self.unet.resnet_block(&x, &t_emb, &prefix, compute, command_buffer)?;
                if has_cross_attn {
                    let attn_prefix = format!("down_blocks.{}.attentions.{}", i, j);
                    x = self.unet.transformer_block(&x, encoder_hidden_states, &attn_prefix, compute, command_buffer)?;
                }
                down_residuals.push(x.clone());
            }
            if !is_final {
                let prefix = format!("down_blocks.{}.downsamplers.0", i);
                x = self.unet.downsample(&x, &prefix, compute, command_buffer)?;
                down_residuals.push(x.clone());
            }
        }
        // Mid block.
        x = self.unet.resnet_block(&x, &t_emb, "mid_block.resnets.0", compute, command_buffer)?;
        x = self.unet.transformer_block(&x, encoder_hidden_states, "mid_block.attentions.0", compute, command_buffer)?;
        x = self.unet.resnet_block(&x, &t_emb, "mid_block.resnets.1", compute, command_buffer)?;
        let mid = x;

        // Zero-conv heads: 1x1 conv with bias for each down + mid.
        let mut residuals = Vec::with_capacity(down_residuals.len() + 1);
        for (i, r) in down_residuals.iter().enumerate() {
            let projected = self.unet.conv2d(
                r,
                &format!("controlnet_down_blocks.{}.weight", i),
                Some(&format!("controlnet_down_blocks.{}.bias", i)),
                compute, command_buffer, 1, 0,
            )?;
            residuals.push(projected);
        }
        let mid_proj = self.unet.conv2d(
            &mid,
            "controlnet_mid_block.weight",
            Some("controlnet_mid_block.bias"),
            compute, command_buffer, 1, 0,
        )?;
        residuals.push(mid_proj);
        Ok(residuals)
    }

    /// 8-layer conditioning embedding stack: `[3, 512, 512]` → `[320, 64, 64]`.
    ///
    /// SD 1.5 ControlNet conditioning embedder is fixed:
    ///   conv_in (3→16, k3 s1 p1) + SiLU
    ///   blocks.0 (16→16, k3 s1 p1) + SiLU
    ///   blocks.1 (16→32, k3 s2 p1) + SiLU   ← downsamples 512→256
    ///   blocks.2 (32→32, k3 s1 p1) + SiLU
    ///   blocks.3 (32→96, k3 s2 p1) + SiLU   ← downsamples 256→128
    ///   blocks.4 (96→96, k3 s1 p1) + SiLU
    ///   blocks.5 (96→256, k3 s2 p1) + SiLU  ← downsamples 128→64
    ///   conv_out (256→320, k3 s1 p1)        (no activation; final projection)
    fn cond_embedding(
        &self,
        control_image: &Tensor,
        compute: &Arc<MetalCompute>,
        cb: &CommandBufferRef,
    ) -> Result<Tensor> {
        let prefix = "controlnet_cond_embedding";
        // Per-stage (weight_suffix, stride, padding, has_silu_after).
        let stages: &[(&str, usize, usize, bool)] = &[
            ("conv_in",   1, 1, true),
            ("blocks.0",  1, 1, true),
            ("blocks.1",  2, 1, true),
            ("blocks.2",  1, 1, true),
            ("blocks.3",  2, 1, true),
            ("blocks.4",  1, 1, true),
            ("blocks.5",  2, 1, true),
            ("conv_out",  1, 1, false),
        ];
        let mut x = control_image.clone();
        for (suffix, stride, padding, silu_after) in stages.iter().copied() {
            let weight_name = format!("{}.{}.weight", prefix, suffix);
            let bias_name = format!("{}.{}.bias", prefix, suffix);
            x = self.unet.conv2d(
                &x,
                &weight_name,
                Some(&bias_name),
                compute, cb, stride, padding,
            )?;
            if silu_after {
                x = self.unet.silu(&x, compute, cb)?;
            }
        }
        Ok(x)
    }
}

// Tiny accessor surface on UNet2DConditionModel needed by ControlNet to
// know the architectural shape (block count, cross-attn flags). Living
// here to keep the unet.rs footprint stable.
//
// (These are no-ops on non-metal builds; the module itself is gated on
// the metal feature.)
