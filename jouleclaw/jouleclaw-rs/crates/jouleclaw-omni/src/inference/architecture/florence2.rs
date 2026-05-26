//! Florence-2: Vision-language model (770M params) for multi-task visual understanding.
//!
//! Architecture:
//!   Image (768×768) → DaViT vision encoder (4-stage hierarchical ViT)
//!   → Stage 0: 128-dim, 1 block, 4 heads, patch_size=4, spatial_window=7
//!   → Stage 1: 256-dim, 1 block, 8 heads, stride=2 downsample
//!   → Stage 2: 512-dim, 9 blocks, 16 heads, stride=2 downsample
//!   → Stage 3: 1024-dim, 1 block, 32 heads, stride=2 downsample
//!   Each block alternates: Spatial Window Attention → Channel Group Attention
//!
//!   → Projection: Linear(1024, 768) maps vision to text dimension
//!
//!   → Text decoder: 6-layer transformer (768-dim, 12 heads, 3072 FFN, GELU)
//!   Cross-attention to projected vision features
//!   Autoregressive token generation
//!
//! Multi-task via task-prefix tokens: <CAPTION>, <DETAILED_CAPTION>, <OCR>,
//! <OD> (object detection), <GROUNDING>, etc.
//!
//! Weight prefixes: `vision_tower.`, `text_decoder.`, `projection.`

use crate::core::{Error, Result};
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;
#[cfg(feature = "metal")]
use crate::inference::gpu_ops::{self, MetalPipeline as _};
#[cfg(feature = "metal")]
use tracing::debug;

// ==================== Configuration ====================

/// Florence-2 configuration (770M "large" variant).
#[derive(Debug, Clone)]
pub struct Florence2Config {
    /// Per-stage hidden dimensions for DaViT encoder.
    pub vision_dims: Vec<usize>,
    /// Per-stage block depths for DaViT encoder.
    pub vision_depths: Vec<usize>,
    /// Per-stage attention head counts for DaViT encoder.
    pub vision_heads: Vec<usize>,
    /// Per-stage channel-attention group counts (DaViT `num_groups`).
    pub vision_groups: Vec<usize>,
    /// Per-stage conv-embed kernel sizes (DaViT `patch_size`). Real
    /// Florence-2-base: [7, 3, 3, 3] — convs interleave with stages and
    /// do the spatial downsampling (stage blocks preserve shape).
    pub conv_kernels: Vec<usize>,
    /// Per-stage conv-embed strides (DaViT `patch_stride`): [4, 2, 2, 2].
    pub conv_strides: Vec<usize>,
    /// Per-stage conv-embed paddings (DaViT `patch_padding`): [3, 1, 1, 1].
    pub conv_paddings: Vec<usize>,
    /// Patch embedding size (legacy scalar; retained until the DaViT
    /// reimplementation switches to per-stage conv params).
    pub patch_size: usize,
    /// Spatial window size for windowed attention (DaViT `window_size`=12).
    pub spatial_window: usize,
    /// Number of channel groups for channel group attention (legacy).
    pub channel_groups: usize,
    /// Text decoder model dimension.
    pub d_model: usize,
    /// Number of text decoder layers.
    pub decoder_layers: usize,
    /// Number of text decoder attention heads.
    pub decoder_heads: usize,
    /// Text decoder feedforward dimension.
    pub decoder_ffn_dim: usize,
    /// Vocabulary size (including special tokens).
    pub vocab_size: usize,
    /// Projection dimension (vision → text).
    pub projection_dim: usize,
    /// Maximum sequence length for decoder.
    pub max_seq_len: usize,
    /// Layer norm epsilon.
    pub layer_norm_eps: f32,
}

impl Default for Florence2Config {
    /// Florence-2-large (770M params).
    fn default() -> Self {
        Self {
            vision_dims: vec![128, 256, 512, 1024],
            vision_depths: vec![1, 1, 9, 1],
            vision_heads: vec![4, 8, 16, 32],
            vision_groups: vec![4, 8, 16, 32],
            conv_kernels: vec![7, 3, 3, 3],
            conv_strides: vec![4, 2, 2, 2],
            conv_paddings: vec![3, 1, 1, 1],
            patch_size: 4,
            spatial_window: 12,
            channel_groups: 4,
            d_model: 768,
            decoder_layers: 6,
            decoder_heads: 12,
            decoder_ffn_dim: 3072,
            vocab_size: 51289,
            projection_dim: 768,
            max_seq_len: 1024,
            layer_norm_eps: 1e-5,
        }
    }
}

impl Florence2Config {
    /// Florence-2-base (231M params). Verified vs the actual
    /// `florence2-base.safetensors` checkpoint (2026-05-19): DaViT vision
    /// dims/heads are identical to -large; conv shapes convs.0=[128,3,7,7],
    /// .1=[256,128,3,3], .2=[512,256,3,3], .3=[1024,512,3,3]; depths
    /// [1,1,9,1]; window 12; projection_dim 768; BART-base 6+6.
    pub fn base() -> Self {
        Self {
            vision_dims: vec![128, 256, 512, 1024],
            vision_depths: vec![1, 1, 9, 1],
            vision_heads: vec![4, 8, 16, 32],
            vision_groups: vec![4, 8, 16, 32],
            conv_kernels: vec![7, 3, 3, 3],
            conv_strides: vec![4, 2, 2, 2],
            conv_paddings: vec![3, 1, 1, 1],
            patch_size: 4,
            spatial_window: 12,
            channel_groups: 4,
            d_model: 768,
            decoder_layers: 6,
            decoder_heads: 12,
            decoder_ffn_dim: 3072,
            vocab_size: 51289,
            projection_dim: 768,
            max_seq_len: 1024,
            layer_norm_eps: 1e-5,
        }
    }

    /// Parse from config.json.
    pub fn from_json(path: &std::path::Path) -> Result<Self> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| Error::internal(format!("failed to read config: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Error::internal(format!("failed to parse config: {}", e)))?;

        let mut c = Self::default();

        // Vision encoder config (nested under "vision_config")
        if let Some(vc) = json.get("vision_config") {
            if let Some(dims) = vc.get("dim_embed").and_then(|v| v.as_array()) {
                c.vision_dims = dims.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect();
            }
            if let Some(depths) = vc.get("depths").and_then(|v| v.as_array()) {
                c.vision_depths = depths.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect();
            }
            if let Some(heads) = vc.get("num_heads").and_then(|v| v.as_array()) {
                c.vision_heads = heads.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect();
            }
            if let Some(v) = vc.get("patch_size").and_then(|v| v.as_u64()) { c.patch_size = v as usize; }
            if let Some(v) = vc.get("window_size").and_then(|v| v.as_u64()) { c.spatial_window = v as usize; }
            if let Some(v) = vc.get("num_groups").and_then(|v| v.as_u64()) { c.channel_groups = v as usize; }
        }

        // Text decoder config (nested under "text_config")
        if let Some(tc) = json.get("text_config") {
            if let Some(v) = tc.get("d_model").and_then(|v| v.as_u64()) { c.d_model = v as usize; }
            if let Some(v) = tc.get("decoder_layers").and_then(|v| v.as_u64()) { c.decoder_layers = v as usize; }
            if let Some(v) = tc.get("decoder_attention_heads").and_then(|v| v.as_u64()) { c.decoder_heads = v as usize; }
            if let Some(v) = tc.get("decoder_ffn_dim").and_then(|v| v.as_u64()) { c.decoder_ffn_dim = v as usize; }
            if let Some(v) = tc.get("vocab_size").and_then(|v| v.as_u64()) { c.vocab_size = v as usize; }
            if let Some(v) = tc.get("max_position_embeddings").and_then(|v| v.as_u64()) { c.max_seq_len = v as usize; }
        }

        if let Some(v) = json.get("projection_dim").and_then(|v| v.as_u64()) { c.projection_dim = v as usize; }

        Ok(c)
    }

    /// Head dimension for a given stage.
    fn head_dim(&self, stage: usize) -> usize {
        self.vision_dims[stage] / self.vision_heads[stage]
    }
}

// ==================== Task Tokens ====================

/// Florence-2 task types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Florence2Task {
    /// Short caption generation.
    Caption,
    /// Detailed caption generation.
    DetailedCaption,
    /// Optical character recognition.
    Ocr,
    /// Object detection with bounding boxes.
    ObjectDetection,
    /// Grounded captioning (text + regions).
    Grounding,
}

impl Florence2Task {
    /// Parse from string.
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "caption" => Ok(Self::Caption),
            "detailed_caption" => Ok(Self::DetailedCaption),
            "ocr" => Ok(Self::Ocr),
            "object_detection" => Ok(Self::ObjectDetection),
            "grounding" => Ok(Self::Grounding),
            _ => Err(Error::internal(format!(
                "unknown Florence-2 task: '{}'. Expected: caption, detailed_caption, ocr, object_detection, grounding",
                s
            ))),
        }
    }

    /// Task prefix string for the decoder prompt.
    pub fn prefix(&self) -> &'static str {
        match self {
            Self::Caption => "<CAPTION>",
            Self::DetailedCaption => "<DETAILED_CAPTION>",
            Self::Ocr => "<OCR>",
            Self::ObjectDetection => "<OD>",
            Self::Grounding => "<GROUNDING>",
        }
    }
}

// ==================== Compiled Kernels ====================

#[cfg(feature = "metal")]
#[allow(dead_code)]
struct Florence2Kernels {
    common: gpu_ops::CommonKernels,
    gelu: Arc<ComputePipeline>,
    /// Used by `davit_downsample` between DaViT stages.
    patch_merge_concat: Arc<ComputePipeline>,
    embedding_lookup: Arc<ComputePipeline>,
    gqa_attention: Arc<ComputePipeline>,
}

// ==================== Florence-2 Pipeline ====================

/// Florence-2 vision-language pipeline.
///
/// Forward pipeline:
/// 1. DaViT vision encoder: image [3, H, W] → hierarchical features → [N, 1024]
///    - 4 stages with spatial window attention + channel group attention
///    - Progressive downsampling via stride-2 convolution
/// 2. Projection: Linear(1024, 768) maps vision features to text space
/// 3. Text decoder: 6-layer transformer with cross-attention to vision features
///    - Autoregressive token generation conditioned on task prefix + vision
#[cfg(feature = "metal")]
pub struct Florence2Pipeline {
    /// Loaded weights. Held as `Arc<Model>` (read-only access pattern matches
    /// every other architecture pipeline in this crate — Hunyuan3D, Trellis,
    /// etc.). The earlier `Arc<RwLock<Model>>` shape was unnecessary; weight
    /// reads through `Model::get_weight` are already interior-mutable.
    model: Arc<Model>,
    compute: Arc<MetalCompute>,
    config: Florence2Config,
    kernels: Florence2Kernels,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for Florence2Pipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl Florence2Pipeline {
    /// Create a new Florence-2 pipeline.
    pub fn new(model: Arc<Model>, config: Florence2Config, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        let kernels = Florence2Kernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            patch_merge_concat: compute.compile_pipeline(
                "patch_merge_concat",
                sources::PATCH_MERGE_CONCAT,
                "patch_merge_concat_f16",
            )?,
            embedding_lookup: compute.compile_pipeline(
                "embedding_lookup",
                sources::EMBEDDING,
                "embedding_lookup_f16",
            )?,
            gqa_attention: compute.compile_pipeline(
                "gqa_attention",
                sources::GQA_ATTENTION,
                "gqa_attention_f16",
            )?,
        };

        Ok(Self { model, compute, config, kernels })
    }

    /// Analyze an image with a specified task.
    ///
    /// `image_rgb`: flat f32 array in [3, H, W] layout, normalized to [0, 1].
    /// `width`, `height`: original image dimensions.
    /// `task`: one of "caption", "detailed_caption", "ocr", "object_detection", "grounding".
    ///
    /// Returns the generated text output.
    pub fn analyze(
        &self,
        image_rgb: &[f32],
        width: usize,
        height: usize,
        task: &str,
    ) -> Result<String> {
        let task = Florence2Task::from_str(task)?;

        debug!(task = ?task, width, height, "Florence-2 analyze");

        // 1. Florence-2 vision encoder (M4 verified) → image features [577, 768].
        let image_features = self.davit_encode(image_rgb, width, height)?;

        // 2. BART input construction (M5b verified): embed task tokens, concat
        //    with image features, add learned positional embed (offset 2), LN.
        let enc_in = self.bart_encoder_input(&image_features, task)?;

        // 3. BART encoder: 6 post-norm layers (M5c). Output [N, d_model].
        let enc_out = self.bart_encoder_layers(&enc_in)?;

        // 4. Greedy generation (M5e). Start with decoder_start_token_id=2 (BART).
        //    Run decoder for the full token sequence each step (no KV cache yet —
        //    correctness-first; optimize later). Stop on EOS=2 or max_new_tokens.
        let dec_start: u32 = 2;
        let eos: u32 = 2;
        let max_new = 64usize;
        let mut ids: Vec<u32> = vec![dec_start];
        let mut generated: Vec<u32> = Vec::new();
        for _ in 0..max_new {
            let logits = self.bart_decoder_step(&enc_out, &ids)?;
            let next = argmax_f32(&logits);
            generated.push(next);
            if next == eos { break; }
            ids.push(next);
        }
        // Dump generated token IDs + detokenize.
        if let Ok(dir) = std::env::var("FLO2_DUMP_DIR") {
            let s = generated.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");
            let _ = std::fs::write(format!("{}/rust_08_generated_ids.txt", dir), s);
            tracing::info!("[flo2-diag] generated {} tokens: {:?}", generated.len(), generated);
        }
        let caption = self.detokenize(&generated);
        tracing::info!("[flo2-diag] caption: {:?}", caption);
        Ok(caption)
    }

    // ==================== DaViT Vision Encoder ====================

    /// DaViT hierarchical vision encoder.
    ///
    /// 4-stage architecture:
    ///   Stage 0: patch embed (4×4 → 128-dim), 1 block
    ///   Stage 1: downsample (stride 2 → 256-dim), 1 block
    ///   Stage 2: downsample (stride 2 → 512-dim), 9 blocks
    ///   Stage 3: downsample (stride 2 → 1024-dim), 1 block
    ///
    /// Each block: spatial window attention → channel group attention.
    fn davit_encode(&self, image_rgb: &[f32], width: usize, height: usize) -> Result<Tensor> {
        let config = &self.config;

        // [DIAG] Element-wise verification harness (mirrors SD's SD_DUMP_DIR).
        // When FLO2_DUMP_DIR is set, the Rust path must consume the EXACT
        // HF-preprocessed pixel_values (`00_pixel_values.f32`, [1,3,768,768])
        // so per-checkpoint cos/relL2 vs `/tmp/flo2_dump` is apples-to-apples;
        // re-preprocessing in Rust would diverge (HF resize+mean/std differ).
        let flo2_dump = std::env::var("FLO2_DUMP_DIR").ok();
        let injected: Option<Vec<f32>> = flo2_dump.as_ref().and_then(|d| {
            let p = format!("{}/00_pixel_values.f32", d);
            std::fs::read(&p).ok().map(|b| {
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            })
        });
        let (image_rgb, width, height): (&[f32], usize, usize) = match injected.as_ref() {
            Some(v) => {
                tracing::info!("[flo2-diag] injected 00_pixel_values.f32 ({} f32 = 3x768x768)", v.len());
                (v.as_slice(), 768, 768)
            }
            None => (image_rgb, width, height),
        };
        let dump_f32 = |name: &str, t: &Tensor| {
            if let Some(dir) = flo2_dump.as_ref() {
                if let Ok(v) = t.to_f32_vec() {
                    let mut bytes = Vec::with_capacity(v.len() * 4);
                    for f in &v { bytes.extend_from_slice(&f.to_le_bytes()); }
                    let _ = std::fs::write(format!("{}/rust_{}.f32", dir, name), &bytes);
                    tracing::info!("[flo2-diag] wrote rust_{}.f32 ({} f32, shape={:?})", name, v.len(), t.shape());
                }
            }
        };

        // Stage 0: real DaViT conv-embed `vision_tower.convs.0`
        // (Conv2d 3→128, k7 s4 p3) → 768→192 spatial, then convs.0.norm.
        let k0 = config.conv_kernels[0];
        let s0 = config.conv_strides[0];
        let p0 = config.conv_paddings[0];
        let mut h = (height + 2 * p0 - k0) / s0 + 1;
        let mut w = (width + 2 * p0 - k0) / s0 + 1;
        let mut dim = config.vision_dims[0];

        debug!(h, w, dim, "conv0");
        let mut hidden = self.davit_patch_embed(image_rgb, width, height, k0, dim)?;
        dump_f32("01_conv0", &hidden);

        // Unified DaViT loop: for each stage i:
        //   if i > 0: convs[i] (seq-input k3 s2 p1 conv-embed + LN; h,w /= 2, dim → next)
        //   for j in 0..depths[i]: spatial_block then channel_block (per modeling_florence2.py)
        let window_size = config.spatial_window;
        for stage in 0..4usize {
            let stage_dim = config.vision_dims[stage];
            let num_heads = config.vision_heads[stage];
            let num_groups = config.vision_groups[stage];
            let depth = config.vision_depths[stage];

            if stage > 0 {
                // convs.{stage}: 3D-seq input → strided conv k3 s2 p1 → LN.
                let prev_dim = config.vision_dims[stage - 1];
                let kk = config.conv_kernels[stage];
                let ss = config.conv_strides[stage];
                let pp = config.conv_paddings[stage];
                hidden = self.davit_conv_embed_seq(
                    &hidden, h, w, prev_dim, stage_dim, kk, ss, pp, stage,
                )?;
                h = (h + 2 * pp - kk) / ss + 1;
                w = (w + 2 * pp - kk) / ss + 1;
                dim = stage_dim;
                dump_f32(&format!("01_conv{}", stage), &hidden);
                debug!(h, w, dim, stage, "conv-embed");
            }

            for j in 0..depth {
                // SpatialBlock
                let sp = format!("vision_tower.blocks.{}.{}.spatial_block", stage, j);
                hidden = self.davit_dw_conv_residual(&hidden, h, w, dim, &format!("{}.conv1", sp))?;
                if stage == 0 && j == 0 { dump_f32("03_sp_conv1", &hidden); }
                hidden = self.davit_window_attn_prenorm(
                    &hidden, h, w, dim, num_heads, window_size,
                    &format!("{}.window_attn", sp),
                )?;
                if stage == 0 && j == 0 { dump_f32("03_sp_wattn", &hidden); }
                hidden = self.davit_dw_conv_residual(&hidden, h, w, dim, &format!("{}.conv2", sp))?;
                if stage == 0 && j == 0 { dump_f32("03_sp_conv2", &hidden); }
                hidden = self.davit_ffn_prenorm(&hidden, h * w, dim, &format!("{}.ffn", sp))?;
                if stage == 0 && j == 0 { dump_f32("03_sp_ffn", &hidden); }

                // ChannelBlock
                let ch = format!("vision_tower.blocks.{}.{}.channel_block", stage, j);
                hidden = self.davit_dw_conv_residual(&hidden, h, w, dim, &format!("{}.conv1", ch))?;
                if stage == 0 && j == 0 { dump_f32("03_ch_conv1", &hidden); }
                hidden = self.davit_channel_attn_prenorm(
                    &hidden, h * w, dim, num_groups,
                    &format!("{}.channel_attn", ch),
                )?;
                if stage == 0 && j == 0 { dump_f32("03_ch_cattn", &hidden); }
                hidden = self.davit_dw_conv_residual(&hidden, h, w, dim, &format!("{}.conv2", ch))?;
                if stage == 0 && j == 0 { dump_f32("03_ch_conv2", &hidden); }
                hidden = self.davit_ffn_prenorm(&hidden, h * w, dim, &format!("{}.ffn", ch))?;
                if stage == 0 && j == 0 { dump_f32("03_ch_ffn", &hidden); }
            }
            dump_f32(&format!("02_stage{}", stage), &hidden);
        }

        // M4d image projection: feeds the 576×1024 DaViT output into BART.
        //   x (1×24×24×1024 view) + image_pos_embed(row, col)
        //   + visual_temporal_embed[0]  (T=1, broadcast over spatial)
        //   spatial_avg_pool = mean over spatial  → [1, 1024]
        //   temporal_avg_pool = x itself           → [576, 1024]
        //   cat([spatial, temporal], dim=tokens)   → [577, 1024]
        //   @ image_projection [1024, 768] (no bias)
        //   LN(image_proj_norm)                    → [577, 768]
        let final_dim = config.vision_dims[3];   // 1024
        let proj_dim = config.projection_dim;     // 768
        let proj = self.davit_image_projection(&hidden, h, w, final_dim, proj_dim)?;
        dump_f32("04_image_proj_norm", &proj);
        debug!(proj_dim, "Florence-2 vision projection complete (577×{})", proj_dim);
        Ok(proj)
    }

    /// DaViT `ConvPosEnc` = `PreNorm(None, DepthWiseConv2d(k=3, p=1, s=1, groups=C))`,
    /// which is just `x = x + dw(x)`. CPU implementation — depthwise so per-
    /// channel weight [C,1,3,3] flat = C*9 elements; bias [C]. Input [N=H*W, C]
    /// row-major (position p has slot p*C + c).
    /// `prefix` is the block path; the weights are at `{prefix}.fn.dw.{weight,bias}`.
    fn davit_dw_conv_residual(
        &self, input: &Tensor, h: usize, w: usize, dim: usize, prefix: &str,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let n = h * w;
        let x: Vec<f32> = input.to_f32_vec()?;
        let w_name = format!("{}.fn.dw.weight", prefix);
        let b_name = format!("{}.fn.dw.bias", prefix);
        let wgt: Vec<f32> = self.weight_vec_f32(&self.model, &w_name)?;
        let bias: Vec<f32> = self.weight_vec_f32(&self.model, &b_name)?;
        // wgt[c*9 + ky*3 + kx] ; PyTorch conv weight shape [C, 1, 3, 3] row-major.
        let mut out = vec![half::f16::ZERO; n * dim];
        for oy in 0..h {
            for ox in 0..w {
                let p = oy * w + ox;
                for c in 0..dim {
                    let mut s = bias[c];
                    for ky in 0..3 {
                        let iy = oy as isize + ky as isize - 1;
                        if iy < 0 || iy >= h as isize { continue; }
                        for kx in 0..3 {
                            let ix = ox as isize + kx as isize - 1;
                            if ix < 0 || ix >= w as isize { continue; }
                            let ip = iy as usize * w + ix as usize;
                            s += x[ip * dim + c] * wgt[c * 9 + ky * 3 + kx];
                        }
                    }
                    // Residual: out = x + dw(x)
                    out[p * dim + c] = half::f16::from_f32(x[p * dim + c] + s);
                }
            }
        }
        Tensor::from_slice(&out, Shape::from([n, dim]), DType::F16, device_id)
    }

    /// DaViT spatial window attention with `PreNorm` and residual.
    ///
    /// `PreNorm(LN {prefix}.norm, WindowAttention({prefix}.fn.qkv, .fn.proj))`:
    ///   shortcut = x
    ///   y = LayerNorm(x; norm.weight, norm.bias)
    ///   y = WindowAttention(y, H, W, ws, num_heads)
    ///   x = shortcut + y
    /// CPU correctness implementation. For stage0 (H=W=192, ws=12, heads=4,
    /// dim=128) there is no padding (192 % 12 == 0); the pad branch is
    /// implemented but exercised only by deeper stages.
    fn davit_window_attn_prenorm(
        &self, input: &Tensor, h: usize, w: usize, dim: usize,
        num_heads: usize, window_size: usize, prefix: &str,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let n = h * w;
        let head_dim = dim / num_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let x: Vec<f32> = input.to_f32_vec()?;

        // --- LayerNorm over C ---
        let ln_w: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.norm.weight", prefix))?;
        let ln_b: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.norm.bias", prefix))?;
        let eps = self.config.layer_norm_eps;
        let mut xn = vec![0.0f32; n * dim];
        for p in 0..n {
            let off = p * dim;
            let mut mean = 0.0f32;
            for c in 0..dim { mean += x[off + c]; }
            mean /= dim as f32;
            let mut var = 0.0f32;
            for c in 0..dim { let d = x[off + c] - mean; var += d * d; }
            var /= dim as f32;
            let inv = 1.0f32 / (var + eps).sqrt();
            for c in 0..dim { xn[off + c] = (x[off + c] - mean) * inv * ln_w[c] + ln_b[c]; }
        }

        // --- Pad to multiple of window_size ---
        let pad_r = (window_size - w % window_size) % window_size;
        let pad_b = (window_size - h % window_size) % window_size;
        let hp = h + pad_b;
        let wp = w + pad_r;
        // padded x_norm view [Hp, Wp, C], zero-padded
        let mut padded = vec![0.0f32; hp * wp * dim];
        for r in 0..h {
            for col in 0..w {
                let src = (r * w + col) * dim;
                let dst = (r * wp + col) * dim;
                padded[dst..dst + dim].copy_from_slice(&xn[src..src + dim]);
            }
        }

        // --- QKV + proj weights (flat row-major: [out, in]) ---
        let qkv_w: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.fn.qkv.weight", prefix))?;
        let qkv_b: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.fn.qkv.bias", prefix))?;
        let proj_w: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.fn.proj.weight", prefix))?;
        let proj_b: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.fn.proj.bias", prefix))?;
        // qkv_w shape [3*dim, dim], proj_w [dim, dim].

        let ws = window_size;
        let nwh = hp / ws;
        let nww = wp / ws;
        let tokens_per_win = ws * ws;

        // Output buffer in PADDED [Hp, Wp, C] layout — we'll crop at the end.
        let mut out_padded = vec![0.0f32; hp * wp * dim];

        // Per-window MHSA. Allocate per-window scratch.
        let mut win = vec![0.0f32; tokens_per_win * dim];
        let mut qkv = vec![0.0f32; tokens_per_win * 3 * dim];
        let mut scores = vec![0.0f32; tokens_per_win * tokens_per_win];
        let mut attn_out = vec![0.0f32; tokens_per_win * dim];
        let mut proj_out = vec![0.0f32; tokens_per_win * dim];

        for wy in 0..nwh {
            for wx in 0..nww {
                // Gather this window's tokens from padded[Hp,Wp,C].
                for ty in 0..ws {
                    for tx in 0..ws {
                        let local = ty * ws + tx;
                        let gy = wy * ws + ty;
                        let gx = wx * ws + tx;
                        let src = (gy * wp + gx) * dim;
                        win[local * dim..local * dim + dim]
                            .copy_from_slice(&padded[src..src + dim]);
                    }
                }
                // qkv = win @ qkv_w^T + qkv_b ; [N=tpw, 3*dim]
                for t in 0..tokens_per_win {
                    for o in 0..3 * dim {
                        let mut s = qkv_b[o];
                        let wrow = o * dim;
                        let trow = t * dim;
                        for c in 0..dim { s += win[trow + c] * qkv_w[wrow + c]; }
                        qkv[t * 3 * dim + o] = s;
                    }
                }
                // Multi-head attention. qkv[t, 0..dim]=q, dim..2dim=k, 2dim..3dim=v.
                // Per head h_i: head occupies channels [h_i*head_dim, (h_i+1)*head_dim).
                for hi in 0..num_heads {
                    let hoff = hi * head_dim;
                    // scores[i,j] = scale * sum_d q[i, hoff+d] * k[j, hoff+d]
                    for i in 0..tokens_per_win {
                        let qbase = i * 3 * dim + 0 * dim + hoff;
                        for j in 0..tokens_per_win {
                            let kbase = j * 3 * dim + 1 * dim + hoff;
                            let mut s = 0.0f32;
                            for d in 0..head_dim { s += qkv[qbase + d] * qkv[kbase + d]; }
                            scores[i * tokens_per_win + j] = s * scale;
                        }
                    }
                    // softmax over j per row i
                    for i in 0..tokens_per_win {
                        let row = i * tokens_per_win;
                        let mut m = f32::NEG_INFINITY;
                        for j in 0..tokens_per_win { if scores[row + j] > m { m = scores[row + j]; } }
                        let mut sum = 0.0f32;
                        for j in 0..tokens_per_win {
                            let e = (scores[row + j] - m).exp();
                            scores[row + j] = e; sum += e;
                        }
                        let inv = 1.0f32 / sum;
                        for j in 0..tokens_per_win { scores[row + j] *= inv; }
                    }
                    // out_h[i, hoff+d] = sum_j scores[i,j] * v[j, hoff+d]
                    for i in 0..tokens_per_win {
                        let row = i * tokens_per_win;
                        for d in 0..head_dim {
                            let mut s = 0.0f32;
                            for j in 0..tokens_per_win {
                                let vbase = j * 3 * dim + 2 * dim + hoff;
                                s += scores[row + j] * qkv[vbase + d];
                            }
                            attn_out[i * dim + hoff + d] = s;
                        }
                    }
                }
                // proj = attn_out @ proj_w^T + proj_b
                for t in 0..tokens_per_win {
                    for o in 0..dim {
                        let mut s = proj_b[o];
                        let wrow = o * dim;
                        let trow = t * dim;
                        for c in 0..dim { s += attn_out[trow + c] * proj_w[wrow + c]; }
                        proj_out[t * dim + o] = s;
                    }
                }
                // Scatter back to padded out
                for ty in 0..ws {
                    for tx in 0..ws {
                        let local = ty * ws + tx;
                        let gy = wy * ws + ty;
                        let gx = wx * ws + tx;
                        let dst = (gy * wp + gx) * dim;
                        out_padded[dst..dst + dim]
                            .copy_from_slice(&proj_out[local * dim..local * dim + dim]);
                    }
                }
            }
        }

        // --- Crop padding + residual add ---
        let mut out = vec![half::f16::ZERO; n * dim];
        for r in 0..h {
            for col in 0..w {
                let src = (r * wp + col) * dim;
                let dst = (r * w + col) * dim;
                for c in 0..dim {
                    // shortcut x (NOT x_norm) + window_attn result
                    let v = x[dst + c] + out_padded[src + c];
                    out[dst + c] = half::f16::from_f32(v);
                }
            }
        }
        Tensor::from_slice(&out, Shape::from([n, dim]), DType::F16, device_id)
    }

    /// M4d Florence-2 image projection: stage3 DaViT output → BART input.
    /// Mirrors `Florence2VisionModelWithProjection.forward` for T=1:
    ///   x [N=h*w, dim] → view [h, w, dim] → + image_pos_embed
    ///                  → + visual_temporal_embed[0]    (broadcast)
    ///   spatial_avg_pool = x.mean(spatial)    → [1, dim]
    ///   temporal_avg_pool = x (for T=1)       → [N, dim]
    ///   concat([spatial, temporal], dim=tok)  → [1+N, dim]
    ///   @ image_projection (Parameter, no bias) → [1+N, proj_dim]
    ///   image_proj_norm (LayerNorm proj_dim)  → [1+N, proj_dim]
    fn davit_image_projection(
        &self, input: &Tensor, h: usize, w: usize, dim: usize, proj_dim: usize,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let n = h * w;
        let mut x: Vec<f32> = input.to_f32_vec()?;

        // image_pos_embed: row + column embeddings concat.
        // row_embeddings [num_pos, dim/2], column_embeddings [num_pos, dim-dim/2].
        let row_w: Vec<f32> = self.weight_vec_f32(&self.model, "image_pos_embed.row_embeddings.weight")?;
        let col_w: Vec<f32> = self.weight_vec_f32(&self.model, "image_pos_embed.column_embeddings.weight")?;
        // Florence-2 default num_pos=50. row_embeddings is [50, dim/2],
        // column_embeddings is [50, dim - dim/2] (split is dim/2 each for even dim).
        let num_pos = 50usize;
        let row_dim = row_w.len() / num_pos;          // dim/2 = 512
        let col_dim = col_w.len() / num_pos;          // dim - dim/2 = 512
        debug_assert_eq!(row_dim + col_dim, dim);
        // Per modeling_florence2.py: pos = cat([col_x, row_y], dim=-1)
        // So pos[r,c] = [column_embeddings[c, 0..col_dim], row_embeddings[r, 0..row_dim]]
        for r in 0..h {
            for c in 0..w {
                let off = (r * w + c) * dim;
                for i in 0..col_dim {
                    x[off + i] += col_w[c * col_dim + i];
                }
                for i in 0..row_dim {
                    x[off + col_dim + i] += row_w[r * row_dim + i];
                }
            }
        }

        // visual_temporal_embed[0]: row 0 of pos_idx_to_embed (T=1 → pos=0).
        let temp_buf: Vec<f32> = self.weight_vec_f32(
            &self.model, "visual_temporal_embed.pos_idx_to_embed")?;
        // shape [max_seq_len, dim]; take first `dim` elements (row 0).
        for i in 0..n {
            let off = i * dim;
            for c in 0..dim { x[off + c] += temp_buf[c]; }
        }

        // Pools.
        let mut spatial_pool = vec![0.0f32; dim];
        for i in 0..n {
            let off = i * dim;
            for c in 0..dim { spatial_pool[c] += x[off + c]; }
        }
        for c in 0..dim { spatial_pool[c] /= n as f32; }
        // temporal_avg_pool for T=1 is just x.

        // Concat: row 0 = spatial_pool, rows 1..n+1 = x.
        let m = n + 1;
        let mut cat = vec![0.0f32; m * dim];
        cat[0..dim].copy_from_slice(&spatial_pool);
        for i in 0..n {
            let src = i * dim;
            let dst = (i + 1) * dim;
            cat[dst..dst + dim].copy_from_slice(&x[src..src + dim]);
        }

        // image_projection (Parameter): shape [dim, proj_dim], no bias.
        let proj_w: Vec<f32> = self.weight_vec_f32(&self.model, "image_projection")?;
        // proj_out[i, o] = Σ_c cat[i, c] * proj_w[c, o]   (proj_w row-major [dim, proj_dim])
        let mut proj_out = vec![0.0f32; m * proj_dim];
        for i in 0..m {
            for o in 0..proj_dim {
                let mut s = 0.0f32;
                let ci = i * dim;
                for c in 0..dim {
                    s += cat[ci + c] * proj_w[c * proj_dim + o];
                }
                proj_out[i * proj_dim + o] = s;
            }
        }

        // image_proj_norm: LayerNorm over proj_dim.
        let ln_w: Vec<f32> = self.weight_vec_f32(&self.model, "image_proj_norm.weight")?;
        let ln_b: Vec<f32> = self.weight_vec_f32(&self.model, "image_proj_norm.bias")?;
        let eps = self.config.layer_norm_eps;
        let mut out = vec![half::f16::ZERO; m * proj_dim];
        for p in 0..m {
            let off = p * proj_dim;
            let mut mean = 0.0f32;
            for c in 0..proj_dim { mean += proj_out[off + c]; }
            mean /= proj_dim as f32;
            let mut var = 0.0f32;
            for c in 0..proj_dim { let d = proj_out[off + c] - mean; var += d * d; }
            var /= proj_dim as f32;
            let inv = 1.0f32 / (var + eps).sqrt();
            for c in 0..proj_dim {
                let v = (proj_out[off + c] - mean) * inv * ln_w[c] + ln_b[c];
                out[off + c] = half::f16::from_f32(v);
            }
        }
        Tensor::from_slice(&out, Shape::from([m, proj_dim]), DType::F16, device_id)
    }

    /// DaViT `ConvEmbed` for stages 1..3 (seq-input, **pre-norm** variant).
    /// Per checkpoint: convs.{1,2,3}.norm.weight shape = `in_chans` (not
    /// out_dim) → Florence-2-base uses `patch_prenorm=(F,T,T,T)`. Flow:
    ///   x [B, N=H*W, in_dim] → LN over in_dim (`convs.{i}.norm`)
    ///                        → reshape to [B, in_dim, H, W]
    ///                        → Conv2d(in→out, k, s, p) (`convs.{i}.proj`)
    ///                        → reshape to [B, N_out, out_dim]
    /// (stage 0 uses POST-norm in davit_patch_embed; cos=1.0 verified there.)
    fn davit_conv_embed_seq(
        &self, input: &Tensor, h: usize, w: usize, in_dim: usize, out_dim: usize,
        k: usize, stride: usize, pad: usize, stage: usize,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let h_out = (h + 2 * pad - k) / stride + 1;
        let w_out = (w + 2 * pad - k) / stride + 1;
        let n_in = h * w;
        let n_out = h_out * w_out;
        let mut x: Vec<f32> = input.to_f32_vec()?;

        // PRE-norm: LN over in_dim, in-place on x.
        let ln_w: Vec<f32> = self.weight_vec_f32(
            &self.model, &format!("vision_tower.convs.{}.norm.weight", stage))?;
        let ln_b: Vec<f32> = self.weight_vec_f32(
            &self.model, &format!("vision_tower.convs.{}.norm.bias", stage))?;
        let eps = self.config.layer_norm_eps;
        for p in 0..n_in {
            let off = p * in_dim;
            let mut mean = 0.0f32;
            for c in 0..in_dim { mean += x[off + c]; }
            mean /= in_dim as f32;
            let mut var = 0.0f32;
            for c in 0..in_dim { let d = x[off + c] - mean; var += d * d; }
            var /= in_dim as f32;
            let inv = 1.0f32 / (var + eps).sqrt();
            for c in 0..in_dim {
                x[off + c] = (x[off + c] - mean) * inv * ln_w[c] + ln_b[c];
            }
        }

        // Conv2d k,s,p on the normalized input.
        let wgt: Vec<f32> = self.weight_vec_f32(
            &self.model, &format!("vision_tower.convs.{}.proj.weight", stage))?;
        let bias: Vec<f32> = self.weight_vec_f32(
            &self.model, &format!("vision_tower.convs.{}.proj.bias", stage))?;
        let mut out_f32 = vec![0.0f32; n_out * out_dim];
        for oy in 0..h_out {
            for ox in 0..w_out {
                let p_out = oy * w_out + ox;
                let out_base = p_out * out_dim;
                for c_out in 0..out_dim {
                    let mut s = bias[c_out];
                    for ky in 0..k {
                        let iy = oy as isize * stride as isize - pad as isize + ky as isize;
                        if iy < 0 || iy >= h as isize { continue; }
                        for kx in 0..k {
                            let ix = ox as isize * stride as isize - pad as isize + kx as isize;
                            if ix < 0 || ix >= w as isize { continue; }
                            let p_in = iy as usize * w + ix as usize;
                            let in_base = p_in * in_dim;
                            // weight[c_out, c_in, ky, kx] flat row-major:
                            // wgt[c_out*in_dim*k*k + c_in*k*k + ky*k + kx]
                            let w_base = c_out * in_dim * k * k + ky * k + kx;
                            for c_in in 0..in_dim {
                                s += x[in_base + c_in] * wgt[w_base + c_in * k * k];
                            }
                        }
                    }
                    out_f32[out_base + c_out] = s;
                }
            }
        }

        let out: Vec<half::f16> = out_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&out, Shape::from([n_out, out_dim]), DType::F16, device_id)
    }

    /// DaViT channel attention with `PreNorm(LN, ChannelAttention)` + residual.
    ///
    /// Reference (modeling_florence2.py ChannelAttention):
    ///   qkv = Linear(x) → reshape [B,N,3,groups,C/g] → permute → q,k,v [B,g,N,C/g]
    ///   q *= N^-0.5    (NB: scale by sequence length N, NOT head_dim!)
    ///   attn = softmax(qᵀ @ k, dim=-1)            # [B,g,C/g,C/g]
    ///   out = (attn @ vᵀ)ᵀ                        # [B,g,N,C/g]
    ///   out.reshape(B,N,C) → Linear proj
    /// Wrapped: `x = shortcut + proj(ChannelAttention(LN(x)))`.
    fn davit_channel_attn_prenorm(
        &self, input: &Tensor, n: usize, dim: usize, groups: usize, prefix: &str,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let head_dim = dim / groups;
        let scale = 1.0f32 / (n as f32).sqrt();

        let x: Vec<f32> = input.to_f32_vec()?;

        // LN
        let ln_w: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.norm.weight", prefix))?;
        let ln_b: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.norm.bias", prefix))?;
        let eps = self.config.layer_norm_eps;
        let mut xn = vec![0.0f32; n * dim];
        for p in 0..n {
            let off = p * dim;
            let mut mean = 0.0f32;
            for c in 0..dim { mean += x[off + c]; }
            mean /= dim as f32;
            let mut var = 0.0f32;
            for c in 0..dim { let d = x[off + c] - mean; var += d * d; }
            var /= dim as f32;
            let inv = 1.0f32 / (var + eps).sqrt();
            for c in 0..dim { xn[off + c] = (x[off + c] - mean) * inv * ln_w[c] + ln_b[c]; }
        }

        // qkv: [n, dim] → [n, 3*dim]. Weight [3*dim, dim].
        let qkv_w: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.fn.qkv.weight", prefix))?;
        let qkv_b: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.fn.qkv.bias", prefix))?;
        let three_dim = 3 * dim;
        let mut qkv = vec![0.0f32; n * three_dim];
        for i in 0..n {
            for o in 0..three_dim {
                let mut s = qkv_b[o];
                let xr = i * dim;
                let wr = o * dim;
                for c in 0..dim { s += xn[xr + c] * qkv_w[wr + c]; }
                qkv[i * three_dim + o] = s;
            }
        }
        // Channel layout in qkv (post-reshape [n,3,groups,head_dim] flat row-major):
        //   q[n,g,d] = qkv[n, 0*dim + g*head_dim + d]
        //   k[n,g,d] = qkv[n, 1*dim + g*head_dim + d]
        //   v[n,g,d] = qkv[n, 2*dim + g*head_dim + d]

        // Per-group: attn [head_dim, head_dim] = scale * qᵀ @ k.
        // Then softmax(-1). Then out[n, d_out] = Σ_{d_in} attn[d_out, d_in] * v[n, g, d_in].
        let mut attn_out = vec![0.0f32; n * dim];
        let mut attn = vec![0.0f32; head_dim * head_dim];
        for g in 0..groups {
            let gqo = 0 * dim + g * head_dim;
            let gko = 1 * dim + g * head_dim;
            let gvo = 2 * dim + g * head_dim;
            // attn[d_out, d_in] = scale * Σ_n q[n,g,d_out] * k[n,g,d_in]
            for d_out in 0..head_dim {
                for d_in in 0..head_dim {
                    let mut s = 0.0f32;
                    for tok in 0..n {
                        s += qkv[tok * three_dim + gqo + d_out]
                           * qkv[tok * three_dim + gko + d_in];
                    }
                    attn[d_out * head_dim + d_in] = s * scale;
                }
            }
            // softmax over d_in per row d_out
            for d_out in 0..head_dim {
                let row = d_out * head_dim;
                let mut m = f32::NEG_INFINITY;
                for j in 0..head_dim { if attn[row + j] > m { m = attn[row + j]; } }
                let mut sum = 0.0f32;
                for j in 0..head_dim {
                    let e = (attn[row + j] - m).exp();
                    attn[row + j] = e; sum += e;
                }
                let inv = 1.0f32 / sum;
                for j in 0..head_dim { attn[row + j] *= inv; }
            }
            // out[n, g, d_out] = Σ_{d_in} attn[d_out, d_in] * v[n, g, d_in]
            for tok in 0..n {
                for d_out in 0..head_dim {
                    let mut s = 0.0f32;
                    let row = d_out * head_dim;
                    for d_in in 0..head_dim {
                        s += attn[row + d_in] * qkv[tok * three_dim + gvo + d_in];
                    }
                    attn_out[tok * dim + g * head_dim + d_out] = s;
                }
            }
        }

        // proj + residual
        let proj_w: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.fn.proj.weight", prefix))?;
        let proj_b: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.fn.proj.bias", prefix))?;
        let mut out = vec![half::f16::ZERO; n * dim];
        for i in 0..n {
            for o in 0..dim {
                let mut s = proj_b[o];
                let xr = i * dim;
                let wr = o * dim;
                for c in 0..dim { s += attn_out[xr + c] * proj_w[wr + c]; }
                out[i * dim + o] = half::f16::from_f32(x[i * dim + o] + s);
            }
        }
        Tensor::from_slice(&out, Shape::from([n, dim]), DType::F16, device_id)
    }

    /// DaViT FFN with `PreNorm(LN {prefix}.norm, Mlp(fc1→GELU→fc2))` + residual.
    /// nn.GELU default approximation is 'none' = exact: 0.5x(1+erf(x/√2)).
    /// Weight prefixes: `{prefix}.norm.{weight,bias}` [dim],
    /// `{prefix}.fn.net.fc1.{weight,bias}` (W [4*dim, dim]; b [4*dim]),
    /// `{prefix}.fn.net.fc2.{weight,bias}` (W [dim, 4*dim]; b [dim]).
    fn davit_ffn_prenorm(
        &self, input: &Tensor, n: usize, dim: usize, prefix: &str,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let x: Vec<f32> = input.to_f32_vec()?;
        let ln_w: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.norm.weight", prefix))?;
        let ln_b: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.norm.bias", prefix))?;
        let eps = self.config.layer_norm_eps;
        let mut xn = vec![0.0f32; n * dim];
        for p in 0..n {
            let off = p * dim;
            let mut mean = 0.0f32;
            for c in 0..dim { mean += x[off + c]; }
            mean /= dim as f32;
            let mut var = 0.0f32;
            for c in 0..dim { let d = x[off + c] - mean; var += d * d; }
            var /= dim as f32;
            let inv = 1.0f32 / (var + eps).sqrt();
            for c in 0..dim { xn[off + c] = (x[off + c] - mean) * inv * ln_w[c] + ln_b[c]; }
        }
        let hidden_dim = 4 * dim;
        let fc1_w: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.fn.net.fc1.weight", prefix))?;
        let fc1_b: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.fn.net.fc1.bias", prefix))?;
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let mut hbuf = vec![0.0f32; n * hidden_dim];
        for i in 0..n {
            for o in 0..hidden_dim {
                let mut s = fc1_b[o];
                let xr = i * dim;
                let wr = o * dim;
                for c in 0..dim { s += xn[xr + c] * fc1_w[wr + c]; }
                hbuf[i * hidden_dim + o] = 0.5 * s * (1.0 + erf_approx_f32(s * inv_sqrt2));
            }
        }
        let fc2_w: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.fn.net.fc2.weight", prefix))?;
        let fc2_b: Vec<f32> = self.weight_vec_f32(&self.model, &format!("{}.fn.net.fc2.bias", prefix))?;
        let mut out = vec![half::f16::ZERO; n * dim];
        for i in 0..n {
            for o in 0..dim {
                let mut s = fc2_b[o];
                let hr = i * hidden_dim;
                let wr = o * hidden_dim;
                for c in 0..hidden_dim { s += hbuf[hr + c] * fc2_w[wr + c]; }
                // residual: x + mlp(LN(x))
                out[i * dim + o] = half::f16::from_f32(x[i * dim + o] + s);
            }
        }
        Tensor::from_slice(&out, Shape::from([n, dim]), DType::F16, device_id)
    }

    /// DaViT patch embedding: image [3, H, W] → patches [num_patches, dim].
    ///
    /// Implements a patch_size × patch_size convolution (im2col + matmul).
    fn davit_patch_embed(
        &self,
        image_rgb: &[f32],
        width: usize,
        height: usize,
        patch_size: usize,
        dim: usize,
    ) -> Result<Tensor> {
        // Real DaViT conv-embed `vision_tower.convs.0`: Conv2d(3→dim, k, s, p).
        // Implemented as overlapping im2col + linear (a conv = im2col @ Wᵀ).
        // `patch_size` carries the kernel size; stride/padding from config[0].
        let device_id = self.compute.device().info().id;
        let kernel = patch_size;
        let stride = self.config.conv_strides[0];
        let pad = self.config.conv_paddings[0];
        let grid_h = (height + 2 * pad - kernel) / stride + 1;
        let grid_w = (width + 2 * pad - kernel) / stride + 1;
        let num_patches = grid_h * grid_w;
        let patch_pixels = 3 * kernel * kernel; // c_in * kh * kw = 3*7*7 = 147

        // im2col with stride/padding. Column order (c, ky, kx) matches the
        // PyTorch conv weight [c_out, c_in, kh, kw] flattened to [c_out, 147].
        let mut im2col = vec![half::f16::ZERO; num_patches * patch_pixels];
        for oy in 0..grid_h {
            for ox in 0..grid_w {
                let patch_idx = oy * grid_w + ox;
                for c in 0..3 {
                    for ky in 0..kernel {
                        for kx in 0..kernel {
                            let iy = oy * stride + ky;
                            let ix = ox * stride + kx;
                            // valid region is [pad, pad+H) in padded coords
                            if iy >= pad && ix >= pad {
                                let ry = iy - pad;
                                let rx = ix - pad;
                                if ry < height && rx < width {
                                    let col = c * kernel * kernel + ky * kernel + kx;
                                    im2col[patch_idx * patch_pixels + col] =
                                        half::f16::from_f32(image_rgb[c * height * width + ry * width + rx]);
                                }
                            }
                        }
                    }
                }
            }
        }

        let input = Tensor::from_slice(&im2col, Shape::from([num_patches, patch_pixels]), DType::F16, device_id)?;

        let cb = self.compute.new_command_buffer();
        // conv: [num_patches, 147] @ convs.0.proj.weight[128,147]ᵀ + bias
        let conv = self.linear_bias(
            &cb, &self.model, &input,
            "vision_tower.convs.0.proj.weight",
            "vision_tower.convs.0.proj.bias",
            num_patches, patch_pixels, dim,
        )?;
        // convs.0.norm: LayerNorm over the dim channels (channels-last).
        let normed = self.layer_norm(
            &cb, &self.model, &conv,
            "vision_tower.convs.0.norm.weight",
            "vision_tower.convs.0.norm.bias",
            num_patches, dim, self.config.layer_norm_eps,
        )?;
        cb.commit();
        cb.wait_until_completed();
        Ok(normed)
    }

    /// DaViT downsample: stride-2 spatial reduction + channel projection.
    ///
    /// Merges 2×2 spatial patches → 1 patch, then projects from 4*prev_dim → new_dim.
    fn davit_downsample(
        &self,
        input: &Tensor,
        h: usize,
        w: usize,
        prev_dim: usize,
        new_dim: usize,
        stage: usize,
    ) -> Result<Tensor> {
        let new_h = h / 2;
        let new_w = w / 2;
        let num_patches = new_h * new_w;
        let merged_dim = prev_dim * 4;

        // GPU patch merge: [H*W, D] → [H/2 * W/2, 4*D] via patch_merge_concat kernel
        let input_hwd = input.reshape([h, w, prev_dim])?;
        let cb_merge = self.compute.new_command_buffer();
        let merged_tensor = gpu_ops::patch_merge_concat_on(
            &self.compute, &self.kernels.patch_merge_concat, cb_merge.as_ref(),
            &input_hwd, h, w, prev_dim,
        );
        cb_merge.commit();
        cb_merge.wait_until_completed();
        let merged_tensor = merged_tensor.reshape([num_patches, merged_dim])?;

        // Linear projection: [num_patches, 4*prev_dim] → [num_patches, new_dim]
        let w_name = format!("vision_tower.downsamples.{}.reduction.weight", stage - 1);
        let b_name = format!("vision_tower.downsamples.{}.reduction.bias", stage - 1);

        let cb = self.compute.new_command_buffer();
        let norm = self.layer_norm(
            &cb, &self.model, &merged_tensor,
            &format!("vision_tower.downsamples.{}.norm.weight", stage - 1),
            &format!("vision_tower.downsamples.{}.norm.bias", stage - 1),
            num_patches, merged_dim, self.config.layer_norm_eps,
        )?;
        let projected = self.linear_bias(
            &cb, &self.model, &norm,
            &w_name, &b_name,
            num_patches, merged_dim, new_dim,
        )?;
        cb.commit();
        cb.wait_until_completed();

        Ok(projected)
    }

    /// Spatial window attention (Swin-style local windowed self-attention).
    ///
    /// Partitions the [H, W] grid into non-overlapping windows of size window_size × window_size.
    /// Applies multi-head self-attention within each window (CPU implementation).
    fn spatial_window_attention(
        &self,
        input: &Tensor,
        h: usize,
        w: usize,
        dim: usize,
        num_heads: usize,
        head_dim: usize,
        prefix: &str,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let num_tokens = h * w;
        let window_size = self.config.spatial_window.min(h).min(w);
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Pre-norm
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(
            &cb, &self.model, input,
            &format!("{}.spatial_norm.weight", prefix),
            &format!("{}.spatial_norm.bias", prefix),
            num_tokens, dim, self.config.layer_norm_eps,
        )?;

        // QKV projection: [num_tokens, dim] → [num_tokens, 3 * dim]
        let qkv = self.linear_bias(
            &cb, &self.model, &normed,
            &format!("{}.spatial_attn.qkv.weight", prefix),
            &format!("{}.spatial_attn.qkv.bias", prefix),
            num_tokens, dim, 3 * dim,
        )?;
        cb.commit();
        cb.wait_until_completed();

        let qkv_data: Vec<half::f16> = qkv.to_vec()?;

        // Partition into windows and apply attention within each
        let num_win_h = (h + window_size - 1) / window_size;
        let num_win_w = (w + window_size - 1) / window_size;
        let mut output_data = vec![half::f16::ZERO; num_tokens * dim];

        for win_y in 0..num_win_h {
            for win_x in 0..num_win_w {
                let y_start = win_y * window_size;
                let x_start = win_x * window_size;
                let y_end = (y_start + window_size).min(h);
                let x_end = (x_start + window_size).min(w);
                let win_h = y_end - y_start;
                let win_w = x_end - x_start;
                let win_tokens = win_h * win_w;

                // Gather window tokens
                let mut win_indices = Vec::with_capacity(win_tokens);
                for wy in y_start..y_end {
                    for wx in x_start..x_end {
                        win_indices.push(wy * w + wx);
                    }
                }

                // CPU attention within this window
                let attn_out = self.cpu_windowed_attention(
                    &qkv_data, &win_indices, dim, num_heads, head_dim, scale,
                );

                // Scatter back
                for (local_idx, &global_idx) in win_indices.iter().enumerate() {
                    for d in 0..dim {
                        output_data[global_idx * dim + d] = attn_out[local_idx * dim + d];
                    }
                }
            }
        }

        let attn_out = Tensor::from_slice(
            &output_data, Shape::from([num_tokens, dim]), DType::F16, device_id,
        )?;

        // Output projection + residual
        let cb2 = self.compute.new_command_buffer();
        let projected = self.linear_bias(
            &cb2, &self.model, &attn_out,
            &format!("{}.spatial_attn.proj.weight", prefix),
            &format!("{}.spatial_attn.proj.bias", prefix),
            num_tokens, dim, dim,
        )?;
        let residual = self.add(&cb2, input, &projected);
        cb2.commit();
        cb2.wait_until_completed();

        Ok(residual)
    }

    /// CPU windowed self-attention on gathered token indices.
    fn cpu_windowed_attention(
        &self,
        qkv_data: &[half::f16],
        indices: &[usize],
        dim: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Vec<half::f16> {
        let win_size = indices.len();
        let mut out = vec![half::f16::ZERO; win_size * dim];

        for h in 0..num_heads {
            let h_offset = h * head_dim;

            // Compute attention scores
            let mut scores = vec![0.0f32; win_size * win_size];
            for qi in 0..win_size {
                let q_idx = indices[qi];
                for ki in 0..win_size {
                    let k_idx = indices[ki];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        let q = qkv_data[q_idx * 3 * dim + h_offset + d].to_f32();
                        let k = qkv_data[k_idx * 3 * dim + dim + h_offset + d].to_f32();
                        dot += q * k;
                    }
                    scores[qi * win_size + ki] = dot * scale;
                }

                // Softmax over this row
                let row = &mut scores[qi * win_size..(qi + 1) * win_size];
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max_val).exp();
                    sum += *v;
                }
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }

            // Weighted sum of values
            for qi in 0..win_size {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for ki in 0..win_size {
                        let v_idx = indices[ki];
                        let v = qkv_data[v_idx * 3 * dim + 2 * dim + h_offset + d].to_f32();
                        sum += scores[qi * win_size + ki] * v;
                    }
                    out[qi * dim + h_offset + d] = half::f16::from_f32(sum);
                }
            }
        }

        out
    }

    /// Channel group attention (DaViT channel attention).
    ///
    /// Reshapes [H*W, C] → groups spatial positions and attends across channel groups.
    /// Instead of attending over spatial positions (like spatial attention),
    /// this attends over channel dimension groups — each group of n_groups channels
    /// attends to all spatial positions together.
    fn channel_group_attention(
        &self,
        input: &Tensor,
        h: usize,
        w: usize,
        dim: usize,
        num_heads: usize,
        prefix: &str,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let num_tokens = h * w;
        let n_groups = self.config.channel_groups;
        let group_dim = dim / n_groups;

        // Pre-norm
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(
            &cb, &self.model, input,
            &format!("{}.channel_norm.weight", prefix),
            &format!("{}.channel_norm.bias", prefix),
            num_tokens, dim, self.config.layer_norm_eps,
        )?;

        // QKV projection: [num_tokens, dim] → [num_tokens, 3 * dim]
        let qkv = self.linear_bias(
            &cb, &self.model, &normed,
            &format!("{}.channel_attn.qkv.weight", prefix),
            &format!("{}.channel_attn.qkv.bias", prefix),
            num_tokens, dim, 3 * dim,
        )?;
        cb.commit();
        cb.wait_until_completed();

        let qkv_data: Vec<half::f16> = qkv.to_vec()?;

        // Channel group attention: transpose to [n_groups, num_tokens, group_dim]
        // and apply attention across spatial positions within each channel group.
        //
        // For each group g, the "sequence" is the num_tokens spatial positions,
        // and features are group_dim-dimensional.
        let heads_per_group = num_heads / n_groups;
        let head_dim_cg = group_dim / heads_per_group;
        let scale_cg = 1.0 / (head_dim_cg as f32).sqrt();

        let mut output_data = vec![half::f16::ZERO; num_tokens * dim];

        for g in 0..n_groups {
            let ch_start = g * group_dim;

            for hh in 0..heads_per_group {
                let hd_start = hh * head_dim_cg;
                let hd_end = hd_start + head_dim_cg;

                // Compute attention scores across spatial tokens within this head
                let mut scores = vec![0.0f32; num_tokens * num_tokens];
                for qi in 0..num_tokens {
                    for ki in 0..num_tokens {
                        let mut dot = 0.0f32;
                        for d in hd_start..hd_end {
                            let q = qkv_data[qi * 3 * dim + ch_start + d].to_f32();
                            let k = qkv_data[ki * 3 * dim + dim + ch_start + d].to_f32();
                            dot += q * k;
                        }
                        scores[qi * num_tokens + ki] = dot * scale_cg;
                    }

                    // Softmax
                    let row = &mut scores[qi * num_tokens..(qi + 1) * num_tokens];
                    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for v in row.iter_mut() {
                        *v = (*v - max_val).exp();
                        sum += *v;
                    }
                    for v in row.iter_mut() {
                        *v /= sum;
                    }
                }

                // Weighted sum
                for qi in 0..num_tokens {
                    for d in hd_start..hd_end {
                        let mut sum = 0.0f32;
                        for ki in 0..num_tokens {
                            let v = qkv_data[ki * 3 * dim + 2 * dim + ch_start + d].to_f32();
                            sum += scores[qi * num_tokens + ki] * v;
                        }
                        output_data[qi * dim + ch_start + d] = half::f16::from_f32(sum);
                    }
                }
            }
        }

        let attn_out = Tensor::from_slice(
            &output_data, Shape::from([num_tokens, dim]), DType::F16, device_id,
        )?;

        // Output projection + residual
        let cb2 = self.compute.new_command_buffer();
        let projected = self.linear_bias(
            &cb2, &self.model, &attn_out,
            &format!("{}.channel_attn.proj.weight", prefix),
            &format!("{}.channel_attn.proj.bias", prefix),
            num_tokens, dim, dim,
        )?;
        let residual = self.add(&cb2, input, &projected);
        cb2.commit();
        cb2.wait_until_completed();

        Ok(residual)
    }

    // ==================== Vision Projection ====================

    /// BART encoder input construction (M5b).
    ///
    /// Mirrors `Florence2ForConditionalGeneration._merge_input_ids_with_image_features`
    /// + the BART encoder's pre-layer embed/positional/LN sequence:
    ///   token_embeds = shared.embed_tokens(task_ids)        # [T, 768]
    ///   inputs_embeds = cat([image_features [577,768], token_embeds [T,768]], dim=tokens)
    ///   embed_pos = embed_positions(positions_0..N-1)       # offset +2
    ///   hidden = inputs_embeds + embed_pos
    ///   hidden = layernorm_embedding(hidden)
    /// Returns [N=577+T, 768]. Under FLO2_DUMP_DIR, reads `00_input_ids.f32`
    /// from the dump dir and uses those exact task IDs (verification path);
    /// otherwise hard-codes the `<CAPTION>` prompt IDs.
    fn bart_encoder_input(
        &self, image_features: &Tensor, _task: Florence2Task,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let d_model = self.config.d_model; // 768
        let img: Vec<f32> = image_features.to_f32_vec()?;
        let img_tokens = img.len() / d_model;

        // Task input_ids: read from /tmp dump if available; else fall back.
        let task_ids: Vec<u32> = std::env::var("FLO2_DUMP_DIR").ok()
            .and_then(|d| std::fs::read(format!("{}/00_input_ids.f32", d)).ok())
            .map(|b| b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).round() as u32)
                .collect::<Vec<u32>>())
            .unwrap_or_else(|| vec![0, 2264, 473, 5, 2274, 6190, 116, 2]); // <CAPTION>
        let t = task_ids.len();

        // shared.weight [vocab, d_model] — gather rows for task_ids.
        let shared: Vec<f32> = self.weight_vec_f32(
            &self.model, "language_model.model.shared.weight")?;
        let mut text_embeds = vec![0.0f32; t * d_model];
        for (i, &id) in task_ids.iter().enumerate() {
            let src = id as usize * d_model;
            text_embeds[i * d_model..(i + 1) * d_model]
                .copy_from_slice(&shared[src..src + d_model]);
        }

        // Concat [image_features, text_embeds] → [N=img_tokens+t, 768].
        let n = img_tokens + t;
        let mut cat = vec![0.0f32; n * d_model];
        cat[..img.len()].copy_from_slice(&img);
        cat[img.len()..].copy_from_slice(&text_embeds);

        // Learned positional embed with BART offset=2:
        //   embed_pos[i] = embed_positions.weight[i + 2]   for i in 0..N
        let pos_w: Vec<f32> = self.weight_vec_f32(
            &self.model, "language_model.model.encoder.embed_positions.weight")?;
        for i in 0..n {
            let src = (i + 2) * d_model;
            let dst = i * d_model;
            for c in 0..d_model { cat[dst + c] += pos_w[src + c]; }
        }

        // layernorm_embedding.
        let ln_w: Vec<f32> = self.weight_vec_f32(
            &self.model, "language_model.model.encoder.layernorm_embedding.weight")?;
        let ln_b: Vec<f32> = self.weight_vec_f32(
            &self.model, "language_model.model.encoder.layernorm_embedding.bias")?;
        let eps = self.config.layer_norm_eps;
        let mut out = vec![half::f16::ZERO; n * d_model];
        for p in 0..n {
            let off = p * d_model;
            let mut mean = 0.0f32;
            for c in 0..d_model { mean += cat[off + c]; }
            mean /= d_model as f32;
            let mut var = 0.0f32;
            for c in 0..d_model { let d = cat[off + c] - mean; var += d * d; }
            var /= d_model as f32;
            let inv = 1.0f32 / (var + eps).sqrt();
            for c in 0..d_model {
                let v = (cat[off + c] - mean) * inv * ln_w[c] + ln_b[c];
                out[off + c] = half::f16::from_f32(v);
            }
        }

        // Dump for verification vs reference 05b_enc_layernorm_embed.
        let result = Tensor::from_slice(
            &out, Shape::from([n, d_model]), DType::F16, device_id)?;
        if let Ok(dir) = std::env::var("FLO2_DUMP_DIR") {
            if let Ok(v) = result.to_f32_vec() {
                let mut bytes = Vec::with_capacity(v.len() * 4);
                for f in &v { bytes.extend_from_slice(&f.to_le_bytes()); }
                let _ = std::fs::write(
                    format!("{}/rust_05b_enc_layernorm_embed.f32", dir), &bytes);
                tracing::info!("[flo2-diag] wrote rust_05b_enc_layernorm_embed.f32 ({} f32, [{}, {}])", v.len(), n, d_model);
            }
        }
        Ok(result)
    }

    /// BART encoder: run 6 post-norm layers over the LN-embed-input [N, d_model].
    /// Returns the final encoder output. Per-layer dumps for verification when
    /// FLO2_DUMP_DIR is set. Final output also dumped as `05_bart_encoder`.
    fn bart_encoder_layers(&self, input: &Tensor) -> Result<Tensor> {
        let d_model = self.config.d_model;          // 768
        let num_heads = 12usize;                    // BART-base
        let ffn_dim = 3072usize;                    // BART-base encoder_ffn_dim
        let num_layers = 6usize;                    // BART-base encoder_layers
        let dims = input.shape().dims();
        let n = dims[0];

        let dump_dir = std::env::var("FLO2_DUMP_DIR").ok();
        let mut x: Vec<f32> = input.to_f32_vec()?;
        for layer in 0..num_layers {
            let prefix = format!("language_model.model.encoder.layers.{}", layer);
            x = self.bart_encoder_layer(&x, n, d_model, num_heads, ffn_dim, &prefix)?;
            if let Some(dir) = &dump_dir {
                let bytes: Vec<u8> = x.iter().flat_map(|&v| v.to_le_bytes()).collect();
                let _ = std::fs::write(
                    format!("{}/rust_05c_enc_layer{}.f32", dir, layer), &bytes);
            }
        }
        // Final encoder output is the last layer's output (no extra norm in BART).
        if let Some(dir) = &dump_dir {
            let bytes: Vec<u8> = x.iter().flat_map(|&v| v.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{}/rust_05_bart_encoder.f32", dir), &bytes);
            tracing::info!("[flo2-diag] wrote rust_05_bart_encoder.f32 ({} f32, [{}, {}])", x.len(), n, d_model);
        }

        let device_id = self.compute.device().info().id;
        let out: Vec<half::f16> = x.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&out, Shape::from([n, d_model]), DType::F16, device_id)
    }

    /// Single BART encoder layer (post-norm):
    ///   residual = x
    ///   x = self_attn(x)
    ///   x = self_attn_layer_norm(residual + x)
    ///   residual = x
    ///   x = fc2(gelu(fc1(x)))
    ///   x = final_layer_norm(residual + x)
    /// CPU correctness-first. Takes/returns a flat [N*d_model] f32 buffer.
    fn bart_encoder_layer(
        &self, x_in: &[f32], n: usize, d_model: usize,
        num_heads: usize, ffn_dim: usize, prefix: &str,
    ) -> Result<Vec<f32>> {
        let head_dim = d_model / num_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let eps = self.config.layer_norm_eps;

        // ---- Self-attention ----
        // q, k, v = Linear(d_model, d_model) with bias.
        let q_w = self.weight_vec_f32(&self.model, &format!("{}.self_attn.q_proj.weight", prefix))?;
        let q_b = self.weight_vec_f32(&self.model, &format!("{}.self_attn.q_proj.bias", prefix))?;
        let k_w = self.weight_vec_f32(&self.model, &format!("{}.self_attn.k_proj.weight", prefix))?;
        let k_b = self.weight_vec_f32(&self.model, &format!("{}.self_attn.k_proj.bias", prefix))?;
        let v_w = self.weight_vec_f32(&self.model, &format!("{}.self_attn.v_proj.weight", prefix))?;
        let v_b = self.weight_vec_f32(&self.model, &format!("{}.self_attn.v_proj.bias", prefix))?;
        let o_w = self.weight_vec_f32(&self.model, &format!("{}.self_attn.out_proj.weight", prefix))?;
        let o_b = self.weight_vec_f32(&self.model, &format!("{}.self_attn.out_proj.bias", prefix))?;

        // Project: q, k, v ∈ [n, d_model]. Weight row-major [d_model_out, d_model_in].
        let proj = |w: &[f32], b: &[f32], src: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; n * d_model];
            for i in 0..n {
                for o in 0..d_model {
                    let mut s = b[o];
                    let wr = o * d_model;
                    let xr = i * d_model;
                    for c in 0..d_model { s += src[xr + c] * w[wr + c]; }
                    out[i * d_model + o] = s;
                }
            }
            out
        };
        let q = proj(&q_w, &q_b, x_in);
        let k = proj(&k_w, &k_b, x_in);
        let v = proj(&v_w, &v_b, x_in);

        // Multi-head attention: per head, scores [n, n] = scale * q[h] @ k[h]^T,
        // softmax along last, out = scores @ v[h].
        let mut attn_out = vec![0.0f32; n * d_model];
        let mut scores = vec![0.0f32; n * n];
        for h in 0..num_heads {
            let hoff = h * head_dim;
            // scores[i, j] = scale * Σ_d q[i, hoff+d] * k[j, hoff+d]
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0f32;
                    let qb = i * d_model + hoff;
                    let kb = j * d_model + hoff;
                    for d in 0..head_dim { s += q[qb + d] * k[kb + d]; }
                    scores[i * n + j] = s * scale;
                }
            }
            // softmax over j per row i
            for i in 0..n {
                let row = i * n;
                let mut m = f32::NEG_INFINITY;
                for j in 0..n { if scores[row + j] > m { m = scores[row + j]; } }
                let mut sum = 0.0f32;
                for j in 0..n {
                    let e = (scores[row + j] - m).exp();
                    scores[row + j] = e; sum += e;
                }
                let inv = 1.0f32 / sum;
                for j in 0..n { scores[row + j] *= inv; }
            }
            // out[i, hoff+d] = Σ_j scores[i, j] * v[j, hoff+d]
            for i in 0..n {
                let row = i * n;
                for d in 0..head_dim {
                    let mut s = 0.0f32;
                    for j in 0..n {
                        let vb = j * d_model + hoff;
                        s += scores[row + j] * v[vb + d];
                    }
                    attn_out[i * d_model + hoff + d] = s;
                }
            }
        }
        // out_proj
        let attn_proj = proj(&o_w, &o_b, &attn_out);

        // Residual + LN (post-norm).
        let mut x: Vec<f32> = vec![0.0f32; n * d_model];
        let ln1_w = self.weight_vec_f32(&self.model, &format!("{}.self_attn_layer_norm.weight", prefix))?;
        let ln1_b = self.weight_vec_f32(&self.model, &format!("{}.self_attn_layer_norm.bias", prefix))?;
        for p in 0..n {
            let off = p * d_model;
            let mut tmp = vec![0.0f32; d_model];
            for c in 0..d_model { tmp[c] = x_in[off + c] + attn_proj[off + c]; }
            let mut mean = 0.0f32;
            for c in 0..d_model { mean += tmp[c]; }
            mean /= d_model as f32;
            let mut var = 0.0f32;
            for c in 0..d_model { let d = tmp[c] - mean; var += d * d; }
            var /= d_model as f32;
            let inv = 1.0f32 / (var + eps).sqrt();
            for c in 0..d_model {
                x[off + c] = (tmp[c] - mean) * inv * ln1_w[c] + ln1_b[c];
            }
        }

        // ---- FFN ----
        let fc1_w = self.weight_vec_f32(&self.model, &format!("{}.fc1.weight", prefix))?;
        let fc1_b = self.weight_vec_f32(&self.model, &format!("{}.fc1.bias", prefix))?;
        let fc2_w = self.weight_vec_f32(&self.model, &format!("{}.fc2.weight", prefix))?;
        let fc2_b = self.weight_vec_f32(&self.model, &format!("{}.fc2.bias", prefix))?;
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let mut hbuf = vec![0.0f32; n * ffn_dim];
        for i in 0..n {
            for o in 0..ffn_dim {
                let mut s = fc1_b[o];
                let wr = o * d_model;
                let xr = i * d_model;
                for c in 0..d_model { s += x[xr + c] * fc1_w[wr + c]; }
                // exact GELU (A&S erf)
                hbuf[i * ffn_dim + o] = 0.5 * s * (1.0 + erf_approx_f32(s * inv_sqrt2));
            }
        }
        let mut ffn_out = vec![0.0f32; n * d_model];
        for i in 0..n {
            for o in 0..d_model {
                let mut s = fc2_b[o];
                let wr = o * ffn_dim;
                let hr = i * ffn_dim;
                for c in 0..ffn_dim { s += hbuf[hr + c] * fc2_w[wr + c]; }
                ffn_out[i * d_model + o] = s;
            }
        }

        // Residual + final LN.
        let ln2_w = self.weight_vec_f32(&self.model, &format!("{}.final_layer_norm.weight", prefix))?;
        let ln2_b = self.weight_vec_f32(&self.model, &format!("{}.final_layer_norm.bias", prefix))?;
        let mut out = vec![0.0f32; n * d_model];
        for p in 0..n {
            let off = p * d_model;
            let mut tmp = vec![0.0f32; d_model];
            for c in 0..d_model { tmp[c] = x[off + c] + ffn_out[off + c]; }
            let mut mean = 0.0f32;
            for c in 0..d_model { mean += tmp[c]; }
            mean /= d_model as f32;
            let mut var = 0.0f32;
            for c in 0..d_model { let d = tmp[c] - mean; var += d * d; }
            var /= d_model as f32;
            let inv = 1.0f32 / (var + eps).sqrt();
            for c in 0..d_model {
                out[off + c] = (tmp[c] - mean) * inv * ln2_w[c] + ln2_b[c];
            }
        }
        Ok(out)
    }

    /// BART decoder for a single generation step. Takes encoder output
    /// `enc_out [n_enc, d_model]` (M5c verified) and decoder input ids
    /// (typically `[2]` for step 0, `[2, prev1, prev2, ...]` for later steps).
    /// Runs the full decoder forward (input embed + 6 post-norm layers with
    /// causal self-attn + cross-attn + FFN) + LM head, returning the logits
    /// for the LAST token in the sequence (the next-token distribution).
    /// Dumps per-layer outputs `06c_dec_layer{i}_step1` and lm-head logits.
    fn bart_decoder_step(
        &self, enc_out: &Tensor, dec_input_ids: &[u32],
    ) -> Result<Vec<f32>> {
        let d_model = self.config.d_model;          // 768
        let num_heads = 12usize;
        let head_dim = d_model / num_heads;         // 64
        let ffn_dim = 3072usize;
        let num_layers = 6usize;
        let eps = self.config.layer_norm_eps;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let n = dec_input_ids.len();

        let enc: Vec<f32> = enc_out.to_f32_vec()?;
        let n_enc = enc.len() / d_model;

        // --- Decoder input embedding (shared.embed_tokens + positions + LN) ---
        let shared: Vec<f32> = self.weight_vec_f32(
            &self.model, "language_model.model.shared.weight")?;
        let pos_w: Vec<f32> = self.weight_vec_f32(
            &self.model, "language_model.model.decoder.embed_positions.weight")?;
        let mut x = vec![0.0f32; n * d_model];
        for (i, &id) in dec_input_ids.iter().enumerate() {
            let src = id as usize * d_model;
            let dst = i * d_model;
            for c in 0..d_model { x[dst + c] = shared[src + c]; }
        }
        // positions 0..n, offset +2
        for i in 0..n {
            let src = (i + 2) * d_model;
            for c in 0..d_model { x[i * d_model + c] += pos_w[src + c]; }
        }
        // layernorm_embedding
        let ln_em_w: Vec<f32> = self.weight_vec_f32(
            &self.model, "language_model.model.decoder.layernorm_embedding.weight")?;
        let ln_em_b: Vec<f32> = self.weight_vec_f32(
            &self.model, "language_model.model.decoder.layernorm_embedding.bias")?;
        for p in 0..n {
            let off = p * d_model;
            let mut mean = 0.0f32;
            for c in 0..d_model { mean += x[off + c]; }
            mean /= d_model as f32;
            let mut var = 0.0f32;
            for c in 0..d_model { let d = x[off + c] - mean; var += d * d; }
            var /= d_model as f32;
            let inv = 1.0f32 / (var + eps).sqrt();
            for c in 0..d_model {
                x[off + c] = (x[off + c] - mean) * inv * ln_em_w[c] + ln_em_b[c];
            }
        }

        let dump_dir = std::env::var("FLO2_DUMP_DIR").ok();
        let write_dump = |name: &str, buf: &[f32]| {
            if let Some(dir) = dump_dir.as_ref() {
                let bytes: Vec<u8> = buf.iter().flat_map(|&v| v.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{}/rust_{}.f32", dir, name), &bytes);
            }
        };

        // --- Decoder layers ---
        for layer in 0..num_layers {
            let prefix = format!("language_model.model.decoder.layers.{}", layer);

            // -- Causal self-attention --
            let q_w = self.weight_vec_f32(&self.model, &format!("{}.self_attn.q_proj.weight", prefix))?;
            let q_b = self.weight_vec_f32(&self.model, &format!("{}.self_attn.q_proj.bias", prefix))?;
            let k_w = self.weight_vec_f32(&self.model, &format!("{}.self_attn.k_proj.weight", prefix))?;
            let k_b = self.weight_vec_f32(&self.model, &format!("{}.self_attn.k_proj.bias", prefix))?;
            let v_w = self.weight_vec_f32(&self.model, &format!("{}.self_attn.v_proj.weight", prefix))?;
            let v_b = self.weight_vec_f32(&self.model, &format!("{}.self_attn.v_proj.bias", prefix))?;
            let o_w = self.weight_vec_f32(&self.model, &format!("{}.self_attn.out_proj.weight", prefix))?;
            let o_b = self.weight_vec_f32(&self.model, &format!("{}.self_attn.out_proj.bias", prefix))?;
            let proj_n = |w: &[f32], b: &[f32], src: &[f32], n_rows: usize, d_in: usize, d_out: usize| -> Vec<f32> {
                let mut out = vec![0.0f32; n_rows * d_out];
                for i in 0..n_rows {
                    for o in 0..d_out {
                        let mut s = b[o];
                        let wr = o * d_in;
                        let xr = i * d_in;
                        for c in 0..d_in { s += src[xr + c] * w[wr + c]; }
                        out[i * d_out + o] = s;
                    }
                }
                out
            };
            let q = proj_n(&q_w, &q_b, &x, n, d_model, d_model);
            let k = proj_n(&k_w, &k_b, &x, n, d_model, d_model);
            let v = proj_n(&v_w, &v_b, &x, n, d_model, d_model);

            // MHA with causal mask (only attend to positions <= i).
            let mut self_out = vec![0.0f32; n * d_model];
            let mut sc = vec![0.0f32; n * n];
            for h in 0..num_heads {
                let hoff = h * head_dim;
                for i in 0..n {
                    for j in 0..n {
                        if j > i { sc[i * n + j] = f32::NEG_INFINITY; continue; }
                        let mut s = 0.0f32;
                        let qb = i * d_model + hoff;
                        let kb = j * d_model + hoff;
                        for d in 0..head_dim { s += q[qb + d] * k[kb + d]; }
                        sc[i * n + j] = s * scale;
                    }
                }
                for i in 0..n {
                    let row = i * n;
                    let mut m = f32::NEG_INFINITY;
                    for j in 0..n { if sc[row + j] > m { m = sc[row + j]; } }
                    let mut sum = 0.0f32;
                    for j in 0..n {
                        let e = if sc[row + j] == f32::NEG_INFINITY { 0.0 } else { (sc[row + j] - m).exp() };
                        sc[row + j] = e; sum += e;
                    }
                    let inv = 1.0f32 / sum;
                    for j in 0..n { sc[row + j] *= inv; }
                }
                for i in 0..n {
                    let row = i * n;
                    for d in 0..head_dim {
                        let mut s = 0.0f32;
                        for j in 0..n {
                            s += sc[row + j] * v[j * d_model + hoff + d];
                        }
                        self_out[i * d_model + hoff + d] = s;
                    }
                }
            }
            let self_proj = proj_n(&o_w, &o_b, &self_out, n, d_model, d_model);
            // Residual + LN.
            let ln1_w = self.weight_vec_f32(&self.model, &format!("{}.self_attn_layer_norm.weight", prefix))?;
            let ln1_b = self.weight_vec_f32(&self.model, &format!("{}.self_attn_layer_norm.bias", prefix))?;
            for p in 0..n {
                let off = p * d_model;
                let mut tmp = vec![0.0f32; d_model];
                for c in 0..d_model { tmp[c] = x[off + c] + self_proj[off + c]; }
                let mut mean = 0.0f32;
                for c in 0..d_model { mean += tmp[c]; }
                mean /= d_model as f32;
                let mut var = 0.0f32;
                for c in 0..d_model { let d = tmp[c] - mean; var += d * d; }
                var /= d_model as f32;
                let inv = 1.0f32 / (var + eps).sqrt();
                for c in 0..d_model {
                    x[off + c] = (tmp[c] - mean) * inv * ln1_w[c] + ln1_b[c];
                }
            }

            // -- Cross-attention to encoder --
            let cq_w = self.weight_vec_f32(&self.model, &format!("{}.encoder_attn.q_proj.weight", prefix))?;
            let cq_b = self.weight_vec_f32(&self.model, &format!("{}.encoder_attn.q_proj.bias", prefix))?;
            let ck_w = self.weight_vec_f32(&self.model, &format!("{}.encoder_attn.k_proj.weight", prefix))?;
            let ck_b = self.weight_vec_f32(&self.model, &format!("{}.encoder_attn.k_proj.bias", prefix))?;
            let cv_w = self.weight_vec_f32(&self.model, &format!("{}.encoder_attn.v_proj.weight", prefix))?;
            let cv_b = self.weight_vec_f32(&self.model, &format!("{}.encoder_attn.v_proj.bias", prefix))?;
            let co_w = self.weight_vec_f32(&self.model, &format!("{}.encoder_attn.out_proj.weight", prefix))?;
            let co_b = self.weight_vec_f32(&self.model, &format!("{}.encoder_attn.out_proj.bias", prefix))?;
            let cq = proj_n(&cq_w, &cq_b, &x, n, d_model, d_model);
            let ck = proj_n(&ck_w, &ck_b, &enc, n_enc, d_model, d_model);
            let cv = proj_n(&cv_w, &cv_b, &enc, n_enc, d_model, d_model);
            let mut cross_out = vec![0.0f32; n * d_model];
            let mut cs = vec![0.0f32; n * n_enc];
            for h in 0..num_heads {
                let hoff = h * head_dim;
                for i in 0..n {
                    for j in 0..n_enc {
                        let mut s = 0.0f32;
                        let qb = i * d_model + hoff;
                        let kb = j * d_model + hoff;
                        for d in 0..head_dim { s += cq[qb + d] * ck[kb + d]; }
                        cs[i * n_enc + j] = s * scale;
                    }
                }
                for i in 0..n {
                    let row = i * n_enc;
                    let mut m = f32::NEG_INFINITY;
                    for j in 0..n_enc { if cs[row + j] > m { m = cs[row + j]; } }
                    let mut sum = 0.0f32;
                    for j in 0..n_enc {
                        let e = (cs[row + j] - m).exp();
                        cs[row + j] = e; sum += e;
                    }
                    let inv = 1.0f32 / sum;
                    for j in 0..n_enc { cs[row + j] *= inv; }
                }
                for i in 0..n {
                    let row = i * n_enc;
                    for d in 0..head_dim {
                        let mut s = 0.0f32;
                        for j in 0..n_enc {
                            s += cs[row + j] * cv[j * d_model + hoff + d];
                        }
                        cross_out[i * d_model + hoff + d] = s;
                    }
                }
            }
            let cross_proj = proj_n(&co_w, &co_b, &cross_out, n, d_model, d_model);
            // Residual + LN.
            let ln2_w = self.weight_vec_f32(&self.model, &format!("{}.encoder_attn_layer_norm.weight", prefix))?;
            let ln2_b = self.weight_vec_f32(&self.model, &format!("{}.encoder_attn_layer_norm.bias", prefix))?;
            for p in 0..n {
                let off = p * d_model;
                let mut tmp = vec![0.0f32; d_model];
                for c in 0..d_model { tmp[c] = x[off + c] + cross_proj[off + c]; }
                let mut mean = 0.0f32;
                for c in 0..d_model { mean += tmp[c]; }
                mean /= d_model as f32;
                let mut var = 0.0f32;
                for c in 0..d_model { let d = tmp[c] - mean; var += d * d; }
                var /= d_model as f32;
                let inv = 1.0f32 / (var + eps).sqrt();
                for c in 0..d_model {
                    x[off + c] = (tmp[c] - mean) * inv * ln2_w[c] + ln2_b[c];
                }
            }

            // -- FFN --
            let fc1_w = self.weight_vec_f32(&self.model, &format!("{}.fc1.weight", prefix))?;
            let fc1_b = self.weight_vec_f32(&self.model, &format!("{}.fc1.bias", prefix))?;
            let fc2_w = self.weight_vec_f32(&self.model, &format!("{}.fc2.weight", prefix))?;
            let fc2_b = self.weight_vec_f32(&self.model, &format!("{}.fc2.bias", prefix))?;
            let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
            let mut hbuf = vec![0.0f32; n * ffn_dim];
            for i in 0..n {
                for o in 0..ffn_dim {
                    let mut s = fc1_b[o];
                    let wr = o * d_model;
                    let xr = i * d_model;
                    for c in 0..d_model { s += x[xr + c] * fc1_w[wr + c]; }
                    hbuf[i * ffn_dim + o] = 0.5 * s * (1.0 + erf_approx_f32(s * inv_sqrt2));
                }
            }
            let mut ffn_out = vec![0.0f32; n * d_model];
            for i in 0..n {
                for o in 0..d_model {
                    let mut s = fc2_b[o];
                    let wr = o * ffn_dim;
                    let hr = i * ffn_dim;
                    for c in 0..ffn_dim { s += hbuf[hr + c] * fc2_w[wr + c]; }
                    ffn_out[i * d_model + o] = s;
                }
            }
            let lnf_w = self.weight_vec_f32(&self.model, &format!("{}.final_layer_norm.weight", prefix))?;
            let lnf_b = self.weight_vec_f32(&self.model, &format!("{}.final_layer_norm.bias", prefix))?;
            for p in 0..n {
                let off = p * d_model;
                let mut tmp = vec![0.0f32; d_model];
                for c in 0..d_model { tmp[c] = x[off + c] + ffn_out[off + c]; }
                let mut mean = 0.0f32;
                for c in 0..d_model { mean += tmp[c]; }
                mean /= d_model as f32;
                let mut var = 0.0f32;
                for c in 0..d_model { let d = tmp[c] - mean; var += d * d; }
                var /= d_model as f32;
                let inv = 1.0f32 / (var + eps).sqrt();
                for c in 0..d_model {
                    x[off + c] = (tmp[c] - mean) * inv * lnf_w[c] + lnf_b[c];
                }
            }

            write_dump(&format!("06c_dec_layer{}_step1", layer), &x);
        }
        write_dump("06_bart_decoder_step1", &x);

        // --- LM head: x @ shared^T + final_logits_bias ---
        let vocab = shared.len() / d_model; // 51289
        let bias: Vec<f32> = self.weight_vec_f32(
            &self.model, "language_model.final_logits_bias")?;
        let last = n - 1; // logits for the LAST token
        let xr = last * d_model;
        let mut logits = vec![0.0f32; vocab];
        for o in 0..vocab {
            let mut s = bias[o];
            let wr = o * d_model;
            for c in 0..d_model { s += x[xr + c] * shared[wr + c]; }
            logits[o] = s;
        }
        write_dump("07_lm_head_logits_step1", &logits);
        tracing::info!("[flo2-diag] BART step1 argmax={}", argmax_f32(&logits));
        Ok(logits)
    }

    /// Project vision features from final DaViT dim to text decoder dim.
    ///
    /// Linear(1024, 768): [num_tokens, vision_dim] → [num_tokens, d_model]
    fn project_vision(&self, vision_features: &Tensor) -> Result<Tensor> {
        let config = &self.config;
        let final_vision_dim = config.vision_dims[3]; // 1024
        let num_tokens = vision_features.numel() / final_vision_dim;

        let cb = self.compute.new_command_buffer();
        let projected = self.linear_bias(
            &cb, &self.model, vision_features,
            "projection.weight",
            "projection.bias",
            num_tokens, final_vision_dim, config.projection_dim,
        )?;
        cb.commit();
        cb.wait_until_completed();

        debug!(num_tokens, projection_dim = config.projection_dim, "vision projected");
        Ok(projected)
    }

    // ==================== Text Decoder ====================

    /// Autoregressive text decoder with cross-attention to vision features.
    ///
    /// Architecture per layer:
    ///   1. Causal self-attention (masked)
    ///   2. Cross-attention to projected vision features
    ///   3. FFN (Linear → GELU → Linear)
    ///
    /// Starts with task prefix tokens, generates until EOS or max_seq_len.
    fn decode(&self, vision_features: &Tensor, task: Florence2Task) -> Result<Vec<u32>> {
        let config = &self.config;
        let d_model = config.d_model;
        let num_heads = config.decoder_heads;
        let head_dim = d_model / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let vision_seq = vision_features.numel() / d_model;
        let device_id = self.compute.device().info().id;

        // Task prefix token (simplified: use a single task token ID)
        let task_token_id = self.task_token_id(task);
        let bos_id = 2u32; // <s> token
        let eos_id = 2u32; // </s> token (same as BOS for BART-based models)
        let pad_id = 1u32;

        // Start sequence: [BOS, task_token]
        let mut token_ids: Vec<u32> = vec![bos_id, task_token_id];
        let max_new_tokens = config.max_seq_len.min(512);

        // Load word embeddings
        let embed_weight = self.weight_f16(&self.model, "text_decoder.embed_tokens.weight")?;
        let embed_data: Vec<half::f16> = embed_weight.to_vec()?;

        // Load positional embeddings
        let pos_embed_weight = self.weight_f16(&self.model, "text_decoder.embed_positions.weight")?;
        let pos_data: Vec<half::f16> = pos_embed_weight.to_vec()?;

        debug!(vision_seq, task = ?task, "decoding");

        // Autoregressive loop
        for _step in 0..max_new_tokens {
            let seq_len = token_ids.len();

            // Embed all tokens (word + position)
            let mut hidden_data = vec![half::f16::ZERO; seq_len * d_model];
            for (pos, &tid) in token_ids.iter().enumerate() {
                let tid = tid as usize;
                for d in 0..d_model {
                    let word = if tid < config.vocab_size {
                        embed_data[tid * d_model + d].to_f32()
                    } else {
                        0.0
                    };
                    let positional = if pos < config.max_seq_len {
                        pos_data[pos * d_model + d].to_f32()
                    } else {
                        0.0
                    };
                    hidden_data[pos * d_model + d] = half::f16::from_f32(word + positional);
                }
            }

            let mut hidden = Tensor::from_slice(
                &hidden_data, Shape::from([seq_len, d_model]), DType::F16, device_id,
            )?;

            // Run through decoder layers
            for layer in 0..config.decoder_layers {
                hidden = self.decoder_layer(
                    &hidden, vision_features, layer,
                    seq_len, vision_seq, d_model, num_heads, head_dim, scale,
                )?;
            }

            // Final layer norm
            let cb = self.compute.new_command_buffer();
            let normed = self.layer_norm(
                &cb, &self.model, &hidden,
                "text_decoder.layer_norm.weight",
                "text_decoder.layer_norm.bias",
                seq_len, d_model, config.layer_norm_eps,
            )?;
            cb.commit();
            cb.wait_until_completed();

            // LM head: project last token to vocabulary logits
            let last_hidden = normed.slice(0, seq_len - 1, seq_len)?;
            let logits = self.lm_head(&last_hidden, &embed_weight, d_model)?;

            // Greedy sampling
            let next_token = argmax_f32(&logits);
            if next_token == eos_id || next_token == pad_id {
                break;
            }
            token_ids.push(next_token);
        }

        // Return generated tokens (skip BOS + task prefix)
        let output_start = 2.min(token_ids.len());
        Ok(token_ids[output_start..].to_vec())
    }

    /// Single decoder layer: self-attn → cross-attn → FFN.
    fn decoder_layer(
        &self,
        hidden: &Tensor,
        vision_features: &Tensor,
        layer: usize,
        seq_len: usize,
        vision_seq: usize,
        d_model: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Tensor> {
        let prefix = format!("text_decoder.layers.{}", layer);
        let eps = self.config.layer_norm_eps;

        // 1. Causal self-attention
        let cb = self.compute.new_command_buffer();
        let normed_sa = self.layer_norm(
            &cb, &self.model, hidden,
            &format!("{}.self_attn_layer_norm.weight", prefix),
            &format!("{}.self_attn_layer_norm.bias", prefix),
            seq_len, d_model, eps,
        )?;

        let q = self.linear_bias(
            &cb, &self.model, &normed_sa,
            &format!("{}.self_attn.q_proj.weight", prefix),
            &format!("{}.self_attn.q_proj.bias", prefix),
            seq_len, d_model, d_model,
        )?;
        let k = self.linear_bias(
            &cb, &self.model, &normed_sa,
            &format!("{}.self_attn.k_proj.weight", prefix),
            &format!("{}.self_attn.k_proj.bias", prefix),
            seq_len, d_model, d_model,
        )?;
        let v = self.linear_bias(
            &cb, &self.model, &normed_sa,
            &format!("{}.self_attn.v_proj.weight", prefix),
            &format!("{}.self_attn.v_proj.bias", prefix),
            seq_len, d_model, d_model,
        )?;
        cb.commit();
        cb.wait_until_completed();

        // Causal self-attention (CPU for causal mask)
        let self_attn_out = self.cpu_causal_attention(
            &q, &k, &v, seq_len, d_model, num_heads, head_dim, scale,
        )?;

        let cb2 = self.compute.new_command_buffer();
        let sa_proj = self.linear_bias(
            &cb2, &self.model, &self_attn_out,
            &format!("{}.self_attn.out_proj.weight", prefix),
            &format!("{}.self_attn.out_proj.bias", prefix),
            seq_len, d_model, d_model,
        )?;
        let h1 = self.add(&cb2, hidden, &sa_proj);

        // 2. Cross-attention to vision features
        let normed_ca = self.layer_norm(
            &cb2, &self.model, &h1,
            &format!("{}.encoder_attn_layer_norm.weight", prefix),
            &format!("{}.encoder_attn_layer_norm.bias", prefix),
            seq_len, d_model, eps,
        )?;

        let q_ca = self.linear_bias(
            &cb2, &self.model, &normed_ca,
            &format!("{}.encoder_attn.q_proj.weight", prefix),
            &format!("{}.encoder_attn.q_proj.bias", prefix),
            seq_len, d_model, d_model,
        )?;
        let k_ca = self.linear_bias(
            &cb2, &self.model, vision_features,
            &format!("{}.encoder_attn.k_proj.weight", prefix),
            &format!("{}.encoder_attn.k_proj.bias", prefix),
            vision_seq, d_model, d_model,
        )?;
        let v_ca = self.linear_bias(
            &cb2, &self.model, vision_features,
            &format!("{}.encoder_attn.v_proj.weight", prefix),
            &format!("{}.encoder_attn.v_proj.bias", prefix),
            vision_seq, d_model, d_model,
        )?;

        // Cross-attention via batched GPU attention (no causal mask needed)
        let ca_out = self.batched_attention(
            &cb2, &q_ca, &k_ca, &v_ca,
            seq_len, vision_seq, num_heads, head_dim, scale,
        )?;

        let ca_proj = self.linear_bias(
            &cb2, &self.model, &ca_out,
            &format!("{}.encoder_attn.out_proj.weight", prefix),
            &format!("{}.encoder_attn.out_proj.bias", prefix),
            seq_len, d_model, d_model,
        )?;
        let h2 = self.add(&cb2, &h1, &ca_proj);

        // 3. FFN: LayerNorm → Linear → GELU → Linear + residual
        let normed_ff = self.layer_norm(
            &cb2, &self.model, &h2,
            &format!("{}.final_layer_norm.weight", prefix),
            &format!("{}.final_layer_norm.bias", prefix),
            seq_len, d_model, eps,
        )?;
        let ff_up = self.linear_bias(
            &cb2, &self.model, &normed_ff,
            &format!("{}.fc1.weight", prefix),
            &format!("{}.fc1.bias", prefix),
            seq_len, d_model, self.config.decoder_ffn_dim,
        )?;
        let ff_act = self.activation(&cb2, &self.kernels.gelu, &ff_up);
        let ff_down = self.linear_bias(
            &cb2, &self.model, &ff_act,
            &format!("{}.fc2.weight", prefix),
            &format!("{}.fc2.bias", prefix),
            seq_len, self.config.decoder_ffn_dim, d_model,
        )?;
        let h3 = self.add(&cb2, &h2, &ff_down);
        cb2.commit();
        cb2.wait_until_completed();

        Ok(h3)
    }

    /// GPU causal self-attention via batched matmul (replaces CPU O(N²) loops).
    fn cpu_causal_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
        _d_model: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Tensor> {
        // Reshape [seq_len, d_model] → [seq_len, num_heads, head_dim] for batched attention
        let q_shd = q.reshape([seq_len, num_heads, head_dim])?;
        let k_shd = k.reshape([seq_len, num_heads, head_dim])?;
        let v_shd = v.reshape([seq_len, num_heads, head_dim])?;

        let cb = self.compute.new_command_buffer();
        let result = self.batched_attention(
            cb.as_ref(), &q_shd, &k_shd, &v_shd,
            seq_len, seq_len, num_heads, head_dim, scale,
        )?;
        cb.commit();
        cb.wait_until_completed();
        Ok(result)
    }

    /// LM head: project hidden state to vocab logits on GPU.
    ///
    /// Uses tied embeddings: logits = hidden @ embed_weight^T (no bias).
    fn lm_head(&self, last_hidden: &Tensor, embed_weight: &Tensor, d_model: usize) -> Result<Vec<f32>> {
        let config = &self.config;
        let device_id = self.compute.device().info().id;

        // GPU matmul: [1, d_model] @ [vocab_size, d_model]^T → [1, vocab_size]
        let zero_bias = Tensor::empty(Shape::from([config.vocab_size]), DType::F16, device_id)?;
        let cb = self.compute.new_command_buffer();
        let logits = self.linear_tensors(
            cb.as_ref(), last_hidden, embed_weight, &zero_bias,
            1, d_model, config.vocab_size,
        );
        cb.commit();
        cb.wait_until_completed();
        logits.to_f32_vec()
    }

    // ==================== Task Token Mapping ====================

    /// Map task to a special token ID.
    ///
    /// Florence-2 uses special tokens added beyond the base vocabulary.
    /// These IDs correspond to the tokenizer's added_tokens for each task.
    fn task_token_id(&self, task: Florence2Task) -> u32 {
        let base = self.config.vocab_size as u32;
        match task {
            Florence2Task::Caption => base.saturating_sub(10),
            Florence2Task::DetailedCaption => base.saturating_sub(9),
            Florence2Task::Ocr => base.saturating_sub(8),
            Florence2Task::ObjectDetection => base.saturating_sub(7),
            Florence2Task::Grounding => base.saturating_sub(6),
        }
    }

    // ==================== Detokenization ====================

    /// Simple detokenization: convert token IDs to string.
    ///
    /// In a full implementation this would use the BPE tokenizer.
    /// For now, returns a placeholder with the token IDs for downstream processing.
    fn detokenize(&self, token_ids: &[u32]) -> String {
        // Florence-2 uses a BART-style (Roberta) byte-level BPE tokenizer.
        // We load vocab.json (token_string → id) from FLO2_TOKENIZER_VOCAB or
        // the default location, build the inverse map, and decode IDs by
        // concatenating token strings and replacing 'Ġ' (U+0120, byte-level
        // BPE's space marker) with a literal space. <s>(0), </s>(2), <pad>(1)
        // and the task tokens are stripped. Caches the inverse map on first
        // call (one-shot init).
        use std::sync::OnceLock;
        static INV_VOCAB: OnceLock<Vec<String>> = OnceLock::new();
        let inv = INV_VOCAB.get_or_init(|| {
            let path = std::env::var("FLO2_TOKENIZER_VOCAB").ok().unwrap_or_else(|| {
                "/Users/dcharlot/sites/efficient-genai/diag-scripts/florence2_vocab.json".to_string()
            });
            match std::fs::read_to_string(&path) {
                Ok(s) => match serde_json::from_str::<serde_json::Value>(&s) {
                    Ok(v) => {
                        let obj = v.as_object().cloned().unwrap_or_default();
                        let max_id = obj.values()
                            .filter_map(|x| x.as_u64()).max().unwrap_or(0) as usize;
                        let mut inv = vec![String::new(); max_id + 1];
                        for (tok, id) in obj {
                            if let Some(i) = id.as_u64() {
                                if (i as usize) < inv.len() { inv[i as usize] = tok; }
                            }
                        }
                        inv
                    }
                    Err(_) => Vec::new(),
                }
                Err(_) => Vec::new(),
            }
        });

        if inv.is_empty() {
            return token_ids.iter().map(|id| format!("[{}]", id)).collect::<Vec<_>>().join("");
        }
        let mut out = String::new();
        for &id in token_ids {
            let t = id as usize;
            // Skip BART specials: <s>=0, <pad>=1, </s>=2, <unk>=3.
            if t < 4 { continue; }
            if t < inv.len() {
                // Replace byte-level BPE space marker.
                let s = inv[t].replace('Ġ', " ").replace('Ċ', "\n");
                out.push_str(&s);
            }
        }
        // Trim leading whitespace introduced by first Ġ-token.
        out.trim_start().to_string()
    }
}

// ==================== Utilities ====================

/// Abramowitz & Stegun 7.1.26 erf approximation (max abs err ≈ 1.5e-7).
/// Matches PyTorch's exact erf to fp16 precision — used by exact GELU
/// in the DaViT FFN (nn.GELU default `approximate='none'`).
fn erf_approx_f32(x: f32) -> f32 {
    let a1 =  0.254829592_f32;
    let a2 = -0.284496736_f32;
    let a3 =  1.421413741_f32;
    let a4 = -1.453152027_f32;
    let a5 =  1.061405429_f32;
    let p  =  0.3275911_f32;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = x.abs();
    let t  = 1.0 / (1.0 + p * xa);
    let y  = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-xa * xa).exp();
    sign * y
}

/// Greedy argmax over f32 logits.
fn argmax_f32(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}
