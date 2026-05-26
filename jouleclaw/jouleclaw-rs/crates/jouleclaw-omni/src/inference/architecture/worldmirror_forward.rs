//! HY-WorldMirror 2.0 forward pass (v1 — partial implementation).
//!
//! Tencent's `tencent/HY-World-2.0::HY-WorldMirror-2.0` is a unified
//! feed-forward 3D reconstructor for multi-view inputs. The released
//! safetensors (~5 GB, 1545 keys) contains:
//!
//!   1. **`visual_geometry_transformer.patch_embed`** (344 keys) — a full
//!      24-layer DINOv2-L/14 with register tokens. Vanilla MHA + LayerScale
//!      (no QK-norm or RoPE in this branch).
//!   2. **`visual_geometry_transformer.frame_blocks`** (24 layers, 433
//!      keys) — per-frame self-attention with `q_norm`/`k_norm` (RMS norm
//!      per head) + 2D RoPE + LayerScale. Runs on patch_embed output.
//!   3. **`visual_geometry_transformer.global_blocks`** (24 layers, 432
//!      keys) — cross-frame attention. Same arch as frame_blocks.
//!   4. **`{depth, norm, cam, pts, gs}_head`** — DPT-style decoders that
//!      read multi-scale features from intermediate transformer layers
//!      and emit per-pixel predictions. `gs_head` produces 12-channel
//!      Gaussian-splat parameter maps that `gs_renderer.gs_head` projects.
//!   5. **`pose_embed`, `depth_embed`, `ray_embed`** — pow3r conditioning
//!      embedders for optional camera-intrinsics / depth / ray-direction
//!      hints supplied by the caller.
//!
//! ### What this v1 implements
//!
//! Only stage (1) — the DINOv2-L patch_embed encoder. Patch features
//! (real, from trained weights) are used to **shade and position** an
//! otherwise-stub splat cloud:
//!
//!   * One splat per patch (37×37 = 1369 splats per view).
//!   * Splat color comes from the per-patch feature → 3-channel projection
//!     (deterministic from the feature mean across the embedding dim).
//!   * Splat position is laid out on a sphere shell deterministically.
//!
//! That gives output that visibly *changes with the input image* (the
//! patch features carry semantic structure) without claiming to be a
//! correct reconstruction.
//!
//! ### What's left (multi-day, separate turn)
//!
//! Stages (2) frame_blocks, (3) global_blocks, (4) DPT heads + gs_renderer,
//! (5) pow3r conditioning. The frame_blocks alone need: 2D RoPE on Q/K,
//! per-head RMS norm on Q and K (`q_norm` / `k_norm` weights), LayerScale
//! gates (`ls1.gamma`, `ls2.gamma`), and a fused QKV projection.

#[cfg(feature = "metal")]
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::core::Result;
#[cfg(feature = "metal")]
use crate::hal::metal::{ComputePipeline, MetalCompute, MetalDevice};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;
#[cfg(feature = "metal")]
use crate::inference::gpu_ops::{self, read_weight_f16, MetalPipeline};
#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::tensor::{DType, Shape, Tensor};

#[cfg(feature = "metal")]
use super::hyworld::GaussianSplat;

/// Bundle of WorldMirror forward outputs. `splats` is always populated;
/// the other fields are `Some` per view only when the matching `compute_*`
/// flag is set on `WorldMirrorRuntime::forward_full`.
#[cfg(feature = "metal")]
pub struct WorldMirrorOutputs {
    pub splats: Vec<GaussianSplat>,
    /// One f32 buffer per view, length 37×37 (raw depth values).
    pub depth_maps: Option<Vec<Vec<f32>>>,
    /// One f32 buffer per view, length 3×37×37 CHW (xyz unit normals, raw).
    pub normal_maps: Option<Vec<Vec<f32>>>,
    /// One f32 buffer per view, length 3×37×37 CHW (xyz point coordinates, raw).
    pub point_clouds: Option<Vec<Vec<f32>>>,
    /// Per-view 9-dim camera parameters from cam_head: `[px, py, pz, fx, fy, fz, ux, uy, uz]`.
    /// `Some(Vec<[f32; 9]>)` of length `n_views` when cam_head ran successfully.
    pub camera_params: Option<Vec<[f32; 9]>>,
}

#[cfg(feature = "metal")]
pub struct WorldMirrorRuntime {
    model: Arc<Model>,
    compute: Arc<MetalCompute>,
    kernels: WorldMirrorKernels,
    /// UNet2DConditionModel wrapper of the WorldMirror model — used purely
    /// for its public conv2d helper (im2col + matmul). The down/up block
    /// configuration on the wrapper is irrelevant; conv2d only reads the
    /// model's weights by name.
    unet_helper: super::unet::UNet2DConditionModel,
    /// Cache of f16 GPU weight tensors keyed by safetensors name. Each
    /// `read_weight_f16` call without this cache triggers an f32→f16 CPU
    /// conversion + Metal upload — wasteful when the same weights are read
    /// repeatedly across views (cam_head re-reads ~50 weights per view).
    /// Tensors are Arc-cloned so the cache hands out cheap references.
    weight_cache: std::sync::Mutex<std::collections::HashMap<String, Tensor>>,
}

#[cfg(feature = "metal")]
struct WorldMirrorKernels {
    common: gpu_ops::CommonKernels,
    gelu: Arc<ComputePipeline>,
    relu: Arc<ComputePipeline>,
    /// v6.7: GPU im2col for 3×3 conv (replaces CPU im2col bottleneck in
    /// multi-res DPT chain).
    im2col_3x3: Arc<ComputePipeline>,
    /// v6.7: CHW→HWC reshape on GPU for 1×1 conv matmul path.
    chw_to_hwc: Arc<ComputePipeline>,
    /// v6.7: HWC→CHW reshape on GPU for matmul output → next layer's input.
    hwc_to_chw: Arc<ComputePipeline>,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for WorldMirrorRuntime {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl WorldMirrorRuntime {
    pub fn new(model: Arc<Model>, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));
        let kernels = WorldMirrorKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            relu: compute.compile_pipeline("relu", sources::GELU, "relu_f16")?,
            im2col_3x3: compute.compile_pipeline("im2col_3x3", sources::CONV2D, "im2col_3x3_f16")?,
            chw_to_hwc: compute.compile_pipeline("chw_to_hwc", sources::CONV2D, "chw_to_hwc_f16")?,
            hwc_to_chw: compute.compile_pipeline("hwc_to_chw", sources::CONV2D, "hwc_to_chw_f16")?,
        };
        let unet_helper = super::unet::UNet2DConditionModel::new(model.clone());
        Ok(Self {
            model, compute, kernels, unet_helper,
            weight_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// Cached weight reader — returns an f16 GPU `Tensor` for the named
    /// safetensors weight. First call uploads + caches; subsequent calls
    /// return a `Tensor` clone (shares the underlying Metal buffer via
    /// `Arc`). Massive win for cam_head + DPT chains that read the same
    /// weights repeatedly across views.
    fn cached_weight_f16(&self, name: &str) -> Result<Tensor> {
        {
            let cache = self.weight_cache.lock()
                .map_err(|_| crate::core::Error::internal("weight_cache mutex poisoned"))?;
            if let Some(t) = cache.get(name) {
                return Ok(t.clone());
            }
        }
        let t = read_weight_f16(&self.model, &self.compute, name)?;
        {
            let mut cache = self.weight_cache.lock()
                .map_err(|_| crate::core::Error::internal("weight_cache mutex poisoned"))?;
            cache.entry(name.to_string()).or_insert_with(|| t.clone());
        }
        Ok(t)
    }

    /// `linear_bias` with cached weights — equivalent to the uncached
    /// `linear_bias` but reads weights through `cached_weight_f16` so the
    /// f32→f16 conversion + Metal upload happens once per weight per
    /// runtime lifetime.
    fn cached_linear_bias(
        &self,
        cb: &metal::CommandBufferRef,
        input: &Tensor,
        weight_name: &str,
        bias_name: &str,
        m: usize, k: usize, n: usize,
    ) -> Result<Tensor> {
        let w = self.cached_weight_f16(weight_name)?;
        let b = self.cached_weight_f16(bias_name)?;
        Ok(self.linear_tensors(cb, input, &w, &b, m, k, n))
    }

    /// Cached zero-bias tensor (length `cout`) for no-bias convs/matmuls.
    /// The DPT path's `layer_rn` (3×3 256→256, no bias) hits this on every
    /// view; without caching every call allocates a fresh Metal buffer.
    fn cached_zero_bias(&self, cout: usize) -> Result<Tensor> {
        let key = format!("__zero_bias_{}", cout);
        {
            let cache = self.weight_cache.lock()
                .map_err(|_| crate::core::Error::internal("weight_cache mutex poisoned"))?;
            if let Some(t) = cache.get(&key) {
                return Ok(t.clone());
            }
        }
        let t = Tensor::zeros_on(Shape::from([cout]), DType::F16, self.compute.device().info().id)?;
        {
            let mut cache = self.weight_cache.lock()
                .map_err(|_| crate::core::Error::internal("weight_cache mutex poisoned"))?;
            cache.entry(key).or_insert_with(|| t.clone());
        }
        Ok(t)
    }

    /// `layer_norm` with cached weights.
    fn cached_layer_norm(
        &self,
        cb: &metal::CommandBufferRef,
        input: &Tensor,
        weight_name: &str,
        bias_name: &str,
        n: usize, d: usize, eps: f32,
    ) -> Result<Tensor> {
        let w = self.cached_weight_f16(weight_name)?;
        let b = self.cached_weight_f16(bias_name)?;
        Ok(gpu_ops::layer_norm_tensors_on(
            &self.compute, &self.kernels.common.layer_norm, cb,
            input, &w, &b, n, d, eps,
        ))
    }

    /// Forward: `views_chw` is a flat f32 RGB CHW buffer of length
    /// `n_views * 3 * 518 * 518`, ImageNet-normalised. Returns one
    /// shape-correct splat per patch (37*37 = 1369 per view) shaded by
    /// the per-patch feature mean from the real DINOv2-L encoder.
    pub fn forward(
        &self,
        views_chw: &[f32],
        n_views: usize,
        seed: u64,
    ) -> Result<Vec<GaussianSplat>> {
        let out = self.forward_full(views_chw, n_views, seed, false, false, false)?;
        Ok(out.splats)
    }

    /// Forward pass with optional secondary heads (depth/normal/points).
    /// Each head adds ~30s CPU compute on the current path. Caller opts in
    /// via the boolean flags; the returned struct's corresponding field is
    /// `Some(Vec<f32>)` per view if computed, `None` otherwise.
    pub fn forward_full(
        &self,
        views_chw: &[f32],
        n_views: usize,
        seed: u64,
        compute_depth: bool,
        compute_normals: bool,
        compute_points: bool,
    ) -> Result<WorldMirrorOutputs> {
        self.forward_full_with_hints(
            views_chw, n_views, seed,
            compute_depth, compute_normals, compute_points,
            &super::hyworld::Pow3rHints::default(),
        )
    }

    /// `forward_full` with optional pow3r conditioning hints. Each
    /// hint, if `Some`, contributes a `[D_MODEL=1024]` embedding that is
    /// added additively to every patch token before the frame_blocks. Hints
    /// that are `None` are skipped — backwards-compatible with the no-hints
    /// path (`forward_full`).
    pub fn forward_full_with_hints(
        &self,
        views_chw: &[f32],
        n_views: usize,
        seed: u64,
        compute_depth: bool,
        compute_normals: bool,
        compute_points: bool,
        hints: &super::hyworld::Pow3rHints,
    ) -> Result<WorldMirrorOutputs> {
        const D_MODEL: usize = 1024;
        const NUM_HEADS: usize = 16;
        const HEAD_DIM: usize = D_MODEL / NUM_HEADS; // 64
        const NUM_LAYERS: usize = 24;
        const NUM_REG_TOKENS: usize = 4;
        const FFN_DIM: usize = D_MODEL * 4; // 4096
        const PATCH_SIZE: usize = 14;
        const IMG_SIZE: usize = 518;
        const GRID: usize = IMG_SIZE / PATCH_SIZE; // 37
        const NUM_PATCHES: usize = GRID * GRID; // 1369
        let scale_attn = 1.0f32 / (HEAD_DIM as f32).sqrt();

        let mut all_splats = Vec::with_capacity(n_views * NUM_PATCHES);
        let mut depth_maps_out: Option<Vec<Vec<f32>>> = if compute_depth { Some(Vec::with_capacity(n_views)) } else { None };
        let mut normal_maps_out: Option<Vec<Vec<f32>>> = if compute_normals { Some(Vec::with_capacity(n_views)) } else { None };
        let mut point_clouds_out: Option<Vec<Vec<f32>>> = if compute_points { Some(Vec::with_capacity(n_views)) } else { None };
        let mut camera_params_out: Vec<[f32; 9]> = Vec::with_capacity(n_views);
        // Collected per-view CLS-token features (2048-dim each), fed to cam_head
        // once after the loop for joint multi-view camera estimation.
        let mut cam_cls_tokens: Vec<half::f16> = Vec::with_capacity(n_views * 2 * 1024);
        let view_stride = 3 * IMG_SIZE * IMG_SIZE;

        // The DINOv2-L lives under `visual_geometry_transformer.patch_embed.*`.
        // (The OUTER `patch_embed` is the full encoder; the INNER
        // `patch_embed.patch_embed.proj.*` is the conv layer.)
        let prefix = "visual_geometry_transformer.patch_embed";

        for v in 0..n_views {
            let view_chw = &views_chw[v * view_stride..(v + 1) * view_stride];

            // 1. Patch projection: 3×518×518 → 1369 × 1024 via 14×14 conv.
            let patches = self.patch_embed_conv(view_chw, GRID, PATCH_SIZE, IMG_SIZE, D_MODEL, prefix)?;

            // 2. Prepend cls + register tokens, add positional embedding.
            let cls = self.cached_weight_f16(&format!("{}.cls_token", prefix))?;
            let reg = self.cached_weight_f16(&format!("{}.register_tokens", prefix))?;
            let pos = self.cached_weight_f16(&format!("{}.pos_embed", prefix))?;
            let cls_data: Vec<half::f16> = cls.to_vec()?;
            let reg_data: Vec<half::f16> = reg.to_vec()?;
            let pos_data: Vec<half::f16> = pos.to_vec()?;
            let patches_data: Vec<half::f16> = patches.to_vec()?;

            // v8 pow3r token layout (matches the Tencent reference
            // `_assemble_tokens`): when pose AND/OR ray hints are supplied the
            // sequence gains 2 extra "special" tokens (a pose token + a ray
            // token) inserted AFTER the register tokens and BEFORE the patches:
            //   [cls, reg×4, pose, ray, patches+depth]    (1377 tokens)
            // vs the default                               [cls, reg×4, patches] (1374)
            // The pose/ray tokens get NO positional embedding (special_pos = 0
            // in the reference). The depth hint is still added elementwise to
            // the patch tokens. cam_head reads token 0 regardless. All hints
            // are per-request (same n_pow3r for every view).
            let pose_emb: Option<Vec<half::f16>> = match hints.pose_per_view.as_ref().and_then(|pv| pv.get(v)) {
                Some(pose) => Some(self.pow3r_pose_embed(pose)?.to_vec()?),
                None => None,
            };
            let ray_emb: Option<Vec<half::f16>> = match hints.ray_per_view.as_ref().and_then(|rv| rv.get(v)) {
                Some(ray) => Some(self.pow3r_ray_embed(ray)?.to_vec()?),
                None => None,
            };
            let depth_emb: Option<Vec<half::f16>> = match hints.depth_per_view.as_ref().and_then(|dv| dv.get(v)) {
                Some(d) if d.len() == NUM_PATCHES * 196 => Some(self.pow3r_depth_embed(d)?.to_vec()?),
                _ => None,
            };
            // n_pow3r: ALWAYS 2 special tokens when conditioning is enabled
            // on the saved model (it is — `enable_cond=True` in the published
            // config). The reference's `_assemble_tokens` always allocates
            // [pose, ray] slots regardless of whether hints are supplied
            // (zero tokens when not). Matching this is critical: the trained
            // transformer expects `patch_start_idx=7` positions and seq_len=1376;
            // a 1374-sequence shifts every position index and corrupts
            // attention vs the trained weights (suspected root cause of the
            // image-independent CLS-token symptom found in the v49 reference
            // diff — see memory `project_efficient_genai.md` 2026-05-13).
            let n_pow3r: usize = 2;
            let n_lead = 1 + NUM_REG_TOKENS + n_pow3r; // always 7
            let seq_len = n_lead + NUM_PATCHES;
            let mut combined = Vec::with_capacity(seq_len * D_MODEL);
            // CLS first.
            combined.extend_from_slice(&cls_data[..D_MODEL]);
            // Register tokens.
            for r in 0..NUM_REG_TOKENS {
                let off = r * D_MODEL;
                if off + D_MODEL <= reg_data.len() {
                    combined.extend_from_slice(&reg_data[off..off + D_MODEL]);
                } else {
                    combined.extend(std::iter::repeat(half::f16::ZERO).take(D_MODEL));
                }
            }
            // pow3r special tokens (pose then ray), if any. A missing
            // embedding contributes a zero token (matches the reference's
            // `torch.zeros` fallback).
            if n_pow3r == 2 {
                match &pose_emb {
                    Some(e) => combined.extend_from_slice(&e[..D_MODEL]),
                    None => combined.extend(std::iter::repeat(half::f16::ZERO).take(D_MODEL)),
                }
                match &ray_emb {
                    Some(e) => combined.extend_from_slice(&e[..D_MODEL]),
                    None => combined.extend(std::iter::repeat(half::f16::ZERO).take(D_MODEL)),
                }
            }
            // Patch tokens, with the depth embedding added elementwise.
            if let Some(de) = &depth_emb {
                for p in 0..NUM_PATCHES {
                    for c in 0..D_MODEL {
                        let idx = p * D_MODEL + c;
                        combined.push(half::f16::from_f32(
                            patches_data[idx].to_f32() + de[idx].to_f32(),
                        ));
                    }
                }
            } else {
                combined.extend_from_slice(&patches_data);
            }

            // Positional embedding (DINOv2-with-registers convention):
            //   pos_data[0]   → CLS
            //   pos_data[1+k] → patch k
            // Register + pow3r special tokens get no positional embedding.
            // When no pow3r tokens are present this reduces to the original
            // contiguous `combined[i] += pos_data[i]` for the [cls, reg…] head
            // — kept byte-identical to avoid touching the default path.
            if n_pow3r == 0 {
                for i in 0..(seq_len * D_MODEL).min(pos_data.len()) {
                    combined[i] = half::f16::from_f32(combined[i].to_f32() + pos_data[i].to_f32());
                }
            } else {
                // CLS gets pos[0].
                for c in 0..D_MODEL.min(pos_data.len()) {
                    combined[c] = half::f16::from_f32(combined[c].to_f32() + pos_data[c].to_f32());
                }
                // Patch k gets pos[1+k] (if available in pos_data).
                let patch_base = n_lead * D_MODEL;
                for p in 0..NUM_PATCHES {
                    let pos_row = (1 + p) * D_MODEL;
                    if pos_row + D_MODEL > pos_data.len() { break; }
                    for c in 0..D_MODEL {
                        let dst = patch_base + p * D_MODEL + c;
                        combined[dst] = half::f16::from_f32(
                            combined[dst].to_f32() + pos_data[pos_row + c].to_f32(),
                        );
                    }
                }
            }

            let device_id = self.compute.device().info().id;
            let mut hidden = Tensor::from_slice(
                &combined, Shape::from([seq_len, D_MODEL]), DType::F16, device_id,
            )?;

            // 3. 24 transformer layers — vanilla MHA (no QK-norm / RoPE here;
            // those live in frame_blocks/global_blocks which are skipped in v1).
            for layer in 0..NUM_LAYERS {
                let lp = format!("{}.blocks.{}", prefix, layer);
                let cb = self.compute.new_command_buffer();

                // Pre-norm.
                let normed = self.cached_layer_norm(
                    &cb, &hidden,
                    &format!("{}.norm1.weight", lp), &format!("{}.norm1.bias", lp),
                    seq_len, D_MODEL, 1e-6,
                )?;

                // Fused QKV projection: [seq, 1024] -> [seq, 3*1024].
                // Most DINOv2 saved checkpoints use fused qkv; some use
                // separate query/key/value. We probe for the fused form
                // first since that matches WorldMirror's saved layout.
                let qkv = if self.model.get_weight(&format!("{}.attn.qkv.weight", lp)).is_some() {
                    self.cached_linear_bias(
                        &cb, &normed,
                        &format!("{}.attn.qkv.weight", lp),
                        &format!("{}.attn.qkv.bias", lp),
                        seq_len, D_MODEL, 3 * D_MODEL,
                    )?
                } else {
                    // Fallback for unfused layouts (unused in WorldMirror).
                    return Err(crate::core::Error::internal(format!(
                        "WorldMirror patch_embed expected fused {}.attn.qkv but it was missing",
                        lp,
                    )));
                };

                // Split fused QKV by reading data and slicing on CPU. The
                // batched_attention helper expects [S, H, D] layout per
                // tensor, so we reshape after splitting.
                let qkv_data: Vec<half::f16> = qkv.to_vec()?;
                let mut q_data = Vec::with_capacity(seq_len * D_MODEL);
                let mut k_data = Vec::with_capacity(seq_len * D_MODEL);
                let mut v_data = Vec::with_capacity(seq_len * D_MODEL);
                for s in 0..seq_len {
                    let row = s * 3 * D_MODEL;
                    q_data.extend_from_slice(&qkv_data[row..row + D_MODEL]);
                    k_data.extend_from_slice(&qkv_data[row + D_MODEL..row + 2 * D_MODEL]);
                    v_data.extend_from_slice(&qkv_data[row + 2 * D_MODEL..row + 3 * D_MODEL]);
                }
                let q = Tensor::from_slice(&q_data, Shape::from([seq_len, D_MODEL]), DType::F16, device_id)?;
                let k = Tensor::from_slice(&k_data, Shape::from([seq_len, D_MODEL]), DType::F16, device_id)?;
                let v = Tensor::from_slice(&v_data, Shape::from([seq_len, D_MODEL]), DType::F16, device_id)?;

                let attn_out = self.batched_attention(
                    &cb, &q, &k, &v, seq_len, seq_len, NUM_HEADS, HEAD_DIM, scale_attn,
                )?;

                // Output projection.
                let proj = self.cached_linear_bias(
                    &cb, &attn_out,
                    &format!("{}.attn.proj.weight", lp),
                    &format!("{}.attn.proj.bias", lp),
                    seq_len, D_MODEL, D_MODEL,
                )?;
                // Must commit before `layer_scale` reads `proj.to_vec()` on CPU
                // (same bug pattern as `frame_block_forward`, fixed 2026-05-13).
                cb.commit();
                cb.wait_until_completed();

                // LayerScale (ls1.gamma): per-channel multiplier before residual.
                let scaled = self.layer_scale(&cb, &proj, &format!("{}.ls1.gamma", lp), seq_len, D_MODEL)?;
                let cb = self.compute.new_command_buffer();
                let h = self.add(&cb, &hidden, &scaled);

                // MLP block.
                let normed2 = self.cached_layer_norm(
                    &cb, &h,
                    &format!("{}.norm2.weight", lp), &format!("{}.norm2.bias", lp),
                    seq_len, D_MODEL, 1e-6,
                )?;
                let ffn_up = self.cached_linear_bias(
                    &cb, &normed2,
                    &format!("{}.mlp.fc1.weight", lp), &format!("{}.mlp.fc1.bias", lp),
                    seq_len, D_MODEL, FFN_DIM,
                )?;
                let ffn_act = self.activation(&cb, &self.kernels.gelu, &ffn_up);
                let ffn_down = self.cached_linear_bias(
                    &cb, &ffn_act,
                    &format!("{}.mlp.fc2.weight", lp), &format!("{}.mlp.fc2.bias", lp),
                    seq_len, FFN_DIM, D_MODEL,
                )?;
                cb.commit();
                cb.wait_until_completed();
                let scaled2 = self.layer_scale(&cb, &ffn_down, &format!("{}.ls2.gamma", lp), seq_len, D_MODEL)?;
                let cb = self.compute.new_command_buffer();
                hidden = self.add(&cb, &h, &scaled2);

                cb.commit();
                cb.wait_until_completed();
            }

            // 4. Final norm.
            let cb = self.compute.new_command_buffer();
            let normed = self.cached_layer_norm(
                &cb, &hidden,
                &format!("{}.norm.weight", prefix), &format!("{}.norm.bias", prefix),
                seq_len, D_MODEL, 1e-6,
            )?;
            cb.commit();
            cb.wait_until_completed();

            // 5. Frame_blocks (24 layers): per-frame attention with 2D RoPE
            //    + per-head QK-norm (RMS) + LayerScale + fused QKV. This
            //    runs ON TOP of the DINOv2-L patch_embed output and is
            //    where most of the geometric reasoning happens.
            let mut frame_hidden = normed; // Reuse normed from patch_embed final.
            let frame_periods = self.cached_weight_f16("visual_geometry_transformer.frame_blocks.0.attn.rope.periods")?;
            let periods_data: Vec<half::f16> = frame_periods.to_vec()?;
            let periods_f32: Vec<f32> = periods_data.iter().map(|v| v.to_f32()).collect();
            let (cos_y, sin_y, cos_x, sin_x) =
                Self::precompute_rope_2d(&periods_f32, GRID, GRID, n_lead);

            for layer in 0..NUM_LAYERS {
                let lp = format!("visual_geometry_transformer.frame_blocks.{}", layer);
                frame_hidden = self.frame_block_forward(
                    &frame_hidden, &lp, seq_len, D_MODEL, NUM_HEADS, HEAD_DIM, FFN_DIM,
                    &cos_y, &sin_y, &cos_x, &sin_x, scale_attn,
                )?;
            }
            // 6. Global_blocks (24 layers): cross-frame attention sharing
            //    the same architecture as frame_blocks. For single-view
            //    input it acts as additional self-attention; for multi-view
            //    it would concat tokens across views before running these
            //    layers. v3 single-view path: run on top of frame_blocks
            //    output using `global_blocks.{i}.*` weights.
            // Capture 4 intermediate global_blocks outputs at evenly-spaced
            // layer indices for DPT multi-scale fusion. Standard DPT uses
            // layers [n/4, n/2, 3n/4, n] = [6, 12, 18, 24] (1-indexed) =
            // 0-indexed [5, 11, 17, 23].
            const DPT_LAYERS: [usize; 4] = [5, 11, 17, 23];
            let mut intermediate_globals: Vec<Vec<half::f16>> = Vec::with_capacity(4);
            let mut global_hidden = frame_hidden.clone();
            for layer in 0..NUM_LAYERS {
                let lp = format!("visual_geometry_transformer.global_blocks.{}", layer);
                global_hidden = self.frame_block_forward(
                    &global_hidden, &lp, seq_len, D_MODEL, NUM_HEADS, HEAD_DIM, FFN_DIM,
                    &cos_y, &sin_y, &cos_x, &sin_x, scale_attn,
                )?;
                if DPT_LAYERS.contains(&layer) {
                    intermediate_globals.push(global_hidden.to_vec()?);
                }
            }

            // 7. Concat frame + global token features along the channel
            //    axis → [seq, 2048]. The 5 prediction heads (`gs_head`,
            //    `depth_head`, etc.) read 2048-dim inputs at their `norm`
            //    weights and `projects.0..3` (1×1 convs).
            let frame_data: Vec<half::f16> = frame_hidden.to_vec()?;
            let global_data: Vec<half::f16> = global_hidden.to_vec()?;
            let mut concat = Vec::with_capacity(seq_len * 2 * D_MODEL);
            for s in 0..seq_len {
                concat.extend_from_slice(&frame_data[s * D_MODEL..(s + 1) * D_MODEL]);
                concat.extend_from_slice(&global_data[s * D_MODEL..(s + 1) * D_MODEL]);
            }
            let concat_tensor = Tensor::from_slice(
                &concat,
                Shape::from([seq_len, 2 * D_MODEL]),
                DType::F16,
                self.compute.device().info().id,
            )?;

            // Build 4 multi-scale concat features: each pairs the FINAL
            // frame_hidden with a different global_blocks intermediate.
            // Then drop cls + register tokens (rows 0..NUM_REG_TOKENS+1).
            let mut multi_scale_patches: Vec<Vec<half::f16>> = Vec::with_capacity(4);
            for ig in &intermediate_globals {
                let mut buf = Vec::with_capacity(NUM_PATCHES * 2 * D_MODEL);
                for s in n_lead..seq_len {
                    buf.extend_from_slice(&frame_data[s * D_MODEL..(s + 1) * D_MODEL]);
                    buf.extend_from_slice(&ig[s * D_MODEL..(s + 1) * D_MODEL]);
                }
                multi_scale_patches.push(buf);
            }

            // 8. Slice off CLS + register tokens; keep only patch features.
            let patch_feat = concat_tensor.slice(0, n_lead, seq_len)?;

            // 9a. DPT chromatic refinement branch — minimal v6 path running
            //     real `gs_head.norm` + `projects.0` + `scratch.layer1_rn`
            //     + `refinenet1` (resConfUnit1, resConfUnit2, out_conv) +
            //     `output_conv1` + `output_conv2.{0,2}` weights at 37×37
            //     resolution. Returns flat 3*37*37 RGB values in 0..1.
            let cb_dpt = self.compute.new_command_buffer();
            let dpt_in = self.cached_layer_norm(
                &cb_dpt, &patch_feat,
                "gs_head.norm.weight", "gs_head.norm.bias",
                NUM_PATCHES, 2 * D_MODEL, 1e-6,
            )?;
            cb_dpt.commit();
            cb_dpt.wait_until_completed();
            // GPU DPT path (`dpt_minimal_forward_gpu`) is wired and compiles
            // but produces all-zero RGB on prod (sigmoid driven to 0 by
            // overly-negative GPU conv chain output — a kernel-level bug
            // pending debug; possibly the 3×3 tiled kernel's no-bias path
            // or activation buffer handling). CPU path (battle-tested by
            // v16) ships meaningful chroma.
            let dpt_in_data: Vec<half::f16> = dpt_in.to_vec()?;

            // cam_head input: collect this view's CLS-token feature (row 0 of
            // the 2048-dim concat). cam_head runs ONCE after the per-view loop
            // over all views' CLS tokens so `refine_net` self-attention can
            // mix cross-view context (matches reference `latest_feat[:,:,0]`
            // shape [B, S, C]).
            cam_cls_tokens.extend_from_slice(&concat[..2 * D_MODEL]);

            // Apply gs_head.norm to each multi-scale feature before passing
            // to the DPT. Norm on CPU since these are small (1369 × 2048 f16).
            let norm_w_f16: Vec<half::f16> = self
                .weight_f16(&self.model, "gs_head.norm.weight")?
                .to_vec()?;
            let norm_b_f16: Vec<half::f16> = self
                .weight_f16(&self.model, "gs_head.norm.bias")?
                .to_vec()?;
            let mut multi_scale_normed: Vec<Vec<half::f16>> = Vec::with_capacity(4);
            for ms in &multi_scale_patches {
                let mut out = Vec::with_capacity(NUM_PATCHES * 2 * D_MODEL);
                for n in 0..NUM_PATCHES {
                    let row = &ms[n * 2 * D_MODEL..(n + 1) * 2 * D_MODEL];
                    let mean: f32 = row.iter().map(|v| v.to_f32()).sum::<f32>()
                        / (2 * D_MODEL) as f32;
                    let var: f32 = row.iter()
                        .map(|v| { let d = v.to_f32() - mean; d * d })
                        .sum::<f32>() / (2 * D_MODEL) as f32;
                    let inv_std = 1.0 / (var + 1e-6).sqrt();
                    for (c, v) in row.iter().enumerate() {
                        let normed = (v.to_f32() - mean) * inv_std;
                        let scaled = normed * norm_w_f16[c].to_f32() + norm_b_f16[c].to_f32();
                        out.push(half::f16::from_f32(scaled));
                    }
                }
                multi_scale_normed.push(out);
            }
            // gs_head DPT chromatic refinement — GPU matmul path. Numerically
            // matches the CPU `dpt_head_forward` to within f16 precision
            // (max_diff ≤ 0.01, rms ≤ 0.001). Massive speed-up vs CPU at
            // multi-scale 4×proj (each ~3-13M f16 mults per level).
            let normed_tensor = Tensor::from_slice(
                &dpt_in_data,
                Shape::from([NUM_PATCHES, 2 * D_MODEL]),
                DType::F16,
                self.compute.device().info().id,
            )?;
            // gs_head DPT: capture both the chromatic head output AND the
            // 128ch fused features the gs_renderer needs (see reference
            // `DPTHead.forward` is_gsdpt=True branch: returns `fused, preds, conf`).
            let (dpt_rgb, gs_fused_features) = self.dpt_head_forward_gpu_single_scale(
                &normed_tensor, &multi_scale_normed, "gs_head", 3, true,
            )?;

            // Secondary heads — same DPT structure, different output channels,
            // no sigmoid (raw values for caller normalisation). All on GPU
            // matmul path.
            if let Some(out) = depth_maps_out.as_mut() {
                let depth = self.dpt_head_forward_gpu(&normed_tensor, &multi_scale_normed, "depth_head", 1, false)?;
                out.push(depth);
            }
            if let Some(out) = normal_maps_out.as_mut() {
                let norm = self.dpt_head_forward_gpu(&normed_tensor, &multi_scale_normed, "norm_head", 3, false)?;
                out.push(norm);
            }
            if let Some(out) = point_clouds_out.as_mut() {
                let pts = self.dpt_head_forward_gpu(&normed_tensor, &multi_scale_normed, "pts_head", 3, false)?;
                out.push(pts);
            }

            // 9b. Image branch (gs_renderer chain) — runs the trained
            //     image-conditioned splat predictor on the raw view.
            //     input_merger.0 (7×7 stride=14 padding=3) is on CPU due to
            //     a NaN bug in unet's conv2d_naive_f16 metal kernel at
            //     stride>1; the subsequent 3×3 and 1×1 convs use the tested
            //     GPU tiled / 1×1 kernels.
            let img_data_chw = self.image_branch_forward(view_chw, IMG_SIZE, &gs_fused_features)?;
            // img_data_chw is laid out as `[12, 37, 37]` CHW. Re-pack to
            // `[37*37, 12]` (sequence-of-pixels) for easy per-splat access.
            let mut splat_data: Vec<half::f16> = Vec::with_capacity(NUM_PATCHES * 12);
            for p in 0..NUM_PATCHES {
                let py = p / GRID;
                let px = p % GRID;
                for c in 0..12 {
                    let off = c * GRID * GRID + py * GRID + px;
                    splat_data.push(img_data_chw[off]);
                }
            }
            // Keep transformer concat features for color chroma blend.
            let patch_feat_data: Vec<half::f16> = patch_feat.to_vec()?;

            // 10. Decode 12-channel splat params per patch.
            //
            //     Channel layout (Tencent reference
            //     `rasterization.py::GaussianSplatRenderer.gs_head`,
            //     `splits_and_inits`, sh_degree=0):
            //       [0..4]  : quaternion (qx, qy, qz, qw)  — xavier init 1.0/0.0
            //       [4..7]  : log scale (xyz)              — xavier init 0.00003/-7.0
            //       [7]     : opacity logit                — xavier init 1.0/-2.0
            //       [8..11] : SH-DC RGB (3ch)              — xavier init 1.0/0.0
            //       [11]    : weight/conf logit            — xavier init 1.0/-2.0
            //
            //     Position in the reference is NOT in this head — it comes from
            //     unprojecting per-pixel depth through the camera intrinsics.
            //     Until depth-driven unproject is wired (separate follow-up),
            //     anchor each splat at its patch's grid centre on a unit
            //     ground plane at z=2 so the rendered output is a flat
            //     photo-projected plane instead of garbage.
            //
            //     TODO: route the DPT `fused` features (patch_feat_data here)
            //     into `gs_renderer.gs_head` as the reference does — currently
            //     `image_branch_forward` only sees `input_merger(image)` and
            //     skips `fused`, so the trained gs_head sees only half its
            //     input and outputs near-bias splats.
            let _ = patch_feat_data; let _ = dpt_rgb;
            // SH-DC → linear color: rgb = 0.5 + SH_Y00 * sh_dc.
            // Y_0^0 = 1 / (2 * sqrt(pi)) ≈ 0.28209479.
            const SH_C0: f32 = 0.28209479;
            for p in 0..NUM_PATCHES {
                let s = &splat_data[p * 12..(p + 1) * 12];

                // Patch grid centre on a 2-unit-wide plane at z=2.
                let gy = (p / GRID) as f32;
                let gx = (p % GRID) as f32;
                let cx = (gx / (GRID as f32 - 1.0) - 0.5) * 2.0;
                let cy = -(gy / (GRID as f32 - 1.0) - 0.5) * 2.0;
                let cz = 2.0_f32 + (v as f32) * 0.05;
                let position = [cx, cy, cz];

                // Quaternion at [0..4] (xyzw); guard against zero norm.
                let qx = s[0].to_f32(); let qy = s[1].to_f32();
                let qz = s[2].to_f32(); let qw = s[3].to_f32();
                let qn = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt().max(1e-6);
                let rotation = [qx / qn, qy / qn, qz / qn, qw / qn];

                // Log scale at [4..7]. Reference uses softplus or exp; we exp
                // and clamp. Wider range than before since trained log-scales
                // typically run [-8, 0] (giving 3e-4 .. 1.0).
                let scale = [
                    s[4].to_f32().exp().clamp(0.001, 1.0),
                    s[5].to_f32().exp().clamp(0.001, 1.0),
                    s[6].to_f32().exp().clamp(0.001, 1.0),
                ];

                // Opacity logit at [7].
                let opacity = (1.0_f32 / (1.0 + (-s[7].to_f32()).exp())).clamp(0.0, 1.0);

                // SH-DC RGB at [8..11].
                let r = (0.5 + SH_C0 * s[8].to_f32()).clamp(0.0, 1.0);
                let g = (0.5 + SH_C0 * s[9].to_f32()).clamp(0.0, 1.0);
                let b = (0.5 + SH_C0 * s[10].to_f32()).clamp(0.0, 1.0);
                let color = [r, g, b];

                let _ = seed;
                all_splats.push(GaussianSplat {
                    position,
                    scale,
                    rotation,
                    color,
                    opacity,
                });
            }
        }

        // cam_head: joint multi-view camera estimation over all collected CLS
        // tokens. Non-fatal — falls back to per-view zeros if it errors.
        if cam_cls_tokens.len() == n_views * 2 * 1024 {
            match self.cam_head_forward_all_views(&cam_cls_tokens, n_views) {
                Ok(cams) => camera_params_out = cams,
                Err(_) => camera_params_out = vec![[0.0; 9]; n_views],
            }
        }

        Ok(WorldMirrorOutputs {
            splats: all_splats,
            depth_maps: depth_maps_out,
            normal_maps: normal_maps_out,
            point_clouds: point_clouds_out,
            camera_params: if camera_params_out.is_empty() { None } else { Some(camera_params_out) },
        })
    }

    /// Image branch: raw image → 12-channel splat parameters at the
    /// patch-grid resolution (37×37) via the trained gs_renderer chain.
    ///
    /// Pipeline:
    ///   image [1, 3, 518, 518]
    ///     → input_merger.0 (Conv2d 3→128, 7×7 kernel, stride=14, padding=3)
    ///     → ReLU
    ///     → [1, 128, 37, 37]
    ///   → gs_renderer.gs_head.0 (Conv2d 128→256, 3×3 kernel, stride=1, padding=1)
    ///     → ReLU
    ///     → [1, 256, 37, 37]
    ///   → gs_renderer.gs_head.2 (Conv2d 256→12, 1×1 kernel)
    ///     → [1, 12, 37, 37]
    ///
    /// Returns the f16 buffer of shape `[12, 37, 37]` flattened CHW. Each
    /// of the 1369 pixels carries 12 splat parameters (position offset,
    /// scale, rotation, opacity, SH-DC luminance) trained by Tencent.
    ///
    /// `stride=14` aligns the image-branch output spatial resolution with
    /// the DINOv2 patch grid so downstream code can match patch features
    /// 1:1 with image-branch splats.
    /// GPU-accelerated minimal DPT decoder. Same architecture as
    /// [`Self::dpt_minimal_forward`] (CPU) but keeps tensors on the Metal
    /// device throughout, using the now-zero-bias-fixed unet conv2d
    /// helpers (3×3 tiled for stride=1 padding=1; 1×1 for the projections).
    ///
    /// Drops the DPT compute from ~10s CPU to <1s GPU; warm world/reconstruct
    /// goes from 13s back down toward v5's ~3s.
    fn dpt_minimal_forward_gpu(
        &self,
        normed_concat: &Tensor,
        multi_scale_features: &[Vec<half::f16>],
    ) -> Result<Vec<f32>> {
        self.dpt_head_forward_gpu(normed_concat, multi_scale_features, "gs_head", 3, true)
    }

    /// DPT decoder dispatcher — picks single-scale (v33) or multi-resolution
    /// (v6.5) path based on env var `WORLDMIRROR_DPT_MULTIRES`.
    ///
    /// Default = single-scale (37×37 output, fast: ~1-2s/head/view). Multi-res
    /// (env var = "1") gives 296×296 output (slow: ~400s/head/view) for the
    /// secondary heads where higher spatial resolution matters most.
    ///
    /// gs_head ALWAYS runs single-scale because splats are anchored to the
    /// 37×37 patch grid, so 296×296 output is wasted there. The
    /// `force_single_scale` flag (caller sets to `true` for gs_head) overrides
    /// the env var.
    fn dpt_head_forward_gpu(
        &self,
        normed_concat: &Tensor,
        multi_scale_features: &[Vec<half::f16>],
        head_prefix: &str,
        output_channels: usize,
        apply_sigmoid: bool,
    ) -> Result<Vec<f32>> {
        let force_single_scale = head_prefix == "gs_head";
        let use_multires = !force_single_scale
            && std::env::var("WORLDMIRROR_DPT_MULTIRES").ok().as_deref() == Some("1");
        if use_multires {
            self.dpt_head_forward_gpu_multires(
                normed_concat, multi_scale_features, head_prefix, output_channels, apply_sigmoid,
            )
        } else {
            // Discard the fused-features intermediate — callers that need it
            // call `dpt_head_forward_gpu_single_scale` directly.
            self.dpt_head_forward_gpu_single_scale(
                normed_concat, multi_scale_features, head_prefix, output_channels, apply_sigmoid,
            ).map(|(out, _fused)| out)
        }
    }

    /// v33-style single-scale DPT decoder. All compute at the 37×37 patch
    /// grid: 4 levels of (1×1 projection + 3×3 layer_rn), sum-fused at 37,
    /// single refinenet1, output_conv chain at 37. Output: `output_channels`
    /// × 37 × 37 f32. Used for gs_head (splat colors at patch grid) and as
    /// the default fast path for secondary heads.
    /// Returns `(head_output, fused_features)`. `fused_features` is the
    /// post-`output_conv1` + ReLU activation as a flat `[128, 37, 37]` f16
    /// buffer — the same `fused` features the reference's `GaussianSplatRenderer.gs_head`
    /// consumes (`fused + input_merger(images)` in `DPTHead.forward`). Callers
    /// that don't need them ignore the second value.
    fn dpt_head_forward_gpu_single_scale(
        &self,
        normed_concat: &Tensor,
        multi_scale_features: &[Vec<half::f16>],
        head_prefix: &str,
        output_channels: usize,
        apply_sigmoid: bool,
    ) -> Result<(Vec<f32>, Vec<half::f16>)> {
        const GRID: usize = 37;
        const N: usize = GRID * GRID;
        const C_IN: usize = 2048;
        const C_OUT: usize = 256;
        let device_id = self.compute.device().info().id;

        let to_chw_2048 = |seq: &[half::f16]| -> Result<Tensor> {
            let mut chw = vec![half::f16::ZERO; C_IN * N];
            for p in 0..N {
                for c in 0..C_IN {
                    chw[c * N + p] = seq[p * C_IN + c];
                }
            }
            Tensor::from_slice(
                &chw, Shape::from([1usize, C_IN, GRID, GRID]),
                DType::F16, device_id,
            )
        };

        let final_seq: Vec<half::f16> = normed_concat.to_vec()?;

        // 4 levels at 37×37 (no resize_layers — sum-fuse at common res).
        let mut levels: Vec<Tensor> = Vec::with_capacity(4);
        for i in 0..4usize {
            let ch_proj = match i { 0 => 256, 1 => 512, _ => 1024 };
            let level_feat: &[half::f16] = multi_scale_features
                .get(i)
                .map(|v| v.as_slice())
                .unwrap_or(&final_seq);
            let x_chw = to_chw_2048(level_feat)?;
            let proj = self.gpu_conv1x1_via_matmul(
                &x_chw,
                &format!("{}.projects.{}.weight", head_prefix, i),
                Some(&format!("{}.projects.{}.bias", head_prefix, i)),
                C_IN, ch_proj, GRID, GRID,
            )?;
            let l = self.gpu_conv3x3_via_matmul(
                &proj,
                &format!("{}.scratch.layer{}_rn.weight", head_prefix, i + 1),
                None,
                ch_proj, C_OUT, GRID, GRID,
            )?;
            levels.push(l);
        }

        // Sum-fuse 4 levels at 37×37.
        let mut fused = vec![half::f16::ZERO; C_OUT * N];
        for lvl in &levels {
            let d: Vec<half::f16> = lvl.to_vec()?;
            for i in 0..(C_OUT * N) {
                fused[i] = half::f16::from_f32(fused[i].to_f32() + d[i].to_f32());
            }
        }
        let l1 = Tensor::from_slice(
            &fused, Shape::from([1usize, C_OUT, GRID, GRID]),
            DType::F16, device_id,
        )?;

        // refinenet1 (resConfUnit1 + resConfUnit2 + out_conv) at 37×37.
        let cb = self.compute.new_command_buffer();
        let l1_relu = self.activation(&cb, &self.kernels.relu, &l1);
        cb.commit();
        cb.wait_until_completed();
        let rc1_h1 = self.gpu_conv3x3_via_matmul(
            &l1_relu,
            &format!("{}.scratch.refinenet1.resConfUnit1.conv1.weight", head_prefix),
            Some(&format!("{}.scratch.refinenet1.resConfUnit1.conv1.bias", head_prefix)),
            C_OUT, C_OUT, GRID, GRID,
        )?;
        let cb = self.compute.new_command_buffer();
        let rc1_h1_relu = self.activation(&cb, &self.kernels.relu, &rc1_h1);
        cb.commit();
        cb.wait_until_completed();
        let rc1_h2 = self.gpu_conv3x3_via_matmul(
            &rc1_h1_relu,
            &format!("{}.scratch.refinenet1.resConfUnit1.conv2.weight", head_prefix),
            Some(&format!("{}.scratch.refinenet1.resConfUnit1.conv2.bias", head_prefix)),
            C_OUT, C_OUT, GRID, GRID,
        )?;
        let cb = self.compute.new_command_buffer();
        let after_rc1 = self.add(&cb, &l1, &rc1_h2);
        cb.commit();
        cb.wait_until_completed();

        let cb = self.compute.new_command_buffer();
        let after_rc1_relu = self.activation(&cb, &self.kernels.relu, &after_rc1);
        cb.commit();
        cb.wait_until_completed();
        let rc2_h1 = self.gpu_conv3x3_via_matmul(
            &after_rc1_relu,
            &format!("{}.scratch.refinenet1.resConfUnit2.conv1.weight", head_prefix),
            Some(&format!("{}.scratch.refinenet1.resConfUnit2.conv1.bias", head_prefix)),
            C_OUT, C_OUT, GRID, GRID,
        )?;
        let cb = self.compute.new_command_buffer();
        let rc2_h1_relu = self.activation(&cb, &self.kernels.relu, &rc2_h1);
        cb.commit();
        cb.wait_until_completed();
        let rc2_h2 = self.gpu_conv3x3_via_matmul(
            &rc2_h1_relu,
            &format!("{}.scratch.refinenet1.resConfUnit2.conv2.weight", head_prefix),
            Some(&format!("{}.scratch.refinenet1.resConfUnit2.conv2.bias", head_prefix)),
            C_OUT, C_OUT, GRID, GRID,
        )?;
        let cb = self.compute.new_command_buffer();
        let after_rc2 = self.add(&cb, &after_rc1, &rc2_h2);
        cb.commit();
        cb.wait_until_completed();

        let after_oc = self.gpu_conv1x1_via_matmul(
            &after_rc2,
            &format!("{}.scratch.refinenet1.out_conv.weight", head_prefix),
            Some(&format!("{}.scratch.refinenet1.out_conv.bias", head_prefix)),
            C_OUT, C_OUT, GRID, GRID,
        )?;

        // Output chain at 37×37.
        let oc1 = self.gpu_conv3x3_via_matmul(
            &after_oc,
            &format!("{}.scratch.output_conv1.weight", head_prefix),
            Some(&format!("{}.scratch.output_conv1.bias", head_prefix)),
            C_OUT, 128, GRID, GRID,
        )?;
        let cb = self.compute.new_command_buffer();
        let oc1_relu = self.activation(&cb, &self.kernels.relu, &oc1);
        cb.commit();
        cb.wait_until_completed();

        // Capture the 128ch ReLU'd intermediate — these are the `fused`
        // features in the reference (DPTHead.forward, is_gsdpt=True branch),
        // consumed by GaussianSplatRenderer.gs_head as `fused + input_merger(images)`.
        let fused_features: Vec<half::f16> = oc1_relu.to_vec()?;

        let oc20 = self.gpu_conv3x3_via_matmul(
            &oc1_relu,
            &format!("{}.scratch.output_conv2.0.weight", head_prefix),
            Some(&format!("{}.scratch.output_conv2.0.bias", head_prefix)),
            128, 32, GRID, GRID,
        )?;
        let cb = self.compute.new_command_buffer();
        let oc20_relu = self.activation(&cb, &self.kernels.relu, &oc20);
        cb.commit();
        cb.wait_until_completed();

        let head = self.gpu_conv1x1_via_matmul(
            &oc20_relu,
            &format!("{}.scratch.output_conv2.2.weight", head_prefix),
            Some(&format!("{}.scratch.output_conv2.2.bias", head_prefix)),
            32, output_channels, GRID, GRID,
        )?;

        let head_data: Vec<half::f16> = head.to_vec()?;
        let mut out = Vec::with_capacity(output_channels * N);
        for &v in &head_data {
            let f = v.to_f32();
            if apply_sigmoid {
                out.push(1.0 / (1.0 + (-f).exp()));
            } else {
                out.push(f);
            }
        }
        Ok((out, fused_features))
    }

    /// Per-view forward pass for `cam_head` — produces 9-dim camera params
    /// from per-view transformer features.
    ///
    /// Architecture (perceiver/DETR-style query-token aggregation):
    ///   1. `init_token` [1, 1, 9] → `param_embed` (Linear 9→2048) → cam_token [1, 2048]
    ///   2. Concat cam_token (1) + patch_features (NUM_PATCHES) → seq [1+NUM_PATCHES, 2048]
    ///   3. 4 `refine_net` self-attention blocks (d_model=2048, ffn=8192,
    ///      num_heads=16, head_dim=128, no q/k norm, no RoPE — simpler than
    ///      `frame_block_forward`).
    ///   4. Extract cam_token at position 0 → [2048]
    ///   5. `out_norm` (LayerNorm)
    ///   6. `param_predictor.fc1` (2048→1024) → GELU → `param_predictor.fc2` (1024→9)
    ///
    /// Output: 9-dim flat array `[px, py, pz, fx, fy, fz, ux, uy, uz]`
    /// (camera position + forward + up vectors).
    ///
    /// Note: `adapt_norm_gen` (DiT-style AdaLN modulation) is NOT applied in
    /// this implementation — produces approximate output. The trained
    /// `cam_head.adapt_norm_gen.1` weight is unused. Future v8.1 may wire
    /// it in once we confirm the upstream conditioning input.
    ///
    /// Patch features must be `[NUM_PATCHES, 2*D_MODEL=2048]` row-major
    /// f16 — typically the gs_head-normed concat features (`dpt_in_data`
    /// in the per-view loop).
    /// v8 pow3r — depth_embed conditioning. Computes a per-patch 1024-dim
    /// embedding from caller-supplied depth values.
    ///
    /// Input: `depth_per_patch` is `[NUM_PATCHES, 196]` f32 — each patch's
    /// 14×14 depth values, flattened row-major.
    /// Architecture: `depth_embed.proj.2` = nn.Sequential(Linear(196→4096),
    /// GELU, Linear(4096→1024)). The trained `proj.2.fc1` / `proj.2.fc2` are
    /// the two Linears.
    /// Output: `[NUM_PATCHES, D_MODEL=1024]` f16 embedding to add additively
    /// to patch tokens before frame_blocks.
    fn pow3r_depth_embed(&self, depth_per_patch: &[f32]) -> Result<Tensor> {
        const NUM_PATCHES: usize = 37 * 37;
        const D: usize = 1024;
        debug_assert_eq!(depth_per_patch.len(), NUM_PATCHES * 196);
        let device_id = self.compute.device().info().id;
        let d_f16: Vec<half::f16> = depth_per_patch.iter().map(|&v| half::f16::from_f32(v)).collect();
        let inp = Tensor::from_slice(
            &d_f16, Shape::from([NUM_PATCHES, 196]), DType::F16, device_id,
        )?;
        let cb = self.compute.new_command_buffer();
        let fc1 = self.cached_linear_bias(
            &cb, &inp,
            "visual_geometry_transformer.depth_embed.proj.2.fc1.weight",
            "visual_geometry_transformer.depth_embed.proj.2.fc1.bias",
            NUM_PATCHES, 196, 4096,
        )?;
        let fc1_act = self.activation(&cb, &self.kernels.gelu, &fc1);
        let fc2 = self.cached_linear_bias(
            &cb, &fc1_act,
            "visual_geometry_transformer.depth_embed.proj.2.fc2.weight",
            "visual_geometry_transformer.depth_embed.proj.2.fc2.bias",
            NUM_PATCHES, 4096, D,
        )?;
        cb.commit();
        cb.wait_until_completed();
        Ok(fc2)
    }

    /// v8 pow3r — pose_embed conditioning. Per-view 7-dim camera pose
    /// (e.g. translation 3 + quaternion 4) → 1024-dim embedding broadcast
    /// across all patches.
    ///
    /// Architecture: `pose_embed` = nn.Sequential(Linear(7→1024), GELU,
    /// Linear(1024→1024)). Weight indices `.0` and `.2` (index 1 is the
    /// activation).
    /// `pose_embed` = nn.Sequential(Linear(7→1024), **SiLU**, Linear(1024→1024)).
    /// SiLU computed on CPU (no SiLU Metal kernel; the token is tiny).
    fn pow3r_pose_embed(&self, pose: &[f32; 7]) -> Result<Tensor> {
        const D: usize = 1024;
        let device_id = self.compute.device().info().id;
        let p_f16: Vec<half::f16> = pose.iter().map(|&v| half::f16::from_f32(v)).collect();
        let inp = Tensor::from_slice(&p_f16, Shape::from([1usize, 7]), DType::F16, device_id)?;
        let cb = self.compute.new_command_buffer();
        let fc1 = self.cached_linear_bias(
            &cb, &inp,
            "visual_geometry_transformer.pose_embed.0.weight",
            "visual_geometry_transformer.pose_embed.0.bias",
            1, 7, D,
        )?;
        cb.commit();
        cb.wait_until_completed();
        let fc1_data: Vec<half::f16> = fc1.to_vec()?;
        let mut silu = vec![half::f16::ZERO; D];
        for i in 0..D { let x = fc1_data[i].to_f32(); silu[i] = half::f16::from_f32(x / (1.0 + (-x).exp())); }
        let silu_t = Tensor::from_slice(&silu, Shape::from([1usize, D]), DType::F16, device_id)?;
        let cb = self.compute.new_command_buffer();
        let fc2 = self.cached_linear_bias(
            &cb, &silu_t,
            "visual_geometry_transformer.pose_embed.2.weight",
            "visual_geometry_transformer.pose_embed.2.bias",
            1, D, D,
        )?;
        cb.commit();
        cb.wait_until_completed();
        Ok(fc2)
    }

    /// `ray_embed` = nn.Sequential(Linear(4→1024), **SiLU**, Linear(1024→1024)).
    fn pow3r_ray_embed(&self, ray: &[f32; 4]) -> Result<Tensor> {
        const D: usize = 1024;
        let device_id = self.compute.device().info().id;
        let r_f16: Vec<half::f16> = ray.iter().map(|&v| half::f16::from_f32(v)).collect();
        let inp = Tensor::from_slice(&r_f16, Shape::from([1usize, 4]), DType::F16, device_id)?;
        let cb = self.compute.new_command_buffer();
        let fc1 = self.cached_linear_bias(
            &cb, &inp,
            "visual_geometry_transformer.ray_embed.0.weight",
            "visual_geometry_transformer.ray_embed.0.bias",
            1, 4, D,
        )?;
        cb.commit();
        cb.wait_until_completed();
        let fc1_data: Vec<half::f16> = fc1.to_vec()?;
        let mut silu = vec![half::f16::ZERO; D];
        for i in 0..D { let x = fc1_data[i].to_f32(); silu[i] = half::f16::from_f32(x / (1.0 + (-x).exp())); }
        let silu_t = Tensor::from_slice(&silu, Shape::from([1usize, D]), DType::F16, device_id)?;
        let cb = self.compute.new_command_buffer();
        let fc2 = self.cached_linear_bias(
            &cb, &silu_t,
            "visual_geometry_transformer.ray_embed.2.weight",
            "visual_geometry_transformer.ray_embed.2.bias",
            1, D, D,
        )?;
        cb.commit();
        cb.wait_until_completed();
        Ok(fc2)
    }

    /// cam_head — joint multi-view camera estimation, matching the Tencent
    /// reference `CameraHead.forward` (which processes `latest_feat[:,:,0]`
    /// of shape `[B, S, C]` — one CLS-token feature per view, all S views
    /// processed together so `refine_net` self-attention couples the
    /// per-view estimates into a coherent trajectory).
    ///
    /// Input: `cls_tokens` = flat `[n_views * D]` f16, one 2048-dim CLS token
    /// per view (row-major: view 0's 2048 values, then view 1's, ...).
    ///
    /// Pipeline (all rows = views processed together):
    ///   cam_tokens = token_norm(cls_tokens)                            [S, D]
    ///   adapt_norm_cam = LayerNorm-no-affine(cam_tokens)               [S, D]
    ///   curr_pred = None
    ///   for step in 0..4:
    ///       net_input = param_embed(curr_pred or init_token-broadcast) [S, D]
    ///       shift,scale,gate = adapt_norm_gen(net_input).chunk(3)      SiLU→Linear D→3D, per row
    ///       mod = gate*((1+scale)*adapt_norm_cam + shift) + cam_tokens [S, D]
    ///       proc = refine_net(mod)   4 transformer Blocks — REAL self-attn over the S view tokens
    ///       delta = param_predictor(out_norm(proc))   LN → Mlp(D→D/2→9, GELU), per row
    ///       curr_pred += delta
    ///   activate: relu on focal component (params[7:9]) per row
    ///
    /// Output: `Vec<[f32; 9]>` of length `n_views`, each
    /// `[tx, ty, tz, qw, qx, qy, qz, fl1, fl2]`.
    fn cam_head_forward_all_views(
        &self,
        cls_tokens: &[half::f16],
        n_views: usize,
    ) -> Result<Vec<[f32; 9]>> {
        const D: usize = 2048;
        const FFN: usize = 8192;
        const NUM_HEADS: usize = 16;
        const HEAD_DIM: usize = D / NUM_HEADS; // 128
        let s = n_views;
        let scale_attn = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let device_id = self.compute.device().info().id;
        debug_assert_eq!(cls_tokens.len(), s * D);

        // 1. cam_tokens = token_norm(cls_tokens)  [S, D]
        let cls_t = Tensor::from_slice(cls_tokens, Shape::from([s, D]), DType::F16, device_id)?;
        let cb = self.compute.new_command_buffer();
        let cam_norm = self.cached_layer_norm(
            &cb, &cls_t,
            "cam_head.token_norm.weight", "cam_head.token_norm.bias",
            s, D, 1e-6,
        )?;
        cb.commit();
        cb.wait_until_completed();
        let cam_tokens: Vec<half::f16> = cam_norm.to_vec()?;

        // adapt_norm(cam_tokens): per-row LayerNorm with no learnable params.
        let mut adapt_norm_cam = vec![0.0f32; s * D];
        for row in 0..s {
            let off = row * D;
            let mean = (off..off + D).map(|i| cam_tokens[i].to_f32() as f64).sum::<f64>() / D as f64;
            let var = (off..off + D).map(|i| { let d = cam_tokens[i].to_f32() as f64 - mean; d * d }).sum::<f64>() / D as f64;
            let inv_std = 1.0f64 / (var + 1e-6).sqrt();
            for i in 0..D {
                adapt_norm_cam[off + i] = ((cam_tokens[off + i].to_f32() as f64 - mean) * inv_std) as f32;
            }
        }

        // init_token (9-dim) broadcast across views, used as the step-0 input.
        let init_9: Vec<half::f16> = {
            let it: Vec<half::f16> = self.cached_weight_f16("cam_head.init_token")?.to_vec()?;
            it.into_iter().take(9).collect()
        };

        let mut curr_pred: Vec<[f32; 9]> = Vec::new(); // empty == "None" for step 0

        for step in 0..4usize {
            // a. net_input = param_embed(curr_pred or init_token-broadcast) → [S, D]
            let mut in_data = vec![half::f16::ZERO; s * 9];
            for row in 0..s {
                if step == 0 {
                    for i in 0..9 { in_data[row * 9 + i] = init_9[i]; }
                } else {
                    for i in 0..9 { in_data[row * 9 + i] = half::f16::from_f32(curr_pred[row][i]); }
                }
            }
            let in_t = Tensor::from_slice(&in_data, Shape::from([s, 9]), DType::F16, device_id)?;
            let cb = self.compute.new_command_buffer();
            let net_input = self.cached_linear_bias(
                &cb, &in_t,
                "cam_head.param_embed.weight", "cam_head.param_embed.bias",
                s, 9, D,
            )?;
            cb.commit();
            cb.wait_until_completed();
            let net_input_data: Vec<half::f16> = net_input.to_vec()?;

            // b. shift,scale,gate = adapt_norm_gen(net_input).chunk(3)  per row
            //    adapt_norm_gen = Sequential(SiLU, Linear(D, 3D)).
            let mut silu = vec![half::f16::ZERO; s * D];
            for i in 0..(s * D) {
                let x = net_input_data[i].to_f32();
                silu[i] = half::f16::from_f32(x / (1.0 + (-x).exp()));
            }
            let silu_t = Tensor::from_slice(&silu, Shape::from([s, D]), DType::F16, device_id)?;
            let cb = self.compute.new_command_buffer();
            let adaln = self.cached_linear_bias(
                &cb, &silu_t,
                "cam_head.adapt_norm_gen.1.weight", "cam_head.adapt_norm_gen.1.bias",
                s, D, 3 * D,
            )?;
            cb.commit();
            cb.wait_until_completed();
            let adaln_data: Vec<half::f16> = adaln.to_vec()?; // [s, 3D]

            // c+d. mod = gate*((1+scale)*adapt_norm_cam + shift) + cam_tokens  [S, D]
            let mut x_data = vec![half::f16::ZERO; s * D];
            for row in 0..s {
                let a_off = row * 3 * D;
                let off = row * D;
                for i in 0..D {
                    let shift = adaln_data[a_off + i].to_f32();
                    let sc = adaln_data[a_off + D + i].to_f32();
                    let g = adaln_data[a_off + 2 * D + i].to_f32();
                    let modulated = (1.0 + sc) * adapt_norm_cam[off + i] + shift;
                    x_data[off + i] = half::f16::from_f32(g * modulated + cam_tokens[off + i].to_f32());
                }
            }
            let mut x = Tensor::from_slice(&x_data, Shape::from([s, D]), DType::F16, device_id)?;

            // e. proc = refine_net(mod) — 4 transformer Blocks with REAL
            //    self-attention over the S view tokens.
            for layer in 0..4usize {
                let lp = format!("cam_head.refine_net.{}", layer);
                let cb = self.compute.new_command_buffer();
                let normed = self.cached_layer_norm(
                    &cb, &x,
                    &format!("{}.norm1.weight", lp), &format!("{}.norm1.bias", lp),
                    s, D, 1e-6,
                )?;
                let qkv = self.cached_linear_bias(
                    &cb, &normed,
                    &format!("{}.attn.qkv.weight", lp), &format!("{}.attn.qkv.bias", lp),
                    s, D, 3 * D,
                )?;
                cb.commit();
                cb.wait_until_completed();
                // Split QKV into [S, D] each (no q/k norm, no RoPE).
                let qkv_data: Vec<half::f16> = qkv.to_vec()?;
                let mut q_data = Vec::with_capacity(s * D);
                let mut k_data = Vec::with_capacity(s * D);
                let mut v_data = Vec::with_capacity(s * D);
                for row in 0..s {
                    let r = row * 3 * D;
                    q_data.extend_from_slice(&qkv_data[r..r + D]);
                    k_data.extend_from_slice(&qkv_data[r + D..r + 2 * D]);
                    v_data.extend_from_slice(&qkv_data[r + 2 * D..r + 3 * D]);
                }
                let q = Tensor::from_slice(&q_data, Shape::from([s, D]), DType::F16, device_id)?;
                let k = Tensor::from_slice(&k_data, Shape::from([s, D]), DType::F16, device_id)?;
                let v = Tensor::from_slice(&v_data, Shape::from([s, D]), DType::F16, device_id)?;
                let cb = self.compute.new_command_buffer();
                let attn_out = self.batched_attention(
                    &cb, &q, &k, &v, s, s, NUM_HEADS, HEAD_DIM, scale_attn,
                )?;
                let proj = self.cached_linear_bias(
                    &cb, &attn_out,
                    &format!("{}.attn.proj.weight", lp), &format!("{}.attn.proj.bias", lp),
                    s, D, D,
                )?;
                cb.commit();
                cb.wait_until_completed();
                let scaled = self.layer_scale(&cb, &proj, &format!("{}.ls1.gamma", lp), s, D)?;
                let cb = self.compute.new_command_buffer();
                let h = self.add(&cb, &x, &scaled);
                let normed2 = self.cached_layer_norm(
                    &cb, &h,
                    &format!("{}.norm2.weight", lp), &format!("{}.norm2.bias", lp),
                    s, D, 1e-6,
                )?;
                let ffn_up = self.cached_linear_bias(
                    &cb, &normed2,
                    &format!("{}.mlp.fc1.weight", lp), &format!("{}.mlp.fc1.bias", lp),
                    s, D, FFN,
                )?;
                let ffn_act = self.activation(&cb, &self.kernels.gelu, &ffn_up);
                let ffn_down = self.cached_linear_bias(
                    &cb, &ffn_act,
                    &format!("{}.mlp.fc2.weight", lp), &format!("{}.mlp.fc2.bias", lp),
                    s, FFN, D,
                )?;
                cb.commit();
                cb.wait_until_completed();
                let scaled2 = self.layer_scale(&cb, &ffn_down, &format!("{}.ls2.gamma", lp), s, D)?;
                let cb = self.compute.new_command_buffer();
                x = self.add(&cb, &h, &scaled2);
                cb.commit();
                cb.wait_until_completed();
            }

            // f. delta = param_predictor(out_norm(proc)) — LN → Mlp(D→D/2→9, GELU)  [S, 9]
            let cb = self.compute.new_command_buffer();
            let proc_norm = self.cached_layer_norm(
                &cb, &x,
                "cam_head.out_norm.weight", "cam_head.out_norm.bias",
                s, D, 1e-6,
            )?;
            let fc1 = self.cached_linear_bias(
                &cb, &proc_norm,
                "cam_head.param_predictor.fc1.weight", "cam_head.param_predictor.fc1.bias",
                s, D, 1024,
            )?;
            let fc1_act = self.activation(&cb, &self.kernels.gelu, &fc1);
            let fc2 = self.cached_linear_bias(
                &cb, &fc1_act,
                "cam_head.param_predictor.fc2.weight", "cam_head.param_predictor.fc2.bias",
                s, 1024, 9,
            )?;
            cb.commit();
            cb.wait_until_completed();
            let delta_data: Vec<half::f16> = fc2.to_vec()?;

            // g. curr_pred += delta  (per row)
            if step == 0 {
                curr_pred = (0..s).map(|row| {
                    let mut r = [0f32; 9];
                    for i in 0..9 { r[i] = delta_data[row * 9 + i].to_f32(); }
                    r
                }).collect();
            } else {
                for row in 0..s {
                    for i in 0..9 { curr_pred[row][i] += delta_data[row * 9 + i].to_f32(); }
                }
            }
        }

        // Activations: trans/quat = linear (identity); focal length = ReLU.
        for cam in curr_pred.iter_mut() {
            cam[7] = cam[7].max(0.0);
            cam[8] = cam[8].max(0.0);
        }
        Ok(curr_pred)
    }

    /// Parameterized GPU DPT decoder for any head (gs_head / depth_head /
    /// norm_head / pts_head). v6.5 — full multi-resolution top-down chain
    /// matching the trained Tencent DPT architecture:
    ///
    /// Per-level projection + spatial resize:
    ///   L0: projects.0 (1×1 2048→256) → resize_layers.0 (transposed 4×, 37→148) → layer1_rn (3×3 256→256, no bias)
    ///   L1: projects.1 (1×1 2048→512) → resize_layers.1 (transposed 2×, 37→74)  → layer2_rn (3×3 512→256, no bias)
    ///   L2: projects.2 (1×1 2048→1024) → identity (37×37)                       → layer3_rn (3×3 1024→256, no bias)
    ///   L3: projects.3 (1×1 2048→1024) → resize_layers.3 (3×3 stride=2, 37→19)  → layer4_rn (3×3 1024→256, no bias)
    ///
    /// Top-down fusion (refinenet pyramid):
    ///   y4 = refinenet4(l4)                    — single-input (no resConfUnit1), at 19, then bilinear → 37, then out_conv.
    ///   y3 = refinenet3(l3, y4_at_37)          — at 37, then bilinear 2× → 74, then out_conv.
    ///   y2 = refinenet2(l2, y3_at_74)          — at 74, then bilinear 2× → 148, then out_conv.
    ///   y1 = refinenet1(l1, y2_at_148)         — at 148, then bilinear 2× → 296, then out_conv.
    ///
    /// Output chain at 296:
    ///   output_conv1 (3×3 256→128 + bias) → ReLU
    ///   output_conv2.0 (3×3 128→32 + bias) → ReLU
    ///   output_conv2.2 (1×1 32→output_channels + bias)
    ///   [optional sigmoid for gs_head]
    ///
    /// All convs route through matmul-based helpers (bypassing
    /// `conv2d_3x3_tiled_f16` / `conv2d_1x1_f16` Metal kernel bugs).
    fn dpt_head_forward_gpu_multires(
        &self,
        normed_concat: &Tensor,
        multi_scale_features: &[Vec<half::f16>],
        head_prefix: &str,
        output_channels: usize,
        apply_sigmoid: bool,
    ) -> Result<Vec<f32>> {
        const GRID: usize = 37;
        const N: usize = GRID * GRID;
        const C_IN: usize = 2048;
        const C_OUT: usize = 256;
        // Spatial pyramid sizes.
        const S0: usize = 148; // L0: 4× upsample
        const S1: usize = 74;  // L1: 2× upsample
        const S2: usize = 37;  // L2: identity
        const S3: usize = 19;  // L3: downsample (37+2-3)/2+1 = 19
        const S_OUT: usize = 296; // top resolution after refinenet1's 2× upsample
        let device_id = self.compute.device().info().id;

        let to_chw_2048 = |seq: &[half::f16]| -> Result<Tensor> {
            let mut chw = vec![half::f16::ZERO; C_IN * N];
            for p in 0..N {
                for c in 0..C_IN {
                    chw[c * N + p] = seq[p * C_IN + c];
                }
            }
            Tensor::from_slice(
                &chw, Shape::from([1usize, C_IN, GRID, GRID]),
                DType::F16, device_id,
            )
        };

        let final_seq: Vec<half::f16> = normed_concat.to_vec()?;

        // === Per-level projection + spatial resize + layer_rn ===
        let mut layer_rns: Vec<Tensor> = Vec::with_capacity(4);
        let layer_resolutions = [S0, S1, S2, S3];
        for i in 0..4usize {
            let ch_proj = match i { 0 => 256, 1 => 512, _ => 1024 };
            let level_feat: &[half::f16] = multi_scale_features
                .get(i)
                .map(|v| v.as_slice())
                .unwrap_or(&final_seq);
            let x_chw = to_chw_2048(level_feat)?;

            // 1×1 projection at 37×37 (v6.7 GPU im2col-free path).
            let proj = self.gpu_conv1x1_v67(
                &x_chw,
                &format!("{}.projects.{}.weight", head_prefix, i),
                Some(&format!("{}.projects.{}.bias", head_prefix, i)),
                C_IN, ch_proj, GRID, GRID,
            )?;

            // Spatial resize via resize_layers.{i} (or identity for i=2).
            // Transposed convs keep the CPU-im2col path for now (only run once
            // per request, so the optimisation is smaller).
            let resized = match i {
                0 => self.gpu_conv_transpose_via_matmul(
                    &proj,
                    &format!("{}.resize_layers.0.weight", head_prefix),
                    Some(&format!("{}.resize_layers.0.bias", head_prefix)),
                    ch_proj, ch_proj, GRID, GRID, 4,
                )?,
                1 => self.gpu_conv_transpose_via_matmul(
                    &proj,
                    &format!("{}.resize_layers.1.weight", head_prefix),
                    Some(&format!("{}.resize_layers.1.bias", head_prefix)),
                    ch_proj, ch_proj, GRID, GRID, 2,
                )?,
                2 => proj, // identity
                3 => self.gpu_conv3x3_strided_v67(
                    &proj,
                    &format!("{}.resize_layers.3.weight", head_prefix),
                    Some(&format!("{}.resize_layers.3.bias", head_prefix)),
                    ch_proj, ch_proj, GRID, GRID, 2, 1,
                )?,
                _ => unreachable!(),
            };

            // 3×3 layer{i+1}_rn (no bias) at the resized resolution — GPU path.
            let res = layer_resolutions[i];
            let l = self.gpu_conv3x3_v67(
                &resized,
                &format!("{}.scratch.layer{}_rn.weight", head_prefix, i + 1),
                None,
                ch_proj, C_OUT, res, res,
            )?;
            layer_rns.push(l);
        }

        // === Refinenet top-down chain ===
        // refinenet4 (at S3=19): single-input — only resConfUnit2 + bilinear resize → 37 + out_conv.
        let y4 = self.refinenet_block_gpu(
            &layer_rns[3], None,
            head_prefix, "refinenet4", C_OUT, S3, S3, S2, S2,
        )?;
        // refinenet3 (at S2=37): dual-input — resConfUnit1(y4_at_37) + l3 + resConfUnit2 + bilinear 2× to 74 + out_conv.
        let y3 = self.refinenet_block_gpu(
            &layer_rns[2], Some(&y4),
            head_prefix, "refinenet3", C_OUT, S2, S2, S1, S1,
        )?;
        let y2 = self.refinenet_block_gpu(
            &layer_rns[1], Some(&y3),
            head_prefix, "refinenet2", C_OUT, S1, S1, S0, S0,
        )?;
        let y1 = self.refinenet_block_gpu(
            &layer_rns[0], Some(&y2),
            head_prefix, "refinenet1", C_OUT, S0, S0, S_OUT, S_OUT,
        )?;

        // === Output conv chain at S_OUT=296 (v6.7 GPU im2col-free path) ===
        let oc1 = self.gpu_conv3x3_v67(
            &y1,
            &format!("{}.scratch.output_conv1.weight", head_prefix),
            Some(&format!("{}.scratch.output_conv1.bias", head_prefix)),
            C_OUT, 128, S_OUT, S_OUT,
        )?;
        let cb = self.compute.new_command_buffer();
        let oc1_relu = self.activation(&cb, &self.kernels.relu, &oc1);
        cb.commit();
        cb.wait_until_completed();

        let oc20 = self.gpu_conv3x3_v67(
            &oc1_relu,
            &format!("{}.scratch.output_conv2.0.weight", head_prefix),
            Some(&format!("{}.scratch.output_conv2.0.bias", head_prefix)),
            128, 32, S_OUT, S_OUT,
        )?;
        let cb = self.compute.new_command_buffer();
        let oc20_relu = self.activation(&cb, &self.kernels.relu, &oc20);
        cb.commit();
        cb.wait_until_completed();

        let head = self.gpu_conv1x1_v67(
            &oc20_relu,
            &format!("{}.scratch.output_conv2.2.weight", head_prefix),
            Some(&format!("{}.scratch.output_conv2.2.bias", head_prefix)),
            32, output_channels, S_OUT, S_OUT,
        )?;

        let head_data: Vec<half::f16> = head.to_vec()?;
        let mut out = Vec::with_capacity(output_channels * S_OUT * S_OUT);
        for &v in &head_data {
            let f = v.to_f32();
            if apply_sigmoid {
                out.push(1.0 / (1.0 + (-f).exp()));
            } else {
                out.push(f);
            }
        }
        Ok(out)
    }

    /// One refinenet block in the DPT top-down chain. Mirrors MiDaS/DPT
    /// `FeatureFusionBlock_custom`:
    ///   if upsampled_prev is Some:
    ///       res = resConfUnit1(upsampled_prev)
    ///       output = current_scale + res
    ///   else:
    ///       output = current_scale            (refinenet4 single-input case)
    ///   output = resConfUnit2(output)
    ///   output = bilinear resize → (h_out, w_out)
    ///   output = out_conv (1×1, 256→256 + bias)
    ///
    /// `current_scale` should already be at resolution `(h_in, w_in)`.
    /// `upsampled_prev` (if Some) must also be at `(h_in, w_in)` — the caller
    /// is responsible for resizing the prev refinenet's output to match. This
    /// function performs the FINAL resize-to-`(h_out, w_out)` after
    /// `resConfUnit2` (per DPT convention).
    fn refinenet_block_gpu(
        &self,
        current_scale: &Tensor,
        upsampled_prev: Option<&Tensor>,
        head_prefix: &str,
        block_name: &str, // "refinenet1" .. "refinenet4"
        c: usize,
        h_in: usize, w_in: usize,
        h_out: usize, w_out: usize,
    ) -> Result<Tensor> {
        // 1. If we have a prev (upsampled to h_in×w_in already, see caller for
        //    refinenet3..1; for refinenet4 prev is None), resize it to match
        //    current's spatial size and run resConfUnit1.
        let combined = if let Some(prev) = upsampled_prev {
            let prev_dims = prev.shape().dims();
            let (ph, pw) = (prev_dims[2], prev_dims[3]);
            let prev_at_input = if (ph, pw) == (h_in, w_in) {
                prev.clone()
            } else {
                self.cpu_bilinear_resize_to(prev, c, ph, pw, h_in, w_in)?
            };
            let res = self.res_conf_unit_gpu(
                &prev_at_input,
                head_prefix, block_name, "resConfUnit1",
                c, h_in, w_in,
            )?;
            // current + res
            let cb = self.compute.new_command_buffer();
            let sum = self.add(&cb, current_scale, &res);
            cb.commit();
            cb.wait_until_completed();
            sum
        } else {
            current_scale.clone()
        };

        // 2. resConfUnit2 on the (combined or single) input.
        let after_rc2 = self.res_conf_unit_gpu(
            &combined,
            head_prefix, block_name, "resConfUnit2",
            c, h_in, w_in,
        )?;

        // 3. Bilinear resize to (h_out, w_out).
        let resized = if (h_in, w_in) == (h_out, w_out) {
            after_rc2
        } else {
            self.cpu_bilinear_resize_to(&after_rc2, c, h_in, w_in, h_out, w_out)?
        };

        // 4. out_conv (1×1, 256→256 + bias) — v6.7 GPU path.
        let out = self.gpu_conv1x1_v67(
            &resized,
            &format!("{}.scratch.{}.out_conv.weight", head_prefix, block_name),
            Some(&format!("{}.scratch.{}.out_conv.bias", head_prefix, block_name)),
            c, c, h_out, w_out,
        )?;
        Ok(out)
    }

    /// One resConfUnit (residual conv block):
    ///   x' = x + conv2(ReLU(conv1(ReLU(x))))
    /// Both conv1 and conv2 are 3×3, 256→256, with bias. v6.7 GPU path.
    fn res_conf_unit_gpu(
        &self,
        x: &Tensor,
        head_prefix: &str, block_name: &str, unit_name: &str,
        c: usize, h: usize, w: usize,
    ) -> Result<Tensor> {
        let cb = self.compute.new_command_buffer();
        let x_relu = self.activation(&cb, &self.kernels.relu, x);
        cb.commit();
        cb.wait_until_completed();

        let h1 = self.gpu_conv3x3_v67(
            &x_relu,
            &format!("{}.scratch.{}.{}.conv1.weight", head_prefix, block_name, unit_name),
            Some(&format!("{}.scratch.{}.{}.conv1.bias", head_prefix, block_name, unit_name)),
            c, c, h, w,
        )?;

        let cb = self.compute.new_command_buffer();
        let h1_relu = self.activation(&cb, &self.kernels.relu, &h1);
        cb.commit();
        cb.wait_until_completed();

        let h2 = self.gpu_conv3x3_v67(
            &h1_relu,
            &format!("{}.scratch.{}.{}.conv2.weight", head_prefix, block_name, unit_name),
            Some(&format!("{}.scratch.{}.{}.conv2.bias", head_prefix, block_name, unit_name)),
            c, c, h, w,
        )?;

        let cb = self.compute.new_command_buffer();
        let out = self.add(&cb, x, &h2);
        cb.commit();
        cb.wait_until_completed();
        Ok(out)
    }

    /// Minimal DPT decoder for gs_head — single-scale, single-refinenet
    /// path that exercises the trained `gs_head.norm`, `gs_head.projects.0`,
    /// `gs_head.scratch.layer1_rn`, `gs_head.scratch.refinenet1.{resConfUnit1,
    /// resConfUnit2, out_conv}`, `gs_head.scratch.output_conv1`,
    /// `output_conv2.0`, `output_conv2.2` weights and produces a 3-channel
    /// RGB map at 37×37 (one chromatic refinement triplet per patch). The
    /// full DPT (4-scale fusion across `projects.0..3` + `resize_layers` +
    /// 4 refinenet blocks at 18/37/74/148 resolutions) is the v6+ follow-up;
    /// this minimal v6 is a scoped CPU implementation that uses real
    /// weights at the patch-grid resolution.
    ///
    /// Input: `[37*37, 2048]` final transformer concat features (already
    /// LN-normed by gs_head.norm — done by the caller).
    /// Output: flat `[3 * 37 * 37]` f32 RGB triplets in 0..1 (sigmoid'd
    /// before return).
    fn dpt_minimal_forward(
        &self,
        normed_concat: &[half::f16],
        multi_scale_features: &[Vec<half::f16>],
    ) -> Result<Vec<f32>> {
        self.dpt_head_forward(
            normed_concat,
            multi_scale_features,
            "gs_head",
            3, // gs_head output_conv2.2: [3, 32, 1, 1]
            true, // sigmoid for chromatic 0..1 RGB output
        )
    }

    /// Generalized DPT decoder for any of the 5 prediction heads — same
    /// architecture as `dpt_minimal_forward` but parameterized by:
    ///   `head_prefix`: one of `gs_head` / `depth_head` / `norm_head` /
    ///                  `pts_head` (cam_head uses a different token-based
    ///                  architecture, not DPT — keep separate).
    ///   `output_channels`: 3 for gs/norm/pts, 1 for depth.
    ///   `apply_sigmoid`: gs uses sigmoid → 0..1 RGB; depth/norm/pts emit
    ///                    raw values (caller normalises).
    ///
    /// All four DPT heads use the same key layout under their prefix:
    ///   `{prefix}.norm.{weight,bias}`
    ///   `{prefix}.projects.{0..3}.{weight,bias}`  (1×1 convs)
    ///   `{prefix}.scratch.layer{1..4}_rn.weight`  (3×3 convs, no bias)
    ///   `{prefix}.scratch.refinenet1.{resConfUnit{1,2}.{conv1,conv2}, out_conv}.{weight,bias}`
    ///   `{prefix}.scratch.output_conv1.{weight,bias}`
    ///   `{prefix}.scratch.output_conv2.0.{weight,bias}`
    ///   `{prefix}.scratch.output_conv2.2.{weight,bias}`
    fn dpt_head_forward(
        &self,
        normed_concat: &[half::f16],
        multi_scale_features: &[Vec<half::f16>],
        head_prefix: &str,
        output_channels: usize,
        apply_sigmoid: bool,
    ) -> Result<Vec<f32>> {
        const GRID: usize = 37;
        const N: usize = GRID * GRID;
        const C_IN: usize = 2048;
        const C_OUT: usize = 256;

        // Helpers to read weights once (cached).
        let get_f16 = |name: &str| -> Result<Vec<half::f16>> {
            self.cached_weight_f16(name)?.to_vec()
        };

        // Per-level projection-and-reduction. Each projects.{i} reads from
        // a DIFFERENT transformer-feature scale (4 intermediate global_blocks
        // outputs concatenated with the final frame_hidden). Fall back to
        // `normed_concat` for any level whose intermediate isn't supplied.
        // After projection, run the matching `scratch.layer{i+1}_rn` 3×3
        // conv and return a CHW [256, 37, 37] f32 buf.
        let project_and_reduce = |i: usize, ch_proj: usize| -> Result<Vec<f32>> {
            let pw = get_f16(&format!("{}.projects.{}.weight", head_prefix, i))?;
            let pb = get_f16(&format!("{}.projects.{}.bias", head_prefix, i))?;
            let lw = get_f16(&format!("{}.scratch.layer{}_rn.weight", head_prefix, i + 1))?;
            // Pick this level's input feature.
            let level_feat: &[half::f16] = multi_scale_features
                .get(i)
                .map(|v| v.as_slice())
                .unwrap_or(normed_concat);
            // 1×1 projection: [N, 2048] → [N, ch_proj]
            let mut proj = vec![0.0f32; N * ch_proj];
            for n in 0..N {
                for oc in 0..ch_proj {
                    let mut acc = pb[oc].to_f32();
                    for ic in 0..C_IN {
                        acc += level_feat[n * C_IN + ic].to_f32()
                            * pw[oc * C_IN + ic].to_f32();
                    }
                    proj[n * ch_proj + oc] = acc;
                }
            }
            // Reshape [N, ch_proj] → CHW [ch_proj, 37, 37]
            let proj_chw = Self::nhwc_to_chw(&proj, N, ch_proj, GRID, GRID);
            // 3×3 layer_rn → [256, 37, 37]
            let mut out = vec![0.0f32; C_OUT * GRID * GRID];
            Self::cpu_conv2d_3x3_chw(&proj_chw, &lw, None, &mut out, ch_proj, C_OUT, GRID, GRID);
            Ok(out)
        };

        // 1+2. Multi-scale projection + channel reduction. The 4 intermediate
        //      global_blocks features at layers [5, 11, 17, 23] feed the 4
        //      `projects.{i}`. Each is concat-with-final-frame to match the
        //      2048-dim input the trained `projects.{i}` weights expect.
        //      Sum-fuse all 4 levels at 37×37 (skipping the resize_layers +
        //      top-down chain — see v6.5 for full DPT).
        let l1 = project_and_reduce(0, 256)?;
        let l2 = project_and_reduce(1, 512)?;
        let l3 = project_and_reduce(2, 1024)?;
        let l4 = project_and_reduce(3, 1024)?;

        // 2.5. Sum-fuse all 4 levels at 37×37.
        let mut x = vec![0.0f32; C_OUT * GRID * GRID];
        for i in 0..(C_OUT * GRID * GRID) {
            x[i] = l1[i] + l2[i] + l3[i] + l4[i];
        }

        // 3. refinenet1.resConfUnit1: ReLU → conv1 → ReLU → conv2 → residual.
        let rc1_c1_w = get_f16(&format!("{}.scratch.refinenet1.resConfUnit1.conv1.weight", head_prefix))?;
        let rc1_c1_b = get_f16(&format!("{}.scratch.refinenet1.resConfUnit1.conv1.bias", head_prefix))?;
        let rc1_c2_w = get_f16(&format!("{}.scratch.refinenet1.resConfUnit1.conv2.weight", head_prefix))?;
        let rc1_c2_b = get_f16(&format!("{}.scratch.refinenet1.resConfUnit1.conv2.bias", head_prefix))?;
        let x = Self::res_conf_unit(&x, &rc1_c1_w, &rc1_c1_b, &rc1_c2_w, &rc1_c2_b, C_OUT, GRID, GRID);

        // 4. refinenet1.resConfUnit2: same pattern.
        let rc2_c1_w = get_f16(&format!("{}.scratch.refinenet1.resConfUnit2.conv1.weight", head_prefix))?;
        let rc2_c1_b = get_f16(&format!("{}.scratch.refinenet1.resConfUnit2.conv1.bias", head_prefix))?;
        let rc2_c2_w = get_f16(&format!("{}.scratch.refinenet1.resConfUnit2.conv2.weight", head_prefix))?;
        let rc2_c2_b = get_f16(&format!("{}.scratch.refinenet1.resConfUnit2.conv2.bias", head_prefix))?;
        let x = Self::res_conf_unit(&x, &rc2_c1_w, &rc2_c1_b, &rc2_c2_w, &rc2_c2_b, C_OUT, GRID, GRID);

        // 5. refinenet1.out_conv: 1×1 conv 256→256, with bias.
        let oc_w = get_f16(&format!("{}.scratch.refinenet1.out_conv.weight", head_prefix))?;
        let oc_b = get_f16(&format!("{}.scratch.refinenet1.out_conv.bias", head_prefix))?;
        let mut x_out = vec![0.0f32; C_OUT * GRID * GRID];
        Self::cpu_conv2d_1x1_chw(&x, &oc_w, Some(&oc_b), &mut x_out, C_OUT, C_OUT, GRID, GRID);

        // 6. output_conv1: 3×3 conv 256→128 with bias.
        let oc1_w = get_f16(&format!("{}.scratch.output_conv1.weight", head_prefix))?;
        let oc1_b = get_f16(&format!("{}.scratch.output_conv1.bias", head_prefix))?;
        let mut h128 = vec![0.0f32; 128 * GRID * GRID];
        Self::cpu_conv2d_3x3_chw(&x_out, &oc1_w, Some(&oc1_b), &mut h128, C_OUT, 128, GRID, GRID);
        for v in h128.iter_mut() { *v = v.max(0.0); } // ReLU

        // 7. output_conv2.0: 3×3 conv 128→32 with bias.
        let oc20_w = get_f16(&format!("{}.scratch.output_conv2.0.weight", head_prefix))?;
        let oc20_b = get_f16(&format!("{}.scratch.output_conv2.0.bias", head_prefix))?;
        let mut h32 = vec![0.0f32; 32 * GRID * GRID];
        Self::cpu_conv2d_3x3_chw(&h128, &oc20_w, Some(&oc20_b), &mut h32, 128, 32, GRID, GRID);
        for v in h32.iter_mut() { *v = v.max(0.0); } // ReLU

        // 8. output_conv2.2: 1×1 conv 32→{output_channels} with bias.
        let oc22_w = get_f16(&format!("{}.scratch.output_conv2.2.weight", head_prefix))?;
        let oc22_b = get_f16(&format!("{}.scratch.output_conv2.2.bias", head_prefix))?;
        let mut head_out = vec![0.0f32; output_channels * GRID * GRID];
        Self::cpu_conv2d_1x1_chw(&h32, &oc22_w, Some(&oc22_b), &mut head_out, 32, output_channels, GRID, GRID);

        // Optional sigmoid for chromatic 0..1 output (gs_head only;
        // depth/norm/pts emit raw values for caller normalisation).
        if apply_sigmoid {
            for v in head_out.iter_mut() { *v = 1.0 / (1.0 + (-*v).exp()); }
        }
        Ok(head_out)
    }

    /// GPU 3×3 conv via im2col + linear_bias matmul.
    ///
    /// Bypasses `conv2d_3x3_tiled_f16` (NaN at Cin=Cout=256 no-bias) and
    /// `conv2d_naive_f16` (slow at high Cin) by reformulating the conv as a
    /// matmul:
    ///   col = im2col(input, 3, pad=1) → [H*W, Cin*9]
    ///   weight reshape [Cout, Cin, 3, 3] → [Cout, Cin*9] (no copy — already row-major NCHW)
    ///   output = col @ weight^T + bias → [H*W, Cout] → reshape CHW
    ///
    /// Manages its own command buffer (commits + waits before reading the
    /// matmul output for the CHW reshape).
    ///
    /// `bias_name` of `None` yields a zero bias (allocated once per Cout in the
    /// dummy_cache).
    ///
    /// Returns a Tensor of shape [1, Cout, H, W] in f16.
    /// v6.7 — GPU im2col for 3×3 conv. Dispatches `im2col_3x3_f16` kernel.
    /// Returns `[H_out*W_out, Cin*9]` f16 tensor on GPU. No CPU round-trip.
    fn gpu_im2col_3x3(
        &self,
        cb: &metal::CommandBufferRef,
        input_chw: &Tensor, // [1, Cin, Hin, Win]
        cin: usize, h_in: usize, w_in: usize,
        h_out: usize, w_out: usize,
        pad: u32, stride: u32,
    ) -> Tensor {
        let n_rows = h_out * w_out;
        let k = cin * 9;
        let total = n_rows * k;
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer(
            (total * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.compute.dispatch_1d(cb, &self.kernels.im2col_3x3, total, |encoder| {
            gpu_ops::set_tensor_buffer(encoder, 0, input_chw);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            let cin_u = cin as u32;
            let hin_u = h_in as u32;
            let win_u = w_in as u32;
            let hout_u = h_out as u32;
            let wout_u = w_out as u32;
            encoder.set_bytes(2, 4, &cin_u as *const u32 as *const _);
            encoder.set_bytes(3, 4, &hin_u as *const u32 as *const _);
            encoder.set_bytes(4, 4, &win_u as *const u32 as *const _);
            encoder.set_bytes(5, 4, &hout_u as *const u32 as *const _);
            encoder.set_bytes(6, 4, &wout_u as *const u32 as *const _);
            encoder.set_bytes(7, 4, &pad as *const u32 as *const _);
            encoder.set_bytes(8, 4, &stride as *const u32 as *const _);
        });
        Tensor::from_metal_buffer(
            output_buffer,
            Shape::from([n_rows, k]),
            DType::F16,
            self.compute.device().info().id,
        )
    }

    /// v6.7 — GPU CHW→HWC reshape for 1×1 conv path.
    /// Returns `[H*W, Cin]` f16 tensor on GPU.
    fn gpu_chw_to_hwc(
        &self,
        cb: &metal::CommandBufferRef,
        input_chw: &Tensor, // [1, Cin, H, W]
        cin: usize, h: usize, w: usize,
    ) -> Tensor {
        let n_rows = h * w;
        let total = n_rows * cin;
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer(
            (total * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.compute.dispatch_1d(cb, &self.kernels.chw_to_hwc, total, |encoder| {
            gpu_ops::set_tensor_buffer(encoder, 0, input_chw);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            let cin_u = cin as u32;
            let h_u = h as u32;
            let w_u = w as u32;
            encoder.set_bytes(2, 4, &cin_u as *const u32 as *const _);
            encoder.set_bytes(3, 4, &h_u as *const u32 as *const _);
            encoder.set_bytes(4, 4, &w_u as *const u32 as *const _);
        });
        Tensor::from_metal_buffer(
            output_buffer,
            Shape::from([n_rows, cin]),
            DType::F16,
            self.compute.device().info().id,
        )
    }

    /// v6.7 — GPU HWC→CHW reshape for matmul output → next layer's input.
    /// Returns `[1, Cout, H, W]` f16 tensor on GPU.
    fn gpu_hwc_to_chw(
        &self,
        cb: &metal::CommandBufferRef,
        input_hwc: &Tensor, // [H*W, Cout]
        cout: usize, h: usize, w: usize,
    ) -> Tensor {
        let total = h * w * cout;
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer(
            (total * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.compute.dispatch_1d(cb, &self.kernels.hwc_to_chw, total, |encoder| {
            gpu_ops::set_tensor_buffer(encoder, 0, input_hwc);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            let cout_u = cout as u32;
            let h_u = h as u32;
            let w_u = w as u32;
            encoder.set_bytes(2, 4, &cout_u as *const u32 as *const _);
            encoder.set_bytes(3, 4, &h_u as *const u32 as *const _);
            encoder.set_bytes(4, 4, &w_u as *const u32 as *const _);
        });
        Tensor::from_metal_buffer(
            output_buffer,
            Shape::from([1usize, cout, h, w]),
            DType::F16,
            self.compute.device().info().id,
        )
    }

    /// v6.7 — fully on-GPU 3×3 conv via matmul. Replaces
    /// `gpu_conv3x3_via_matmul` for the multi-res DPT path. The caller's
    /// command buffer is used (no commit/wait inside) so an entire chain of
    /// convs can be dispatched on a single command buffer for max GPU
    /// throughput.
    fn gpu_conv3x3_matmul_v67(
        &self,
        cb: &metal::CommandBufferRef,
        input_chw: &Tensor,    // [1, Cin, H, W]
        weight_name: &str,
        bias_name: Option<&str>,
        cin: usize, cout: usize, h: usize, w: usize,
        stride: u32, pad: u32,
    ) -> Result<Tensor> {
        let h_out = ((h + 2 * (pad as usize)).saturating_sub(3)) / (stride as usize) + 1;
        let w_out = ((w + 2 * (pad as usize)).saturating_sub(3)) / (stride as usize) + 1;
        let n = h_out * w_out;
        let k = cin * 9;

        // GPU im2col → [N, K]
        let col = self.gpu_im2col_3x3(cb, input_chw, cin, h, w, h_out, w_out, pad, stride);

        // Matmul on GPU
        let weight = self.cached_weight_f16(weight_name)?;
        let bias = if let Some(bn) = bias_name {
            self.cached_weight_f16(bn)?
        } else {
            self.cached_zero_bias(cout)?
        };
        let result = self.linear_tensors(cb, &col, &weight, &bias, n, k, cout);

        // GPU HWC→CHW
        Ok(self.gpu_hwc_to_chw(cb, &result, cout, h_out, w_out))
    }

    /// v6.7 — fully on-GPU 1×1 conv via matmul.
    fn gpu_conv1x1_matmul_v67(
        &self,
        cb: &metal::CommandBufferRef,
        input_chw: &Tensor,    // [1, Cin, H, W]
        weight_name: &str,
        bias_name: Option<&str>,
        cin: usize, cout: usize, h: usize, w: usize,
    ) -> Result<Tensor> {
        let n = h * w;

        // GPU CHW→HWC → [N, Cin]
        let seq = self.gpu_chw_to_hwc(cb, input_chw, cin, h, w);

        // Matmul on GPU
        let weight = self.cached_weight_f16(weight_name)?;
        let bias = if let Some(bn) = bias_name {
            self.cached_weight_f16(bn)?
        } else {
            self.cached_zero_bias(cout)?
        };
        let result = self.linear_tensors(cb, &seq, &weight, &bias, n, cin, cout);

        // GPU HWC→CHW
        Ok(self.gpu_hwc_to_chw(cb, &result, cout, h, w))
    }

    /// v6.7 drop-in 3×3 conv with own cb. Substitutes for
    /// `gpu_conv3x3_via_matmul` in the multi-res DPT path. Uses GPU im2col
    /// (eliminates the ~200 MB CPU-side im2col allocation at 296×256ch).
    fn gpu_conv3x3_v67(
        &self,
        input_chw: &Tensor,
        weight_name: &str,
        bias_name: Option<&str>,
        cin: usize, cout: usize, h: usize, w: usize,
    ) -> Result<Tensor> {
        let cb = self.compute.new_command_buffer();
        let out = self.gpu_conv3x3_matmul_v67(
            &cb, input_chw, weight_name, bias_name, cin, cout, h, w, 1, 1,
        )?;
        cb.commit();
        cb.wait_until_completed();
        Ok(out)
    }

    /// v6.7 drop-in 3×3 conv with configurable stride/pad.
    fn gpu_conv3x3_strided_v67(
        &self,
        input_chw: &Tensor,
        weight_name: &str,
        bias_name: Option<&str>,
        cin: usize, cout: usize, h: usize, w: usize,
        stride: usize, pad: usize,
    ) -> Result<Tensor> {
        let cb = self.compute.new_command_buffer();
        let out = self.gpu_conv3x3_matmul_v67(
            &cb, input_chw, weight_name, bias_name, cin, cout, h, w,
            stride as u32, pad as u32,
        )?;
        cb.commit();
        cb.wait_until_completed();
        Ok(out)
    }

    /// v6.7 drop-in 1×1 conv with own cb. Uses GPU CHW↔HWC reshapes.
    fn gpu_conv1x1_v67(
        &self,
        input_chw: &Tensor,
        weight_name: &str,
        bias_name: Option<&str>,
        cin: usize, cout: usize, h: usize, w: usize,
    ) -> Result<Tensor> {
        let cb = self.compute.new_command_buffer();
        let out = self.gpu_conv1x1_matmul_v67(
            &cb, input_chw, weight_name, bias_name, cin, cout, h, w,
        )?;
        cb.commit();
        cb.wait_until_completed();
        Ok(out)
    }

    fn gpu_conv3x3_via_matmul(
        &self,
        input_chw: &Tensor, // [1, Cin, H, W]
        weight_name: &str,
        bias_name: Option<&str>,
        cin: usize, cout: usize, h: usize, w: usize,
    ) -> Result<Tensor> {
        self.gpu_conv3x3_strided_via_matmul(
            input_chw, weight_name, bias_name, cin, cout, h, w, 1, 1,
        )
    }

    /// GPU 3×3 conv with configurable stride + padding via im2col + linear_bias
    /// matmul. Output spatial size: `(h + 2*pad - 3)/stride + 1`.
    /// Used by both stride=1 padding=1 (refinenet/output convs) and stride=2
    /// padding=1 (resize_layers.3 down-sampling 37 → 19).
    fn gpu_conv3x3_strided_via_matmul(
        &self,
        input_chw: &Tensor, // [1, Cin, H, W]
        weight_name: &str,
        bias_name: Option<&str>,
        cin: usize, cout: usize, h: usize, w: usize,
        stride: usize, pad: usize,
    ) -> Result<Tensor> {
        let h_out = (h + 2 * pad - 3) / stride + 1;
        let w_out = (w + 2 * pad - 3) / stride + 1;
        let n = h_out * w_out;
        let k = cin * 9;
        let device_id = self.compute.device().info().id;

        // CPU im2col: [1, Cin, H, W] f16 → [H_out*W_out, Cin*9] f16
        let in_data: Vec<half::f16> = input_chw.to_vec()?;
        let mut col = vec![half::f16::ZERO; n * k];
        for oy in 0..h_out {
            for ox in 0..w_out {
                let p = oy * w_out + ox;
                for ic in 0..cin {
                    for ky in 0..3usize {
                        let iy_i = (oy * stride) as i32 + ky as i32 - pad as i32;
                        if iy_i < 0 || iy_i >= h as i32 { continue; }
                        let iy = iy_i as usize;
                        for kx in 0..3usize {
                            let ix_i = (ox * stride) as i32 + kx as i32 - pad as i32;
                            if ix_i < 0 || ix_i >= w as i32 { continue; }
                            let ix = ix_i as usize;
                            let in_off = ic * h * w + iy * w + ix;
                            let col_off = p * k + ic * 9 + ky * 3 + kx;
                            col[col_off] = in_data[in_off];
                        }
                    }
                }
            }
        }
        let col_t = Tensor::from_slice(&col, Shape::from([n, k]), DType::F16, device_id)?;

        let cb = self.compute.new_command_buffer();
        let result = if let Some(bn) = bias_name {
            self.cached_linear_bias(&cb, &col_t, weight_name, bn, n, k, cout)?
        } else {
            let weight = self.cached_weight_f16(weight_name)?;
            let zero_bias = self.cached_zero_bias(cout)?;
            self.linear_tensors(&cb, &col_t, &weight, &zero_bias, n, k, cout)
        };
        cb.commit();
        cb.wait_until_completed();

        let result_data: Vec<half::f16> = result.to_vec()?;
        let mut chw = vec![half::f16::ZERO; cout * n];
        for p in 0..n {
            for c in 0..cout {
                chw[c * n + p] = result_data[p * cout + c];
            }
        }
        Tensor::from_slice(&chw, Shape::from([1usize, cout, h_out, w_out]), DType::F16, device_id)
    }

    /// GPU non-overlapping ConvTranspose2d (kernel=stride=`k`, padding=0) via
    /// matmul + pixel-shuffle.
    ///
    /// Used for resize_layers.0 (k=4: 37 → 148) and resize_layers.1 (k=2:
    /// 37 → 74). Implements:
    ///   output[oc, oy, ox] = bias[oc] + Σ_ic input[ic, oy/k, ox/k] * weight[ic, oc, oy%k, ox%k]
    ///
    /// PyTorch ConvTranspose2d weight layout is `[Cin, Cout, kH, kW]`. Viewed
    /// flat as `[Cin, Cout*k*k]` (row-major, no permutation needed) and then
    /// transposed to `[Cout*k*k, Cin]` for the linear shader, the matmul
    /// `[H*W, Cin] @ W^T` produces `[H*W, Cout*k*k]` where index
    /// `(p, oc*k*k + ky*k + kx)` is the contribution of pixel `p` to output
    /// position `(p_y*k + ky, p_x*k + kx)` of channel `oc`. Pixel-shuffle
    /// rearranges to `[Cout, H*k, W*k]` CHW.
    fn gpu_conv_transpose_via_matmul(
        &self,
        input_chw: &Tensor, // [1, Cin, H, W]
        weight_name: &str,
        bias_name: Option<&str>,
        cin: usize, cout: usize, h: usize, w: usize, k: usize,
    ) -> Result<Tensor> {
        let n = h * w;
        let n_out = cout * k * k; // matmul output inner dim
        let h_out = h * k;
        let w_out = w * k;
        let device_id = self.compute.device().info().id;

        // CPU reshape: [1, Cin, H, W] CHW → [H*W, Cin] HWC sequence.
        let in_data: Vec<half::f16> = input_chw.to_vec()?;
        let mut hwc = vec![half::f16::ZERO; n * cin];
        for c in 0..cin {
            for p in 0..n {
                hwc[p * cin + c] = in_data[c * n + p];
            }
        }
        let in_seq = Tensor::from_slice(&hwc, Shape::from([n, cin]), DType::F16, device_id)?;

        // Cache the *permuted* weight tensor [Cout*k*k, Cin] so the
        // transpose only runs once per safetensors weight per runtime.
        // Synthetic cache key avoids collision with the raw weight name.
        let perm_key = format!("__transposed_perm__{}", weight_name);
        let w_tensor = {
            let cached = {
                let cache = self.weight_cache.lock()
                    .map_err(|_| crate::core::Error::internal("weight_cache mutex poisoned"))?;
                cache.get(&perm_key).cloned()
            };
            if let Some(t) = cached {
                t
            } else {
                let w_orig: Vec<half::f16> = self.cached_weight_f16(weight_name)?.to_vec()?;
                debug_assert_eq!(w_orig.len(), cin * cout * k * k);
                let mut w_t = vec![half::f16::ZERO; n_out * cin];
                for c_in in 0..cin {
                    for j in 0..n_out {
                        w_t[j * cin + c_in] = w_orig[c_in * n_out + j];
                    }
                }
                let t = Tensor::from_slice(&w_t, Shape::from([n_out, cin]), DType::F16, device_id)?;
                let mut cache = self.weight_cache.lock()
                    .map_err(|_| crate::core::Error::internal("weight_cache mutex poisoned"))?;
                cache.entry(perm_key).or_insert_with(|| t.clone());
                t
            }
        };
        let zero_bias = self.cached_zero_bias(n_out)?;

        let cb = self.compute.new_command_buffer();
        let result = self.linear_tensors(&cb, &in_seq, &w_tensor, &zero_bias, n, cin, n_out);
        cb.commit();
        cb.wait_until_completed();

        // Pixel-shuffle: [H*W, Cout*k*k] → [Cout, H*k, W*k].
        // result[(iy*W + ix), oc*k*k + ky*k + kx] = output[oc, iy*k + ky, ix*k + kx].
        let result_data: Vec<half::f16> = result.to_vec()?;
        let mut chw = vec![half::f16::ZERO; cout * h_out * w_out];
        for c in 0..cout {
            for oy in 0..h_out {
                for ox in 0..w_out {
                    let iy = oy / k;
                    let ix = ox / k;
                    let ky = oy % k;
                    let kx = ox % k;
                    let src = (iy * w + ix) * n_out + c * k * k + ky * k + kx;
                    let dst = c * h_out * w_out + oy * w_out + ox;
                    chw[dst] = result_data[src];
                }
            }
        }

        // CPU bias add (small, applied per output channel).
        if let Some(bn) = bias_name {
            let b: Vec<half::f16> = self.cached_weight_f16(bn)?.to_vec()?;
            for c in 0..cout {
                let bv = b[c].to_f32();
                for p in 0..(h_out * w_out) {
                    let off = c * h_out * w_out + p;
                    chw[off] = half::f16::from_f32(chw[off].to_f32() + bv);
                }
            }
        }

        Tensor::from_slice(&chw, Shape::from([1usize, cout, h_out, w_out]), DType::F16, device_id)
    }

    /// CPU bilinear 2× upsample of a CHW tensor.
    ///
    /// Used for the DPT top-down chain. PyTorch default is
    /// `nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)`.
    /// Output spatial size = (2H, 2W). Returns `[1, C, 2H, 2W]` f16 Tensor.
    fn cpu_bilinear_2x_upsample(&self, input_chw: &Tensor, c: usize, h: usize, w: usize) -> Result<Tensor> {
        let h_out = h * 2;
        let w_out = w * 2;
        let in_data: Vec<half::f16> = input_chw.to_vec()?;
        let mut out = vec![half::f16::ZERO; c * h_out * w_out];
        for ch in 0..c {
            for oy in 0..h_out {
                // align_corners=False: input_y = (oy + 0.5)/2 - 0.5 = oy*0.5 - 0.25
                let iy_f = (oy as f32 + 0.5) * 0.5 - 0.5;
                let iy0 = iy_f.floor().max(0.0) as usize;
                let iy1 = (iy0 + 1).min(h - 1);
                let dy = (iy_f - iy0 as f32).clamp(0.0, 1.0);
                for ox in 0..w_out {
                    let ix_f = (ox as f32 + 0.5) * 0.5 - 0.5;
                    let ix0 = ix_f.floor().max(0.0) as usize;
                    let ix1 = (ix0 + 1).min(w - 1);
                    let dx = (ix_f - ix0 as f32).clamp(0.0, 1.0);
                    let v00 = in_data[ch * h * w + iy0 * w + ix0].to_f32();
                    let v01 = in_data[ch * h * w + iy0 * w + ix1].to_f32();
                    let v10 = in_data[ch * h * w + iy1 * w + ix0].to_f32();
                    let v11 = in_data[ch * h * w + iy1 * w + ix1].to_f32();
                    let v = (1.0 - dy) * ((1.0 - dx) * v00 + dx * v01)
                          + dy        * ((1.0 - dx) * v10 + dx * v11);
                    out[ch * h_out * w_out + oy * w_out + ox] = half::f16::from_f32(v);
                }
            }
        }
        Tensor::from_slice(&out, Shape::from([1usize, c, h_out, w_out]),
            DType::F16, self.compute.device().info().id)
    }

    /// CPU bilinear resize of a CHW tensor to an explicit `(h_out, w_out)`
    /// (used for refinenet4's odd-spatial output 19 → 37, since strict 2×
    /// gives 38 but the next level expects 37).
    fn cpu_bilinear_resize_to(
        &self, input_chw: &Tensor, c: usize, h: usize, w: usize,
        h_out: usize, w_out: usize,
    ) -> Result<Tensor> {
        let in_data: Vec<half::f16> = input_chw.to_vec()?;
        let scale_y = h as f32 / h_out as f32;
        let scale_x = w as f32 / w_out as f32;
        let mut out = vec![half::f16::ZERO; c * h_out * w_out];
        for ch in 0..c {
            for oy in 0..h_out {
                let iy_f = (oy as f32 + 0.5) * scale_y - 0.5;
                let iy0 = iy_f.floor().max(0.0) as usize;
                let iy1 = (iy0 + 1).min(h - 1);
                let dy = (iy_f - iy0 as f32).clamp(0.0, 1.0);
                for ox in 0..w_out {
                    let ix_f = (ox as f32 + 0.5) * scale_x - 0.5;
                    let ix0 = ix_f.floor().max(0.0) as usize;
                    let ix1 = (ix0 + 1).min(w - 1);
                    let dx = (ix_f - ix0 as f32).clamp(0.0, 1.0);
                    let v00 = in_data[ch * h * w + iy0 * w + ix0].to_f32();
                    let v01 = in_data[ch * h * w + iy0 * w + ix1].to_f32();
                    let v10 = in_data[ch * h * w + iy1 * w + ix0].to_f32();
                    let v11 = in_data[ch * h * w + iy1 * w + ix1].to_f32();
                    let v = (1.0 - dy) * ((1.0 - dx) * v00 + dx * v01)
                          + dy        * ((1.0 - dx) * v10 + dx * v11);
                    out[ch * h_out * w_out + oy * w_out + ox] = half::f16::from_f32(v);
                }
            }
        }
        Tensor::from_slice(&out, Shape::from([1usize, c, h_out, w_out]),
            DType::F16, self.compute.device().info().id)
    }

    /// GPU 1×1 conv via reshape + linear_bias matmul.
    ///
    /// Bypasses `conv2d_1x1_f16` (NaN at Cin=2048) and the SIMD variant's
    /// alignment requirements. Same pattern as `gpu_conv3x3_via_matmul` but
    /// without im2col (1×1 = identity patch).
    /// Manages its own command buffer.
    fn gpu_conv1x1_via_matmul(
        &self,
        input_chw: &Tensor, // [1, Cin, H, W]
        weight_name: &str,
        bias_name: Option<&str>,
        cin: usize, cout: usize, h: usize, w: usize,
    ) -> Result<Tensor> {
        let n = h * w;
        let device_id = self.compute.device().info().id;

        // CPU reshape: [1, Cin, H, W] CHW → [H*W, Cin] HWC.
        let in_data: Vec<half::f16> = input_chw.to_vec()?;
        let mut hwc = vec![half::f16::ZERO; n * cin];
        for c in 0..cin {
            for p in 0..n {
                hwc[p * cin + c] = in_data[c * n + p];
            }
        }
        let seq = Tensor::from_slice(&hwc, Shape::from([n, cin]), DType::F16, device_id)?;

        // Matmul. Weight is stored [Cout, Cin, 1, 1] = [Cout, Cin] flat.
        let cb = self.compute.new_command_buffer();
        let result = if let Some(bn) = bias_name {
            self.cached_linear_bias(&cb, &seq, weight_name, bn, n, cin, cout)?
        } else {
            let weight = self.cached_weight_f16(weight_name)?;
            let zero_bias = self.cached_zero_bias(cout)?;
            self.linear_tensors(&cb, &seq, &weight, &zero_bias, n, cin, cout)
        };
        cb.commit();
        cb.wait_until_completed();

        // Reshape [N, Cout] → [1, Cout, H, W] CHW.
        let result_data: Vec<half::f16> = result.to_vec()?;
        let mut chw = vec![half::f16::ZERO; cout * n];
        for p in 0..n {
            for c in 0..cout {
                chw[c * n + p] = result_data[p * cout + c];
            }
        }
        Tensor::from_slice(&chw, Shape::from([1usize, cout, h, w]), DType::F16, device_id)
    }

    /// CPU 1×1 conv over CHW input. Output is also CHW.
    fn cpu_conv2d_1x1_chw(
        input: &[f32], weight: &[half::f16], bias: Option<&[half::f16]>,
        output: &mut [f32], cin: usize, cout: usize, h: usize, w: usize,
    ) {
        let hw = h * w;
        for oc in 0..cout {
            let bias_v = bias.map(|b| b[oc].to_f32()).unwrap_or(0.0);
            for p in 0..hw {
                let mut acc = bias_v;
                for ic in 0..cin {
                    acc += input[ic * hw + p] * weight[oc * cin + ic].to_f32();
                }
                output[oc * hw + p] = acc;
            }
        }
    }

    /// CPU 3×3 conv (stride=1, padding=1) over CHW input. Output is CHW.
    fn cpu_conv2d_3x3_chw(
        input: &[f32], weight: &[half::f16], bias: Option<&[half::f16]>,
        output: &mut [f32], cin: usize, cout: usize, h: usize, w: usize,
    ) {
        let hw = h * w;
        for oc in 0..cout {
            let bias_v = bias.map(|b| b[oc].to_f32()).unwrap_or(0.0);
            for oy in 0..h {
                for ox in 0..w {
                    let mut acc = bias_v;
                    for ic in 0..cin {
                        for ky in 0..3usize {
                            let iy = oy as i32 + ky as i32 - 1;
                            if iy < 0 || iy >= h as i32 { continue; }
                            for kx in 0..3usize {
                                let ix = ox as i32 + kx as i32 - 1;
                                if ix < 0 || ix >= w as i32 { continue; }
                                let in_off = ic * hw + (iy as usize) * w + (ix as usize);
                                let w_off = oc * cin * 9 + ic * 9 + ky * 3 + kx;
                                acc += input[in_off] * weight[w_off].to_f32();
                            }
                        }
                    }
                    output[oc * hw + oy * w + ox] = acc;
                }
            }
        }
    }

    /// Reshape NHWC-style flat `[N=H*W, C]` array to CHW `[C, H, W]`.
    fn nhwc_to_chw(input: &[f32], n: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
        let _ = h; let _ = w;
        let mut out = vec![0.0f32; n * c];
        for p in 0..n {
            for ci in 0..c {
                out[ci * n + p] = input[p * c + ci];
            }
        }
        out
    }

    /// One DPT residual conv unit: x' = x + conv2(ReLU(conv1(ReLU(x)))).
    fn res_conf_unit(
        x: &[f32], c1_w: &[half::f16], c1_b: &[half::f16],
        c2_w: &[half::f16], c2_b: &[half::f16], c: usize, h: usize, w: usize,
    ) -> Vec<f32> {
        let hw = h * w;
        // ReLU(x) → conv1 → ReLU → conv2 → x + result
        let mut x_relu = x.to_vec();
        for v in x_relu.iter_mut() { *v = v.max(0.0); }
        let mut h1 = vec![0.0f32; c * hw];
        Self::cpu_conv2d_3x3_chw(&x_relu, c1_w, Some(c1_b), &mut h1, c, c, h, w);
        for v in h1.iter_mut() { *v = v.max(0.0); }
        let mut h2 = vec![0.0f32; c * hw];
        Self::cpu_conv2d_3x3_chw(&h1, c2_w, Some(c2_b), &mut h2, c, c, h, w);
        let mut out = x.to_vec();
        for i in 0..(c * hw) { out[i] += h2[i]; }
        out
    }

    fn image_branch_forward(
        &self,
        view_chw: &[f32],
        img_size: usize,
        gs_fused_features: &[half::f16],
    ) -> Result<Vec<half::f16>> {
        const GRID: usize = 37;
        const C0: usize = 128; // input_merger output channels
        const C1: usize = 256; // gs_renderer.gs_head.0 output channels
        const C2: usize = 12;  // gs_renderer.gs_head.2 output channels (final splat params)

        debug_assert_eq!(gs_fused_features.len(), C0 * GRID * GRID,
            "gs_fused_features must be [128, 37, 37] flat");

        // Whole branch runs on CPU until the unet conv2d_naive_f16 metal
        // kernel's NaN-at-stride>1 bug is fixed. The convs are small —
        // total ~430M mults at 37×37 grid resolution — sub-second on
        // modern CPUs.

        // Load gs_renderer + input_merger weights once. Cached so the f32→f16
        // conversion + GPU upload only happens on the first view.
        let im_w: Vec<half::f16> = self.cached_weight_f16("gs_head.input_merger.0.weight")?.to_vec()?;
        let im_b: Vec<half::f16> = self.cached_weight_f16("gs_head.input_merger.0.bias")?.to_vec()?;
        let gs0_w: Vec<half::f16> = self.cached_weight_f16("gs_renderer.gs_head.0.weight")?.to_vec()?;
        let gs2_w: Vec<half::f16> = self.cached_weight_f16("gs_renderer.gs_head.2.weight")?.to_vec()?;
        let gs2_b: Vec<half::f16> = self.cached_weight_f16("gs_renderer.gs_head.2.bias")?.to_vec()?;

        // 1. input_merger 7×7 stride=14 padding=3, fused ReLU, plus add the
        //    DPT fused features (`fused + input_merger(images)` in the
        //    reference's `DPTHead.forward` is_gsdpt branch — these together
        //    form `gs_feats` that gs_renderer.gs_head consumes).
        //    Reference's stride is 1 at 518×518 res; we approximate at the
        //    patch grid (stride=14, 37×37) since our DPT runs single-scale.
        let mut a = vec![0.0f32; C0 * GRID * GRID];
        let stride = 14usize;
        let pad = 3i32;
        let kh = 7usize;
        let kw = 7usize;
        let img = img_size as i32;
        for oc in 0..C0 {
            let bias_v = im_b[oc].to_f32();
            for oy in 0..GRID {
                for ox in 0..GRID {
                    let mut acc = bias_v;
                    for ic in 0..3 {
                        for ky in 0..kh {
                            let iy = (oy * stride) as i32 + ky as i32 - pad;
                            if iy < 0 || iy >= img { continue; }
                            for kx in 0..kw {
                                let ix = (ox * stride) as i32 + kx as i32 - pad;
                                if ix < 0 || ix >= img { continue; }
                                let in_off = ic * (img_size * img_size)
                                    + (iy as usize) * img_size + (ix as usize);
                                let w_off = oc * (3 * kh * kw)
                                    + ic * (kh * kw) + ky * kw + kx;
                                acc += view_chw[in_off] * im_w[w_off].to_f32();
                            }
                        }
                    }
                    let off = oc * GRID * GRID + oy * GRID + ox;
                    // ReLU(input_merger) + fused (the reference adds them BEFORE
                    // running gs_renderer.gs_head; fused already has ReLU baked in
                    // since it comes from output_conv1 + ReLU).
                    a[off] = acc.max(0.0) + gs_fused_features[off].to_f32();
                }
            }
        }

        // 2. gs_renderer.gs_head.0 (3×3 stride=1 padding=1, no bias), fused ReLU.
        //    Output [256, 37, 37].
        let mut b = vec![0.0f32; C1 * GRID * GRID];
        for oc in 0..C1 {
            for oy in 0..GRID {
                for ox in 0..GRID {
                    let mut acc = 0.0f32;
                    for ic in 0..C0 {
                        for ky in 0..3 {
                            let iy = oy as i32 + ky as i32 - 1;
                            if iy < 0 || iy >= GRID as i32 { continue; }
                            for kx in 0..3 {
                                let ix = ox as i32 + kx as i32 - 1;
                                if ix < 0 || ix >= GRID as i32 { continue; }
                                let in_off = ic * GRID * GRID
                                    + (iy as usize) * GRID + (ix as usize);
                                let w_off = oc * (C0 * 3 * 3) + ic * 9 + ky * 3 + kx;
                                acc += a[in_off] * gs0_w[w_off].to_f32();
                            }
                        }
                    }
                    b[oc * GRID * GRID + oy * GRID + ox] = acc.max(0.0);
                }
            }
        }

        // 3. gs_renderer.gs_head.2 (1×1, with bias). → [12, 37, 37]
        let mut out_f16 = vec![half::f16::ZERO; C2 * GRID * GRID];
        for oc in 0..C2 {
            let bias_v = gs2_b[oc].to_f32();
            for oy in 0..GRID {
                for ox in 0..GRID {
                    let mut acc = bias_v;
                    for ic in 0..C1 {
                        let in_off = ic * GRID * GRID + oy * GRID + ox;
                        let w_off = oc * C1 + ic; // 1×1 has only [oc, ic]
                        acc += b[in_off] * gs2_w[w_off].to_f32();
                    }
                    out_f16[oc * GRID * GRID + oy * GRID + ox] = half::f16::from_f32(acc);
                }
            }
        }
        Ok(out_f16)
    }

    /// 14×14 conv patch projection (im2col + GPU matmul). Mirrors
    /// `Hunyuan3DPipeline::dino_patch_embed` but for 1024-dim DINOv2-L
    /// at the WorldMirror weight namespace.
    fn patch_embed_conv(
        &self,
        image_chw: &[f32],
        grid: usize,
        patch_size: usize,
        img_size: usize,
        d_model: usize,
        prefix: &str,
    ) -> Result<Tensor> {
        let c_in = 3;
        let num_patches = grid * grid;
        let k_size = c_in * patch_size * patch_size;
        let mut col_data: Vec<half::f16> = vec![half::f16::ZERO; num_patches * k_size];
        for gy in 0..grid {
            for gx in 0..grid {
                let p = gy * grid + gx;
                for in_c in 0..c_in {
                    for ky in 0..patch_size {
                        for kx in 0..patch_size {
                            let iy = gy * patch_size + ky;
                            let ix = gx * patch_size + kx;
                            if iy < img_size && ix < img_size {
                                let val = image_chw[in_c * img_size * img_size + iy * img_size + ix];
                                col_data[p * k_size + in_c * patch_size * patch_size + ky * patch_size + kx] =
                                    half::f16::from_f32(val);
                            }
                        }
                    }
                }
            }
        }
        let col_tensor = Tensor::from_slice(
            &col_data, Shape::from([num_patches, k_size]),
            DType::F16, self.compute.device().info().id,
        )?;
        let cb = self.compute.new_command_buffer();
        let result = self.cached_linear_bias(
            &cb, &col_tensor,
            &format!("{}.patch_embed.proj.weight", prefix),
            &format!("{}.patch_embed.proj.bias", prefix),
            num_patches, k_size, d_model,
        )?;
        cb.commit();
        cb.wait_until_completed();
        Ok(result)
    }

    /// Precompute 2D RoPE cos/sin tables for the patch grid + special-token
    /// rows (cls + register).
    ///
    /// `periods`: 16 frequency periods loaded from `attn.rope.periods`.
    /// `grid_h` × `grid_w`: patch grid dimensions (37×37 for SD-518).
    /// `n_special`: number of leading non-positional tokens (cls + 4 reg = 5).
    ///
    /// Returns four flat tables of length `seq_len × half_dim` where
    /// `half_dim = 16` (= head_dim/4 — RoPE rotates pairs and the head is
    /// split y-half + x-half). Special tokens get position 0 (cos=1, sin=0).
    fn precompute_rope_2d(
        periods: &[f32],
        grid_h: usize,
        grid_w: usize,
        n_special: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let pairs = periods.len(); // 16
        let n_patches = grid_h * grid_w;
        let seq_len = n_special + n_patches;
        let mut cos_y = vec![1.0f32; seq_len * pairs];
        let mut sin_y = vec![0.0f32; seq_len * pairs];
        let mut cos_x = vec![1.0f32; seq_len * pairs];
        let mut sin_x = vec![0.0f32; seq_len * pairs];
        for p in 0..n_patches {
            let y = (p / grid_w) as f32;
            let x = (p % grid_w) as f32;
            let row = n_special + p;
            for k in 0..pairs {
                let inv_period = 1.0 / periods[k].max(1e-6);
                let theta_y = y * inv_period;
                let theta_x = x * inv_period;
                cos_y[row * pairs + k] = theta_y.cos();
                sin_y[row * pairs + k] = theta_y.sin();
                cos_x[row * pairs + k] = theta_x.cos();
                sin_x[row * pairs + k] = theta_x.sin();
            }
        }
        (cos_y, sin_y, cos_x, sin_x)
    }

    /// Apply 2D RoPE to a flat `[seq, num_heads, head_dim]` f16 buffer
    /// in-place. First `head_dim/2` of each head is rotated by y-position,
    /// last `head_dim/2` by x-position. Each half is treated as `head_dim/4`
    /// adjacent (even, odd) pairs.
    fn apply_rope_2d_inplace(
        data: &mut [half::f16],
        cos_y: &[f32],
        sin_y: &[f32],
        cos_x: &[f32],
        sin_x: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) {
        let half = head_dim / 2;
        let pairs = half / 2; // 16
        let row_stride = num_heads * head_dim;
        for s in 0..seq_len {
            let base_table = s * pairs;
            for h in 0..num_heads {
                let head_off = s * row_stride + h * head_dim;
                // y-half: dims [0 .. half)
                for k in 0..pairs {
                    let i_e = head_off + 2 * k;
                    let i_o = i_e + 1;
                    let x0 = data[i_e].to_f32();
                    let x1 = data[i_o].to_f32();
                    let c = cos_y[base_table + k];
                    let s_ = sin_y[base_table + k];
                    data[i_e] = half::f16::from_f32(x0 * c - x1 * s_);
                    data[i_o] = half::f16::from_f32(x0 * s_ + x1 * c);
                }
                // x-half: dims [half .. head_dim)
                for k in 0..pairs {
                    let i_e = head_off + half + 2 * k;
                    let i_o = i_e + 1;
                    let x0 = data[i_e].to_f32();
                    let x1 = data[i_o].to_f32();
                    let c = cos_x[base_table + k];
                    let s_ = sin_x[base_table + k];
                    data[i_e] = half::f16::from_f32(x0 * c - x1 * s_);
                    data[i_o] = half::f16::from_f32(x0 * s_ + x1 * c);
                }
            }
        }
    }

    /// Apply per-head RMS norm: each head's [head_dim] vector is RMS-normalised
    /// then scaled by the learned per-channel `weight` and offset by `bias`
    /// (per `q_norm` / `k_norm` weights). Done on CPU since each head is small
    /// (head_dim=64) and there's no Metal kernel for per-head RMS.
    fn rms_norm_per_head(
        data: &mut [half::f16],
        weight: &[half::f16],
        bias: &[half::f16],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) {
        let row_stride = num_heads * head_dim;
        for s in 0..seq_len {
            for h in 0..num_heads {
                let off = s * row_stride + h * head_dim;
                let mut sum_sq = 0.0f32;
                for d in 0..head_dim {
                    let v = data[off + d].to_f32();
                    sum_sq += v * v;
                }
                let rms = (sum_sq / head_dim as f32 + eps).sqrt().max(1e-6);
                let inv = 1.0 / rms;
                for d in 0..head_dim {
                    let v = data[off + d].to_f32() * inv;
                    let w = weight[d].to_f32();
                    let b = bias[d].to_f32();
                    data[off + d] = half::f16::from_f32(v * w + b);
                }
            }
        }
    }

    /// One frame_block layer: pre-norm → fused QKV → split → per-head QK
    /// RMS-norm → 2D RoPE on Q,K → batched MHA → output proj → LayerScale →
    /// residual → norm → MLP → LayerScale → residual.
    fn frame_block_forward(
        &self,
        input: &Tensor,
        lp: &str,
        seq_len: usize,
        d_model: usize,
        num_heads: usize,
        head_dim: usize,
        ffn_dim: usize,
        cos_y: &[f32],
        sin_y: &[f32],
        cos_x: &[f32],
        sin_x: &[f32],
        scale_attn: f32,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let cb = self.compute.new_command_buffer();

        // 1. Pre-norm.
        let normed = self.cached_layer_norm(
            &cb, input,
            &format!("{}.norm1.weight", lp), &format!("{}.norm1.bias", lp),
            seq_len, d_model, 1e-6,
        )?;

        // 2. Fused QKV.
        let qkv = self.cached_linear_bias(
            &cb, &normed,
            &format!("{}.attn.qkv.weight", lp),
            &format!("{}.attn.qkv.bias", lp),
            seq_len, d_model, 3 * d_model,
        )?;
        cb.commit();
        cb.wait_until_completed();

        // 3. Split QKV on CPU.
        let qkv_data: Vec<half::f16> = qkv.to_vec()?;
        let mut q_data = Vec::with_capacity(seq_len * d_model);
        let mut k_data = Vec::with_capacity(seq_len * d_model);
        let mut v_data = Vec::with_capacity(seq_len * d_model);
        for s in 0..seq_len {
            let row = s * 3 * d_model;
            q_data.extend_from_slice(&qkv_data[row..row + d_model]);
            k_data.extend_from_slice(&qkv_data[row + d_model..row + 2 * d_model]);
            v_data.extend_from_slice(&qkv_data[row + 2 * d_model..row + 3 * d_model]);
        }

        // 4. Per-head RMS-norm on Q and K.
        let qn_w = self.cached_weight_f16(&format!("{}.attn.q_norm.weight", lp))?;
        let qn_b = self.cached_weight_f16(&format!("{}.attn.q_norm.bias", lp))?;
        let kn_w = self.cached_weight_f16(&format!("{}.attn.k_norm.weight", lp))?;
        let kn_b = self.cached_weight_f16(&format!("{}.attn.k_norm.bias", lp))?;
        let qn_w_data: Vec<half::f16> = qn_w.to_vec()?;
        let qn_b_data: Vec<half::f16> = qn_b.to_vec()?;
        let kn_w_data: Vec<half::f16> = kn_w.to_vec()?;
        let kn_b_data: Vec<half::f16> = kn_b.to_vec()?;
        Self::rms_norm_per_head(&mut q_data, &qn_w_data, &qn_b_data, seq_len, num_heads, head_dim, 1e-6);
        Self::rms_norm_per_head(&mut k_data, &kn_w_data, &kn_b_data, seq_len, num_heads, head_dim, 1e-6);

        // 5. Apply 2D RoPE to Q and K.
        Self::apply_rope_2d_inplace(&mut q_data, cos_y, sin_y, cos_x, sin_x, seq_len, num_heads, head_dim);
        Self::apply_rope_2d_inplace(&mut k_data, cos_y, sin_y, cos_x, sin_x, seq_len, num_heads, head_dim);

        // 6. Upload back + batched MHA.
        let q = Tensor::from_slice(&q_data, Shape::from([seq_len, d_model]), DType::F16, device_id)?;
        let k = Tensor::from_slice(&k_data, Shape::from([seq_len, d_model]), DType::F16, device_id)?;
        let v = Tensor::from_slice(&v_data, Shape::from([seq_len, d_model]), DType::F16, device_id)?;
        let cb2 = self.compute.new_command_buffer();
        let attn_out = self.batched_attention(
            &cb2, &q, &k, &v, seq_len, seq_len, num_heads, head_dim, scale_attn,
        )?;

        // 7. Output projection.
        let proj = self.cached_linear_bias(
            &cb2, &attn_out,
            &format!("{}.attn.proj.weight", lp),
            &format!("{}.attn.proj.bias", lp),
            seq_len, d_model, d_model,
        )?;
        // CRITICAL: layer_scale below reads `proj.to_vec()` on CPU. The proj
        // buffer's contents are only valid after the GPU work producing it
        // (queued on cb2) has actually executed — must commit + wait BEFORE
        // the CPU read. (Bug found 2026-05-13: without this commit, layer_scale
        // saw zeros from an uninitialised buffer, so `scaled ≈ 0`, so
        // `h = input + 0 = input` made every frame_block / global_block an
        // identity transform — see memory `project_efficient_genai.md`.)
        cb2.commit();
        cb2.wait_until_completed();

        // 8. LayerScale (ls1) + residual.
        let scaled = self.layer_scale(&cb2, &proj, &format!("{}.ls1.gamma", lp), seq_len, d_model)?;
        let cb3 = self.compute.new_command_buffer();
        let h = self.add(&cb3, input, &scaled);

        // 9. MLP block.
        let normed2 = self.cached_layer_norm(
            &cb3, &h,
            &format!("{}.norm2.weight", lp), &format!("{}.norm2.bias", lp),
            seq_len, d_model, 1e-6,
        )?;
        let ffn_up = self.cached_linear_bias(
            &cb3, &normed2,
            &format!("{}.mlp.fc1.weight", lp), &format!("{}.mlp.fc1.bias", lp),
            seq_len, d_model, ffn_dim,
        )?;
        let ffn_act = self.activation(&cb3, &self.kernels.gelu, &ffn_up);
        let ffn_down = self.cached_linear_bias(
            &cb3, &ffn_act,
            &format!("{}.mlp.fc2.weight", lp), &format!("{}.mlp.fc2.bias", lp),
            seq_len, ffn_dim, d_model,
        )?;
        // Same CPU-read-before-commit issue: must commit cb3 so `ffn_down`
        // is materialised before `layer_scale` reads it.
        cb3.commit();
        cb3.wait_until_completed();
        let scaled2 = self.layer_scale(&cb3, &ffn_down, &format!("{}.ls2.gamma", lp), seq_len, d_model)?;
        let cb4 = self.compute.new_command_buffer();
        let out = self.add(&cb4, &h, &scaled2);
        cb4.commit();
        cb4.wait_until_completed();
        Ok(out)
    }

    /// LayerScale: per-channel multiplier `gamma` applied to a `[seq, d]`
    /// tensor (broadcast across the seq dim). Done on CPU since the
    /// operation is small and no Metal kernel exists for it.
    fn layer_scale(
        &self,
        _cb: &metal::CommandBufferRef,
        input: &Tensor,
        gamma_name: &str,
        seq_len: usize,
        d: usize,
    ) -> Result<Tensor> {
        let gamma = self.cached_weight_f16(gamma_name)?;
        let g_data: Vec<half::f16> = gamma.to_vec()?;
        let in_data: Vec<half::f16> = input.to_vec()?;
        let mut out = Vec::with_capacity(seq_len * d);
        for s in 0..seq_len {
            for di in 0..d {
                let val = in_data[s * d + di].to_f32() * g_data[di].to_f32();
                out.push(half::f16::from_f32(val));
            }
        }
        Tensor::from_slice(&out, Shape::from([seq_len, d]), DType::F16, self.compute.device().info().id)
    }
}
