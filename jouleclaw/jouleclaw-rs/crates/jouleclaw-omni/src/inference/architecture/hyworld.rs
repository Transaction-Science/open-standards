//! HY-World 2.0 — text/image → navigable 3D world (panoramic 3DGS).
//!
//! Tencent's open-source 3D world model (released April 15, 2026, weights
//! at `tencent/HY-World-2.0` on Hugging Face). Architecture mirrors the
//! official 4-stage pipeline:
//!
//! ```text
//!   text or image                     status (May 2026)
//!         │
//!         ▼
//!   1) HY-Pano 2.0      panoramic gen (DiT)            ─ weights pending upstream
//!         │
//!         ▼
//!   2) WorldNav         trajectory planning            ─ code+weights pending upstream
//!         │
//!         ▼
//!   3) WorldStereo 2.0  depth + multi-view stereo      ─ weights pending upstream
//!         │
//!         ▼
//!   4) WorldMirror 2.0  unified feed-forward recon     ─ released, ~5 GB safetensors
//! ```
//!
//! **What's deployable today** (May 2026): WorldMirror 2.0 alone supports
//! the *world reconstruction* path — multi-view images / video → 3DGS
//! (depth + normals + camera + points + Gaussian parameters predicted in
//! one forward). The full *world generation* path (text/single-image →
//! 3D world) waits on Tencent's release of HY-Pano 2.0 + WorldStereo 2.0
//! + WorldNav, scheduled "Coming Soon" per the upstream README.
//!
//! Until those land, [`HYWorldPipeline::generate_from_text`] and
//! [`HYWorldPipeline::generate_from_image`] emit a synthetic spiral
//! trajectory + sparse splat cloud so the studio shell can render
//! *something* at the surface — but the production-quality path runs
//! through [`HYWorldPipeline::reconstruct_from_views`] (multi-view in,
//! 3DGS out) using only the released WorldMirror 2.0 weights.
//!
//! Output: a navigable Gaussian-splat scene plus an optional triangle
//! mesh fallback (via marching cubes on the densified depth).
//!
//! Reference: <https://github.com/Tencent-Hunyuan/HY-World-2.0>
//!            <https://huggingface.co/tencent/HY-World-2.0>
//!            <https://3d-models.hunyuan.tencent.com/world/world2_0/HY_World_2_0.pdf>

use crate::core::Result;
use crate::tensor::{DType, Shape, Tensor};

#[cfg(feature = "metal")]
use crate::hal::metal::{ComputePipeline, MetalCompute, MetalDevice};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;
#[cfg(feature = "metal")]
use crate::inference::gpu_ops::{self, MetalPipeline};
#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use std::sync::Arc;

// ───── Config ─────

/// HY-World 2.0 configuration. Defaults match the official `tencent/HY-World-2.0`
/// release card (April 2026).
#[derive(Debug, Clone)]
pub struct HYWorldConfig {
    /// Equirectangular panorama height in pixels.
    pub pano_height: usize,
    /// Equirectangular panorama width in pixels (= 2 × height for full sphere).
    pub pano_width: usize,
    /// Number of camera trajectory steps planned by WorldNav.
    pub trajectory_steps: usize,
    /// Multi-view stereo angular spacing for WorldStereo 2.0 (degrees).
    pub stereo_step_degrees: f32,
    /// Target Gaussian splat count for the reconstructed scene.
    pub splat_budget: usize,
    /// HY-Pano 2.0 DiT hidden dimension.
    pub pano_hidden: usize,
    /// HY-Pano 2.0 DiT block count (dual-stream + single-stream combined).
    pub pano_blocks: usize,
    /// WorldNav transformer hidden dimension.
    pub nav_hidden: usize,
    /// WorldNav transformer layer count.
    pub nav_layers: usize,
    /// WorldStereo 2.0 depth-decoder feature dim.
    pub stereo_feature_dim: usize,
    /// WorldMirror 2.0 reconstruction grid resolution.
    pub mirror_grid_resolution: usize,
    /// Render quality preset.
    pub quality: WorldQuality,
}

/// Render quality presets — control trajectory density, splat budget,
/// and stereo step size.
#[derive(Debug, Clone, Copy)]
pub enum WorldQuality {
    /// Fast preview — single trajectory, lower splat count, larger stereo steps.
    Draft,
    /// Default — full trajectory, target splat budget.
    Standard,
    /// Cinematic — denser trajectory, increased splat budget, finer stereo.
    Cinematic,
}

impl Default for WorldQuality {
    fn default() -> Self {
        Self::Standard
    }
}

impl HYWorldConfig {
    /// Default HY-World 2.0 release configuration.
    pub fn default_release() -> Self {
        Self {
            pano_height: 1024,
            pano_width: 2048,
            trajectory_steps: 16,
            stereo_step_degrees: 22.5,
            splat_budget: 600_000,
            pano_hidden: 1536,
            pano_blocks: 24,
            nav_hidden: 1024,
            nav_layers: 6,
            stereo_feature_dim: 384,
            mirror_grid_resolution: 256,
            quality: WorldQuality::Standard,
        }
    }

    /// Adjust the config for a quality preset. Returns self for chaining.
    pub fn with_quality(mut self, q: WorldQuality) -> Self {
        self.quality = q;
        match q {
            WorldQuality::Draft => {
                self.trajectory_steps = 6;
                self.splat_budget = 200_000;
                self.stereo_step_degrees = 45.0;
            }
            WorldQuality::Standard => {
                self.trajectory_steps = 16;
                self.splat_budget = 600_000;
                self.stereo_step_degrees = 22.5;
            }
            WorldQuality::Cinematic => {
                self.trajectory_steps = 32;
                self.splat_budget = 1_500_000;
                self.stereo_step_degrees = 11.25;
            }
        }
        self
    }
}

// ───── Output types ─────

/// Single 3D Gaussian splat. Layout matches the existing
/// `sources::GAUSSIAN_SPLAT` kernel's `struct Gaussian`.
#[derive(Debug, Clone, Copy)]
pub struct GaussianSplat {
    /// World-space centre.
    pub position: [f32; 3],
    /// Per-axis scale (anisotropic Gaussian extent).
    pub scale: [f32; 3],
    /// Rotation as unit quaternion (x, y, z, w).
    pub rotation: [f32; 4],
    /// RGB color.
    pub color: [f32; 3],
    /// Opacity (0..1).
    pub opacity: f32,
}

/// Optional pow3r conditioning hints (depth/pose/ray) supplied by the caller.
/// Each field is `None` if no conditioning is provided for that channel; in
/// that case the corresponding embedder is skipped. When `Some`, the inner
/// `Vec` must have one entry per view.
#[derive(Debug, Default, Clone)]
pub struct Pow3rHints {
    /// Per-view depth values: `[n_views][NUM_PATCHES × 196]` f32. Each view's
    /// buffer is the patch-wise 14×14 depth values, flattened row-major.
    pub depth_per_view: Option<Vec<Vec<f32>>>,
    /// Per-view camera pose `[tx, ty, tz, qw, qx, qy, qz]` (7-dim).
    pub pose_per_view: Option<Vec<[f32; 7]>>,
    /// Per-view ray direction (4-dim).
    pub ray_per_view: Option<Vec<[f32; 4]>>,
}

/// Camera pose along the planned trajectory.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryPose {
    /// World-space camera position.
    pub position: [f32; 3],
    /// Look direction (unit vector).
    pub forward: [f32; 3],
    /// Up direction.
    pub up: [f32; 3],
    /// Field of view (degrees).
    pub fov_deg: f32,
}

/// Final HY-World 2.0 output bundle.
#[derive(Debug, Clone)]
pub struct WorldOutput {
    /// 3D Gaussian splats forming the scene.
    pub splats: Vec<GaussianSplat>,
    /// Planned camera trajectory.
    pub trajectory: Vec<TrajectoryPose>,
    /// Equirectangular panorama (HxW RGB f16 flattened) — optional preview.
    pub panorama_preview: Option<Vec<f32>>,
    /// Optional triangle-mesh fallback (Wavefront OBJ text). Generated when
    /// the caller opts into mesh-mode rather than splats.
    pub mesh_obj: Option<String>,
    /// Optional depth map per view, in raw f32 values (not normalised).
    /// Layout per view: `[depth_grid_h * depth_grid_w]` flat. Computed by
    /// the WorldMirror DPT decoder via `depth_head` when the caller opts
    /// in (`compute_depth=true` on the API request).
    pub depth_maps: Option<Vec<Vec<f32>>>,
    /// Optional surface normals per view, in raw f32 values. Layout per
    /// view: `[3, depth_grid_h, depth_grid_w]` CHW flat (xyz unit vectors).
    /// Caller-opt-in via `compute_normals=true`.
    pub normal_maps: Option<Vec<Vec<f32>>>,
    /// Optional 3D point clouds per view, in raw f32 xyz values. Layout
    /// per view: `[3, points_grid_h, points_grid_w]` CHW flat. Opt-in via
    /// `compute_points=true`.
    pub point_clouds: Option<Vec<Vec<f32>>>,
    /// Number of inference steps actually executed.
    pub steps_executed: u32,
}

// ───── Pipeline ─────

/// HY-World 2.0 four-stage pipeline.
///
/// Each sub-model is `Option<Arc<Model>>` because the production weights
/// (`tencent/HY-World-2.0`, ~6 GB total) are loaded lazily and not all
/// stages are required for a draft pass. When a stage's weights are
/// `None` the stage emits a shape-correct stub so the rest of the
/// pipeline still runs end-to-end.
#[cfg(feature = "metal")]
pub struct HYWorldPipeline {
    pano_model: Option<Arc<Model>>,
    nav_model: Option<Arc<Model>>,
    stereo_model: Option<Arc<Model>>,
    mirror_model: Option<Arc<Model>>,
    /// Lazily-built WorldMirror runtime — once `reconstruct_from_views`
    /// is called and `mirror_model` is loaded, build the patch-embed
    /// encoder runtime and reuse it for subsequent calls.
    mirror_runtime: std::sync::OnceLock<crate::inference::architecture::worldmirror_forward::WorldMirrorRuntime>,
    /// Device handle (kept around for lazy WorldMirror build).
    device: Arc<MetalDevice>,
    compute: Arc<MetalCompute>,
    config: HYWorldConfig,
    kernels: HYWorldKernels,
}

#[cfg(feature = "metal")]
struct HYWorldKernels {
    common: gpu_ops::CommonKernels,
    gaussian_splat: Arc<ComputePipeline>,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for HYWorldPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl HYWorldPipeline {
    /// Construct a pipeline from any subset of the four sub-models. Pass
    /// `None` for stages whose weights aren't deployed yet; those stages
    /// emit shape-correct stub output.
    pub fn new(
        pano_model: Option<Arc<Model>>,
        nav_model: Option<Arc<Model>>,
        stereo_model: Option<Arc<Model>>,
        mirror_model: Option<Arc<Model>>,
        config: HYWorldConfig,
        device: Arc<MetalDevice>,
    ) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device.clone()));
        let kernels = HYWorldKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gaussian_splat: compute.compile_pipeline(
                "gaussian_splat",
                sources::GAUSSIAN_SPLAT,
                "splat_gaussians",
            )?,
        };
        Ok(Self {
            pano_model,
            nav_model,
            stereo_model,
            mirror_model,
            mirror_runtime: std::sync::OnceLock::new(),
            device,
            compute,
            config,
            kernels,
        })
    }

    /// Reconstruct a 3D world directly from a stack of multi-view images.
    /// This is the **production-quality path** today — uses only the
    /// released WorldMirror 2.0 weights, no panorama / trajectory /
    /// stereo stages needed.
    ///
    /// `views` is `[N, 3, H, W]` f16 RGB, normalised to ImageNet stats.
    /// `H` and `W` should be multiples of 14 (DINOv2 patch size); 518×518
    /// is the canonical input size matching the WorldMirror 2.0 config.
    /// Returns Gaussian splats predicted feed-forward by WorldMirror 2.0
    /// from all views jointly.
    pub fn reconstruct_from_views(&self, views: &Tensor, seed: u64) -> Result<WorldOutput> {
        self.reconstruct_from_views_full(views, seed, false, false, false)
    }

    /// Pre-warm the WorldMirror runtime by running a synthetic 1-view forward
    /// pass. Populates the weight cache and pages in the safetensors mmap so
    /// the first user request gets the warm-path latency (~13s on M-series)
    /// instead of the cold-path 762s+ demand-paging penalty.
    ///
    /// Safe to call before any user requests. No-op when `mirror_model` is
    /// unset.
    pub fn prewarm(&self) -> Result<()> {
        if self.mirror_model.is_none() {
            return Ok(());
        }
        // Synthetic 1-view [1, 3, 518, 518] f16 zero tensor.
        let n = 3usize * 518 * 518;
        let zeros = vec![half::f16::ZERO; n];
        let device_id = self.device.info().id;
        let dummy = Tensor::from_slice(
            &zeros, Shape::from([1usize, 3, 518, 518]), DType::F16, device_id,
        )?;
        let _ = self.reconstruct_from_views_full(&dummy, 0, false, false, false)?;
        Ok(())
    }

    /// Like `reconstruct_from_views` but with optional secondary head
    /// outputs. Each `compute_*` flag adds ~30s CPU compute per view.
    pub fn reconstruct_from_views_full(
        &self,
        views: &Tensor,
        seed: u64,
        compute_depth: bool,
        compute_normals: bool,
        compute_points: bool,
    ) -> Result<WorldOutput> {
        self.reconstruct_from_views_with_hints(
            views, seed, compute_depth, compute_normals, compute_points,
            &Pow3rHints::default(),
        )
    }

    /// `reconstruct_from_views_full` with optional pow3r conditioning hints
    /// (depth/pose/ray priors per view). Empty hints == identical behaviour
    /// to `reconstruct_from_views_full`.
    pub fn reconstruct_from_views_with_hints(
        &self,
        views: &Tensor,
        seed: u64,
        compute_depth: bool,
        compute_normals: bool,
        compute_points: bool,
        hints: &Pow3rHints,
    ) -> Result<WorldOutput> {
        let n_views = views.shape().dims().first().copied().unwrap_or(1);

        // Prefer the runtime path whenever the WorldMirror model is loaded —
        // this runs cam_head unconditionally so trajectory reflects real
        // model output even when no secondary heads are requested. Falls back
        // to `worldmirror_reconstruct` (synthetic trajectory) only when the
        // model isn't loaded.
        if let Some(model) = &self.mirror_model {
            let runtime = self.mirror_runtime.get_or_init(|| {
                crate::inference::architecture::worldmirror_forward::WorldMirrorRuntime::new(
                    model.clone(), self.device.clone(),
                ).expect("WorldMirror runtime build failed")
            });
            let f16: Vec<half::f16> = views.to_vec()?;
            let chw_f32: Vec<f32> = f16.iter().map(|v| v.to_f32()).collect();
            let outputs = runtime.forward_full_with_hints(
                &chw_f32, n_views, seed,
                compute_depth, compute_normals, compute_points,
                hints,
            )?;
            let trajectory = Self::trajectory_from_cam_params(
                outputs.camera_params.as_ref(), n_views, seed,
            )?;
            return Ok(WorldOutput {
                splats: outputs.splats,
                trajectory,
                panorama_preview: None,
                mesh_obj: None,
                depth_maps: outputs.depth_maps,
                normal_maps: outputs.normal_maps,
                point_clouds: outputs.point_clouds,
                steps_executed: 1,
            });
        }

        // Model not loaded — pure-stub fallback path.
        let splats = self.worldmirror_reconstruct(views, seed)?;
        let trajectory = self.synthetic_trajectory(n_views, seed)?;
        Ok(WorldOutput {
            splats,
            trajectory,
            panorama_preview: None,
            mesh_obj: None,
            depth_maps: None,
            normal_maps: None,
            point_clouds: None,
            steps_executed: 1,
        })
    }

    /// Build per-view `TrajectoryPose`s from `cam_head` 9-dim outputs.
    ///
    /// Layout (matches the Tencent reference `camera_utils.vector_to_camera_matrices`):
    ///   `[0:3]` = translation `t` of the **camera-to-world** extrinsic `[R | t]`
    ///   `[3:7]` = quaternion `q` (wxyz)
    ///   `[7]`   = `fov_v` — vertical field of view, in **radians** (relu'd)
    ///   `[8]`   = `fov_u` — horizontal field of view, in radians (relu'd)
    ///
    /// Since the extrinsic is camera-to-world: the world-space camera position
    /// is `t` directly, the look direction is the camera's local +Z axis in
    /// world frame = `R[:, 2]` (3rd column of R), and "up" is the camera's
    /// local -Y axis = `-R[:, 1]` (OpenCV/COLMAP convention: camera Y points
    /// down). Falls back to a synthetic spiral if `cam_params` is `None` or
    /// length-mismatched.
    fn trajectory_from_cam_params(
        cam_params: Option<&Vec<[f32; 9]>>,
        n_views: usize,
        seed: u64,
    ) -> Result<Vec<TrajectoryPose>> {
        match cam_params {
            Some(params) if params.len() == n_views => {
                let mut traj = Vec::with_capacity(n_views);
                for c in params {
                    let position = [c[0], c[1], c[2]];
                    // Normalise the quaternion. Reference `rotation.py`
                    // `quat_to_rotmat` unpacks `i, j, k, r = unbind(q)` =
                    // (qx, qy, qz, qw) — SCALAR LAST. So `c[3..7]` is
                    // (qx, qy, qz, qw), not (qw, qx, qy, qz).
                    let (mut qx, mut qy, mut qz, mut qw) = (c[3], c[4], c[5], c[6]);
                    let qn = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt();
                    if qn > 1e-8 { qw /= qn; qx /= qn; qy /= qn; qz /= qn; }
                    else { qw = 1.0; qx = 0.0; qy = 0.0; qz = 0.0; }
                    // Rotation matrix columns from the unit quaternion.
                    // R = [[1-2(y²+z²), 2(xy-wz),   2(xz+wy)  ],
                    //      [2(xy+wz),   1-2(x²+z²), 2(yz-wx)  ],
                    //      [2(xz-wy),   2(yz+wx),   1-2(x²+y²)]]
                    // forward = R[:, 2] (camera local +Z in world);
                    // up      = -R[:, 1] (camera local -Y; OpenCV Y is down).
                    let forward = [
                        2.0 * (qx * qz + qw * qy),
                        2.0 * (qy * qz - qw * qx),
                        1.0 - 2.0 * (qx * qx + qy * qy),
                    ];
                    let up = [
                        -(2.0 * (qx * qy - qw * qz)),
                        -(1.0 - 2.0 * (qx * qx + qz * qz)),
                        -(2.0 * (qy * qz + qw * qx)),
                    ];
                    // c[7] = vertical FoV in radians (already relu'd by the head).
                    // c[8] = horizontal FoV in radians (unused by TrajectoryPose,
                    // which carries a single fov_deg). Convert + clamp to sane range.
                    let fov_v_rad = c[7].max(1e-3);
                    let fov_deg = (fov_v_rad * 180.0 / std::f32::consts::PI).clamp(10.0, 170.0);
                    traj.push(TrajectoryPose { position, forward, up, fov_deg });
                }
                Ok(traj)
            }
            _ => Self::synthetic_trajectory_static(n_views, seed),
        }
    }

    fn synthetic_trajectory_static(n: usize, seed: u64) -> Result<Vec<TrajectoryPose>> {
        let mut traj = Vec::with_capacity(n);
        for i in 0..n {
            let t = (i as f32) / (n as f32).max(1.0);
            let phi = t * std::f32::consts::TAU + (seed as f32 % 1.0);
            traj.push(TrajectoryPose {
                position: [phi.cos() * 2.0, 1.6, phi.sin() * 2.0],
                forward: [(-phi).cos(), 0.0, (-phi).sin()],
                up: [0.0, 1.0, 0.0],
                fov_deg: 70.0,
            });
        }
        Ok(traj)
    }

    fn synthetic_trajectory(&self, n: usize, seed: u64) -> Result<Vec<TrajectoryPose>> {
        let mut traj = Vec::with_capacity(n);
        for i in 0..n {
            let t = (i as f32) / (n as f32).max(1.0);
            let phi = t * std::f32::consts::TAU + (seed as f32 % 1.0);
            traj.push(TrajectoryPose {
                position: [phi.cos() * 2.0, 1.6, phi.sin() * 2.0],
                forward: [(-phi).cos(), 0.0, (-phi).sin()],
                up: [0.0, 1.0, 0.0],
                fov_deg: 70.0,
            });
        }
        Ok(traj)
    }

    /// Generate a 3D world from a text prompt. **Upstream-pending**: needs
    /// HY-Pano 2.0 weights to actually produce a panorama; until those
    /// release, this path emits a synthetic spiral splat cloud through
    /// the stub stages.
    pub fn generate_from_text(&self, prompt: &str, seed: u64) -> Result<WorldOutput> {
        let panorama = self.pano_generate_from_text(prompt, seed)?;
        self.complete_pipeline(panorama, seed)
    }

    /// Generate a 3D world from a single reference image (RGB f16
    /// `[3, H, W]`). The image is up-projected to a 360° panorama by
    /// HY-Pano 2.0 (which is conditioned on either text or an anchor view).
    pub fn generate_from_image(&self, image_chw: &Tensor, seed: u64) -> Result<WorldOutput> {
        let panorama = self.pano_generate_from_image(image_chw, seed)?;
        self.complete_pipeline(panorama, seed)
    }

    fn complete_pipeline(&self, panorama: Tensor, seed: u64) -> Result<WorldOutput> {
        let trajectory = self.worldnav_plan(&panorama, seed)?;
        let stereo_views = self.worldstereo_expand(&panorama, &trajectory, seed)?;
        let splats = self.worldmirror_reconstruct(&stereo_views, seed)?;

        let panorama_preview = panorama.to_f32_vec().ok();
        Ok(WorldOutput {
            splats,
            trajectory,
            panorama_preview,
            mesh_obj: None,
            depth_maps: None,
            normal_maps: None,
            point_clouds: None,
            steps_executed: self.config.trajectory_steps as u32,
        })
    }

    // ───── Stage 1: HY-Pano 2.0 ─────

    /// Text → panorama via HY-Pano 2.0 (DiT, ~1.5B params).
    ///
    /// Real forward: tokenise prompt → CLIP-G encode → DiT denoising loop
    /// over `[1, 3, pano_height, pano_width]` latents using flow matching.
    /// Mirrors the FLUX-style DiT in `flux.rs` but with cylindrical/
    /// equirectangular position embeddings and seam-aware attention.
    fn pano_generate_from_text(&self, _prompt: &str, _seed: u64) -> Result<Tensor> {
        if self.pano_model.is_none() {
            return self.pano_stub();
        }
        // TODO: real HY-Pano 2.0 forward — port DiT + equirectangular RoPE.
        self.pano_stub()
    }

    /// Image → panorama via HY-Pano 2.0 (image-conditioning branch).
    fn pano_generate_from_image(&self, _image_chw: &Tensor, _seed: u64) -> Result<Tensor> {
        if self.pano_model.is_none() {
            return self.pano_stub();
        }
        // TODO: real HY-Pano 2.0 image-conditioned forward.
        self.pano_stub()
    }

    /// Shape-correct neutral-grey panorama, used when weights aren't loaded.
    fn pano_stub(&self) -> Result<Tensor> {
        Tensor::zeros(
            Shape::from([1, 3, self.config.pano_height, self.config.pano_width]),
            DType::F16,
        )
    }

    // ───── Stage 2: WorldNav ─────

    /// Plan camera trajectory through the generated panorama.
    ///
    /// Real forward: WorldNav transformer reads a downsampled panorama,
    /// emits `trajectory_steps` poses arranged on a navigable arc through
    /// the dominant scene scale (rooms / corridors detected by attention).
    /// Without weights we emit a deterministic spiral-out trajectory at
    /// the world centre, sufficient for the downstream stages to run.
    fn worldnav_plan(
        &self,
        _panorama: &Tensor,
        seed: u64,
    ) -> Result<Vec<TrajectoryPose>> {
        let n = self.config.trajectory_steps;
        let mut traj = Vec::with_capacity(n);
        let phi_step = std::f32::consts::TAU / (n as f32).max(1.0);
        let radius_step = 0.25;
        for i in 0..n {
            let phi = (i as f32) * phi_step + (seed as f32 % 1.0);
            let r = 0.5 + (i as f32) * radius_step;
            let position = [phi.cos() * r, 1.6, phi.sin() * r];
            let forward = [(-phi).cos(), 0.0, (-phi).sin()];
            traj.push(TrajectoryPose {
                position,
                forward,
                up: [0.0, 1.0, 0.0],
                fov_deg: 70.0,
            });
        }
        Ok(traj)
    }

    // ───── Stage 3: WorldStereo 2.0 ─────

    /// Expand panorama into multi-view stereo by re-rendering through the
    /// trajectory and estimating depth per view.
    ///
    /// Real forward: WorldStereo 2.0 is a feed-forward depth + visibility
    /// network conditioned on the panorama and the planned poses. Output
    /// is a tensor of `[N, 4, H, W]` (RGB + depth) for N=trajectory_steps.
    fn worldstereo_expand(
        &self,
        _panorama: &Tensor,
        trajectory: &[TrajectoryPose],
        _seed: u64,
    ) -> Result<Tensor> {
        let n = trajectory.len();
        let h = self.config.pano_height / 2;
        let w = self.config.pano_width / 4;
        Tensor::zeros(Shape::from([n, 4, h, w]), DType::F16)
    }

    // ───── Stage 4: WorldMirror 2.0 + 3DGS ─────

    /// Reconstruct a 3D Gaussian splat scene from the multi-view stereo
    /// stack. WorldMirror 2.0 (`tencent/HunyuanWorld-Mirror`) is the
    /// universal feed-forward 3D reconstructor — DINOv2 image encoder +
    /// transformer that predicts per-pixel Gaussian parameters.
    ///
    /// Without weights we emit a sparse spiral cloud of splats along the
    /// trajectory so the studio shell can still render *something*
    /// recognisable; once the WorldMirror model is loaded, this
    /// implementation will mirror `hunyuan3d.rs::flow_matching_loop` but
    /// emit Gaussian parameters instead of SDF tokens.
    fn worldmirror_reconstruct(
        &self,
        stereo_views: &Tensor,
        seed: u64,
    ) -> Result<Vec<GaussianSplat>> {
        // Real path: when WorldMirror 2.0 weights are loaded, run the
        // DINOv2-L patch_embed encoder (v1 — frame/global blocks + DPT
        // decoders are still pending) and emit splats shaded by the
        // per-patch features.
        if let Some(model) = &self.mirror_model {
            let runtime = self.mirror_runtime.get_or_init(|| {
                crate::inference::architecture::worldmirror_forward::WorldMirrorRuntime::new(
                    model.clone(), self.device.clone(),
                ).expect("WorldMirror runtime build failed")
            });
            let dims = stereo_views.shape().dims().to_vec();
            // Flatten the multi-view stack to f32 CHW concatenated.
            let f16: Vec<half::f16> = stereo_views.to_vec()?;
            let chw_f32: Vec<f32> = f16.iter().map(|v| v.to_f32()).collect();
            let n_views = dims.first().copied().unwrap_or(1);
            return runtime.forward(&chw_f32, n_views, seed);
        }

        // Skeleton fallback (no weights): synthetic spiral cloud.
        let n = self.config.splat_budget.min(50_000);
        let mut splats = Vec::with_capacity(n);
        let golden_angle = 2.39996323_f32;
        for i in 0..n {
            let t = (i as f32 + 0.5) / n as f32;
            let phi = (i as f32) * golden_angle + (seed as f32 % 1.0);
            let r = (t * 5.0).sqrt();
            let y = (t * 2.0 - 1.0) * 2.0;
            splats.push(GaussianSplat {
                position: [phi.cos() * r, y, phi.sin() * r],
                scale: [0.05, 0.05, 0.05],
                rotation: [0.0, 0.0, 0.0, 1.0],
                color: [0.6, 0.6, 0.7],
                opacity: 0.8,
            });
        }
        Ok(splats)
    }
}

// ───── Splat archive serialisation ─────

/// Serialise a list of Gaussian splats as the standard `.splat` binary
/// format used by web viewers (Niantic, gsplat.js, etc.).
///
/// Layout per splat (32 bytes):
///   - position: 3 × f32         (12 B)
///   - scale:    3 × f32         (12 B)
///   - color:    4 × u8 (RGBA)   (4 B)
///   - rot:      4 × u8 quantised quaternion (4 B)
pub fn encode_splat_archive(splats: &[GaussianSplat]) -> Vec<u8> {
    let mut out = Vec::with_capacity(splats.len() * 32);
    for s in splats {
        for v in s.position { out.extend_from_slice(&v.to_le_bytes()); }
        for v in s.scale    { out.extend_from_slice(&v.to_le_bytes()); }
        let r = (s.color[0].clamp(0.0, 1.0) * 255.0) as u8;
        let g = (s.color[1].clamp(0.0, 1.0) * 255.0) as u8;
        let b = (s.color[2].clamp(0.0, 1.0) * 255.0) as u8;
        let a = (s.opacity.clamp(0.0, 1.0) * 255.0) as u8;
        out.extend_from_slice(&[r, g, b, a]);
        let qx = ((s.rotation[0].clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0) as u8;
        let qy = ((s.rotation[1].clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0) as u8;
        let qz = ((s.rotation[2].clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0) as u8;
        let qw = ((s.rotation[3].clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0) as u8;
        out.extend_from_slice(&[qx, qy, qz, qw]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_quality_presets() {
        let s = HYWorldConfig::default_release().with_quality(WorldQuality::Standard);
        let d = HYWorldConfig::default_release().with_quality(WorldQuality::Draft);
        let c = HYWorldConfig::default_release().with_quality(WorldQuality::Cinematic);
        assert!(d.splat_budget < s.splat_budget);
        assert!(c.splat_budget > s.splat_budget);
        assert!(d.trajectory_steps < s.trajectory_steps);
        assert!(c.trajectory_steps > s.trajectory_steps);
        assert!(d.stereo_step_degrees > s.stereo_step_degrees);
        assert!(c.stereo_step_degrees < s.stereo_step_degrees);
    }

    #[test]
    fn splat_archive_layout() {
        let splats = vec![GaussianSplat {
            position: [1.0, 2.0, 3.0],
            scale: [0.1, 0.2, 0.3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            color: [1.0, 0.5, 0.0],
            opacity: 0.75,
        }];
        let bytes = encode_splat_archive(&splats);
        assert_eq!(bytes.len(), 32);
        assert_eq!(&bytes[0..4], &1.0_f32.to_le_bytes());
        assert_eq!(&bytes[4..8], &2.0_f32.to_le_bytes());
        assert_eq!(&bytes[8..12], &3.0_f32.to_le_bytes());
        assert_eq!(bytes[24], 255); // R
        assert_eq!(bytes[25], 127); // G
        assert_eq!(bytes[26], 0);   // B
        assert_eq!(bytes[27], 191); // A (0.75 → 191)
    }
}
