//! Main inference engine.

use super::{InferenceConfig, Model, Session, SessionConfig};
use super::config::{TextParams, ImageParams, ThreeDParams};
use crate::core::{Error, Result};
use crate::runtime::{ResourceMonitor, Runtime, StreamingOutput};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::hal::{LazyLoader, MetalDevice};

/// The main inference engine.
///
/// Manages models, sessions, and inference execution with
/// optimal resource usage for real-time generation.
pub struct Engine {
    /// Runtime
    runtime: Arc<Runtime>,
    /// Resource monitor
    monitor: Arc<ResourceMonitor>,
    /// Loaded models
    models: parking_lot::RwLock<HashMap<String, Arc<Model>>>,
    /// Active sessions
    sessions: parking_lot::RwLock<HashMap<String, Arc<Session>>>,
    /// Configuration
    config: InferenceConfig,
    /// Metal device (macOS)
    #[cfg(feature = "metal")]
    metal_device: Arc<MetalDevice>,
    /// Lazy loader (macOS)
    #[cfg(feature = "metal")]
    lazy_loader: Arc<LazyLoader>,
}

impl Engine {
    /// Create a new inference engine with default configuration.
    pub fn new() -> Result<Self> {
        Self::with_config(InferenceConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: InferenceConfig) -> Result<Self> {
        let runtime = Arc::new(Runtime::new()?);
        let monitor = Arc::new(ResourceMonitor::new());

        #[cfg(feature = "metal")]
        let metal_device = Arc::new(MetalDevice::new()?);

        #[cfg(feature = "metal")]
        let lazy_loader = Arc::new(LazyLoader::new(Arc::clone(&metal_device)));

        tracing::info!("Inference engine initialized");
        tracing::info!("  Primary device: {}", runtime.primary_device());
        #[cfg(feature = "metal")]
        tracing::info!("  Apple Silicon: {:?}", metal_device.generation());

        Ok(Self {
            runtime,
            monitor,
            models: parking_lot::RwLock::new(HashMap::new()),
            sessions: parking_lot::RwLock::new(HashMap::new()),
            config,
            #[cfg(feature = "metal")]
            metal_device,
            #[cfg(feature = "metal")]
            lazy_loader,
        })
    }

    /// Get the runtime.
    pub fn runtime(&self) -> &Arc<Runtime> {
        &self.runtime
    }

    /// Get the resource monitor.
    pub fn monitor(&self) -> &Arc<ResourceMonitor> {
        &self.monitor
    }

    /// Load a model from a path.
    ///
    /// Models are loaded lazily - weights are memory-mapped
    /// and only loaded into RAM when accessed by the GPU.
    pub fn load_model(&self, name: &str, path: &Path) -> Result<Arc<Model>> {
        tracing::info!("Loading model '{}' from {}", name, path.display());

        let model = Model::load(
            name,
            path,
            #[cfg(feature = "metal")]
            Arc::clone(&self.lazy_loader),
        )?;

        let model = Arc::new(model);
        self.models.write().insert(name.to_string(), Arc::clone(&model));

        tracing::info!(
            "Model '{}' ready ({} parameters, {:.1} MB on disk)",
            name,
            model.info().num_parameters,
            model.info().size_bytes as f64 / (1024.0 * 1024.0)
        );

        Ok(model)
    }

    /// Get a loaded model.
    pub fn get_model(&self, name: &str) -> Option<Arc<Model>> {
        self.models.read().get(name).cloned()
    }

    /// Unload a model.
    pub fn unload_model(&self, name: &str) -> bool {
        self.models.write().remove(name).is_some()
    }

    /// List loaded models.
    pub fn list_models(&self) -> Vec<String> {
        self.models.read().keys().cloned().collect()
    }

    /// Create a new inference session.
    pub fn create_session(&self, model_name: &str, config: SessionConfig) -> Result<Arc<Session>> {
        let model = self.get_model(model_name)
            .ok_or_else(|| Error::model_load(model_name, "model not loaded"))?;

        let session = Session::new(
            Arc::clone(&model),
            config,
            Arc::clone(&self.monitor),
        )?;

        let session = Arc::new(session);
        let session_id = session.id().to_string();
        self.sessions.write().insert(session_id, Arc::clone(&session));

        Ok(session)
    }

    /// Get an active session.
    pub fn get_session(&self, id: &str) -> Option<Arc<Session>> {
        self.sessions.read().get(id).cloned()
    }

    /// Close a session.
    pub fn close_session(&self, id: &str) -> bool {
        self.sessions.write().remove(id).is_some()
    }

    // ========== Text Generation ==========

    /// Generate text (streaming).
    pub fn generate_text(
        &self,
        session: &Session,
        prompt: &str,
        params: TextParams,
    ) -> StreamingOutput<TextToken> {
        let (output, sender) = crate::runtime::stream::StreamBuilder::new()
            .buffer_size(params.max_tokens.min(256))
            .build();

        // Spawn generation task
        let session = session.clone();
        let prompt = prompt.to_string();
        let monitor = Arc::clone(&self.monitor);

        tokio::spawn(async move {
            let result = session.generate_text_internal(&prompt, params, &sender, &monitor).await;
            if let Err(e) = result {
                let _ = sender.send_error(e).await;
            }
            sender.complete();
        });

        output
    }

    /// Generate text (blocking, returns full response).
    pub async fn generate_text_sync(
        &self,
        session: &Session,
        prompt: &str,
        params: TextParams,
    ) -> Result<String> {
        let stream = self.generate_text(session, prompt, params);
        let tokens: Vec<TextToken> = stream.collect().await?;
        Ok(tokens.into_iter().map(|t| t.text).collect())
    }

    // ========== Image Generation ==========

    /// Generate an image.
    pub async fn generate_image(
        &self,
        session: &Session,
        prompt: &str,
        params: ImageParams,
    ) -> Result<Tensor> {
        session.generate_image_internal(prompt, params, &self.monitor).await
    }

    /// Generate image with progressive output.
    pub fn generate_image_progressive(
        &self,
        session: &Session,
        prompt: &str,
        params: ImageParams,
    ) -> StreamingOutput<ImageProgress> {
        let (output, sender) = crate::runtime::stream::StreamBuilder::new()
            .buffer_size(params.num_steps as usize)
            .build();

        let session = session.clone();
        let prompt = prompt.to_string();
        let monitor = Arc::clone(&self.monitor);

        tokio::spawn(async move {
            let result = session.generate_image_progressive_internal(
                &prompt, params, &sender, &monitor
            ).await;
            if let Err(e) = result {
                let _ = sender.send_error(e).await;
            }
            sender.complete();
        });

        output
    }

    // ========== 3D Generation ==========

    /// Generate 3D from image.
    pub async fn image_to_3d(
        &self,
        session: &Session,
        image: &Tensor,
        params: ThreeDParams,
    ) -> Result<Object3D> {
        session.image_to_3d_internal(image, params, &self.monitor).await
    }

    /// Render a view of a 3D object.
    pub fn render_3d_view(
        &self,
        object: &Object3D,
        camera: &Camera3D,
    ) -> Result<Tensor> {
        // This should be <10ms on Apple Silicon
        let _timer = crate::runtime::monitor::ScopedTimer::new(
            self.monitor.compute(),
            (object.num_gaussians * 100) as u64, // Estimate FLOPs
        );

        object.render(camera)
    }

    /// Stream 3D views along a camera path.
    pub fn stream_3d_views(
        &self,
        object: &Object3D,
        path: CameraPath,
        fps: f32,
    ) -> StreamingOutput<Tensor> {
        let (output, sender) = crate::runtime::stream::StreamBuilder::new()
            .buffer_size(8)
            .build();

        let object = object.clone();
        let _monitor = Arc::clone(&self.monitor);

        tokio::spawn(async move {
            let frame_duration = std::time::Duration::from_secs_f32(1.0 / fps);
            let total_frames = (path.duration() * fps) as usize;

            for i in 0..total_frames {
                if sender.is_cancelled() {
                    break;
                }

                let t = i as f32 / fps;
                let camera = path.sample(t);

                match object.render(&camera) {
                    Ok(frame) => {
                        if sender.send(frame).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = sender.send_error(e).await;
                        break;
                    }
                }

                // Rate limit to target FPS
                tokio::time::sleep(frame_duration).await;
            }

            sender.complete();
        });

        output
    }

    // ========== Resource Management ==========

    /// Get memory statistics.
    pub fn memory_stats(&self) -> crate::runtime::monitor::MemoryStats {
        self.monitor.memory().stats()
    }

    /// Get compute statistics.
    pub fn compute_stats(&self) -> crate::runtime::ComputeStats {
        self.monitor.compute().stats()
    }

    /// Get full resource snapshot.
    pub fn resource_snapshot(&self) -> crate::runtime::ResourceSnapshot {
        self.monitor.snapshot()
    }

    /// Check if within memory budget.
    pub fn is_within_memory_budget(&self) -> bool {
        #[cfg(feature = "metal")]
        {
            self.metal_device.is_within_memory_budget()
        }
        #[cfg(not(feature = "metal"))]
        {
            true
        }
    }

    /// Trigger garbage collection.
    pub fn gc(&self) {
        self.runtime.clear_cache();
        tracing::debug!("Garbage collection complete");
    }

    /// Prefetch model weights.
    pub fn prefetch_model(&self, name: &str) -> Result<()> {
        let model = self.get_model(name)
            .ok_or_else(|| Error::model_load(name, "model not loaded"))?;

        model.prefetch();
        Ok(())
    }
}

/// A generated text token.
#[derive(Debug, Clone)]
pub struct TextToken {
    /// Token ID
    pub id: u32,
    /// Decoded text
    pub text: String,
    /// Log probability
    pub logprob: Option<f32>,
    /// Is this the final token?
    pub is_final: bool,
}

/// Image generation progress.
#[derive(Debug)]
pub struct ImageProgress {
    /// Current step
    pub step: u32,
    /// Total steps
    pub total_steps: u32,
    /// Preview image (low-res)
    pub preview: Option<Tensor>,
    /// Final image (on completion)
    pub final_image: Option<Tensor>,
}

/// A 3D object (Gaussian splats).
#[derive(Debug, Clone)]
pub struct Object3D {
    /// Gaussian positions [N, 3]
    pub positions: Tensor,
    /// Gaussian scales [N, 3]
    pub scales: Tensor,
    /// Gaussian rotations [N, 4]
    pub rotations: Tensor,
    /// Gaussian colors [N, 3]
    pub colors: Tensor,
    /// Gaussian opacities [N, 1]
    pub opacities: Tensor,
    /// Number of gaussians
    pub num_gaussians: usize,
}

impl Object3D {
    /// Render from a camera viewpoint using CPU Gaussian splatting.
    ///
    /// Projects each Gaussian to screen space, sorts by depth (back to front),
    /// and rasterizes with alpha blending. Output is [1, 3, 512, 512] RGB tensor.
    pub fn render(&self, camera: &Camera3D) -> Result<Tensor> {
        let width = 512usize;
        let height = 512usize;
        let mut image_data = vec![0.0f32; 3 * height * width];

        if self.num_gaussians == 0 {
            let shape = crate::core::Shape::from([1, 3, height, width]);
            return Tensor::from_slice(&image_data, shape, crate::tensor::DType::F32, crate::hal::DeviceId::cpu());
        }

        let positions: Vec<f32> = self.positions.to_vec()?;
        let colors: Vec<f32> = self.colors.to_vec()?;
        let opacities: Vec<f32> = self.opacities.to_vec()?;

        // View matrix from camera
        let (eye, target, up) = (camera.position, camera.target, camera.up);
        let fwd = normalize([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
        let right = normalize(cross(fwd, up));
        let cam_up = cross(right, fwd);

        // Project and sort
        let mut projected: Vec<(usize, f32)> = Vec::with_capacity(self.num_gaussians);
        for i in 0..self.num_gaussians {
            let px = positions.get(i * 3).copied().unwrap_or(0.0) - eye[0];
            let py = positions.get(i * 3 + 1).copied().unwrap_or(0.0) - eye[1];
            let pz = positions.get(i * 3 + 2).copied().unwrap_or(0.0) - eye[2];

            let depth = px * fwd[0] + py * fwd[1] + pz * fwd[2];
            if depth > 0.1 {
                projected.push((i, depth));
            }
        }
        projected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let fov_rad = camera.fov.to_radians();
        let focal = (width as f32 / 2.0) / (fov_rad / 2.0).tan();

        for &(idx, depth) in &projected {
            let px = positions.get(idx * 3).copied().unwrap_or(0.0) - eye[0];
            let py = positions.get(idx * 3 + 1).copied().unwrap_or(0.0) - eye[1];
            let pz = positions.get(idx * 3 + 2).copied().unwrap_or(0.0) - eye[2];

            let cam_x = px * right[0] + py * right[1] + pz * right[2];
            let cam_y = px * cam_up[0] + py * cam_up[1] + pz * cam_up[2];

            let screen_x = (focal * cam_x / depth + width as f32 / 2.0) as i32;
            let screen_y = (height as f32 / 2.0 - focal * cam_y / depth) as i32;

            let r = colors.get(idx * 3).copied().unwrap_or(0.5);
            let g = colors.get(idx * 3 + 1).copied().unwrap_or(0.5);
            let b = colors.get(idx * 3 + 2).copied().unwrap_or(0.5);
            let alpha = opacities.get(idx).copied().unwrap_or(0.5).clamp(0.0, 1.0);

            let radius = (3.0 / depth.max(0.1)) as i32;
            let radius = radius.clamp(1, 20);

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let sx = screen_x + dx;
                    let sy = screen_y + dy;
                    if sx < 0 || sx >= width as i32 || sy < 0 || sy >= height as i32 {
                        continue;
                    }
                    let dist_sq = (dx * dx + dy * dy) as f32;
                    let sigma_sq = (radius as f32 * 0.5).powi(2);
                    let weight = alpha * (-dist_sq / (2.0 * sigma_sq)).exp();
                    if weight < 0.001 { continue; }

                    let pixel = sy as usize * width + sx as usize;
                    image_data[pixel] = image_data[pixel] * (1.0 - weight) + r * weight;
                    image_data[height * width + pixel] = image_data[height * width + pixel] * (1.0 - weight) + g * weight;
                    image_data[2 * height * width + pixel] = image_data[2 * height * width + pixel] * (1.0 - weight) + b * weight;
                }
            }
        }

        let shape = crate::core::Shape::from([1, 3, height, width]);
        Tensor::from_slice(&image_data, shape, crate::tensor::DType::F32, crate::hal::DeviceId::cpu())
    }

    /// Export Gaussians to triangle mesh.
    ///
    /// Samples the Gaussian density field on a 3D grid and extracts the
    /// isosurface at a density threshold using a simplified marching cubes.
    pub fn to_mesh(&self) -> Result<Mesh> {
        if self.num_gaussians == 0 {
            let device = crate::hal::DeviceId::cpu();
            return Ok(Mesh {
                vertices: Tensor::from_slice(&[0.0f32; 3], crate::core::Shape::from([1, 3]), crate::tensor::DType::F32, device)?,
                faces: Tensor::from_slice(&[0.0f32; 3], crate::core::Shape::from([1, 3]), crate::tensor::DType::F32, device)?,
            });
        }

        let positions: Vec<f32> = self.positions.to_vec()?;

        // Compute bounding box
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for i in 0..self.num_gaussians {
            for d in 0..3 {
                let v = positions.get(i * 3 + d).copied().unwrap_or(0.0);
                min[d] = min[d].min(v);
                max[d] = max[d].max(v);
            }
        }

        // Expand bounds slightly
        for d in 0..3 {
            let margin = (max[d] - min[d]) * 0.1 + 0.1;
            min[d] -= margin;
            max[d] += margin;
        }

        // Voxelize Gaussian density
        let grid_res = 32usize;
        let mut density = vec![0.0f32; grid_res * grid_res * grid_res];

        for i in 0..self.num_gaussians {
            let gx = positions.get(i * 3).copied().unwrap_or(0.0);
            let gy = positions.get(i * 3 + 1).copied().unwrap_or(0.0);
            let gz = positions.get(i * 3 + 2).copied().unwrap_or(0.0);

            // Map to grid
            let vx = ((gx - min[0]) / (max[0] - min[0]) * (grid_res - 1) as f32) as usize;
            let vy = ((gy - min[1]) / (max[1] - min[1]) * (grid_res - 1) as f32) as usize;
            let vz = ((gz - min[2]) / (max[2] - min[2]) * (grid_res - 1) as f32) as usize;

            let vx = vx.min(grid_res - 1);
            let vy = vy.min(grid_res - 1);
            let vz = vz.min(grid_res - 1);

            // Splat Gaussian into nearby voxels
            let spread = 2i32;
            for dz in -spread..=spread {
                for dy in -spread..=spread {
                    for dx in -spread..=spread {
                        let nx = vx as i32 + dx;
                        let ny = vy as i32 + dy;
                        let nz = vz as i32 + dz;
                        if nx >= 0 && nx < grid_res as i32 && ny >= 0 && ny < grid_res as i32 && nz >= 0 && nz < grid_res as i32 {
                            let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                            let weight = (-dist_sq * 0.5).exp();
                            density[nz as usize * grid_res * grid_res + ny as usize * grid_res + nx as usize] += weight;
                        }
                    }
                }
            }
        }

        // Extract isosurface (simplified marching cubes)
        let threshold = 0.5f32;
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        for z in 0..grid_res - 1 {
            for y in 0..grid_res - 1 {
                for x in 0..grid_res - 1 {
                    let idx = z * grid_res * grid_res + y * grid_res + x;
                    let v = density[idx];

                    // Check neighbors for crossings
                    let vx_next = density.get(idx + 1).copied().unwrap_or(0.0);
                    let vy_next = density.get(idx + grid_res).copied().unwrap_or(0.0);
                    let vz_next = density.get(idx + grid_res * grid_res).copied().unwrap_or(0.0);

                    if (v > threshold) != (vx_next > threshold) {
                        let t = if vx_next != v { (threshold - v) / (vx_next - v) } else { 0.5 };
                        let wx = min[0] + ((x as f32 + t) / grid_res as f32) * (max[0] - min[0]);
                        let wy = min[1] + (y as f32 / grid_res as f32) * (max[1] - min[1]);
                        let wz = min[2] + (z as f32 / grid_res as f32) * (max[2] - min[2]);
                        let step_y = (max[1] - min[1]) / grid_res as f32;
                        let step_z = (max[2] - min[2]) / grid_res as f32;

                        let base = vertices.len() / 3;
                        vertices.extend_from_slice(&[wx, wy, wz, wx, wy + step_y, wz, wx, wy, wz + step_z, wx, wy + step_y, wz + step_z]);
                        let b = base as f32;
                        faces.extend_from_slice(&[b, b + 1.0, b + 2.0, b + 1.0, b + 3.0, b + 2.0]);
                    }
                }
            }
        }

        let device = crate::hal::DeviceId::cpu();
        let num_verts = vertices.len() / 3;
        let num_faces = faces.len() / 3;

        if num_verts == 0 {
            return Ok(Mesh {
                vertices: Tensor::from_slice(&[0.0f32; 3], crate::core::Shape::from([1, 3]), crate::tensor::DType::F32, device)?,
                faces: Tensor::from_slice(&[0.0f32; 3], crate::core::Shape::from([1, 3]), crate::tensor::DType::F32, device)?,
            });
        }

        Ok(Mesh {
            vertices: Tensor::from_slice(&vertices, crate::core::Shape::from([num_verts, 3]), crate::tensor::DType::F32, device)?,
            faces: Tensor::from_slice(&faces, crate::core::Shape::from([num_faces, 3]), crate::tensor::DType::F32, device)?,
        })
    }
}

/// Normalize a 3D vector.
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-8);
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Cross product of two 3D vectors.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// 3D camera.
#[derive(Debug, Clone)]
pub struct Camera3D {
    /// Position
    pub position: [f32; 3],
    /// Look-at target
    pub target: [f32; 3],
    /// Up vector
    pub up: [f32; 3],
    /// Field of view (degrees)
    pub fov: f32,
}

impl Camera3D {
    /// Create an orbit camera.
    pub fn orbit(center: [f32; 3], radius: f32, theta: f32, phi: f32) -> Self {
        let x = radius * phi.cos() * theta.cos();
        let y = radius * phi.sin();
        let z = radius * phi.cos() * theta.sin();

        Self {
            position: [center[0] + x, center[1] + y, center[2] + z],
            target: center,
            up: [0.0, 1.0, 0.0],
            fov: 45.0,
        }
    }
}

/// Camera path for animations.
#[derive(Debug, Clone)]
pub struct CameraPath {
    /// Duration in seconds
    duration_seconds: f32,
    /// Path type
    path_type: CameraPathType,
    /// Center point
    center: [f32; 3],
    /// Radius
    radius: f32,
}

impl CameraPath {
    /// Create an orbit path.
    pub fn orbit(center: [f32; 3], radius: f32, duration: f32) -> Self {
        Self {
            duration_seconds: duration,
            path_type: CameraPathType::Orbit,
            center,
            radius,
        }
    }

    /// Get duration.
    pub fn duration(&self) -> f32 {
        self.duration_seconds
    }

    /// Sample camera at time t.
    pub fn sample(&self, t: f32) -> Camera3D {
        match self.path_type {
            CameraPathType::Orbit => {
                let theta = (t / self.duration_seconds) * std::f32::consts::TAU;
                Camera3D::orbit(self.center, self.radius, theta, 0.3)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum CameraPathType {
    Orbit,
}

/// Triangle mesh.
#[derive(Debug)]
pub struct Mesh {
    /// Vertices [N, 3]
    pub vertices: Tensor,
    /// Faces [M, 3]
    pub faces: Tensor,
}
