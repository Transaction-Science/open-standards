//! Fusion Pipeline — hybrid neural-procedural generation.
//!
//! Combines lightweight neural encoding (semantic understanding)
//! with deterministic procedural execution (content synthesis).
//!
//! The neural encoder maps input prompts to a Generation Parameter Vector (GPV),
//! which the procedural engine expands into full-resolution output via Metal
//! compute shaders (SDF ray marching, Perlin noise, Karplus-Strong waveguide).

#[cfg(feature = "metal")]
use crate::core::Result;
#[cfg(feature = "metal")]
use crate::hal::metal::MetalOps;
#[cfg(feature = "metal")]
use crate::inference::gpv::*;
#[cfg(feature = "metal")]
use std::sync::Arc;

/// Configuration for the fusion pipeline.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Default image output width.
    pub width: u32,
    /// Default image output height.
    pub height: u32,
    /// Default audio sample rate.
    pub sample_rate: u32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            sample_rate: 44100,
        }
    }
}

/// Output from the fusion pipeline.
#[derive(Debug)]
pub enum FusionOutput {
    /// RGB image as f32 [height, width, 3] in [0, 1].
    Image {
        /// Pixel data as interleaved RGB floats.
        data: Vec<f32>,
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
    /// PCM audio as f32 samples in [-1, 1].
    Audio {
        /// PCM sample data.
        data: Vec<f32>,
        /// Audio sample rate in Hz.
        sample_rate: u32,
    },
}

/// Number of continuous parameters that define an image GPV.
///
/// Layout (60 floats total):
///   [0..9]   = noise: type(1), octaves(1), lacunarity(1), persistence(1), scale(1), amplitude(1), blend(1), seed(1), reserved(1)
///   [9..29]  = palette: 5 stops × 4 floats each (pos, r, g, b)
///   [29..48] = SDF params: 3 primitive radii(3), 3 translations(9), 3 smooth_k(3), camera_eye(3)
///   [48..60] = composition: fov(1), light_dir(3), background(3), reserved(5)
pub const IMAGE_GPV_DIM: usize = 60;

/// Number of continuous parameters for an audio GPV.
///
/// Layout (32 floats):
///   [0..4]   = envelope: attack, decay, sustain, release
///   [4..8]   = waveguide: damping, brightness, excitation_pos, excitation_vel
///   [8..32]  = sequence: bpm(1), 7 notes × (midi_note, velocity, duration)(3) = 21, pad(2)
pub const AUDIO_GPV_DIM: usize = 32;

/// A small MLP encoder that maps embeddings to GPV parameter vectors.
///
/// Architecture: input_dim → hidden1 → hidden2 → output_dim
/// with SiLU activations between layers. All f32 on CPU (fast for small vectors).
pub struct GPVEncoder {
    /// Layer 1: weight [hidden1 × input_dim], bias [hidden1]
    w1: Vec<f32>,
    b1: Vec<f32>,
    /// Layer 2: weight [hidden2 × hidden1], bias [hidden2]
    w2: Vec<f32>,
    b2: Vec<f32>,
    /// Layer 3 (output): weight [output_dim × hidden2], bias [output_dim]
    w3: Vec<f32>,
    b3: Vec<f32>,
    /// Dimensions
    input_dim: usize,
    hidden1: usize,
    hidden2: usize,
    output_dim: usize,
}

impl GPVEncoder {
    /// Create a new encoder with Xavier-uniform initialization.
    pub fn new(input_dim: usize, hidden1: usize, hidden2: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let w1 = xavier_init(&mut rng, hidden1, input_dim);
        let b1 = vec![0.0; hidden1];
        let w2 = xavier_init(&mut rng, hidden2, hidden1);
        let b2 = vec![0.0; hidden2];
        let w3 = xavier_init(&mut rng, output_dim, hidden2);
        let b3 = vec![0.0; output_dim];
        Self { w1, b1, w2, b2, w3, b3, input_dim, hidden1, hidden2, output_dim }
    }

    /// Forward pass: embedding → GPV parameter vector.
    pub fn forward(&self, embedding: &[f32]) -> Vec<f32> {
        assert_eq!(embedding.len(), self.input_dim);
        let h1 = linear_silu(&self.w1, &self.b1, embedding, self.hidden1);
        let h2 = linear_silu(&self.w2, &self.b2, &h1, self.hidden2);
        linear_forward(&self.w3, &self.b3, &h2, self.output_dim)
    }

    /// Total number of trainable parameters.
    pub fn num_params(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len() + self.w3.len() + self.b3.len()
    }

    /// Flatten all parameters into a single vector (for optimization).
    pub fn flatten_params(&self) -> Vec<f32> {
        let mut params = Vec::with_capacity(self.num_params());
        params.extend_from_slice(&self.w1);
        params.extend_from_slice(&self.b1);
        params.extend_from_slice(&self.w2);
        params.extend_from_slice(&self.b2);
        params.extend_from_slice(&self.w3);
        params.extend_from_slice(&self.b3);
        params
    }

    /// Restore parameters from a flat vector.
    pub fn load_params(&mut self, params: &[f32]) {
        assert_eq!(params.len(), self.num_params());
        let mut offset = 0;
        for dst in [&mut self.w1, &mut self.b1, &mut self.w2, &mut self.b2, &mut self.w3, &mut self.b3] {
            let len = dst.len();
            dst.copy_from_slice(&params[offset..offset + len]);
            offset += len;
        }
    }
}

/// Inverse fitting: optimize GPV parameters to match a target output.
///
/// Uses Adam optimizer on CPU to minimize MSE between
/// `procedural(gpv_params)` and `target` pixel/sample data.
pub struct InverseFitter {
    /// Learning rate.
    lr: f32,
    /// Number of optimization steps.
    steps: usize,
    /// Adam beta1.
    beta1: f32,
    /// Adam beta2.
    beta2: f32,
}

impl Default for InverseFitter {
    fn default() -> Self {
        Self { lr: 0.01, steps: 200, beta1: 0.9, beta2: 0.999 }
    }
}

impl InverseFitter {
    /// Create a fitter with custom hyperparameters.
    pub fn new(lr: f32, steps: usize) -> Self {
        Self { lr, steps, ..Default::default() }
    }

    /// Fit image GPV parameters to a target image.
    ///
    /// `target` is [height * width * 3] f32 RGB in [0,1].
    /// Returns optimized continuous parameter vector and final loss.
    #[cfg(feature = "metal")]
    pub fn fit_image(
        &self,
        pipeline: &FusionPipeline,
        target: &[f32],
        width: u32,
        height: u32,
    ) -> (Vec<f32>, f32) {
        let dim = IMAGE_GPV_DIM;
        let mut params = vec![0.5f32; dim];

        // Initialize with reasonable defaults
        // Noise params
        params[0] = 0.0; // Perlin
        params[1] = 6.0; // octaves
        params[2] = 2.1; // lacunarity
        params[3] = 0.5; // persistence
        params[4] = 4.0; // scale
        params[5] = 1.0; // amplitude
        params[6] = 0.0; // Replace blend
        params[7] = 42.0; // seed
        // Palette: sunset-like
        params[9] = 0.0; params[10] = 0.05; params[11] = 0.1; params[12] = 0.3;
        params[13] = 0.3; params[14] = 0.2; params[15] = 0.5; params[16] = 0.8;
        params[17] = 0.6; params[18] = 0.9; params[19] = 0.7; params[20] = 0.3;
        params[21] = 0.8; params[22] = 1.0; params[23] = 0.4; params[24] = 0.1;
        params[25] = 1.0; params[26] = 1.0; params[27] = 1.0; params[28] = 0.95;
        // Camera
        params[45] = 0.0; params[46] = 1.5; params[47] = 4.0;
        params[48] = 1.0; // fov
        params[49] = 0.577; params[50] = 0.577; params[51] = -0.577; // light

        // Adam state
        let mut m = vec![0.0f32; dim];
        let mut v = vec![0.0f32; dim];
        let eps = 1e-8f32;
        let mut best_loss = f32::MAX;
        let mut best_params = params.clone();

        // Save original pipeline config
        let orig_w = pipeline.config.width;
        let orig_h = pipeline.config.height;

        for step in 0..self.steps {
            // Forward: params → GPV → render
            let gpv = params_to_image_gpv(&params);

            // Temporarily set pipeline dimensions (safe: single-threaded fitting)
            let pipeline_ptr = pipeline as *const FusionPipeline as *mut FusionPipeline;
            unsafe {
                (*pipeline_ptr).config.width = width;
                (*pipeline_ptr).config.height = height;
            }

            let output = match pipeline.generate(&gpv) {
                Ok(FusionOutput::Image { data, .. }) => data,
                _ => break,
            };

            // Restore
            unsafe {
                (*pipeline_ptr).config.width = orig_w;
                (*pipeline_ptr).config.height = orig_h;
            }

            // MSE loss (downsampled for speed)
            let stride = (target.len() / 3000).max(1); // ~1000 pixel samples
            let mut loss = 0.0f32;
            let mut count = 0usize;
            for i in (0..target.len()).step_by(stride) {
                let diff = output.get(i).copied().unwrap_or(0.0) - target[i];
                loss += diff * diff;
                count += 1;
            }
            loss /= count.max(1) as f32;

            if loss < best_loss {
                best_loss = loss;
                best_params = params.clone();
            }

            if step % 50 == 0 {
                eprintln!("  [fit] step {}/{}: loss={:.6}", step, self.steps, loss);
            }

            // Finite-difference gradients for continuous params
            let delta = 0.01f32;
            let mut grad = vec![0.0f32; dim];
            // Only optimize palette/noise continuous params (skip discrete like noise_type)
            for p in [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 45, 46, 47, 48] {
                if p >= dim { continue; }
                let old = params[p];
                params[p] = old + delta;
                let gpv_plus = params_to_image_gpv(&params);

                unsafe {
                    (*pipeline_ptr).config.width = width;
                    (*pipeline_ptr).config.height = height;
                }
                let out_plus = match pipeline.generate(&gpv_plus) {
                    Ok(FusionOutput::Image { data, .. }) => data,
                    _ => { params[p] = old; continue; }
                };
                unsafe {
                    (*pipeline_ptr).config.width = orig_w;
                    (*pipeline_ptr).config.height = orig_h;
                }

                let mut loss_plus = 0.0f32;
                let mut cnt = 0usize;
                for i in (0..target.len()).step_by(stride) {
                    let diff = out_plus.get(i).copied().unwrap_or(0.0) - target[i];
                    loss_plus += diff * diff;
                    cnt += 1;
                }
                loss_plus /= cnt.max(1) as f32;
                grad[p] = (loss_plus - loss) / delta;
                params[p] = old;
            }

            // Adam update
            let t = (step + 1) as f32;
            for p in 0..dim {
                m[p] = self.beta1 * m[p] + (1.0 - self.beta1) * grad[p];
                v[p] = self.beta2 * v[p] + (1.0 - self.beta2) * grad[p] * grad[p];
                let m_hat = m[p] / (1.0 - self.beta1.powf(t));
                let v_hat = v[p] / (1.0 - self.beta2.powf(t));
                params[p] -= self.lr * m_hat / (v_hat.sqrt() + eps);
            }

            // Clamp palette colors to [0,1]
            for i in 9..29 {
                if i < dim {
                    params[i] = params[i].clamp(0.0, 1.0);
                }
            }
        }

        (best_params, best_loss)
    }
}

/// Fusion pipeline: neural encoding → procedural execution → output.
#[cfg(feature = "metal")]
pub struct FusionPipeline {
    ops: Arc<MetalOps>,
    config: FusionConfig,
    /// Optional GPV encoder for text→procedural generation.
    encoder: Option<GPVEncoder>,
}

#[cfg(feature = "metal")]
impl FusionPipeline {
    /// Create a new FusionPipeline with the given MetalOps and config.
    pub fn new(ops: Arc<MetalOps>, config: FusionConfig) -> Self {
        Self { ops, config, encoder: None }
    }

    /// Attach a trained GPV encoder for text-to-procedural generation.
    pub fn set_encoder(&mut self, encoder: GPVEncoder) {
        self.encoder = Some(encoder);
    }

    /// Generate from an embedding vector using the GPV encoder.
    ///
    /// The embedding is passed through the encoder MLP to predict GPV parameters,
    /// then the procedural engine renders the output.
    pub fn generate_from_embedding(&self, embedding: &[f32], modality: GPVModality) -> Result<FusionOutput> {
        let encoder = self.encoder.as_ref().ok_or_else(|| {
            crate::core::Error::unsupported("No GPV encoder loaded — call set_encoder() first")
        })?;
        let params = encoder.forward(embedding);
        let gpv = match modality {
            GPVModality::Image => params_to_image_gpv(&params),
            GPVModality::Audio => params_to_audio_gpv(&params),
            _ => return Err(crate::core::Error::unsupported(format!("Fusion modality {:?} not yet supported", modality))),
        };
        self.generate(&gpv)
    }

    /// Generate output from a GPV.
    ///
    /// Routes to the appropriate procedural engine based on modality.
    pub fn generate(&self, gpv: &GPV) -> Result<FusionOutput> {
        match &gpv.params {
            GPVParams::Image { sdf_ops, noise_layers, palette, composition } => {
                self.generate_image(sdf_ops, noise_layers, palette, composition)
            }
            GPVParams::Audio { waveguide, excitation, envelope, sequence } => {
                self.generate_audio(waveguide, excitation, envelope, sequence)
            }
        }
    }

    /// Generate an image via SDF ray marching + noise compositing.
    fn generate_image(
        &self,
        sdf_ops: &[SDFOp],
        noise_layers: &[NoiseLayer],
        palette: &ColorPalette,
        composition: &Composition,
    ) -> Result<FusionOutput> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let compute = self.ops.compute();
        let device = compute.device().raw();

        // Encode SDF ops into flat buffer (16 floats per op)
        let scene_data = encode_sdf_scene(sdf_ops);
        let num_ops = scene_data.len() / 16;

        // Camera params: [eye(3), target(3), fov, max_dist, max_steps, bg(3), light(3)]
        let camera_data: Vec<f32> = vec![
            composition.camera_eye[0], composition.camera_eye[1], composition.camera_eye[2],
            composition.camera_target[0], composition.camera_target[1], composition.camera_target[2],
            composition.fov,
            composition.max_distance,
            composition.max_steps as f32,
            composition.background.r, composition.background.g, composition.background.b,
            composition.light_dir[0], composition.light_dir[1], composition.light_dir[2],
        ];

        // Allocate output buffer
        let output_bytes = (width * height * 3 * 4) as u64; // f32 RGB
        let output_buf = device.new_buffer(output_bytes, metal::MTLResourceOptions::StorageModeShared);

        // If we have SDF ops, ray march them first
        if num_ops > 0 {
            let scene_buf = device.new_buffer_with_data(
                scene_data.as_ptr() as *const _,
                (scene_data.len() * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let camera_buf = device.new_buffer_with_data(
                camera_data.as_ptr() as *const _,
                (camera_data.len() * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let cb = compute.new_command_buffer();
            let w = self.config.width;
            let h = self.config.height;
            let n = num_ops as u32;
            compute.dispatch_2d(&cb, self.ops.sdf_raymarch_pipeline(), width, height, |encoder| {
                encoder.set_buffer(0, Some(&scene_buf), 0);
                encoder.set_buffer(1, Some(&camera_buf), 0);
                encoder.set_buffer(2, Some(&output_buf), 0);
                encoder.set_bytes(3, 4, &w as *const u32 as *const _);
                encoder.set_bytes(4, 4, &h as *const u32 as *const _);
                encoder.set_bytes(5, 4, &n as *const u32 as *const _);
            });
            cb.commit();
            cb.wait_until_completed();
        }

        // Apply noise layers
        if !noise_layers.is_empty() {
            let noise_bytes = (width * height * 4) as u64; // f32 single channel
            let noise_buf = device.new_buffer(noise_bytes, metal::MTLResourceOptions::StorageModeShared);

            // Zero the noise buffer
            let noise_ptr = noise_buf.contents() as *mut f32;
            unsafe {
                std::ptr::write_bytes(noise_ptr, 0, width * height);
            }

            for layer in noise_layers {
                let cb = compute.new_command_buffer();
                let w = self.config.width;
                let h = self.config.height;
                let noise_type = match layer.noise_type {
                    NoiseType::Perlin => 0u32,
                    NoiseType::Simplex => 1u32,
                    NoiseType::Worley => 2u32,
                    NoiseType::Value => 3u32,
                };
                let blend = match layer.blend {
                    BlendMode::Replace => 0u32,
                    BlendMode::Add => 1u32,
                    BlendMode::Multiply => 2u32,
                    BlendMode::Screen => 3u32,
                    BlendMode::Overlay => 4u32,
                };
                compute.dispatch_2d(&cb, self.ops.perlin_noise_pipeline(), width, height, |encoder| {
                    encoder.set_buffer(0, Some(&noise_buf), 0);
                    encoder.set_bytes(1, 4, &w as *const u32 as *const _);
                    encoder.set_bytes(2, 4, &h as *const u32 as *const _);
                    encoder.set_bytes(3, 4, &noise_type as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &layer.octaves as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &layer.lacunarity as *const f32 as *const _);
                    encoder.set_bytes(6, 4, &layer.persistence as *const f32 as *const _);
                    encoder.set_bytes(7, 4, &layer.scale as *const f32 as *const _);
                    encoder.set_bytes(8, 4, &layer.amplitude as *const f32 as *const _);
                    encoder.set_bytes(9, 4, &layer.seed as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &blend as *const u32 as *const _);
                });
                cb.commit();
                cb.wait_until_completed();
            }

            // Apply palette to the noise to get colored output
            let palette_data = encode_palette(palette);
            let num_stops = palette.stops.len() as u32;
            let palette_buf = device.new_buffer_with_data(
                palette_data.as_ptr() as *const _,
                (palette_data.len() * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            // If we had no SDF ops, the noise IS the image — apply palette to output_buf
            // If we had SDF ops, composite noise on top (additive blend of palette-mapped noise)
            let color_buf = if num_ops > 0 {
                // Allocate separate color buffer for noise color, then blend
                device.new_buffer(output_bytes, metal::MTLResourceOptions::StorageModeShared)
            } else {
                // Palette goes directly to output
                output_buf.clone()
            };

            let cb = compute.new_command_buffer();
            let w = self.config.width;
            let h = self.config.height;
            compute.dispatch_2d(&cb, self.ops.apply_palette_pipeline(), width, height, |encoder| {
                encoder.set_buffer(0, Some(&noise_buf), 0);
                encoder.set_buffer(1, Some(&palette_buf), 0);
                encoder.set_buffer(2, Some(&color_buf), 0);
                encoder.set_bytes(3, 4, &w as *const u32 as *const _);
                encoder.set_bytes(4, 4, &h as *const u32 as *const _);
                encoder.set_bytes(5, 4, &num_stops as *const u32 as *const _);
            });
            cb.commit();
            cb.wait_until_completed();

            // If we had both SDF and noise, blend them
            if num_ops > 0 {
                let out_ptr = output_buf.contents() as *mut f32;
                let color_ptr = color_buf.contents() as *const f32;
                let pixel_count = width * height * 3;
                unsafe {
                    for i in 0..pixel_count {
                        let sdf_val = *out_ptr.add(i);
                        let noise_val = *color_ptr.add(i);
                        // Screen blend: 1 - (1-a)(1-b)
                        *out_ptr.add(i) = 1.0 - (1.0 - sdf_val) * (1.0 - noise_val * 0.3);
                    }
                }
            }
        }

        // Read back result
        let out_ptr = output_buf.contents() as *const f32;
        let data = unsafe { std::slice::from_raw_parts(out_ptr, width * height * 3).to_vec() };

        Ok(FusionOutput::Image {
            data,
            width: self.config.width,
            height: self.config.height,
        })
    }

    /// Generate audio via Karplus-Strong waveguide synthesis.
    fn generate_audio(
        &self,
        waveguide: &WaveguideParams,
        excitation: &ExcitationParams,
        envelope: &ADSREnvelope,
        sequence: &NoteSequence,
    ) -> Result<FusionOutput> {
        let sample_rate = sequence.sample_rate;
        let compute = self.ops.compute();
        let device = compute.device().raw();

        if sequence.notes.is_empty() {
            return Ok(FusionOutput::Audio {
                data: vec![],
                sample_rate,
            });
        }

        // Calculate total duration
        let total_beats: f32 = sequence.notes.iter()
            .map(|n| n.start + n.duration)
            .fold(0.0f32, f32::max);
        let total_seconds = total_beats * 60.0 / sequence.bpm + envelope.release;
        let total_samples = (total_seconds * sample_rate as f32) as usize;

        let mut output = vec![0.0f32; total_samples];

        // Synthesize each note
        for (note_idx, note) in sequence.notes.iter().enumerate() {
            // Convert MIDI note to delay length
            let freq = 440.0 * (2.0f32).powf((note.midi_note as f32 - 69.0) / 12.0);
            let delay_len = (sample_rate as f32 / freq).round() as u32;
            if delay_len < 2 { continue; }

            let note_start = (note.start * 60.0 / sequence.bpm * sample_rate as f32) as usize;
            let note_duration = (note.duration * 60.0 / sequence.bpm * sample_rate as f32) as usize;
            let release_samples = (envelope.release * sample_rate as f32) as usize;
            let note_samples = note_duration + release_samples;
            if note_start + note_samples > total_samples { continue; }

            // Get per-string parameters (or use defaults)
            let damping = waveguide.damping.get(note_idx).copied()
                .unwrap_or_else(|| waveguide.damping.first().copied().unwrap_or(0.996));
            let brightness = waveguide.brightness.get(note_idx).copied()
                .unwrap_or_else(|| waveguide.brightness.first().copied().unwrap_or(0.5));

            // Allocate delay line + output buffers
            let delay_buf = device.new_buffer(
                (delay_len as u64) * 4,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let note_buf = device.new_buffer(
                (note_samples as u64) * 4,
                metal::MTLResourceOptions::StorageModeShared,
            );

            // Excite the string
            let excitation_type = match excitation.excitation_type {
                ExcitationType::Pluck => 0u32,
                ExcitationType::Bow => 1u32,
                ExcitationType::Strike => 2u32,
            };
            let seed = note_idx as u32 * 12345 + note.midi_note as u32;

            let cb = compute.new_command_buffer();
            compute.dispatch_1d(&cb, self.ops.ks_excite_pipeline(), delay_len as usize, |encoder| {
                encoder.set_buffer(0, Some(&delay_buf), 0);
                encoder.set_bytes(1, 4, &delay_len as *const u32 as *const _);
                encoder.set_bytes(2, 4, &excitation_type as *const u32 as *const _);
                encoder.set_bytes(3, 4, &excitation.position as *const f32 as *const _);
                let vel = note.velocity * excitation.velocity;
                encoder.set_bytes(4, 4, &vel as *const f32 as *const _);
                encoder.set_bytes(5, 4, &seed as *const u32 as *const _);
            });
            cb.commit();
            cb.wait_until_completed();

            // Synthesize (sequential — waveguide is inherently serial)
            let ns = note_samples as u32;
            let cb = compute.new_command_buffer();
            compute.dispatch_1d(&cb, self.ops.ks_synthesize_pipeline(), 1, |encoder| {
                encoder.set_buffer(0, Some(&delay_buf), 0);
                encoder.set_buffer(1, Some(&note_buf), 0);
                encoder.set_bytes(2, 4, &delay_len as *const u32 as *const _);
                encoder.set_bytes(3, 4, &ns as *const u32 as *const _);
                encoder.set_bytes(4, 4, &damping as *const f32 as *const _);
                encoder.set_bytes(5, 4, &brightness as *const f32 as *const _);
            });
            cb.commit();
            cb.wait_until_completed();

            // Mix note into output with ADSR envelope
            let note_ptr = note_buf.contents() as *const f32;
            let note_data = unsafe { std::slice::from_raw_parts(note_ptr, note_samples) };
            let attack_samples = (envelope.attack * sample_rate as f32) as usize;
            let decay_samples = (envelope.decay * sample_rate as f32) as usize;

            for (i, &sample) in note_data.iter().enumerate() {
                let out_idx = note_start + i;
                if out_idx >= total_samples { break; }

                // ADSR envelope
                let env = if i < attack_samples {
                    i as f32 / attack_samples.max(1) as f32
                } else if i < attack_samples + decay_samples {
                    let t = (i - attack_samples) as f32 / decay_samples.max(1) as f32;
                    1.0 - t * (1.0 - envelope.sustain)
                } else if i < note_duration {
                    envelope.sustain
                } else {
                    // Release
                    let t = (i - note_duration) as f32 / release_samples.max(1) as f32;
                    envelope.sustain * (1.0 - t).max(0.0)
                };

                output[out_idx] += sample * env;
            }
        }

        // Normalize if needed
        let peak = output.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        if peak > 1.0 {
            let inv = 1.0 / peak;
            for s in &mut output {
                *s *= inv;
            }
        }

        Ok(FusionOutput::Audio {
            data: output,
            sample_rate,
        })
    }
}

/// Encode SDF ops into flat f32 buffer (16 floats per op).
#[cfg(feature = "metal")]
fn encode_sdf_scene(ops: &[SDFOp]) -> Vec<f32> {
    let mut data = Vec::with_capacity(ops.len() * 16);
    for op in ops {
        let mut node = [0.0f32; 16];
        match op {
            SDFOp::Sphere { radius } => {
                node[0] = 0.0; // op type
                node[1] = *radius;
            }
            SDFOp::Box { half_extents } => {
                node[0] = 1.0;
                node[1] = half_extents[0];
                node[2] = half_extents[1];
                node[3] = half_extents[2];
            }
            SDFOp::Torus { major_radius, minor_radius } => {
                node[0] = 2.0;
                node[1] = *major_radius;
                node[2] = *minor_radius;
            }
            SDFOp::Plane { normal, offset } => {
                node[0] = 3.0;
                node[1] = normal[0];
                node[2] = normal[1];
                node[3] = normal[2];
                node[4] = *offset;
            }
            SDFOp::Cylinder { radius, height } => {
                node[0] = 4.0;
                node[1] = *radius;
                node[2] = *height;
            }
            SDFOp::Cone { radius, height } => {
                node[0] = 5.0;
                node[1] = *radius;
                node[2] = *height;
            }
            SDFOp::Translate { offset, child } => {
                node[0] = 10.0;
                node[1] = offset[0];
                node[2] = offset[1];
                node[3] = offset[2];
                node[7] = *child as f32;
            }
            SDFOp::Rotate { axis, angle, child } => {
                node[0] = 11.0;
                node[1] = axis[0];
                node[2] = axis[1];
                node[3] = axis[2];
                node[4] = *angle;
                node[7] = *child as f32;
            }
            SDFOp::Scale { factor, child } => {
                node[0] = 12.0;
                node[1] = *factor;
                node[7] = *child as f32;
            }
            SDFOp::SmoothUnion { a, b, k } => {
                node[0] = 20.0;
                node[7] = *a as f32;
                node[8] = *b as f32;
                node[9] = *k;
            }
            SDFOp::SmoothSubtraction { a, b, k } => {
                node[0] = 21.0;
                node[7] = *a as f32;
                node[8] = *b as f32;
                node[9] = *k;
            }
            SDFOp::SmoothIntersection { a, b, k } => {
                node[0] = 22.0;
                node[7] = *a as f32;
                node[8] = *b as f32;
                node[9] = *k;
            }
        }
        data.extend_from_slice(&node);
    }
    data
}

/// Encode a color palette into flat f32 buffer (4 floats per stop: position, r, g, b).
#[cfg(feature = "metal")]
fn encode_palette(palette: &ColorPalette) -> Vec<f32> {
    let mut data = Vec::with_capacity(palette.stops.len() * 4);
    for stop in &palette.stops {
        data.push(stop.position);
        data.push(stop.color.r);
        data.push(stop.color.g);
        data.push(stop.color.b);
    }
    data
}

// ============================================================================
// CPU MLP helpers (f32, used for GPV encoder — tiny network, CPU is fast)
// ============================================================================

/// SiLU activation: x * sigmoid(x)
fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Linear layer + SiLU: output = silu(W @ input + bias)
fn linear_silu(w: &[f32], bias: &[f32], input: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = input.len();
    let mut output = Vec::with_capacity(out_dim);
    for i in 0..out_dim {
        let mut sum = bias[i];
        let row = &w[i * in_dim..(i + 1) * in_dim];
        for (j, &x) in input.iter().enumerate() {
            sum += row[j] * x;
        }
        output.push(silu_f32(sum));
    }
    output
}

/// Linear layer (no activation): output = W @ input + bias
fn linear_forward(w: &[f32], bias: &[f32], input: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = input.len();
    let mut output = Vec::with_capacity(out_dim);
    for i in 0..out_dim {
        let mut sum = bias[i];
        let row = &w[i * in_dim..(i + 1) * in_dim];
        for (j, &x) in input.iter().enumerate() {
            sum += row[j] * x;
        }
        output.push(sum);
    }
    output
}

/// Simple deterministic RNG (xoshiro128+).
struct SimpleRng {
    state: [u32; 4],
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: [
                seed as u32,
                (seed >> 32) as u32,
                seed.wrapping_mul(6364136223846793005) as u32,
                (seed.wrapping_mul(6364136223846793005) >> 32) as u32,
            ],
        }
    }

    fn next_u32(&mut self) -> u32 {
        let result = self.state[0].wrapping_add(self.state[3]);
        let t = self.state[1] << 9;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(11);
        result
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Xavier uniform initialization.
fn xavier_init(rng: &mut SimpleRng, out_dim: usize, in_dim: usize) -> Vec<f32> {
    let limit = (6.0 / (in_dim + out_dim) as f32).sqrt();
    (0..out_dim * in_dim)
        .map(|_| rng.next_f32() * 2.0 * limit - limit)
        .collect()
}

/// Public wrapper for params_to_image_gpv (used by examples/tests).
pub fn params_to_image_gpv_pub(params: &[f32]) -> GPV {
    params_to_image_gpv(params)
}

/// Convert continuous parameter vector to an image GPV.
fn params_to_image_gpv(params: &[f32]) -> GPV {
    // Noise layer from params[0..9]
    let noise_type = match params.get(0).copied().unwrap_or(0.0) as u32 {
        1 => NoiseType::Simplex,
        2 => NoiseType::Worley,
        3 => NoiseType::Value,
        _ => NoiseType::Perlin,
    };
    let blend = match params.get(6).copied().unwrap_or(0.0) as u32 {
        1 => BlendMode::Add,
        2 => BlendMode::Multiply,
        3 => BlendMode::Screen,
        4 => BlendMode::Overlay,
        _ => BlendMode::Replace,
    };
    let noise_layers = vec![NoiseLayer {
        noise_type,
        octaves: (params.get(1).copied().unwrap_or(6.0).round() as u32).clamp(1, 12),
        lacunarity: params.get(2).copied().unwrap_or(2.1),
        persistence: params.get(3).copied().unwrap_or(0.5),
        scale: params.get(4).copied().unwrap_or(4.0).max(0.1),
        amplitude: params.get(5).copied().unwrap_or(1.0).max(0.0),
        blend,
        seed: params.get(7).copied().unwrap_or(42.0) as u32,
    }];

    // Palette from params[9..29] — 5 stops × 4 floats
    let mut stops = Vec::with_capacity(5);
    for s in 0..5 {
        let base = 9 + s * 4;
        stops.push(GradientStop {
            position: params.get(base).copied().unwrap_or(s as f32 / 4.0).clamp(0.0, 1.0),
            color: Color {
                r: params.get(base + 1).copied().unwrap_or(0.5).clamp(0.0, 1.0),
                g: params.get(base + 2).copied().unwrap_or(0.5).clamp(0.0, 1.0),
                b: params.get(base + 3).copied().unwrap_or(0.5).clamp(0.0, 1.0),
            },
        });
    }
    // Ensure stops are sorted by position
    stops.sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap_or(std::cmp::Ordering::Equal));

    // Camera from params[45..48]
    let cam_eye = [
        params.get(45).copied().unwrap_or(0.0),
        params.get(46).copied().unwrap_or(1.5),
        params.get(47).copied().unwrap_or(4.0),
    ];

    let composition = Composition {
        camera_eye: cam_eye,
        camera_target: [0.0, 0.0, 0.0],
        fov: params.get(48).copied().unwrap_or(1.0).max(0.1),
        max_distance: 50.0,
        max_steps: 128,
        background: Color {
            r: params.get(52).copied().unwrap_or(0.05).clamp(0.0, 1.0),
            g: params.get(53).copied().unwrap_or(0.05).clamp(0.0, 1.0),
            b: params.get(54).copied().unwrap_or(0.12).clamp(0.0, 1.0),
        },
        light_dir: [
            params.get(49).copied().unwrap_or(0.577),
            params.get(50).copied().unwrap_or(0.577),
            params.get(51).copied().unwrap_or(-0.577),
        ],
    };

    // For now, no SDF ops from continuous params (noise-only mode).
    // SDF requires discrete topology (number/type of ops) which needs
    // a separate discrete optimization or learned structure predictor.
    GPV {
        modality: GPVModality::Image,
        procedural_confidence: 1.0,
        params: GPVParams::Image {
            sdf_ops: vec![],
            noise_layers,
            palette: ColorPalette { stops },
            composition,
        },
    }
}

/// Convert continuous parameter vector to an audio GPV.
fn params_to_audio_gpv(params: &[f32]) -> GPV {
    let attack = params.get(0).copied().unwrap_or(0.002).max(0.0);
    let decay = params.get(1).copied().unwrap_or(0.05).max(0.0);
    let sustain = params.get(2).copied().unwrap_or(0.6).clamp(0.0, 1.0);
    let release = params.get(3).copied().unwrap_or(0.8).max(0.0);
    let damping = params.get(4).copied().unwrap_or(0.996).clamp(0.9, 1.0);
    let brightness = params.get(5).copied().unwrap_or(0.5).clamp(0.0, 1.0);
    let exc_pos = params.get(6).copied().unwrap_or(0.15).clamp(0.01, 0.99);
    let exc_vel = params.get(7).copied().unwrap_or(0.8).clamp(0.0, 1.0);
    let bpm = params.get(8).copied().unwrap_or(120.0).clamp(40.0, 300.0);

    // Decode up to 7 notes from params[9..30]
    let mut notes = Vec::new();
    for n in 0..7 {
        let base = 9 + n * 3;
        let midi = params.get(base).copied().unwrap_or(0.0);
        if midi < 20.0 { continue; } // Skip silent/invalid notes
        let vel = params.get(base + 1).copied().unwrap_or(0.8).clamp(0.0, 1.0);
        let dur = params.get(base + 2).copied().unwrap_or(1.0).max(0.1);
        notes.push(NoteEvent {
            midi_note: (midi.round() as u8).clamp(21, 108),
            velocity: vel,
            duration: dur,
            start: n as f32 * 0.5, // evenly spaced
        });
    }

    if notes.is_empty() {
        // Default single note
        notes.push(NoteEvent { midi_note: 60, velocity: 0.8, duration: 1.0, start: 0.0 });
    }

    GPV {
        modality: GPVModality::Audio,
        procedural_confidence: 1.0,
        params: GPVParams::Audio {
            waveguide: WaveguideParams {
                delay_samples: vec![],
                damping: vec![damping; notes.len()],
                brightness: vec![brightness; notes.len()],
            },
            excitation: ExcitationParams {
                excitation_type: ExcitationType::Pluck,
                position: exc_pos,
                velocity: exc_vel,
            },
            envelope: ADSREnvelope { attack, decay, sustain, release },
            sequence: NoteSequence { bpm, sample_rate: 44100, notes },
        },
    }
}
