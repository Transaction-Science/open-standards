//! Audio generation modality handler.

use super::{CacheStrategy, ModalityHandler, PrefetchPattern};
use crate::core::{Modality, Result};
use crate::tensor::Tensor;
use std::sync::Arc;

// ============================================================================
// Biquad Filter (Robert Bristow-Johnson Audio EQ Cookbook)
// ============================================================================

/// Biquad filter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiquadType {
    /// Low-pass filter: passes frequencies below cutoff.
    Lowpass,
    /// High-pass filter: passes frequencies above cutoff.
    Highpass,
    /// Band-pass filter: passes frequencies near center.
    Bandpass,
    /// Notch (band-reject) filter: rejects frequencies near center.
    Notch,
    /// Peak/bell EQ: boost or cut at center frequency.
    PeakEQ,
    /// Low-shelf: boost or cut below shelf frequency.
    LowShelf,
    /// High-shelf: boost or cut above shelf frequency.
    HighShelf,
}

/// Second-order IIR (biquad) filter using Transposed Direct Form II.
///
/// Implements all 7 standard filter types from the Audio EQ Cookbook.
/// Coefficients are pre-normalized (a0 = 1).
#[derive(Debug, Clone)]
pub struct BiquadFilter {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl BiquadFilter {
    /// Create a new biquad filter with the given parameters.
    ///
    /// - `filter_type`: Type of filter
    /// - `frequency`: Center/cutoff frequency in Hz
    /// - `sample_rate`: Sample rate in Hz
    /// - `q`: Quality factor (bandwidth control, typically 0.707 for Butterworth)
    /// - `gain_db`: Gain in dB (only used for PeakEQ, LowShelf, HighShelf)
    pub fn new(filter_type: BiquadType, frequency: f32, sample_rate: f32, q: f32, gain_db: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * frequency / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);
        let a_gain = 10.0f32.powf(gain_db / 40.0); // sqrt of voltage gain

        let (b0, b1, b2, a0, a1, a2) = match filter_type {
            BiquadType::Lowpass => {
                let b1 = 1.0 - cos_w0;
                let b0 = b1 / 2.0;
                let b2 = b0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::Highpass => {
                let b1 = -(1.0 + cos_w0);
                let b0 = (1.0 + cos_w0) / 2.0;
                let b2 = b0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::Bandpass => {
                let b0 = alpha;
                let b1 = 0.0;
                let b2 = -alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::Notch => {
                let b0 = 1.0;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::PeakEQ => {
                let b0 = 1.0 + alpha * a_gain;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 - alpha * a_gain;
                let a0 = 1.0 + alpha / a_gain;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha / a_gain;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::LowShelf => {
                let two_sqrt_a_alpha = 2.0 * a_gain.sqrt() * alpha;
                let b0 = a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * cos_w0 + two_sqrt_a_alpha);
                let b1 = 2.0 * a_gain * ((a_gain - 1.0) - (a_gain + 1.0) * cos_w0);
                let b2 = a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * cos_w0 - two_sqrt_a_alpha);
                let a0 = (a_gain + 1.0) + (a_gain - 1.0) * cos_w0 + two_sqrt_a_alpha;
                let a1 = -2.0 * ((a_gain - 1.0) + (a_gain + 1.0) * cos_w0);
                let a2 = (a_gain + 1.0) + (a_gain - 1.0) * cos_w0 - two_sqrt_a_alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::HighShelf => {
                let two_sqrt_a_alpha = 2.0 * a_gain.sqrt() * alpha;
                let b0 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_w0 + two_sqrt_a_alpha);
                let b1 = -2.0 * a_gain * ((a_gain - 1.0) + (a_gain + 1.0) * cos_w0);
                let b2 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_w0 - two_sqrt_a_alpha);
                let a0 = (a_gain + 1.0) - (a_gain - 1.0) * cos_w0 + two_sqrt_a_alpha;
                let a1 = 2.0 * ((a_gain - 1.0) - (a_gain + 1.0) * cos_w0);
                let a2 = (a_gain + 1.0) - (a_gain - 1.0) * cos_w0 - two_sqrt_a_alpha;
                (b0, b1, b2, a0, a1, a2)
            }
        };

        // Normalize by a0
        let inv_a0 = 1.0 / a0;
        Self {
            b0: b0 * inv_a0,
            b1: b1 * inv_a0,
            b2: b2 * inv_a0,
            a1: a1 * inv_a0,
            a2: a2 * inv_a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Process a single sample using Transposed Direct Form II.
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Process a block of samples.
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        for (x, y) in input.iter().zip(output.iter_mut()) {
            *y = self.process_sample(*x);
        }
    }

    /// Reset the filter state (delay elements).
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

// ============================================================================
// MFCC Feature Extraction
// ============================================================================

/// Mel-Frequency Cepstral Coefficient extractor.
///
/// Pipeline: Hanning window → DFT → power spectrum → mel filterbank → log → DCT → MFCCs.
pub struct MfccExtractor {
    sample_rate: u32,
    n_fft: usize,
    hop_size: usize,
    n_mels: usize,
    n_mfcc: usize,
    mel_filterbank: Vec<f32>,  // [n_mels * (n_fft/2+1)]
    dct_matrix: Vec<f32>,      // [n_mfcc * n_mels]
    window: Vec<f32>,          // Hanning window [n_fft]
}

impl MfccExtractor {
    /// Create a new MFCC extractor with the given parameters.
    pub fn new(sample_rate: u32, n_fft: usize, hop_size: usize, n_mels: usize, n_mfcc: usize) -> Self {
        let n_freqs = n_fft / 2 + 1;

        // Hanning window
        let window: Vec<f32> = (0..n_fft)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos()))
            .collect();

        // Mel filterbank
        let f_min = 0.0f32;
        let f_max = sample_rate as f32 / 2.0;
        let mel_min = hz_to_mel(f_min);
        let mel_max = hz_to_mel(f_max);

        // n_mels + 2 mel points (edges of filterbank)
        let mel_points: Vec<f32> = (0..n_mels + 2)
            .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
            .collect();

        // Convert Hz to FFT bin indices
        let bin_points: Vec<f32> = mel_points.iter()
            .map(|&hz| hz * n_fft as f32 / sample_rate as f32)
            .collect();

        let mut mel_filterbank = vec![0.0f32; n_mels * n_freqs];
        for m in 0..n_mels {
            let f_left = bin_points[m];
            let f_center = bin_points[m + 1];
            let f_right = bin_points[m + 2];

            for k in 0..n_freqs {
                let kf = k as f32;
                if kf >= f_left && kf <= f_center && f_center > f_left {
                    mel_filterbank[m * n_freqs + k] = (kf - f_left) / (f_center - f_left);
                } else if kf > f_center && kf <= f_right && f_right > f_center {
                    mel_filterbank[m * n_freqs + k] = (f_right - kf) / (f_right - f_center);
                }
            }
        }

        // DCT-II matrix
        let mut dct_matrix = vec![0.0f32; n_mfcc * n_mels];
        for i in 0..n_mfcc {
            for j in 0..n_mels {
                dct_matrix[i * n_mels + j] = (std::f32::consts::PI * i as f32 * (j as f32 + 0.5) / n_mels as f32).cos();
            }
        }

        Self {
            sample_rate,
            n_fft,
            hop_size,
            n_mels,
            n_mfcc,
            mel_filterbank,
            dct_matrix,
            window,
        }
    }

    /// Extract MFCC features from audio samples.
    /// Returns a vec of frames, each containing `n_mfcc` coefficients.
    pub fn extract(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        if audio.len() < self.n_fft {
            return Vec::new();
        }

        let n_frames = (audio.len() - self.n_fft) / self.hop_size + 1;
        let n_freqs = self.n_fft / 2 + 1;
        let mut result = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_size;
            let frame = &audio[start..start + self.n_fft];

            // Step 1: Windowed DFT → power spectrum
            let power = self.compute_power_spectrum(frame);

            // Step 2: Apply mel filterbank
            let mut mel_energies = vec![0.0f32; self.n_mels];
            for m in 0..self.n_mels {
                let mut sum = 0.0f32;
                for k in 0..n_freqs {
                    sum += self.mel_filterbank[m * n_freqs + k] * power[k];
                }
                mel_energies[m] = (sum + 1e-10).ln(); // Log mel energies
            }

            // Step 3: DCT → MFCCs
            let mut mfcc = vec![0.0f32; self.n_mfcc];
            for i in 0..self.n_mfcc {
                let mut sum = 0.0f32;
                for j in 0..self.n_mels {
                    sum += self.dct_matrix[i * self.n_mels + j] * mel_energies[j];
                }
                mfcc[i] = sum;
            }

            result.push(mfcc);
        }

        result
    }

    /// Compute power spectrum from a windowed frame using DFT.
    fn compute_power_spectrum(&self, frame: &[f32]) -> Vec<f32> {
        let n = self.n_fft;
        let n_freqs = n / 2 + 1;
        let mut power = vec![0.0f32; n_freqs];

        for k in 0..n_freqs {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for t in 0..n {
                let windowed = frame[t] * self.window[t];
                let angle = -2.0 * std::f32::consts::PI * k as f32 * t as f32 / n as f32;
                real += windowed * angle.cos();
                imag += windowed * angle.sin();
            }
            power[k] = (real * real + imag * imag) / n as f32;
        }

        power
    }
}

/// Compute delta (temporal derivative) features using a regression window.
pub fn compute_deltas(features: &[Vec<f32>], width: usize) -> Vec<Vec<f32>> {
    let n_frames = features.len();
    if n_frames == 0 {
        return Vec::new();
    }

    let n_coeffs = features[0].len();
    let mut deltas = vec![vec![0.0f32; n_coeffs]; n_frames];

    let denom: f32 = (1..=width).map(|n| 2.0 * (n * n) as f32).sum();
    if denom < 1e-10 {
        return deltas;
    }

    for t in 0..n_frames {
        for c in 0..n_coeffs {
            let mut sum = 0.0f32;
            for n in 1..=width {
                let t_plus = (t + n).min(n_frames - 1);
                let t_minus = if t >= n { t - n } else { 0 };
                sum += n as f32 * (features[t_plus][c] - features[t_minus][c]);
            }
            deltas[t][c] = sum / denom;
        }
    }

    deltas
}

/// Convert frequency in Hz to mel scale.
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale to frequency in Hz.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

// ============================================================================
// Voice Activity Detection
// ============================================================================

/// Voice Activity Detector using energy, zero-crossing rate, and hangover logic.
pub struct VoiceActivityDetector {
    sample_rate: u32,
    frame_size: usize,
    energy_threshold: f32,
    zcr_threshold: f32,
    hangover_frames: usize,
}

impl VoiceActivityDetector {
    /// Create a new VAD with sensible defaults.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            frame_size: (sample_rate as f32 * 0.03) as usize, // 30ms frames
            energy_threshold: 0.01,
            zcr_threshold: 0.5,
            hangover_frames: 10,
        }
    }

    /// Detect voice activity per frame.
    /// Returns a boolean for each frame indicating whether speech is detected.
    pub fn detect(&self, audio: &[f32]) -> Vec<bool> {
        if audio.len() < self.frame_size {
            return Vec::new();
        }

        let n_frames = audio.len() / self.frame_size;
        let mut activity = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let start = i * self.frame_size;
            let frame = &audio[start..start + self.frame_size];

            let energy = Self::frame_energy(frame);
            let zcr = Self::frame_zcr(frame);

            // Speech: high energy AND reasonable ZCR (not noise)
            let is_speech = energy > self.energy_threshold && zcr < self.zcr_threshold;
            activity.push(is_speech);
        }

        // Apply hangover: bridge brief silences
        self.apply_hangover(&mut activity);

        activity
    }

    /// Detect voice activity segments as sample ranges.
    pub fn detect_segments(&self, audio: &[f32]) -> Vec<(usize, usize)> {
        let activity = self.detect(audio);
        let mut segments = Vec::new();
        let mut in_segment = false;
        let mut start = 0usize;

        for (i, &active) in activity.iter().enumerate() {
            if active && !in_segment {
                start = i * self.frame_size;
                in_segment = true;
            } else if !active && in_segment {
                let end = i * self.frame_size;
                segments.push((start, end.min(audio.len())));
                in_segment = false;
            }
        }

        if in_segment {
            segments.push((start, audio.len()));
        }

        segments
    }

    /// Compute RMS energy of a frame.
    pub fn frame_energy(frame: &[f32]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = frame.iter().map(|&x| x * x).sum();
        (sum_sq / frame.len() as f32).sqrt()
    }

    /// Compute zero-crossing rate of a frame.
    pub fn frame_zcr(frame: &[f32]) -> f32 {
        if frame.len() < 2 {
            return 0.0;
        }
        let crossings: usize = frame.windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        crossings as f32 / (frame.len() - 1) as f32
    }

    /// Compute spectral centroid of a frame.
    pub fn spectral_centroid(frame: &[f32], sample_rate: u32) -> f32 {
        let n = frame.len();
        if n == 0 {
            return 0.0;
        }

        let n_freqs = n / 2 + 1;
        let mut magnitude_sum = 0.0f32;
        let mut weighted_sum = 0.0f32;

        for k in 0..n_freqs {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for t in 0..n {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * t as f32 / n as f32;
                real += frame[t] * angle.cos();
                imag += frame[t] * angle.sin();
            }
            let mag = (real * real + imag * imag).sqrt();
            let freq = k as f32 * sample_rate as f32 / n as f32;
            weighted_sum += freq * mag;
            magnitude_sum += mag;
        }

        if magnitude_sum > 1e-10 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }

    /// Apply hangover smoothing: bridge gaps shorter than `hangover_frames`.
    fn apply_hangover(&self, activity: &mut [bool]) {
        let n = activity.len();
        if n == 0 {
            return;
        }

        // Forward pass: count consecutive silent frames
        let mut silence_count = 0usize;
        let mut last_speech = false;

        for i in 0..n {
            if activity[i] {
                // If we just came out of a short silence, fill it in
                if last_speech && silence_count > 0 && silence_count <= self.hangover_frames {
                    for j in (i - silence_count)..i {
                        activity[j] = true;
                    }
                }
                silence_count = 0;
                last_speech = true;
            } else {
                silence_count += 1;
            }
        }
    }
}

// ============================================================================
// Physical Modeling Synthesis
// ============================================================================

/// Karplus-Strong plucked string synthesis.
///
/// A noise burst is injected into a circular delay line. Each sample cycle,
/// the output is the averaged pair of adjacent delay samples, multiplied by
/// a feedback coefficient. The delay length determines pitch; the averaging
/// lowpass produces the characteristic decaying timbre of a plucked string.
pub struct PluckedString {
    sample_rate: u32,
    delay_line: Vec<f32>,
    write_pos: usize,
    feedback: f32,
    pick_position: f32,
}

impl PluckedString {
    /// Create a new plucked string at the given frequency.
    /// The delay line is pre-filled with a noise burst (excitation).
    pub fn new(frequency: f32, sample_rate: u32) -> Self {
        let delay_len = (sample_rate as f32 / frequency).round().max(2.0) as usize;
        let mut delay_line = vec![0.0f32; delay_len];

        // Fill with deterministic pseudo-random noise burst
        let mut seed: u32 = 42;
        for sample in delay_line.iter_mut() {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *sample = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }

        Self {
            sample_rate,
            delay_line,
            write_pos: 0,
            feedback: 0.996,
            pick_position: 0.5,
        }
    }

    /// Re-excite the string with a noise burst.
    pub fn pluck(&mut self) {
        let mut seed: u32 = 137;
        for sample in self.delay_line.iter_mut() {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *sample = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
    }

    /// Set damping (0.0 = no damping, 1.0 = maximum damping).
    pub fn set_damping(&mut self, damping: f32) {
        self.feedback = 1.0 - damping.clamp(0.0, 0.99);
    }

    /// Set pick position (0.0-1.0, affects harmonic content).
    pub fn set_pick_position(&mut self, position: f32) {
        self.pick_position = position.clamp(0.01, 0.99);
    }

    /// Process one sample: read from delay, apply averaging lowpass, write back.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        let len = self.delay_line.len();
        let read_pos = self.write_pos;
        let next_pos = (read_pos + 1) % len;

        // Averaging lowpass: y = 0.5 * (delay[n] + delay[n+1])
        let filtered = 0.5 * (self.delay_line[read_pos] + self.delay_line[next_pos]);

        // Write back with feedback (passive loop, comb is output-only)
        self.delay_line[self.write_pos] = filtered * self.feedback;
        self.write_pos = (self.write_pos + 1) % len;

        // Pick position comb filter applied to output only (not in feedback loop)
        let pick_offset = ((len as f32 * self.pick_position) as usize).max(1).min(len - 1);
        let pick_idx = (read_pos + pick_offset) % len;
        filtered - 0.1 * self.delay_line[pick_idx]
    }

    /// Generate a block of samples.
    pub fn generate(&mut self, num_samples: usize) -> Vec<f32> {
        let mut output = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            output.push(self.tick());
        }
        output
    }
}

/// Bowed string synthesis using bidirectional waveguide.
///
/// Two delay lines model traveling waves in each direction. A nonlinear
/// friction function (Stribeck model) at the bow point injects energy from
/// the bow velocity into both waves. A body resonance filter colors the output.
pub struct BowedString {
    sample_rate: u32,
    delay_upper: Vec<f32>,
    delay_lower: Vec<f32>,
    pos_upper: usize,
    pos_lower: usize,
    bow_velocity: f32,
    bow_force: f32,
    body_filter: BiquadFilter,
    loss: f32,
}

impl BowedString {
    /// Create a new bowed string at the given frequency.
    pub fn new(frequency: f32, sample_rate: u32) -> Self {
        let total_delay = (sample_rate as f32 / frequency).round().max(4.0) as usize;
        // Split delay at bow position (approximately 1/3 from nut)
        let upper_len = (total_delay * 2 / 3).max(2);
        let lower_len = (total_delay - upper_len).max(2);

        // Body resonance filter: broad peak at ~200Hz
        let body_filter = BiquadFilter::new(BiquadType::PeakEQ, 200.0, sample_rate as f32, 0.5, 6.0);

        Self {
            sample_rate,
            delay_upper: vec![0.0; upper_len],
            delay_lower: vec![0.0; lower_len],
            pos_upper: 0,
            pos_lower: 0,
            bow_velocity: 0.0,
            bow_force: 0.0,
            body_filter,
            loss: 0.995,
        }
    }

    /// Set bow parameters.
    pub fn set_bow(&mut self, velocity: f32, force: f32) {
        self.bow_velocity = velocity;
        self.bow_force = force.clamp(0.0, 1.0);
    }

    /// Process one sample.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        // Read from both delay lines at bow position
        let from_upper = self.delay_upper[self.pos_upper];
        let from_lower = self.delay_lower[self.pos_lower];

        // Velocity at bow point (sum of incoming waves)
        let v_string = from_upper + from_lower;

        // Velocity difference between bow and string
        let v_diff = self.bow_velocity - v_string;

        // Stribeck friction model: f = force * v_diff * exp(-v_diff^2 * gain)
        let friction = self.bow_force * v_diff * (-v_diff * v_diff * 100.0).exp();

        // Inject friction into both delay lines
        let to_upper = from_lower + friction;
        let to_lower = from_upper + friction;

        // Write with loss (simulates string damping at endpoints)
        self.delay_upper[self.pos_upper] = to_upper * self.loss;
        self.delay_lower[self.pos_lower] = to_lower * self.loss;

        // Advance positions
        self.pos_upper = (self.pos_upper + 1) % self.delay_upper.len();
        self.pos_lower = (self.pos_lower + 1) % self.delay_lower.len();

        // Output: bridge pickup = end of lower delay, filtered by body resonance
        let raw = from_lower;
        self.body_filter.process_sample(raw)
    }

    /// Generate a block of samples.
    pub fn generate(&mut self, num_samples: usize) -> Vec<f32> {
        let mut output = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            output.push(self.tick());
        }
        output
    }
}

/// Waveguide tube resonator.
///
/// Models a cylindrical acoustic tube (e.g., flute, organ pipe) using
/// bidirectional delay lines with reflection at endpoints. Open-end
/// reflection inverts the wave; closed-end reflects in phase.
pub struct TubeResonator {
    sample_rate: u32,
    delay_forward: Vec<f32>,
    delay_backward: Vec<f32>,
    pos_forward: usize,
    pos_backward: usize,
    end_reflection: f32,
    loss_factor: f32,
    pending_excitation: f32,
}

impl TubeResonator {
    /// Create a new tube resonator.
    /// `open_end`: true for open tube (flute), false for closed tube (clarinet).
    pub fn new(frequency: f32, sample_rate: u32, open_end: bool) -> Self {
        // Open tube: half-wavelength resonance (L = v/2f)
        // Closed tube: quarter-wavelength resonance (L = v/4f)
        let delay_len = if open_end {
            (sample_rate as f32 / (2.0 * frequency)).round().max(2.0) as usize
        } else {
            (sample_rate as f32 / (4.0 * frequency)).round().max(2.0) as usize
        };

        let end_reflection = if open_end { -1.0 } else { 1.0 };

        Self {
            sample_rate,
            delay_forward: vec![0.0; delay_len],
            delay_backward: vec![0.0; delay_len],
            pos_forward: 0,
            pos_backward: 0,
            end_reflection,
            loss_factor: 0.995,
            pending_excitation: 0.0,
        }
    }

    /// Inject excitation at the input end (buffered until next tick).
    pub fn excite(&mut self, sample: f32) {
        self.pending_excitation += sample;
    }

    /// Process one sample.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        let fwd_len = self.delay_forward.len();
        let bwd_len = self.delay_backward.len();

        // Read from end of forward delay (reaching the far end)
        let fwd_end = (self.pos_forward + fwd_len - 1) % fwd_len;
        let arriving_forward = self.delay_forward[fwd_end];

        // Read from end of backward delay (reaching the input end)
        let bwd_end = (self.pos_backward + bwd_len - 1) % bwd_len;
        let arriving_backward = self.delay_backward[bwd_end];

        // Reflection at far end: reflect with sign and loss
        let reflected_at_far = arriving_forward * self.end_reflection * self.loss_factor;
        self.delay_backward[self.pos_backward] = reflected_at_far;

        // Reflection at input end + pending excitation
        let reflected_at_input = arriving_backward * self.loss_factor;
        self.delay_forward[self.pos_forward] = reflected_at_input + self.pending_excitation;
        self.pending_excitation = 0.0;

        // Output: radiated sound at the open end
        let output = arriving_forward * (1.0 - self.end_reflection.abs() * 0.5);

        // Advance positions
        self.pos_forward = (self.pos_forward + 1) % fwd_len;
        self.pos_backward = (self.pos_backward + 1) % bwd_len;

        output
    }

    /// Generate output with external excitation signal.
    pub fn generate(&mut self, excitation: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(excitation.len());
        for &ex in excitation {
            self.excite(ex);
            output.push(self.tick());
        }
        output
    }
}

// ============================================================================
// Granular Synthesis
// ============================================================================

/// Individual grain in a granular cloud.
struct Grain {
    /// Position in source buffer (fractional sample index).
    source_position: f32,
    /// Playback rate for this grain.
    pitch: f32,
    /// Current read phase within the grain (incremented by pitch each sample).
    phase: f32,
    /// Total grain size in samples.
    grain_size: usize,
    /// Samples remaining before this grain expires.
    remaining: usize,
}

/// Granular synthesis engine.
///
/// Spawns overlapping grains from a source buffer at a configurable density.
/// Each grain reads from the source with independent position and pitch jitter,
/// windowed by a Hanning envelope for smooth overlap-add.
pub struct GrainCloud {
    source: Vec<f32>,
    sample_rate: u32,
    density_hz: f32,
    grain_size: usize,
    position: f32,
    position_jitter: f32,
    pitch: f32,
    pitch_jitter: f32,
    envelope: Vec<f32>,
    active_grains: Vec<Grain>,
    spawn_counter: f32,
    rng_state: u32,
}

impl GrainCloud {
    /// Create a new grain cloud from a source buffer.
    pub fn new(source: Vec<f32>, sample_rate: u32, grain_size: usize, density_hz: f32) -> Self {
        // Pre-compute Hanning envelope
        let envelope: Vec<f32> = (0..grain_size)
            .map(|n| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / grain_size as f32).cos())
            })
            .collect();

        Self {
            source,
            sample_rate,
            density_hz,
            grain_size,
            position: 0.5,
            position_jitter: 0.05,
            pitch: 1.0,
            pitch_jitter: 0.0,
            envelope,
            active_grains: Vec::new(),
            spawn_counter: 0.0,
            rng_state: 12345,
        }
    }

    /// Set playback position in source (0.0-1.0).
    pub fn set_position(&mut self, position: f32) {
        self.position = position.clamp(0.0, 1.0);
    }

    /// Set grain density (grains per second).
    pub fn set_density(&mut self, hz: f32) {
        self.density_hz = hz.max(0.1);
    }

    /// Set pitch and jitter.
    pub fn set_pitch(&mut self, pitch: f32, jitter: f32) {
        self.pitch = pitch.max(0.01);
        self.pitch_jitter = jitter.max(0.0);
    }

    /// Simple deterministic pseudo-random number in [-1, 1].
    fn next_random(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    /// Spawn a new grain with jittered parameters.
    fn spawn_grain(&mut self) {
        let pos_offset = self.next_random() * self.position_jitter;
        let source_pos = (self.position + pos_offset).clamp(0.0, 1.0)
            * (self.source.len().saturating_sub(1)) as f32;

        let pitch_offset = self.next_random() * self.pitch_jitter;
        let grain_pitch = (self.pitch + pitch_offset).max(0.01);

        self.active_grains.push(Grain {
            source_position: source_pos,
            pitch: grain_pitch,
            phase: 0.0,
            grain_size: self.grain_size,
            remaining: self.grain_size,
        });
    }

    /// Process one sample.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        // Check if we should spawn a new grain
        self.spawn_counter += self.density_hz / self.sample_rate as f32;
        while self.spawn_counter >= 1.0 {
            self.spawn_counter -= 1.0;
            self.spawn_grain();
        }

        // Accumulate contributions from all active grains
        let mut output = 0.0f32;
        let source_len = self.source.len();

        let mut i = 0;
        while i < self.active_grains.len() {
            let grain = &mut self.active_grains[i];

            // Read source with linear interpolation
            let read_pos = grain.source_position + grain.phase;
            let idx0 = read_pos as usize;
            let frac = read_pos - idx0 as f32;

            let sample = if idx0 < source_len {
                let s0 = self.source[idx0];
                let s1 = if idx0 + 1 < source_len { self.source[idx0 + 1] } else { s0 };
                s0 + frac * (s1 - s0)
            } else {
                0.0
            };

            // Apply envelope
            let env_idx = grain.grain_size - grain.remaining;
            let env = if env_idx < self.envelope.len() { self.envelope[env_idx] } else { 0.0 };
            output += sample * env;

            // Advance grain
            grain.phase += grain.pitch;
            grain.remaining -= 1;

            if grain.remaining == 0 {
                self.active_grains.swap_remove(i);
            } else {
                i += 1;
            }
        }

        output
    }

    /// Generate a block of samples.
    pub fn generate(&mut self, num_samples: usize) -> Vec<f32> {
        let mut output = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            output.push(self.tick());
        }
        output
    }

    /// Get the number of currently active grains.
    pub fn active_grain_count(&self) -> usize {
        self.active_grains.len()
    }
}

/// Audio input configuration.
#[derive(Debug, Clone)]
pub struct AudioInput {
    /// Audio source
    pub source: AudioSource,
    /// Output configuration
    pub output: AudioOutputConfig,
}

/// Audio source.
#[derive(Debug, Clone)]
pub enum AudioSource {
    /// Generate from text (TTS, music generation)
    Text(alloc::string::String),
    /// Audio to audio (voice conversion, enhancement)
    Audio(Tensor),
}

/// Audio output configuration.
#[derive(Debug, Clone)]
pub struct AudioOutputConfig {
    /// Sample rate
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration_seconds: f32,
    /// Number of channels
    pub channels: u32,
}

impl Default for AudioOutputConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            duration_seconds: 5.0,
            channels: 2,
        }
    }
}

/// Audio output.
#[derive(Debug, Default)]
pub struct AudioOutput {
    /// Generated audio waveform
    pub waveform: Option<Tensor>,
    /// Statistics
    pub stats: AudioStats,
}

/// Audio statistics.
#[derive(Debug, Default, Clone)]
pub struct AudioStats {
    /// Time to first audio (ms)
    pub time_to_first_ms: f32,
    /// Real-time factor (>1 means faster than real-time)
    pub realtime_factor: f32,
}

/// Audio handler.
pub struct AudioHandler {
    /// Chunk duration in ms
    chunk_duration_ms: usize,
    /// Optional inference pipeline for model-backed generation
    pipeline: Option<Arc<crate::inference::AudioPipeline>>,
}

impl core::fmt::Debug for AudioHandler {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AudioHandler")
            .field("chunk_duration_ms", &self.chunk_duration_ms)
            .field("pipeline", &self.pipeline.as_ref().map(|_| "AudioPipeline"))
            .finish()
    }
}

impl AudioHandler {
    /// Create a new audio handler.
    pub fn new() -> Self {
        Self {
            chunk_duration_ms: 100,
            pipeline: None,
        }
    }

    /// Set the inference pipeline for model-backed audio generation.
    pub fn set_pipeline(&mut self, pipeline: Arc<crate::inference::AudioPipeline>) {
        self.pipeline = Some(pipeline);
    }

    /// Generate audio from input.
    ///
    /// Pipeline:
    /// 1. Encode the input (text → token embeddings, or audio → mel spectrogram)
    /// 2. Generate mel spectrogram via diffusion/autoregressive model
    /// 3. Decode mel spectrogram to waveform via vocoder (Griffin-Lim or HiFi-GAN)
    pub async fn generate(&self, input: AudioInput) -> Result<AudioOutput> {
        let start = std::time::Instant::now();

        let sample_rate = input.output.sample_rate;
        let duration = input.output.duration_seconds;
        let channels = input.output.channels as usize;
        let total_samples = (sample_rate as f32 * duration) as usize;

        // Step 1: Encode input to conditioning
        let conditioning = match &input.source {
            AudioSource::Text(text) => self.encode_text(text)?,
            AudioSource::Audio(audio_tensor) => self.encode_audio(audio_tensor)?,
        };

        let time_to_first = start.elapsed().as_secs_f32() * 1000.0;

        // Step 2: Generate mel spectrogram
        // Parameters: 80 mel bins, hop_size = 256
        let n_mels = 80usize;
        let hop_size = 256usize;
        let n_frames = total_samples / hop_size;

        let mel_data = self.generate_mel_spectrogram(&conditioning, n_mels, n_frames)?;

        // Step 3: Vocoder - convert mel spectrogram to waveform
        // Using Griffin-Lim approximation
        let waveform_data = self.mel_to_waveform(&mel_data, n_mels, n_frames, hop_size, channels)?;

        let waveform_shape = crate::core::Shape::from([1, channels, total_samples]);
        let waveform = Tensor::from_slice(
            &waveform_data,
            waveform_shape,
            crate::tensor::DType::F32,
            crate::hal::DeviceId::cpu(),
        )?;

        let total_time = start.elapsed().as_secs_f32() * 1000.0;
        let realtime_factor = if total_time > 0.0 {
            (duration * 1000.0) / total_time
        } else {
            0.0
        };

        Ok(AudioOutput {
            waveform: Some(waveform),
            stats: AudioStats {
                time_to_first_ms: time_to_first,
                realtime_factor,
            },
        })
    }

    /// Stream audio chunks as they are generated.
    ///
    /// Generates audio in fixed-duration chunks and streams them,
    /// enabling real-time playback during generation.
    pub fn generate_stream(
        &self,
        input: AudioInput,
    ) -> crate::runtime::StreamingOutput<AudioChunk> {
        let (output, sender) = crate::runtime::stream::StreamBuilder::new()
            .buffer_size(32)
            .build();

        let chunk_duration_ms = self.chunk_duration_ms;

        tokio::spawn(async move {
            let sample_rate = input.output.sample_rate;
            let duration = input.output.duration_seconds;
            let channels = input.output.channels as usize;
            let samples_per_chunk = (sample_rate as usize * chunk_duration_ms) / 1000;
            let total_samples = (sample_rate as f32 * duration) as usize;
            let num_chunks = (total_samples + samples_per_chunk - 1) / samples_per_chunk;

            let n_mels = 80usize;
            let hop_size = 256usize;

            // Generate mel spectrogram for the full duration
            let n_frames = total_samples / hop_size;
            let mel_data = vec![0.0f32; n_mels * n_frames];

            // Without loaded model weights, mel_data remains silence (zeros).
            // With a loaded acoustic model, this would run the forward pass
            // to predict mel frames from the text/audio conditioning.
            let _ = &mel_data; // silence unused warning

            // Stream chunks
            let mut timestamp_ms = 0u64;
            for chunk_idx in 0..num_chunks {
                if sender.is_cancelled() {
                    break;
                }

                let start_sample = chunk_idx * samples_per_chunk;
                let end_sample = (start_sample + samples_per_chunk).min(total_samples);
                let chunk_samples = end_sample - start_sample;

                // Extract mel frames for this chunk
                let start_frame = start_sample / hop_size;
                let end_frame = (end_sample / hop_size).min(n_frames);
                let chunk_frames = end_frame - start_frame;

                // Griffin-Lim vocoder for this chunk's mel frames
                let n_fft = hop_size * 4;
                let n_iter = 32;
                let mel_ref = &mel_data;
                let chunk_mel_data: Vec<f32> = (0..n_mels)
                    .flat_map(|m| {
                        let mel = mel_ref;
                        (start_frame..end_frame)
                            .map(move |f| mel.get(m * n_frames + f).copied().unwrap_or(0.0))
                    })
                    .collect();

                let mono = AudioHandler::griffin_lim(
                    &chunk_mel_data,
                    n_mels,
                    chunk_frames.max(1),
                    sample_rate as usize,
                    hop_size,
                    n_fft,
                    n_iter,
                );

                let mut chunk_data = vec![0.0f32; channels * chunk_samples];
                for ch in 0..channels {
                    for s in 0..chunk_samples {
                        chunk_data[ch * chunk_samples + s] =
                            mono.get(s).copied().unwrap_or(0.0);
                    }
                }

                let chunk_shape = crate::core::Shape::from([1, channels, chunk_samples]);
                let samples_tensor = match Tensor::from_slice(
                    &chunk_data,
                    chunk_shape,
                    crate::tensor::DType::F32,
                    crate::hal::DeviceId::cpu(),
                ) {
                    Ok(t) => t,
                    Err(e) => { let _ = sender.send_error(e).await; return; }
                };

                let chunk = AudioChunk {
                    samples: samples_tensor,
                    timestamp_ms,
                    duration_ms: chunk_duration_ms as u32,
                };

                if sender.send(chunk).await.is_err() {
                    break;
                }

                timestamp_ms += chunk_duration_ms as u64;
            }

            sender.complete();
        });

        output
    }

    /// Encode text input to conditioning embeddings.
    fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        // Character-level embedding for TTS
        // Each character maps to an embedding vector
        let embed_dim = 256;
        let mut embedding = vec![0.0f32; embed_dim];

        for (i, ch) in text.chars().enumerate() {
            // Simple learned-like embedding: distribute character info across dimensions
            let char_val = ch as u32 as f32;

            // Sinusoidal position encoding + character value
            for d in 0..embed_dim {
                let freq = ((d as f32) / embed_dim as f32 * 10.0).exp();
                let pos_enc = ((i as f32) / freq).sin();
                let char_enc = ((char_val + d as f32) / 128.0).sin();
                embedding[d] += (pos_enc * 0.5 + char_enc * 0.5) / (text.len() as f32).sqrt();
            }
        }

        Ok(embedding)
    }

    /// Encode audio input to conditioning features.
    fn encode_audio(&self, audio: &Tensor) -> Result<Vec<f32>> {
        // Extract features from input audio (simplified)
        let data: Vec<f32> = audio.to_vec()?;
        let embed_dim = 256;
        let mut embedding = vec![0.0f32; embed_dim];

        // Downsample audio to embedding
        let step = data.len().max(1) / embed_dim.max(1);
        for i in 0..embed_dim {
            let idx = (i * step).min(data.len().saturating_sub(1));
            embedding[i] = data.get(idx).copied().unwrap_or(0.0);
        }

        Ok(embedding)
    }

    /// Generate mel spectrogram from text embedding.
    ///
    /// Mel spectrogram prediction requires loaded acoustic model weights.
    /// Without weights, generate silence (zeros).
    ///
    /// With loaded weights, this would be:
    /// 1. Text encoder: characters/phonemes -> hidden states
    /// 2. Duration predictor: hidden states -> frame durations
    /// 3. Length regulator: expand hidden states to frame-level
    /// 4. Mel decoder: frame-level features -> mel spectrogram
    fn generate_mel_spectrogram(
        &self,
        _text_embedding: &[f32],
        n_mels: usize,
        n_frames: usize,
    ) -> Result<Vec<f32>> {
        if self.model_loaded() {
            // Placeholder: return silent frames until acoustic model weights are loaded
            Ok(vec![0.0; n_mels * n_frames])
        } else {
            Ok(vec![0.0; n_mels * n_frames])
        }
    }

    /// Convert mel spectrogram to waveform using Griffin-Lim algorithm.
    fn mel_to_waveform(
        &self,
        mel_data: &[f32],
        n_mels: usize,
        n_frames: usize,
        hop_size: usize,
        channels: usize,
    ) -> Result<Vec<f32>> {
        let n_fft = hop_size * 4; // Typical ratio
        let n_iter = 32; // Griffin-Lim iterations
        let sample_rate = 44100; // Default sample rate

        let mono = Self::griffin_lim(mel_data, n_mels, n_frames, sample_rate, hop_size, n_fft, n_iter);

        let total_samples = n_frames * hop_size;
        let mut waveform = vec![0.0f32; channels * total_samples];

        // Copy mono signal to all channels, truncating/padding to total_samples
        for ch in 0..channels {
            for s in 0..total_samples {
                waveform[ch * total_samples + s] = mono.get(s).copied().unwrap_or(0.0);
            }
        }

        Ok(waveform)
    }

    /// Griffin-Lim algorithm: iterative phase estimation from magnitude spectrogram.
    ///
    /// Steps:
    /// 1. Initialize random phase
    /// 2. Iterate: ISTFT -> STFT -> replace magnitude -> ISTFT
    fn griffin_lim(
        mel_spectrogram: &[f32],
        n_mels: usize,
        n_frames: usize,
        _sample_rate: usize,
        hop_size: usize,
        n_fft: usize,
        n_iter: usize,
    ) -> Vec<f32> {
        let n_freqs = n_fft / 2 + 1;
        let total_samples = (n_frames - 1) * hop_size + n_fft;

        // Convert mel spectrogram to linear magnitude spectrogram
        // (simplified: use mel filterbank inverse)
        let mut magnitudes = vec![0.0f32; n_freqs * n_frames];
        for f in 0..n_freqs.min(n_mels) {
            for t in 0..n_frames {
                magnitudes[f * n_frames + t] = mel_spectrogram[f.min(n_mels - 1) * n_frames + t].max(0.0);
            }
        }

        // Initialize phase randomly
        let mut phase = vec![0.0f32; n_freqs * n_frames];
        let mut rng_state: u64 = 42;
        for p in phase.iter_mut() {
            // Simple LCG random
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *p = (rng_state as f32 / u64::MAX as f32) * 2.0 * std::f32::consts::PI;
        }

        let mut signal = vec![0.0f32; total_samples];

        // Griffin-Lim iterations
        for _iter in 0..n_iter {
            // Reconstruct signal from magnitude + phase (inverse STFT)
            signal.fill(0.0);
            let mut window_sum = vec![0.0f32; total_samples];

            for t in 0..n_frames {
                let offset = t * hop_size;
                for n in 0..n_fft.min(total_samples - offset) {
                    let mut sample = 0.0f32;
                    for f in 0..n_freqs {
                        let angle = phase[f * n_frames + t]
                            + 2.0 * std::f32::consts::PI * f as f32 * n as f32 / n_fft as f32;
                        sample += magnitudes[f * n_frames + t] * angle.cos();
                    }
                    // Apply Hann window
                    let window = 0.5
                        * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos());
                    signal[offset + n] += sample * window;
                    window_sum[offset + n] += window * window;
                }
            }

            // Normalize by window sum
            for i in 0..total_samples {
                if window_sum[i] > 1e-8 {
                    signal[i] /= window_sum[i];
                }
            }

            // Re-estimate phase from signal (forward STFT)
            for t in 0..n_frames {
                let offset = t * hop_size;
                for f in 0..n_freqs {
                    let mut real = 0.0f32;
                    let mut imag = 0.0f32;
                    for n in 0..n_fft.min(total_samples - offset) {
                        let window = 0.5
                            * (1.0
                                - (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos());
                        let angle = -2.0 * std::f32::consts::PI * f as f32 * n as f32
                            / n_fft as f32;
                        real += signal[offset + n] * window * angle.cos();
                        imag += signal[offset + n] * window * angle.sin();
                    }
                    phase[f * n_frames + t] = imag.atan2(real);
                }
            }
        }

        // Normalize to [-1, 1]
        let max_abs = signal.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max_abs > 1e-8 {
            for s in signal.iter_mut() {
                *s /= max_abs;
            }
        }

        signal
    }

    /// Check if acoustic model weights are loaded.
    fn model_loaded(&self) -> bool {
        self.pipeline.is_some()
    }
}

impl Default for AudioHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl ModalityHandler for AudioHandler {
    fn modality(&self) -> Modality {
        Modality::Audio
    }

    fn optimal_chunk_size(&self, available_memory: usize) -> usize {
        // Chunk by audio duration
        // 44100 samples/sec * 2 bytes * 100ms = ~8.8KB per chunk
        let chunk_bytes = 44100 * 2 * self.chunk_duration_ms / 1000;
        available_memory / chunk_bytes
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn prefetch_pattern(&self) -> PrefetchPattern {
        PrefetchPattern::Sequential
    }

    fn cache_strategy(&self) -> CacheStrategy {
        CacheStrategy::Recent(8)
    }
}

/// Audio chunk for streaming.
#[derive(Debug)]
pub struct AudioChunk {
    /// Waveform samples
    pub samples: Tensor,
    /// Timestamp in ms
    pub timestamp_ms: u64,
    /// Duration in ms
    pub duration_ms: u32,
}
