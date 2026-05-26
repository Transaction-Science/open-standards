//! Audio quality metrics for generated audio / music / speech outputs.
//!
//! Operates on `&[f32]` PCM samples (expected range [-1.0, 1.0]).

use super::stats;

/// Quality report for generated audio.
#[derive(Debug, Clone)]
pub struct AudioReport {
    /// Root mean square energy
    pub rms_energy: f32,
    /// Peak absolute amplitude
    pub peak_amplitude: f32,
    /// DC offset (mean of all samples)
    pub dc_offset: f32,
    /// Crest factor: peak / RMS (natural audio ≈ 3-20)
    pub crest_factor: f32,
    /// Zero-crossing rate (crossings per sample)
    pub zero_crossing_rate: f32,
    /// True if RMS < 0.001 (effectively silent)
    pub is_silence: bool,
    /// True if peak > 0.99 (clipping / saturation)
    pub is_clipping: bool,
    /// True if data is finite and non-empty
    pub is_valid: bool,
    /// Number of samples
    pub sample_count: usize,
    /// NaN count
    pub nan_count: usize,
    /// Inf count
    pub inf_count: usize,
}

impl AudioReport {
    /// Generate human-readable warnings.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if !self.is_valid {
            w.push("Audio contains NaN or Inf values".into());
        }
        if self.is_silence {
            w.push(format!("Audio is silent (RMS={:.6})", self.rms_energy));
        }
        if self.is_clipping {
            w.push(format!("Audio is clipping (peak={:.4})", self.peak_amplitude));
        }
        if self.dc_offset.abs() > 0.05 {
            w.push(format!("DC offset detected: {:.4}", self.dc_offset));
        }
        if self.crest_factor > 30.0 && !self.is_silence {
            w.push(format!("Abnormal crest factor: {:.1} (possible impulse artifacts)", self.crest_factor));
        }
        if self.zero_crossing_rate < 0.001 && !self.is_silence {
            w.push("Very low zero-crossing rate (possible DC or sub-sonic content)".into());
        }
        w
    }

    /// One-line summary.
    pub fn summary(&self) -> String {
        let status = if self.is_valid && !self.is_silence && !self.is_clipping { "PASS" } else { "WARN" };
        format!(
            "[{status}] {} samples RMS={:.4} peak={:.4} crest={:.1} ZCR={:.4}",
            self.sample_count, self.rms_energy, self.peak_amplitude, self.crest_factor, self.zero_crossing_rate
        )
    }
}

/// Compute audio quality metrics from PCM f32 samples.
///
/// Samples are expected in [-1.0, 1.0] range.
pub fn audio_report(samples: &[f32], _sample_rate: u32) -> AudioReport {
    if samples.is_empty() {
        return AudioReport {
            rms_energy: 0.0,
            peak_amplitude: 0.0,
            dc_offset: 0.0,
            crest_factor: 0.0,
            zero_crossing_rate: 0.0,
            is_silence: true,
            is_clipping: false,
            is_valid: false,
            sample_count: 0,
            nan_count: 0,
            inf_count: 0,
        };
    }

    let s = stats::stats(samples);
    let (nan_count, inf_count) = s.as_ref()
        .map(|s| (s.nan_count, s.inf_count))
        .unwrap_or((0, 0));
    let is_valid = nan_count == 0 && inf_count == 0;

    // RMS energy
    let mut sum_sq = 0.0_f64;
    let mut peak = 0.0_f32;
    let mut sum = 0.0_f64;
    let mut finite_count = 0usize;
    let mut zero_crossings = 0usize;
    let mut prev_sign = false; // true = positive

    for (i, &sample) in samples.iter().enumerate() {
        if !sample.is_finite() {
            continue;
        }

        sum += sample as f64;
        sum_sq += (sample as f64) * (sample as f64);
        let abs_val = sample.abs();
        if abs_val > peak {
            peak = abs_val;
        }
        finite_count += 1;

        // Zero-crossing detection
        let cur_sign = sample >= 0.0;
        if i > 0 && cur_sign != prev_sign {
            zero_crossings += 1;
        }
        prev_sign = cur_sign;
    }

    let rms = if finite_count > 0 {
        ((sum_sq / finite_count as f64) as f32).sqrt()
    } else {
        0.0
    };

    let dc_offset = if finite_count > 0 {
        (sum / finite_count as f64) as f32
    } else {
        0.0
    };

    let crest_factor = if rms > 1e-10 { peak / rms } else { 0.0 };

    let zcr = if finite_count > 1 {
        zero_crossings as f32 / (finite_count - 1) as f32
    } else {
        0.0
    };

    AudioReport {
        rms_energy: rms,
        peak_amplitude: peak,
        dc_offset,
        crest_factor,
        zero_crossing_rate: zcr,
        is_silence: rms < 0.001,
        is_clipping: {
            // True clipping: many samples clamped at ±peak (flat tops).
            // A clean sine naturally spends a few % near its peaks.
            // Clipped signals have flat tops → much higher ratio at bounds.
            // Count samples within 0.1% of peak.
            if peak > 0.99 {
                let bound = peak * 0.999;
                let at_bounds = samples.iter().filter(|&&s| s.is_finite() && s.abs() >= bound).count();
                at_bounds as f32 / finite_count.max(1) as f32 > 0.08 // > 8% at peak = flat tops
            } else {
                false
            }
        },
        is_valid,
        sample_count: samples.len(),
        nan_count,
        inf_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    fn sine_wave(freq: f32, sample_rate: u32, duration: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn silence_detected() {
        let samples = vec![0.0f32; 16000];
        let report = audio_report(&samples, 16000);
        assert!(report.is_silence, "zeros should be silence");
        assert!(report.is_valid, "zeros should be valid");
        assert!(report.rms_energy < 0.001);
    }

    #[test]
    fn sine_wave_valid() {
        let samples = sine_wave(440.0, 16000, 1.0);
        let report = audio_report(&samples, 16000);
        assert!(!report.is_silence, "sine should not be silence");
        assert!(!report.is_clipping, "unit sine should not clip");
        assert!(report.is_valid);
        // RMS of sine wave = 1/sqrt(2) ≈ 0.707
        assert!((report.rms_energy - 0.707).abs() < 0.02, "sine RMS should be ~0.707, got {}", report.rms_energy);
        // Crest factor of sine = sqrt(2) ≈ 1.414
        assert!((report.crest_factor - 1.414).abs() < 0.1, "sine crest should be ~1.414, got {}", report.crest_factor);
    }

    #[test]
    fn clipping_detected() {
        let mut samples = sine_wave(440.0, 16000, 0.1);
        // Amplify to cause clipping
        for s in &mut samples {
            *s = (*s * 2.0).clamp(-1.0, 1.0);
        }
        let report = audio_report(&samples, 16000);
        assert!(report.is_clipping, "clipped sine should detect clipping");
    }

    #[test]
    fn dc_offset_detected() {
        let samples: Vec<f32> = sine_wave(440.0, 16000, 0.5)
            .iter()
            .map(|&s| s + 0.3)
            .collect();
        let report = audio_report(&samples, 16000);
        assert!((report.dc_offset - 0.3).abs() < 0.02, "DC offset should be ~0.3, got {}", report.dc_offset);
    }

    #[test]
    fn nan_audio_invalid() {
        let mut samples = vec![0.5f32; 1000];
        samples[500] = f32::NAN;
        let report = audio_report(&samples, 16000);
        assert!(!report.is_valid, "NaN audio should not be valid");
        assert_eq!(report.nan_count, 1);
    }
}
