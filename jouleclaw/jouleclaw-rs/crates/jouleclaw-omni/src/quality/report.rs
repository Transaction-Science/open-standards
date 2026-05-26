//! Unified quality report types.

use super::audio::AudioReport;
use super::image::ImageReport;
use super::text::TextReport;

/// Unified quality report across modalities.
#[derive(Debug, Clone)]
pub enum QualityReport {
    /// Image generation quality
    Image(ImageReport),
    /// Audio generation quality
    Audio(AudioReport),
    /// Text generation quality
    Text(TextReport),
    /// Video: per-frame image reports + temporal coherence
    Video {
        /// Per-frame image reports
        frames: Vec<ImageReport>,
        /// Mean SSIM-like similarity between consecutive frames (0.0-1.0)
        temporal_coherence: f32,
    },
}

impl QualityReport {
    /// Whether the output passes basic validity checks.
    pub fn is_valid(&self) -> bool {
        match self {
            Self::Image(r) => r.is_valid && !r.is_uniform,
            Self::Audio(r) => r.is_valid && !r.is_silence,
            Self::Text(r) => r.is_valid && !r.is_degenerate,
            Self::Video { frames, .. } => {
                !frames.is_empty() && frames.iter().all(|f| f.is_valid)
            }
        }
    }

    /// Collect all warnings from the report.
    pub fn warnings(&self) -> Vec<String> {
        match self {
            Self::Image(r) => r.warnings(),
            Self::Audio(r) => r.warnings(),
            Self::Text(r) => r.warnings(),
            Self::Video { frames, temporal_coherence } => {
                let mut w: Vec<String> = frames.iter()
                    .enumerate()
                    .flat_map(|(i, f)| {
                        f.warnings().into_iter().map(move |msg| format!("frame {i}: {msg}"))
                    })
                    .collect();
                if *temporal_coherence < 0.3 && frames.len() > 1 {
                    w.push(format!("Low temporal coherence: {temporal_coherence:.2} (flickering)"));
                }
                w
            }
        }
    }

    /// One-line human-readable summary.
    pub fn summary(&self) -> String {
        match self {
            Self::Image(r) => r.summary(),
            Self::Audio(r) => r.summary(),
            Self::Text(r) => r.summary(),
            Self::Video { frames, temporal_coherence } => {
                let valid = frames.iter().filter(|f| f.is_valid).count();
                let status = if valid == frames.len() { "PASS" } else { "WARN" };
                format!("[{status}] {} frames, {valid} valid, coherence={temporal_coherence:.2}", frames.len())
            }
        }
    }
}

/// Compute temporal coherence between consecutive video frames.
///
/// Uses mean absolute difference of luminance as a simple coherence proxy.
/// Returns 1.0 for identical frames, 0.0 for completely different frames.
pub fn temporal_coherence(frames: &[Vec<f32>], h: usize, w: usize) -> f32 {
    if frames.len() < 2 || h == 0 || w == 0 {
        return 1.0;
    }

    let area = h * w;
    let expected_len = 3 * area;
    let mut total_sim = 0.0_f64;
    let mut pair_count = 0usize;

    for pair in frames.windows(2) {
        let a = &pair[0];
        let b = &pair[1];

        if a.len() < expected_len || b.len() < expected_len {
            continue;
        }

        // Compute luminance MAD (mean absolute difference)
        let mut diff_sum = 0.0_f64;
        let mut count = 0usize;

        for i in 0..area {
            let luma_a = a[i] * 0.299 + a[area + i] * 0.587 + a[2 * area + i] * 0.114;
            let luma_b = b[i] * 0.299 + b[area + i] * 0.587 + b[2 * area + i] * 0.114;

            if luma_a.is_finite() && luma_b.is_finite() {
                diff_sum += (luma_a - luma_b).abs() as f64;
                count += 1;
            }
        }

        if count > 0 {
            let mad = diff_sum / count as f64;
            // Convert MAD to similarity: 1.0 = identical, 0.0 = max difference
            // Assuming values in [0, 1], max MAD ≈ 1.0
            total_sim += (1.0 - mad).max(0.0);
            pair_count += 1;
        }
    }

    if pair_count > 0 {
        (total_sim / pair_count as f64) as f32
    } else {
        1.0
    }
}

/// Serialize a quality report to a JSON-compatible value.
///
/// Returns key-value pairs suitable for embedding in API responses.
pub fn report_to_json(report: &QualityReport) -> serde_json::Value {
    match report {
        QualityReport::Image(r) => serde_json::json!({
            "type": "image",
            "is_valid": r.is_valid,
            "is_uniform": r.is_uniform,
            "spatial_entropy": format!("{:.2}", r.spatial_entropy),
            "sharpness": format!("{:.4}", r.laplacian_sharpness),
            "saturation_pct": format!("{:.1}", r.saturation_pct),
            "warnings": r.warnings(),
            "summary": r.summary(),
        }),
        QualityReport::Audio(r) => serde_json::json!({
            "type": "audio",
            "is_valid": r.is_valid,
            "is_silence": r.is_silence,
            "is_clipping": r.is_clipping,
            "rms_energy": format!("{:.4}", r.rms_energy),
            "peak_amplitude": format!("{:.4}", r.peak_amplitude),
            "crest_factor": format!("{:.1}", r.crest_factor),
            "dc_offset": format!("{:.4}", r.dc_offset),
            "zero_crossing_rate": format!("{:.4}", r.zero_crossing_rate),
            "warnings": r.warnings(),
            "summary": r.summary(),
        }),
        QualityReport::Text(r) => serde_json::json!({
            "type": "text",
            "is_valid": r.is_valid,
            "is_degenerate": r.is_degenerate,
            "token_count": r.token_count,
            "unique_ratio": format!("{:.2}", r.unique_ratio),
            "max_consecutive_repeat": r.max_consecutive_repeat,
            "oov_count": r.oov_count,
            "mean_logprob": r.mean_logprob.map(|v| format!("{v:.2}")),
            "warnings": r.warnings(),
            "summary": r.summary(),
        }),
        QualityReport::Video { frames, temporal_coherence } => {
            let valid_count = frames.iter().filter(|f| f.is_valid).count();
            serde_json::json!({
                "type": "video",
                "frame_count": frames.len(),
                "valid_frames": valid_count,
                "temporal_coherence": format!("{temporal_coherence:.2}"),
                "warnings": report.warnings(),
                "summary": report.summary(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_coherence_identical() {
        let frame = vec![0.5f32; 3 * 4 * 4];
        let tc = temporal_coherence(&[frame.clone(), frame], 4, 4);
        assert!((tc - 1.0).abs() < 1e-5, "identical frames should have coherence 1.0, got {tc}");
    }

    #[test]
    fn temporal_coherence_different() {
        let frame_a = vec![0.0f32; 3 * 4 * 4];
        let frame_b = vec![1.0f32; 3 * 4 * 4];
        let tc = temporal_coherence(&[frame_a, frame_b], 4, 4);
        assert!(tc < 0.1, "opposite frames should have low coherence, got {tc}");
    }

    #[test]
    fn report_json_roundtrip() {
        let report = QualityReport::Text(TextReport {
            token_count: 10,
            unique_ratio: 0.9,
            max_consecutive_repeat: 2,
            oov_count: 0,
            mean_logprob: Some(-1.5),
            min_logprob: Some(-3.0),
            is_degenerate: false,
            is_valid: true,
        });
        let json = report_to_json(&report);
        assert_eq!(json["type"], "text");
        assert_eq!(json["is_valid"], true);
        assert_eq!(json["token_count"], 10);
    }
}
