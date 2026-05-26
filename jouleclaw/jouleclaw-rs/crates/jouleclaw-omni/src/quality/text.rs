//! Text generation quality metrics.
//!
//! Operates on token IDs and optional log-probabilities.

/// Quality report for generated text.
#[derive(Debug, Clone)]
pub struct TextReport {
    /// Total number of tokens generated
    pub token_count: usize,
    /// Fraction of distinct tokens (0.0-1.0)
    pub unique_ratio: f32,
    /// Longest consecutive run of the same token
    pub max_consecutive_repeat: u32,
    /// Number of tokens with id >= vocab_size
    pub oov_count: usize,
    /// Mean log-probability (higher = more confident)
    pub mean_logprob: Option<f32>,
    /// Minimum log-probability (lowest-confidence token)
    pub min_logprob: Option<f32>,
    /// True if output shows degenerate repetition
    pub is_degenerate: bool,
    /// True if all tokens are in-vocab and no pathological patterns
    pub is_valid: bool,
}

impl TextReport {
    /// Generate human-readable warnings.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if self.oov_count > 0 {
            w.push(format!("{} out-of-vocabulary tokens detected", self.oov_count));
        }
        if self.is_degenerate {
            w.push(format!(
                "Degenerate output: unique_ratio={:.2}, max_repeat={}",
                self.unique_ratio, self.max_consecutive_repeat
            ));
        }
        if self.token_count == 0 {
            w.push("Empty output (zero tokens generated)".into());
        }
        if let Some(mean_lp) = self.mean_logprob {
            if mean_lp < -5.0 {
                w.push(format!("Low confidence: mean logprob={:.2}", mean_lp));
            }
        }
        w
    }

    /// One-line summary.
    pub fn summary(&self) -> String {
        let status = if self.is_valid && !self.is_degenerate { "PASS" } else { "WARN" };
        let lp = self.mean_logprob
            .map(|v| format!(" lp={v:.2}"))
            .unwrap_or_default();
        format!(
            "[{status}] {} tokens unique={:.0}% max_repeat={}{lp}",
            self.token_count,
            self.unique_ratio * 100.0,
            self.max_consecutive_repeat
        )
    }
}

/// Compute text generation quality metrics.
///
/// - `tokens`: generated token IDs
/// - `vocab_size`: maximum valid token ID (exclusive)
/// - `logprobs`: optional per-token log-probabilities
pub fn text_report(tokens: &[u32], vocab_size: u32, logprobs: Option<&[f32]>) -> TextReport {
    if tokens.is_empty() {
        return TextReport {
            token_count: 0,
            unique_ratio: 0.0,
            max_consecutive_repeat: 0,
            oov_count: 0,
            mean_logprob: None,
            min_logprob: None,
            is_degenerate: false,
            is_valid: false,
        };
    }

    // Unique ratio
    let mut sorted = tokens.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let unique_ratio = sorted.len() as f32 / tokens.len() as f32;

    // Max consecutive repeat
    let mut max_repeat = 1u32;
    let mut current_repeat = 1u32;
    for i in 1..tokens.len() {
        if tokens[i] == tokens[i - 1] {
            current_repeat += 1;
            if current_repeat > max_repeat {
                max_repeat = current_repeat;
            }
        } else {
            current_repeat = 1;
        }
    }

    // Out-of-vocab tokens
    let oov_count = tokens.iter().filter(|&&t| t >= vocab_size).count();

    // Logprob statistics
    let (mean_logprob, min_logprob) = if let Some(lps) = logprobs {
        let finite: Vec<f32> = lps.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            (None, None)
        } else {
            let mean = finite.iter().sum::<f32>() / finite.len() as f32;
            let min = finite.iter().copied().fold(f32::INFINITY, f32::min);
            (Some(mean), Some(min))
        }
    } else {
        (None, None)
    };

    // Degenerate detection thresholds
    let is_degenerate = unique_ratio < 0.1 || max_repeat > 10;
    let is_valid = oov_count == 0 && tokens.len() > 0;

    TextReport {
        token_count: tokens.len(),
        unique_ratio,
        max_consecutive_repeat: max_repeat,
        oov_count,
        mean_logprob,
        min_logprob,
        is_degenerate,
        is_valid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetitive_detected() {
        let tokens = vec![42u32; 100];
        let report = text_report(&tokens, 50000, None);
        assert!(report.is_degenerate, "all-same tokens should be degenerate");
        assert!((report.unique_ratio - 0.01).abs() < 0.02);
        assert_eq!(report.max_consecutive_repeat, 100);
    }

    #[test]
    fn diverse_tokens_valid() {
        let tokens: Vec<u32> = (0..100).collect();
        let report = text_report(&tokens, 50000, None);
        assert!(!report.is_degenerate, "sequential tokens should not be degenerate");
        assert!((report.unique_ratio - 1.0).abs() < 1e-5);
        assert_eq!(report.max_consecutive_repeat, 1);
        assert!(report.is_valid);
    }

    #[test]
    fn oov_detected() {
        let tokens = vec![0, 1, 2, 99999];
        let report = text_report(&tokens, 1000, None);
        assert_eq!(report.oov_count, 1);
    }

    #[test]
    fn logprob_stats() {
        let tokens = vec![1, 2, 3, 4, 5];
        let logprobs = vec![-1.0, -2.0, -3.0, -0.5, -1.5];
        let report = text_report(&tokens, 50000, Some(&logprobs));
        let mean = report.mean_logprob.unwrap();
        assert!((mean - (-1.6)).abs() < 0.01, "mean logprob should be -1.6, got {mean}");
        assert!((report.min_logprob.unwrap() - (-3.0)).abs() < 1e-5);
    }

    #[test]
    fn empty_output() {
        let report = text_report(&[], 50000, None);
        assert_eq!(report.token_count, 0);
        assert!(!report.is_valid);
    }

    #[test]
    fn moderate_repetition_ok() {
        // Some repetition is natural (e.g., "the the" → repeat=2)
        let tokens = vec![1, 2, 2, 3, 4, 5, 5, 5, 6, 7];
        let report = text_report(&tokens, 50000, None);
        assert!(!report.is_degenerate, "moderate repetition should not be degenerate");
        assert_eq!(report.max_consecutive_repeat, 3);
    }
}
