//! Architectural constants for DeBERTa-v3-large with NLI head.
//!
//! Sourced from the public configs:
//!
//! - microsoft/deberta-v3-large/config.json
//! - MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli/config.json
//!
//! Verified against the real config.json after download (see
//! [`ModelConfig::from_config_json`]); the constants below are the
//! defaults if no config is present.

use serde::Deserialize;

use jouleclaw_schema::EntailmentLabel;

/// The DeBERTa-v3-large architectural defaults. NLI head adds
/// `num_labels = 3` on top.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f32,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub initializer_range: f32,
    pub relative_attention: bool,
    /// Position-bucket size for the relative-position embedding
    /// table. DeBERTa-v3-large uses 256.
    pub position_buckets: usize,
    /// `max_relative_positions` in the HF config; DeBERTa-v3-large
    /// sets this equal to `position_buckets`.
    pub max_relative_positions: usize,
    /// Which disentangled-attention paths are enabled. DeBERTa-v3
    /// uses `["p2c", "c2p"]`.
    pub pos_att_type: Vec<String>,
    /// Hidden activation. DeBERTa-v3 uses `"gelu"`.
    pub hidden_act: String,
    /// LayerNorm convention for the relative-position embeddings.
    /// DeBERTa-v3 uses `"layer_norm"`.
    pub norm_rel_ebd: String,
    /// Position-embedding type. DeBERTa-v3 uses `"relative"`
    /// (absolute embeddings are zeroed out).
    pub position_embedding_type: String,
    /// `type_vocab_size = 0` on DeBERTa-v3 (token_type_ids removed).
    pub type_vocab_size: usize,
    /// v3-specific: when `true`, the same K-projection is used for
    /// both the content pathway and the position pathway in
    /// disentangled attention. v2 used separate projections; v3
    /// shares them as a parameter-count optimization. Confirmed
    /// `true` in the MoritzLaurer config.
    pub share_att_key: bool,
    /// v3-specific: when `false`, absolute position embeddings are
    /// not added to the input — relative position is the sole
    /// positional signal. Confirmed `false` in the MoritzLaurer
    /// config.
    pub position_biased_input: bool,
    /// NLI fine-tune head — number of output labels. 3 for the
    /// canonical entailment/neutral/contradiction setup.
    pub num_labels: usize,
    /// Index-to-label mapping the fine-tune ships with; needed to
    /// map model output to [`EntailmentLabel`] correctly. Different
    /// fine-tunes use different orderings.
    pub label_layout: NliLabelLayout,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::deberta_v3_large_mnli_fever_anli_ling_wanli()
    }
}

impl ModelConfig {
    /// Defaults for `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
    /// (verified against the model's config.json on download).
    pub fn deberta_v3_large_mnli_fever_anli_ling_wanli() -> Self {
        Self {
            vocab_size: 128100,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            max_position_embeddings: 512,
            layer_norm_eps: 1e-7,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            initializer_range: 0.02,
            relative_attention: true,
            position_buckets: 256,
            max_relative_positions: 256,
            pos_att_type: vec!["p2c".into(), "c2p".into()],
            hidden_act: "gelu".into(),
            norm_rel_ebd: "layer_norm".into(),
            position_embedding_type: "relative".into(),
            type_vocab_size: 0,
            share_att_key: true,
            position_biased_input: false,
            num_labels: 3,
            label_layout: NliLabelLayout::EntailmentNeutralContradiction,
        }
    }

    /// Parse a HuggingFace `config.json` against this crate's
    /// expectations. Validates the architecture matches what we've
    /// built support for; returns an error if the file describes a
    /// different model family.
    pub fn from_config_json(json: &str) -> Result<Self, ConfigError> {
        let raw: RawConfig =
            serde_json::from_str(json).map_err(|e| ConfigError::ParseFailed(e.to_string()))?;
        if raw.model_type.as_deref() != Some("deberta-v2")
            && raw.model_type.as_deref() != Some("deberta-v3")
        {
            return Err(ConfigError::UnsupportedArchitecture(
                raw.model_type.unwrap_or_default(),
            ));
        }
        let label_layout = NliLabelLayout::from_id2label(raw.id2label.as_ref());
        Ok(Self {
            vocab_size: raw.vocab_size.unwrap_or(128100),
            hidden_size: raw.hidden_size.unwrap_or(1024),
            num_hidden_layers: raw.num_hidden_layers.unwrap_or(24),
            num_attention_heads: raw.num_attention_heads.unwrap_or(16),
            intermediate_size: raw.intermediate_size.unwrap_or(4096),
            max_position_embeddings: raw.max_position_embeddings.unwrap_or(512),
            layer_norm_eps: raw.layer_norm_eps.unwrap_or(1e-7),
            hidden_dropout_prob: raw.hidden_dropout_prob.unwrap_or(0.1),
            attention_probs_dropout_prob: raw.attention_probs_dropout_prob.unwrap_or(0.1),
            initializer_range: raw.initializer_range.unwrap_or(0.02),
            relative_attention: raw.relative_attention.unwrap_or(true),
            position_buckets: raw.position_buckets.unwrap_or(256),
            // HF: -1 sentinel means "use position_buckets".
            max_relative_positions: match raw.max_relative_positions {
                Some(n) if n > 0 => n as usize,
                _ => raw.position_buckets.unwrap_or(256),
            },
            pos_att_type: raw
                .pos_att_type
                .unwrap_or_else(|| vec!["p2c".into(), "c2p".into()]),
            hidden_act: raw.hidden_act.unwrap_or_else(|| "gelu".into()),
            norm_rel_ebd: raw.norm_rel_ebd.unwrap_or_else(|| "layer_norm".into()),
            position_embedding_type: raw
                .position_embedding_type
                .unwrap_or_else(|| "relative".into()),
            type_vocab_size: raw.type_vocab_size.unwrap_or(0),
            share_att_key: raw.share_att_key.unwrap_or(true),
            position_biased_input: raw.position_biased_input.unwrap_or(false),
            num_labels: raw.id2label.as_ref().map(|m| m.len()).unwrap_or(3),
            label_layout,
        })
    }

    /// Per-head dim, derived. `hidden_size` must divide
    /// `num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug)]
pub enum ConfigError {
    ParseFailed(String),
    UnsupportedArchitecture(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseFailed(s) => write!(f, "parse: {s}"),
            Self::UnsupportedArchitecture(s) => write!(f, "unsupported model_type: {s}"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// HF config.json shape — only the fields we care about.
#[derive(Debug, Deserialize)]
struct RawConfig {
    model_type: Option<String>,
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    intermediate_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    layer_norm_eps: Option<f32>,
    hidden_dropout_prob: Option<f32>,
    attention_probs_dropout_prob: Option<f32>,
    initializer_range: Option<f32>,
    relative_attention: Option<bool>,
    position_buckets: Option<usize>,
    /// HF convention: `-1` means "use `position_buckets`". Parse as
    /// signed so we don't reject the canonical sentinel value, then
    /// fold it into the planner config below.
    max_relative_positions: Option<i64>,
    pos_att_type: Option<Vec<String>>,
    hidden_act: Option<String>,
    norm_rel_ebd: Option<String>,
    position_embedding_type: Option<String>,
    type_vocab_size: Option<usize>,
    share_att_key: Option<bool>,
    position_biased_input: Option<bool>,
    id2label: Option<std::collections::BTreeMap<String, String>>,
}

/// Which output index means what label. Different NLI fine-tunes
/// use different orderings (HF MNLI = {0: contradiction, 1: neutral,
/// 2: entailment}; FEVER often reverses; the MoritzLaurer model uses
/// {0: entailment, 1: neutral, 2: contradiction} per its config).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NliLabelLayout {
    /// {0: entailment, 1: neutral, 2: contradiction}.
    EntailmentNeutralContradiction,
    /// {0: contradiction, 1: neutral, 2: entailment} — the
    /// HuggingFace MNLI default before MoritzLaurer's relabeling.
    ContradictionNeutralEntailment,
}

impl NliLabelLayout {
    pub fn from_id2label(map: Option<&std::collections::BTreeMap<String, String>>) -> Self {
        let Some(m) = map else {
            return Self::EntailmentNeutralContradiction;
        };
        let label_for = |idx: &str| {
            m.get(idx)
                .map(|s| s.to_lowercase())
                .unwrap_or_default()
        };
        let l0 = label_for("0");
        if l0.contains("contradiction") {
            Self::ContradictionNeutralEntailment
        } else {
            Self::EntailmentNeutralContradiction
        }
    }

    /// Map an output index to the schema's [`EntailmentLabel`].
    pub fn label_at(self, idx: usize) -> NliLabel {
        match (self, idx) {
            (Self::EntailmentNeutralContradiction, 0) => NliLabel::Entailment,
            (Self::EntailmentNeutralContradiction, 1) => NliLabel::Neutral,
            (Self::EntailmentNeutralContradiction, 2) => NliLabel::Contradiction,
            (Self::ContradictionNeutralEntailment, 0) => NliLabel::Contradiction,
            (Self::ContradictionNeutralEntailment, 1) => NliLabel::Neutral,
            (Self::ContradictionNeutralEntailment, 2) => NliLabel::Entailment,
            (_, _) => NliLabel::Neutral,
        }
    }
}

/// Local NLI label enum bridging to the schema's
/// [`EntailmentLabel`]. We carry our own so this crate can be used
/// without jouleclaw-schema if needed (though the conversion is free).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NliLabel {
    Entailment,
    Neutral,
    Contradiction,
}

impl From<NliLabel> for EntailmentLabel {
    fn from(l: NliLabel) -> Self {
        match l {
            NliLabel::Entailment => Self::Entails,
            NliLabel::Neutral => Self::Neutral,
            NliLabel::Contradiction => Self::Contradicts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_deberta_v3_large() {
        let c = ModelConfig::default();
        assert_eq!(c.num_hidden_layers, 24);
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.head_dim(), 64);
        assert!(c.relative_attention);
        assert_eq!(c.position_buckets, 256);
        assert_eq!(c.num_labels, 3);
    }

    #[test]
    fn parses_minimal_config_json() {
        let json = r#"{
            "model_type": "deberta-v2",
            "vocab_size": 128100,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096
        }"#;
        let c = ModelConfig::from_config_json(json).unwrap();
        assert_eq!(c.vocab_size, 128100);
        assert_eq!(c.num_hidden_layers, 24);
    }

    #[test]
    fn rejects_unsupported_arch() {
        let json = r#"{"model_type": "llama"}"#;
        assert!(matches!(
            ModelConfig::from_config_json(json),
            Err(ConfigError::UnsupportedArchitecture(_))
        ));
    }

    #[test]
    fn detects_entailment_first_layout() {
        let json = r#"{
            "model_type": "deberta-v3",
            "id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"}
        }"#;
        let c = ModelConfig::from_config_json(json).unwrap();
        assert_eq!(c.label_layout, NliLabelLayout::EntailmentNeutralContradiction);
        assert_eq!(c.label_layout.label_at(0), NliLabel::Entailment);
        assert_eq!(c.label_layout.label_at(2), NliLabel::Contradiction);
    }

    #[test]
    fn detects_contradiction_first_layout() {
        let json = r#"{
            "model_type": "deberta-v3",
            "id2label": {"0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"}
        }"#;
        let c = ModelConfig::from_config_json(json).unwrap();
        assert_eq!(c.label_layout, NliLabelLayout::ContradictionNeutralEntailment);
        assert_eq!(c.label_layout.label_at(0), NliLabel::Contradiction);
        assert_eq!(c.label_layout.label_at(2), NliLabel::Entailment);
    }

    #[test]
    fn label_bridges_to_schema() {
        let e: jouleclaw_schema::EntailmentLabel = NliLabel::Entailment.into();
        assert!(matches!(e, jouleclaw_schema::EntailmentLabel::Entails));
        let c: jouleclaw_schema::EntailmentLabel = NliLabel::Contradiction.into();
        assert!(matches!(c, jouleclaw_schema::EntailmentLabel::Contradicts));
    }
}
