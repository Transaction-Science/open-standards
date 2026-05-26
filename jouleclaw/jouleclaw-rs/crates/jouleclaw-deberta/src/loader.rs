//! Safetensors loader (DeBERTa-v3 weight inventory + validation).
//!
//! The MoritzLaurer/DeBERTa-v3-large-mnli safetensors file contains
//! 395 tensors:
//!
//!   - 384 = 24 layers × 16 tensors per layer (Q/K/V projections,
//!           attention output dense + LayerNorm, FFN intermediate
//!           dense, FFN output dense + LayerNorm — each with weight
//!           and bias).
//!   - 11 non-layer tensors: word embeddings, embedding LayerNorm,
//!           a stored position_ids buffer, encoder-level
//!           LayerNorm + relative position embeddings, pooler dense,
//!           and the NLI classifier head.
//!
//! All weights are fp16. The pipeline upcasts to fp32 on load for
//! deterministic accumulation; the storage is fp16 to keep the
//! resident-set down to 830 MB.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::config::ModelConfig;

/// On-disk file locations within a HuggingFace-style model dir.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub config_json: PathBuf,
    pub safetensors: PathBuf,
    pub tokenizer_json: PathBuf,
    pub spm_model: PathBuf,
}

impl ModelFiles {
    pub fn from_dir(dir: impl AsRef<Path>) -> Self {
        let d = dir.as_ref();
        Self {
            config_json: d.join("config.json"),
            safetensors: d.join("model.safetensors"),
            tokenizer_json: d.join("tokenizer.json"),
            spm_model: d.join("spm.model"),
        }
    }
}

#[derive(Debug)]
pub enum LoaderError {
    Io(String),
    Parse(String),
    UnexpectedShape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    UnexpectedDtype {
        name: String,
        expected: Dtype,
        actual: Dtype,
    },
    MissingTensor(String),
    MissingFile(PathBuf),
    Config(crate::config::ConfigError),
}

impl std::fmt::Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(s) => write!(f, "io: {s}"),
            Self::Parse(s) => write!(f, "parse: {s}"),
            Self::UnexpectedShape { name, expected, actual } => write!(
                f,
                "{name}: shape mismatch, expected {expected:?}, got {actual:?}"
            ),
            Self::UnexpectedDtype { name, expected, actual } => write!(
                f,
                "{name}: dtype mismatch, expected {expected:?}, got {actual:?}"
            ),
            Self::MissingTensor(s) => write!(f, "missing tensor: {s}"),
            Self::MissingFile(p) => write!(f, "missing file: {}", p.display()),
            Self::Config(e) => write!(f, "config: {e}"),
        }
    }
}

impl std::error::Error for LoaderError {}

impl From<crate::config::ConfigError> for LoaderError {
    fn from(e: crate::config::ConfigError) -> Self {
        Self::Config(e)
    }
}

/// Summary of what was found in a safetensors file. Cheap to
/// produce; doesn't materialize tensor data.
#[derive(Debug, Clone)]
pub struct ModelInventory {
    pub config: ModelConfig,
    /// `(name, shape, dtype)` per tensor.
    pub tensors: BTreeMap<String, TensorMeta>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorMeta {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl ModelInventory {
    /// Read the model directory and validate the tensor inventory
    /// against [`ModelConfig`]. Doesn't load tensor data.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, LoaderError> {
        let files = ModelFiles::from_dir(dir);

        if !files.config_json.exists() {
            return Err(LoaderError::MissingFile(files.config_json.clone()));
        }
        if !files.safetensors.exists() {
            return Err(LoaderError::MissingFile(files.safetensors.clone()));
        }

        let config_text = std::fs::read_to_string(&files.config_json)
            .map_err(|e| LoaderError::Io(format!("read config.json: {e}")))?;
        let config = ModelConfig::from_config_json(&config_text)?;

        let file = File::open(&files.safetensors)
            .map_err(|e| LoaderError::Io(format!("open safetensors: {e}")))?;
        // SAFETY: we hold a read-only mmap for the duration of the
        // call. The OS guarantees stability so long as no one truncates
        // the file underneath us.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| LoaderError::Io(format!("mmap: {e}")))?;
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| LoaderError::Parse(format!("safetensors deserialize: {e}")))?;

        let mut tensors = BTreeMap::new();
        for (name, view) in st.tensors() {
            tensors.insert(
                name.clone(),
                TensorMeta {
                    shape: view.shape().to_vec(),
                    dtype: view.dtype(),
                },
            );
        }

        let inv = Self { config, tensors };
        inv.validate()?;
        Ok(inv)
    }

    /// Validate that the inventory contains every tensor we need to
    /// run the model, and that each tensor has the expected shape.
    pub fn validate(&self) -> Result<(), LoaderError> {
        let c = &self.config;

        // Non-layer tensors.
        let h = c.hidden_size;
        let v = c.vocab_size;
        let m = c.max_position_embeddings;
        let p = c.position_buckets * 2;
        let labels = c.num_labels;
        let ffn = c.intermediate_size;

        let mut expected: Vec<(&str, Vec<usize>)> = vec![
            ("deberta.embeddings.word_embeddings.weight", vec![v, h]),
            ("deberta.embeddings.LayerNorm.weight", vec![h]),
            ("deberta.embeddings.LayerNorm.bias", vec![h]),
            // The `position_ids` buffer is metadata-only (the canonical
            // 0..512 sequence); it's an int64 tensor we never read from
            // safetensors. We don't include it in the strict inventory
            // but tolerate its presence.
            ("deberta.encoder.LayerNorm.weight", vec![h]),
            ("deberta.encoder.LayerNorm.bias", vec![h]),
            ("deberta.encoder.rel_embeddings.weight", vec![p, h]),
            ("pooler.dense.weight", vec![h, h]),
            ("pooler.dense.bias", vec![h]),
            ("classifier.weight", vec![labels, h]),
            ("classifier.bias", vec![labels]),
        ];
        let _ = m; // max_position_embeddings is referenced but not in tensor inventory directly.

        // Per-layer tensors.
        for layer in 0..c.num_hidden_layers {
            let l = layer.to_string();
            for proj in ["query_proj", "key_proj", "value_proj"] {
                let w = leak(format!(
                    "deberta.encoder.layer.{l}.attention.self.{proj}.weight"
                ));
                let b = leak(format!(
                    "deberta.encoder.layer.{l}.attention.self.{proj}.bias"
                ));
                expected.push((w, vec![h, h]));
                expected.push((b, vec![h]));
            }
            expected.push((
                leak(format!(
                    "deberta.encoder.layer.{l}.attention.output.dense.weight"
                )),
                vec![h, h],
            ));
            expected.push((
                leak(format!(
                    "deberta.encoder.layer.{l}.attention.output.dense.bias"
                )),
                vec![h],
            ));
            expected.push((
                leak(format!(
                    "deberta.encoder.layer.{l}.attention.output.LayerNorm.weight"
                )),
                vec![h],
            ));
            expected.push((
                leak(format!(
                    "deberta.encoder.layer.{l}.attention.output.LayerNorm.bias"
                )),
                vec![h],
            ));
            expected.push((
                leak(format!(
                    "deberta.encoder.layer.{l}.intermediate.dense.weight"
                )),
                vec![ffn, h],
            ));
            expected.push((
                leak(format!(
                    "deberta.encoder.layer.{l}.intermediate.dense.bias"
                )),
                vec![ffn],
            ));
            expected.push((
                leak(format!("deberta.encoder.layer.{l}.output.dense.weight")),
                vec![h, ffn],
            ));
            expected.push((
                leak(format!("deberta.encoder.layer.{l}.output.dense.bias")),
                vec![h],
            ));
            expected.push((
                leak(format!(
                    "deberta.encoder.layer.{l}.output.LayerNorm.weight"
                )),
                vec![h],
            ));
            expected.push((
                leak(format!("deberta.encoder.layer.{l}.output.LayerNorm.bias")),
                vec![h],
            ));
        }

        for (name, shape) in &expected {
            let meta = self
                .tensors
                .get(*name)
                .ok_or_else(|| LoaderError::MissingTensor(name.to_string()))?;
            if &meta.shape != shape {
                return Err(LoaderError::UnexpectedShape {
                    name: name.to_string(),
                    expected: shape.clone(),
                    actual: meta.shape.clone(),
                });
            }
            // DeBERTa-v3-large ships fp16. Tolerate fp32 too (re-saved
            // weights are sometimes upcast).
            if meta.dtype != Dtype::F16 && meta.dtype != Dtype::F32 {
                return Err(LoaderError::UnexpectedDtype {
                    name: name.to_string(),
                    expected: Dtype::F16,
                    actual: meta.dtype,
                });
            }
        }

        Ok(())
    }

    pub fn total_tensors(&self) -> usize {
        self.tensors.len()
    }
}

/// Leak a String into a `&'static str` so error variants can carry
/// the formatted name. Bounded by validate()'s ~395 calls.
fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .expect("workspace root")
    }

    fn model_dir_present() -> Option<PathBuf> {
        let p = workspace_root().join("models/deberta-v3-large-mnli");
        if p.join("model.safetensors").exists() {
            Some(p)
        } else {
            None
        }
    }

    #[test]
    fn inventory_loads_real_model_when_available() {
        let Some(dir) = model_dir_present() else {
            eprintln!("skip: model not downloaded");
            return;
        };
        let inv = ModelInventory::from_dir(&dir).expect("inventory");
        // 395 = 384 layer tensors (24×16) + 11 non-layer.
        assert!(
            inv.total_tensors() >= 395,
            "expected ≥395 tensors, got {}",
            inv.total_tensors()
        );
        assert_eq!(inv.config.num_hidden_layers, 24);
        assert_eq!(inv.config.hidden_size, 1024);
        assert_eq!(inv.config.num_labels, 3);
        // Spot-check a tensor.
        let cls = inv
            .tensors
            .get("classifier.weight")
            .expect("classifier present");
        assert_eq!(cls.shape, vec![3, 1024]);
    }

    #[test]
    fn missing_dir_errors() {
        let res = ModelInventory::from_dir("/tmp/does-not-exist-deberta");
        assert!(matches!(res, Err(LoaderError::MissingFile(_))));
    }
}
