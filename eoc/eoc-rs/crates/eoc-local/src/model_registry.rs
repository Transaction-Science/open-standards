//! Local catalogue of installed inference models.
//!
//! `~/.eoc/models/` (overridable) is the well-known location. Each
//! sub-directory or `.gguf` file is treated as one model; the registry
//! holds a JSON manifest mapping logical model names to backend / file
//! path / memory requirement / quantization / context window.
//!
//! Discovery is best-effort. A directory scan parses each `.gguf`
//! header to populate the manifest; files that fail to parse are
//! reported but do not abort the scan.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{LocalError, LocalResult};

#[cfg(any(feature = "gguf", feature = "llamacpp"))]
use crate::gguf::{GgufFile, GgufMetadataValue};

/// Which on-host inference stack a model is intended for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendKind {
    /// llama.cpp via GGUF.
    LlamaCpp,
    /// Apple MLX.
    Mlx,
    /// MLC-LLM (TVM).
    Mlc,
    /// ONNX Runtime.
    Onnx,
    /// Stack unknown / mixed.
    Unknown,
}

impl BackendKind {
    /// Stable string identifier — used in receipts and CLI output.
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendKind::LlamaCpp => "llamacpp",
            BackendKind::Mlx => "mlx",
            BackendKind::Mlc => "mlc",
            BackendKind::Onnx => "onnx",
            BackendKind::Unknown => "unknown",
        }
    }
}

/// Quantization tier of an on-disk model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quantization {
    /// 4-bit, type 0.
    Q4_0,
    /// 4-bit K-quants, medium.
    Q4KM,
    /// 5-bit K-quants, medium.
    Q5KM,
    /// 6-bit K-quants.
    Q6K,
    /// 8-bit.
    Q8_0,
    /// 16-bit floats.
    F16,
    /// 32-bit floats.
    F32,
    /// Quantization not known / not relevant (e.g. ONNX).
    Other,
}

impl Quantization {
    /// Stable string identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Quantization::Q4_0 => "Q4_0",
            Quantization::Q4KM => "Q4_K_M",
            Quantization::Q5KM => "Q5_K_M",
            Quantization::Q6K => "Q6_K",
            Quantization::Q8_0 => "Q8_0",
            Quantization::F16 => "F16",
            Quantization::F32 => "F32",
            Quantization::Other => "Other",
        }
    }

    /// Parse from the textual tag in a GGUF `general.file_type` /
    /// `general.quantization_version` / filename. Best-effort.
    pub fn from_tag(tag: &str) -> Quantization {
        let t = tag.to_ascii_uppercase();
        if t.contains("Q4_K_M") || t.contains("Q4KM") {
            Quantization::Q4KM
        } else if t.contains("Q4_0") {
            Quantization::Q4_0
        } else if t.contains("Q5_K_M") || t.contains("Q5KM") {
            Quantization::Q5KM
        } else if t.contains("Q6_K") {
            Quantization::Q6K
        } else if t.contains("Q8_0") {
            Quantization::Q8_0
        } else if t.contains("F16") {
            Quantization::F16
        } else if t.contains("F32") {
            Quantization::F32
        } else {
            Quantization::Other
        }
    }
}

/// One row in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Logical model name (registry key).
    pub name: String,
    /// Which backend stack should be used to run it.
    pub backend: BackendKind,
    /// Absolute path to the model file or directory.
    pub path: PathBuf,
    /// Total file size in bytes — order-of-magnitude memory requirement.
    pub size_bytes: u64,
    /// Quantization tier.
    pub quantization: Quantization,
    /// Context window in tokens, if known.
    pub context_window: Option<u32>,
    /// Architecture string (`llama`, `qwen2`, `mistral`, ...).
    pub architecture: Option<String>,
}

/// In-memory model registry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelRegistry {
    /// Logical name → entry.
    pub models: BTreeMap<String, ModelEntry>,
}

impl ModelRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Default models directory: `$EOC_MODELS_DIR` if set, else
    /// `~/.eoc/models/`.
    pub fn default_models_dir() -> PathBuf {
        if let Ok(dir) = std::env::var("EOC_MODELS_DIR") {
            return PathBuf::from(dir);
        }
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".eoc").join("models")
    }

    /// Scan a directory and populate the registry from any recognisable
    /// model files within. Non-recursive. Files that fail to parse are
    /// skipped (a `tracing::warn!` is emitted).
    pub fn scan(dir: impl AsRef<Path>) -> LocalResult<Self> {
        let dir = dir.as_ref();
        let mut reg = ModelRegistry::new();
        if !dir.exists() {
            return Ok(reg);
        }
        let entries =
            std::fs::read_dir(dir).map_err(|e| LocalError::Registry(e.to_string()))?;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            match Self::classify(&path) {
                Ok(Some(e)) => {
                    reg.models.insert(e.name.clone(), e);
                }
                Ok(None) => {}
                Err(err) => {
                    tracing::warn!(path = %path.display(), error = %err,
                                   "skipping unrecognised model file");
                }
            }
        }
        Ok(reg)
    }

    /// Classify a single file. Returns `Ok(None)` for "this file is not
    /// a model we recognise; skip it silently".
    pub fn classify(path: &Path) -> LocalResult<Option<ModelEntry>> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase());
        let size_bytes = std::fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or(0);

        match ext.as_deref() {
            Some("gguf") => {
                #[cfg(any(feature = "gguf", feature = "llamacpp"))]
                {
                    let f = GgufFile::open(path)?;
                    let arch = f.architecture().map(|s| s.to_string());
                    // Quantization tag — `general.file_type` is the
                    // canonical key. Fallback to filename.
                    let quant_tag = match f.meta("general.file_type") {
                        Some(GgufMetadataValue::String(s)) => s.clone(),
                        Some(other) => other.render(),
                        None => path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("")
                            .to_string(),
                    };
                    let quantization = Quantization::from_tag(&quant_tag);
                    // Context window — try `<arch>.context_length`,
                    // fallback to `llama.context_length`.
                    let ctx_key = arch
                        .as_ref()
                        .map(|a| format!("{a}.context_length"))
                        .unwrap_or_else(|| "llama.context_length".into());
                    let context_window = match f.meta(&ctx_key) {
                        Some(GgufMetadataValue::U32(v)) => Some(*v),
                        Some(GgufMetadataValue::U64(v)) => Some(*v as u32),
                        Some(GgufMetadataValue::I32(v)) => Some(*v as u32),
                        Some(GgufMetadataValue::I64(v)) => Some(*v as u32),
                        _ => None,
                    };
                    let name = f
                        .name()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| stem_or(path, "unnamed"));
                    Ok(Some(ModelEntry {
                        name,
                        backend: BackendKind::LlamaCpp,
                        path: path.to_path_buf(),
                        size_bytes,
                        quantization,
                        context_window,
                        architecture: arch,
                    }))
                }
                #[cfg(not(any(feature = "gguf", feature = "llamacpp")))]
                {
                    // Without GGUF support compiled in we still record
                    // the file's existence — backend chosen by extension.
                    Ok(Some(ModelEntry {
                        name: stem_or(path, "gguf-model"),
                        backend: BackendKind::LlamaCpp,
                        path: path.to_path_buf(),
                        size_bytes,
                        quantization: Quantization::Other,
                        context_window: None,
                        architecture: None,
                    }))
                }
            }
            Some("onnx") => Ok(Some(ModelEntry {
                name: stem_or(path, "onnx-model"),
                backend: BackendKind::Onnx,
                path: path.to_path_buf(),
                size_bytes,
                quantization: Quantization::Other,
                context_window: None,
                architecture: None,
            })),
            Some("safetensors") => Ok(Some(ModelEntry {
                name: stem_or(path, "mlx-or-mlc-model"),
                backend: BackendKind::Unknown,
                path: path.to_path_buf(),
                size_bytes,
                quantization: Quantization::Other,
                context_window: None,
                architecture: None,
            })),
            _ => Ok(None),
        }
    }

    /// Insert / overwrite an entry.
    pub fn insert(&mut self, entry: ModelEntry) {
        self.models.insert(entry.name.clone(), entry);
    }

    /// Look up an entry by logical name.
    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.models.get(name)
    }

    /// Iterate over entries.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ModelEntry)> {
        self.models.iter()
    }

    /// Serialize the manifest as pretty JSON.
    pub fn to_json(&self) -> LocalResult<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Parse a manifest from JSON.
    pub fn from_json(s: &str) -> LocalResult<Self> {
        Ok(serde_json::from_str(s)?)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

fn stem_or(path: &Path, default: &str) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| default.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[cfg(any(feature = "gguf", feature = "llamacpp"))]
    use std::io::Write;

    #[test]
    fn quantization_parsing_from_tags() {
        assert_eq!(Quantization::from_tag("model-Q4_K_M.gguf"), Quantization::Q4KM);
        assert_eq!(Quantization::from_tag("model.Q5_K_M"), Quantization::Q5KM);
        assert_eq!(Quantization::from_tag("foo-q8_0"), Quantization::Q8_0);
        assert_eq!(Quantization::from_tag("f16"), Quantization::F16);
        assert_eq!(Quantization::from_tag("unknown-quant"), Quantization::Other);
    }

    #[test]
    fn scan_empty_dir_yields_empty_registry() {
        let dir = tempfile::tempdir().unwrap();
        let r = ModelRegistry::scan(dir.path()).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn scan_missing_dir_does_not_fail() {
        // A missing directory is treated as "no models installed yet"
        // — the registry returns empty.
        let r = ModelRegistry::scan("/nope/never/should/exist").unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn scan_picks_up_onnx() {
        let dir = tempfile::tempdir().unwrap();
        let onnx = dir.path().join("modelA.onnx");
        fs::File::create(&onnx).unwrap();

        let r = ModelRegistry::scan(dir.path()).unwrap();
        let onnx_entry = r.get("modelA").unwrap();
        assert_eq!(onnx_entry.backend, BackendKind::Onnx);
    }

    #[cfg(any(feature = "gguf", feature = "llamacpp"))]
    #[test]
    fn scan_picks_up_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("modelB.gguf");
        let mut f = fs::File::create(&gguf).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&crate::gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        f.write_all(&buf).unwrap();
        drop(f);
        let r = ModelRegistry::scan(dir.path()).unwrap();
        let gguf_entry = r.get("modelB").unwrap();
        assert_eq!(gguf_entry.backend, BackendKind::LlamaCpp);
    }

    #[test]
    fn json_round_trip() {
        let mut r = ModelRegistry::new();
        r.insert(ModelEntry {
            name: "tinyllama".into(),
            backend: BackendKind::LlamaCpp,
            path: PathBuf::from("/tmp/tinyllama.gguf"),
            size_bytes: 1_000_000_000,
            quantization: Quantization::Q4KM,
            context_window: Some(2048),
            architecture: Some("llama".into()),
        });
        let j = r.to_json().unwrap();
        let back = ModelRegistry::from_json(&j).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back.get("tinyllama").unwrap().context_window, Some(2048));
    }

    #[test]
    fn default_dir_falls_back_to_home() {
        // Don't touch the process env (would require unsafe in
        // edition-2024). Just verify the fallback path shape when the
        // var is unset.
        if std::env::var("EOC_MODELS_DIR").is_ok() {
            // Test environment has the override set; just check the
            // function produces a path under it.
            let d = ModelRegistry::default_models_dir();
            assert!(d.is_absolute() || d.starts_with("."));
            return;
        }
        let d = ModelRegistry::default_models_dir();
        // Should end in `.eoc/models`.
        assert!(d.ends_with("models"));
        assert!(d.to_string_lossy().contains(".eoc"));
    }
}
