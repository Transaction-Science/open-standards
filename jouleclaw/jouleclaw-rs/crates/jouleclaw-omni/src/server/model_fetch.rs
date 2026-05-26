//! Remote model fetching — download models from Hugging Face Hub or S3.
//!
//! Three-layer resolution:
//! 1. **Local cache** — check `MODEL_CACHE_DIR` first
//! 2. **S3** — download from `create-model-artifacts` bucket (if configured)
//! 3. **Hugging Face Hub** — fallback via direct HTTPS
//!
//! Each model directory contains a `manifest.json` with file list and SHA-256 hashes
//! for integrity verification.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

/// Model manifest describing files in a remote model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Model identifier (e.g. "tinyllama-1.1b-chat")
    pub model_id: String,
    /// Hugging Face repo ID (e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    pub hf_repo: Option<String>,
    /// S3 key prefix (e.g. "models/tinyllama-1.1b-chat")
    pub s3_prefix: Option<String>,
    /// Pipeline type: "llm", "diffusion", "whisper", etc.
    pub pipeline_type: String,
    /// Total size in bytes (all files combined)
    pub total_size_bytes: u64,
    /// Individual files with their checksums
    pub files: Vec<ModelFile>,
}

/// A single file within a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    /// Relative path within the model directory
    pub name: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// SHA-256 hash (hex-encoded) for verification
    pub sha256: Option<String>,
}

/// Download progress callback.
pub type ProgressCallback = Box<dyn Fn(DownloadProgress) + Send + Sync>;

/// Download progress info.
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Current file being downloaded
    pub file_name: String,
    /// Bytes downloaded for current file
    pub bytes_downloaded: u64,
    /// Total bytes for current file
    pub bytes_total: u64,
    /// File index (0-based)
    pub file_index: usize,
    /// Total number of files
    pub total_files: usize,
}

/// Configuration for model fetching.
#[derive(Debug, Clone)]
pub struct FetchConfig {
    /// Local cache directory
    pub cache_dir: PathBuf,
    /// S3 bucket name (optional)
    pub s3_bucket: Option<String>,
    /// S3 region (optional, default us-east-1)
    pub s3_region: Option<String>,
    /// Hugging Face Hub token (optional, for gated models)
    pub hf_token: Option<String>,
}

impl FetchConfig {
    /// Create from environment variables.
    pub fn from_env() -> Self {
        Self {
            cache_dir: PathBuf::from(
                std::env::var("MODEL_CACHE_DIR")
                    .unwrap_or_else(|_| "/Volumes/DataVolume/model_cache".into()),
            ),
            s3_bucket: std::env::var("MODEL_S3_BUCKET").ok(),
            s3_region: std::env::var("MODEL_S3_REGION").ok(),
            hf_token: std::env::var("HF_TOKEN").ok(),
        }
    }
}

/// Ensure a model is available locally, downloading if necessary.
///
/// Returns the local path to the model directory.
pub async fn ensure_model_available(
    model_id: &str,
    config: &FetchConfig,
    client: &reqwest::Client,
    on_progress: Option<&ProgressCallback>,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    let local_path = config.cache_dir.join(model_id);

    // Layer 1: Check local cache
    if local_path.exists() {
        let manifest_path = local_path.join("manifest.json");
        if manifest_path.exists() {
            let manifest = read_manifest(&manifest_path).await?;
            if verify_local_files(&local_path, &manifest).await {
                tracing::info!(model_id = %model_id, "Model available in local cache");
                return Ok(local_path);
            }
            tracing::warn!(model_id = %model_id, "Local cache incomplete, re-downloading");
        } else {
            // Directory exists but no manifest — assume it's a valid local model
            tracing::info!(model_id = %model_id, "Model directory exists (no manifest, assuming valid)");
            return Ok(local_path);
        }
    }

    // Create cache directory
    tokio::fs::create_dir_all(&local_path).await?;

    // Try to find model manifest
    let manifest = find_manifest(model_id, config, client).await?;

    // Download files
    let total_files = manifest.files.len();
    for (i, file) in manifest.files.iter().enumerate() {
        let dest = local_path.join(&file.name);

        // Create parent directories if needed
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Skip if file already exists and matches expected size
        if dest.exists() {
            let metadata = tokio::fs::metadata(&dest).await?;
            if metadata.len() == file.size_bytes {
                tracing::debug!(file = %file.name, "File already cached, skipping");
                continue;
            }
        }

        tracing::info!(
            model_id = %model_id,
            file = %file.name,
            size = file.size_bytes,
            "[{}/{}] Downloading",
            i + 1, total_files
        );

        // Try S3 first, then HF Hub
        let downloaded = if let Some(ref bucket) = config.s3_bucket {
            if let Some(ref prefix) = manifest.s3_prefix {
                let s3_url = format!(
                    "https://{bucket}.s3.{region}.amazonaws.com/{prefix}/{filename}",
                    region = config.s3_region.as_deref().unwrap_or("us-east-1"),
                    filename = file.name,
                );
                download_file(client, &s3_url, &dest, file, i, total_files, on_progress, None).await.ok()
            } else {
                None
            }
        } else {
            None
        };

        if downloaded.is_none() {
            // Fall back to Hugging Face Hub
            if let Some(ref hf_repo) = manifest.hf_repo {
                let hf_url = format!(
                    "https://huggingface.co/{}/resolve/main/{}",
                    hf_repo, file.name
                );
                download_file(
                    client, &hf_url, &dest, file, i, total_files, on_progress,
                    config.hf_token.as_deref(),
                ).await?;
            } else {
                return Err(format!(
                    "No download source for {}/{} (no S3 bucket or HF repo configured)",
                    model_id, file.name
                ).into());
            }
        }
    }

    // Save manifest to cache
    let manifest_path = local_path.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    tokio::fs::write(&manifest_path, manifest_json).await?;

    tracing::info!(model_id = %model_id, path = %local_path.display(), "Model downloaded successfully");
    Ok(local_path)
}

/// Maximum number of download retry attempts for transient failures.
const MAX_DOWNLOAD_RETRIES: u32 = 3;

/// Download a single file with progress reporting and automatic retry.
///
/// Retries up to `MAX_DOWNLOAD_RETRIES` times with exponential backoff
/// for transient failures (network errors, 5xx server errors, 429 rate limits).
async fn download_file(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    file: &ModelFile,
    file_index: usize,
    total_files: usize,
    on_progress: Option<&ProgressCallback>,
    auth_token: Option<&str>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut last_error: Option<Box<dyn std::error::Error + Send + Sync>> = None;

    for attempt in 0..=MAX_DOWNLOAD_RETRIES {
        if attempt > 0 {
            let delay = std::time::Duration::from_millis(500 * 2u64.pow(attempt - 1));
            tracing::warn!(
                url = %url,
                attempt = attempt + 1,
                delay_ms = delay.as_millis() as u64,
                "Retrying download after transient failure"
            );
            tokio::time::sleep(delay).await;
        }

        match download_file_once(client, url, dest, file, file_index, total_files, on_progress, auth_token).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                if is_retryable_error(&e) && attempt < MAX_DOWNLOAD_RETRIES {
                    tracing::warn!(url = %url, error = %e, "Transient download error, will retry");
                    last_error = Some(e);
                    continue;
                }
                return Err(e);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| "download failed after retries".into()))
}

/// Check if an error is transient and worth retrying.
fn is_retryable_error(error: &Box<dyn std::error::Error + Send + Sync>) -> bool {
    let msg = error.to_string();
    // Network-level errors
    if msg.contains("connection") || msg.contains("timeout") || msg.contains("reset")
        || msg.contains("broken pipe") || msg.contains("dns")
    {
        return true;
    }
    // HTTP 5xx / 429 in error message
    if msg.contains("500") || msg.contains("502") || msg.contains("503")
        || msg.contains("504") || msg.contains("429")
    {
        return true;
    }
    false
}

/// Single download attempt (no retry).
async fn download_file_once(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    file: &ModelFile,
    file_index: usize,
    total_files: usize,
    on_progress: Option<&ProgressCallback>,
    auth_token: Option<&str>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut request = client.get(url);
    if let Some(token) = auth_token {
        request = request.header("Authorization", format!("Bearer {token}"));
    }

    let response = request.send().await?;
    let status = response.status();

    if !status.is_success() {
        return Err(format!(
            "Download failed: {} returned {}",
            url, status
        ).into());
    }

    let mut output = tokio::fs::File::create(dest).await?;
    let mut bytes_downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        output.write_all(&chunk).await?;
        bytes_downloaded += chunk.len() as u64;

        if let Some(cb) = on_progress {
            cb(DownloadProgress {
                file_name: file.name.clone(),
                bytes_downloaded,
                bytes_total: file.size_bytes,
                file_index,
                total_files,
            });
        }
    }

    output.flush().await?;
    Ok(())
}

/// Find the model manifest — check S3, then try to build one from HF Hub API.
async fn find_manifest(
    model_id: &str,
    config: &FetchConfig,
    client: &reqwest::Client,
) -> Result<ModelManifest, Box<dyn std::error::Error + Send + Sync>> {
    // Check for manifest in S3
    if let Some(ref bucket) = config.s3_bucket {
        let s3_url = format!(
            "https://{bucket}.s3.{region}.amazonaws.com/models/{model_id}/manifest.json",
            region = config.s3_region.as_deref().unwrap_or("us-east-1"),
        );
        if let Ok(resp) = client.get(&s3_url).send().await {
            if resp.status().is_success() {
                let manifest: ModelManifest = resp.json().await?;
                return Ok(manifest);
            }
        }
    }

    // Check for manifest in local cache dir
    let local_manifest = config.cache_dir.join(model_id).join("manifest.json");
    if local_manifest.exists() {
        return read_manifest(&local_manifest).await;
    }

    // Build manifest from Hugging Face Hub API
    build_manifest_from_hf(model_id, config, client).await
}

/// Build a manifest by querying the Hugging Face Hub API.
async fn build_manifest_from_hf(
    model_id: &str,
    config: &FetchConfig,
    client: &reqwest::Client,
) -> Result<ModelManifest, Box<dyn std::error::Error + Send + Sync>> {
    // Try common HF repo naming conventions
    let hf_repo = resolve_hf_repo(model_id, config, client).await?;

    // Query the HF API for the file tree
    let api_url = format!("https://huggingface.co/api/models/{hf_repo}/tree/main");
    let mut request = client.get(&api_url);
    if let Some(ref token) = config.hf_token {
        request = request.header("Authorization", format!("Bearer {token}"));
    }

    let response = request.send().await?;
    if !response.status().is_success() {
        return Err(format!("HF API returned {} for {}", response.status(), hf_repo).into());
    }

    let tree: Vec<HfTreeEntry> = response.json().await?;

    // Filter to relevant model files
    let files: Vec<ModelFile> = tree
        .iter()
        .filter(|e| e.entry_type == "file")
        .filter(|e| is_model_file(&e.path))
        .map(|e| ModelFile {
            name: e.path.clone(),
            size_bytes: e.size.unwrap_or(0),
            sha256: e.lfs.as_ref().map(|l| l.sha256.clone()),
        })
        .collect();

    let total_size: u64 = files.iter().map(|f| f.size_bytes).sum();
    let pipeline_type = infer_pipeline_type(&files);

    Ok(ModelManifest {
        model_id: model_id.to_string(),
        hf_repo: Some(hf_repo),
        s3_prefix: Some(format!("models/{model_id}")),
        pipeline_type,
        total_size_bytes: total_size,
        files,
    })
}

/// Resolve a model_id to a Hugging Face repo path.
async fn resolve_hf_repo(
    model_id: &str,
    config: &FetchConfig,
    client: &reqwest::Client,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // If model_id contains '/', it's already a full repo path
    if model_id.contains('/') {
        return Ok(model_id.to_string());
    }

    // Common prefixes to try
    let candidates = [
        model_id.to_string(),
        format!("meta-llama/{model_id}"),
        format!("mistralai/{model_id}"),
        format!("Qwen/{model_id}"),
        format!("google/{model_id}"),
        format!("openai/{model_id}"),
        format!("stabilityai/{model_id}"),
        format!("facebook/{model_id}"),
        format!("TinyLlama/{model_id}"),
    ];

    for candidate in &candidates {
        let url = format!("https://huggingface.co/api/models/{candidate}");
        let mut request = client.get(&url);
        if let Some(ref token) = config.hf_token {
            request = request.header("Authorization", format!("Bearer {token}"));
        }

        if let Ok(resp) = request.send().await {
            if resp.status().is_success() {
                return Ok(candidate.clone());
            }
        }
    }

    Err(format!("Could not resolve HF repo for model_id: {model_id}").into())
}

/// HF API tree entry.
#[derive(Debug, Deserialize)]
struct HfTreeEntry {
    #[serde(rename = "type")]
    entry_type: String,
    path: String,
    size: Option<u64>,
    lfs: Option<HfLfsInfo>,
}

/// HF LFS info (for large files stored in Git LFS).
#[derive(Debug, Deserialize)]
struct HfLfsInfo {
    sha256: String,
}

/// Check if a file is a model-relevant file worth downloading.
fn is_model_file(path: &str) -> bool {
    let dominated = [
        ".safetensors", ".bin", ".gguf", ".pt", ".pth",
        "config.json", "tokenizer.json", "tokenizer.model",
        "tokenizer_config.json", "special_tokens_map.json",
        "generation_config.json", "preprocessor_config.json",
        "vocab.json", "merges.txt", "added_tokens.json",
        "model_index.json",
    ];
    dominated.iter().any(|ext| path.ends_with(ext))
}

/// Infer pipeline type from the list of files.
fn infer_pipeline_type(files: &[ModelFile]) -> String {
    let has_file = |name: &str| files.iter().any(|f| f.name.contains(name));

    if has_file("model_index.json") || has_file("unet") || has_file("vae") {
        "diffusion".into()
    } else if has_file("whisper") || has_file("preprocessor_config.json") {
        "whisper".into()
    } else if has_file(".gguf") {
        "llm".into()
    } else {
        "llm".into() // Default assumption
    }
}

/// Read a manifest from disk.
async fn read_manifest(
    path: &Path,
) -> Result<ModelManifest, Box<dyn std::error::Error + Send + Sync>> {
    let json = tokio::fs::read_to_string(path).await?;
    let manifest: ModelManifest = serde_json::from_str(&json)?;
    Ok(manifest)
}

/// Verify that all files in a manifest exist locally with correct sizes.
async fn verify_local_files(model_dir: &Path, manifest: &ModelManifest) -> bool {
    for file in &manifest.files {
        let path = model_dir.join(&file.name);
        match tokio::fs::metadata(&path).await {
            Ok(meta) => {
                if meta.len() != file.size_bytes {
                    return false;
                }
            }
            Err(_) => return false,
        }
    }
    true
}

/// Format bytes as a human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.0} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.0} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{bytes} B")
    }
}

/// Create a manifest for a local model directory (for uploading to S3).
pub async fn create_manifest_from_local(
    model_id: &str,
    model_dir: &Path,
    hf_repo: Option<&str>,
    pipeline_type: &str,
) -> Result<ModelManifest, Box<dyn std::error::Error + Send + Sync>> {
    let mut files = Vec::new();
    let mut total_size: u64 = 0;

    let mut entries = tokio::fs::read_dir(model_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.is_file() && is_model_file(&path.to_string_lossy()) {
            let metadata = tokio::fs::metadata(&path).await?;
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            total_size += metadata.len();
            files.push(ModelFile {
                name,
                size_bytes: metadata.len(),
                sha256: None, // Could compute but expensive for large files
            });
        }
    }

    Ok(ModelManifest {
        model_id: model_id.to_string(),
        hf_repo: hf_repo.map(|s| s.to_string()),
        s3_prefix: Some(format!("models/{model_id}")),
        pipeline_type: pipeline_type.to_string(),
        total_size_bytes: total_size,
        files,
    })
}
