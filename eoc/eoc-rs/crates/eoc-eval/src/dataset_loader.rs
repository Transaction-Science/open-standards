//! HuggingFace Hub dataset fetcher.
//!
//! Behind the `download` feature. Caches to
//! `~/.eoc/datasets/<repo-slug>/<split>.json`.

#![cfg(feature = "download")]

use std::path::PathBuf;

use crate::error::{EvalError, Result};

/// Resolve the cache directory: `$HOME/.eoc/datasets/<repo-slug>/`.
fn cache_dir(repo: &str) -> PathBuf {
    let mut p = match std::env::var_os("HOME") {
        Some(h) => PathBuf::from(h),
        None => PathBuf::from("."),
    };
    p.push(".eoc");
    p.push("datasets");
    p.push(repo.replace('/', "__"));
    p
}

/// Fetch the JSON-Lines split of a HuggingFace dataset. Returns the raw
/// text. Cached at `~/.eoc/datasets/<repo-slug>/<split>.json` on first
/// fetch.
pub async fn fetch_huggingface(repo: &str, split: &str) -> Result<String> {
    let dir = cache_dir(repo);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| EvalError::Io {
            path: dir.clone(),
            source: e,
        })?;
    let mut cache_path = dir.clone();
    cache_path.push(format!("{split}.json"));

    if cache_path.exists() {
        return tokio::fs::read_to_string(&cache_path)
            .await
            .map_err(|e| EvalError::Io {
                path: cache_path,
                source: e,
            });
    }

    let url = format!(
        "https://datasets-server.huggingface.co/rows?dataset={repo}&config=default&split={split}&offset=0&length=100"
    );
    let body = reqwest::get(&url)
        .await
        .map_err(|e| EvalError::Fetch(format!("GET {url}: {e}")))?
        .error_for_status()
        .map_err(|e| EvalError::Fetch(format!("status: {e}")))?
        .text()
        .await
        .map_err(|e| EvalError::Fetch(format!("read body: {e}")))?;

    tokio::fs::write(&cache_path, &body)
        .await
        .map_err(|e| EvalError::Io {
            path: cache_path,
            source: e,
        })?;
    Ok(body)
}
