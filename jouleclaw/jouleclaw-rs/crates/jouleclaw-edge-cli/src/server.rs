//! Long-running daemon: load DeBERTa once, serve queries over a
//! Unix socket. Amortizes the model-load cost across requests —
//! subsequent queries return in roughly verify_time (sub-second
//! with parallel entailment).
//!
//! Protocol (newline-delimited JSON, one request per connection):
//!
//!   →  request:  `{"query": "...", "json": bool?, "no_verify":
//!                  bool?, "no_cache": bool?}`
//!   ←  response: the full envelope from `render::render_json` if
//!                 the request's `json` is true, otherwise the
//!                 pretty-text rendering.
//!
//! The server is stateless across requests — each query reads the
//! current cache, runs the pipeline (skipping DeBERTa load), and
//! returns the result. The cache directory is configurable per
//! request via the `cache_dir` field (falls back to the server's
//! default if absent).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};

use jouleclaw_deberta::NliEngine;
use jouleclaw_diagnose::DebertaEntailer;

use crate::cli::Options;
use crate::pipeline::{run_with_entailer, EntailerKind, PipelineError};
use crate::render;

/// Wire format for a request. Optional flags fall back to server
/// defaults when absent.
///
/// Endpoint fields, when `Some`, override the server-wide defaults
/// for this one request. The resolved endpoints are part of the
/// answer cache key (see `jouleclaw_compose::cache_key_for_plan`), so a
/// request against the public Wikidata endpoint and the same query
/// against a local mirror don't alias — fixed in lever #2 of the
/// open-items round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServeRequest {
    pub query: String,
    #[serde(default)]
    pub json: bool,
    #[serde(default)]
    pub no_verify: bool,
    #[serde(default)]
    pub no_cache: bool,
    #[serde(default)]
    pub verbose: bool,
    /// Per-request Wikidata SPARQL endpoint override. `None` falls
    /// back to the server's default; `Some(url)` is used for this
    /// request only.
    #[serde(default)]
    pub wikidata_endpoint: Option<String>,
    /// Per-request Wikipedia REST endpoint override.
    #[serde(default)]
    pub wikipedia_endpoint: Option<String>,
}

/// Wire format for an error reply.
#[derive(Debug, Clone, Serialize)]
struct ServeError {
    error: String,
}

/// Configuration for [`serve`].
pub struct ServerConfig {
    pub socket_path: PathBuf,
    pub model_dir: PathBuf,
    pub cache_dir: PathBuf,
    /// Optional Wikidata SPARQL endpoint applied to every request
    /// (server-wide; clients can't override per-request). Useful
    /// for local Oxigraph / Qlever mirrors.
    pub wikidata_endpoint: Option<String>,
    /// Optional Wikipedia REST summary endpoint.
    pub wikipedia_endpoint: Option<String>,
}

/// Spawn the listener and handle requests forever. Removes any
/// stale socket file at the configured path before binding;
/// removes the socket again on graceful shutdown.
pub async fn serve(cfg: ServerConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Bind. If a stale socket file exists (from a previous run
    // that didn't shut down cleanly), remove it first.
    if cfg.socket_path.exists() {
        eprintln!(
            "jouleclaw-edge-cli serve: removing stale socket at {}",
            cfg.socket_path.display()
        );
        std::fs::remove_file(&cfg.socket_path)?;
    }
    if let Some(parent) = cfg.socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let listener = UnixListener::bind(&cfg.socket_path)?;
    eprintln!(
        "jouleclaw-edge-cli serve: listening on {}",
        cfg.socket_path.display()
    );

    // Load the DeBERTa engine once. This is the entire point of
    // running as a server — every subsequent request reuses it.
    eprintln!("jouleclaw-edge-cli serve: loading DeBERTa from {} …", cfg.model_dir.display());
    let load_started = std::time::Instant::now();
    let engine = NliEngine::from_dir(&cfg.model_dir)?;
    let entailer: Arc<DebertaEntailer> = Arc::new(DebertaEntailer::new(engine));
    eprintln!(
        "jouleclaw-edge-cli serve: model loaded in {:.2}s — ready",
        load_started.elapsed().as_secs_f64()
    );

    let cache_dir = Arc::new(cfg.cache_dir);
    let model_dir = Arc::new(cfg.model_dir);
    let wikidata_endpoint = Arc::new(cfg.wikidata_endpoint);
    let wikipedia_endpoint = Arc::new(cfg.wikipedia_endpoint);
    if let Some(url) = wikidata_endpoint.as_deref() {
        eprintln!("jouleclaw-edge-cli serve: wikidata-endpoint={url}");
    }
    if let Some(url) = wikipedia_endpoint.as_deref() {
        eprintln!("jouleclaw-edge-cli serve: wikipedia-endpoint={url}");
    }

    loop {
        let (stream, _addr) = listener.accept().await?;
        let entailer = Arc::clone(&entailer);
        let cache_dir = Arc::clone(&cache_dir);
        let model_dir = Arc::clone(&model_dir);
        let wd = Arc::clone(&wikidata_endpoint);
        let wp = Arc::clone(&wikipedia_endpoint);
        tokio::spawn(async move {
            if let Err(e) =
                handle_one_request(stream, &entailer, &cache_dir, &model_dir, &wd, &wp)
                    .await
            {
                eprintln!("jouleclaw-edge-cli serve: request error: {e}");
            }
        });
    }
}

async fn handle_one_request(
    mut stream: UnixStream,
    entailer: &DebertaEntailer,
    cache_dir: &Path,
    model_dir: &Path,
    wikidata_endpoint: &Option<String>,
    wikipedia_endpoint: &Option<String>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut buf = Vec::with_capacity(2048);
    let mut tmp = [0u8; 1024];
    // Read until newline (one request per connection).
    loop {
        let n = stream.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.contains(&b'\n') {
            break;
        }
    }
    let line = match std::str::from_utf8(&buf) {
        Ok(s) => s.trim_end_matches(['\n', '\r']),
        Err(e) => {
            let body = serde_json::to_string(&ServeError {
                error: format!("invalid utf-8: {e}"),
            })?;
            stream.write_all(body.as_bytes()).await?;
            stream.write_all(b"\n").await?;
            return Ok(());
        }
    };
    let req: ServeRequest = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(e) => {
            let body = serde_json::to_string(&ServeError {
                error: format!("invalid request json: {e}"),
            })?;
            stream.write_all(body.as_bytes()).await?;
            stream.write_all(b"\n").await?;
            return Ok(());
        }
    };

    // Endpoint merge: per-request override wins over server default
    // when present, server default otherwise. The resolved value is
    // hashed into the answer cache key by `cache_key_for_plan`, so
    // distinct endpoints can never alias to the same cached envelope.
    let resolved_wikidata = req
        .wikidata_endpoint
        .clone()
        .or_else(|| wikidata_endpoint.clone());
    let resolved_wikipedia = req
        .wikipedia_endpoint
        .clone()
        .or_else(|| wikipedia_endpoint.clone());

    let opts = Options {
        query: req.query.clone(),
        json: req.json,
        no_verify: req.no_verify,
        model_dir: model_dir.to_path_buf(),
        cache_dir: cache_dir.to_path_buf(),
        no_cache: req.no_cache,
        verbose: req.verbose,
        wikidata_endpoint: resolved_wikidata,
        wikipedia_endpoint: resolved_wikipedia,
    };

    let log: Box<dyn Fn(&str) + Send + Sync> = if req.verbose {
        Box::new(|s: &str| eprintln!("    [req] {s}"))
    } else {
        Box::new(|_: &str| {})
    };

    let outcome = if req.no_verify {
        // Reuse the existing inline build path for the no-verify
        // case (it's already permissive FixtureEntailer and cheap).
        crate::pipeline::run(&opts, log.as_ref()).await
    } else {
        run_with_entailer(&opts, entailer, EntailerKind::Deberta, log.as_ref()).await
    };

    match outcome {
        Ok(output) => {
            let body = if req.json {
                render::render_json_to_string(&output)
            } else {
                render::render_text_to_string(&output)
            };
            stream.write_all(body.as_bytes()).await?;
            if !body.ends_with('\n') {
                stream.write_all(b"\n").await?;
            }
        }
        Err(PipelineError::Execute(_))
        | Err(PipelineError::Plan(_))
        | Err(PipelineError::Verify(_))
        | Err(PipelineError::Draft(_))
        | Err(PipelineError::Atomize(_))
        | Err(PipelineError::Compose(_))
        | Err(PipelineError::Understanding(_))
        | Err(PipelineError::LoadModel(_)) => {
            let body = serde_json::to_string(&ServeError {
                error: format!("pipeline error: {:?}", "see server log"),
            })?;
            stream.write_all(body.as_bytes()).await?;
            stream.write_all(b"\n").await?;
        }
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────
// Client side
// ──────────────────────────────────────────────────────────────────

/// Client-side helper: send a request over a Unix socket and print
/// the response to stdout. Used by `jouleclaw-edge-cli --socket ...`.
pub async fn send_query(
    socket: &Path,
    req: &ServeRequest,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mut stream = UnixStream::connect(socket).await?;
    let line = serde_json::to_string(req)?;
    stream.write_all(line.as_bytes()).await?;
    stream.write_all(b"\n").await?;
    let mut out = Vec::with_capacity(8192);
    let mut tmp = [0u8; 8192];
    loop {
        let n = stream.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        out.extend_from_slice(&tmp[..n]);
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}
