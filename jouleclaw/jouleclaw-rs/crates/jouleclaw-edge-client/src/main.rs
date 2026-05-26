//! `jouleclaw-edge-client` — minimal client for a running
//! `jouleclaw-edge-cli serve` daemon.
//!
//! Why a separate binary? The full `jouleclaw-edge-cli` links in
//! tokenizers + safetensors + half + memmap2 + chrono + ... +
//! Apple Accelerate, all of which take ~600 ms of dyld time on
//! startup *even for a cache hit*. The client doesn't actually
//! need any of those — just a Unix socket and a JSON encoder.
//! This crate ships exactly that.
//!
//! USAGE
//!     jouleclaw-edge-client [OPTIONS] <QUERY>...
//!
//! OPTIONS
//!     --socket <PATH>           server socket
//!                               (default: $HOME/.cache/joule-edge/server.sock)
//!     --json                    ask the server for JSON output
//!     --no-verify               ask the server to skip DeBERTa
//!     --no-cache                ask the server to skip the answer cache
//!     --verbose                 ask the server to log per-stage progress
//!     --wikidata-endpoint URL   override the server's Wikidata SPARQL
//!                               endpoint for this request only
//!     --wikipedia-endpoint URL  override the server's Wikipedia REST
//!                               endpoint for this request only
//!     -h, --help                show this help and exit
//!
//! The flags map 1:1 onto the wire `ServeRequest` schema. See
//! `jouleclaw-edge-cli/src/server.rs` for the canonical definition;
//! kept inline here to avoid pulling that crate's heavy deps.
//!
//! Endpoint overrides are safe to mix: the answer cache key is
//! computed against the *resolved* endpoint, so a query against
//! `query.wikidata.org` and the same query against a local mirror
//! land in distinct cache entries.

use std::path::PathBuf;

use serde::Serialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

/// Wire request schema — must stay in sync with
/// `jouleclaw-edge-cli/src/server.rs::ServeRequest`.
#[derive(Debug, Clone, Serialize)]
struct ServeRequest {
    query: String,
    #[serde(default)]
    json: bool,
    #[serde(default)]
    no_verify: bool,
    #[serde(default)]
    no_cache: bool,
    #[serde(default)]
    verbose: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    wikidata_endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    wikipedia_endpoint: Option<String>,
}

const HELP: &str = "\
jouleclaw-edge-client — minimal client for `jouleclaw-edge-cli serve`

USAGE
    jouleclaw-edge-client [OPTIONS] <QUERY>...

OPTIONS
    --socket <PATH>           server socket
                              (default: $HOME/.cache/joule-edge/server.sock)
    --json                    request JSON output from the server
    --no-verify               skip DeBERTa entailment on the server side
    --no-cache                skip the server's answer cache
    --verbose                 server logs per-stage progress to its stderr
    --wikidata-endpoint URL   override the server's Wikidata endpoint
                              for this request only
    --wikipedia-endpoint URL  override the server's Wikipedia endpoint
                              for this request only
    -h, --help                show this help and exit

This binary has no DeBERTa / safetensors / tokenizers
dependencies and starts in ~50 ms. For the full inline pipeline
(load DeBERTa, run plan → execute → diagnose → compose), use
`jouleclaw-edge-cli` directly.

Endpoint overrides are safe to mix: the answer cache key includes
the resolved endpoint so the public Wikidata + a local mirror
land in distinct cache entries.
";

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut socket = default_socket_path();
    let mut json = false;
    let mut no_verify = false;
    let mut no_cache = false;
    let mut verbose = false;
    let mut wikidata_endpoint: Option<String> = None;
    let mut wikipedia_endpoint: Option<String> = None;
    let mut tokens: Vec<String> = Vec::new();

    let mut iter = argv.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                return;
            }
            "--socket" => match iter.next() {
                Some(v) => socket = PathBuf::from(v),
                None => {
                    eprintln!("--socket requires a value\n");
                    eprint!("{HELP}");
                    std::process::exit(2);
                }
            },
            "--json" => json = true,
            "--no-verify" => no_verify = true,
            "--no-cache" => no_cache = true,
            "--verbose" => verbose = true,
            "--wikidata-endpoint" => match iter.next() {
                Some(v) => wikidata_endpoint = Some(v),
                None => {
                    eprintln!("--wikidata-endpoint requires a URL\n");
                    eprint!("{HELP}");
                    std::process::exit(2);
                }
            },
            "--wikipedia-endpoint" => match iter.next() {
                Some(v) => wikipedia_endpoint = Some(v),
                None => {
                    eprintln!("--wikipedia-endpoint requires a URL\n");
                    eprint!("{HELP}");
                    std::process::exit(2);
                }
            },
            s if s.starts_with("--") => {
                eprintln!("unknown flag: {s}\n");
                eprint!("{HELP}");
                std::process::exit(2);
            }
            _ => tokens.push(arg),
        }
    }
    if tokens.is_empty() {
        eprintln!("no query supplied\n");
        eprint!("{HELP}");
        std::process::exit(2);
    }
    let req = ServeRequest {
        query: tokens.join(" "),
        json,
        no_verify,
        no_cache,
        verbose,
        wikidata_endpoint,
        wikipedia_endpoint,
    };

    // Single-thread runtime — we only do one socket round-trip.
    // current_thread is ~10× cheaper to init than multi_thread.
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: cannot build tokio runtime: {e}");
            std::process::exit(1);
        }
    };

    let code = rt.block_on(async {
        match send_query(&socket, &req).await {
            Ok(body) => {
                use std::io::Write;
                let stdout = std::io::stdout();
                let mut lock = stdout.lock();
                let _ = lock.write_all(body.as_bytes());
                if !body.ends_with('\n') {
                    let _ = lock.write_all(b"\n");
                }
                0
            }
            Err(e) => {
                eprintln!("client error: {e}");
                1
            }
        }
    });
    std::process::exit(code);
}

fn default_socket_path() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("joule-edge")
            .join("server.sock");
    }
    PathBuf::from("/tmp/joule-edge.sock")
}

async fn send_query(
    socket: &std::path::Path,
    req: &ServeRequest,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut stream = UnixStream::connect(socket).await?;
    let line = serde_json::to_string(req)?;
    stream.write_all(line.as_bytes()).await?;
    stream.write_all(b"\n").await?;
    // Half-close so the server reads EOF after our request and
    // starts sending the response without waiting for more input.
    stream.shutdown().await?;
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
