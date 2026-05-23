//! `eoc` — reference CLI for the EOC four-stage cascade.
//!
//! Subcommands:
//!
//! - `eoc query "<prompt>"` — run a prompt through the cascade and print
//!   the stage that resolved it plus the attributed joule cost.
//! - `eoc meter` — detect available hardware energy counters and report
//!   the micro-joules consumed during one second of idle.

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use eoc_cache::LruCache;
use eoc_cascade::Cascade;
use eoc_core::Query;
use eoc_graph::{GraphStage, Triple};
use eoc_kv::{KvBackend, KvStage, MemoryKvBackend};
use eoc_meter::detect;
use eoc_neural::{EchoBackend, NeuralStage};

#[derive(Parser, Debug)]
#[command(
    name = "eoc",
    version,
    about = "EOC — Energy-Optimized Compute reference CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run a query through the cascade.
    Query {
        /// The prompt to resolve.
        prompt: String,
    },
    /// Detect the joule counter and report idle micro-joules per second.
    Meter,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Query { prompt } => run_query(prompt).await,
        Cmd::Meter => run_meter().await,
    }
}

fn build_cascade() -> Cascade {
    let cache = Arc::new(LruCache::new(1024));

    let kv_backend = Box::new(MemoryKvBackend::new());
    // Seed a few exact-match KV entries for demo purposes.
    kv_backend.put("ping", b"pong".to_vec());
    kv_backend.put("what is 2+2", b"4".to_vec());
    let kv = Arc::new(KvStage::new(kv_backend));

    let graph = Arc::new(GraphStage::new());
    graph.extend([
        Triple::new("Paris", "capital of", "France"),
        Triple::new("Tokyo", "capital of", "Japan"),
        Triple::new("Mars", "fourth planet from", "the Sun"),
    ]);

    let neural = Arc::new(NeuralStage::new(Box::new(
        EchoBackend::new().with_prefix("[neural fallback] "),
    )));

    Cascade::new(cache, kv, graph, neural).with_meter(Arc::from(detect()))
}

async fn run_query(prompt: String) -> Result<()> {
    let cascade = build_cascade();
    let q = Query::new(prompt);
    let r = cascade.resolve(q).await;
    println!("stage  : {}", r.stage);
    println!("payload: {}", r.payload);
    println!("cost   : {}", r.joule_cost);
    println!("receipt: {}", r.receipt);
    Ok(())
}

async fn run_meter() -> Result<()> {
    let counter = detect();
    println!("counter: {}", counter.name());
    let start = counter.read_microjoules().unwrap_or(0);
    tokio::time::sleep(Duration::from_secs(1)).await;
    let end = counter.read_microjoules().unwrap_or(0);
    let delta = end.saturating_sub(start);
    println!("idle   : {} µJ over 1.0 s ({:.3} W)", delta, (delta as f64) / 1_000_000.0);
    Ok(())
}
