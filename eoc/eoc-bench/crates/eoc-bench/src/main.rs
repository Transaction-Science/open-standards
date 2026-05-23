//! `eoc-bench` — CLI driver for the EOC benchmark harness.
//!
//! ```text
//! eoc-bench run --suite {mt|router|all}     # run a suite, print BenchReport JSON
//! eoc-bench compare --baseline a.json --candidate b.json
//! ```
//!
//! `run` is only available when the binary is built with the `eoc-rs`
//! feature (default on). `compare` works either way — it only needs the
//! report types from `eoc-bench-runner`.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use eoc_bench_runner::BenchReport;

#[derive(Parser, Debug)]
#[command(
    name = "eoc-bench",
    version,
    about = "EOC benchmark harness — joules per MT-Bench point"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run a benchmark suite against the cascade and print a JSON report.
    Run {
        /// Which suite to run.
        #[arg(long, value_enum, default_value_t = Suite::All)]
        suite: Suite,
        /// Optional output file. Defaults to stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Compare two reports and print the percent deltas that matter
    /// (joules-per-correct, latency-p95, accuracy).
    Compare {
        /// Baseline report.
        #[arg(long)]
        baseline: PathBuf,
        /// Candidate report.
        #[arg(long)]
        candidate: PathBuf,
    },
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Suite {
    Mt,
    Router,
    All,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Run { suite, out } => run_cmd(suite, out).await,
        Cmd::Compare {
            baseline,
            candidate,
        } => compare_cmd(&baseline, &candidate),
    }
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

#[cfg(feature = "eoc-rs")]
async fn run_cmd(suite: Suite, out: Option<PathBuf>) -> Result<()> {
    use std::sync::Arc;

    use eoc_bench_runner::{aggregate, run, BenchCase};
    use eoc_cache::LruCache;
    use eoc_cascade::Cascade;
    use eoc_graph::{GraphStage, Triple};
    use eoc_kv::{KvBackend, KvStage, MemoryKvBackend};
    use eoc_neural::{EchoBackend, NeuralStage};

    let cases: Vec<BenchCase> = match suite {
        Suite::Mt => eoc_bench_mt::load(),
        Suite::Router => eoc_bench_router::load(),
        Suite::All => {
            let mut v = eoc_bench_mt::load();
            v.extend(eoc_bench_router::load());
            v
        }
    };

    let cache = Arc::new(LruCache::new(1024));

    // Pre-populate the KV stage with a handful of well-known facts so
    // router cases that *should* land at KV actually land at KV under
    // the reference cascade. This is deliberate fixture data — real
    // deployments wire real backends.
    let kv_backend: Box<dyn KvBackend> = Box::new(MemoryKvBackend::new());
    let kv_seed: &[(&str, &str)] = &[
        ("speed of light in m/s", "299792458"),
        ("default port for HTTPS", "443"),
        ("default port for SSH", "22"),
        ("atomic number of carbon", "6"),
        ("ISO 3166-1 alpha-2 code for Japan", "JP"),
        ("number of bytes in a TCP/IPv4 header (no options)", "20"),
        ("SI prefix for 10^-6", "micro"),
    ];
    for (k, v) in kv_seed {
        kv_backend.put(k, v.as_bytes().to_vec());
    }
    let kv = Arc::new(KvStage::new(kv_backend));

    let graph = Arc::new(GraphStage::new());
    graph.extend([
        Triple::new("Mars", "fourth planet from", "the Sun"),
        Triple::new("Hamlet", "written by", "Shakespeare"),
        Triple::new("Japan", "capital", "Tokyo"),
        Triple::new("Au", "chemical symbol of", "gold"),
        Triple::new("Paris", "river running through", "Seine"),
        Triple::new("Apollo 11", "landed on the Moon in", "1969"),
        Triple::new("The Origin of Species", "author", "Darwin"),
        Triple::new("M1 chip", "designed by", "Apple"),
    ]);

    let neural = Arc::new(NeuralStage::new(Box::new(
        EchoBackend::new().with_cost(50_000_000),
    )));

    let cascade = Cascade::new(cache, kv, graph, neural);

    let results = run(&cases, &cascade).await;
    let report = aggregate(&results);

    let json = serde_json::to_string_pretty(&report)
        .context("failed to serialise BenchReport as JSON")?;
    match out {
        Some(path) => {
            fs::write(&path, &json)
                .with_context(|| format!("failed to write report to {}", path.display()))?;
        }
        None => println!("{json}"),
    }
    Ok(())
}

#[cfg(not(feature = "eoc-rs"))]
async fn run_cmd(_suite: Suite, _out: Option<PathBuf>) -> Result<()> {
    anyhow::bail!(
        "eoc-bench was built without the `eoc-rs` feature; \
         rebuild with `cargo build --features eoc-rs` to enable `run`."
    )
}

// ---------------------------------------------------------------------------
// compare
// ---------------------------------------------------------------------------

fn compare_cmd(baseline: &PathBuf, candidate: &PathBuf) -> Result<()> {
    let baseline = read_report(baseline)?;
    let candidate = read_report(candidate)?;
    let delta = ReportDelta::from_pair(&baseline, &candidate);
    let json =
        serde_json::to_string_pretty(&delta).context("failed to serialise comparison as JSON")?;
    println!("{json}");
    Ok(())
}

fn read_report(path: &PathBuf) -> Result<BenchReport> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read report file {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse report file {} as BenchReport", path.display()))
}

#[derive(Debug, serde::Serialize)]
struct ReportDelta {
    baseline: ReportSummary,
    candidate: ReportSummary,
    /// Percent delta in joules-per-correct. Negative is better.
    joules_per_correct_pct: f64,
    /// Percent delta in p95 latency. Negative is better.
    latency_p95_pct: f64,
    /// Absolute delta in accuracy percentage points. Positive is better.
    accuracy_pp_delta: f64,
}

#[derive(Debug, serde::Serialize)]
struct ReportSummary {
    joules_per_correct: f64,
    latency_p95_ms: u64,
    accuracy_pct: f64,
}

impl ReportDelta {
    fn from_pair(baseline: &BenchReport, candidate: &BenchReport) -> Self {
        Self {
            baseline: summarise(baseline),
            candidate: summarise(candidate),
            joules_per_correct_pct: pct_delta(
                baseline.joules_per_correct,
                candidate.joules_per_correct,
            ),
            latency_p95_pct: pct_delta(
                baseline.latency_p95_ms as f64,
                candidate.latency_p95_ms as f64,
            ),
            accuracy_pp_delta: candidate.accuracy_pct - baseline.accuracy_pct,
        }
    }
}

fn summarise(r: &BenchReport) -> ReportSummary {
    ReportSummary {
        joules_per_correct: r.joules_per_correct,
        latency_p95_ms: r.latency_p95_ms,
        accuracy_pct: r.accuracy_pct,
    }
}

fn pct_delta(baseline: f64, candidate: f64) -> f64 {
    if !baseline.is_finite() || baseline == 0.0 {
        return f64::NAN;
    }
    (candidate - baseline) * 100.0 / baseline
}
