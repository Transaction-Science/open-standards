//! # jclaw — JouleClaw's Pi-class minimal harness
//!
//! The CLI binary an operator runs to drive a JouleClaw cascade
//! interactively. Mario Zechner's `pi` proved that a tiny tool surface
//! (read / write / edit / bash) + tree-forked sessions + hot-reload
//! extensions is enough to support sophisticated agentic work; jclaw
//! adopts that design but adds the load-bearing JouleClaw addition:
//! every action accounts joules against an explicit budget, and the
//! breaker trips when the budget exhausts.
//!
//! ## v0.1 scope
//!
//! - Subcommands: `version`, `tier-walk`, `meter`, `receipt`
//! - Reads + prints the four cascade tiers (`L0:Cache`, `L1:Lawful`,
//!   `L2:Embed`, `L3:Model`, `L4:Wire`) and their joule classes
//! - Emits a Smart-Byte-compatible `jouleclaw-prov::Receipt` per run
//! - Hook-shape compatible with Claude Code (`PreToolUse` /
//!   `PostToolUse` / `Stop` JSON over stdin)
//!
//! ## Later
//!
//! - Hot-reload Rust extensions (cargo-script style)
//! - Tree-forked session history
//! - Full MCP client (depends on `rmcp` downstream)
//! - The four Pi-class tool primitives (read/write/edit/bash) wired
//!   through `jouleclaw-mcp::dispatch_metered`
//!
//! Run `jclaw --help` for the current command surface.

use clap::{Parser, Subcommand};
use jouleclaw_prov::{input_hash, CascadeTier, ReceiptBuilder, ToolTouch};
use jouleclaw_energy::Provenance;

/// jclaw — JouleClaw's minimal energy-metered harness.
#[derive(Parser, Debug)]
#[command(name = "jclaw", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Print the JouleClaw runtime version + the five cascade tiers
    /// with their nominal energy classes.
    Version,
    /// Walk the cascade for an input string and print which tier
    /// (would have) closed the query. v0.1 ships a heuristic walker;
    /// the full router lands when `jouleclaw-router` ports.
    TierWalk {
        /// The input string to resolve.
        input: String,
        /// Maximum allowed energy in microjoules. The walker trips
        /// the breaker if any single tier exceeds this.
        #[arg(long, default_value_t = 100_000)]
        budget_uj: u64,
    },
    /// Emit a sample receipt that closed at the named tier.
    Receipt {
        /// L0 / L1 / L2 / L3 / L4 (case-insensitive).
        #[arg(long)]
        tier: String,
        /// The input that was resolved.
        #[arg(long)]
        input: String,
    },
    /// Print the active energy counter's honest resolution + minimum
    /// window. Useful for verifying a deployment's circuit breaker
    /// can enforce at the granularity the operator expects.
    Meter,
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match run(cli).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("jclaw: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error("unknown tier `{0}` — expected L0, L1, L2, L3, or L4")]
    UnknownTier(String),
    #[error("receipt seal: {0}")]
    Seal(#[from] jouleclaw_prov::ReceiptError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

async fn run(cli: Cli) -> Result<(), CliError> {
    match cli.cmd {
        Cmd::Version => {
            println!("jclaw {} — JouleClaw reference harness", env!("CARGO_PKG_VERSION"));
            println!();
            println!("Cascade tiers (nominal joule class per resolution):");
            println!("  L0:Cache    picojoules     content-addressed hit");
            println!("  L1:Lawful   nanojoules     deterministic primitive (text/code)");
            println!("  L2:Embed    sub-mJ         Matryoshka + hybrid search");
            println!("  L3:Model    joules         local SSM / ternary / multimodal / diffusion");
            println!("  L4:Wire     tens of J      remote frontier (escape hatch)");
        }
        Cmd::TierWalk { input, budget_uj } => {
            // v0.1 heuristic: deterministic-looking inputs (math, dates,
            // unit conversions) hit L1; question-shaped inputs go to L2;
            // generation-shaped inputs (write/draw/sing/animate) go to
            // L3. Real routing arrives when jouleclaw-router lands.
            let tier = heuristic_tier(&input);
            let nominal_uj = nominal_cost_uj(tier);
            println!("input        {input}");
            println!("tier (heuristic)  {} ({})", tier.wire_tag(), tier.name());
            println!("nominal cost      {nominal_uj} μJ");
            println!("budget            {budget_uj} μJ");
            if nominal_uj > budget_uj {
                println!("breaker           ⚠ would trip: nominal > budget");
            } else {
                println!("breaker           ✓ within budget");
            }
        }
        Cmd::Receipt { tier, input } => {
            let tier = parse_tier(&tier)?;
            let receipt = ReceiptBuilder::new()
                .input_hash(input_hash(&input))
                .tier(tier)
                .account_tool(ToolTouch {
                    tool_id: format!("demo:{}", tier.wire_tag().to_lowercase()),
                    joules_uj: nominal_cost_uj(tier),
                    energy_provenance: Provenance::Estimator,
                })
                .seal()?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        Cmd::Meter => {
            // v0.1 reports the protocol intent without binding to a
            // specific counter backend (those are feature-gated and
            // platform-specific). Once a deployment wires the right
            // EnergyCounter impl, this command reads the live one.
            println!("Energy counter (not bound in v0.1)");
            println!("  resolution   varies by platform: 1 μJ on Intel/AMD RAPL,");
            println!("               ~1 mJ on Apple IOReport (model-based),");
            println!("               ~1 mJ on NVIDIA NVML,");
            println!("               ~10 mW on Jetson INA3221 (shunt; integrate).");
            println!("  honesty      `Provenance::HwShunt` on RAPL/NVML/INA3221,");
            println!("               `Provenance::ModelBased` on Apple IOReport / ROCm SMI,");
            println!("               `Provenance::Estimator` on consumer AMD GPU / ARM PMU.");
            println!();
            println!("Bind a counter at startup with `jouleclaw_energy::EnergyCounter`.");
        }
    }
    Ok(())
}

/// Parse `L0` / `l0` / `0` / `cache` / etc. into a [`CascadeTier`].
fn parse_tier(s: &str) -> Result<CascadeTier, CliError> {
    let n = s.trim().to_ascii_lowercase();
    let n = n.strip_prefix("l").unwrap_or(&n);
    match n {
        "0" | "cache" => Ok(CascadeTier::L0Cache),
        "1" | "lawful" => Ok(CascadeTier::L1Lawful),
        "2" | "embed" => Ok(CascadeTier::L2Embed),
        "3" | "model" => Ok(CascadeTier::L3Model),
        "4" | "wire" => Ok(CascadeTier::L4Wire),
        _ => Err(CliError::UnknownTier(s.to_string())),
    }
}

/// Nominal joule cost per tier for the v0.1 heuristic walker.
/// The real cost comes from `jouleclaw-pack` measurements once a
/// model is loaded — this is the placeholder for the CLI demo.
fn nominal_cost_uj(tier: CascadeTier) -> u64 {
    match tier {
        CascadeTier::L0Cache => 1,
        CascadeTier::L1Lawful => 100,
        CascadeTier::L2Embed => 10_000,
        CascadeTier::L3Model => 5_000_000,
        CascadeTier::L4Wire => 30_000_000,
    }
}

/// Toy router — picks a tier from input shape. Production routing
/// arrives when `jouleclaw-router` ports.
fn heuristic_tier(input: &str) -> CascadeTier {
    let q = input.trim();
    if q.is_empty() {
        return CascadeTier::L0Cache;
    }
    let lower = q.to_ascii_lowercase();
    // Generation-shaped → model.
    for kw in ["draw", "paint", "render", "animate", "sing", "compose",
               "write a", "generate", "create"] {
        if lower.contains(kw) {
            return CascadeTier::L3Model;
        }
    }
    // Arithmetic / unit conversion / date → lawful.
    if q.chars().all(|c| c.is_ascii_digit() || "+-*/.()= ".contains(c)) && !q.is_empty() {
        return CascadeTier::L1Lawful;
    }
    for kw in ["gcd", "lcm", "factor", "convert", "what time"] {
        if lower.contains(kw) {
            return CascadeTier::L1Lawful;
        }
    }
    // Question-shaped → embed (retrieval first).
    if q.ends_with('?') || lower.starts_with("what") || lower.starts_with("who")
        || lower.starts_with("where") || lower.starts_with("when") {
        return CascadeTier::L2Embed;
    }
    // Default to model — but with the breaker watching the budget.
    CascadeTier::L3Model
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_parse_accepts_all_forms() {
        assert_eq!(parse_tier("L0").unwrap(), CascadeTier::L0Cache);
        assert_eq!(parse_tier("l1").unwrap(), CascadeTier::L1Lawful);
        assert_eq!(parse_tier("2").unwrap(), CascadeTier::L2Embed);
        assert_eq!(parse_tier("model").unwrap(), CascadeTier::L3Model);
        assert_eq!(parse_tier("wire").unwrap(), CascadeTier::L4Wire);
        assert!(parse_tier("L99").is_err());
    }

    #[test]
    fn heuristic_arithmetic_to_lawful() {
        assert_eq!(heuristic_tier("12 + 8"), CascadeTier::L1Lawful);
        assert_eq!(heuristic_tier("gcd(12, 8)"), CascadeTier::L1Lawful);
    }

    #[test]
    fn heuristic_generation_to_model() {
        assert_eq!(heuristic_tier("draw a cat"), CascadeTier::L3Model);
        assert_eq!(heuristic_tier("write a poem"), CascadeTier::L3Model);
    }

    #[test]
    fn heuristic_question_to_embed() {
        assert_eq!(heuristic_tier("what is the capital of france?"), CascadeTier::L2Embed);
    }

    #[test]
    fn nominal_costs_grow_by_tier() {
        let l0 = nominal_cost_uj(CascadeTier::L0Cache);
        let l1 = nominal_cost_uj(CascadeTier::L1Lawful);
        let l2 = nominal_cost_uj(CascadeTier::L2Embed);
        let l3 = nominal_cost_uj(CascadeTier::L3Model);
        let l4 = nominal_cost_uj(CascadeTier::L4Wire);
        assert!(l0 < l1 && l1 < l2 && l2 < l3 && l3 < l4);
    }
}
