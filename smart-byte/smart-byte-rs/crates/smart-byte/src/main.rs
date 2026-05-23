//! `smart-byte` — reference CLI for the Smart Byte substrate.
//!
//! Subcommands:
//!   - `envelope new`     — build a fresh envelope from CLI args
//!   - `envelope verify`  — re-derive a SAID and check it matches
//!   - `cluster simulate` — run the deterministic lockstep demo
//!
//! The lockstep demo's output is deterministic: the same `--nodes` /
//! `--frames` pair produces byte-identical output across runs.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};
use clap::{Parser, Subcommand, ValueEnum};

use smart_byte_core::{
    Cargo, Envelope, JouleCost, OwnershipChain, Provenance, Said,
};
use smart_byte_lockstep::{Cluster, synthetic_transitions};

#[derive(Parser, Debug)]
#[command(name = "smart-byte", version, about = "Smart Byte reference CLI")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Envelope operations.
    Envelope {
        #[command(subcommand)]
        sub: EnvelopeCmd,
    },
    /// Cluster operations.
    Cluster {
        #[command(subcommand)]
        sub: ClusterCmd,
    },
}

#[derive(Subcommand, Debug)]
enum EnvelopeCmd {
    /// Construct a new envelope and print its SAID + CBOR length.
    New {
        #[arg(long = "cargo-type", value_enum, default_value_t = CargoType::Usd)]
        cargo_type: CargoType,
        /// USD minor units (cents). Required for cargo-type=usd.
        #[arg(long = "amount-minor", default_value_t = 0)]
        amount_minor: i64,
        /// Microjoules. Used for cargo-type=joule-claim.
        #[arg(long = "microjoules", default_value_t = 0)]
        microjoules: u64,
        /// Optional path to write the CBOR-encoded envelope to.
        #[arg(long = "out")]
        out: Option<PathBuf>,
    },
    /// Verify the SAID of an envelope read from disk.
    Verify {
        /// Path to the CBOR-encoded envelope.
        file: PathBuf,
    },
}

#[derive(Subcommand, Debug)]
enum ClusterCmd {
    /// Run a deterministic lockstep simulation.
    Simulate {
        #[arg(long, default_value_t = 4)]
        nodes: usize,
        #[arg(long, default_value_t = 10)]
        frames: u64,
        /// Optional number of Byzantine nodes (must be < nodes / 3).
        #[arg(long = "faulty", default_value_t = 0)]
        faulty: usize,
    },
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum CargoType {
    Bytes,
    Usd,
    JouleClaim,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Envelope { sub } => match sub {
            EnvelopeCmd::New {
                cargo_type,
                amount_minor,
                microjoules,
                out,
            } => envelope_new(cargo_type, amount_minor, microjoules, out),
            EnvelopeCmd::Verify { file } => envelope_verify(file),
        },
        Cmd::Cluster { sub } => match sub {
            ClusterCmd::Simulate {
                nodes,
                frames,
                faulty,
            } => cluster_simulate(nodes, frames, faulty),
        },
    }
}

fn envelope_new(
    cargo_type: CargoType,
    amount_minor: i64,
    microjoules: u64,
    out: Option<PathBuf>,
) -> Result<()> {
    let issuer = Said::hash(b"cli-issuer");
    // Fixed issuance time so that repeated `envelope new` invocations
    // with the same flags produce the same SAID. Real issuers stamp
    // wall-clock time; the CLI's job is to demonstrate the schema, not
    // to imitate issuer behavior.
    let issued_at = Utc
        .with_ymd_and_hms(2026, 5, 23, 0, 0, 0)
        .single()
        .context("fixed timestamp is valid")?;
    let provenance = Provenance::new(issuer, issued_at, b"cli-auth".to_vec());

    let cargo = match cargo_type {
        CargoType::Bytes => Cargo::Bytes(b"demo".to_vec()),
        CargoType::Usd => Cargo::Usd {
            minor: amount_minor,
        },
        CargoType::JouleClaim => Cargo::JouleClaim { microjoules },
    };

    let envelope = Envelope::new(
        provenance,
        OwnershipChain::empty(),
        cargo,
        JouleCost::measured(microjoules),
    )?;

    let bytes = envelope.to_cbor()?;
    println!("said       : {}", envelope.id);
    println!("cargo_kind : {}", envelope.cargo.kind());
    println!("cbor_bytes : {}", bytes.len());
    if let Some(path) = out {
        fs::write(&path, &bytes).context("write envelope to disk")?;
        println!("written_to : {}", path.display());
    }
    Ok(())
}

fn envelope_verify(file: PathBuf) -> Result<()> {
    let bytes = fs::read(&file).context("read envelope")?;
    let env = Envelope::from_cbor(&bytes).context("decode envelope")?;
    env.verify_said().context("verify SAID")?;
    println!("ok  : {}", env.id);
    Ok(())
}

fn cluster_simulate(nodes: usize, frames: u64, faulty: usize) -> Result<()> {
    if faulty >= nodes {
        anyhow::bail!("--faulty must be < --nodes");
    }
    let mut cluster = if faulty == 0 {
        Cluster::new_honest(nodes)
    } else {
        Cluster::new_with_faulty(nodes, faulty)
    };
    println!(
        "cluster: nodes={} faulty={} frames={}",
        nodes, faulty, frames
    );
    for f in 0..frames {
        let transitions = synthetic_transitions(f, 3);
        let commit = cluster.step(transitions)?;
        let hex_head: String = commit
            .state_hash
            .iter()
            .take(8)
            .map(|b| format!("{:02x}", b))
            .collect();
        println!(
            "frame {:>3}: state_hash={}... committers={}/{}",
            commit.frame,
            hex_head,
            commit.committed_by.len(),
            nodes
        );
    }
    Ok(())
}
