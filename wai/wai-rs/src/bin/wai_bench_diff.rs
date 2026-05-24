//! Diff two `wai_bench --report` JSON files. Flags any per-codec
//! regression in failure count, byte ratio, or encode-time. Designed
//! for production CI: `wai_bench_diff baseline.json head.json` returns
//! non-zero exit if any threshold is exceeded.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::process::ExitCode;

#[derive(serde::Deserialize)]
struct Report {
    wai_version: String,
    timestamp_unix: u64,
    mode: String,
    host: String,
    total_seconds: f64,
    codecs: BTreeMap<String, CodecStats>,
}

#[derive(serde::Deserialize, Default, Clone)]
struct CodecStats {
    n_runs: usize,
    n_ok: usize,
    n_fail: usize,
    total_wai_bytes: u64,
    #[serde(default)]
    total_ref_bytes: u64,
    #[serde(default)]
    sum_psnr_delta: f64,
    #[serde(default)]
    n_psnr: usize,
    encode_ms: u128,
}

fn ratio_pct(a: u64, b: u64) -> f64 {
    if b == 0 { return 0.0; }
    (a as f64 / b as f64 - 1.0) * 100.0
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: wai_bench_diff <baseline.json> <head.json> [--bytes-tol PCT] [--time-tol PCT]");
        eprintln!();
        eprintln!("Compares head against baseline; non-zero exit if any codec");
        eprintln!("regressed beyond the tolerance. Defaults: bytes ±5%, time ±25%.");
        return ExitCode::from(2);
    }
    let base: Report = match fs::read_to_string(&args[0])
        .map_err(|e| e.to_string())
        .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))
    {
        Ok(r) => r,
        Err(e) => { eprintln!("baseline parse: {e}"); return ExitCode::from(2); }
    };
    let head: Report = match fs::read_to_string(&args[1])
        .map_err(|e| e.to_string())
        .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))
    {
        Ok(r) => r,
        Err(e) => { eprintln!("head parse: {e}"); return ExitCode::from(2); }
    };
    let bytes_tol: f64 = args.iter().position(|a| a == "--bytes-tol")
        .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let time_tol: f64 = args.iter().position(|a| a == "--time-tol")
        .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(25.0);

    println!("baseline: wai {} on {} @ {} ({:.1}s, mode={})",
             base.wai_version, base.host, base.timestamp_unix, base.total_seconds, base.mode);
    println!("head:     wai {} on {} @ {} ({:.1}s, mode={})",
             head.wai_version, head.host, head.timestamp_unix, head.total_seconds, head.mode);
    if base.mode != head.mode {
        eprintln!("WARNING: comparing different modes ({} vs {})", base.mode, head.mode);
    }
    println!("\n  {:14} {:>8} {:>8} {:>10} {:>10} {:>10}",
             "codec", "Δ fail", "Δ runs", "Δ bytes %", "Δ enc-ms %", "verdict");
    let mut regressed = false;
    let all: std::collections::BTreeSet<&String> =
        base.codecs.keys().chain(head.codecs.keys()).collect();
    for name in &all {
        let b = base.codecs.get(*name).cloned().unwrap_or_default();
        let h = head.codecs.get(*name).cloned().unwrap_or_default();
        let dfail = h.n_fail as i64 - b.n_fail as i64;
        let druns = h.n_runs as i64 - b.n_runs as i64;
        let dbytes = ratio_pct(h.total_wai_bytes, b.total_wai_bytes);
        let dtime = ratio_pct(h.encode_ms as u64, b.encode_ms as u64);
        let mut verdict = vec![];
        if dfail > 0 { verdict.push("MORE FAILURES"); regressed = true; }
        if dbytes > bytes_tol { verdict.push("BIGGER BYTES"); regressed = true; }
        if dtime > time_tol { verdict.push("SLOWER ENCODE"); regressed = true; }
        let verdict = if verdict.is_empty() { "ok".to_string() } else { verdict.join(", ") };
        println!("  {:14} {:>+8} {:>+8} {:>+9.1}% {:>+9.1}% {:>10}",
                 name, dfail, druns, dbytes, dtime, verdict);
    }
    if regressed {
        eprintln!("\nREGRESSION detected (bytes tol {bytes_tol}%, time tol {time_tol}%)");
        ExitCode::from(1)
    } else {
        println!("\nno regressions over tolerances (bytes {bytes_tol}%, time {time_tol}%)");
        ExitCode::SUCCESS
    }
}
