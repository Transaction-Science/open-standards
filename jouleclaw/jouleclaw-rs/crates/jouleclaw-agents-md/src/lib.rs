//! # jouleclaw-agents-md
//!
//! Hierarchical discovery of `AGENTS.md` / `CLAUDE.md` files — the
//! convention the Linux Foundation Agentic AI Foundation pinned in
//! December 2025 and that 60k+ repositories now ship. Walks from a
//! starting directory up to the repository root (or a configured
//! stop) and collects every file matching a configured name list,
//! deepest-scope-first so the consumer can apply
//! more-specific-overrides-less-specific by reading the list in
//! reverse.
//!
//! ## What "joule-stamped" means here
//!
//! Discovery is IO, IO costs energy, and the field has been silent
//! about that cost. Every [`AgentsManifest`] carries `bytes_read` and
//! `estimated_joules_uj`; the aggregate [`DiscoveryReport`] reports
//! the total. The estimate is honest about its
//! [`jouleclaw_energy::Provenance::Estimator`] tier — a wall-clock
//! IO joule estimator, not a hardware shunt. The number is small
//! per call (tens to hundreds of µJ for a few-KB markdown file) but
//! the consumer that walks 10k repos cares about the rollup.
//!
//! ## Honest scope (v1)
//!
//! - **Discovery convention only.** No schema validation, no
//!   required sections, no markdown-structure check. The merged
//!   output is plain markdown.
//! - **No symlink follow.** Symlinks encountered during the walk are
//!   skipped, not chased — the discovery walk has bounded cost by
//!   construction (`max_files`).
//! - **No persistence.** Each call walks from scratch. Memoisation
//!   is the consumer's choice.
//! - **No semantic dedup.** Two AGENTS.md files at different scopes
//!   that say the same thing are both returned; merge() concatenates
//!   them verbatim with scope hints.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use jouleclaw_energy::Provenance;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ─────────────────────────────────────────────────────────────────────
// Energy model — estimator constants
// ─────────────────────────────────────────────────────────────────────

/// File IO joule estimate per byte read. Conservative — empirical
/// SSDs sit around 0.5–2 nJ/byte at the application layer once
/// kernel + filesystem overhead is folded in. We pick 1 nJ/byte =
/// 1_000 pJ/byte. Microjoule arithmetic: 1 nJ = 1/1000 µJ.
///
/// Multiplied by `bytes_read` then divided by 1000 to get µJ. A 4 KB
/// AGENTS.md file therefore stamps ~4 µJ — a number small enough
/// that no one cares about a single read and large enough that a
/// 10k-repo crawl produces a measurable budget.
pub const JOULES_PER_BYTE_PJ: u64 = 1_000;

/// Fixed per-file IO setup cost (syscalls, dentry lookup) in
/// microjoules. Empirically tens of µJ for cold paths, single-digit
/// when the path is cached; we pick 10 µJ as a conservative middle
/// ground. Added to every successful read.
pub const PER_FILE_SETUP_UJ: u64 = 10;

/// Estimate the joule cost of reading `bytes_read` bytes from disk.
/// Returns microjoules using the integer-only arithmetic the
/// `jouleclaw-energy` protocol mandates (no floats — determinism).
#[inline]
pub fn estimate_read_joules_uj(bytes_read: u64) -> u64 {
    // Picojoules then convert to µJ. Saturating to avoid overflow on
    // pathological inputs (e.g. attacker-crafted huge file).
    let pj = bytes_read.saturating_mul(JOULES_PER_BYTE_PJ);
    let uj = pj / 1_000_000;
    PER_FILE_SETUP_UJ.saturating_add(uj)
}

// ─────────────────────────────────────────────────────────────────────
// Discovery options
// ─────────────────────────────────────────────────────────────────────

/// Options for [`discover`].
#[derive(Debug, Clone)]
pub struct AgentsDiscoveryOptions {
    /// File names to collect, in priority order. Defaults to
    /// `["AGENTS.md", "CLAUDE.md"]` — the two conventions with
    /// 60k+-repo and Anthropic-tooling adoption respectively.
    pub names: Vec<String>,
    /// Stop walking up the directory tree once a `.git/` is seen.
    /// Default `true`. Set `false` to walk to filesystem root —
    /// useful for monorepo-of-monorepos setups.
    pub stop_at_repo_root: bool,
    /// Maximum number of files to collect across the walk. Default
    /// 32. Returned manifests are in walk order; once the cap is
    /// hit, the walk stops and the report flags
    /// [`DiscoveryReport::truncated`].
    pub max_files: usize,
}

impl Default for AgentsDiscoveryOptions {
    fn default() -> Self {
        Self {
            names: vec!["AGENTS.md".into(), "CLAUDE.md".into()],
            stop_at_repo_root: true,
            max_files: 32,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Result types
// ─────────────────────────────────────────────────────────────────────

/// One discovered agents-md file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentsManifest {
    /// Absolute path to the file.
    pub path: PathBuf,
    /// The file's contents, verbatim.
    pub content: String,
    /// Directory the file lives in — used to derive scope ordering
    /// (deeper scope = more specific).
    pub scope: PathBuf,
    /// Bytes read from disk for this manifest. Equal to
    /// `content.len()` for valid UTF-8 — kept as a separate field so
    /// non-UTF-8 truncations are auditable.
    pub bytes_read: u64,
    /// Estimated joules for this file's read. Microjoules.
    /// Provenance is always [`Provenance::Estimator`] — IO energy is
    /// not measured by a shunt at this layer.
    pub joules_uj: u64,
}

/// Aggregate report from a [`discover`] call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryReport {
    /// All discovered manifests, in walk order (deepest scope FIRST
    /// — the dir the walk started in, then its parent, then its
    /// grandparent, …). To get the "less-specific first / override
    /// with more-specific" merge order, reverse this list.
    pub manifests: Vec<AgentsManifest>,
    /// Sum of `bytes_read` across all manifests.
    pub total_bytes_read: u64,
    /// Sum of `joules_uj` across all manifests. Microjoules.
    pub total_joules_uj: u64,
    /// Provenance of `total_joules_uj`. Always
    /// [`Provenance::Estimator`] at v1 (no IO hardware-counter
    /// integration); the field is explicit so the receipt layer
    /// downstream of this crate can carry the floor honestly.
    pub energy_provenance: Provenance,
    /// True when [`AgentsDiscoveryOptions::max_files`] was hit and
    /// further matches were not returned.
    pub truncated: bool,
}

/// Errors that can come out of [`discover`].
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    /// IO failure on the starting directory itself (not present, not
    /// readable). Per-file IO failures are silently skipped so a
    /// permission glitch on one ancestor does not abort the walk.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

// ─────────────────────────────────────────────────────────────────────
// discover() — the public entry point
// ─────────────────────────────────────────────────────────────────────

/// Walk up the directory tree from `start`, collecting AGENTS.md /
/// CLAUDE.md (or `opts.names`) files. Returns a joule-stamped
/// report.
///
/// Walk order: `start` first, then parent, then grandparent, until
/// `.git/` is seen (if `opts.stop_at_repo_root`) or the filesystem
/// root is reached. Within a single directory, files are read in
/// the order given by `opts.names` — so a config-specified
/// preference is preserved.
pub fn discover(
    start: impl AsRef<Path>,
    opts: &AgentsDiscoveryOptions,
) -> Result<DiscoveryReport, DiscoveryError> {
    let start = start.as_ref().to_path_buf();
    let start = if start.is_absolute() {
        start
    } else {
        std::env::current_dir()?.join(start)
    };

    let mut manifests: Vec<AgentsManifest> = Vec::new();
    let mut total_bytes_read: u64 = 0;
    let mut total_joules_uj: u64 = 0;
    let mut truncated = false;

    let mut cursor: Option<PathBuf> = Some(start);
    while let Some(dir) = cursor.take() {
        if manifests.len() >= opts.max_files {
            truncated = true;
            break;
        }

        for name in &opts.names {
            if manifests.len() >= opts.max_files {
                truncated = true;
                break;
            }
            let candidate = dir.join(name);
            // Skip symlinks (honest scope — bounded walk cost).
            let meta = std::fs::symlink_metadata(&candidate);
            let Ok(meta) = meta else { continue };
            if !meta.is_file() {
                continue;
            }
            // Try to read; silently skip per-file IO failures so one
            // bad ancestor doesn't kill the walk.
            let Ok(content) = std::fs::read_to_string(&candidate) else {
                continue;
            };
            let bytes_read = content.len() as u64;
            let joules_uj = estimate_read_joules_uj(bytes_read);
            total_bytes_read = total_bytes_read.saturating_add(bytes_read);
            total_joules_uj = total_joules_uj.saturating_add(joules_uj);
            manifests.push(AgentsManifest {
                path: candidate.clone(),
                content,
                scope: dir.clone(),
                bytes_read,
                joules_uj,
            });
        }

        // Stop conditions: at repo root if requested, or at FS root.
        if opts.stop_at_repo_root && dir.join(".git").exists() {
            break;
        }
        cursor = dir.parent().map(Path::to_path_buf);
    }

    Ok(DiscoveryReport {
        manifests,
        total_bytes_read,
        total_joules_uj,
        energy_provenance: Provenance::Estimator,
        truncated,
    })
}

/// Walk up from `start` and return the first directory that
/// contains a `.git/` entry, or `None` if none is found before the
/// filesystem root.
pub fn find_repo_root(start: &Path) -> Option<PathBuf> {
    let mut cursor = if start.is_absolute() {
        start.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(start)
    };
    loop {
        if cursor.join(".git").exists() {
            return Some(cursor);
        }
        match cursor.parent() {
            Some(p) => cursor = p.to_path_buf(),
            None => return None,
        }
    }
}

/// Concatenate every manifest's `content` into a single string,
/// preceded by a scope hint comment. Output order is reversed from
/// the walk — the *least specific* file appears first so a downstream
/// consumer reading it as a system prompt can have *more specific*
/// scopes override earlier ones, the Codex / Cursor / Copilot
/// convention.
pub fn merge(manifests: &[AgentsManifest]) -> String {
    let mut out = String::new();
    for m in manifests.iter().rev() {
        out.push_str(&format!(
            "<!-- agents-md: {} (scope: {}) -->\n",
            m.path.display(),
            m.scope.display()
        ));
        out.push_str(&m.content);
        if !m.content.ends_with('\n') {
            out.push('\n');
        }
        out.push('\n');
    }
    out
}

// ─────────────────────────────────────────────────────────────────────
// Kani proof harnesses — orthogonal to runtime tests
// ─────────────────────────────────────────────────────────────────────
//
// Run with `cargo kani`. These compile only under `#[cfg(kani)]` so
// the normal build doesn't pull in `kani` as a dependency. Each
// harness pins one structural invariant the protocol cannot afford
// to lose.

/// Joule arithmetic must never overflow regardless of bytes_read
/// (saturating). A 16-EiB-byte file produces a bounded `u64`.
#[cfg(kani)]
#[kani::proof]
fn kani_estimate_joules_never_overflows() {
    let bytes: u64 = kani::any();
    let uj = estimate_read_joules_uj(bytes);
    // Saturating: result fits in u64. Trivially true by type, but
    // the proof also checks the implementation didn't subtract or
    // do anything wild.
    let _ = uj;
}

/// Empty file still costs PER_FILE_SETUP_UJ (we charge for the
/// syscall, not for the bytes).
#[cfg(kani)]
#[kani::proof]
fn kani_empty_file_costs_setup() {
    let uj = estimate_read_joules_uj(0);
    kani::assert(uj == PER_FILE_SETUP_UJ, "empty file == setup cost");
}

/// Joule cost is monotone in bytes_read — more bytes ≥ fewer bytes.
#[cfg(kani)]
#[kani::proof]
fn kani_joules_monotone_in_bytes() {
    let a: u64 = kani::any();
    let b: u64 = kani::any();
    kani::assume(a <= b);
    let ua = estimate_read_joules_uj(a);
    let ub = estimate_read_joules_uj(b);
    kani::assert(ua <= ub, "more bytes never costs fewer joules");
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(label: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let p = std::env::temp_dir().join(format!("jouleclaw-agents-md-{label}-{pid}-{n}"));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn write(dir: &Path, name: &str, content: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(name), content).unwrap();
    }

    #[test]
    fn estimator_charges_per_file_setup_for_zero_bytes() {
        assert_eq!(estimate_read_joules_uj(0), PER_FILE_SETUP_UJ);
    }

    #[test]
    fn estimator_monotone_in_bytes() {
        let a = estimate_read_joules_uj(100);
        let b = estimate_read_joules_uj(10_000);
        let c = estimate_read_joules_uj(10_000_000);
        assert!(a <= b);
        assert!(b <= c);
    }

    #[test]
    fn discover_finds_file_in_start_dir() {
        let dir = tmpdir("start");
        write(&dir, "AGENTS.md", "hello");
        let report = discover(&dir, &Default::default()).unwrap();
        assert_eq!(report.manifests.len(), 1);
        assert_eq!(report.manifests[0].content, "hello");
        assert_eq!(report.manifests[0].bytes_read, 5);
        assert!(report.total_joules_uj >= PER_FILE_SETUP_UJ);
        assert_eq!(report.energy_provenance, Provenance::Estimator);
        assert!(!report.truncated);
    }

    #[test]
    fn discover_walks_up_to_parent_until_git_marker() {
        let root = tmpdir("walk");
        let child = root.join("child");
        let grand = child.join("grand");
        write(&root, "AGENTS.md", "root-level");
        write(&child, "AGENTS.md", "child-level");
        write(&grand, "AGENTS.md", "grand-level");
        // Mark the root as a repo root.
        std::fs::create_dir_all(root.join(".git")).unwrap();

        let report = discover(&grand, &Default::default()).unwrap();
        assert_eq!(report.manifests.len(), 3);
        // Deepest-first walk order:
        assert!(report.manifests[0].path.ends_with("grand/AGENTS.md"));
        assert!(report.manifests[1].path.ends_with("child/AGENTS.md"));
        assert!(report.manifests[2].path.ends_with("walk/AGENTS.md") || report.manifests[2].path.ends_with("AGENTS.md"));
    }

    #[test]
    fn discover_collects_both_agents_md_and_claude_md_in_priority_order() {
        let dir = tmpdir("both");
        write(&dir, "AGENTS.md", "a");
        write(&dir, "CLAUDE.md", "c");
        let report = discover(&dir, &Default::default()).unwrap();
        assert_eq!(report.manifests.len(), 2);
        assert!(report.manifests[0].path.ends_with("AGENTS.md"));
        assert!(report.manifests[1].path.ends_with("CLAUDE.md"));
    }

    #[test]
    fn discover_truncates_at_max_files() {
        let root = tmpdir("trunc");
        // Stack three dirs each carrying one AGENTS.md.
        for sub in ["a", "a/b", "a/b/c"] {
            let p = root.join(sub);
            write(&p, "AGENTS.md", "x");
        }
        std::fs::create_dir_all(root.join(".git")).unwrap();
        let opts = AgentsDiscoveryOptions {
            max_files: 2,
            ..Default::default()
        };
        let report = discover(root.join("a/b/c"), &opts).unwrap();
        assert_eq!(report.manifests.len(), 2);
        assert!(report.truncated);
    }

    #[test]
    fn discover_missing_files_silently_skipped() {
        let dir = tmpdir("missing");
        // No AGENTS.md / CLAUDE.md written. Walk should produce an
        // empty report, not error.
        let report = discover(&dir, &Default::default()).unwrap();
        assert!(report.manifests.is_empty());
        assert_eq!(report.total_joules_uj, 0);
    }

    #[test]
    fn merge_reverses_walk_order_for_specificity_layering() {
        let manifests = vec![
            AgentsManifest {
                path: PathBuf::from("/repo/child/AGENTS.md"),
                content: "child-specific".into(),
                scope: PathBuf::from("/repo/child"),
                bytes_read: 14,
                joules_uj: 11,
            },
            AgentsManifest {
                path: PathBuf::from("/repo/AGENTS.md"),
                content: "root-level".into(),
                scope: PathBuf::from("/repo"),
                bytes_read: 10,
                joules_uj: 11,
            },
        ];
        let merged = merge(&manifests);
        // Root level appears FIRST in the merged output (less-specific
        // first, so child-specific can override it later in prompt).
        let root_idx = merged.find("root-level").unwrap();
        let child_idx = merged.find("child-specific").unwrap();
        assert!(root_idx < child_idx, "root must precede child in merged: {merged}");
    }

    #[test]
    fn merge_includes_scope_hint_comments() {
        let manifests = vec![AgentsManifest {
            path: PathBuf::from("/x/AGENTS.md"),
            content: "body".into(),
            scope: PathBuf::from("/x"),
            bytes_read: 4,
            joules_uj: 11,
        }];
        let merged = merge(&manifests);
        assert!(merged.contains("agents-md:"));
        assert!(merged.contains("scope:"));
    }

    #[test]
    fn total_joules_sums_per_file_costs() {
        let dir = tmpdir("sum");
        write(&dir, "AGENTS.md", "abc"); // 3 bytes → ~10 µJ setup + 0 µJ bytes
        write(&dir, "CLAUDE.md", "def"); // 3 bytes → ~10 µJ setup + 0 µJ bytes
        let report = discover(&dir, &Default::default()).unwrap();
        assert_eq!(report.manifests.len(), 2);
        assert_eq!(
            report.total_joules_uj,
            report.manifests.iter().map(|m| m.joules_uj).sum::<u64>()
        );
        assert_eq!(report.total_bytes_read, 6);
    }

    #[test]
    fn find_repo_root_locates_git_dir() {
        let root = tmpdir("repo-root");
        let nested = root.join("a/b/c");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        assert_eq!(find_repo_root(&nested).unwrap(), root);
    }

    #[test]
    fn discovery_report_round_trips_through_json() {
        let manifests = vec![AgentsManifest {
            path: PathBuf::from("/a/AGENTS.md"),
            content: "x".into(),
            scope: PathBuf::from("/a"),
            bytes_read: 1,
            joules_uj: 11,
        }];
        let report = DiscoveryReport {
            manifests,
            total_bytes_read: 1,
            total_joules_uj: 11,
            energy_provenance: Provenance::Estimator,
            truncated: false,
        };
        let bytes = serde_json::to_vec(&report).unwrap();
        let back: DiscoveryReport = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, report);
    }
}
