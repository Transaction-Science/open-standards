//! Durable promotion store — promoted facts survive process restarts.
//!
//! Pattern borrowed (conceptually, not in code) from RCE's `EngineStore`:
//! mirror every state change to an append-only journal and replay it on
//! open. Here the journal is `promotions.jsonl` — one JSON record per
//! promoted fact — under a caller-chosen directory.
//!
//! ```text
//! <root>/
//!   promotions.jsonl   append-only, one PersistedRecord per line
//! ```
//!
//! On [`FilePromotionStore::open`] the journal is replayed to rebuild
//! the in-memory fact map. On [`record`](super::PromotionStore::record)
//! the new fact is both inserted in memory and appended to the journal.
//! The in-memory map is always authoritative for the live session;
//! journal writes are best-effort and surfaced via
//! [`FilePromotionStore::write_errors`] (the trait's `record` is
//! infallible by contract, so a disk error cannot abort a promotion —
//! it is counted instead).
//!
//! Reuse counts (`hits`) are session-scoped and intentionally NOT
//! persisted: persisting every lookup would turn a read into a write.
//! Facts are durable; the "invocations avoided" tally resets per process.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use jouleclaw_cascade::types::{AnswerOutput, L3ModelId, L4ModelId, TierId};

use crate::{PromotedEntry, PromotionKey, PromotionLogEntry, PromotionStore};

const JOURNAL: &str = "promotions.jsonl";

/// Serializable form of a promoted answer payload. `Refused` is never
/// promoted (the gate rejects it), so only the two answer-bearing
/// variants need a wire form.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PersistedOutput {
    Text { text: String },
    Structured { bytes: Vec<u8> },
}

/// One journal line.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedRecord {
    key_hex: String,
    output: PersistedOutput,
    confidence: f32,
    origin_tier_tag: String,
    promoted_at_secs: u64,
}

/// Errors opening a durable store.
#[derive(Debug, thiserror::Error)]
pub enum DiskError {
    #[error("create promotion-store dir: {0}")]
    CreateDir(String),
    #[error("read promotion journal: {0}")]
    ReadJournal(String),
}

/// A durable [`PromotionStore`] backed by an append-only JSON-lines
/// journal under a directory.
pub struct FilePromotionStore {
    root: PathBuf,
    facts: HashMap<[u8; 32], PromotedEntry>,
    log: Vec<PromotionLogEntry>,
    write_errors: u64,
}

impl FilePromotionStore {
    /// Open (creating if needed) a durable store rooted at `dir`,
    /// replaying any existing journal into memory.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, DiskError> {
        let root = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&root).map_err(|e| DiskError::CreateDir(e.to_string()))?;
        let mut store = Self {
            root,
            facts: HashMap::new(),
            log: Vec::new(),
            write_errors: 0,
        };
        store.replay()?;
        Ok(store)
    }

    fn journal_path(&self) -> PathBuf {
        self.root.join(JOURNAL)
    }

    /// Replay the journal into the in-memory map. Malformed lines are
    /// skipped (a corrupt line never blocks recovery of the rest).
    fn replay(&mut self) -> Result<(), DiskError> {
        let path = self.journal_path();
        if !path.exists() {
            return Ok(());
        }
        let file = File::open(&path).map_err(|e| DiskError::ReadJournal(e.to_string()))?;
        for line in BufReader::new(file).lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.trim().is_empty() {
                continue;
            }
            let Ok(rec) = serde_json::from_str::<PersistedRecord>(&line) else {
                continue;
            };
            let Some(key) = PromotionKey::from_hex(&rec.key_hex) else {
                continue;
            };
            let output = match rec.output {
                PersistedOutput::Text { text } => AnswerOutput::Text(text),
                PersistedOutput::Structured { bytes } => AnswerOutput::Structured(bytes),
            };
            let entry = PromotedEntry {
                output,
                confidence: rec.confidence,
                origin_tier: tier_from_tag(&rec.origin_tier_tag),
                promoted_at_secs: rec.promoted_at_secs,
                hits: 0,
            };
            self.facts.insert(*key.as_bytes(), entry);
            self.log.push(PromotionLogEntry {
                key_hex: rec.key_hex,
                origin_tier: rec.origin_tier_tag,
                confidence: rec.confidence,
                promoted_at_secs: rec.promoted_at_secs,
            });
        }
        Ok(())
    }

    /// Append one record to the journal. Best-effort: returns `false`
    /// and bumps [`write_errors`](Self::write_errors) on I/O failure so
    /// the infallible trait `record` can keep its contract.
    fn append(&mut self, rec: &PersistedRecord) -> bool {
        let line = match serde_json::to_string(rec) {
            Ok(l) => l,
            Err(_) => {
                self.write_errors += 1;
                return false;
            }
        };
        let mut file = match OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.journal_path())
        {
            Ok(f) => f,
            Err(_) => {
                self.write_errors += 1;
                return false;
            }
        };
        if writeln!(file, "{line}").is_err() {
            self.write_errors += 1;
            return false;
        }
        true
    }

    /// Count of journal-write failures this session. Non-zero means some
    /// promoted facts live only in memory and will not survive a restart.
    pub fn write_errors(&self) -> u64 {
        self.write_errors
    }

    /// The append-only promotion log (replayed + this session's).
    pub fn log(&self) -> &[PromotionLogEntry] {
        &self.log
    }
}

impl PromotionStore for FilePromotionStore {
    fn lookup(&mut self, key: &PromotionKey) -> Option<PromotedEntry> {
        let entry = self.facts.get_mut(key.as_bytes())?;
        entry.hits += 1;
        Some(entry.clone())
    }

    fn record(&mut self, key: PromotionKey, entry: PromotedEntry, log: PromotionLogEntry) {
        if self.facts.contains_key(key.as_bytes()) {
            return; // permanent + idempotent: first fact wins
        }
        // Refused outputs are never promoted; if one slips through, keep
        // it in memory but do not journal an unrepresentable payload.
        let persisted_output = match &entry.output {
            AnswerOutput::Text(t) => Some(PersistedOutput::Text { text: t.clone() }),
            AnswerOutput::Structured(b) => Some(PersistedOutput::Structured { bytes: b.clone() }),
            AnswerOutput::Refused(_) => None,
        };
        if let Some(output) = persisted_output {
            let rec = PersistedRecord {
                key_hex: key.to_hex(),
                output,
                confidence: entry.confidence,
                origin_tier_tag: entry.origin_tier.wire_tag().to_string(),
                promoted_at_secs: entry.promoted_at_secs,
            };
            self.append(&rec);
        }
        self.facts.insert(*key.as_bytes(), entry);
        self.log.push(log);
    }

    fn len(&self) -> usize {
        self.facts.len()
    }

    fn invocations_avoided(&self) -> u64 {
        self.facts.values().map(|e| e.hits).sum()
    }
}

/// Map a wire tag back to a representative [`TierId`]. Lossy for the
/// parameterised model tiers (the model id is not journaled) — origin
/// tier is informational provenance, so a representative is sufficient.
fn tier_from_tag(tag: &str) -> TierId {
    match tag {
        "L3" => TierId::L3(L3ModelId(0)),
        "L4" => TierId::L4(L4ModelId(0)),
        "L2.5" => TierId::L2_5NeuralRerank,
        _ => TierId::L3(L3ModelId(0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PromotionGate};
    use jouleclaw_cascade::types::{
        Answer, ContextRef, ExecutionTrace, JouleBudget, QualityFloor, Query, QueryInput,
    };
    use jouleclaw_cascade::verification::VerificationStatus;

    fn tmpdir(tag: &str) -> PathBuf {
        let mut d = std::env::temp_dir();
        d.push(format!(
            "jouleclaw-promote-test-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        d
    }

    fn query(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.to_string()),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    fn model_answer(out: &str, conf: f32) -> Answer {
        Answer {
            output: AnswerOutput::Text(out.to_string()),
            tier_used: TierId::L3(L3ModelId(0)),
            joules_spent: 2.0,
            confidence: conf,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        }
    }

    #[test]
    fn facts_survive_reopen() {
        let dir = tmpdir("survive");
        // First "process": open, promote a fact, drop.
        {
            let store = std::sync::Arc::new(std::sync::Mutex::new(
                FilePromotionStore::open(&dir).expect("open"),
            ));
            let mut gate = PromotionGate::new(store.clone());
            assert!(gate.consider(&query("capital of france"), &model_answer("Paris", 0.97), true, 1000));
            assert_eq!(store.lock().unwrap().len(), 1);
        }
        // Second "process": reopen the same dir, the fact is replayed.
        {
            let mut reopened = FilePromotionStore::open(&dir).expect("reopen");
            assert_eq!(reopened.len(), 1);
            let key = PromotionKey::of(&query("capital of france"));
            let hit = reopened.lookup(&key).expect("replayed fact present");
            match hit.output {
                AnswerOutput::Text(t) => assert_eq!(t, "Paris"),
                other => panic!("expected text, got {other:?}"),
            }
            assert!((hit.confidence - 0.97).abs() < 1e-6);
            assert_eq!(reopened.log().len(), 1);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn miss_after_reopen_for_unknown_key() {
        let dir = tmpdir("miss");
        {
            let store = std::sync::Arc::new(std::sync::Mutex::new(
                FilePromotionStore::open(&dir).expect("open"),
            ));
            let mut gate = PromotionGate::new(store);
            gate.consider(&query("known"), &model_answer("yes", 0.95), true, 1);
        }
        let mut reopened = FilePromotionStore::open(&dir).expect("reopen");
        assert!(reopened.lookup(&PromotionKey::of(&query("unknown"))).is_none());
        assert!(reopened.lookup(&PromotionKey::of(&query("known"))).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn idempotent_across_session_and_disk() {
        let dir = tmpdir("idem");
        let store = std::sync::Arc::new(std::sync::Mutex::new(
            FilePromotionStore::open(&dir).expect("open"),
        ));
        let mut gate = PromotionGate::new(store.clone());
        assert!(gate.consider(&query("q"), &model_answer("first", 0.95), true, 1));
        assert!(!gate.consider(&query("q"), &model_answer("second", 0.99), true, 2));
        assert_eq!(store.lock().unwrap().len(), 1);
        // Reopen: still one fact, still "first".
        drop(store);
        let mut reopened = FilePromotionStore::open(&dir).expect("reopen");
        let hit = reopened.lookup(&PromotionKey::of(&query("q"))).unwrap();
        match hit.output {
            AnswerOutput::Text(t) => assert_eq!(t, "first"),
            o => panic!("{o:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn key_hex_round_trips() {
        let k = PromotionKey::of(&query("anything"));
        let k2 = PromotionKey::from_hex(&k.to_hex()).expect("round trip");
        assert_eq!(k, k2);
        assert!(PromotionKey::from_hex("nothex").is_none());
    }
}
