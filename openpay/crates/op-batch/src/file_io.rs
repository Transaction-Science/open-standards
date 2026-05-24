//! File submission and naming conventions.
//!
//! Every bank's SFTP folder wants files named a specific way; if
//! the name's wrong the bank's ingestion job ignores it. This
//! module centralises the conventions:
//!
//! | Rail | Pattern | Example |
//! |------|---------|---------|
//! | NACHA | `<odfi-routing>.<yyyymmdd>.<seq>` | `121000248.20260601.001` |
//! | SEPA  | `<unique>.xml`                    | `MSG-2026-06-01-0001.xml` |
//! | Bacs  | `<sun>.<yyddd>`                    | `123456.26152` |
//! | Wire (MT)  | `<sender-ref>.txt`            | `REF12345.txt` |
//! | Wire (pacs)| `<msg-id>.xml`                | `REF12345.xml` |
//!
//! ## Sink trait
//!
//! Operators implement [`SubmissionSink`] to drop the file
//! wherever their plumbing wants it: a local spool watched by an
//! external SFTP agent, an in-process SFTP push, or — for tests —
//! an in-memory map. We ship [`SpoolSink`] (writes to a directory
//! the operator's SFTP daemon watches) as the default.
//!
//! A `sftp` feature flag is reserved for direct SSH push via
//! `russh` / `ssh2`; we don't enable it by default because
//! production deployments overwhelmingly already run their own
//! file-mover.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, Datelike, Utc};
use serde::{Deserialize, Serialize};

use crate::BatchRail;
use crate::error::{Error, Result};

/// File-naming convention helpers, one method per rail.
pub struct FileNaming;

impl FileNaming {
    /// NACHA ACH filename: `<odfi-routing>.<yyyymmdd>.<seq>`.
    #[must_use]
    pub fn nacha(odfi_routing: &str, date: DateTime<Utc>, seq: u32) -> String {
        format!(
            "{odfi_routing}.{:04}{:02}{:02}.{seq:03}",
            date.year(),
            date.month(),
            date.day()
        )
    }

    /// SEPA pain.001 / pain.008 filename: `<msg-id>.xml`. Caller
    /// supplies `MsgId` (≤35 chars).
    #[must_use]
    pub fn sepa(message_id: &str) -> String {
        format!("{message_id}.xml")
    }

    /// Bacs filename: `<sun>.<yyddd>` (julian processing day).
    #[must_use]
    pub fn bacs(sun: &str, day_julian: &str) -> String {
        format!("{sun}.{day_julian}")
    }

    /// Wire MT103/MT202 filename: `<sender-reference>.txt`.
    #[must_use]
    pub fn wire_mt(sender_reference: &str) -> String {
        format!("{sender_reference}.txt")
    }

    /// Wire pacs.008/pacs.009 filename: `<msg-id>.xml`.
    #[must_use]
    pub fn wire_pacs(msg_id: &str) -> String {
        format!("{msg_id}.xml")
    }
}

/// A submission packet — what gets handed to the [`SubmissionSink`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Submission {
    /// Source rail.
    pub rail: BatchRail,
    /// File contents (already encoded; UTF-8).
    pub contents: String,
    /// Filename, per the rail's convention.
    pub filename: String,
}

/// Anywhere the operator wants the file to land.
///
/// `Sync` so a single sink can be shared across the orchestrator's
/// per-rail processors.
pub trait SubmissionSink: Send + Sync {
    /// Persist the submission. Implementations must be **idempotent**
    /// on `filename`: re-submission of the identical bytes is a
    /// no-op; submission of *different* bytes under the same name
    /// should be rejected with [`Error::Submission`].
    ///
    /// # Errors
    /// [`Error::Submission`] on any persistence failure.
    fn submit(&self, packet: Submission) -> Result<()>;
}

/// File-based sink. Writes to `dir` with the convention'd filename.
/// Operators point this at the folder their SFTP-to-bank script
/// watches.
pub struct SpoolSink {
    dir: PathBuf,
}

impl SpoolSink {
    /// Construct against `dir`. `dir` must already exist.
    ///
    /// # Errors
    /// [`Error::Submission`] if `dir` is not a directory.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        if !dir.is_dir() {
            return Err(Error::Submission(format!(
                "spool dir does not exist: {}",
                dir.display()
            )));
        }
        Ok(Self { dir })
    }

    fn path_for(&self, filename: &str) -> PathBuf {
        let mut p = self.dir.clone();
        p.push(filename);
        p
    }
}

impl SubmissionSink for SpoolSink {
    fn submit(&self, packet: Submission) -> Result<()> {
        let path = self.path_for(&packet.filename);
        if path.exists() {
            // Idempotence: identical bytes are fine, different
            // bytes are a hard error.
            let existing = std::fs::read_to_string(&path)?;
            if existing == packet.contents {
                tracing::info!(filename = %packet.filename, "submission idempotent no-op");
                return Ok(());
            }
            return Err(Error::Submission(format!(
                "file `{}` already exists with different contents",
                packet.filename
            )));
        }
        std::fs::write(&path, packet.contents.as_bytes())?;
        tracing::info!(filename = %packet.filename, rail = ?packet.rail, "submitted");
        Ok(())
    }
}

/// In-memory sink for tests.
pub struct MemorySink {
    inner: Mutex<HashMap<String, Submission>>,
}

impl MemorySink {
    /// Construct an empty in-memory sink.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Snapshot the stored submissions.
    ///
    /// # Errors
    /// `Error::Submission` if the inner lock is poisoned.
    pub fn snapshot(&self) -> Result<Vec<Submission>> {
        let g = self
            .inner
            .lock()
            .map_err(|_| Error::Submission("mutex poisoned".into()))?;
        Ok(g.values().cloned().collect())
    }

    /// Borrow a stored submission by filename.
    ///
    /// # Errors
    /// `Error::Submission` if the inner lock is poisoned.
    pub fn get(&self, filename: &str) -> Result<Option<Submission>> {
        let g = self
            .inner
            .lock()
            .map_err(|_| Error::Submission("mutex poisoned".into()))?;
        Ok(g.get(filename).cloned())
    }
}

impl Default for MemorySink {
    fn default() -> Self {
        Self::new()
    }
}

impl SubmissionSink for MemorySink {
    fn submit(&self, packet: Submission) -> Result<()> {
        let mut g = self
            .inner
            .lock()
            .map_err(|_| Error::Submission("mutex poisoned".into()))?;
        if let Some(existing) = g.get(&packet.filename) {
            if existing.contents != packet.contents {
                return Err(Error::Submission(format!(
                    "file `{}` already exists with different contents",
                    packet.filename
                )));
            }
            return Ok(());
        }
        g.insert(packet.filename.clone(), packet);
        Ok(())
    }
}

/// Reserve a place for an SFTP-direct sink. The `sftp` cargo
/// feature must be enabled to use any such impl; this struct
/// merely tags the intent so operators can probe whether direct
/// delivery is wired without taking a transitive dep on `russh`.
#[cfg(feature = "sftp")]
pub struct SftpSink {
    /// Host (e.g. `bank-sftp.example.com:22`).
    pub host: String,
    /// Remote directory.
    pub remote_dir: String,
}

#[cfg(feature = "sftp")]
impl SubmissionSink for SftpSink {
    fn submit(&self, _packet: Submission) -> Result<()> {
        // Wire-up belongs in the operator's deployment — we surface
        // the trait so swapping in an ssh2-backed impl is a
        // recompile, not a refactor. The default build refuses to
        // pretend it can SFTP without a key wired in.
        Err(Error::Submission(
            "sftp feature is enabled but no transport impl is wired".into(),
        ))
    }
}

/// Tiny helper for tests: persist a packet, then read it back.
///
/// # Errors
/// Forwards any submission / IO error.
pub fn round_trip_through_spool(
    dir: &Path,
    packet: Submission,
) -> Result<String> {
    let sink = SpoolSink::new(dir.to_path_buf())?;
    sink.submit(packet.clone())?;
    let path = dir.join(&packet.filename);
    Ok(std::fs::read_to_string(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn dt() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 6, 1, 10, 30, 0).unwrap()
    }

    #[test]
    fn nacha_naming() {
        assert_eq!(
            FileNaming::nacha("121000248", dt(), 1),
            "121000248.20260601.001"
        );
    }

    #[test]
    fn sepa_naming() {
        assert_eq!(FileNaming::sepa("MSG-001"), "MSG-001.xml");
    }

    #[test]
    fn bacs_naming() {
        assert_eq!(FileNaming::bacs("123456", "26152"), "123456.26152");
    }

    #[test]
    fn wire_naming() {
        assert_eq!(FileNaming::wire_mt("REF12345"), "REF12345.txt");
        assert_eq!(FileNaming::wire_pacs("REF12345"), "REF12345.xml");
    }

    #[test]
    fn memory_sink_round_trip() {
        let sink = MemorySink::new();
        let pkt = Submission {
            rail: BatchRail::Nacha,
            contents: "hello".into(),
            filename: "f1".into(),
        };
        sink.submit(pkt.clone()).unwrap();
        assert_eq!(sink.get("f1").unwrap().unwrap().contents, "hello");
    }

    #[test]
    fn memory_sink_idempotent() {
        let sink = MemorySink::new();
        let pkt = Submission {
            rail: BatchRail::Nacha,
            contents: "hello".into(),
            filename: "f1".into(),
        };
        sink.submit(pkt.clone()).unwrap();
        sink.submit(pkt).unwrap();
        assert_eq!(sink.snapshot().unwrap().len(), 1);
    }

    #[test]
    fn memory_sink_rejects_collision() {
        let sink = MemorySink::new();
        let p1 = Submission {
            rail: BatchRail::Nacha,
            contents: "hello".into(),
            filename: "f1".into(),
        };
        let p2 = Submission {
            rail: BatchRail::Nacha,
            contents: "DIFFERENT".into(),
            filename: "f1".into(),
        };
        sink.submit(p1).unwrap();
        assert!(sink.submit(p2).is_err());
    }
}
