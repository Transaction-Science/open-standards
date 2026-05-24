//! Pluggable storage for event logs and witness receipts.
//!
//! Two reference implementations ship with the crate:
//!
//! * [`MemoryStorage`] — a [`DashMap`]-backed in-memory log; useful
//!   for tests and ephemeral verifier processes.
//! * [`FileStorage`] — one CBOR file per controller for the events and
//!   one per receipted-event SAID for receipts.

use std::path::PathBuf;

use async_trait::async_trait;
use dashmap::DashMap;
use smart_byte_core::Said;
use tokio::fs;

use crate::error::{KeriError, Result};
use crate::events::{ControllerAid, KeyEvent};
use crate::witness::WitnessReceipt;

/// Abstract event-log + receipt storage.
#[async_trait]
pub trait EventLogStorage: Send + Sync {
    /// Append an event to the given controller's log.
    async fn append(&self, controller: &ControllerAid, event: KeyEvent) -> Result<()>;
    /// Append a witness receipt for the given event SAID.
    async fn append_receipt(&self, event_said: Said, receipt: WitnessReceipt) -> Result<()>;
    /// Fetch the controller's full event log in sequence order.
    async fn fetch_log(&self, controller: &ControllerAid) -> Result<Vec<KeyEvent>>;
    /// Fetch every receipt collected for the given event SAID.
    async fn fetch_receipts(&self, event_said: Said) -> Result<Vec<WitnessReceipt>>;
}

/// In-memory reference implementation.
#[derive(Default)]
pub struct MemoryStorage {
    logs: DashMap<ControllerAid, Vec<KeyEvent>>,
    receipts: DashMap<Said, Vec<WitnessReceipt>>,
}

impl MemoryStorage {
    /// Build an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl EventLogStorage for MemoryStorage {
    async fn append(&self, controller: &ControllerAid, event: KeyEvent) -> Result<()> {
        self.logs.entry(controller.clone()).or_default().push(event);
        Ok(())
    }

    async fn append_receipt(&self, event_said: Said, receipt: WitnessReceipt) -> Result<()> {
        self.receipts.entry(event_said).or_default().push(receipt);
        Ok(())
    }

    async fn fetch_log(&self, controller: &ControllerAid) -> Result<Vec<KeyEvent>> {
        Ok(self
            .logs
            .get(controller)
            .map(|e| e.value().clone())
            .unwrap_or_default())
    }

    async fn fetch_receipts(&self, event_said: Said) -> Result<Vec<WitnessReceipt>> {
        Ok(self
            .receipts
            .get(&event_said)
            .map(|e| e.value().clone())
            .unwrap_or_default())
    }
}

/// On-disk CBOR storage. Layout under `root`:
///
/// ```text
/// root/
///   logs/<controller-aid>.cbor       # Vec<KeyEvent>
///   receipts/<event-said>.cbor       # Vec<WitnessReceipt>
/// ```
pub struct FileStorage {
    root: PathBuf,
}

impl FileStorage {
    /// Build a [`FileStorage`] rooted at `root`. Subdirectories are
    /// created lazily on first write.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    fn logs_dir(&self) -> PathBuf {
        self.root.join("logs")
    }

    fn receipts_dir(&self) -> PathBuf {
        self.root.join("receipts")
    }

    fn log_path(&self, controller: &ControllerAid) -> PathBuf {
        self.logs_dir().join(format!("{}.cbor", controller.0))
    }

    fn receipt_path(&self, said: &Said) -> PathBuf {
        self.receipts_dir().join(format!("{}.cbor", said.to_base32()))
    }
}

#[async_trait]
impl EventLogStorage for FileStorage {
    async fn append(&self, controller: &ControllerAid, event: KeyEvent) -> Result<()> {
        fs::create_dir_all(self.logs_dir()).await?;
        let path = self.log_path(controller);
        let mut existing: Vec<KeyEvent> = if path.exists() {
            let bytes = fs::read(&path).await?;
            serde_cbor::from_slice(&bytes)?
        } else {
            Vec::new()
        };
        existing.push(event);
        let bytes = serde_cbor::to_vec(&existing)?;
        fs::write(&path, bytes).await?;
        Ok(())
    }

    async fn append_receipt(&self, event_said: Said, receipt: WitnessReceipt) -> Result<()> {
        fs::create_dir_all(self.receipts_dir()).await?;
        let path = self.receipt_path(&event_said);
        let mut existing: Vec<WitnessReceipt> = if path.exists() {
            let bytes = fs::read(&path).await?;
            serde_cbor::from_slice(&bytes)?
        } else {
            Vec::new()
        };
        existing.push(receipt);
        let bytes = serde_cbor::to_vec(&existing)?;
        fs::write(&path, bytes).await?;
        Ok(())
    }

    async fn fetch_log(&self, controller: &ControllerAid) -> Result<Vec<KeyEvent>> {
        let path = self.log_path(controller);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let bytes = fs::read(&path).await?;
        Ok(serde_cbor::from_slice(&bytes)?)
    }

    async fn fetch_receipts(&self, event_said: Said) -> Result<Vec<WitnessReceipt>> {
        let path = self.receipt_path(&event_said);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let bytes = fs::read(&path).await?;
        Ok(serde_cbor::from_slice(&bytes)?)
    }
}

// Touch KeriError so the import is not flagged as unused when both
// reference impls happen to compile clean.
#[allow(dead_code)]
fn _touch_error() -> Option<KeriError> {
    None
}
