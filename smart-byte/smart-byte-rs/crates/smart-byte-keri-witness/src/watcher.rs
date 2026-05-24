//! Watchers: third-party observers who detect duplicity.
//!
//! A watcher does not sign anything. It collects observations of
//! `(controller, sequence, event_said)` and, when it sees two distinct
//! events at the same `(controller, sequence)`, raises a
//! [`DuplicityDetected`] signal. Multiple watchers around the network
//! make it impossible for a colluding controller-plus-subset-of-witnesses
//! to maintain a forked log without being noticed.

use chrono::{DateTime, Utc};
use dashmap::{DashMap, DashSet};
use serde::{Deserialize, Serialize};
use smart_byte_core::Said;

use crate::error::{KeriError, Result};
use crate::events::{ControllerAid, KeyEvent, WatcherAid};
use crate::witness::WitnessReceipt;

/// One observed event plus the receipts a watcher saw alongside it.
#[derive(Clone, Debug)]
pub struct ObservedEntry {
    /// The event itself.
    pub event: KeyEvent,
    /// Witness receipts the watcher saw bundled with the event.
    pub receipts: Vec<WitnessReceipt>,
    /// When the watcher first saw the event.
    pub first_seen: DateTime<Utc>,
}

/// In-memory log of observations for a single controller.
#[derive(Default, Debug, Clone)]
pub struct ObservedLog {
    /// Map from sequence number to every distinct event seen at that sequence.
    pub by_sequence: std::collections::BTreeMap<u64, Vec<ObservedEntry>>,
}

/// Signal emitted when the watcher detects two distinct events at the
/// same `(controller, sequence)`. Carries enough evidence for a
/// downstream verifier to confirm the fork.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DuplicityDetected {
    /// The forked controller.
    pub controller: ControllerAid,
    /// Sequence number at which two distinct events were observed.
    pub sequence: u64,
    /// SAIDs of the two distinct events forming the fork.
    pub fork: (Said, Said),
    /// Wall-clock timestamp of detection.
    pub detected_at: DateTime<Utc>,
}

/// Third-party duplicity-detector.
pub struct Watcher {
    /// Watcher AID — opaque to this crate.
    pub aid: WatcherAid,
    /// Observed event logs, keyed by controller.
    pub observed_controllers: DashMap<ControllerAid, ObservedLog>,
    /// Set of controllers for which duplicity has already been raised.
    pub duplicity_detected: DashSet<ControllerAid>,
    /// All duplicity signals raised so far, in detection order.
    pub signals: std::sync::Mutex<Vec<DuplicityDetected>>,
}

impl Watcher {
    /// Construct a watcher with the given AID.
    #[must_use]
    pub fn new(aid: WatcherAid) -> Self {
        Self {
            aid,
            observed_controllers: DashMap::new(),
            duplicity_detected: DashSet::new(),
            signals: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Observe an event plus its witness receipts.
    ///
    /// On observing a second distinct event at the same
    /// `(controller, sequence)`, a [`DuplicityDetected`] is recorded
    /// and the controller is added to [`Self::duplicity_detected`].
    pub async fn observe(
        &self,
        event: KeyEvent,
        witness_receipts: &[WitnessReceipt],
    ) -> Result<()> {
        // SAID consistency before any further processing.
        event.validate_said()?;
        // Verify every supplied receipt before letting it influence
        // duplicity state.
        for r in witness_receipts {
            if r.signed_event_said != event.said() || r.sequence != event.sequence() {
                return Err(KeriError::Malformed(
                    "receipt does not match supplied event".into(),
                ));
            }
            r.verify_signature()?;
        }

        let controller = event.controller().clone();
        let sequence = event.sequence();
        let said = event.said();

        let mut entry = self
            .observed_controllers
            .entry(controller.clone())
            .or_default();
        let log = entry.value_mut();
        let bucket = log.by_sequence.entry(sequence).or_default();
        let already_have = bucket.iter().any(|e| e.event.said() == said);
        if !already_have {
            bucket.push(ObservedEntry {
                event,
                receipts: witness_receipts.to_vec(),
                first_seen: Utc::now(),
            });
            // If now >1 distinct event at this sequence → duplicity.
            if bucket.len() > 1 {
                let a = bucket[0].event.said();
                let b = bucket[1].event.said();
                let signal = DuplicityDetected {
                    controller: controller.clone(),
                    sequence,
                    fork: (a, b),
                    detected_at: Utc::now(),
                };
                self.duplicity_detected.insert(controller);
                if let Ok(mut sigs) = self.signals.lock() {
                    sigs.push(signal);
                }
            }
        }
        Ok(())
    }

    /// Snapshot of all duplicity signals raised so far.
    #[must_use]
    pub fn signals(&self) -> Vec<DuplicityDetected> {
        self.signals.lock().map(|v| v.clone()).unwrap_or_default()
    }
}
