//! Per-rail orchestration: dispatch a batch to the right processor,
//! respect cutoff windows, fetch returns.
//!
//! ## Roles
//!
//! - [`BatchProcessor`] — operator-implemented trait, one impl per
//!   rail. Owns the actual mechanics of encoding a batch and
//!   poking the bank (or just writing the spool file).
//! - [`BatchOrchestrator`] — registry of processors plus a
//!   [`Scheduler`] that knows each rail's cutoff windows. Routes
//!   on [`crate::BatchRail`].
//! - [`Scheduler`] — pure function on `(rail, now)` → next cutoff.
//!   No background tasks, no clock injection — operator code calls
//!   `next_cutoff` when it wants to know.

use std::collections::HashMap;

use chrono::{DateTime, Datelike, NaiveTime, TimeZone, Utc, Weekday};
use serde::{Deserialize, Serialize};

use crate::BatchRail;
use crate::error::{Error, Result};
use crate::exception::Exception;

/// What the bank handed back when we submitted a batch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RailReceipt {
    /// Which rail produced this.
    pub rail: BatchRail,
    /// Bank's id for our submission (e.g. ack ref, filename).
    pub bank_reference: String,
    /// Number of entries in the submitted file.
    pub entry_count: u32,
    /// Time we recorded the submission.
    pub submitted_at: DateTime<Utc>,
}

/// Anything that can encode, submit, and reconcile a batch on a
/// specific rail.
///
/// Operators implement this for their NACHA / SEPA / Wire / Bacs
/// rail by composing the file-format modules in this crate with
/// their own bank connectivity.
pub trait BatchProcessor: Send + Sync {
    /// The rail this processor speaks.
    fn rail(&self) -> BatchRail;

    /// Submit `batch_bytes` (already encoded by the rail-specific
    /// encoder) under `filename`. Returns the bank receipt.
    ///
    /// # Errors
    /// Submission-layer errors surface as [`Error::Submission`].
    fn submit(&self, filename: &str, batch_bytes: &str, entry_count: u32) -> Result<RailReceipt>;

    /// Fetch returns / exceptions raised since `since`. May be a
    /// pull from a return-file SFTP folder, an API call, or just
    /// reading a directory.
    ///
    /// # Errors
    /// Submission or parsing errors.
    fn fetch_returns(&self, since: DateTime<Utc>) -> Result<Vec<Exception>>;
}

/// Per-rail cutoff windows (operator times, typically in the rail's
/// canonical timezone — ET for NACHA, CET for SEPA, GMT for Bacs).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CutoffSchedule {
    /// Rail.
    pub rail: BatchRail,
    /// Sorted `(hour, minute)` cutoffs in the rail's canonical TZ.
    pub windows: Vec<(u32, u32)>,
    /// Days the rail operates (NACHA: Mon-Fri excluding fed
    /// holidays; SEPA: target2 calendar; Bacs: Mon-Fri).
    pub operating_days: Vec<Weekday>,
}

impl CutoffSchedule {
    /// Hard-coded defaults per the rail's operating rules. Operators
    /// override `operating_days` to apply their holiday calendar
    /// (we don't ship one — holidays drift between years and are
    /// outside the scope of this crate).
    #[must_use]
    pub fn defaults(rail: BatchRail) -> Self {
        let weekdays = vec![
            Weekday::Mon,
            Weekday::Tue,
            Weekday::Wed,
            Weekday::Thu,
            Weekday::Fri,
        ];
        let windows = match rail {
            // NACHA Same-Day ACH windows (ET) per Operating Rules.
            BatchRail::Nacha => vec![(10, 30), (14, 45), (16, 45)],
            // SEPA CT runs ~4 cycles/day (CET). Approximate values
            // reflecting STEP2-T's daily-cycle schedule.
            BatchRail::SepaCt | BatchRail::SepaDd => {
                vec![(7, 30), (10, 0), (12, 30), (14, 0)]
            }
            // Bacs: input-day cutoff 22:30 GMT for next-day
            // processing → entry T+2.
            BatchRail::Bacs => vec![(22, 30)],
            // Fedwire: 17:00 ET cutoff for value-today.
            BatchRail::Fedwire => vec![(17, 0)],
            // SWIFT FIN cross-border: 23:00 GMT generic.
            BatchRail::Swift => vec![(23, 0)],
            // CHIPS: 17:00 ET.
            BatchRail::Chips => vec![(17, 0)],
        };
        Self {
            rail,
            windows,
            operating_days: weekdays,
        }
    }

    /// True if `dt` falls on one of the rail's operating days.
    #[must_use]
    pub fn is_operating_day(&self, dt: DateTime<Utc>) -> bool {
        self.operating_days.contains(&dt.weekday())
    }

    /// The next cutoff strictly after `now`, expressed in UTC. The
    /// rail's canonical timezone is approximated by treating the
    /// supplied windows as already in UTC — operators pre-convert
    /// when they care about local cutoffs.
    #[must_use]
    pub fn next_cutoff(&self, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        if !self.is_operating_day(now) {
            // Roll to next operating day.
            return self.first_window_of_next_operating_day(now);
        }
        let today = now.date_naive();
        for (h, m) in &self.windows {
            let t = NaiveTime::from_hms_opt(*h, *m, 0)?;
            let candidate = today.and_time(t);
            let candidate_utc = Utc.from_utc_datetime(&candidate);
            if candidate_utc > now {
                return Some(candidate_utc);
            }
        }
        self.first_window_of_next_operating_day(now)
    }

    fn first_window_of_next_operating_day(
        &self,
        now: DateTime<Utc>,
    ) -> Option<DateTime<Utc>> {
        let (h, m) = *self.windows.first()?;
        let mut day = now.date_naive();
        for _ in 0..14 {
            day = day.succ_opt()?;
            let wd = day.weekday();
            if self.operating_days.contains(&wd) {
                let t = NaiveTime::from_hms_opt(h, m, 0)?;
                return Some(Utc.from_utc_datetime(&day.and_time(t)));
            }
        }
        None
    }
}

/// Pure cutoff calendar across rails.
#[derive(Clone, Debug, Default)]
pub struct Scheduler {
    schedules: HashMap<BatchRail, CutoffSchedule>,
}

impl Scheduler {
    /// Construct with default windows for the listed rails.
    #[must_use]
    pub fn with_defaults(rails: &[BatchRail]) -> Self {
        let mut s = Self::default();
        for r in rails {
            s.schedules.insert(*r, CutoffSchedule::defaults(*r));
        }
        s
    }

    /// Register / override the schedule for `rail`.
    pub fn register(&mut self, schedule: CutoffSchedule) {
        self.schedules.insert(schedule.rail, schedule);
    }

    /// Next cutoff after `now` for `rail`.
    ///
    /// # Errors
    /// [`Error::NotFound`] if `rail` has no registered schedule.
    pub fn next_cutoff(&self, rail: BatchRail, now: DateTime<Utc>) -> Result<Option<DateTime<Utc>>> {
        let s = self
            .schedules
            .get(&rail)
            .ok_or_else(|| Error::NotFound(format!("no schedule for {rail:?}")))?;
        Ok(s.next_cutoff(now))
    }
}

/// Maps a [`BatchRail`] to a [`BatchProcessor`] and owns the
/// [`Scheduler`].
pub struct BatchOrchestrator {
    processors: HashMap<BatchRail, Box<dyn BatchProcessor>>,
    scheduler: Scheduler,
}

impl BatchOrchestrator {
    /// Construct empty.
    #[must_use]
    pub fn new(scheduler: Scheduler) -> Self {
        Self {
            processors: HashMap::new(),
            scheduler,
        }
    }

    /// Register a processor for its rail. Replaces any previously
    /// registered processor for that rail.
    pub fn register(&mut self, processor: Box<dyn BatchProcessor>) {
        self.processors.insert(processor.rail(), processor);
    }

    /// Borrow the underlying scheduler.
    #[must_use]
    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Submit `bytes` to `rail`.
    ///
    /// # Errors
    /// [`Error::NotFound`] if no processor is registered for the rail.
    pub fn submit(
        &self,
        rail: BatchRail,
        filename: &str,
        bytes: &str,
        entry_count: u32,
    ) -> Result<RailReceipt> {
        let p = self
            .processors
            .get(&rail)
            .ok_or_else(|| Error::NotFound(format!("no processor for {rail:?}")))?;
        p.submit(filename, bytes, entry_count)
    }

    /// Pull returns for every registered rail since `since`,
    /// concatenated into one list. Operators that want to drive
    /// per-rail polling separately can call `processors().get(rail)`
    /// directly.
    ///
    /// # Errors
    /// Forwards the first [`BatchProcessor::fetch_returns`] error.
    pub fn fetch_all_returns(&self, since: DateTime<Utc>) -> Result<Vec<Exception>> {
        let mut out = Vec::new();
        for p in self.processors.values() {
            out.extend(p.fetch_returns(since)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file_io::{MemorySink, Submission, SubmissionSink};
    use chrono::Timelike;
    use std::sync::Arc;

    /// Test processor: drop into a memory sink, no returns.
    struct TestProcessor {
        rail: BatchRail,
        sink: Arc<MemorySink>,
    }

    impl BatchProcessor for TestProcessor {
        fn rail(&self) -> BatchRail {
            self.rail
        }
        fn submit(
            &self,
            filename: &str,
            bytes: &str,
            entry_count: u32,
        ) -> Result<RailReceipt> {
            self.sink.submit(Submission {
                rail: self.rail,
                contents: bytes.to_string(),
                filename: filename.to_string(),
            })?;
            Ok(RailReceipt {
                rail: self.rail,
                bank_reference: format!("ack-{filename}"),
                entry_count,
                submitted_at: Utc::now(),
            })
        }
        fn fetch_returns(&self, _since: DateTime<Utc>) -> Result<Vec<Exception>> {
            Ok(vec![])
        }
    }

    #[test]
    fn orchestrator_dispatches_to_right_rail() {
        let sink = Arc::new(MemorySink::new());
        let mut o = BatchOrchestrator::new(Scheduler::with_defaults(&[
            BatchRail::Nacha,
            BatchRail::SepaCt,
        ]));
        o.register(Box::new(TestProcessor {
            rail: BatchRail::Nacha,
            sink: Arc::clone(&sink),
        }));
        o.register(Box::new(TestProcessor {
            rail: BatchRail::SepaCt,
            sink: Arc::clone(&sink),
        }));
        let r = o
            .submit(BatchRail::Nacha, "x.001", "BODY", 1)
            .unwrap();
        assert_eq!(r.rail, BatchRail::Nacha);
        assert_eq!(r.bank_reference, "ack-x.001");
        assert_eq!(sink.snapshot().unwrap().len(), 1);
    }

    #[test]
    fn unknown_rail_errors() {
        let o = BatchOrchestrator::new(Scheduler::with_defaults(&[BatchRail::Nacha]));
        assert!(o.submit(BatchRail::SepaCt, "x", "y", 1).is_err());
    }

    #[test]
    fn scheduler_returns_next_window_today() {
        let s = Scheduler::with_defaults(&[BatchRail::Nacha]);
        // Pick a Wednesday so we don't bounce to next operating
        // day on a weekend.
        let now = Utc.with_ymd_and_hms(2026, 6, 3, 9, 0, 0).unwrap();
        let next = s.next_cutoff(BatchRail::Nacha, now).unwrap().unwrap();
        assert_eq!(next.hour(), 10);
        assert_eq!(next.minute(), 30);
    }

    #[test]
    fn scheduler_rolls_to_next_operating_day_after_last_window() {
        let s = Scheduler::with_defaults(&[BatchRail::Nacha]);
        // Friday 18:00 → next Monday 10:30 (skipping Sat/Sun).
        let now = Utc.with_ymd_and_hms(2026, 6, 5, 18, 0, 0).unwrap();
        let next = s.next_cutoff(BatchRail::Nacha, now).unwrap().unwrap();
        assert_eq!(next.weekday(), Weekday::Mon);
        assert_eq!(next.hour(), 10);
    }

    #[test]
    fn sepa_has_four_daily_cycles() {
        let s = CutoffSchedule::defaults(BatchRail::SepaCt);
        assert_eq!(s.windows.len(), 4);
    }

    #[test]
    fn bacs_has_one_evening_cutoff() {
        let s = CutoffSchedule::defaults(BatchRail::Bacs);
        assert_eq!(s.windows, vec![(22, 30)]);
    }
}
