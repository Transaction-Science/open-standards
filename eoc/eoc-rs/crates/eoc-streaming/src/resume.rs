//! Resumable streams via `Last-Event-ID`.
//!
//! When a client reconnects to a streaming endpoint it may include the
//! `Last-Event-ID` HTTP header. If the server still has a replay
//! window covering that id, it can send the missed events first and
//! then resume normal streaming. This module owns the in-memory event
//! log + replay logic; the actual transport binding (HTTP header
//! plumbing, etc.) is left to the host.

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::error::{StreamError, StreamResult};

/// An identifier carried in the SSE `id:` field.
pub type LastEventId = String;

/// Bounded ring-buffer of recently emitted events keyed on id.
#[derive(Debug)]
pub struct EventLog<T: Clone + Send + 'static> {
    inner: Mutex<EventLogInner<T>>,
    capacity: usize,
}

#[derive(Debug)]
struct EventLogInner<T> {
    /// Pairs of (id, event). Oldest first.
    entries: VecDeque<(LastEventId, T)>,
}

impl<T: Clone + Send + 'static> EventLog<T> {
    /// Construct a log retaining at most `capacity` entries.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(EventLogInner {
                entries: VecDeque::with_capacity(capacity.max(1)),
            }),
            capacity: capacity.max(1),
        }
    }

    /// Append a new event. Drops the oldest entry when capacity is hit.
    pub fn push(&self, id: LastEventId, ev: T) -> StreamResult<()> {
        let mut g = self.inner.lock().map_err(|_| {
            StreamError::Backend("event log mutex poisoned".into())
        })?;
        if g.entries.len() == self.capacity {
            g.entries.pop_front();
        }
        g.entries.push_back((id, ev));
        Ok(())
    }

    /// Return all events strictly newer than `last_id`. If `last_id` is
    /// `None`, returns the entire current window. Errors with
    /// [`StreamError::ResumeOutOfRange`] when `last_id` is supplied but
    /// not in the window (i.e. it predates the oldest retained event).
    pub fn replay_after(&self, last_id: Option<&str>) -> StreamResult<Vec<T>> {
        let g = self.inner.lock().map_err(|_| {
            StreamError::Backend("event log mutex poisoned".into())
        })?;
        let Some(last_id) = last_id else {
            return Ok(g.entries.iter().map(|(_, e)| e.clone()).collect());
        };
        // If the oldest entry's id is already newer than last_id we
        // cannot guarantee no gap, so report out-of-range.
        let mut found = false;
        let mut out = Vec::new();
        for (id, ev) in g.entries.iter() {
            if found {
                out.push(ev.clone());
            } else if id == last_id {
                found = true;
            }
        }
        if !found {
            return Err(StreamError::ResumeOutOfRange(last_id.to_string()));
        }
        Ok(out)
    }

    /// Current window length.
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .map(|g| g.entries.len())
            .unwrap_or(0)
    }

    /// True if the window is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replays_after_id() {
        let log = EventLog::<&'static str>::new(4);
        log.push("1".into(), "a").unwrap();
        log.push("2".into(), "b").unwrap();
        log.push("3".into(), "c").unwrap();
        let r = log.replay_after(Some("1")).unwrap();
        assert_eq!(r, vec!["b", "c"]);
    }

    #[test]
    fn out_of_range_reports() {
        let log = EventLog::<&'static str>::new(2);
        log.push("1".into(), "a").unwrap();
        log.push("2".into(), "b").unwrap();
        log.push("3".into(), "c").unwrap(); // evicts "1"
        let err = log.replay_after(Some("1")).unwrap_err();
        assert!(matches!(err, StreamError::ResumeOutOfRange(_)));
    }
}
