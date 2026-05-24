//! Per-stream joule accounting.
//!
//! A [`StreamMeter`] samples a [`JouleCounter`](eoc_meter::JouleCounter)
//! at the start of a stream and again at the end (or on each
//! checkpoint). The difference is attributed to the stream and surfaced
//! as a [`JouleCost`](eoc_core::JouleCost). Token counts can be
//! accumulated alongside so the cascade sees both axes.

use std::sync::Arc;
use std::sync::Mutex;

use eoc_core::{JouleCost, JouleSource};
use eoc_meter::JouleCounter;

use crate::error::{StreamError, StreamResult};

/// Per-stream accounting record.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamAccount {
    /// Tokens streamed so far.
    pub tokens: u64,
    /// Cumulative micro-joule delta attributed to this stream.
    pub microjoules: u64,
    /// Whether the joule reading is from a hardware counter.
    pub measured: bool,
}

impl StreamAccount {
    /// Convert to the canonical [`JouleCost`] type.
    pub fn joule_cost(&self) -> JouleCost {
        JouleCost {
            microjoules: self.microjoules,
            source: if self.measured {
                JouleSource::Measured
            } else {
                JouleSource::Estimated
            },
        }
    }
}

/// Stream-scoped joule meter.
///
/// Construct with [`StreamMeter::start`]. Call [`StreamMeter::on_token`]
/// per emitted token (or per delta). Call [`StreamMeter::checkpoint`]
/// to fold any accumulated hardware-counter delta into the account.
/// Call [`StreamMeter::finish`] to capture the final reading.
#[derive(Clone)]
pub struct StreamMeter {
    counter: Arc<dyn JouleCounter>,
    start_uj: u64,
    measured: bool,
    state: Arc<Mutex<StreamAccount>>,
}

impl std::fmt::Debug for StreamMeter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamMeter")
            .field("counter", &self.counter.name())
            .field("start_uj", &self.start_uj)
            .field("measured", &self.measured)
            .finish()
    }
}

impl StreamMeter {
    /// Start metering a stream. Reads the counter once; if the read
    /// fails the meter degrades to `Estimated` and continues.
    pub fn start(counter: Arc<dyn JouleCounter>) -> Self {
        let (start_uj, measured) = match counter.read_microjoules() {
            Ok(uj) => (uj, counter.name() != "stub"),
            Err(_) => (0, false),
        };
        Self {
            counter,
            start_uj,
            measured,
            state: Arc::new(Mutex::new(StreamAccount {
                tokens: 0,
                microjoules: 0,
                measured,
            })),
        }
    }

    /// Increment the token count by `n`.
    pub fn on_token(&self, n: u64) -> StreamResult<()> {
        let mut g = self.state.lock().map_err(poisoned)?;
        g.tokens = g.tokens.saturating_add(n);
        Ok(())
    }

    /// Refresh the joule delta from the counter. Cheap and safe to
    /// call at every checkpoint; the cost is one counter read.
    pub fn checkpoint(&self) -> StreamResult<StreamAccount> {
        let now_uj = match self.counter.read_microjoules() {
            Ok(uj) => uj,
            Err(_) => {
                // No measurement available — leave the existing delta in place.
                let g = self.state.lock().map_err(poisoned)?;
                return Ok(*g);
            }
        };
        let mut g = self.state.lock().map_err(poisoned)?;
        g.microjoules = now_uj.saturating_sub(self.start_uj);
        Ok(*g)
    }

    /// Finalize and return the final account. Equivalent to a
    /// checkpoint followed by a snapshot.
    pub fn finish(self) -> StreamResult<StreamAccount> {
        self.checkpoint()
    }

    /// Snapshot the current account without touching the counter.
    pub fn snapshot(&self) -> StreamResult<StreamAccount> {
        let g = self.state.lock().map_err(poisoned)?;
        Ok(*g)
    }
}

fn poisoned<T>(_: T) -> StreamError {
    StreamError::Meter("stream meter mutex poisoned".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eoc_meter::StubCounter;

    #[test]
    fn stub_counter_estimated() {
        let meter = StreamMeter::start(Arc::new(StubCounter));
        meter.on_token(5).unwrap();
        let acct = meter.finish().unwrap();
        assert_eq!(acct.tokens, 5);
        assert_eq!(acct.microjoules, 0);
        assert!(!acct.measured);
        assert_eq!(acct.joule_cost().source, JouleSource::Estimated);
    }
}
