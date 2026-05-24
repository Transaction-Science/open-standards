//! Metric primitives: Counter, Gauge, Histogram.
//!
//! These are intentionally minimal — no async, no labels-as-strings — because
//! exposing the right shape to Prometheus/StatsD requires structural
//! attributes anyway. Each metric carries:
//!
//! - a stable `name` (e.g. `"eoc.cache.hits"`)
//! - a stable `unit` (e.g. `"j"`, `"ms"`, `"By"`, `"1"`)
//! - attributes (string-keyed scalars used as labels)
//!
//! Histograms use *explicit* bucket boundaries; the default boundaries match
//! the OTel "duration" suggestion (5, 10, 25, 50, 75, 100, 250, 500, 750,
//! 1000, 2500, 5000, 7500, 10000 ms).

use crate::span::AttrValue;
use std::sync::Mutex;

/// Default histogram bucket boundaries (milliseconds, OTel duration shape).
pub const DEFAULT_DURATION_BUCKETS_MS: &[f64] = &[
    5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 2500.0, 5000.0, 7500.0,
    10000.0,
];

/// A monotonically-increasing counter (u64).
#[derive(Debug)]
pub struct Counter {
    name: String,
    unit: String,
    description: String,
    attributes: Vec<(String, AttrValue)>,
    value: Mutex<u64>,
}

impl Counter {
    /// Construct a new counter.
    pub fn new(name: impl Into<String>, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            description: String::new(),
            attributes: Vec::new(),
            value: Mutex::new(0),
        }
    }

    /// Set the human-readable description.
    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }

    /// Attach a static attribute.
    pub fn with_attribute(mut self, key: impl Into<String>, val: impl Into<AttrValue>) -> Self {
        self.attributes.push((key.into(), val.into()));
        self
    }

    /// Increment by `n`.
    pub fn add(&self, n: u64) {
        if let Ok(mut v) = self.value.lock() {
            *v = v.saturating_add(n);
        }
    }

    /// Read the current value.
    pub fn get(&self) -> u64 {
        self.value.lock().map(|g| *g).unwrap_or(0)
    }

    /// Metric name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Unit string.
    pub fn unit(&self) -> &str {
        &self.unit
    }
    /// Description.
    pub fn description(&self) -> &str {
        &self.description
    }
    /// Attribute view.
    pub fn attributes(&self) -> &[(String, AttrValue)] {
        &self.attributes
    }
}

/// A signed gauge (f64) — last-write-wins.
#[derive(Debug)]
pub struct Gauge {
    name: String,
    unit: String,
    description: String,
    attributes: Vec<(String, AttrValue)>,
    value: Mutex<f64>,
}

impl Gauge {
    /// Construct a new gauge.
    pub fn new(name: impl Into<String>, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            description: String::new(),
            attributes: Vec::new(),
            value: Mutex::new(0.0),
        }
    }

    /// Set the human-readable description.
    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }

    /// Attach a static attribute.
    pub fn with_attribute(mut self, key: impl Into<String>, val: impl Into<AttrValue>) -> Self {
        self.attributes.push((key.into(), val.into()));
        self
    }

    /// Record a new value (replaces previous).
    pub fn set(&self, v: f64) {
        if let Ok(mut slot) = self.value.lock() {
            *slot = v;
        }
    }

    /// Read the most-recent value.
    pub fn get(&self) -> f64 {
        self.value.lock().map(|g| *g).unwrap_or(0.0)
    }

    /// Metric name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Unit string.
    pub fn unit(&self) -> &str {
        &self.unit
    }
    /// Description.
    pub fn description(&self) -> &str {
        &self.description
    }
    /// Attribute view.
    pub fn attributes(&self) -> &[(String, AttrValue)] {
        &self.attributes
    }
}

/// A bucketed histogram.
#[derive(Debug)]
pub struct Histogram {
    name: String,
    unit: String,
    description: String,
    attributes: Vec<(String, AttrValue)>,
    boundaries: Vec<f64>,
    state: Mutex<HistogramState>,
}

#[derive(Debug)]
struct HistogramState {
    /// Count per bucket. Length = boundaries.len() + 1 (last is +inf).
    counts: Vec<u64>,
    /// Total observations.
    count: u64,
    /// Sum of observations.
    sum: f64,
    /// Smallest observation seen.
    min: f64,
    /// Largest observation seen.
    max: f64,
}

/// A snapshot of a histogram's current state, useful for export.
#[derive(Debug, Clone)]
pub struct HistogramSnapshot {
    /// Cumulative counts: `cumulative[i]` = observations ≤ `boundaries[i]`.
    pub cumulative: Vec<u64>,
    /// Bucket boundaries.
    pub boundaries: Vec<f64>,
    /// Total observation count.
    pub count: u64,
    /// Sum of observations.
    pub sum: f64,
    /// Min observation.
    pub min: f64,
    /// Max observation.
    pub max: f64,
}

impl Histogram {
    /// Construct with explicit boundaries (must be sorted ascending).
    pub fn new(name: impl Into<String>, unit: impl Into<String>, boundaries: Vec<f64>) -> Self {
        let n = boundaries.len() + 1;
        Self {
            name: name.into(),
            unit: unit.into(),
            description: String::new(),
            attributes: Vec::new(),
            boundaries,
            state: Mutex::new(HistogramState {
                counts: vec![0u64; n],
                count: 0,
                sum: 0.0,
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
            }),
        }
    }

    /// Construct with the default duration buckets.
    pub fn duration_ms(name: impl Into<String>) -> Self {
        Self::new(name, "ms", DEFAULT_DURATION_BUCKETS_MS.to_vec())
    }

    /// Set the human-readable description.
    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }

    /// Attach a static attribute.
    pub fn with_attribute(mut self, key: impl Into<String>, val: impl Into<AttrValue>) -> Self {
        self.attributes.push((key.into(), val.into()));
        self
    }

    /// Record one observation.
    pub fn record(&self, value: f64) {
        let mut idx = self.boundaries.len();
        for (i, b) in self.boundaries.iter().enumerate() {
            if value <= *b {
                idx = i;
                break;
            }
        }
        if let Ok(mut s) = self.state.lock() {
            s.counts[idx] = s.counts[idx].saturating_add(1);
            s.count = s.count.saturating_add(1);
            s.sum += value;
            if value < s.min {
                s.min = value;
            }
            if value > s.max {
                s.max = value;
            }
        }
    }

    /// Snapshot the current state, computing cumulative bucket counts.
    pub fn snapshot(&self) -> HistogramSnapshot {
        match self.state.lock() {
            Ok(g) => {
                let mut cumulative = Vec::with_capacity(self.boundaries.len() + 1);
                let mut running = 0u64;
                for c in g.counts.iter() {
                    running = running.saturating_add(*c);
                    cumulative.push(running);
                }
                HistogramSnapshot {
                    cumulative,
                    boundaries: self.boundaries.clone(),
                    count: g.count,
                    sum: g.sum,
                    min: if g.min == f64::INFINITY { 0.0 } else { g.min },
                    max: if g.max == f64::NEG_INFINITY { 0.0 } else { g.max },
                }
            }
            Err(_) => HistogramSnapshot {
                cumulative: vec![0u64; self.boundaries.len() + 1],
                boundaries: self.boundaries.clone(),
                count: 0,
                sum: 0.0,
                min: 0.0,
                max: 0.0,
            },
        }
    }

    /// Bucket boundaries.
    pub fn boundaries(&self) -> &[f64] {
        &self.boundaries
    }
    /// Metric name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Unit string.
    pub fn unit(&self) -> &str {
        &self.unit
    }
    /// Description.
    pub fn description(&self) -> &str {
        &self.description
    }
    /// Attribute view.
    pub fn attributes(&self) -> &[(String, AttrValue)] {
        &self.attributes
    }
}
