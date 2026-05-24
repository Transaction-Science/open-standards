//! `eoc-observability` — OpenTelemetry-compatible tracing, metrics, and logs
//! for energy-optimized AI compute.
//!
//! This crate is the observability seam for the EOC cascade. It is
//! deliberately *not* a vendored copy of `opentelemetry-rust`; it is a
//! minimal, self-contained, runtime-agnostic implementation of the same
//! data model, so EOC can deploy on `wasm32-unknown-unknown` and embedded
//! targets where the upstream SDK is too heavy.
//!
//! # What's included
//!
//! - Spans + [`SpanContext`] + [`SpanKind`] + [`Status`] ([`span`]).
//! - W3C `traceparent` + `tracestate` + `baggage` ([`context`]).
//! - Samplers: [`sampler::AlwaysOnSampler`], [`sampler::AlwaysOffSampler`],
//!   [`sampler::TraceIdRatioBased`], [`sampler::ParentBased`].
//! - [`processor::BatchSpanProcessor`].
//! - [`exporter::SpanExporter`] trait + [`exporter::InMemoryExporter`].
//! - Metrics: [`metric::Counter`], [`metric::Gauge`], [`metric::Histogram`].
//! - Prometheus text exposition ([`prometheus`]).
//! - StatsD / DogStatsD line format ([`statsd`]).
//! - GenAI semantic conventions ([`genai_conventions`]).
//! - Cost attribution: USD + gCO2e ([`cost`]).
//! - LangSmith ([`langsmith`]) and LangFuse ([`langfuse`]) exporters.
//! - Log records with trace correlation ([`log`]).
//!
//! # Integration with `eoc-meter` and `eoc-carbon`
//!
//! `Span::set_joule_cost` accepts an [`eoc_core::JouleCost`] coming from
//! `eoc-meter`. The [`cost::attribute`] helper turns that joule reading
//! into USD + gCO2e for a given grid intensity and PUE. Stage-level joule
//! counters are typically driven directly from `eoc-meter`; this crate just
//! shapes them into OTel attributes.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod context;
pub mod cost;
pub mod error;
pub mod exporter;
pub mod genai_conventions;
pub mod langfuse;
pub mod langsmith;
pub mod log;
pub mod metric;
pub mod processor;
pub mod prometheus;
pub mod sampler;
pub mod span;
pub mod statsd;

pub use context::{Baggage, SpanContext, SpanId, TraceFlags, TraceId};
pub use cost::{
    CarbonIntensityGCo2ePerKwh, CostAttribution, EnergyPriceUsdPerKwh, Pue, attribute,
    joules_to_kwh,
};
pub use error::{ObsError, ObsResult};
pub use exporter::{InMemoryExporter, SpanExporter};
pub use genai_conventions::GenAiAttributes;
pub use langfuse::{LangFuseExporter, span_to_observation};
pub use langsmith::{LangSmithExporter, span_to_run};
pub use log::{LogRecord, Severity};
pub use metric::{
    Counter, DEFAULT_DURATION_BUCKETS_MS, Gauge, Histogram, HistogramSnapshot,
};
pub use processor::{BatchConfig, BatchSpanProcessor};
pub use prometheus::PrometheusExposer;
pub use sampler::{
    AlwaysOffSampler, AlwaysOnSampler, ParentBased, Sampler, SamplingDecision,
    TraceIdRatioBased,
};
pub use span::{AttrValue, Span, SpanEvent, SpanKind, SpanLink, Status, StatusCode};
pub use statsd::StatsdLine;
