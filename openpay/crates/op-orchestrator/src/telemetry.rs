//! OpenTelemetry instrumentation for the typestate machine.
//!
//! This module is the **lens**, not the machine. It exposes:
//!
//! - [`instrument_transition`] — emit a `tracing::Span` tagged with
//!   the semantic-conventions attributes operators expect on every
//!   `Payment<S>` state transition: `payment.id`, `payment.state.from`,
//!   `payment.state.to`, `payment.rail`, `payment.amount.minor`,
//!   `payment.amount.currency`. Callers wrap their transition body in
//!   `let _g = instrument_transition(...).entered();`.
//! - [`init_telemetry`] — opt-in initializer that wires the
//!   `tracing` subscriber to an OTLP exporter when the `telemetry`
//!   feature is enabled. Without the feature, it returns `Ok(())`
//!   silently so the same call site compiles in both modes.
//!
//! ## Why a helper instead of inline `tracing::span!`?
//!
//! The orchestrator's state machine lives across multiple files
//! (`engine.rs`, `circuit_breaker.rs`, the rail adapters). Centralising
//! the attribute names here means:
//!
//! 1. Every transition span carries the same schema — Grafana
//!    dashboards built on this schema don't break when a new
//!    transition is added.
//! 2. Semantic-conventions are documented in one place and reviewed
//!    once.
//! 3. The follow-up issue that wires this in to every state
//!    transition has a single function to call.
//!
//! ## Append-only by issue contract
//!
//! Per the spec, this module **does not modify any existing
//! orchestrator code**. It is additive: a sibling module exposing a
//! helper. The follow-up issue threads the helper through
//! `engine::Orchestrator::run` and the typestate-transition methods.
//!
//! ## Semantic conventions
//!
//! The attribute namespace `payment.*` follows the OpenTelemetry
//! semantic-conventions pattern (`http.*`, `db.*`, `messaging.*`).
//! Amounts are recorded as **minor units + currency** rather than
//! a formatted string so dashboards can aggregate without parsing.

use op_core::Money;
use tracing::{Level, Span, span};

/// Emit a structured span describing a single `Payment<S>` state
/// transition.
///
/// The span uses [`Level::INFO`] (transitions are first-class events,
/// not debug noise) and carries the following attributes:
///
/// | Attribute                  | Type      | Meaning                                      |
/// | -------------------------- | --------- | -------------------------------------------- |
/// | `payment.id`               | string    | Idempotency key / opaque payment id          |
/// | `payment.state.from`       | string    | Source typestate (e.g. `Authorized`)         |
/// | `payment.state.to`         | string    | Destination typestate (e.g. `Captured`)      |
/// | `payment.rail`             | string    | Rail name (`card`, `a2a`, `crypto`)          |
/// | `payment.amount.minor`     | i64       | Amount in minor units (cents for USD)        |
/// | `payment.amount.currency`  | string    | ISO 4217 alpha-3 currency code               |
///
/// Callers entered the span around the transition body:
///
/// ```ignore
/// use op_core::Money;
/// use op_orchestrator::telemetry::instrument_transition;
///
/// let amount = Money::from_minor(1_999, op_core::Currency::USD);
/// let span = instrument_transition("pay_123", "Authorized", "Captured", "card", &amount);
/// let _guard = span.enter();
/// // ... do the transition ...
/// ```
///
/// When no subscriber is registered the span is a no-op (zero cost
/// modulo a few atomic loads). When a subscriber is registered the
/// attributes appear on every emitted event inside the guard's scope.
#[must_use]
pub fn instrument_transition(
    payment_id: &str,
    from_state: &str,
    to_state: &str,
    rail: &str,
    amount: &Money,
) -> Span {
    span!(
        Level::INFO,
        "payment.transition",
        payment.id = payment_id,
        payment.state.from = from_state,
        payment.state.to = to_state,
        payment.rail = rail,
        payment.amount.minor = amount.minor_units,
        payment.amount.currency = amount.currency.code(),
    )
}

/// Errors that [`init_telemetry`] can surface.
#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    /// OTLP exporter construction failed (bad endpoint, missing
    /// tonic features, etc.).
    #[error("failed to build OTLP exporter: {0}")]
    ExporterBuild(String),
    /// Subscriber registration failed (another global subscriber is
    /// already installed).
    #[error("failed to install global tracing subscriber: {0}")]
    SubscriberInstall(String),
}

/// Result alias for telemetry initialization.
pub type TelemetryResult<T> = std::result::Result<T, TelemetryError>;

// ============================================================
// `telemetry` feature ON — wire `tracing` to an OTLP exporter.
// ============================================================

/// Initialize the OpenTelemetry pipeline.
///
/// When the `telemetry` feature is **enabled**, this:
///
/// 1. Builds an OTLP gRPC exporter pointed at `otlp_endpoint`
///    (defaults to `http://127.0.0.1:4317` if `None`).
/// 2. Wraps it in a batch span processor.
/// 3. Installs a `tracing-subscriber` layer that forwards every
///    `tracing` span (including the ones emitted by
///    [`instrument_transition`]) to the exporter.
///
/// When the feature is **disabled**, this is a `Ok(())` no-op so
/// the same call site compiles in both modes.
///
/// # Errors
/// Returns [`TelemetryError`] if the exporter or subscriber install
/// fails.
#[cfg(feature = "telemetry")]
pub fn init_telemetry(otlp_endpoint: Option<&str>) -> TelemetryResult<()> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::WithExportConfig;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let endpoint = otlp_endpoint.unwrap_or("http://127.0.0.1:4317");

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
        .map_err(|e| TelemetryError::ExporterBuild(e.to_string()))?;

    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .build();
    let tracer = provider.tracer("op-orchestrator");

    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
    tracing_subscriber::registry()
        .with(otel_layer)
        .try_init()
        .map_err(|e| TelemetryError::SubscriberInstall(e.to_string()))?;
    Ok(())
}

// ============================================================
// `telemetry` feature OFF — no-op stub.
// ============================================================

/// Initialize the OpenTelemetry pipeline.
///
/// See the feature-enabled version above for the real documentation;
/// this is the stub used when the `telemetry` feature is **off**.
#[cfg(not(feature = "telemetry"))]
#[allow(unused_variables)]
pub fn init_telemetry(otlp_endpoint: Option<&str>) -> TelemetryResult<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money};

    #[test]
    fn instrument_transition_emits_named_span() {
        let amount = Money::from_minor(1_999, Currency::USD);
        let span = instrument_transition("pay_abc", "Pending", "Authorized", "card", &amount);
        // The span has the canonical name; subscribers downstream
        // will see the attributes. We only assert the name here
        // because attribute introspection is not stable in
        // `tracing::Span`.
        assert_eq!(span.metadata().map(|m| m.name()), Some("payment.transition"));
    }

    #[test]
    fn init_telemetry_no_feature_is_ok() {
        // With the `telemetry` feature off, init is a no-op and
        // never fails. With the feature on, this test would only
        // succeed if an OTLP collector is reachable, so we just
        // assert the no-op contract.
        #[cfg(not(feature = "telemetry"))]
        {
            assert!(init_telemetry(None).is_ok());
        }
    }
}
