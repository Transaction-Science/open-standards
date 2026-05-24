//! Joule-attribution integration test.
//!
//! Verifies the contract: when the meter is a [`StubCounter`] the
//! reported joule cost is 0; when the meter is a non-stub counter
//! whose readings monotonically advance, the reported joule cost is
//! non-zero.
//!
//! These tests exercise the *plumbing* — the backends themselves use
//! their respective placeholder implementations in the reference
//! build. The contract is the same regardless of whether the backend
//! is wired to a real native runtime or to a placeholder.

#![forbid(unsafe_code)]
#![allow(unused_imports)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use eoc_core::{JouleSource, Query};
use eoc_meter::{JouleCounter, StubCounter};

/// A counter that returns a monotonically increasing value each time
/// it is read. Useful for asserting that the sandwich-meter logic in
/// the backends actually picks up a non-zero delta.
#[derive(Debug, Default)]
struct TickingCounter {
    state: AtomicU64,
    step: u64,
}

impl TickingCounter {
    fn new(step: u64) -> Self {
        Self {
            state: AtomicU64::new(0),
            step,
        }
    }
}

impl JouleCounter for TickingCounter {
    fn read_microjoules(&self) -> eoc_core::Result<u64> {
        Ok(self.state.fetch_add(self.step, Ordering::Relaxed) + self.step)
    }
    fn name(&self) -> &'static str {
        "ticking-test-counter"
    }
}

#[cfg(feature = "mlc")]
#[tokio::test]
async fn mlc_stub_counter_yields_zero_cost() {
    use eoc_local::MlcBackend;
    use eoc_neural::NeuralBackend;
    let dir = tempfile::tempdir().unwrap();
    let b = MlcBackend::new(dir.path(), Arc::new(StubCounter)).unwrap();
    let q = Query::new("hello");
    let r = b.infer(&q).await;
    assert_eq!(r.joule_cost.microjoules, 0);
}

#[cfg(feature = "mlc")]
#[tokio::test]
async fn mlc_ticking_counter_yields_measured_nonzero_cost() {
    use eoc_local::MlcBackend;
    use eoc_neural::NeuralBackend;
    let dir = tempfile::tempdir().unwrap();
    let meter: Arc<dyn JouleCounter> = Arc::new(TickingCounter::new(10_000));
    let b = MlcBackend::new(dir.path(), meter).unwrap();
    let q = Query::new("hello");
    let r = b.infer(&q).await;
    assert!(r.joule_cost.microjoules > 0);
    assert_eq!(r.joule_cost.source, JouleSource::Measured);
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[tokio::test]
async fn mlx_stub_counter_uses_estimated_fallback() {
    use eoc_local::MlxBackend;
    use eoc_neural::NeuralBackend;
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model");
    std::fs::create_dir(&model).unwrap();
    let tok = model.join("tokenizer.json");
    std::fs::write(&tok, b"{}").unwrap();
    let b = MlxBackend::new(&model, &tok, Arc::new(StubCounter))
        .unwrap()
        .with_max_tokens(10)
        .with_fallback_joules_per_token(0.05);
    let q = Query::new("ping");
    let r = b.infer(&q).await;
    // With a stub counter the backend should fall back to its estimated
    // coefficient: 10 * 0.05 J = 500,000 µJ.
    assert_eq!(r.joule_cost.microjoules, 500_000);
    assert_eq!(r.joule_cost.source, JouleSource::Estimated);
}

// Note: an onnx-backend test could mirror the mlc one above, but the
// `ort-sys` crate requires either a `download-binaries` feature or an
// ORT_LIB_PATH pointing at libonnxruntime to *link* a test binary.
// Library compilation succeeds without that — see the unit tests in
// `src/onnx.rs` — but integration tests cannot be linked in a
// hermetic sandbox without the runtime present. Operators with ONNX
// Runtime installed can re-enable the test by flipping the `ort`
// dependency's `download-binaries` feature in `Cargo.toml`.

/// When no backend feature is enabled, this test crate still compiles
/// and one trivial test runs so cargo test does not complain about an
/// empty suite.
#[test]
fn ticking_counter_advances() {
    let c = TickingCounter::new(7);
    let a = c.read_microjoules().unwrap();
    let b = c.read_microjoules().unwrap();
    assert!(b > a);
}
