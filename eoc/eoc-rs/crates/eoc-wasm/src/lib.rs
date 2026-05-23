//! Browser-runnable EOC cascade.
//!
//! This crate exposes a minimal `wasm-bindgen` surface so the EOC cascade
//! can be driven from JavaScript. The build target is
//! `wasm32-unknown-unknown`:
//!
//! ```text
//! cargo build -p eoc-wasm --target wasm32-unknown-unknown --release
//! wasm-bindgen target/wasm32-unknown-unknown/release/eoc_wasm.wasm \
//!     --out-dir ./pkg --target web
//! ```
//!
//! The meter is omitted in WASM — RAPL/NVML/powermetrics do not compile
//! to `wasm32-unknown-unknown`. The cascade falls back to estimated
//! joule cost reported by each stage.

#![forbid(unsafe_code)]

use std::sync::Arc;

use eoc_cache::LruCache;
use eoc_cascade::Cascade;
use eoc_core::Query;
use eoc_graph::{GraphStage, Triple};
use eoc_kv::{KvBackend, KvStage, MemoryKvBackend};
use eoc_neural::{EchoBackend, NeuralStage};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct JsResponse {
    stage: String,
    payload: String,
    microjoules: u64,
    source: String,
    receipt_hex: String,
}

/// A WASM-friendly handle to a fully wired cascade.
#[wasm_bindgen]
pub struct WasmCascade {
    inner: Cascade,
}

#[wasm_bindgen]
impl WasmCascade {
    /// Construct a demo cascade with a seeded graph and KV.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmCascade {
        let cache = Arc::new(LruCache::new(256));
        let kv_backend = Box::new(MemoryKvBackend::new());
        kv_backend.put("ping", b"pong".to_vec());
        let kv = Arc::new(KvStage::new(kv_backend));
        let graph = Arc::new(GraphStage::new());
        graph.insert(Triple::new("Paris", "capital of", "France"));
        let neural = Arc::new(NeuralStage::new(Box::new(EchoBackend::new())));
        let inner = Cascade::new(cache, kv, graph, neural);
        Self { inner }
    }

    /// Resolve a prompt through the cascade. Returns a JS object
    /// `{ stage, payload, microjoules, source, receipt_hex }`.
    #[wasm_bindgen(js_name = resolve)]
    pub async fn resolve_js(&self, prompt: String) -> Result<JsValue, JsValue> {
        let q = Query::new(prompt);
        let r = self.inner.resolve(q).await;
        let out = JsResponse {
            stage: r.stage.to_string(),
            payload: r.payload,
            microjoules: r.joule_cost.microjoules,
            source: match r.joule_cost.source {
                eoc_core::JouleSource::Measured => "measured".to_string(),
                eoc_core::JouleSource::Estimated => "estimated".to_string(),
            },
            receipt_hex: r.receipt.to_hex(),
        };
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for WasmCascade {
    fn default() -> Self {
        Self::new()
    }
}
