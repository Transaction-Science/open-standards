//! # jouleclaw-runtime
//!
//! Phase 1.1: minimum viable runtime end-to-end on the reference backend.
//! Phase 1.2: validation harness + Apple Silicon scaffolding.

pub mod compile;
pub mod execute;
pub mod validate;
pub mod generate;
pub mod streaming;
pub mod embed;
pub mod chat;
pub mod pld;
pub mod drafter;

pub use compile::{compile, CompiledGraph, NodePlan};
pub use execute::{execute, ExecutionOptions, ExecutionResult};
pub use validate::{
    run_kernel, validate_against_reference, validate_against_reference_fp,
    validate_all_against_reference, FpTolerance,
    ValidationReport, ValidationStatus,
};
pub use generate::{generate, GenerateConfig, GenerateError, GenerateResult, KvCacheKind, TokenizerKind};
pub use streaming::{generate_stream, Conversation, ConversationStream, GenerateStream, StreamedToken};
pub use embed::{embed, cosine_similarity, EmbedConfig, EmbedResult, Pooling};
pub use chat::{
    encode_user_turn, encode_with_specials, ChatMessage, ChatRole, ChatTemplate,
};
pub use pld::{find_draft, PldConfig, PldOutcome};
pub use drafter::{DrafterConfig, DrafterOutcome};
pub use streaming::extend_with_drafter;

use jouleclaw_core::kernel::Kernel;
use jouleclaw_topology::Topology;
use std::sync::Arc;

/// The runtime. One per process (typically).
pub struct Runtime {
    pub topology: Topology,
    pub kernels: KernelRegistry,
}

impl Runtime {
    /// Build a runtime by discovering the host's topology and registering
    /// available backends.
    pub fn boot() -> Self {
        let mut kernels = KernelRegistry::new();

        // Reference backend: always present, the determinism oracle.
        for k in jouleclaw_backend_reference::all_kernels() {
            kernels.register(k);
        }

        // Apple Silicon backend: registers only on macOS aarch64; no-op
        // elsewhere.
        for k in jouleclaw_backend_apple::all_kernels() {
            kernels.register(k);
        }

        Self {
            topology: jouleclaw_topology::discover(),
            kernels,
        }
    }

    /// Build a runtime with **only** the reference backend registered —
    /// the determinism oracle, no accelerators. Use this when an
    /// accelerator backend has a known correctness gap (e.g. the
    /// AppleAmx matmul kernel's batched inner-dim bug) or when
    /// bit-exact cross-platform reproducibility is required over speed.
    pub fn reference_only() -> Self {
        let mut kernels = KernelRegistry::new();
        for k in jouleclaw_backend_reference::all_kernels() {
            kernels.register(k);
        }
        Self {
            topology: jouleclaw_topology::discover(),
            kernels,
        }
    }

    /// Build a runtime with no kernels registered (for tests).
    pub fn empty() -> Self {
        Self {
            topology: jouleclaw_topology::discover(),
            kernels: KernelRegistry::new(),
        }
    }
}

/// Registry of available kernels.
pub struct KernelRegistry {
    kernels: Vec<Arc<dyn Kernel>>,
}

impl KernelRegistry {
    pub fn new() -> Self { Self { kernels: Vec::new() } }
    pub fn register(&mut self, kernel: Arc<dyn Kernel>) { self.kernels.push(kernel); }
    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn Kernel>> { self.kernels.iter() }
}

impl Default for KernelRegistry {
    fn default() -> Self { Self::new() }
}
