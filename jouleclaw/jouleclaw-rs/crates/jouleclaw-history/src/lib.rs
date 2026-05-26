//! Joule history layer — concrete `HistoryLayer` implementations.
//!
//! The `HistoryLayer` trait lives in `jouleclaw-cascade`. This crate
//! provides two implementations:
//!
//!   `InMemoryHistory`  — HashMap-backed, ephemeral.
//!   `DiskHistory`      — append-only log file, durable across runtime
//!                        restarts.
//!
//! Both implement the same trait so the cascade `Runtime` can use
//! either via `Runtime::new_with_history`.

pub mod in_memory;
pub mod disk;
pub mod semantic;
pub mod warm;
pub mod tiered;

pub use in_memory::{InMemoryHistory, L0Tier};
pub use disk::DiskHistory;
pub use semantic::{SemanticHistory, IndexedHistory, estimate_semantic_lookup_cost};
pub use warm::WarmCache;
pub use tiered::TieredMemory;
