//! # Efficient GenAI
//!
//! Hardware-first generative AI infrastructure for real-time generation.
//!
//! ## Design Principles
//!
//! 1. **Memory as Performance Lever** - Streaming execution, memory-aware scheduling
//! 2. **True Hardware Parallelization** - Saturate all execution units
//! 3. **Runtime Discovery** - JIT optimization based on actual hardware
//! 4. **Streaming-First** - Results emerge as inputs arrive
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     High-Level API                          │
//! │  (Text, Image, Video, Audio, 3D generation interfaces)      │
//! ├─────────────────────────────────────────────────────────────┤
//! │                   Modality Handlers                         │
//! │  (Specialized processing for each media type)               │
//! ├─────────────────────────────────────────────────────────────┤
//! │                      Runtime                                │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
//! │  │  Scheduler  │  │   Memory    │  │  Execution  │         │
//! │  │             │  │   Manager   │  │   Engine    │         │
//! │  └─────────────┘  └─────────────┘  └─────────────┘         │
//! ├─────────────────────────────────────────────────────────────┤
//! │              Hardware Abstraction Layer                     │
//! │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐    │
//! │  │  CUDA  │ │ Metal  │ │ ROCm   │ │Vulkan  │ │ RISC-V │    │
//! │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘    │
//! └─────────────────────────────────────────────────────────────┘
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
// Enable compiler warnings for code quality
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
// Only allow these specific patterns where necessary
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(dead_code)]
#![allow(unused_variables)]

extern crate alloc;

pub mod core;
pub mod hal;
pub mod runtime;
pub mod tensor;
pub mod modalities;
pub mod inference;
/// JouleDB weight store bridge for on-demand tensor loading.
#[cfg(feature = "jouledb")]
pub mod weight_store;
#[cfg(feature = "server")]
pub mod server;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::core::{Error, Result};
    pub use crate::tensor::{Tensor, TensorView, DType};
    pub use crate::hal::{Device, DeviceInfo};
    pub use crate::runtime::{Runtime, StreamingOutput};
    pub use crate::inference::{Engine, TextParams, ImageParams};
}

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
