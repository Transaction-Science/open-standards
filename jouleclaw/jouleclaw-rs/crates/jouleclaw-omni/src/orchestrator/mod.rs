//! Orchestrator — API gateway and worker fleet management for Create.
//!
//! The orchestrator is a lightweight Axum HTTP service that sits in front of
//! one or more Metal GPU inference workers. It handles:
//! - Worker registration and health monitoring
//! - Request routing based on loaded models and queue depth
//! - SSE stream proxying (text/image generation)
//! - Auto-scaling via cloud provider APIs
//! - Authentication, rate limiting, circuit breaking, and metrics
//!
//! The orchestrator does NOT depend on the `metal` feature and can run on any platform.

pub mod types;
pub mod registry;
pub mod health;
pub mod middleware;
pub mod router;
pub mod scaler;
pub mod server;
