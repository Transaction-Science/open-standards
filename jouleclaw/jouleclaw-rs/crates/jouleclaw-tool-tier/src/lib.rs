//! JouleClaw L0.5 — deterministic tool-compute tier.
//!
//! Wraps the 484-tool [`jouleclaw_tools`] registry as a
//! [`jouleclaw_cascade::Tier`]. When the incoming query parses as a pure-
//! function call (numeric / string / date / hash / unit / percentage / etc.)
//! the tier dispatches to the corresponding tool and returns a structured
//! answer with confidence `1.0`. The L0.25 formula tier above and the L0.75
//! SSM router below never see the query.
//!
//! Cost model (flat — pure CPU, zero network, zero GPU, zero LLM):
//!
//! ```text
//! joules           = 15 µJ
//! latency          = 5 µs
//! confidence_floor = 1.0   (deterministic; matched queries cannot lie)
//! ```
//!
//! The cost is constant rather than measured because the work is bounded
//! (microsecond pure computation) and dwarfed by the surrounding tier walk;
//! a sampler here would consume more energy than it accounts for.
//!
//! ## Ported subset
//!
//! The donor `verity-cascade::layers::l05_tool_compute` ships a 10k-LOC
//! query→tool matcher across the full 484-tool surface. This port carries
//! the *adapter* faithfully and a focused, well-tested subset of routers
//! covering the high-traffic shapes: math, unit conversion, percentage,
//! UUID, SHA-256 / MD5 / generic hash, base64 / URL en-/de-code, case
//! transforms, word count, statistics, day-of-week, "now", and bytes-
//! formatting. The remainder of the 484 variants remain reachable via
//! [`ToolRouter::register_tool`] (programmatic registration) so downstream
//! crates can extend the router without re-implementing dispatch.

#![forbid(unsafe_code)]

mod router;
mod tier;

pub use router::{ToolMatch, ToolRouter};
pub use tier::{ToolTier, ToolTierError, TOOL_TIER_CONFIDENCE_FLOOR, TOOL_TIER_JOULES, TOOL_TIER_LATENCY};
