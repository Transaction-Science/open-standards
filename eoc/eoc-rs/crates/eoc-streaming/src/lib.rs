//! EOC — streaming inference primitives.
//!
//! Provider streaming has settled on a handful of patterns: line-oriented
//! SSE (`data: ...\n\n`), Anthropic's typed `event:` / `data:` pairs,
//! OpenAI's `chat.completion.chunk` deltas, the Vercel AI SDK protocol
//! (text-delta / tool-call / tool-result / finish), and WebSocket frames
//! carrying the same payloads. This crate normalizes all of those into a
//! single [`Event`] enum and a single [`TokenStream`] sink so the rest of
//! the EOC stack (cascade, agent, vendor-api) can treat streams as a
//! first-class resource with:
//!
//! * **Backpressure** via bounded channels with high/low watermarks
//!   ([`backpressure`]).
//! * **Cancellation** propagated from client disconnects to upstream
//!   providers ([`cancel`]).
//! * **Resumable streams** keyed on `Last-Event-ID` ([`resume`]).
//! * **Per-stream joule accounting** anchored on [`eoc_meter`]
//!   ([`account`]).
//!
//! ## Discipline
//!
//! `#![forbid(unsafe_code)]`. No `.unwrap()` outside `#[cfg(test)]`. The
//! crate is `serde`-typed end-to-end; transport bindings (HTTP server,
//! WebSocket library) are intentionally **out of scope** so the same
//! primitives can ride on whatever runtime the host picks.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod account;
pub mod anthropic;
pub mod backpressure;
pub mod cancel;
pub mod error;
pub mod openai;
pub mod resume;
pub mod sse;
pub mod stream;
pub mod vercel_ai;
pub mod ws;

pub use account::{StreamAccount, StreamMeter};
pub use anthropic::AnthropicMapper;
pub use backpressure::{BoundedStream, Watermarks};
pub use cancel::CancelToken;
pub use error::{StreamError, StreamResult};
pub use openai::OpenAiMapper;
pub use resume::{EventLog, LastEventId};
pub use sse::{SseEvent, SseParser};
pub use stream::{Event, FinishReason, Role, Sink, TokenStream};
pub use vercel_ai::VercelAiMapper;
pub use ws::{WsFrame, WsOpcode};
