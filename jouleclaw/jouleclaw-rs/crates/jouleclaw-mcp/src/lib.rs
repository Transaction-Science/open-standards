//! # jouleclaw-mcp
//!
//! JouleClaw's MCP (Model Context Protocol) tool surface.
//!
//! ## Why this crate
//!
//! Standardising on MCP makes JouleClaw plug into the Claude Code /
//! Codex / Goose ecosystem without bespoke glue. But raw MCP measures
//! nothing about energy. This crate adds two things:
//!
//! 1. **Joule-metered tool calls.** Every dispatch routes through a
//!    [`MeteredTool`] wrapper that brackets the call with energy
//!    counter reads and pushes a `jouleclaw-prov::ToolTouch` onto the
//!    current cascade receipt.
//! 2. **The `joule-mcp` transport profile.** A capability-negotiated
//!    extension that allows two JouleClaw-aware endpoints to switch
//!    from JSON-RPC to length-prefixed CBOR for inner loops. Stays
//!    spec-compliant on the wire (any non-JouleClaw MCP client sees
//!    plain JSON-RPC); kicks in only when both sides advertise
//!    `x-jouleclaw/joule-mcp@1` in the handshake.
//!
//! ## What this crate is NOT
//!
//! - Not a re-implementation of MCP. The official Rust SDK is
//!   `modelcontextprotocol/rust-sdk` (`rmcp`). The recommended
//!   composition is to depend on `rmcp` downstream and wrap each
//!   server / client in `MeteredTool` from this crate. We don't
//!   pin `rmcp` here to keep `jouleclaw-mcp` reqwest- and
//!   tokio-feature-free.
//! - Not a transport. The CBOR profile defines envelope shape and
//!   negotiation, not the socket layer.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use async_trait::async_trait;
use jouleclaw_energy::{EnergyCounter, EnergyReading, Provenance};
use jouleclaw_prov::ToolTouch;
use serde::{Deserialize, Serialize};

/// The capability tag two JouleClaw-aware MCP endpoints advertise in
/// their handshake to opt into the binary-transport profile.
pub const JOULE_MCP_CAPABILITY: &str = "x-jouleclaw/joule-mcp@1";

/// Errors a metered MCP call can produce.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    /// The underlying tool returned an error.
    #[error("tool: {0}")]
    Tool(String),
    /// Failure reading the energy counter — the call still ran, but
    /// the receipt's joules_uj entry is `0` and provenance falls back
    /// to [`Provenance::Estimator`] with explicit drift band.
    #[error("energy counter: {0}")]
    EnergyCounter(String),
    /// Wire encoding failure (JSON or CBOR).
    #[error("encode: {0}")]
    Encode(String),
}

/// One pre-execution + post-execution energy reading pair, used to
/// price a single tool call.
#[derive(Debug, Clone)]
pub struct EnergyBracket {
    /// Counter reading taken just before the call dispatched.
    pub before: EnergyReading,
    /// Counter reading taken just after the call returned.
    pub after: EnergyReading,
    /// Provenance of the counter — load-bearing for the receipt.
    pub provenance: Provenance,
}

impl EnergyBracket {
    /// Microjoules consumed inside the bracket. Saturates on counter
    /// wrap rather than panicking.
    pub fn delta_uj(&self) -> u64 {
        self.after.uj.saturating_sub(self.before.uj)
    }
}

/// A tool wrapped with energy metering and receipt emission.
///
/// Implementations supply the inner tool dispatch; this trait drives
/// the bracket-and-account dance.
#[async_trait]
pub trait MeteredTool: Send + Sync {
    /// Stable tool identifier — `"mcp:filesystem/read"`,
    /// `"mcp:github/issue.create"`, etc. Appears verbatim in the
    /// receipt's `tools_touched` ledger.
    fn tool_id(&self) -> &str;

    /// Dispatch the inner tool. Implementations should be pure
    /// dispatch — no metering, no receipt mutation. This crate's
    /// [`dispatch_metered`] handles those.
    async fn dispatch(&self, request: &[u8]) -> Result<Vec<u8>, McpError>;
}

/// Drive a metered dispatch: pre-read the counter, call the tool,
/// post-read the counter, return the response bytes paired with a
/// [`ToolTouch`] ready to push into a `jouleclaw-prov::ReceiptBuilder`.
///
/// If the counter read fails, the response is still returned but the
/// `ToolTouch.joules_uj = 0` and `energy_provenance = Estimator` so
/// the receipt remains shape-correct (and the worst-counter floor
/// downgrades, alerting the auditor that this call wasn't honestly
/// measured).
pub async fn dispatch_metered<C: EnergyCounter + ?Sized>(
    tool: &dyn MeteredTool,
    counter: &C,
    request: &[u8],
) -> Result<(Vec<u8>, ToolTouch), McpError> {
    let before = counter.read().ok();
    let resp = tool.dispatch(request).await?;
    let after = counter.read().ok();

    let (joules_uj, provenance) = match (before, after) {
        (Some(a), Some(b)) => (b.uj.saturating_sub(a.uj), counter.provenance()),
        // Counter unavailable — receipt records zero cost and flags
        // Estimator so downstream auditors discount the entry.
        _ => (0, Provenance::Estimator),
    };

    let touch = ToolTouch {
        tool_id: tool.tool_id().to_string(),
        joules_uj,
        energy_provenance: provenance,
    };
    Ok((resp, touch))
}

/// Wire encoding for the joule-mcp transport profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WireEncoding {
    /// Plain MCP JSON-RPC over stdio/HTTP/SSE/WebSocket. Always
    /// available; required for non-JouleClaw clients.
    JsonRpc,
    /// Length-prefixed CBOR, advertised under
    /// [`JOULE_MCP_CAPABILITY`]. Roughly 30–50% lower per-call
    /// serialisation tax than JSON-RPC at the cost of opacity to
    /// generic tools.
    Cbor,
}

/// Choose the wire encoding given each side's advertised capabilities.
///
/// Returns [`WireEncoding::Cbor`] iff both endpoints advertise
/// [`JOULE_MCP_CAPABILITY`], otherwise [`WireEncoding::JsonRpc`].
pub fn negotiate(local_caps: &[String], remote_caps: &[String]) -> WireEncoding {
    let cap = JOULE_MCP_CAPABILITY.to_string();
    if local_caps.contains(&cap) && remote_caps.contains(&cap) {
        WireEncoding::Cbor
    } else {
        WireEncoding::JsonRpc
    }
}

/// Encode a value under the negotiated wire encoding.
pub fn encode<T: Serialize>(value: &T, wire: WireEncoding) -> Result<Vec<u8>, McpError> {
    match wire {
        WireEncoding::JsonRpc => {
            serde_json::to_vec(value).map_err(|e| McpError::Encode(e.to_string()))
        }
        WireEncoding::Cbor => {
            serde_cbor::to_vec(value).map_err(|e| McpError::Encode(e.to_string()))
        }
    }
}

/// Decode a value from the negotiated wire encoding.
pub fn decode<T: for<'de> Deserialize<'de>>(bytes: &[u8], wire: WireEncoding) -> Result<T, McpError> {
    match wire {
        WireEncoding::JsonRpc => {
            serde_json::from_slice(bytes).map_err(|e| McpError::Encode(e.to_string()))
        }
        WireEncoding::Cbor => {
            serde_cbor::from_slice(bytes).map_err(|e| McpError::Encode(e.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_energy::{EnergyDomain, EnergyError};

    struct EchoTool;
    #[async_trait]
    impl MeteredTool for EchoTool {
        fn tool_id(&self) -> &str { "test:echo" }
        async fn dispatch(&self, request: &[u8]) -> Result<Vec<u8>, McpError> {
            Ok(request.to_vec())
        }
    }

    /// A counter that emits ascending readings 0, 100, 200, … so the
    /// delta between two consecutive reads is exactly 100 μJ.
    struct StepCounter {
        cell: std::sync::atomic::AtomicU64,
        prov: Provenance,
    }
    impl EnergyCounter for StepCounter {
        fn domain(&self) -> EnergyDomain { EnergyDomain::SocTotal }
        fn provenance(&self) -> Provenance { self.prov }
        fn resolution_uj(&self) -> u64 { 1 }
        fn min_window_ns(&self) -> u64 { 1_000 }
        fn read(&self) -> Result<EnergyReading, EnergyError> {
            let prev = self.cell.fetch_add(100, std::sync::atomic::Ordering::Relaxed);
            Ok(EnergyReading {
                uj: prev,
                timestamp_ns: 0,
                domain: EnergyDomain::SocTotal,
                provenance: self.prov,
            })
        }
    }

    #[test]
    fn negotiate_falls_back_to_jsonrpc() {
        let local = vec![JOULE_MCP_CAPABILITY.to_string()];
        let remote: Vec<String> = vec![];
        assert_eq!(negotiate(&local, &remote), WireEncoding::JsonRpc);
    }

    #[test]
    fn negotiate_picks_cbor_when_both_advertise() {
        let cap = JOULE_MCP_CAPABILITY.to_string();
        assert_eq!(negotiate(&[cap.clone()], &[cap]), WireEncoding::Cbor);
    }

    #[test]
    fn encode_decode_round_trip_jsonrpc() {
        let v = serde_json::json!({"hello": "world"});
        let bytes = encode(&v, WireEncoding::JsonRpc).expect("encode");
        let back: serde_json::Value = decode(&bytes, WireEncoding::JsonRpc).expect("decode");
        assert_eq!(back, v);
    }

    #[test]
    fn encode_decode_round_trip_cbor() {
        let v = serde_json::json!({"hello": "world"});
        let bytes = encode(&v, WireEncoding::Cbor).expect("encode");
        let back: serde_json::Value = decode(&bytes, WireEncoding::Cbor).expect("decode");
        assert_eq!(back, v);
    }

    #[tokio::test]
    async fn dispatch_metered_emits_tool_touch() {
        let tool = EchoTool;
        let counter = StepCounter {
            cell: std::sync::atomic::AtomicU64::new(0),
            prov: Provenance::HwShunt,
        };
        let (resp, touch) = dispatch_metered(&tool, &counter, b"ping").await.expect("dispatch");
        assert_eq!(resp, b"ping");
        assert_eq!(touch.tool_id, "test:echo");
        assert_eq!(touch.joules_uj, 100);
        assert_eq!(touch.energy_provenance, Provenance::HwShunt);
    }
}
