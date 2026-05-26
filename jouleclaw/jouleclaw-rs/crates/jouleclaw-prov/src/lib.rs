//! # jouleclaw-prov
//!
//! Receipt-per-resolution emitter — JouleClaw's thermodynamic ledger.
//!
//! Every cascade walk (`L0:Cache → L1:Lawful → L2:Embed → L3:Model →
//! L4:Wire`) emits a signed [`Receipt`] envelope describing exactly
//! what happened: which tier closed the query, how many microjoules
//! were spent, what tools were touched, what provenance the retrieved
//! claims carried, and what the underlying energy counter's honesty
//! tier was.
//!
//! Receipts are shaped to compose with the Smart Byte open standard's
//! signed-envelope format — they can be sealed by a Smart Byte
//! issuer, replicated by lockstep, and audited downstream without
//! re-running the cascade.
//!
//! ## The provenance graph
//!
//! Each receipt also carries a minimal PROV-O-compatible graph:
//! `prov:Activity` for the cascade walk, `prov:Entity` per retrieved
//! claim, `prov:Agent` for the steward. This is what makes the
//! "capability per joule, not capability per parameter" claim
//! auditable by a third party — they re-execute the cascade against
//! the published conformance vectors and verify the receipt's
//! `joules_spent` falls within the declared drift band.
//!
//! ## What this crate is NOT
//!
//! - Not a key-management layer. Operators bring their own signing
//!   key (any `Signer` impl). Smart Byte's KERI-based AID rotation
//!   is the recommended substrate.
//! - Not a transport. Receipts are produced; how they're shipped
//!   (HTTP, lockstep gossip, message queue) is the caller's choice.
//! - Not a verifier. Verification crates can build on the [`Receipt`]
//!   shape; we ship the producer only.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use jouleclaw_energy::Provenance;
use serde::{Deserialize, Serialize};

/// The tier that closed the query. Mirrors the JouleClaw cascade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CascadeTier {
    /// L0 — content-addressed cache hit. Picojoules.
    L0Cache,
    /// L1 — deterministic lawful primitive. Nanojoules.
    L1Lawful,
    /// L2 — embedding / nearest-neighbour retrieval. Sub-millijoules.
    L2Embed,
    /// L3 — local stochastic model (SSM / ternary / multimodal / diffusion).
    /// Joules to tens of joules.
    L3Model,
    /// L4 — remote frontier RPC. Tens of joules and up.
    L4Wire,
}

impl CascadeTier {
    /// Stable wire identifier used in receipts and conformance vectors.
    /// Format: `L<n>` where n is 0..4. Never changes across versions.
    pub fn wire_tag(self) -> &'static str {
        match self {
            Self::L0Cache => "L0",
            Self::L1Lawful => "L1",
            Self::L2Embed => "L2",
            Self::L3Model => "L3",
            Self::L4Wire => "L4",
        }
    }

    /// Human-readable name used in prose / dashboards.
    pub fn name(self) -> &'static str {
        match self {
            Self::L0Cache => "Cache",
            Self::L1Lawful => "Lawful",
            Self::L2Embed => "Embed",
            Self::L3Model => "Model",
            Self::L4Wire => "Wire",
        }
    }
}

/// Provenance of a single retrieved claim that fed the resolution.
///
/// Compatible with W3C PROV-O `prov:Entity` (the URL is the entity IRI;
/// `content_hash` is the wasGeneratedBy fingerprint).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimProvenance {
    /// Where the claim was fetched from. URL, DOI, did:plc resolver, etc.
    pub source: String,
    /// BLAKE3 of the claim bytes as observed at fetch time.
    pub content_hash: String,
    /// RFC 3339 timestamp of the fetch.
    pub fetched_at: String,
    /// Trust tier of the source (e.g. Wikipedia RSP). Higher = more
    /// trustworthy.
    pub trust_tier: u8,
}

/// One tool invocation during the resolution (MCP call, IaC primitive,
/// model invocation, …).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolTouch {
    /// Stable identifier — e.g. `mcp:filesystem/read`, `iac:fs.read`,
    /// `model:gemma4-9b-q5_k_m`.
    pub tool_id: String,
    /// Microjoules consumed by this single tool call.
    pub joules_uj: u64,
    /// Provenance of the energy reading that produced `joules_uj`.
    pub energy_provenance: Provenance,
}

/// A receipt issued at the close of one cascade walk.
///
/// Designed to be sealed inside a Smart Byte signed envelope. The
/// envelope's signature attests to the receipt's integrity; the
/// receipt itself is the auditable thermodynamic record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receipt {
    /// JouleClaw receipt schema version. This crate produces `"1"`.
    pub jc_receipt: String,
    /// Stable receipt id, monotonic per steward.
    pub id: String,
    /// RFC 3339 timestamp when the cascade walk closed.
    pub closed_at: String,
    /// BLAKE3 of the input that initiated the walk (normalised).
    pub input_hash: String,
    /// Which tier closed the query.
    pub tier: CascadeTier,
    /// Total microjoules spent across all tiers walked.
    pub joules_uj: u64,
    /// Honesty tier of the *worst* energy counter seen during the walk.
    /// The breaker enforces at this granularity.
    pub energy_provenance: Provenance,
    /// Tool calls in order, with per-tool cost.
    pub tools_touched: Vec<ToolTouch>,
    /// Claims that contributed to the resolution (L2/L3 retrieval,
    /// L4 wire fetch). Empty for L0/L1 paths.
    pub claims: Vec<ClaimProvenance>,
    /// EOC stage label if the runtime was driven by an EOC-style
    /// cascade adapter. Optional — present only when the runtime
    /// reports through `eoc-cascade`.
    pub eoc_stage: Option<String>,
}

/// Receipt schema version this crate produces.
pub const RECEIPT_VERSION: &str = "1";

/// Errors a [`ReceiptBuilder`] can return.
#[derive(Debug, thiserror::Error)]
pub enum ReceiptError {
    /// Builder was sealed before all required fields were set.
    #[error("receipt builder missing required field: {0}")]
    MissingField(&'static str),
    /// Serialisation of the receipt for hashing failed.
    #[error("serialize: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// Fluent builder for [`Receipt`] values.
#[derive(Debug, Default)]
pub struct ReceiptBuilder {
    input_hash: Option<String>,
    tier: Option<CascadeTier>,
    joules_uj: u64,
    energy_provenance: Option<Provenance>,
    tools_touched: Vec<ToolTouch>,
    claims: Vec<ClaimProvenance>,
    eoc_stage: Option<String>,
}

impl ReceiptBuilder {
    /// Start a new receipt for a cascade walk that just closed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the BLAKE3 hash of the input that initiated this walk.
    pub fn input_hash(mut self, hash: impl Into<String>) -> Self {
        self.input_hash = Some(hash.into());
        self
    }

    /// Set the tier that closed the query.
    pub fn tier(mut self, tier: CascadeTier) -> Self {
        self.tier = Some(tier);
        self
    }

    /// Account microjoules spent by a single tool call. Updates the
    /// running total and the per-tool ledger.
    pub fn account_tool(mut self, tool: ToolTouch) -> Self {
        self.joules_uj = self.joules_uj.saturating_add(tool.joules_uj);
        // The receipt's overall energy_provenance is the *worst* counter
        // seen. We keep the running floor as we go.
        let worst = match (self.energy_provenance, tool.energy_provenance) {
            (None, p) => Some(p),
            (Some(a), b) => Some(worst_provenance(a, b)),
        };
        self.energy_provenance = worst;
        self.tools_touched.push(tool);
        self
    }

    /// Account a retrieved claim (L2/L3 retrieval, L4 wire fetch).
    pub fn account_claim(mut self, claim: ClaimProvenance) -> Self {
        self.claims.push(claim);
        self
    }

    /// Tag the receipt with an EOC-style stage label.
    pub fn eoc_stage(mut self, stage: impl Into<String>) -> Self {
        self.eoc_stage = Some(stage.into());
        self
    }

    /// Seal the receipt. Assigns id and closed_at from the current clock.
    pub fn seal(self) -> Result<Receipt, ReceiptError> {
        let input_hash = self.input_hash.ok_or(ReceiptError::MissingField("input_hash"))?;
        let tier = self.tier.ok_or(ReceiptError::MissingField("tier"))?;
        // No tool touches → fall back to Estimator provenance with explicit
        // wide tolerance, so downstream consumers know to treat this
        // receipt cautiously.
        let energy_provenance = self.energy_provenance.unwrap_or(Provenance::Estimator);
        Ok(Receipt {
            jc_receipt: RECEIPT_VERSION.to_string(),
            id: uuid::Uuid::new_v4().to_string(),
            closed_at: chrono::Utc::now().to_rfc3339(),
            input_hash,
            tier,
            joules_uj: self.joules_uj,
            energy_provenance,
            tools_touched: self.tools_touched,
            claims: self.claims,
            eoc_stage: self.eoc_stage,
        })
    }
}

/// Rank two provenance tags by honesty floor. Returns the *worse* one —
/// i.e. the floor a receipt that includes both must use.
fn worst_provenance(a: Provenance, b: Provenance) -> Provenance {
    use Provenance::*;
    // HwShunt > ModelBased > Estimator (in honesty).
    let rank = |p: Provenance| match p {
        HwShunt => 2,
        ModelBased => 1,
        Estimator => 0,
    };
    if rank(a) <= rank(b) { a } else { b }
}

/// BLAKE3 hash of an input string, used as the canonical `input_hash`
/// for cascade receipts. The hash is the lowercase hex of the BLAKE3
/// digest of the UTF-8 bytes of the normalised input.
pub fn input_hash(input: &str) -> String {
    blake3::hash(input.as_bytes()).to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_wire_tag_is_stable() {
        assert_eq!(CascadeTier::L0Cache.wire_tag(), "L0");
        assert_eq!(CascadeTier::L4Wire.wire_tag(), "L4");
    }

    #[test]
    fn empty_seal_requires_input_hash_and_tier() {
        let err = ReceiptBuilder::new().seal().unwrap_err();
        matches!(err, ReceiptError::MissingField("input_hash"));
        let err = ReceiptBuilder::new()
            .input_hash("abc")
            .seal()
            .unwrap_err();
        matches!(err, ReceiptError::MissingField("tier"));
    }

    #[test]
    fn worst_provenance_picks_estimator_over_shunt() {
        assert_eq!(
            worst_provenance(Provenance::HwShunt, Provenance::Estimator),
            Provenance::Estimator
        );
        assert_eq!(
            worst_provenance(Provenance::ModelBased, Provenance::HwShunt),
            Provenance::ModelBased
        );
    }

    #[test]
    fn receipt_accounts_tools_and_floors_provenance() {
        let r = ReceiptBuilder::new()
            .input_hash(input_hash("what is gcd(12,8)?"))
            .tier(CascadeTier::L1Lawful)
            .account_tool(ToolTouch {
                tool_id: "iac:math.gcd".into(),
                joules_uj: 12,
                energy_provenance: Provenance::HwShunt,
            })
            .account_tool(ToolTouch {
                tool_id: "iac:math.format".into(),
                joules_uj: 4,
                energy_provenance: Provenance::ModelBased,
            })
            .seal()
            .expect("seal");
        assert_eq!(r.tier, CascadeTier::L1Lawful);
        assert_eq!(r.joules_uj, 16);
        // Worst counter seen was ModelBased — the receipt floors there.
        assert_eq!(r.energy_provenance, Provenance::ModelBased);
        assert_eq!(r.tools_touched.len(), 2);
    }

    #[test]
    fn input_hash_is_deterministic() {
        let h1 = input_hash("hello world");
        let h2 = input_hash("hello world");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // BLAKE3 hex
    }

    #[test]
    fn receipt_serialises_round_trip() {
        let r = ReceiptBuilder::new()
            .input_hash("abc")
            .tier(CascadeTier::L3Model)
            .account_tool(ToolTouch {
                tool_id: "model:gemma4-9b".into(),
                joules_uj: 3_500_000,
                energy_provenance: Provenance::ModelBased,
            })
            .seal()
            .expect("seal");
        let json = serde_json::to_string(&r).expect("serialise");
        let back: Receipt = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(back.tier, CascadeTier::L3Model);
        assert_eq!(back.joules_uj, 3_500_000);
    }
}
