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
    /// Inline span-level citations linking output byte ranges to the
    /// retrieval chunks that grounded them. Empty for L0/L1 paths that
    /// were not retrieval-grounded; the field is `#[serde(default)]` so
    /// receipts produced before this field existed still deserialise.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub citations: Vec<Citation>,
    /// HMAC-signed tool-execution receipts minted at the MCP boundary.
    /// Each entry attests that the gateway — not the model — invoked the
    /// tool and observed the joule cost. Empty when the resolution
    /// touched no gateway-metered tools.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_receipts: Vec<ToolReceipt>,
    /// Per-token attributions, one row per non-trivial output token.
    /// Computed by a [`TokenAttributor`] the model backend supplies;
    /// this crate carries the data, not the algorithm. Empty when the
    /// resolution path is deterministic / non-model (L0–L2) or when
    /// the runtime did not run an attributor.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_attributions: Vec<TokenAttribution>,
}

// ─────────────────────────────────────────────────────────────────────
// Span-level citations
// ─────────────────────────────────────────────────────────────────────

/// A half-open byte range `[start, end)` over an output text. The
/// receipt does not embed the output itself — `start..end` is a slice
/// the consumer applies to the artifact they hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextSpan {
    pub start: u32,
    pub end: u32,
}

impl TextSpan {
    pub fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }
    pub fn len(&self) -> u32 {
        self.end.saturating_sub(self.start)
    }
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// One inline span-level citation. Links a byte range in the output
/// to the retrieval chunk that grounded it and the exact quoted text
/// that justifies the link. Three pieces are the load-bearing trio
/// the field has converged on: `(text_block, source_chunk_id,
/// exact_quote)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Citation {
    /// Byte range in the output artefact this citation grounds.
    pub text_span: TextSpan,
    /// Opaque chunk id from the retrieval store (URL+offset, vector-DB
    /// row id, content-address, …). Identifies which source supplied
    /// the grounding.
    pub source_chunk_id: String,
    /// The exact text quoted from the source chunk that justifies the
    /// span. Verbatim — auditors should be able to find this substring
    /// in the chunk's contents.
    pub exact_quote: String,
}

impl Citation {
    pub fn new(
        text_span: TextSpan,
        source_chunk_id: impl Into<String>,
        exact_quote: impl Into<String>,
    ) -> Self {
        Self {
            text_span,
            source_chunk_id: source_chunk_id.into(),
            exact_quote: exact_quote.into(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Per-token attribution (MIRAGE-class structural surface)
// ─────────────────────────────────────────────────────────────────────

/// One per-token attribution record. Says: for the output token at
/// `token_idx` (counting from 0 in the generated sequence), the model
/// drew on these retrieval chunks with these contribution scores.
///
/// Scores are unit-less and provider-defined — MIRAGE-class methods
/// emit normalised attention/gradient saliency over chunks; consumers
/// SHOULD treat them as ordinal, not absolute. The contract is "the
/// non-zero scores are the chunks that mattered for this token,"
/// not "score X means importance Y."
///
/// Honest scope: this crate ships only the **data type and trait**.
/// Computing the attribution requires access to model internals
/// (attention weights, gradient-based saliency, IRR-style internal
/// retrieval flags) and is therefore the model backend's concern,
/// not this crate's. The [`TokenAttributor`] trait is the seam where
/// a model-aware implementation plugs in; the absence of a reference
/// implementation in this open-standard repo is intentional —
/// shipping a hard-coded one would lie about which method was used.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenAttribution {
    /// Output-token index (0-based in the generated sequence).
    pub token_idx: u32,
    /// Per-chunk contribution scores. Keyed by retrieval chunk id
    /// (the same identifier carried in [`Citation::source_chunk_id`]
    /// so a span-level citation and a token-level attribution can be
    /// joined). Values are provider-defined; consumers SHOULD treat
    /// the *order* (highest-first) as load-bearing and the absolute
    /// numbers as opaque.
    pub chunk_scores: std::collections::BTreeMap<String, f64>,
    /// Optional opaque method tag (e.g. `"mirage:v1"`,
    /// `"attention-rollout"`, `"grad-x-input"`) so auditors know what
    /// produced this attribution. Different methods are NOT
    /// comparable across rows; consumers SHOULD filter by method
    /// before aggregating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
}

impl TokenAttribution {
    /// Construct an attribution for a single output token from an
    /// iterable of (chunk_id, score) pairs. Pairs with zero score are
    /// dropped — they carry no signal and bloat the wire form.
    pub fn new<I, S>(token_idx: u32, scores: I, method: Option<S>) -> Self
    where
        I: IntoIterator<Item = (String, f64)>,
        S: Into<String>,
    {
        let chunk_scores = scores
            .into_iter()
            .filter(|(_, v)| *v != 0.0)
            .collect();
        Self {
            token_idx,
            chunk_scores,
            method: method.map(|s| s.into()),
        }
    }

    /// The chunk id with the highest score, if any. Lexically-smaller
    /// key wins on a score tie (the same stable rule
    /// `CheapestCapable` uses in `jouleclaw-federation`).
    pub fn top_chunk(&self) -> Option<&str> {
        self.chunk_scores
            .iter()
            .max_by(|(ak, av), (bk, bv)| {
                // Compare scores first; on tie, the lexically-smaller
                // key should win — so we reverse the key comparison
                // (smaller key → "greater" in max_by's order).
                av.partial_cmp(bv)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(bk.cmp(ak))
            })
            .map(|(k, _)| k.as_str())
    }

    /// Total contribution mass — sum of all per-chunk scores.
    pub fn total_mass(&self) -> f64 {
        self.chunk_scores.values().sum()
    }
}

/// Computes per-token attributions for an output text given the
/// retrieval chunks the model could draw from. The contract is
/// intentionally narrow — implementations need access to model
/// internals, and JouleClaw is not in the business of dictating
/// which internals.
///
/// Consumers that want span-level citations + token-level attribution
/// on the same receipt run both an attributor (this trait) and a
/// citation builder (separately); the receipt carries both via
/// [`ReceiptBuilder::account_citation`] and a future
/// `account_token_attribution`.
pub trait TokenAttributor: Send + Sync {
    /// Compute attributions for `output_text` (token sequence the
    /// model generated) against `chunks` (the retrieval chunks
    /// available to it). Returns one [`TokenAttribution`] per
    /// generated output token, in order. May return fewer rows than
    /// tokens if the implementation only emits non-trivial rows
    /// (e.g. tokens that are deterministic / EOS get dropped).
    fn attribute(
        &self,
        output_text: &str,
        chunks: &[(String, String)],
    ) -> Vec<TokenAttribution>;
}

// ─────────────────────────────────────────────────────────────────────
// HMAC-signed tool-gateway receipts
// ─────────────────────────────────────────────────────────────────────

/// A tool-execution receipt minted by the MCP boundary (the *gateway*).
/// Sealed with HMAC-SHA-256 using a gateway-held secret, so a downstream
/// auditor can verify the tool was invoked through the gateway — even
/// when the artifact carrying the receipt is replayed in a different
/// context. The model never holds the HMAC key; the receipt's integrity
/// is the gateway's, not the model's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolReceipt {
    /// Stable identifier — e.g. `mcp:filesystem/read`. Matches the
    /// [`ToolTouch::tool_id`] when both are emitted for the same call.
    pub tool_id: String,
    /// Microjoules observed by the gateway for this call.
    pub joules_uj: u64,
    /// Honesty tier of the energy reading.
    pub energy_provenance: Provenance,
    /// RFC 3339 timestamp the gateway minted the receipt.
    pub minted_at: String,
    /// Opaque gateway identifier — which gateway minted this receipt.
    /// Verifiers use this to look up the corresponding HMAC key.
    pub gateway_id: String,
    /// Lowercase hex of the HMAC-SHA-256 over the canonical bytes
    /// (`tool_id || \0 || joules_uj || \0 || energy_provenance || \0 ||
    /// minted_at || \0 || gateway_id`). 64 hex chars.
    pub hmac_sha256_hex: String,
}

/// An MCP-boundary gateway that mints HMAC-signed [`ToolReceipt`]s. The
/// key never leaves the gateway; only the receipt does.
pub struct ToolGateway {
    gateway_id: String,
    key: Vec<u8>,
}

impl ToolGateway {
    pub fn new(gateway_id: impl Into<String>, key: impl Into<Vec<u8>>) -> Self {
        Self {
            gateway_id: gateway_id.into(),
            key: key.into(),
        }
    }

    pub fn gateway_id(&self) -> &str {
        &self.gateway_id
    }

    /// Mint a receipt for a tool call observed by this gateway.
    pub fn mint_receipt(
        &self,
        tool_id: impl Into<String>,
        joules_uj: u64,
        energy_provenance: Provenance,
        minted_at: impl Into<String>,
    ) -> ToolReceipt {
        let tool_id = tool_id.into();
        let minted_at = minted_at.into();
        let payload = canonical_tool_receipt_bytes(
            &tool_id,
            joules_uj,
            energy_provenance,
            &minted_at,
            &self.gateway_id,
        );
        let mac = hmac_sha256(&self.key, &payload);
        ToolReceipt {
            tool_id,
            joules_uj,
            energy_provenance,
            minted_at,
            gateway_id: self.gateway_id.clone(),
            hmac_sha256_hex: hex_encode(&mac),
        }
    }

    /// Verify that a receipt was minted by this gateway. Constant-time
    /// over the HMAC bytes via [`subtle_eq`]; any tampered field fails.
    pub fn verify_receipt(&self, r: &ToolReceipt) -> bool {
        if r.gateway_id != self.gateway_id {
            return false;
        }
        let payload = canonical_tool_receipt_bytes(
            &r.tool_id,
            r.joules_uj,
            r.energy_provenance,
            &r.minted_at,
            &r.gateway_id,
        );
        let Ok(expected_bytes) = hex_decode(&r.hmac_sha256_hex) else {
            return false;
        };
        let actual = hmac_sha256(&self.key, &payload);
        subtle_eq(&actual, &expected_bytes)
    }
}

/// Canonical byte encoding for the HMAC payload — stable, NUL-delimited,
/// matches the documented ordering on [`ToolReceipt::hmac_sha256_hex`].
fn canonical_tool_receipt_bytes(
    tool_id: &str,
    joules_uj: u64,
    energy_provenance: Provenance,
    minted_at: &str,
    gateway_id: &str,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        tool_id.len() + minted_at.len() + gateway_id.len() + 32,
    );
    out.extend_from_slice(tool_id.as_bytes());
    out.push(0);
    out.extend_from_slice(joules_uj.to_string().as_bytes());
    out.push(0);
    let prov_tag = match energy_provenance {
        Provenance::HwShunt => "HwShunt",
        Provenance::ModelBased => "ModelBased",
        Provenance::Estimator => "Estimator",
    };
    out.extend_from_slice(prov_tag.as_bytes());
    out.push(0);
    out.extend_from_slice(minted_at.as_bytes());
    out.push(0);
    out.extend_from_slice(gateway_id.as_bytes());
    out
}

/// Hand-rolled HMAC-SHA-256 over `sha2` so the crate doesn't pull in an
/// extra dependency for one call site. Standard RFC 2104 construction:
/// `H((K' XOR opad) || H((K' XOR ipad) || M))` with block size 64.
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    const BLOCK: usize = 64;
    // Step 1: normalise key to BLOCK bytes.
    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK {
        let hashed = Sha256::digest(key);
        k[..32].copy_from_slice(&hashed);
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    // Step 2: inner = H((K' XOR 0x36) || M).
    let mut ipad = [0u8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] = k[i] ^ 0x36;
    }
    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(message);
    let inner_digest = inner.finalize();
    // Step 3: outer = H((K' XOR 0x5c) || inner).
    let mut opad = [0u8; BLOCK];
    for i in 0..BLOCK {
        opad[i] = k[i] ^ 0x5c;
    }
    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner_digest);
    let out = outer.finalize();
    let mut result = [0u8; 32];
    result.copy_from_slice(&out);
    result
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

fn hex_decode(s: &str) -> Result<Vec<u8>, &'static str> {
    if s.len() % 2 != 0 {
        return Err("odd hex length");
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let val = |c: u8| -> Result<u8, &'static str> {
        match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            b'A'..=b'F' => Ok(c - b'A' + 10),
            _ => Err("bad hex char"),
        }
    };
    for chunk in bytes.chunks(2) {
        out.push((val(chunk[0])? << 4) | val(chunk[1])?);
    }
    Ok(out)
}

/// Constant-time byte comparison so HMAC verification does not leak the
/// expected bytes through timing.
fn subtle_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
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
    citations: Vec<Citation>,
    tool_receipts: Vec<ToolReceipt>,
    token_attributions: Vec<TokenAttribution>,
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

    /// Attach an inline span-level [`Citation`] grounding a slice of the
    /// output to a retrieval chunk + exact quote.
    pub fn account_citation(mut self, citation: Citation) -> Self {
        self.citations.push(citation);
        self
    }

    /// Attach a gateway-minted, HMAC-signed [`ToolReceipt`] for a tool
    /// call observed at the MCP boundary. Adds the call's joules into
    /// the running total and lowers the receipt's overall provenance
    /// floor to the worst counter seen.
    pub fn account_tool_receipt(mut self, receipt: ToolReceipt) -> Self {
        self.joules_uj = self.joules_uj.saturating_add(receipt.joules_uj);
        let worst = match (self.energy_provenance, receipt.energy_provenance) {
            (None, p) => Some(p),
            (Some(a), b) => Some(worst_provenance(a, b)),
        };
        self.energy_provenance = worst;
        self.tool_receipts.push(receipt);
        self
    }

    /// Attach a per-token [`TokenAttribution`] row. Order matters —
    /// the seal preserves call order so `token_idx` sequencing is
    /// auditable. The attribution itself carries no joule cost (the
    /// attributor's spend is the model backend's accounting).
    pub fn account_token_attribution(mut self, attr: TokenAttribution) -> Self {
        self.token_attributions.push(attr);
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
            citations: self.citations,
            tool_receipts: self.tool_receipts,
            token_attributions: self.token_attributions,
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

    // ─── Citation + ToolReceipt ────────────────────────────────────

    #[test]
    fn text_span_basic() {
        let s = TextSpan::new(0, 10);
        assert_eq!(s.len(), 10);
        assert!(!s.is_empty());
        let z = TextSpan::new(5, 5);
        assert!(z.is_empty());
    }

    #[test]
    fn citation_round_trips_through_json() {
        let c = Citation::new(
            TextSpan::new(12, 27),
            "chunk:wikipedia/Paris#offset=345",
            "Paris is the capital of France",
        );
        let j = serde_json::to_string(&c).expect("ser");
        let back: Citation = serde_json::from_str(&j).expect("deser");
        assert_eq!(back, c);
    }

    #[test]
    fn receipt_with_citations_round_trips() {
        let r = ReceiptBuilder::new()
            .input_hash("abc")
            .tier(CascadeTier::L2Embed)
            .account_citation(Citation::new(
                TextSpan::new(0, 5),
                "chunk:1",
                "hello",
            ))
            .seal()
            .expect("seal");
        let j = serde_json::to_string(&r).expect("ser");
        let back: Receipt = serde_json::from_str(&j).expect("deser");
        assert_eq!(back.citations.len(), 1);
        assert_eq!(back.citations[0].source_chunk_id, "chunk:1");
    }

    #[test]
    fn receipt_without_citations_does_not_emit_field() {
        // skip_serializing_if = "Vec::is_empty" means the JSON omits the
        // citations array when none are attached — older readers see the
        // same shape as before this field existed.
        let r = ReceiptBuilder::new()
            .input_hash("abc")
            .tier(CascadeTier::L0Cache)
            .seal()
            .expect("seal");
        let j = serde_json::to_string(&r).expect("ser");
        assert!(!j.contains("\"citations\""), "got: {j}");
        assert!(!j.contains("\"tool_receipts\""), "got: {j}");
    }

    #[test]
    fn old_receipt_without_new_fields_deserialises() {
        // A receipt JSON written before the new fields existed — it
        // should still deserialise (the new fields default to empty).
        let old = r#"{
            "jc_receipt": "1",
            "id": "deadbeef",
            "closed_at": "2026-05-30T00:00:00Z",
            "input_hash": "abc",
            "tier": "l0-cache",
            "joules_uj": 0,
            "energy_provenance": "Estimator",
            "tools_touched": [],
            "claims": [],
            "eoc_stage": null
        }"#;
        let r: Receipt = serde_json::from_str(old).expect("deser old shape");
        assert!(r.citations.is_empty());
        assert!(r.tool_receipts.is_empty());
    }

    #[test]
    fn hmac_sha256_matches_known_test_vector() {
        // RFC 4231 test case 1: key = 0x0b × 20, data = "Hi There".
        let key = [0x0bu8; 20];
        let mac = hmac_sha256(&key, b"Hi There");
        let expected = "b0344c61d8db38535ca8afceaf0bf12b\
                        881dc200c9833da726e9376c2e32cff7";
        assert_eq!(hex_encode(&mac), expected);
    }

    #[test]
    fn hmac_handles_oversized_key_by_hashing_it() {
        // Per RFC 4231 case 4: key longer than 64 bytes must be hashed
        // first. Just check that an oversized key produces a 32-byte tag
        // (no panic, no wrap-around).
        let key = vec![0x77; 200];
        let mac = hmac_sha256(&key, b"some message");
        assert_eq!(mac.len(), 32);
    }

    #[test]
    fn tool_gateway_mints_and_verifies_receipt() {
        let gw = ToolGateway::new("mcp:gateway-a", b"sekret-key-32-bytes-or-more-here!".to_vec());
        let r = gw.mint_receipt(
            "mcp:filesystem/read",
            120,
            Provenance::HwShunt,
            "2026-05-30T12:34:56Z",
        );
        assert_eq!(r.tool_id, "mcp:filesystem/read");
        assert_eq!(r.joules_uj, 120);
        assert_eq!(r.gateway_id, "mcp:gateway-a");
        assert_eq!(r.hmac_sha256_hex.len(), 64);
        assert!(gw.verify_receipt(&r));
    }

    #[test]
    fn tool_receipt_tampering_breaks_verification() {
        let gw = ToolGateway::new("gw", b"key-bytes".to_vec());
        let r = gw.mint_receipt("t", 100, Provenance::ModelBased, "now");

        // Tampering with joules_uj invalidates the HMAC.
        let mut t1 = r.clone();
        t1.joules_uj = 1;
        assert!(!gw.verify_receipt(&t1));

        // Tampering with tool_id invalidates.
        let mut t2 = r.clone();
        t2.tool_id = "other".into();
        assert!(!gw.verify_receipt(&t2));

        // Wrong gateway key rejects.
        let other = ToolGateway::new("gw", b"different-key".to_vec());
        assert!(!other.verify_receipt(&r));

        // Wrong gateway_id on the receipt rejects.
        let mut t3 = r.clone();
        t3.gateway_id = "gw-spoofed".into();
        assert!(!gw.verify_receipt(&t3));
    }

    #[test]
    fn account_tool_receipt_accumulates_joules_and_floor() {
        let gw = ToolGateway::new("gw", b"k".to_vec());
        let r1 = gw.mint_receipt("a", 100, Provenance::HwShunt, "t1");
        let r2 = gw.mint_receipt("b", 200, Provenance::Estimator, "t2");
        let receipt = ReceiptBuilder::new()
            .input_hash("abc")
            .tier(CascadeTier::L2Embed)
            .account_tool_receipt(r1)
            .account_tool_receipt(r2)
            .seal()
            .expect("seal");
        assert_eq!(receipt.joules_uj, 300);
        // Worst counter wins: Estimator < ModelBased < HwShunt by honesty.
        assert_eq!(receipt.energy_provenance, Provenance::Estimator);
        assert_eq!(receipt.tool_receipts.len(), 2);
    }

    // ─── TokenAttribution ────────────────────────────────────────────

    #[test]
    fn token_attribution_drops_zero_scores() {
        let a = TokenAttribution::new(
            0,
            vec![
                ("chunk:a".to_string(), 0.0),
                ("chunk:b".to_string(), 0.7),
                ("chunk:c".to_string(), 0.0),
            ],
            Some("mirage:v1"),
        );
        assert_eq!(a.chunk_scores.len(), 1);
        assert!(a.chunk_scores.contains_key("chunk:b"));
    }

    #[test]
    fn token_attribution_top_chunk_picks_highest_with_lexical_tiebreak() {
        let a = TokenAttribution::new(
            5,
            vec![
                ("chunk:b".to_string(), 0.5),
                ("chunk:a".to_string(), 0.5),
                ("chunk:c".to_string(), 0.3),
            ],
            None::<String>,
        );
        // Tie at 0.5; "chunk:a" < "chunk:b" lexically.
        assert_eq!(a.top_chunk(), Some("chunk:a"));
        assert!((a.total_mass() - 1.3).abs() < 1e-9);
    }

    #[test]
    fn token_attribution_round_trips_through_json() {
        let a = TokenAttribution::new(
            3,
            vec![("c1".to_string(), 0.6), ("c2".to_string(), 0.4)],
            Some("attention-rollout"),
        );
        let j = serde_json::to_value(&a).unwrap();
        assert_eq!(j["token_idx"], 3);
        assert_eq!(j["method"], "attention-rollout");
        let back: TokenAttribution = serde_json::from_value(j).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn token_attribution_method_is_optional_on_wire() {
        let a = TokenAttribution::new(0, vec![("c1".to_string(), 1.0)], None::<String>);
        let j = serde_json::to_value(&a).unwrap();
        // Optional method should be absent when None.
        assert!(j.get("method").is_none(), "got: {j:?}");
    }

    #[test]
    fn receipt_carries_token_attributions_through_seal_and_wire() {
        let attr0 = TokenAttribution::new(
            0,
            vec![("chunk:doc-1".to_string(), 0.9)],
            Some("mirage:v1"),
        );
        let attr1 = TokenAttribution::new(
            1,
            vec![("chunk:doc-2".to_string(), 0.55)],
            Some("mirage:v1"),
        );
        let receipt = ReceiptBuilder::new()
            .input_hash("a".repeat(64))
            .tier(CascadeTier::L3Model)
            .account_token_attribution(attr0.clone())
            .account_token_attribution(attr1.clone())
            .seal()
            .expect("seal");
        assert_eq!(receipt.token_attributions.len(), 2);
        assert_eq!(receipt.token_attributions[0], attr0);
        let j = serde_json::to_value(&receipt).unwrap();
        assert!(j.get("token_attributions").is_some());
        let back: Receipt = serde_json::from_value(j).unwrap();
        assert_eq!(back.token_attributions, receipt.token_attributions);
    }

    #[test]
    fn receipt_omits_empty_token_attributions_on_wire() {
        let receipt = ReceiptBuilder::new()
            .input_hash("a".repeat(64))
            .tier(CascadeTier::L0Cache)
            .seal()
            .expect("seal");
        let j = serde_json::to_string(&receipt).unwrap();
        assert!(
            !j.contains("token_attributions"),
            "empty list must be skipped on wire to keep L0/L1 receipts terse: {j}"
        );
    }

    /// Test attributor that scores chunks by simple substring overlap
    /// with the output text. Exercises the TokenAttributor trait via a
    /// deterministic, no-model implementation — useful for tests but
    /// NOT a real attribution method.
    struct OverlapAttributor;
    impl TokenAttributor for OverlapAttributor {
        fn attribute(
            &self,
            output_text: &str,
            chunks: &[(String, String)],
        ) -> Vec<TokenAttribution> {
            let words: Vec<&str> = output_text.split_whitespace().collect();
            let mut out = Vec::new();
            for (i, w) in words.iter().enumerate() {
                let mut scores = Vec::new();
                for (id, text) in chunks {
                    if text.contains(w) {
                        scores.push((id.clone(), 1.0));
                    }
                }
                if !scores.is_empty() {
                    out.push(TokenAttribution::new(
                        i as u32,
                        scores,
                        Some("test:overlap"),
                    ));
                }
            }
            out
        }
    }

    #[test]
    fn token_attributor_trait_is_implementable() {
        let chunks = vec![
            ("chunk:1".to_string(), "the sky is blue".to_string()),
            ("chunk:2".to_string(), "grass is green".to_string()),
        ];
        let attrs = OverlapAttributor.attribute("sky is blue", &chunks);
        // Three tokens: "sky", "is", "blue". "sky" + "blue" only in
        // chunk:1; "is" in both.
        assert_eq!(attrs.len(), 3);
        assert_eq!(attrs[0].top_chunk(), Some("chunk:1"));
        assert_eq!(attrs[1].chunk_scores.len(), 2);
        assert_eq!(attrs[2].top_chunk(), Some("chunk:1"));
    }
}
