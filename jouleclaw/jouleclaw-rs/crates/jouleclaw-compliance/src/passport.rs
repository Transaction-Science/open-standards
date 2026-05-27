//! AI Service Passport — a machine-readable energy + carbon + compliance
//! record for an AI inference endpoint.
//!
//! Passports are designed to be signed (or in this lightweight default,
//! BLAKE3-hashed) so a downstream consumer can verify the bundle has
//! not been tampered with after publication. They bundle:
//!
//! 1. **Model identity** (parameters, quantization, version)
//! 2. **Hardware description** (accelerator, CPU, interconnect)
//! 3. **Energy profile** (joules per input/output token, measurement
//!    method) — measurement method is mapped from
//!    [`jouleclaw_energy::Provenance`]
//! 4. **Carbon profile** (SCI score, grid intensity, renewable share)
//! 5. **Compliance record** (EU AI Act, CSRD Scope 3, EED, ESPR DPP)
//! 6. **Pricing** — optional, lightweight
//!
//! ## Sealing
//!
//! Use [`AiServicePassport::sealed_hash`] to derive a BLAKE3 hash of
//! the JSON-canonicalised passport bytes. For full cryptographic
//! signing, pair this with a `jouleclaw-prov` `Signer`.

#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::fmt;

use chrono::{DateTime, Utc};
use jouleclaw_energy::Provenance;
use serde::{Deserialize, Serialize};

// ── EnergyLabel (self-contained) ───────────────────────────────────

/// EU-style energy efficiency rating, A (best) through G (worst).
///
/// Ordering: `EnergyLabel::A < EnergyLabel::G` (A is *less than* G —
/// "lower is better"). This matches the donor's semantic so existing
/// catalog filtering by "rating at most C" works.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnergyLabel {
    /// Most efficient.
    A,
    /// Very efficient.
    B,
    /// Efficient.
    C,
    /// Average.
    D,
    /// Below average.
    E,
    /// Inefficient.
    F,
    /// Least efficient.
    G,
}

impl EnergyLabel {
    /// One-word description of the rating tier.
    pub fn description(&self) -> &'static str {
        match self {
            Self::A => "Most efficient",
            Self::B => "Very efficient",
            Self::C => "Efficient",
            Self::D => "Average",
            Self::E => "Below average",
            Self::F => "Inefficient",
            Self::G => "Least efficient",
        }
    }

    fn rank(&self) -> u8 {
        match self {
            Self::A => 0,
            Self::B => 1,
            Self::C => 2,
            Self::D => 3,
            Self::E => 4,
            Self::F => 5,
            Self::G => 6,
        }
    }
}

impl fmt::Display for EnergyLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::A => f.write_str("A"),
            Self::B => f.write_str("B"),
            Self::C => f.write_str("C"),
            Self::D => f.write_str("D"),
            Self::E => f.write_str("E"),
            Self::F => f.write_str("F"),
            Self::G => f.write_str("G"),
        }
    }
}

impl PartialOrd for EnergyLabel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EnergyLabel {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank().cmp(&other.rank())
    }
}

// ── MeasurementMethod ───────────────────────────────────────────────

/// How the inference-energy profile was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeasurementMethod {
    /// Intel / AMD RAPL counters.
    Rapl,
    /// Apple Silicon IOReport (or estimator until FFI ships).
    IoReport,
    /// NVIDIA Management Library.
    Nvml,
    /// Combined GPU (NVML) + CPU (RAPL) measurement.
    NvmlRapl,
    /// Derived from billing data or hardware specs — not measured.
    Estimated,
}

impl MeasurementMethod {
    /// Honest provenance tag for this measurement method.
    /// Pure-Rust translation from the JouleClaw energy contract.
    pub fn provenance(&self) -> Provenance {
        match self {
            Self::Rapl | Self::Nvml | Self::NvmlRapl => Provenance::HwShunt,
            Self::IoReport => Provenance::Estimator,
            Self::Estimated => Provenance::Estimator,
        }
    }
}

impl fmt::Display for MeasurementMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rapl => write!(f, "RAPL"),
            Self::IoReport => write!(f, "IOReport"),
            Self::Nvml => write!(f, "NVML"),
            Self::NvmlRapl => write!(f, "NVML+RAPL"),
            Self::Estimated => write!(f, "Estimated"),
        }
    }
}

// ── Sub-structs ─────────────────────────────────────────────────────

/// Model identity / configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    /// Model identifier.
    pub id: String,
    /// Model family.
    pub family: String,
    /// Parameter count.
    pub parameters: u64,
    /// Quantization (`"FP8"`, `"FP16"`, `"INT4"`, ...).
    pub quantization: String,
    /// Version tag.
    pub version: String,
}

/// Hardware running the inference service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// Accelerator model.
    pub accelerator: String,
    /// CPU model.
    pub cpu: String,
    /// Memory in GiB.
    pub memory_gb: u32,
    /// Interconnect (`"NVLink 4.0"`, `"PCIe 5.0"`, ...).
    pub interconnect: String,
}

/// Measured energy characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyProfile {
    /// Energy per input token (joules).
    pub joules_per_input_token: f64,
    /// Energy per output token (joules).
    pub joules_per_output_token: f64,
    /// How the measurement was obtained.
    pub measurement_method: MeasurementMethod,
    /// Number of inference requests in the sample.
    pub sample_size: u64,
    /// Benchmark date.
    pub benchmark_date: DateTime<Utc>,
    /// Idle power draw (watts).
    pub idle_power_watts: f64,
    /// Peak observed power draw (watts).
    pub peak_power_watts: f64,
}

/// Geographic location of the inference service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationInfo {
    /// Cloud region or data-centre identifier.
    pub region: String,
    /// ISO 3166-1 alpha-2 country code.
    pub country: String,
    /// Electricity grid zone identifier.
    pub grid_zone: String,
}

/// Carbon profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonProfile {
    /// SCI score: gCO2 per request.
    pub sci_score_gco2_per_request: f64,
    /// Grid carbon intensity (gCO2 / kWh).
    pub grid_carbon_intensity_gco2_kwh: f64,
    /// Source of the carbon-intensity data.
    pub carbon_intensity_source: String,
    /// Renewable share (0–100).
    pub renewable_percentage: f64,
    /// Amortised embodied carbon per request (gCO2).
    pub embodied_carbon_gco2_per_request: f64,
    /// Location.
    pub location: LocationInfo,
}

/// A-G label record with methodology + baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassportLabel {
    /// A-G rating.
    pub rating: EnergyLabel,
    /// Labelling methodology description.
    pub methodology: String,
    /// Model used as the efficiency baseline.
    pub baseline_model: String,
    /// Ratio of measured energy to baseline (lower is better).
    pub efficiency_ratio: f64,
}

/// EU AI Act disclosure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiActCompliance {
    /// Training energy publicly disclosed.
    pub training_energy_disclosed: bool,
    /// Total training energy (kWh), if known.
    pub training_energy_kwh: Option<f64>,
    /// Total training emissions (tonnes CO2), if known.
    pub training_emissions_tco2: Option<f64>,
    /// Inference energy is measured, not estimated.
    pub inference_energy_measured: bool,
}

/// CSRD Scope 3 status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrdCompliance {
    /// Independently verified.
    pub verified: bool,
    /// Methodology.
    pub methodology: String,
    /// Audit-trail reference.
    pub audit_trail: String,
}

/// EED data-centre metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EedCompliance {
    /// Power Usage Effectiveness.
    pub facility_pue: f64,
    /// Water Usage Effectiveness.
    pub facility_wue: f64,
    /// Waste-heat recovery share (0–100).
    pub waste_heat_recovery_pct: f64,
    /// Renewable-energy factor (0.0–1.0).
    pub renewable_energy_factor: f64,
}

/// ESPR Digital Product Passport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EsprCompliance {
    /// URL of the hosted DPP.
    pub passport_url: String,
    /// Submitted to the EU registry.
    pub registry_submitted: bool,
}

/// Aggregate compliance record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRecord {
    /// EU AI Act compliance.
    pub eu_ai_act: AiActCompliance,
    /// CSRD Scope 3 compliance.
    pub csrd_scope_3: CsrdCompliance,
    /// EED compliance.
    pub eed: EedCompliance,
    /// ESPR DPP compliance.
    pub espr_dpp: EsprCompliance,
}

/// Pricing + settlement information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingInfo {
    /// Price per input token (USDC).
    pub usdc_per_input_token: f64,
    /// Price per output token (USDC).
    pub usdc_per_output_token: f64,
    /// Price per joule (USDC).
    pub usdc_per_joule: f64,
    /// CAIP-2 settlement chain identifier.
    pub settlement_network: String,
    /// HTTP endpoint for x402 payment negotiation.
    pub x402_endpoint: String,
    /// Free-tier daily energy budget (joules).
    pub free_tier_joules_per_day: f64,
}

impl Default for PricingInfo {
    fn default() -> Self {
        Self {
            usdc_per_input_token: 0.0,
            usdc_per_output_token: 0.0,
            usdc_per_joule: 0.0,
            settlement_network: "eip155:8453".to_string(),
            x402_endpoint: String::new(),
            free_tier_joules_per_day: 0.0,
        }
    }
}

// ── AiServicePassport ───────────────────────────────────────────────

/// Errors building a passport.
#[derive(Debug, thiserror::Error)]
pub enum PassportError {
    /// A required field was missing.
    #[error("required field missing: {0}")]
    MissingField(&'static str),
    /// JSON encoding failed.
    #[error("json encoding failed: {0}")]
    JsonEncode(#[from] serde_json::Error),
}

/// Complete AI Service Passport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiServicePassport {
    /// Passport schema version (e.g. `"1.0"`).
    pub passport_version: String,
    /// Unique service identifier.
    pub service_id: String,
    /// Infrastructure provider.
    pub provider: String,
    /// Issuance timestamp.
    pub issued_at: DateTime<Utc>,
    /// Expiry timestamp.
    pub valid_until: DateTime<Utc>,
    /// Served model.
    pub model: ModelProfile,
    /// Hardware running the service.
    pub hardware: HardwareProfile,
    /// Energy measurements.
    pub energy_profile: EnergyProfile,
    /// Carbon footprint.
    pub carbon_profile: CarbonProfile,
    /// EU-style energy label.
    pub energy_label: PassportLabel,
    /// Regulatory compliance status.
    pub compliance: ComplianceRecord,
    /// Pricing and settlement.
    pub pricing: PricingInfo,
}

impl AiServicePassport {
    /// Start a builder for a passport with the given service id.
    pub fn builder(service_id: &str) -> PassportBuilder {
        PassportBuilder {
            service_id: service_id.to_string(),
            passport_version: "1.0".to_string(),
            provider: String::new(),
            issued_at: None,
            valid_until: None,
            model: None,
            hardware: None,
            energy_profile: None,
            carbon_profile: None,
            energy_label: None,
            compliance: None,
            pricing: None,
        }
    }

    /// BLAKE3 hash over the canonical JSON encoding of this passport.
    /// Suitable as a lightweight tamper-evidence seal; for full
    /// signing use a `jouleclaw-prov` `Signer`.
    pub fn sealed_hash(&self) -> Result<[u8; 32], PassportError> {
        let bytes = serde_json::to_vec(self)?;
        let h = blake3::hash(&bytes);
        Ok(*h.as_bytes())
    }

    /// Hex-encoded BLAKE3 hash convenience accessor.
    pub fn sealed_hash_hex(&self) -> Result<String, PassportError> {
        let h = self.sealed_hash()?;
        let mut s = String::with_capacity(64);
        for byte in h.iter() {
            s.push_str(&format!("{:02x}", byte));
        }
        Ok(s)
    }
}

// ── Builder ─────────────────────────────────────────────────────────

/// Incremental builder for [`AiServicePassport`].
#[derive(Debug)]
pub struct PassportBuilder {
    service_id: String,
    passport_version: String,
    provider: String,
    issued_at: Option<DateTime<Utc>>,
    valid_until: Option<DateTime<Utc>>,
    model: Option<ModelProfile>,
    hardware: Option<HardwareProfile>,
    energy_profile: Option<EnergyProfile>,
    carbon_profile: Option<CarbonProfile>,
    energy_label: Option<PassportLabel>,
    compliance: Option<ComplianceRecord>,
    pricing: Option<PricingInfo>,
}

impl PassportBuilder {
    /// Set the passport schema version.
    pub fn passport_version(mut self, v: &str) -> Self {
        self.passport_version = v.to_string();
        self
    }
    /// Set the provider.
    pub fn provider(mut self, p: &str) -> Self {
        self.provider = p.to_string();
        self
    }
    /// Set the issuance timestamp.
    pub fn issued_at(mut self, t: DateTime<Utc>) -> Self {
        self.issued_at = Some(t);
        self
    }
    /// Set the expiry.
    pub fn valid_until(mut self, t: DateTime<Utc>) -> Self {
        self.valid_until = Some(t);
        self
    }
    /// Set the model profile.
    pub fn model(mut self, m: ModelProfile) -> Self {
        self.model = Some(m);
        self
    }
    /// Set the hardware profile.
    pub fn hardware(mut self, h: HardwareProfile) -> Self {
        self.hardware = Some(h);
        self
    }
    /// Set the energy profile.
    pub fn energy_profile(mut self, e: EnergyProfile) -> Self {
        self.energy_profile = Some(e);
        self
    }
    /// Set the carbon profile.
    pub fn carbon_profile(mut self, c: CarbonProfile) -> Self {
        self.carbon_profile = Some(c);
        self
    }
    /// Set the energy label.
    pub fn energy_label(mut self, l: PassportLabel) -> Self {
        self.energy_label = Some(l);
        self
    }
    /// Set the compliance record.
    pub fn compliance(mut self, c: ComplianceRecord) -> Self {
        self.compliance = Some(c);
        self
    }
    /// Set pricing.
    pub fn pricing(mut self, p: PricingInfo) -> Self {
        self.pricing = Some(p);
        self
    }

    /// Consume the builder and return a passport. Errors on missing
    /// required fields.
    pub fn build(self) -> Result<AiServicePassport, PassportError> {
        Ok(AiServicePassport {
            passport_version: self.passport_version,
            service_id: self.service_id,
            provider: self.provider,
            issued_at: self.issued_at.ok_or(PassportError::MissingField("issued_at"))?,
            valid_until: self
                .valid_until
                .ok_or(PassportError::MissingField("valid_until"))?,
            model: self.model.ok_or(PassportError::MissingField("model"))?,
            hardware: self
                .hardware
                .ok_or(PassportError::MissingField("hardware"))?,
            energy_profile: self
                .energy_profile
                .ok_or(PassportError::MissingField("energy_profile"))?,
            carbon_profile: self
                .carbon_profile
                .ok_or(PassportError::MissingField("carbon_profile"))?,
            energy_label: self
                .energy_label
                .ok_or(PassportError::MissingField("energy_label"))?,
            compliance: self
                .compliance
                .ok_or(PassportError::MissingField("compliance"))?,
            pricing: self.pricing.unwrap_or_default(),
        })
    }
}

// ── Catalog ─────────────────────────────────────────────────────────

/// Catalog of available service passports for discovery and comparison.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PassportCatalog {
    /// All registered passports.
    pub passports: Vec<AiServicePassport>,
}

impl PassportCatalog {
    /// Empty catalog.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a passport.
    pub fn add(&mut self, p: AiServicePassport) {
        self.passports.push(p);
    }

    /// Find a passport by exact service id.
    pub fn find_by_service_id(&self, id: &str) -> Option<&AiServicePassport> {
        self.passports.iter().find(|p| p.service_id == id)
    }

    /// All passports serving a given model id.
    pub fn find_by_model(&self, model_id: &str) -> Vec<&AiServicePassport> {
        self.passports.iter().filter(|p| p.model.id == model_id).collect()
    }

    /// All passports with rating ≤ `min_rating` (A is best). E.g.
    /// `find_by_rating(EnergyLabel::C)` returns A, B, and C.
    pub fn find_by_rating(&self, min_rating: EnergyLabel) -> Vec<&AiServicePassport> {
        self.passports
            .iter()
            .filter(|p| p.energy_label.rating <= min_rating)
            .collect()
    }

    /// Passports sorted by `efficiency_ratio` ascending (best first).
    pub fn leaderboard(&self) -> Vec<&AiServicePassport> {
        let mut sorted: Vec<&AiServicePassport> = self.passports.iter().collect();
        sorted.sort_by(|a, b| {
            a.energy_label
                .efficiency_ratio
                .partial_cmp(&b.energy_label.efficiency_ratio)
                .unwrap_or(Ordering::Equal)
        });
        sorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn sample_model() -> ModelProfile {
        ModelProfile {
            id: "meta-llama/Llama-3.3-70B-Instruct".to_string(),
            family: "Llama 3.3".to_string(),
            parameters: 70_000_000_000,
            quantization: "FP8".to_string(),
            version: "1.0".to_string(),
        }
    }

    fn sample_hardware() -> HardwareProfile {
        HardwareProfile {
            accelerator: "NVIDIA H100 SXM".to_string(),
            cpu: "AMD EPYC 9654".to_string(),
            memory_gb: 80,
            interconnect: "NVLink 4.0".to_string(),
        }
    }

    fn sample_energy_profile() -> EnergyProfile {
        EnergyProfile {
            joules_per_input_token: 0.001,
            joules_per_output_token: 0.003,
            measurement_method: MeasurementMethod::NvmlRapl,
            sample_size: 10_000,
            benchmark_date: Utc::now(),
            idle_power_watts: 75.0,
            peak_power_watts: 700.0,
        }
    }

    fn sample_carbon_profile() -> CarbonProfile {
        CarbonProfile {
            sci_score_gco2_per_request: 0.42,
            grid_carbon_intensity_gco2_kwh: 45.0,
            carbon_intensity_source: "electricitymaps.com".to_string(),
            renewable_percentage: 92.0,
            embodied_carbon_gco2_per_request: 0.005,
            location: LocationInfo {
                region: "eu-west-1".to_string(),
                country: "IE".to_string(),
                grid_zone: "IE-SEM".to_string(),
            },
        }
    }

    fn sample_label(rating: EnergyLabel, ratio: f64) -> PassportLabel {
        PassportLabel {
            rating,
            methodology: "jouleclaw-compliance v0.1".to_string(),
            baseline_model: "baseline".to_string(),
            efficiency_ratio: ratio,
        }
    }

    fn sample_compliance() -> ComplianceRecord {
        ComplianceRecord {
            eu_ai_act: AiActCompliance {
                training_energy_disclosed: true,
                training_energy_kwh: Some(6_500_000.0),
                training_emissions_tco2: Some(2_290.0),
                inference_energy_measured: true,
            },
            csrd_scope_3: CsrdCompliance {
                verified: true,
                methodology: "GHG Protocol".to_string(),
                audit_trail: "https://example.com/audit/2025-Q4".to_string(),
            },
            eed: EedCompliance {
                facility_pue: 1.08,
                facility_wue: 0.3,
                waste_heat_recovery_pct: 45.0,
                renewable_energy_factor: 0.92,
            },
            espr_dpp: EsprCompliance {
                passport_url: "https://example.com/dpp/llama".to_string(),
                registry_submitted: true,
            },
        }
    }

    fn sample_pricing() -> PricingInfo {
        PricingInfo {
            usdc_per_input_token: 0.0000015,
            usdc_per_output_token: 0.000006,
            usdc_per_joule: 0.0001,
            settlement_network: "eip155:8453".to_string(),
            x402_endpoint: "https://example.com/x402".to_string(),
            free_tier_joules_per_day: 1000.0,
        }
    }

    fn build_sample(id: &str, ratio: f64, rating: EnergyLabel) -> AiServicePassport {
        let now = Utc::now();
        AiServicePassport::builder(id)
            .provider("jouleclaw.dev")
            .issued_at(now)
            .valid_until(now + Duration::days(90))
            .model(sample_model())
            .hardware(sample_hardware())
            .energy_profile(sample_energy_profile())
            .carbon_profile(sample_carbon_profile())
            .energy_label(sample_label(rating, ratio))
            .compliance(sample_compliance())
            .pricing(sample_pricing())
            .build()
            .expect("build")
    }

    #[test]
    fn energy_label_ordering_a_lt_g() {
        assert!(EnergyLabel::A < EnergyLabel::G);
        assert!(EnergyLabel::A < EnergyLabel::B);
        assert_eq!(EnergyLabel::B.description(), "Very efficient");
        assert_eq!(format!("{}", EnergyLabel::A), "A");
    }

    #[test]
    fn measurement_method_maps_to_provenance() {
        assert_eq!(MeasurementMethod::Rapl.provenance(), Provenance::HwShunt);
        assert_eq!(MeasurementMethod::Nvml.provenance(), Provenance::HwShunt);
        assert_eq!(
            MeasurementMethod::NvmlRapl.provenance(),
            Provenance::HwShunt
        );
        // IOReport is Estimator until the FFI is wired — see apple.rs.
        assert_eq!(
            MeasurementMethod::IoReport.provenance(),
            Provenance::Estimator
        );
        assert_eq!(
            MeasurementMethod::Estimated.provenance(),
            Provenance::Estimator
        );
    }

    #[test]
    fn build_passport_succeeds_with_all_fields() {
        let p = build_sample("svc-1", 0.35, EnergyLabel::A);
        assert_eq!(p.service_id, "svc-1");
        assert_eq!(p.energy_label.rating, EnergyLabel::A);
        assert_eq!(p.model.parameters, 70_000_000_000);
    }

    #[test]
    fn build_passport_errors_without_required_field() {
        let now = Utc::now();
        let r = AiServicePassport::builder("svc-missing")
            .provider("jouleclaw.dev")
            .issued_at(now)
            .valid_until(now + Duration::days(30))
            .model(sample_model())
            .build();
        assert!(matches!(r, Err(PassportError::MissingField(_))));
    }

    #[test]
    fn passport_round_trip() {
        let p = build_sample("svc-rt", 0.35, EnergyLabel::A);
        let json = serde_json::to_string(&p).expect("serialize");
        let d: AiServicePassport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(d.service_id, "svc-rt");
        assert_eq!(d.energy_label.rating, EnergyLabel::A);
    }

    #[test]
    fn sealed_hash_changes_with_payload() {
        let p1 = build_sample("svc-a", 0.35, EnergyLabel::A);
        let mut p2 = p1.clone();
        p2.service_id = "svc-b".to_string();
        let h1 = p1.sealed_hash().expect("h1");
        let h2 = p2.sealed_hash().expect("h2");
        assert_ne!(h1, h2);
        let hex1 = p1.sealed_hash_hex().expect("hex");
        assert_eq!(hex1.len(), 64);
    }

    #[test]
    fn measurement_method_display() {
        assert_eq!(format!("{}", MeasurementMethod::Rapl), "RAPL");
        assert_eq!(format!("{}", MeasurementMethod::IoReport), "IOReport");
        assert_eq!(format!("{}", MeasurementMethod::Nvml), "NVML");
        assert_eq!(format!("{}", MeasurementMethod::NvmlRapl), "NVML+RAPL");
        assert_eq!(format!("{}", MeasurementMethod::Estimated), "Estimated");
    }

    #[test]
    fn catalog_find_by_service_id() {
        let mut cat = PassportCatalog::new();
        cat.add(build_sample("svc-alpha", 0.30, EnergyLabel::A));
        cat.add(build_sample("svc-beta", 0.60, EnergyLabel::B));
        assert!(cat.find_by_service_id("svc-alpha").is_some());
        assert!(cat.find_by_service_id("missing").is_none());
    }

    #[test]
    fn catalog_find_by_model() {
        let mut cat = PassportCatalog::new();
        cat.add(build_sample("svc-1", 0.30, EnergyLabel::A));
        cat.add(build_sample("svc-2", 0.60, EnergyLabel::B));
        let results = cat.find_by_model("meta-llama/Llama-3.3-70B-Instruct");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn catalog_find_by_rating() {
        let mut cat = PassportCatalog::new();
        cat.add(build_sample("svc-a", 0.30, EnergyLabel::A));
        cat.add(build_sample("svc-b", 0.60, EnergyLabel::B));
        cat.add(build_sample("svc-d", 1.00, EnergyLabel::D));
        cat.add(build_sample("svc-g", 2.00, EnergyLabel::G));
        assert_eq!(cat.find_by_rating(EnergyLabel::C).len(), 2);
        assert_eq!(cat.find_by_rating(EnergyLabel::G).len(), 4);
        assert_eq!(cat.find_by_rating(EnergyLabel::A).len(), 1);
    }

    #[test]
    fn catalog_leaderboard_orders_best_first() {
        let mut cat = PassportCatalog::new();
        cat.add(build_sample("svc-worst", 1.80, EnergyLabel::G));
        cat.add(build_sample("svc-best", 0.20, EnergyLabel::A));
        cat.add(build_sample("svc-mid", 0.85, EnergyLabel::C));
        let board = cat.leaderboard();
        assert_eq!(board[0].service_id, "svc-best");
        assert_eq!(board[2].service_id, "svc-worst");
    }

    #[test]
    fn pricing_default_uses_base_chain() {
        let p = PricingInfo::default();
        assert_eq!(p.settlement_network, "eip155:8453");
        assert!(p.x402_endpoint.is_empty());
    }
}
