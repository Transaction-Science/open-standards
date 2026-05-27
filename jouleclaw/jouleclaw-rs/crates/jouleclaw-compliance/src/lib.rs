//! # jouleclaw-compliance
//!
//! Regulatory-compliance wedge for JouleClaw.
//!
//! This crate translates the honest energy telemetry produced by
//! [`jouleclaw_energy`] into the structures regulators care about:
//!
//! - **SCI** — Software Carbon Intensity per ISO/IEC 21031 / Green
//!   Software Foundation. See [`sci`].
//! - **AI Service Passport** — machine-readable energy + carbon +
//!   compliance bundle, BLAKE3-sealable for tamper evidence. See
//!   [`passport`].
//! - **EU compliance reports** — EED (Energy Efficiency Directive),
//!   CSRD Scope 3, EU AI Act energy disclosure, ESPR Digital Product
//!   Passport. See [`eu_compliance`].
//!
//! Every passport links a [`passport::MeasurementMethod`] to a
//! [`jouleclaw_energy::Provenance`] tag — measurement honesty travels
//! all the way from the platform counter into the compliance report.
//! Apple Silicon (until the IOReport FFI ships) flows through as
//! `Provenance::Estimator`; RAPL and NVML flow through as
//! `Provenance::HwShunt`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod sci;
pub mod passport;
pub mod eu_compliance;

pub use eu_compliance::{
    AiActReport, CsrdScope3Report, CustomerAllocation, EedReport, EsprReport,
    generate_ai_act_report, generate_csrd_scope3, generate_eed_report,
};
pub use passport::{
    AiActCompliance, AiServicePassport, CarbonProfile, ComplianceRecord, CsrdCompliance,
    EedCompliance, EnergyLabel, EnergyProfile, EsprCompliance, HardwareProfile, LocationInfo,
    MeasurementMethod, ModelProfile, PassportBuilder, PassportCatalog, PassportError,
    PassportLabel, PricingInfo,
};
pub use sci::{
    EmbodiedCarbonEstimate, SciConfig, SciError, SciMeasurements, SciScore, calculate_sci,
    calculate_sci_checked,
};
