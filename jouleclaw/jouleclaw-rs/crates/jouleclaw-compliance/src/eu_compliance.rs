//! EU regulatory-compliance reporting.
//!
//! Generates reports for:
//! - **EED** — Energy Efficiency Directive (data-centre PUE / renewables)
//! - **CSRD Scope 3** — corporate downstream emissions per customer
//! - **EU AI Act** — model training + inference energy disclosure
//! - **ESPR** — Ecodesign for Sustainable Products / Digital Product Passport

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

// ── EED ─────────────────────────────────────────────────────────────

/// EED-compliant data-centre energy report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EedReport {
    /// Reporting period (e.g. `"2025-Q1"`).
    pub reporting_period: String,
    /// Power Usage Effectiveness — total energy / IT energy.
    pub pue: f64,
    /// Total facility energy (kWh).
    pub total_energy_kwh: f64,
    /// IT-equipment energy (kWh).
    pub it_energy_kwh: f64,
    /// Renewable share (0–100).
    pub renewable_pct: f64,
    /// Waste-heat recovery share (0–100).
    pub waste_heat_recovery_pct: f64,
    /// Cooling water usage (litres).
    pub water_usage_liters: f64,
}

impl EedReport {
    /// `true` when PUE < 1.5 *and* renewable share ≥ 50 %.
    pub fn is_compliant(&self) -> bool {
        self.pue < 1.5 && self.renewable_pct >= 50.0
    }
}

// ── CSRD Scope 3 ────────────────────────────────────────────────────

/// Per-customer Scope 3 emissions allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerAllocation {
    /// Opaque customer identifier.
    pub customer_id: String,
    /// Energy attributed to this customer (kWh).
    pub energy_kwh: f64,
    /// Emissions attributed to this customer (tonnes CO2).
    pub emissions_tco2: f64,
    /// Number of workloads run for this customer.
    pub workload_count: u32,
}

/// CSRD Scope 3 emissions report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrdScope3Report {
    /// Organisation name.
    pub organization: String,
    /// Reporting period.
    pub reporting_period: String,
    /// Total Scope 3 emissions (tonnes CO2).
    pub total_emissions_tco2: f64,
    /// Per-customer allocations.
    pub per_customer_allocations: Vec<CustomerAllocation>,
    /// Allocation methodology.
    pub methodology: String,
}

// ── EU AI Act ───────────────────────────────────────────────────────

/// EU AI Act energy / emissions disclosure for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiActReport {
    /// Model identifier.
    pub model_name: String,
    /// Total training energy (kWh).
    pub training_energy_kwh: f64,
    /// Total training emissions (tonnes CO2).
    pub training_emissions_tco2: f64,
    /// Inference energy per request (kWh).
    pub inference_energy_kwh_per_request: f64,
    /// Total inference requests served.
    pub total_inference_requests: u64,
    /// Free-text hardware description.
    pub hardware_description: String,
}

impl AiActReport {
    /// Total inference energy (kWh) over the reporting period.
    pub fn total_inference_energy_kwh(&self) -> f64 {
        self.inference_energy_kwh_per_request * self.total_inference_requests as f64
    }
}

// ── ESPR ────────────────────────────────────────────────────────────

/// ESPR Digital Product Passport report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EsprReport {
    /// Product / model identifier.
    pub product_name: String,
    /// EU energy label (`"A"`..`"G"`).
    pub energy_label: String,
    /// Repairability score on a 0.0–10.0 scale.
    pub repairability_score: f64,
    /// Expected product lifetime (years).
    pub expected_lifetime_years: f64,
    /// Recyclable-materials share (0–100).
    pub recyclable_materials_pct: f64,
}

// ── Generators ──────────────────────────────────────────────────────

/// Build an [`EedReport`] from raw facility metrics.
pub fn generate_eed_report(
    period: &str,
    total_kwh: f64,
    it_kwh: f64,
    renewable_pct: f64,
    waste_heat_pct: f64,
    water_liters: f64,
) -> EedReport {
    let pue = if it_kwh > 0.0 {
        total_kwh / it_kwh
    } else {
        0.0
    };
    EedReport {
        reporting_period: period.to_string(),
        pue,
        total_energy_kwh: total_kwh,
        it_energy_kwh: it_kwh,
        renewable_pct,
        waste_heat_recovery_pct: waste_heat_pct,
        water_usage_liters: water_liters,
    }
}

/// Build a [`CsrdScope3Report`] — total emissions are summed from
/// the per-customer allocations.
pub fn generate_csrd_scope3(
    org: &str,
    period: &str,
    allocations: Vec<CustomerAllocation>,
    methodology: &str,
) -> CsrdScope3Report {
    let total_emissions_tco2: f64 = allocations.iter().map(|a| a.emissions_tco2).sum();
    CsrdScope3Report {
        organization: org.to_string(),
        reporting_period: period.to_string(),
        total_emissions_tco2,
        per_customer_allocations: allocations,
        methodology: methodology.to_string(),
    }
}

/// Build an [`AiActReport`] from training + inference metrics.
pub fn generate_ai_act_report(
    model: &str,
    training_kwh: f64,
    training_tco2: f64,
    inference_kwh_per_req: f64,
    total_requests: u64,
    hardware: &str,
) -> AiActReport {
    AiActReport {
        model_name: model.to_string(),
        training_energy_kwh: training_kwh,
        training_emissions_tco2: training_tco2,
        inference_energy_kwh_per_request: inference_kwh_per_req,
        total_inference_requests: total_requests,
        hardware_description: hardware.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eed_compliant_when_pue_low_and_renewable_high() {
        let r = EedReport {
            reporting_period: "2025-Q1".to_string(),
            pue: 1.2,
            total_energy_kwh: 120_000.0,
            it_energy_kwh: 100_000.0,
            renewable_pct: 80.0,
            waste_heat_recovery_pct: 30.0,
            water_usage_liters: 500_000.0,
        };
        assert!(r.is_compliant());
    }

    #[test]
    fn eed_non_compliant_high_pue() {
        let r = EedReport {
            reporting_period: "2025-Q2".to_string(),
            pue: 1.8,
            total_energy_kwh: 180_000.0,
            it_energy_kwh: 100_000.0,
            renewable_pct: 60.0,
            waste_heat_recovery_pct: 10.0,
            water_usage_liters: 800_000.0,
        };
        assert!(!r.is_compliant());
    }

    #[test]
    fn eed_non_compliant_low_renewable() {
        let r = EedReport {
            reporting_period: "2025-Q3".to_string(),
            pue: 1.3,
            total_energy_kwh: 130_000.0,
            it_energy_kwh: 100_000.0,
            renewable_pct: 40.0,
            waste_heat_recovery_pct: 20.0,
            water_usage_liters: 600_000.0,
        };
        assert!(!r.is_compliant());
    }

    #[test]
    fn csrd_sums_allocations() {
        let allocations = vec![
            CustomerAllocation {
                customer_id: "cust-1".to_string(),
                energy_kwh: 1000.0,
                emissions_tco2: 0.5,
                workload_count: 10,
            },
            CustomerAllocation {
                customer_id: "cust-2".to_string(),
                energy_kwh: 2000.0,
                emissions_tco2: 1.0,
                workload_count: 20,
            },
        ];
        let r = generate_csrd_scope3("AcmeCorp", "2025-H1", allocations, "energy-based");
        assert!((r.total_emissions_tco2 - 1.5).abs() < 1e-10);
    }

    #[test]
    fn csrd_empty_allocations_zero_total() {
        let r = generate_csrd_scope3("AcmeCorp", "2025-H1", Vec::new(), "energy-based");
        assert!(r.total_emissions_tco2.abs() < 1e-10);
        assert!(r.per_customer_allocations.is_empty());
    }

    #[test]
    fn ai_act_inference_total() {
        let r = AiActReport {
            model_name: "llm-v1".to_string(),
            training_energy_kwh: 50_000.0,
            training_emissions_tco2: 25.0,
            inference_energy_kwh_per_request: 0.001,
            total_inference_requests: 1_000_000,
            hardware_description: "8xA100".to_string(),
        };
        assert!((r.total_inference_energy_kwh() - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn ai_act_zero_requests_zero_total() {
        let r = AiActReport {
            model_name: "llm-v2".to_string(),
            training_energy_kwh: 10_000.0,
            training_emissions_tco2: 5.0,
            inference_energy_kwh_per_request: 0.002,
            total_inference_requests: 0,
            hardware_description: "4xH100".to_string(),
        };
        assert!(r.total_inference_energy_kwh().abs() < 1e-10);
    }

    #[test]
    fn espr_creation() {
        let r = EsprReport {
            product_name: "JouleNode-1".to_string(),
            energy_label: "B".to_string(),
            repairability_score: 7.5,
            expected_lifetime_years: 6.0,
            recyclable_materials_pct: 85.0,
        };
        assert_eq!(r.energy_label, "B");
        assert!((r.repairability_score - 7.5).abs() < 1e-10);
    }

    #[test]
    fn eed_round_trip() {
        let r = EedReport {
            reporting_period: "2025-Q1".to_string(),
            pue: 1.3,
            total_energy_kwh: 130_000.0,
            it_energy_kwh: 100_000.0,
            renewable_pct: 70.0,
            waste_heat_recovery_pct: 25.0,
            water_usage_liters: 400_000.0,
        };
        let json = serde_json::to_string(&r).expect("serialize");
        let d: EedReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(d.reporting_period, "2025-Q1");
    }

    #[test]
    fn csrd_round_trip() {
        let r = CsrdScope3Report {
            organization: "Org".to_string(),
            reporting_period: "2025-H2".to_string(),
            total_emissions_tco2: 3.5,
            per_customer_allocations: vec![CustomerAllocation {
                customer_id: "c1".to_string(),
                energy_kwh: 500.0,
                emissions_tco2: 3.5,
                workload_count: 15,
            }],
            methodology: "energy-based".to_string(),
        };
        let json = serde_json::to_string(&r).expect("serialize");
        let d: CsrdScope3Report = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(d.per_customer_allocations.len(), 1);
    }

    #[test]
    fn ai_act_round_trip() {
        let r = generate_ai_act_report("model-x", 1000.0, 0.5, 0.0001, 500_000, "TPUv5");
        let json = serde_json::to_string(&r).expect("serialize");
        let d: AiActReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(d.total_inference_requests, 500_000);
    }

    #[test]
    fn espr_round_trip() {
        let r = EsprReport {
            product_name: "Node-2".to_string(),
            energy_label: "A".to_string(),
            repairability_score: 9.0,
            expected_lifetime_years: 8.0,
            recyclable_materials_pct: 92.0,
        };
        let json = serde_json::to_string(&r).expect("serialize");
        let d: EsprReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(d.energy_label, "A");
    }

    #[test]
    fn generate_eed_pue_arithmetic() {
        let r = generate_eed_report("2025-Q4", 150_000.0, 100_000.0, 75.0, 20.0, 300_000.0);
        assert!((r.pue - 1.5).abs() < 1e-10);
        // pue == 1.5 → not strictly < 1.5 → non-compliant
        assert!(!r.is_compliant());
    }

    #[test]
    fn generate_eed_zero_it_energy_zero_pue() {
        let r = generate_eed_report("2025-Q4", 0.0, 0.0, 75.0, 20.0, 300_000.0);
        assert!(r.pue.abs() < 1e-10);
    }
}
