//! IEA per-country carbon-intensity baselines (annual averages, gCO2e/kWh).
//!
//! Used as a last-resort fallback when a live provider is unavailable or
//! the requested zone is out of catalog. Values come from publicly
//! reported IEA grid emission factors for electricity generation (2024
//! edition). Numbers are intentionally rounded — this is a fallback,
//! not a measurement.

use chrono::Utc;

use crate::error::{CarbonError, Result};
use crate::intensity::{CarbonIntensity, IntensityKind, ProviderKind, Zone};

/// One IEA baseline row.
#[derive(Debug, Clone, Copy)]
pub struct IeaRow {
    /// ISO 3166-1 alpha-2 country code (or special tag like "WORLD").
    pub country: &'static str,
    /// Annual grid intensity in gCO2e per kWh.
    pub g_co2e_per_kwh: f64,
}

/// The static IEA baseline table. Sorted by country code for easy diffs.
pub const IEA_TABLE: &[IeaRow] = &[
    IeaRow { country: "AU", g_co2e_per_kwh: 540.0 }, // Australia (coal-heavy)
    IeaRow { country: "BR", g_co2e_per_kwh: 90.0 },  // Brazil (hydro-heavy)
    IeaRow { country: "CA", g_co2e_per_kwh: 130.0 }, // Canada (hydro + nuclear)
    IeaRow { country: "CH", g_co2e_per_kwh: 35.0 },  // Switzerland
    IeaRow { country: "CN", g_co2e_per_kwh: 555.0 }, // China
    IeaRow { country: "DE", g_co2e_per_kwh: 380.0 }, // Germany
    IeaRow { country: "ES", g_co2e_per_kwh: 175.0 }, // Spain
    IeaRow { country: "FI", g_co2e_per_kwh: 110.0 }, // Finland
    IeaRow { country: "FR", g_co2e_per_kwh: 60.0 },  // France (nuclear)
    IeaRow { country: "GB", g_co2e_per_kwh: 220.0 }, // United Kingdom
    IeaRow { country: "IE", g_co2e_per_kwh: 290.0 }, // Ireland
    IeaRow { country: "IN", g_co2e_per_kwh: 635.0 }, // India
    IeaRow { country: "IS", g_co2e_per_kwh: 28.0 },  // Iceland (geothermal/hydro)
    IeaRow { country: "IT", g_co2e_per_kwh: 270.0 }, // Italy
    IeaRow { country: "JP", g_co2e_per_kwh: 470.0 }, // Japan
    IeaRow { country: "KR", g_co2e_per_kwh: 430.0 }, // South Korea
    IeaRow { country: "NL", g_co2e_per_kwh: 320.0 }, // Netherlands
    IeaRow { country: "NO", g_co2e_per_kwh: 30.0 },  // Norway (hydro)
    IeaRow { country: "PL", g_co2e_per_kwh: 660.0 }, // Poland
    IeaRow { country: "SE", g_co2e_per_kwh: 40.0 },  // Sweden
    IeaRow { country: "SG", g_co2e_per_kwh: 415.0 }, // Singapore
    IeaRow { country: "US", g_co2e_per_kwh: 370.0 }, // United States (avg)
    IeaRow { country: "WORLD", g_co2e_per_kwh: 481.0 },
    IeaRow { country: "ZA", g_co2e_per_kwh: 900.0 }, // South Africa (coal)
];

/// Look up an IEA row by country code (case-insensitive). Returns
/// `UnknownZone` if the code is not in the table.
pub fn lookup(country: &str) -> Result<IeaRow> {
    IEA_TABLE
        .iter()
        .find(|row| row.country.eq_ignore_ascii_case(country))
        .copied()
        .ok_or_else(|| CarbonError::UnknownZone(country.to_string()))
}

/// Look up an IEA row by country code and wrap it as a [`CarbonIntensity`].
pub fn intensity_for(country: &str) -> Result<CarbonIntensity> {
    let row = lookup(country)?;
    Ok(CarbonIntensity::new(
        Zone::new(row.country),
        row.g_co2e_per_kwh,
        IntensityKind::Average,
        ProviderKind::IeaBaseline,
        Utc::now(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_sorted_and_unique() {
        let mut keys: Vec<&str> = IEA_TABLE.iter().map(|r| r.country).collect();
        let original = keys.clone();
        keys.sort();
        assert_eq!(keys, original, "IEA_TABLE must stay sorted by country");
        keys.dedup();
        assert_eq!(keys.len(), IEA_TABLE.len(), "duplicate country in table");
    }

    #[test]
    fn france_is_low_carbon() {
        let fr = lookup("FR").expect("FR present");
        assert!(fr.g_co2e_per_kwh < 100.0);
    }

    #[test]
    fn unknown_country_errors() {
        assert!(matches!(lookup("XX"), Err(CarbonError::UnknownZone(_))));
    }

    #[test]
    fn intensity_for_world() {
        let ci = intensity_for("WORLD").expect("ok");
        assert_eq!(ci.provider, ProviderKind::IeaBaseline);
        assert!(ci.g_co2e_per_kwh > 400.0);
    }
}
