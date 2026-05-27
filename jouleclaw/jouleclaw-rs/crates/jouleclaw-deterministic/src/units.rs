//! Unit conversion primitives. Numeric outputs round to 4 decimal
//! places via the `format!("{:.4}", v)` formatter. Trailing zeros are
//! preserved so receipts remain deterministic.

use jouleclaw_cascade::LawfulPrimitive;
use std::sync::Arc;

pub fn primitives() -> Vec<Arc<dyn LawfulPrimitive>> {
    vec![
        Arc::new(FahrenheitToCelsius),
        Arc::new(CelsiusToFahrenheit),
        Arc::new(CelsiusToKelvin),
        Arc::new(KelvinToCelsius),
        Arc::new(MilesToKm),
        Arc::new(KmToMiles),
        Arc::new(KgToLb),
        Arc::new(LbToKg),
        Arc::new(MetersToFeet),
        Arc::new(FeetToMeters),
        Arc::new(LitresToGallons),
        Arc::new(GallonsToLitres),
    ]
}

fn parse_f64(s: &str) -> Option<f64> {
    let parsed = s.trim().parse::<f64>().ok()?;
    if parsed.is_finite() { Some(parsed) } else { None }
}

fn fmt4(v: f64) -> String {
    format!("{v:.4}")
}

/// Recognise either `"<keyword> <num>"` or `"<num> <keyword>"`. Keyword
/// matching is case-insensitive. Returns the parsed number on a match.
fn extract_value(query: &str, keyword: &str) -> Option<f64> {
    let q = query.trim();
    let lower = q.to_ascii_lowercase();
    let kw_lower = keyword.to_ascii_lowercase();

    // "<keyword> <num>"
    if let Some(rest) = lower.strip_prefix(&kw_lower) {
        if rest.starts_with(|c: char| c.is_whitespace()) {
            return parse_f64(rest.trim());
        }
    }
    // "<num> <keyword>"
    if let Some(rest) = lower.strip_suffix(&kw_lower) {
        if rest.ends_with(|c: char| c.is_whitespace()) {
            return parse_f64(rest.trim());
        }
    }
    None
}

// ---- temperature --------------------------------------------------------

pub struct FahrenheitToCelsius;
impl LawfulPrimitive for FahrenheitToCelsius {
    fn id(&self) -> &str {
        "lawful:units:f-to-c"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let v = extract_value(query, "f to c")?;
        Some(fmt4((v - 32.0) * 5.0 / 9.0))
    }
    fn declared_cost_uj(&self) -> u64 {
        60
    }
}

pub struct CelsiusToFahrenheit;
impl LawfulPrimitive for CelsiusToFahrenheit {
    fn id(&self) -> &str {
        "lawful:units:c-to-f"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let v = extract_value(query, "c to f")?;
        Some(fmt4(v * 9.0 / 5.0 + 32.0))
    }
    fn declared_cost_uj(&self) -> u64 {
        60
    }
}

pub struct CelsiusToKelvin;
impl LawfulPrimitive for CelsiusToKelvin {
    fn id(&self) -> &str {
        "lawful:units:c-to-k"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let v = extract_value(query, "c to k")?;
        Some(fmt4(v + 273.15))
    }
    fn declared_cost_uj(&self) -> u64 {
        55
    }
}

pub struct KelvinToCelsius;
impl LawfulPrimitive for KelvinToCelsius {
    fn id(&self) -> &str {
        "lawful:units:k-to-c"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let v = extract_value(query, "k to c")?;
        Some(fmt4(v - 273.15))
    }
    fn declared_cost_uj(&self) -> u64 {
        55
    }
}

// ---- distance -----------------------------------------------------------

pub struct MilesToKm;
impl LawfulPrimitive for MilesToKm {
    fn id(&self) -> &str {
        "lawful:units:miles-to-km"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let v = extract_value(query, "miles to km")?;
        Some(fmt4(v * 1.609344))
    }
    fn declared_cost_uj(&self) -> u64 {
        60
    }
}

pub struct KmToMiles;
impl LawfulPrimitive for KmToMiles {
    fn id(&self) -> &str {
        "lawful:units:km-to-miles"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let v = extract_value(query, "km to miles")?;
        Some(fmt4(v / 1.609344))
    }
    fn declared_cost_uj(&self) -> u64 {
        60
    }
}

// ---- mass ---------------------------------------------------------------

pub struct KgToLb;
impl LawfulPrimitive for KgToLb {
    fn id(&self) -> &str {
        "lawful:units:kg-to-lb"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let v = extract_value(query, "kg to lb")?;
        Some(fmt4(v * 2.2046226218))
    }
    fn declared_cost_uj(&self) -> u64 {
        60
    }
}

pub struct LbToKg;
impl LawfulPrimitive for LbToKg {
    fn id(&self) -> &str {
        "lawful:units:lb-to-kg"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let v = extract_value(query, "lb to kg")?;
        Some(fmt4(v / 2.2046226218))
    }
    fn declared_cost_uj(&self) -> u64 {
        60
    }
}

// ---- length -------------------------------------------------------------

pub struct MetersToFeet;
impl LawfulPrimitive for MetersToFeet {
    fn id(&self) -> &str {
        "lawful:units:m-to-ft"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let v = extract_value(query, "m to ft")?;
        Some(fmt4(v * 3.28083989501))
    }
    fn declared_cost_uj(&self) -> u64 {
        60
    }
}

pub struct FeetToMeters;
impl LawfulPrimitive for FeetToMeters {
    fn id(&self) -> &str {
        "lawful:units:ft-to-m"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let v = extract_value(query, "ft to m")?;
        Some(fmt4(v * 0.3048))
    }
    fn declared_cost_uj(&self) -> u64 {
        60
    }
}

// ---- volume (US gallons) ------------------------------------------------

pub struct LitresToGallons;
impl LawfulPrimitive for LitresToGallons {
    fn id(&self) -> &str {
        "lawful:units:l-to-gal"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let v = extract_value(query, "l to gal")?;
        Some(fmt4(v * 0.2641720524))
    }
    fn declared_cost_uj(&self) -> u64 {
        60
    }
}

pub struct GallonsToLitres;
impl LawfulPrimitive for GallonsToLitres {
    fn id(&self) -> &str {
        "lawful:units:gal-to-l"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let v = extract_value(query, "gal to l")?;
        Some(fmt4(v * 3.785411784))
    }
    fn declared_cost_uj(&self) -> u64 {
        60
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fahrenheit_celsius_round_trip() {
        assert_eq!(FahrenheitToCelsius.try_resolve("f to c 212").as_deref(), Some("100.0000"));
        assert_eq!(FahrenheitToCelsius.try_resolve("212 f to c").as_deref(), Some("100.0000"));
        assert_eq!(CelsiusToFahrenheit.try_resolve("c to f 100").as_deref(), Some("212.0000"));
        assert_eq!(CelsiusToFahrenheit.try_resolve("c to f 0").as_deref(), Some("32.0000"));
    }

    #[test]
    fn celsius_kelvin() {
        assert_eq!(CelsiusToKelvin.try_resolve("c to k 0").as_deref(), Some("273.1500"));
        assert_eq!(KelvinToCelsius.try_resolve("k to c 273.15").as_deref(), Some("0.0000"));
    }

    #[test]
    fn miles_km() {
        assert_eq!(MilesToKm.try_resolve("miles to km 1").as_deref(), Some("1.6093"));
        assert_eq!(KmToMiles.try_resolve("km to miles 1.609344").as_deref(), Some("1.0000"));
    }

    #[test]
    fn kg_lb() {
        assert_eq!(KgToLb.try_resolve("kg to lb 1").as_deref(), Some("2.2046"));
        // 1 / 2.2046226218 ≈ 0.45359237 → 0.4536 to 4dp
        assert_eq!(LbToKg.try_resolve("lb to kg 1").as_deref(), Some("0.4536"));
    }

    #[test]
    fn m_ft() {
        let s = MetersToFeet.try_resolve("m to ft 1");
        assert!(s.as_deref().map(|x| x.starts_with("3.2808")).unwrap_or(false));
        let s = FeetToMeters.try_resolve("ft to m 1");
        assert_eq!(s.as_deref(), Some("0.3048"));
    }

    #[test]
    fn l_gal() {
        // 1 L ≈ 0.26417205 gal → 0.2642 to 4dp
        assert_eq!(LitresToGallons.try_resolve("l to gal 1").as_deref(), Some("0.2642"));
        assert_eq!(GallonsToLitres.try_resolve("gal to l 1").as_deref(), Some("3.7854"));
    }

    #[test]
    fn malformed_returns_none() {
        assert!(FahrenheitToCelsius.try_resolve("f to c hot").is_none());
        assert!(MilesToKm.try_resolve("miles to km").is_none());
        assert!(KgToLb.try_resolve("how heavy is the moon").is_none());
    }

    #[test]
    fn category_count() {
        assert!(primitives().len() >= 10);
    }
}
