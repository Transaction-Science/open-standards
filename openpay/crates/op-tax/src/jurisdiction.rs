//! Jurisdictional addressing.
//!
//! Tax authority is hierarchical: a transaction in Seattle is
//! simultaneously taxable by the United States (federal — none on
//! retail sales, but excise applies), Washington State, King County,
//! the City of Seattle, and the King County Transit Special District.
//!
//! We model that hierarchy as four nested optional levels:
//! `country > region > locality > special_district`. A jurisdiction
//! with `region = None` means "country-level only" — useful for VAT,
//! where the tax-bearing layer is the country.
//!
//! ## Identifier discipline
//!
//! Country codes are ISO 3166-1 alpha-2 (`US`, `DE`, `IN`).
//!
//! Region codes are ISO 3166-2 subdivisions (`US-WA`, `DE-BY`,
//! `IN-MH`). We store only the subdivision part (`WA`, `BY`, `MH`)
//! since the country is already on the parent.
//!
//! Locality and special-district codes are free-form ASCII strings.
//! In production deployments they are typically GEOID (US Census),
//! INSEE (FR), or vendor-specific (Avalara `JurisCode`, Vertex
//! `LocationCode`). We do not enforce a scheme — the rate table is
//! the source of truth for which locality strings are populated.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;

/// ISO 3166-1 alpha-2 country code (`US`, `GB`, `JP`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CountryCode(pub String);

impl CountryCode {
    /// Construct from a 2-letter code. Forces uppercase.
    #[must_use]
    pub fn new(code: &str) -> Self {
        Self(code.to_ascii_uppercase())
    }
}

impl fmt::Display for CountryCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl PartialOrd for CountryCode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CountryCode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

/// ISO 3166-2 subdivision code, without the country prefix
/// (`WA` for Washington, `BY` for Bavaria, `MH` for Maharashtra).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct RegionCode(pub String);

impl RegionCode {
    /// Construct from a subdivision string. Forces uppercase.
    #[must_use]
    pub fn new(code: &str) -> Self {
        Self(code.to_ascii_uppercase())
    }
}

impl fmt::Display for RegionCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Locality identifier — city, town, parish, county. Free-form.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct LocalityCode(pub String);

impl LocalityCode {
    /// Construct from any locality identifier (no canonicalization).
    #[must_use]
    pub fn new(code: &str) -> Self {
        Self(code.to_owned())
    }
}

impl fmt::Display for LocalityCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Special-district identifier — transit, school, stadium, hospital,
/// resort. Free-form. Many US sales-tax jurisdictions add 0.1%–1.0%
/// on top of state+county+city for one or more named districts.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct DistrictCode(pub String);

impl DistrictCode {
    /// Construct from any district identifier.
    #[must_use]
    pub fn new(code: &str) -> Self {
        Self(code.to_owned())
    }
}

impl fmt::Display for DistrictCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A point in the tax-authority hierarchy.
///
/// `country` is always present; the lower levels are optional.
/// Equality is structural — `(US, Some(WA), None, None)` is a
/// different jurisdiction from `(US, Some(WA), Some(Seattle), None)`.
///
/// Sort order matches the natural hierarchy: country, then region,
/// then locality, then special district — which means rate-table
/// lookups iterating in order naturally see broader scopes first.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Jurisdiction {
    /// ISO 3166-1 alpha-2 country.
    pub country: CountryCode,
    /// ISO 3166-2 subdivision (state, province, prefecture, Land, …).
    pub region: Option<RegionCode>,
    /// City / town / county / parish.
    pub locality: Option<LocalityCode>,
    /// Transit / school / stadium / etc. special district.
    pub special_district: Option<DistrictCode>,
}

impl Jurisdiction {
    /// Construct a country-only jurisdiction (typical for VAT).
    #[must_use]
    pub fn country(country: &str) -> Self {
        Self {
            country: CountryCode::new(country),
            region: None,
            locality: None,
            special_district: None,
        }
    }

    /// Construct a country+region jurisdiction (typical for US
    /// state-level sales-tax base rate).
    #[must_use]
    pub fn region(country: &str, region: &str) -> Self {
        Self {
            country: CountryCode::new(country),
            region: Some(RegionCode::new(region)),
            locality: None,
            special_district: None,
        }
    }

    /// Construct a country+region+locality jurisdiction (city sales tax).
    #[must_use]
    pub fn locality(country: &str, region: &str, locality: &str) -> Self {
        Self {
            country: CountryCode::new(country),
            region: Some(RegionCode::new(region)),
            locality: Some(LocalityCode::new(locality)),
            special_district: None,
        }
    }

    /// Construct a full four-level jurisdiction.
    #[must_use]
    pub fn full(country: &str, region: &str, locality: &str, district: &str) -> Self {
        Self {
            country: CountryCode::new(country),
            region: Some(RegionCode::new(region)),
            locality: Some(LocalityCode::new(locality)),
            special_district: Some(DistrictCode::new(district)),
        }
    }

    /// Yields the ancestors of this jurisdiction, broadest first.
    ///
    /// For `(US, WA, Seattle, KingTransit)` the chain is:
    ///   1. `(US, None, None, None)`
    ///   2. `(US, WA, None, None)`
    ///   3. `(US, WA, Seattle, None)`
    ///   4. `(US, WA, Seattle, KingTransit)` (self)
    ///
    /// This is the iteration order the native calculator uses when
    /// compounding US sales-tax layers.
    #[must_use]
    pub fn ancestors(&self) -> Vec<Self> {
        let mut out = Vec::with_capacity(4);
        out.push(Self {
            country: self.country.clone(),
            region: None,
            locality: None,
            special_district: None,
        });
        if let Some(region) = &self.region {
            out.push(Self {
                country: self.country.clone(),
                region: Some(region.clone()),
                locality: None,
                special_district: None,
            });
            if let Some(locality) = &self.locality {
                out.push(Self {
                    country: self.country.clone(),
                    region: Some(region.clone()),
                    locality: Some(locality.clone()),
                    special_district: None,
                });
                if let Some(district) = &self.special_district {
                    out.push(Self {
                        country: self.country.clone(),
                        region: Some(region.clone()),
                        locality: Some(locality.clone()),
                        special_district: Some(district.clone()),
                    });
                }
            }
        }
        out
    }

    /// Convenience: is this jurisdiction inside the European Union?
    /// Used for reverse-charge detection.
    ///
    /// We hard-code the EU-27 membership as of 2026 (post-Brexit).
    #[must_use]
    pub fn is_eu_member(&self) -> bool {
        matches!(
            self.country.0.as_str(),
            "AT" | "BE"
                | "BG"
                | "HR"
                | "CY"
                | "CZ"
                | "DK"
                | "EE"
                | "FI"
                | "FR"
                | "DE"
                | "GR"
                | "HU"
                | "IE"
                | "IT"
                | "LV"
                | "LT"
                | "LU"
                | "MT"
                | "NL"
                | "PL"
                | "PT"
                | "RO"
                | "SK"
                | "SI"
                | "ES"
                | "SE"
        )
    }
}

impl fmt::Display for Jurisdiction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.country)?;
        if let Some(r) = &self.region {
            write!(f, "-{r}")?;
        }
        if let Some(l) = &self.locality {
            write!(f, "/{l}")?;
        }
        if let Some(d) = &self.special_district {
            write!(f, "#{d}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ancestors_four_levels() {
        let j = Jurisdiction::full("US", "WA", "Seattle", "KingTransit");
        let a = j.ancestors();
        assert_eq!(a.len(), 4);
        assert_eq!(a[0], Jurisdiction::country("US"));
        assert_eq!(a[1], Jurisdiction::region("US", "WA"));
        assert_eq!(a[2], Jurisdiction::locality("US", "WA", "Seattle"));
        assert_eq!(a[3], j);
    }

    #[test]
    fn ancestors_country_only() {
        let j = Jurisdiction::country("DE");
        let a = j.ancestors();
        assert_eq!(a.len(), 1);
        assert_eq!(a[0], j);
    }

    #[test]
    fn eu_membership_post_brexit() {
        assert!(Jurisdiction::country("DE").is_eu_member());
        assert!(Jurisdiction::country("IE").is_eu_member());
        assert!(!Jurisdiction::country("GB").is_eu_member());
        assert!(!Jurisdiction::country("US").is_eu_member());
        assert!(!Jurisdiction::country("CH").is_eu_member());
    }

    #[test]
    fn display_format() {
        assert_eq!(Jurisdiction::country("US").to_string(), "US");
        assert_eq!(Jurisdiction::region("US", "WA").to_string(), "US-WA");
        assert_eq!(
            Jurisdiction::locality("US", "WA", "Seattle").to_string(),
            "US-WA/Seattle"
        );
        assert_eq!(
            Jurisdiction::full("US", "WA", "Seattle", "KingTransit").to_string(),
            "US-WA/Seattle#KingTransit"
        );
    }

    #[test]
    fn sort_order_hierarchical() {
        let mut v = [
            Jurisdiction::locality("US", "WA", "Seattle"),
            Jurisdiction::region("US", "WA"),
            Jurisdiction::country("US"),
        ];
        v.sort();
        assert_eq!(v[0], Jurisdiction::country("US"));
        assert_eq!(v[1], Jurisdiction::region("US", "WA"));
        assert_eq!(v[2], Jurisdiction::locality("US", "WA", "Seattle"));
    }
}
