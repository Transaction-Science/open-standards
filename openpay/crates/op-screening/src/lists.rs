//! Sanctions list domain model.
//!
//! Each authority publishes a slightly different schema, but they all
//! agree on the same skeleton: an entity (person, organisation, vessel,
//! or aircraft) carries a primary name, aliases, identifying numbers,
//! addresses, and one-or-more sanctions programs it is named under.
//! [`SanctionedEntity`] is the union we ingest into.

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

/// Which authority published the entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SanctionsList {
    /// US Treasury — Specially Designated Nationals (SDN).
    OfacSdn,
    /// US Treasury — Non-SDN Consolidated Sanctions list.
    OfacConsolidated,
    /// EU Council — consolidated financial sanctions list.
    EuConsolidated,
    /// UN Security Council — consolidated list.
    UnConsolidated,
    /// HM Treasury (UK) — consolidated list of financial sanctions targets.
    HmtUk,
    /// Australia DFAT — consolidated list.
    AustraliaDfat,
    /// Canada SEMA — Special Economic Measures Act regulations.
    CanadaSema,
    /// Japan Ministry of Finance.
    JapanMof,
}

impl SanctionsList {
    /// Human-readable short name.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::OfacSdn => "OFAC SDN",
            Self::OfacConsolidated => "OFAC Consolidated",
            Self::EuConsolidated => "EU Consolidated",
            Self::UnConsolidated => "UN Consolidated",
            Self::HmtUk => "HMT UK",
            Self::AustraliaDfat => "Australia DFAT",
            Self::CanadaSema => "Canada SEMA",
            Self::JapanMof => "Japan MOF",
        }
    }
}

/// What kind of thing the entry names.
///
/// Sanctions programs apply to natural persons, legal entities, and to
/// specific transportation assets (vessel IMO numbers, aircraft tail
/// numbers) that have themselves been designated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityType {
    /// A natural person.
    Individual,
    /// A company, organisation, or other legal entity.
    Entity,
    /// A maritime vessel (named by IMO number and ship name).
    Vessel,
    /// An aircraft (named by registration / tail number).
    Aircraft,
}

/// ISO 3166-1 alpha-2 country code.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CountryCode(pub String);

/// Postal / business address attached to an entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Address {
    /// Street address line(s), joined with newlines.
    pub street: Option<String>,
    /// City or locality.
    pub city: Option<String>,
    /// State, province, or region.
    pub region: Option<String>,
    /// Postal code.
    pub postal_code: Option<String>,
    /// ISO 3166-1 alpha-2 country, if known.
    pub country: Option<CountryCode>,
}

/// The kind of identifying document or number attached to an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentificationKind {
    /// National identity card.
    NationalId,
    /// Passport.
    Passport,
    /// Driver's licence.
    DriversLicence,
    /// Tax identification number.
    TaxId,
    /// Maritime IMO number (vessels).
    Imo,
    /// Aircraft registration / tail number.
    AircraftTail,
    /// Anything not in the enum above; carry the raw label.
    Other,
}

/// A piece of identifying information published with an entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identification {
    /// What sort of identifier this is.
    pub kind: IdentificationKind,
    /// Free-text label as published (e.g. `"Cedula No."`, `"DNI"`).
    pub label: Option<String>,
    /// The identifier value itself.
    pub value: String,
    /// Issuing country, if known.
    pub country: Option<CountryCode>,
}

/// A single sanctioned entity, normalised across all upstream list formats.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SanctionedEntity {
    /// Stable per-list identifier (e.g. OFAC's `uid`, EU `logicalId`).
    pub id: String,
    /// Canonical primary name as published.
    pub name: String,
    /// AKAs, FKAs, and other published aliases.
    pub name_aliases: Vec<String>,
    /// Individual / Entity / Vessel / Aircraft.
    pub entity_type: EntityType,
    /// Date of birth (individuals only).
    pub dob: Option<NaiveDate>,
    /// Place of birth, free-text (individuals only).
    pub place_of_birth: Option<String>,
    /// Known addresses.
    pub addresses: Vec<Address>,
    /// Nationalities / citizenships.
    pub nationalities: Vec<CountryCode>,
    /// Published identifications.
    pub identifications: Vec<Identification>,
    /// Sanctions programmes the entry sits under (e.g. `"SDGT"`, `"UKRAINE-EO13662"`).
    pub programs: Vec<String>,
    /// When the upstream list last updated this entry.
    pub last_updated: DateTime<Utc>,
    /// Which source list this entry came from.
    pub source_list: SanctionsList,
}
