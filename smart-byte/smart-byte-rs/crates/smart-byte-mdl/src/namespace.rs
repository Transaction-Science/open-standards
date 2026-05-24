//! ISO/IEC 18013-5 namespaces and canonical mDL data elements.
//!
//! The mDL namespace `org.iso.18013.5.1` carries the bulk of holder
//! identity claims, the AAMVA add-on `org.iso.18013.5.1.aamva` carries
//! US-specific extensions, and the eIDAS namespaces carry EU additions.
//!
//! This module exposes the canonical element identifiers as constants
//! and provides strongly typed accessors over a CBOR claim map. Callers
//! that prefer to walk the raw map directly may; the typed accessors
//! exist so application code does not stringly-type element names.

use std::collections::BTreeMap;

use ciborium::value::Value as CborValue;
use chrono::NaiveDate;

use crate::error::MdlError;

/// Canonical mDL namespace identifier (ISO/IEC 18013-5 clause 7.2).
pub const NS_MDL: &str = "org.iso.18013.5.1";

/// AAMVA US driver-licence add-on namespace (AAMVA mDL Implementation
/// Guidelines, mapped onto the ISO 18013-5 add-on namespace registry).
pub const NS_MDL_AAMVA: &str = "org.iso.18013.5.1.aamva";

/// eIDAS 2.0 European Digital Identity Wallet PID namespace
/// (ARF v1.x).
pub const NS_EIDAS_PID: &str = "eu.europa.ec.eudi.pid.1";

/// eIDAS 2.0 European Digital Identity Wallet mDL namespace
/// (ARF v1.x — the EU profile of ISO 18013-5).
pub const NS_EIDAS_MDL: &str = "eu.europa.ec.eudi.mdl.1";

/// Canonical mDL element identifiers, ISO/IEC 18013-5 §7.2.1.
pub mod mdl {
    /// Holder's family (surname) name.
    pub const FAMILY_NAME: &str = "family_name";
    /// Holder's given (first) names.
    pub const GIVEN_NAME: &str = "given_name";
    /// Holder's date of birth (ISO 8601 full-date).
    pub const BIRTH_DATE: &str = "birth_date";
    /// Date the credential was issued.
    pub const ISSUE_DATE: &str = "issue_date";
    /// Date the credential expires.
    pub const EXPIRY_DATE: &str = "expiry_date";
    /// ISO 3166-1 alpha-2 issuing country code.
    pub const ISSUING_COUNTRY: &str = "issuing_country";
    /// Plain-text issuing authority name.
    pub const ISSUING_AUTHORITY: &str = "issuing_authority";
    /// Driver-licence document number.
    pub const DOCUMENT_NUMBER: &str = "document_number";
    /// JPEG-2000-encoded portrait of the holder.
    pub const PORTRAIT: &str = "portrait";
    /// CBOR array of driving privilege records.
    pub const DRIVING_PRIVILEGES: &str = "driving_privileges";
    /// UN distinguishing sign (vehicle-licence country code).
    pub const UN_DISTINGUISHING_SIGN: &str = "un_distinguishing_sign";
    /// Issuer-assigned administrative number.
    pub const ADMINISTRATIVE_NUMBER: &str = "administrative_number";
    /// Sex (ISO/IEC 5218: 0 unknown, 1 male, 2 female, 9 N/A).
    pub const SEX: &str = "sex";
    /// Height in cm.
    pub const HEIGHT: &str = "height";
    /// Weight in kg.
    pub const WEIGHT: &str = "weight";
    /// Eye colour (free text).
    pub const EYE_COLOUR: &str = "eye_colour";
    /// Hair colour (free text).
    pub const HAIR_COLOUR: &str = "hair_colour";
    /// Place of birth (free text).
    pub const BIRTH_PLACE: &str = "birth_place";
    /// Holder's resident address (single-line).
    pub const RESIDENT_ADDRESS: &str = "resident_address";
    /// Date the portrait was captured.
    pub const PORTRAIT_CAPTURE_DATE: &str = "portrait_capture_date";
    /// Age in years (integer).
    pub const AGE_IN_YEARS: &str = "age_in_years";
    /// Year of birth.
    pub const AGE_BIRTH_YEAR: &str = "age_birth_year";
    /// Issuing jurisdiction (state/province within country).
    pub const ISSUING_JURISDICTION: &str = "issuing_jurisdiction";
    /// Nationality (ISO 3166-1 alpha-2).
    pub const NATIONALITY: &str = "nationality";
    /// Resident city.
    pub const RESIDENT_CITY: &str = "resident_city";
    /// Resident state.
    pub const RESIDENT_STATE: &str = "resident_state";
    /// Resident postal code.
    pub const RESIDENT_POSTAL_CODE: &str = "resident_postal_code";
    /// Resident country.
    pub const RESIDENT_COUNTRY: &str = "resident_country";
    /// Family name in national characters.
    pub const FAMILY_NAME_NATIONAL_CHARACTER: &str =
        "family_name_national_character";
    /// Given name in national characters.
    pub const GIVEN_NAME_NATIONAL_CHARACTER: &str =
        "given_name_national_character";
    /// Holder's usual mark / signature image.
    pub const SIGNATURE_USUAL_MARK: &str = "signature_usual_mark";

    /// Returns the canonical `age_over_NN` element identifier for `n`.
    /// ISO 18013-5 reserves these for age-over disclosure (commonly
    /// 18, 21, 25, 65).
    pub fn age_over(n: u8) -> String {
        format!("age_over_{n:02}")
    }

    /// Returns the canonical `biometric_template_xx` element identifier
    /// for an arbitrary alphabetic modality tag (e.g. "face").
    pub fn biometric_template(modality: &str) -> String {
        format!("biometric_template_{modality}")
    }
}

/// AAMVA add-on namespace elements (US driver-licence specifics).
/// The full set is enumerated in the AAMVA mDL Implementation Guidelines.
pub mod aamva {
    /// AAMVA card-revision date (DD).
    pub const DOMESTIC_VEHICLE_CLASS: &str = "domestic_vehicle_class";
    /// AAMVA ED — expiration date.
    pub const EDL_CREDENTIAL: &str = "EDL_credential";
    /// AAMVA family-name-truncation indicator (T, N, U).
    pub const FAMILY_NAME_TRUNCATION: &str = "family_name_truncation";
    /// AAMVA given-name-truncation indicator (T, N, U).
    pub const GIVEN_NAME_TRUNCATION: &str = "given_name_truncation";
    /// AAMVA name-suffix (e.g. "JR").
    pub const NAME_SUFFIX: &str = "name_suffix";
    /// AAMVA AKA family name.
    pub const AKA_FAMILY_NAME: &str = "aka_family_name";
    /// AAMVA AKA given name.
    pub const AKA_GIVEN_NAME: &str = "aka_given_name";
    /// AAMVA weight range (1..9).
    pub const WEIGHT_RANGE: &str = "weight_range";
    /// AAMVA race / ethnicity.
    pub const RACE_ETHNICITY: &str = "race_ethnicity";
    /// AAMVA presence of REAL ID Act compliance.
    pub const DHS_COMPLIANCE: &str = "DHS_compliance";
    /// AAMVA presence of organ-donor flag.
    pub const ORGAN_DONOR: &str = "organ_donor";
    /// AAMVA presence of veteran indicator.
    pub const VETERAN: &str = "veteran";
}

/// EU eIDAS 2.0 PID-namespace elements (ARF v1.x).
pub mod eidas_pid {
    /// Family name at birth.
    pub const FAMILY_NAME_BIRTH: &str = "family_name_birth";
    /// Given name at birth.
    pub const GIVEN_NAME_BIRTH: &str = "given_name_birth";
    /// Issuance authority (eIDAS).
    pub const ISSUING_AUTHORITY: &str = "issuing_authority";
    /// Document issuance timestamp.
    pub const ISSUANCE_DATE: &str = "issuance_date";
    /// Document expiry timestamp.
    pub const EXPIRY_DATE: &str = "expiry_date";
    /// EU country code (ISO 3166-1 alpha-2).
    pub const ISSUING_COUNTRY: &str = "issuing_country";
}

/// Strongly typed accessor over a CBOR claim map for the canonical mDL
/// namespace `org.iso.18013.5.1`.
///
/// The accessor never copies large fields (portrait, biometric_template)
/// — they are borrowed out of the underlying map. Callers that need
/// owned bytes can `.to_vec()` themselves.
#[derive(Debug, Clone)]
pub struct MdlClaims<'a> {
    inner: &'a BTreeMap<String, CborValue>,
}

impl<'a> MdlClaims<'a> {
    /// Wrap a claim map.
    pub fn new(claims: &'a BTreeMap<String, CborValue>) -> Self {
        Self { inner: claims }
    }

    fn text(&self, key: &str) -> Result<Option<&'a str>, MdlError> {
        match self.inner.get(key) {
            None => Ok(None),
            Some(CborValue::Text(s)) => Ok(Some(s.as_str())),
            Some(_) => Err(MdlError::Type(format!("{key} is not text"))),
        }
    }

    fn bytes(&self, key: &str) -> Result<Option<&'a [u8]>, MdlError> {
        match self.inner.get(key) {
            None => Ok(None),
            Some(CborValue::Bytes(b)) => Ok(Some(b.as_slice())),
            Some(_) => Err(MdlError::Type(format!("{key} is not bytes"))),
        }
    }

    fn integer(&self, key: &str) -> Result<Option<i128>, MdlError> {
        match self.inner.get(key) {
            None => Ok(None),
            Some(CborValue::Integer(i)) => Ok(Some(i128::from(*i))),
            Some(_) => Err(MdlError::Type(format!("{key} is not an integer"))),
        }
    }

    fn bool(&self, key: &str) -> Result<Option<bool>, MdlError> {
        match self.inner.get(key) {
            None => Ok(None),
            Some(CborValue::Bool(b)) => Ok(Some(*b)),
            Some(_) => Err(MdlError::Type(format!("{key} is not boolean"))),
        }
    }

    fn date(&self, key: &str) -> Result<Option<NaiveDate>, MdlError> {
        let Some(s) = self.text(key)? else {
            return Ok(None);
        };
        NaiveDate::parse_from_str(s, "%Y-%m-%d")
            .map(Some)
            .map_err(|e| MdlError::Type(format!("{key}: {e}")))
    }

    /// `family_name`.
    pub fn family_name(&self) -> Result<Option<&'a str>, MdlError> {
        self.text(mdl::FAMILY_NAME)
    }
    /// `given_name`.
    pub fn given_name(&self) -> Result<Option<&'a str>, MdlError> {
        self.text(mdl::GIVEN_NAME)
    }
    /// `birth_date` parsed as a calendar date.
    pub fn birth_date(&self) -> Result<Option<NaiveDate>, MdlError> {
        self.date(mdl::BIRTH_DATE)
    }
    /// `issue_date` parsed as a calendar date.
    pub fn issue_date(&self) -> Result<Option<NaiveDate>, MdlError> {
        self.date(mdl::ISSUE_DATE)
    }
    /// `expiry_date` parsed as a calendar date.
    pub fn expiry_date(&self) -> Result<Option<NaiveDate>, MdlError> {
        self.date(mdl::EXPIRY_DATE)
    }
    /// `issuing_country` (ISO 3166-1 alpha-2).
    pub fn issuing_country(&self) -> Result<Option<&'a str>, MdlError> {
        self.text(mdl::ISSUING_COUNTRY)
    }
    /// `issuing_authority`.
    pub fn issuing_authority(&self) -> Result<Option<&'a str>, MdlError> {
        self.text(mdl::ISSUING_AUTHORITY)
    }
    /// `document_number`.
    pub fn document_number(&self) -> Result<Option<&'a str>, MdlError> {
        self.text(mdl::DOCUMENT_NUMBER)
    }
    /// `portrait` — JPEG-2000 (or JPEG) encoded portrait bytes.
    pub fn portrait(&self) -> Result<Option<&'a [u8]>, MdlError> {
        self.bytes(mdl::PORTRAIT)
    }
    /// `driving_privileges` raw CBOR array — typed parsing left to the
    /// caller because privilege records are jurisdiction-specific.
    pub fn driving_privileges(&self) -> Option<&'a CborValue> {
        self.inner.get(mdl::DRIVING_PRIVILEGES)
    }
    /// `un_distinguishing_sign`.
    pub fn un_distinguishing_sign(&self) -> Result<Option<&'a str>, MdlError> {
        self.text(mdl::UN_DISTINGUISHING_SIGN)
    }
    /// `sex` as ISO/IEC 5218 integer.
    pub fn sex(&self) -> Result<Option<i128>, MdlError> {
        self.integer(mdl::SEX)
    }
    /// `height` in cm.
    pub fn height(&self) -> Result<Option<i128>, MdlError> {
        self.integer(mdl::HEIGHT)
    }
    /// `age_in_years`.
    pub fn age_in_years(&self) -> Result<Option<i128>, MdlError> {
        self.integer(mdl::AGE_IN_YEARS)
    }
    /// `age_birth_year`.
    pub fn age_birth_year(&self) -> Result<Option<i128>, MdlError> {
        self.integer(mdl::AGE_BIRTH_YEAR)
    }
    /// `age_over_NN` — true if the holder's age is over `n`.
    pub fn age_over(&self, n: u8) -> Result<Option<bool>, MdlError> {
        let key = mdl::age_over(n);
        self.bool(&key)
    }
    /// `nationality`.
    pub fn nationality(&self) -> Result<Option<&'a str>, MdlError> {
        self.text(mdl::NATIONALITY)
    }
    /// `resident_address`.
    pub fn resident_address(&self) -> Result<Option<&'a str>, MdlError> {
        self.text(mdl::RESIDENT_ADDRESS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> BTreeMap<String, CborValue> {
        let mut m = BTreeMap::new();
        m.insert(mdl::FAMILY_NAME.into(), CborValue::Text("Doe".into()));
        m.insert(mdl::GIVEN_NAME.into(), CborValue::Text("Jane".into()));
        m.insert(mdl::BIRTH_DATE.into(), CborValue::Text("1990-04-12".into()));
        m.insert(mdl::SEX.into(), CborValue::Integer(2.into()));
        m.insert(mdl::age_over(21), CborValue::Bool(true));
        m.insert(mdl::PORTRAIT.into(), CborValue::Bytes(vec![1, 2, 3, 4]));
        m
    }

    #[test]
    fn typed_accessors_round_trip() {
        let map = fixture();
        let c = MdlClaims::new(&map);
        assert_eq!(c.family_name().unwrap(), Some("Doe"));
        assert_eq!(c.given_name().unwrap(), Some("Jane"));
        assert_eq!(
            c.birth_date().unwrap(),
            Some(NaiveDate::from_ymd_opt(1990, 4, 12).unwrap())
        );
        assert_eq!(c.sex().unwrap(), Some(2));
        assert_eq!(c.age_over(21).unwrap(), Some(true));
        assert_eq!(c.age_over(65).unwrap(), None);
        assert_eq!(c.portrait().unwrap(), Some(&[1, 2, 3, 4][..]));
    }

    #[test]
    fn typed_accessor_type_mismatch() {
        let mut m = BTreeMap::new();
        m.insert(mdl::FAMILY_NAME.into(), CborValue::Integer(7.into()));
        let c = MdlClaims::new(&m);
        assert!(c.family_name().is_err());
    }

    #[test]
    fn age_over_format_pads() {
        assert_eq!(mdl::age_over(18), "age_over_18");
        assert_eq!(mdl::age_over(8), "age_over_08");
    }
}
