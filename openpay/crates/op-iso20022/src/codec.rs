//! XML codec for ISO 20022 documents.
//!
//! ISO 20022 messages travel as XML on most rails (`FedNow` uses XML over
//! ICOM; PIX uses XML over the SPI mTLS connection; SEPA Instant runs
//! XML over EBICS / EPC). We use `quick-xml` for performance and
//! `serde`-based round-trip via the upstream crate's annotations.
//!
//! ## Round-trip contract
//!
//! For every supported message kind, the following must hold:
//!
//! ```text
//! canonical_xml == to_xml(from_xml(canonical_xml)?)?
//! ```
//!
//! This is the conformance test we run against every sample in the
//! `vectors/` directory. A failure means we have either misunderstood
//! the schema or the upstream crate has a regression.

use quick_xml::de::from_str as xml_from_str;
use quick_xml::se::to_string as xml_to_string;
use serde::{Serialize, de::DeserializeOwned};

use crate::error::{Error, Result};

/// Decode an ISO 20022 XML document into a typed value.
///
/// `T` is one of the upstream message types
/// (e.g. `FIToFICustomerCreditTransferV08`), selected by the caller
/// based on the message kind they expect.
///
/// # Errors
/// Returns `Error::XmlDecode` if the XML cannot be parsed or does not
/// match the expected shape.
pub fn from_xml<T: DeserializeOwned>(xml: &str) -> Result<T> {
    xml_from_str(xml).map_err(|e| Error::XmlDecode(e.to_string()))
}

/// Encode a typed message back into ISO 20022 XML.
///
/// The output has no XML declaration; callers prepending it themselves
/// is convention because some rails require specific declaration
/// attributes (encoding, standalone) that vary.
///
/// # Errors
/// Returns `Error::XmlEncode` if serialization fails (e.g. an unknown
/// enum variant that the upstream serde derive can't represent).
pub fn to_xml<T: Serialize>(message: &T) -> Result<String> {
    xml_to_string(message).map_err(|e| Error::XmlEncode(e.to_string()))
}

/// Convenience: round-trip a message and assert the canonical form is
/// stable. Used by conformance tests.
///
/// Returns the canonical (post-round-trip) XML on success.
///
/// # Errors
/// Any codec failure, or `RoundTripMismatch` if a second round-trip
/// produces different output (indicates non-determinism).
pub fn round_trip_canonical<T>(xml: &str) -> Result<String>
where
    T: Serialize + DeserializeOwned,
{
    let first: T = from_xml(xml)?;
    let canonical = to_xml(&first)?;
    let second: T = from_xml(&canonical)?;
    let second_xml = to_xml(&second)?;
    if canonical != second_xml {
        return Err(Error::RoundTripMismatch);
    }
    Ok(canonical)
}
