//! `Iso8583Message`: the typed in-memory representation of an ISO 8583
//! message frame plus its encode/decode entry points.
//!
//! On the wire, every message is:
//!
//! ```text
//! [ MTI (4 digits BCD) ][ primary bitmap (8 bytes) ][ secondary bitmap (8 bytes if needed) ]
//! [ each present data element, in ascending DE order, in its own encoding ]
//! ```
//!
//! Some links prepend a 2-byte length header (the "TPDU length" used by
//! TCP socket framing). That is a transport concern, not a message
//! concern — `Iso8583Message::encode` / `decode` operate on the message
//! bytes only. Wire framers in `op-rails-card` add the length prefix.

use core::fmt;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bitmap::Bitmaps;
use crate::codec::{bcd_decode, bcd_encode};
use crate::error::{Error, Result};
use crate::fields::{
    FieldSpec, FieldValue, decode_field, default_catalog, encode_field, lookup,
};

/// Message Type Indicator. Four digits identifying the ISO 8583 message
/// family and direction.
///
/// Common values:
///
/// | MTI    | Meaning                                  |
/// |--------|------------------------------------------|
/// | `0100` | Authorization request                    |
/// | `0110` | Authorization response                   |
/// | `0200` | Financial transaction request            |
/// | `0210` | Financial transaction response           |
/// | `0220` | Financial transaction advice             |
/// | `0230` | Financial transaction advice response    |
/// | `0400` | Reversal request                         |
/// | `0410` | Reversal response                        |
/// | `0420` | Reversal advice                          |
/// | `0800` | Network management request (sign-on / echo / key change) |
/// | `0810` | Network management response              |
///
/// Note: ISO 8583:2003 uses a `1xxx` prefix instead of `0xxx`; we
/// accept any four-digit value and let the dialect interpret it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Mti(pub u16);

impl Mti {
    /// Authorization request (`0100`).
    pub const AUTH_REQUEST: Self = Self(0x0100);
    /// Authorization response (`0110`).
    pub const AUTH_RESPONSE: Self = Self(0x0110);
    /// Financial transaction request (`0200`).
    pub const FINANCIAL_REQUEST: Self = Self(0x0200);
    /// Financial transaction response (`0210`).
    pub const FINANCIAL_RESPONSE: Self = Self(0x0210);
    /// Financial advice (`0220`).
    pub const FINANCIAL_ADVICE: Self = Self(0x0220);
    /// Reversal advice (`0420`).
    pub const REVERSAL_ADVICE: Self = Self(0x0420);
    /// Network-management request (`0800`).
    pub const NETWORK_REQUEST: Self = Self(0x0800);
    /// Network-management response (`0810`).
    pub const NETWORK_RESPONSE: Self = Self(0x0810);

    /// Render the MTI as a 4-digit string (e.g. `0100`).
    #[must_use]
    pub fn as_str4(self) -> String {
        format!("{:04x}", self.0)
    }

    /// True iff this is a response message (last digit is `0` and the
    /// second-from-last is one of `1, 3, 5, 7, 9` per ISO 8583).
    #[must_use]
    pub const fn is_response(self) -> bool {
        // ISO 8583: position 3 (0=request, 1=request response, 2=advice,
        //                       3=advice response, 4=notification, 5=notification ack, ...)
        let position3 = (self.0 >> 4) & 0x0F;
        matches!(position3, 1 | 3 | 5 | 7 | 9)
    }
}

impl fmt::Display for Mti {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04x}", self.0)
    }
}

/// One ISO 8583 message: MTI + bitmaps + ordered map of present fields.
///
/// `fields` is a `BTreeMap` so iteration is in ascending DE order, which
/// matches the on-the-wire layout exactly.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Iso8583Message {
    /// Message type indicator.
    pub mti: Mti,
    /// Primary + (optionally) secondary bitmap.
    pub bitmaps: Bitmaps,
    /// Decoded data elements keyed by DE number (1..=128).
    pub fields: BTreeMap<u8, FieldValue>,
}

impl Iso8583Message {
    /// Create a new empty message with the given MTI.
    #[must_use]
    pub fn new(mti: Mti) -> Self {
        Self {
            mti,
            bitmaps: Bitmaps::new(),
            fields: BTreeMap::new(),
        }
    }

    /// Set a data element. Updates the bitmap automatically.
    pub fn set(&mut self, de: u8, value: FieldValue) {
        self.bitmaps.set(de);
        self.fields.insert(de, value);
    }

    /// Remove a data element. Updates the bitmap automatically.
    pub fn unset(&mut self, de: u8) {
        self.bitmaps.clear(de);
        self.fields.remove(&de);
    }

    /// Get a data element by number.
    #[must_use]
    pub fn get(&self, de: u8) -> Option<&FieldValue> {
        self.fields.get(&de)
    }

    // ----------- Typed accessors for well-known DEs -----------

    /// DE 2 — Primary Account Number (PAN) digit string.
    #[must_use]
    pub fn pan(&self) -> Option<&str> {
        self.get(2).and_then(FieldValue::as_numeric)
    }

    /// DE 3 — six-digit processing code.
    #[must_use]
    pub fn processing_code(&self) -> Option<&str> {
        self.get(3).and_then(FieldValue::as_numeric)
    }

    /// DE 4 — transaction amount, in the currency's minor units, as a
    /// 12-digit string.
    #[must_use]
    pub fn amount_tx(&self) -> Option<&str> {
        self.get(4).and_then(FieldValue::as_numeric)
    }

    /// DE 11 — Systems Trace Audit Number (STAN), 6 digits.
    #[must_use]
    pub fn stan(&self) -> Option<&str> {
        self.get(11).and_then(FieldValue::as_numeric)
    }

    /// DE 37 — 12-character Retrieval Reference Number.
    #[must_use]
    pub fn rrn(&self) -> Option<&str> {
        self.get(37).and_then(FieldValue::as_text)
    }

    /// DE 38 — 6-character approval (authorization) code.
    #[must_use]
    pub fn approval_code(&self) -> Option<&str> {
        self.get(38).and_then(FieldValue::as_text)
    }

    /// DE 39 — 2-character response code (`"00"` = approved on most networks).
    #[must_use]
    pub fn response_code(&self) -> Option<&str> {
        self.get(39).and_then(FieldValue::as_text)
    }

    /// DE 41 — 8-character terminal ID.
    #[must_use]
    pub fn terminal_id(&self) -> Option<&str> {
        self.get(41).and_then(FieldValue::as_text)
    }

    /// DE 42 — 15-character merchant (card acceptor) ID.
    #[must_use]
    pub fn merchant_id(&self) -> Option<&str> {
        self.get(42).and_then(FieldValue::as_text)
    }

    /// DE 49 — 3-digit ISO 4217 numeric currency code.
    #[must_use]
    pub fn currency_tx(&self) -> Option<&str> {
        self.get(49).and_then(FieldValue::as_numeric)
    }

    /// DE 55 — ICC / EMV TLV blob.
    #[must_use]
    pub fn emv_blob(&self) -> Option<&[u8]> {
        self.get(55).and_then(FieldValue::as_bytes)
    }

    /// DE 64 — primary MAC (8 bytes).
    #[must_use]
    pub fn mac_primary(&self) -> Option<&[u8]> {
        self.get(64).and_then(FieldValue::as_bytes)
    }

    /// DE 90 — original data elements (for reversal/chargeback). Reads
    /// out the original MTI, STAN, transmission date/time, acquirer ID,
    /// forwarding institution ID.
    #[must_use]
    pub fn original_data_elements(&self) -> Option<&str> {
        self.get(90).and_then(FieldValue::as_numeric)
    }

    // ----------- Encode / decode -----------

    /// Encode this message into ISO 8583 wire bytes using the default
    /// catalog.
    ///
    /// # Errors
    /// Any [`Error`] from the underlying codec — typically
    /// [`Error::InvalidDataElement`] if a field value violates its
    /// encoding constraints.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let catalog = default_catalog();
        self.encode_with_catalog(&catalog)
    }

    /// Encode using a caller-supplied catalog (used by dialects that
    /// override specific field encodings).
    ///
    /// # Errors
    /// As [`Self::encode`].
    pub fn encode_with_catalog(&self, catalog: &[Option<FieldSpec>]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        // MTI as 4 BCD digits.
        let mti_str = format!("{:04x}", self.mti.0);
        out.extend(bcd_encode(&mti_str)?);
        // Refresh bitmap from the keyed fields so the wire representation
        // matches actual presence even if `bitmaps` is stale.
        let mut bm = Bitmaps::new();
        for de in self.fields.keys() {
            bm.set(*de);
        }
        out.extend(bm.encode_binary());
        // Encode each present DE in ascending order. Skip DE 1 (continuation flag).
        for de in bm.iter_des() {
            let value = self
                .fields
                .get(&de)
                .ok_or(Error::MissingDataElement(de))?;
            let spec = lookup(catalog, de)?;
            let bytes = encode_field(&spec, value)?;
            out.extend(bytes);
        }
        Ok(out)
    }

    /// Decode a message from wire bytes using the default catalog.
    ///
    /// # Errors
    /// As [`Self::encode`].
    pub fn decode(input: &[u8]) -> Result<Self> {
        let catalog = default_catalog();
        Self::decode_with_catalog(input, &catalog)
    }

    /// Decode using a caller-supplied catalog.
    ///
    /// # Errors
    /// As [`Self::encode`].
    pub fn decode_with_catalog(
        input: &[u8],
        catalog: &[Option<FieldSpec>],
    ) -> Result<Self> {
        if input.len() < 2 {
            return Err(Error::UnexpectedEof {
                offset: 0,
                needed: 2,
            });
        }
        let (mti_str, mut offset) = bcd_decode(input, 0, 4)?;
        let mti_val = u16::from_str_radix(&mti_str, 16)
            .map_err(|_| Error::InvalidMti(mti_str.clone()))?;
        let mti = Mti(mti_val);
        let (bitmaps, new_offset) = Bitmaps::decode_binary(input, offset)?;
        offset = new_offset;
        let mut fields = BTreeMap::new();
        for de in bitmaps.iter_des() {
            let spec = lookup(catalog, de)?;
            let (value, new_offset) = decode_field(&spec, input, offset)?;
            offset = new_offset;
            fields.insert(de, value);
        }
        Ok(Self {
            mti,
            bitmaps,
            fields,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mti_constants_match_iso_8583_table() {
        assert_eq!(Mti::AUTH_REQUEST.0, 0x0100);
        assert_eq!(Mti::AUTH_RESPONSE.0, 0x0110);
        assert_eq!(Mti::FINANCIAL_REQUEST.0, 0x0200);
        assert_eq!(Mti::REVERSAL_ADVICE.0, 0x0420);
        assert_eq!(Mti::NETWORK_REQUEST.0, 0x0800);
    }

    #[test]
    fn mti_response_classification() {
        assert!(!Mti::AUTH_REQUEST.is_response());
        assert!(Mti::AUTH_RESPONSE.is_response());
        assert!(Mti::FINANCIAL_RESPONSE.is_response());
        assert!(!Mti::REVERSAL_ADVICE.is_response()); // advice is x2xx
    }

    #[test]
    fn empty_message_round_trip() {
        let m = Iso8583Message::new(Mti::AUTH_REQUEST);
        let bytes = m.encode().unwrap();
        // 4 BCD digits (2 bytes) + 8-byte primary bitmap = 10 bytes
        assert_eq!(bytes.len(), 10);
        let back = Iso8583Message::decode(&bytes).unwrap();
        assert_eq!(back.mti, m.mti);
    }
}
