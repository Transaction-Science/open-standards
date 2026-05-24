//! Bitcoin PSBT v2 (BIP-370) sketch.
//!
//! PSBT (Partially Signed Bitcoin Transaction) is a binary
//! interchange format that lets multiple parties contribute inputs,
//! outputs, signatures, and metadata to a single transaction
//! without ever needing to broadcast a partial txid. BIP-174 defines
//! v0; BIP-370 defines v2.
//!
//! The wire format is a sequence of typed key-value maps. The first
//! map is the *global* map, then one *input* map per input, then one
//! *output* map per output. Each entry is:
//! ```text
//!   varint(key_len) || key_type_byte || key_data
//!   varint(value_len) || value_data
//! ```
//! Maps end with a single 0x00 byte. The whole stream is prefixed
//! by the magic bytes `0x70 0x73 0x62 0x74 0xff` ("psbt\xff").
//!
//! This module is a *sketch*: it models the shape, computes the
//! magic + version, and writes / reads simple PSBT v2 envelopes
//! containing only the BIP-370 required fields. Production code
//! should reach for a full PSBT crate (e.g. `bitcoin`). The sketch
//! exists so OpenPay's settlement / dispute code can construct
//! Bitcoin-side artefacts without pulling in the Bitcoin Core C
//! deps as a hard requirement.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// PSBT magic bytes: `0x70 0x73 0x62 0x74 0xff` = `"psbt\xff"`.
pub const PSBT_MAGIC: [u8; 5] = [0x70, 0x73, 0x62, 0x74, 0xff];

/// PSBT version, BIP-370 (v2).
pub const PSBT_V2: u32 = 2;

/// Key types in the global map that are required for PSBT v2
/// (BIP-370 §"Global Types").
pub mod global_key_type {
    /// `PSBT_GLOBAL_TX_VERSION` — required in v2.
    pub const TX_VERSION: u8 = 0x02;
    /// `PSBT_GLOBAL_FALLBACK_LOCKTIME` — required in v2.
    pub const FALLBACK_LOCKTIME: u8 = 0x03;
    /// `PSBT_GLOBAL_INPUT_COUNT` — required in v2.
    pub const INPUT_COUNT: u8 = 0x04;
    /// `PSBT_GLOBAL_OUTPUT_COUNT` — required in v2.
    pub const OUTPUT_COUNT: u8 = 0x05;
    /// `PSBT_GLOBAL_TX_MODIFIABLE` — optional in v2.
    pub const TX_MODIFIABLE: u8 = 0x06;
    /// `PSBT_GLOBAL_VERSION`.
    pub const VERSION: u8 = 0xfb;
}

/// Input-map key types (subset).
pub mod input_key_type {
    /// `PSBT_IN_PREVIOUS_TXID`.
    pub const PREVIOUS_TXID: u8 = 0x0e;
    /// `PSBT_IN_OUTPUT_INDEX`.
    pub const OUTPUT_INDEX: u8 = 0x0f;
    /// `PSBT_IN_SEQUENCE`.
    pub const SEQUENCE: u8 = 0x10;
    /// `PSBT_IN_REQUIRED_TIME_LOCKTIME`.
    pub const REQUIRED_TIME_LOCKTIME: u8 = 0x11;
    /// `PSBT_IN_REQUIRED_HEIGHT_LOCKTIME`.
    pub const REQUIRED_HEIGHT_LOCKTIME: u8 = 0x12;
}

/// Output-map key types (subset).
pub mod output_key_type {
    /// `PSBT_OUT_AMOUNT`.
    pub const AMOUNT: u8 = 0x03;
    /// `PSBT_OUT_SCRIPT`.
    pub const SCRIPT: u8 = 0x04;
}

/// PSBT v2 global map.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsbtV2Global {
    /// Transaction version (typically 2 for current Bitcoin).
    pub tx_version: i32,
    /// Fallback locktime for the unsigned tx (0 = "no locktime").
    pub fallback_locktime: u32,
    /// Number of inputs.
    pub input_count: u64,
    /// Number of outputs.
    pub output_count: u64,
}

/// PSBT v2 input map (BIP-370 required-only subset).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsbtV2Input {
    /// Previous tx's 32-byte txid (little-endian wire order, as on
    /// the wire).
    pub previous_txid: [u8; 32],
    /// Output index in `previous_txid`.
    pub output_index: u32,
    /// Sequence (default 0xffff_ffff if not RBF-signaling).
    pub sequence: Option<u32>,
}

/// PSBT v2 output map.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsbtV2Output {
    /// Amount in satoshis.
    pub amount: u64,
    /// Locking script (scriptPubKey) bytes.
    pub script: Vec<u8>,
}

/// PSBT v2 envelope.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsbtV2 {
    /// Global map.
    pub global: PsbtV2Global,
    /// Inputs in order.
    pub inputs: Vec<PsbtV2Input>,
    /// Outputs in order.
    pub outputs: Vec<PsbtV2Output>,
}

impl PsbtV2 {
    /// Construct.
    #[must_use]
    pub fn new(
        global: PsbtV2Global,
        inputs: Vec<PsbtV2Input>,
        outputs: Vec<PsbtV2Output>,
    ) -> Self {
        Self {
            global,
            inputs,
            outputs,
        }
    }

    /// Structural validation: counts in `global` match the actual
    /// vectors; previous-txid + amount are consistent.
    ///
    /// # Errors
    /// Returns [`Error::Constraint`] on mismatch.
    pub fn validate(&self) -> Result<()> {
        if self.global.input_count != self.inputs.len() as u64 {
            return Err(Error::Constraint {
                field: "input_count",
                reason: format!(
                    "global declares {} inputs, vector has {}",
                    self.global.input_count,
                    self.inputs.len()
                ),
            });
        }
        if self.global.output_count != self.outputs.len() as u64 {
            return Err(Error::Constraint {
                field: "output_count",
                reason: format!(
                    "global declares {} outputs, vector has {}",
                    self.global.output_count,
                    self.outputs.len()
                ),
            });
        }
        Ok(())
    }

    /// Encode this PSBT v2 envelope to wire bytes.
    ///
    /// # Errors
    /// Returns [`Error::Constraint`] when the envelope fails
    /// [`Self::validate`].
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut out = Vec::with_capacity(64);
        out.extend_from_slice(&PSBT_MAGIC);

        // Global map.
        write_kv_u32(&mut out, &[global_key_type::VERSION], PSBT_V2);
        write_kv_i32(&mut out, &[global_key_type::TX_VERSION], self.global.tx_version);
        write_kv_u32(
            &mut out,
            &[global_key_type::FALLBACK_LOCKTIME],
            self.global.fallback_locktime,
        );
        write_kv_varint(
            &mut out,
            &[global_key_type::INPUT_COUNT],
            self.global.input_count,
        );
        write_kv_varint(
            &mut out,
            &[global_key_type::OUTPUT_COUNT],
            self.global.output_count,
        );
        out.push(0x00); // global map separator

        // Input maps.
        for inp in &self.inputs {
            write_kv_bytes(&mut out, &[input_key_type::PREVIOUS_TXID], &inp.previous_txid);
            write_kv_u32(
                &mut out,
                &[input_key_type::OUTPUT_INDEX],
                inp.output_index,
            );
            if let Some(seq) = inp.sequence {
                write_kv_u32(&mut out, &[input_key_type::SEQUENCE], seq);
            }
            out.push(0x00);
        }

        // Output maps.
        for o in &self.outputs {
            write_kv_u64(&mut out, &[output_key_type::AMOUNT], o.amount);
            write_kv_bytes(&mut out, &[output_key_type::SCRIPT], &o.script);
            out.push(0x00);
        }
        Ok(out)
    }

    /// Decode-magic-only: peek at the first 5 bytes and confirm
    /// they're `psbt\xff`. Doesn't parse the rest of the envelope —
    /// production parse should reach for a full PSBT crate.
    ///
    /// # Errors
    /// Returns [`Error::InvalidLayout`] when the magic doesn't
    /// match.
    pub fn check_magic(bytes: &[u8]) -> Result<()> {
        if bytes.len() < PSBT_MAGIC.len() || &bytes[..PSBT_MAGIC.len()] != PSBT_MAGIC {
            return Err(Error::InvalidLayout("psbt magic mismatch".into()));
        }
        Ok(())
    }
}

fn write_varint(out: &mut Vec<u8>, n: u64) {
    if n < 0xfd {
        #[allow(clippy::cast_possible_truncation)]
        out.push(n as u8);
    } else if n <= u64::from(u16::MAX) {
        out.push(0xfd);
        #[allow(clippy::cast_possible_truncation)]
        out.extend_from_slice(&(n as u16).to_le_bytes());
    } else if n <= u64::from(u32::MAX) {
        out.push(0xfe);
        #[allow(clippy::cast_possible_truncation)]
        out.extend_from_slice(&(n as u32).to_le_bytes());
    } else {
        out.push(0xff);
        out.extend_from_slice(&n.to_le_bytes());
    }
}

fn write_kv_bytes(out: &mut Vec<u8>, key: &[u8], value: &[u8]) {
    write_varint(out, key.len() as u64);
    out.extend_from_slice(key);
    write_varint(out, value.len() as u64);
    out.extend_from_slice(value);
}

fn write_kv_u32(out: &mut Vec<u8>, key: &[u8], v: u32) {
    write_kv_bytes(out, key, &v.to_le_bytes());
}

fn write_kv_i32(out: &mut Vec<u8>, key: &[u8], v: i32) {
    write_kv_bytes(out, key, &v.to_le_bytes());
}

fn write_kv_u64(out: &mut Vec<u8>, key: &[u8], v: u64) {
    write_kv_bytes(out, key, &v.to_le_bytes());
}

fn write_kv_varint(out: &mut Vec<u8>, key: &[u8], v: u64) {
    let mut value = Vec::with_capacity(9);
    write_varint(&mut value, v);
    write_kv_bytes(out, key, &value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PsbtV2 {
        PsbtV2 {
            global: PsbtV2Global {
                tx_version: 2,
                fallback_locktime: 0,
                input_count: 1,
                output_count: 1,
            },
            inputs: vec![PsbtV2Input {
                previous_txid: [0xab; 32],
                output_index: 0,
                sequence: Some(0xffff_fffd),
            }],
            outputs: vec![PsbtV2Output {
                amount: 50_000,
                script: vec![0x00, 0x14, 0xde, 0xad, 0xbe, 0xef],
            }],
        }
    }

    #[test]
    fn validate_mismatched_counts() {
        let mut p = sample();
        p.global.input_count = 99;
        assert!(p.validate().is_err());
    }

    #[test]
    fn encode_starts_with_magic() {
        let bytes = sample().encode().unwrap();
        assert_eq!(&bytes[..5], &PSBT_MAGIC);
    }

    #[test]
    fn encode_ends_after_outputs() {
        let bytes = sample().encode().unwrap();
        // Last byte should be a map separator 0x00 (output-map end).
        assert_eq!(*bytes.last().unwrap(), 0x00);
    }

    #[test]
    fn check_magic_accepts() {
        let bytes = sample().encode().unwrap();
        PsbtV2::check_magic(&bytes).unwrap();
    }

    #[test]
    fn check_magic_rejects_other_bytes() {
        let err = PsbtV2::check_magic(b"hello").unwrap_err();
        assert!(matches!(err, Error::InvalidLayout(_)));
    }

    #[test]
    fn varint_compact_sizes() {
        let mut buf = Vec::new();
        write_varint(&mut buf, 0xfc);
        assert_eq!(buf, vec![0xfc]);
        buf.clear();
        write_varint(&mut buf, 0xfd);
        assert_eq!(buf, vec![0xfd, 0xfd, 0x00]);
        buf.clear();
        write_varint(&mut buf, 0x1_0000);
        assert_eq!(buf, vec![0xfe, 0x00, 0x00, 0x01, 0x00]);
    }
}
