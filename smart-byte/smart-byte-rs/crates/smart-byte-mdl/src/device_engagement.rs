//! ISO/IEC 18013-5 §8.2 device engagement (QR / NFC).
//!
//! Device engagement is the bootstrap the holder hands to a reader. It
//! carries the holder's ephemeral key (for ECDH key agreement) and the
//! list of transports the holder is willing to speak (BLE, NFC,
//! Wi-Fi Aware). This crate ships:
//!
//! * The CBOR encoder/decoder for the canonical engagement structure.
//! * Transport descriptors as transparent records — actual radio I/O
//!   belongs in a downstream platform-specific crate.
//! * A helper for the QR-code URL form: `mdoc:` followed by the
//!   base64url of the engagement bytes.
//!
//! Wire form (ISO 18013-5 §8.2.1.1.1):
//!
//! ```text
//! DeviceEngagement = {
//!   0: tstr "1.0",                     ; version
//!   1: [ 1, COSE_Key ],                ; security
//!   2: [+ DeviceRetrievalMethod],      ; transports
//!   ? 3: ServerRetrievalMethods,
//!   ? 4: ProtocolInfo,
//! }
//! DeviceRetrievalMethod = [ uint type, uint version, RetrievalOptions ]
//! ```
//!
//! Transport type IDs (ISO 18013-5 §8.2.2):
//! `1` = NFC, `2` = BLE, `3` = Wi-Fi Aware.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ciborium::value::{Integer, Value as CborValue};

use crate::error::MdlError;
use crate::mdoc::{decode_cbor, encode_cbor};

/// Transport descriptor for device retrieval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceRetrievalMethod {
    /// NFC engagement (type 1).
    Nfc {
        /// Max command data length.
        max_len_command: u16,
        /// Max response data length.
        max_len_response: u16,
    },
    /// Bluetooth LE engagement (type 2).
    Ble {
        /// Whether the device acts as the central role.
        supports_central: bool,
        /// Whether the device acts as the peripheral role.
        supports_peripheral: bool,
        /// BLE service UUID for central mode (16 bytes).
        central_uuid: Option<[u8; 16]>,
        /// BLE service UUID for peripheral mode (16 bytes).
        peripheral_uuid: Option<[u8; 16]>,
    },
    /// Wi-Fi Aware engagement (type 3).
    WifiAware {
        /// Passphrase / service name.
        passphrase: Option<String>,
    },
}

impl DeviceRetrievalMethod {
    fn type_id(&self) -> u64 {
        match self {
            DeviceRetrievalMethod::Nfc { .. } => 1,
            DeviceRetrievalMethod::Ble { .. } => 2,
            DeviceRetrievalMethod::WifiAware { .. } => 3,
        }
    }

    fn options(&self) -> CborValue {
        match self {
            DeviceRetrievalMethod::Nfc {
                max_len_command,
                max_len_response,
            } => CborValue::Map(vec![
                (
                    CborValue::Integer(Integer::from(0i64)),
                    CborValue::Integer(Integer::from(*max_len_command)),
                ),
                (
                    CborValue::Integer(Integer::from(1i64)),
                    CborValue::Integer(Integer::from(*max_len_response)),
                ),
            ]),
            DeviceRetrievalMethod::Ble {
                supports_central,
                supports_peripheral,
                central_uuid,
                peripheral_uuid,
            } => {
                let mut m: Vec<(CborValue, CborValue)> = vec![
                    (
                        CborValue::Integer(Integer::from(0i64)),
                        CborValue::Bool(*supports_peripheral),
                    ),
                    (
                        CborValue::Integer(Integer::from(1i64)),
                        CborValue::Bool(*supports_central),
                    ),
                ];
                if let Some(uuid) = peripheral_uuid {
                    m.push((
                        CborValue::Integer(Integer::from(10i64)),
                        CborValue::Bytes(uuid.to_vec()),
                    ));
                }
                if let Some(uuid) = central_uuid {
                    m.push((
                        CborValue::Integer(Integer::from(11i64)),
                        CborValue::Bytes(uuid.to_vec()),
                    ));
                }
                CborValue::Map(m)
            }
            DeviceRetrievalMethod::WifiAware { passphrase } => {
                let mut m: Vec<(CborValue, CborValue)> = Vec::new();
                if let Some(p) = passphrase {
                    m.push((
                        CborValue::Integer(Integer::from(0i64)),
                        CborValue::Text(p.clone()),
                    ));
                }
                CborValue::Map(m)
            }
        }
    }

    fn to_value(&self) -> CborValue {
        CborValue::Array(vec![
            CborValue::Integer(Integer::from(self.type_id())),
            CborValue::Integer(Integer::from(1u64)), // method version
            self.options(),
        ])
    }
}

/// Engagement structure handed to the reader (typically as a QR code).
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceEngagement {
    /// Protocol version string. Always `1.0` for ISO 18013-5:2021.
    pub version: String,
    /// Holder's ephemeral COSE_Key (P-256 EC2). Caller supplies the map.
    pub device_key: CborValue,
    /// Transports the holder offers.
    pub transports: Vec<DeviceRetrievalMethod>,
}

impl DeviceEngagement {
    /// Build a new engagement with version `1.0`.
    pub fn new(device_key: CborValue, transports: Vec<DeviceRetrievalMethod>) -> Self {
        Self {
            version: "1.0".into(),
            device_key,
            transports,
        }
    }

    /// Encode to canonical CBOR.
    pub fn to_cbor(&self) -> Result<Vec<u8>, MdlError> {
        let security = CborValue::Array(vec![
            CborValue::Integer(Integer::from(1u64)),
            self.device_key.clone(),
        ]);
        let transports = CborValue::Array(
            self.transports.iter().map(|t| t.to_value()).collect(),
        );
        let map = CborValue::Map(vec![
            (
                CborValue::Integer(Integer::from(0i64)),
                CborValue::Text(self.version.clone()),
            ),
            (CborValue::Integer(Integer::from(1i64)), security),
            (CborValue::Integer(Integer::from(2i64)), transports),
        ]);
        encode_cbor(&map)
    }

    /// Decode from CBOR bytes. (Lossy: transport options beyond the
    /// fields modeled here are preserved as raw CBOR is dropped.)
    pub fn from_cbor(bytes: &[u8]) -> Result<Self, MdlError> {
        let v: CborValue = decode_cbor(bytes)?;
        let entries = match v {
            CborValue::Map(m) => m,
            _ => {
                return Err(MdlError::Type(
                    "DeviceEngagement not a map".into(),
                ));
            }
        };
        let mut version: Option<String> = None;
        let mut device_key: Option<CborValue> = None;
        let mut transports: Vec<DeviceRetrievalMethod> = Vec::new();
        for (k, val) in entries {
            let n: i128 = match k {
                CborValue::Integer(i) => i.into(),
                _ => continue,
            };
            match n {
                0 => {
                    if let CborValue::Text(s) = val {
                        version = Some(s);
                    }
                }
                1 => {
                    if let CborValue::Array(arr) = val
                        && arr.len() == 2
                        && let Some(second) = arr.into_iter().nth(1)
                    {
                        device_key = Some(second);
                    }
                }
                2 => {
                    if let CborValue::Array(arr) = val {
                        for item in arr {
                            if let Some(t) = parse_transport(&item) {
                                transports.push(t);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(Self {
            version: version.unwrap_or_else(|| "1.0".into()),
            device_key: device_key
                .ok_or_else(|| MdlError::Missing("deviceEngagement deviceKey".into()))?,
            transports,
        })
    }

    /// Format as the `mdoc:` URL the reader scans from QR.
    pub fn to_qr_url(&self) -> Result<String, MdlError> {
        let bytes = self.to_cbor()?;
        let b64 = URL_SAFE_NO_PAD.encode(bytes);
        Ok(format!("mdoc:{b64}"))
    }

    /// Parse from a `mdoc:` URL.
    pub fn from_qr_url(url: &str) -> Result<Self, MdlError> {
        let payload = url
            .strip_prefix("mdoc:")
            .ok_or_else(|| MdlError::Type("expected mdoc: scheme".into()))?;
        let bytes = URL_SAFE_NO_PAD.decode(payload)?;
        Self::from_cbor(&bytes)
    }
}

fn parse_transport(v: &CborValue) -> Option<DeviceRetrievalMethod> {
    let arr = match v {
        CborValue::Array(a) => a,
        _ => return None,
    };
    if arr.len() != 3 {
        return None;
    }
    let type_id: i128 = match &arr[0] {
        CborValue::Integer(i) => (*i).into(),
        _ => return None,
    };
    let opts = match &arr[2] {
        CborValue::Map(m) => m,
        _ => return None,
    };
    match type_id {
        1 => {
            let mut cmd: u16 = 0;
            let mut resp: u16 = 0;
            for (k, val) in opts {
                if let (CborValue::Integer(ki), CborValue::Integer(vi)) = (k, val) {
                    let kn: i128 = (*ki).into();
                    let vn: i128 = (*vi).into();
                    match kn {
                        0 => cmd = vn as u16,
                        1 => resp = vn as u16,
                        _ => {}
                    }
                }
            }
            Some(DeviceRetrievalMethod::Nfc {
                max_len_command: cmd,
                max_len_response: resp,
            })
        }
        2 => {
            let mut supports_peripheral = false;
            let mut supports_central = false;
            let mut peripheral_uuid: Option<[u8; 16]> = None;
            let mut central_uuid: Option<[u8; 16]> = None;
            for (k, val) in opts {
                let kn: i128 = match k {
                    CborValue::Integer(i) => (*i).into(),
                    _ => continue,
                };
                match (kn, val) {
                    (0, CborValue::Bool(b)) => supports_peripheral = *b,
                    (1, CborValue::Bool(b)) => supports_central = *b,
                    (10, CborValue::Bytes(b)) if b.len() == 16 => {
                        let mut u = [0u8; 16];
                        u.copy_from_slice(b);
                        peripheral_uuid = Some(u);
                    }
                    (11, CborValue::Bytes(b)) if b.len() == 16 => {
                        let mut u = [0u8; 16];
                        u.copy_from_slice(b);
                        central_uuid = Some(u);
                    }
                    _ => {}
                }
            }
            Some(DeviceRetrievalMethod::Ble {
                supports_central,
                supports_peripheral,
                central_uuid,
                peripheral_uuid,
            })
        }
        3 => {
            let mut pass: Option<String> = None;
            for (k, val) in opts {
                if let (CborValue::Integer(ki), CborValue::Text(t)) = (k, val) {
                    let kn: i128 = (*ki).into();
                    if kn == 0 {
                        pass = Some(t.clone());
                    }
                }
            }
            Some(DeviceRetrievalMethod::WifiAware { passphrase: pass })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::issuer::IssuerKey;

    #[test]
    fn engagement_round_trip() {
        let key = IssuerKey::generate_es256();
        let eng = DeviceEngagement::new(
            key.cose_public_key(),
            vec![
                DeviceRetrievalMethod::Ble {
                    supports_central: false,
                    supports_peripheral: true,
                    central_uuid: None,
                    peripheral_uuid: Some([0xAA; 16]),
                },
                DeviceRetrievalMethod::Nfc {
                    max_len_command: 256,
                    max_len_response: 1024,
                },
            ],
        );
        let bytes = eng.to_cbor().unwrap();
        let back = DeviceEngagement::from_cbor(&bytes).unwrap();
        assert_eq!(back.transports.len(), 2);
        assert_eq!(back.version, "1.0");
    }

    #[test]
    fn qr_url_round_trip() {
        let key = IssuerKey::generate_es256();
        let eng = DeviceEngagement::new(
            key.cose_public_key(),
            vec![DeviceRetrievalMethod::Nfc {
                max_len_command: 256,
                max_len_response: 1024,
            }],
        );
        let url = eng.to_qr_url().unwrap();
        assert!(url.starts_with("mdoc:"));
        let back = DeviceEngagement::from_qr_url(&url).unwrap();
        assert_eq!(back.transports.len(), 1);
    }
}
