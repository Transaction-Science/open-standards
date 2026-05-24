//! safetensors read sketch.
//!
//! safetensors layout:
//!
//! ```text
//!   u64  header_len (little-endian)
//!   JSON header of length header_len    (utf8, no trailing data)
//!   tensor data (concatenated, offsets named in the header)
//! ```
//!
//! The JSON header is a dict whose keys are tensor names and whose
//! values look like:
//!
//! ```json
//! { "dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64] }
//! ```
//!
//! plus an optional `"__metadata__"` entry. We provide a minimal
//! splitter that returns the header substring and the data offset —
//! enough to drive a tensor walker without pulling in a full JSON
//! parser.

use crate::error::QuantError;

/// Split a safetensors blob into (header_json, payload_start_offset).
///
/// The caller is responsible for parsing the header JSON.
pub fn split_header(bytes: &[u8]) -> Result<(&str, usize), QuantError> {
    if bytes.len() < 8 {
        return Err(QuantError::Truncated {
            needed: 8,
            got: bytes.len(),
        });
    }
    let mut len_buf = [0u8; 8];
    len_buf.copy_from_slice(&bytes[0..8]);
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let end = 8 + header_len;
    if bytes.len() < end {
        return Err(QuantError::Truncated {
            needed: end,
            got: bytes.len(),
        });
    }
    let json = core::str::from_utf8(&bytes[8..end])
        .map_err(|_| QuantError::InvalidFormat("non-utf8 safetensors header"))?;
    Ok((json, end))
}

/// Build a safetensors blob with a given header JSON and tensor
/// payload. Useful for tests and tooling.
pub fn build_blob(header_json: &str, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + header_json.len() + payload.len());
    let len = header_json.len() as u64;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(header_json.as_bytes());
    out.extend_from_slice(payload);
    out
}

/// Cheap key probe: returns true if a top-level JSON object string
/// contains a given tensor name as a key. This is a substring check;
/// for production use, plug in a real JSON parser.
pub fn header_mentions_tensor(header_json: &str, tensor_name: &str) -> bool {
    let needle = format!("\"{tensor_name}\"");
    header_json.contains(&needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_built_blob() {
        let header = r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let payload = vec![0u8; 16];
        let blob = build_blob(header, &payload);
        let (h, off) = split_header(&blob).expect("split");
        assert_eq!(h, header);
        assert_eq!(&blob[off..], payload.as_slice());
    }

    #[test]
    fn rejects_truncated() {
        let bytes = vec![0u8; 3];
        let err = split_header(&bytes);
        assert!(matches!(err, Err(QuantError::Truncated { .. })));
    }

    #[test]
    fn tensor_name_probe() {
        let h = r#"{"layer.weight":{"dtype":"F32"}}"#;
        assert!(header_mentions_tensor(h, "layer.weight"));
        assert!(!header_mentions_tensor(h, "missing"));
    }
}
