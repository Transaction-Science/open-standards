//! GGUF v3 header parse test.

use eoc_quantize::gguf::{parse_header, write_header, GgufHeader, GGUF_VERSION_V3};

#[test]
fn parses_synthesized_header() {
    let h = GgufHeader {
        version: GGUF_VERSION_V3,
        tensor_count: 1024,
        metadata_kv_count: 16,
    };
    let bytes = write_header(&h);
    let parsed = parse_header(&bytes).expect("parse header");
    assert_eq!(parsed, h);
}

#[test]
fn rejects_v2_header() {
    let mut bytes = write_header(&GgufHeader {
        version: GGUF_VERSION_V3,
        tensor_count: 0,
        metadata_kv_count: 0,
    });
    // Patch version to 2.
    bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
    assert!(parse_header(&bytes).is_err());
}
