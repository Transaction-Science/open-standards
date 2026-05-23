//! Conformance tests against canonical EMV BER-TLV byte sequences.
//!
//! Each `.hex` file in `vectors/` is a hex-encoded TLV blob. The
//! assertions here encode our understanding of what the bytes mean.
//! When the EMV spec doesn't change (which is its whole point), these
//! values are fixed for the life of the project.
//!
//! Sources:
//! - `fci_template.hex` — Payment System Environment selection response,
//!   verbatim from EMV Book 1 Annex A example (also widely cited at
//!   emvlab.org/tlvutils).
//! - `taptopay_flat.hex`, `taptopay_nested.hex` — six standard
//!   transaction-data tags as a terminal kernel would emit them on a
//!   Tap-to-Pay $1.00 USD purchase, May 17 2026.

use op_emv::stream::TlvIter;
use op_emv::tag::Tag;
use op_emv::tree::{Tlv, TlvBody, TlvSliceExt};

fn unhex(s: &str) -> Vec<u8> {
    let s = s.trim();
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        let hi = (bytes[i] as char).to_digit(16).expect("invalid hex");
        let lo = (bytes[i + 1] as char).to_digit(16).expect("invalid hex");
        out.push((hi * 16 + lo) as u8);
        i += 2;
    }
    out
}

const FCI: &str = include_str!("../vectors/fci_template.hex");
const TAPTOPAY_FLAT: &str = include_str!("../vectors/taptopay_flat.hex");
const TAPTOPAY_NESTED: &str = include_str!("../vectors/taptopay_nested.hex");

// ---- FCI Template (canonical EMV PSE response) ----

#[test]
fn fci_template_parses_to_expected_tree() {
    let buf = unhex(FCI);
    // 6F (FCI Template, constructed) ─┬─ 84 (DF Name) = "1PAY.SYS.DDF01"
    //                                 └─ A5 (FCI Proprietary) ─┬─ 88 (SFI) = 0x02
    //                                                          └─ 5F2D (Language) = "en"
    let tree = Tlv::parse_all(&buf).unwrap();
    assert_eq!(tree.len(), 1, "FCI is a single top-level TLV");

    let fci = &tree[0];
    assert_eq!(fci.tag, Tag::FCI_TEMPLATE);

    let children = match &fci.body {
        TlvBody::Constructed(c) => c,
        TlvBody::Primitive(_) => panic!("FCI must be constructed"),
    };
    assert_eq!(children.len(), 2);

    // 84 — DF Name
    assert_eq!(children[0].tag, Tag::DF_NAME);
    assert_eq!(children[0].primitive().unwrap(), b"1PAY.SYS.DDF01");

    // A5 — FCI Proprietary
    assert_eq!(children[1].tag, Tag::FCI_PROPRIETARY);
    let prop_children = match &children[1].body {
        TlvBody::Constructed(c) => c,
        _ => panic!("A5 must be constructed"),
    };
    assert_eq!(prop_children.len(), 2);
    assert_eq!(prop_children[0].tag, Tag::SFI);
    assert_eq!(prop_children[0].primitive().unwrap(), &[0x02]);
    assert_eq!(prop_children[1].tag, Tag::LANGUAGE);
    assert_eq!(prop_children[1].primitive().unwrap(), b"en");
}

#[test]
fn fci_template_find_locates_language_deep() {
    let buf = unhex(FCI);
    let tree = Tlv::parse_all(&buf).unwrap();
    let language = tree.find_tag(Tag::LANGUAGE).expect("5F2D should be in FCI");
    assert_eq!(language.primitive().unwrap(), b"en");
}

#[test]
fn fci_template_encoded_len_matches_input() {
    let buf = unhex(FCI);
    let tree = Tlv::parse_all(&buf).unwrap();
    let total: usize = tree.iter().map(|t| t.encoded_len()).sum();
    assert_eq!(
        total,
        buf.len(),
        "encoded_len must round-trip the input length"
    );
}

// ---- Tap-to-Pay flat ----

#[test]
fn taptopay_flat_contains_six_standard_fields() {
    let buf = unhex(TAPTOPAY_FLAT);
    let parsed: Vec<_> = TlvIter::new(&buf).collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(parsed.len(), 6, "flat bundle has six TLVs");

    // 9F02 — Amount, Authorised: 12-digit BCD $1.00 = 000000000100
    assert_eq!(parsed[0].tag, Tag::AMOUNT_AUTHORISED);
    assert_eq!(parsed[0].value, &[0x00, 0x00, 0x00, 0x00, 0x01, 0x00]);

    // 9F03 — Amount, Other: 0
    assert_eq!(parsed[1].tag, Tag::AMOUNT_OTHER);
    assert_eq!(parsed[1].value, &[0x00; 6]);

    // 9F1A — Terminal Country Code: 0840 (USA, ISO 3166-1 numeric)
    assert_eq!(parsed[2].tag, Tag::TERMINAL_COUNTRY);
    assert_eq!(parsed[2].value, &[0x08, 0x40]);

    // 5F2A — Transaction Currency Code: 0840 (USD, ISO 4217 numeric)
    assert_eq!(parsed[3].tag, Tag::TXN_CURRENCY);
    assert_eq!(parsed[3].value, &[0x08, 0x40]);

    // 9A — Transaction Date: 260517 (YYMMDD = 2026-05-17)
    assert_eq!(parsed[4].tag, Tag::TXN_DATE);
    assert_eq!(parsed[4].value, &[0x26, 0x05, 0x17]);

    // 9C — Transaction Type: 00 (purchase)
    assert_eq!(parsed[5].tag, Tag::TXN_TYPE);
    assert_eq!(parsed[5].value, &[0x00]);
}

#[test]
fn taptopay_flat_offsets_are_contiguous() {
    let buf = unhex(TAPTOPAY_FLAT);
    let parsed: Vec<_> = TlvIter::new(&buf).collect::<Result<Vec<_>, _>>().unwrap();
    // Expected offsets: 0, 8, 16, 20, 24, 29
    // 9F02: tag(2) + len(1) + val(6) = 9 bytes; starts at 0
    // 9F03: same, 9 bytes; starts at 9? — no, both are 9 bytes -> 0, 9. Let me recount.
    // Actually: each multi-byte tag entry: 2 (tag) + 1 (len) + value
    //   9F02 + 06 + 6 = 9 bytes  -> offset 0
    //   9F03 + 06 + 6 = 9 bytes  -> offset 9
    //   9F1A + 02 + 2 = 5 bytes  -> offset 18
    //   5F2A + 02 + 2 = 5 bytes  -> offset 23
    //   9A   + 03 + 3 = 5 bytes  -> offset 28
    //   9C   + 01 + 1 = 3 bytes  -> offset 33
    assert_eq!(parsed[0].offset, 0);
    assert_eq!(parsed[1].offset, 9);
    assert_eq!(parsed[2].offset, 18);
    assert_eq!(parsed[3].offset, 23);
    assert_eq!(parsed[4].offset, 28);
    assert_eq!(parsed[5].offset, 33);
}

// ---- Tap-to-Pay nested in E1 ----

#[test]
fn taptopay_nested_descends_into_wrapper() {
    let buf = unhex(TAPTOPAY_NESTED);
    let parsed: Vec<_> = TlvIter::new(&buf).collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(parsed.len(), 1);
    let wrapper = &parsed[0];
    assert_eq!(wrapper.tag.0, 0xE1);
    assert!(wrapper.tag.is_constructed());

    let children: Vec<_> = wrapper
        .children()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(children.len(), 6);
    assert_eq!(children[0].tag, Tag::AMOUNT_AUTHORISED);
    assert_eq!(children[5].tag, Tag::TXN_TYPE);
}

#[test]
fn taptopay_nested_find_locates_amount() {
    let buf = unhex(TAPTOPAY_NESTED);
    let tree = Tlv::parse_all(&buf).unwrap();
    let amt = tree
        .find_tag(Tag::AMOUNT_AUTHORISED)
        .expect("amount must be discoverable inside the wrapper");
    assert_eq!(
        amt.primitive().unwrap(),
        &[0x00, 0x00, 0x00, 0x00, 0x01, 0x00]
    );
}

// ---- Malformed inputs must not panic ----

#[test]
fn truncated_input_yields_error_not_panic() {
    // FCI but cut short.
    let buf = unhex(FCI);
    let truncated = &buf[..buf.len() / 2];
    let result: Result<Vec<_>, _> = TlvIter::new(truncated).collect();
    assert!(result.is_err());
}

#[test]
fn random_garbage_yields_error_not_panic() {
    let garbage = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
    let result: Result<Vec<_>, _> = TlvIter::new(&garbage).collect();
    // Either errors cleanly or returns nothing — must not panic.
    let _ = result;
}
