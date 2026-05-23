//! Integration tests: round-trip canonical message vectors, dialect MAC
//! reproducibility, and EMV DE 55 parsing.
//!
//! Each fixture under `tests/fixtures/` is a hex-text file with one byte
//! per pair of hex chars. Comments are `# ...` to end of line, blank
//! lines are ignored. This keeps the on-disk artifact human-inspectable
//! (network engineers love being able to grep a fixture file).

use op_iso8583::{
    Dialect, EmvTag, FieldValue, Iso8583Message, Mti, NetworkMgmtCode, VisaBaseI,
    dialect::{AmexGns, DiscoverCard, Jcb, MastercardMds},
    emv, network_mgmt,
};

fn parse_hex_fixture(path: &str) -> Vec<u8> {
    let raw = std::fs::read_to_string(path).expect("fixture readable");
    let mut out = Vec::new();
    for line in raw.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let cleaned: String = line.chars().filter(|c| !c.is_whitespace()).collect();
        let mut chars = cleaned.chars();
        loop {
            match (chars.next(), chars.next()) {
                (Some(a), Some(b)) => {
                    let nib_a = a.to_digit(16).expect("hex digit") as u8;
                    let nib_b = b.to_digit(16).expect("hex digit") as u8;
                    out.push((nib_a << 4) | nib_b);
                }
                (None, _) => break,
                (Some(_), None) => panic!("odd hex length in {path}"),
            }
        }
    }
    out
}

#[test]
fn fixture_auth_request_0100_decodes_and_round_trips() {
    let bytes = parse_hex_fixture("tests/fixtures/visa_0100_auth_request.hex");
    let msg = Iso8583Message::decode(&bytes).expect("decode 0100");
    assert_eq!(msg.mti, Mti::AUTH_REQUEST);
    // Verify well-known fields populated.
    assert!(msg.pan().is_some(), "PAN present");
    assert_eq!(msg.processing_code(), Some("000000"));
    assert_eq!(msg.amount_tx().map(str::len), Some(12));
    assert!(msg.stan().is_some(), "STAN present");
    assert_eq!(msg.response_code(), None); // request has no DE 39
    assert_eq!(msg.currency_tx(), Some("840")); // USD numeric
    // Round-trip: re-encode and confirm bytes match.
    let re = msg.encode().expect("encode 0100");
    assert_eq!(re, bytes, "0100 round trip");
}

#[test]
fn fixture_auth_response_0110_decodes_and_round_trips() {
    let bytes = parse_hex_fixture("tests/fixtures/visa_0110_auth_response.hex");
    let msg = Iso8583Message::decode(&bytes).expect("decode 0110");
    assert_eq!(msg.mti, Mti::AUTH_RESPONSE);
    assert!(msg.mti.is_response());
    assert_eq!(msg.response_code(), Some("00"));
    assert_eq!(msg.approval_code().map(str::len), Some(6));
    assert!(msg.rrn().is_some(), "RRN present");
    // Round-trip.
    let re = msg.encode().expect("encode 0110");
    assert_eq!(re, bytes, "0110 round trip");
}

#[test]
fn fixture_emv_de55_parses_real_tags() {
    let blob = parse_hex_fixture("tests/fixtures/emv_de55_minimal.hex");
    let tlvs = emv::parse_de55(&blob).expect("parse DE 55");
    let map = emv::tlv_map(&tlvs);
    // We expect at least AC, CID, TVR, TX_DATE.
    assert!(map.contains_key(&EmvTag::AC));
    assert!(map.contains_key(&EmvTag::CID));
    assert!(map.contains_key(&EmvTag::TVR));
    assert!(map.contains_key(&EmvTag::TX_DATE));
    assert_eq!(map[&EmvTag::AC].len(), 8);
    // Re-encode and confirm fidelity.
    let re = emv::encode_de55(&tlvs).expect("encode DE 55");
    assert_eq!(re, blob, "DE 55 round trip");
}

#[test]
fn network_mgmt_sign_on_then_response() {
    let req = network_mgmt::sign_on("0523120000", "000001").expect("sign on");
    assert_eq!(req.mti, Mti::NETWORK_REQUEST);
    assert_eq!(
        req.get(70).and_then(FieldValue::as_numeric),
        Some(NetworkMgmtCode::SignOn.as_str3().as_str())
    );
    let resp = network_mgmt::build_response(&req, "00").expect("build response");
    assert_eq!(resp.mti, Mti::NETWORK_RESPONSE);
    assert_eq!(resp.response_code(), Some("00"));
}

// ---- Dialect MAC reproducibility against pinned vectors. The "vector"
// is the hex bytes the reference Feistel MAC must emit for a fixed
// key+message tuple. These are *our* canonical values — regenerated
// here as tests so any future change to the MAC primitive (e.g.
// swapping in a real TDES implementation) becomes a visible bump
// rather than a silent drift. Each dialect gets its own vector.

#[test]
fn visa_base_i_mac_pinned_vector() {
    let d = VisaBaseI;
    let key = [0x01_u8; 16];
    let data = b"0100";
    let mac = d.mac(&key, data).expect("visa mac");
    // Pin the bytes. This is the canonical MAC the reference Feistel
    // implementation emits for this key+data. Any change implies the
    // primitive moved.
    let expected: [u8; 8] = [0x75, 0x75, 0x75, 0x74, 0xea, 0xea, 0xeb, 0xea];
    assert_eq!(mac, expected, "Visa Base I MAC drift");
}

#[test]
fn mastercard_mds_mac_pinned_vector() {
    let d = MastercardMds;
    let key = [0x02_u8; 16];
    let data = b"0200";
    let mac = d.mac(&key, data).expect("mc mac");
    let expected: [u8; 8] = [0x76, 0x76, 0x76, 0x74, 0xea, 0xea, 0xe8, 0xea];
    assert_eq!(mac, expected, "Mastercard MDS MAC drift");
}

#[test]
fn amex_gns_mac_pinned_vector() {
    let d = AmexGns;
    let key = [0x03_u8; 16];
    let data = b"0100AMEX";
    let mac = d.mac(&key, data).expect("amex mac");
    let expected: [u8; 8] = [0x2f, 0x36, 0x3a, 0x33, 0xee, 0xff, 0xef, 0xff];
    assert_eq!(mac, expected, "Amex GNS MAC drift");
}

#[test]
fn discover_aes_mac_pinned_vector() {
    let d = DiscoverCard;
    let key = [0x04_u8; 16];
    let data = b"0100DISC";
    let mac = d.mac(&key, data).expect("disc mac");
    let expected: [u8; 8] = [0x99, 0x9f, 0x93, 0x88, 0xf9, 0xe9, 0xef, 0xe3];
    assert_eq!(mac, expected, "Discover AES MAC drift");
}

#[test]
fn jcb_mac_pinned_vector() {
    let d = Jcb;
    let key = [0x05_u8; 16];
    let data = b"0100JCB";
    let mac = d.mac(&key, data).expect("jcb mac");
    let expected: [u8; 8] = [0x71, 0x3b, 0x32, 0x32, 0xe2, 0xa9, 0xe3, 0xa9];
    assert_eq!(mac, expected, "JCB MAC drift");
}

#[test]
fn dialect_response_codes_diverge() {
    // 82 means "Negative CAM/dCVV/iCVV/CVV" on Mastercard but is *not*
    // a Visa code. This shape-check makes sure dialects are actually
    // separate tables, not aliases.
    assert!(MastercardMds.response_code_meaning("82").is_some());
    assert!(VisaBaseI.response_code_meaning("82").is_none());
    // Amex 109 = "Invalid merchant", a 3-digit Amex variant.
    assert!(AmexGns.response_code_meaning("109").is_some());
    assert!(VisaBaseI.response_code_meaning("109").is_none());
}
