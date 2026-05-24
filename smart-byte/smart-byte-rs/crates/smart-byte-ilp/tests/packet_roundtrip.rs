//! Encode every ILPv4 packet kind, decode it back, and verify the wire
//! shape matches what we expect.

use smart_byte_ilp::{
    Address, Condition, Fulfill, Fulfillment, IlpPacket, Prepare, Reject, RejectCode, Result,
};

#[test]
fn prepare_wire_starts_with_type_12() -> Result<()> {
    let f = Fulfillment::new([5u8; 32]);
    let pkt = IlpPacket::Prepare(Prepare {
        amount: 12_345,
        expires_at: *b"20260524180000000",
        condition: f.condition(),
        destination: Address::parse("g.us.bank.alice")?,
        data: b"payload".to_vec(),
    });
    let wire = pkt.encode();
    assert_eq!(wire[0], 12);
    let back = IlpPacket::decode(&wire)?;
    assert_eq!(pkt, back);
    Ok(())
}

#[test]
fn fulfill_wire_starts_with_type_13() -> Result<()> {
    let pkt = IlpPacket::Fulfill(Fulfill {
        fulfillment: Fulfillment::new([7u8; 32]),
        data: vec![],
    });
    let wire = pkt.encode();
    assert_eq!(wire[0], 13);
    let back = IlpPacket::decode(&wire)?;
    assert_eq!(pkt, back);
    Ok(())
}

#[test]
fn reject_wire_starts_with_type_14() -> Result<()> {
    let pkt = IlpPacket::Reject(Reject {
        code: RejectCode::T04InsufficientLiquidity,
        triggered_by: "g.us.bank".into(),
        message: "out of pennies".into(),
        data: vec![0xde, 0xad, 0xbe, 0xef],
    });
    let wire = pkt.encode();
    assert_eq!(wire[0], 14);
    let back = IlpPacket::decode(&wire)?;
    assert_eq!(pkt, back);
    Ok(())
}

#[test]
fn large_prepare_uses_long_form_length() -> Result<()> {
    let f = Fulfillment::new([0u8; 32]);
    let pkt = IlpPacket::Prepare(Prepare {
        amount: 1,
        expires_at: *b"20260524180000000",
        condition: f.condition(),
        destination: Address::parse("g.us.bank")?,
        data: vec![0xa5; 300],
    });
    let wire = pkt.encode();
    assert_eq!(wire[0], 12);
    // Long-form length-determinant: high bit set on the second byte.
    assert!(wire[1] >= 0x80);
    let back = IlpPacket::decode(&wire)?;
    assert_eq!(pkt, back);
    Ok(())
}

#[test]
fn truncated_input_errors() {
    let f = Fulfillment::new([1u8; 32]);
    let pkt = IlpPacket::Prepare(Prepare {
        amount: 1,
        expires_at: *b"20260524180000000",
        condition: f.condition(),
        destination: Address::parse("g.us.bank").unwrap(),
        data: vec![],
    });
    let wire = pkt.encode();
    let truncated = &wire[..wire.len() - 5];
    assert!(IlpPacket::decode(truncated).is_err());
}

#[test]
fn condition_round_trips_through_prepare() -> Result<()> {
    let secret = Fulfillment::new([0xaa; 32]);
    let cond: Condition = secret.condition();
    let pkt = IlpPacket::Prepare(Prepare {
        amount: 1,
        expires_at: *b"20260524180000000",
        condition: cond,
        destination: Address::parse("g.us.bank.alice")?,
        data: vec![],
    });
    let wire = pkt.encode();
    let back = IlpPacket::decode(&wire)?;
    match back {
        IlpPacket::Prepare(p) => assert_eq!(p.condition, cond),
        _ => panic!("not a prepare"),
    }
    Ok(())
}
