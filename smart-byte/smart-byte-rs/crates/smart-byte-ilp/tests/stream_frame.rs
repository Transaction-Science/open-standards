//! STREAM frame multiplex tests.

use smart_byte_ilp::{Frame, FrameType, Result, StreamPacket};

#[test]
fn multiplexed_money_plus_data_round_trip() -> Result<()> {
    let pkt = StreamPacket {
        version: 1,
        ilp_packet_type: 12,
        sequence: 42,
        prepare_amount: 1_000,
        frames: vec![
            Frame::stream_money(1, 500),
            Frame::stream_money(2, 500),
            Frame::stream_data(1, 0, b"first chunk"),
            Frame::stream_data(2, 0, b"second chunk"),
        ],
    };
    let wire = pkt.encode();
    let back = StreamPacket::decode(&wire)?;
    assert_eq!(pkt, back);
    assert_eq!(back.frames.len(), 4);
    assert_eq!(back.frames[0].frame_type, FrameType::StreamMoney);
    assert_eq!(back.frames[3].frame_type, FrameType::StreamData);
    Ok(())
}

#[test]
fn empty_frame_list_decodes() -> Result<()> {
    let pkt = StreamPacket {
        version: 1,
        ilp_packet_type: 13,
        sequence: 0,
        prepare_amount: 0,
        frames: vec![],
    };
    let wire = pkt.encode();
    let back = StreamPacket::decode(&wire)?;
    assert_eq!(back.frames.len(), 0);
    Ok(())
}

#[test]
fn unknown_frame_type_errors() {
    // Build a one-frame packet with a forbidden frame-type byte.
    let mut wire = vec![1u8, 12]; // version + ilp packet type
    wire.extend_from_slice(&0u64.to_be_bytes()); // sequence
    wire.extend_from_slice(&0u64.to_be_bytes()); // prepare amount
    wire.push(1); // n_frames == 1 (short form)
    wire.push(0xff); // frame_type byte
    wire.push(0); // empty payload
    assert!(StreamPacket::decode(&wire).is_err());
}
