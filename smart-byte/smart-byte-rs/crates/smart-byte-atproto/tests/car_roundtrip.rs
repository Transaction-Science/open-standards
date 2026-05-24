//! CAR file encode → decode roundtrip.

use smart_byte_atproto::{CarBlock, CarFile, Cid};

#[test]
fn empty_car_roundtrip() {
    let car = CarFile::new(vec![Cid::dag_cbor(b"root")]);
    let bytes = car.encode().unwrap();
    let back = CarFile::decode(&bytes).unwrap();
    assert_eq!(back.roots.len(), 1);
    assert_eq!(back.roots[0], car.roots[0]);
    assert!(back.blocks.is_empty());
}

#[test]
fn roundtrip_with_blocks() {
    let payload_a = b"alpha".to_vec();
    let payload_b = b"beta-block".to_vec();
    let block_a = CarBlock::dag_cbor(payload_a.clone());
    let block_b = CarBlock::dag_cbor(payload_b.clone());
    let mut car = CarFile::new(vec![block_a.cid.clone()]);
    car.push(block_a.clone());
    car.push(block_b.clone());

    let encoded = car.encode().unwrap();
    let decoded = CarFile::decode(&encoded).unwrap();
    assert_eq!(decoded.blocks.len(), 2);
    assert_eq!(decoded.blocks[0].cid, block_a.cid);
    assert_eq!(decoded.blocks[0].data, payload_a);
    assert_eq!(decoded.blocks[1].cid, block_b.cid);
    assert_eq!(decoded.blocks[1].data, payload_b);
    assert_eq!(decoded.roots, vec![block_a.cid]);
}

#[test]
fn truncated_car_rejected() {
    let block = CarBlock::dag_cbor(b"x".to_vec());
    let mut car = CarFile::new(vec![block.cid.clone()]);
    car.push(block);
    let mut encoded = car.encode().unwrap();
    encoded.truncate(encoded.len() - 4);
    assert!(CarFile::decode(&encoded).is_err());
}
