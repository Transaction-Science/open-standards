//! End-to-end SSE encode → parse round-trip.

use eoc_streaming::sse::{SseEvent, SseParser, parse_all};

#[test]
fn encode_then_parse_recovers_event() {
    let original = SseEvent {
        event: "content_block_delta".into(),
        data: r#"{"index":0,"delta":{"type":"text_delta","text":"hi"}}"#.into(),
        id: Some("evt-7".into()),
        retry_ms: Some(1500),
    };
    let wire = original.encode();
    let parsed = parse_all(&wire).expect("parse");
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0], original);
}

#[test]
fn incremental_feed_preserves_event_boundaries() {
    let wire =
        "event: a\ndata: 1\n\nevent: b\ndata: 2\ndata: 3\n\n";
    let mut p = SseParser::new();
    for chunk in wire.as_bytes().chunks(3) {
        let s = std::str::from_utf8(chunk).expect("ascii");
        p.feed(s).expect("feed");
    }
    let evs = p.drain();
    assert_eq!(evs.len(), 2);
    assert_eq!(evs[0].event, "a");
    assert_eq!(evs[0].data, "1");
    assert_eq!(evs[1].event, "b");
    assert_eq!(evs[1].data, "2\n3");
}

#[test]
fn default_event_type_is_message() {
    let wire = "data: hello\n\n";
    let evs = parse_all(wire).unwrap();
    assert_eq!(evs[0].event, "message");
}
