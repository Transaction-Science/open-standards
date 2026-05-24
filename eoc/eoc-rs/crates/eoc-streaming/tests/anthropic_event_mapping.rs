//! Anthropic SSE → normalized event mapping.

use eoc_streaming::anthropic::AnthropicMapper;
use eoc_streaming::sse::SseEvent;
use eoc_streaming::stream::{Event, FinishReason, Role};

fn sse(event: &str, data: &str) -> SseEvent {
    SseEvent {
        event: event.into(),
        data: data.into(),
        id: None,
        retry_ms: None,
    }
}

#[test]
fn full_text_message_lifecycle() {
    let m = AnthropicMapper;
    let frames = vec![
        sse(
            "message_start",
            r#"{"message":{"id":"msg_01","role":"assistant"}}"#,
        ),
        sse(
            "content_block_start",
            r#"{"index":0,"content_block":{"type":"text"}}"#,
        ),
        sse(
            "content_block_delta",
            r#"{"index":0,"delta":{"type":"text_delta","text":"Hel"}}"#,
        ),
        sse(
            "content_block_delta",
            r#"{"index":0,"delta":{"type":"text_delta","text":"lo"}}"#,
        ),
        sse("content_block_stop", r#"{"index":0}"#),
        sse(
            "message_delta",
            r#"{"delta":{"stop_reason":"end_turn"}}"#,
        ),
        sse("message_stop", r#"{}"#),
    ];

    let mut events: Vec<Event> = Vec::new();
    for f in frames.iter() {
        events.extend(m.map(f).expect("map"));
    }

    assert!(matches!(
        events[0],
        Event::MessageStart {
            role: Role::Assistant,
            ..
        }
    ));
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            Event::TextDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hello");
    assert!(events.iter().any(|e| matches!(e, Event::ContentBlockStop { .. })));
    assert!(events.iter().any(|e| matches!(
        e,
        Event::MessageDelta {
            stop_reason: Some(FinishReason::EndTurn)
        }
    )));
    assert!(events.iter().any(|e| matches!(e, Event::MessageStop { .. })));
}

#[test]
fn tool_use_block_emits_start_and_arg_deltas() {
    let m = AnthropicMapper;
    let start = sse(
        "content_block_start",
        r#"{"index":1,"content_block":{"type":"tool_use","id":"tool_01","name":"calc"}}"#,
    );
    let delta = sse(
        "content_block_delta",
        r#"{"index":1,"delta":{"type":"input_json_delta","partial_json":"{\"x\":"}}"#,
    );
    let out: Vec<Event> = m
        .map(&start)
        .unwrap()
        .into_iter()
        .chain(m.map(&delta).unwrap())
        .collect();
    assert!(matches!(
        out[0],
        Event::ToolCallStart { ref name, .. } if name == "calc"
    ));
    assert!(matches!(out[1], Event::ToolCallDelta { .. }));
}

#[test]
fn ping_passes_through() {
    let m = AnthropicMapper;
    let ev = sse("ping", "{}");
    assert_eq!(m.map(&ev).unwrap(), vec![Event::Ping]);
}

#[test]
fn unknown_event_errors() {
    let m = AnthropicMapper;
    let ev = sse("brand_new_event", "{}");
    let err = m.map(&ev).unwrap_err();
    assert!(matches!(
        err,
        eoc_streaming::error::StreamError::UnknownEvent(_)
    ));
}
