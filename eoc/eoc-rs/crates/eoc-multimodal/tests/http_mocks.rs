//! End-to-end HTTP behaviour tests using `wiremock`.
//!
//! Covers Whisper transcription and OpenAI TTS — the two vendor calls
//! that don't share the chat-completions body schema and therefore need
//! their own fixtures.

use eoc_multimodal::audio::tts::{OpenAiTtsBackend, Synthesizer, VoiceSpec};
use eoc_multimodal::audio::whisper_api::{Transcriber, WhisperApiBackend};
use eoc_multimodal::modality::AudioRef;
use serde_json::json;
use wiremock::matchers::{header, header_exists, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn whisper_known_bytes_returns_expected_text() {
    let server = MockServer::start().await;
    let body = json!({
        "text": "hello world",
        "language": "en",
        "duration": 1.5,
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "hello world"}
        ]
    });
    Mock::given(method("POST"))
        .and(path("/"))
        .and(header("Authorization", "Bearer sk-test"))
        .and(header_exists("Content-Type"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;

    let backend = WhisperApiBackend::new("sk-test", "whisper-1").with_endpoint(server.uri());
    let audio = AudioRef::Bytes {
        content_type: "audio/wav".to_string(),
        bytes: vec![0x00, 0x01, 0x02, 0x03],
    };
    let result = backend.transcribe(&audio).await.expect("ok");
    assert_eq!(result.text, "hello world");
    assert_eq!(result.language.as_deref(), Some("en"));
    assert_eq!(result.segments.len(), 1);
    // 1.5 s * 5 J/s = 7.5 J = 7_500_000 µJ.
    assert_eq!(result.joule_cost.microjoules, 7_500_000);
}

#[tokio::test]
async fn whisper_401_returns_invalid_api_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(401).set_body_string("denied"))
        .mount(&server)
        .await;

    let backend = WhisperApiBackend::new("sk-bad", "whisper-1").with_endpoint(server.uri());
    let audio = AudioRef::Bytes {
        content_type: "audio/wav".to_string(),
        bytes: vec![0x00],
    };
    let err = backend.transcribe(&audio).await.expect_err("should fail");
    assert!(matches!(
        err,
        eoc_multimodal::MultimodalError::InvalidApiKey
    ));
}

#[tokio::test]
async fn tts_returns_audio_bytes_with_content_type() {
    let server = MockServer::start().await;
    let synthetic_wav: Vec<u8> = b"RIFF....WAVEfmt ".to_vec();
    Mock::given(method("POST"))
        .and(path("/"))
        .and(header("Authorization", "Bearer sk-test"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Content-Type", "audio/wav")
                .set_body_bytes(synthetic_wav.clone()),
        )
        .mount(&server)
        .await;

    let backend = OpenAiTtsBackend::new("sk-test", "tts-1").with_endpoint(server.uri());
    let out = backend
        .synthesize("hello world", VoiceSpec::default_alloy())
        .await
        .expect("ok");
    assert_eq!(out.content_type, "audio/wav");
    assert_eq!(out.bytes, synthetic_wav);
    // "hello world" is 11 chars / 150 cps = 0.0733 s * 1.5 J/s ≈ 110_000 µJ.
    assert!(out.joule_cost.microjoules > 0);
    assert!(out.joule_cost.microjoules < 1_000_000);
}
