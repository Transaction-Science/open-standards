//! Audio transcription and synthesis.
//!
//! * [`whisper_api`] — vendor transcription endpoints (OpenAI `whisper-1`,
//!   Gemini audio inline).
//! * [`whisper_local`] *(feature `local`)* — ONNX-exported Whisper.
//! * [`mms`] *(feature `local`)* — Meta MMS multilingual ASR.
//! * [`tts`] — text-to-speech: OpenAI `tts-1` / `tts-1-hd`, optional Bark
//!   and MMS-TTS behind `local`.

pub mod mms;
pub mod tts;
pub mod whisper_api;

#[cfg(feature = "local")]
pub mod whisper_local;

pub use tts::{OpenAiTtsBackend, Synthesizer, VoiceSpec};
pub use whisper_api::{Segment, Transcriber, TranscriptionResult, WhisperApiBackend};
