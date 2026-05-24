//! Shared decoder for the four neural-audio capabilities:
//! `wai.neural.{encodec32, dac, mimi, wavtokenizer}`.
//!
//! Wire format (uniform across all four):
//!   <III>  q  t  n_samples         u32 little-endian × 3
//!   q*t × u16 codes (row-major, LE)
//!
//! ONNX decoder contract (also uniform — guaranteed by the exporters):
//!   inputs:  audio_codes  int64 [1, 1, q, t]
//!            audio_scales float [1]
//!   output:  audio_values float [1, 1, samples]

use std::path::Path;

use ndarray::{Array1, Array4, IxDyn};
use ort::value::Value;

use super::{runtime::load_session, DecodeError};

/// Decoded audio: float samples in [-1, 1], mono.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub(crate) fn decode(payload: &[u8], model_path: &Path, sample_rate: u32)
    -> Result<DecodedAudio, DecodeError>
{
    if payload.len() < 12 {
        return Err(DecodeError::InvalidPayload(
            format!("payload too short ({} B); need ≥ 12 B header", payload.len())));
    }
    let q = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
    let n_samples = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
    let codes_bytes = 2 * q * t;
    if payload.len() != 12 + codes_bytes {
        return Err(DecodeError::InvalidPayload(format!(
            "payload length {} != header q={q} t={t} (expected {})",
            payload.len(), 12 + codes_bytes)));
    }

    // Parse u16 codes → i64 ndarray [1, 1, q, t].
    let mut codes = Array4::<i64>::zeros((1, 1, q, t));
    {
        let slice = codes.as_slice_mut().unwrap();
        let raw = &payload[12..12 + codes_bytes];
        for i in 0..(q * t) {
            slice[i] = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]) as i64;
        }
    }
    let scales = Array1::<f32>::from_elem(1, 1.0_f32);

    let mut sess = load_session(model_path)?;
    let codes_v  = Value::from_array(codes)
        .map_err(|e| DecodeError::Ort(format!("codes tensor: {e}")))?;
    let scales_v = Value::from_array(scales)
        .map_err(|e| DecodeError::Ort(format!("scales tensor: {e}")))?;
    let outputs = sess.run(ort::inputs![
        "audio_codes"  => codes_v,
        "audio_scales" => scales_v,
    ]).map_err(|e| DecodeError::Ort(format!("session.run: {e}")))?;

    let out = outputs.get("audio_values").ok_or_else(||
        DecodeError::Ort("output 'audio_values' missing".into()))?;
    let view = out.try_extract_array::<f32>()
        .map_err(|e| DecodeError::Ort(format!("extract audio_values: {e}")))?;
    let _shape: &[usize] = view.shape();
    let view: ndarray::ArrayViewD<f32> = view.view().into_dyn().into_dimensionality::<IxDyn>()
        .map_err(|e| DecodeError::Ort(format!("reshape audio_values: {e}")))?;
    // Output is [1, 1, samples]; flatten to Vec<f32>, trim to n_samples.
    let mut samples: Vec<f32> = view.iter().copied().collect();
    if samples.len() > n_samples { samples.truncate(n_samples); }
    Ok(DecodedAudio { samples, sample_rate })
}
