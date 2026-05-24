//! Decoder for `wai.neural.bmshj2018` (v0.4 wire format).
//!
//! Wire format:
//!   <IIIIBB>  H, W, L, quality, C, S_log2
//!   L bytes   CompressAI rANS bitstream
//!
//! Decode pipeline:
//!   1. Load `cdfs_q<quality>.json` from the same dir as decoder.onnx
//!      (cached per-process).
//!   2. Range-decode L bytes → C*S*S int32 symbols (channel-major order).
//!   3. Add per-channel medians → float32 [1, C, S, S] latents.
//!   4. Run g_s through ONNX → float32 [1, 3, H, W] in [0, 1].

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ndarray::Array4;
use ort::value::Value;
use serde::Deserialize;

use super::{rans::RansDecoder, runtime::load_session, DecodeError};

/// Decoded image: RGB row-major u8, H × W × 3.
#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct CdfBlob {
    channels:   usize,
    cdf_length: Vec<usize>,
    offset:     Vec<i32>,
    medians:    Vec<f32>,
    cdf:        Vec<Vec<i32>>,
}

fn cdf_cache() -> &'static Mutex<HashMap<String, &'static CdfBlob>> {
    static CACHE: OnceLock<Mutex<HashMap<String, &'static CdfBlob>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn load_cdfs_for(model_path: &Path, quality: u32) -> Result<&'static CdfBlob, DecodeError> {
    let dir = model_path.parent()
        .ok_or_else(|| DecodeError::InvalidPayload("model path has no parent".into()))?;
    let cdf_path = dir.join(format!("cdfs_q{quality}.json"));
    let key = cdf_path.to_string_lossy().into_owned();
    {
        let cache = cdf_cache().lock().unwrap();
        if let Some(b) = cache.get(&key) { return Ok(*b); }
    }
    let raw = std::fs::read(&cdf_path)
        .map_err(|e| DecodeError::InvalidPayload(format!("cdf file {key}: {e}")))?;
    let blob: CdfBlob = serde_json::from_slice(&raw)
        .map_err(|e| DecodeError::InvalidPayload(format!("cdf parse {key}: {e}")))?;
    let leaked: &'static CdfBlob = Box::leak(Box::new(blob));
    cdf_cache().lock().unwrap().insert(key, leaked);
    Ok(leaked)
}

pub(crate) fn decode(payload: &[u8], model_path: &Path) -> Result<DecodedImage, DecodeError> {
    if payload.len() < 18 {
        return Err(DecodeError::InvalidPayload(
            format!("payload too short ({} B); need ≥ 18 B header", payload.len())));
    }
    let h = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    let w = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
    let l = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
    let q = u32::from_le_bytes(payload[12..16].try_into().unwrap());
    let c = payload[16] as usize;
    let s = 1usize << payload[17];
    if payload.len() != 18 + l {
        return Err(DecodeError::InvalidPayload(format!(
            "payload length {} != 18 + L={l}", payload.len())));
    }
    let rans_bytes = &payload[18..18 + l];

    let cdfs = load_cdfs_for(model_path, q)?;
    if cdfs.channels != c {
        return Err(DecodeError::InvalidPayload(format!(
            "envelope C={c} != cdf channels={}", cdfs.channels)));
    }

    // Range-decode C * S * S symbols, channel-major (matches
    // EntropyBottleneck._build_indexes flatten order).
    let mut latents = Array4::<f32>::zeros((1, c, s, s));
    let lat_slice = latents.as_slice_mut().unwrap();
    let mut dec = RansDecoder::new(rans_bytes)
        .map_err(|e| DecodeError::InvalidPayload(e.into()))?;
    let mut idx = 0;
    for ch in 0..c {
        let cdf     = &cdfs.cdf[ch];
        let cdf_len = cdfs.cdf_length[ch];
        let offset  = cdfs.offset[ch];
        let median  = cdfs.medians[ch];
        for _ in 0..(s * s) {
            let sym = dec.decode(cdf, cdf_len, offset);
            lat_slice[idx] = sym as f32 + median;
            idx += 1;
        }
    }

    let mut sess = load_session(model_path)?;
    let input_name  = sess.inputs().first().map(|i| i.name().to_string())
        .unwrap_or_else(|| "latents".into());
    let output_name = sess.outputs().first().map(|o| o.name().to_string())
        .unwrap_or_else(|| "image".into());
    let latents_v = Value::from_array(latents)
        .map_err(|e| DecodeError::Ort(format!("latents tensor: {e}")))?;
    let outputs = sess.run(ort::inputs![input_name => latents_v])
        .map_err(|e| DecodeError::Ort(format!("session.run: {e}")))?;
    let out = outputs.get(output_name.as_str()).ok_or_else(||
        DecodeError::Ort(format!("output {output_name} missing")))?;
    let view = out.try_extract_array::<f32>()
        .map_err(|e| DecodeError::Ort(format!("extract image: {e}")))?;
    let view = view.view();
    let dims = view.shape();
    if dims.len() != 4 || dims[1] != 3 || dims[2] != h || dims[3] != w {
        return Err(DecodeError::Ort(format!(
            "unexpected output shape {dims:?}; want [1, 3, {h}, {w}]")));
    }
    let data = view.as_slice().expect("contiguous");
    let mut rgb = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let pi = y * w + x;
            for ch in 0..3 {
                let v = data[ch * h * w + pi].clamp(0.0, 1.0);
                rgb[pi * 3 + ch] = (v * 255.0).round() as u8;
            }
        }
    }
    Ok(DecodedImage { rgb, width: w as u32, height: h as u32 })
}
