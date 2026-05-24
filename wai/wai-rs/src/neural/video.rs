//! Decoder for `wai.neural.video_bmshj2018` — per-frame neural video.
//!
//! Wire format:
//!   <IIIIBB>   H, W, n_frames, fps_x_1000, C, S_log2
//!   n × <I>    per-frame zstd-payload length L_i
//!   n × L_i    zstd-compressed int8 latents [C, S, S]
//!
//! Reuses the same `bmshj2018-factorized` synthesis-transform ONNX as
//! the still-image capability — one model on disk, two capabilities.

use std::path::Path;

use ndarray::Array4;
use ort::session::Session;
use ort::value::Value;

use super::{runtime::load_session, DecodeError};

#[derive(Debug, Clone)]
pub struct DecodedVideo {
    pub frames_rgb: Vec<Vec<u8>>,   // n_frames × (H*W*3 bytes)
    pub width: u32,
    pub height: u32,
    pub fps: f32,
}

pub(crate) fn decode(payload: &[u8], model_path: &Path) -> Result<DecodedVideo, DecodeError> {
    if payload.len() < 18 {
        return Err(DecodeError::InvalidPayload(
            format!("payload too short ({} B); need ≥ 18 B header", payload.len())));
    }
    let h = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    let w = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
    let fps_x_1000 = u32::from_le_bytes(payload[12..16].try_into().unwrap());
    let c = payload[16] as usize;
    let s = 1usize << payload[17];
    let header_len = 18;
    let table_len  = n * 4;
    if payload.len() < header_len + table_len {
        return Err(DecodeError::InvalidPayload("frame table out of bounds".into()));
    }

    let mut lens = Vec::with_capacity(n);
    for i in 0..n {
        let off = header_len + i * 4;
        lens.push(u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize);
    }

    let mut sess = load_session(model_path)?;
    let in_name  = sess.inputs().first().map(|i| i.name().to_string()).unwrap_or_else(|| "latents".into());
    let out_name = sess.outputs().first().map(|o| o.name().to_string()).unwrap_or_else(|| "image".into());

    let mut offset = header_len + table_len;
    let mut frames_rgb = Vec::with_capacity(n);
    for (i, &l) in lens.iter().enumerate() {
        if offset + l > payload.len() {
            return Err(DecodeError::InvalidPayload(
                format!("frame {i} out of bounds (offset {offset} + len {l} > payload {})",
                        payload.len())));
        }
        let zbytes = &payload[offset..offset + l];
        offset += l;
        let expected_raw = c * s * s;
        let mut raw = vec![0u8; expected_raw];
        let written = zstd_safe::decompress(&mut raw, zbytes)
            .map_err(|e| DecodeError::Zstd(format!("frame {i} decompress: code {e:?}")))?;
        if written != expected_raw {
            return Err(DecodeError::Zstd(format!(
                "frame {i} decompressed {written} != C*S*S = {expected_raw}")));
        }
        frames_rgb.push(decode_one_frame(&mut sess, &raw, c, s, h, w, &in_name, &out_name)?);
    }

    Ok(DecodedVideo {
        frames_rgb, width: w as u32, height: h as u32,
        fps: fps_x_1000 as f32 / 1000.0,
    })
}

fn decode_one_frame(sess: &mut Session, raw_i8: &[u8], c: usize, s: usize, h: usize, w: usize,
                    in_name: &str, out_name: &str) -> Result<Vec<u8>, DecodeError> {
    let mut latents = Array4::<f32>::zeros((1, c, s, s));
    {
        let slice = latents.as_slice_mut().unwrap();
        for (i, &b) in raw_i8.iter().enumerate() {
            slice[i] = (b as i8) as f32;
        }
    }
    let v = Value::from_array(latents)
        .map_err(|e| DecodeError::Ort(format!("latents tensor: {e}")))?;
    let outputs = sess.run(ort::inputs![in_name => v])
        .map_err(|e| DecodeError::Ort(format!("session.run: {e}")))?;
    let out = outputs.get(out_name).ok_or_else(||
        DecodeError::Ort(format!("output {out_name} missing")))?;
    let view = out.try_extract_array::<f32>()
        .map_err(|e| DecodeError::Ort(format!("extract image: {e}")))?;
    let view = view.view();
    let data = view.as_slice().expect("contiguous");
    let mut rgb = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            for ch in 0..3 {
                let v = data[ch * h * w + idx].clamp(0.0, 1.0);
                rgb[idx * 3 + ch] = (v * 255.0).round() as u8;
            }
        }
    }
    Ok(rgb)
}
