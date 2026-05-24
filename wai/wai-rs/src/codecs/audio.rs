//! Audio codec wrappers. Opus (lossy, modern SOTA) + FLAC (lossless,
//! universal). Mono only for v1; multichannel is a follow-up.
//!
//! Sample format: f32 in [-1, 1] for the WAI-facing API; codecs may
//! convert internally to i16/i24 as needed.

use byteorder::{LittleEndian, WriteBytesExt};
use ogg::{PacketReader, PacketWriteEndInfo, PacketWriter};
use opus::{Application, Channels, Decoder, Encoder};
use std::io::Cursor;

#[derive(Debug)]
pub struct AudioError(pub String);

impl<E: std::fmt::Display> From<E> for AudioError {
    fn from(e: E) -> Self { AudioError(e.to_string()) }
}

pub type Result<T> = std::result::Result<T, AudioError>;

// ---- Opus (lossy, modern SOTA) --------------------------------------
// Opus only supports specific sample rates (8/12/16/24/48 kHz) and
// fixed frame sizes. We frame at 20 ms (the Opus default) and resample
// upstream if needed (caller's job).
const OPUS_FRAME_MS: usize = 20;

fn opus_frame_size(sr: u32) -> usize { (sr as usize) * OPUS_FRAME_MS / 1000 }

/// Encode mono f32 PCM as standard Ogg-Opus (RFC 7845). The bytes are
/// a real `.opus` file — drops into any Opus-aware tool (ffmpeg, VLC,
/// the browser <audio> tag, etc.) without re-framing.
///
/// libopus only accepts 8/12/16/24/48 kHz. Other rates MUST be
/// resampled upstream.
pub fn opus_encode(samples: &[f32], sr: u32, bitrate_bps: i32) -> Result<Vec<u8>> {
    if !matches!(sr, 8_000 | 12_000 | 16_000 | 24_000 | 48_000) {
        return Err(AudioError(format!(
            "opus only supports sr in {{8000,12000,16000,24000,48000}} \
             — got {sr}; resample upstream")));
    }
    let frame = opus_frame_size(sr);
    let mut enc = Encoder::new(sr, Channels::Mono, Application::Audio)?;
    enc.set_bitrate(opus::Bitrate::Bits(bitrate_bps))?;

    let mut out = Cursor::new(Vec::new());
    let serial: u32 = 0xC0FF_EE01;            // stream serial number
    let mut pw = PacketWriter::new(&mut out);

    // === OpusHead packet (RFC 7845 §5.1) ===
    let mut head = Vec::with_capacity(19);
    head.extend_from_slice(b"OpusHead");
    head.push(1);                             // version
    head.push(1);                             // channel count = 1
    head.write_u16::<LittleEndian>(0)?;       // pre-skip (no compensation)
    head.write_u32::<LittleEndian>(sr)?;      // original input sample rate
    head.write_i16::<LittleEndian>(0)?;       // output gain (Q7.8 dB)
    head.push(0);                             // channel mapping family 0
    pw.write_packet(head, serial, PacketWriteEndInfo::EndPage, 0)
        .map_err(|e| AudioError(format!("ogg header: {e:?}")))?;

    // === OpusTags packet (RFC 7845 §5.2) ===
    let vendor = b"wai-rs";
    let mut tags = Vec::with_capacity(8 + 4 + vendor.len() + 4);
    tags.extend_from_slice(b"OpusTags");
    tags.write_u32::<LittleEndian>(vendor.len() as u32)?;
    tags.extend_from_slice(vendor);
    tags.write_u32::<LittleEndian>(0)?;       // 0 user comments
    pw.write_packet(tags, serial, PacketWriteEndInfo::EndPage, 0)
        .map_err(|e| AudioError(format!("ogg tags: {e:?}")))?;

    // === Audio data packets ===
    // Granule positions are in 48 kHz samples regardless of input rate
    // (RFC 7845 §4); the encoder always operates at the input sr and
    // libopus internally upsamples to 48 kHz.
    let mut buf = vec![0u8; 4000];
    let nframes = (samples.len() + frame - 1) / frame;
    let granule_per_frame = (frame as u64 * 48_000) / sr as u64;
    for f in 0..nframes {
        let lo = f * frame;
        let hi = (lo + frame).min(samples.len());
        let mut in_frame = vec![0f32; frame];
        in_frame[..hi - lo].copy_from_slice(&samples[lo..hi]);
        let n = enc.encode_float(&in_frame, &mut buf)?;
        let granule = (f as u64 + 1) * granule_per_frame;
        let end = if f + 1 == nframes {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::NormalPacket
        };
        pw.write_packet(buf[..n].to_vec(), serial, end, granule)
            .map_err(|e| AudioError(format!("ogg packet: {e:?}")))?;
    }
    Ok(out.into_inner())
}

pub fn opus_decode(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let mut rd = PacketReader::new(Cursor::new(bytes));
    // First packet: OpusHead — parse for sample-rate + pre-skip
    let head_pkt = rd.read_packet_expected()
        .map_err(|e| AudioError(format!("ogg read OpusHead: {e:?}")))?;
    if head_pkt.data.len() < 19 || &head_pkt.data[..8] != b"OpusHead" {
        return Err(AudioError("not an Ogg-Opus stream".into()));
    }
    let pre_skip = u16::from_le_bytes(head_pkt.data[10..12].try_into().unwrap()) as usize;
    let sr = u32::from_le_bytes(head_pkt.data[12..16].try_into().unwrap());
    // Second packet: OpusTags — skip its content
    let _ = rd.read_packet_expected()
        .map_err(|e| AudioError(format!("ogg read OpusTags: {e:?}")))?;

    let mut dec = Decoder::new(sr, Channels::Mono)?;
    let frame = opus_frame_size(sr);
    let mut buf = vec![0f32; frame * 6];
    let mut out: Vec<f32> = Vec::new();
    let mut last_granule: u64 = 0;
    loop {
        let pkt = match rd.read_packet().map_err(|e|
            AudioError(format!("ogg packet read: {e:?}")))? {
            Some(p) => p,
            None => break,
        };
        let got = dec.decode_float(&pkt.data, &mut buf, false)?;
        out.extend_from_slice(&buf[..got]);
        last_granule = pkt.absgp_page();
    }
    // Trim pre-skip (algorithmic delay padding) and any trailing pad
    // implied by the final granule position. Granule is in 48 kHz; convert
    // back to input-rate samples.
    let drop = (pre_skip as u64 * sr as u64 / 48_000) as usize;
    if drop < out.len() { out.drain(..drop); } else { out.clear(); }
    let total_at_sr = (last_granule * sr as u64 / 48_000) as usize;
    let total_at_sr = total_at_sr.saturating_sub(drop);
    if total_at_sr < out.len() { out.truncate(total_at_sr); }
    Ok((out, sr))
}

// ---- FLAC (lossless) -------------------------------------------------
// libFLAC bindings for encode; pure-Rust claxon for decode.

pub fn flac_encode(samples: &[f32], sr: u32) -> Result<Vec<u8>> {
    use flac_bound::{FlacEncoder, WriteWrapper};
    // FLAC takes int32 samples; widen i16-equivalent quant to i32 in 16-bit
    let i32_samples: Vec<i32> = samples.iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i32)
        .collect();
    let mut out: Vec<u8> = Vec::new();
    {
        let mut ww = WriteWrapper(&mut out);
        let mut enc = FlacEncoder::new()
            .ok_or_else(|| AudioError("flac encoder init failed".into()))?
            .channels(1)
            .bits_per_sample(16)
            .sample_rate(sr)
            .compression_level(8)
            .init_write(&mut ww)
            .map_err(|e| AudioError(format!("flac init: {e:?}")))?;
        enc.process_interleaved(&i32_samples, i32_samples.len() as u32)
            .map_err(|e| AudioError(format!("flac process: {e:?}")))?;
        enc.finish()
            .map_err(|e| AudioError(format!("flac finish: {e:?}")))?;
    }
    Ok(out)
}

pub fn flac_decode(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let mut rdr = claxon::FlacReader::new(Cursor::new(bytes))
        .map_err(|e| AudioError(format!("flac open: {e}")))?;
    let info = rdr.streaminfo();
    let sr = info.sample_rate;
    let mut out = Vec::with_capacity(info.samples.unwrap_or(0) as usize);
    for s in rdr.samples() {
        out.push((s.map_err(|e| AudioError(format!("flac sample: {e}")))? as f32) / 32768.0);
    }
    Ok((out, sr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn synth_signal(sr: u32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n).map(|i| {
            let t = i as f32 / sr as f32;
            0.4 * (2.0 * PI * 440.0 * t).sin()
                + 0.2 * (2.0 * PI * 1320.0 * t).sin()
        }).collect()
    }

    #[test]
    fn opus_round_trip_sane() {
        let sr = 48_000;
        let sig = synth_signal(sr, 0.5);
        let bytes = opus_encode(&sig, sr, 64_000).unwrap();
        assert!(bytes.len() > 0);
        let (rec, rsr) = opus_decode(&bytes).unwrap();
        assert_eq!(rsr, sr);
        assert_eq!(rec.len(), sig.len());
    }

    #[test]
    fn flac_lossless_round_trip() {
        let sr = 44_100;
        let sig = synth_signal(sr, 0.25);
        let bytes = flac_encode(&sig, sr).unwrap();
        let (rec, rsr) = flac_decode(&bytes).unwrap();
        assert_eq!(rsr, sr);
        assert_eq!(rec.len(), sig.len());
        // 16-bit quant is the only loss; samples should match exactly
        // after going through the same int16 rounding.
        let q: Vec<f32> = sig.iter()
            .map(|&s| ((s.clamp(-1.0, 1.0) * 32767.0).round() / 32768.0))
            .collect();
        for (a, b) in q.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-4, "FLAC must be lossless on the quantized samples");
        }
    }
}
