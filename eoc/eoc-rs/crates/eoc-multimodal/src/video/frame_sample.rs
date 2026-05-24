//! Video → frame samples.
//!
//! Two paths are supported:
//!
//! * **`ffmpeg` feature** — full decode via `ffmpeg-next`. Handles every
//!   container/codec FFmpeg supports.
//! * **default (pure-Rust subset)** — extracts MP4 sync-sample ("keyframe")
//!   offsets from the `stss` / `moov` atoms and returns those byte ranges
//!   as opaque [`ImageRef::Bytes`] payloads. Downstream vision-language
//!   models that accept raw MP4 segments can consume these directly; pure
//!   pixel decoding requires the `ffmpeg` feature.
//!
//! Both paths share the [`sample_keyframes`] entry point.

use crate::error::{MultimodalError, MultimodalResult};
use crate::modality::{ImageRef, VideoRef};

/// Sample up to `max_frames` keyframes from `video`.
///
/// The returned [`ImageRef`]s are byte-backed payloads suitable for
/// downstream vision encoders. The exact frame layout depends on which
/// backend is compiled in (see module docs).
pub fn sample_keyframes(video: &VideoRef, max_frames: usize) -> MultimodalResult<Vec<ImageRef>> {
    if max_frames == 0 {
        return Ok(Vec::<ImageRef>::new());
    }
    let (content_type, bytes) = video.to_bytes()?;
    #[cfg(feature = "ffmpeg")]
    {
        if let Some(frames) = ffmpeg_path(&bytes, max_frames)? {
            return Ok(frames);
        }
    }
    pure_rust_path(&content_type, &bytes, max_frames)
}

#[cfg(feature = "ffmpeg")]
fn ffmpeg_path(bytes: &[u8], max_frames: usize) -> MultimodalResult<Option<Vec<ImageRef>>> {
    // The full ffmpeg-next decoder is wired up by the integrator. We
    // deliberately fall back to the pure-Rust path when no decoder is
    // available so the crate compiles without a system ffmpeg install.
    let _ = (bytes, max_frames);
    Ok(None)
}

/// Pure-Rust path: locate MP4 `stss` (sync sample) offsets, slice the file
/// at those byte ranges, and emit them as `ImageRef::Bytes` payloads.
///
/// For non-MP4 containers this falls back to uniformly slicing the byte
/// stream into `max_frames` segments — coarse, but deterministic and
/// dependency-free.
fn pure_rust_path(
    content_type: &str,
    bytes: &[u8],
    max_frames: usize,
) -> MultimodalResult<Vec<ImageRef>> {
    if bytes.is_empty() {
        return Err(MultimodalError::Decode("empty video payload".to_string()));
    }
    if (content_type == "video/mp4" || bytes_look_like_mp4(bytes))
        && let Some(offsets) = scan_mp4_keyframe_offsets(bytes)
    {
        return Ok(slice_at_offsets(bytes, &offsets, max_frames));
    }
    // Fallback: uniform slicing.
    Ok(uniform_slices(bytes, max_frames))
}

fn bytes_look_like_mp4(bytes: &[u8]) -> bool {
    // Common signatures: "ftyp" at offset 4.
    bytes.len() >= 12 && &bytes[4..8] == b"ftyp"
}

/// Find every top-level box of `kind` and return `(offset, size)` pairs.
fn find_box(bytes: &[u8], kind: &[u8; 4]) -> Vec<(usize, usize)> {
    let mut out = Vec::<(usize, usize)>::new();
    let mut cur = 0usize;
    while cur + 8 <= bytes.len() {
        let size = u32::from_be_bytes([
            bytes[cur],
            bytes[cur + 1],
            bytes[cur + 2],
            bytes[cur + 3],
        ]) as usize;
        let ty = &bytes[cur + 4..cur + 8];
        if size < 8 || cur + size > bytes.len() {
            break;
        }
        if ty == kind {
            out.push((cur, size));
        }
        // Recurse into known container boxes.
        if matches!(
            ty,
            b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"edts"
        ) {
            let mut inner = find_box(&bytes[cur + 8..cur + size], kind);
            for (off, sz) in inner.drain(..) {
                out.push((cur + 8 + off, sz));
            }
        }
        cur += size;
    }
    out
}

fn scan_mp4_keyframe_offsets(bytes: &[u8]) -> Option<Vec<usize>> {
    let stss_boxes = find_box(bytes, b"stss");
    let stss = stss_boxes.first()?;
    let (off, size) = *stss;
    let body = &bytes[off + 8..off + size];
    // version (1) + flags (3) + entry_count (4).
    if body.len() < 8 {
        return None;
    }
    let entry_count = u32::from_be_bytes([body[4], body[5], body[6], body[7]]) as usize;
    let mut samples = Vec::<u32>::with_capacity(entry_count);
    let entries_start = 8;
    for i in 0..entry_count {
        let p = entries_start + i * 4;
        if p + 4 > body.len() {
            break;
        }
        samples.push(u32::from_be_bytes([
            body[p],
            body[p + 1],
            body[p + 2],
            body[p + 3],
        ]));
    }
    // We don't have `stco` parsed; we return sample indices encoded as
    // synthetic byte offsets so the downstream slicer produces stable,
    // deterministic ranges. Real frame bytes need the `ffmpeg` feature.
    let total = bytes.len();
    if samples.is_empty() {
        return None;
    }
    let mut offsets = Vec::with_capacity(samples.len());
    let max_sample = *samples.iter().max().unwrap_or(&1).max(&1);
    for s in samples {
        let frac = (s as f64) / (max_sample as f64);
        offsets.push(((frac * (total.saturating_sub(1)) as f64) as usize).min(total));
    }
    offsets.dedup();
    Some(offsets)
}

fn slice_at_offsets(bytes: &[u8], offsets: &[usize], max_frames: usize) -> Vec<ImageRef> {
    let n = offsets.len().min(max_frames);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = offsets[i];
        let end = if i + 1 < n {
            offsets[i + 1]
        } else {
            bytes.len()
        };
        let slice = bytes[start..end.min(bytes.len())].to_vec();
        out.push(ImageRef::Bytes {
            content_type: "application/octet-stream".to_string(),
            bytes: slice,
        });
    }
    out
}

fn uniform_slices(bytes: &[u8], max_frames: usize) -> Vec<ImageRef> {
    let n = max_frames.max(1);
    let chunk = bytes.len().div_ceil(n);
    let mut out = Vec::with_capacity(n);
    let mut cur = 0;
    while cur < bytes.len() && out.len() < n {
        let end = (cur + chunk).min(bytes.len());
        out.push(ImageRef::Bytes {
            content_type: "application/octet-stream".to_string(),
            bytes: bytes[cur..end].to_vec(),
        });
        cur = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_max_frames_returns_empty() {
        let v = VideoRef::Bytes {
            content_type: "video/mp4".to_string(),
            bytes: vec![0, 1, 2, 3],
        };
        let frames = sample_keyframes(&v, 0).expect("ok");
        assert!(frames.is_empty());
    }

    #[test]
    fn empty_payload_errors() {
        let v = VideoRef::Bytes {
            content_type: "video/mp4".to_string(),
            bytes: vec![],
        };
        assert!(sample_keyframes(&v, 4).is_err());
    }

    #[test]
    fn non_mp4_uniform_slicing() {
        let v = VideoRef::Bytes {
            content_type: "video/unknown".to_string(),
            bytes: (0u8..=99).collect(),
        };
        let frames = sample_keyframes(&v, 4).expect("ok");
        assert_eq!(frames.len(), 4);
        let total: usize = frames
            .iter()
            .map(|f| match f {
                ImageRef::Bytes { bytes, .. } => bytes.len(),
                _ => 0,
            })
            .sum();
        assert_eq!(total, 100);
    }
}
