//! Text/general-binary compression wrappers. zstd (universal balance)
//! and XZ/LZMA (max classical ratio). Both lossless by construction.

use std::io::{Read, Write};

#[derive(Debug)]
pub struct TextError(pub String);

impl<E: std::fmt::Display> From<E> for TextError {
    fn from(e: E) -> Self { TextError(e.to_string()) }
}

pub type Result<T> = std::result::Result<T, TextError>;

/// zstd at the given compression level (1..=22; 22 is "ultra", slow).
pub fn zstd_encode(data: &[u8], level: i32) -> Result<Vec<u8>> {
    Ok(zstd::stream::encode_all(data, level)?)
}

pub fn zstd_decode(bytes: &[u8]) -> Result<Vec<u8>> {
    Ok(zstd::stream::decode_all(bytes)?)
}

/// XZ/LZMA2 at the given preset (0..=9; 6 is the default, 9 is "extreme").
pub fn xz_encode(data: &[u8], level: u32) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len() / 4);
    let mut enc = xz2::write::XzEncoder::new(&mut out, level);
    enc.write_all(data)?;
    enc.finish()?;
    Ok(out)
}

pub fn xz_decode(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    xz2::read::XzDecoder::new(bytes).read_to_end(&mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zstd_lossless_round_trip() {
        // mix of structure and PRNG — zstd should compress the structured
        // part well. Pure PRNG never compresses (high entropy) so use a
        // partly-repetitive payload to exercise both paths.
        let mut data = b"WAI standard zeroth text path test payload. ".repeat(500);
        data.extend((0..10_000u32).flat_map(|i| {
            (i.wrapping_mul(1_103_515_245).wrapping_add(12_345)).to_le_bytes()
        }));
        let e = zstd_encode(&data, 19).unwrap();
        let d = zstd_decode(&e).unwrap();
        assert_eq!(d, data, "zstd must be lossless");
        assert!(e.len() < data.len(), "zstd should compress mixed-entropy data");
    }

    #[test]
    fn xz_lossless_round_trip() {
        let data = b"the quick brown fox ".repeat(2_000);
        let e = xz_encode(&data, 6).unwrap();
        let d = xz_decode(&e).unwrap();
        assert_eq!(d, data);
        assert!(e.len() < data.len() / 5, "XZ should crush highly repetitive data");
    }
}
