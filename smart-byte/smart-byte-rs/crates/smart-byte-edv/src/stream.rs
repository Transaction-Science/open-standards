//! Chunked stream encryption (DIF EDV v0.10 § 4.5 — Streams).
//!
//! Large payloads can exceed reasonable single-AEAD limits or simply be
//! too large to materialise in memory. Streams sidestep both problems by
//! splitting the payload into fixed-size plaintext chunks, sealing each
//! independently with AES-256-GCM, and recording per-chunk IV / tag bytes
//! in a manifest. Consumers fetch chunks in any order and decrypt them
//! independently, then concatenate the plaintext.
//!
//! The chunk format is:
//!
//! ```text
//! chunk[i] = AES-256-GCM(
//!   key   = CEK,
//!   nonce = nonce_prefix || u32_be(chunk_index),
//!   aad   = stream_id || u32_be(chunk_index),
//!   pt    = plaintext_slice[i]
//! )
//! ```
//!
//! The `nonce_prefix` (8 bytes) is chosen once per stream and stored in
//! the [`StreamManifest`]; the per-chunk `chunk_index` keeps every nonce
//! unique under the same key for up to 2³² chunks, matching the standard
//! AES-GCM nonce-construction guidance.

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::error::EdvError;

/// One sealed chunk in a stream.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamChunk {
    /// Zero-based chunk index.
    pub index: u32,
    /// AES-GCM ciphertext (without the trailing 16-byte tag — the tag is
    /// returned separately for convenience).
    pub ciphertext: Vec<u8>,
    /// 16-byte GCM authentication tag.
    pub tag: Vec<u8>,
}

/// Public manifest for a sealed stream.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamManifest {
    /// Stream id (URN), mirrored into
    /// [`crate::spec::Stream::id`].
    pub id: String,
    /// 8-byte random nonce prefix; concatenated with `u32_be(index)` to
    /// form each per-chunk AES-GCM nonce.
    pub nonce_prefix: [u8; 8],
    /// Plaintext bytes per chunk (the final chunk may be smaller).
    pub chunk_size: usize,
    /// Number of chunks in the stream.
    pub chunk_count: usize,
    /// Plaintext byte length of the stream.
    pub plaintext_len: usize,
}

impl StreamManifest {
    /// Reify the manifest as a [`crate::spec::Stream`] descriptor that can
    /// be stored on an [`crate::spec::EncryptedDocument`].
    pub fn as_spec(&self) -> crate::spec::Stream {
        crate::spec::Stream {
            id: self.id.clone(),
            chunk_count: self.chunk_count,
            chunk_size: self.chunk_size,
            sequence: 0,
        }
    }
}

/// Seal a plaintext into `chunk_size`-sized chunks under `cek`.
pub fn seal(
    id: impl Into<String>,
    cek: &[u8; 32],
    plaintext: &[u8],
    chunk_size: usize,
) -> Result<(StreamManifest, Vec<StreamChunk>), EdvError> {
    if chunk_size == 0 {
        return Err(EdvError::Stream("chunk_size must be > 0".into()));
    }
    let mut nonce_prefix = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut nonce_prefix);
    let id = id.into();
    let cipher = Aes256Gcm::new(cek.into());

    let chunk_count = plaintext.len().div_ceil(chunk_size).max(1);
    if (chunk_count as u64) > u32::MAX as u64 {
        return Err(EdvError::Stream(
            "too many chunks for u32 index".into(),
        ));
    }
    let mut chunks = Vec::with_capacity(chunk_count);
    for i in 0..chunk_count {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(plaintext.len());
        let pt_slice = if start >= plaintext.len() {
            &[][..]
        } else {
            &plaintext[start..end]
        };
        let nonce = chunk_nonce(&nonce_prefix, i as u32);
        let aad = chunk_aad(&id, i as u32);
        let ct_tag = cipher
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: pt_slice,
                    aad: &aad,
                },
            )
            .map_err(|e| EdvError::Crypto(format!("aes-gcm seal: {e}")))?;
        if ct_tag.len() < 16 {
            return Err(EdvError::Crypto(
                "aes-gcm chunk output too short".into(),
            ));
        }
        let (ct, tag) = ct_tag.split_at(ct_tag.len() - 16);
        chunks.push(StreamChunk {
            index: i as u32,
            ciphertext: ct.to_vec(),
            tag: tag.to_vec(),
        });
    }

    let manifest = StreamManifest {
        id,
        nonce_prefix,
        chunk_size,
        chunk_count,
        plaintext_len: plaintext.len(),
    };
    Ok((manifest, chunks))
}

/// Open a sealed stream, reassembling the plaintext.
///
/// Chunks may be supplied in any order — they are sorted internally.
pub fn open(
    manifest: &StreamManifest,
    cek: &[u8; 32],
    chunks: &[StreamChunk],
) -> Result<Vec<u8>, EdvError> {
    if chunks.len() != manifest.chunk_count {
        return Err(EdvError::Stream(format!(
            "expected {} chunks, got {}",
            manifest.chunk_count,
            chunks.len()
        )));
    }
    let mut sorted: Vec<&StreamChunk> = chunks.iter().collect();
    sorted.sort_by_key(|c| c.index);
    for (i, c) in sorted.iter().enumerate() {
        if c.index as usize != i {
            return Err(EdvError::Stream(format!(
                "missing or duplicate chunk at index {i}"
            )));
        }
    }

    let cipher = Aes256Gcm::new(cek.into());
    let mut out = Vec::with_capacity(manifest.plaintext_len);
    for chunk in sorted {
        if chunk.tag.len() != 16 {
            return Err(EdvError::Jose(format!(
                "chunk {} has bad tag length",
                chunk.index
            )));
        }
        let mut ct = chunk.ciphertext.clone();
        ct.extend_from_slice(&chunk.tag);
        let nonce = chunk_nonce(&manifest.nonce_prefix, chunk.index);
        let aad = chunk_aad(&manifest.id, chunk.index);
        let pt = cipher
            .decrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: &ct,
                    aad: &aad,
                },
            )
            .map_err(|e| {
                EdvError::Crypto(format!(
                    "aes-gcm open chunk {}: {e}",
                    chunk.index
                ))
            })?;
        out.extend_from_slice(&pt);
    }

    if out.len() != manifest.plaintext_len {
        return Err(EdvError::Stream(format!(
            "reassembled plaintext length {} != manifest length {}",
            out.len(),
            manifest.plaintext_len
        )));
    }
    Ok(out)
}

fn chunk_nonce(prefix: &[u8; 8], index: u32) -> [u8; 12] {
    let mut n = [0u8; 12];
    n[..8].copy_from_slice(prefix);
    n[8..].copy_from_slice(&index.to_be_bytes());
    n
}

fn chunk_aad(id: &str, index: u32) -> Vec<u8> {
    let mut aad = Vec::with_capacity(id.len() + 4);
    aad.extend_from_slice(id.as_bytes());
    aad.extend_from_slice(&index.to_be_bytes());
    aad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty() {
        let cek = [3u8; 32];
        let (m, c) = seal("urn:stream:empty", &cek, &[], 16).expect("seal");
        assert_eq!(m.chunk_count, 1);
        let pt = open(&m, &cek, &c).expect("open");
        assert!(pt.is_empty());
    }

    #[test]
    fn round_trip_exact_chunks() {
        let cek = [5u8; 32];
        let pt: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let (m, c) = seal("urn:stream:1", &cek, &pt, 16).expect("seal");
        assert_eq!(m.chunk_count, 4);
        let out = open(&m, &cek, &c).expect("open");
        assert_eq!(out, pt);
    }

    #[test]
    fn round_trip_partial_final_chunk() {
        let cek = [9u8; 32];
        let pt: Vec<u8> = (0..70).map(|i| (i % 251) as u8).collect();
        let (m, c) = seal("urn:stream:2", &cek, &pt, 16).expect("seal");
        assert_eq!(m.chunk_count, 5);
        let out = open(&m, &cek, &c).expect("open");
        assert_eq!(out, pt);
    }

    #[test]
    fn shuffled_chunks_reassemble() {
        let cek = [11u8; 32];
        let pt: Vec<u8> = (0..40).map(|i| i as u8).collect();
        let (m, mut c) = seal("urn:stream:3", &cek, &pt, 8).expect("seal");
        c.reverse();
        let out = open(&m, &cek, &c).expect("open");
        assert_eq!(out, pt);
    }

    #[test]
    fn tamper_detection() {
        let cek = [13u8; 32];
        let pt = b"some bytes here for tamper test, please verify";
        let (m, mut c) = seal("urn:stream:4", &cek, pt, 8).expect("seal");
        if let Some(first) = c.first_mut() {
            if !first.ciphertext.is_empty() {
                first.ciphertext[0] ^= 0x01;
            }
        }
        let res = open(&m, &cek, &c);
        assert!(res.is_err());
    }
}
