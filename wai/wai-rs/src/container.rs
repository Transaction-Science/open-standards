//! The WAI envelope: magic "WAI1" | u32 manifest_len | manifest JSON |
//! payload bytes. The manifest declares media, intent, the capability
//! REQUIREMENT (never weights, never a hash — a name the sink resolves
//! against its registered codecs) and an optional fallback. Wire-
//! identical to `wai/container.py`.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"WAI1";        // v1.0 single-payload
pub const MAGIC_V2: &[u8; 4] = b"WAI2";     // v1.1 multi-rendition

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequirement {
    pub capability: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conditioning {
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub wai: String,
    pub media: String,
    pub intent: String,
    pub model_requirement: ModelRequirement,
    pub conditioning: Conditioning,
    #[serde(default)]
    pub target: serde_json::Value,
}

pub struct Wai {
    pub manifest: Manifest,
    pub payload: Vec<u8>,
}

impl Wai {
    pub fn new(manifest: Manifest, payload: Vec<u8>) -> Self {
        Self { manifest, payload }
    }

    pub fn write<P: AsRef<Path>>(&self, path: P) -> std::io::Result<usize> {
        let bytes = self.to_bytes()?;
        fs::write(path, &bytes)?;
        Ok(bytes.len())
    }

    pub fn to_bytes(&self) -> std::io::Result<Vec<u8>> {
        // Manifest must be serialized with no whitespace to match Python's
        // json.dumps(separators=(",", ":")), so the bytes round-trip
        // identically. serde_json's default `to_vec` already does this.
        let m = serde_json::to_vec(&self.manifest)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut out = Vec::with_capacity(12 + m.len() + self.payload.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(m.len() as u32).to_le_bytes());
        out.extend_from_slice(&m);
        out.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.payload);
        Ok(out)
    }

    pub fn read<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let bytes = fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        // Every field is bounds-checked: malformed input (truncated /
        // wrong-magic / length-prefix overrunning the buffer / etc.)
        // must return Err, never panic. Fuzz smoke test enforces this.
        let need = |off: usize, n: usize, what: &str| -> std::io::Result<()> {
            if off.saturating_add(n) > bytes.len() {
                Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    format!("truncated WAI ({what} needs {n} bytes at offset {off}, have {})",
                            bytes.len())))
            } else { Ok(()) }
        };
        need(0, 4, "magic")?;
        if &bytes[..4] != MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                "not a WAI file (magic mismatch)"));
        }
        need(4, 4, "manifest length")?;
        let ml = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        need(8, ml, "manifest body")?;
        let m: Manifest = serde_json::from_slice(&bytes[8..8 + ml])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let p_off = 8 + ml;
        need(p_off, 4, "payload length")?;
        let pl = u32::from_le_bytes(bytes[p_off..p_off + 4].try_into().unwrap()) as usize;
        need(p_off + 4, pl, "payload body")?;
        let payload = bytes[p_off + 4..p_off + 4 + pl].to_vec();
        Ok(Self { manifest: m, payload })
    }

    pub fn capability(&self) -> &str {
        &self.manifest.model_requirement.capability
    }

    pub fn fallback(&self) -> Option<&str> {
        self.manifest.model_requirement.fallback.as_deref()
    }

    pub fn media(&self) -> &str {
        &self.manifest.media
    }

    pub fn intent(&self) -> &str {
        &self.manifest.intent
    }

    pub fn kind(&self) -> &str {
        &self.manifest.conditioning.kind
    }
}

// ---- v1.1: multi-rendition envelope (WAI2) ----------------------
//
// One envelope carries multiple renditions of the SAME content; the
// sink picks one by deployer-defined policy (compute budget, bandwidth,
// preferred runtime). In a controlled-deployment ecosystem every sink
// has every required capability — this is not "fallback for missing
// capability", it's policy selection. See SPEC.md §6.1.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenditionMeta {
    pub capability: String,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestV2 {
    pub wai: String,                          // "1.1"
    pub media: String,
    pub intent: String,
    pub renditions: Vec<RenditionMeta>,
    #[serde(default)]
    pub target: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct Rendition {
    pub capability: String,
    pub kind: String,
    pub payload: Vec<u8>,
}

pub struct WaiMulti {
    pub manifest: ManifestV2,
    pub renditions: Vec<Vec<u8>>,             // payload bytes, parallel to manifest.renditions
}

impl WaiMulti {
    pub fn new(media: impl Into<String>, intent: impl Into<String>,
               target: serde_json::Value, renditions: Vec<Rendition>) -> Self {
        let metas: Vec<RenditionMeta> = renditions.iter()
            .map(|r| RenditionMeta { capability: r.capability.clone(),
                                     kind: r.kind.clone() })
            .collect();
        let payloads: Vec<Vec<u8>> = renditions.into_iter().map(|r| r.payload).collect();
        Self {
            manifest: ManifestV2 {
                wai: "1.1".into(),
                media: media.into(),
                intent: intent.into(),
                renditions: metas,
                target,
            },
            renditions: payloads,
        }
    }

    pub fn write<P: AsRef<Path>>(&self, path: P) -> std::io::Result<usize> {
        let bytes = self.to_bytes()?;
        fs::write(path, &bytes)?;
        Ok(bytes.len())
    }

    pub fn read<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let bytes = fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    pub fn to_bytes(&self) -> std::io::Result<Vec<u8>> {
        if self.manifest.renditions.len() != self.renditions.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("manifest has {} renditions but payload list has {}",
                        self.manifest.renditions.len(), self.renditions.len())));
        }
        if self.renditions.len() > u16::MAX as usize {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                "too many renditions for u16 count field"));
        }
        let m = serde_json::to_vec(&self.manifest)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let table_bytes = 2 + self.renditions.len() * 8;
        let payload_bytes: usize = self.renditions.iter().map(|p| p.len()).sum();

        let mut out = Vec::with_capacity(8 + m.len() + table_bytes + payload_bytes);
        out.extend_from_slice(MAGIC_V2);
        out.extend_from_slice(&(m.len() as u32).to_le_bytes());
        out.extend_from_slice(&m);
        out.extend_from_slice(&(self.renditions.len() as u16).to_le_bytes());

        // Rendition table: (u32 offset, u32 length); offsets are relative to
        // the start of the payloads block (the byte right after the table).
        let mut off: u32 = 0;
        for p in &self.renditions {
            out.extend_from_slice(&off.to_le_bytes());
            out.extend_from_slice(&(p.len() as u32).to_le_bytes());
            off = off.checked_add(p.len() as u32).ok_or_else(|| std::io::Error::new(
                std::io::ErrorKind::InvalidData, "payload sum overflows u32"))?;
        }
        for p in &self.renditions {
            out.extend_from_slice(p);
        }
        Ok(out)
    }

    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        let need = |off: usize, n: usize, what: &str| -> std::io::Result<()> {
            if off.saturating_add(n) > bytes.len() {
                Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    format!("truncated WAI2 ({what} needs {n} bytes at offset {off}, have {})",
                            bytes.len())))
            } else { Ok(()) }
        };
        need(0, 4, "magic")?;
        if &bytes[..4] != MAGIC_V2 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                "not a WAI2 file (magic mismatch)"));
        }
        need(4, 4, "manifest length")?;
        let ml = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        need(8, ml, "manifest body")?;
        let manifest: ManifestV2 = serde_json::from_slice(&bytes[8..8 + ml])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut o = 8 + ml;
        need(o, 2, "rendition count")?;
        let n = u16::from_le_bytes(bytes[o..o + 2].try_into().unwrap()) as usize;
        o += 2;
        if n != manifest.renditions.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("rendition table count {} != manifest renditions {}",
                        n, manifest.renditions.len())));
        }
        need(o, n * 8, "rendition table")?;
        let mut entries: Vec<(u32, u32)> = Vec::with_capacity(n);
        for i in 0..n {
            let off = u32::from_le_bytes(bytes[o + i*8..o + i*8 + 4].try_into().unwrap());
            let len = u32::from_le_bytes(bytes[o + i*8 + 4..o + i*8 + 8].try_into().unwrap());
            entries.push((off, len));
        }
        o += n * 8;
        let payload_block_start = o;
        let total_payloads: usize = entries.iter().map(|(_, l)| *l as usize).sum();
        need(payload_block_start, total_payloads, "payload block")?;

        // Validate offsets: each entry's (off, len) must lie within the
        // payload block and not overlap previous entries' ranges (we
        // require monotonic contiguous packing to keep parsing simple).
        let mut expected = 0u32;
        let mut renditions = Vec::with_capacity(n);
        for (off, len) in &entries {
            if *off != expected {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    format!("rendition offset {off} not contiguous (expected {expected})")));
            }
            let start = payload_block_start + *off as usize;
            let end = start + *len as usize;
            if end > bytes.len() {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    format!("rendition byte range [{start}, {end}) exceeds file length {}",
                            bytes.len())));
            }
            renditions.push(bytes[start..end].to_vec());
            expected = expected.checked_add(*len).ok_or_else(|| std::io::Error::new(
                std::io::ErrorKind::InvalidData, "offset arithmetic overflows"))?;
        }
        Ok(Self { manifest, renditions })
    }

    /// Pick the first rendition whose capability matches `pred`.
    pub fn pick<F: Fn(&str) -> bool>(&self, pred: F) -> Option<(&RenditionMeta, &[u8])> {
        for (meta, payload) in self.manifest.renditions.iter().zip(self.renditions.iter()) {
            if pred(&meta.capability) {
                return Some((meta, payload));
            }
        }
        None
    }

    /// First rendition in deployer-preferred order (the canonical default
    /// pick when the sink has no further policy).
    pub fn primary(&self) -> Option<(&RenditionMeta, &[u8])> {
        self.manifest.renditions.first()
            .zip(self.renditions.first().map(|v| v.as_slice()))
    }

    pub fn media(&self) -> &str { &self.manifest.media }
    pub fn intent(&self) -> &str { &self.manifest.intent }
}

/// Sniff the magic and return which envelope version is in this byte
/// blob. Lets callers branch on `Wai::from_bytes` vs `WaiMulti::from_bytes`
/// without parsing twice.
pub fn detect_version(bytes: &[u8]) -> std::io::Result<u8> {
    if bytes.len() < 4 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            "too short for magic"));
    }
    match &bytes[..4] {
        b"WAI1" => Ok(1),
        b"WAI2" => Ok(2),
        _ => Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            "unknown magic (not WAI1/WAI2)")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_envelope() {
        let m = Manifest {
            wai: "1.0".into(),
            media: "image".into(),
            intent: "replicate".into(),
            model_requirement: ModelRequirement {
                capability: "wai.zeroth.dct".into(),
                fallback: Some("zeroth".into()),
            },
            conditioning: Conditioning { kind: "image_zeroth".into() },
            target: serde_json::json!({"w": 512, "h": 512}),
        };
        let w = Wai::new(m, b"hello world".to_vec());
        let bytes = w.to_bytes().unwrap();
        let w2 = Wai::from_bytes(&bytes).unwrap();
        assert_eq!(w2.capability(), "wai.zeroth.dct");
        assert_eq!(w2.payload, b"hello world");
        assert_eq!(w2.media(), "image");
    }

    // ---- v1.1: multi-rendition WAI2 ----

    fn sample_multi() -> WaiMulti {
        WaiMulti::new("audio", "replicate",
            serde_json::json!({"sr": 48000, "dur": 5.0}),
            vec![
                Rendition { capability: "wai.neural.encodec32".into(),
                            kind: "encodec_tokens".into(),
                            payload: b"NEURAL-TOKENS-2317-BYTES".to_vec() },
                Rendition { capability: "wai.audio.opus".into(),
                            kind: "opus".into(),
                            payload: b"OPUS-PAYLOAD-4411-BYTES".to_vec() },
            ])
    }

    #[test]
    fn multi_rendition_round_trip() {
        let m = sample_multi();
        let bytes = m.to_bytes().unwrap();
        assert_eq!(detect_version(&bytes).unwrap(), 2);
        let dec = WaiMulti::from_bytes(&bytes).unwrap();
        assert_eq!(dec.manifest.wai, "1.1");
        assert_eq!(dec.renditions.len(), 2);
        assert_eq!(dec.renditions[0], b"NEURAL-TOKENS-2317-BYTES");
        assert_eq!(dec.renditions[1], b"OPUS-PAYLOAD-4411-BYTES");
        assert_eq!(dec.manifest.renditions[0].capability, "wai.neural.encodec32");
        assert_eq!(dec.manifest.renditions[1].capability, "wai.audio.opus");
    }

    #[test]
    fn multi_pick_by_capability() {
        let m = sample_multi();
        let (meta, payload) = m.pick(|c| c == "wai.audio.opus").unwrap();
        assert_eq!(meta.capability, "wai.audio.opus");
        assert_eq!(payload, b"OPUS-PAYLOAD-4411-BYTES");
        // primary = first rendition (deployer-preferred order)
        let (pmeta, _) = m.primary().unwrap();
        assert_eq!(pmeta.capability, "wai.neural.encodec32");
    }

    #[test]
    fn version_detection_and_cross_rejection() {
        let v1 = Wai::new(
            Manifest {
                wai: "1.0".into(), media: "image".into(), intent: "replicate".into(),
                model_requirement: ModelRequirement {
                    capability: "wai.image.png".into(), fallback: None },
                conditioning: Conditioning { kind: "png".into() },
                target: serde_json::Value::Null,
            }, b"x".to_vec(),
        ).to_bytes().unwrap();
        let v2 = sample_multi().to_bytes().unwrap();
        assert_eq!(detect_version(&v1).unwrap(), 1);
        assert_eq!(detect_version(&v2).unwrap(), 2);
        // a v1 reader on a v2 envelope MUST refuse cleanly (no silent
        // fallback — that would hide a major-version mismatch)
        assert!(Wai::from_bytes(&v2).is_err());
        // and vice versa
        assert!(WaiMulti::from_bytes(&v1).is_err());
    }

    #[test]
    fn multi_rejects_truncation() {
        let bytes = sample_multi().to_bytes().unwrap();
        for cut in [3, 7, 30, bytes.len() - 5] {
            assert!(WaiMulti::from_bytes(&bytes[..cut]).is_err(),
                    "truncation at {cut} should be rejected");
        }
    }
}
