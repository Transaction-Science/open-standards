//! WAI envelope parser + capability dispatch for the browser.
//!
//! What this crate IS:
//!   - A wasm-bindgen surface over the standard's container format
//!     (`WAI1` magic + length-prefixed manifest + length-prefixed payload)
//!   - A capability menu (the registered `wai.*` strings) the JS shim
//!     uses to route to its handler map
//!
//! What this crate is NOT:
//!   - A codec backend. The sink-side compute fabric is heterogeneous
//!     (WASM, WebGPU, WebNN, native ONNX, transformers.js, custom). The
//!     deployer picks. WAI dispatches to whatever handler the JS shim
//!     registers for a given capability.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

const MAGIC: &[u8; 4] = b"WAI1";
const MAGIC_V2: &[u8; 4] = b"WAI2";

// ---- Capability menu (kept in sync with wai-rs/src/codecs/mod.rs) ----
// Exported so JS can match capability strings without typing them
// from memory. This is the same menu as the native SDK.
#[wasm_bindgen]
pub fn known_capabilities() -> Box<[JsValue]> {
    [
        // image
        "wai.image.png", "wai.image.jpeg", "wai.image.avif", "wai.image.jxl",
        // audio
        "wai.audio.opus", "wai.audio.flac",
        // video
        "wai.video.av1", "wai.video.av1.lossless",
        // text
        "wai.text.zstd", "wai.text.xz",
        // neural (sink-installed model) — see SPEC.md §5 "Neural capabilities"
        "wai.neural.encodec32",      // Meta EnCodec, 32 kHz audio
        "wai.neural.dac",            // Descript Audio Codec, 44.1 kHz music
        "wai.neural.mimi",           // Kyutai Mimi, real-time speech
        "wai.neural.wavtokenizer",   // WavTokenizer, ultra-low bitrate audio
        "wai.neural.bmshj2018",        // bmshj2018-factorized image codec (~30x via zstd-packed latents)
        "wai.neural.video_bmshj2018",  // per-frame bmshj2018 video, browser-decodable
        "wai.neural.glc",              // (future) Generative Latent Coding, ultra-low bpp images
        "wai.neural.dcvc_rt",          // DCVC-RT (native-sink only, requires CUDA)
    ].iter().map(|s| JsValue::from_str(s)).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelRequirement {
    capability: String,
    #[serde(default)]
    fallback: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Conditioning {
    kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    wai: String,
    media: String,
    intent: String,
    model_requirement: ModelRequirement,
    conditioning: Conditioning,
    #[serde(default)]
    target: serde_json::Value,
}

/// Parsed WAI envelope returned to JS. The payload is a separate
/// Uint8Array (zero-copy view via wasm-bindgen) so handlers can feed
/// it straight into whatever decoder they choose.
#[wasm_bindgen]
pub struct WaiEnvelope {
    manifest_json: String,
    payload: Vec<u8>,
    capability: String,
    fallback: Option<String>,
    media: String,
    intent: String,
    kind: String,
}

#[wasm_bindgen]
impl WaiEnvelope {
    #[wasm_bindgen(getter)]
    pub fn capability(&self) -> String { self.capability.clone() }

    #[wasm_bindgen(getter)]
    pub fn fallback(&self) -> Option<String> { self.fallback.clone() }

    #[wasm_bindgen(getter)]
    pub fn media(&self) -> String { self.media.clone() }

    #[wasm_bindgen(getter)]
    pub fn intent(&self) -> String { self.intent.clone() }

    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String { self.kind.clone() }

    #[wasm_bindgen(getter)]
    pub fn manifest(&self) -> String { self.manifest_json.clone() }

    /// Returns the payload bytes — the codec-specific bytes the
    /// handler for `capability` knows how to consume.
    #[wasm_bindgen(getter)]
    pub fn payload(&self) -> Box<[u8]> { self.payload.clone().into_boxed_slice() }

    #[wasm_bindgen(getter)]
    pub fn payload_len(&self) -> usize { self.payload.len() }
}

/// Parse a `Uint8Array` containing a complete WAI envelope. Throws a
/// JS Error with a descriptive message on bad magic / truncation /
/// invalid manifest JSON — the JS shim catches and surfaces it.
#[wasm_bindgen(js_name = parse)]
pub fn parse(bytes: &[u8]) -> Result<WaiEnvelope, JsError> {
    // Bounds-checked every step (same defensive pattern as wai-rs
    // container.rs — fuzz-tested).
    let need = |off: usize, n: usize, what: &str| -> Result<(), JsError> {
        if off.saturating_add(n) > bytes.len() {
            Err(JsError::new(&format!(
                "truncated WAI ({what} needs {n} bytes at offset {off}, have {})",
                bytes.len())))
        } else { Ok(()) }
    };
    need(0, 4, "magic")?;
    if &bytes[..4] != MAGIC {
        return Err(JsError::new("not a WAI v1 file (magic mismatch)"));
    }
    need(4, 4, "manifest length")?;
    let ml = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    need(8, ml, "manifest body")?;
    let m: Manifest = serde_json::from_slice(&bytes[8..8 + ml])
        .map_err(|e| JsError::new(&format!("manifest parse: {e}")))?;
    let p_off = 8 + ml;
    need(p_off, 4, "payload length")?;
    let pl = u32::from_le_bytes(bytes[p_off..p_off + 4].try_into().unwrap()) as usize;
    need(p_off + 4, pl, "payload body")?;
    let payload = bytes[p_off + 4..p_off + 4 + pl].to_vec();

    Ok(WaiEnvelope {
        manifest_json: String::from_utf8_lossy(&bytes[8..8 + ml]).into_owned(),
        payload,
        capability: m.model_requirement.capability,
        fallback: m.model_requirement.fallback,
        media: m.media,
        intent: m.intent,
        kind: m.conditioning.kind,
    })
}

/// Build a WAI envelope from a manifest JSON string + a payload byte
/// array. Mirrors `wai_envelope_pack` from the native SDK so JS code
/// can produce .wai files in the browser too.
#[wasm_bindgen(js_name = pack)]
pub fn pack(manifest_json: &str, payload: &[u8]) -> Result<Box<[u8]>, JsError> {
    let _: Manifest = serde_json::from_str(manifest_json)
        .map_err(|e| JsError::new(&format!("manifest parse: {e}")))?;
    let mb = manifest_json.as_bytes();
    let mut out = Vec::with_capacity(12 + mb.len() + payload.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(mb.len() as u32).to_le_bytes());
    out.extend_from_slice(mb);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    Ok(out.into_boxed_slice())
}

// ---- v1.1 (WAI2): multi-rendition envelope ----------------------

/// Sniff the magic. Returns 1 for WAI1, 2 for WAI2, throws otherwise.
/// Use this to route between `parse` and `parse_multi`.
#[wasm_bindgen(js_name = detectVersion)]
pub fn detect_version(bytes: &[u8]) -> Result<u8, JsError> {
    if bytes.len() < 4 {
        return Err(JsError::new("too short for magic"));
    }
    match &bytes[..4] {
        b"WAI1" => Ok(1),
        b"WAI2" => Ok(2),
        _ => Err(JsError::new("unknown magic (not WAI1/WAI2)")),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RenditionMeta {
    capability: String,
    kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManifestV2 {
    wai: String,
    media: String,
    intent: String,
    renditions: Vec<RenditionMeta>,
    #[serde(default)]
    target: serde_json::Value,
}

/// One alternative rendition within a WAI2 multi-rendition envelope.
/// `payload` is the codec bytes the handler for `capability` consumes.
#[wasm_bindgen]
pub struct Rendition {
    capability: String,
    kind: String,
    payload: Vec<u8>,
}

#[wasm_bindgen]
impl Rendition {
    #[wasm_bindgen(getter)]
    pub fn capability(&self) -> String { self.capability.clone() }
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String { self.kind.clone() }
    #[wasm_bindgen(getter)]
    pub fn payload(&self) -> Box<[u8]> { self.payload.clone().into_boxed_slice() }
    #[wasm_bindgen(getter)]
    pub fn payload_len(&self) -> usize { self.payload.len() }
}

/// Parsed WAI2 envelope. Hand the result of `renditions()` to the JS
/// shim's policy picker, then dispatch the chosen rendition's payload
/// to the handler registered for its capability.
#[wasm_bindgen]
pub struct WaiMulti {
    manifest_json: String,
    media: String,
    intent: String,
    renditions: Vec<Rendition>,
}

#[wasm_bindgen]
impl WaiMulti {
    #[wasm_bindgen(getter)]
    pub fn manifest(&self) -> String { self.manifest_json.clone() }
    #[wasm_bindgen(getter)]
    pub fn media(&self) -> String { self.media.clone() }
    #[wasm_bindgen(getter)]
    pub fn intent(&self) -> String { self.intent.clone() }
    #[wasm_bindgen(getter)]
    pub fn n_renditions(&self) -> usize { self.renditions.len() }

    /// Returns the i-th rendition (0-indexed by deployer-preferred
    /// order). Throws on out-of-range index.
    #[wasm_bindgen(js_name = rendition)]
    pub fn rendition(&self, i: usize) -> Result<Rendition, JsError> {
        self.renditions.get(i)
            .map(|r| Rendition { capability: r.capability.clone(),
                                 kind: r.kind.clone(),
                                 payload: r.payload.clone() })
            .ok_or_else(|| JsError::new(&format!(
                "rendition index {i} out of range (have {})", self.renditions.len())))
    }

    /// Return the JSON-array list of `{capability, kind}` metadata so
    /// the JS shim's policy picker can score them without copying payloads.
    #[wasm_bindgen(js_name = renditionsMeta)]
    pub fn renditions_meta(&self) -> Result<JsValue, JsError> {
        let metas: Vec<RenditionMeta> = self.renditions.iter()
            .map(|r| RenditionMeta { capability: r.capability.clone(),
                                     kind: r.kind.clone() })
            .collect();
        serde_wasm_bindgen::to_value(&metas)
            .map_err(|e| JsError::new(&format!("{e}")))
    }
}

#[wasm_bindgen(js_name = parseMulti)]
pub fn parse_multi(bytes: &[u8]) -> Result<WaiMulti, JsError> {
    let need = |off: usize, n: usize, what: &str| -> Result<(), JsError> {
        if off.saturating_add(n) > bytes.len() {
            Err(JsError::new(&format!(
                "truncated WAI2 ({what} needs {n} bytes at offset {off}, have {})",
                bytes.len())))
        } else { Ok(()) }
    };
    need(0, 4, "magic")?;
    if &bytes[..4] != MAGIC_V2 {
        return Err(JsError::new("not a WAI2 file (magic mismatch)"));
    }
    need(4, 4, "manifest length")?;
    let ml = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    need(8, ml, "manifest body")?;
    let manifest: ManifestV2 = serde_json::from_slice(&bytes[8..8 + ml])
        .map_err(|e| JsError::new(&format!("manifest parse: {e}")))?;

    let mut o = 8 + ml;
    need(o, 2, "rendition count")?;
    let n = u16::from_le_bytes(bytes[o..o + 2].try_into().unwrap()) as usize;
    o += 2;
    if n != manifest.renditions.len() {
        return Err(JsError::new(&format!(
            "rendition table count {n} != manifest renditions {}",
            manifest.renditions.len())));
    }
    need(o, n * 8, "rendition table")?;
    let mut entries: Vec<(u32, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        let off = u32::from_le_bytes(bytes[o + i * 8..o + i * 8 + 4].try_into().unwrap());
        let len = u32::from_le_bytes(bytes[o + i * 8 + 4..o + i * 8 + 8].try_into().unwrap());
        entries.push((off, len));
    }
    o += n * 8;
    let payload_block_start = o;
    let total: usize = entries.iter().map(|(_, l)| *l as usize).sum();
    need(payload_block_start, total, "payload block")?;

    let mut expected = 0u32;
    let mut renditions = Vec::with_capacity(n);
    for ((off, len), meta) in entries.iter().zip(manifest.renditions.iter()) {
        if *off != expected {
            return Err(JsError::new(&format!(
                "rendition offset {off} not contiguous (expected {expected})")));
        }
        let start = payload_block_start + *off as usize;
        let end = start + *len as usize;
        renditions.push(Rendition {
            capability: meta.capability.clone(),
            kind: meta.kind.clone(),
            payload: bytes[start..end].to_vec(),
        });
        expected = expected.checked_add(*len).ok_or_else(||
            JsError::new("offset arithmetic overflows"))?;
    }
    Ok(WaiMulti {
        manifest_json: String::from_utf8_lossy(&bytes[8..8 + ml]).into_owned(),
        media: manifest.media,
        intent: manifest.intent,
        renditions,
    })
}

/// Pack a WAI2 envelope from a manifest JSON + array of (capability,
/// kind, payload-bytes) tuples. Manifest's `renditions` array MUST be
/// the same length and order as the provided payloads.
#[wasm_bindgen(js_name = packMulti)]
pub fn pack_multi(manifest_json: &str, payloads: js_sys::Array) -> Result<Box<[u8]>, JsError> {
    let manifest: ManifestV2 = serde_json::from_str(manifest_json)
        .map_err(|e| JsError::new(&format!("manifest parse: {e}")))?;
    let n = payloads.length() as usize;
    if n != manifest.renditions.len() {
        return Err(JsError::new(&format!(
            "{} payloads but manifest declares {} renditions",
            n, manifest.renditions.len())));
    }
    if n > u16::MAX as usize {
        return Err(JsError::new("too many renditions for u16 count field"));
    }
    let mut raw_payloads: Vec<Vec<u8>> = Vec::with_capacity(n);
    for i in 0..n {
        let val = payloads.get(i as u32);
        let arr: js_sys::Uint8Array = val.dyn_into()
            .map_err(|_| JsError::new(&format!(
                "payload {i} is not a Uint8Array")))?;
        raw_payloads.push(arr.to_vec());
    }
    let mb = manifest_json.as_bytes();
    let table_bytes = 2 + n * 8;
    let payload_bytes: usize = raw_payloads.iter().map(|p| p.len()).sum();
    let mut out = Vec::with_capacity(8 + mb.len() + table_bytes + payload_bytes);
    out.extend_from_slice(MAGIC_V2);
    out.extend_from_slice(&(mb.len() as u32).to_le_bytes());
    out.extend_from_slice(mb);
    out.extend_from_slice(&(n as u16).to_le_bytes());
    let mut off: u32 = 0;
    for p in &raw_payloads {
        out.extend_from_slice(&off.to_le_bytes());
        out.extend_from_slice(&(p.len() as u32).to_le_bytes());
        off = off.checked_add(p.len() as u32).ok_or_else(||
            JsError::new("payload sum overflows u32"))?;
    }
    for p in &raw_payloads {
        out.extend_from_slice(p);
    }
    Ok(out.into_boxed_slice())
}
