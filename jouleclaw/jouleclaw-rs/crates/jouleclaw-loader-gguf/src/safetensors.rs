//! Safetensors loader → [`GgufModel`].
//!
//! Most HuggingFace checkpoints (Gemma, Llama, Mistral, …) ship as
//! `.safetensors` long before a GGUF conversion exists. This module
//! reads that format directly into the *same* `GgufModel` structure
//! the rest of the substrate consumes, so once an arch adapter maps
//! the tensor names + config, those models flow through
//! `LlamaConfig` / `tensor_from_gguf` / `build_block` unchanged.
//!
//! ## Format
//!
//! A `.safetensors` file is:
//!   1. `u64` LE — header byte length `N`
//!   2. `N` bytes — a JSON object: `name → { dtype, shape,
//!      data_offsets:[start,end] }`, plus an optional `"__metadata__"`
//!      string→string map. `start`/`end` are relative to the start of
//!      the data blob (byte `8 + N`).
//!   3. the rest — the raw tensor data blob.
//!
//! Sharded models ship a `*.safetensors.index.json`:
//!   `{ "metadata": {...}, "weight_map": { tensor → shard_file } }`.
//!   We load each referenced shard and merge.
//!
//! ## Scope
//!
//! This is the *loader* — faithful format reading + dtype widening
//! (F32/F16/BF16 → kept as `GgmlType` for `tensor_from_gguf`). Tensor
//! names are preserved verbatim (HF naming, e.g.
//! `model.layers.0.self_attn.q_proj.weight`). Mapping those to the
//! GGUF block convention and synthesising arch metadata from
//! `config.json` is the per-architecture adapter's job (next step) —
//! deliberately *not* done here so the loader stays generic.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::{GgmlType, GgufModel, GgufValue, ParseError, TensorInfo};

fn st_dtype(s: &str) -> Option<GgmlType> {
    match s {
        "F32" => Some(GgmlType::F32),
        "F16" => Some(GgmlType::F16),
        "BF16" => Some(GgmlType::BF16),
        "F64" => Some(GgmlType::F64),
        "I32" => Some(GgmlType::I32),
        "I16" => Some(GgmlType::I16),
        "I8" => Some(GgmlType::I8),
        // F8 / bool / U-types: no widening path yet — surfaced as an
        // explicit error rather than silently mis-loaded.
        _ => None,
    }
}

/// Parse one `.safetensors` blob (already in memory). Appends each
/// tensor's bytes to `out_data` (recording its absolute offset) and
/// pushes a [`TensorInfo`]. Also returns any `__metadata__` string
/// entries.
fn parse_one(
    buf: &[u8],
    out_data: &mut Vec<u8>,
    out_tensors: &mut Vec<TensorInfo>,
    out_meta: &mut HashMap<String, GgufValue>,
) -> Result<(), ParseError> {
    if buf.len() < 8 {
        return Err(ParseError::Truncated);
    }
    let header_len = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_len;
    if buf.len() < header_end {
        return Err(ParseError::Truncated);
    }
    let header: serde_json::Value =
        serde_json::from_slice(&buf[8..header_end])
            .map_err(|e| ParseError::Safetensors(format!("header JSON: {e}")))?;
    let obj = header.as_object()
        .ok_or_else(|| ParseError::Safetensors("header is not an object".into()))?;
    let data_base = header_end;

    for (name, v) in obj {
        if name == "__metadata__" {
            if let Some(m) = v.as_object() {
                for (k, mv) in m {
                    if let Some(s) = mv.as_str() {
                        out_meta.insert(
                            format!("safetensors.{k}"),
                            GgufValue::String(s.to_string()),
                        );
                    }
                }
            }
            continue;
        }
        let t = v.as_object().ok_or_else(|| {
            ParseError::Safetensors(format!("tensor {name} entry not an object"))
        })?;
        let dtype_s = t.get("dtype").and_then(|x| x.as_str()).ok_or_else(|| {
            ParseError::Safetensors(format!("{name}: missing dtype"))
        })?;
        let dtype = st_dtype(dtype_s).ok_or_else(|| {
            ParseError::Safetensors(format!("{name}: unsupported dtype {dtype_s}"))
        })?;
        let shape: Vec<u64> = t.get("shape")
            .and_then(|x| x.as_array())
            .ok_or_else(|| ParseError::Safetensors(format!("{name}: missing shape")))?
            .iter()
            .map(|d| d.as_u64().unwrap_or(0))
            .collect();
        let offs = t.get("data_offsets")
            .and_then(|x| x.as_array())
            .ok_or_else(|| ParseError::Safetensors(format!("{name}: missing data_offsets")))?;
        if offs.len() != 2 {
            return Err(ParseError::Safetensors(format!("{name}: bad data_offsets")));
        }
        let s = offs[0].as_u64().unwrap_or(0) as usize;
        let e = offs[1].as_u64().unwrap_or(0) as usize;
        if e < s || data_base + e > buf.len() {
            return Err(ParseError::Safetensors(format!(
                "{name}: data_offsets [{s},{e}] out of bounds")));
        }
        // Append bytes to the unified data blob; offset is the
        // position within that blob (GgufModel convention).
        let abs_off = out_data.len() as u64;
        out_data.extend_from_slice(&buf[data_base + s..data_base + e]);
        out_tensors.push(TensorInfo {
            name: name.clone(),
            shape,
            dtype,
            offset: abs_off,
        });
    }
    Ok(())
}

/// Load a single `.safetensors` file (no sharding) into a
/// [`GgufModel`]. `metadata` holds only the file's `__metadata__`
/// entries (as `safetensors.*`); arch hyperparameters come from
/// `config.json` via the per-arch adapter, not from here.
pub fn read_safetensors_file<P: AsRef<Path>>(
    path: P,
) -> Result<GgufModel, ParseError> {
    let mut f = std::fs::File::open(path.as_ref()).map_err(crate::io_err)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(crate::io_err)?;

    let mut data = Vec::new();
    let mut tensors = Vec::new();
    let mut metadata = HashMap::new();
    parse_one(&buf, &mut data, &mut tensors, &mut metadata)?;
    Ok(GgufModel { version: 0, metadata, tensors, buf: crate::GgufBuffer::Owned(data) })
}

/// Load a (possibly sharded) safetensors model. `path` may point at
/// either a single `*.safetensors` or a `*.safetensors.index.json`.
/// For the index form, every shard in `weight_map` is loaded and
/// merged in first-seen order.
pub fn read_safetensors_model<P: AsRef<Path>>(
    path: P,
) -> Result<GgufModel, ParseError> {
    let path = path.as_ref();
    let is_index = path.extension().map(|e| e == "json").unwrap_or(false)
        || path.to_string_lossy().ends_with(".index.json");
    if !is_index {
        return read_safetensors_file(path);
    }

    let idx_txt = std::fs::read_to_string(path).map_err(crate::io_err)?;
    let idx: serde_json::Value = serde_json::from_str(&idx_txt)
        .map_err(|e| ParseError::Safetensors(format!("index JSON: {e}")))?;
    let wm = idx.get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| ParseError::Safetensors("index missing weight_map".into()))?;

    // Distinct shard files, in first-seen order.
    let dir: PathBuf = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let mut shard_order: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for v in wm.values() {
        if let Some(s) = v.as_str() {
            if seen.insert(s) {
                shard_order.push(s.to_string());
            }
        }
    }

    let mut data = Vec::new();
    let mut tensors = Vec::new();
    let mut metadata = HashMap::new();
    for shard in &shard_order {
        let mut f = std::fs::File::open(dir.join(shard))
            .map_err(|e| ParseError::Safetensors(format!("shard {shard}: {e}")))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).map_err(crate::io_err)?;
        parse_one(&buf, &mut data, &mut tensors, &mut metadata)?;
    }
    Ok(GgufModel { version: 0, metadata, tensors, buf: crate::GgufBuffer::Owned(data) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal in-memory `.safetensors` blob: header u64 +
    /// JSON + data. Used to round-trip the parser without a real file.
    fn make_st(entries: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
        let mut data = Vec::new();
        let mut hdr = serde_json::Map::new();
        for (name, dt, shape, bytes) in entries {
            let s = data.len();
            data.extend_from_slice(bytes);
            let e = data.len();
            hdr.insert(name.to_string(), serde_json::json!({
                "dtype": dt,
                "shape": shape,
                "data_offsets": [s, e],
            }));
        }
        let hj = serde_json::to_vec(&serde_json::Value::Object(hdr)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(hj.len() as u64).to_le_bytes());
        out.extend_from_slice(&hj);
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn parses_f32_and_bf16_roundtrip() {
        // tensor A: F32 [2] = [1.0, -2.5]
        let a_bytes: Vec<u8> = [1.0f32, -2.5]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        // tensor B: BF16 [2] = high-16-bits of [3.0, 0.5]
        let b_bytes: Vec<u8> = [3.0f32, 0.5]
            .iter()
            .flat_map(|v| {
                let hi = (v.to_bits() >> 16) as u16;
                hi.to_le_bytes()
            })
            .collect();
        let blob = make_st(&[
            ("w.a", "F32", vec![2], a_bytes),
            ("w.b", "BF16", vec![2], b_bytes),
        ]);

        let mut data = Vec::new();
        let mut tensors = Vec::new();
        let mut meta = HashMap::new();
        parse_one(&blob, &mut data, &mut tensors, &mut meta).unwrap();
        assert_eq!(tensors.len(), 2);

        let model = GgufModel { version: 0, metadata: meta, tensors, buf: crate::GgufBuffer::Owned(data) };
        let a = model.tensor_by_name("w.a").unwrap();
        assert_eq!(a.dtype, GgmlType::F32);
        let at = crate::tensor_from_gguf(&model, a).unwrap();
        assert_eq!(at.as_f32_vec(), vec![1.0, -2.5]);

        let b = model.tensor_by_name("w.b").unwrap();
        assert_eq!(b.dtype, GgmlType::BF16);
        let bt = crate::tensor_from_gguf(&model, b).unwrap();
        // BF16 truncates the low 16 mantissa bits; 3.0 and 0.5 are
        // exactly representable, so widening is lossless here.
        assert_eq!(bt.as_f32_vec(), vec![3.0, 0.5]);
    }

    #[test]
    fn rejects_unsupported_dtype_and_truncation() {
        let blob = make_st(&[("x", "F8_E4M3", vec![1], vec![0u8])]);
        let mut d = Vec::new(); let mut t = Vec::new(); let mut m = HashMap::new();
        assert!(matches!(
            parse_one(&blob, &mut d, &mut t, &mut m),
            Err(ParseError::Safetensors(_))));
        // Truncated (header claims more than present).
        let bad = vec![0xFFu8; 4];
        let mut d2 = Vec::new(); let mut t2 = Vec::new(); let mut m2 = HashMap::new();
        assert!(matches!(
            parse_one(&bad, &mut d2, &mut t2, &mut m2),
            Err(ParseError::Truncated)));
    }

    #[test]
    fn metadata_entries_captured() {
        let mut hdr = serde_json::Map::new();
        hdr.insert("__metadata__".into(),
            serde_json::json!({ "format": "pt" }));
        hdr.insert("w".into(), serde_json::json!({
            "dtype": "F32", "shape": [1], "data_offsets": [0, 4],
        }));
        let hj = serde_json::to_vec(&serde_json::Value::Object(hdr)).unwrap();
        let mut blob = (hj.len() as u64).to_le_bytes().to_vec();
        blob.extend_from_slice(&hj);
        blob.extend_from_slice(&1.0f32.to_le_bytes());

        let mut d = Vec::new(); let mut t = Vec::new(); let mut m = HashMap::new();
        parse_one(&blob, &mut d, &mut t, &mut m).unwrap();
        assert_eq!(m.get("safetensors.format").and_then(|v| v.as_string()),
            Some("pt"));
        assert_eq!(t.len(), 1);
    }
}
