//! GGUF/SafeTensors tensor index — metadata extraction without loading data.

use crate::core::{DType, Error, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Metadata for a single tensor in a model file.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// Tensor name (e.g., "model.layers.47.self_attn.q_proj.weight").
    pub name: String,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Data type.
    pub dtype: DType,
    /// Quantization type string (e.g., "Q4_K", "Q3_K_S", "F16", "BF16").
    pub quant_type: String,
    /// Byte offset within the file where tensor data begins.
    pub file_offset: u64,
    /// Size of tensor data in bytes.
    pub size_bytes: u64,
    /// Path to the file containing this tensor.
    pub file_path: PathBuf,
    /// Parsed layer index (None if not a layer tensor, e.g., embeddings).
    pub layer_idx: Option<usize>,
    /// Parsed expert index (None if not an MoE expert tensor).
    pub expert_idx: Option<usize>,
    /// Component type (attention, ffn, moe, embed, norm, lm_head, etc.).
    pub component: String,
}

/// Index of all tensors in a model, with structural metadata.
pub struct ModelIndex {
    /// All tensors by name.
    pub tensors: HashMap<String, TensorMeta>,
    /// Total bytes across all tensors.
    pub total_bytes: u64,
    /// Tensors grouped by layer index.
    layers: HashMap<usize, Vec<String>>,
    /// Tensors grouped by (layer_idx, expert_idx).
    experts: HashMap<(usize, usize), Vec<String>>,
    /// Tensors that aren't part of any layer (embeddings, final norm, lm_head).
    global_tensors: Vec<String>,
    /// GGUF metadata key-value pairs.
    pub metadata: HashMap<String, MetaValue>,
}

/// A metadata value from a GGUF file.
#[derive(Debug, Clone)]
pub enum MetaValue {
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    Bool(bool),
    String(String),
    Array(Vec<MetaValue>),
}

impl ModelIndex {
    pub fn get_tensor(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.get(name)
    }

    pub fn tensors_for_layer(&self, layer_idx: usize) -> Vec<&TensorMeta> {
        self.layers
            .get(&layer_idx)
            .map(|names| names.iter().filter_map(|n| self.tensors.get(n)).collect())
            .unwrap_or_default()
    }

    pub fn tensors_for_expert(&self, layer_idx: usize, expert_idx: usize) -> Vec<&TensorMeta> {
        self.experts
            .get(&(layer_idx, expert_idx))
            .map(|names| names.iter().filter_map(|n| self.tensors.get(n)).collect())
            .unwrap_or_default()
    }

    pub fn global_tensors(&self) -> Vec<&TensorMeta> {
        self.global_tensors
            .iter()
            .filter_map(|n| self.tensors.get(n))
            .collect()
    }

    pub fn layer_count(&self) -> usize {
        self.layers.keys().max().map(|m| m + 1).unwrap_or(0)
    }

    pub fn expert_count(&self) -> usize {
        // First check per-expert tensors (standard MoE naming)
        let per_expert = self.experts.keys().map(|(_, e)| e + 1).max().unwrap_or(0);
        if per_expert > 0 {
            return per_expert;
        }
        // Check fused expert tensors (Nemotron-style: last dim = n_experts)
        for meta in self.tensors.values() {
            if meta.component == "moe.fused_experts" && meta.shape.len() == 3 {
                return meta.shape[2]; // [inner, outer, n_experts]
            }
        }
        0
    }

    /// Count of MoE layers (layers that have expert tensors).
    pub fn moe_layer_count(&self) -> usize {
        let mut moe_layers = std::collections::HashSet::new();
        for meta in self.tensors.values() {
            if meta.component.starts_with("moe.") {
                if let Some(l) = meta.layer_idx {
                    moe_layers.insert(l);
                }
            }
        }
        moe_layers.len()
    }

    /// Count of SSM layers.
    pub fn ssm_layer_count(&self) -> usize {
        let mut ssm_layers = std::collections::HashSet::new();
        for meta in self.tensors.values() {
            if meta.component == "ssm" {
                if let Some(l) = meta.layer_idx {
                    ssm_layers.insert(l);
                }
            }
        }
        ssm_layers.len()
    }
}

/// Parse a tensor name to extract structural metadata.
///
/// Handles common naming conventions:
/// - `model.layers.{N}.self_attn.q_proj.weight` → layer=N, component="attention.q_proj"
/// - `model.layers.{N}.mlp.experts.{E}.gate_proj.weight` → layer=N, expert=E
/// - `model.layers.{N}.block_sparse_moe.experts.{E}.w1.weight` → layer=N, expert=E
/// - `model.embed_tokens.weight` → global, component="embed"
/// - `lm_head.weight` → global, component="lm_head"
/// - `blk.{N}.attn_q.weight` → GGUF-style layer naming
fn parse_tensor_name(name: &str) -> (Option<usize>, Option<usize>, String) {
    let parts: Vec<&str> = name.split('.').collect();

    let mut layer_idx = None;
    let mut expert_idx = None;
    let mut component = String::new();

    // Find layer index
    for (i, part) in parts.iter().enumerate() {
        // HuggingFace style: "model.layers.47.self_attn..."
        if (*part == "layers" || *part == "blk") && i + 1 < parts.len() {
            if let Ok(idx) = parts[i + 1].parse::<usize>() {
                layer_idx = Some(idx);
            }
        }
        // Expert index: "experts.13" or "expert.13" (per-expert tensors)
        if (*part == "experts" || *part == "expert") && i + 1 < parts.len() {
            if let Ok(idx) = parts[i + 1].parse::<usize>() {
                expert_idx = Some(idx);
            }
        }
        // Fused expert tensors: "ffn_down_exps", "ffn_up_exps" (Nemotron-style)
        // These contain ALL experts in one tensor. The last dim is n_experts.
        // We mark expert_idx = Some(0) as a sentinel for "fused expert tensor".
        if part.ends_with("_exps") || part.ends_with("_shexp") {
            // Fused: don't set expert_idx (handled at component level)
        }
    }

    // Determine component type
    let name_lower = name.to_lowercase();
    if name_lower.contains("embed") {
        component = "embed".to_string();
    } else if name_lower.contains("lm_head") || name_lower.contains("output.weight") {
        component = "lm_head".to_string();
    } else if name_lower.contains("final_norm") || name_lower.contains("model.norm") || name_lower.contains("output_norm") {
        component = "final_norm".to_string();
    } else if expert_idx.is_some() {
        component = "moe.expert".to_string();
    } else if name_lower.contains("attn") || name_lower.contains("self_attn") {
        if name_lower.contains("q_proj") || name_lower.contains("attn_q") {
            component = "attention.q_proj".to_string();
        } else if name_lower.contains("k_proj") || name_lower.contains("attn_k") {
            component = "attention.k_proj".to_string();
        } else if name_lower.contains("v_proj") || name_lower.contains("attn_v") {
            component = "attention.v_proj".to_string();
        } else if name_lower.contains("o_proj") || name_lower.contains("attn_output") {
            component = "attention.o_proj".to_string();
        } else {
            component = "attention".to_string();
        }
    } else if name_lower.contains("mlp") || name_lower.contains("ffn") {
        if name_lower.contains("_exps") {
            // Fused expert tensor (Nemotron): ffn_down_exps, ffn_up_exps
            component = "moe.fused_experts".to_string();
        } else if name_lower.contains("_shexp") {
            // Shared expert (Nemotron): ffn_down_shexp, ffn_up_shexp
            component = "moe.shared_expert".to_string();
        } else if name_lower.contains("gate_inp") {
            // MoE gate/router
            component = "moe.gate".to_string();
        } else if name_lower.contains("latent") {
            // Latent MoE projection (Nemotron)
            component = "moe.latent".to_string();
        } else if name_lower.contains("gate") {
            component = "ffn.gate".to_string();
        } else if name_lower.contains("up") || name_lower.contains("w1") {
            component = "ffn.up".to_string();
        } else if name_lower.contains("down") || name_lower.contains("w2") {
            component = "ffn.down".to_string();
        } else {
            component = "ffn".to_string();
        }
    } else if name_lower.contains("norm") || name_lower.contains("layernorm") {
        component = "norm".to_string();
    } else if name_lower.contains("moe") || name_lower.contains("gate") {
        component = "moe.gate".to_string();
    } else if name_lower.contains("ssm") || name_lower.contains("mamba") {
        component = "ssm".to_string();
    } else {
        component = "other".to_string();
    }

    (layer_idx, expert_idx, component)
}

/// GGUF-specific quantization type sizes.
fn ggml_type_size(type_id: u32) -> (u64, &'static str) {
    match type_id {
        0 => (4, "F32"),        // F32: 4 bytes
        1 => (2, "F16"),        // F16: 2 bytes
        2 => (18, "Q4_0"),      // Q4_0: 32 values in 18 bytes (block)
        3 => (20, "Q4_1"),      // Q4_1: 32 values in 20 bytes
        6 => (22, "Q5_0"),      // Q5_0: 32 values in 22 bytes
        7 => (24, "Q5_1"),      // Q5_1: 32 values in 24 bytes
        8 => (34, "Q8_0"),      // Q8_0: 32 values in 34 bytes
        9 => (36, "Q8_1"),      // Q8_1: 32 values in 36 bytes
        10 => (256, "Q2_K"),    // Q2_K: block size 256 bytes for 256 values
        11 => (110, "Q3_K"),    // Q3_K: ~110 bytes per 256 values
        12 => (144, "Q4_K"),    // Q4_K: 144 bytes per 256 values
        13 => (176, "Q5_K"),    // Q5_K: 176 bytes per 256 values
        14 => (210, "Q6_K"),    // Q6_K: 210 bytes per 256 values
        15 => (292, "Q8_K"),    // Q8_K: 292 bytes per 256 values
        _ => (1, "unknown"),
    }
}

/// Block sizes for GGML quantization types.
fn ggml_block_size(type_id: u32) -> u64 {
    match type_id {
        0 | 1 => 1,                         // F32, F16: per-element
        2 | 3 | 6 | 7 | 8 | 9 => 32,       // Q4/Q5/Q8 variants: 32 values/block
        10 | 11 | 12 | 13 | 14 | 15 => 256, // K-quants: 256 values/block
        _ => 1,
    }
}

/// Compute tensor data size from shape and GGML type.
fn compute_tensor_size(shape: &[usize], type_id: u32) -> u64 {
    let n_elements: u64 = shape.iter().map(|&d| d as u64).product();
    if n_elements == 0 {
        return 0;
    }

    let block_size = ggml_block_size(type_id);
    let (type_size, _) = ggml_type_size(type_id);

    if block_size == 1 {
        // Per-element types (F32, F16)
        n_elements * type_size
    } else {
        // Block quantization
        let n_blocks = (n_elements + block_size - 1) / block_size;
        n_blocks * type_size
    }
}

/// Index a GGUF file: read header + tensor info without loading tensor data.
pub fn index_gguf_file(path: &Path) -> Result<ModelIndex> {
    let data = std::fs::read(path).map_err(|e| {
        Error::Io { operation: "read".into(), message: format!("'{}': {}", path.display(), e), #[cfg(feature = "std")] source: None }
    })?;

    if data.len() < 24 {
        return Err(Error::InvalidArgument { name: "gguf".into(), message: "GGUF file too small".into() });
    }

    // Parse header
    let magic = &data[0..4];
    if magic != b"GGUF" {
        return Err(Error::InvalidArgument { name: "gguf".into(), message: format!(
            "invalid GGUF magic: {:?}", magic
        ) });
    }

    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version < 2 || version > 3 {
        return Err(Error::InvalidArgument { name: "gguf".into(), message: format!(
            "unsupported GGUF version: {}", version
        ) });
    }

    let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    let metadata_kv_count = u64::from_le_bytes(data[16..24].try_into().unwrap()) as usize;

    let mut offset = 24usize;
    let mut metadata = HashMap::new();

    // Parse metadata key-value pairs
    for _ in 0..metadata_kv_count {
        if offset + 8 > data.len() { break; }
        let (key, val, new_offset) = parse_gguf_kv(&data, offset, version)?;
        metadata.insert(key, val);
        offset = new_offset;
    }

    // Parse tensor infos
    let mut tensor_infos = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        if offset + 8 > data.len() { break; }
        let (info, new_offset) = parse_gguf_tensor_info(&data, offset, version)?;
        tensor_infos.push(info);
        offset = new_offset;
    }

    // Alignment (default 32 bytes)
    let alignment = match metadata.get("general.alignment") {
        Some(MetaValue::U32(a)) => *a as usize,
        Some(MetaValue::U64(a)) => *a as usize,
        _ => 32,
    };

    // Data section starts after alignment
    let data_offset = (offset + alignment - 1) / alignment * alignment;

    // Build index
    let mut tensors = HashMap::new();
    let mut layers: HashMap<usize, Vec<String>> = HashMap::new();
    let mut experts: HashMap<(usize, usize), Vec<String>> = HashMap::new();
    let mut global_tensors = Vec::new();
    let mut total_bytes = 0u64;

    for info in tensor_infos {
        let tensor_data_offset = data_offset as u64 + info.offset;
        let size_bytes = compute_tensor_size(&info.shape, info.type_id);
        let (_, quant_name) = ggml_type_size(info.type_id);
        let (layer_idx, expert_idx, component) = parse_tensor_name(&info.name);

        let dtype = match info.type_id {
            0 => DType::F32,
            1 => DType::F16,
            _ => DType::U8, // Quantized types represented as raw bytes
        };

        let meta = TensorMeta {
            name: info.name.clone(),
            shape: info.shape.clone(),
            dtype,
            quant_type: quant_name.to_string(),
            file_offset: tensor_data_offset,
            size_bytes,
            file_path: path.to_path_buf(),
            layer_idx,
            expert_idx,
            component,
        };

        total_bytes += size_bytes;

        // Group by layer/expert
        match (layer_idx, expert_idx) {
            (Some(l), Some(e)) => {
                experts.entry((l, e)).or_default().push(info.name.clone());
                layers.entry(l).or_default().push(info.name.clone());
            }
            (Some(l), None) => {
                layers.entry(l).or_default().push(info.name.clone());
            }
            _ => {
                global_tensors.push(info.name.clone());
            }
        }

        tensors.insert(info.name, meta);
    }

    Ok(ModelIndex {
        tensors,
        total_bytes,
        layers,
        experts,
        global_tensors,
        metadata,
    })
}

/// Index sharded GGUF files in a directory.
pub fn index_gguf_sharded(dir: &Path) -> Result<ModelIndex> {
    let mut shard_paths: Vec<PathBuf> = Vec::new();

    // Find all .gguf files, sorted by name (shard order)
    for entry in std::fs::read_dir(dir).map_err(|e| Error::Io {
        operation: "read_dir".into(), message: format!("{}: {}", dir.display(), e),
        #[cfg(feature = "std")] source: None,
    })? {
        let entry = entry.map_err(|e| Error::Io {
            operation: "read_dir".into(), message: e.to_string(),
            #[cfg(feature = "std")] source: None,
        })?;
        let path = entry.path();
        if path.extension().map(|e| e == "gguf").unwrap_or(false) {
            shard_paths.push(path);
        }
    }

    shard_paths.sort();

    if shard_paths.is_empty() {
        return Err(Error::ModelLoad {
            model: dir.display().to_string(),
            message: "no .gguf files found".into(),
            #[cfg(feature = "std")]
            source: None,
        });
    }

    // If only one shard, just index it directly
    if shard_paths.len() == 1 {
        return index_gguf_file(&shard_paths[0]);
    }

    // Multiple shards: index each and merge
    let mut combined = index_gguf_file(&shard_paths[0])?;

    for shard_path in &shard_paths[1..] {
        let shard_index = index_gguf_file(shard_path)?;

        for (name, meta) in shard_index.tensors {
            combined.total_bytes += meta.size_bytes;

            match (meta.layer_idx, meta.expert_idx) {
                (Some(l), Some(e)) => {
                    combined.experts.entry((l, e)).or_default().push(name.clone());
                    combined.layers.entry(l).or_default().push(name.clone());
                }
                (Some(l), None) => {
                    combined.layers.entry(l).or_default().push(name.clone());
                }
                _ => {
                    combined.global_tensors.push(name.clone());
                }
            }

            combined.tensors.insert(name, meta);
        }

        // Merge metadata (later shards don't usually have unique metadata)
        for (k, v) in shard_index.metadata {
            combined.metadata.entry(k).or_insert(v);
        }
    }

    Ok(combined)
}

// --- GGUF parsing helpers ---

struct GgufTensorInfo {
    name: String,
    shape: Vec<usize>,
    type_id: u32,
    offset: u64,
}

fn read_string(data: &[u8], offset: usize, _version: u32) -> Result<(String, usize)> {
    if offset + 8 > data.len() {
        return Err(Error::InvalidArgument { name: "gguf".into(), message: "string length out of bounds".into() });
    }
    let len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
    let str_start = offset + 8;
    if str_start + len > data.len() {
        return Err(Error::InvalidArgument { name: "gguf".into(), message: format!(
            "string data out of bounds: offset={}, len={}, file_size={}",
            str_start, len, data.len()
        ) });
    }
    let s = String::from_utf8_lossy(&data[str_start..str_start + len]).to_string();
    Ok((s, str_start + len))
}

fn parse_gguf_kv(data: &[u8], offset: usize, version: u32) -> Result<(String, MetaValue, usize)> {
    let (key, mut pos) = read_string(data, offset, version)?;

    if pos + 4 > data.len() {
        return Err(Error::InvalidArgument { name: "gguf".into(), message: "kv type out of bounds".into() });
    }
    let value_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;

    let (val, new_pos) = parse_gguf_value(data, pos, value_type, version)?;
    Ok((key, val, new_pos))
}

fn parse_gguf_value(data: &[u8], offset: usize, type_id: u32, version: u32) -> Result<(MetaValue, usize)> {
    match type_id {
        // UINT8
        0 => {
            Ok((MetaValue::U32(data[offset] as u32), offset + 1))
        }
        // INT8
        1 => {
            Ok((MetaValue::I32(data[offset] as i8 as i32), offset + 1))
        }
        // UINT16
        2 => {
            let v = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
            Ok((MetaValue::U32(v as u32), offset + 2))
        }
        // INT16
        3 => {
            let v = i16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
            Ok((MetaValue::I32(v as i32), offset + 2))
        }
        // UINT32
        4 => {
            let v = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            Ok((MetaValue::U32(v), offset + 4))
        }
        // INT32
        5 => {
            let v = i32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            Ok((MetaValue::I32(v), offset + 4))
        }
        // FLOAT32
        6 => {
            let v = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            Ok((MetaValue::F32(v), offset + 4))
        }
        // BOOL
        7 => {
            Ok((MetaValue::Bool(data[offset] != 0), offset + 1))
        }
        // STRING
        8 => {
            let (s, new_pos) = read_string(data, offset, version)?;
            Ok((MetaValue::String(s), new_pos))
        }
        // ARRAY
        9 => {
            if offset + 12 > data.len() {
                return Err(Error::InvalidArgument { name: "gguf".into(), message: "array header out of bounds".into() });
            }
            let elem_type = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            let count = u64::from_le_bytes(data[offset + 4..offset + 12].try_into().unwrap()) as usize;
            let mut pos = offset + 12;
            let mut items = Vec::with_capacity(count.min(1024)); // Cap to prevent OOM
            for _ in 0..count {
                let (val, new_pos) = parse_gguf_value(data, pos, elem_type, version)?;
                items.push(val);
                pos = new_pos;
            }
            Ok((MetaValue::Array(items), pos))
        }
        // UINT64
        10 => {
            let v = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            Ok((MetaValue::U64(v), offset + 8))
        }
        // INT64
        11 => {
            let v = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            Ok((MetaValue::U64(v as u64), offset + 8))
        }
        // FLOAT64
        12 => {
            let _v = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            Ok((MetaValue::F32(_v as f32), offset + 8))
        }
        _ => {
            Err(Error::InvalidArgument { name: "gguf".into(), message: format!("unknown GGUF value type: {}", type_id) })
        }
    }
}

fn parse_gguf_tensor_info(data: &[u8], offset: usize, version: u32) -> Result<(GgufTensorInfo, usize)> {
    let (name, mut pos) = read_string(data, offset, version)?;

    if pos + 4 > data.len() {
        return Err(Error::InvalidArgument { name: "gguf".into(), message: "tensor n_dims out of bounds".into() });
    }
    let n_dims = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    let mut shape = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        if pos + 8 > data.len() {
            return Err(Error::InvalidArgument { name: "gguf".into(), message: "tensor dim out of bounds".into() });
        }
        let dim = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        shape.push(dim);
        pos += 8;
    }

    if pos + 4 > data.len() {
        return Err(Error::InvalidArgument { name: "gguf".into(), message: "tensor type out of bounds".into() });
    }
    let type_id = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;

    if pos + 8 > data.len() {
        return Err(Error::InvalidArgument { name: "gguf".into(), message: "tensor offset out of bounds".into() });
    }
    let tensor_offset = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;

    Ok((GgufTensorInfo {
        name,
        shape,
        type_id,
        offset: tensor_offset,
    }, pos))
}
