// NEON intrinsics + raw-pointer SIMD in `matmul_q8_0` and
// `kv_cache_inplace` were written under Rust 2021 where `unsafe fn`
// implicitly carried the unsafe context. Rust 2024 requires explicit
// `unsafe { }` blocks even inside `unsafe fn`. Suppressing here while
// the kernels stand untouched; the proper fix is wrapping each
// intrinsic call individually — tracked as a kernel-quality task,
// not standards work.
#![allow(unsafe_op_in_unsafe_fn)]

//! # jouleclaw-loader-gguf
//!
//! GGUF format parser. GGUF is the de facto interchange format for open-weight
//! LLMs (Llama, Mistral, Qwen, etc.) maintained by the llama.cpp community.
//!
//! Phase 1.3 supports:
//! - GGUF v3 layout (the version most distributions ship)
//! - F32 and F16 tensor types
//! - All metadata value types
//!
//! Phase 1.4 adds:
//! - Llama-style architecture loader (`llama` module)
//! - Synthetic model factory for tests (`synthetic` module)
//!
//! Phase 1.5+ will add:
//! - Q4_K, Q5_K, Q8_0 quantized tensor types
//! - mmap-backed tensor reads (zero-copy)
//! - Streaming load for large models
//!
//! Reference:
//! https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

pub mod llama;
pub mod synthetic;
pub mod dequant;
pub mod matmul_q8_0;
pub mod tokenizer;
pub mod sample;
pub mod kv_cache;
pub mod kv_cache_inplace;
pub mod decode;
pub mod safetensors;
pub mod gemma;
pub mod gemma4;
pub mod gemma4_q8;
pub mod gemma4_q8g;
pub mod gemma4_q5;
pub mod gemma_tokenizer;

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian

/// GGML tensor data types as encoded in GGUF.
///
/// Names mirror upstream `ggml.h` exactly (e.g. `Q4_K`, `Q5_K`) so they
/// stay diff-able against llama.cpp. The non-camel-case lint is silenced
/// for that reason; do not rename.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 30,
    /// Microsoft bitnet.cpp ternary packing. Empirically (measured
    /// against `bitnet-b1.58-2B-4T` GGUF byte offsets): 4 ternary
    /// weights per byte (2 bits each) + a trailing f32 scale, padded
    /// to GGUF alignment. Type id 36 in that file.
    I2_S = 36,
    /// PrismML Bonsai 1-bit packing (`Bonsai-*-Q1_0.gguf`, g128).
    /// Verified against the model card + PrismML's llama.cpp fork
    /// (`PrismML-Eng/llama.cpp@prism`, `block_q1_0` struct + kernel) +
    /// direct byte inspection of `Bonsai-1.7B-Q1_0`. Layout: 128-element
    /// blocks, 18 bytes each = `{ f16 d (offset 0); u8 qs[16]
    /// (offset 2) }`. Bits are LSB-first (8 elements/byte): `byte =
    /// j/8`, `bit = j%8`, `q = (qs[byte] >> bit) & 1`, `w = q ? d : -d`
    /// (binary, no zero). Effective 1.125 bpw. Type id 41. Dequant in
    /// [`dequant::dequantize_q1_0`].
    Q1_0 = 41,
    /// PrismML Bonsai ternary packing (`Ternary-Bonsai-*-Q2_0.gguf`),
    /// "Q2_0 g128". Verified three ways: the model card spec, the
    /// `block_q2_0` struct + `dequantize_row_q2_0` kernel in PrismML's
    /// llama.cpp fork (`PrismML-Eng/llama.cpp` `prism` branch), and
    /// direct byte inspection of `Ternary-Bonsai-1.7B-Q2_0`. Layout:
    /// 128-element blocks, 34 bytes each = `{ f16 d (offset 0); u8
    /// qs[32] (offset 2) }`. Codes are LSB-first 2-bit (4/byte):
    /// `byte = j/4`, `bit = (j%4)*2`, `q = (qs[byte] >> bit) & 3`,
    /// `w = (q - 1) * d`, so {0,1,2,3} → {-1,0,+1,+2}·d; ternary uses
    /// {0,1,2} only (q=3 confirmed absent in real weights). Distinct
    /// from I2_S (whole-tensor single f32 scale). Type id 42 in that
    /// file. Dequant in [`dequant::dequantize_q2_0`].
    Q2_0 = 42,
    /// Tencent / AngelSlim "Sherry" sparse-ternary packing
    /// (`Hy-MT1.5-*-1.25bit-GGUF`). Triple-verified: model card,
    /// llama.cpp PR #22836 (`block_stq1_0` struct + kernel), and direct
    /// byte inspection of `Hy-MT1.5-1.8B-1.25bit`. Layout: 256-element
    /// blocks, 42 bytes each = `{ u8 qs[32]; u8 sign[8]; f16 d; }`.
    /// 3:4 sparse ternary — every 4-lane group has exactly one zero;
    /// the other three are `{-d, +d}`. Encoded as a 4-bit slot index +
    /// 1-bit table-select via a 32-entry codebook (see
    /// [`dequant::dequantize_stq1_0`]). **The on-disk ggml type id
    /// collides with [`Self::Q2_0`] (both 42)**; the GGUF parser
    /// disambiguates by byte stride — Q2_0 is 34 B/128 elem, STQ1_0
    /// is 42 B/256 elem — and rewrites the per-tensor dtype after
    /// loading. The internal enum value here (142) is just a marker;
    /// nothing writes it back to disk.
    STQ1_0 = 142,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32), 1 => Some(Self::F16),
            2 => Some(Self::Q4_0), 3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0), 7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0), 9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K), 11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K), 13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K), 15 => Some(Self::Q8_K),
            24 => Some(Self::I8), 25 => Some(Self::I16),
            26 => Some(Self::I32), 27 => Some(Self::I64),
            28 => Some(Self::F64), 30 => Some(Self::BF16),
            36 => Some(Self::I2_S),
            41 => Some(Self::Q1_0),
            42 => Some(Self::Q2_0),
            _ => None,
        }
    }

    /// Element size for non-quantized types. Returns `None` for quantized.
    pub fn element_size(self) -> Option<usize> {
        match self {
            Self::F32 | Self::I32 => Some(4),
            Self::F16 | Self::I16 | Self::BF16 => Some(2),
            Self::I8 => Some(1),
            Self::I64 | Self::F64 => Some(8),
            _ => None, // Quantized types have a block-based layout.
        }
    }

    pub fn is_quantized(self) -> bool {
        self.element_size().is_none()
    }

    /// Block layout for quantized types: `(bytes_per_block, elements_per_block)`.
    /// Returns `None` for non-quantized types and for quantization formats
    /// not yet implemented in this loader.
    pub fn block_layout(self) -> Option<(usize, usize)> {
        match self {
            Self::Q8_0 => Some((34, 32)),     // 2 (FP16 d) + 32 (i8 quants)
            Self::Q4_K => Some((144, 256)),   // 2 + 2 + 12 + 128
            Self::Q5_K => Some((176, 256)),   // 2 + 2 + 12 + 32 + 128
            Self::Q6_K => Some((210, 256)),   // 128 (ql) + 64 (qh) + 16 (scales) + 2 (FP16 d)
            Self::Q1_0 => Some((18, 128)),    // 2 (FP16 d) + 16 (1-bit codes), g128
            Self::Q2_0 => Some((34, 128)),    // 2 (FP16 d) + 32 (2-bit codes), g128
            Self::STQ1_0 => Some((42, 256)),  // 32 (4-bit slots) + 8 (1-bit signs) + 2 (FP16 d), g256
            _ => None,
        }
    }
}

/// GGUF metadata value types.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8), I8(i8),
    U16(u16), I16(i16),
    U32(u32), I32(i32),
    U64(u64), I64(i64),
    F32(f32), F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::I64(v) => Some(*v as u64),
            Self::U32(v) => Some(*v as u64),
            // LFM2's `attention.head_count_kv` array ships as i32
            // (with 0 marking recurrent layers). Accept it too — kv
            // counts are non-negative by construction.
            Self::I32(v) => Some(*v as u64),
            Self::U16(v) => Some(*v as u64),
            Self::I16(v) => Some(*v as u64),
            Self::U8(v) => Some(*v as u64),
            Self::I8(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        match self {
            Self::Array(a) => a.iter().map(|v| v.as_string()).collect(),
            _ => None,
        }
    }
}

/// Information about one tensor stored in a GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    /// Offset from the start of the tensor-data section.
    pub offset: u64,
}

/// Parsed GGUF file (header + tensor info table). Tensor data is read on
/// demand; the file-path entry point [`read_gguf_file`] mmaps the
/// file so tensor reads are zero-copy into Apple Silicon's unified
/// memory / Linux's page cache.
///
/// On Apple Silicon, this is the key UMA optimization: Metal /
/// Accelerate kernels read directly from the page-cached pages the
/// CPU saw; no PCIe transfer (because there isn't one), no
/// host-to-device copy stage. Same property holds on Pi 4/5,
/// Android arm64, and any other UMA target. On a discrete GPU
/// the mmap still saves the host-side copy, leaving only the
/// device transfer the runtime would do anyway.
#[derive(Debug, Default)]
pub struct GgufModel {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    /// Tensor data buffer. Mmapped from disk when loaded via
    /// [`read_gguf_file`]; owned `Vec<u8>` when loaded from a
    /// reader via [`read_gguf`] (synthetic / in-memory cases).
    /// Read via [`Self::data`].
    pub(crate) buf: GgufBuffer,
}

/// The tensor-data byte buffer. Owned (`Vec<u8>`) for in-memory /
/// synthetic loads; mmapped for file loads. The mmap variant also
/// remembers the byte offset where the tensor data section begins
/// after the GGUF header, so [`GgufBuffer::data`] returns a slice
/// whose index 0 matches `info.offset` exactly — identical
/// semantics to the owned path.
pub enum GgufBuffer {
    Owned(Vec<u8>),
    Mapped { mmap: std::sync::Arc<memmap2::Mmap>, data_offset: usize },
}

/// Zero-copy byte source for tensors loaded from a GGUF file. Holds the
/// `Arc<Mmap>` keeping the mapped pages alive; `bytes()` returns the
/// data-section slice (i.e. tensor offsets index from 0 here, same
/// semantics as `GgufModel::data()`).
#[derive(Debug)]
pub struct GgufMmapBacking {
    mmap: std::sync::Arc<memmap2::Mmap>,
    data_offset: usize,
}

impl jouleclaw_core::tensor::ByteBacking for GgufMmapBacking {
    fn bytes(&self) -> &[u8] {
        &self.mmap[self.data_offset..]
    }
}

impl Default for GgufBuffer {
    fn default() -> Self {
        Self::Owned(Vec::new())
    }
}

impl std::fmt::Debug for GgufBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned(v) => write!(f, "Owned({} bytes)", v.len()),
            Self::Mapped { mmap, data_offset } => write!(
                f,
                "Mapped({} bytes, data_offset={data_offset})",
                mmap.len()
            ),
        }
    }
}

impl GgufBuffer {
    /// View the tensor data section as a byte slice — indexed by
    /// `info.offset`. The mmap variant returns a sub-slice starting
    /// at the data section; the owned variant returns the whole
    /// vec (which already starts at the data section by construction).
    pub fn data(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v,
            Self::Mapped { mmap, data_offset } => &mmap[*data_offset..],
        }
    }

    /// Total byte length of the visible tensor data section.
    pub fn len(&self) -> usize {
        self.data().len()
    }

    /// True if the data section is empty (no tensors).
    pub fn is_empty(&self) -> bool {
        self.data().is_empty()
    }
}

impl GgufModel {
    /// Tensor data section as a byte slice. Index 0 is the first
    /// byte of the first tensor's data; `info.offset` indexes into
    /// this slice. Backed by mmap when the model came from a file,
    /// by a `Vec<u8>` otherwise.
    pub fn data(&self) -> &[u8] {
        self.buf.data()
    }

    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        self.metadata.get(key)?.as_string()
    }

    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key)?.as_u32()
    }

    pub fn metadata_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key)?.as_u64()
    }

    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key)?.as_f32()
    }

    /// Read a metadata value that is a `GgufValue::Array` of u32/u64
    /// integers, coerced to `usize`. Returns `None` if the key is
    /// missing or the array element type isn't an unsigned integer.
    /// Used by LFM2's per-layer `attention.head_count_kv` (an array
    /// whose 0-entries mark recurrent / conv layers, and non-zero
    /// entries mark attention layers).
    pub fn metadata_u_array(&self, key: &str) -> Option<Vec<usize>> {
        match self.metadata.get(key)? {
            GgufValue::Array(items) => items.iter().map(|v| {
                v.as_u64().map(|x| x as usize)
            }).collect(),
            _ => None,
        }
    }

    pub fn tensor_by_name(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Zero-copy mmap backing for this model's tensor data section,
    /// or `None` if the model was loaded from an owned buffer (synthetic
    /// / in-memory case). Used by `embed_weight` to construct
    /// mapped-tensor constants without copying weights into RAM.
    pub fn mmap_backing(&self) -> Option<std::sync::Arc<GgufMmapBacking>> {
        match &self.buf {
            GgufBuffer::Owned(_) => None,
            GgufBuffer::Mapped { mmap, data_offset } => {
                Some(std::sync::Arc::new(GgufMmapBacking {
                    mmap: mmap.clone(),
                    data_offset: *data_offset,
                }))
            }
        }
    }

    /// Bytes of one tensor as raw storage.
    pub fn tensor_bytes(&self, info: &TensorInfo) -> &[u8] {
        let start = info.offset as usize;
        let n_elems = info.shape.iter().map(|&d| d as usize).product::<usize>();
        let bytes_count = if let Some(sz) = info.dtype.element_size() {
            sz * n_elems
        } else if let Some((block_bytes, elems_per_block)) = info.dtype.block_layout() {
            // Quantized: round up to block boundary (GGUF tensors are
            // always block-aligned in practice).
            let n_blocks = (n_elems + elems_per_block - 1) / elems_per_block;
            n_blocks * block_bytes
        } else {
            0
        };
        &self.data()[start..start + bytes_count]
    }
}

/// Parse a GGUF file from a reader.
pub fn read_gguf<R: Read + Seek>(mut r: R) -> Result<GgufModel, ParseError> {
    let magic = read_u32(&mut r)?;
    if magic != GGUF_MAGIC {
        return Err(ParseError::BadMagic { got: magic });
    }
    let version = read_u32(&mut r)?;
    if !(1..=3).contains(&version) {
        return Err(ParseError::UnsupportedVersion { version });
    }
    let tensor_count = read_u64(&mut r)?;
    let metadata_kv_count = read_u64(&mut r)?;

    // Metadata.
    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_string(&mut r)?;
        let value = read_value(&mut r)?;
        metadata.insert(key, value);
    }

    // Tensor info table.
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut r)?;
        let n_dims = read_u32(&mut r)?;
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(read_u64(&mut r)?);
        }
        let dtype_raw = read_u32(&mut r)?;
        let dtype = GgmlType::from_u32(dtype_raw)
            .ok_or(ParseError::UnknownDtype { value: dtype_raw })?;
        let offset = read_u64(&mut r)?;
        tensors.push(TensorInfo { name, shape, dtype, offset });
    }

    // GGUF aligns the data section to `general.alignment` (default 32).
    let alignment = metadata.get("general.alignment")
        .and_then(|v| v.as_u64()).unwrap_or(32);
    let pos = r.stream_position().map_err(io_err)?;
    let aligned = (pos + alignment - 1) / alignment * alignment;
    if aligned > pos {
        r.seek(SeekFrom::Start(aligned)).map_err(io_err)?;
    }

    // Read all remaining bytes into `data`.
    let mut data = Vec::new();
    r.read_to_end(&mut data).map_err(io_err)?;

    // Disambiguate the ggml-type-42 collision between PrismML's Q2_0
    // (PrismML-Eng/llama.cpp@prism, 34 B / 128 elem ternary) and
    // Tencent/AngelSlim's STQ1_0 (ggml-org/llama.cpp PR #22836,
    // 42 B / 256 elem sparse-ternary). Both reserve id 42 in their
    // forks. We measure the byte stride of the first type-42 tensor
    // against the next tensor's offset and rewrite all type-42
    // tensors' internal dtype to whichever layout the file actually
    // uses. This is empirical, not metadata-trust-based, so it stays
    // correct even if a future model swaps the architecture tag.
    if tensors.iter().any(|t| t.dtype == GgmlType::Q2_0) {
        let mut sorted: Vec<usize> = (0..tensors.len()).collect();
        sorted.sort_by_key(|&i| tensors[i].offset);
        let first42_idx_in_sorted = sorted.iter()
            .position(|&i| tensors[i].dtype == GgmlType::Q2_0);
        if let Some(p) = first42_idx_in_sorted {
            let curr = &tensors[sorted[p]];
            let next_off = sorted.iter().skip(p + 1)
                .map(|&i| tensors[i].offset)
                .next()
                .unwrap_or(curr.offset + data.len() as u64);
            let span = (next_off - curr.offset) as usize;
            let n: usize = curr.shape.iter().map(|&d| d as usize).product();
            // STQ1_0: 42 B per 256-elem block. Q2_0: 34 B per 128-elem block.
            let detected = if n > 0 && n % 256 == 0 && span == (n / 256) * 42 {
                Some(GgmlType::STQ1_0)
            } else if n > 0 && n % 128 == 0 && span == (n / 128) * 34 {
                Some(GgmlType::Q2_0)
            } else {
                None
            };
            if let Some(d) = detected {
                for t in tensors.iter_mut().filter(|t| t.dtype == GgmlType::Q2_0) {
                    t.dtype = d;
                }
            }
        }
    }

    Ok(GgufModel { version, metadata, tensors, buf: GgufBuffer::Owned(data) })
}

/// Parse a GGUF file from a path, **mmapping** the tensor data
/// section. On UMA targets (Apple Silicon, Pi 4/5, Android arm64,
/// Intel Lunar Lake / AMD Strix Halo iGPUs) this is true zero-copy:
/// the kernel maps file pages into virtual memory, and tensor
/// reads page-fault them in on demand. There's no host-side copy
/// and, on UMA, no host-to-device copy either — the GPU reads the
/// same physical pages the CPU sees.
///
/// Trade-off vs the `Read + Seek` path: the metadata table and
/// per-tensor info still parse via a `Cursor` over the mmap (small
/// reads, no I/O cost). The win is on the tensor **data** section
/// — the multi-hundred-megabyte to multi-gigabyte chunk that
/// previously copied into a fresh `Vec<u8>`.
pub fn read_gguf_file<P: AsRef<Path>>(path: P) -> Result<GgufModel, ParseError> {
    let f = std::fs::File::open(path).map_err(io_err)?;
    // SAFETY: We're mmapping a regular file read-only. Concurrent
    // modification of the file under us would invalidate this, but
    // model files on disk are static-distribution artifacts that
    // shouldn't be modified while loaded. memmap2's `Mmap` enforces
    // read-only at the API level.
    let mmap = unsafe { memmap2::Mmap::map(&f) }.map_err(io_err)?;
    read_gguf_from_mmap(std::sync::Arc::new(mmap))
}

/// Parse a GGUF model whose tensor data is already mmapped. Header
/// is parsed via a `Cursor` over the mmap; the tensor data section
/// is exposed without copying.
pub fn read_gguf_from_mmap(mmap: std::sync::Arc<memmap2::Mmap>) -> Result<GgufModel, ParseError> {
    use std::io::Cursor;
    let mut cur = Cursor::new(&mmap[..]);

    // Inlined header parser — same logic as read_gguf, but we need
    // to track stream position to compute the data_offset before
    // we lose mutable access to the mmap.
    let magic = read_u32(&mut cur)?;
    if magic != GGUF_MAGIC {
        return Err(ParseError::BadMagic { got: magic });
    }
    let version = read_u32(&mut cur)?;
    if !(1..=3).contains(&version) {
        return Err(ParseError::UnsupportedVersion { version });
    }
    let tensor_count = read_u64(&mut cur)?;
    let metadata_kv_count = read_u64(&mut cur)?;

    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_string(&mut cur)?;
        let value = read_value(&mut cur)?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut cur)?;
        let n_dims = read_u32(&mut cur)?;
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(read_u64(&mut cur)?);
        }
        let dtype_raw = read_u32(&mut cur)?;
        let dtype = GgmlType::from_u32(dtype_raw)
            .ok_or(ParseError::UnknownDtype { value: dtype_raw })?;
        let offset = read_u64(&mut cur)?;
        tensors.push(TensorInfo { name, shape, dtype, offset });
    }

    let alignment = metadata.get("general.alignment")
        .and_then(|v| v.as_u64()).unwrap_or(32);
    let pos = cur.position();
    let aligned = (pos + alignment - 1) / alignment * alignment;
    let data_offset = aligned as usize;
    if data_offset > mmap.len() {
        return Err(ParseError::UnexpectedEof);
    }

    // Q2_0 / STQ1_0 dtype disambiguation (same logic as read_gguf).
    if tensors.iter().any(|t| t.dtype == GgmlType::Q2_0) {
        let mut sorted: Vec<usize> = (0..tensors.len()).collect();
        sorted.sort_by_key(|&i| tensors[i].offset);
        let first42_idx_in_sorted = sorted.iter()
            .position(|&i| tensors[i].dtype == GgmlType::Q2_0);
        if let Some(p) = first42_idx_in_sorted {
            let curr = &tensors[sorted[p]];
            let next_off = sorted.iter().skip(p + 1)
                .map(|&i| tensors[i].offset)
                .next()
                .unwrap_or(curr.offset + (mmap.len() - data_offset) as u64);
            let span = (next_off - curr.offset) as usize;
            let n: usize = curr.shape.iter().map(|&d| d as usize).product();
            let detected = if n > 0 && n % 256 == 0 && span == (n / 256) * 42 {
                Some(GgmlType::STQ1_0)
            } else if n > 0 && n % 128 == 0 && span == (n / 128) * 34 {
                Some(GgmlType::Q2_0)
            } else {
                None
            };
            if let Some(d) = detected {
                for t in tensors.iter_mut().filter(|t| t.dtype == GgmlType::Q2_0) {
                    t.dtype = d;
                }
            }
        }
    }

    Ok(GgufModel {
        version,
        metadata,
        tensors,
        buf: GgufBuffer::Mapped { mmap, data_offset },
    })
}

// ---- Errors ----

#[derive(Debug)]
pub enum ParseError {
    Io(std::io::Error),
    BadMagic { got: u32 },
    UnsupportedVersion { version: u32 },
    UnknownDtype { value: u32 },
    UnknownValueType { value: u32 },
    InvalidString,
    UnexpectedEof,
    /// A `.safetensors` blob ended before its declared header/data.
    Truncated,
    /// Safetensors-specific structural error (bad JSON header,
    /// unsupported dtype, out-of-bounds offsets, missing shard, …).
    Safetensors(String),
}

fn io_err(e: std::io::Error) -> ParseError { ParseError::Io(e) }

// ---- Primitive readers (little-endian) ----

fn read_u8<R: Read>(r: &mut R) -> Result<u8, ParseError> {
    let mut b = [0u8; 1]; r.read_exact(&mut b).map_err(io_err)?; Ok(b[0])
}
fn read_i8<R: Read>(r: &mut R) -> Result<i8, ParseError> {
    Ok(read_u8(r)? as i8)
}
fn read_u16<R: Read>(r: &mut R) -> Result<u16, ParseError> {
    let mut b = [0u8; 2]; r.read_exact(&mut b).map_err(io_err)?; Ok(u16::from_le_bytes(b))
}
fn read_i16<R: Read>(r: &mut R) -> Result<i16, ParseError> {
    let mut b = [0u8; 2]; r.read_exact(&mut b).map_err(io_err)?; Ok(i16::from_le_bytes(b))
}
fn read_u32<R: Read>(r: &mut R) -> Result<u32, ParseError> {
    let mut b = [0u8; 4]; r.read_exact(&mut b).map_err(io_err)?; Ok(u32::from_le_bytes(b))
}
fn read_i32<R: Read>(r: &mut R) -> Result<i32, ParseError> {
    let mut b = [0u8; 4]; r.read_exact(&mut b).map_err(io_err)?; Ok(i32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R) -> Result<u64, ParseError> {
    let mut b = [0u8; 8]; r.read_exact(&mut b).map_err(io_err)?; Ok(u64::from_le_bytes(b))
}
fn read_i64<R: Read>(r: &mut R) -> Result<i64, ParseError> {
    let mut b = [0u8; 8]; r.read_exact(&mut b).map_err(io_err)?; Ok(i64::from_le_bytes(b))
}
fn read_f32<R: Read>(r: &mut R) -> Result<f32, ParseError> {
    let mut b = [0u8; 4]; r.read_exact(&mut b).map_err(io_err)?; Ok(f32::from_le_bytes(b))
}
fn read_f64<R: Read>(r: &mut R) -> Result<f64, ParseError> {
    let mut b = [0u8; 8]; r.read_exact(&mut b).map_err(io_err)?; Ok(f64::from_le_bytes(b))
}
fn read_bool<R: Read>(r: &mut R) -> Result<bool, ParseError> {
    Ok(read_u8(r)? != 0)
}

fn read_string<R: Read>(r: &mut R) -> Result<String, ParseError> {
    let len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; len];
    r.read_exact(&mut bytes).map_err(io_err)?;
    String::from_utf8(bytes).map_err(|_| ParseError::InvalidString)
}

fn read_value<R: Read>(r: &mut R) -> Result<GgufValue, ParseError> {
    let ty = read_u32(r)?;
    read_value_of_type(r, ty)
}

fn read_value_of_type<R: Read>(r: &mut R, ty: u32) -> Result<GgufValue, ParseError> {
    match ty {
        0 => Ok(GgufValue::U8(read_u8(r)?)),
        1 => Ok(GgufValue::I8(read_i8(r)?)),
        2 => Ok(GgufValue::U16(read_u16(r)?)),
        3 => Ok(GgufValue::I16(read_i16(r)?)),
        4 => Ok(GgufValue::U32(read_u32(r)?)),
        5 => Ok(GgufValue::I32(read_i32(r)?)),
        6 => Ok(GgufValue::F32(read_f32(r)?)),
        7 => Ok(GgufValue::Bool(read_bool(r)?)),
        8 => Ok(GgufValue::String(read_string(r)?)),
        9 => {
            let elem_ty = read_u32(r)?;
            let len = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_value_of_type(r, elem_ty)?);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::U64(read_u64(r)?)),
        11 => Ok(GgufValue::I64(read_i64(r)?)),
        12 => Ok(GgufValue::F64(read_f64(r)?)),
        other => Err(ParseError::UnknownValueType { value: other }),
    }
}

// ---- Tensor extraction into the runtime's Tensor type ----

use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta, TensorStorage};

/// Convert a GGUF tensor into a runtime `Tensor`.
///
/// Phase 1.6 supports F32, F16 (widened to F32), Q8_0, Q4_K, Q5_K. All
/// quantized types are dequantized to F32 at load time. Other quantized
/// formats return `UnknownDtype`.
pub fn tensor_from_gguf(model: &GgufModel, info: &TensorInfo) -> Result<Tensor, ParseError> {
    // GGUF/GGML stores tensor dims in `ne` order: fastest-varying axis
    // first. A weight that is logically [out, in] (row-major, contiguous
    // in `in`) is stored with ne = [in, out]. The graph ops here use
    // logical [out, in] (matmul_bt wants b = [N, K]; `lookup` wants
    // [vocab, d_model]). So the logical shape is the *reverse* of the
    // GGUF ne array. The data bytes are already row-major over the
    // logical shape, so reversing only the shape metadata — not the
    // bytes — recovers the correct tensor. (Square synthetic tensors
    // masked this; real GQA / FFN weights are non-square and expose it.)
    let shape: Vec<usize> = info.shape.iter().rev().map(|&d| d as usize).collect();
    let n_elems = shape.iter().product::<usize>();

    match info.dtype {
        GgmlType::F32 => {
            let bytes = model.tensor_bytes(info).to_vec();
            Ok(Tensor {
                meta: TensorMeta::new(Dtype::F32, &shape),
                storage: std::sync::Arc::new(TensorStorage { bytes, mapped: None }),
            })
        }
        GgmlType::F16 => {
            let raw = model.tensor_bytes(info);
            let n = raw.len() / 2;
            let mut out = Vec::with_capacity(n * 4);
            for i in 0..n {
                let mut b = [0u8; 2];
                b.copy_from_slice(&raw[i * 2..i * 2 + 2]);
                let bits = u16::from_le_bytes(b);
                let f = f16_to_f32(bits);
                out.extend_from_slice(&f.to_le_bytes());
            }
            Ok(Tensor {
                meta: TensorMeta::new(Dtype::F32, &shape),
                storage: std::sync::Arc::new(TensorStorage { bytes: out, mapped: None }),
            })
        }
        GgmlType::BF16 => {
            // bfloat16 is the top 16 bits of an IEEE f32 (sign + 8-bit
            // exponent + 7-bit mantissa). Widening is just a left
            // shift by 16 — no table, exact. This is the native weight
            // dtype for most HuggingFace safetensors checkpoints
            // (Gemma, Llama, Mistral, …).
            let raw = model.tensor_bytes(info);
            let n = raw.len() / 2;
            let mut out = Vec::with_capacity(n * 4);
            for i in 0..n {
                let bits = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
                let f = f32::from_bits((bits as u32) << 16);
                out.extend_from_slice(&f.to_le_bytes());
            }
            Ok(Tensor {
                meta: TensorMeta::new(Dtype::F32, &shape),
                storage: std::sync::Arc::new(TensorStorage { bytes: out, mapped: None }),
            })
        }
        GgmlType::Q8_0
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::I2_S
        | GgmlType::Q1_0
        | GgmlType::Q2_0
        | GgmlType::STQ1_0 => {
            let raw = model.tensor_bytes(info);
            let dequant = match info.dtype {
                GgmlType::Q8_0 => dequant::dequantize_q8_0(raw, n_elems),
                GgmlType::Q4_K => dequant::dequantize_q4_k(raw, n_elems),
                GgmlType::Q5_K => dequant::dequantize_q5_k(raw, n_elems),
                GgmlType::Q6_K => dequant::dequantize_q6_k(raw, n_elems),
                GgmlType::I2_S => dequant::dequantize_i2_s(raw, n_elems),
                GgmlType::Q1_0 => dequant::dequantize_q1_0(raw, n_elems),
                GgmlType::Q2_0 => dequant::dequantize_q2_0(raw, n_elems),
                GgmlType::STQ1_0 => dequant::dequantize_stq1_0(raw, n_elems),
                _ => unreachable!(),
            }.map_err(|_| ParseError::UnknownDtype { value: info.dtype as u32 })?;
            let mut bytes = Vec::with_capacity(dequant.len() * 4);
            for v in &dequant { bytes.extend_from_slice(&v.to_le_bytes()); }
            Ok(Tensor {
                meta: TensorMeta::new(Dtype::F32, &shape),
                storage: std::sync::Arc::new(TensorStorage { bytes, mapped: None }),
            })
        }
        _ => Err(ParseError::UnknownDtype { value: info.dtype as u32 }),
    }
}

/// Convert IEEE 754 binary16 to binary32. Pure software implementation;
/// deterministic across platforms.
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;

    let f32_bits = if exp == 0 {
        if mant == 0 {
            // Zero (signed).
            (sign as u32) << 31
        } else {
            // Subnormal -> normalize.
            let mut e: i32 = -14;
            let mut m: u32 = mant as u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            ((sign as u32) << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        // Inf or NaN.
        ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13)
    } else {
        // Normal.
        ((sign as u32) << 31)
            | (((exp as i32 - 15 + 127) as u32) << 23)
            | ((mant as u32) << 13)
    };
    f32::from_bits(f32_bits)
}
