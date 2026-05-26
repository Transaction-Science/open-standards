//! GGUF format parser for quantized models.
//!
//! GGUF (GGML Universal Format) is used by llama.cpp for storing quantized models.
//! Format specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//!
//! Structure:
//! - Magic: "GGUF" (4 bytes)
//! - Version: u32
//! - Tensor count: u64
//! - Metadata KV count: u64
//! - Metadata key-value pairs
//! - Tensor infos
//! - Padding to alignment
//! - Tensor data

use crate::core::{DType, Error, Result, Shape};
use std::collections::HashMap;
use std::path::Path;

/// GGUF magic bytes.
const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF"

/// GGUF version we support.
const GGUF_VERSION: u32 = 3;

/// GGUF data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    /// 32-bit floating point.
    F32 = 0,
    /// 16-bit floating point.
    F16 = 1,
    /// 4-bit quantization (type 0).
    Q4_0 = 2,
    /// 4-bit quantization (type 1).
    Q4_1 = 3,
    /// 5-bit quantization (type 0).
    Q5_0 = 6,
    /// 5-bit quantization (type 1).
    Q5_1 = 7,
    /// 8-bit quantization (type 0).
    Q8_0 = 8,
    /// 8-bit quantization (type 1).
    Q8_1 = 9,
    /// 2-bit K-quantization.
    Q2K = 10,
    /// 3-bit K-quantization.
    Q3K = 11,
    /// 4-bit K-quantization.
    Q4K = 12,
    /// 5-bit K-quantization.
    Q5K = 13,
    /// 6-bit K-quantization.
    Q6K = 14,
    /// 8-bit K-quantization.
    Q8K = 15,
    /// 2-bit importance quantization (XXS).
    IQ2XXS = 16,
    /// 2-bit importance quantization (XS).
    IQ2XS = 17,
    /// 3-bit importance quantization (XXS).
    IQ3XXS = 18,
    /// 1-bit importance quantization (S).
    IQ1S = 19,
    /// 4-bit importance quantization (NL).
    IQ4NL = 20,
    /// 3-bit importance quantization (S).
    IQ3S = 21,
    /// 2-bit importance quantization (S).
    IQ2S = 22,
    /// 4-bit importance quantization (XS).
    IQ4XS = 23,
    /// 8-bit integer.
    I8 = 24,
    /// 16-bit integer.
    I16 = 25,
    /// 32-bit integer.
    I32 = 26,
    /// 64-bit integer.
    I64 = 27,
    /// 64-bit floating point.
    F64 = 28,
    /// Brain float 16-bit.
    BF16 = 29,
}

impl GgmlType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Convert to DType (for non-quantized types).
    pub fn to_dtype(&self) -> Option<DType> {
        match self {
            Self::F32 => Some(DType::F32),
            Self::F16 => Some(DType::F16),
            Self::BF16 => Some(DType::BF16),
            Self::I8 => Some(DType::I8),
            Self::I32 => Some(DType::I32),
            Self::I64 => Some(DType::I64),
            _ => None, // Quantized types don't map directly
        }
    }

    /// Get block size for quantized types.
    pub fn block_size(&self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            _ => 1,
        }
    }

    /// Get bytes per block.
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,   // 32 values = 2 bytes scale + 16 bytes data
            Self::Q4_1 => 20,   // 32 values = 2 bytes scale + 2 bytes min + 16 bytes data
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,   // 32 values = 2 bytes scale + 32 bytes data
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            _ => 1,
        }
    }

    /// Check if this is a quantized type.
    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::BF16 | Self::F64 | Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }
}

/// Parsed GGUF file.
#[derive(Debug)]
pub struct GgufFile {
    /// Version
    pub version: u32,
    /// Metadata
    pub metadata: GgufMetadata,
    /// Tensor infos
    pub tensors: HashMap<String, GgufTensorInfo>,
    /// Data section offset
    pub data_offset: usize,
    /// Alignment
    pub alignment: usize,
}

/// GGUF metadata.
#[derive(Debug, Default)]
pub struct GgufMetadata {
    /// Architecture (e.g., "llama", "mistral", "phi")
    pub architecture: Option<String>,
    /// Model name
    pub name: Option<String>,
    /// Context length
    pub context_length: Option<u64>,
    /// Embedding length (hidden size)
    pub embedding_length: Option<u64>,
    /// Block count (layers)
    pub block_count: Option<u64>,
    /// Feed-forward hidden size (intermediate size)
    pub feed_forward_length: Option<u64>,
    /// Head count
    pub head_count: Option<u64>,
    /// Head count for KV (for GQA)
    pub head_count_kv: Option<u64>,
    /// Vocab size
    pub vocab_size: Option<u64>,
    /// Rope dimension count
    pub rope_dimension_count: Option<u64>,
    /// Rope frequency base (theta)
    pub rope_freq_base: Option<f32>,
    /// Rope frequency scale
    pub rope_freq_scale: Option<f32>,
    /// RMS norm epsilon
    pub rms_norm_eps: Option<f32>,
    /// Quantization version
    pub quantization_version: Option<u64>,
    /// File type (quantization)
    pub file_type: Option<u64>,
    /// All key-value pairs
    pub raw: HashMap<String, MetadataValue>,
}

/// Metadata value types.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Signed 8-bit integer.
    I8(i8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Signed 16-bit integer.
    I16(i16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 32-bit integer.
    I32(i32),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 64-bit integer.
    I64(i64),
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
    /// Boolean value.
    Bool(bool),
    /// String value.
    String(String),
    /// Array of metadata values.
    Array(Vec<MetadataValue>),
}

/// Extract u64 from metadata value (handles u32/u64/i32/i64).
fn extract_u64(value: &MetadataValue) -> u64 {
    match value {
        MetadataValue::U64(v) => *v,
        MetadataValue::U32(v) => *v as u64,
        MetadataValue::I64(v) => *v as u64,
        MetadataValue::I32(v) => *v as u64,
        MetadataValue::U16(v) => *v as u64,
        MetadataValue::U8(v) => *v as u64,
        _ => 0,
    }
}

/// Extract f32 from metadata value (handles f32/f64).
fn extract_f32(value: &MetadataValue) -> f32 {
    match value {
        MetadataValue::F32(v) => *v,
        MetadataValue::F64(v) => *v as f32,
        _ => 0.0,
    }
}

/// Tensor information.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name
    pub name: String,
    /// Number of dimensions
    pub n_dims: u32,
    /// Shape (dimensions)
    pub shape: Shape,
    /// Data type
    pub ggml_type: GgmlType,
    /// Offset from data section start
    pub offset: u64,
    /// Size in bytes
    pub size: usize,
}

/// Reader helper.
struct GgufReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> GgufReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(Error::model_load("gguf", "unexpected end of file"));
        }
        let bytes = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8> {
        let bytes = self.read_bytes(1)?;
        Ok(bytes[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32> {
        let bytes = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let bytes = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| Error::model_load("gguf", format!("invalid UTF-8: {}", e)))
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn align(&mut self, alignment: usize) {
        let remainder = self.pos % alignment;
        if remainder != 0 {
            self.pos += alignment - remainder;
        }
    }
}

impl GgufFile {
    /// Parse a GGUF file from memory.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut reader = GgufReader::new(data);

        // Read magic
        let magic = reader.read_bytes(4)?;
        if magic != GGUF_MAGIC {
            return Err(Error::model_load("gguf", "invalid magic bytes"));
        }

        // Read version
        let version = reader.read_u32()?;
        if version != GGUF_VERSION && version != 2 {
            return Err(Error::model_load("gguf", format!("unsupported version: {}", version)));
        }

        // Read counts
        let tensor_count = reader.read_u64()? as usize;
        let metadata_kv_count = reader.read_u64()? as usize;

        // Read metadata
        let mut metadata = GgufMetadata::default();
        for _ in 0..metadata_kv_count {
            let key = reader.read_string()?;
            let value = read_metadata_value(&mut reader)?;

            // Extract known fields
            match key.as_str() {
                "general.architecture" => {
                    if let MetadataValue::String(s) = &value {
                        metadata.architecture = Some(s.clone());
                    }
                }
                "general.name" => {
                    if let MetadataValue::String(s) = &value {
                        metadata.name = Some(s.clone());
                    }
                }
                _ if key.ends_with(".context_length") => {
                    metadata.context_length = Some(extract_u64(&value));
                }
                _ if key.ends_with(".embedding_length") => {
                    metadata.embedding_length = Some(extract_u64(&value));
                }
                _ if key.ends_with(".block_count") => {
                    metadata.block_count = Some(extract_u64(&value));
                }
                _ if key.ends_with(".feed_forward_length") => {
                    metadata.feed_forward_length = Some(extract_u64(&value));
                }
                _ if key.ends_with(".head_count") && !key.contains("_kv") => {
                    metadata.head_count = Some(extract_u64(&value));
                }
                _ if key.ends_with(".head_count_kv") => {
                    metadata.head_count_kv = Some(extract_u64(&value));
                }
                _ if key.ends_with(".vocab_size") => {
                    metadata.vocab_size = Some(extract_u64(&value));
                }
                _ if key.ends_with(".rope.dimension_count") || key.ends_with(".rope_dimension_count") => {
                    metadata.rope_dimension_count = Some(extract_u64(&value));
                }
                _ if key.ends_with(".rope.freq_base") => {
                    metadata.rope_freq_base = Some(extract_f32(&value));
                }
                _ if key.ends_with(".rope.scale_linear") || key.ends_with(".rope.freq_scale") => {
                    metadata.rope_freq_scale = Some(extract_f32(&value));
                }
                _ if key.ends_with(".attention.layer_norm_rms_epsilon") || key.ends_with(".layer_norm_rms_epsilon") => {
                    metadata.rms_norm_eps = Some(extract_f32(&value));
                }
                "general.quantization_version" => {
                    metadata.quantization_version = Some(extract_u64(&value));
                }
                "general.file_type" => {
                    metadata.file_type = Some(extract_u64(&value));
                }
                _ => {}
            }

            metadata.raw.insert(key, value);
        }

        // Read tensor infos
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = reader.read_string()?;
            let n_dims = reader.read_u32()?;

            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(reader.read_u64()? as usize);
            }

            let type_id = reader.read_u32()?;
            let ggml_type = GgmlType::from_u32(type_id)
                .ok_or_else(|| Error::model_load("gguf", format!("unknown tensor type: {}", type_id)))?;

            let offset = reader.read_u64()?;

            // Calculate size
            let numel: usize = dims.iter().product();
            let size = if ggml_type.is_quantized() {
                let block_size = ggml_type.block_size();
                let num_blocks = (numel + block_size - 1) / block_size;
                num_blocks * ggml_type.type_size()
            } else {
                numel * ggml_type.type_size()
            };

            let info = GgufTensorInfo {
                name: name.clone(),
                n_dims,
                shape: Shape::new(dims),
                ggml_type,
                offset,
                size,
            };

            tensors.insert(name, info);
        }

        // Align to 32 bytes for data section
        let alignment = 32;
        reader.align(alignment);
        let data_offset = reader.pos;

        Ok(Self {
            version,
            metadata,
            tensors,
            data_offset,
            alignment,
        })
    }

    /// Parse from a file path.
    pub fn from_path(path: &Path) -> Result<(Self, memmap2::Mmap)> {
        let file = std::fs::File::open(path)
            .map_err(|e| Error::io("open", format!("{}: {}", path.display(), e)))?;

        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| Error::io("mmap", format!("{}: {}", path.display(), e)))?;

        let parsed = Self::parse(&mmap)?;
        Ok((parsed, mmap))
    }

    /// Get tensor data slice.
    pub fn tensor_data<'a>(&self, name: &str, data: &'a [u8]) -> Option<&'a [u8]> {
        let info = self.tensors.get(name)?;
        let start = self.data_offset + info.offset as usize;
        let end = start + info.size;

        if end <= data.len() {
            Some(&data[start..end])
        } else {
            None
        }
    }

    /// Iterate over tensors.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &GgufTensorInfo)> {
        self.tensors.iter().map(|(k, v)| (k.as_str(), v))
    }
}

/// Read a metadata value.
fn read_metadata_value(reader: &mut GgufReader) -> Result<MetadataValue> {
    let value_type = reader.read_u32()?;

    match value_type {
        0 => Ok(MetadataValue::U8(reader.read_u8()?)),
        1 => Ok(MetadataValue::I8(reader.read_i8()?)),
        2 => Ok(MetadataValue::U16(reader.read_u16()?)),
        3 => Ok(MetadataValue::I16(reader.read_i16()?)),
        4 => Ok(MetadataValue::U32(reader.read_u32()?)),
        5 => Ok(MetadataValue::I32(reader.read_i32()?)),
        6 => Ok(MetadataValue::F32(reader.read_f32()?)),
        7 => Ok(MetadataValue::Bool(reader.read_bool()?)),
        8 => Ok(MetadataValue::String(reader.read_string()?)),
        9 => {
            // Array
            let element_type = reader.read_u32()?;
            let len = reader.read_u64()? as usize;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                // Read elements based on type
                let val = match element_type {
                    0 => MetadataValue::U8(reader.read_u8()?),
                    1 => MetadataValue::I8(reader.read_i8()?),
                    2 => MetadataValue::U16(reader.read_u16()?),
                    3 => MetadataValue::I16(reader.read_i16()?),
                    4 => MetadataValue::U32(reader.read_u32()?),
                    5 => MetadataValue::I32(reader.read_i32()?),
                    6 => MetadataValue::F32(reader.read_f32()?),
                    7 => MetadataValue::Bool(reader.read_bool()?),
                    8 => MetadataValue::String(reader.read_string()?),
                    10 => MetadataValue::U64(reader.read_u64()?),
                    11 => MetadataValue::I64(reader.read_i64()?),
                    12 => MetadataValue::F64(reader.read_f64()?),
                    _ => return Err(Error::model_load("gguf", format!("unsupported array element type: {}", element_type))),
                };
                values.push(val);
            }
            Ok(MetadataValue::Array(values))
        }
        10 => Ok(MetadataValue::U64(reader.read_u64()?)),
        11 => Ok(MetadataValue::I64(reader.read_i64()?)),
        12 => Ok(MetadataValue::F64(reader.read_f64()?)),
        _ => Err(Error::model_load("gguf", format!("unsupported metadata type: {}", value_type))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_sizes() {
        assert_eq!(GgmlType::F32.type_size(), 4);
        assert_eq!(GgmlType::F16.type_size(), 2);
        assert_eq!(GgmlType::Q4_0.type_size(), 18);
        assert_eq!(GgmlType::Q8_0.type_size(), 34);
    }

    #[test]
    fn test_ggml_type_quantized() {
        assert!(!GgmlType::F32.is_quantized());
        assert!(!GgmlType::F16.is_quantized());
        assert!(GgmlType::Q4_0.is_quantized());
        assert!(GgmlType::Q8_0.is_quantized());
    }
}
