//! Native SafeTensors format parser.
//!
//! SafeTensors is a simple, safe format for storing tensors:
//! - 8 bytes: header size (u64 little endian)
//! - N bytes: JSON header with tensor metadata
//! - Rest: raw tensor data (aligned)
//!
//! This parser is zero-copy friendly - it returns offsets into the mmap'd file.

use crate::core::{DType, Error, Result, Shape};
use std::collections::HashMap;
use std::path::Path;

/// Parsed SafeTensors file.
#[derive(Debug)]
pub struct SafeTensorsFile {
    /// Tensor metadata
    pub tensors: HashMap<String, SafeTensorInfo>,
    /// Global metadata (if any)
    pub metadata: HashMap<String, String>,
    /// Header size in bytes
    pub header_size: usize,
    /// Total file size
    pub file_size: usize,
}

/// Information about a single tensor.
#[derive(Debug, Clone)]
pub struct SafeTensorInfo {
    /// Tensor name
    pub name: String,
    /// Data type
    pub dtype: DType,
    /// Shape
    pub shape: Shape,
    /// Offset into file (after header)
    pub data_offset: usize,
    /// Size in bytes
    pub size: usize,
}

impl SafeTensorsFile {
    /// Parse a SafeTensors file from a memory-mapped buffer.
    ///
    /// This only parses the header - tensor data is accessed via offsets.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(Error::model_load("safetensors", "file too small"));
        }

        // Read header size (8 bytes, little endian)
        let header_size = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]) as usize;

        if header_size > data.len() - 8 {
            return Err(Error::model_load(
                "safetensors",
                format!("header size {} exceeds file size {}", header_size, data.len()),
            ));
        }

        // Parse JSON header
        let header_json = &data[8..8 + header_size];
        let header_str = std::str::from_utf8(header_json)
            .map_err(|e| Error::model_load("safetensors", format!("invalid UTF-8 in header: {}", e)))?;

        // Parse the JSON manually (avoiding serde_json dependency for core)
        let (tensors, metadata) = parse_header(header_str, 8 + header_size)?;

        Ok(Self {
            tensors,
            metadata,
            header_size,
            file_size: data.len(),
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

    /// Get tensor data slice from the mmap'd buffer.
    pub fn tensor_data<'a>(&self, name: &str, data: &'a [u8]) -> Option<&'a [u8]> {
        let info = self.tensors.get(name)?;
        let start = info.data_offset;
        let end = start + info.size;
        if end <= data.len() {
            Some(&data[start..end])
        } else {
            None
        }
    }

    /// Iterate over all tensors.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &SafeTensorInfo)> {
        self.tensors.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Get total data size (excluding header).
    pub fn data_size(&self) -> usize {
        self.tensors.values().map(|t| t.size).sum()
    }
}

/// Parse the JSON header without external dependencies.
fn parse_header(json: &str, data_start: usize) -> Result<(HashMap<String, SafeTensorInfo>, HashMap<String, String>)> {
    let mut tensors = HashMap::new();
    let mut metadata = HashMap::new();

    // Trim whitespace
    let json = json.trim();

    // Must start with {
    if !json.starts_with('{') || !json.ends_with('}') {
        return Err(Error::model_load("safetensors", "header must be a JSON object"));
    }

    // Simple JSON parser for SafeTensors format
    // Format: { "tensor_name": {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, 8]}, ... }

    let inner = &json[1..json.len()-1];
    let mut offset = 0;

    while offset < inner.len() {
        // Skip whitespace and commas
        while offset < inner.len() {
            let c = inner.as_bytes()[offset];
            if c == b' ' || c == b'\n' || c == b'\r' || c == b'\t' || c == b',' {
                offset += 1;
            } else {
                break;
            }
        }

        if offset >= inner.len() {
            break;
        }

        // Parse key
        let (key, new_offset) = parse_json_string(&inner[offset..])?;
        offset += new_offset;

        // Skip : and whitespace
        while offset < inner.len() {
            let c = inner.as_bytes()[offset];
            if c == b' ' || c == b'\n' || c == b'\r' || c == b'\t' || c == b':' {
                offset += 1;
            } else {
                break;
            }
        }

        // Handle __metadata__ specially
        if key == "__metadata__" {
            let (meta, new_offset) = parse_json_object(&inner[offset..])?;
            offset += new_offset;
            metadata = meta;
            continue;
        }

        // Parse tensor info object
        let (tensor_info, new_offset) = parse_tensor_info(&inner[offset..], &key, data_start)?;
        offset += new_offset;
        tensors.insert(key, tensor_info);
    }

    Ok((tensors, metadata))
}

/// Parse a JSON string, returning the string content and bytes consumed.
fn parse_json_string(s: &str) -> Result<(String, usize)> {
    if !s.starts_with('"') {
        return Err(Error::model_load("safetensors", "expected string"));
    }

    let mut result = String::new();
    let mut i = 1;
    let bytes = s.as_bytes();

    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                return Ok((result, i + 1));
            }
            b'\\' if i + 1 < bytes.len() => {
                match bytes[i + 1] {
                    b'"' => result.push('"'),
                    b'\\' => result.push('\\'),
                    b'n' => result.push('\n'),
                    b'r' => result.push('\r'),
                    b't' => result.push('\t'),
                    _ => {
                        result.push('\\');
                        result.push(bytes[i + 1] as char);
                    }
                }
                i += 2;
            }
            c => {
                result.push(c as char);
                i += 1;
            }
        }
    }

    Err(Error::model_load("safetensors", "unterminated string"))
}

/// Parse a simple JSON object with string values.
fn parse_json_object(s: &str) -> Result<(HashMap<String, String>, usize)> {
    if !s.starts_with('{') {
        return Err(Error::model_load("safetensors", "expected object"));
    }

    let result = HashMap::new();
    let mut depth = 0;
    let mut i = 0;

    for (idx, c) in s.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    i = idx + 1;
                    break;
                }
            }
            _ => {}
        }
    }

    // Simple extraction - for metadata we just skip it for now
    // Real implementation would parse key-value pairs

    Ok((result, i))
}

/// Parse tensor info from JSON object.
fn parse_tensor_info(s: &str, name: &str, data_start: usize) -> Result<(SafeTensorInfo, usize)> {
    if !s.starts_with('{') {
        return Err(Error::model_load("safetensors", format!("expected object for tensor '{}'", name)));
    }

    // Find matching closing brace
    let mut depth = 0;
    let mut end = 0;
    for (idx, c) in s.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = idx + 1;
                    break;
                }
            }
            _ => {}
        }
    }

    let obj = &s[1..end-1];

    // Parse fields
    let mut dtype = None;
    let mut shape = None;
    let mut data_offsets = None;

    // Find "dtype"
    if let Some(pos) = obj.find("\"dtype\"") {
        let after = &obj[pos + 7..];
        let after = after.trim_start_matches(|c: char| c.is_whitespace() || c == ':');
        let (dtype_str, _) = parse_json_string(after)?;
        dtype = Some(parse_dtype(&dtype_str)?);
    }

    // Find "shape"
    if let Some(pos) = obj.find("\"shape\"") {
        let after = &obj[pos + 7..];
        let after = after.trim_start_matches(|c: char| c.is_whitespace() || c == ':');
        shape = Some(parse_shape(after)?);
    }

    // Find "data_offsets"
    if let Some(pos) = obj.find("\"data_offsets\"") {
        let after = &obj[pos + 14..];
        let after = after.trim_start_matches(|c: char| c.is_whitespace() || c == ':');
        data_offsets = Some(parse_offsets(after)?);
    }

    let dtype = dtype.ok_or_else(|| Error::model_load("safetensors", format!("missing dtype for '{}'", name)))?;
    let shape = shape.ok_or_else(|| Error::model_load("safetensors", format!("missing shape for '{}'", name)))?;
    let (start_offset, end_offset) = data_offsets.ok_or_else(|| Error::model_load("safetensors", format!("missing data_offsets for '{}'", name)))?;

    let info = SafeTensorInfo {
        name: name.to_string(),
        dtype,
        shape,
        data_offset: data_start + start_offset,
        size: end_offset - start_offset,
    };

    Ok((info, end))
}

/// Parse dtype string to DType.
fn parse_dtype(s: &str) -> Result<DType> {
    match s {
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "I32" => Ok(DType::I32),
        "I64" => Ok(DType::I64),
        "I8" => Ok(DType::I8),
        "U8" => Ok(DType::U8),
        "U32" => Ok(DType::U32),
        "BOOL" => Ok(DType::Bool),
        "F8_E4M3" => Ok(DType::F8E4M3),
        "F8_E5M2" => Ok(DType::F8E5M2),
        _ => Err(Error::model_load("safetensors", format!("unknown dtype: {}", s))),
    }
}

/// Parse shape array.
fn parse_shape(s: &str) -> Result<Shape> {
    let s = s.trim();
    if !s.starts_with('[') {
        return Err(Error::model_load("safetensors", "shape must be an array"));
    }

    let end = s.find(']').ok_or_else(|| Error::model_load("safetensors", "unterminated shape array"))?;
    let inner = &s[1..end];

    let dims: Vec<usize> = inner
        .split(',')
        .filter(|x| !x.trim().is_empty())
        .map(|x| x.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| Error::model_load("safetensors", format!("invalid shape dimension: {}", e)))?;

    Ok(Shape::new(dims))
}

/// Parse data_offsets array [start, end].
fn parse_offsets(s: &str) -> Result<(usize, usize)> {
    let s = s.trim();
    if !s.starts_with('[') {
        return Err(Error::model_load("safetensors", "data_offsets must be an array"));
    }

    let end = s.find(']').ok_or_else(|| Error::model_load("safetensors", "unterminated offsets array"))?;
    let inner = &s[1..end];

    let parts: Vec<usize> = inner
        .split(',')
        .map(|x| x.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| Error::model_load("safetensors", format!("invalid offset: {}", e)))?;

    if parts.len() != 2 {
        return Err(Error::model_load("safetensors", "data_offsets must have exactly 2 elements"));
    }

    Ok((parts[0], parts[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dtype() {
        assert!(matches!(parse_dtype("F32"), Ok(DType::F32)));
        assert!(matches!(parse_dtype("F16"), Ok(DType::F16)));
        assert!(matches!(parse_dtype("BF16"), Ok(DType::BF16)));
    }

    #[test]
    fn test_parse_shape() {
        let shape = parse_shape("[1, 2, 3]").unwrap();
        assert_eq!(shape.dims(), &[1, 2, 3]);

        let shape = parse_shape("[768]").unwrap();
        assert_eq!(shape.dims(), &[768]);
    }

    #[test]
    fn test_parse_offsets() {
        let (start, end) = parse_offsets("[0, 1024]").unwrap();
        assert_eq!(start, 0);
        assert_eq!(end, 1024);
    }
}
