//! GGUF file format reader.
//!
//! GGUF is the on-disk format used by llama.cpp. It is a single binary
//! file containing a header, a key/value metadata block, and a tensor
//! directory followed by raw tensor bytes. This module ships a *read-only*
//! parser for the header + metadata + tensor directory. It does **not**
//! load or run the tensors — that is what the `llamacpp` backend is for.
//!
//! The parser is useful for:
//!
//! * Introspection (`eoc-local list-models`).
//! * Populating the model registry.
//! * Sanity-checking files before sending them to llama.cpp.
//!
//! Reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
//!
//! ## WASM
//!
//! This module is portable Rust and compiles to `wasm32-unknown-unknown`.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::error::{LocalError, LocalResult};

/// GGUF magic number (`GGUF` little-endian, ASCII).
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Minimum GGUF version this parser understands. v1 and v2 are
/// historical; v3 is the format used by llama.cpp from 2024 onwards.
pub const MIN_SUPPORTED_VERSION: u32 = 1;
/// Maximum GGUF version this parser understands.
pub const MAX_SUPPORTED_VERSION: u32 = 3;

/// Top-level GGUF file representation.
#[derive(Debug, Clone)]
pub struct GgufFile {
    /// Path on disk the file was loaded from.
    pub path: PathBuf,
    /// Parsed header.
    pub header: GgufHeader,
    /// Parsed metadata (key → typed value).
    pub metadata: Vec<(String, GgufMetadataValue)>,
    /// Tensor directory entries.
    pub tensors: Vec<GgufTensorInfo>,
}

/// GGUF header.
#[derive(Debug, Clone, Copy)]
pub struct GgufHeader {
    /// Format magic (`GGUF`).
    pub magic: u32,
    /// Format version.
    pub version: u32,
    /// Number of tensors declared in the file.
    pub tensor_count: u64,
    /// Number of metadata key/value pairs.
    pub metadata_kv_count: u64,
}

/// GGUF tensor directory entry. Tensor *data* is not loaded.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name (e.g. `blk.0.attn_q.weight`).
    pub name: String,
    /// Shape (dimensions, fastest-varying last).
    pub dims: Vec<u64>,
    /// Underlying ggml type code (see the GGUF spec).
    pub ggml_type: u32,
    /// Byte offset of the tensor data inside the file (relative to the
    /// start of the tensor-data section).
    pub offset: u64,
}

/// Strongly-typed GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    /// u8 scalar.
    U8(u8),
    /// i8 scalar.
    I8(i8),
    /// u16 scalar.
    U16(u16),
    /// i16 scalar.
    I16(i16),
    /// u32 scalar.
    U32(u32),
    /// i32 scalar.
    I32(i32),
    /// u64 scalar.
    U64(u64),
    /// i64 scalar.
    I64(i64),
    /// f32 scalar.
    F32(f32),
    /// f64 scalar.
    F64(f64),
    /// Bool scalar.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Heterogeneous array — element type encoded inline.
    Array(Vec<GgufMetadataValue>),
}

impl GgufMetadataValue {
    /// Render as a short string suitable for diagnostic output.
    pub fn render(&self) -> String {
        match self {
            GgufMetadataValue::U8(v) => v.to_string(),
            GgufMetadataValue::I8(v) => v.to_string(),
            GgufMetadataValue::U16(v) => v.to_string(),
            GgufMetadataValue::I16(v) => v.to_string(),
            GgufMetadataValue::U32(v) => v.to_string(),
            GgufMetadataValue::I32(v) => v.to_string(),
            GgufMetadataValue::U64(v) => v.to_string(),
            GgufMetadataValue::I64(v) => v.to_string(),
            GgufMetadataValue::F32(v) => format!("{v}"),
            GgufMetadataValue::F64(v) => format!("{v}"),
            GgufMetadataValue::Bool(v) => v.to_string(),
            GgufMetadataValue::String(s) => format!("\"{s}\""),
            GgufMetadataValue::Array(a) => format!("[array; len={}]", a.len()),
        }
    }
}

impl GgufFile {
    /// Open and fully parse the header / metadata / tensor directory of
    /// the GGUF file at `path`. Tensor data is **not** loaded; only the
    /// directory entries are read.
    pub fn open(path: impl AsRef<Path>) -> LocalResult<Self> {
        let path = path.as_ref().to_path_buf();
        let f = File::open(&path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => LocalError::ModelNotFound(path.display().to_string()),
            _ => LocalError::Io(e.to_string()),
        })?;
        let mut r = BufReader::new(f);
        Self::parse(&path, &mut r)
    }

    fn parse<R: Read + Seek>(path: &Path, r: &mut R) -> LocalResult<Self> {
        let header = read_header(r)?;
        if header.magic != GGUF_MAGIC {
            return Err(LocalError::InvalidModelFormat(format!(
                "bad magic 0x{:08x}",
                header.magic
            )));
        }
        if !(MIN_SUPPORTED_VERSION..=MAX_SUPPORTED_VERSION).contains(&header.version) {
            return Err(LocalError::InvalidModelFormat(format!(
                "unsupported gguf version {}",
                header.version
            )));
        }
        let metadata = read_metadata(r, header.metadata_kv_count, header.version)?;
        let tensors = read_tensor_directory(r, header.tensor_count, header.version)?;
        Ok(GgufFile {
            path: path.to_path_buf(),
            header,
            metadata,
            tensors,
        })
    }

    /// Look up a metadata value by exact key. Returns `None` if absent.
    pub fn meta(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.metadata.iter().find_map(|(k, v)| (k == key).then_some(v))
    }

    /// Convenience: read the architecture string
    /// (`general.architecture`), if present.
    pub fn architecture(&self) -> Option<&str> {
        match self.meta("general.architecture")? {
            GgufMetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Convenience: read the model name (`general.name`), if present.
    pub fn name(&self) -> Option<&str> {
        match self.meta("general.name")? {
            GgufMetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------
// Binary parsing.

fn read_header<R: Read>(r: &mut R) -> LocalResult<GgufHeader> {
    let magic = read_u32(r)?;
    let version = read_u32(r)?;
    // v1 used u32 counts; v2+ uses u64. We attempt to discriminate by
    // version.
    let (tensor_count, metadata_kv_count) = if version == 1 {
        (read_u32(r)? as u64, read_u32(r)? as u64)
    } else {
        (read_u64(r)?, read_u64(r)?)
    };
    Ok(GgufHeader {
        magic,
        version,
        tensor_count,
        metadata_kv_count,
    })
}

fn read_metadata<R: Read + Seek>(
    r: &mut R,
    count: u64,
    version: u32,
) -> LocalResult<Vec<(String, GgufMetadataValue)>> {
    let mut out = Vec::with_capacity(count.min(4096) as usize);
    for _ in 0..count {
        let key = read_string(r, version)?;
        let value_type = read_u32(r)?;
        let value = read_value(r, value_type, version)?;
        out.push((key, value));
    }
    Ok(out)
}

fn read_tensor_directory<R: Read + Seek>(
    r: &mut R,
    count: u64,
    version: u32,
) -> LocalResult<Vec<GgufTensorInfo>> {
    let mut out = Vec::with_capacity(count.min(4096) as usize);
    for _ in 0..count {
        let name = read_string(r, version)?;
        let n_dims = read_u32(r)?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            // Dim sizes: u32 in v1, u64 in v2+.
            if version == 1 {
                dims.push(read_u32(r)? as u64);
            } else {
                dims.push(read_u64(r)?);
            }
        }
        let ggml_type = read_u32(r)?;
        let offset = read_u64(r)?;
        out.push(GgufTensorInfo {
            name,
            dims,
            ggml_type,
            offset,
        });
    }
    Ok(out)
}

fn read_value<R: Read + Seek>(
    r: &mut R,
    type_code: u32,
    version: u32,
) -> LocalResult<GgufMetadataValue> {
    // GGUF value-type codes (from the spec):
    //   0=u8 1=i8 2=u16 3=i16 4=u32 5=i32 6=f32 7=bool 8=string 9=array
    //  10=u64 11=i64 12=f64
    match type_code {
        0 => Ok(GgufMetadataValue::U8(read_u8(r)?)),
        1 => Ok(GgufMetadataValue::I8(read_u8(r)? as i8)),
        2 => Ok(GgufMetadataValue::U16(read_u16(r)?)),
        3 => Ok(GgufMetadataValue::I16(read_u16(r)? as i16)),
        4 => Ok(GgufMetadataValue::U32(read_u32(r)?)),
        5 => Ok(GgufMetadataValue::I32(read_u32(r)? as i32)),
        6 => Ok(GgufMetadataValue::F32(f32::from_bits(read_u32(r)?))),
        7 => Ok(GgufMetadataValue::Bool(read_u8(r)? != 0)),
        8 => Ok(GgufMetadataValue::String(read_string(r, version)?)),
        9 => {
            let elem_type = read_u32(r)?;
            // Array length: u32 in v1, u64 in v2+.
            let n = if version == 1 {
                read_u32(r)? as u64
            } else {
                read_u64(r)?
            };
            // Bound the in-memory array to a sane limit so a corrupted
            // file can't make us allocate the universe.
            let bounded = n.min(1_000_000) as usize;
            let mut elems = Vec::with_capacity(bounded);
            for _ in 0..n {
                elems.push(read_value(r, elem_type, version)?);
            }
            Ok(GgufMetadataValue::Array(elems))
        }
        10 => Ok(GgufMetadataValue::U64(read_u64(r)?)),
        11 => Ok(GgufMetadataValue::I64(read_u64(r)? as i64)),
        12 => Ok(GgufMetadataValue::F64(f64::from_bits(read_u64(r)?))),
        other => Err(LocalError::InvalidModelFormat(format!(
            "unknown gguf value type {other}"
        ))),
    }
}

fn read_string<R: Read>(r: &mut R, version: u32) -> LocalResult<String> {
    let len = if version == 1 {
        read_u32(r)? as u64
    } else {
        read_u64(r)?
    };
    // Bound to keep things sane.
    if len > 16 * 1024 * 1024 {
        return Err(LocalError::InvalidModelFormat(format!(
            "gguf string length {len} exceeds 16 MiB cap"
        )));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).map_err(|e| LocalError::Io(e.to_string()))?;
    String::from_utf8(buf).map_err(|e| LocalError::InvalidModelFormat(format!("utf-8: {e}")))
}

fn read_u8<R: Read>(r: &mut R) -> LocalResult<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(|e| LocalError::Io(e.to_string()))?;
    Ok(b[0])
}

fn read_u16<R: Read>(r: &mut R) -> LocalResult<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(|e| LocalError::Io(e.to_string()))?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32<R: Read>(r: &mut R) -> LocalResult<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| LocalError::Io(e.to_string()))?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> LocalResult<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| LocalError::Io(e.to_string()))?;
    Ok(u64::from_le_bytes(b))
}

// Allow `Seek` import — currently unused but kept for future tensor
// data offset jumps.
#[allow(dead_code)]
fn skip_to<R: Read + Seek>(r: &mut R, offset: u64) -> LocalResult<()> {
    r.seek(SeekFrom::Start(offset))
        .map_err(|e| LocalError::Io(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    /// Build a minimal valid GGUF v3 byte buffer with no metadata and
    /// no tensors. Useful for exercising the parser without a real file.
    fn synthetic_gguf_v3_empty() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes()); // magic
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&0u64.to_le_bytes()); // metadata kv count
        buf
    }

    /// Build a synthetic GGUF with one string metadata key/value and
    /// one fake tensor entry.
    fn synthetic_gguf_v3_simple() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata kv

        // Metadata: key = "general.architecture", value = String("llama")
        write_v3_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes()); // type = string
        write_v3_string(&mut buf, "llama");

        // Tensor directory: name="t0", dims=[2,3], ggml_type=0, offset=0
        write_v3_string(&mut buf, "t0");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // ggml type
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        buf
    }

    fn write_v3_string(buf: &mut Vec<u8>, s: &str) {
        let bytes = s.as_bytes();
        buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(bytes);
    }

    #[test]
    fn parse_empty_gguf_v3() {
        let bytes = synthetic_gguf_v3_empty();
        let mut cur = Cursor::new(bytes);
        let f = GgufFile::parse(Path::new("synth.gguf"), &mut cur).unwrap();
        assert_eq!(f.header.magic, GGUF_MAGIC);
        assert_eq!(f.header.version, 3);
        assert_eq!(f.header.tensor_count, 0);
        assert!(f.metadata.is_empty());
        assert!(f.tensors.is_empty());
    }

    #[test]
    fn parse_simple_gguf_v3() {
        let bytes = synthetic_gguf_v3_simple();
        let mut cur = Cursor::new(bytes);
        let f = GgufFile::parse(Path::new("synth.gguf"), &mut cur).unwrap();
        assert_eq!(f.tensors.len(), 1);
        assert_eq!(f.tensors[0].name, "t0");
        assert_eq!(f.tensors[0].dims, vec![2, 3]);
        assert_eq!(f.architecture(), Some("llama"));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let mut cur = Cursor::new(buf);
        assert!(GgufFile::parse(Path::new("synth.gguf"), &mut cur).is_err());
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&999u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let mut cur = Cursor::new(buf);
        assert!(GgufFile::parse(Path::new("synth.gguf"), &mut cur).is_err());
    }

    #[test]
    fn round_trip_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        let mut f = File::create(&path).unwrap();
        f.write_all(&synthetic_gguf_v3_simple()).unwrap();
        drop(f);
        let parsed = GgufFile::open(&path).unwrap();
        assert_eq!(parsed.architecture(), Some("llama"));
    }

    #[test]
    fn missing_file_yields_model_not_found() {
        let r = GgufFile::open("/no/such/path.gguf");
        assert!(matches!(r, Err(LocalError::ModelNotFound(_))));
    }
}
