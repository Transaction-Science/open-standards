//! Lazy loading for model weights.
//!
//! ## How It Works
//!
//! 1. Model file is memory-mapped (mmap) - no initial RAM usage
//! 2. Metal buffer is created pointing to mmap'd memory
//! 3. When GPU accesses the buffer, OS triggers page faults
//! 4. Pages are loaded from disk on-demand
//! 5. Unused pages can be evicted under memory pressure
//!
//! ## Benefits
//!
//! - **Near-instant model "loading"**: Just mmap, don't read
//! - **Memory efficiency**: Only used weights in RAM
//! - **Automatic eviction**: OS handles memory pressure
//! - **UMA advantage**: No separate CPU→GPU copy needed
//!
//! ## Example
//!
//! ```rust,ignore
//! // "Load" a 70B model in milliseconds
//! let loader = LazyLoader::new(device);
//! let weights = loader.load_safetensors("model.safetensors")?;
//!
//! // First access triggers page faults - weights loaded on demand
//! let output = model.forward(&input, &weights)?;
//! ```

use super::MetalDevice;
use crate::core::{DType, Error, Result, Shape};
use half::f16;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Convert a BF16 value (as u16 bits) to F16 (as u16 bits).
///
/// BF16: 1 sign, 8 exponent, 7 mantissa (same exp range as F32)
/// F16:  1 sign, 5 exponent, 10 mantissa
///
/// Since BF16 has a much larger exponent range than F16, values may
/// overflow to infinity or underflow to zero during conversion.
#[inline]
fn bf16_to_f16(bf16_bits: u16) -> u16 {
    let sign = ((bf16_bits >> 15) & 1) as u32;
    let exp = ((bf16_bits >> 7) & 0xff) as u32;  // 8-bit exponent
    let mant = (bf16_bits & 0x7f) as u32;         // 7-bit mantissa

    if exp == 0 {
        // Zero or subnormal BF16 -> zero in F16 (subnormals too small)
        return (sign << 15) as u16;
    }

    if exp == 0xff {
        // Inf or NaN
        if mant == 0 {
            // Infinity
            return ((sign << 15) | 0x7c00) as u16;
        } else {
            // NaN - preserve some mantissa bits
            return ((sign << 15) | 0x7e00) as u16;
        }
    }

    // Normal number: convert exponent from BF16 bias (127) to F16 bias (15)
    let bf16_exp_unbiased = exp as i32 - 127;
    let f16_exp = bf16_exp_unbiased + 15;

    if f16_exp >= 31 {
        // Overflow to infinity
        return ((sign << 15) | 0x7c00) as u16;
    }

    if f16_exp <= 0 {
        // Underflow to zero or subnormal F16
        if f16_exp < -10 {
            // Too small, flush to zero
            return (sign << 15) as u16;
        }
        // Subnormal: shift mantissa right using saturating arithmetic to prevent underflow
        let shift = (1i32.saturating_sub(f16_exp)) as u32;
        let full_mant = (1u32 << 7) | mant; // Add implicit leading 1
        let subnorm_mant = full_mant.checked_shr(shift).unwrap_or(0);
        let f16_mant = subnorm_mant.saturating_mul(8); // equivalent to << 3, but saturating
        return ((sign << 15) | (f16_mant & 0x03FF)) as u16;
    }

    // Normal F16: expand 7-bit mantissa to 10-bit (pad with zeros)
    let f16_mant = mant << 3;
    ((sign << 15) | ((f16_exp as u32) << 10) | f16_mant) as u16
}

/// Convert a buffer of BF16 values to F16 values in-place or to new buffer.
fn convert_bf16_to_f16_buffer(src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    assert!(src.len() % 2 == 0);

    let src_u16 = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u16, src.len() / 2)
    };
    let dst_u16 = unsafe {
        std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u16, dst.len() / 2)
    };

    for (i, &bf16) in src_u16.iter().enumerate() {
        dst_u16[i] = bf16_to_f16(bf16);
    }
}

/// Convert a buffer of F32 values to F16 values in-place or to new buffer.
fn convert_f32_to_f16_buffer(src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len() / 4, dst.len() / 2);
    let src_f32 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
    let dst_f16 = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f16, dst.len() / 2) };
    
    for (i, &f) in src_f32.iter().enumerate() {
        dst_f16[i] = f16::from_f32(f);
    }
}

/// A lazily-loaded tensor backed by mmap.
///
/// For BF16/F32 weights, the raw buffer holds the original mmap'd data
/// and `buffer()` lazily converts to F16 on first access (cached via OnceLock).
/// For F16 weights, `buffer()` returns the raw mmap'd buffer directly (zero-copy).
#[cfg(feature = "metal")]
pub struct LazyTensor {
    /// Raw Metal buffer — may be BF16/F32/F16 or quantized blocks (Q5_0/Q4_K/etc), typically mmap-backed
    raw_buffer: metal::Buffer,
    /// Shape
    shape: Shape,
    /// Logical data type (after conversion). For native-quant weights this is F16
    /// (legacy compat), but `ggml_type` carries the true on-disk format.
    dtype: DType,
    /// Original GGUF block format. Q5_0/Q4_K/Q5_K/Q6_K/Q8_0 stay native and are
    /// matmul'd directly via the corresponding kernel; F32/BF16/F16 are stored
    /// in raw_buffer as F16.
    ggml_type: crate::inference::formats::GgmlType,
    /// Offset into the mmap'd file
    offset: usize,
    /// Size in bytes (of raw buffer)
    size: usize,
    /// Reference to keep mmap alive
    _mmap: Option<Arc<memmap2::Mmap>>,
    /// Source file path (for debugging)
    source: PathBuf,
    /// Tensor name
    name: String,
    /// Cached F16 buffer for BF16/F32 tensors (lazy one-time conversion)
    f16_cache: std::sync::OnceLock<metal::Buffer>,
}

#[cfg(feature = "metal")]
impl LazyTensor {
    /// Get the F16 Metal buffer for GPU compute.
    ///
    /// For F16 tensors: returns the raw mmap'd buffer directly (zero-copy).
    /// For BF16/F32 tensors: lazily converts to F16 on first call, caches the result.
    /// For integer types: returns the raw buffer as-is.
    pub fn buffer(&self) -> &metal::Buffer {
        match self.dtype {
            DType::F16 | DType::I32 | DType::U32 | DType::I8 | DType::U8 => &self.raw_buffer,
            DType::BF16 | DType::F32 => {
                self.f16_cache.get_or_init(|| self.convert_to_f16())
            }
            _ => &self.raw_buffer,
        }
    }

    /// Get the raw Metal buffer (original dtype, mmap-backed).
    ///
    /// Use this for operations that handle multiple dtypes directly
    /// (e.g., `to_f32_vec()`, madvise hints). For GPU compute, use `buffer()`.
    pub fn raw_buffer(&self) -> &metal::Buffer {
        &self.raw_buffer
    }

    /// Convert raw BF16/F32 buffer to F16.
    /// Called once per tensor, result cached in `f16_cache`.
    fn convert_to_f16(&self) -> metal::Buffer {
        let numel = self.shape.numel();
        let f16_size = numel * 2;
        let device = self.raw_buffer.device();
        let new_buffer = device.new_buffer(
            f16_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            let src = std::slice::from_raw_parts(
                self.raw_buffer.contents() as *const u8,
                self.raw_buffer.length() as usize,
            );
            let dst = std::slice::from_raw_parts_mut(
                new_buffer.contents() as *mut u8,
                f16_size,
            );
            match self.dtype {
                DType::BF16 => convert_bf16_to_f16_buffer(src, dst),
                DType::F32 => convert_f32_to_f16_buffer(src, dst),
                _ => unreachable!(),
            }
        }
        new_buffer.did_modify_range(metal::NSRange::new(0, f16_size as u64));
        new_buffer
    }

    /// Get shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the original GGUF block format (Q5_0, Q4_K, F16, etc).
    /// Used by callers that dispatch quant-aware matmul kernels.
    pub fn ggml_type(&self) -> crate::inference::formats::GgmlType {
        self.ggml_type
    }

    /// Get size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Create a resident tensor (not backed by file).
    pub fn new_resident(
        buffer: metal::Buffer,
        shape: Shape,
        dtype: DType,
        name: String,
    ) -> Self {
        let size = shape.numel() * dtype.size_bytes();
        Self {
            raw_buffer: buffer,
            shape,
            dtype,
            ggml_type: crate::inference::formats::GgmlType::F16,
            offset: 0,
            size,
            _mmap: None,
            source: PathBuf::from("resident"),
            name,
            f16_cache: std::sync::OnceLock::new(),
        }
    }

    /// Get the device pointer.
    pub fn device_ptr(&self) -> Option<crate::hal::DevicePtr> {
        use metal::foreign_types::ForeignType;
        Some(crate::hal::DevicePtr::new(self.raw_buffer.as_ptr() as u64))
    }

    /// Check if data is resident in memory.
    ///
    /// Returns approximate percentage of pages in RAM.
    pub fn residency(&self) -> f64 {
        #[cfg(unix)]
        {
            let page_size: usize = 16384; // Apple Silicon uses 16KB pages
            if self.size == 0 { return 1.0; }
            let ptr = self.raw_buffer.contents();
            let aligned_ptr = (ptr as usize & !(page_size - 1)) as *mut libc::c_void;
            let end = ptr as usize + self.size;
            let aligned_len = end - (aligned_ptr as usize);
            let num_pages = (aligned_len + page_size - 1) / page_size;
            let mut vec = vec![0i8; num_pages];
            let ret = unsafe {
                libc::mincore(aligned_ptr, aligned_len, vec.as_mut_ptr() as *mut _)
            };
            if ret != 0 { return 0.0; }
            let resident = vec.iter().filter(|&&v| v != 0).count();
            resident as f64 / num_pages as f64
        }
        #[cfg(not(unix))]
        { 0.0 }
    }

    /// Prefetch data into memory.
    pub fn prefetch(&self) {
        // Touch all pages to load them
        // This is optional - GPU access will load them anyway
        unsafe {
            let ptr = self.raw_buffer.contents() as *const u8;
            let page_size = 16384; // 16KB pages on Apple Silicon
            for i in (0..self.size).step_by(page_size) {
                std::ptr::read_volatile(ptr.add(i));
            }
        }
    }

    /// Advise OS about access pattern.
    pub fn advise_sequential(&self) {
        #[cfg(unix)]
        unsafe {
            let ptr = self.raw_buffer.contents();
            libc::madvise(ptr, self.size, libc::MADV_SEQUENTIAL);
        }
    }

    /// Advise the OS to asynchronously prefetch this tensor's pages from NVMe.
    ///
    /// Unlike `prefetch()` which blocks while touching every page, this uses
    /// `madvise(MADV_WILLNEED)` to issue an asynchronous read-ahead.
    /// Only applies to mmap-backed tensors; no-op for copied buffers.
    pub fn advise_willneed(&self) {
        if self._mmap.is_none() { return; }
        #[cfg(unix)]
        unsafe {
            let ptr = self.raw_buffer.contents();
            libc::madvise(ptr, self.size, libc::MADV_WILLNEED);
        }
    }

    /// Read buffer contents as f32 vector.
    ///
    /// Converts from the stored dtype (F16/BF16/F32) to F32.
    /// Reads from the raw mmap'd buffer directly (handles all dtypes).
    /// On Apple Silicon UMA, reads directly from shared memory.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        let numel = self.shape.numel();
        unsafe {
            let ptr = self.raw_buffer.contents() as *const u8;
            match self.dtype {
                DType::F32 => {
                    let f32_ptr = ptr as *const f32;
                    let slice = std::slice::from_raw_parts(f32_ptr, numel);
                    Ok(slice.to_vec())
                }
                DType::F16 => {
                    let f16_ptr = ptr as *const f16;
                    let slice = std::slice::from_raw_parts(f16_ptr, numel);
                    Ok(slice.iter().map(|v| v.to_f32()).collect())
                }
                DType::BF16 => {
                    let u16_ptr = ptr as *const u16;
                    let slice = std::slice::from_raw_parts(u16_ptr, numel);
                    Ok(slice.iter().map(|&bits| {
                        f32::from_bits((bits as u32) << 16)
                    }).collect())
                }
                _ => Err(Error::internal(format!(
                    "Unsupported dtype {:?} for to_f32_vec", self.dtype
                ))),
            }
        }
    }

    /// Advise OS this data won't be needed soon.
    pub fn advise_dontneed(&self) {
        #[cfg(unix)]
        unsafe {
            let ptr = self.raw_buffer.contents();
            libc::madvise(ptr, self.size, libc::MADV_DONTNEED);
        }
    }
}

/// Lazy loader for model weights.
#[cfg(feature = "metal")]
pub struct LazyLoader {
    /// Metal device
    device: Arc<MetalDevice>,
    /// Cached mmaps (to keep them alive)
    mmaps: parking_lot::RwLock<HashMap<PathBuf, Arc<memmap2::Mmap>>>,
    /// Statistics
    stats: parking_lot::RwLock<LazyLoadStats>,
}

/// Lazy loading statistics.
#[derive(Debug, Clone, Default)]
pub struct LazyLoadStats {
    /// Files mapped
    pub files_mapped: usize,
    /// Total bytes mapped
    pub bytes_mapped: usize,
    /// Tensors created
    pub tensors_created: usize,
}

#[cfg(feature = "metal")]
impl LazyLoader {
    /// Create a new lazy loader.
    pub fn new(device: Arc<MetalDevice>) -> Self {
        Self {
            device,
            mmaps: parking_lot::RwLock::new(HashMap::new()),
            stats: parking_lot::RwLock::new(LazyLoadStats::default()),
        }
    }

    /// Get or create mmap for a file.
    fn get_or_create_mmap(&self, path: &Path) -> Result<Arc<memmap2::Mmap>> {
        let canonical = path.canonicalize().map_err(|e| Error::Io {
            operation: "canonicalize".into(),
            message: format!("{}: {}", path.display(), e),
            #[cfg(feature = "std")]
            source: None,
        })?;

        // Check cache
        {
            let mmaps = self.mmaps.read();
            if let Some(mmap) = mmaps.get(&canonical) {
                return Ok(Arc::clone(mmap));
            }
        }

        // Create new mmap
        let file = std::fs::File::open(&canonical).map_err(|e| Error::Io {
            operation: "open".into(),
            message: format!("{}: {}", path.display(), e),
            #[cfg(feature = "std")]
            source: None,
        })?;

        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Error::Io {
            operation: "mmap".into(),
            message: format!("{}: {}", path.display(), e),
            #[cfg(feature = "std")]
            source: None,
        })?;

        // Advise OS about access pattern
        #[cfg(unix)]
        unsafe {
            libc::madvise(
                mmap.as_ptr() as *mut _,
                mmap.len(),
                libc::MADV_RANDOM, // Random access for model weights
            );
        }

        let mmap = Arc::new(mmap);

        // Cache it
        {
            let mut mmaps = self.mmaps.write();
            let mut stats = self.stats.write();
            stats.files_mapped += 1;
            stats.bytes_mapped += mmap.len();
            mmaps.insert(canonical, Arc::clone(&mmap));
        }

        Ok(mmap)
    }

    /// Load a safetensors file lazily.
    #[cfg(feature = "safetensors")]
    pub fn load_safetensors(&self, path: &Path) -> Result<HashMap<String, LazyTensor>> {
        use safetensors::SafeTensors;

        let mmap = self.get_or_create_mmap(path)?;

        // Parse safetensors header (only reads header, not data)
        let tensors = SafeTensors::deserialize(&mmap).map_err(|e| Error::ModelLoad {
            model: path.display().to_string(),
            message: e.to_string(),
            #[cfg(feature = "std")]
            source: None,
        })?;

        let mut result = HashMap::new();

        for (name, view) in tensors.tensors() {
            let dtype = safetensors_dtype_to_dtype(view.dtype());
            let shape = Shape::new(view.shape().to_vec());
            let data = view.data();

            // Calculate offset into mmap
            let offset = data.as_ptr() as usize - mmap.as_ptr() as usize;
            let size = data.len();

            // Use zero-copy when data pointer is page-aligned (16KB on Apple Silicon),
            // otherwise fall back to a copy to ensure Metal buffer alignment.
            let page_size = 16384usize; // Apple Silicon page size
            let buffer = if (data.as_ptr() as usize) % page_size == 0 && size >= page_size {
                self.device.raw().new_buffer_with_bytes_no_copy(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    metal::MTLResourceOptions::StorageModeShared
                        | metal::MTLResourceOptions::CPUCacheModeWriteCombined
                        | metal::MTLResourceOptions::HazardTrackingModeUntracked,
                    None,
                )
            } else {
                let buf = self.device.raw().new_buffer(
                    data.len() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        buf.contents() as *mut u8,
                        data.len(),
                    );
                }
                buf
            };

            let tensor = LazyTensor {
                raw_buffer: buffer,
                shape,
                dtype,
                ggml_type: crate::inference::formats::GgmlType::F16,
                offset,
                size,
                _mmap: Some(Arc::clone(&mmap)),
                source: path.to_path_buf(),
                name: name.to_string(),
                f16_cache: std::sync::OnceLock::new(),
            };

            // Update stats
            {
                let mut stats = self.stats.write();
                stats.tensors_created += 1;
            }

            result.insert(name.to_string(), tensor);
        }

        Ok(result)
    }

    /// Load a safetensors file lazily using our custom parser.
    ///
    /// All dtypes (BF16/F32/F16) are loaded via zero-copy mmap.
    /// BF16→F16 conversion happens lazily on first `buffer()` access.
    #[cfg(not(feature = "safetensors"))]
    pub fn load_safetensors(&self, path: &Path) -> Result<HashMap<String, LazyTensor>> {
        use crate::inference::formats::SafeTensorsFile;

        // Memory-map the file instead of reading it entirely into RAM
        let file = std::fs::File::open(path).map_err(|e| Error::Io { operation: "open".to_string(), message: format!("{}: {}", path.display(), e), #[cfg(feature = "std")] source: None })?;
        let file_bytes = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Error::Io { operation: "mmap".to_string(), message: format!("{}: {}", path.display(), e), #[cfg(feature = "std")] source: None })?;

        // Wrap in Arc so multiple LazyTensors can keep the mmap alive
        let mmap_arc = Arc::new(file_bytes);

        // Parse safetensors header
        let parsed = SafeTensorsFile::parse(&mmap_arc).map_err(|e| Error::ModelLoad {
            model: path.display().to_string(),
            message: e.to_string(),
            #[cfg(feature = "std")]
            source: None,
        })?;

        let mut result = HashMap::new();

        for (name, info) in &parsed.tensors {
            // Get data slice from bytes
            let data = parsed.tensor_data(name, &mmap_arc).ok_or_else(|| Error::ModelLoad {
                model: path.display().to_string(),
                message: format!("tensor {} not found", name),
                #[cfg(feature = "std")]
                source: None,
            })?;

            // Zero-copy mmap for all dtypes. BF16→F16 conversion is deferred
            // to first buffer() access via OnceLock cache.
            let page_size = 16384usize;
            let (buffer, is_mmap) = if (data.as_ptr() as usize) % page_size == 0 && data.len() >= page_size {
                let buf = self.device.raw().new_buffer_with_bytes_no_copy(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    metal::MTLResourceOptions::StorageModeShared
                        | metal::MTLResourceOptions::CPUCacheModeWriteCombined
                        | metal::MTLResourceOptions::HazardTrackingModeUntracked,
                    None,
                );
                (buf, true)
            } else {
                let buf = self.device.raw().new_buffer_with_data(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                (buf, false)
            };

            let tensor = LazyTensor {
                raw_buffer: buffer,
                shape: info.shape.clone(),
                dtype: info.dtype,
                ggml_type: crate::inference::formats::GgmlType::F16,
                offset: info.data_offset,
                size: info.size,
                _mmap: if is_mmap { Some(Arc::clone(&mmap_arc)) } else { None },
                source: path.to_path_buf(),
                name: name.clone(),
                f16_cache: std::sync::OnceLock::new(),
            };

            // Track stats
            {
                let mut stats = self.stats.write();
                // stats.bytes_mapped += data.len();
                stats.tensors_created += 1;
            }

            result.insert(name.clone(), tensor);
        }

        Ok(result)
    }

    /// Load a safetensors file with zero-copy mmap for all dtypes.
    ///
    /// All tensors (BF16/F32/F16) are memory-mapped directly from NVMe.
    /// BF16→F16 and F32→F16 conversion happens lazily on first `buffer()` access,
    /// avoiding up-front allocation of converted buffers. This enables loading
    /// 93GB+ models without OOM on Apple Silicon UMA.
    pub fn load_safetensors_f16(&self, path: &Path) -> Result<HashMap<String, LazyTensor>> {
        use crate::inference::formats::SafeTensorsFile;

        // Memory-map the file instead of reading it entirely into RAM.
        // On Apple Silicon UMA, mmap'd pages are loaded from NVMe on demand
        // by the kernel — no need to read the entire file into physical RAM.
        let file = std::fs::File::open(path).map_err(|e| Error::Io { operation: "open".to_string(), message: format!("{}: {}", path.display(), e), #[cfg(feature = "std")] source: None })?;
        let file_bytes = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Error::Io { operation: "mmap".to_string(), message: format!("{}: {}", path.display(), e), #[cfg(feature = "std")] source: None })?;

        // Advise random access — model weights are accessed per-layer, not sequentially
        #[cfg(unix)]
        unsafe {
            libc::madvise(
                file_bytes.as_ptr() as *mut _,
                file_bytes.len(),
                libc::MADV_RANDOM,
            );
        }

        // Wrap in Arc so multiple LazyTensors can keep the mmap alive
        let mmap_arc = Arc::new(file_bytes);

        // Parse safetensors header
        let parsed = SafeTensorsFile::parse(&mmap_arc).map_err(|e| Error::ModelLoad {
            model: path.display().to_string(),
            message: e.to_string(),
            #[cfg(feature = "std")]
            source: None,
        })?;

        let page_size = 16384usize; // Apple Silicon 16KB pages
        let mut result = HashMap::new();

        for (name, info) in &parsed.tensors {
            // Get data slice from bytes
            let data = parsed.tensor_data(name, &mmap_arc).ok_or_else(|| Error::ModelLoad {
                model: path.display().to_string(),
                message: format!("tensor {} not found", name),
                #[cfg(feature = "std")]
                source: None,
            })?;

            // Zero-copy mmap for ALL dtypes (BF16/F32/F16/etc).
            // BF16/F32→F16 conversion happens lazily on first buffer() access.
            // This avoids allocating 93GB+ of converted buffers at load time.
            let (buffer, is_mmap) = if (data.as_ptr() as usize) % page_size == 0
                && data.len() >= page_size
            {
                // Zero-copy mmap: data is page-aligned and large enough.
                // Metal buffer points directly into the mmap'd file —
                // pages are loaded from NVMe on demand by the kernel.
                let buffer = self.device.raw().new_buffer_with_bytes_no_copy(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    metal::MTLResourceOptions::StorageModeShared
                        | metal::MTLResourceOptions::CPUCacheModeWriteCombined
                        | metal::MTLResourceOptions::HazardTrackingModeUntracked,
                    None,
                );
                (buffer, true)
            } else {
                // Fallback: copy for unaligned or tiny tensors (keep original dtype)
                let buffer = self.device.raw().new_buffer_with_data(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                (buffer, false)
            };

            let tensor = LazyTensor {
                raw_buffer: buffer,
                shape: info.shape.clone(),
                dtype: info.dtype,
                ggml_type: crate::inference::formats::GgmlType::F16,
                offset: info.data_offset,
                size: info.size,
                _mmap: if is_mmap { Some(Arc::clone(&mmap_arc)) } else { None },
                source: path.to_path_buf(),
                name: name.clone(),
                f16_cache: std::sync::OnceLock::new(),
            };

            // Track stats
            {
                let mut stats = self.stats.write();
                stats.tensors_created += 1;
            }

            result.insert(name.clone(), tensor);
        }

        Ok(result)
    }

    /// Load sharded safetensors (multiple files) with zero-copy mmap.
    ///
    /// Reads the index JSON (e.g. `model.safetensors.index.json`) to find which
    /// shard file each tensor lives in, then loads all shards via mmap.
    /// BF16/F32→F16 conversion is deferred to first `buffer()` access.
    pub fn load_safetensors_sharded_f16(&self, index_path: &Path) -> Result<HashMap<String, LazyTensor>> {
        // Read and parse the index JSON
        let index_bytes = std::fs::read(index_path).map_err(|e| Error::Io {
            operation: "read".to_string(),
            message: format!("{}: {}", index_path.display(), e),
            #[cfg(feature = "std")]
            source: None,
        })?;
        let index_json: serde_json::Value = serde_json::from_slice(&index_bytes).map_err(|e| Error::ModelLoad {
            model: index_path.display().to_string(),
            message: format!("failed to parse index JSON: {}", e),
            #[cfg(feature = "std")]
            source: None,
        })?;

        let weight_map = index_json.get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::ModelLoad {
                model: index_path.display().to_string(),
                message: "missing 'weight_map' in index JSON".to_string(),
                #[cfg(feature = "std")]
                source: None,
            })?;

        // Group tensor names by shard file
        let parent_dir = index_path.parent().unwrap_or(Path::new("."));
        let mut shard_files: HashMap<String, Vec<String>> = HashMap::new();
        for (tensor_name, shard_value) in weight_map {
            if let Some(shard_file) = shard_value.as_str() {
                shard_files.entry(shard_file.to_string())
                    .or_default()
                    .push(tensor_name.clone());
            }
        }

        // Load each shard and merge tensors
        let mut result = HashMap::new();
        for (shard_file, _) in &shard_files {
            let shard_path = parent_dir.join(shard_file);
            let shard_tensors = self.load_safetensors_f16(&shard_path)?;
            result.extend(shard_tensors);
        }

        Ok(result)
    }

    /// Load a raw binary file as a single tensor.
    pub fn load_raw(
        &self,
        path: &Path,
        shape: Shape,
        dtype: DType,
    ) -> Result<LazyTensor> {
        let mmap = self.get_or_create_mmap(path)?;

        let expected_size = shape.numel() * dtype.size_bytes();
        if mmap.len() != expected_size {
            return Err(Error::ModelLoad {
                model: path.display().to_string(),
                message: format!(
                    "size mismatch: expected {} bytes, got {}",
                    expected_size,
                    mmap.len()
                ),
                #[cfg(feature = "std")]
                source: None,
            });
        }

        let buffer = self.device.raw().new_buffer_with_bytes_no_copy(
            mmap.as_ptr() as *const _,
            mmap.len() as u64,
            metal::MTLResourceOptions::StorageModeShared
                | metal::MTLResourceOptions::CPUCacheModeWriteCombined
                | metal::MTLResourceOptions::HazardTrackingModeUntracked,
            None,
        );

        let tensor = LazyTensor {
            raw_buffer: buffer,
            shape,
            dtype,
            ggml_type: crate::inference::formats::GgmlType::F16,
            offset: 0,
            size: mmap.len(),
            _mmap: Some(mmap),
            source: path.to_path_buf(),
            name: path.file_name().unwrap_or_default().to_string_lossy().to_string(),
            f16_cache: std::sync::OnceLock::new(),
        };

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.tensors_created += 1;
        }

        Ok(tensor)
    }

    /// Get loading statistics.
    pub fn stats(&self) -> LazyLoadStats {
        self.stats.read().clone()
    }

    /// Clear mmap cache (releases file handles).
    pub fn clear_cache(&self) {
        let mut mmaps = self.mmaps.write();
        mmaps.clear();
    }

    /// Load a GGUF file with dequantization to F16.
    ///
    /// GGUF files typically use quantized formats (Q4_K, Q8_0, etc.) that
    /// need to be dequantized for GPU compute. This method handles common
    /// quantization formats and converts them to F16.
    pub fn load_gguf_f16(&self, path: &Path) -> Result<(HashMap<String, LazyTensor>, crate::inference::formats::GgufMetadata)> {
        use crate::inference::formats::{GgufFile, GgmlType};

        let mmap = self.get_or_create_mmap(path)?;

        // Parse GGUF header
        let parsed = GgufFile::parse(&mmap).map_err(|e| Error::ModelLoad {
            model: path.display().to_string(),
            message: e.to_string(),
            #[cfg(feature = "std")]
            source: None,
        })?;

        let mut result = HashMap::new();
        // let mut quantized_count = 0usize;

        for (name, info) in &parsed.tensors {
            // Get data slice from mmap
            let data = parsed.tensor_data(name, &mmap).ok_or_else(|| Error::ModelLoad {
                model: path.display().to_string(),
                message: format!("tensor {} not found", name),
                #[cfg(feature = "std")]
                source: None,
            })?;

            // Dequantize if needed.
            //
            // For quant types we have native Metal matmul kernels for, KEEP the
            // original block format zero-copy from mmap. The graph builder picks
            // the right kernel using LazyTensor::ggml_type().
            //
            // Token embeddings (GET_ROWS path) and any tensor with no matching
            // matmul kernel still get dequantized to F16 here.
            let needs_f16_for_kernels = name == "token_embd.weight";
            let kept_native_quant = info.ggml_type.is_quantized()
                && !needs_f16_for_kernels
                && matches!(
                    info.ggml_type,
                    GgmlType::Q5_0
                        | GgmlType::Q8_0
                        | GgmlType::Q4K
                        | GgmlType::Q5K
                        | GgmlType::Q6K
                );

            let (buffer, final_dtype) = if kept_native_quant {
                // Zero-copy from mmap — kernel reads quant blocks directly.
                let buffer = self.device.raw().new_buffer_with_bytes_no_copy(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    metal::MTLResourceOptions::StorageModeShared
                        | metal::MTLResourceOptions::CPUCacheModeWriteCombined
                        | metal::MTLResourceOptions::HazardTrackingModeUntracked,
                    None,
                );
                (buffer, DType::F16)
            } else if info.ggml_type.is_quantized() {
                // Allocate F16 buffer for dequantized data
                let numel = info.shape.numel();
                let f16_size = numel * 2;
                let new_buffer = self.device.raw().new_buffer(
                    f16_size as u64,
                    metal::MTLResourceOptions::StorageModeShared
                        | metal::MTLResourceOptions::CPUCacheModeWriteCombined,
                );

                // Dequantize
                unsafe {
                    let dst = std::slice::from_raw_parts_mut(
                        new_buffer.contents() as *mut u16,
                        numel,
                    );
                    dequantize_to_f16(data, dst, info.ggml_type, numel)?;
                }

                (new_buffer, DType::F16)
            } else if info.ggml_type == GgmlType::BF16 {
                // Convert BF16 to F16
                let new_buffer = self.device.raw().new_buffer(
                    data.len() as u64,
                    metal::MTLResourceOptions::StorageModeShared
                        | metal::MTLResourceOptions::CPUCacheModeWriteCombined,
                );
                unsafe {
                    let dst = std::slice::from_raw_parts_mut(
                        new_buffer.contents() as *mut u8,
                        data.len(),
                    );
                    convert_bf16_to_f16_buffer(data, dst);
                }
                (new_buffer, DType::F16)
            } else if info.ggml_type == GgmlType::F32 {
                let numel = info.shape.numel();
                if data.len() >= numel * 4 {
                    // True F32 data — convert to F16
                    let f16_size = numel * 2;
                    let new_buffer = self.device.raw().new_buffer(
                        f16_size as u64,
                        metal::MTLResourceOptions::StorageModeShared
                            | metal::MTLResourceOptions::CPUCacheModeWriteCombined,
                    );
                    unsafe {
                        let src = std::slice::from_raw_parts(data.as_ptr() as *const f32, numel);
                        let dst = std::slice::from_raw_parts_mut(
                            new_buffer.contents() as *mut u16,
                            numel,
                        );
                        for i in 0..numel {
                            dst[i] = f32_to_f16(src[i]);
                        }
                    }
                    (new_buffer, DType::F16)
                } else if data.len() >= numel * 2 {
                    // Data is actually F16 (GGUF type mismatch) — zero-copy
                    let buffer = self.device.raw().new_buffer_with_bytes_no_copy(
                        data.as_ptr() as *const _,
                        (numel * 2) as u64,
                        metal::MTLResourceOptions::StorageModeShared
                            | metal::MTLResourceOptions::CPUCacheModeWriteCombined
                            | metal::MTLResourceOptions::HazardTrackingModeUntracked,
                        None,
                    );
                    (buffer, DType::F16)
                } else {
                    // Undersized data — zero-copy whatever we have
                    let buffer = self.device.raw().new_buffer_with_bytes_no_copy(
                        data.as_ptr() as *const _,
                        data.len() as u64,
                        metal::MTLResourceOptions::StorageModeShared
                            | metal::MTLResourceOptions::CPUCacheModeWriteCombined
                            | metal::MTLResourceOptions::HazardTrackingModeUntracked,
                        None,
                    );
                    (buffer, DType::F16)
                }
            } else {
                // Zero-copy for F16
                let buffer = self.device.raw().new_buffer_with_bytes_no_copy(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    metal::MTLResourceOptions::StorageModeShared
                        | metal::MTLResourceOptions::CPUCacheModeWriteCombined
                        | metal::MTLResourceOptions::HazardTrackingModeUntracked,
                    None,
                );
                (buffer, info.ggml_type.to_dtype().unwrap_or(DType::F16))
            };

            // ggml_type stored on the tensor: native quant format if kept,
            // otherwise F16 (since the buffer holds dequantized F16 data).
            let stored_ggml_type = if kept_native_quant {
                info.ggml_type
            } else {
                GgmlType::F16
            };

            let tensor = LazyTensor {
                raw_buffer: buffer,
                shape: info.shape.clone(),
                dtype: final_dtype,
                ggml_type: stored_ggml_type,
                offset: info.offset as usize,
                size: data.len(),
                _mmap: Some(Arc::clone(&mmap)),
                source: path.to_path_buf(),
                name: name.clone(),
                f16_cache: std::sync::OnceLock::new(),
            };

            // Track stats
            {
                let mut stats = self.stats.write();
                stats.bytes_mapped += data.len();
                stats.tensors_created += 1;
            }

            result.insert(name.clone(), tensor);
        }

        Ok((result, parsed.metadata))
    }

    /// Load a GGUF file keeping quantized weights in native format.
    ///
    /// Unlike `load_gguf_f16`, this keeps Q4_K/Q6_K weights in their quantized
    /// format for use with fused dequantize-matmul kernels. This dramatically
    /// reduces memory for 7B+ models.
    ///
    /// Returns a map of (tensor_name -> (buffer, is_quantized, ggml_type)).
    pub fn load_gguf_quantized(&self, path: &Path) -> Result<(HashMap<String, QuantizedTensor>, crate::inference::formats::GgufMetadata)> {
        use crate::inference::formats::{GgufFile, GgmlType};

        let mmap = self.get_or_create_mmap(path)?;

        // Parse GGUF header
        let parsed = GgufFile::parse(&mmap).map_err(|e| Error::ModelLoad {
            model: path.display().to_string(),
            message: e.to_string(),
            #[cfg(feature = "std")]
            source: None,
        })?;

        let mut result = HashMap::new();

        for (name, info) in &parsed.tensors {
            // Get data slice from mmap
            let data = parsed.tensor_data(name, &mmap).ok_or_else(|| Error::ModelLoad {
                model: path.display().to_string(),
                message: format!("tensor {} not found", name),
                #[cfg(feature = "std")]
                source: None,
            })?;

            let force_f16 = name == "token_embd.weight";

            // Only keep Q4_K as quantized because we only have a matmul_q4k_f16 kernel.
            // All other quantized formats (Q6_K, Q8_0, Q4_0, etc.) must be dequantized to F16
            // so they can be used with the standard F16 matmul kernel.
            // 
            // Warning: Mistral Q4_K_M uses Q6_K for v.weight and output.weight!
            // If we treat them as Q4_K, we get garbage.
            let keep_quantized = info.ggml_type == GgmlType::Q4K && !force_f16;

            let (buffer, is_quantized, ggml_type) = if keep_quantized {
                // Keep quantized data as-is (zero-copy from mmap)
                let buffer = self.device.raw().new_buffer_with_bytes_no_copy(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    metal::MTLResourceOptions::StorageModeShared
                        | metal::MTLResourceOptions::CPUCacheModeWriteCombined
                        | metal::MTLResourceOptions::HazardTrackingModeUntracked,
                    None,
                );
                (buffer, true, info.ggml_type)
            } else if info.ggml_type.is_quantized() {
                // Dequantize to F16 (since we don't have a GPU kernel for this quantized type)
                let numel = info.shape.numel();
                let f16_size = numel * 2;
                let new_buffer = self.device.raw().new_buffer(
                    f16_size as u64,
                    metal::MTLResourceOptions::StorageModeShared
                        | metal::MTLResourceOptions::CPUCacheModeWriteCombined,
                );

                unsafe {
                    let dst = std::slice::from_raw_parts_mut(
                        new_buffer.contents() as *mut u16,
                        numel,
                    );
                    dequantize_to_f16(data, dst, info.ggml_type, numel)?;
                }

                // It is now an F16 tensor
                (new_buffer, false, GgmlType::F16)
            } else if info.ggml_type == GgmlType::BF16 {
                // Convert BF16 to F16 (same size)
                let new_buffer = self.device.raw().new_buffer(
                    data.len() as u64,
                    metal::MTLResourceOptions::StorageModeShared
                        | metal::MTLResourceOptions::CPUCacheModeWriteCombined,
                );
                unsafe {
                    let dst = std::slice::from_raw_parts_mut(
                        new_buffer.contents() as *mut u8,
                        data.len(),
                    );
                    convert_bf16_to_f16_buffer(data, dst);
                }
                (new_buffer, false, GgmlType::F16)
            } else if info.ggml_type == GgmlType::F32 {
                // Convert F32 to F16
                let numel = info.shape.numel();
                let f16_size = numel * 2;
                let new_buffer = self.device.raw().new_buffer(
                    f16_size as u64,
                    metal::MTLResourceOptions::StorageModeShared
                        | metal::MTLResourceOptions::CPUCacheModeWriteCombined,
                );
                unsafe {
                    let src = std::slice::from_raw_parts(data.as_ptr() as *const f32, numel);
                    let dst = std::slice::from_raw_parts_mut(
                        new_buffer.contents() as *mut u16,
                        numel,
                    );
                    for i in 0..numel {
                        dst[i] = f32_to_f16(src[i]);
                    }
                }
                (new_buffer, false, GgmlType::F16)
            } else {
                // F16 - zero copy
                let buffer = self.device.raw().new_buffer_with_bytes_no_copy(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    metal::MTLResourceOptions::StorageModeShared
                        | metal::MTLResourceOptions::CPUCacheModeWriteCombined
                        | metal::MTLResourceOptions::HazardTrackingModeUntracked,
                    None,
                );
                (buffer, false, GgmlType::F16)
            };

            let tensor = QuantizedTensor {
                buffer,
                shape: info.shape.clone(),
                ggml_type,
                is_quantized,
                _mmap: Arc::clone(&mmap),
            };

            result.insert(name.clone(), tensor);
        }

        Ok((result, parsed.metadata))
    }
}

/// Tensor that may be in quantized format.
pub struct QuantizedTensor {
    /// The Metal buffer containing tensor data.
    pub buffer: metal::Buffer,
    /// The tensor shape.
    pub shape: crate::core::Shape,
    /// The GGML quantization type.
    pub ggml_type: crate::inference::formats::GgmlType,
    /// Whether the tensor uses quantized storage.
    pub is_quantized: bool,
    _mmap: Arc<memmap2::Mmap>,
}

impl QuantizedTensor {
    /// Get a reference to the Metal buffer.
    pub fn buffer(&self) -> &metal::Buffer {
        &self.buffer
    }
}

/// Dequantize data to F16.
fn dequantize_to_f16(data: &[u8], dst: &mut [u16], ggml_type: crate::inference::formats::GgmlType, numel: usize) -> Result<()> {
    use crate::inference::formats::GgmlType;

    match ggml_type {
        GgmlType::Q4_0 => dequantize_q4_0(data, dst, numel),
        GgmlType::Q8_0 => dequantize_q8_0(data, dst, numel),
        GgmlType::Q5_0 => dequantize_q5_0(data, dst, numel),
        GgmlType::Q4K => dequantize_q4_k(data, dst, numel),
        GgmlType::Q5K => dequantize_q5_k(data, dst, numel),
        GgmlType::Q6K => dequantize_q6_k(data, dst, numel),
        _ => Err(Error::ModelLoad {
            model: "gguf".into(),
            message: format!("unsupported quantization type: {:?}", ggml_type),
            #[cfg(feature = "std")]
            source: None,
        }),
    }
}

/// Dequantize Q4_0: 32 values per block, 2 bytes scale + 16 bytes data.
fn dequantize_q4_0(data: &[u8], dst: &mut [u16], numel: usize) -> Result<()> {
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 18; // 2 (scale) + 16 (data)

    if numel % BLOCK_SIZE != 0 {
        return Err(Error::ModelLoad {
            model: "gguf".into(),
            message: format!(
                "Q4_0 dequantization requires numel ({}) to be a multiple of block size ({})",
                numel, BLOCK_SIZE
            ),
            #[cfg(feature = "std")]
            source: None,
        });
    }

    let num_blocks = numel / BLOCK_SIZE;

    // Reference: ggml/src/ggml-quants.c dequantize_row_q4_0
    // Split-half layout: y[j] = low nibble, y[j+16] = high nibble. NOT interleaved!
    for block_idx in 0..num_blocks {
        let block_data = &data[block_idx * BYTES_PER_BLOCK..];
        let scale = f16_to_f32(u16::from_le_bytes([block_data[0], block_data[1]]));

        // 16 bytes = 32 x 4-bit values in split layout
        for j in 0..BLOCK_SIZE/2 { // j = 0..15
            let byte = block_data[2 + j];
            let x0 = (byte & 0x0f) as i32 - 8;
            let x1 = ((byte >> 4) & 0x0f) as i32 - 8;
            let idx_lo = block_idx * BLOCK_SIZE + j;
            let idx_hi = idx_lo + BLOCK_SIZE/2;
            if idx_lo < numel {
                dst[idx_lo] = f32_to_f16(x0 as f32 * scale);
            }
            if idx_hi < numel {
                dst[idx_hi] = f32_to_f16(x1 as f32 * scale);
            }
        }
    }

    Ok(())
}

/// Dequantize Q8_0: 32 values per block, 2 bytes scale + 32 bytes data.
fn dequantize_q8_0(data: &[u8], dst: &mut [u16], numel: usize) -> Result<()> {
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 34; // 2 (scale) + 32 (data)

    let num_blocks = numel / BLOCK_SIZE;

    for block_idx in 0..num_blocks {
        let block_data = &data[block_idx * BYTES_PER_BLOCK..];

        // Scale is F16
        let scale_bits = u16::from_le_bytes([block_data[0], block_data[1]]);
        let scale = f16_to_f32(scale_bits);

        // 32 bytes = 32 x 8-bit signed values
        for i in 0..32 {
            let val = block_data[2 + i] as i8 as i32;
            let idx = block_idx * BLOCK_SIZE + i;
            if idx < numel {
                dst[idx] = f32_to_f16(val as f32 * scale);
            }
        }
    }

    Ok(())
}

/// Dequantize Q5_0: 32 values per block. 2B scale + 4B high bits + 16B low nibbles = 22B.
/// Dequantize Q5_0: 32 values per block. 2B scale + 4B high bits + 16B packed nibbles = 22B.
/// Each byte in qs holds TWO 4-bit values: lo nibble (j%2==0) and hi nibble (j%2==1).
fn dequantize_q5_0(data: &[u8], dst: &mut [u16], numel: usize) -> Result<()> {
    const BS: usize = 32;
    const BPB: usize = 22;
    let nb = numel / BS;
    // Reference: ggml/src/ggml-quants.c dequantize_row_q5_0
    // Layout: y[j] = low nibble + high bit from qh[j], y[j+16] = high nibble + high bit from qh[j+12]
    // NOT interleaved! The 32 output values are split: first 16 from low nibbles, next 16 from high nibbles.
    for b in 0..nb {
        let blk = &data[b * BPB..];
        let scale = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = &blk[6..22]; // 16 bytes = 32 packed 4-bit values
        for j in 0..BS/2 { // j = 0..15
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = ((qs[j] & 0x0F) as i32 | xh_0 as i32) - 16;
            let x1 = ((qs[j] >> 4) as i32 | xh_1 as i32) - 16;
            dst[b * BS + j] = half::f16::from_f32(x0 as f32 * scale).to_bits();
            dst[b * BS + j + BS/2] = half::f16::from_f32(x1 as f32 * scale).to_bits();
        }
    }
    Ok(())
}

/// Dequantize Q5_K: 256 values per block. 176 bytes per block.
fn dequantize_q5_k(data: &[u8], dst: &mut [u16], numel: usize) -> Result<()> {
    const BS: usize = 256;
    const BPB: usize = 176;
    let nb = numel / BS;
    for bi in 0..nb {
        let blk = &data[bi * BPB..];
        let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        let dmin = half::f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])).to_f32();
        let sr = &blk[4..16];
        let qh = &blk[16..48];
        let qs = &blk[48..176];
        let mut sc = [0u8; 8];
        let mut m = [0u8; 8];
        for i in 0..4 { sc[i] = sr[i] & 0x3F; m[i] = sr[i + 4] & 0x3F; }
        for i in 0..2 {
            sc[4+i] = (sr[8+i] & 0x0F) | ((sr[i] >> 6) << 4);
            sc[6+i] = (sr[8+i] >> 4) | ((sr[i+2] >> 6) << 4);
            m[4+i] = (sr[10+i] & 0x0F) | ((sr[i+4] >> 6) << 4);
            m[6+i] = (sr[10+i] >> 4) | ((sr[i+6] >> 6) << 4);
        }
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for g in 0..4 {
            let si = g * 2;
            let (d1, m1) = (d * sc[si] as f32, dmin * m[si] as f32);
            let (d2, m2) = (d * sc[si+1] as f32, dmin * m[si+1] as f32);
            for j in 0..32 {
                let qi = g * 32 + j;
                let lo = qs[qi] & 0x0F;
                let hb = if (qh[j] & u1) != 0 { 16u8 } else { 0 };
                dst[bi*BS + g*64 + j] = half::f16::from_f32((lo+hb) as f32 * d1 - m1).to_bits();
            }
            for j in 0..32 {
                let qi = g * 32 + j;
                let hi = (qs[qi] >> 4) & 0x0F;
                let hb = if (qh[j] & u2) != 0 { 16u8 } else { 0 };
                dst[bi*BS + g*64 + 32 + j] = half::f16::from_f32((hi+hb) as f32 * d2 - m2).to_bits();
            }
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(())
}

/// Dequantize Q4_K: 256 values per block.
/// Block structure: d (f16) + dmin (f16) + scales (12 bytes) + qs (128 bytes)
fn dequantize_q4_k(data: &[u8], dst: &mut [u16], numel: usize) -> Result<()> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 144; // 2 + 2 + 12 + 128

    let num_blocks = numel / BLOCK_SIZE;
    let actual_blocks = std::cmp::min(num_blocks, data.len() / BYTES_PER_BLOCK);

    for block_idx in 0..actual_blocks {
        let block_data = &data[block_idx * BYTES_PER_BLOCK..];

        // d and dmin are f16
        let d = f16_to_f32(u16::from_le_bytes([block_data[0], block_data[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block_data[2], block_data[3]]));

        let scales = &block_data[4..16];
        let qs = &block_data[16..144];

        // Process 4 groups of 64 values each
        let mut out_idx = block_idx * BLOCK_SIZE;
        let mut q_offset = 0;
        let mut is = 0;

        for _ in 0..4 {
            // Extract scale/min for low nibbles (is+0)
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;

            // Extract scale/min for high nibbles (is+1)
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;

            // First 32 outputs: low nibbles
            for l in 0..32 {
                let q_val = (qs[q_offset + l] & 0x0F) as f32;
                if out_idx + l < numel {
                    dst[out_idx + l] = f32_to_f16(d1 * q_val - min1);
                }
            }

            // Next 32 outputs: high nibbles
            for l in 0..32 {
                let q_val = ((qs[q_offset + l] >> 4) & 0x0F) as f32;
                if out_idx + 32 + l < numel {
                    dst[out_idx + 32 + l] = f32_to_f16(d2 * q_val - min2);
                }
            }

            q_offset += 32;
            out_idx += 64;
            is += 2;
        }
    }

    Ok(())
}

/// Extract 6-bit scale and min values for Q4_K.
/// Matches llama.cpp's get_scale_min_k4 exactly.
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        let sc = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (sc, m)
    } else {
        // j = 4,5,6,7
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// Dequantize Q6_K: 256 values per block.
fn dequantize_q6_k(data: &[u8], dst: &mut [u16], numel: usize) -> Result<()> {
    // Q6_K block structure (from ggml-quants.h):
    // - ql: QK_K/2 = 128 bytes (lower 4 bits of each quant)
    // - qh: QK_K/4 = 64 bytes (upper 2 bits of each quant)
    // - scales: QK_K/16 = 16 bytes (8-bit signed scales)
    // - d: 2 bytes (f16 super-block scale)
    // Total: 128 + 64 + 16 + 2 = 210 bytes per block of 256 values
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 210;

    let num_blocks = numel / BLOCK_SIZE;

    for block_idx in 0..num_blocks {
        let block_data = &data[block_idx * BYTES_PER_BLOCK..];

        // Parse Q6_K block structure per ggml memory layout
        let ql = &block_data[0..128];      // lower 4 bits (128 bytes)
        let qh = &block_data[128..192];    // upper 2 bits (64 bytes)
        let scales = &block_data[192..208]; // 8-bit signed scales (16 bytes)
        let d = f16_to_f32(u16::from_le_bytes([block_data[208], block_data[209]]));

        // Dequantize exactly matching llama.cpp dequantize_row_q6_K
        // Process in two halves (n=0: first 128, n=128: second 128)
        let base = block_idx * BLOCK_SIZE;
        for n in [0usize, 128] {
            let ql_off = if n == 0 { 0 } else { 64 };
            let qh_off = if n == 0 { 0 } else { 32 };
            let sc_off = if n == 0 { 0 } else { 8 };

            for l in 0..32 {
                let is = l / 16;  // 0 or 1
                let sc0 = (scales[sc_off + is + 0] as i8) as f32;
                let sc1 = (scales[sc_off + is + 2] as i8) as f32;
                let sc2 = (scales[sc_off + is + 4] as i8) as f32;
                let sc3 = (scales[sc_off + is + 6] as i8) as f32;

                // Extract 4 quantized values from interleaved positions
                let q1 = ((ql[ql_off + l] & 0x0f) | ((qh[qh_off + l] & 0x03) << 4)) as i32 - 32;
                let q2 = ((ql[ql_off + l + 32] & 0x0f) | (((qh[qh_off + l] >> 2) & 0x03) << 4)) as i32 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 0x03) << 4)) as i32 - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 0x03) << 4)) as i32 - 32;

                dst[base + n + l + 0] = f32_to_f16(d * sc0 * q1 as f32);
                dst[base + n + l + 32] = f32_to_f16(d * sc1 * q2 as f32);
                dst[base + n + l + 64] = f32_to_f16(d * sc2 * q3 as f32);
                dst[base + n + l + 96] = f32_to_f16(d * sc3 * q4 as f32);
            }
        }
    }

    Ok(())
}

/// Convert F32 to F16 bits.
#[inline]
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;

    if exp == 0xff {
        // Inf or NaN
        if mant == 0 {
            return ((sign << 15) | 0x7c00) as u16;
        } else {
            return ((sign << 15) | 0x7e00 | (mant >> 13)) as u16;
        }
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | 0x7c00) as u16;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let m = (mant | 0x800000) >> (1 - new_exp);
        return ((sign << 15) | (m >> 13)) as u16;
    }

    ((sign << 15) | ((new_exp as u32) << 10) | (mant >> 13)) as u16
}

/// Convert F16 bits to F32.
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut e = -14i32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            f32::from_bits((sign << 31) | (((127 + e) as u32) << 23) | (m << 13))
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits((sign << 31) | 0x7f800000)
        } else {
            f32::from_bits((sign << 31) | 0x7fc00000 | (mant << 13))
        }
    } else {
        let new_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (new_exp << 23) | (mant << 13))
    }
}

#[cfg(feature = "safetensors")]
fn safetensors_dtype_to_dtype(dtype: safetensors::Dtype) -> DType {
    match dtype {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::I32 => DType::I32,
        safetensors::Dtype::I64 => DType::I64,
        safetensors::Dtype::I8 => DType::I8,
        safetensors::Dtype::U8 => DType::U8,
        safetensors::Dtype::BOOL => DType::Bool,
        _ => DType::F32, // Fallback
    }
}

/// Model weights container with lazy loading.
#[cfg(feature = "metal")]
pub struct LazyModel {
    /// Tensor map
    tensors: HashMap<String, LazyTensor>,
    /// Source path
    source: PathBuf,
    /// Loader reference
    loader: Arc<LazyLoader>,
}

#[cfg(feature = "metal")]
impl LazyModel {
    /// Load a model lazily.
    #[cfg(feature = "safetensors")]
    pub fn load(loader: Arc<LazyLoader>, path: &Path) -> Result<Self> {
        let tensors = loader.load_safetensors(path)?;

        Ok(Self {
            tensors,
            source: path.to_path_buf(),
            loader,
        })
    }

    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> Option<&LazyTensor> {
        self.tensors.get(name)
    }

    /// Iterate over all tensors.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &LazyTensor)> {
        self.tensors.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Get total size in bytes.
    pub fn total_size(&self) -> usize {
        self.tensors.values().map(|t| t.size()).sum()
    }

    /// Prefetch all tensors (load into RAM).
    pub fn prefetch_all(&self) {
        for tensor in self.tensors.values() {
            tensor.prefetch();
        }
    }

    /// Prefetch specific tensors by prefix.
    pub fn prefetch_prefix(&self, prefix: &str) {
        for (name, tensor) in &self.tensors {
            if name.starts_with(prefix) {
                tensor.prefetch();
            }
        }
    }

    /// Asynchronously prefetch tensors matching a name prefix via MADV_WILLNEED.
    ///
    /// Call for layer N+1 while layer N runs on the GPU to overlap I/O with compute.
    pub fn advise_willneed_prefix(&self, prefix: &str) {
        for (name, tensor) in &self.tensors {
            if name.starts_with(prefix) {
                tensor.advise_willneed();
            }
        }
    }

    /// Mark tensors as not needed (hint to OS).
    pub fn evict_prefix(&self, prefix: &str) {
        for (name, tensor) in &self.tensors {
            if name.starts_with(prefix) {
                tensor.advise_dontneed();
            }
        }
    }
}

// Stubs for non-macOS
#[cfg(not(feature = "metal"))]
pub struct LazyTensor;

#[cfg(not(feature = "metal"))]
pub struct LazyLoader;

#[cfg(not(feature = "metal"))]
impl LazyLoader {
    pub fn new(_device: Arc<super::MetalDevice>) -> Self {
        Self
    }
}

#[cfg(not(feature = "metal"))]
pub struct LazyModel;
