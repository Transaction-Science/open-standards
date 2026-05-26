//! Core types and error handling for efficient-genai.

mod error;
mod config;

pub use error::{Error, Result};
pub use config::{Config, MemoryConfig, ExecutionConfig};


/// Unique identifier for tensors, operations, and resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(u64);

impl Id {
    /// Create a new unique ID.
    #[inline]
    pub fn new() -> Self {
        use core::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value.
    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Create an ID from a raw value.
    ///
    /// This is useful for recreating IDs from stored values.
    #[inline]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
}

impl Default for Id {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Display for Id {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Data types supported by the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    /// 32-bit floating point
    F32 = 0,
    /// 16-bit floating point (IEEE 754)
    F16 = 1,
    /// Brain floating point (16-bit)
    BF16 = 2,
    /// 8-bit floating point (E4M3)
    F8E4M3 = 3,
    /// 8-bit floating point (E5M2)
    F8E5M2 = 4,
    /// 32-bit signed integer
    I32 = 5,
    /// 64-bit signed integer
    I64 = 6,
    /// 8-bit signed integer
    I8 = 7,
    /// 8-bit unsigned integer
    U8 = 8,
    /// 32-bit unsigned integer
    U32 = 9,
    /// Boolean
    Bool = 10,
}

impl DType {
    /// Size of this data type in bytes.
    #[inline]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F8E4M3 | Self::F8E5M2 | Self::I8 | Self::U8 | Self::Bool => 1,
            Self::I64 => 8,
        }
    }

    /// Whether this is a floating-point type.
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::F32 | Self::F16 | Self::BF16 | Self::F8E4M3 | Self::F8E5M2
        )
    }

    /// Whether this is an integer type.
    #[inline]
    pub const fn is_integer(self) -> bool {
        matches!(self, Self::I32 | Self::I64 | Self::I8 | Self::U8 | Self::U32)
    }
}

/// Shape of a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: alloc::vec::Vec<usize>,
}

impl Shape {
    /// Create a new shape from dimensions.
    #[inline]
    pub fn new(dims: impl Into<alloc::vec::Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    /// Create a scalar shape (0 dimensions).
    #[inline]
    pub fn scalar() -> Self {
        Self { dims: alloc::vec::Vec::new() }
    }

    /// Number of dimensions (rank).
    #[inline]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Get dimensions as a slice.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Total number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Get a specific dimension.
    #[inline]
    pub fn dim(&self, idx: usize) -> Option<usize> {
        self.dims.get(idx).copied()
    }

    /// Compute strides for contiguous layout (row-major).
    pub fn strides(&self) -> alloc::vec::Vec<usize> {
        let mut strides = alloc::vec![1; self.rank()];
        for i in (0..self.rank().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Get dimensions as (d0, d1, d2, d3) tuple for 4D shapes.
    #[inline]
    pub fn dims4(&self) -> Option<(usize, usize, usize, usize)> {
        if self.dims.len() >= 4 {
            Some((self.dims[0], self.dims[1], self.dims[2], self.dims[3]))
        } else {
            None
        }
    }

    /// Get dimensions as (d0, d1, d2) tuple for 3D shapes.
    #[inline]
    pub fn dims3(&self) -> Option<(usize, usize, usize)> {
        if self.dims.len() >= 3 {
            Some((self.dims[0], self.dims[1], self.dims[2]))
        } else {
            None
        }
    }

    /// Get dimensions as (d0, d1) tuple for 2D shapes.
    #[inline]
    pub fn dims2(&self) -> Option<(usize, usize)> {
        if self.dims.len() >= 2 {
            Some((self.dims[0], self.dims[1]))
        } else {
            None
        }
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Self::new(dims.to_vec())
    }
}

/// Memory layout of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Row-major (C-style) contiguous
    RowMajor,
    /// Column-major (Fortran-style) contiguous
    ColMajor,
    /// Custom strides (possibly non-contiguous)
    Strided,
}

/// Modality types supported by the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    /// Text/language models
    Text,
    /// Image generation/processing
    Image,
    /// Video generation/processing
    Video,
    /// Audio generation/processing
    Audio,
    /// 3D generation (Gaussian splats, meshes)
    ThreeD,
}

impl core::fmt::Display for Modality {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Image => write!(f, "image"),
            Self::Video => write!(f, "video"),
            Self::Audio => write!(f, "audio"),
            Self::ThreeD => write!(f, "3d"),
        }
    }
}
