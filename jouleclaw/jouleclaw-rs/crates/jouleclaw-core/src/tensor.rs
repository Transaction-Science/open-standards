//! Tensor types and metadata.
//!
//! A `Tensor` is a typed multidimensional array with an explicit lifetime tier.
//! The tier indicates how the runtime should place the tensor in memory.

use std::sync::Arc;

/// Element type of a tensor.
///
/// Phase 0 declares the full set; Phase 1 implements F32 and F16 only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl Dtype {
    /// Bytes per element.
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::F8E4M3 | Self::F8E5M2 | Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }
}

/// Shape of a tensor. Each entry is the size of one dimension.
pub type Shape = Vec<usize>;

/// Lifetime tier of a tensor. Determines memory placement.
///
/// See spec 00 and the upcoming memory layer spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LifetimeTier {
    /// Cold: weights. Persistent, mostly read-only, long-lived.
    /// Place in highest-capacity tier (DRAM, SSD-mapped).
    Cold,

    /// Warm: KV cache, intermediate state. Per-request lifetime, recoverable.
    /// Place in mid-tier (SLC, DRAM with locality hints).
    Warm,

    /// Hot: activations. Ephemeral, ideally register- or L1-resident.
    /// Place in fastest tier available.
    Hot,

    /// Persistent: history, indexes. Survives across requests.
    /// May be backed by SSD or network storage with caching.
    Persistent,
}

/// Metadata describing a tensor's type and shape, without owning data.
///
/// Used in graph definitions, type-checking, and joule estimation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorMeta {
    pub dtype: Dtype,
    pub shape: Shape,
    pub tier: LifetimeTier,
}

impl TensorMeta {
    pub fn new(dtype: Dtype, shape: &[usize]) -> Self {
        Self {
            dtype,
            shape: shape.to_vec(),
            tier: LifetimeTier::Hot,
        }
    }

    pub fn with_tier(mut self, tier: LifetimeTier) -> Self {
        self.tier = tier;
        self
    }

    /// Total element count.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total bytes.
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }
}

/// An owned reference to a tensor's storage.
///
/// In Phase 1 this wraps a `Vec<u8>`; in Phase 2+ it will be a memory-tier-aware
/// allocation tracked by the memory subsystem.
#[derive(Debug, Clone)]
pub struct TensorRef {
    pub meta: TensorMeta,
    pub storage: Arc<TensorStorage>,
}

/// External read-only byte source — typically an mmap of a GGUF/safetensors
/// file. The runtime hands the kernel a `&[u8]` into the mapped pages
/// without copying.
///
/// `Send + Sync` so an `Arc<dyn ByteBacking>` can travel between graph
/// build and kernel execute. Implementations must guarantee the returned
/// slice stays valid for the lifetime of `Self` (typical for mmap'd
/// files held by Arc).
pub trait ByteBacking: Send + Sync + std::fmt::Debug {
    fn bytes(&self) -> &[u8];
}

/// Zero-copy mmap-backed tensor data: a sub-range of an external byte
/// source. The backing `Arc` keeps the mmap alive for as long as any
/// tensor references it.
#[derive(Debug, Clone)]
pub struct MappedBacking {
    pub backing: Arc<dyn ByteBacking>,
    pub offset: usize,
    pub len: usize,
}

/// Backing storage for a tensor. Either owned (default) or a zero-copy
/// reference into an external mmap. Outputs/activations are always
/// owned; constants (model weights loaded from a GGUF file) can be
/// mapped, saving a copy of ~hundreds of MB per model.
#[derive(Debug)]
pub struct TensorStorage {
    /// Owned bytes. Empty when [`Self::mapped`] is `Some` (the kernel
    /// reads from `mapped.backing` instead). Outputs and intermediate
    /// activations always use this path.
    pub bytes: Vec<u8>,
    /// Optional zero-copy backing. When set, `view_bytes()` returns
    /// `backing.bytes()[offset..offset+len]` instead of `&bytes`.
    /// Set by GGUF loader's `embed_weight` packed paths.
    pub mapped: Option<MappedBacking>,
}

impl TensorStorage {
    /// Owned-bytes constructor (the historical path). All call sites
    /// that did `TensorStorage { bytes, mapped: None }` keep working.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes, mapped: None }
    }

    /// Mmap-backed constructor — zero-copy. The returned storage owns
    /// no bytes; reads go through `backing[offset..offset+len]`.
    pub fn from_mapped(backing: Arc<dyn ByteBacking>, offset: usize, len: usize) -> Self {
        Self {
            bytes: Vec::new(),
            mapped: Some(MappedBacking { backing, offset, len }),
        }
    }

    /// Read-only byte view — picks the mmap path if available, else the
    /// owned Vec. Callers that previously did `&storage.bytes` for
    /// possibly-mapped inputs must switch to this accessor; owned-only
    /// paths (outputs, KV cache) can keep using `&bytes` directly.
    pub fn view_bytes(&self) -> &[u8] {
        match &self.mapped {
            Some(m) => &m.backing.bytes()[m.offset..m.offset + m.len],
            None => &self.bytes,
        }
    }
}

/// An immutable view of a tensor (or slice thereof).
///
/// Lifetime-bound to its owning storage. Used by kernels at execution time.
pub struct TensorView<'a> {
    pub meta: &'a TensorMeta,
    pub bytes: &'a [u8],
}

/// A mutable view of a tensor.
///
/// Lifetime-bound to its owning storage. Output buffers are passed as `TensorViewMut`.
pub struct TensorViewMut<'a> {
    pub meta: &'a TensorMeta,
    pub bytes: &'a mut [u8],
}

/// A tensor: metadata plus storage.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub meta: TensorMeta,
    pub storage: Arc<TensorStorage>,
}

impl Tensor {
    /// Allocate a zeroed tensor with the given metadata.
    pub fn zeros(meta: TensorMeta) -> Self {
        let bytes = vec![0u8; meta.size_bytes()];
        Self { meta, storage: Arc::new(TensorStorage::from_bytes(bytes)) }
    }

    /// Construct from an f32 slice. Panics if dtype is not F32 or size mismatches.
    pub fn from_f32(meta: TensorMeta, data: &[f32]) -> Self {
        assert_eq!(meta.dtype, Dtype::F32, "from_f32 requires F32 dtype");
        assert_eq!(data.len(), meta.numel(), "data length mismatch");
        let mut bytes = vec![0u8; meta.size_bytes()];
        for (i, &v) in data.iter().enumerate() {
            let b = v.to_le_bytes();
            bytes[i * 4..i * 4 + 4].copy_from_slice(&b);
        }
        Self { meta, storage: Arc::new(TensorStorage::from_bytes(bytes)) }
    }

    /// Zero-copy mmap-backed constructor. The tensor owns no bytes;
    /// reads go through `backing[offset..offset+len]` and stay valid
    /// for as long as the `Arc<dyn ByteBacking>` is alive.
    pub fn from_mapped(
        meta: TensorMeta,
        backing: Arc<dyn ByteBacking>,
        offset: usize,
        len: usize,
    ) -> Self {
        assert_eq!(meta.size_bytes(), len,
            "mapped tensor: meta size_bytes {} != mapped len {}",
            meta.size_bytes(), len);
        Self {
            meta,
            storage: Arc::new(TensorStorage::from_mapped(backing, offset, len)),
        }
    }

    pub fn view(&self) -> TensorView<'_> {
        TensorView { meta: &self.meta, bytes: self.storage.view_bytes() }
    }

    /// Read as an f32 vector. Panics if dtype is not F32.
    /// Phase 1.1 uses byte copies; a future allocator will give us alignment
    /// guarantees that allow zero-copy `&[f32]` views.
    pub fn as_f32_vec(&self) -> Vec<f32> {
        assert_eq!(self.meta.dtype, Dtype::F32, "as_f32_vec requires F32 dtype");
        let n = self.meta.numel();
        let src = self.storage.view_bytes();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut b = [0u8; 4];
            b.copy_from_slice(&src[i * 4..i * 4 + 4]);
            out.push(f32::from_le_bytes(b));
        }
        out
    }
}

impl<'a> TensorView<'a> {
    pub fn as_f32_vec(&self) -> Vec<f32> {
        assert_eq!(self.meta.dtype, Dtype::F32);
        let n = self.meta.numel();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut b = [0u8; 4];
            b.copy_from_slice(&self.bytes[i * 4..i * 4 + 4]);
            out.push(f32::from_le_bytes(b));
        }
        out
    }
}

impl<'a> TensorViewMut<'a> {
    pub fn write_f32(&mut self, data: &[f32]) {
        assert_eq!(self.meta.dtype, Dtype::F32);
        assert_eq!(data.len(), self.meta.numel());
        for (i, &v) in data.iter().enumerate() {
            let b = v.to_le_bytes();
            self.bytes[i * 4..i * 4 + 4].copy_from_slice(&b);
        }
    }
}
