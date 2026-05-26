//! Device memory management.

use crate::core::{Error, Result};
use alloc::vec::Vec;
use core::ptr::NonNull;

/// Raw pointer to device memory.
#[derive(Debug, Clone, Copy)]
pub struct DevicePtr {
    ptr: u64,
}

impl DevicePtr {
    /// Create from raw pointer value.
    pub const fn new(ptr: u64) -> Self {
        Self { ptr }
    }

    /// Null pointer.
    pub const fn null() -> Self {
        Self { ptr: 0 }
    }

    /// Check if null.
    pub const fn is_null(self) -> bool {
        self.ptr == 0
    }

    /// Get raw pointer value.
    pub const fn raw(self) -> u64 {
        self.ptr
    }

    /// Offset by bytes.
    pub const fn offset(self, bytes: usize) -> Self {
        Self {
            ptr: self.ptr + bytes as u64,
        }
    }
}

/// Buffer allocated on a device.
#[derive(Debug)]
pub struct DeviceBuffer {
    /// Pointer to device memory
    ptr: DevicePtr,
    /// Size in bytes
    size: usize,
    /// Device that owns this buffer
    device_id: super::DeviceId,
    /// Whether this buffer owns the memory
    owned: bool,
}

impl DeviceBuffer {
    /// Create a new device buffer (internal use).
    pub(crate) fn new(ptr: DevicePtr, size: usize, device_id: super::DeviceId) -> Self {
        Self {
            ptr,
            size,
            device_id,
            owned: true,
        }
    }

    /// Get the device pointer.
    pub fn ptr(&self) -> DevicePtr {
        self.ptr
    }

    /// Get buffer size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the device ID.
    pub fn device_id(&self) -> super::DeviceId {
        self.device_id
    }

    /// Create a view into this buffer.
    pub fn slice(&self, offset: usize, size: usize) -> Result<DeviceBuffer> {
        if offset + size > self.size {
            return Err(Error::InvalidArgument {
                name: "slice".into(),
                message: format!(
                    "slice [{}, {}) out of bounds for buffer of size {}",
                    offset,
                    offset + size,
                    self.size
                ),
            });
        }

        Ok(DeviceBuffer {
            ptr: self.ptr.offset(offset),
            size,
            device_id: self.device_id,
            owned: false, // Views don't own memory
        })
    }
}

/// Memory pool for efficient allocation.
#[derive(Debug)]
pub struct MemoryPool {
    /// Device this pool manages
    device_id: super::DeviceId,
    /// Page size for allocations
    page_size: usize,
    /// Free pages
    free_pages: Vec<DeviceBuffer>,
    /// Total allocated bytes
    total_allocated: usize,
    /// Peak usage
    peak_usage: usize,
}

impl MemoryPool {
    /// Create a new memory pool.
    pub fn new(device_id: super::DeviceId, page_size: usize) -> Self {
        Self {
            device_id,
            page_size,
            free_pages: Vec::new(),
            total_allocated: 0,
            peak_usage: 0,
        }
    }

    /// Get page size.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Get total allocated memory.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Get peak memory usage.
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Round size up to page boundary.
    fn round_to_page(&self, size: usize) -> usize {
        (size + self.page_size - 1) / self.page_size * self.page_size
    }

    /// Return a buffer to the pool.
    pub fn return_buffer(&mut self, buffer: DeviceBuffer) {
        if buffer.owned && buffer.size >= self.page_size {
            self.free_pages.push(buffer);
        }
        // Small buffers or views are dropped
    }

    /// Clear all cached pages.
    pub fn clear(&mut self) {
        self.free_pages.clear();
    }
}

/// Pinned host memory for fast transfers.
#[derive(Debug)]
pub struct PinnedBuffer {
    ptr: NonNull<u8>,
    size: usize,
}

impl PinnedBuffer {
    /// Allocate pinned memory.
    #[cfg(feature = "std")]
    pub fn allocate(size: usize) -> Result<Self> {
        // For now, use regular allocation
        // TODO: Use CUDA pinned memory or OS-specific mechanisms
        let layout = std::alloc::Layout::from_size_align(size, 64)
            .map_err(|_| Error::internal("invalid layout"))?;

        let ptr = unsafe { std::alloc::alloc(layout) };

        NonNull::new(ptr)
            .map(|ptr| Self { ptr, size })
            .ok_or_else(|| Error::out_of_memory(size, 0))
    }

    /// Get as slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Get as mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

#[cfg(feature = "std")]
impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.size, 64).unwrap();
        unsafe {
            std::alloc::dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

/// Staging buffer for host-device transfers.
#[derive(Debug)]
pub struct StagingBuffer {
    /// Host-side pinned memory
    host: PinnedBuffer,
    /// Device-side buffer
    device: DeviceBuffer,
}

impl StagingBuffer {
    /// Create a new staging buffer.
    pub fn new(host: PinnedBuffer, device: DeviceBuffer) -> Self {
        Self { host, device }
    }

    /// Get host buffer.
    pub fn host(&self) -> &PinnedBuffer {
        &self.host
    }

    /// Get mutable host buffer.
    pub fn host_mut(&mut self) -> &mut PinnedBuffer {
        &mut self.host
    }

    /// Get device buffer.
    pub fn device(&self) -> &DeviceBuffer {
        &self.device
    }
}
