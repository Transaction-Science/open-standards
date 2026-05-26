//! Zero-copy Metal buffers for UMA.
//!
//! With Unified Memory Architecture, CPU and GPU share the same physical memory.
//! This module provides buffer types that exploit this for zero-copy operations.

use super::{MetalDevice, ResourceOptions};
use crate::core::{DType, Error, Result, Shape};
use std::sync::Arc;

#[cfg(feature = "metal")]
use metal::Buffer;

/// A Metal buffer with UMA-aware semantics.
#[cfg(feature = "metal")]
pub struct MetalBuffer {
    /// Underlying Metal buffer
    buffer: Buffer,
    /// Size in bytes
    size: usize,
    /// Data type
    dtype: DType,
    /// Shape (if tensor-like)
    shape: Option<Shape>,
    /// Whether this is a view (doesn't own memory)
    is_view: bool,
    /// Device reference for cleanup
    device: Arc<MetalDevice>,
}

#[cfg(feature = "metal")]
impl MetalBuffer {
    /// Create a new buffer.
    pub fn new(
        device: Arc<MetalDevice>,
        size: usize,
        options: ResourceOptions,
    ) -> Result<Self> {
        let buffer = device.create_buffer(size, options)?;

        Ok(Self {
            buffer,
            size,
            dtype: DType::U8,
            shape: None,
            is_view: false,
            device,
        })
    }

    /// Create a buffer with specific dtype and shape.
    pub fn with_shape(
        device: Arc<MetalDevice>,
        shape: Shape,
        dtype: DType,
        options: ResourceOptions,
    ) -> Result<Self> {
        let size = shape.numel() * dtype.size_bytes();
        let buffer = device.create_buffer(size, options)?;

        Ok(Self {
            buffer,
            size,
            dtype,
            shape: Some(shape),
            is_view: false,
            device,
        })
    }

    /// Create from existing data (zero-copy with UMA).
    pub fn from_slice<T: bytemuck::Pod>(
        device: Arc<MetalDevice>,
        data: &[T],
        dtype: DType,
    ) -> Result<Self> {
        let bytes = bytemuck::cast_slice(data);
        let buffer = device.create_buffer_with_data(bytes, ResourceOptions::default())?;

        Ok(Self {
            buffer,
            size: bytes.len(),
            dtype,
            shape: Some(Shape::new([data.len()])),
            is_view: false,
            device,
        })
    }

    /// Get the underlying Metal buffer.
    pub fn raw(&self) -> &Buffer {
        &self.buffer
    }

    /// Get size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get shape.
    pub fn shape(&self) -> Option<&Shape> {
        self.shape.as_ref()
    }

    /// Get contents as slice (zero-copy for Shared storage).
    ///
    /// # Safety
    /// Must ensure GPU is not writing to this buffer.
    pub unsafe fn as_slice<T: bytemuck::Pod>(&self) -> &[T] {
        let ptr = self.buffer.contents() as *const T;
        let count = self.size / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(ptr, count) }
    }

    /// Get mutable contents (zero-copy for Shared storage).
    ///
    /// # Safety
    /// Must ensure GPU is not accessing this buffer.
    pub unsafe fn as_mut_slice<T: bytemuck::Pod>(&self) -> &mut [T] {
        let ptr = self.buffer.contents() as *mut T;
        let count = self.size / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(ptr, count) }
    }

    /// Copy data to this buffer.
    ///
    /// # Safety
    /// Must ensure GPU is not reading from this buffer.
    pub unsafe fn copy_from_slice<T: bytemuck::Pod>(&self, data: &[T]) -> Result<()> {
        let bytes = bytemuck::cast_slice(data);
        if bytes.len() > self.size {
            return Err(Error::InvalidArgument {
                name: "data".into(),
                message: format!("data size {} exceeds buffer size {}", bytes.len(), self.size),
            });
        }

        let dst = self.buffer.contents() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }

        Ok(())
    }

    /// Create a view into this buffer.
    pub fn slice(&self, offset: usize, length: usize) -> Result<MetalBufferView<'_>> {
        if offset + length > self.size {
            return Err(Error::InvalidArgument {
                name: "slice".into(),
                message: format!(
                    "slice [{}, {}) out of bounds for buffer of size {}",
                    offset,
                    offset + length,
                    self.size
                ),
            });
        }

        Ok(MetalBufferView {
            buffer: &self.buffer,
            offset,
            length,
        })
    }

    /// GPU address for compute kernels.
    pub fn gpu_address(&self) -> u64 {
        self.buffer.gpu_address()
    }
}

#[cfg(feature = "metal")]
impl Drop for MetalBuffer {
    fn drop(&mut self) {
        if !self.is_view {
            self.device.record_buffer_free(self.size);
        }
    }
}

/// A view into a Metal buffer (no ownership).
#[cfg(feature = "metal")]
pub struct MetalBufferView<'a> {
    buffer: &'a Buffer,
    offset: usize,
    length: usize,
}

#[cfg(feature = "metal")]
impl<'a> MetalBufferView<'a> {
    /// Get the underlying buffer.
    pub fn buffer(&self) -> &Buffer {
        self.buffer
    }

    /// Get offset.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get length.
    pub fn length(&self) -> usize {
        self.length
    }
}

/// Buffer pool for efficient allocation.
#[cfg(feature = "metal")]
pub struct BufferPool {
    /// Device
    device: Arc<MetalDevice>,
    /// Free buffers by size class
    free_buffers: parking_lot::Mutex<std::collections::BTreeMap<usize, Vec<Buffer>>>,
    /// Total pooled bytes
    pooled_bytes: std::sync::atomic::AtomicUsize,
    /// Options for new allocations
    options: ResourceOptions,
}

#[cfg(feature = "metal")]
impl BufferPool {
    /// Create a new buffer pool.
    pub fn new(device: Arc<MetalDevice>, options: ResourceOptions) -> Self {
        Self {
            device,
            free_buffers: parking_lot::Mutex::new(std::collections::BTreeMap::new()),
            pooled_bytes: std::sync::atomic::AtomicUsize::new(0),
            options,
        }
    }

    /// Get or allocate a buffer of at least the requested size.
    pub fn acquire(&self, min_size: usize) -> Result<PooledBuffer<'_>> {
        let size_class = size_class(min_size);

        // Try to get from pool
        {
            let mut free = self.free_buffers.lock();
            if let Some(buffers) = free.get_mut(&size_class) {
                if let Some(buffer) = buffers.pop() {
                    self.pooled_bytes.fetch_sub(size_class, std::sync::atomic::Ordering::Relaxed);
                    return Ok(PooledBuffer {
                        buffer: std::mem::ManuallyDrop::new(buffer),
                        size: size_class,
                        pool: self,
                    });
                }
            }
        }

        // Allocate new buffer
        let buffer = self.device.create_buffer(size_class, self.options)?;
        Ok(PooledBuffer {
            buffer: std::mem::ManuallyDrop::new(buffer),
            size: size_class,
            pool: self,
        })
    }

    /// Return a buffer to the pool.
    fn release(&self, buffer: Buffer, size: usize) {
        let mut free = self.free_buffers.lock();
        free.entry(size).or_default().push(buffer);
        self.pooled_bytes.fetch_add(size, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get pooled bytes.
    pub fn pooled_bytes(&self) -> usize {
        self.pooled_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Clear the pool.
    pub fn clear(&self) {
        let mut free = self.free_buffers.lock();
        free.clear();
        self.pooled_bytes.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// A buffer from a pool (returns on drop).
#[cfg(feature = "metal")]
pub struct PooledBuffer<'a> {
    buffer: std::mem::ManuallyDrop<Buffer>,
    size: usize,
    pool: &'a BufferPool,
}

#[cfg(feature = "metal")]
impl<'a> PooledBuffer<'a> {
    /// Get the underlying buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get size.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Take ownership (won't return to pool).
    pub fn take(mut self) -> Buffer {
        // SAFETY: We take the buffer out and then forget self so Drop won't run.
        let buffer = unsafe { std::mem::ManuallyDrop::take(&mut self.buffer) };
        std::mem::forget(self);
        buffer
    }
}

#[cfg(feature = "metal")]
impl<'a> Drop for PooledBuffer<'a> {
    fn drop(&mut self) {
        // SAFETY: This is the only place that takes the buffer out during drop.
        let buffer = unsafe { std::mem::ManuallyDrop::take(&mut self.buffer) };
        self.pool.release(buffer, self.size);
    }
}

/// Round up to power of 2 size class.
fn size_class(size: usize) -> usize {
    const MIN_SIZE: usize = 256; // Minimum allocation
    size.max(MIN_SIZE).next_power_of_two()
}

/// A non-owning reference to a Metal buffer, created from a raw device pointer.
///
/// This replaces the unsafe pattern of `Buffer::from_ptr()` + `std::mem::forget()`
/// used throughout the codebase for passing tensor buffers to compute encoders.
/// `ManuallyDrop` prevents the buffer from being released when this wrapper drops.
///
/// # Safety
/// The caller must ensure the underlying tensor/buffer remains alive for the
/// lifetime of this wrapper and any GPU command buffers that reference it.
#[cfg(feature = "metal")]
pub struct BorrowedMetalBuffer {
    inner: std::mem::ManuallyDrop<Buffer>,
}

#[cfg(feature = "metal")]
impl BorrowedMetalBuffer {
    /// Create a non-owning buffer reference from a `DevicePtr`.
    ///
    /// # Safety
    /// - `ptr` must be a valid Metal buffer pointer
    /// - The underlying buffer must remain alive for the lifetime of this wrapper
    pub unsafe fn from_device_ptr(ptr: crate::hal::DevicePtr) -> Self {
        use metal::foreign_types::ForeignType;
        let buffer = unsafe { Buffer::from_ptr(ptr.raw() as *mut _) };
        Self {
            inner: std::mem::ManuallyDrop::new(buffer),
        }
    }

    /// Get a reference to the Metal buffer for passing to encoders.
    pub fn as_ref(&self) -> &Buffer {
        &self.inner
    }
}

// Stub for non-macOS
#[cfg(not(feature = "metal"))]
pub struct MetalBuffer;

#[cfg(not(feature = "metal"))]
pub struct BufferPool;
