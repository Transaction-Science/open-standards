//! Tensor storage backends.

use crate::core::{Error, Result};
use crate::hal::{DeviceId, DevicePtr, DeviceType};
use alloc::vec::Vec;

#[cfg(feature = "metal")]
use metal::foreign_types::ForeignType;

/// Safe wrapper for Metal buffer that properly manages lifetime.
/// This avoids the fragile mem::forget pattern by keeping the buffer alive.
#[cfg(feature = "metal")]
#[derive(Debug)]
pub struct MetalBufferHandle {
    /// The underlying Metal buffer (kept alive)
    buffer: metal::Buffer,
    /// Raw pointer for GPU access
    ptr: u64,
    /// Buffer size
    size: usize,
}

#[cfg(feature = "metal")]
impl MetalBufferHandle {
    /// Create a new handle from a Metal buffer.
    pub fn new(buffer: metal::Buffer) -> Self {
        let ptr = buffer.as_ptr() as u64;
        let size = buffer.length() as usize;
        Self { buffer, ptr, size }
    }

    /// Get the raw pointer for GPU access.
    #[inline]
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// Get the underlying buffer reference.
    #[inline]
    pub fn buffer(&self) -> &metal::Buffer {
        &self.buffer
    }

    /// Get the buffer size.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get contents pointer for CPU access (unified memory).
    #[inline]
    pub fn contents(&self) -> *mut std::ffi::c_void {
        self.buffer.contents()
    }
}

/// Storage backend for tensor data.
#[derive(Debug)]
pub enum TensorStorage {
    /// CPU memory (Vec<u8>)
    Cpu(CpuStorage),
    /// Device memory (GPU buffer)
    Device(DeviceStorage),
    /// Memory-mapped file
    Mmap(MmapStorage),
}

impl TensorStorage {
    /// Allocate storage on a device.
    pub fn allocate(size: usize, device: DeviceId) -> Result<Self> {
        match device.device_type {
            DeviceType::Cpu => Ok(Self::Cpu(CpuStorage::allocate(size)?)),
            _ => Ok(Self::Device(DeviceStorage::allocate(size, device)?)),
        }
    }

    /// Create storage from bytes.
    pub fn from_bytes(data: &[u8], device: DeviceId) -> Result<Self> {
        match device.device_type {
            DeviceType::Cpu => Ok(Self::Cpu(CpuStorage::from_bytes(data))),
            _ => {
                // Allocate on device and copy
                let mut storage = DeviceStorage::allocate(data.len(), device)?;
                storage.copy_from_host(data)?;
                Ok(Self::Device(storage))
            }
        }
    }

    /// Get the device this storage resides on.
    pub fn device(&self) -> DeviceId {
        match self {
            Self::Cpu(_) => DeviceId::cpu(),
            Self::Device(d) => d.device,
            Self::Mmap(_) => DeviceId::cpu(),
        }
    }

    /// Get the size in bytes.
    pub fn size(&self) -> usize {
        match self {
            Self::Cpu(c) => c.data.len(),
            Self::Device(d) => d.size,
            Self::Mmap(m) => m.size,
        }
    }

    /// Copy data to a host vector.
    pub fn to_vec<T: bytemuck::Pod>(&self, count: usize) -> Result<Vec<T>> {
        match self {
            Self::Cpu(c) => c.to_vec(count),
            Self::Device(d) => d.to_vec(count),
            Self::Mmap(m) => m.to_vec(count),
        }
    }

    /// Get raw pointer (for CPU storage).
    pub fn as_ptr(&self) -> Option<*const u8> {
        match self {
            Self::Cpu(c) => Some(c.data.as_ptr()),
            Self::Mmap(m) => Some(m.ptr),
            Self::Device(_) => None,
        }
    }

    /// Get device pointer (for GPU storage).
    pub fn device_ptr(&self) -> Option<DevicePtr> {
        match self {
            Self::Device(d) => Some(d.ptr),
            _ => None,
        }
    }

    /// Create from Metal buffer.
    #[cfg(feature = "metal")]
    pub fn from_metal_buffer(buffer: metal::Buffer, device: DeviceId) -> Self {
        Self::Device(DeviceStorage::from_metal_buffer(buffer, device))
    }
}

/// CPU memory storage.
#[derive(Debug)]
pub struct CpuStorage {
    data: Vec<u8>,
}

impl CpuStorage {
    /// Allocate CPU storage.
    pub fn allocate(size: usize) -> Result<Self> {
        Ok(Self {
            data: vec![0u8; size],
        })
    }

    /// Create from bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }

    /// Copy to typed vector.
    pub fn to_vec<T: bytemuck::Pod>(&self, count: usize) -> Result<Vec<T>> {
        let byte_size = count * core::mem::size_of::<T>();
        if byte_size > self.data.len() {
            return Err(Error::InvalidArgument {
                name: "to_vec".into(),
                message: format!(
                    "requested {} bytes but storage only has {}",
                    byte_size,
                    self.data.len()
                ),
            });
        }

        let slice = &self.data[..byte_size];
        Ok(bytemuck::cast_slice(slice).to_vec())
    }

    /// Get as byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get as mutable byte slice.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

/// Device (GPU) memory storage.
#[derive(Debug)]
pub struct DeviceStorage {
    ptr: DevicePtr,
    size: usize,
    device: DeviceId,
    /// Safe Metal buffer handle (keeps buffer alive)
    #[cfg(feature = "metal")]
    metal_handle: Option<MetalBufferHandle>,
}

#[cfg(feature = "metal")]
static METAL_DEVICE: std::sync::OnceLock<std::sync::Arc<crate::hal::metal::MetalDevice>> = std::sync::OnceLock::new();

/// Get or initialize the global Metal device.
#[cfg(feature = "metal")]
pub fn get_metal_device() -> Result<&'static std::sync::Arc<crate::hal::metal::MetalDevice>> {
    Ok(METAL_DEVICE.get_or_init(|| {
        crate::hal::metal::MetalDevice::new()
            .map(std::sync::Arc::new)
            .expect("Failed to initialize Metal device")
    }))
}

impl DeviceStorage {
    /// Allocate device storage.
    pub fn allocate(size: usize, device: DeviceId) -> Result<Self> {
        match device.device_type {
            #[cfg(feature = "metal")]
            DeviceType::Metal => {
                let metal_dev = get_metal_device()?;
                // Use default resource options for now
                let options = crate::hal::metal::ResourceOptions::default();
                let buffer = metal_dev.create_buffer(size, options)?;

                // Use safe wrapper to manage buffer lifetime
                let handle = MetalBufferHandle::new(buffer);
                let ptr_val = handle.ptr();

                Ok(Self {
                    ptr: DevicePtr::new(ptr_val),
                    size,
                    device,
                    metal_handle: Some(handle),
                })
            }
            #[cfg(not(feature = "metal"))]
            DeviceType::Metal => Err(Error::unsupported("Metal backend not enabled")),

            _ => {
                Err(Error::unsupported(format!(
                    "device storage allocation not implemented for {:?}",
                    device.device_type
                )))
            }
        }
    }

    /// Create from existing Metal buffer.
    #[cfg(feature = "metal")]
    pub fn from_metal_buffer(buffer: metal::Buffer, device: DeviceId) -> Self {
        let handle = MetalBufferHandle::new(buffer);
        let size = handle.size();
        let ptr_val = handle.ptr();

        Self {
            ptr: DevicePtr::new(ptr_val),
            size,
            device,
            metal_handle: Some(handle),
        }
    }

    /// Get the Metal buffer handle if available.
    #[cfg(feature = "metal")]
    pub fn metal_buffer(&self) -> Option<&metal::Buffer> {
        self.metal_handle.as_ref().map(|h| h.buffer())
    }

    /// Copy from host memory.
    pub fn copy_from_host(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(Error::InvalidArgument {
                name: "copy_from_host".into(),
                message: format!("data size {} exceeds buffer size {}", data.len(), self.size),
            });
        }

        match self.device.device_type {
            #[cfg(feature = "metal")]
            DeviceType::Metal => {
                // Use safe handle instead of reconstructing from raw pointer
                let handle = self.metal_handle.as_ref().ok_or_else(|| {
                    Error::internal("Metal buffer handle not available")
                })?;

                // Copy data using the safe handle
                let dst = handle.contents() as *mut u8;
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
                }

                Ok(())
            }
            _ => Err(Error::unsupported(format!(
                "copy_from_host not implemented for {:?}",
                self.device.device_type
            ))),
        }
    }

    /// Copy to host memory.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        if dst.len() > self.size {
            return Err(Error::InvalidArgument {
                name: "copy_to_host".into(),
                message: format!("dst size {} exceeds buffer size {}", dst.len(), self.size),
            });
        }

        match self.device.device_type {
            #[cfg(feature = "metal")]
            DeviceType::Metal => {
                // Use safe handle instead of reconstructing from raw pointer
                let handle = self.metal_handle.as_ref().ok_or_else(|| {
                    Error::internal("Metal buffer handle not available")
                })?;

                let src = handle.contents() as *const u8;
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
                }

                Ok(())
            }
            _ => Err(Error::unsupported(format!(
                "copy_to_host not implemented for {:?}",
                self.device.device_type
            ))),
        }
    }

    /// Copy to typed vector.
    pub fn to_vec<T: bytemuck::Pod>(&self, count: usize) -> Result<Vec<T>> {
        let byte_size = count * core::mem::size_of::<T>();
        let mut bytes = vec![0u8; byte_size];
        self.copy_to_host(&mut bytes)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }
}

impl Drop for DeviceStorage {
    fn drop(&mut self) {
        // The MetalBufferHandle (if present) will automatically release
        // the Metal buffer when it goes out of scope. No manual cleanup needed.
        // This is safe because we now properly own the buffer through the handle.
        #[cfg(feature = "metal")]
        {
            // metal_handle will be dropped automatically, releasing the buffer
            self.metal_handle = None;
        }
    }
}

/// Memory-mapped file storage.
#[derive(Debug)]
pub struct MmapStorage {
    ptr: *const u8,
    size: usize,
    #[cfg(feature = "std")]
    _mmap: memmap2::Mmap,
}

impl MmapStorage {
    /// Create from a memory-mapped file.
    #[cfg(feature = "std")]
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| {
            Error::io_with_source("open", format!("{}", path.display()), e)
        })?;

        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
            Error::io_with_source("mmap", format!("{}", path.display()), e)
        })?;

        let ptr = mmap.as_ptr();
        let size = mmap.len();

        Ok(Self {
            ptr,
            size,
            _mmap: mmap,
        })
    }

    /// Copy to typed vector.
    pub fn to_vec<T: bytemuck::Pod>(&self, count: usize) -> Result<Vec<T>> {
        let byte_size = count * core::mem::size_of::<T>();
        if byte_size > self.size {
            return Err(Error::InvalidArgument {
                name: "to_vec".into(),
                message: format!(
                    "requested {} bytes but mmap only has {}",
                    byte_size, self.size
                ),
            });
        }

        let slice = unsafe { core::slice::from_raw_parts(self.ptr, byte_size) };
        Ok(bytemuck::cast_slice(slice).to_vec())
    }
}

// Safety: MmapStorage is read-only and the underlying file handle keeps it valid
unsafe impl Send for MmapStorage {}
unsafe impl Sync for MmapStorage {}
