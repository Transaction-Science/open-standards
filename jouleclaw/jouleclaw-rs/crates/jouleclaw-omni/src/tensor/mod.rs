//! Tensor types and operations.
//!
//! This module provides the core tensor abstraction with support for:
//! - Multiple data types (F32, F16, BF16, F8, INT8)
//! - Device-agnostic storage (CPU, GPU)
//! - Zero-copy views and slicing
//! - Memory-efficient operations

mod storage;
mod view;
/// Tensor operations module.
pub mod ops;

pub use storage::TensorStorage;
#[cfg(feature = "metal")]
pub use storage::get_metal_device;
pub use view::TensorView;

use crate::core::{Error, Id, Result};
pub use crate::core::{DType, Shape};
use crate::hal::DeviceId;
use alloc::sync::Arc;
use alloc::vec::Vec;

/// A multi-dimensional array stored on a compute device.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Unique identifier
    id: Id,
    /// Shape of the tensor
    shape: Shape,
    /// Data type
    dtype: DType,
    /// Storage (shared, reference counted)
    storage: Arc<TensorStorage>,
    /// Offset into storage (for views)
    offset: usize,
    /// Strides for each dimension
    strides: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor with given shape and dtype.
    pub fn new(shape: impl Into<Shape>, dtype: DType, storage: TensorStorage) -> Self {
        let shape = shape.into();
        let strides = shape.strides();

        Self {
            id: Id::new(),
            shape,
            dtype,
            storage: Arc::new(storage),
            offset: 0,
            strides,
        }
    }

    /// Create an uninitialized tensor on a device.
    pub fn empty(shape: impl Into<Shape>, dtype: DType, device: DeviceId) -> Result<Self> {
        let shape = shape.into();
        let size_bytes = shape.numel() * dtype.size_bytes();
        let storage = TensorStorage::allocate(size_bytes, device)?;
        Ok(Self::new(shape, dtype, storage))
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: impl Into<Shape>, dtype: DType) -> Result<Self> {
        Self::zeros_on(shape, dtype, DeviceId::cpu())
    }

    /// Create a tensor filled with zeros on a specific device.
    ///
    /// Note: `TensorStorage::allocate` already zeroes CPU memory via `vec![0u8; size]`.
    /// For device memory, Metal buffers are allocated with default options which zero memory.
    pub fn zeros_on(shape: impl Into<Shape>, dtype: DType, device: DeviceId) -> Result<Self> {
        let tensor = Self::empty(shape, dtype, device)?;
        // CPU storage is already zeroed by Vec allocation.
        // Metal buffers with default resource options are also zeroed.
        Ok(tensor)
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: impl Into<Shape>, dtype: DType) -> Result<Self> {
        Self::ones_on(shape, dtype, DeviceId::cpu())
    }

    /// Create a tensor filled with ones on a specific device.
    pub fn ones_on(shape: impl Into<Shape>, dtype: DType, device: DeviceId) -> Result<Self> {
        let shape = shape.into();
        let numel = shape.numel();

        // Build the byte representation of "1" for this dtype and fill
        match dtype {
            DType::F32 => {
                let data = vec![1.0f32; numel];
                Self::from_slice(&data, shape, dtype, device)
            }
            DType::F16 => {
                use half::f16;
                let data = vec![f16::from_f32(1.0); numel];
                Self::from_slice(&data, shape, dtype, device)
            }
            DType::BF16 => {
                use half::bf16;
                let data = vec![bf16::from_f32(1.0); numel];
                Self::from_slice(&data, shape, dtype, device)
            }
            DType::I32 => {
                let data = vec![1i32; numel];
                Self::from_slice(&data, shape, dtype, device)
            }
            DType::I64 => {
                let data = vec![1i64; numel];
                Self::from_slice(&data, shape, dtype, device)
            }
            DType::I8 => {
                let data = vec![1i8; numel];
                Self::from_slice(&data, shape, dtype, device)
            }
            DType::U8 => {
                let data = vec![1u8; numel];
                Self::from_slice(&data, shape, dtype, device)
            }
            DType::U32 => {
                let data = vec![1u32; numel];
                Self::from_slice(&data, shape, dtype, device)
            }
            DType::Bool => {
                let data = vec![1u8; numel]; // true
                Self::from_slice(&data, shape, dtype, device)
            }
            DType::F8E4M3 | DType::F8E5M2 => {
                // F8 types don't have standard Rust types
                // Use byte representation of 1.0 in the respective format
                let data = vec![0x3Cu8; numel]; // Approximate 1.0 in E4M3/E5M2
                Self::from_slice(&data, shape, dtype, device)
            }
        }
    }

    /// Create a tensor with random normal values.
    pub fn randn(shape: impl Into<Shape>, dtype: DType) -> Result<Self> {
        Self::randn_on(shape, dtype, DeviceId::cpu())
    }

    /// Create a tensor with random normal values on a specific device.
    pub fn randn_on(shape: impl Into<Shape>, dtype: DType, device: DeviceId) -> Result<Self> {
        let shape: Shape = shape.into();
        let numel = shape.numel();
        let mut data = Vec::with_capacity(numel);
        let mut rng = rand::rng();

        use half::f16;
        use rand::Rng;

        // Standard normal via Box–Muller (pair of independent N(0,1) per
        // uniform pair). Until this fix, this function returned `Uniform[-1,1]`
        // — a comment in the source explicitly said "Uniform -1 to 1 is easier
        // and visible" — which gave std ≈ 0.577 and a hard [-1,1] range.
        // Stable-diffusion latents need true Gaussian noise: SD 1.5 / SDXL
        // sample x_T ~ N(0, 1) at the terminal training timestep ᾱ ≈ 0.
        // Uniform init plus the DDPM scheduler's σ=1 produced latents with
        // a bounded, kurtotic distribution rather than the unbounded Gaussian
        // the U-Net was trained on — visible as the tight ±1 latent ceiling
        // hitting conv_in too peakedly and then drifting up under DDPM steps
        // into the NaN regime.
        let mut i = 0;
        while i < numel {
            // Avoid u1=0 (log(0) = -inf).
            let mut u1: f32 = rng.random_range(0.0..1.0);
            if u1 < 1e-7 { u1 = 1e-7; }
            let u2: f32 = rng.random_range(0.0..1.0);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            let z0 = r * theta.cos();
            data.push(f16::from_f32(z0));
            i += 1;
            if i < numel {
                let z1 = r * theta.sin();
                data.push(f16::from_f32(z1));
                i += 1;
            }
        }

        Self::from_slice(&data, shape, dtype, device)
    }

    /// Get size in bytes.
    pub fn size(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }

    /// Create a tensor from host data.
    pub fn from_slice<T: bytemuck::Pod>(
        data: &[T],
        shape: impl Into<Shape>,
        dtype: DType,
        device: DeviceId,
    ) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.numel() {
            return Err(Error::shape_mismatch(
                format!("{} elements", shape.numel()),
                format!("{} elements", data.len()),
            ));
        }

        let bytes = bytemuck::cast_slice(data);
        let storage = TensorStorage::from_bytes(bytes, device)?;
        Ok(Self::new(shape, dtype, storage))
    }

    /// Get tensor ID.
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get tensor shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get tensor data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Get total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }

    /// Get the device this tensor is stored on.
    pub fn device(&self) -> DeviceId {
        self.storage.device()
    }

    /// Get the device pointer (if on device).
    pub fn device_ptr(&self) -> Option<crate::hal::DevicePtr> {
        self.storage.device_ptr()
    }

    /// Get the byte offset into storage (non-zero for sliced/view tensors).
    pub fn byte_offset(&self) -> usize {
        self.offset
    }

    /// Create from Metal buffer.
    #[cfg(feature = "metal")]
    pub fn from_metal_buffer(
        buffer: metal::Buffer,
        shape: impl Into<Shape>,
        dtype: DType,
        device: DeviceId,
    ) -> Self {
        let shape = shape.into();
        let strides = shape.strides();
        let storage = TensorStorage::from_metal_buffer(buffer, device);
        
        Self {
            id: Id::new(),
            shape,
            dtype,
            storage: Arc::new(storage),
            offset: 0,
            strides,
        }
    }

    /// Get strides.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Check if tensor is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        self.strides == self.shape.strides()
    }

    /// Get a dimension by index.
    pub fn dim(&self, idx: usize) -> Option<usize> {
        self.shape.dim(idx)
    }

    /// Create a view of this tensor.
    pub fn view(&self) -> TensorView<'_> {
        TensorView::new(self)
    }

    /// Reshape the tensor (returns a view if possible).
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Result<Tensor> {
        let new_shape = new_shape.into();

        if new_shape.numel() != self.numel() {
            return Err(Error::shape_mismatch(
                format!("{} elements", self.numel()),
                format!("{} elements", new_shape.numel()),
            ));
        }

        if !self.is_contiguous() {
            return Err(Error::unsupported(
                "reshape of non-contiguous tensor requires copy",
            ));
        }

        let strides = new_shape.strides();
        Ok(Tensor {
            id: Id::new(),
            shape: new_shape,
            dtype: self.dtype,
            storage: Arc::clone(&self.storage),
            offset: self.offset,
            strides,
        })
    }

    /// Transpose dimensions.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Tensor> {
        if dim0 >= self.rank() || dim1 >= self.rank() {
            return Err(Error::InvalidArgument {
                name: "transpose".into(),
                message: format!(
                    "dimensions ({}, {}) out of range for tensor of rank {}",
                    dim0,
                    dim1,
                    self.rank()
                ),
            });
        }

        let mut new_dims: Vec<usize> = self.shape.dims().to_vec();
        let mut new_strides = self.strides.clone();

        new_dims.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        Ok(Tensor {
            id: Id::new(),
            shape: Shape::new(new_dims),
            dtype: self.dtype,
            storage: Arc::clone(&self.storage),
            offset: self.offset,
            strides: new_strides,
        })
    }

    /// Slice the tensor along a dimension.
    pub fn slice(&self, dim: usize, start: usize, end: usize) -> Result<Tensor> {
        if dim >= self.rank() {
            return Err(Error::InvalidArgument {
                name: "slice".into(),
                message: format!("dimension {} out of range for rank {}", dim, self.rank()),
            });
        }

        let dim_size = self.shape.dim(dim).unwrap();
        if start >= dim_size || end > dim_size || start >= end {
            return Err(Error::InvalidArgument {
                name: "slice".into(),
                message: format!(
                    "invalid slice [{}, {}) for dimension of size {}",
                    start, end, dim_size
                ),
            });
        }

        let mut new_dims: Vec<usize> = self.shape.dims().to_vec();
        new_dims[dim] = end - start;

        let new_offset = self.offset + start * self.strides[dim] * self.dtype.size_bytes();

        Ok(Tensor {
            id: Id::new(),
            shape: Shape::new(new_dims),
            dtype: self.dtype,
            storage: Arc::clone(&self.storage),
            offset: new_offset,
            strides: self.strides.clone(),
        })
    }

    /// Make a contiguous copy if necessary.
    ///
    /// If the tensor is already contiguous, returns a clone (shared storage).
    /// Otherwise, copies data into a new contiguous buffer preserving the
    /// logical layout defined by the current shape and strides.
    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        // Only CPU tensors supported for now
        if self.device().device_type != crate::hal::DeviceType::Cpu {
            return Err(Error::unsupported(
                "contiguous copy of device tensors not yet supported",
            ));
        }

        let numel = self.numel();
        let elem_size = self.dtype.size_bytes();
        let total_bytes = numel * elem_size;

        // Read raw bytes from storage
        let src_ptr = self
            .storage
            .as_ptr()
            .ok_or_else(|| Error::internal("cannot get CPU pointer for contiguous copy"))?;

        let mut dst_bytes = vec![0u8; total_bytes];
        let dims = self.shape.dims();
        let rank = dims.len();

        // Iterate over all elements using the strided layout and write contiguously
        let contiguous_strides = self.shape.strides();
        let mut indices = vec![0usize; rank];
        for flat_idx in 0..numel {
            // Compute source offset using the (possibly non-contiguous) strides
            let src_offset = self.offset
                + indices
                    .iter()
                    .zip(self.strides.iter())
                    .map(|(&idx, &stride)| idx * stride * elem_size)
                    .sum::<usize>();

            let dst_offset = flat_idx * elem_size;

            // Copy one element
            unsafe {
                core::ptr::copy_nonoverlapping(
                    src_ptr.add(src_offset),
                    dst_bytes.as_mut_ptr().add(dst_offset),
                    elem_size,
                );
            }

            // Increment indices (row-major order)
            for d in (0..rank).rev() {
                indices[d] += 1;
                if indices[d] < dims[d] {
                    break;
                }
                indices[d] = 0;
            }
        }

        let storage = TensorStorage::from_bytes(&dst_bytes, self.device())?;
        Ok(Tensor::new(self.shape.clone(), self.dtype, storage))
    }

    /// Copy data to host memory.
    pub fn to_vec<T: bytemuck::Pod>(&self) -> Result<Vec<T>> {
        if core::mem::size_of::<T>() != self.dtype.size_bytes() {
            return Err(Error::DTypeMismatch {
                expected: format!("{:?}", self.dtype),
                got: format!("T with size {}", core::mem::size_of::<T>()),
            });
        }

        if !self.is_contiguous() {
            return Err(Error::unsupported(
                "to_vec on non-contiguous tensor requires copy",
            ));
        }

        self.storage.to_vec(self.numel())
    }

    /// Copy data to host memory as f32, converting from F16/BF16 if needed.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        if !self.is_contiguous() {
            return Err(Error::unsupported(
                "to_f32_vec on non-contiguous tensor requires copy",
            ));
        }

        match self.dtype {
            DType::F32 => self.storage.to_vec(self.numel()),
            DType::F16 => {
                let f16_data: Vec<half::f16> = self.storage.to_vec(self.numel())?;
                Ok(f16_data.iter().map(|v| v.to_f32()).collect())
            }
            DType::BF16 => {
                let u16_data: Vec<u16> = self.storage.to_vec(self.numel())?;
                Ok(u16_data.iter().map(|&bits| f32::from_bits((bits as u32) << 16)).collect())
            }
            _ => Err(Error::DTypeMismatch {
                expected: "F32, F16, or BF16".to_string(),
                got: format!("{:?}", self.dtype),
            }),
        }
    }

    /// Cast to a different data type.
    ///
    /// Supports casting between F32, F16, BF16, I32, and U8.
    /// Requires the tensor to be contiguous.
    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor> {
        if dtype == self.dtype {
            return Ok(self.clone());
        }

        // Ensure contiguous
        let src = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()?
        };

        let numel = src.numel();

        // Read source as f32 intermediary
        let f32_data: Vec<f32> = match src.dtype {
            DType::F32 => src.to_vec()?,
            DType::F16 => {
                let data: Vec<half::f16> = src.to_vec()?;
                data.iter().map(|v| v.to_f32()).collect()
            }
            DType::BF16 => {
                let data: Vec<half::bf16> = src.to_vec()?;
                data.iter().map(|v| v.to_f32()).collect()
            }
            DType::I32 => {
                let data: Vec<i32> = src.to_vec()?;
                data.iter().map(|&v| v as f32).collect()
            }
            DType::I64 => {
                let data: Vec<i64> = src.to_vec()?;
                data.iter().map(|&v| v as f32).collect()
            }
            DType::I8 => {
                let data: Vec<i8> = src.to_vec()?;
                data.iter().map(|&v| v as f32).collect()
            }
            DType::U8 | DType::Bool => {
                let data: Vec<u8> = src.to_vec()?;
                data.iter().map(|&v| v as f32).collect()
            }
            DType::U32 => {
                let data: Vec<u32> = src.to_vec()?;
                data.iter().map(|&v| v as f32).collect()
            }
            _ => return Err(Error::unsupported(format!("cast from {:?}", src.dtype))),
        };

        // Convert f32 intermediary to target dtype
        match dtype {
            DType::F32 => {
                Self::from_slice(&f32_data, src.shape.clone(), dtype, src.device())
            }
            DType::F16 => {
                let data: Vec<half::f16> = f32_data.iter().map(|&v| half::f16::from_f32(v)).collect();
                Self::from_slice(&data, src.shape.clone(), dtype, src.device())
            }
            DType::BF16 => {
                let data: Vec<half::bf16> = f32_data.iter().map(|&v| half::bf16::from_f32(v)).collect();
                Self::from_slice(&data, src.shape.clone(), dtype, src.device())
            }
            DType::I32 => {
                let data: Vec<i32> = f32_data.iter().map(|&v| v as i32).collect();
                Self::from_slice(&data, src.shape.clone(), dtype, src.device())
            }
            DType::I64 => {
                let data: Vec<i64> = f32_data.iter().map(|&v| v as i64).collect();
                Self::from_slice(&data, src.shape.clone(), dtype, src.device())
            }
            DType::I8 => {
                let data: Vec<i8> = f32_data.iter().map(|&v| v as i8).collect();
                Self::from_slice(&data, src.shape.clone(), dtype, src.device())
            }
            DType::U8 => {
                let data: Vec<u8> = f32_data.iter().map(|&v| v as u8).collect();
                Self::from_slice(&data, src.shape.clone(), dtype, src.device())
            }
            DType::U32 => {
                let data: Vec<u32> = f32_data.iter().map(|&v| v as u32).collect();
                Self::from_slice(&data, src.shape.clone(), dtype, src.device())
            }
            DType::Bool => {
                let data: Vec<u8> = f32_data.iter().map(|&v| if v != 0.0 { 1 } else { 0 }).collect();
                Self::from_slice(&data, src.shape.clone(), dtype, src.device())
            }
            _ => Err(Error::unsupported(format!("cast to {:?}", dtype))),
        }
    }

    /// Move tensor to a different device.
    ///
    /// Copies the tensor data to a new device. Supports CPU-to-CPU (no-op),
    /// CPU-to-Metal, and Metal-to-CPU transfers.
    pub fn to_device(&self, device: DeviceId) -> Result<Tensor> {
        if device == self.device() {
            return Ok(self.clone());
        }

        // Ensure contiguous before device transfer
        let src = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()?
        };

        let size_bytes = src.size_bytes();

        match (src.device().device_type, device.device_type) {
            // CPU -> CPU (shouldn't happen due to equality check, but handle it)
            (crate::hal::DeviceType::Cpu, crate::hal::DeviceType::Cpu) => Ok(src),

            // CPU -> Device: read bytes from CPU, create storage on device
            (crate::hal::DeviceType::Cpu, _) => {
                let src_ptr = src
                    .storage
                    .as_ptr()
                    .ok_or_else(|| Error::internal("cannot get CPU pointer for device transfer"))?;

                let bytes =
                    unsafe { core::slice::from_raw_parts(src_ptr.add(src.offset), size_bytes) };

                let storage = TensorStorage::from_bytes(bytes, device)?;
                Ok(Tensor::new(src.shape.clone(), src.dtype, storage))
            }

            // Device -> CPU: copy bytes from device to CPU
            (_, crate::hal::DeviceType::Cpu) => {
                let mut bytes = vec![0u8; size_bytes];
                match &*src.storage {
                    TensorStorage::Device(d) => d.copy_to_host(&mut bytes)?,
                    _ => {
                        return Err(Error::internal(
                            "expected device storage for device-to-cpu transfer",
                        ))
                    }
                }

                let storage = TensorStorage::from_bytes(&bytes, device)?;
                Ok(Tensor::new(src.shape.clone(), src.dtype, storage))
            }

            // Device -> Device: copy via CPU as intermediary
            (_, _) => {
                let cpu_tensor = src.to_device(DeviceId::cpu())?;
                cpu_tensor.to_device(device)
            }
        }
    }

    /// Concatenate tensors along a dimension.
    pub fn cat(tensors: &[Tensor], dim: usize) -> Result<Tensor> {
        let refs: Vec<&Tensor> = tensors.iter().collect();
        ops::cat(&refs, dim as i64)
    }
}

