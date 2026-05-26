//! Tensor operations.
//!
//! Operations are implemented as standalone functions that work
//! with any tensor, regardless of device. The actual implementation
//! is dispatched based on the tensor's storage location.

use super::{DType, Shape, Tensor};
use crate::core::{Error, Result};
use crate::hal::DeviceType;
use alloc::vec;
use alloc::vec::Vec;
use half;

// ============================================================================
// Metal helpers (cached MetalOps + buffer extraction)
// ============================================================================

#[cfg(feature = "metal")]
use crate::hal::metal::{MetalOps, BorrowedMetalBuffer};

/// Cached MetalOps instance to avoid recompiling pipelines on every call.
#[cfg(feature = "metal")]
static METAL_OPS: std::sync::OnceLock<MetalOps> = std::sync::OnceLock::new();

/// Get or initialise the global MetalOps instance.
#[cfg(feature = "metal")]
fn get_metal_ops() -> Result<&'static MetalOps> {
    if let Some(ops) = METAL_OPS.get() {
        return Ok(ops);
    }
    let dev = crate::tensor::storage::get_metal_device()?;
    let ops = MetalOps::new(dev.clone())?;
    // Another thread may have initialized it between our get() and set().
    // OnceLock::set returns Err(value) on conflict but the already-set instance is fine.
    let _ = METAL_OPS.set(ops);
    Ok(METAL_OPS.get().expect("OnceLock was just set"))
}

/// Extract a borrowed Metal buffer from a tensor that is known to reside on a
/// Metal device. The returned `BorrowedMetalBuffer` is non-owning: it will
/// *not* release the underlying buffer when dropped, so the tensor must stay
/// alive for as long as the buffer is in use.
#[cfg(feature = "metal")]
fn borrow_metal_buffer(tensor: &Tensor) -> Result<BorrowedMetalBuffer> {
    let ptr = tensor.device_ptr().ok_or_else(|| {
        Error::internal("tensor on Metal device has no device pointer")
    })?;
    Ok(unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
}

/// Allocate an output tensor on the Metal device and return both the tensor and
/// a borrowed reference to its Metal buffer. The tensor is uninitialized.
#[cfg(feature = "metal")]
fn alloc_metal_output(
    shape: Shape,
    dtype: DType,
    device: crate::hal::DeviceId,
) -> Result<Tensor> {
    Tensor::empty(shape, dtype, device)
}

fn tensor_from_f32_as(data: Vec<f32>, shape: Shape, dtype: DType, device: crate::hal::DeviceId) -> Result<Tensor> {
    match dtype {
        DType::F16 => {
            let f16_data: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
            Tensor::from_slice(&f16_data, shape, DType::F16, device)
        }
        DType::BF16 => {
            let bf16_data: Vec<u16> = data.iter().map(|&v| {
                let bits = v.to_bits();
                (bits >> 16) as u16
            }).collect();
            Tensor::from_slice(&bf16_data, shape, DType::BF16, device)
        }
        _ => Tensor::from_slice(&data, shape, DType::F32, device),
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Check if two shapes can be broadcast together.
fn broadcast_shapes(a: &Shape, b: &Shape) -> Result<Shape> {
    let a_dims = a.dims();
    let b_dims = b.dims();
    let max_rank = a_dims.len().max(b_dims.len());

    let mut result = vec![0usize; max_rank];

    for i in 0..max_rank {
        let a_dim = if i < max_rank - a_dims.len() {
            1
        } else {
            a_dims[i - (max_rank - a_dims.len())]
        };
        let b_dim = if i < max_rank - b_dims.len() {
            1
        } else {
            b_dims[i - (max_rank - b_dims.len())]
        };

        if a_dim == b_dim {
            result[i] = a_dim;
        } else if a_dim == 1 {
            result[i] = b_dim;
        } else if b_dim == 1 {
            result[i] = a_dim;
        } else {
            return Err(Error::shape_mismatch(
                format!("{:?}", a_dims),
                format!("{:?}", b_dims),
            ));
        }
    }

    Ok(Shape::new(result))
}

/// Get linear index from multi-dimensional indices with broadcasting.
fn broadcast_index(indices: &[usize], shape: &[usize], full_shape: &[usize]) -> usize {
    debug_assert!(
        full_shape.len() >= shape.len(),
        "full_shape rank must be >= shape rank"
    );
    let offset = full_shape.len() - shape.len();
    let mut idx = 0;
    let mut stride = 1;

    for i in (0..shape.len()).rev() {
        let dim_idx = if i + offset < indices.len() {
            indices[i + offset]
        } else {
            0
        };
        // Handle broadcasting: if dim is 1, always use index 0
        let actual_idx = if shape[i] == 1 { 0 } else { dim_idx };
        idx += actual_idx * stride;
        stride *= shape[i];
    }
    idx
}

/// Iterate over all indices of a shape.
///
/// Returns a reference to the current indices on each call to avoid
/// allocating a new Vec per iteration. Callers that need the indices
/// to outlive the borrow should copy them.
struct ShapeIterator {
    shape: Vec<usize>,
    current: Vec<usize>,
    done: bool,
    /// True before the first call to `advance()`.
    first: bool,
}

impl ShapeIterator {
    fn new(shape: &[usize]) -> Self {
        let done = shape.iter().any(|&d| d == 0);
        Self {
            shape: shape.to_vec(),
            current: vec![0; shape.len()],
            done,
            first: true,
        }
    }

    /// Access the current indices without allocation.
    fn current_indices(&self) -> &[usize] {
        &self.current
    }

    /// Advance to the next index set. Returns `false` when exhausted.
    fn advance(&mut self) -> bool {
        if self.done {
            return false;
        }

        // On the very first call, (0,0,...,0) is already valid — just emit it.
        if self.first {
            self.first = false;
            return true;
        }

        // Increment indices (odometer-style, last dimension fastest).
        for i in (0..self.shape.len()).rev() {
            self.current[i] += 1;
            if self.current[i] < self.shape[i] {
                return true;
            }
            self.current[i] = 0;
            if i == 0 {
                self.done = true;
                return false;
            }
        }
        false
    }
}

/// Compatibility: also implement Iterator so existing `for indices in ShapeIterator` still works.
impl Iterator for ShapeIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.advance() {
            Some(self.current.clone())
        } else {
            None
        }
    }
}

// ============================================================================
// Matrix multiplication
// ============================================================================

/// Matrix multiplication.
///
/// Computes C = A @ B where:
/// - A has shape [..., M, K]
/// - B has shape [..., K, N]
/// - C has shape [..., M, N]
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Validate shapes
    if a.rank() < 2 || b.rank() < 2 {
        return Err(Error::shape_mismatch(
            "at least 2D tensors",
            format!("{}D and {}D", a.rank(), b.rank()),
        ));
    }

    let a_shape = a.shape();
    let b_shape = b.shape();
    let a_dims = a_shape.dims();
    let b_dims = b_shape.dims();

    let m = a_dims[a_dims.len() - 2];
    let k = a_dims[a_dims.len() - 1];
    let k2 = b_dims[b_dims.len() - 2];
    let n = b_dims[b_dims.len() - 1];

    if k != k2 {
        return Err(Error::shape_mismatch(
            format!("inner dimensions to match ({} vs {})", k, k2),
            "mismatched dimensions".to_string(),
        ));
    }

    // Calculate output shape with batch dimensions
    let a_batch: Vec<usize> = a_dims[..a_dims.len() - 2].to_vec();
    let b_batch: Vec<usize> = b_dims[..b_dims.len() - 2].to_vec();

    // Broadcast batch dimensions
    let batch_shape = if a_batch.is_empty() && b_batch.is_empty() {
        vec![]
    } else {
        let a_batch_shape = Shape::new(if a_batch.is_empty() {
            vec![1]
        } else {
            a_batch.clone()
        });
        let b_batch_shape = Shape::new(if b_batch.is_empty() {
            vec![1]
        } else {
            b_batch.clone()
        });
        let broadcast = broadcast_shapes(&a_batch_shape, &b_batch_shape)?;
        broadcast.dims().to_vec()
    };

    let mut out_dims = batch_shape.clone();
    out_dims.push(m);
    out_dims.push(n);

    // Dispatch based on device
    match a.device().device_type {
        DeviceType::Cpu => matmul_cpu(a, b, &out_dims, m, k, n, &batch_shape),
        #[cfg(feature = "metal")]
        DeviceType::Metal => matmul_metal(a, b, &out_dims, m, k, n, &batch_shape),
        _ => Err(Error::unsupported(format!(
            "matmul not implemented for {:?}",
            a.device().device_type
        ))),
    }
}

// Apple Accelerate BLAS for AMX-accelerated matmul on Apple Silicon.
#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32,
        a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32,
        c: *mut f32, ldc: i32,
    );
}

#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(target_os = "macos")]
const CBLAS_NO_TRANS: i32 = 111;
#[cfg(target_os = "macos")]
const CBLAS_TRANS: i32 = 112;

/// AMX-accelerated f32 matmul: C = A @ B (row-major).
///
/// A: [m, k], B: [k, n], C: [m, n].
/// On non-macOS, falls back to a triple-nested loop.
pub fn sgemm_cpu(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    #[cfg(target_os = "macos")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0,
            a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            0.0,
            c.as_mut_ptr(), n as i32,
        );
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
}

/// AMX-accelerated f32 matmul: C = A^T @ B (row-major).
///
/// A: [k, m] (stored row-major), B: [k, n], C: [m, n].
/// On non-macOS, falls back to a triple-nested loop.
pub fn sgemm_transa_cpu(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    #[cfg(target_os = "macos")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0,
            a.as_ptr(), m as i32,
            b.as_ptr(), n as i32,
            0.0,
            c.as_mut_ptr(), n as i32,
        );
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[l * m + i] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
}

/// AMX-accelerated f32 linear: C = A @ B^T (row-major), equivalent to Y = X @ W^T.
///
/// A: [m, k], B: [n, k] (weight, row-major), C: [m, n].
/// On non-macOS, falls back to a triple-nested loop.
pub fn sgemm_transb_cpu(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    #[cfg(target_os = "macos")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
            m as i32, n as i32, k as i32,
            1.0,
            a.as_ptr(), k as i32,
            b.as_ptr(), k as i32,
            0.0,
            c.as_mut_ptr(), n as i32,
        );
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[j * k + l];
                }
                c[i * n + j] = sum;
            }
        }
    }
}

/// AMX-accelerated linear with bias: output[i,j] = sum_k(input[i,k] * weight[j,k]) + bias[j].
///
/// input: [m, k], weight: [n, k], bias: [n], output: [m, n].
pub fn linear_amx(
    input: &[f32], weight: &[f32], bias: &[f32], output: &mut [f32],
    m: usize, k: usize, n: usize,
) {
    sgemm_transb_cpu(input, weight, output, m, n, k);
    for i in 0..m {
        for j in 0..n {
            output[i * n + j] += bias[j];
        }
    }
}

/// CPU matmul using Apple Accelerate BLAS (AMX-accelerated on M-series).
fn matmul_cpu(
    a: &Tensor,
    b: &Tensor,
    out_dims: &[usize],
    m: usize,
    k: usize,
    n: usize,
    batch_shape: &[usize],
) -> Result<Tensor> {
    let a_data: Vec<f32> = a.to_f32_vec()?;
    let b_data: Vec<f32> = b.to_f32_vec()?;

    let batch_size: usize = batch_shape.iter().product::<usize>().max(1);
    let mut out_data = vec![0.0f32; batch_size * m * n];

    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    let out_batch_stride = m * n;

    for batch in 0..batch_size {
        let a_offset = batch * a_batch_stride;
        let b_offset = batch * b_batch_stride;
        let out_offset = batch * out_batch_stride;

        #[cfg(target_os = "macos")]
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                m as i32, n as i32, k as i32,
                1.0,
                a_data[a_offset..].as_ptr(), k as i32,
                b_data[b_offset..].as_ptr(), n as i32,
                0.0,
                out_data[out_offset..].as_mut_ptr(), n as i32,
            );
        }

        #[cfg(not(target_os = "macos"))]
        {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        sum += a_data[a_offset + i * k + l] * b_data[b_offset + l * n + j];
                    }
                    out_data[out_offset + i * n + j] = sum;
                }
            }
        }
    }

    tensor_from_f32_as(out_data, Shape::new(out_dims.to_vec()), a.dtype(), a.device())
}

#[cfg(feature = "metal")]
fn matmul_metal(
    a: &Tensor,
    b: &Tensor,
    out_dims: &[usize],
    m: usize,
    k: usize,
    n: usize,
    batch_shape: &[usize],
) -> Result<Tensor> {
    // Only dispatch to Metal for non-batched matmul with supported dtypes.
    // Batched matmul would need per-batch offset handling in the kernel.
    let batch_size: usize = batch_shape.iter().product::<usize>().max(1);
    if batch_size > 1 {
        // Batched matmul routes to CPU/AMX — single dispatch per batch via cblas_sgemm.
        return matmul_cpu(a, b, out_dims, m, k, n, batch_shape);
    }

    let dtype = a.dtype();
    if dtype != DType::F16 && dtype != DType::F32 {
        // Non-F16/F32 dtypes route to CPU/AMX (rare: only for integer/bf16 tensors).
        return matmul_cpu(a, b, out_dims, m, k, n, batch_shape);
    }

    let ops = get_metal_ops()?;
    let metal_dev = crate::tensor::storage::get_metal_device()?;

    let a_buf = borrow_metal_buffer(a)?;
    let b_buf = borrow_metal_buffer(b)?;

    let out_shape = Shape::new(out_dims.to_vec());
    let out_tensor = alloc_metal_output(out_shape, dtype, a.device())?;
    let c_buf = borrow_metal_buffer(&out_tensor)?;

    ops.matmul(
        a_buf.as_ref(),
        b_buf.as_ref(),
        c_buf.as_ref(),
        m,
        n,
        k,
        dtype,
        metal_dev.as_ref(),
    )?;

    Ok(out_tensor)
}

/// Batch matrix multiplication.
pub fn bmm(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.rank() != 3 || b.rank() != 3 {
        return Err(Error::shape_mismatch("3D tensors", "non-3D tensors"));
    }

    matmul(a, b)
}

// ============================================================================
// Element-wise operations
// ============================================================================

/// Element-wise addition with broadcasting.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if a.device().device_type == DeviceType::Metal {
        if let Some(result) = try_metal_binary_add(a, b)? {
            return Ok(result);
        }
    }
    binary_op(a, b, |x, y| x + y)
}

/// Element-wise subtraction with broadcasting.
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(a, b, |x, y| x - y)
}

/// Element-wise multiplication with broadcasting.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if a.device().device_type == DeviceType::Metal {
        if let Some(result) = try_metal_binary_mul(a, b)? {
            return Ok(result);
        }
    }
    binary_op(a, b, |x, y| x * y)
}

/// Element-wise division with broadcasting.
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(a, b, |x, y| x / y)
}

/// Scalar multiplication.
pub fn mul_scalar(a: &Tensor, scalar: f32) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if a.device().device_type == DeviceType::Metal {
        if let Some(result) = try_metal_scale(a, scalar)? {
            return Ok(result);
        }
    }
    unary_op(a, |x| x * scalar)
}

/// Scalar addition.
pub fn add_scalar(a: &Tensor, scalar: f32) -> Result<Tensor> {
    unary_op(a, |x| x + scalar)
}

/// Try Metal-accelerated element-wise add (same-shape F16 only).
/// Returns `Ok(None)` if conditions are not met so caller falls back to CPU.
#[cfg(feature = "metal")]
fn try_metal_binary_add(a: &Tensor, b: &Tensor) -> Result<Option<Tensor>> {
    if a.dtype() != DType::F16 || a.shape() != b.shape() {
        return Ok(None);
    }
    let ops = get_metal_ops()?;
    let dev = crate::tensor::storage::get_metal_device()?;

    let a_buf = borrow_metal_buffer(a)?;
    let b_buf = borrow_metal_buffer(b)?;

    let out = alloc_metal_output(a.shape().clone(), a.dtype(), a.device())?;
    let c_buf = borrow_metal_buffer(&out)?;

    ops.add(a_buf.as_ref(), b_buf.as_ref(), c_buf.as_ref(), a.numel(), dev.as_ref())?;
    Ok(Some(out))
}

/// Try Metal-accelerated element-wise mul (same-shape F16 only).
#[cfg(feature = "metal")]
fn try_metal_binary_mul(a: &Tensor, b: &Tensor) -> Result<Option<Tensor>> {
    if a.dtype() != DType::F16 || a.shape() != b.shape() {
        return Ok(None);
    }
    let ops = get_metal_ops()?;
    let dev = crate::tensor::storage::get_metal_device()?;

    let a_buf = borrow_metal_buffer(a)?;
    let b_buf = borrow_metal_buffer(b)?;

    let out = alloc_metal_output(a.shape().clone(), a.dtype(), a.device())?;
    let c_buf = borrow_metal_buffer(&out)?;

    ops.mul(a_buf.as_ref(), b_buf.as_ref(), c_buf.as_ref(), a.numel(), dev.as_ref())?;
    Ok(Some(out))
}

/// Try Metal-accelerated scalar scale (F16 only).
#[cfg(feature = "metal")]
fn try_metal_scale(a: &Tensor, scalar: f32) -> Result<Option<Tensor>> {
    if a.dtype() != DType::F16 {
        return Ok(None);
    }
    let ops = get_metal_ops()?;
    let dev = crate::tensor::storage::get_metal_device()?;

    let a_buf = borrow_metal_buffer(a)?;

    let out = alloc_metal_output(a.shape().clone(), a.dtype(), a.device())?;
    let o_buf = borrow_metal_buffer(&out)?;

    ops.scale(a_buf.as_ref(), o_buf.as_ref(), scalar, a.numel(), dev.as_ref())?;
    Ok(Some(out))
}

/// Generic binary operation with broadcasting.
fn binary_op<F>(a: &Tensor, b: &Tensor, op: F) -> Result<Tensor>
where
    F: Fn(f32, f32) -> f32,
{
    let out_shape = broadcast_shapes(a.shape(), b.shape())?;

    match a.device().device_type {
        DeviceType::Cpu => binary_op_cpu(a, b, &out_shape, op),
        #[cfg(feature = "metal")]
        // Generic binary closures fall back to CPU. Specific ops (add, mul)
        // dispatch to Metal before reaching this function.
        DeviceType::Metal => binary_op_cpu(a, b, &out_shape, op),
        _ => Err(Error::unsupported(format!(
            "binary op not implemented for {:?}",
            a.device().device_type
        ))),
    }
}

fn binary_op_cpu<F>(a: &Tensor, b: &Tensor, out_shape: &Shape, op: F) -> Result<Tensor>
where
    F: Fn(f32, f32) -> f32,
{
    let a_data: Vec<f32> = a.to_f32_vec()?;
    let b_data: Vec<f32> = b.to_f32_vec()?;

    let out_dims = out_shape.dims();
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();

    let mut out_data = Vec::with_capacity(out_shape.numel());

    let mut iter = ShapeIterator::new(out_dims);
    while iter.advance() {
        let indices = iter.current_indices();
        let a_idx = broadcast_index(indices, a_dims, out_dims);
        let b_idx = broadcast_index(indices, b_dims, out_dims);

        let a_val = a_data.get(a_idx).copied().unwrap_or(0.0);
        let b_val = b_data.get(b_idx).copied().unwrap_or(0.0);

        out_data.push(op(a_val, b_val));
    }

    tensor_from_f32_as(out_data, out_shape.clone(), a.dtype(), a.device())
}

/// Generic unary operation.
fn unary_op<F>(a: &Tensor, op: F) -> Result<Tensor>
where
    F: Fn(f32) -> f32,
{
    match a.device().device_type {
        DeviceType::Cpu => unary_op_cpu(a, op),
        #[cfg(feature = "metal")]
        // Generic unary closures fall back to CPU. Named activations
        // (silu, etc.) dispatch to Metal before reaching this function.
        DeviceType::Metal => unary_op_cpu(a, op),
        _ => Err(Error::unsupported(format!(
            "unary op not implemented for {:?}",
            a.device().device_type
        ))),
    }
}

fn unary_op_cpu<F>(a: &Tensor, op: F) -> Result<Tensor>
where
    F: Fn(f32) -> f32,
{
    let data: Vec<f32> = a.to_f32_vec()?;
    let out_data: Vec<f32> = data.iter().map(|&x| op(x)).collect();
    tensor_from_f32_as(out_data, a.shape().clone(), a.dtype(), a.device())
}

// ============================================================================
// Activation functions
// ============================================================================

/// GELU activation (Gaussian Error Linear Unit).
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(input: &Tensor) -> Result<Tensor> {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const COEFF: f32 = 0.044715;

    unary_op(input, |x| {
        let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    })
}

/// SiLU (Swish) activation.
/// silu(x) = x * sigmoid(x)
pub fn silu(input: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if input.device().device_type == DeviceType::Metal {
        let dtype = input.dtype();
        if dtype == DType::F16 || dtype == DType::F32 {
            return silu_metal(input);
        }
    }
    unary_op(input, |x| x * (1.0 / (1.0 + (-x).exp())))
}

#[cfg(feature = "metal")]
fn silu_metal(input: &Tensor) -> Result<Tensor> {
    let ops = get_metal_ops()?;
    let dev = crate::tensor::storage::get_metal_device()?;

    let in_buf = borrow_metal_buffer(input)?;
    let out = alloc_metal_output(input.shape().clone(), input.dtype(), input.device())?;
    let out_buf = borrow_metal_buffer(&out)?;

    ops.silu(
        in_buf.as_ref(),
        out_buf.as_ref(),
        input.numel(),
        input.dtype(),
        dev.as_ref(),
    )?;

    Ok(out)
}

/// ReLU activation.
/// relu(x) = max(0, x)
pub fn relu(input: &Tensor) -> Result<Tensor> {
    unary_op(input, |x| x.max(0.0))
}

/// Leaky ReLU activation.
/// leaky_relu(x) = max(alpha * x, x)
pub fn leaky_relu(input: &Tensor, alpha: f32) -> Result<Tensor> {
    unary_op(input, |x| if x > 0.0 { x } else { alpha * x })
}

/// Sigmoid activation.
/// sigmoid(x) = 1 / (1 + exp(-x))
pub fn sigmoid(input: &Tensor) -> Result<Tensor> {
    unary_op(input, |x| 1.0 / (1.0 + (-x).exp()))
}

/// Tanh activation.
pub fn tanh(input: &Tensor) -> Result<Tensor> {
    unary_op(input, |x| x.tanh())
}

// ============================================================================
// Normalization
// ============================================================================

/// Softmax along a dimension.
pub fn softmax(input: &Tensor, dim: i64) -> Result<Tensor> {
    let rank = input.rank();
    let dim = if dim < 0 {
        (rank as i64 + dim) as usize
    } else {
        dim as usize
    };

    if dim >= rank {
        return Err(Error::InvalidArgument {
            name: "dim".into(),
            message: format!("dimension {} out of range for rank {}", dim, rank),
        });
    }

    match input.device().device_type {
        DeviceType::Cpu => softmax_cpu(input, dim),
        #[cfg(feature = "metal")]
        DeviceType::Metal => softmax_metal(input, dim),
        _ => Err(Error::unsupported(format!(
            "softmax not implemented for {:?}",
            input.device().device_type
        ))),
    }
}

fn softmax_cpu(input: &Tensor, dim: usize) -> Result<Tensor> {
    let data: Vec<f32> = input.to_f32_vec()?;
    let shape = input.shape();
    let dims = shape.dims();

    let mut out_data = vec![0.0f32; data.len()];

    // Calculate strides
    let mut strides = vec![1usize; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    let dim_size = dims[dim];
    let dim_stride = strides[dim];

    // Number of softmax operations to perform
    let outer_size: usize = dims[..dim].iter().product();
    let inner_size: usize = dims[dim + 1..].iter().product();
    let outer_size = outer_size.max(1);
    let inner_size = inner_size.max(1);

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..dim_size {
                let idx = outer * strides.get(dim.wrapping_sub(1)).copied().unwrap_or(dim_size * inner_size)
                    + i * dim_stride
                    + inner;
                if idx < data.len() {
                    max_val = max_val.max(data[idx]);
                }
            }

            // Compute exp and sum
            let mut sum = 0.0f32;
            for i in 0..dim_size {
                let idx = outer * strides.get(dim.wrapping_sub(1)).copied().unwrap_or(dim_size * inner_size)
                    + i * dim_stride
                    + inner;
                if idx < data.len() {
                    let exp_val = (data[idx] - max_val).exp();
                    out_data[idx] = exp_val;
                    sum += exp_val;
                }
            }

            // Normalize
            for i in 0..dim_size {
                let idx = outer * strides.get(dim.wrapping_sub(1)).copied().unwrap_or(dim_size * inner_size)
                    + i * dim_stride
                    + inner;
                if idx < out_data.len() {
                    out_data[idx] /= sum;
                }
            }
        }
    }

    tensor_from_f32_as(out_data, shape.clone(), input.dtype(), input.device())
}

/// Metal-accelerated softmax along the last dimension (F16 only).
/// Falls back to CPU for non-F16 dtypes or when softmax is not along the
/// last dimension.
#[cfg(feature = "metal")]
fn softmax_metal(input: &Tensor, dim: usize) -> Result<Tensor> {
    let rank = input.rank();
    let last_dim = rank.saturating_sub(1);

    // The Metal kernel only supports softmax along the last dimension and F16.
    if dim != last_dim || input.dtype() != DType::F16 {
        return softmax_cpu(input, dim);
    }

    let dims = input.shape().dims();
    let softmax_dim = dims[last_dim];
    let batch_size: usize = dims[..last_dim].iter().product::<usize>().max(1);

    let ops = get_metal_ops()?;
    let dev = crate::tensor::storage::get_metal_device()?;

    let in_buf = borrow_metal_buffer(input)?;
    let out = alloc_metal_output(input.shape().clone(), input.dtype(), input.device())?;
    let out_buf = borrow_metal_buffer(&out)?;

    ops.softmax(
        in_buf.as_ref(),
        out_buf.as_ref(),
        batch_size,
        softmax_dim,
        dev.as_ref(),
    )?;

    Ok(out)
}

/// Layer normalization.
pub fn layer_norm(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    match input.device().device_type {
        DeviceType::Cpu => layer_norm_cpu(input, normalized_shape, weight, bias, eps),
        #[cfg(feature = "metal")]
        // No dedicated Metal layer_norm kernel yet; fall back to CPU.
        DeviceType::Metal => layer_norm_cpu(input, normalized_shape, weight, bias, eps),
        _ => Err(Error::unsupported(format!(
            "layer_norm not implemented for {:?}",
            input.device().device_type
        ))),
    }
}

fn layer_norm_cpu(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let data: Vec<f32> = input.to_f32_vec()?;
    let shape = input.shape();
    let dims = shape.dims();

    // Number of elements to normalize over
    let norm_size: usize = normalized_shape.iter().product();
    let batch_size = data.len() / norm_size;

    let weight_data: Option<Vec<f32>> = weight.map(|w| w.to_f32_vec()).transpose()?;
    let bias_data: Option<Vec<f32>> = bias.map(|b| b.to_f32_vec()).transpose()?;

    let mut out_data = vec![0.0f32; data.len()];

    for b in 0..batch_size {
        let offset = b * norm_size;
        let slice = &data[offset..offset + norm_size];

        // Compute mean
        let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;

        // Compute variance
        let var: f32 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / norm_size as f32;

        // Normalize
        let inv_std = 1.0 / (var + eps).sqrt();

        for i in 0..norm_size {
            let normalized = (slice[i] - mean) * inv_std;
            let scaled = match &weight_data {
                Some(w) => normalized * w.get(i).copied().unwrap_or(1.0),
                None => normalized,
            };
            let shifted = match &bias_data {
                Some(b) => scaled + b.get(i).copied().unwrap_or(0.0),
                None => scaled,
            };
            out_data[offset + i] = shifted;
        }
    }

    tensor_from_f32_as(out_data, shape.clone(), input.dtype(), input.device())
}

/// RMS normalization (used in LLaMA, etc.).
pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    match input.device().device_type {
        DeviceType::Cpu => rms_norm_cpu(input, weight, eps),
        #[cfg(feature = "metal")]
        DeviceType::Metal => rms_norm_metal(input, weight, eps),
        _ => Err(Error::unsupported(format!(
            "rms_norm not implemented for {:?}",
            input.device().device_type
        ))),
    }
}

#[cfg(feature = "metal")]
fn rms_norm_metal(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    if input.dtype() != DType::F16 {
        return rms_norm_cpu(input, weight, eps);
    }

    let dims = input.shape().dims();
    let norm_size = dims.last().copied().unwrap_or(1);
    let batch_size = input.numel() / norm_size;

    let ops = get_metal_ops()?;
    let dev = crate::tensor::storage::get_metal_device()?;

    let in_buf = borrow_metal_buffer(input)?;
    let w_buf = borrow_metal_buffer(weight)?;
    let out = alloc_metal_output(input.shape().clone(), input.dtype(), input.device())?;
    let out_buf = borrow_metal_buffer(&out)?;

    ops.rms_norm(
        in_buf.as_ref(),
        w_buf.as_ref(),
        out_buf.as_ref(),
        batch_size,
        norm_size,
        eps,
        dev.as_ref(),
    )?;

    Ok(out)
}

fn rms_norm_cpu(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let data: Vec<f32> = input.to_f32_vec()?;
    let weight_data: Vec<f32> = weight.to_f32_vec()?;
    let shape = input.shape();
    let dims = shape.dims();

    // Normalize over the last dimension
    let norm_size = dims.last().copied().unwrap_or(1);
    let batch_size = data.len() / norm_size;

    let mut out_data = vec![0.0f32; data.len()];

    for b in 0..batch_size {
        let offset = b * norm_size;
        let slice = &data[offset..offset + norm_size];

        // Compute RMS
        let rms: f32 = (slice.iter().map(|&x| x * x).sum::<f32>() / norm_size as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Normalize and scale
        for i in 0..norm_size {
            let w = weight_data.get(i).copied().unwrap_or(1.0);
            out_data[offset + i] = slice[i] * inv_rms * w;
        }
    }

    tensor_from_f32_as(out_data, shape.clone(), input.dtype(), input.device())
}

/// Group normalization.
pub fn group_norm(
    input: &Tensor,
    num_groups: usize,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    match input.device().device_type {
        DeviceType::Cpu => group_norm_cpu(input, num_groups, weight, bias, eps),
        #[cfg(feature = "metal")]
        // No dedicated Metal group_norm kernel yet; fall back to CPU.
        DeviceType::Metal => group_norm_cpu(input, num_groups, weight, bias, eps),
        _ => Err(Error::unsupported(format!(
            "group_norm not implemented for {:?}",
            input.device().device_type
        ))),
    }
}

fn group_norm_cpu(
    input: &Tensor,
    num_groups: usize,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let data: Vec<f32> = input.to_f32_vec()?;
    let shape = input.shape();
    let dims = shape.dims();

    // Input shape: [N, C, ...] or [N, C, H, W]
    if dims.len() < 2 {
        return Err(Error::shape_mismatch(
            "at least 2D tensor [N, C, ...]",
            format!("{}D tensor", dims.len()),
        ));
    }

    let batch = dims[0];
    let channels = dims[1];
    let spatial: usize = dims[2..].iter().product();
    let spatial = spatial.max(1);

    if channels % num_groups != 0 {
        return Err(Error::InvalidArgument {
            name: "num_groups".into(),
            message: format!(
                "num_groups {} must divide num_channels {}",
                num_groups, channels
            ),
        });
    }

    let channels_per_group = channels / num_groups;
    let group_size = channels_per_group * spatial;

    let weight_data: Option<Vec<f32>> = weight.map(|w| w.to_f32_vec()).transpose()?;
    let bias_data: Option<Vec<f32>> = bias.map(|b| b.to_f32_vec()).transpose()?;

    let mut out_data = vec![0.0f32; data.len()];

    for n in 0..batch {
        for g in 0..num_groups {
            // Calculate mean and variance for this group
            let mut sum = 0.0f32;
            let mut sq_sum = 0.0f32;

            for c in 0..channels_per_group {
                let channel = g * channels_per_group + c;
                for s in 0..spatial {
                    let idx = n * channels * spatial + channel * spatial + s;
                    if idx < data.len() {
                        let val = data[idx];
                        sum += val;
                        sq_sum += val * val;
                    }
                }
            }

            let mean = sum / group_size as f32;
            let var = sq_sum / group_size as f32 - mean * mean;
            let inv_std = 1.0 / (var + eps).sqrt();

            // Normalize
            for c in 0..channels_per_group {
                let channel = g * channels_per_group + c;
                let w = weight_data
                    .as_ref()
                    .and_then(|w| w.get(channel).copied())
                    .unwrap_or(1.0);
                let b = bias_data
                    .as_ref()
                    .and_then(|b| b.get(channel).copied())
                    .unwrap_or(0.0);

                for s in 0..spatial {
                    let idx = n * channels * spatial + channel * spatial + s;
                    if idx < data.len() {
                        out_data[idx] = (data[idx] - mean) * inv_std * w + b;
                    }
                }
            }
        }
    }

    tensor_from_f32_as(out_data, shape.clone(), input.dtype(), input.device())
}

// ============================================================================
// Attention
// ============================================================================

/// Scaled dot-product attention.
///
/// Computes: softmax(Q @ K^T / sqrt(d)) @ V
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
    scale: Option<f32>,
) -> Result<Tensor> {
    // Validate shapes
    if query.rank() < 2 || key.rank() < 2 || value.rank() < 2 {
        return Err(Error::shape_mismatch(
            "at least 2D tensors for Q, K, V",
            format!(
                "{}D, {}D, {}D",
                query.rank(),
                key.rank(),
                value.rank()
            ),
        ));
    }

    match query.device().device_type {
        DeviceType::Cpu => sdpa_cpu(query, key, value, mask, scale),
        #[cfg(feature = "metal")]
        DeviceType::Metal => sdpa_metal(query, key, value, mask, scale),
        _ => Err(Error::unsupported(format!(
            "scaled_dot_product_attention not implemented for {:?}",
            query.device().device_type
        ))),
    }
}

/// Metal-accelerated fused attention.
///
/// The Metal kernel supports the common case of 4-D [batch, heads, seq, dim]
/// F16 tensors with no mask. For anything else we fall back to CPU.
#[cfg(feature = "metal")]
fn sdpa_metal(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
    scale: Option<f32>,
) -> Result<Tensor> {
    // The fused kernel only supports 4-D F16, no mask.
    if query.dtype() != DType::F16 || query.rank() != 4 || mask.is_some() {
        return sdpa_cpu(query, key, value, mask, scale);
    }

    let q_dims = query.shape().dims();
    // q_dims = [batch, num_heads, seq_len, head_dim]
    let _batch = q_dims[0];
    let num_heads = q_dims[1];
    let seq_len = q_dims[2];
    let head_dim = q_dims[3];

    // The Metal attention kernel currently handles batch=1 via num_heads dispatch.
    // For batch > 1 we would need to loop or extend the kernel.
    if _batch > 1 {
        return sdpa_cpu(query, key, value, mask, scale);
    }

    let ops = get_metal_ops()?;
    let dev = crate::tensor::storage::get_metal_device()?;

    let q_buf = borrow_metal_buffer(query)?;
    let k_buf = borrow_metal_buffer(key)?;
    let v_buf = borrow_metal_buffer(value)?;

    let out = alloc_metal_output(query.shape().clone(), query.dtype(), query.device())?;
    let out_buf = borrow_metal_buffer(&out)?;

    ops.attention(
        q_buf.as_ref(),
        k_buf.as_ref(),
        v_buf.as_ref(),
        out_buf.as_ref(),
        seq_len,
        head_dim,
        num_heads,
        dev.as_ref(),
    )?;

    Ok(out)
}

fn sdpa_cpu(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
    scale: Option<f32>,
) -> Result<Tensor> {
    let q_dims = query.shape().dims();
    let k_dims = key.shape().dims();
    let v_dims = value.shape().dims();

    // Get dimensions: [..., seq_len, head_dim]
    let seq_q = q_dims[q_dims.len() - 2];
    let head_dim = q_dims[q_dims.len() - 1];
    let seq_k = k_dims[k_dims.len() - 2];

    // Scale factor
    let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

    // Q @ K^T -> [batch, heads, seq_q, seq_k]
    // First transpose K
    let key_t = transpose(key, -1, -2)?;

    // Compute attention scores
    let scores = matmul(query, &key_t)?;
    let scores = mul_scalar(&scores, scale)?;

    // Apply mask if provided
    let scores = if let Some(m) = mask {
        // Mask should be broadcastable to scores shape
        // Where mask is 0, set to -inf
        let mask_data: Vec<f32> = m.to_f32_vec()?;
        let scores_data: Vec<f32> = scores.to_f32_vec()?;

        let mut masked_scores = scores_data.clone();
        for i in 0..masked_scores.len() {
            let mask_val = mask_data.get(i % mask_data.len()).copied().unwrap_or(1.0);
            if mask_val == 0.0 {
                masked_scores[i] = f32::NEG_INFINITY;
            }
        }
        tensor_from_f32_as(masked_scores, scores.shape().clone(), query.dtype(), query.device())?
    } else {
        scores
    };

    // Softmax over last dimension
    let attn_weights = softmax(&scores, -1)?;

    // Attention @ V
    matmul(&attn_weights, value)
}

/// Transpose two dimensions of a tensor.
pub fn transpose(input: &Tensor, dim0: i64, dim1: i64) -> Result<Tensor> {
    let rank = input.rank();
    let dim0 = if dim0 < 0 {
        (rank as i64 + dim0) as usize
    } else {
        dim0 as usize
    };
    let dim1 = if dim1 < 0 {
        (rank as i64 + dim1) as usize
    } else {
        dim1 as usize
    };

    if dim0 >= rank || dim1 >= rank {
        return Err(Error::InvalidArgument {
            name: "dim".into(),
            message: format!("dimensions out of range for rank {}", rank),
        });
    }

    if dim0 == dim1 {
        // No-op
        return Ok(input.clone());
    }

    let data: Vec<f32> = input.to_f32_vec()?;
    let shape = input.shape();
    let dims = shape.dims();

    // New shape with dimensions swapped
    let mut new_dims = dims.to_vec();
    new_dims.swap(dim0, dim1);

    // Calculate strides
    let mut old_strides = vec![1usize; rank];
    let mut new_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        old_strides[i] = old_strides[i + 1] * dims[i + 1];
        new_strides[i] = new_strides[i + 1] * new_dims[i + 1];
    }

    let mut out_data = vec![0.0f32; data.len()];

    for new_idx in 0..data.len() {
        // Convert linear index to multi-dimensional
        let mut new_coords = vec![0usize; rank];
        let mut remaining = new_idx;
        for i in 0..rank {
            new_coords[i] = remaining / new_strides[i];
            remaining %= new_strides[i];
        }

        // Swap coordinates
        let mut old_coords = new_coords.clone();
        old_coords.swap(dim0, dim1);

        // Convert back to linear index
        let old_idx: usize = old_coords
            .iter()
            .zip(old_strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();

        if old_idx < data.len() {
            out_data[new_idx] = data[old_idx];
        }
    }

    tensor_from_f32_as(out_data, Shape::new(new_dims), input.dtype(), input.device())
}

/// Rotary positional embedding.
pub fn rotary_embedding(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    match input.device().device_type {
        DeviceType::Cpu => rope_cpu(input, cos, sin),
        #[cfg(feature = "metal")]
        DeviceType::Metal => rope_metal(input, cos, sin),
        _ => Err(Error::unsupported(format!(
            "rotary_embedding not implemented for {:?}",
            input.device().device_type
        ))),
    }
}

#[cfg(feature = "metal")]
fn rope_metal(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    if input.dtype() != DType::F16 {
        return rope_cpu(input, cos, sin);
    }

    let dims = input.shape().dims();
    // Input shape: [..., seq_len, head_dim]
    let head_dim = dims.last().copied().unwrap_or(1);
    let seq_len = if dims.len() >= 2 { dims[dims.len() - 2] } else { 1 };

    let ops = get_metal_ops()?;
    let dev = crate::tensor::storage::get_metal_device()?;

    // The Metal RoPE kernel operates in-place on x, so we need to copy
    // input to the output first, then apply RoPE in-place.
    let out = alloc_metal_output(input.shape().clone(), input.dtype(), input.device())?;

    // Copy input data to output buffer via unified memory
    {
        let in_buf = borrow_metal_buffer(input)?;
        let out_buf = borrow_metal_buffer(&out)?;

        // Use a blit encoder to copy
        let cmd = dev.new_command_buffer();
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(
            in_buf.as_ref(),
            0,
            out_buf.as_ref(),
            0,
            input.size_bytes() as u64,
        );
        blit.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let out_buf = borrow_metal_buffer(&out)?;
    let cos_buf = borrow_metal_buffer(cos)?;
    let sin_buf = borrow_metal_buffer(sin)?;

    ops.rope(
        out_buf.as_ref(),
        cos_buf.as_ref(),
        sin_buf.as_ref(),
        seq_len,
        head_dim,
        dev.as_ref(),
    )?;

    Ok(out)
}

fn rope_cpu(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let data: Vec<f32> = input.to_f32_vec()?;
    let cos_data: Vec<f32> = cos.to_f32_vec()?;
    let sin_data: Vec<f32> = sin.to_f32_vec()?;

    let shape = input.shape();
    let dims = shape.dims();

    // Assume input is [..., seq_len, head_dim]
    let head_dim = dims.last().copied().unwrap_or(1);
    let half_dim = head_dim / 2;

    let mut out_data = vec![0.0f32; data.len()];

    // Apply rotary embedding: split into two halves, rotate
    for i in (0..data.len()).step_by(head_dim) {
        let pos_idx = (i / head_dim) % cos_data.len().max(1);

        for j in 0..half_dim {
            let cos_val = cos_data.get(pos_idx * half_dim + j).copied().unwrap_or(1.0);
            let sin_val = sin_data.get(pos_idx * half_dim + j).copied().unwrap_or(0.0);

            let x0 = data.get(i + j).copied().unwrap_or(0.0);
            let x1 = data.get(i + j + half_dim).copied().unwrap_or(0.0);

            // Rotate
            out_data[i + j] = x0 * cos_val - x1 * sin_val;
            out_data[i + j + half_dim] = x0 * sin_val + x1 * cos_val;
        }
    }

    tensor_from_f32_as(out_data, shape.clone(), input.dtype(), input.device())
}

// ============================================================================
// Convolution
// ============================================================================

/// Convolution 2D.
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> Result<Tensor> {
    match input.device().device_type {
        DeviceType::Cpu => conv2d_cpu(input, weight, bias, stride, padding),
        #[cfg(feature = "metal")]
        // No dedicated Metal conv2d kernel yet; fall back to CPU.
        DeviceType::Metal => conv2d_cpu(input, weight, bias, stride, padding),
        _ => Err(Error::unsupported(format!(
            "conv2d not implemented for {:?}",
            input.device().device_type
        ))),
    }
}

fn conv2d_cpu(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> Result<Tensor> {
    let input_data: Vec<f32> = input.to_f32_vec()?;
    let weight_data: Vec<f32> = weight.to_f32_vec()?;
    let bias_data: Option<Vec<f32>> = bias.map(|b| b.to_f32_vec()).transpose()?;

    let i_dims = input.shape().dims();
    let w_dims = weight.shape().dims();

    // Input: [N, C_in, H, W]
    // Weight: [C_out, C_in, kH, kW]
    if i_dims.len() != 4 || w_dims.len() != 4 {
        return Err(Error::shape_mismatch(
            "4D tensors [N, C, H, W]",
            format!("{}D and {}D", i_dims.len(), w_dims.len()),
        ));
    }

    let (batch, c_in, h_in, w_in) = (i_dims[0], i_dims[1], i_dims[2], i_dims[3]);
    let (c_out, c_in_w, kh, kw) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);

    if c_in != c_in_w {
        return Err(Error::shape_mismatch(
            format!("input channels {} to match weight channels {}", c_in, c_in_w),
            "channel mismatch".to_string(),
        ));
    }

    let h_out = (h_in + 2 * padding[0] - kh) / stride[0] + 1;
    let w_out = (w_in + 2 * padding[1] - kw) / stride[1] + 1;

    let mut out_data = vec![0.0f32; batch * c_out * h_out * w_out];

    for n in 0..batch {
        for oc in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0f32;

                    for ic in 0..c_in {
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * stride[0] + khi;
                                let iw = ow * stride[1] + kwi;

                                // Handle padding
                                let ih_real = ih as isize - padding[0] as isize;
                                let iw_real = iw as isize - padding[1] as isize;

                                if ih_real >= 0
                                    && ih_real < h_in as isize
                                    && iw_real >= 0
                                    && iw_real < w_in as isize
                                {
                                    let i_idx = n * c_in * h_in * w_in
                                        + ic * h_in * w_in
                                        + (ih_real as usize) * w_in
                                        + iw_real as usize;
                                    let w_idx =
                                        oc * c_in * kh * kw + ic * kh * kw + khi * kw + kwi;

                                    if i_idx < input_data.len() && w_idx < weight_data.len() {
                                        sum += input_data[i_idx] * weight_data[w_idx];
                                    }
                                }
                            }
                        }
                    }

                    // Add bias
                    if let Some(ref b) = bias_data {
                        sum += b.get(oc).copied().unwrap_or(0.0);
                    }

                    let o_idx = n * c_out * h_out * w_out + oc * h_out * w_out + oh * w_out + ow;
                    out_data[o_idx] = sum;
                }
            }
        }
    }

    tensor_from_f32_as(
        out_data,
        Shape::new(vec![batch, c_out, h_out, w_out]),
        input.dtype(),
        input.device(),
    )
}

// ============================================================================
// Embedding
// ============================================================================

/// Embedding lookup.
pub fn embedding(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    match input.device().device_type {
        DeviceType::Cpu => embedding_cpu(input, weight),
        #[cfg(feature = "metal")]
        // No dedicated Metal embedding kernel yet; fall back to CPU.
        DeviceType::Metal => embedding_cpu(input, weight),
        _ => Err(Error::unsupported(format!(
            "embedding not implemented for {:?}",
            input.device().device_type
        ))),
    }
}

fn embedding_cpu(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    // Input: indices tensor of shape [...]
    // Weight: embedding table of shape [vocab_size, embed_dim]
    // Output: shape [..., embed_dim]

    let indices: Vec<f32> = input.to_f32_vec()?;
    let weight_data: Vec<f32> = weight.to_f32_vec()?;

    let w_dims = weight.shape().dims();
    if w_dims.len() != 2 {
        return Err(Error::shape_mismatch(
            "2D weight tensor [vocab_size, embed_dim]",
            format!("{}D tensor", w_dims.len()),
        ));
    }

    let embed_dim = w_dims[1];

    let mut out_shape = input.shape().dims().to_vec();
    out_shape.push(embed_dim);

    let mut out_data = Vec::with_capacity(indices.len() * embed_dim);

    for &idx in &indices {
        let idx = idx as usize;
        for e in 0..embed_dim {
            let w_idx = idx * embed_dim + e;
            out_data.push(weight_data.get(w_idx).copied().unwrap_or(0.0));
        }
    }

    tensor_from_f32_as(out_data, Shape::new(out_shape), weight.dtype(), input.device())
}

// ============================================================================
// Tensor manipulation
// ============================================================================

/// Concatenate tensors along a dimension.
pub fn cat(tensors: &[&Tensor], dim: i64) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(Error::InvalidArgument {
            name: "tensors".into(),
            message: "cannot concatenate empty list".into(),
        });
    }

    let first = tensors[0];
    let rank = first.rank();

    // Normalize dim
    let dim = if dim < 0 {
        (rank as i64 + dim) as usize
    } else {
        dim as usize
    };

    if dim >= rank {
        return Err(Error::InvalidArgument {
            name: "dim".into(),
            message: format!("dimension {} out of range for rank {}", dim, rank),
        });
    }

    // Validate shapes and calculate output shape
    let mut new_dims: Vec<usize> = first.shape().dims().to_vec();
    new_dims[dim] = 0;

    for (i, t) in tensors.iter().enumerate() {
        if t.rank() != rank {
            return Err(Error::shape_mismatch(
                format!("rank {}", rank),
                format!("rank {} at index {}", t.rank(), i),
            ));
        }

        for (d, &size) in t.shape().dims().iter().enumerate() {
            if d == dim {
                new_dims[dim] += size;
            } else if i > 0 && size != first.shape().dims()[d] {
                return Err(Error::shape_mismatch(
                    format!("dim {} size {}", d, first.shape().dims()[d]),
                    format!("size {} at index {}", size, i),
                ));
            }
        }
    }

    // Actually concatenate the data
    match first.device().device_type {
        DeviceType::Cpu => cat_cpu(tensors, dim, &new_dims),
        #[cfg(feature = "metal")]
        // No dedicated Metal cat kernel yet; fall back to CPU.
        DeviceType::Metal => cat_cpu(tensors, dim, &new_dims),
        _ => Tensor::empty(Shape::new(new_dims), first.dtype(), first.device()),
    }
}

fn cat_cpu(tensors: &[&Tensor], dim: usize, out_dims: &[usize]) -> Result<Tensor> {
    let first = tensors[0];
    let out_size: usize = out_dims.iter().product();
    let mut out_data = vec![0.0f32; out_size];

    // Calculate strides
    let mut out_strides = vec![1usize; out_dims.len()];
    for i in (0..out_dims.len() - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    }

    let mut dim_offset = 0;

    for t in tensors {
        let t_data: Vec<f32> = t.to_f32_vec()?;
        let t_dims = t.shape().dims();

        let mut t_strides = vec![1usize; t_dims.len()];
        for i in (0..t_dims.len() - 1).rev() {
            t_strides[i] = t_strides[i + 1] * t_dims[i + 1];
        }

        for t_idx in 0..t_data.len() {
            // Convert linear index to multi-dimensional
            let mut coords = vec![0usize; t_dims.len()];
            let mut remaining = t_idx;
            for i in 0..t_dims.len() {
                coords[i] = remaining / t_strides[i];
                remaining %= t_strides[i];
            }

            // Adjust coordinate in concat dimension
            coords[dim] += dim_offset;

            // Convert to output linear index
            let out_idx: usize = coords
                .iter()
                .zip(out_strides.iter())
                .map(|(&c, &s)| c * s)
                .sum();

            if out_idx < out_data.len() {
                out_data[out_idx] = t_data[t_idx];
            }
        }

        dim_offset += t_dims[dim];
    }

    tensor_from_f32_as(out_data, Shape::new(out_dims.to_vec()), first.dtype(), first.device())
}

/// Stack tensors along a new dimension.
pub fn stack(tensors: &[&Tensor], dim: i64) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(Error::InvalidArgument {
            name: "tensors".into(),
            message: "cannot stack empty list".into(),
        });
    }

    let first = tensors[0];
    let rank = first.rank();

    // Normalize dim (can be 0 to rank inclusive)
    let dim = if dim < 0 {
        (rank as i64 + 1 + dim) as usize
    } else {
        dim as usize
    };

    if dim > rank {
        return Err(Error::InvalidArgument {
            name: "dim".into(),
            message: format!("dimension {} out of range for stacking rank {}", dim, rank),
        });
    }

    // Validate all tensors have the same shape
    for (i, t) in tensors.iter().enumerate() {
        if t.shape() != first.shape() {
            return Err(Error::shape_mismatch(
                format!("{:?}", first.shape().dims()),
                format!("{:?} at index {}", t.shape().dims(), i),
            ));
        }
    }

    // Unsqueeze each tensor and concatenate
    let mut unsqueezed: Vec<Tensor> = Vec::with_capacity(tensors.len());
    for t in tensors {
        unsqueezed.push(unsqueeze(t, dim as i64)?);
    }

    let refs: Vec<&Tensor> = unsqueezed.iter().collect();
    cat(&refs, dim as i64)
}

/// Add a dimension of size 1.
pub fn unsqueeze(input: &Tensor, dim: i64) -> Result<Tensor> {
    let rank = input.rank();
    let dim = if dim < 0 {
        (rank as i64 + 1 + dim) as usize
    } else {
        dim as usize
    };

    if dim > rank {
        return Err(Error::InvalidArgument {
            name: "dim".into(),
            message: format!("dimension {} out of range for rank {}", dim, rank),
        });
    }

    let mut new_dims = input.shape().dims().to_vec();
    new_dims.insert(dim, 1);

    // Data doesn't change, just shape
    let data: Vec<f32> = input.to_f32_vec()?;
    tensor_from_f32_as(data, Shape::new(new_dims), input.dtype(), input.device())
}

/// Remove dimensions of size 1.
pub fn squeeze(input: &Tensor, dim: Option<i64>) -> Result<Tensor> {
    let dims = input.shape().dims();

    let new_dims: Vec<usize> = match dim {
        Some(d) => {
            let d = if d < 0 {
                (dims.len() as i64 + d) as usize
            } else {
                d as usize
            };

            if d >= dims.len() {
                return Err(Error::InvalidArgument {
                    name: "dim".into(),
                    message: format!("dimension {} out of range", d),
                });
            }

            if dims[d] == 1 {
                let mut new_dims = dims.to_vec();
                new_dims.remove(d);
                new_dims
            } else {
                dims.to_vec()
            }
        }
        None => dims.iter().copied().filter(|&d| d != 1).collect(),
    };

    // Handle scalar case
    let new_dims = if new_dims.is_empty() {
        vec![1]
    } else {
        new_dims
    };

    let data: Vec<f32> = input.to_f32_vec()?;
    tensor_from_f32_as(data, Shape::new(new_dims), input.dtype(), input.device())
}

/// Reshape tensor to new shape.
pub fn reshape(input: &Tensor, new_shape: &[i64]) -> Result<Tensor> {
    let numel = input.shape().numel();

    // Handle -1 dimension
    let mut inferred_dim = None;
    let mut product: usize = 1;

    for (i, &d) in new_shape.iter().enumerate() {
        if d == -1 {
            if inferred_dim.is_some() {
                return Err(Error::InvalidArgument {
                    name: "shape".into(),
                    message: "only one dimension can be -1".into(),
                });
            }
            inferred_dim = Some(i);
        } else if d < 0 {
            return Err(Error::InvalidArgument {
                name: "shape".into(),
                message: format!("invalid dimension {}", d),
            });
        } else {
            product *= d as usize;
        }
    }

    let new_dims: Vec<usize> = match inferred_dim {
        Some(i) => {
            if numel % product != 0 {
                return Err(Error::shape_mismatch(
                    format!("{} elements", numel),
                    format!("incompatible shape {:?}", new_shape),
                ));
            }
            let inferred = numel / product;
            new_shape
                .iter()
                .enumerate()
                .map(|(j, &d)| if j == i { inferred } else { d as usize })
                .collect()
        }
        None => {
            if product != numel {
                return Err(Error::shape_mismatch(
                    format!("{} elements", numel),
                    format!("{} elements in new shape", product),
                ));
            }
            new_shape.iter().map(|&d| d as usize).collect()
        }
    };

    let data: Vec<f32> = input.to_f32_vec()?;
    tensor_from_f32_as(data, Shape::new(new_dims), input.dtype(), input.device())
}

// ============================================================================
// Reductions
// ============================================================================

/// Sum over dimensions.
pub fn sum(input: &Tensor, dims: &[i64], keep_dim: bool) -> Result<Tensor> {
    reduce_op(input, dims, keep_dim, |acc, x| acc + x, 0.0)
}

/// Mean over dimensions.
pub fn mean(input: &Tensor, dims: &[i64], keep_dim: bool) -> Result<Tensor> {
    let result = sum(input, dims, keep_dim)?;

    // Calculate reduction factor
    let input_dims = input.shape().dims();
    let mut factor: usize = 1;
    for &d in dims {
        let d = if d < 0 {
            (input_dims.len() as i64 + d) as usize
        } else {
            d as usize
        };
        if d < input_dims.len() {
            factor *= input_dims[d];
        }
    }

    mul_scalar(&result, 1.0 / factor as f32)
}

/// Maximum over dimensions.
pub fn max(input: &Tensor, dims: &[i64], keep_dim: bool) -> Result<Tensor> {
    reduce_op(input, dims, keep_dim, |acc, x| acc.max(x), f32::NEG_INFINITY)
}

/// Minimum over dimensions.
pub fn min(input: &Tensor, dims: &[i64], keep_dim: bool) -> Result<Tensor> {
    reduce_op(input, dims, keep_dim, |acc, x| acc.min(x), f32::INFINITY)
}

/// Argmax over a dimension.
pub fn argmax(input: &Tensor, dim: i64, keep_dim: bool) -> Result<Tensor> {
    let rank = input.rank();
    let dim = if dim < 0 {
        (rank as i64 + dim) as usize
    } else {
        dim as usize
    };

    if dim >= rank {
        return Err(Error::InvalidArgument {
            name: "dim".into(),
            message: format!("dimension {} out of range for rank {}", dim, rank),
        });
    }

    match input.device().device_type {
        DeviceType::Cpu => argmax_cpu(input, dim, keep_dim),
        #[cfg(feature = "metal")]
        // Argmax on CPU — small output (single index per row), no GPU kernel overhead needed.
        DeviceType::Metal => argmax_cpu(input, dim, keep_dim),
        _ => Err(Error::unsupported(format!(
            "argmax not implemented for {:?}",
            input.device().device_type
        ))),
    }
}

fn argmax_cpu(input: &Tensor, dim: usize, keep_dim: bool) -> Result<Tensor> {
    let data: Vec<f32> = input.to_f32_vec()?;
    let shape = input.shape();
    let dims = shape.dims();

    // Calculate output shape
    let mut out_dims: Vec<usize> = dims.to_vec();
    if keep_dim {
        out_dims[dim] = 1;
    } else {
        out_dims.remove(dim);
    }

    // Ensure at least 1D
    if out_dims.is_empty() {
        out_dims.push(1);
    }

    // Calculate strides
    let mut strides = vec![1usize; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    let dim_size = dims[dim];
    let dim_stride = strides[dim];

    // Number of reductions
    let outer_size: usize = dims[..dim].iter().product::<usize>().max(1);
    let inner_size: usize = dims[dim + 1..].iter().product::<usize>().max(1);

    let mut out_data = Vec::with_capacity(outer_size * inner_size);

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut max_idx = 0usize;
            let mut max_val = f32::NEG_INFINITY;

            for i in 0..dim_size {
                let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                if idx < data.len() && data[idx] > max_val {
                    max_val = data[idx];
                    max_idx = i;
                }
            }

            out_data.push(max_idx as f32);
        }
    }

    tensor_from_f32_as(out_data, Shape::new(out_dims), input.dtype(), input.device())
}

/// Generic reduction operation.
fn reduce_op<F>(
    input: &Tensor,
    dims: &[i64],
    keep_dim: bool,
    op: F,
    init: f32,
) -> Result<Tensor>
where
    F: Fn(f32, f32) -> f32,
{
    match input.device().device_type {
        DeviceType::Cpu => reduce_op_cpu(input, dims, keep_dim, op, init),
        #[cfg(feature = "metal")]
        // Reductions (sum/mean/max/min) on CPU — scalar output, UMA zero-copy, no GPU dispatch overhead.
        DeviceType::Metal => reduce_op_cpu(input, dims, keep_dim, op, init),
        _ => Err(Error::unsupported(format!(
            "reduce not implemented for {:?}",
            input.device().device_type
        ))),
    }
}

fn reduce_op_cpu<F>(
    input: &Tensor,
    dims: &[i64],
    keep_dim: bool,
    op: F,
    init: f32,
) -> Result<Tensor>
where
    F: Fn(f32, f32) -> f32,
{
    let data: Vec<f32> = input.to_f32_vec()?;
    let shape = input.shape();
    let in_dims = shape.dims();
    let rank = in_dims.len();

    // Normalize and deduplicate dims
    let mut reduce_dims: Vec<usize> = dims
        .iter()
        .map(|&d| {
            if d < 0 {
                (rank as i64 + d) as usize
            } else {
                d as usize
            }
        })
        .collect();
    reduce_dims.sort_unstable();
    reduce_dims.dedup();

    // Validate dims
    for &d in &reduce_dims {
        if d >= rank {
            return Err(Error::InvalidArgument {
                name: "dims".into(),
                message: format!("dimension {} out of range for rank {}", d, rank),
            });
        }
    }

    // Calculate output shape
    let out_dims: Vec<usize> = in_dims
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| {
            if reduce_dims.contains(&i) {
                if keep_dim {
                    Some(1)
                } else {
                    None
                }
            } else {
                Some(d)
            }
        })
        .collect();

    let out_dims = if out_dims.is_empty() {
        vec![1]
    } else {
        out_dims
    };

    let out_size: usize = out_dims.iter().product();
    let mut out_data = vec![init; out_size];

    // Calculate strides
    let mut in_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
    }

    let mut out_strides = vec![1usize; out_dims.len()];
    for i in (0..out_dims.len().saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    }

    // Perform reduction
    for in_idx in 0..data.len() {
        // Convert to coords
        let mut in_coords = vec![0usize; rank];
        let mut remaining = in_idx;
        for i in 0..rank {
            in_coords[i] = remaining / in_strides[i];
            remaining %= in_strides[i];
        }

        // Map to output coords
        let out_coords: Vec<usize> = in_coords
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| {
                if reduce_dims.contains(&i) {
                    if keep_dim {
                        Some(0)
                    } else {
                        None
                    }
                } else {
                    Some(c)
                }
            })
            .collect();

        // Convert to output index
        let out_idx: usize = out_coords
            .iter()
            .zip(out_strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();

        if out_idx < out_data.len() {
            out_data[out_idx] = op(out_data[out_idx], data[in_idx]);
        }
    }

    tensor_from_f32_as(out_data, Shape::new(out_dims), input.dtype(), input.device())
}

// ============================================================================
// Tensor creation
// ============================================================================

/// Create a tensor filled with zeros.
pub fn zeros(shape: &[usize], dtype: DType, device: crate::hal::DeviceId) -> Result<Tensor> {
    let numel: usize = shape.iter().product();
    let data = vec![0.0f32; numel];
    tensor_from_f32_as(data, Shape::new(shape.to_vec()), dtype, device)
}

/// Create a tensor filled with ones.
pub fn ones(shape: &[usize], dtype: DType, device: crate::hal::DeviceId) -> Result<Tensor> {
    let numel: usize = shape.iter().product();
    let data = vec![1.0f32; numel];
    tensor_from_f32_as(data, Shape::new(shape.to_vec()), dtype, device)
}

/// Create a tensor filled with a scalar value.
pub fn full(shape: &[usize], value: f32, dtype: DType, device: crate::hal::DeviceId) -> Result<Tensor> {
    let numel: usize = shape.iter().product();
    let data = vec![value; numel];
    tensor_from_f32_as(data, Shape::new(shape.to_vec()), dtype, device)
}

/// Create an identity matrix.
pub fn eye(n: usize, dtype: DType, device: crate::hal::DeviceId) -> Result<Tensor> {
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    tensor_from_f32_as(data, Shape::new(vec![n, n]), dtype, device)
}

/// Create a 1D tensor with values from start to end (exclusive).
pub fn arange(start: f32, end: f32, step: f32, dtype: DType, device: crate::hal::DeviceId) -> Result<Tensor> {
    if step == 0.0 {
        return Err(Error::InvalidArgument {
            name: "step".into(),
            message: "step cannot be zero".into(),
        });
    }

    let mut data = Vec::new();
    let mut val = start;

    if step > 0.0 {
        while val < end {
            data.push(val);
            val += step;
        }
    } else {
        while val > end {
            data.push(val);
            val += step;
        }
    }

    let len = data.len();
    tensor_from_f32_as(data, Shape::new(vec![len]), dtype, device)
}

/// Cast tensor to a different dtype.
pub fn cast(input: &Tensor, dtype: DType) -> Result<Tensor> {
    // For now, we only support F32 internally, so this is mostly a no-op
    // In the future, this would handle actual dtype conversion
    let data: Vec<f32> = input.to_f32_vec()?;
    tensor_from_f32_as(data, input.shape().clone(), dtype, input.device())
}

/// Clone a tensor.
pub fn clone_tensor(input: &Tensor) -> Result<Tensor> {
    let data: Vec<f32> = input.to_f32_vec()?;
    tensor_from_f32_as(data, input.shape().clone(), input.dtype(), input.device())
}
