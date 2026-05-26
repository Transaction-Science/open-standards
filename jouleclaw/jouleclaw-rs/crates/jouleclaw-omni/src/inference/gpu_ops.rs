//! Shared GPU dispatch operations for architecture pipelines.
//!
//! Eliminates duplication of identical helper functions across architecture files.
//! All functions take explicit model/compute/kernel parameters (no `self`).

#[cfg(feature = "metal")]
use crate::core::{Error, Result};
#[cfg(feature = "metal")]
use crate::hal::metal::{BorrowedMetalBuffer, ComputePipeline, MetalCompute};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;
#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::tensor::{DType, Shape, Tensor};
#[cfg(feature = "metal")]
use std::sync::Arc;

// ==================== Buffer Utilities ====================

/// Set a Metal buffer on a compute command encoder at the given index.
#[cfg(feature = "metal")]
pub fn set_tensor_buffer(encoder: &metal::ComputeCommandEncoderRef, index: u64, tensor: &Tensor) {
    if let Some(ptr) = tensor.device_ptr() {
        let b = unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) };
        encoder.set_buffer(index, Some(b.as_ref()), tensor.byte_offset() as u64);
    }
}

// ==================== Weight Loading ====================

/// Read a weight from the model, converting to f16 GPU tensor.
#[cfg(feature = "metal")]
pub fn read_weight_f16(model: &Model, compute: &MetalCompute, name: &str) -> Result<Tensor> {
    let lt = model
        .get_weight(name)
        .ok_or_else(|| Error::internal(format!("weight not found: {}", name)))?;
    let numel = lt.shape().numel();
    let is_f32 = lt.buffer().length() as usize >= numel * 4;
    let device_id = compute.device().info().id;
    if is_f32 {
        let ptr = lt.buffer().contents() as *const f32;
        let f32_data = unsafe { std::slice::from_raw_parts(ptr, numel) };
        let f16_data: Vec<half::f16> = f32_data.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&f16_data, lt.shape().clone(), DType::F16, device_id)
    } else {
        let ptr = lt.buffer().contents() as *const half::f16;
        let f16_data = unsafe { std::slice::from_raw_parts(ptr, numel).to_vec() };
        Tensor::from_slice(&f16_data, lt.shape().clone(), DType::F16, device_id)
    }
}

/// Read a weight from the model as f32 Vec (converting from f16 if needed).
#[cfg(feature = "metal")]
pub fn read_weight_vec_f32(model: &Model, name: &str) -> Result<Vec<f32>> {
    let lt = model
        .get_weight(name)
        .ok_or_else(|| Error::internal(format!("weight not found: {}", name)))?;
    let numel = lt.shape().numel();
    let is_f32 = lt.buffer().length() as usize >= numel * 4;
    if is_f32 {
        let ptr = lt.buffer().contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(ptr, numel).to_vec() })
    } else {
        let ptr = lt.buffer().contents() as *const half::f16;
        let f16_data = unsafe { std::slice::from_raw_parts(ptr, numel) };
        Ok(f16_data.iter().map(|v| v.to_f32()).collect())
    }
}

// ==================== Activation Dispatches ====================

/// GPU activation dispatch (gelu, silu, relu — parameterized by kernel).
#[cfg(feature = "metal")]
pub fn activation_on(
    compute: &MetalCompute,
    kernel: &ComputePipeline,
    cb: &metal::CommandBufferRef,
    input: &Tensor,
) -> Tensor {
    let numel = input.numel();
    let device = compute.device().raw();
    let output_buffer =
        device.new_buffer((numel * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
    compute.dispatch_1d(cb, kernel, numel, |encoder| {
        set_tensor_buffer(encoder, 0, input);
        encoder.set_buffer(1, Some(&output_buffer), 0);
    });
    Tensor::from_metal_buffer(
        output_buffer,
        input.shape().clone(),
        DType::F16,
        compute.device().info().id,
    )
}

// ==================== Linear Dispatch ====================

/// GPU linear with bias: output = input @ W^T + bias.
///
/// Loads weights by name from the model, dispatches the linear kernel.
/// Returns [m, n] f16 tensor.
#[cfg(feature = "metal")]
pub fn linear_bias_on(
    compute: &MetalCompute,
    linear_kernel: &ComputePipeline,
    cb: &metal::CommandBufferRef,
    model: &Model,
    input: &Tensor,
    weight_name: &str,
    bias_name: &str,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor> {
    let w_f16 = read_weight_f16(model, compute, weight_name)?;
    let b_f16 = read_weight_f16(model, compute, bias_name)?;
    let device = compute.device().raw();
    let output_buffer =
        device.new_buffer((m * n * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
    let tile: usize = 16;
    compute.dispatch(
        cb,
        linear_kernel,
        ((n + tile - 1) / tile, (m + tile - 1) / tile, 1),
        (tile, tile, 1),
        |encoder| {
            set_tensor_buffer(encoder, 0, input);
            set_tensor_buffer(encoder, 1, &w_f16);
            set_tensor_buffer(encoder, 2, &b_f16);
            encoder.set_buffer(3, Some(&output_buffer), 0);
            let vals: [u32; 4] = [m as u32, n as u32, k as u32, 1];
            for (i, v) in vals.iter().enumerate() {
                encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
            }
        },
    );
    Ok(Tensor::from_metal_buffer(
        output_buffer,
        Shape::from([m, n]),
        DType::F16,
        compute.device().info().id,
    ))
}

/// GPU linear with pre-loaded weight/bias tensors (no model access needed).
///
/// Returns [m, n] f16 tensor.
#[cfg(feature = "metal")]
pub fn linear_tensors_on(
    compute: &MetalCompute,
    linear_kernel: &ComputePipeline,
    cb: &metal::CommandBufferRef,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    m: usize,
    k: usize,
    n: usize,
) -> Tensor {
    let device = compute.device().raw();
    let output_buffer =
        device.new_buffer((m * n * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
    let tile: usize = 16;
    compute.dispatch(
        cb,
        linear_kernel,
        ((n + tile - 1) / tile, (m + tile - 1) / tile, 1),
        (tile, tile, 1),
        |encoder| {
            set_tensor_buffer(encoder, 0, input);
            set_tensor_buffer(encoder, 1, weight);
            set_tensor_buffer(encoder, 2, bias);
            encoder.set_buffer(3, Some(&output_buffer), 0);
            let vals: [u32; 4] = [m as u32, n as u32, k as u32, 1];
            for (i, v) in vals.iter().enumerate() {
                encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
            }
        },
    );
    Tensor::from_metal_buffer(
        output_buffer,
        Shape::from([m, n]),
        DType::F16,
        compute.device().info().id,
    )
}

// ==================== Layer Norm Dispatch ====================

/// GPU layer normalization.
///
/// Loads gamma/beta by name from the model, dispatches the layer_norm kernel.
/// Returns [n, d] f16 tensor.
#[cfg(feature = "metal")]
pub fn layer_norm_on(
    compute: &MetalCompute,
    ln_kernel: &ComputePipeline,
    cb: &metal::CommandBufferRef,
    model: &Model,
    input: &Tensor,
    weight_name: &str,
    bias_name: &str,
    n: usize,
    d: usize,
    eps: f32,
) -> Result<Tensor> {
    let w_f16 = read_weight_f16(model, compute, weight_name)?;
    let b_f16 = read_weight_f16(model, compute, bias_name)?;
    let device = compute.device().raw();
    let output_buffer =
        device.new_buffer((n * d * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
    compute.dispatch_1d(cb, ln_kernel, n, |encoder| {
        set_tensor_buffer(encoder, 0, input);
        set_tensor_buffer(encoder, 1, &w_f16);
        set_tensor_buffer(encoder, 2, &b_f16);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        let n_u32 = n as u32;
        let d_u32 = d as u32;
        encoder.set_bytes(4, 4, &n_u32 as *const u32 as *const _);
        encoder.set_bytes(5, 4, &d_u32 as *const u32 as *const _);
        encoder.set_bytes(6, 4, &eps as *const f32 as *const _);
    });
    Ok(Tensor::from_metal_buffer(
        output_buffer,
        Shape::from([n, d]),
        DType::F16,
        compute.device().info().id,
    ))
}

/// GPU layer normalization with pre-loaded weight/bias tensors (no model
/// access needed). Returns `[n, d]` f16 tensor.
#[cfg(feature = "metal")]
pub fn layer_norm_tensors_on(
    compute: &MetalCompute,
    ln_kernel: &ComputePipeline,
    cb: &metal::CommandBufferRef,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    n: usize,
    d: usize,
    eps: f32,
) -> Tensor {
    let device = compute.device().raw();
    let output_buffer =
        device.new_buffer((n * d * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
    compute.dispatch_1d(cb, ln_kernel, n, |encoder| {
        set_tensor_buffer(encoder, 0, input);
        set_tensor_buffer(encoder, 1, weight);
        set_tensor_buffer(encoder, 2, bias);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        let n_u32 = n as u32;
        let d_u32 = d as u32;
        encoder.set_bytes(4, 4, &n_u32 as *const u32 as *const _);
        encoder.set_bytes(5, 4, &d_u32 as *const u32 as *const _);
        encoder.set_bytes(6, 4, &eps as *const f32 as *const _);
    });
    Tensor::from_metal_buffer(
        output_buffer,
        Shape::from([n, d]),
        DType::F16,
        compute.device().info().id,
    )
}

// ==================== Transpose Dispatches ====================

/// GPU transpose: [S, H, D] → [H, S, D].
#[cfg(feature = "metal")]
pub fn transpose_shd_to_hsd_on(
    compute: &MetalCompute,
    kernel: &ComputePipeline,
    cb: &metal::CommandBufferRef,
    input: &Tensor,
    output: &Tensor,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) {
    let tile: usize = 16;
    compute.dispatch(
        cb,
        kernel,
        (
            (head_dim + tile - 1) / tile,
            (seq_len + tile - 1) / tile,
            num_heads,
        ),
        (tile, tile, 1),
        |encoder| {
            set_tensor_buffer(encoder, 0, input);
            set_tensor_buffer(encoder, 1, output);
            let vals: [u32; 3] = [seq_len as u32, num_heads as u32, head_dim as u32];
            for (i, v) in vals.iter().enumerate() {
                encoder.set_bytes((2 + i) as u64, 4, v as *const u32 as *const _);
            }
        },
    );
}

/// GPU transpose: [H, S, D] → [S, H, D].
#[cfg(feature = "metal")]
pub fn transpose_hsd_to_shd_on(
    compute: &MetalCompute,
    kernel: &ComputePipeline,
    cb: &metal::CommandBufferRef,
    input: &Tensor,
    output: &Tensor,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) {
    let tile: usize = 16;
    compute.dispatch(
        cb,
        kernel,
        (
            (head_dim + tile - 1) / tile,
            (seq_len + tile - 1) / tile,
            num_heads,
        ),
        (tile, tile, 1),
        |encoder| {
            set_tensor_buffer(encoder, 0, input);
            set_tensor_buffer(encoder, 1, output);
            let vals: [u32; 3] = [seq_len as u32, num_heads as u32, head_dim as u32];
            for (i, v) in vals.iter().enumerate() {
                encoder.set_bytes((2 + i) as u64, 4, v as *const u32 as *const _);
            }
        },
    );
}

// ==================== Element-wise Dispatches ====================

/// GPU element-wise binary operation (add, sub, mul — parameterized by kernel).
#[cfg(feature = "metal")]
pub fn elementwise_binary_on(
    compute: &MetalCompute,
    kernel: &ComputePipeline,
    cb: &metal::CommandBufferRef,
    a: &Tensor,
    b: &Tensor,
) -> Tensor {
    let numel = a.numel();
    let device = compute.device().raw();
    let output_buffer =
        device.new_buffer((numel * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
    compute.dispatch_1d(cb, kernel, numel, |encoder| {
        set_tensor_buffer(encoder, 0, a);
        set_tensor_buffer(encoder, 1, b);
        encoder.set_buffer(2, Some(&output_buffer), 0);
    });
    Tensor::from_metal_buffer(
        output_buffer,
        a.shape().clone(),
        DType::F16,
        compute.device().info().id,
    )
}

/// GPU scalar multiply: output[i] = input[i] * scale.
#[cfg(feature = "metal")]
pub fn scale_tensor_on(
    compute: &MetalCompute,
    kernel: &ComputePipeline,
    cb: &metal::CommandBufferRef,
    input: &Tensor,
    s: f32,
) -> Tensor {
    let numel = input.numel();
    let device = compute.device().raw();
    let output_buffer =
        device.new_buffer((numel * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
    compute.dispatch_1d(cb, kernel, numel, |encoder| {
        set_tensor_buffer(encoder, 0, input);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(2, 4, &s as *const f32 as *const _);
    });
    Tensor::from_metal_buffer(
        output_buffer,
        input.shape().clone(),
        DType::F16,
        compute.device().info().id,
    )
}

// ==================== DaViT / Swin patch merge ====================

/// GPU patch-merge concat: `[H, W, D]` → `[H/2 * W/2, 4*D]`.
///
/// Used by DaViT's downsample stages between vision-encoder layers.
/// Channel order matches the standard reference implementation
/// (top-left, bottom-left, top-right, bottom-right) so the trained
/// `reduction.weight` linear projection that follows is correct without
/// extra channel shuffling.
#[cfg(feature = "metal")]
pub fn patch_merge_concat_on(
    compute: &MetalCompute,
    kernel: &ComputePipeline,
    cb: &metal::CommandBufferRef,
    input: &Tensor,
    in_h: usize,
    in_w: usize,
    dim: usize,
) -> Tensor {
    let out_h = in_h / 2;
    let out_w = in_w / 2;
    let num_out = out_h * out_w;
    let merged_dim = dim * 4;

    let device = compute.device().raw();
    let output_buffer = device.new_buffer(
        (num_out * merged_dim * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let tile_x: usize = 16;
    let tile_y: usize = 16;
    compute.dispatch(
        cb,
        kernel,
        (
            (num_out + tile_x - 1) / tile_x,
            (dim + tile_y - 1) / tile_y,
            1,
        ),
        (tile_x, tile_y, 1),
        |encoder| {
            set_tensor_buffer(encoder, 0, input);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            let vals: [u32; 3] = [in_h as u32, in_w as u32, dim as u32];
            for (i, v) in vals.iter().enumerate() {
                encoder.set_bytes((2 + i) as u64, 4, v as *const u32 as *const _);
            }
        },
    );

    Tensor::from_metal_buffer(
        output_buffer,
        Shape::from([num_out, merged_dim]),
        DType::F16,
        compute.device().info().id,
    )
}

// ==================== Canny edge detection ====================

/// Compiled pipelines for the 6-stage Canny edge detection kernel suite.
///
/// Built once from `sources::CANNY_EDGE` (compiled into the shader library
/// as `"canny_edge"` at startup). Reused for every ControlNet preprocessing
/// call.
#[cfg(feature = "metal")]
pub struct CannyKernels {
    /// RGB CHW → grayscale HW.
    pub rgb_to_gray: Arc<ComputePipeline>,
    /// Grayscale HW → magnitude HW + quantised direction HW.
    pub sobel: Arc<ComputePipeline>,
    /// Non-maximum suppression along quantised direction.
    pub nms: Arc<ComputePipeline>,
    /// Double thresholding into {0, 0.5, 1.0}.
    pub double_threshold: Arc<ComputePipeline>,
    /// One pass of 8-neighbour hysteresis (caller iterates).
    pub hysteresis: Arc<ComputePipeline>,
    /// Replicate single-channel edge map to RGB CHW for ControlNet input.
    pub gray_to_rgb: Arc<ComputePipeline>,
}

#[cfg(feature = "metal")]
impl CannyKernels {
    /// Compile pipelines from the precompiled `canny_edge` library. Each
    /// kernel needs a unique cache name (`compile_pipeline` keys on `name`)
    /// even though they share one source bundle.
    pub fn new(compute: &MetalCompute) -> Result<Self> {
        Ok(Self {
            rgb_to_gray: compute.compile_pipeline("canny_rgb_to_gray", sources::CANNY_EDGE, "canny_rgb_to_gray_f16")?,
            sobel: compute.compile_pipeline("canny_sobel", sources::CANNY_EDGE, "canny_sobel_f16")?,
            nms: compute.compile_pipeline("canny_nms", sources::CANNY_EDGE, "canny_nms_f16")?,
            double_threshold: compute.compile_pipeline("canny_double_threshold", sources::CANNY_EDGE, "canny_double_threshold_f16")?,
            hysteresis: compute.compile_pipeline("canny_hysteresis", sources::CANNY_EDGE, "canny_hysteresis_f16")?,
            gray_to_rgb: compute.compile_pipeline("canny_gray_to_rgb", sources::CANNY_EDGE, "canny_gray_to_rgb_f16")?,
        })
    }
}

/// GPU Canny edge detection.
///
/// `rgb_chw` is `[3, H, W]` f16 in 0..1 range. Returns `[3, H, W]` f16 with
/// edge map replicated across all three channels (white-on-black) — the
/// shape SD ControlNet expects as conditioning input.
///
/// `low_threshold` and `high_threshold` operate on Sobel magnitude already
/// normalised to 0..1. Typical defaults: `low=0.1, high=0.2`. Lower the
/// thresholds for sketch input (line art is already binary-like).
///
/// `hysteresis_iterations` controls how aggressively weak edges are connected
/// to strong ones. 4–8 iterations are sufficient for typical 512×512 inputs;
/// pass 0 to skip hysteresis entirely.
#[cfg(feature = "metal")]
pub fn canny_edge_on(
    compute: &MetalCompute,
    kernels: &CannyKernels,
    cb: &metal::CommandBufferRef,
    rgb_chw: &Tensor,
    height: usize,
    width: usize,
    low_threshold: f32,
    high_threshold: f32,
    hysteresis_iterations: u32,
) -> Result<Tensor> {
    let device = compute.device().raw();
    let device_id = compute.device().info().id;
    let hw = height * width;

    let alloc_hw = || device.new_buffer((hw * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
    let gray = alloc_hw();
    let mag = alloc_hw();
    let dir = alloc_hw();
    let nms = alloc_hw();
    let dbl = alloc_hw();
    let mut hyst_a = alloc_hw();
    let mut hyst_b = alloc_hw();
    let out_rgb = device.new_buffer((3 * hw * 2) as u64, metal::MTLResourceOptions::StorageModeShared);

    let tile_x: usize = 16;
    let tile_y: usize = 16;
    let grid = (
        (width + tile_x - 1) / tile_x,
        (height + tile_y - 1) / tile_y,
        1usize,
    );
    let block = (tile_x, tile_y, 1usize);

    // 1) RGB → gray.
    compute.dispatch(cb, &kernels.rgb_to_gray, grid, block, |encoder| {
        set_tensor_buffer(encoder, 0, rgb_chw);
        encoder.set_buffer(1, Some(&gray), 0);
        let h = height as u32;
        let w = width as u32;
        encoder.set_bytes(2, 4, &h as *const u32 as *const _);
        encoder.set_bytes(3, 4, &w as *const u32 as *const _);
    });

    // 2) Sobel.
    compute.dispatch(cb, &kernels.sobel, grid, block, |encoder| {
        encoder.set_buffer(0, Some(&gray), 0);
        encoder.set_buffer(1, Some(&mag), 0);
        encoder.set_buffer(2, Some(&dir), 0);
        let h = height as u32;
        let w = width as u32;
        encoder.set_bytes(3, 4, &h as *const u32 as *const _);
        encoder.set_bytes(4, 4, &w as *const u32 as *const _);
    });

    // 3) NMS.
    compute.dispatch(cb, &kernels.nms, grid, block, |encoder| {
        encoder.set_buffer(0, Some(&mag), 0);
        encoder.set_buffer(1, Some(&dir), 0);
        encoder.set_buffer(2, Some(&nms), 0);
        let h = height as u32;
        let w = width as u32;
        encoder.set_bytes(3, 4, &h as *const u32 as *const _);
        encoder.set_bytes(4, 4, &w as *const u32 as *const _);
    });

    // 4) Double threshold.
    compute.dispatch(cb, &kernels.double_threshold, grid, block, |encoder| {
        encoder.set_buffer(0, Some(&nms), 0);
        encoder.set_buffer(1, Some(&dbl), 0);
        encoder.set_bytes(2, 4, &low_threshold as *const f32 as *const _);
        encoder.set_bytes(3, 4, &high_threshold as *const f32 as *const _);
        let h = height as u32;
        let w = width as u32;
        encoder.set_bytes(4, 4, &h as *const u32 as *const _);
        encoder.set_bytes(5, 4, &w as *const u32 as *const _);
    });

    // 5) Hysteresis — ping-pong between hyst_a / hyst_b. Seed from `dbl`.
    let n_iter = hysteresis_iterations.max(1);
    {
        let blit = cb.new_blit_command_encoder();
        blit.copy_from_buffer(&dbl, 0, &hyst_a, 0, (hw * 2) as u64);
        blit.end_encoding();
    }
    for _ in 0..n_iter {
        compute.dispatch(cb, &kernels.hysteresis, grid, block, |encoder| {
            encoder.set_buffer(0, Some(&hyst_a), 0);
            encoder.set_buffer(1, Some(&hyst_b), 0);
            let h = height as u32;
            let w = width as u32;
            encoder.set_bytes(2, 4, &h as *const u32 as *const _);
            encoder.set_bytes(3, 4, &w as *const u32 as *const _);
        });
        std::mem::swap(&mut hyst_a, &mut hyst_b);
    }

    // 6) Replicate edge map to RGB.
    compute.dispatch(cb, &kernels.gray_to_rgb, grid, block, |encoder| {
        encoder.set_buffer(0, Some(&hyst_a), 0);
        encoder.set_buffer(1, Some(&out_rgb), 0);
        let h = height as u32;
        let w = width as u32;
        encoder.set_bytes(2, 4, &h as *const u32 as *const _);
        encoder.set_bytes(3, 4, &w as *const u32 as *const _);
    });

    Ok(Tensor::from_metal_buffer(
        out_rgb,
        Shape::from([3, height, width]),
        DType::F16,
        device_id,
    ))
}

// ==================== Batched Attention Dispatches ====================

/// GPU batched Q@K^T: scores[h] = Q[h] @ K[h]^T.
///
/// Inputs are in [H, S, D] layout. Returns raw Metal buffer of scores [H, q_seq, kv_seq].
#[cfg(feature = "metal")]
pub fn batched_qk_on(
    compute: &MetalCompute,
    kernel: &ComputePipeline,
    cb: &metal::CommandBufferRef,
    q: &Tensor,
    k: &Tensor,
    num_heads: usize,
    q_seq: usize,
    kv_seq: usize,
    head_dim: usize,
) -> metal::Buffer {
    let device = compute.device().raw();
    let buf = device.new_buffer(
        (num_heads * q_seq * kv_seq * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let tile: usize = 16;
    compute.dispatch(
        cb,
        kernel,
        ((kv_seq + tile - 1) / tile, (q_seq + tile - 1) / tile, num_heads),
        (tile, tile, 1),
        |encoder| {
            set_tensor_buffer(encoder, 0, q);
            set_tensor_buffer(encoder, 1, k);
            encoder.set_buffer(2, Some(&buf), 0);
            let vals: [u32; 3] = [q_seq as u32, kv_seq as u32, head_dim as u32];
            for (i, v) in vals.iter().enumerate() {
                encoder.set_bytes((3 + i) as u64, 4, v as *const u32 as *const _);
            }
        },
    );
    buf
}

/// GPU row-wise scaled softmax (in-place on scores buffer).
#[cfg(feature = "metal")]
pub fn row_softmax_on(
    compute: &MetalCompute,
    kernel: &ComputePipeline,
    cb: &metal::CommandBufferRef,
    scores: &metal::Buffer,
    num_rows: usize,
    num_cols: usize,
    scale: f32,
) {
    compute.dispatch_1d(cb, kernel, num_rows, |encoder| {
        encoder.set_buffer(0, Some(scores), 0);
        let rows = num_rows as u32;
        let cols = num_cols as u32;
        encoder.set_bytes(1, 4, &rows as *const u32 as *const _);
        encoder.set_bytes(2, 4, &cols as *const u32 as *const _);
        encoder.set_bytes(3, 4, &scale as *const f32 as *const _);
    });
}

/// GPU batched Scores@V: output[h] = Scores[h] @ V[h].
///
/// Returns tensor in [H, q_seq, head_dim] layout.
#[cfg(feature = "metal")]
pub fn batched_sv_on(
    compute: &MetalCompute,
    kernel: &ComputePipeline,
    cb: &metal::CommandBufferRef,
    scores: &metal::Buffer,
    v: &Tensor,
    num_heads: usize,
    q_seq: usize,
    kv_seq: usize,
    head_dim: usize,
) -> Tensor {
    let device_id = compute.device().info().id;
    let output = Tensor::empty(
        Shape::from([num_heads, q_seq, head_dim]),
        DType::F16,
        device_id,
    )
    .unwrap();
    let tile: usize = 16;
    compute.dispatch(
        cb,
        kernel,
        ((head_dim + tile - 1) / tile, (q_seq + tile - 1) / tile, num_heads),
        (tile, tile, 1),
        |encoder| {
            encoder.set_buffer(0, Some(scores), 0);
            set_tensor_buffer(encoder, 1, v);
            set_tensor_buffer(encoder, 2, &output);
            let vals: [u32; 3] = [q_seq as u32, head_dim as u32, kv_seq as u32];
            for (i, v) in vals.iter().enumerate() {
                encoder.set_bytes((3 + i) as u64, 4, v as *const u32 as *const _);
            }
        },
    );
    output
}

/// Full multi-head batched attention: Q,K,V in [S,H,D] → output [q_seq, hidden].
///
/// Orchestrates: transpose SHD→HSD → Q@K^T → softmax → S@V → transpose HSD→SHD → reshape.
#[cfg(feature = "metal")]
pub fn batched_attention_on(
    compute: &MetalCompute,
    kernels: &CommonKernels,
    cb: &metal::CommandBufferRef,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_seq: usize,
    kv_seq: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Tensor> {
    let device_id = compute.device().info().id;
    let q_hsd = Tensor::empty(Shape::from([num_heads, q_seq, head_dim]), DType::F16, device_id)?;
    let k_hsd = Tensor::empty(Shape::from([num_heads, kv_seq, head_dim]), DType::F16, device_id)?;
    let v_hsd = Tensor::empty(Shape::from([num_heads, kv_seq, head_dim]), DType::F16, device_id)?;
    transpose_shd_to_hsd_on(compute, &kernels.transpose_shd_to_hsd, cb, q, &q_hsd, q_seq, num_heads, head_dim);
    transpose_shd_to_hsd_on(compute, &kernels.transpose_shd_to_hsd, cb, k, &k_hsd, kv_seq, num_heads, head_dim);
    transpose_shd_to_hsd_on(compute, &kernels.transpose_shd_to_hsd, cb, v, &v_hsd, kv_seq, num_heads, head_dim);

    let scores = batched_qk_on(compute, &kernels.batched_linear, cb, &q_hsd, &k_hsd, num_heads, q_seq, kv_seq, head_dim);
    row_softmax_on(compute, &kernels.row_softmax_scale, cb, &scores, num_heads * q_seq, kv_seq, scale);
    let attn_hsd = batched_sv_on(compute, &kernels.batched_matmul_nn, cb, &scores, &v_hsd, num_heads, q_seq, kv_seq, head_dim);

    let attn_shd = Tensor::empty(Shape::from([q_seq, num_heads, head_dim]), DType::F16, device_id)?;
    transpose_hsd_to_shd_on(compute, &kernels.transpose_hsd_to_shd, cb, &attn_hsd, &attn_shd, q_seq, num_heads, head_dim);
    let hidden = num_heads * head_dim;
    attn_shd.reshape([q_seq, hidden])
}

// ==================== Common Kernels ====================

/// Pre-compiled kernel pipelines shared by all GPU architecture pipelines.
///
/// Contains the 8 kernels that appear in all architecture files:
/// linear, layer_norm, add, batched_linear, batched_matmul_nn,
/// row_softmax_scale, transpose_shd_to_hsd, transpose_hsd_to_shd.
#[cfg(feature = "metal")]
pub struct CommonKernels {
    pub linear: Arc<ComputePipeline>,
    pub layer_norm: Arc<ComputePipeline>,
    pub add: Arc<ComputePipeline>,
    pub batched_linear: Arc<ComputePipeline>,
    pub batched_matmul_nn: Arc<ComputePipeline>,
    pub row_softmax_scale: Arc<ComputePipeline>,
    pub transpose_shd_to_hsd: Arc<ComputePipeline>,
    pub transpose_hsd_to_shd: Arc<ComputePipeline>,
}

#[cfg(feature = "metal")]
impl CommonKernels {
    /// Compile all common kernel pipelines.
    pub fn new(compute: &MetalCompute) -> Result<Self> {
        Ok(Self {
            linear: compute.compile_pipeline("linear", sources::LINEAR, "linear_f16")?,
            layer_norm: compute.compile_pipeline("layer_norm", sources::LAYER_NORM, "layer_norm_f16")?,
            add: compute.compile_pipeline("add", sources::ELEMENTWISE, "add_f16")?,
            batched_linear: compute.compile_pipeline("batched_linear", sources::LINEAR, "batched_linear_f16")?,
            batched_matmul_nn: compute.compile_pipeline("batched_matmul_nn", sources::LINEAR, "batched_matmul_nn_f16")?,
            row_softmax_scale: compute.compile_pipeline("row_softmax_scale", sources::LINEAR, "row_softmax_scale_f16")?,
            transpose_shd_to_hsd: compute.compile_pipeline("transpose_shd_to_hsd", sources::LINEAR, "transpose_shd_to_hsd_f16")?,
            transpose_hsd_to_shd: compute.compile_pipeline("transpose_hsd_to_shd", sources::LINEAR, "transpose_hsd_to_shd_f16")?,
        })
    }
}

// ==================== Pipeline Trait ====================

/// Trait for GPU architecture pipelines with shared Metal operations.
///
/// Provides default implementations for common GPU dispatches using
/// `CommonKernels`. Implementors only need to provide `compute()` and
/// `common_kernels()`.
#[cfg(feature = "metal")]
pub trait MetalPipeline {
    /// Access to the Metal compute context.
    fn compute(&self) -> &MetalCompute;

    /// Access to the common kernel pipelines.
    fn common_kernels(&self) -> &CommonKernels;

    /// GPU linear with bias: output = input @ W^T + bias.
    fn linear_bias(
        &self, cb: &metal::CommandBufferRef, model: &Model, input: &Tensor,
        weight_name: &str, bias_name: &str, m: usize, k: usize, n: usize,
    ) -> Result<Tensor> {
        linear_bias_on(self.compute(), &self.common_kernels().linear, cb, model, input, weight_name, bias_name, m, k, n)
    }

    /// GPU linear with pre-loaded weight/bias tensors.
    fn linear_tensors(
        &self, cb: &metal::CommandBufferRef, input: &Tensor,
        weight: &Tensor, bias: &Tensor, m: usize, k: usize, n: usize,
    ) -> Tensor {
        linear_tensors_on(self.compute(), &self.common_kernels().linear, cb, input, weight, bias, m, k, n)
    }

    /// GPU layer normalization.
    fn layer_norm(
        &self, cb: &metal::CommandBufferRef, model: &Model, input: &Tensor,
        weight_name: &str, bias_name: &str, n: usize, d: usize, eps: f32,
    ) -> Result<Tensor> {
        layer_norm_on(self.compute(), &self.common_kernels().layer_norm, cb, model, input, weight_name, bias_name, n, d, eps)
    }

    /// GPU activation (parameterized by kernel — gelu, silu, relu, etc.).
    fn activation(&self, cb: &metal::CommandBufferRef, kernel: &ComputePipeline, input: &Tensor) -> Tensor {
        activation_on(self.compute(), kernel, cb, input)
    }

    /// Read a weight from model as f16 GPU tensor.
    fn weight_f16(&self, model: &Model, name: &str) -> Result<Tensor> {
        read_weight_f16(model, self.compute(), name)
    }

    /// Read a weight from model as f32 Vec.
    fn weight_vec_f32(&self, model: &Model, name: &str) -> Result<Vec<f32>> {
        read_weight_vec_f32(model, name)
    }

    /// GPU transpose: [S, H, D] → [H, S, D].
    fn transpose_shd_to_hsd(
        &self, cb: &metal::CommandBufferRef, input: &Tensor, output: &Tensor,
        seq_len: usize, num_heads: usize, head_dim: usize,
    ) {
        transpose_shd_to_hsd_on(self.compute(), &self.common_kernels().transpose_shd_to_hsd, cb, input, output, seq_len, num_heads, head_dim)
    }

    /// GPU transpose: [H, S, D] → [S, H, D].
    fn transpose_hsd_to_shd(
        &self, cb: &metal::CommandBufferRef, input: &Tensor, output: &Tensor,
        seq_len: usize, num_heads: usize, head_dim: usize,
    ) {
        transpose_hsd_to_shd_on(self.compute(), &self.common_kernels().transpose_hsd_to_shd, cb, input, output, seq_len, num_heads, head_dim)
    }

    /// GPU element-wise add.
    fn add(&self, cb: &metal::CommandBufferRef, a: &Tensor, b: &Tensor) -> Tensor {
        elementwise_binary_on(self.compute(), &self.common_kernels().add, cb, a, b)
    }

    /// GPU element-wise binary op (parameterized by kernel — sub, mul, etc.).
    fn elementwise_binary(&self, cb: &metal::CommandBufferRef, kernel: &ComputePipeline, a: &Tensor, b: &Tensor) -> Tensor {
        elementwise_binary_on(self.compute(), kernel, cb, a, b)
    }

    /// GPU scalar multiply: output[i] = input[i] * scale.
    fn scale_tensor(&self, cb: &metal::CommandBufferRef, kernel: &ComputePipeline, input: &Tensor, s: f32) -> Tensor {
        scale_tensor_on(self.compute(), kernel, cb, input, s)
    }

    /// GPU batched Q@K^T. Inputs in [H,S,D] layout. Returns raw scores buffer.
    fn batched_qk(&self, cb: &metal::CommandBufferRef, q: &Tensor, k: &Tensor,
        num_heads: usize, q_seq: usize, kv_seq: usize, head_dim: usize,
    ) -> metal::Buffer {
        batched_qk_on(self.compute(), &self.common_kernels().batched_linear, cb, q, k, num_heads, q_seq, kv_seq, head_dim)
    }

    /// GPU row-wise scaled softmax (in-place).
    fn row_softmax(&self, cb: &metal::CommandBufferRef, scores: &metal::Buffer,
        num_rows: usize, num_cols: usize, scale: f32,
    ) {
        row_softmax_on(self.compute(), &self.common_kernels().row_softmax_scale, cb, scores, num_rows, num_cols, scale)
    }

    /// GPU batched Scores@V. Returns [H, q_seq, head_dim] tensor.
    fn batched_sv(&self, cb: &metal::CommandBufferRef, scores: &metal::Buffer, v: &Tensor,
        num_heads: usize, q_seq: usize, kv_seq: usize, head_dim: usize,
    ) -> Tensor {
        batched_sv_on(self.compute(), &self.common_kernels().batched_matmul_nn, cb, scores, v, num_heads, q_seq, kv_seq, head_dim)
    }

    /// Full multi-head batched attention. Q,K,V in [S,H,D] → [q_seq, hidden].
    fn batched_attention(
        &self, cb: &metal::CommandBufferRef,
        q: &Tensor, k: &Tensor, v: &Tensor,
        q_seq: usize, kv_seq: usize, num_heads: usize, head_dim: usize, scale: f32,
    ) -> Result<Tensor> {
        batched_attention_on(self.compute(), self.common_kernels(), cb, q, k, v, q_seq, kv_seq, num_heads, head_dim, scale)
    }
}
