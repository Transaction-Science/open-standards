//! Shared DiT (Diffusion Transformer) building blocks.
//!
//! Used by AuraFlow and Flux architectures. Provides GPU-accelerated:
//! - AdaLN modulation (adaptive layer norm from timestep embedding)
//! - Timestep embedding (sinusoidal + MLP)
//! - Helper functions for patchify/unpatchify dispatch

use crate::core::Result;
use crate::tensor::{DType, Shape, Tensor};

#[cfg(feature = "metal")]
use crate::hal::{MetalCompute};
#[cfg(feature = "metal")]
use crate::hal::metal::{BorrowedMetalBuffer, ComputePipeline};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;
use std::sync::Arc;

/// Compiled kernel pipelines shared by DiT models.
#[cfg(feature = "metal")]
pub struct DiTKernels {
    /// AdaLN modulate: (1+scale)*x + shift
    pub adaln_modulate: Arc<ComputePipeline>,
    /// AdaLN gate: x + gate * residual
    pub adaln_gate: Arc<ComputePipeline>,
    /// Patchify: [C,H,W] -> [num_patches, C*4]
    pub patchify: Arc<ComputePipeline>,
    /// Unpatchify: [num_patches, C*4] -> [C,H,W]
    pub unpatchify: Arc<ComputePipeline>,
    /// Linear Y=X@W^T
    pub linear: Arc<ComputePipeline>,
    /// SiLU activation
    pub silu: Arc<ComputePipeline>,
    /// Element-wise add
    pub add: Arc<ComputePipeline>,
    /// RMSNorm
    pub rms_norm: Arc<ComputePipeline>,
    /// LayerNorm
    pub layer_norm: Arc<ComputePipeline>,
    /// GEGLU (gated GELU)
    pub geglu: Arc<ComputePipeline>,
}

#[cfg(feature = "metal")]
impl DiTKernels {
    /// Compile all shared DiT kernels.
    pub fn new(compute: &Arc<MetalCompute>) -> Result<Self> {
        Ok(Self {
            adaln_modulate: compute.compile_pipeline("adaln_modulate", sources::ADALN, "adaln_modulate_f16")?,
            adaln_gate: compute.compile_pipeline("adaln_gate", sources::ADALN, "adaln_gate_f16")?,
            patchify: compute.compile_pipeline("patchify", sources::PATCHIFY, "patchify_f16")?,
            unpatchify: compute.compile_pipeline("unpatchify", sources::PATCHIFY, "unpatchify_f16")?,
            linear: compute.compile_pipeline("linear", sources::LINEAR, "linear_f16")?,
            silu: compute.compile_pipeline("silu", sources::SILU, "silu_f16")?,
            add: compute.compile_pipeline("add", sources::ELEMENTWISE, "add_f16")?,
            rms_norm: compute.compile_pipeline("rms_norm", sources::RMS_NORM, "rms_norm_f16")?,
            layer_norm: compute.compile_pipeline("layer_norm", sources::LAYER_NORM, "layer_norm_f16")?,
            geglu: compute.compile_pipeline("geglu", sources::GELU, "geglu_f16")?,
        })
    }
}

/// GPU helper functions for DiT operations.
#[cfg(feature = "metal")]
pub struct DiTOps;

#[cfg(feature = "metal")]
impl DiTOps {
    /// Apply AdaLN modulation: output = (1 + scale) * x + shift
    /// x: [seq_len, hidden_size], scale/shift: [hidden_size]
    pub fn adaln_modulate(
        compute: &MetalCompute,
        kernels: &DiTKernels,
        x: &Tensor,
        scale: &Tensor,
        shift: &Tensor,
        hidden_size: usize,
    ) -> Result<Tensor> {
        let count = x.shape().numel();
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let c_hidden = hidden_size as u32;
        let c_count = count as u32;

        let cb = compute.new_command_buffer();
        let x_buf = borrow(x)?;
        let s_buf = borrow(scale)?;
        let sh_buf = borrow(shift)?;
        let o_buf = borrow(&output)?;

        compute.dispatch_async(cb.as_ref(), &kernels.adaln_modulate,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(x_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(s_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(sh_buf.as_ref()), 0);
                encoder.set_buffer(3, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(4, 4, &c_hidden as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_count as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// Apply gated residual: output = x + gate * residual
    /// x, residual: [seq_len, hidden_size], gate: [hidden_size]
    pub fn adaln_gate(
        compute: &MetalCompute,
        kernels: &DiTKernels,
        x: &Tensor,
        residual: &Tensor,
        gate: &Tensor,
        hidden_size: usize,
    ) -> Result<Tensor> {
        let count = x.shape().numel();
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let c_hidden = hidden_size as u32;
        let c_count = count as u32;

        let cb = compute.new_command_buffer();
        let x_buf = borrow(x)?;
        let r_buf = borrow(residual)?;
        let g_buf = borrow(gate)?;
        let o_buf = borrow(&output)?;

        compute.dispatch_async(cb.as_ref(), &kernels.adaln_gate,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(x_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(r_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(g_buf.as_ref()), 0);
                encoder.set_buffer(3, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(4, 4, &c_hidden as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_count as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// Patchify: [channels, height, width] -> [num_patches, channels*4]
    pub fn patchify(
        compute: &MetalCompute,
        kernels: &DiTKernels,
        input: &Tensor,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let num_patches = (height / 2) * (width / 2);
        let output = Tensor::empty(
            Shape::from([num_patches, channels * 4]),
            DType::F16,
            input.device(),
        )?;

        let cb = compute.new_command_buffer();
        let i_buf = borrow(input)?;
        let o_buf = borrow(&output)?;
        let c_ch = channels as u32;
        let c_h = height as u32;
        let c_w = width as u32;

        compute.dispatch_async(cb.as_ref(), &kernels.patchify,
            (num_patches, channels, 1), (1, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(i_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(2, 4, &c_ch as *const u32 as *const _);
                encoder.set_bytes(3, 4, &c_h as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_w as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// Unpatchify: [num_patches, channels*4] -> [channels, height, width]
    pub fn unpatchify(
        compute: &MetalCompute,
        kernels: &DiTKernels,
        input: &Tensor,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let num_patches = (height / 2) * (width / 2);
        let output = Tensor::empty(
            Shape::from([channels, height, width]),
            DType::F16,
            input.device(),
        )?;

        let cb = compute.new_command_buffer();
        let i_buf = borrow(input)?;
        let o_buf = borrow(&output)?;
        let c_ch = channels as u32;
        let c_h = height as u32;
        let c_w = width as u32;

        compute.dispatch_async(cb.as_ref(), &kernels.unpatchify,
            (num_patches, channels, 1), (1, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(i_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(2, 4, &c_ch as *const u32 as *const _);
                encoder.set_bytes(3, 4, &c_h as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_w as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// Compute sinusoidal timestep embedding.
    /// Returns [1, dim] tensor with sin/cos features.
    pub fn timestep_embedding(timestep: f32, dim: usize, device: crate::hal::DeviceId) -> Result<Tensor> {
        let half_dim = dim / 2;
        let mut emb = vec![0.0f32; dim];

        for i in 0..half_dim {
            let freq = (-(i as f32) / half_dim as f32 * (10000.0f32).ln()).exp();
            let angle = timestep * freq;
            emb[i] = angle.sin();
            emb[i + half_dim] = angle.cos();
        }

        // Convert to F16 tensor
        let f16_data: Vec<half::f16> = emb.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&f16_data, Shape::from([1, dim]), DType::F16, device)
    }
}

#[cfg(feature = "metal")]
fn borrow(tensor: &Tensor) -> Result<BorrowedMetalBuffer> {
    let ptr = tensor.device_ptr()
        .ok_or_else(|| crate::core::Error::internal("tensor not on device"))?;
    Ok(unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
}
