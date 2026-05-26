//! Real-ESRGAN: 4x image super-resolution via Residual-in-Residual Dense Blocks.
//!
//! Architecture (RRDBNet, ~17M params):
//!   RGB [3, H, W] -> conv_first [64, H, W]
//!   -> 23 RRDB blocks (each: 3 ResidualDenseBlocks with dense connections)
//!   -> conv_body [64, H, W] + skip from conv_first
//!   -> 2x nearest upsample + conv_up1 [64, 2H, 2W]
//!   -> 2x nearest upsample + conv_up2 [64, 4H, 4W]
//!   -> conv_hr [64, 4H, 4W] -> LeakyReLU -> conv_last [3, 4H, 4W]
//!
//! All convolutions are 3x3 with padding=1, stride=1.
//! Activation is LeakyReLU(negative_slope=0.2) throughout.
//! ResidualDenseBlock uses dense (concatenation) connections with growth_channels=32.
//! Each RRDB applies residual scaling of 0.2.
//!
//! Weight format: safetensors with keys like:
//!   conv_first.weight, conv_first.bias,
//!   body.{0-22}.rdb{1-3}.conv{1-5}.weight/.bias,
//!   conv_body.weight/.bias, conv_up1.weight/.bias,
//!   conv_up2.weight/.bias, conv_hr.weight/.bias,
//!   conv_last.weight/.bias

use crate::core::{Error, Result};

#[cfg(feature = "metal")]
use crate::tensor::{DType, Shape, Tensor};
#[cfg(feature = "metal")]
use std::sync::Arc;
#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline, BorrowedMetalBuffer};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;
#[cfg(feature = "metal")]
use crate::inference::gpu_ops;
#[cfg(feature = "metal")]
use tracing::debug;

// ==================== Configuration ====================

/// Real-ESRGAN configuration.
#[derive(Debug, Clone)]
pub struct EsrganConfig {
    /// Number of RRDB blocks (default: 23).
    pub num_blocks: usize,
    /// Base feature channels (default: 64).
    pub num_features: usize,
    /// Growth channels per dense layer (default: 32).
    pub growth_channels: usize,
    /// Upscaling factor (default: 4, achieved via 2x upsample applied twice).
    pub scale: usize,
    /// LeakyReLU negative slope (default: 0.2).
    pub leaky_slope: f32,
    /// Residual scaling factor for RRDB blocks (default: 0.2).
    pub residual_scale: f32,
}

impl Default for EsrganConfig {
    fn default() -> Self {
        Self {
            num_blocks: 23,
            num_features: 64,
            growth_channels: 32,
            scale: 4,
            leaky_slope: 0.2,
            residual_scale: 0.2,
        }
    }
}

// ==================== Metal GPU Pipeline ====================

#[cfg(feature = "metal")]
struct EsrganKernels {
    conv2d: Arc<ComputePipeline>,
    leaky_relu: Arc<ComputePipeline>,
    upsample_nearest: Arc<ComputePipeline>,
    add: Arc<ComputePipeline>,
    scale: Arc<ComputePipeline>,
}

/// Real-ESRGAN pipeline for 4x image super-resolution.
///
/// Forward pipeline (pure feedforward, no attention or normalization):
/// 1. conv_first: [3, H, W] -> [64, H, W]
/// 2. 23 RRDB blocks: [64, H, W] -> [64, H, W] (residual dense blocks)
/// 3. conv_body + skip: [64, H, W]
/// 4. upsample 2x + conv_up1: [64, 2H, 2W]
/// 5. upsample 2x + conv_up2: [64, 4H, 4W]
/// 6. conv_hr -> LeakyReLU -> conv_last: [3, 4H, 4W]
#[cfg(feature = "metal")]
pub struct EsrganPipeline {
    model: Arc<parking_lot::RwLock<Model>>,
    compute: Arc<MetalCompute>,
    config: EsrganConfig,
    kernels: EsrganKernels,
}

#[cfg(feature = "metal")]
impl EsrganPipeline {
    /// Create a new Real-ESRGAN pipeline.
    pub fn new(model: Arc<parking_lot::RwLock<Model>>, config: EsrganConfig, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        let kernels = EsrganKernels {
            conv2d: compute.compile_pipeline(
                "conv2d_3x3_tiled",
                sources::CONV2D,
                "conv2d_3x3_tiled_f16",
            )?,
            leaky_relu: compute.compile_pipeline(
                "leaky_relu",
                sources::GELU,
                "leaky_relu_f16",
            )?,
            upsample_nearest: compute.compile_pipeline(
                "upsample_nearest",
                sources::UPSAMPLE,
                "upsample_nearest_f16",
            )?,
            add: compute.compile_pipeline("add", sources::ELEMENTWISE, "add_f16")?,
            scale: compute.compile_pipeline("scale", sources::ELEMENTWISE, "scale_f16")?,
        };

        Ok(Self { model, compute, config, kernels })
    }

    /// Upscale an image by 4x using Real-ESRGAN.
    ///
    /// Input: `image_rgb` is a flat f32 array in [C, H, W] layout with C=3, values in [0, 1].
    /// Output: flat f32 array in [C, H*4, W*4] layout with C=3, values clamped to [0, 1].
    pub fn upscale(&self, image_rgb: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
        let config = &self.config;
        let nf = config.num_features;
        let device_id = self.compute.device().info().id;

        debug!(width, height, num_blocks = config.num_blocks, "ESRGAN upscale start");

        // Convert input f32 [3, H, W] -> f16 tensor
        let input_f16: Vec<half::f16> = image_rgb.iter().map(|&v| half::f16::from_f32(v)).collect();
        let input = Tensor::from_slice(
            &input_f16,
            Shape::from([1, 3, height, width]),
            DType::F16,
            device_id,
        )?;

        // 1. conv_first: [1, 3, H, W] -> [1, 64, H, W]
        let cb = self.compute.new_command_buffer();
        let feat = self.conv2d(&cb, &input, "conv_first", 3, nf, height, width, 3, 1, 1)?;
        let feat = self.leaky_relu(&cb, &feat);
        cb.commit();
        cb.wait_until_completed();

        debug!("conv_first done");

        // Save for trunk skip connection
        let trunk_input = feat.clone();

        // 2. 23 RRDB blocks
        let mut x = feat;
        for block_idx in 0..config.num_blocks {
            x = self.rrdb_block(&x, block_idx, nf, height, width)?;
            if block_idx % 5 == 0 {
                debug!(block = block_idx, "RRDB block done");
            }
        }

        // 3. conv_body + trunk skip
        let cb = self.compute.new_command_buffer();
        let body = self.conv2d(&cb, &x, "conv_body", nf, nf, height, width, 3, 1, 1)?;
        x = self.add_tensors(&cb, &body, &trunk_input);
        cb.commit();
        cb.wait_until_completed();

        debug!("trunk conv + skip done");

        // 4. Upsample 2x + conv_up1: [1, 64, H, W] -> [1, 64, 2H, 2W]
        let cb = self.compute.new_command_buffer();
        let up1 = self.upsample_nearest_2x(&cb, &x, 1, nf, height, width)?;
        cb.commit();
        cb.wait_until_completed();

        let h2 = height * 2;
        let w2 = width * 2;
        let cb = self.compute.new_command_buffer();
        let up1_conv = self.conv2d(&cb, &up1, "conv_up1", nf, nf, h2, w2, 3, 1, 1)?;
        let up1_act = self.leaky_relu(&cb, &up1_conv);
        cb.commit();
        cb.wait_until_completed();

        debug!("upsample 2x (stage 1) done");

        // 5. Upsample 2x + conv_up2: [1, 64, 2H, 2W] -> [1, 64, 4H, 4W]
        let cb = self.compute.new_command_buffer();
        let up2 = self.upsample_nearest_2x(&cb, &up1_act, 1, nf, h2, w2)?;
        cb.commit();
        cb.wait_until_completed();

        let h4 = height * 4;
        let w4 = width * 4;
        let cb = self.compute.new_command_buffer();
        let up2_conv = self.conv2d(&cb, &up2, "conv_up2", nf, nf, h4, w4, 3, 1, 1)?;
        let up2_act = self.leaky_relu(&cb, &up2_conv);
        cb.commit();
        cb.wait_until_completed();

        debug!("upsample 2x (stage 2) done");

        // 6. conv_hr -> LeakyReLU -> conv_last
        let cb = self.compute.new_command_buffer();
        let hr = self.conv2d(&cb, &up2_act, "conv_hr", nf, nf, h4, w4, 3, 1, 1)?;
        let hr_act = self.leaky_relu(&cb, &hr);
        let out = self.conv2d(&cb, &hr_act, "conv_last", nf, 3, h4, w4, 3, 1, 1)?;
        cb.commit();
        cb.wait_until_completed();

        debug!(out_h = h4, out_w = w4, "ESRGAN upscale complete");

        // Read back and convert f16 -> f32, clamp to [0, 1]
        let out_data: Vec<half::f16> = out.to_vec()?;
        let result: Vec<f32> = out_data
            .iter()
            .map(|v| v.to_f32().clamp(0.0, 1.0))
            .collect();

        Ok(result)
    }

    // ==================== RRDB Block ====================

    /// Single RRDB (Residual-in-Residual Dense Block).
    ///
    /// RRDB = rdb1 -> rdb2 -> rdb3, with residual scaling 0.2 applied to output.
    fn rrdb_block(
        &self,
        input: &Tensor,
        block_idx: usize,
        nf: usize,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let prefix = format!("body.{}", block_idx);

        let rdb1 = self.residual_dense_block(input, &format!("{}.rdb1", prefix), nf, h, w)?;
        let rdb2 = self.residual_dense_block(&rdb1, &format!("{}.rdb2", prefix), nf, h, w)?;
        let rdb3 = self.residual_dense_block(&rdb2, &format!("{}.rdb3", prefix), nf, h, w)?;

        // Residual scaling: output = input + 0.2 * rdb3
        let cb = self.compute.new_command_buffer();
        let scaled = self.scale_tensor(&cb, &rdb3, self.config.residual_scale);
        let out = self.add_tensors(&cb, input, &scaled);
        cb.commit();
        cb.wait_until_completed();

        Ok(out)
    }

    // ==================== Residual Dense Block ====================

    /// Single ResidualDenseBlock with 5 conv layers and dense connections.
    ///
    /// Dense connection pattern (channels grow):
    ///   x1 = lrelu(conv1(input))              // 64 -> 32
    ///   x2 = lrelu(conv2(cat(input, x1)))     // 96 -> 32
    ///   x3 = lrelu(conv3(cat(input, x1, x2))) // 128 -> 32
    ///   x4 = lrelu(conv4(cat(input, x1, x2, x3)))   // 160 -> 32
    ///   x5 = conv5(cat(input, x1, x2, x3, x4))      // 192 -> 64
    ///   output = input + 0.2 * x5
    fn residual_dense_block(
        &self,
        input: &Tensor,
        prefix: &str,
        nf: usize,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let gc = self.config.growth_channels;
        let device_id = self.compute.device().info().id;

        // conv1: nf -> gc
        let cb = self.compute.new_command_buffer();
        let x1 = self.conv2d(&cb, input, &format!("{}.conv1", prefix), nf, gc, h, w, 3, 1, 1)?;
        let x1 = self.leaky_relu(&cb, &x1);
        cb.commit();
        cb.wait_until_completed();

        // cat(input, x1): nf + gc channels
        let cat1 = self.concat_channels(input, &x1, nf, gc, h, w, device_id)?;

        // conv2: (nf + gc) -> gc
        let cb = self.compute.new_command_buffer();
        let x2 = self.conv2d(&cb, &cat1, &format!("{}.conv2", prefix), nf + gc, gc, h, w, 3, 1, 1)?;
        let x2 = self.leaky_relu(&cb, &x2);
        cb.commit();
        cb.wait_until_completed();

        // cat(input, x1, x2): nf + 2*gc channels
        let cat2 = self.concat_channels(&cat1, &x2, nf + gc, gc, h, w, device_id)?;

        // conv3: (nf + 2*gc) -> gc
        let cb = self.compute.new_command_buffer();
        let x3 = self.conv2d(&cb, &cat2, &format!("{}.conv3", prefix), nf + 2 * gc, gc, h, w, 3, 1, 1)?;
        let x3 = self.leaky_relu(&cb, &x3);
        cb.commit();
        cb.wait_until_completed();

        // cat(input, x1, x2, x3): nf + 3*gc channels
        let cat3 = self.concat_channels(&cat2, &x3, nf + 2 * gc, gc, h, w, device_id)?;

        // conv4: (nf + 3*gc) -> gc
        let cb = self.compute.new_command_buffer();
        let x4 = self.conv2d(&cb, &cat3, &format!("{}.conv4", prefix), nf + 3 * gc, gc, h, w, 3, 1, 1)?;
        let x4 = self.leaky_relu(&cb, &x4);
        cb.commit();
        cb.wait_until_completed();

        // cat(input, x1, x2, x3, x4): nf + 4*gc channels
        let cat4 = self.concat_channels(&cat3, &x4, nf + 3 * gc, gc, h, w, device_id)?;

        // conv5: (nf + 4*gc) -> nf (no activation)
        let cb = self.compute.new_command_buffer();
        let x5 = self.conv2d(&cb, &cat4, &format!("{}.conv5", prefix), nf + 4 * gc, nf, h, w, 3, 1, 1)?;
        cb.commit();
        cb.wait_until_completed();

        // Residual: input + 0.2 * x5
        let cb = self.compute.new_command_buffer();
        let scaled = self.scale_tensor(&cb, &x5, self.config.residual_scale);
        let out = self.add_tensors(&cb, input, &scaled);
        cb.commit();
        cb.wait_until_completed();

        Ok(out)
    }

    // ==================== GPU Kernel Dispatches ====================

    /// GPU 3x3 Conv2d with padding=1, stride=1.
    ///
    /// Loads weight and bias from model by name, dispatches conv2d_3x3_tiled_f16.
    fn conv2d(
        &self,
        cb: &metal::CommandBufferRef,
        input: &Tensor,
        weight_prefix: &str,
        cin: usize,
        cout: usize,
        hin: usize,
        win: usize,
        kernel_size: usize,
        padding: usize,
        stride: usize,
    ) -> Result<Tensor> {
        let w_name = format!("{}.weight", weight_prefix);
        let b_name = format!("{}.bias", weight_prefix);

        let weight = gpu_ops::read_weight_f16(&self.model, &self.compute, &w_name)?;
        let bias = gpu_ops::read_weight_f16(&self.model, &self.compute, &b_name)?;

        let hout = (hin + 2 * padding - kernel_size) / stride + 1;
        let wout = (win + 2 * padding - kernel_size) / stride + 1;
        let n: usize = 1;

        let output = Tensor::empty(
            Shape::from([n, cout, hout, wout]),
            DType::F16,
            self.compute.device().info().id,
        )?;

        let in_ptr = input.device_ptr().ok_or(Error::internal("conv2d: input not on device"))?;
        let w_ptr = weight.device_ptr().ok_or(Error::internal("conv2d: weight not on device"))?;
        let b_ptr = bias.device_ptr().ok_or(Error::internal("conv2d: bias not on device"))?;
        let out_ptr = output.device_ptr().ok_or(Error::internal("conv2d: output not on device"))?;

        let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
        let w_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(w_ptr) };
        let b_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(b_ptr) };
        let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };

        let kh = kernel_size as u32;
        let kw = kernel_size as u32;

        let grid_size = ((wout + 15) / 16, (hout + 15) / 16, cout * n);
        let threadgroup_size = (16, 16, 1);

        self.compute.dispatch(
            cb,
            &self.kernels.conv2d,
            grid_size,
            threadgroup_size,
            |encoder| {
                encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(w_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(b_buf.as_ref()), 0);
                encoder.set_buffer(3, Some(out_buf.as_ref()), 0);
                encoder.set_bytes(4, 4, &(cin as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &(hin as u32) as *const u32 as *const _);
                encoder.set_bytes(6, 4, &(win as u32) as *const u32 as *const _);
                encoder.set_bytes(7, 4, &(cout as u32) as *const u32 as *const _);
                encoder.set_bytes(8, 4, &(hout as u32) as *const u32 as *const _);
                encoder.set_bytes(9, 4, &(wout as u32) as *const u32 as *const _);
                encoder.set_bytes(10, 4, &kw as *const u32 as *const _);
                encoder.set_bytes(11, 4, &kh as *const u32 as *const _);
                encoder.set_bytes(12, 4, &(padding as u32) as *const u32 as *const _);
                encoder.set_bytes(13, 4, &(padding as u32) as *const u32 as *const _);
                encoder.set_bytes(14, 4, &(stride as u32) as *const u32 as *const _);
                encoder.set_bytes(15, 4, &(stride as u32) as *const u32 as *const _);
                encoder.set_bytes(16, 4, &(n as u32) as *const u32 as *const _);
            },
        );

        Ok(output)
    }

    /// GPU LeakyReLU with configurable alpha.
    ///
    /// Uses leaky_relu_f16 kernel with alpha parameter (default 0.2).
    fn leaky_relu(&self, cb: &metal::CommandBufferRef, input: &Tensor) -> Tensor {
        let numel = input.numel();
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer(
            (numel * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let alpha = self.config.leaky_slope;

        self.compute.dispatch_1d(cb, &self.kernels.leaky_relu, numel, |encoder| {
            gpu_ops::set_tensor_buffer(encoder, 0, input);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_bytes(2, 4, &alpha as *const f32 as *const _);
        });

        Tensor::from_metal_buffer(
            output_buffer,
            input.shape().clone(),
            DType::F16,
            self.compute.device().info().id,
        )
    }

    /// GPU 2x nearest-neighbor upsample.
    fn upsample_nearest_2x(
        &self,
        cb: &metal::CommandBufferRef,
        input: &Tensor,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let new_h = h * 2;
        let new_w = w * 2;
        let output = Tensor::empty(
            Shape::from([n, c, new_h, new_w]),
            DType::F16,
            self.compute.device().info().id,
        )?;

        let in_ptr = input.device_ptr().ok_or(Error::internal("upsample: input not on device"))?;
        let out_ptr = output.device_ptr().ok_or(Error::internal("upsample: output not on device"))?;

        let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
        let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };

        let grid_size = ((new_w + 7) / 8, (new_h + 7) / 8, n * c);
        let threadgroup_size = (8, 8, 1);

        self.compute.dispatch(
            cb,
            &self.kernels.upsample_nearest,
            grid_size,
            threadgroup_size,
            |encoder| {
                encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(out_buf.as_ref()), 0);
                encoder.set_bytes(2, 4, &(n as u32) as *const u32 as *const _);
                encoder.set_bytes(3, 4, &(c as u32) as *const u32 as *const _);
                encoder.set_bytes(4, 4, &(h as u32) as *const u32 as *const _);
                encoder.set_bytes(5, 4, &(w as u32) as *const u32 as *const _);
            },
        );

        Ok(output)
    }

    /// GPU element-wise add: c = a + b.
    fn add_tensors(&self, cb: &metal::CommandBufferRef, a: &Tensor, b: &Tensor) -> Tensor {
        gpu_ops::elementwise_binary_on(
            &self.compute,
            &self.kernels.add,
            cb,
            a,
            b,
        )
    }

    /// GPU scalar multiply: output = input * s.
    fn scale_tensor(&self, cb: &metal::CommandBufferRef, input: &Tensor, s: f32) -> Tensor {
        gpu_ops::scale_tensor_on(
            &self.compute,
            &self.kernels.scale,
            cb,
            input,
            s,
        )
    }

    /// Concatenate two tensors along the channel dimension (CPU readback + reassembly).
    ///
    /// Input a: [1, C_a, H, W], b: [1, C_b, H, W]
    /// Output: [1, C_a + C_b, H, W]
    fn concat_channels(
        &self,
        a: &Tensor,
        b: &Tensor,
        c_a: usize,
        c_b: usize,
        h: usize,
        w: usize,
        device_id: crate::hal::DeviceId,
    ) -> Result<Tensor> {
        let hw = h * w;
        let a_data: Vec<half::f16> = a.to_vec()?;
        let b_data: Vec<half::f16> = b.to_vec()?;

        let total_c = c_a + c_b;
        let mut out = vec![half::f16::ZERO; total_c * hw];

        // Copy channels from a
        for c in 0..c_a {
            let src_offset = c * hw;
            let dst_offset = c * hw;
            out[dst_offset..dst_offset + hw].copy_from_slice(&a_data[src_offset..src_offset + hw]);
        }
        // Copy channels from b
        for c in 0..c_b {
            let src_offset = c * hw;
            let dst_offset = (c_a + c) * hw;
            out[dst_offset..dst_offset + hw].copy_from_slice(&b_data[src_offset..src_offset + hw]);
        }

        Tensor::from_slice(
            &out,
            Shape::from([1, total_c, h, w]),
            DType::F16,
            device_id,
        )
    }
}

// ==================== CPU Fallback ====================

/// CPU-only Real-ESRGAN pipeline (no Metal dependency).
///
/// Implements the same architecture with pure Rust loops.
/// Viable for the 17M-param model since it is relatively small.
pub struct EsrganCpuPipeline {
    /// All weights stored as f32 keyed by name.
    weights: std::collections::HashMap<String, Vec<f32>>,
    config: EsrganConfig,
}

impl EsrganCpuPipeline {
    /// Create a new CPU pipeline from a weight map.
    ///
    /// `weights` maps weight names (e.g. "conv_first.weight") to flat f32 arrays.
    pub fn new(
        weights: std::collections::HashMap<String, Vec<f32>>,
        config: EsrganConfig,
    ) -> Self {
        Self { weights, config }
    }

    /// Upscale an image by 4x on CPU.
    ///
    /// Input: `image_rgb` is flat f32 [C, H, W] with C=3, values in [0, 1].
    /// Output: flat f32 [C, H*4, W*4] with C=3, clamped to [0, 1].
    pub fn upscale(&self, image_rgb: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
        let nf = self.config.num_features;
        let gc = self.config.growth_channels;
        let alpha = self.config.leaky_slope;
        let rs = self.config.residual_scale;

        // 1. conv_first + LeakyReLU
        let mut feat = self.cpu_conv2d(image_rgb, "conv_first", 3, nf, height, width)?;
        cpu_leaky_relu_inplace(&mut feat, alpha);

        let trunk_input = feat.clone();

        // 2. RRDB blocks
        for block_idx in 0..self.config.num_blocks {
            feat = self.cpu_rrdb(&feat, block_idx, nf, gc, height, width, alpha, rs)?;
        }

        // 3. conv_body + trunk skip
        let body = self.cpu_conv2d(&feat, "conv_body", nf, nf, height, width)?;
        feat = cpu_add(&body, &trunk_input);

        // 4. Upsample 2x + conv_up1 + LeakyReLU
        let up1 = cpu_upsample_nearest_2x(&feat, nf, height, width);
        let (h2, w2) = (height * 2, width * 2);
        let mut up1_conv = self.cpu_conv2d(&up1, "conv_up1", nf, nf, h2, w2)?;
        cpu_leaky_relu_inplace(&mut up1_conv, alpha);

        // 5. Upsample 2x + conv_up2 + LeakyReLU
        let up2 = cpu_upsample_nearest_2x(&up1_conv, nf, h2, w2);
        let (h4, w4) = (height * 4, width * 4);
        let mut up2_conv = self.cpu_conv2d(&up2, "conv_up2", nf, nf, h4, w4)?;
        cpu_leaky_relu_inplace(&mut up2_conv, alpha);

        // 6. conv_hr + LeakyReLU + conv_last
        let mut hr = self.cpu_conv2d(&up2_conv, "conv_hr", nf, nf, h4, w4)?;
        cpu_leaky_relu_inplace(&mut hr, alpha);
        let out = self.cpu_conv2d(&hr, "conv_last", nf, 3, h4, w4)?;

        // Clamp to [0, 1]
        let result: Vec<f32> = out.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
        Ok(result)
    }

    /// CPU RRDB block.
    fn cpu_rrdb(
        &self,
        input: &[f32],
        block_idx: usize,
        nf: usize,
        gc: usize,
        h: usize,
        w: usize,
        alpha: f32,
        rs: f32,
    ) -> Result<Vec<f32>> {
        let prefix = format!("body.{}", block_idx);
        let rdb1 = self.cpu_rdb(input, &format!("{}.rdb1", prefix), nf, gc, h, w, alpha)?;
        let rdb2 = self.cpu_rdb(&rdb1, &format!("{}.rdb2", prefix), nf, gc, h, w, alpha)?;
        let rdb3 = self.cpu_rdb(&rdb2, &format!("{}.rdb3", prefix), nf, gc, h, w, alpha)?;

        // input + rs * rdb3
        let result: Vec<f32> = input.iter().zip(rdb3.iter()).map(|(&a, &b)| a + rs * b).collect();
        Ok(result)
    }

    /// CPU ResidualDenseBlock.
    fn cpu_rdb(
        &self,
        input: &[f32],
        prefix: &str,
        nf: usize,
        gc: usize,
        h: usize,
        w: usize,
        alpha: f32,
    ) -> Result<Vec<f32>> {
        let hw = h * w;

        // conv1: nf -> gc
        let mut x1 = self.cpu_conv2d(input, &format!("{}.conv1", prefix), nf, gc, h, w)?;
        cpu_leaky_relu_inplace(&mut x1, alpha);

        // cat(input, x1)
        let cat1 = cpu_concat_channels(input, &x1, nf, gc, hw);

        // conv2: (nf+gc) -> gc
        let mut x2 = self.cpu_conv2d(&cat1, &format!("{}.conv2", prefix), nf + gc, gc, h, w)?;
        cpu_leaky_relu_inplace(&mut x2, alpha);

        // cat(input, x1, x2)
        let cat2 = cpu_concat_channels(&cat1, &x2, nf + gc, gc, hw);

        // conv3: (nf+2gc) -> gc
        let mut x3 = self.cpu_conv2d(&cat2, &format!("{}.conv3", prefix), nf + 2 * gc, gc, h, w)?;
        cpu_leaky_relu_inplace(&mut x3, alpha);

        // cat(input, x1, x2, x3)
        let cat3 = cpu_concat_channels(&cat2, &x3, nf + 2 * gc, gc, hw);

        // conv4: (nf+3gc) -> gc
        let mut x4 = self.cpu_conv2d(&cat3, &format!("{}.conv4", prefix), nf + 3 * gc, gc, h, w)?;
        cpu_leaky_relu_inplace(&mut x4, alpha);

        // cat(input, x1, x2, x3, x4)
        let cat4 = cpu_concat_channels(&cat3, &x4, nf + 3 * gc, gc, hw);

        // conv5: (nf+4gc) -> nf (no activation)
        let x5 = self.cpu_conv2d(&cat4, &format!("{}.conv5", prefix), nf + 4 * gc, nf, h, w)?;

        // input + 0.2 * x5
        let result: Vec<f32> = input
            .iter()
            .zip(x5.iter())
            .map(|(&a, &b)| a + self.config.residual_scale * b)
            .collect();
        Ok(result)
    }

    /// CPU Conv2d 3x3, padding=1, stride=1.
    ///
    /// Input: [Cin, H, W] flat f32 (batch dimension squeezed).
    /// Weight: [Cout, Cin, 3, 3], Bias: [Cout].
    /// Output: [Cout, H, W].
    fn cpu_conv2d(
        &self,
        input: &[f32],
        prefix: &str,
        cin: usize,
        cout: usize,
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>> {
        let w_key = format!("{}.weight", prefix);
        let b_key = format!("{}.bias", prefix);

        let weight = self.weights.get(&w_key).ok_or_else(|| {
            Error::internal(format!("CPU ESRGAN: weight not found: {}", w_key))
        })?;
        let bias = self.weights.get(&b_key).ok_or_else(|| {
            Error::internal(format!("CPU ESRGAN: bias not found: {}", b_key))
        })?;

        cpu_conv2d_3x3(input, weight, bias, cin, cout, h, w)
    }
}

// ==================== CPU Helper Functions ====================

/// CPU 3x3 convolution with padding=1, stride=1.
///
/// input: [cin, h, w], weight: [cout, cin, 3, 3], bias: [cout]
/// output: [cout, h, w]
fn cpu_conv2d_3x3(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    cin: usize,
    cout: usize,
    h: usize,
    w: usize,
) -> Result<Vec<f32>> {
    let hw = h * w;
    let mut output = vec![0.0f32; cout * hw];

    for oc in 0..cout {
        let b = bias[oc];
        for oh in 0..h {
            for ow in 0..w {
                let mut sum = b;
                for ic in 0..cin {
                    for kh in 0..3usize {
                        for kw in 0..3usize {
                            let ih = oh as isize + kh as isize - 1;
                            let iw = ow as isize + kw as isize - 1;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                let in_idx = ic * hw + ih as usize * w + iw as usize;
                                let w_idx = oc * cin * 9 + ic * 9 + kh * 3 + kw;
                                sum += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                }
                output[oc * hw + oh * w + ow] = sum;
            }
        }
    }

    Ok(output)
}

/// In-place LeakyReLU: x = max(x, alpha * x).
fn cpu_leaky_relu_inplace(data: &mut [f32], alpha: f32) {
    for v in data.iter_mut() {
        if *v < 0.0 {
            *v *= alpha;
        }
    }
}

/// Element-wise add: c[i] = a[i] + b[i].
fn cpu_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// 2x nearest-neighbor upsample on CPU.
///
/// Input: [C, H, W], output: [C, 2H, 2W].
fn cpu_upsample_nearest_2x(input: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let new_h = h * 2;
    let new_w = w * 2;
    let mut output = vec![0.0f32; c * new_h * new_w];

    for ch in 0..c {
        for oh in 0..new_h {
            for ow in 0..new_w {
                let ih = oh / 2;
                let iw = ow / 2;
                output[ch * new_h * new_w + oh * new_w + ow] =
                    input[ch * h * w + ih * w + iw];
            }
        }
    }

    output
}

/// Concatenate two tensors along channel dimension on CPU.
///
/// a: [C_a, HW], b: [C_b, HW] -> [C_a + C_b, HW].
fn cpu_concat_channels(a: &[f32], b: &[f32], c_a: usize, c_b: usize, hw: usize) -> Vec<f32> {
    let total = (c_a + c_b) * hw;
    let mut out = vec![0.0f32; total];

    out[..c_a * hw].copy_from_slice(&a[..c_a * hw]);
    out[c_a * hw..(c_a + c_b) * hw].copy_from_slice(&b[..c_b * hw]);

    out
}
