// SANA-WM GPU dispatchers — draft from agent run 2026-05-26.
//
// Integration plan:
//   1. Add `ones_weight: Tensor` field to SanaWmPipeline (allocated in new, f16, len = max(hidden_size, ffn_inner), all 1.0)
//   2. Add `zeros_token: Tensor` field (f16, len = max activation tile, all 0.0)
//   3. Paste these methods into `impl SanaWmPipeline { ... }` in sana_wm.rs
//   4. swiglu_split_f16 needs either a caller-side channel-axis permute + half-swap OR
//      a new `swiglu_split_swapped_f16` kernel — 2-line shader.rs change.
//
// Kernel signatures verified (shader.rs file:line):
//   - rms_norm_f16:        shader.rs:413-439  buffers: input(0), weight(1), output(2), N(3), D(4), eps(5)
//   - adaln_modulate_f16:  shader.rs:3728-3743  buffers: x(0), scale(1), shift(2), output(3), hidden(4), count(5)
//   - adaln_gate_f16:      shader.rs:3746-3761  buffers: x(0), residual(1), gate(2), output(3), hidden(4), count(5)
//                          IMPORTANT: this is FUSED gated residual `out = x + gate*residual`, not pure gate
//   - swiglu_split_f16:    shader.rs:2114-2129  buffers: input(0), output(1), half_dim(2), count(3)
//                          IMPORTANT: splits along innermost dim, reads [gate, up] (gate gets SiLU)

#![allow(dead_code)]

use std::sync::Arc;
use crate::core::Result;
use crate::tensor::{DType, Shape, Tensor};
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline};
use crate::inference::gpu_ops::{self, MetalPipeline};

// Pseudo-impl block — actual integration happens in sana_wm.rs's
// `impl SanaWmPipeline` block. This file exists only as a draft reference.

/*

impl SanaWmPipeline {
    /// RMS normalization over the last dim. Input `[n, d]`, output `[n, d]`.
    ///
    /// SANA-WM uses RMSNorm with no learned scale; we feed `self.ones_weight`
    /// (pipeline-cached, all-ones, length >= d).
    fn rms_norm_on(
        &self, cb: &metal::CommandBufferRef,
        input: &Tensor, n: usize, d: usize, eps: f32,
    ) -> Tensor {
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer(
            (n * d * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.compute.dispatch_1d(
            cb, &self.kernels.rms_norm, n,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                gpu_ops::set_tensor_buffer(encoder, 1, &self.ones_weight);
                encoder.set_buffer(2, Some(&output_buffer), 0);
                let n_u32 = n as u32;
                let d_u32 = d as u32;
                encoder.set_bytes(3, 4, &n_u32 as *const u32 as *const _);
                encoder.set_bytes(4, 4, &d_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
            },
        );
        Tensor::from_metal_buffer(
            output_buffer,
            Shape::from([n, d]),
            DType::F16,
            self.compute.device().info().id,
        )
    }

    /// AdaLN modulation: `out[b, ni, ci] = x[b, ni, ci] * (1 + scale[b, ci]) + shift[b, ci]`.
    /// Kernel broadcast: `gid % hidden_size`, where hidden = batch*c for batch=1.
    fn adaln_modulate_on(
        &self, cb: &metal::CommandBufferRef,
        x: &Tensor, shift: &Tensor, scale: &Tensor,
        batch: usize, n_tokens: usize, c: usize,
    ) -> Tensor {
        debug_assert_eq!(batch, 1, "adaln_modulate_on broadcast assumes batch=1");
        let device = self.compute.device().raw();
        let count = batch * n_tokens * c;
        let output_buffer = device.new_buffer(
            (count * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.compute.dispatch_1d(
            cb, &self.kernels.adaln_modulate, count,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, x);
                gpu_ops::set_tensor_buffer(encoder, 1, scale);
                gpu_ops::set_tensor_buffer(encoder, 2, shift);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let hidden_u32 = (batch * c) as u32;
                let count_u32 = count as u32;
                encoder.set_bytes(4, 4, &hidden_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &count_u32 as *const u32 as *const _);
            },
        );
        Tensor::from_metal_buffer(
            output_buffer,
            Shape::from([batch, n_tokens, c]),
            DType::F16,
            self.compute.device().info().id,
        )
    }

    /// AdaLN gate (pure): `out = x * gate`. WORKAROUND for the fused
    /// kernel: dispatch as `out = zeros + gate*x` with `self.zeros_token`.
    /// Costs one fictitious add per element. Cleaner long-term: add a
    /// `gate_only_f16` kernel.
    fn adaln_gate_on(
        &self, cb: &metal::CommandBufferRef,
        x: &Tensor, gate: &Tensor,
        batch: usize, n_tokens: usize, c: usize,
    ) -> Tensor {
        debug_assert_eq!(batch, 1, "adaln_gate_on broadcast assumes batch=1");
        let device = self.compute.device().raw();
        let count = batch * n_tokens * c;
        let output_buffer = device.new_buffer(
            (count * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.compute.dispatch_1d(
            cb, &self.kernels.adaln_gate, count,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, &self.zeros_token);
                gpu_ops::set_tensor_buffer(encoder, 1, x);
                gpu_ops::set_tensor_buffer(encoder, 2, gate);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let hidden_u32 = (batch * c) as u32;
                let count_u32 = count as u32;
                encoder.set_bytes(4, 4, &hidden_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &count_u32 as *const u32 as *const _);
            },
        );
        Tensor::from_metal_buffer(
            output_buffer,
            Shape::from([batch, n_tokens, c]),
            DType::F16,
            self.compute.device().info().id,
        )
    }

    /// GLU split with SiLU. CAVEAT: kernel splits along INNERMOST dim and
    /// reads [gate, up] order (gate gets SiLU). SANA-WM's GLUMBConvTemp
    /// produces [B, 2C, N] and uses [a, gate] order.
    ///
    /// Caller must:
    ///   1. permute input from [B, 2C, N] -> [B, N, 2C] before calling
    ///   2. swap halves to match kernel's [gate, up] order — OR add a
    ///      new `swiglu_split_swapped_f16` kernel and use that instead
    ///   3. permute output back from [B, N, C] -> [B, C, N] if needed
    fn glu_split_silu_on(
        &self, cb: &metal::CommandBufferRef,
        x: &Tensor, batch: usize, n_pixels: usize, expand: usize,
    ) -> Tensor {
        assert!(expand % 2 == 0, "glu_split_silu_on: expand must be even");
        let half_dim = expand / 2;
        let device = self.compute.device().raw();
        let count = batch * n_pixels * half_dim;
        let output_buffer = device.new_buffer(
            (count * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.compute.dispatch_1d(
            cb, &self.kernels.swiglu_split, count,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, x);
                encoder.set_buffer(1, Some(&output_buffer), 0);
                let half_u32 = half_dim as u32;
                let count_u32 = count as u32;
                encoder.set_bytes(2, 4, &half_u32 as *const u32 as *const _);
                encoder.set_bytes(3, 4, &count_u32 as *const u32 as *const _);
            },
        );
        Tensor::from_metal_buffer(
            output_buffer,
            Shape::from([batch, n_pixels, half_dim]),
            DType::F16,
            self.compute.device().info().id,
        )
    }

    /// Residual add: thin wrapper over MetalPipeline::add.
    fn residual_add_on(
        &self, cb: &metal::CommandBufferRef,
        a: &Tensor, b: &Tensor,
    ) -> Tensor {
        self.add(cb, a, b)
    }
}

*/
