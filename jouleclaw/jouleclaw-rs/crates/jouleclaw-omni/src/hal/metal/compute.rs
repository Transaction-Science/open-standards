//! Metal compute pipelines and dispatch.

use super::MetalDevice;
use crate::core::Result;
use std::sync::Arc;

#[cfg(feature = "metal")]
use metal::{CommandBuffer, CommandBufferRef, ComputeCommandEncoderRef, ComputePipelineState};

/// Compute pipeline wrapper.
#[cfg(feature = "metal")]
pub struct ComputePipeline {
    /// Pipeline state
    pipeline: ComputePipelineState,
    /// Thread execution width (SIMD group size)
    thread_execution_width: usize,
    /// Max total threads per threadgroup
    max_threads_per_threadgroup: usize,
    /// Name for debugging
    name: String,
}

#[cfg(feature = "metal")]
impl ComputePipeline {
    /// Create from a compiled pipeline state.
    pub fn new(pipeline: ComputePipelineState, name: impl Into<String>) -> Self {
        let thread_execution_width = pipeline.thread_execution_width() as usize;
        let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;

        Self {
            pipeline,
            thread_execution_width,
            max_threads_per_threadgroup: max_threads,
            name: name.into(),
        }
    }

    /// Get the underlying pipeline state.
    pub fn raw(&self) -> &ComputePipelineState {
        &self.pipeline
    }

    /// Get thread execution width (SIMD size).
    pub fn thread_execution_width(&self) -> usize {
        self.thread_execution_width
    }

    /// Get max threads per threadgroup.
    pub fn max_threads_per_threadgroup(&self) -> usize {
        self.max_threads_per_threadgroup
    }

    /// Calculate optimal threadgroup size for 1D dispatch.
    pub fn optimal_threadgroup_1d(&self, total: usize) -> usize {
        let max = self.max_threads_per_threadgroup;
        let simd = self.thread_execution_width;

        // Use multiple of SIMD width, capped at max
        let optimal = ((total + simd - 1) / simd) * simd;
        optimal.min(max)
    }

    /// Calculate optimal threadgroup size for 2D dispatch.
    pub fn optimal_threadgroup_2d(&self, width: usize, height: usize) -> (usize, usize) {
        let max = self.max_threads_per_threadgroup;
        let simd = self.thread_execution_width;

        // Try to balance width and height while respecting max
        let w = simd.min(width);
        let h = (max / w).min(height);

        (w, h)
    }
}

/// High-level compute dispatch helper.
#[cfg(feature = "metal")]
pub struct MetalCompute {
    /// Device reference
    device: Arc<MetalDevice>,
    /// Compiled pipelines cache
    pipelines: dashmap::DashMap<String, Arc<ComputePipeline>>,
}

#[cfg(feature = "metal")]
impl MetalCompute {
    /// Create a new compute helper.
    pub fn new(device: Arc<MetalDevice>) -> Self {
        Self {
            device,
            pipelines: dashmap::DashMap::new(),
        }
    }

    /// Get the underlying device.
    pub fn device(&self) -> &Arc<MetalDevice> {
        &self.device
    }

    /// Create a new command buffer.
    pub fn new_command_buffer(&self) -> CommandBuffer {
        self.device.new_command_buffer()
    }

    /// Compile and cache a compute pipeline.
    pub fn compile_pipeline(&self, name: &str, source: &str, entry: &str) -> Result<Arc<ComputePipeline>> {
        // Check cache
        if let Some(pipeline) = self.pipelines.get(name) {
            return Ok(Arc::clone(&pipeline));
        }

        // Compile
        let library = self.device.compile_library(source)?;
        let pipeline_state = self.device.create_compute_pipeline(&library, entry)?;
        let pipeline = Arc::new(ComputePipeline::new(pipeline_state, name));

        self.pipelines.insert(name.to_string(), Arc::clone(&pipeline));
        Ok(pipeline)
    }

    /// Dispatch a compute kernel.
    pub fn dispatch(
        &self,
        command_buffer: &CommandBufferRef,
        pipeline: &ComputePipeline,
        grid_size: (usize, usize, usize),
        threadgroup_size: (usize, usize, usize),
        setup: impl FnOnce(&ComputeCommandEncoderRef),
    ) {
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline.raw());

        // Let caller set buffers and other state
        setup(&encoder);

        // Dispatch
        let grid = metal::MTLSize::new(
            grid_size.0 as u64,
            grid_size.1 as u64,
            grid_size.2 as u64,
        );
        let threadgroup = metal::MTLSize::new(
            threadgroup_size.0 as u64,
            threadgroup_size.1 as u64,
            threadgroup_size.2 as u64,
        );

        encoder.dispatch_thread_groups(grid, threadgroup);
        encoder.end_encoding();
    }

    /// Dispatch a compute kernel without waiting.
    pub fn dispatch_async(
        &self,
        command_buffer: &CommandBufferRef,
        pipeline: &ComputePipeline,
        grid_size: (usize, usize, usize),
        threadgroup_size: (usize, usize, usize),
        setup: impl FnOnce(&ComputeCommandEncoderRef),
    ) {
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline.raw());

        // Let caller set buffers and other state
        setup(&encoder);

        // Dispatch
        let grid = metal::MTLSize::new(
            grid_size.0 as u64,
            grid_size.1 as u64,
            grid_size.2 as u64,
        );
        let threadgroup = metal::MTLSize::new(
            threadgroup_size.0 as u64,
            threadgroup_size.1 as u64,
            threadgroup_size.2 as u64,
        );

        encoder.dispatch_thread_groups(grid, threadgroup);
        encoder.end_encoding();
    }

    /// Dispatch with automatic threadgroup sizing.
    pub fn dispatch_1d(
        &self,
        command_buffer: &CommandBufferRef,
        pipeline: &ComputePipeline,
        total_threads: usize,
        setup: impl FnOnce(&ComputeCommandEncoderRef),
    ) {
        let threadgroup_size = pipeline.optimal_threadgroup_1d(total_threads);
        let grid_size = (total_threads + threadgroup_size - 1) / threadgroup_size;

        self.dispatch(
            command_buffer,
            pipeline,
            (grid_size, 1, 1),
            (threadgroup_size, 1, 1),
            setup,
        );
    }

    /// Dispatch 2D with automatic sizing.
    pub fn dispatch_2d(
        &self,
        command_buffer: &CommandBufferRef,
        pipeline: &ComputePipeline,
        width: usize,
        height: usize,
        setup: impl FnOnce(&ComputeCommandEncoderRef),
    ) {
        let (tg_w, tg_h) = pipeline.optimal_threadgroup_2d(width, height);
        let grid_w = (width + tg_w - 1) / tg_w;
        let grid_h = (height + tg_h - 1) / tg_h;

        self.dispatch(
            command_buffer,
            pipeline,
            (grid_w, grid_h, 1),
            (tg_w, tg_h, 1),
            setup,
        );
    }

    /// Execute and wait for completion.
    pub fn execute_sync<T>(
        &self,
        pipeline: &ComputePipeline,
        grid_size: (usize, usize, usize),
        threadgroup_size: (usize, usize, usize),
        setup: impl FnOnce(&ComputeCommandEncoderRef),
        result: impl FnOnce() -> T,
    ) -> T {
        let command_buffer = self.device.new_command_buffer();

        self.dispatch(&command_buffer, pipeline, grid_size, threadgroup_size, setup);

        command_buffer.commit();
        command_buffer.wait_until_completed();

        result()
    }
}

/// Batched compute for multiple operations.
#[cfg(feature = "metal")]
pub struct ComputeBatch {
    /// Command buffer
    command_buffer: CommandBuffer,
    /// Operations queued
    operations: usize,
}

#[cfg(feature = "metal")]
impl ComputeBatch {
    /// Create a new batch.
    pub fn new(device: &MetalDevice) -> Self {
        Self {
            command_buffer: device.new_command_buffer(),
            operations: 0,
        }
    }

    /// Add an operation to the batch.
    pub fn add<F>(&mut self, pipeline: &ComputePipeline, setup: F)
    where
        F: FnOnce(&ComputeCommandEncoderRef),
    {
        let encoder = self.command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline.raw());
        setup(&encoder);
        encoder.end_encoding();
        self.operations += 1;
    }

    /// Submit the batch.
    pub fn submit(self) {
        self.command_buffer.commit();
    }

    /// Submit and wait for completion.
    pub fn submit_and_wait(self) {
        self.command_buffer.commit();
        self.command_buffer.wait_until_completed();
    }

    /// Get number of queued operations.
    pub fn len(&self) -> usize {
        self.operations
    }

    /// Check if batch is empty.
    pub fn is_empty(&self) -> bool {
        self.operations == 0
    }
}

/// Grid configuration helpers.
pub mod grid {
    /// Calculate grid size for 1D dispatch.
    pub fn linear(total: usize, threadgroup: usize) -> usize {
        (total + threadgroup - 1) / threadgroup
    }

    /// Calculate grid size for 2D dispatch.
    pub fn grid_2d(width: usize, height: usize, tg_w: usize, tg_h: usize) -> (usize, usize) {
        (
            (width + tg_w - 1) / tg_w,
            (height + tg_h - 1) / tg_h,
        )
    }

    /// Calculate grid for matrix operation.
    pub fn matrix(m: usize, n: usize, tile_m: usize, tile_n: usize) -> (usize, usize) {
        grid_2d(n, m, tile_n, tile_m)
    }
}

// Stubs for non-macOS
#[cfg(not(feature = "metal"))]
pub struct ComputePipeline;

#[cfg(not(feature = "metal"))]
pub struct MetalCompute;
