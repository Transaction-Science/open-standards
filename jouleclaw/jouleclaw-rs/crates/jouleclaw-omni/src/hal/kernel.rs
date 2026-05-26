//! Kernel compilation and execution.

use crate::core::{Error, Result};
use alloc::string::String;
use alloc::vec::Vec;

/// A compiled kernel ready for execution.
#[derive(Debug)]
pub struct Kernel {
    /// Kernel name
    name: String,
    /// Target device type
    device_type: super::DeviceType,
    /// Compiled binary (PTX, SPIR-V, Metal IR, etc.)
    binary: Vec<u8>,
    /// Entry point name
    entry_point: String,
    /// Shared memory requirement
    shared_memory: usize,
    /// Register usage per thread
    registers_per_thread: usize,
}

impl Kernel {
    /// Get kernel name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get target device type.
    pub fn device_type(&self) -> super::DeviceType {
        self.device_type
    }

    /// Get compiled binary.
    pub fn binary(&self) -> &[u8] {
        &self.binary
    }

    /// Get entry point name.
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// Get shared memory requirement.
    pub fn shared_memory(&self) -> usize {
        self.shared_memory
    }
}

/// Builder for kernel compilation.
#[derive(Debug)]
pub struct KernelBuilder {
    name: String,
    source: KernelSource,
    device_type: super::DeviceType,
    defines: Vec<(String, String)>,
    shared_memory: usize,
}

/// Kernel source representation.
#[derive(Debug, Clone)]
pub enum KernelSource {
    /// CUDA source code
    Cuda(String),
    /// Metal Shading Language
    Metal(String),
    /// SPIR-V binary
    SpirV(Vec<u8>),
    /// Pre-compiled PTX
    Ptx(String),
    /// Generic IR (for JIT)
    Ir(Vec<u8>),
}

impl KernelBuilder {
    /// Create a new kernel builder.
    pub fn new(name: impl Into<String>, source: KernelSource) -> Self {
        let device_type = match &source {
            KernelSource::Cuda(_) | KernelSource::Ptx(_) => super::DeviceType::Cuda,
            KernelSource::Metal(_) => super::DeviceType::Metal,
            KernelSource::SpirV(_) => super::DeviceType::Vulkan,
            KernelSource::Ir(_) => super::DeviceType::Cpu,
        };

        Self {
            name: name.into(),
            source,
            device_type,
            defines: Vec::new(),
            shared_memory: 0,
        }
    }

    /// Add a preprocessor define.
    pub fn define(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.defines.push((name.into(), value.into()));
        self
    }

    /// Set shared memory requirement.
    pub fn shared_memory(mut self, bytes: usize) -> Self {
        self.shared_memory = bytes;
        self
    }

    /// Set target device type explicitly.
    pub fn device_type(mut self, device_type: super::DeviceType) -> Self {
        self.device_type = device_type;
        self
    }

    /// Compile the kernel.
    pub fn build(self) -> Result<Kernel> {
        let binary = self.compile_source()?;

        Ok(Kernel {
            name: self.name.clone(),
            device_type: self.device_type,
            binary,
            entry_point: self.name,
            shared_memory: self.shared_memory,
            registers_per_thread: 0, // Set by driver after compilation
        })
    }

    fn compile_source(&self) -> Result<Vec<u8>> {
        match &self.source {
            KernelSource::Ptx(ptx) => Ok(ptx.as_bytes().to_vec()),
            KernelSource::SpirV(spirv) => Ok(spirv.clone()),
            KernelSource::Metal(msl) => Ok(msl.as_bytes().to_vec()),
            KernelSource::Cuda(_cuda) => {
                // TODO: Compile CUDA using NVRTC
                Err(Error::unsupported("CUDA compilation not yet implemented"))
            }
            KernelSource::Ir(_ir) => {
                // TODO: JIT compile using Cranelift
                Err(Error::unsupported("IR JIT compilation not yet implemented"))
            }
        }
    }
}

/// Arguments for kernel execution.
#[derive(Debug, Default)]
pub struct KernelArgs {
    /// Raw argument data
    args: Vec<KernelArg>,
}

/// A single kernel argument.
#[derive(Debug, Clone)]
pub enum KernelArg {
    /// Device buffer pointer
    Buffer(super::DevicePtr),
    /// Scalar i32
    I32(i32),
    /// Scalar u32
    U32(u32),
    /// Scalar i64
    I64(i64),
    /// Scalar u64
    U64(u64),
    /// Scalar f32
    F32(f32),
    /// Scalar f64
    F64(f64),
}

impl KernelArgs {
    /// Create empty arguments.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a buffer argument.
    pub fn buffer(mut self, ptr: super::DevicePtr) -> Self {
        self.args.push(KernelArg::Buffer(ptr));
        self
    }

    /// Add an i32 argument.
    pub fn i32(mut self, value: i32) -> Self {
        self.args.push(KernelArg::I32(value));
        self
    }

    /// Add a u32 argument.
    pub fn u32(mut self, value: u32) -> Self {
        self.args.push(KernelArg::U32(value));
        self
    }

    /// Add an i64 argument.
    pub fn i64(mut self, value: i64) -> Self {
        self.args.push(KernelArg::I64(value));
        self
    }

    /// Add a u64 argument.
    pub fn u64(mut self, value: u64) -> Self {
        self.args.push(KernelArg::U64(value));
        self
    }

    /// Add an f32 argument.
    pub fn f32(mut self, value: f32) -> Self {
        self.args.push(KernelArg::F32(value));
        self
    }

    /// Add an f64 argument.
    pub fn f64(mut self, value: f64) -> Self {
        self.args.push(KernelArg::F64(value));
        self
    }

    /// Get the arguments.
    pub fn args(&self) -> &[KernelArg] {
        &self.args
    }

    /// Number of arguments.
    pub fn len(&self) -> usize {
        self.args.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.args.is_empty()
    }
}

/// Launch configuration for a kernel.
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    /// Grid dimensions (blocks)
    pub grid: [u32; 3],
    /// Block dimensions (threads per block)
    pub block: [u32; 3],
    /// Dynamic shared memory size
    pub shared_memory: usize,
    /// Stream/queue for async execution
    pub stream: Option<u64>,
}

impl LaunchConfig {
    /// Create a 1D launch configuration.
    pub fn linear(total_threads: u32, block_size: u32) -> Self {
        let grid_x = (total_threads + block_size - 1) / block_size;
        Self {
            grid: [grid_x, 1, 1],
            block: [block_size, 1, 1],
            shared_memory: 0,
            stream: None,
        }
    }

    /// Create a 2D launch configuration.
    pub fn grid_2d(grid_x: u32, grid_y: u32, block_x: u32, block_y: u32) -> Self {
        Self {
            grid: [grid_x, grid_y, 1],
            block: [block_x, block_y, 1],
            shared_memory: 0,
            stream: None,
        }
    }

    /// Set shared memory.
    pub fn with_shared_memory(mut self, bytes: usize) -> Self {
        self.shared_memory = bytes;
        self
    }

    /// Set stream.
    pub fn with_stream(mut self, stream: u64) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Total number of threads.
    pub fn total_threads(&self) -> usize {
        (self.grid[0] as usize)
            * (self.grid[1] as usize)
            * (self.grid[2] as usize)
            * (self.block[0] as usize)
            * (self.block[1] as usize)
            * (self.block[2] as usize)
    }

    /// Total number of blocks.
    pub fn total_blocks(&self) -> usize {
        (self.grid[0] as usize) * (self.grid[1] as usize) * (self.grid[2] as usize)
    }
}
