//! Metal device management.

use super::{AppleSiliconGen, MetalMemoryStats, ResourceOptions, StorageMode};
use crate::core::{Error, Result};
use crate::hal::{Capabilities, DeviceId, DeviceInfo, DeviceType, MatrixUnit, MemoryInfo};

#[cfg(feature = "metal")]
use metal::{Device, MTLResourceOptions};

/// Metal device wrapper with UMA optimizations.
#[cfg(feature = "metal")]
pub struct MetalDevice {
    /// Underlying Metal device
    device: Device,
    /// Device info cache
    info: DeviceInfo,
    /// Chip generation
    generation: AppleSiliconGen,
    /// Command queue for compute
    command_queue: metal::CommandQueue,
    /// Memory statistics
    stats: parking_lot::RwLock<MetalMemoryStats>,
    /// Shader library cache
    library_cache: dashmap::DashMap<String, metal::Library>,
    /// Pipeline cache
    pipeline_cache: dashmap::DashMap<String, metal::ComputePipelineState>,
}

#[cfg(feature = "metal")]
impl MetalDevice {
    /// Create a new Metal device.
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| Error::device_not_available("Metal", "no Metal device found"))?;

        Self::from_device(device)
    }

    /// Create from an existing Metal device.
    pub fn from_device(device: Device) -> Result<Self> {
        let name = device.name().to_string();
        let generation = AppleSiliconGen::detect(&name);

        // Create command queue
        let command_queue = device.new_command_queue();

        // Query system memory (UMA shares all memory)
        let total_memory = get_system_memory();
        let available_memory = get_available_memory();

        let info = DeviceInfo {
            id: DeviceId::new(DeviceType::Metal, 0),
            name: name.clone(),
            device_type: DeviceType::Metal,
            memory: MemoryInfo {
                total: total_memory,
                available: available_memory,
                bandwidth_gbps: generation.memory_bandwidth_gbps(),
            },
            capabilities: Capabilities {
                compute_units: estimate_gpu_cores(&name),
                max_threads_per_unit: 1024,
                supports_f16: true,
                supports_bf16: true,
                supports_f8: false,
                supports_int8: true,
                matrix_unit: Some(MatrixUnit { m: 32, n: 32, k: 32 }),
                max_shared_memory: 32768,
                warp_size: 32, // SIMD group size
            },
        };

        Ok(Self {
            device,
            info,
            generation,
            command_queue,
            stats: parking_lot::RwLock::new(MetalMemoryStats::default()),
            library_cache: dashmap::DashMap::new(),
            pipeline_cache: dashmap::DashMap::new(),
        })
    }

    /// Get the underlying Metal device.
    pub fn raw(&self) -> &Device {
        &self.device
    }

    /// Get device name.
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Check if device has unified memory (UMA).
    pub fn has_unified_memory(&self) -> bool {
        self.device.has_unified_memory()
    }

    /// Get device info.
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Get chip generation.
    pub fn generation(&self) -> AppleSiliconGen {
        self.generation
    }

    /// Get the command queue.
    pub fn command_queue(&self) -> &metal::CommandQueue {
        &self.command_queue
    }

    /// Get current memory stats.
    pub fn memory_stats(&self) -> MetalMemoryStats {
        *self.stats.read()
    }

    /// Recommended working set size (memory limit before paging).
    pub fn recommended_working_set_size(&self) -> usize {
        // Metal API: device.recommendedMaxWorkingSetSize()
        // For now, use 75% of available memory
        (self.info.memory.available * 3) / 4
    }

    /// Current working set size.
    pub fn current_working_set_size(&self) -> usize {
        self.stats.read().total_in_use()
    }

    /// Check if we're within recommended memory limits.
    pub fn is_within_memory_budget(&self) -> bool {
        self.current_working_set_size() < self.recommended_working_set_size()
    }

    /// Create a new buffer with UMA-optimized settings.
    pub fn create_buffer(&self, size: usize, options: ResourceOptions) -> Result<metal::Buffer> {
        let metal_options = options_to_metal(options);

        let buffer = self.device.new_buffer(size as u64, metal_options);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.allocated += size;
            stats.buffer_count += 1;
            if stats.allocated > stats.peak {
                stats.peak = stats.allocated;
            }
        }

        Ok(buffer)
    }

    /// Create a buffer from existing data (zero-copy for UMA).
    pub fn create_buffer_with_data(
        &self,
        data: &[u8],
        options: ResourceOptions,
    ) -> Result<metal::Buffer> {
        let metal_options = options_to_metal(options);

        // With UMA, this is essentially zero-copy for Shared storage mode
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() as u64,
            metal_options,
        );

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.allocated += data.len();
            stats.buffer_count += 1;
            if stats.allocated > stats.peak {
                stats.peak = stats.allocated;
            }
        }

        Ok(buffer)
    }

    /// Create a buffer backed by memory-mapped file (lazy loading).
    pub fn create_buffer_from_mmap(
        &self,
        mmap: &memmap2::Mmap,
    ) -> Result<metal::Buffer> {
        // Create buffer that points to mmap'd memory
        // GPU will trigger page faults as needed
        let buffer = self.device.new_buffer_with_bytes_no_copy(
            mmap.as_ptr() as *const _,
            mmap.len() as u64,
            metal::MTLResourceOptions::StorageModeShared
                | metal::MTLResourceOptions::CPUCacheModeWriteCombined,
            None, // No deallocation callback (mmap owns memory)
        );

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.mmap_bytes += mmap.len();
            stats.buffer_count += 1;
        }

        Ok(buffer)
    }

    /// Compile a Metal shader library.
    pub fn compile_library(&self, source: &str) -> Result<metal::Library> {
        let options = metal::CompileOptions::new();
        // Use default language version for compatibility
        options.set_fast_math_enabled(true);

        self.device
            .new_library_with_source(source, &options)
            .map_err(|e| Error::KernelCompilation {
                kernel: "library".into(),
                message: e.to_string(),
            })
    }

    /// Get or compile a shader library (cached).
    pub fn get_or_compile_library(&self, name: &str, source: &str) -> Result<metal::Library> {
        if let Some(lib) = self.library_cache.get(name) {
            return Ok(lib.clone());
        }

        let library = self.compile_library(source)?;
        self.library_cache.insert(name.to_string(), library.clone());
        Ok(library)
    }

    /// Create a compute pipeline.
    pub fn create_compute_pipeline(
        &self,
        library: &metal::Library,
        function_name: &str,
    ) -> Result<metal::ComputePipelineState> {
        let function = library
            .get_function(function_name, None)
            .map_err(|e| Error::KernelCompilation {
                kernel: function_name.into(),
                message: e.to_string(),
            })?;

        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| Error::KernelCompilation {
                kernel: function_name.into(),
                message: e.to_string(),
            })
    }

    /// Get or create a compute pipeline (cached).
    pub fn get_or_create_pipeline(
        &self,
        cache_key: &str,
        library: &metal::Library,
        function_name: &str,
    ) -> Result<metal::ComputePipelineState> {
        if let Some(pipeline) = self.pipeline_cache.get(cache_key) {
            return Ok(pipeline.clone());
        }

        let pipeline = self.create_compute_pipeline(library, function_name)?;
        self.pipeline_cache.insert(cache_key.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    /// Create a new command buffer.
    pub fn new_command_buffer(&self) -> metal::CommandBuffer {
        self.command_queue.new_command_buffer().to_owned()
    }
    
    /// Create a buffer with data.
    pub fn new_buffer_with_data(
        &self,
        data: &[u8],
        options: MTLResourceOptions,
    ) -> metal::Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() as u64,
            options,
        )
    }
    
    /// Create an empty buffer.
    pub fn new_buffer(
        &self,
        length: usize,
        options: MTLResourceOptions,
    ) -> metal::Buffer {
        self.device.new_buffer(length as u64, options)
    }
    
    /// Get the underlying Metal device.
    pub fn raw_device(&self) -> &Device {
        &self.device
    }
    
    /// Get the command queue.
    pub fn queue(&self) -> &metal::CommandQueue {
        &self.command_queue
    }

    /// Synchronize (wait for all GPU work to complete).
    pub fn synchronize(&self) -> Result<()> {
        let command_buffer = self.new_command_buffer();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }

    /// Record buffer deallocation.
    pub fn record_buffer_free(&self, size: usize) {
        let mut stats = self.stats.write();
        stats.allocated = stats.allocated.saturating_sub(size);
        stats.buffer_count = stats.buffer_count.saturating_sub(1);
    }
}

#[cfg(feature = "metal")]
fn options_to_metal(options: ResourceOptions) -> metal::MTLResourceOptions {
    let mut metal_opts = metal::MTLResourceOptions::empty();

    metal_opts |= match options.storage_mode {
        StorageMode::Shared => metal::MTLResourceOptions::StorageModeShared,
        StorageMode::Private => metal::MTLResourceOptions::StorageModePrivate,
        StorageMode::Managed => metal::MTLResourceOptions::StorageModeManaged,
        StorageMode::Memoryless => metal::MTLResourceOptions::StorageModeMemoryless,
    };

    metal_opts |= match options.cpu_cache_mode {
        super::CpuCacheMode::DefaultCache => metal::MTLResourceOptions::CPUCacheModeDefaultCache,
        super::CpuCacheMode::WriteCombined => metal::MTLResourceOptions::CPUCacheModeWriteCombined,
    };

    metal_opts |= match options.hazard_tracking {
        super::HazardTracking::Tracked => metal::MTLResourceOptions::HazardTrackingModeTracked,
        super::HazardTracking::Untracked => metal::MTLResourceOptions::HazardTrackingModeUntracked,
    };

    metal_opts
}

fn get_system_memory() -> usize {
    #[cfg(feature = "metal")]
    {
        use std::process::Command;
        Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(8 * 1024 * 1024 * 1024)
    }

    #[cfg(not(feature = "metal"))]
    {
        8 * 1024 * 1024 * 1024
    }
}

fn get_available_memory() -> usize {
    // Use ~80% of system memory as "available" for UMA
    (get_system_memory() * 8) / 10
}

fn estimate_gpu_cores(name: &str) -> usize {
    // Parse GPU core count from device name or estimate
    if name.contains("Max") {
        40
    } else if name.contains("Pro") {
        18
    } else if name.contains("Ultra") {
        76
    } else {
        10
    }
}

#[cfg(not(feature = "metal"))]
pub struct MetalDevice;

#[cfg(not(feature = "metal"))]
impl MetalDevice {
    pub fn new() -> Result<Self> {
        Err(Error::device_not_available("Metal", "not on macOS"))
    }
}
