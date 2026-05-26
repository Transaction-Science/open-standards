//! Runtime execution engine.
//!
//! The runtime provides:
//! - Streaming execution with bounded memory
//! - Automatic device selection and memory management
//! - Async execution with proper resource cleanup
//! - Resource efficiency monitoring

mod executor;
pub mod stream;
mod scheduler;
mod memory_manager;
pub mod monitor;

pub use executor::Executor;
pub use stream::{StreamingOutput, StreamHandle, StreamSender, StreamBuilder};
pub use scheduler::{Scheduler, Task, TaskPriority};
pub use memory_manager::{MemoryManager, Allocation};
pub use monitor::{ResourceMonitor, ResourceSnapshot, ComputeStats, ScopedTimer};

use crate::core::{Config, Error, Result};
use crate::hal::{self, DeviceInfo, DeviceId};
use alloc::sync::Arc;
use alloc::vec::Vec;
use parking_lot::RwLock;

/// The main runtime for efficient-genai.
///
/// Manages devices, memory, and execution across all backends.
#[derive(Debug)]
pub struct Runtime {
    /// Configuration
    config: Config,
    /// Discovered devices
    devices: Vec<DeviceInfo>,
    /// Primary device for execution
    primary_device: DeviceId,
    /// Memory manager
    memory: Arc<MemoryManager>,
    /// Scheduler
    scheduler: Arc<RwLock<Scheduler>>,
    /// Executor
    executor: Arc<Executor>,
}

impl Runtime {
    /// Create a new runtime with default configuration.
    pub fn new() -> Result<Self> {
        Self::with_config(Config::default())
    }

    /// Create a new runtime with custom configuration.
    pub fn with_config(config: Config) -> Result<Self> {
        // Discover available devices
        let devices = hal::discover_devices()?;

        if devices.is_empty() {
            return Err(Error::device_not_available(
                "any",
                "no compute devices found",
            ));
        }

        // Select primary device (prefer GPU)
        let primary_device = devices
            .iter()
            .find(|d| d.device_type != hal::DeviceType::Cpu)
            .unwrap_or(&devices[0])
            .id;

        tracing::info!(
            "Runtime initialized with {} devices, primary: {}",
            devices.len(),
            primary_device
        );

        for device in &devices {
            tracing::debug!(
                "  {} ({}): {} MB memory, {} compute units",
                device.name,
                device.id,
                device.memory.total / (1024 * 1024),
                device.capabilities.compute_units
            );
        }

        let memory = Arc::new(MemoryManager::new(&config.memory));
        let scheduler = Arc::new(RwLock::new(Scheduler::new(&config.execution)));
        let executor = Arc::new(Executor::new(
            Arc::clone(&memory),
            config.execution.num_threads,
        ));

        Ok(Self {
            config,
            devices,
            primary_device,
            memory,
            scheduler,
            executor,
        })
    }

    /// Get runtime configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get all available devices.
    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// Get the primary device.
    pub fn primary_device(&self) -> DeviceId {
        self.primary_device
    }

    /// Set the primary device.
    pub fn set_primary_device(&mut self, device: DeviceId) -> Result<()> {
        if !self.devices.iter().any(|d| d.id == device) {
            return Err(Error::device_not_available(
                format!("{}", device),
                "device not found",
            ));
        }
        self.primary_device = device;
        Ok(())
    }

    /// Get device info by ID.
    pub fn device_info(&self, id: DeviceId) -> Option<&DeviceInfo> {
        self.devices.iter().find(|d| d.id == id)
    }

    /// Get the memory manager.
    pub fn memory(&self) -> &Arc<MemoryManager> {
        &self.memory
    }

    /// Get the executor.
    pub fn executor(&self) -> &Arc<Executor> {
        &self.executor
    }

    /// Synchronize all devices.
    ///
    /// Metal devices synchronize via command buffer `wait_until_completed()`
    /// which is already called after each kernel dispatch in MetalOps.
    /// CPU operations are synchronous by nature. This is a barrier for
    /// API completeness — all dispatches are already synchronous.
    pub fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    /// Get memory statistics.
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            allocated: self.memory.allocated(),
            cached: self.memory.cached(),
            peak: self.memory.peak(),
        }
    }

    /// Clear memory caches.
    pub fn clear_cache(&self) {
        self.memory.clear_cache();
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new().expect("failed to initialize runtime")
    }
}

/// Memory usage statistics.
#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    /// Currently allocated bytes
    pub allocated: usize,
    /// Cached (reusable) bytes
    pub cached: usize,
    /// Peak allocation
    pub peak: usize,
}

impl MemoryStats {
    /// Total memory in use (allocated + cached).
    pub fn total(&self) -> usize {
        self.allocated + self.cached
    }
}

/// Global runtime instance (optional convenience).
static GLOBAL_RUNTIME: parking_lot::RwLock<Option<Arc<Runtime>>> = parking_lot::RwLock::new(None);

/// Initialize the global runtime.
pub fn init() -> Result<()> {
    init_with_config(Config::default())
}

/// Initialize the global runtime with configuration.
pub fn init_with_config(config: Config) -> Result<()> {
    let runtime = Runtime::with_config(config)?;
    *GLOBAL_RUNTIME.write() = Some(Arc::new(runtime));
    Ok(())
}

/// Get the global runtime.
pub fn get() -> Option<Arc<Runtime>> {
    GLOBAL_RUNTIME.read().clone()
}

/// Get the global runtime, initializing if necessary.
pub fn get_or_init() -> Result<Arc<Runtime>> {
    {
        let guard = GLOBAL_RUNTIME.read();
        if let Some(runtime) = guard.as_ref() {
            return Ok(Arc::clone(runtime));
        }
    }

    init()?;
    Ok(GLOBAL_RUNTIME.read().clone().unwrap())
}
