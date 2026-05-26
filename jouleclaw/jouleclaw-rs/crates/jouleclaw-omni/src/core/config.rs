//! Configuration types for the runtime.

use super::{Error, Result};

/// Main configuration for the efficient-genai runtime.
#[derive(Debug, Clone)]
pub struct Config {
    /// Memory management configuration
    pub memory: MemoryConfig,
    /// Execution configuration
    pub execution: ExecutionConfig,
    /// Enable JIT compilation when available
    pub enable_jit: bool,
    /// Enable tracing/profiling
    pub enable_tracing: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            memory: MemoryConfig::default(),
            execution: ExecutionConfig::default(),
            enable_jit: true,
            enable_tracing: false,
        }
    }
}

impl Config {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns an error if any configuration values are invalid.
    pub fn validate(&self) -> Result<()> {
        self.memory.validate()?;
        self.execution.validate()?;
        Ok(())
    }

    /// Create a validated configuration, returning an error if invalid.
    pub fn new_validated(memory: MemoryConfig, execution: ExecutionConfig) -> Result<Self> {
        let config = Self {
            memory,
            execution,
            enable_jit: true,
            enable_tracing: false,
        };
        config.validate()?;
        Ok(config)
    }
}

/// Memory management configuration.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum GPU memory to use (in bytes). None = use all available.
    pub max_gpu_memory: Option<usize>,
    /// Maximum CPU memory for model offloading (in bytes).
    pub max_cpu_memory: Option<usize>,
    /// Enable memory-mapped file loading for large models.
    pub enable_mmap: bool,
    /// Arena allocator initial size (bytes).
    pub arena_initial_size: usize,
    /// Enable aggressive memory reclamation.
    pub aggressive_gc: bool,
    /// Memory pool page size (bytes).
    pub pool_page_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_gpu_memory: None,
            max_cpu_memory: None,
            enable_mmap: true,
            arena_initial_size: 64 * 1024 * 1024, // 64 MB
            aggressive_gc: false,
            pool_page_size: 2 * 1024 * 1024, // 2 MB pages
        }
    }
}

/// Minimum allowed page size (4 KB).
const MIN_PAGE_SIZE: usize = 4 * 1024;

/// Maximum allowed page size (1 GB).
const MAX_PAGE_SIZE: usize = 1024 * 1024 * 1024;

/// Minimum arena size (1 MB).
const MIN_ARENA_SIZE: usize = 1024 * 1024;

impl MemoryConfig {
    /// Configuration optimized for low-memory devices.
    pub fn low_memory() -> Self {
        Self {
            max_gpu_memory: Some(4 * 1024 * 1024 * 1024), // 4 GB
            max_cpu_memory: Some(8 * 1024 * 1024 * 1024), // 8 GB
            enable_mmap: true,
            arena_initial_size: 16 * 1024 * 1024, // 16 MB
            aggressive_gc: true,
            pool_page_size: 512 * 1024, // 512 KB pages
        }
    }

    /// Configuration for high-memory servers.
    pub fn high_memory() -> Self {
        Self {
            max_gpu_memory: None,
            max_cpu_memory: None,
            enable_mmap: true,
            arena_initial_size: 256 * 1024 * 1024, // 256 MB
            aggressive_gc: false,
            pool_page_size: 8 * 1024 * 1024, // 8 MB pages
        }
    }

    /// Validate the memory configuration.
    ///
    /// # Errors
    /// Returns an error if any values are invalid.
    pub fn validate(&self) -> Result<()> {
        // Validate page size
        if self.pool_page_size == 0 {
            return Err(Error::invalid_argument(
                "pool_page_size",
                "must be greater than 0",
            ));
        }
        if self.pool_page_size < MIN_PAGE_SIZE {
            return Err(Error::invalid_argument(
                "pool_page_size",
                format!("must be at least {} bytes (4 KB)", MIN_PAGE_SIZE),
            ));
        }
        if self.pool_page_size > MAX_PAGE_SIZE {
            return Err(Error::invalid_argument(
                "pool_page_size",
                format!("must be at most {} bytes (1 GB)", MAX_PAGE_SIZE),
            ));
        }
        if !self.pool_page_size.is_power_of_two() {
            return Err(Error::invalid_argument(
                "pool_page_size",
                "must be a power of two",
            ));
        }

        // Validate arena size
        if self.arena_initial_size == 0 {
            return Err(Error::invalid_argument(
                "arena_initial_size",
                "must be greater than 0",
            ));
        }
        if self.arena_initial_size < MIN_ARENA_SIZE {
            return Err(Error::invalid_argument(
                "arena_initial_size",
                format!("must be at least {} bytes (1 MB)", MIN_ARENA_SIZE),
            ));
        }

        // Validate memory limits make sense
        if let (Some(gpu), Some(cpu)) = (self.max_gpu_memory, self.max_cpu_memory) {
            if gpu == 0 && cpu == 0 {
                return Err(Error::invalid_argument(
                    "memory limits",
                    "at least one memory pool must have non-zero capacity",
                ));
            }
        }

        Ok(())
    }
}

/// Execution configuration.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Number of worker threads for CPU operations.
    pub num_threads: usize,
    /// Enable async execution where possible.
    pub async_execution: bool,
    /// Default batch size for operations.
    pub default_batch_size: usize,
    /// Enable operation fusion.
    pub enable_fusion: bool,
    /// Maximum number of concurrent streams.
    pub max_streams: usize,
    /// Chunk size for streaming operations.
    pub stream_chunk_size: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        Self {
            num_threads: num_cpus,
            async_execution: true,
            default_batch_size: 1,
            enable_fusion: true,
            max_streams: 4,
            stream_chunk_size: 1024,
        }
    }
}

/// Maximum allowed threads.
const MAX_THREADS: usize = 1024;

/// Maximum allowed streams.
const MAX_STREAMS: usize = 64;

/// Maximum batch size.
const MAX_BATCH_SIZE: usize = 4096;

impl ExecutionConfig {
    /// Configuration for latency-sensitive workloads.
    pub fn low_latency() -> Self {
        Self {
            num_threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4),
            async_execution: true,
            default_batch_size: 1,
            enable_fusion: true,
            max_streams: 8,
            stream_chunk_size: 256,
        }
    }

    /// Configuration for throughput-optimized workloads.
    pub fn high_throughput() -> Self {
        Self {
            num_threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4),
            async_execution: true,
            default_batch_size: 8,
            enable_fusion: true,
            max_streams: 2,
            stream_chunk_size: 4096,
        }
    }

    /// Validate the execution configuration.
    ///
    /// # Errors
    /// Returns an error if any values are invalid.
    pub fn validate(&self) -> Result<()> {
        // Validate thread count
        if self.num_threads == 0 {
            return Err(Error::invalid_argument(
                "num_threads",
                "must be at least 1",
            ));
        }
        if self.num_threads > MAX_THREADS {
            return Err(Error::invalid_argument(
                "num_threads",
                format!("must be at most {}", MAX_THREADS),
            ));
        }

        // Validate batch size
        if self.default_batch_size == 0 {
            return Err(Error::invalid_argument(
                "default_batch_size",
                "must be at least 1",
            ));
        }
        if self.default_batch_size > MAX_BATCH_SIZE {
            return Err(Error::invalid_argument(
                "default_batch_size",
                format!("must be at most {}", MAX_BATCH_SIZE),
            ));
        }

        // Validate streams
        if self.max_streams == 0 {
            return Err(Error::invalid_argument(
                "max_streams",
                "must be at least 1",
            ));
        }
        if self.max_streams > MAX_STREAMS {
            return Err(Error::invalid_argument(
                "max_streams",
                format!("must be at most {}", MAX_STREAMS),
            ));
        }

        // Validate chunk size
        if self.stream_chunk_size == 0 {
            return Err(Error::invalid_argument(
                "stream_chunk_size",
                "must be at least 1",
            ));
        }

        Ok(())
    }
}
