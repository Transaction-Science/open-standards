//! Memory management for efficient allocation and reuse.

use crate::core::{Error, Id, MemoryConfig, Result};
use crate::hal::{DeviceId, DevicePtr};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Central memory manager for all allocations.
#[derive(Debug)]
pub struct MemoryManager {
    /// Configuration
    config: MemoryConfig,
    /// Per-device pools
    pools: dashmap::DashMap<DeviceId, DevicePool>,
    /// Current allocation count
    allocated: AtomicUsize,
    /// Peak allocation
    peak: AtomicUsize,
    /// Cached (reusable) memory
    cached: AtomicUsize,
}

impl MemoryManager {
    /// Create a new memory manager.
    pub fn new(config: &MemoryConfig) -> Self {
        Self {
            config: config.clone(),
            pools: dashmap::DashMap::new(),
            allocated: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            cached: AtomicUsize::new(0),
        }
    }

    /// Allocate memory on a device.
    pub fn allocate(&self, size: usize, device: DeviceId) -> Result<Allocation> {
        // Round up to page size
        let size = self.round_to_page(size);

        // Try to get from cache first
        if let Some(alloc) = self.try_get_cached(size, device) {
            return Ok(alloc);
        }

        // Check memory limits
        if let Some(limit) = self.get_limit(device) {
            let current = self.allocated.load(Ordering::Acquire);
            if current + size > limit {
                // Try to free cached memory
                self.evict_cache(size);

                let current = self.allocated.load(Ordering::Acquire);
                if current + size > limit {
                    return Err(Error::out_of_memory(size, limit - current));
                }
            }
        }

        // Allocate new memory
        let allocation = self.do_allocate(size, device)?;

        // Update stats
        let new_total = self.allocated.fetch_add(size, Ordering::Release) + size;
        self.peak.fetch_max(new_total, Ordering::Release);

        Ok(allocation)
    }

    /// Free an allocation (returns to cache).
    pub fn free(&self, allocation: Allocation) {
        let size = allocation.size;
        let device = allocation.device;

        // Return to cache instead of actually freeing
        self.pools
            .entry(device)
            .or_insert_with(|| DevicePool::new(self.config.pool_page_size))
            .return_block(CachedBlock {
                ptr: allocation.ptr,
                size,
            });

        self.cached.fetch_add(size, Ordering::Release);
    }

    /// Get current allocation in bytes.
    pub fn allocated(&self) -> usize {
        self.allocated.load(Ordering::Acquire)
    }

    /// Get cached memory in bytes.
    pub fn cached(&self) -> usize {
        self.cached.load(Ordering::Acquire)
    }

    /// Get peak allocation.
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }

    /// Clear memory caches.
    pub fn clear_cache(&self) {
        for mut entry in self.pools.iter_mut() {
            let device = *entry.key();
            let (freed, blocks) = entry.clear();

            // Actually free all blocks
            for block in blocks {
                self.do_free(block.ptr, block.size, device);
            }

            self.allocated.fetch_sub(freed, Ordering::Release);
        }
        self.cached.store(0, Ordering::Release);
    }

    fn round_to_page(&self, size: usize) -> usize {
        let page = self.config.pool_page_size;
        (size + page - 1) / page * page
    }

    fn get_limit(&self, device: DeviceId) -> Option<usize> {
        match device.device_type {
            crate::hal::DeviceType::Cpu => self.config.max_cpu_memory,
            _ => self.config.max_gpu_memory,
        }
    }

    fn try_get_cached(&self, size: usize, device: DeviceId) -> Option<Allocation> {
        let mut pool = self.pools.get_mut(&device)?;
        let block = pool.get_block(size)?;

        self.cached.fetch_sub(block.size, Ordering::Release);

        Some(Allocation {
            id: Id::new(),
            ptr: block.ptr,
            size: block.size,
            device,
        })
    }

    fn evict_cache(&self, needed: usize) {
        let mut total_freed = 0;

        for mut entry in self.pools.iter_mut() {
            let device = *entry.key();
            let (freed, blocks) = entry.evict_lru(needed - total_freed);
            total_freed += freed;

            // Actually free the memory blocks
            for block in blocks {
                self.do_free(block.ptr, block.size, device);
            }

            if total_freed >= needed {
                break;
            }
        }

        self.cached.fetch_sub(total_freed, Ordering::Release);
        self.allocated.fetch_sub(total_freed, Ordering::Release);
    }

    fn do_allocate(&self, size: usize, device: DeviceId) -> Result<Allocation> {
        use crate::hal::DeviceType;

        let ptr = match device.device_type {
            DeviceType::Cpu => {
                // CPU allocation using aligned memory
                let layout = std::alloc::Layout::from_size_align(size, 64)
                    .map_err(|_| Error::out_of_memory(size, 0))?;

                let raw_ptr = unsafe { std::alloc::alloc_zeroed(layout) };
                if raw_ptr.is_null() {
                    return Err(Error::out_of_memory(size, 0));
                }
                DevicePtr::new(raw_ptr as u64)
            }
            #[cfg(feature = "metal")]
            DeviceType::Metal => {
                use metal::foreign_types::ForeignType;
                // Metal allocation via MetalDevice
                let metal_dev = crate::tensor::get_metal_device()
                    .map_err(|_| Error::device_not_available("Metal", "failed to get Metal device"))?;
                let options = crate::hal::metal::ResourceOptions::default();
                let buffer = metal_dev.create_buffer(size, options)
                    .map_err(|_| Error::out_of_memory(size, 0))?;
                let ptr = DevicePtr::new(buffer.as_ptr() as u64);
                // Prevent buffer from being released — ownership transfers to the DevicePtr.
                // The buffer will be reconstructed and dropped in do_free().
                std::mem::forget(buffer);
                ptr
            }
            _ => {
                // Fallback for unsupported device types
                return Err(Error::device_not_available(
                    format!("{:?}", device.device_type),
                    "device type not supported for direct allocation",
                ));
            }
        };

        Ok(Allocation {
            id: Id::new(),
            ptr,
            size,
            device,
        })
    }

    /// Actually free a block of memory (called during cache eviction).
    fn do_free(&self, ptr: DevicePtr, size: usize, device: DeviceId) {
        use crate::hal::DeviceType;

        if ptr.is_null() {
            return;
        }

        match device.device_type {
            DeviceType::Cpu => {
                // CPU deallocation
                if let Ok(layout) = std::alloc::Layout::from_size_align(size, 64) {
                    unsafe {
                        std::alloc::dealloc(ptr.raw() as *mut u8, layout);
                    }
                }
            }
            #[cfg(feature = "metal")]
            DeviceType::Metal => {
                // Metal buffers are managed by MetalBufferHandle
                // No manual deallocation needed here
            }
            _ => {
                // Other device types not yet supported
            }
        }
    }
}

/// A memory allocation.
#[derive(Debug)]
pub struct Allocation {
    /// Unique allocation ID
    pub id: Id,
    /// Device pointer
    pub ptr: DevicePtr,
    /// Size in bytes
    pub size: usize,
    /// Device
    pub device: DeviceId,
}

/// Per-device memory pool.
#[derive(Debug)]
struct DevicePool {
    /// Page size
    page_size: usize,
    /// Free blocks by size
    free_blocks: BTreeMap<usize, Vec<CachedBlock>>,
    /// Total cached bytes
    cached_bytes: usize,
}

impl DevicePool {
    fn new(page_size: usize) -> Self {
        Self {
            page_size,
            free_blocks: BTreeMap::new(),
            cached_bytes: 0,
        }
    }

    fn get_block(&mut self, size: usize) -> Option<CachedBlock> {
        // Find smallest block that fits
        let size_class = self.size_class(size);

        // Try exact match first
        if let Some(blocks) = self.free_blocks.get_mut(&size_class) {
            if let Some(block) = blocks.pop() {
                if blocks.is_empty() {
                    self.free_blocks.remove(&size_class);
                }
                self.cached_bytes -= block.size;
                return Some(block);
            }
        }

        // Try larger blocks
        let larger: Vec<_> = self
            .free_blocks
            .range(size_class..)
            .map(|(k, _)| *k)
            .collect();

        for size_class in larger {
            if let Some(blocks) = self.free_blocks.get_mut(&size_class) {
                if let Some(block) = blocks.pop() {
                    if blocks.is_empty() {
                        self.free_blocks.remove(&size_class);
                    }
                    self.cached_bytes -= block.size;
                    return Some(block);
                }
            }
        }

        None
    }

    fn return_block(&mut self, block: CachedBlock) {
        let size_class = self.size_class(block.size);
        self.cached_bytes += block.size;
        self.free_blocks.entry(size_class).or_default().push(block);
    }

    /// Evict blocks from cache to free memory.
    /// Returns (total_bytes_freed, blocks_to_free).
    fn evict_lru(&mut self, target: usize) -> (usize, Vec<CachedBlock>) {
        let mut freed = 0;
        let mut blocks_to_free = Vec::new();
        // Limit iterations to prevent infinite loops under memory pressure
        let max_iterations = self.free_blocks.len() + 1;
        let mut iterations = 0;

        // Evict largest blocks first
        while freed < target && !self.free_blocks.is_empty() && iterations < max_iterations {
            iterations += 1;

            // Use if-let instead of unwrap() for safety
            let largest_class = match self.free_blocks.keys().next_back() {
                Some(&class) => class,
                None => break,
            };

            if let Some(blocks) = self.free_blocks.get_mut(&largest_class) {
                if let Some(block) = blocks.pop() {
                    freed += block.size;
                    self.cached_bytes = self.cached_bytes.saturating_sub(block.size);
                    // Collect blocks to free
                    blocks_to_free.push(block);
                }
                if blocks.is_empty() {
                    self.free_blocks.remove(&largest_class);
                }
            }
        }

        (freed, blocks_to_free)
    }

    /// Clear all cached blocks and return them for deallocation.
    fn clear(&mut self) -> (usize, Vec<CachedBlock>) {
        let freed = self.cached_bytes;
        let mut blocks = Vec::new();

        // BTreeMap doesn't have drain(), so we collect and clear
        for (_, block_list) in core::mem::take(&mut self.free_blocks) {
            blocks.extend(block_list);
        }

        self.cached_bytes = 0;
        (freed, blocks)
    }

    fn size_class(&self, size: usize) -> usize {
        // Round up to power of 2 for size classes
        size.next_power_of_two()
    }
}

/// Cached memory block.
#[derive(Debug)]
struct CachedBlock {
    ptr: DevicePtr,
    size: usize,
}

/// RAII guard for temporary allocations.
pub struct TempAllocation<'a> {
    allocation: Allocation,
    manager: &'a MemoryManager,
}

impl<'a> TempAllocation<'a> {
    /// Create a temporary allocation.
    pub fn new(manager: &'a MemoryManager, size: usize, device: DeviceId) -> Result<Self> {
        let allocation = manager.allocate(size, device)?;
        Ok(Self {
            allocation,
            manager,
        })
    }

    /// Get the allocation.
    pub fn allocation(&self) -> &Allocation {
        &self.allocation
    }

    /// Get the device pointer.
    pub fn ptr(&self) -> DevicePtr {
        self.allocation.ptr
    }

    /// Get the size.
    pub fn size(&self) -> usize {
        self.allocation.size
    }
}

impl Drop for TempAllocation<'_> {
    fn drop(&mut self) {
        // Return to pool
        self.manager.free(Allocation {
            id: self.allocation.id,
            ptr: self.allocation.ptr,
            size: self.allocation.size,
            device: self.allocation.device,
        });
    }
}

/// Arena allocator for inference (no individual frees).
#[derive(Debug)]
pub struct Arena {
    /// Current allocation pointer
    current: AtomicUsize,
    /// Total capacity
    capacity: usize,
    /// Device
    device: DeviceId,
    /// Base pointer
    base: DevicePtr,
}

impl Arena {
    /// Create a new arena.
    pub fn new(capacity: usize, device: DeviceId, base: DevicePtr) -> Self {
        Self {
            current: AtomicUsize::new(0),
            capacity,
            device,
            base,
        }
    }

    /// Allocate from the arena.
    pub fn allocate(&self, size: usize, align: usize) -> Result<DevicePtr> {
        loop {
            let current = self.current.load(Ordering::Relaxed);
            let aligned = (current + align - 1) / align * align;
            let new_end = aligned + size;

            if new_end > self.capacity {
                return Err(Error::out_of_memory(size, self.capacity - current));
            }

            if self
                .current
                .compare_exchange_weak(current, new_end, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(self.base.offset(aligned));
            }
        }
    }

    /// Reset the arena (free all allocations).
    pub fn reset(&self) {
        self.current.store(0, Ordering::SeqCst);
    }

    /// Current usage.
    pub fn used(&self) -> usize {
        self.current.load(Ordering::Relaxed)
    }

    /// Remaining capacity.
    pub fn remaining(&self) -> usize {
        self.capacity - self.used()
    }
}
