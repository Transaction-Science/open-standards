//! Metal resource heaps for efficient allocation.
//!
//! Heaps pre-allocate memory in large chunks, allowing fast sub-allocation
//! without per-buffer allocation overhead.

use super::{MetalDevice, ResourceOptions, StorageMode};
use crate::core::{Error, Result};
use std::sync::Arc;

#[cfg(feature = "metal")]
use metal::{Buffer, Heap, HeapDescriptor, MTLResourceOptions};

/// A heap for efficient buffer allocation.
#[cfg(feature = "metal")]
pub struct MetalHeap {
    /// Underlying Metal heap
    heap: Heap,
    /// Total size
    size: usize,
    /// Used size
    used: std::sync::atomic::AtomicUsize,
    /// Device reference
    device: Arc<MetalDevice>,
}

#[cfg(feature = "metal")]
impl MetalHeap {
    /// Create a new heap with the given size.
    pub fn new(device: Arc<MetalDevice>, size: usize, options: ResourceOptions) -> Result<Self> {
        let descriptor = HeapDescriptor::new();
        descriptor.set_size(size as u64);
        descriptor.set_storage_mode(match options.storage_mode {
            StorageMode::Shared => metal::MTLStorageMode::Shared,
            StorageMode::Private => metal::MTLStorageMode::Private,
            StorageMode::Managed => metal::MTLStorageMode::Managed,
            StorageMode::Memoryless => metal::MTLStorageMode::Memoryless,
        });
        descriptor.set_cpu_cache_mode(match options.cpu_cache_mode {
            super::CpuCacheMode::DefaultCache => metal::MTLCPUCacheMode::DefaultCache,
            super::CpuCacheMode::WriteCombined => metal::MTLCPUCacheMode::WriteCombined,
        });
        // Purgeable for memory pressure handling
        descriptor.set_hazard_tracking_mode(match options.hazard_tracking {
            super::HazardTracking::Tracked => metal::MTLHazardTrackingMode::Tracked,
            super::HazardTracking::Untracked => metal::MTLHazardTrackingMode::Untracked,
        });

        let heap = device.raw().new_heap(&descriptor);

        Ok(Self {
            heap,
            size,
            used: std::sync::atomic::AtomicUsize::new(0),
            device,
        })
    }

    /// Create a heap sized for a specific model.
    pub fn for_model(device: Arc<MetalDevice>, model_size: usize) -> Result<Self> {
        // Add 10% headroom for activations
        let size = model_size + model_size / 10;

        Self::new(device, size, ResourceOptions::weights())
    }

    /// Allocate a buffer from the heap.
    pub fn allocate_buffer(&self, size: usize) -> Result<HeapBuffer<'_>> {
        // Check space
        let current = self.used.load(std::sync::atomic::Ordering::Relaxed);
        if current + size > self.size {
            return Err(Error::out_of_memory(size, self.size - current));
        }

        // Allocate from heap
        let options = MTLResourceOptions::StorageModeShared
            | MTLResourceOptions::CPUCacheModeWriteCombined
            | MTLResourceOptions::HazardTrackingModeUntracked;

        let buffer = self.heap.new_buffer(size as u64, options)
            .ok_or_else(|| Error::out_of_memory(size, self.size - current))?;

        // Update used
        self.used.fetch_add(size, std::sync::atomic::Ordering::Relaxed);

        Ok(HeapBuffer {
            buffer,
            size,
            heap: self,
        })
    }

    /// Get current usage.
    pub fn used(&self) -> usize {
        self.used.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total size.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get remaining space.
    pub fn remaining(&self) -> usize {
        self.size - self.used()
    }

    /// Get usage percentage.
    pub fn usage_percent(&self) -> f64 {
        (self.used() as f64 / self.size as f64) * 100.0
    }

    /// Make the heap purgeable (can be evicted under memory pressure).
    pub fn make_purgeable(&self) {
        self.heap.set_purgeable_state(metal::MTLPurgeableState::Volatile);
    }

    /// Make the heap non-purgeable (lock in memory).
    pub fn make_non_purgeable(&self) -> bool {
        let state = self.heap.set_purgeable_state(metal::MTLPurgeableState::NonVolatile);
        state != metal::MTLPurgeableState::Empty
    }
}

/// A buffer allocated from a heap.
#[cfg(feature = "metal")]
pub struct HeapBuffer<'a> {
    buffer: Buffer,
    size: usize,
    heap: &'a MetalHeap,
}

#[cfg(feature = "metal")]
impl<'a> HeapBuffer<'a> {
    /// Get the underlying buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get size.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Make alias (for memory reuse without reallocation).
    pub fn make_aliasable(&self) {
        self.buffer.make_aliasable();
    }
}

#[cfg(feature = "metal")]
impl<'a> Drop for HeapBuffer<'a> {
    fn drop(&mut self) {
        // Buffers are automatically reclaimed when dropped
        self.heap.used.fetch_sub(self.size, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Arena-style allocator using a heap.
#[cfg(feature = "metal")]
pub struct HeapArena {
    /// Underlying heap
    heap: MetalHeap,
    /// Current offset
    offset: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "metal")]
impl HeapArena {
    /// Create a new arena.
    pub fn new(device: Arc<MetalDevice>, size: usize) -> Result<Self> {
        let heap = MetalHeap::new(
            device,
            size,
            ResourceOptions {
                storage_mode: StorageMode::Private,
                cpu_cache_mode: super::CpuCacheMode::DefaultCache,
                hazard_tracking: super::HazardTracking::Untracked,
            },
        )?;

        Ok(Self {
            heap,
            offset: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Allocate from the arena (bump allocation).
    pub fn allocate(&self, size: usize, align: usize) -> Result<ArenaAllocation<'_>> {
        loop {
            let current = self.offset.load(std::sync::atomic::Ordering::Relaxed);
            let aligned = (current + align - 1) & !(align - 1);
            let new_offset = aligned + size;

            if new_offset > self.heap.size() {
                return Err(Error::out_of_memory(size, self.heap.size() - current));
            }

            if self.offset
                .compare_exchange_weak(
                    current,
                    new_offset,
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::Relaxed,
                )
                .is_ok()
            {
                // Allocate buffer at this offset
                let buffer = self.heap.allocate_buffer(size)?;
                return Ok(ArenaAllocation {
                    buffer,
                    offset: aligned,
                });
            }
        }
    }

    /// Reset the arena (reuse all memory).
    pub fn reset(&self) {
        self.offset.store(0, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get current usage.
    pub fn used(&self) -> usize {
        self.offset.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total size.
    pub fn size(&self) -> usize {
        self.heap.size()
    }
}

/// An allocation from an arena.
#[cfg(feature = "metal")]
pub struct ArenaAllocation<'a> {
    buffer: HeapBuffer<'a>,
    offset: usize,
}

#[cfg(feature = "metal")]
impl<'a> ArenaAllocation<'a> {
    /// Get the buffer.
    pub fn buffer(&self) -> &Buffer {
        self.buffer.buffer()
    }

    /// Get offset in arena.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get size.
    pub fn size(&self) -> usize {
        self.buffer.size()
    }
}

// Stubs for non-macOS
#[cfg(not(feature = "metal"))]
pub struct MetalHeap;

#[cfg(not(feature = "metal"))]
pub struct HeapArena;
