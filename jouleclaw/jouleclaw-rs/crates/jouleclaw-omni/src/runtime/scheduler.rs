//! Task scheduling for efficient execution.

use crate::core::{ExecutionConfig, Id};
use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Ordering;

/// Task scheduler that manages execution order.
#[derive(Debug)]
pub struct Scheduler {
    /// Ready queue (priority ordered)
    ready_queue: BinaryHeap<ScheduledTask>,
    /// Tasks waiting on dependencies
    waiting: Vec<Task>,
    /// Configuration
    config: ExecutionConfig,
    /// Total tasks scheduled
    total_scheduled: usize,
    /// Total tasks completed
    total_completed: usize,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(config: &ExecutionConfig) -> Self {
        Self {
            ready_queue: BinaryHeap::new(),
            waiting: Vec::new(),
            config: config.clone(),
            total_scheduled: 0,
            total_completed: 0,
        }
    }

    /// Submit a task for scheduling.
    pub fn submit(&mut self, task: Task) -> Id {
        let id = task.id;
        self.total_scheduled += 1;

        if task.dependencies.is_empty() {
            // No dependencies, immediately ready
            self.ready_queue.push(ScheduledTask {
                task,
                effective_priority: 0,
            });
        } else {
            // Has dependencies, wait
            self.waiting.push(task);
        }

        id
    }

    /// Get the next task to execute.
    pub fn next(&mut self) -> Option<Task> {
        self.ready_queue.pop().map(|st| st.task)
    }

    /// Mark a task as complete and update dependencies.
    pub fn complete(&mut self, task_id: Id) {
        self.total_completed += 1;

        // Find tasks that were waiting on this one
        let mut newly_ready = Vec::new();
        self.waiting.retain(|task| {
            let still_waiting: Vec<_> = task
                .dependencies
                .iter()
                .filter(|d| **d != task_id)
                .cloned()
                .collect();

            if still_waiting.is_empty() {
                newly_ready.push(Task {
                    id: task.id,
                    priority: task.priority,
                    memory_estimate: task.memory_estimate,
                    dependencies: Vec::new(),
                    operation: task.operation.clone(),
                });
                false
            } else {
                true
            }
        });

        // Add newly ready tasks to queue
        for task in newly_ready {
            self.ready_queue.push(ScheduledTask {
                task,
                effective_priority: 0,
            });
        }
    }

    /// Number of ready tasks.
    pub fn ready_count(&self) -> usize {
        self.ready_queue.len()
    }

    /// Number of waiting tasks.
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    /// Total tasks in scheduler.
    pub fn total_pending(&self) -> usize {
        self.ready_queue.len() + self.waiting.len()
    }

    /// Statistics.
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            total_scheduled: self.total_scheduled,
            total_completed: self.total_completed,
            ready: self.ready_queue.len(),
            waiting: self.waiting.len(),
        }
    }

    /// Clear all pending tasks.
    pub fn clear(&mut self) {
        self.ready_queue.clear();
        self.waiting.clear();
    }
}

/// A task to be scheduled.
#[derive(Debug, Clone)]
pub struct Task {
    /// Unique task ID
    pub id: Id,
    /// Task priority
    pub priority: TaskPriority,
    /// Estimated memory requirement
    pub memory_estimate: usize,
    /// Task dependencies (must complete before this runs)
    pub dependencies: Vec<Id>,
    /// The operation to perform
    pub operation: TaskOperation,
}

impl Task {
    /// Create a new task.
    pub fn new(priority: TaskPriority, operation: TaskOperation) -> Self {
        Self {
            id: Id::new(),
            priority,
            memory_estimate: 0,
            dependencies: Vec::new(),
            operation,
        }
    }

    /// Set memory estimate.
    pub fn with_memory(mut self, bytes: usize) -> Self {
        self.memory_estimate = bytes;
        self
    }

    /// Add a dependency.
    pub fn depends_on(mut self, task_id: Id) -> Self {
        self.dependencies.push(task_id);
        self
    }
}

/// Task priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Lowest priority (background work)
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// High priority
    High = 2,
    /// Critical priority (user-facing latency)
    Critical = 3,
}

impl Default for TaskPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Types of task operations.
#[derive(Debug, Clone)]
pub enum TaskOperation {
    /// Compute operation
    Compute(super::executor::Operation),
    /// Memory transfer
    Transfer {
        src_device: crate::hal::DeviceId,
        dst_device: crate::hal::DeviceId,
        size: usize,
    },
    /// Synchronization barrier
    Barrier,
    /// Custom operation
    Custom(alloc::string::String),
}

/// Internal scheduled task with computed priority.
#[derive(Debug)]
struct ScheduledTask {
    task: Task,
    effective_priority: i64,
}

impl PartialEq for ScheduledTask {
    fn eq(&self, other: &Self) -> bool {
        self.task.id == other.task.id
    }
}

impl Eq for ScheduledTask {}

impl PartialOrd for ScheduledTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first
        self.task
            .priority
            .cmp(&other.task.priority)
            .then_with(|| {
                // Lower memory requirement preferred
                other.task.memory_estimate.cmp(&self.task.memory_estimate)
            })
    }
}

/// Scheduler statistics.
#[derive(Debug, Clone, Copy)]
pub struct SchedulerStats {
    /// Total tasks scheduled since start
    pub total_scheduled: usize,
    /// Total tasks completed
    pub total_completed: usize,
    /// Currently ready tasks
    pub ready: usize,
    /// Tasks waiting on dependencies
    pub waiting: usize,
}

/// Memory-aware scheduler that considers available memory.
#[derive(Debug)]
pub struct MemoryAwareScheduler {
    inner: Scheduler,
    available_memory: usize,
    reserved_memory: usize,
}

impl MemoryAwareScheduler {
    /// Create a new memory-aware scheduler.
    pub fn new(config: &ExecutionConfig, available_memory: usize) -> Self {
        Self {
            inner: Scheduler::new(config),
            available_memory,
            reserved_memory: 0,
        }
    }

    /// Submit a task.
    pub fn submit(&mut self, task: Task) -> Id {
        self.inner.submit(task)
    }

    /// Get the next task if sufficient memory is available.
    pub fn next_if_memory_available(&mut self) -> Option<Task> {
        // Peek at the next task
        if let Some(scheduled) = self.inner.ready_queue.peek() {
            let required = scheduled.task.memory_estimate;
            let available = self.available_memory.saturating_sub(self.reserved_memory);

            if required <= available {
                let task = self.inner.next()?;
                self.reserved_memory += task.memory_estimate;
                return Some(task);
            }
        }
        None
    }

    /// Mark task complete and release memory.
    pub fn complete(&mut self, task_id: Id, memory_used: usize) {
        self.reserved_memory = self.reserved_memory.saturating_sub(memory_used);
        self.inner.complete(task_id);
    }

    /// Update available memory.
    pub fn update_available_memory(&mut self, bytes: usize) {
        self.available_memory = bytes;
    }
}
