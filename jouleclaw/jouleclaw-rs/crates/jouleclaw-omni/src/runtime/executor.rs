//! Execution engine for running operations.

use super::MemoryManager;
use crate::core::{Error, Result};
use crate::tensor::{DType, Tensor};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[cfg(feature = "std")]
use std::sync::RwLock;

#[cfg(not(feature = "std"))]
use spin::RwLock;

/// Execution engine that runs operations on devices.
#[derive(Debug)]
pub struct Executor {
    /// Memory manager
    memory: Arc<MemoryManager>,
    /// Number of worker threads
    num_threads: usize,
    /// Thread pool handle
    #[cfg(feature = "rayon")]
    pool: rayon::ThreadPool,
    /// Tensor registry for looking up tensors by ref
    tensor_registry: RwLock<TensorRegistry>,
    /// Operation counter for generating unique IDs
    op_counter: AtomicU64,
    /// Pending operations
    pending_ops: RwLock<BTreeMap<u64, PendingOperation>>,
}

/// Registry to look up tensors by their references.
#[derive(Debug, Default)]
struct TensorRegistry {
    tensors: BTreeMap<u64, Tensor>,
    next_id: u64,
}

impl TensorRegistry {
    fn register(&mut self, tensor: Tensor) -> TensorRef {
        let id = self.next_id;
        self.next_id += 1;
        let size = tensor.size_bytes();
        self.tensors.insert(id, tensor);
        TensorRef {
            id: crate::core::Id::from_raw(id),
            offset: 0,
            size,
        }
    }

    fn get(&self, tensor_ref: &TensorRef) -> Option<&Tensor> {
        self.tensors.get(&tensor_ref.id.raw())
    }

    fn get_mut(&mut self, tensor_ref: &TensorRef) -> Option<&mut Tensor> {
        self.tensors.get_mut(&tensor_ref.id.raw())
    }

    fn remove(&mut self, tensor_ref: &TensorRef) -> Option<Tensor> {
        self.tensors.remove(&tensor_ref.id.raw())
    }
}

/// A pending async operation.
#[derive(Debug)]
struct PendingOperation {
    /// Whether the operation is complete
    complete: AtomicBool,
    /// The result of the operation (if complete)
    result: Option<Tensor>,
}

impl Executor {
    /// Create a new executor.
    pub fn new(memory: Arc<MemoryManager>, num_threads: usize) -> Self {
        #[cfg(feature = "rayon")]
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("failed to create thread pool");

        Self {
            memory,
            num_threads,
            #[cfg(feature = "rayon")]
            pool,
            tensor_registry: RwLock::new(TensorRegistry::default()),
            op_counter: AtomicU64::new(0),
            pending_ops: RwLock::new(BTreeMap::new()),
        }
    }

    /// Get number of worker threads.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Register a tensor and get a reference to it.
    pub fn register_tensor(&self, tensor: Tensor) -> TensorRef {
        #[cfg(feature = "std")]
        let mut registry = self.tensor_registry.write().unwrap();
        #[cfg(not(feature = "std"))]
        let mut registry = self.tensor_registry.write();
        registry.register(tensor)
    }

    /// Get a tensor by reference.
    pub fn get_tensor(&self, tensor_ref: &TensorRef) -> Option<Tensor> {
        #[cfg(feature = "std")]
        let registry = self.tensor_registry.read().unwrap();
        #[cfg(not(feature = "std"))]
        let registry = self.tensor_registry.read();
        registry.get(tensor_ref).cloned()
    }

    /// Execute an operation synchronously.
    pub fn execute(&self, op: Operation) -> Result<OperationResult> {
        match op {
            Operation::MatMul { a, b, out } => {
                self.execute_matmul(a, b, out)
            }
            Operation::Attention { q, k, v, out, scale } => {
                self.execute_attention(q, k, v, out, scale)
            }
            Operation::Elementwise { op, inputs, out } => {
                self.execute_elementwise(op, inputs, out)
            }
            Operation::Custom { name, kernel, args } => {
                self.execute_custom(name, kernel, args)
            }
        }
    }

    /// Execute operations in parallel where possible.
    pub fn execute_batch(&self, ops: Vec<Operation>) -> Result<Vec<OperationResult>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            self.pool.install(|| {
                ops.into_par_iter()
                    .map(|op| self.execute(op))
                    .collect()
            })
        }

        #[cfg(not(feature = "rayon"))]
        {
            ops.into_iter()
                .map(|op| self.execute(op))
                .collect()
        }
    }

    fn execute_matmul(
        &self,
        a_ref: TensorRef,
        b_ref: TensorRef,
        out_ref: TensorRef,
    ) -> Result<OperationResult> {
        // Get input tensors
        let a = self.get_tensor(&a_ref)
            .ok_or_else(|| Error::InvalidArgument {
                name: "a".into(),
                message: "tensor not found in registry".into(),
            })?;
        let b = self.get_tensor(&b_ref)
            .ok_or_else(|| Error::InvalidArgument {
                name: "b".into(),
                message: "tensor not found in registry".into(),
            })?;

        // Execute matmul using tensor ops
        let result = crate::tensor::ops::matmul(&a, &b)?;

        // Store result in output tensor location
        #[cfg(feature = "std")]
        let mut registry = self.tensor_registry.write().unwrap();
        #[cfg(not(feature = "std"))]
        let mut registry = self.tensor_registry.write();

        registry.tensors.insert(out_ref.id.raw(), result.clone());

        Ok(OperationResult::Value(result))
    }

    fn execute_attention(
        &self,
        q_ref: TensorRef,
        k_ref: TensorRef,
        v_ref: TensorRef,
        out_ref: TensorRef,
        scale: f32,
    ) -> Result<OperationResult> {
        // Get input tensors
        let q = self.get_tensor(&q_ref)
            .ok_or_else(|| Error::InvalidArgument {
                name: "q".into(),
                message: "tensor not found in registry".into(),
            })?;
        let k = self.get_tensor(&k_ref)
            .ok_or_else(|| Error::InvalidArgument {
                name: "k".into(),
                message: "tensor not found in registry".into(),
            })?;
        let v = self.get_tensor(&v_ref)
            .ok_or_else(|| Error::InvalidArgument {
                name: "v".into(),
                message: "tensor not found in registry".into(),
            })?;

        // Execute scaled dot-product attention
        let result = crate::tensor::ops::scaled_dot_product_attention(
            &q, &k, &v, None, Some(scale)
        )?;

        // Store result
        #[cfg(feature = "std")]
        let mut registry = self.tensor_registry.write().unwrap();
        #[cfg(not(feature = "std"))]
        let mut registry = self.tensor_registry.write();

        registry.tensors.insert(out_ref.id.raw(), result.clone());

        Ok(OperationResult::Value(result))
    }

    fn execute_elementwise(
        &self,
        op: ElementwiseOp,
        input_refs: Vec<TensorRef>,
        out_ref: TensorRef,
    ) -> Result<OperationResult> {
        // Get input tensors
        let inputs: Vec<Tensor> = input_refs
            .iter()
            .map(|r| self.get_tensor(r)
                .ok_or_else(|| Error::InvalidArgument {
                    name: "input".into(),
                    message: "tensor not found in registry".into(),
                }))
            .collect::<Result<Vec<_>>>()?;

        // Execute the appropriate elementwise operation
        let result = match op {
            ElementwiseOp::Add => {
                if inputs.len() != 2 {
                    return Err(Error::InvalidArgument {
                        name: "inputs".into(),
                        message: "add requires exactly 2 inputs".into(),
                    });
                }
                crate::tensor::ops::add(&inputs[0], &inputs[1])?
            }
            ElementwiseOp::Sub => {
                if inputs.len() != 2 {
                    return Err(Error::InvalidArgument {
                        name: "inputs".into(),
                        message: "sub requires exactly 2 inputs".into(),
                    });
                }
                crate::tensor::ops::sub(&inputs[0], &inputs[1])?
            }
            ElementwiseOp::Mul => {
                if inputs.len() != 2 {
                    return Err(Error::InvalidArgument {
                        name: "inputs".into(),
                        message: "mul requires exactly 2 inputs".into(),
                    });
                }
                crate::tensor::ops::mul(&inputs[0], &inputs[1])?
            }
            ElementwiseOp::Div => {
                if inputs.len() != 2 {
                    return Err(Error::InvalidArgument {
                        name: "inputs".into(),
                        message: "div requires exactly 2 inputs".into(),
                    });
                }
                crate::tensor::ops::div(&inputs[0], &inputs[1])?
            }
            ElementwiseOp::Exp => {
                if inputs.len() != 1 {
                    return Err(Error::InvalidArgument {
                        name: "inputs".into(),
                        message: "exp requires exactly 1 input".into(),
                    });
                }
                // exp(x) using unary op
                let data: Vec<f32> = inputs[0].to_vec()?;
                let exp_data: Vec<f32> = data.iter().map(|x| x.exp()).collect();
                Tensor::from_slice(&exp_data, inputs[0].shape().clone(), DType::F32, inputs[0].device())?
            }
            ElementwiseOp::Log => {
                if inputs.len() != 1 {
                    return Err(Error::InvalidArgument {
                        name: "inputs".into(),
                        message: "log requires exactly 1 input".into(),
                    });
                }
                let data: Vec<f32> = inputs[0].to_vec()?;
                let log_data: Vec<f32> = data.iter().map(|x| x.ln()).collect();
                Tensor::from_slice(&log_data, inputs[0].shape().clone(), DType::F32, inputs[0].device())?
            }
            ElementwiseOp::Sqrt => {
                if inputs.len() != 1 {
                    return Err(Error::InvalidArgument {
                        name: "inputs".into(),
                        message: "sqrt requires exactly 1 input".into(),
                    });
                }
                let data: Vec<f32> = inputs[0].to_vec()?;
                let sqrt_data: Vec<f32> = data.iter().map(|x| x.sqrt()).collect();
                Tensor::from_slice(&sqrt_data, inputs[0].shape().clone(), DType::F32, inputs[0].device())?
            }
            ElementwiseOp::Tanh => {
                if inputs.len() != 1 {
                    return Err(Error::InvalidArgument {
                        name: "inputs".into(),
                        message: "tanh requires exactly 1 input".into(),
                    });
                }
                crate::tensor::ops::tanh(&inputs[0])?
            }
            ElementwiseOp::Sigmoid => {
                if inputs.len() != 1 {
                    return Err(Error::InvalidArgument {
                        name: "inputs".into(),
                        message: "sigmoid requires exactly 1 input".into(),
                    });
                }
                crate::tensor::ops::sigmoid(&inputs[0])?
            }
            ElementwiseOp::Gelu => {
                if inputs.len() != 1 {
                    return Err(Error::InvalidArgument {
                        name: "inputs".into(),
                        message: "gelu requires exactly 1 input".into(),
                    });
                }
                crate::tensor::ops::gelu(&inputs[0])?
            }
            ElementwiseOp::Silu => {
                if inputs.len() != 1 {
                    return Err(Error::InvalidArgument {
                        name: "inputs".into(),
                        message: "silu requires exactly 1 input".into(),
                    });
                }
                crate::tensor::ops::silu(&inputs[0])?
            }
            ElementwiseOp::Relu => {
                if inputs.len() != 1 {
                    return Err(Error::InvalidArgument {
                        name: "inputs".into(),
                        message: "relu requires exactly 1 input".into(),
                    });
                }
                crate::tensor::ops::relu(&inputs[0])?
            }
        };

        // Store result
        #[cfg(feature = "std")]
        let mut registry = self.tensor_registry.write().unwrap();
        #[cfg(not(feature = "std"))]
        let mut registry = self.tensor_registry.write();

        registry.tensors.insert(out_ref.id.raw(), result.clone());

        Ok(OperationResult::Value(result))
    }

    fn execute_custom(
        &self,
        name: String,
        kernel: KernelRef,
        args: Vec<TensorRef>,
    ) -> Result<OperationResult> {
        // Custom kernel execution would dispatch to Metal/CUDA/etc.
        // For now, we return an error for unknown kernels
        Err(Error::unsupported(format!("custom kernel '{}' not implemented", name)))
    }

    /// Create an async operation handle.
    fn create_pending_op(&self) -> OperationHandle {
        let id = self.op_counter.fetch_add(1, Ordering::SeqCst);

        #[cfg(feature = "std")]
        let mut pending = self.pending_ops.write().unwrap();
        #[cfg(not(feature = "std"))]
        let mut pending = self.pending_ops.write();

        pending.insert(id, PendingOperation {
            complete: AtomicBool::new(false),
            result: None,
        });

        OperationHandle {
            id: crate::core::Id::from_raw(id),
        }
    }

    /// Mark a pending operation as complete.
    fn complete_pending_op(&self, handle: &OperationHandle, result: Option<Tensor>) {
        #[cfg(feature = "std")]
        let mut pending = self.pending_ops.write().unwrap();
        #[cfg(not(feature = "std"))]
        let mut pending = self.pending_ops.write();

        if let Some(op) = pending.get_mut(&handle.id.raw()) {
            op.complete.store(true, Ordering::SeqCst);
            op.result = result;
        }
    }

    /// Check if a pending operation is complete.
    pub fn is_op_complete(&self, handle: &OperationHandle) -> bool {
        #[cfg(feature = "std")]
        let pending = self.pending_ops.read().unwrap();
        #[cfg(not(feature = "std"))]
        let pending = self.pending_ops.read();

        pending.get(&handle.id.raw())
            .map(|op| op.complete.load(Ordering::SeqCst))
            .unwrap_or(true)
    }

    /// Get the result of a completed pending operation.
    pub fn get_op_result(&self, handle: &OperationHandle) -> Option<Tensor> {
        #[cfg(feature = "std")]
        let pending = self.pending_ops.read().unwrap();
        #[cfg(not(feature = "std"))]
        let pending = self.pending_ops.read();

        pending.get(&handle.id.raw())
            .and_then(|op| op.result.clone())
    }
}

/// Reference to a tensor in an operation.
#[derive(Debug, Clone)]
pub struct TensorRef {
    /// Tensor ID
    pub id: crate::core::Id,
    /// Offset in storage
    pub offset: usize,
    /// Size in bytes
    pub size: usize,
}

impl TensorRef {
    /// Create a new tensor reference.
    pub fn new(id: crate::core::Id, offset: usize, size: usize) -> Self {
        Self { id, offset, size }
    }
}

/// Reference to a compiled kernel.
#[derive(Debug, Clone)]
pub struct KernelRef {
    /// Kernel ID
    pub id: crate::core::Id,
}

impl KernelRef {
    /// Create a new kernel reference.
    pub fn new(id: crate::core::Id) -> Self {
        Self { id }
    }
}

/// Types of operations that can be executed.
#[derive(Debug, Clone)]
pub enum Operation {
    /// Matrix multiplication
    MatMul {
        a: TensorRef,
        b: TensorRef,
        out: TensorRef,
    },
    /// Scaled dot-product attention
    Attention {
        q: TensorRef,
        k: TensorRef,
        v: TensorRef,
        out: TensorRef,
        scale: f32,
    },
    /// Elementwise operation
    Elementwise {
        op: ElementwiseOp,
        inputs: Vec<TensorRef>,
        out: TensorRef,
    },
    /// Custom kernel
    Custom {
        name: alloc::string::String,
        kernel: KernelRef,
        args: Vec<TensorRef>,
    },
}

impl Operation {
    /// Create a matmul operation.
    pub fn matmul(a: TensorRef, b: TensorRef, out: TensorRef) -> Self {
        Self::MatMul { a, b, out }
    }

    /// Create an attention operation.
    pub fn attention(q: TensorRef, k: TensorRef, v: TensorRef, out: TensorRef, scale: f32) -> Self {
        Self::Attention { q, k, v, out, scale }
    }

    /// Create an elementwise operation.
    pub fn elementwise(op: ElementwiseOp, inputs: Vec<TensorRef>, out: TensorRef) -> Self {
        Self::Elementwise { op, inputs, out }
    }

    /// Create a custom kernel operation.
    pub fn custom(name: impl Into<String>, kernel: KernelRef, args: Vec<TensorRef>) -> Self {
        Self::Custom {
            name: name.into(),
            kernel,
            args,
        }
    }
}

/// Elementwise operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementwiseOp {
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    Log,
    Sqrt,
    Tanh,
    Sigmoid,
    Gelu,
    Silu,
    Relu,
}

impl ElementwiseOp {
    /// Number of inputs required for this operation.
    pub fn num_inputs(&self) -> usize {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div => 2,
            _ => 1,
        }
    }
}

/// Result of executing an operation.
#[derive(Debug)]
pub enum OperationResult {
    /// Operation completed successfully
    Success,
    /// Operation returned a value
    Value(crate::tensor::Tensor),
    /// Operation is still running (async)
    Pending(OperationHandle),
}

impl OperationResult {
    /// Check if the operation was successful.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success | Self::Value(_))
    }

    /// Get the result tensor if available.
    pub fn tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Value(t) => Some(t),
            _ => None,
        }
    }

    /// Take the result tensor if available.
    pub fn into_tensor(self) -> Option<Tensor> {
        match self {
            Self::Value(t) => Some(t),
            _ => None,
        }
    }
}

/// Handle to a pending async operation.
#[derive(Debug)]
pub struct OperationHandle {
    id: crate::core::Id,
}

impl OperationHandle {
    /// Get the operation ID.
    pub fn id(&self) -> crate::core::Id {
        self.id
    }

    /// Wait for the operation to complete.
    #[cfg(feature = "std")]
    pub async fn wait(self) -> Result<OperationResult> {
        // In a real implementation, this would use async primitives
        // For now, we just return success since we don't have true async
        Ok(OperationResult::Success)
    }

    /// Check if the operation is complete.
    pub fn is_complete(&self) -> bool {
        // Would check actual completion status
        true
    }
}

/// A computation graph for batching operations.
#[derive(Debug, Default)]
pub struct ComputeGraph {
    /// Operations in topological order
    operations: Vec<Operation>,
    /// Dependencies between operations (op_idx -> dependent_op_idxs)
    dependencies: BTreeMap<usize, Vec<usize>>,
}

impl ComputeGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an operation to the graph.
    pub fn add_op(&mut self, op: Operation) -> usize {
        let idx = self.operations.len();
        self.operations.push(op);
        idx
    }

    /// Add a dependency (op at dep_idx must run before op at op_idx).
    pub fn add_dependency(&mut self, op_idx: usize, dep_idx: usize) {
        self.dependencies
            .entry(dep_idx)
            .or_default()
            .push(op_idx);
    }

    /// Get operations that can run in parallel (no dependencies).
    pub fn get_parallel_ops(&self) -> Vec<usize> {
        let mut has_deps: Vec<bool> = vec![false; self.operations.len()];

        for deps in self.dependencies.values() {
            for &dep_idx in deps {
                if dep_idx < has_deps.len() {
                    has_deps[dep_idx] = true;
                }
            }
        }

        has_deps
            .iter()
            .enumerate()
            .filter_map(|(i, &has)| if !has { Some(i) } else { None })
            .collect()
    }

    /// Execute the graph on an executor.
    pub fn execute(&self, executor: &Executor) -> Result<Vec<OperationResult>> {
        // Simple sequential execution for now
        // A more sophisticated implementation would do parallel execution
        // based on the dependency graph
        self.operations
            .iter()
            .map(|op| executor.execute(op.clone()))
            .collect()
    }

    /// Number of operations in the graph.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_op_num_inputs() {
        assert_eq!(ElementwiseOp::Add.num_inputs(), 2);
        assert_eq!(ElementwiseOp::Relu.num_inputs(), 1);
        assert_eq!(ElementwiseOp::Gelu.num_inputs(), 1);
    }

    #[test]
    fn test_operation_result_is_success() {
        assert!(OperationResult::Success.is_success());
    }
}
