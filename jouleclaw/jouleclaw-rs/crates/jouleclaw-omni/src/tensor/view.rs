//! Zero-copy tensor views.

use super::Tensor;
use crate::core::{DType, Shape};

/// A borrowed view of a tensor.
///
/// Views are zero-copy and share the underlying storage.
/// They're useful for slicing, transposing, and other
/// operations that don't modify data.
#[derive(Debug)]
pub struct TensorView<'a> {
    tensor: &'a Tensor,
}

impl<'a> TensorView<'a> {
    /// Create a new view of a tensor.
    pub fn new(tensor: &'a Tensor) -> Self {
        Self { tensor }
    }

    /// Get the shape.
    pub fn shape(&self) -> &Shape {
        self.tensor.shape()
    }

    /// Get the data type.
    pub fn dtype(&self) -> DType {
        self.tensor.dtype()
    }

    /// Get the rank.
    pub fn rank(&self) -> usize {
        self.tensor.rank()
    }

    /// Get the number of elements.
    pub fn numel(&self) -> usize {
        self.tensor.numel()
    }

    /// Get the strides.
    pub fn strides(&self) -> &[usize] {
        self.tensor.strides()
    }

    /// Check if contiguous.
    pub fn is_contiguous(&self) -> bool {
        self.tensor.is_contiguous()
    }

    /// Get the underlying tensor.
    pub fn tensor(&self) -> &Tensor {
        self.tensor
    }
}

/// Iterator over tensor elements (for CPU tensors).
pub struct TensorIter<'a, T> {
    data: &'a [T],
    shape: &'a Shape,
    strides: &'a [usize],
    indices: Vec<usize>,
    done: bool,
}

impl<'a, T> TensorIter<'a, T> {
    /// Create a new iterator.
    pub fn new(data: &'a [T], shape: &'a Shape, strides: &'a [usize]) -> Self {
        let rank = shape.rank();
        Self {
            data,
            shape,
            strides,
            indices: vec![0; rank],
            done: rank == 0 && data.is_empty(),
        }
    }

    fn linear_index(&self) -> usize {
        self.indices
            .iter()
            .zip(self.strides.iter())
            .map(|(i, s)| i * s)
            .sum()
    }

    fn advance(&mut self) {
        for i in (0..self.indices.len()).rev() {
            self.indices[i] += 1;
            if self.indices[i] < self.shape.dim(i).unwrap_or(0) {
                return;
            }
            self.indices[i] = 0;
        }
        self.done = true;
    }
}

impl<'a, T: Copy> Iterator for TensorIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let idx = self.linear_index();
        let value = self.data.get(idx).copied();
        self.advance();
        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.shape.numel();
        (remaining, Some(remaining))
    }
}

impl<'a, T: Copy> ExactSizeIterator for TensorIter<'a, T> {}

/// A mutable view of a tensor.
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    tensor: &'a mut Tensor,
}

impl<'a> TensorViewMut<'a> {
    /// Create a new mutable view.
    pub fn new(tensor: &'a mut Tensor) -> Self {
        Self { tensor }
    }

    /// Get the shape.
    pub fn shape(&self) -> &Shape {
        self.tensor.shape()
    }

    /// Get the data type.
    pub fn dtype(&self) -> DType {
        self.tensor.dtype()
    }

    /// Get as immutable view.
    pub fn as_view(&self) -> TensorView<'_> {
        TensorView::new(self.tensor)
    }
}

/// Chunk iterator for streaming processing.
pub struct ChunkIter<'a> {
    tensor: &'a Tensor,
    chunk_size: usize,
    current: usize,
    total: usize,
}

impl<'a> ChunkIter<'a> {
    /// Create a chunk iterator.
    pub fn new(tensor: &'a Tensor, chunk_size: usize) -> Self {
        let total = tensor.dim(0).unwrap_or(0);
        Self {
            tensor,
            chunk_size,
            current: 0,
            total,
        }
    }
}

impl<'a> Iterator for ChunkIter<'a> {
    type Item = crate::core::Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.total {
            return None;
        }

        let end = (self.current + self.chunk_size).min(self.total);
        let result = self.tensor.slice(0, self.current, end);
        self.current = end;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.total - self.current + self.chunk_size - 1) / self.chunk_size;
        (remaining, Some(remaining))
    }
}
