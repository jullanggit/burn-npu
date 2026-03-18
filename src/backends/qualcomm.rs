//! Qualcomm Hexagon NPU backend via QNN SDK.
//!
//! This module provides `QnnFloatTensor`, a tensor type that stores data as
//! `Vec<f32>` with shape metadata. All operations currently execute on CPU.
//! The structure is ready for QNN SDK FFI integration when hardware and SDK
//! are available.
//!
//! # Future QNN Integration
//!
//! The Qualcomm AI Engine Direct (QNN) SDK provides:
//! - `QnnContext_create` / `QnnGraph_create` for building compute graphs
//! - `QnnGraph_addNode` for adding operations (MatMul, Add, etc.)
//! - `QnnGraph_execute` for running on Hexagon NPU
//! - `QnnTensor` for tensor I/O
//!
//! When integrating, the pattern would be similar to OpenVINO:
//! - Build a QNN graph for compute-bound ops (matmul)
//! - Execute on HTP (Hexagon Tensor Processor) backend
//! - Fall back to CPU for elementwise ops

use burn_tensor::{DType, Shape};

// ---------------------------------------------------------------------------
// QnnFloatTensor
// ---------------------------------------------------------------------------

/// A float tensor backed by a `Vec<f32>` with shape metadata.
///
/// When QNN SDK integration is complete, compute-heavy ops like matmul will
/// dispatch to the Hexagon NPU. Currently all ops execute on CPU.
#[derive(Debug, Clone)]
pub struct QnnFloatTensor {
    /// Flat f32 data in row-major order.
    pub data: Vec<f32>,
    /// Shape dimensions.
    pub shape: Vec<usize>,
}

impl QnnFloatTensor {
    /// Create a tensor from flat data and shape.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "QnnFloatTensor::new: data.len() != product(shape)"
        );
        Self { data, shape }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        Self {
            data: vec![0.0; total],
            shape,
        }
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        Self {
            data: vec![1.0; total],
            shape,
        }
    }

    /// Create a tensor filled with a constant value.
    pub fn full(shape: Vec<usize>, value: f32) -> Self {
        let total: usize = shape.iter().product();
        Self {
            data: vec![value; total],
            shape,
        }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

impl burn_tensor::TensorMetadata for QnnFloatTensor {
    fn dtype(&self) -> DType {
        DType::F32
    }

    fn shape(&self) -> Shape {
        Shape::from(self.shape.clone())
    }
}

// ---------------------------------------------------------------------------
// CPU matmul (placeholder for future QNN HTP dispatch)
// ---------------------------------------------------------------------------

/// CPU matmul implementation.
///
/// TODO: QNN SDK integration — when the QNN SDK is available, large matmuls
/// should be dispatched to the Hexagon Tensor Processor (HTP) via:
///   1. QnnContext_create() with HTP backend
///   2. QnnGraph_create() + QnnGraph_addNode("MatMul", ...)
///   3. QnnGraph_finalize() + QnnGraph_execute()
///   4. Read output QnnTensor back
pub fn cpu_matmul(lhs: &QnnFloatTensor, rhs: &QnnFloatTensor) -> QnnFloatTensor {
    // Support batched matmul: [..., M, K] x [..., K, N] -> [..., M, N]
    let lhs_ndim = lhs.shape.len();
    let rhs_ndim = rhs.shape.len();

    assert!(
        lhs_ndim >= 2 && rhs_ndim >= 2,
        "matmul requires at least 2D tensors"
    );

    let m = lhs.shape[lhs_ndim - 2];
    let k = lhs.shape[lhs_ndim - 1];
    let n = rhs.shape[rhs_ndim - 1];
    assert_eq!(
        rhs.shape[rhs_ndim - 2],
        k,
        "matmul inner dimensions mismatch"
    );

    // Compute batch dimensions.
    let lhs_batch: usize = lhs.shape[..lhs_ndim - 2].iter().product();
    let rhs_batch: usize = rhs.shape[..rhs_ndim - 2].iter().product();
    let batch = lhs_batch.max(rhs_batch);

    let mut out_shape: Vec<usize> = if lhs_ndim >= rhs_ndim {
        lhs.shape[..lhs_ndim - 2].to_vec()
    } else {
        rhs.shape[..rhs_ndim - 2].to_vec()
    };
    out_shape.push(m);
    out_shape.push(n);

    let mut result = vec![0.0f32; batch * m * n];

    for b in 0..batch {
        let lhs_offset = (b % lhs_batch) * m * k;
        let rhs_offset = (b % rhs_batch) * k * n;
        let out_offset = b * m * n;

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += lhs.data[lhs_offset + i * k + p] * rhs.data[rhs_offset + p * n + j];
                }
                result[out_offset + i * n + j] = sum;
            }
        }
    }

    QnnFloatTensor::new(result, out_shape)
}

// ---------------------------------------------------------------------------
// Conversion helpers for NdArray interop
// ---------------------------------------------------------------------------

/// Convert QnnFloatTensor -> NdArrayTensor (for delegating ops to burn-ndarray).
pub fn qnn_to_ndarray(tensor: &QnnFloatTensor) -> burn_ndarray::NdArrayTensor {
    let array =
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&tensor.shape), tensor.data.clone())
            .unwrap()
            .into_shared();
    burn_ndarray::NdArrayTensor::from(array)
}

/// Convert NdArrayTensor (f32) -> QnnFloatTensor.
pub fn ndarray_to_qnn(tensor: &burn_ndarray::NdArrayTensor) -> QnnFloatTensor {
    if let burn_ndarray::NdArrayTensor::F32(ref storage) = tensor {
        let view = storage.view();
        let contig = view.as_standard_layout();
        let data = contig.as_slice().unwrap().to_vec();
        let shape: Vec<usize> = view.shape().to_vec();
        QnnFloatTensor::new(data, shape)
    } else {
        panic!("ndarray_to_qnn: expected F32 NdArrayTensor");
    }
}
