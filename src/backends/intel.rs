//! Intel NPU backend via OpenVINO.
//!
//! This module provides `IntelFloatTensor`, a tensor type that stores data as
//! `Vec<f32>` with shape metadata. For compute-bound operations like matmul,
//! it attempts to use the OpenVINO runtime to dispatch to NPU > GPU > CPU.
//! For all other operations, it executes directly on CPU using simple f32 loops.
//!
//! The `openvino` crate is used with `runtime-linking`, so the code compiles
//! everywhere even if the OpenVINO runtime is not installed. NPU acceleration
//! is opportunistic: if the runtime or device is unavailable, we fall back to
//! a CPU implementation transparently.

use burn_tensor::{DType, Shape};
use std::collections::HashMap;
use std::sync::Mutex;

// Cache compiled OpenVINO models by (m, k, n) shape to avoid recompilation per call.
#[cfg(feature = "intel")]
static OV_CACHE: std::sync::LazyLock<Mutex<HashMap<(usize, usize, usize), OvCompiledMatmul>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "intel")]
struct OvCompiledMatmul {
    compiled: openvino::CompiledModel,
}

#[cfg(feature = "intel")]
fn ensure_openvino_loaded() -> Result<(), ()> {
    openvino_sys::load().map_err(|_| ())
}

// ---------------------------------------------------------------------------
// IntelFloatTensor
// ---------------------------------------------------------------------------

/// A float tensor backed by a `Vec<f32>` with shape metadata.
///
/// When the `intel` feature is active, `float_matmul` (and potentially other
/// compute-heavy ops) will attempt to use the OpenVINO NPU. All other ops
/// execute as simple CPU loops on the underlying Vec.
#[derive(Debug, Clone)]
pub struct IntelFloatTensor {
    /// Flat f32 data in row-major order.
    pub data: Vec<f32>,
    /// Shape dimensions.
    pub shape: Vec<usize>,
}

// Vec<f32> is Send + Sync, so IntelFloatTensor is too.
// (Derived automatically, but explicit for clarity.)

impl IntelFloatTensor {
    /// Create a tensor from flat data and shape.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "IntelFloatTensor::new: data.len() != product(shape)"
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

impl burn_tensor::TensorMetadata for IntelFloatTensor {
    fn dtype(&self) -> DType {
        DType::F32
    }

    fn shape(&self) -> Shape {
        Shape::from(self.shape.clone())
    }
}

// ---------------------------------------------------------------------------
// OpenVINO matmul (best-effort NPU acceleration)
// ---------------------------------------------------------------------------

/// Attempt to perform matmul via OpenVINO, targeting NPU > GPU > CPU.
///
/// Dispatch matmul to OpenVINO NPU. Handles batched (3D+) tensors by looping
/// over batch dims and running each 2D slice on NPU.
/// Returns `Err(())` if OpenVINO runtime is unavailable.
#[cfg(feature = "intel")]
pub fn openvino_matmul(
    lhs: &IntelFloatTensor,
    rhs: &IntelFloatTensor,
) -> Result<IntelFloatTensor, ()> {
    use openvino::{Core, DeviceType, ElementType, Shape as OvShape, Tensor as OvTensor};

    // The runtime-linking backend keeps the loaded library in thread-local state.
    ensure_openvino_loaded()?;

    let lhs_ndim = lhs.shape.len();
    let rhs_ndim = rhs.shape.len();
    if lhs_ndim < 2 || rhs_ndim < 2 {
        return Err(());
    }

    let m = lhs.shape[lhs_ndim - 2];
    let k = lhs.shape[lhs_ndim - 1];
    let n = rhs.shape[rhs_ndim - 1];

    // Skip OpenVINO overhead for small matmuls
    if m * k * n < 4096 {
        return Err(());
    }

    // Compute batch dimensions: everything before the last 2 dims
    let lhs_batch: Vec<usize> = lhs.shape[..lhs_ndim - 2].to_vec();
    let rhs_batch: Vec<usize> = rhs.shape[..rhs_ndim - 2].to_vec();
    // Batch shapes must match (or be empty for 2D)
    if lhs_batch != rhs_batch {
        return Err(());
    }
    let batch_size: usize = lhs_batch.iter().product::<usize>().max(1);

    let lhs_stride = m * k; // elements per batch slice in lhs
    let rhs_stride = k * n; // elements per batch slice in rhs
    let out_stride = m * n;

    let lhs_ov_shape = OvShape::new(&[m as i64, k as i64]).map_err(|_| ())?;
    let rhs_ov_shape = OvShape::new(&[k as i64, n as i64]).map_err(|_| ())?;

    // Get or compile the model for this (m, k, n) shape — cached across calls
    let cache_key = (m, k, n);
    let mut cache = OV_CACHE.lock().map_err(|_| ())?;
    if !cache.contains_key(&cache_key) {
        let ir_xml = format!(
            r#"<?xml version="1.0"?>
<net name="matmul" version="11">
  <layers>
    <layer id="0" name="lhs" type="Parameter" version="opset1">
      <data shape="{m},{k}" element_type="f32"/>
      <output><port id="0" precision="FP32"><dim>{m}</dim><dim>{k}</dim></port></output>
    </layer>
    <layer id="1" name="rhs" type="Parameter" version="opset1">
      <data shape="{k},{n}" element_type="f32"/>
      <output><port id="0" precision="FP32"><dim>{k}</dim><dim>{n}</dim></port></output>
    </layer>
    <layer id="2" name="mm" type="MatMul" version="opset1">
      <data transpose_a="false" transpose_b="false"/>
      <input>
        <port id="0"><dim>{m}</dim><dim>{k}</dim></port>
        <port id="1"><dim>{k}</dim><dim>{n}</dim></port>
      </input>
      <output><port id="2" precision="FP32"><dim>{m}</dim><dim>{n}</dim></port></output>
    </layer>
    <layer id="3" name="result" type="Result" version="opset1">
      <input><port id="0"><dim>{m}</dim><dim>{n}</dim></port></input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
  </edges>
</net>"#
        );

        let mut core = Core::new().map_err(|_| ())?;
        let model = core
            .read_model_from_buffer(ir_xml.as_bytes(), None)
            .map_err(|_| ())?;

        let devices = [DeviceType::NPU, DeviceType::GPU, DeviceType::CPU];
        let mut compiled = None;
        for dev in &devices {
            if let Ok(c) = core.compile_model(&model, dev.to_owned()) {
                compiled = Some(c);
                break;
            }
        }
        cache.insert(
            cache_key,
            OvCompiledMatmul {
                compiled: compiled.ok_or(())?,
            },
        );
    }
    let entry = cache.get_mut(&cache_key).unwrap();
    let mut request = entry.compiled.create_infer_request().map_err(|_| ())?;
    drop(cache); // release lock before running inference

    // Run each batch slice through the compiled model
    let mut result_data = Vec::with_capacity(batch_size * out_stride);
    for b in 0..batch_size {
        let lhs_off = b * lhs_stride;
        let rhs_off = b * rhs_stride;

        let mut lt = OvTensor::new(ElementType::F32, &lhs_ov_shape).map_err(|_| ())?;
        let lt_data = lt.get_data_mut::<f32>().map_err(|_| ())?;
        if lt_data.len() != lhs_stride {
            return Err(());
        }
        lt_data.copy_from_slice(&lhs.data[lhs_off..lhs_off + lhs_stride]);

        let mut rt = OvTensor::new(ElementType::F32, &rhs_ov_shape).map_err(|_| ())?;
        let rt_data = rt.get_data_mut::<f32>().map_err(|_| ())?;
        if rt_data.len() != rhs_stride {
            return Err(());
        }
        rt_data.copy_from_slice(&rhs.data[rhs_off..rhs_off + rhs_stride]);

        request.set_input_tensor_by_index(0, &lt).map_err(|_| ())?;
        request.set_input_tensor_by_index(1, &rt).map_err(|_| ())?;
        request.infer().map_err(|_| ())?;

        let output = request.get_output_tensor_by_index(0).map_err(|_| ())?;
        let output_data = output.get_data::<f32>().map_err(|_| ())?;
        if output_data.len() != out_stride {
            return Err(());
        }
        result_data.extend_from_slice(output_data);
    }

    // Output shape: [...batch_dims, m, n]
    let mut out_shape = lhs_batch;
    out_shape.push(m);
    out_shape.push(n);

    Ok(IntelFloatTensor::new(result_data, out_shape))
}

/// CPU fallback matmul (simple triple loop, no SIMD).
pub fn cpu_matmul(lhs: &IntelFloatTensor, rhs: &IntelFloatTensor) -> IntelFloatTensor {
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

    IntelFloatTensor::new(result, out_shape)
}

// ---------------------------------------------------------------------------
// Conversion helpers for NdArray interop
// ---------------------------------------------------------------------------

/// Convert IntelFloatTensor -> NdArrayTensor (for delegating ops to burn-ndarray).
pub fn intel_to_ndarray(tensor: &IntelFloatTensor) -> burn_ndarray::NdArrayTensor {
    let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&tensor.shape), tensor.data.clone())
        .unwrap()
        .into_shared();
    burn_ndarray::NdArrayTensor::from(array)
}

/// Convert NdArrayTensor (f32) -> IntelFloatTensor.
pub fn ndarray_to_intel(tensor: &burn_ndarray::NdArrayTensor) -> IntelFloatTensor {
    if let burn_ndarray::NdArrayTensor::F32(ref storage) = tensor {
        let view = storage.view();
        let contig = view.as_standard_layout();
        let data = contig.as_slice().unwrap().to_vec();
        let shape: Vec<usize> = view.shape().to_vec();
        IntelFloatTensor::new(data, shape)
    } else {
        panic!("ndarray_to_intel: expected F32 NdArrayTensor");
    }
}
