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
/// Returns `Ok(result)` if OpenVINO executed successfully, `Err(())` if the
/// runtime is unavailable or the operation failed for any reason.
#[cfg(feature = "intel")]
pub fn openvino_matmul(
    lhs: &IntelFloatTensor,
    rhs: &IntelFloatTensor,
) -> Result<IntelFloatTensor, ()> {
    use openvino::{Core, DeviceType, ElementType, Shape as OvShape, Tensor as OvTensor};

    // Only worth dispatching to NPU for large matmuls (heuristic threshold).
    let m = if lhs.shape.len() >= 2 {
        lhs.shape[lhs.shape.len() - 2]
    } else {
        1
    };
    let n = if rhs.shape.len() >= 2 {
        rhs.shape[rhs.shape.len() - 1]
    } else {
        1
    };
    let k = *lhs.shape.last().unwrap_or(&1);

    // Skip OpenVINO overhead for small matrices.
    if m * n * k < 4096 {
        return Err(());
    }

    // Build OpenVINO IR XML for MatMul.
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

    // Read model from XML buffer (no weights tensor needed for MatMul).
    let model = core
        .read_model_from_buffer(ir_xml.as_bytes(), None)
        .map_err(|_| ())?;

    // Try devices in priority order: NPU, GPU, CPU.
    let devices: [DeviceType; 3] = [DeviceType::NPU, DeviceType::GPU, DeviceType::CPU];
    let mut compiled = None;
    for device in &devices {
        if let Ok(c) = core.compile_model(&model, device.to_owned()) {
            compiled = Some(c);
            break;
        }
    }
    let mut compiled = compiled.ok_or(())?;
    let mut request = compiled.create_infer_request().map_err(|_| ())?;

    // Set input tensors.
    let lhs_shape = OvShape::new(&[m as i64, k as i64]).map_err(|_| ())?;
    let lhs_tensor = {
        let mut t = OvTensor::new(ElementType::F32, &lhs_shape).map_err(|_| ())?;
        {
            let buf = t.get_data_mut::<f32>().map_err(|_| ())?;
            buf.copy_from_slice(&lhs.data[..m * k]);
        }
        t
    };
    let rhs_shape = OvShape::new(&[k as i64, n as i64]).map_err(|_| ())?;
    let rhs_tensor = {
        let mut t = OvTensor::new(ElementType::F32, &rhs_shape).map_err(|_| ())?;
        {
            let buf = t.get_data_mut::<f32>().map_err(|_| ())?;
            buf.copy_from_slice(&rhs.data[..k * n]);
        }
        t
    };

    request
        .set_input_tensor_by_index(0, &lhs_tensor)
        .map_err(|_| ())?;
    request
        .set_input_tensor_by_index(1, &rhs_tensor)
        .map_err(|_| ())?;

    request.infer().map_err(|_| ())?;

    let output = request.get_output_tensor_by_index(0).map_err(|_| ())?;
    let out_data: Vec<f32> = output.get_data::<f32>().map_err(|_| ())?.to_vec();

    Ok(IntelFloatTensor::new(out_data, vec![m, n]))
}

/// CPU fallback matmul (simple triple loop, no SIMD).
pub fn cpu_matmul(lhs: &IntelFloatTensor, rhs: &IntelFloatTensor) -> IntelFloatTensor {
    // Support batched matmul: [..., M, K] x [..., K, N] -> [..., M, N]
    let lhs_ndim = lhs.shape.len();
    let rhs_ndim = rhs.shape.len();

    assert!(lhs_ndim >= 2 && rhs_ndim >= 2, "matmul requires at least 2D tensors");

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
    let array =
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&tensor.shape), tensor.data.clone())
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
