//! Burn `Backend` implementation for NPU.
//!
//! Platform-specific float tensor primitives:
//!
//! - **`apple`**: `NpuFloatTensor` wraps an `i32` MLTensor handle. All float ops
//!   pass handles through FFI; no data leaves the NPU between ops.
//! - **`intel`**: `NpuFloatTensor` is `IntelFloatTensor` (`Vec<f32>` + shape).
//!   Matmul attempts OpenVINO NPU dispatch; all other ops run on CPU or delegate
//!   to burn-ndarray.
//! - **`qualcomm`**: `NpuFloatTensor` is `QnnFloatTensor` (`Vec<f32>` + shape).
//!   All ops currently run on CPU. Ready for QNN SDK integration.
//! - **no feature**: `NpuFloatTensor` is `NdArrayTensor` (pure CPU fallback).
//!
//! Int/Bool tensor primitives always remain `NdArrayTensor` (delegated to burn-ndarray).

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use burn_ndarray::{NdArray, NdArrayDevice, NdArrayQTensor, NdArrayTensor};
use burn_tensor::backend::{Backend, DeviceId, DeviceOps, ExecutionError};
use burn_tensor::ops::*;
use burn_tensor::ops::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use burn_tensor::quantization::{QuantizationParametersPrimitive, QuantScheme};
use burn_tensor::{DType, Distribution, FloatDType, IntDType, Shape, Slice, TensorData};

// ---------------------------------------------------------------------------
// Type alias for the NdArray backend we delegate to.
// ---------------------------------------------------------------------------
type Nd = NdArray<f32, i64, i8>;

// ===========================================================================
// FFI declarations (apple only, declared once)
// ===========================================================================
#[cfg(feature = "apple")]
#[allow(dead_code)]
extern "C" {
    fn npu_create_tensor(shape: *const i32, dims: i32, data: *const f32, len: i32) -> i32;
    fn npu_create_int_tensor(shape: *const i32, dims: i32, data: *const i32, len: i32) -> i32;
    fn npu_free_tensor(id: i32);
    fn npu_get_shape(id: i32, out: *mut i32, max: i32) -> i32;
    fn npu_get_data(id: i32, out: *mut f32, max: i32) -> i32;
    fn npu_get_int_data(id: i32, out: *mut i32, max: i32) -> i32;
    fn npu_clone(id: i32) -> i32;

    // Matmul
    fn npu_matmul(a: i32, b: i32) -> i32;

    // Binary arithmetic
    fn npu_add(a: i32, b: i32) -> i32;
    fn npu_sub(a: i32, b: i32) -> i32;
    fn npu_mul(a: i32, b: i32) -> i32;
    fn npu_div(a: i32, b: i32) -> i32;

    // Scalar arithmetic
    fn npu_add_scalar(a: i32, s: f32) -> i32;
    fn npu_sub_scalar(a: i32, s: f32) -> i32;
    fn npu_mul_scalar(a: i32, s: f32) -> i32;
    fn npu_div_scalar(a: i32, s: f32) -> i32;

    // Unary math
    fn npu_neg(a: i32) -> i32;
    fn npu_exp(a: i32) -> i32;
    fn npu_log(a: i32) -> i32;
    fn npu_sqrt(a: i32) -> i32;
    fn npu_abs(a: i32) -> i32;
    fn npu_tanh(a: i32) -> i32;
    fn npu_sin(a: i32) -> i32;
    fn npu_cos(a: i32) -> i32;
    fn npu_floor(a: i32) -> i32;
    fn npu_ceil(a: i32) -> i32;
    fn npu_erf(a: i32) -> i32;

    // Power
    fn npu_pow(a: i32, b: i32) -> i32;
    fn npu_pow_scalar(a: i32, p: f32) -> i32;

    // Clamp
    fn npu_clamp_min(a: i32, min: f32) -> i32;
    fn npu_clamp_max(a: i32, max: f32) -> i32;
    fn npu_clamp(a: i32, min: f32, max: f32) -> i32;

    // Softmax
    fn npu_softmax(a: i32, dim: i32) -> i32;

    // Reductions
    fn npu_sum(a: i32) -> i32;
    fn npu_sum_dim(a: i32, dim: i32) -> i32;
    fn npu_mean_all(a: i32) -> i32;
    fn npu_mean(a: i32, dim: i32) -> i32;
    fn npu_max(a: i32) -> i32;
    fn npu_max_dim(a: i32, dim: i32) -> i32;
    fn npu_min(a: i32) -> i32;
    fn npu_min_dim(a: i32, dim: i32) -> i32;

    // Argmax / argmin (return int tensor handles)
    fn npu_argmax(a: i32, dim: i32) -> i32;
    fn npu_argmin(a: i32, dim: i32) -> i32;

    // Shape ops
    fn npu_reshape(a: i32, shape: *const i32, dims: i32) -> i32;
    fn npu_transpose(a: i32, dim0: i32, dim1: i32) -> i32;
    fn npu_permute(a: i32, perm: *const i32, len: i32) -> i32;
    fn npu_narrow(a: i32, dim: i32, start: i32, length: i32) -> i32;
    fn npu_expand(a: i32, shape: *const i32, dims: i32) -> i32;

    // Cat
    fn npu_cat(ids: *const i32, count: i32, dim: i32) -> i32;

    // Indexing
    fn npu_index_select(a: i32, indices: *const i32, len: i32) -> i32;
    fn npu_gather(a: i32, dim: i32, indices: i32) -> i32;
    fn npu_slice(a: i32, ranges: *const i32, num_ranges: i32) -> i32;

    // Comparison (return 0/1 float handles)
    fn npu_equal(a: i32, b: i32) -> i32;
    fn npu_greater(a: i32, b: i32) -> i32;
    fn npu_less(a: i32, b: i32) -> i32;

    // Mask
    fn npu_mask_fill(a: i32, mask: i32, value: f32) -> i32;
    fn npu_mask_where(a: i32, mask: i32, source: i32) -> i32;

    // Creation
    fn npu_zeros(shape: *const i32, dims: i32) -> i32;
    fn npu_ones(shape: *const i32, dims: i32) -> i32;
    fn npu_full(shape: *const i32, dims: i32, value: f32) -> i32;
    fn npu_scalar_tensor(value: f32) -> i32;

    // Cast
    fn npu_cast_to_int(a: i32) -> i32;
    fn npu_cast_to_float(a: i32) -> i32;
}

// ---------------------------------------------------------------------------
// NpuBurnDevice
// ---------------------------------------------------------------------------
/// Device type for the NPU burn backend. There is only one logical device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NpuBurnDevice {
    /// The default device (routes to ANE when available, falls back to CPU).
    Default,
}

impl DeviceOps for NpuBurnDevice {}

impl burn_tensor::backend::Device for NpuBurnDevice {
    fn from_id(_device_id: DeviceId) -> Self {
        Self::Default
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 1,
            index_id: 0,
        }
    }

    fn device_count(_type_id: u16) -> usize {
        1
    }
}

impl Default for NpuBurnDevice {
    fn default() -> Self {
        Self::Default
    }
}

// ===========================================================================
// NpuFloatTensor — MLTensor handle (apple only)
// ===========================================================================

#[cfg(feature = "apple")]
#[derive(Debug)]
pub struct NpuFloatTensor {
    pub(crate) handle: i32,
}

// SAFETY: MLTensor handles are thread-safe integers into a Swift-side table.
#[cfg(feature = "apple")]
unsafe impl Send for NpuFloatTensor {}
#[cfg(feature = "apple")]
unsafe impl Sync for NpuFloatTensor {}

#[cfg(feature = "apple")]
impl Clone for NpuFloatTensor {
    fn clone(&self) -> Self {
        Self {
            handle: unsafe { npu_clone(self.handle) },
        }
    }
}

#[cfg(feature = "apple")]
impl Drop for NpuFloatTensor {
    fn drop(&mut self) {
        unsafe { npu_free_tensor(self.handle) };
    }
}

#[cfg(feature = "apple")]
impl burn_tensor::TensorMetadata for NpuFloatTensor {
    fn dtype(&self) -> DType {
        DType::F32
    }

    fn shape(&self) -> Shape {
        let mut buf = [0i32; 8];
        let ndim = unsafe { npu_get_shape(self.handle, buf.as_mut_ptr(), 8) } as usize;
        Shape::from(buf[..ndim].iter().map(|&d| d as usize).collect::<Vec<_>>())
    }
}

// ===========================================================================
// NpuFloatTensor — IntelFloatTensor wrapper (intel feature)
// ===========================================================================

#[cfg(feature = "intel")]
pub type NpuFloatTensor = crate::backends::intel::IntelFloatTensor;

// ===========================================================================
// NpuFloatTensor — QnnFloatTensor wrapper (qualcomm feature)
// ===========================================================================

#[cfg(feature = "qualcomm")]
pub type NpuFloatTensor = crate::backends::qualcomm::QnnFloatTensor;

// ===========================================================================
// Conversion helpers (intel)
// ===========================================================================

#[cfg(feature = "intel")]
fn npu_to_ndarray(tensor: &NpuFloatTensor) -> NdArrayTensor {
    crate::backends::intel::intel_to_ndarray(tensor)
}

#[cfg(feature = "intel")]
fn ndarray_to_npu(tensor: &NdArrayTensor) -> NpuFloatTensor {
    crate::backends::intel::ndarray_to_intel(tensor)
}

// ===========================================================================
// Conversion helpers (qualcomm)
// ===========================================================================

#[cfg(feature = "qualcomm")]
fn npu_to_ndarray(tensor: &NpuFloatTensor) -> NdArrayTensor {
    crate::backends::qualcomm::qnn_to_ndarray(tensor)
}

#[cfg(feature = "qualcomm")]
fn ndarray_to_npu(tensor: &NdArrayTensor) -> NpuFloatTensor {
    crate::backends::qualcomm::ndarray_to_qnn(tensor)
}

// ===========================================================================
// Helper: read f32 data from an NPU handle
// ===========================================================================
#[cfg(feature = "apple")]
fn read_f32(handle: i32) -> (Vec<f32>, Vec<usize>) {
    let mut shape_buf = [0i32; 8];
    let ndim = unsafe { npu_get_shape(handle, shape_buf.as_mut_ptr(), 8) } as usize;
    let shape: Vec<usize> = shape_buf[..ndim].iter().map(|&d| d as usize).collect();
    let total: usize = shape.iter().product();
    let mut data = vec![0.0f32; total];
    unsafe { npu_get_data(handle, data.as_mut_ptr(), total as i32) };
    (data, shape)
}

/// Helper: read i32 data from an NPU handle (for argmax/argmin results).
#[cfg(feature = "apple")]
fn read_int(handle: i32) -> (Vec<i32>, Vec<usize>) {
    let mut shape_buf = [0i32; 8];
    let ndim = unsafe { npu_get_shape(handle, shape_buf.as_mut_ptr(), 8) } as usize;
    let shape: Vec<usize> = shape_buf[..ndim].iter().map(|&d| d as usize).collect();
    let total: usize = shape.iter().product();
    let mut data = vec![0i32; total];
    unsafe { npu_get_int_data(handle, data.as_mut_ptr(), total as i32) };
    (data, shape)
}

/// Extract i64 data from NdArrayTensor (int variant).
#[cfg(feature = "apple")]
fn extract_i64(tensor: &NdArrayTensor) -> Vec<i64> {
    if let NdArrayTensor::I64(ref storage) = tensor {
        let view = storage.view();
        let contig = view.as_standard_layout();
        return contig.as_slice().unwrap().to_vec();
    }
    if let NdArrayTensor::I32(ref storage) = tensor {
        let view = storage.view();
        let contig = view.as_standard_layout();
        return contig.as_slice().unwrap().iter().map(|&v| v as i64).collect();
    }
    panic!("extract_i64: expected I64 or I32 NdArrayTensor");
}

/// Convert NpuFloatTensor -> NdArrayTensor by reading data back (slow, use sparingly).
#[cfg(feature = "apple")]
fn npu_to_ndarray(tensor: &NpuFloatTensor) -> NdArrayTensor {
    let (data, shape) = read_f32(tensor.handle);
    let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), data)
        .unwrap()
        .into_shared();
    NdArrayTensor::from(array)
}

/// Convert NdArrayTensor (f32) -> NpuFloatTensor by sending data to MLTensor.
#[cfg(feature = "apple")]
fn ndarray_to_npu(tensor: &NdArrayTensor) -> NpuFloatTensor {
    if let NdArrayTensor::F32(ref storage) = tensor {
        let view = storage.view();
        let contig = view.as_standard_layout();
        let data = contig.as_slice().unwrap();
        let shape: Vec<i32> = view.shape().iter().map(|&d| d as i32).collect();
        NpuFloatTensor {
            handle: unsafe {
                npu_create_tensor(
                    shape.as_ptr(),
                    shape.len() as i32,
                    data.as_ptr(),
                    data.len() as i32,
                )
            },
        }
    } else {
        panic!("ndarray_to_npu: expected F32 NdArrayTensor");
    }
}

/// Helper: convert shape dims to i32 vec
#[cfg(feature = "apple")]
#[inline]
fn shape_i32(shape: &Shape) -> Vec<i32> {
    shape.dims.iter().map(|&d| d as i32).collect()
}

/// Helper: convert i32 NPU int result handle -> NdArrayTensor<i64>
#[cfg(feature = "apple")]
fn int_handle_to_ndarray(handle: i32) -> NdArrayTensor {
    let (int_data, shape) = read_int(handle);
    unsafe { npu_free_tensor(handle) };
    let i64_data: Vec<i64> = int_data.iter().map(|&v| v as i64).collect();
    let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), i64_data)
        .unwrap()
        .into_shared();
    NdArrayTensor::from(array)
}

/// Helper: convert float NPU result handle -> NdArrayTensor<bool> (0/1 float -> bool)
#[cfg(feature = "apple")]
fn float_handle_to_bool_ndarray(handle: i32) -> NdArrayTensor {
    let (data, shape) = read_f32(handle);
    unsafe { npu_free_tensor(handle) };
    let bool_data: Vec<bool> = data.iter().map(|&v| v != 0.0).collect();
    let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), bool_data)
        .unwrap()
        .into_shared();
    NdArrayTensor::from(array)
}

/// Helper: map NpuBurnDevice -> NdArrayDevice for forwarding.
#[inline(always)]
fn nd_dev() -> NdArrayDevice {
    NdArrayDevice::Cpu
}

// ---------------------------------------------------------------------------
// NpuBurnBackend
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Default, Debug)]
pub struct NpuBurnBackend;

// ===========================================================================
// Backend impl — apple feature: FloatTensorPrimitive = NpuFloatTensor
// ===========================================================================
#[cfg(feature = "apple")]
impl Backend for NpuBurnBackend {
    type Device = NpuBurnDevice;

    type FloatTensorPrimitive = NpuFloatTensor;
    type FloatElem = f32;

    type IntTensorPrimitive = NdArrayTensor;
    type IntElem = i64;

    type BoolTensorPrimitive = NdArrayTensor;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = NdArrayQTensor;

    fn ad_enabled() -> bool {
        false
    }

    fn name(_device: &Self::Device) -> String {
        String::from("npu (MLTensor)")
    }

    fn seed(_device: &Self::Device, seed: u64) {
        <Nd as Backend>::seed(&NdArrayDevice::Cpu, seed);
    }

    fn supports_dtype(_device: &Self::Device, dtype: DType) -> bool {
        <Nd as Backend>::supports_dtype(&NdArrayDevice::Cpu, dtype)
    }
}

// ===========================================================================
// Backend impl — no apple: full NdArray delegation
// ===========================================================================
#[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
impl Backend for NpuBurnBackend {
    type Device = NpuBurnDevice;

    type FloatTensorPrimitive = NdArrayTensor;
    type FloatElem = f32;

    type IntTensorPrimitive = NdArrayTensor;
    type IntElem = i64;

    type BoolTensorPrimitive = NdArrayTensor;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = NdArrayQTensor;

    fn ad_enabled() -> bool {
        false
    }

    fn name(_device: &Self::Device) -> String {
        String::from("npu (ndarray)")
    }

    fn seed(_device: &Self::Device, seed: u64) {
        <Nd as Backend>::seed(&NdArrayDevice::Cpu, seed);
    }

    fn supports_dtype(_device: &Self::Device, dtype: DType) -> bool {
        <Nd as Backend>::supports_dtype(&NdArrayDevice::Cpu, dtype)
    }
}

// ===========================================================================
// FloatTensorOps — apple: all ops go through MLTensor handles
// ===========================================================================
#[cfg(feature = "apple")]
impl FloatTensorOps<Self> for NpuBurnBackend {
    fn float_from_data(data: TensorData, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        let floats: Vec<f32> = data.to_vec().unwrap();
        let shape: Vec<i32> = data.shape.iter().map(|&d| d as i32).collect();
        NpuFloatTensor {
            handle: unsafe {
                npu_create_tensor(
                    shape.as_ptr(),
                    shape.len() as i32,
                    floats.as_ptr(),
                    floats.len() as i32,
                )
            },
        }
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        _device: &NpuBurnDevice,
    ) -> FloatTensor<Self> {
        // Generate random data on CPU, then send to NPU
        let nd_tensor = <Nd as FloatTensorOps<Nd>>::float_random(shape, distribution, &nd_dev());
        ndarray_to_npu(&nd_tensor)
    }

    fn float_zeros(shape: Shape, _device: &NpuBurnDevice, _dtype: FloatDType) -> FloatTensor<Self> {
        let s = shape_i32(&shape);
        NpuFloatTensor {
            handle: unsafe { npu_zeros(s.as_ptr(), s.len() as i32) },
        }
    }

    fn float_ones(shape: Shape, _device: &NpuBurnDevice, _dtype: FloatDType) -> FloatTensor<Self> {
        let s = shape_i32(&shape);
        NpuFloatTensor {
            handle: unsafe { npu_ones(s.as_ptr(), s.len() as i32) },
        }
    }

    fn float_full(
        shape: Shape,
        fill_value: f32,
        _device: &NpuBurnDevice,
        _dtype: FloatDType,
    ) -> FloatTensor<Self> {
        let s = shape_i32(&shape);
        NpuFloatTensor {
            handle: unsafe { npu_full(s.as_ptr(), s.len() as i32, fill_value) },
        }
    }

    fn float_device(_tensor: &FloatTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }

    fn float_to_device(tensor: FloatTensor<Self>, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        tensor
    }

    fn float_empty(shape: Shape, device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        Self::float_zeros(shape, device, dtype)
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        let shape = burn_tensor::TensorMetadata::shape(&tensor);
        let total: usize = shape.dims.iter().product();
        let mut data = vec![0.0f32; total];
        unsafe { npu_get_data(tensor.handle, data.as_mut_ptr(), total as i32) };
        // tensor drops here, freeing the MLTensor handle
        Ok(TensorData::new(data, shape))
    }

    // ── Matmul ──────────────────────────────────────────────────────────

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_matmul(lhs.handle, rhs.handle) },
        }
    }

    fn float_cross(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        // No FFI for cross product — round-trip through NdArray
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_cross(nd_lhs, nd_rhs, dim);
        ndarray_to_npu(&result)
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> IntTensor<Self> {
        let (data, shape) = read_f32(tensor.handle);
        let int_data: Vec<i64> = data.iter().map(|&v| v as i64).collect();
        let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), int_data)
            .unwrap()
            .into_shared();
        NdArrayTensor::from(array)
    }

    // ── Arithmetic ──────────────────────────────────────────────────────

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_add(lhs.handle, rhs.handle) },
        }
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_add_scalar(lhs.handle, rhs) },
        }
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sub(lhs.handle, rhs.handle) },
        }
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sub_scalar(lhs.handle, rhs) },
        }
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_mul(lhs.handle, rhs.handle) },
        }
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_mul_scalar(lhs.handle, rhs) },
        }
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_div(lhs.handle, rhs.handle) },
        }
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_div_scalar(lhs.handle, rhs) },
        }
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        // remainder = lhs - (lhs / rhs).floor() * rhs
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_remainder(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_remainder_scalar(nd_lhs, rhs);
        ndarray_to_npu(&result)
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        // recip = 1.0 / tensor
        NpuFloatTensor {
            handle: unsafe {
                let one = npu_scalar_tensor(1.0);
                let r = npu_div(one, tensor.handle);
                npu_free_tensor(one);
                r
            },
        }
    }

    // ── Shape / layout ──────────────────────────────────────────────────

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_transpose(tensor.handle, dim1 as i32, dim2 as i32) },
        }
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let perm: Vec<i32> = axes.iter().map(|&a| a as i32).collect();
        NpuFloatTensor {
            handle: unsafe { npu_permute(tensor.handle, perm.as_ptr(), perm.len() as i32) },
        }
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        // No direct FFI — round-trip
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_flip(nd, axes);
        ndarray_to_npu(&result)
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let s = shape_i32(&shape);
        NpuFloatTensor {
            handle: unsafe { npu_reshape(tensor.handle, s.as_ptr(), s.len() as i32) },
        }
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let s = shape_i32(&shape);
        NpuFloatTensor {
            handle: unsafe { npu_expand(tensor.handle, s.as_ptr(), s.len() as i32) },
        }
    }

    // ── Gather / scatter / select ───────────────────────────────────────

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        let idx_data = extract_i64(&indices);
        let idx_i32: Vec<i32> = idx_data.iter().map(|&v| v as i32).collect();
        let idx_len = idx_data.len(); let idx_shape: Vec<i32> = vec![idx_len as i32];
        unsafe {
            let idx_handle = npu_create_int_tensor(idx_shape.as_ptr(), idx_shape.len() as i32, idx_i32.as_ptr(), idx_i32.len() as i32);
            let result = npu_gather(tensor.handle, dim as i32, idx_handle);
            npu_free_tensor(idx_handle);
            NpuFloatTensor { handle: result }
        }
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result = <Nd as FloatTensorOps<Nd>>::float_scatter_add(dim, nd_tensor, indices, nd_value);
        ndarray_to_npu(&result)
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        // Native NPU gather — no readback of weight tensor
        let idx_data = extract_i64(&indices);
        let idx_i32: Vec<i32> = idx_data.iter().map(|&v| v as i32).collect();
        let idx_len = idx_data.len(); let idx_shape: Vec<i32> = vec![idx_len as i32];
        unsafe {
            let idx_handle = npu_create_int_tensor(idx_shape.as_ptr(), idx_shape.len() as i32, idx_i32.as_ptr(), idx_i32.len() as i32);
            let result = npu_gather(tensor.handle, dim as i32, idx_handle);
            npu_free_tensor(idx_handle);
            NpuFloatTensor { handle: result }
        }
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result =
            <Nd as FloatTensorOps<Nd>>::float_select_add(nd_tensor, dim, indices, nd_value);
        ndarray_to_npu(&result)
    }

    // ── Slice ───────────────────────────────────────────────────────────

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        // Fast path: simple contiguous ranges (step=1, no negative indices)
        // This covers narrow() which is the common case
        let ndim = slices.len();
        if ndim <= 3 {
            let all_simple = slices.iter().all(|s| s.step == 1 && s.start >= 0);
            if all_simple {
                let mut ranges = Vec::with_capacity(ndim * 2);
                let shape = {
                    let mut buf = [0i32; 8];
                    let n = unsafe { npu_get_shape(tensor.handle, buf.as_mut_ptr(), 8) } as usize;
                    buf[..n].iter().map(|&d| d as usize).collect::<Vec<_>>()
                };
                for (i, s) in slices.iter().enumerate() {
                    let start = s.start as i32;
                    let end = s.end.map(|e| e as i32).unwrap_or(shape[i] as i32);
                    ranges.push(start);
                    ranges.push(end);
                }
                let result = unsafe { npu_slice(tensor.handle, ranges.as_ptr(), ndim as i32) };
                if result >= 0 {
                    return NpuFloatTensor { handle: result };
                }
            }
        }
        // Fallback for complex slices
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_slice(nd, slices);
        ndarray_to_npu(&result)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // No direct FFI for slice_assign — round-trip
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result = <Nd as FloatTensorOps<Nd>>::float_slice_assign(nd_tensor, slices, nd_value);
        ndarray_to_npu(&result)
    }

    // ── Mask ────────────────────────────────────────────────────────────

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // Round-trip: mask is NdArrayTensor<bool>, needs conversion
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result = <Nd as FloatTensorOps<Nd>>::float_mask_where(nd_tensor, mask, nd_value);
        ndarray_to_npu(&result)
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: f32,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_mask_fill(nd_tensor, mask, value);
        ndarray_to_npu(&result)
    }

    // ── Comparison (return BoolTensor = NdArrayTensor<bool>) ────────────

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let h = unsafe { npu_equal(lhs.handle, rhs.handle) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let rhs_h = unsafe { npu_scalar_tensor(rhs) };
        let h = unsafe { npu_equal(lhs.handle, rhs_h) };
        unsafe { npu_free_tensor(rhs_h) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let h = unsafe { npu_greater(lhs.handle, rhs.handle) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let rhs_h = unsafe { npu_scalar_tensor(rhs) };
        let h = unsafe { npu_greater(lhs.handle, rhs_h) };
        unsafe { npu_free_tensor(rhs_h) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        // greater_equal = NOT less
        let h = unsafe { npu_less(lhs.handle, rhs.handle) };
        let (data, shape) = read_f32(h);
        unsafe { npu_free_tensor(h) };
        let bool_data: Vec<bool> = data.iter().map(|&v| v == 0.0).collect(); // invert
        let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), bool_data)
            .unwrap()
            .into_shared();
        NdArrayTensor::from(array)
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let rhs_h = unsafe { npu_scalar_tensor(rhs) };
        let h = unsafe { npu_less(lhs.handle, rhs_h) };
        unsafe { npu_free_tensor(rhs_h) };
        let (data, shape) = read_f32(h);
        unsafe { npu_free_tensor(h) };
        let bool_data: Vec<bool> = data.iter().map(|&v| v == 0.0).collect();
        let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), bool_data)
            .unwrap()
            .into_shared();
        NdArrayTensor::from(array)
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let h = unsafe { npu_less(lhs.handle, rhs.handle) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let rhs_h = unsafe { npu_scalar_tensor(rhs) };
        let h = unsafe { npu_less(lhs.handle, rhs_h) };
        unsafe { npu_free_tensor(rhs_h) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        // lower_equal = NOT greater
        let h = unsafe { npu_greater(lhs.handle, rhs.handle) };
        let (data, shape) = read_f32(h);
        unsafe { npu_free_tensor(h) };
        let bool_data: Vec<bool> = data.iter().map(|&v| v == 0.0).collect();
        let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), bool_data)
            .unwrap()
            .into_shared();
        NdArrayTensor::from(array)
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let rhs_h = unsafe { npu_scalar_tensor(rhs) };
        let h = unsafe { npu_greater(lhs.handle, rhs_h) };
        unsafe { npu_free_tensor(rhs_h) };
        let (data, shape) = read_f32(h);
        unsafe { npu_free_tensor(h) };
        let bool_data: Vec<bool> = data.iter().map(|&v| v == 0.0).collect();
        let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), bool_data)
            .unwrap()
            .into_shared();
        NdArrayTensor::from(array)
    }

    // ── Reductions ──────────────────────────────────────────────────────

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sum(tensor.handle) },
        }
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sum_dim(tensor.handle, dim as i32) },
        }
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_mean_all(tensor.handle) },
        }
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_mean(tensor.handle, dim as i32) },
        }
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_prod(nd);
        ndarray_to_npu(&result)
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_prod_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cumsum(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cumprod(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cummin(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cummax(nd, dim);
        ndarray_to_npu(&result)
    }

    // ── Argmax / Argmin (return IntTensor = NdArrayTensor) ──────────────

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        let h = unsafe { npu_argmax(tensor.handle, dim as i32) };
        int_handle_to_ndarray(h)
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        let h = unsafe { npu_argmin(tensor.handle, dim as i32) };
        int_handle_to_ndarray(h)
    }

    // ── Max / Min ───────────────────────────────────────────────────────

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_max(tensor.handle) },
        }
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_max_dim(tensor.handle, dim as i32) },
        }
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_min(tensor.handle) },
        }
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_min_dim(tensor.handle, dim as i32) },
        }
    }

    // ── Unary math ──────────────────────────────────────────────────────

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_exp(tensor.handle) },
        }
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_log(tensor.handle) },
        }
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        // log1p(x) = log(1 + x)
        let one = unsafe { npu_scalar_tensor(1.0) };
        let sum = unsafe { npu_add(tensor.handle, one) };
        let result = unsafe { npu_log(sum) };
        unsafe {
            npu_free_tensor(one);
            npu_free_tensor(sum);
        }
        NpuFloatTensor { handle: result }
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_pow(lhs.handle, rhs.handle) },
        }
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_pow_scalar(tensor.handle, value) },
        }
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sqrt(tensor.handle) },
        }
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_abs(tensor.handle) },
        }
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_cos(tensor.handle) },
        }
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sin(tensor.handle) },
        }
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_tanh(tensor.handle) },
        }
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_erf(tensor.handle) },
        }
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_floor(tensor.handle) },
        }
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_ceil(tensor.handle) },
        }
    }

    fn float_neg(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_neg(tensor.handle) },
        }
    }

    // Trig ops without direct FFI — round-trip through NdArray
    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_tan(nd);
        ndarray_to_npu(&result)
    }

    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cosh(nd);
        ndarray_to_npu(&result)
    }

    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sinh(nd);
        ndarray_to_npu(&result)
    }

    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_acos(nd);
        ndarray_to_npu(&result)
    }

    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_acosh(nd);
        ndarray_to_npu(&result)
    }

    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_asin(nd);
        ndarray_to_npu(&result)
    }

    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_asinh(nd);
        ndarray_to_npu(&result)
    }

    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_atan(nd);
        ndarray_to_npu(&result)
    }

    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_atanh(nd);
        ndarray_to_npu(&result)
    }

    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_atan2(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_round(nd);
        ndarray_to_npu(&result)
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_trunc(nd);
        ndarray_to_npu(&result)
    }

    // ── Clamp ───────────────────────────────────────────────────────────

    fn float_clamp_min(tensor: FloatTensor<Self>, min: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_clamp_min(tensor.handle, min) },
        }
    }

    fn float_clamp_max(tensor: FloatTensor<Self>, max: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_clamp_max(tensor.handle, max) },
        }
    }

    fn float_clamp(tensor: FloatTensor<Self>, min: f32, max: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_clamp(tensor.handle, min, max) },
        }
    }

    // ── Cat ─────────────────────────────────────────────────────────────

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        let handles: Vec<i32> = tensors.iter().map(|t| t.handle).collect();
        NpuFloatTensor {
            handle: unsafe { npu_cat(handles.as_ptr(), handles.len() as i32, dim as i32) },
        }
    }

    // ── Sign ────────────────────────────────────────────────────────────

    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sign(nd);
        ndarray_to_npu(&result)
    }

    // ── Cast ────────────────────────────────────────────────────────────

    fn float_cast(tensor: FloatTensor<Self>, _dtype: FloatDType) -> FloatTensor<Self> {
        // MLTensor only supports f32; casting is a no-op
        tensor
    }

    // ── Grid sample ─────────────────────────────────────────────────────

    fn float_grid_sample_2d(
        tensor: FloatTensor<Self>,
        grid: FloatTensor<Self>,
        options: GridSampleOptions,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_grid = npu_to_ndarray(&grid);
        let result = <Nd as FloatTensorOps<Nd>>::float_grid_sample_2d(nd_tensor, nd_grid, options);
        ndarray_to_npu(&result)
    }

    // ── Unfold ──────────────────────────────────────────────────────────

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_unfold(nd, dim, size, step);
        ndarray_to_npu(&result)
    }
}

// ===========================================================================
// FloatTensorOps — no apple: full NdArray delegation
// ===========================================================================
#[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
impl FloatTensorOps<Self> for NpuBurnBackend {
    fn float_from_data(data: TensorData, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_from_data(data, &nd_dev())
    }

    fn float_random(shape: Shape, distribution: Distribution, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_random(shape, distribution, &nd_dev())
    }

    fn float_zeros(shape: Shape, _device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_zeros(shape, &nd_dev(), dtype)
    }

    fn float_ones(shape: Shape, _device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_ones(shape, &nd_dev(), dtype)
    }

    fn float_full(shape: Shape, fill_value: f32, _device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_full(shape, fill_value, &nd_dev(), dtype)
    }

    fn float_device(_tensor: &FloatTensor<Self>) -> NpuBurnDevice { NpuBurnDevice::Default }
    fn float_to_device(tensor: FloatTensor<Self>, _device: &NpuBurnDevice) -> FloatTensor<Self> { tensor }

    fn float_empty(shape: Shape, _device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_empty(shape, &nd_dev(), dtype)
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as FloatTensorOps<Nd>>::float_into_data(tensor).await
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_matmul(lhs, rhs)
    }
    fn float_cross(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cross(lhs, rhs, dim)
    }
    fn float_into_int(tensor: FloatTensor<Self>) -> IntTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_into_int(tensor)
    }
    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_add(lhs, rhs)
    }
    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_add_scalar(lhs, rhs)
    }
    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sub(lhs, rhs)
    }
    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sub_scalar(lhs, rhs)
    }
    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_mul(lhs, rhs)
    }
    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_mul_scalar(lhs, rhs)
    }
    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_div(lhs, rhs)
    }
    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_div_scalar(lhs, rhs)
    }
    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_remainder(lhs, rhs)
    }
    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_remainder_scalar(lhs, rhs)
    }
    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_recip(tensor)
    }
    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_swap_dims(tensor, dim1, dim2)
    }
    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_permute(tensor, axes)
    }
    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_flip(tensor, axes)
    }
    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_reshape(tensor, shape)
    }
    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_expand(tensor, shape)
    }
    fn float_gather(dim: usize, tensor: FloatTensor<Self>, indices: IntTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_gather(dim, tensor, indices)
    }
    fn float_scatter_add(dim: usize, tensor: FloatTensor<Self>, indices: IntTensor<Self>, value: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_scatter_add(dim, tensor, indices, value)
    }
    fn float_select(tensor: FloatTensor<Self>, dim: usize, indices: IntTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_select(tensor, dim, indices)
    }
    fn float_select_add(tensor: FloatTensor<Self>, dim: usize, indices: IntTensor<Self>, value: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_select_add(tensor, dim, indices, value)
    }
    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_slice(tensor, slices)
    }
    fn float_slice_assign(tensor: FloatTensor<Self>, slices: &[Slice], value: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_slice_assign(tensor, slices, value)
    }
    fn float_mask_where(tensor: FloatTensor<Self>, mask: BoolTensor<Self>, value: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_mask_where(tensor, mask, value)
    }
    fn float_mask_fill(tensor: FloatTensor<Self>, mask: BoolTensor<Self>, value: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_mask_fill(tensor, mask, value)
    }
    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_equal(lhs, rhs)
    }
    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_equal_elem(lhs, rhs)
    }
    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_greater(lhs, rhs)
    }
    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_greater_elem(lhs, rhs)
    }
    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_greater_equal(lhs, rhs)
    }
    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_greater_equal_elem(lhs, rhs)
    }
    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_lower(lhs, rhs)
    }
    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_lower_elem(lhs, rhs)
    }
    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_lower_equal(lhs, rhs)
    }
    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_lower_equal_elem(lhs, rhs)
    }
    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sum(tensor)
    }
    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sum_dim(tensor, dim)
    }
    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_mean(tensor)
    }
    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_mean_dim(tensor, dim)
    }
    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_prod(tensor)
    }
    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_prod_dim(tensor, dim)
    }
    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cumsum(tensor, dim)
    }
    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cumprod(tensor, dim)
    }
    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cummin(tensor, dim)
    }
    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cummax(tensor, dim)
    }
    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_argmax(tensor, dim)
    }
    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_argmin(tensor, dim)
    }
    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_exp(tensor)
    }
    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_log(tensor)
    }
    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_log1p(tensor)
    }
    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_powf(lhs, rhs)
    }
    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_powf_scalar_impl(tensor, value)
    }
    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sqrt(tensor)
    }
    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_abs(tensor)
    }
    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cos(tensor)
    }
    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sin(tensor)
    }
    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_tan(tensor)
    }
    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cosh(tensor)
    }
    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sinh(tensor)
    }
    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_tanh(tensor)
    }
    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_acos(tensor)
    }
    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_acosh(tensor)
    }
    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_asin(tensor)
    }
    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_asinh(tensor)
    }
    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_atan(tensor)
    }
    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_atanh(tensor)
    }
    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_atan2(lhs, rhs)
    }
    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_round(tensor)
    }
    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_floor(tensor)
    }
    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_ceil(tensor)
    }
    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_trunc(tensor)
    }
    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_erf(tensor)
    }
    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cat(tensors, dim)
    }
    fn float_clamp_min(tensor: FloatTensor<Self>, min: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_clamp_min(tensor, min)
    }
    fn float_clamp_max(tensor: FloatTensor<Self>, max: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_clamp_max(tensor, max)
    }
    fn float_clamp(tensor: FloatTensor<Self>, min: f32, max: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_clamp(tensor, min, max)
    }
    fn float_neg(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_neg(tensor)
    }
    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sign(tensor)
    }
    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cast(tensor, dtype)
    }
    fn float_grid_sample_2d(tensor: FloatTensor<Self>, grid: FloatTensor<Self>, options: GridSampleOptions) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_grid_sample_2d(tensor, grid, options)
    }
    fn float_unfold(tensor: FloatTensor<Self>, dim: usize, size: usize, step: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_unfold(tensor, dim, size, step)
    }
}

// ===========================================================================
// IntTensorOps — apple: same as before but int_into_float returns NpuFloatTensor
// ===========================================================================
#[cfg(feature = "apple")]
impl IntTensorOps<Self> for NpuBurnBackend {
    fn int_from_data(data: TensorData, _device: &NpuBurnDevice) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_from_data(data, &nd_dev())
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as IntTensorOps<Nd>>::int_into_data(tensor).await
    }

    fn int_device(_tensor: &IntTensor<Self>) -> NpuBurnDevice { NpuBurnDevice::Default }
    fn int_to_device(tensor: IntTensor<Self>, _device: &NpuBurnDevice) -> IntTensor<Self> { tensor }

    fn int_empty(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_empty(shape, &nd_dev(), dtype)
    }
    fn int_zeros(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_zeros(shape, &nd_dev(), dtype)
    }
    fn int_ones(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_ones(shape, &nd_dev(), dtype)
    }
    fn int_full(shape: Shape, fill_value: i64, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_full(shape, fill_value, &nd_dev(), dtype)
    }
    fn int_random(shape: Shape, distribution: Distribution, _device: &NpuBurnDevice) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_random(shape, distribution, &nd_dev())
    }
    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_reshape(tensor, shape)
    }
    fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_slice(tensor, slices)
    }
    fn int_slice_assign(tensor: IntTensor<Self>, slices: &[Slice], value: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_slice_assign(tensor, slices, value)
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        // Convert NdArrayTensor<i64> -> NpuFloatTensor
        // Read i64 data from ndarray, convert to f32, send to NPU
        let nd_float = <Nd as IntTensorOps<Nd>>::int_into_float(tensor);
        ndarray_to_npu(&nd_float)
    }

    fn int_mask_where(tensor: IntTensor<Self>, mask: BoolTensor<Self>, source: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mask_where(tensor, mask, source)
    }
    fn int_mask_fill(tensor: IntTensor<Self>, mask: BoolTensor<Self>, value: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mask_fill(tensor, mask, value)
    }
    fn int_gather(dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_gather(dim, tensor, indices)
    }
    fn int_scatter_add(dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>, value: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_scatter_add(dim, tensor, indices, value)
    }
    fn int_select(tensor: IntTensor<Self>, dim: usize, indices: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_select(tensor, dim, indices)
    }
    fn int_select_add(tensor: IntTensor<Self>, dim: usize, indices: IntTensor<Self>, value: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_select_add(tensor, dim, indices, value)
    }
    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cat(tensors, dim)
    }
    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_equal(lhs, rhs)
    }
    fn int_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_equal_elem(lhs, rhs)
    }
    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater(lhs, rhs)
    }
    fn int_greater_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_elem(lhs, rhs)
    }
    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_equal(lhs, rhs)
    }
    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_equal_elem(lhs, rhs)
    }
    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower(lhs, rhs)
    }
    fn int_lower_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_elem(lhs, rhs)
    }
    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_equal(lhs, rhs)
    }
    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_equal_elem(lhs, rhs)
    }
    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_add(lhs, rhs)
    }
    fn int_add_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_add_scalar(lhs, rhs)
    }
    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sub(lhs, rhs)
    }
    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sub_scalar(lhs, rhs)
    }
    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mul(lhs, rhs)
    }
    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mul_scalar(lhs, rhs)
    }
    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_div(lhs, rhs)
    }
    fn int_div_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_div_scalar(lhs, rhs)
    }
    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_remainder(lhs, rhs)
    }
    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_remainder_scalar(lhs, rhs)
    }
    fn int_neg(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_neg(tensor)
    }
    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sum(tensor)
    }
    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sum_dim(tensor, dim)
    }
    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_prod(tensor)
    }
    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_prod_dim(tensor, dim)
    }
    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mean(tensor)
    }
    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mean_dim(tensor, dim)
    }
    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_max(tensor)
    }
    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_min(tensor)
    }
    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cumsum(tensor, dim)
    }
    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cumprod(tensor, dim)
    }
    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cummin(tensor, dim)
    }
    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cummax(tensor, dim)
    }
    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_matmul(lhs, rhs)
    }
    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_argmax(tensor, dim)
    }
    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_argmin(tensor, dim)
    }
    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_abs(tensor)
    }
    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_swap_dims(tensor, dim1, dim2)
    }
    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_permute(tensor, axes)
    }
    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_flip(tensor, axes)
    }
    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_expand(tensor, shape)
    }
    fn int_sign(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sign(tensor)
    }
    fn int_powi(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_powi(lhs, rhs)
    }
    fn int_powf(lhs: IntTensor<Self>, rhs: FloatTensor<Self>) -> IntTensor<Self> {
        // rhs is NpuFloatTensor, convert to NdArrayTensor for NdArray
        let nd_rhs = npu_to_ndarray(&rhs);
        <Nd as IntTensorOps<Nd>>::int_powf(lhs, nd_rhs)
    }
    fn int_powf_scalar_impl(lhs: IntTensor<Self>, rhs: f32) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_powf_scalar_impl(lhs, rhs)
    }
    fn int_clamp_min(tensor: IntTensor<Self>, min: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp_min(tensor, min)
    }
    fn int_clamp_max(tensor: IntTensor<Self>, max: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp_max(tensor, max)
    }
    fn int_clamp(tensor: IntTensor<Self>, min: i64, max: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp(tensor, min, max)
    }
    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_and(lhs, rhs)
    }
    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_and_scalar(lhs, rhs)
    }
    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_or(lhs, rhs)
    }
    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_or_scalar(lhs, rhs)
    }
    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_xor(lhs, rhs)
    }
    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_xor_scalar(lhs, rhs)
    }
    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_not(tensor)
    }
    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_left_shift(lhs, rhs)
    }
    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_left_shift_scalar(lhs, rhs)
    }
    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_right_shift(lhs, rhs)
    }
    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_right_shift_scalar(lhs, rhs)
    }
    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cast(tensor, dtype)
    }
    fn int_unfold(tensor: IntTensor<Self>, dim: usize, size: usize, step: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_unfold(tensor, dim, size, step)
    }
}

// ===========================================================================
// IntTensorOps — no apple: full delegation
// ===========================================================================
#[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
impl IntTensorOps<Self> for NpuBurnBackend {
    fn int_from_data(data: TensorData, _device: &NpuBurnDevice) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_from_data(data, &nd_dev())
    }
    async fn int_into_data(tensor: IntTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as IntTensorOps<Nd>>::int_into_data(tensor).await
    }
    fn int_device(_tensor: &IntTensor<Self>) -> NpuBurnDevice { NpuBurnDevice::Default }
    fn int_to_device(tensor: IntTensor<Self>, _device: &NpuBurnDevice) -> IntTensor<Self> { tensor }
    fn int_empty(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_empty(shape, &nd_dev(), dtype)
    }
    fn int_zeros(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_zeros(shape, &nd_dev(), dtype)
    }
    fn int_ones(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_ones(shape, &nd_dev(), dtype)
    }
    fn int_full(shape: Shape, fill_value: i64, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_full(shape, fill_value, &nd_dev(), dtype)
    }
    fn int_random(shape: Shape, distribution: Distribution, _device: &NpuBurnDevice) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_random(shape, distribution, &nd_dev())
    }
    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_reshape(tensor, shape)
    }
    fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_slice(tensor, slices)
    }
    fn int_slice_assign(tensor: IntTensor<Self>, slices: &[Slice], value: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_slice_assign(tensor, slices, value)
    }
    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_into_float(tensor)
    }
    fn int_mask_where(tensor: IntTensor<Self>, mask: BoolTensor<Self>, source: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mask_where(tensor, mask, source)
    }
    fn int_mask_fill(tensor: IntTensor<Self>, mask: BoolTensor<Self>, value: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mask_fill(tensor, mask, value)
    }
    fn int_gather(dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_gather(dim, tensor, indices)
    }
    fn int_scatter_add(dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>, value: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_scatter_add(dim, tensor, indices, value)
    }
    fn int_select(tensor: IntTensor<Self>, dim: usize, indices: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_select(tensor, dim, indices)
    }
    fn int_select_add(tensor: IntTensor<Self>, dim: usize, indices: IntTensor<Self>, value: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_select_add(tensor, dim, indices, value)
    }
    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cat(tensors, dim)
    }
    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_equal(lhs, rhs)
    }
    fn int_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_equal_elem(lhs, rhs)
    }
    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater(lhs, rhs)
    }
    fn int_greater_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_elem(lhs, rhs)
    }
    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_equal(lhs, rhs)
    }
    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_equal_elem(lhs, rhs)
    }
    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower(lhs, rhs)
    }
    fn int_lower_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_elem(lhs, rhs)
    }
    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_equal(lhs, rhs)
    }
    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_equal_elem(lhs, rhs)
    }
    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_add(lhs, rhs)
    }
    fn int_add_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_add_scalar(lhs, rhs)
    }
    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sub(lhs, rhs)
    }
    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sub_scalar(lhs, rhs)
    }
    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mul(lhs, rhs)
    }
    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mul_scalar(lhs, rhs)
    }
    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_div(lhs, rhs)
    }
    fn int_div_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_div_scalar(lhs, rhs)
    }
    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_remainder(lhs, rhs)
    }
    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_remainder_scalar(lhs, rhs)
    }
    fn int_neg(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_neg(tensor)
    }
    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sum(tensor)
    }
    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sum_dim(tensor, dim)
    }
    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_prod(tensor)
    }
    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_prod_dim(tensor, dim)
    }
    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mean(tensor)
    }
    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mean_dim(tensor, dim)
    }
    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_max(tensor)
    }
    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_min(tensor)
    }
    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cumsum(tensor, dim)
    }
    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cumprod(tensor, dim)
    }
    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cummin(tensor, dim)
    }
    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cummax(tensor, dim)
    }
    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_matmul(lhs, rhs)
    }
    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_argmax(tensor, dim)
    }
    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_argmin(tensor, dim)
    }
    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_abs(tensor)
    }
    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_swap_dims(tensor, dim1, dim2)
    }
    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_permute(tensor, axes)
    }
    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_flip(tensor, axes)
    }
    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_expand(tensor, shape)
    }
    fn int_sign(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sign(tensor)
    }
    fn int_powi(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_powi(lhs, rhs)
    }
    fn int_powf(lhs: IntTensor<Self>, rhs: FloatTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_powf(lhs, rhs)
    }
    fn int_powf_scalar_impl(lhs: IntTensor<Self>, rhs: f32) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_powf_scalar_impl(lhs, rhs)
    }
    fn int_clamp_min(tensor: IntTensor<Self>, min: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp_min(tensor, min)
    }
    fn int_clamp_max(tensor: IntTensor<Self>, max: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp_max(tensor, max)
    }
    fn int_clamp(tensor: IntTensor<Self>, min: i64, max: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp(tensor, min, max)
    }
    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_and(lhs, rhs)
    }
    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_and_scalar(lhs, rhs)
    }
    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_or(lhs, rhs)
    }
    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_or_scalar(lhs, rhs)
    }
    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_xor(lhs, rhs)
    }
    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_xor_scalar(lhs, rhs)
    }
    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_not(tensor)
    }
    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_left_shift(lhs, rhs)
    }
    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_left_shift_scalar(lhs, rhs)
    }
    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_right_shift(lhs, rhs)
    }
    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_right_shift_scalar(lhs, rhs)
    }
    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cast(tensor, dtype)
    }
    fn int_unfold(tensor: IntTensor<Self>, dim: usize, size: usize, step: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_unfold(tensor, dim, size, step)
    }
}

// ===========================================================================
// BoolTensorOps — apple: bool_into_float returns NpuFloatTensor
// ===========================================================================
#[cfg(feature = "apple")]
impl BoolTensorOps<Self> for NpuBurnBackend {
    fn bool_from_data(data: TensorData, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_from_data(data, &nd_dev())
    }
    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as BoolTensorOps<Nd>>::bool_into_data(tensor).await
    }
    fn bool_device(_tensor: &BoolTensor<Self>) -> NpuBurnDevice { NpuBurnDevice::Default }
    fn bool_to_device(tensor: BoolTensor<Self>, _device: &NpuBurnDevice) -> BoolTensor<Self> { tensor }
    fn bool_empty(shape: Shape, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_empty(shape, &nd_dev())
    }
    fn bool_zeros(shape: Shape, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_zeros(shape, &nd_dev())
    }
    fn bool_ones(shape: Shape, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_ones(shape, &nd_dev())
    }
    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_into_int(tensor)
    }

    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        // Convert NdArrayTensor<bool> -> f32 -> NpuFloatTensor
        let nd_float = <Nd as BoolTensorOps<Nd>>::bool_into_float(tensor);
        ndarray_to_npu(&nd_float)
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_reshape(tensor, shape)
    }
    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_slice(tensor, slices)
    }
    fn bool_slice_assign(tensor: BoolTensor<Self>, slices: &[Slice], value: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_slice_assign(tensor, slices, value)
    }
    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_equal(lhs, rhs)
    }
    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_not(tensor)
    }
    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_and(lhs, rhs)
    }
    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_or(lhs, rhs)
    }
    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_swap_dims(tensor, dim1, dim2)
    }
    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_permute(tensor, axes)
    }
    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_flip(tensor, axes)
    }
    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_expand(tensor, shape)
    }
    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_cat(tensors, dim)
    }
    fn bool_select(tensor: BoolTensor<Self>, dim: usize, indices: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_select(tensor, dim, indices)
    }
    fn bool_select_or(tensor: BoolTensor<Self>, dim: usize, indices: IntTensor<Self>, value: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_select_or(tensor, dim, indices, value)
    }
    fn bool_unfold(tensor: BoolTensor<Self>, dim: usize, size: usize, step: usize) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_unfold(tensor, dim, size, step)
    }
    fn bool_mask_where(tensor: BoolTensor<Self>, mask: BoolTensor<Self>, value: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_mask_where(tensor, mask, value)
    }
    fn bool_mask_fill(tensor: BoolTensor<Self>, mask: BoolTensor<Self>, value: bool) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_mask_fill(tensor, mask, value)
    }
    fn bool_gather(dim: usize, tensor: BoolTensor<Self>, indices: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_gather(dim, tensor, indices)
    }
    fn bool_scatter_or(dim: usize, tensor: BoolTensor<Self>, indices: IntTensor<Self>, value: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_scatter_or(dim, tensor, indices, value)
    }
    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: bool) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_equal_elem(lhs, rhs)
    }
    fn bool_any(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_any(tensor)
    }
    fn bool_all(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_all(tensor)
    }
}

// ===========================================================================
// BoolTensorOps — no apple: full delegation
// ===========================================================================
#[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
impl BoolTensorOps<Self> for NpuBurnBackend {
    fn bool_from_data(data: TensorData, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_from_data(data, &nd_dev())
    }
    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as BoolTensorOps<Nd>>::bool_into_data(tensor).await
    }
    fn bool_device(_tensor: &BoolTensor<Self>) -> NpuBurnDevice { NpuBurnDevice::Default }
    fn bool_to_device(tensor: BoolTensor<Self>, _device: &NpuBurnDevice) -> BoolTensor<Self> { tensor }
    fn bool_empty(shape: Shape, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_empty(shape, &nd_dev())
    }
    fn bool_zeros(shape: Shape, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_zeros(shape, &nd_dev())
    }
    fn bool_ones(shape: Shape, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_ones(shape, &nd_dev())
    }
    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_into_int(tensor)
    }
    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_into_float(tensor)
    }
    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_reshape(tensor, shape)
    }
    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_slice(tensor, slices)
    }
    fn bool_slice_assign(tensor: BoolTensor<Self>, slices: &[Slice], value: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_slice_assign(tensor, slices, value)
    }
    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_equal(lhs, rhs)
    }
    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_not(tensor)
    }
    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_and(lhs, rhs)
    }
    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_or(lhs, rhs)
    }
    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_swap_dims(tensor, dim1, dim2)
    }
    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_permute(tensor, axes)
    }
    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_flip(tensor, axes)
    }
    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_expand(tensor, shape)
    }
    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_cat(tensors, dim)
    }
    fn bool_select(tensor: BoolTensor<Self>, dim: usize, indices: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_select(tensor, dim, indices)
    }
    fn bool_select_or(tensor: BoolTensor<Self>, dim: usize, indices: IntTensor<Self>, value: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_select_or(tensor, dim, indices, value)
    }
    fn bool_unfold(tensor: BoolTensor<Self>, dim: usize, size: usize, step: usize) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_unfold(tensor, dim, size, step)
    }
    fn bool_mask_where(tensor: BoolTensor<Self>, mask: BoolTensor<Self>, value: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_mask_where(tensor, mask, value)
    }
    fn bool_mask_fill(tensor: BoolTensor<Self>, mask: BoolTensor<Self>, value: bool) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_mask_fill(tensor, mask, value)
    }
    fn bool_gather(dim: usize, tensor: BoolTensor<Self>, indices: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_gather(dim, tensor, indices)
    }
    fn bool_scatter_or(dim: usize, tensor: BoolTensor<Self>, indices: IntTensor<Self>, value: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_scatter_or(dim, tensor, indices, value)
    }
    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: bool) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_equal_elem(lhs, rhs)
    }
    fn bool_any(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_any(tensor)
    }
    fn bool_all(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_all(tensor)
    }
}

// ===========================================================================
// ModuleOps — apple: round-trip through NdArray for conv/pool/interpolate
// ===========================================================================
#[cfg(feature = "apple")]
impl ModuleOps<Self> for NpuBurnBackend {
    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv2d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_off = npu_to_ndarray(&offset);
        let nd_w = npu_to_ndarray(&weight);
        let nd_m = mask.as_ref().map(npu_to_ndarray);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::deform_conv2d(nd_x, nd_off, nd_w, nd_m, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn deform_conv2d_backward(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_off = npu_to_ndarray(&offset);
        let nd_w = npu_to_ndarray(&weight);
        let nd_m = mask.as_ref().map(npu_to_ndarray);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let nd_g = npu_to_ndarray(&output_grad);
        let r = <Nd as ModuleOps<Nd>>::deform_conv2d_backward(
            nd_x, nd_off, nd_w, nd_m, nd_b, nd_g, options,
        );
        DeformConv2dBackward::new(
            ndarray_to_npu(&r.x_grad),
            ndarray_to_npu(&r.offset_grad),
            ndarray_to_npu(&r.weight_grad),
            r.mask_grad.map(|g| ndarray_to_npu(&g)),
            r.bias_grad.map(|g| ndarray_to_npu(&g)),
        )
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv3d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv_transpose2d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv_transpose3d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::avg_pool2d(nd_x, kernel_size, stride, padding, count_include_pad, ceil_mode);
        ndarray_to_npu(&result)
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&grad);
        let result = <Nd as ModuleOps<Nd>>::avg_pool2d_backward(
            nd_x, nd_g, kernel_size, stride, padding, count_include_pad, ceil_mode,
        );
        ndarray_to_npu(&result)
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::adaptive_avg_pool2d(nd_x, output_size);
        ndarray_to_npu(&result)
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&grad);
        let result = <Nd as ModuleOps<Nd>>::adaptive_avg_pool2d_backward(nd_x, nd_g);
        ndarray_to_npu(&result)
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::max_pool2d(nd_x, kernel_size, stride, padding, dilation, ceil_mode);
        ndarray_to_npu(&result)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::max_pool2d_with_indices(
            nd_x, kernel_size, stride, padding, dilation, ceil_mode,
        );
        MaxPool2dWithIndices::new(ndarray_to_npu(&result.output), result.indices)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool2dBackward<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&output_grad);
        let result = <Nd as ModuleOps<Nd>>::max_pool2d_with_indices_backward(
            nd_x, kernel_size, stride, padding, dilation, ceil_mode, nd_g, indices,
        );
        MaxPool2dBackward::new(ndarray_to_npu(&result.x_grad))
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::interpolate(nd_x, output_size, options);
        ndarray_to_npu(&result)
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&grad);
        let result = <Nd as ModuleOps<Nd>>::interpolate_backward(nd_x, nd_g, output_size, options);
        ndarray_to_npu(&result)
    }
}

// ===========================================================================
// ModuleOps — no apple: full delegation
// ===========================================================================
#[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
impl ModuleOps<Self> for NpuBurnBackend {
    fn conv2d(x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvOptions<2>) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::conv2d(x, weight, bias, options)
    }
    fn deform_conv2d(x: FloatTensor<Self>, offset: FloatTensor<Self>, weight: FloatTensor<Self>, mask: Option<FloatTensor<Self>>, bias: Option<FloatTensor<Self>>, options: DeformConvOptions<2>) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::deform_conv2d(x, offset, weight, mask, bias, options)
    }
    fn deform_conv2d_backward(x: FloatTensor<Self>, offset: FloatTensor<Self>, weight: FloatTensor<Self>, mask: Option<FloatTensor<Self>>, bias: Option<FloatTensor<Self>>, output_grad: FloatTensor<Self>, options: DeformConvOptions<2>) -> DeformConv2dBackward<Self> {
        let r = <Nd as ModuleOps<Nd>>::deform_conv2d_backward(x, offset, weight, mask, bias, output_grad, options);
        DeformConv2dBackward::new(r.x_grad, r.offset_grad, r.weight_grad, r.mask_grad, r.bias_grad)
    }
    fn conv3d(x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvOptions<3>) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::conv3d(x, weight, bias, options)
    }
    fn conv_transpose2d(x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvTransposeOptions<2>) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::conv_transpose2d(x, weight, bias, options)
    }
    fn conv_transpose3d(x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvTransposeOptions<3>) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::conv_transpose3d(x, weight, bias, options)
    }
    fn avg_pool2d(x: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], count_include_pad: bool, ceil_mode: bool) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::avg_pool2d(x, kernel_size, stride, padding, count_include_pad, ceil_mode)
    }
    fn avg_pool2d_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], count_include_pad: bool, ceil_mode: bool) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::avg_pool2d_backward(x, grad, kernel_size, stride, padding, count_include_pad, ceil_mode)
    }
    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::adaptive_avg_pool2d(x, output_size)
    }
    fn adaptive_avg_pool2d_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::adaptive_avg_pool2d_backward(x, grad)
    }
    fn max_pool2d(x: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2], ceil_mode: bool) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
    }
    fn max_pool2d_with_indices(x: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2], ceil_mode: bool) -> MaxPool2dWithIndices<Self> {
        let r = <Nd as ModuleOps<Nd>>::max_pool2d_with_indices(x, kernel_size, stride, padding, dilation, ceil_mode);
        MaxPool2dWithIndices::new(r.output, r.indices)
    }
    fn max_pool2d_with_indices_backward(x: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2], ceil_mode: bool, output_grad: FloatTensor<Self>, indices: IntTensor<Self>) -> MaxPool2dBackward<Self> {
        let r = <Nd as ModuleOps<Nd>>::max_pool2d_with_indices_backward(x, kernel_size, stride, padding, dilation, ceil_mode, output_grad, indices);
        MaxPool2dBackward::new(r.x_grad)
    }
    fn interpolate(x: FloatTensor<Self>, output_size: [usize; 2], options: InterpolateOptions) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::interpolate(x, output_size, options)
    }
    fn interpolate_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>, output_size: [usize; 2], options: InterpolateOptions) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::interpolate_backward(x, grad, output_size, options)
    }
}

// ===========================================================================
// ActivationOps (all methods have defaults)
// ===========================================================================
impl ActivationOps<Self> for NpuBurnBackend {}

// ===========================================================================
// QTensorOps — apple: quantize/dequantize bridge NpuFloatTensor <-> NdArray
// ===========================================================================
#[cfg(feature = "apple")]
impl QTensorOps<Self> for NpuBurnBackend {
    fn q_from_data(data: TensorData, _device: &NpuBurnDevice) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_from_data(data, &nd_dev())
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        // Convert NpuFloatTensor -> NdArrayTensor for NdArray's quantize
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_scales = npu_to_ndarray(&qparams.scales);
        let nd_qparams = QuantizationParametersPrimitive::<Nd> {
            scales: nd_scales,
        };
        <Nd as QTensorOps<Nd>>::quantize(nd_tensor, scheme, nd_qparams)
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        let nd_result = <Nd as QTensorOps<Nd>>::dequantize(tensor);
        ndarray_to_npu(&nd_result)
    }

    fn q_device(_tensor: &QuantizedTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, _device: &NpuBurnDevice) -> QuantizedTensor<Self> {
        tensor
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as QTensorOps<Nd>>::q_into_data(tensor).await
    }

    fn q_swap_dims(tensor: QuantizedTensor<Self>, dim1: usize, dim2: usize) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_permute(tensor, axes)
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_flip(tensor, axes)
    }

    fn q_gather(dim: usize, tensor: QuantizedTensor<Self>, indices: IntTensor<Self>) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_gather(dim, tensor, indices)
    }

    fn q_select(tensor: QuantizedTensor<Self>, dim: usize, indices: IntTensor<Self>) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_select(tensor, dim, indices)
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_slice(tensor, slices)
    }

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_argmax(tensor, dim)
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_argmin(tensor, dim)
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_expand(tensor, shape)
    }
}

// ===========================================================================
// QTensorOps — no apple: full delegation
// ===========================================================================
#[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
impl QTensorOps<Self> for NpuBurnBackend {
    fn q_from_data(data: TensorData, _device: &NpuBurnDevice) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_from_data(data, &nd_dev())
    }
    fn quantize(tensor: FloatTensor<Self>, scheme: &QuantScheme, qparams: QuantizationParametersPrimitive<Self>) -> QuantizedTensor<Self> {
        let nd_qparams = QuantizationParametersPrimitive::<Nd> { scales: qparams.scales };
        <Nd as QTensorOps<Nd>>::quantize(tensor, scheme, nd_qparams)
    }
    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        <Nd as QTensorOps<Nd>>::dequantize(tensor)
    }
    fn q_device(_tensor: &QuantizedTensor<Self>) -> NpuBurnDevice { NpuBurnDevice::Default }
    fn q_to_device(tensor: QuantizedTensor<Self>, _device: &NpuBurnDevice) -> QuantizedTensor<Self> { tensor }
    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_reshape(tensor, shape)
    }
    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as QTensorOps<Nd>>::q_into_data(tensor).await
    }
    fn q_swap_dims(tensor: QuantizedTensor<Self>, dim1: usize, dim2: usize) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_swap_dims(tensor, dim1, dim2)
    }
    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_permute(tensor, axes)
    }
    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_flip(tensor, axes)
    }
    fn q_gather(dim: usize, tensor: QuantizedTensor<Self>, indices: IntTensor<Self>) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_gather(dim, tensor, indices)
    }
    fn q_select(tensor: QuantizedTensor<Self>, dim: usize, indices: IntTensor<Self>) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_select(tensor, dim, indices)
    }
    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_slice(tensor, slices)
    }
    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_argmax(tensor, dim)
    }
    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_argmin(tensor, dim)
    }
    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_expand(tensor, shape)
    }
}

// ===========================================================================
// TransactionOps (all methods have defaults)
// ===========================================================================
impl TransactionOps<Self> for NpuBurnBackend {}

// ###########################################################################
// Intel / Qualcomm backends — Vec<f32>-based tensor with NdArray delegation
// ###########################################################################

// ===========================================================================
// Backend impl — intel/qualcomm: FloatTensorPrimitive = NpuFloatTensor (Vec<f32>)
// ===========================================================================
#[cfg(any(feature = "intel", feature = "qualcomm"))]
impl Backend for NpuBurnBackend {
    type Device = NpuBurnDevice;

    type FloatTensorPrimitive = NpuFloatTensor;
    type FloatElem = f32;

    type IntTensorPrimitive = NdArrayTensor;
    type IntElem = i64;

    type BoolTensorPrimitive = NdArrayTensor;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = NdArrayQTensor;

    fn ad_enabled() -> bool {
        false
    }

    fn name(_device: &Self::Device) -> String {
        #[cfg(feature = "intel")]
        { String::from("npu (Intel OpenVINO)") }
        #[cfg(feature = "qualcomm")]
        { String::from("npu (Qualcomm QNN)") }
    }

    fn seed(_device: &Self::Device, seed: u64) {
        <Nd as Backend>::seed(&NdArrayDevice::Cpu, seed);
    }

    fn supports_dtype(_device: &Self::Device, dtype: DType) -> bool {
        <Nd as Backend>::supports_dtype(&NdArrayDevice::Cpu, dtype)
    }
}

// ===========================================================================
// FloatTensorOps — intel/qualcomm: Vec<f32> tensor, NdArray delegation
// ===========================================================================
#[cfg(any(feature = "intel", feature = "qualcomm"))]
impl FloatTensorOps<Self> for NpuBurnBackend {
    fn float_from_data(data: TensorData, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        let floats: Vec<f32> = data.to_vec().unwrap();
        let shape: Vec<usize> = data.shape.to_vec();
        NpuFloatTensor::new(floats, shape)
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        _device: &NpuBurnDevice,
    ) -> FloatTensor<Self> {
        let nd_tensor = <Nd as FloatTensorOps<Nd>>::float_random(shape, distribution, &nd_dev());
        ndarray_to_npu(&nd_tensor)
    }

    fn float_zeros(shape: Shape, _device: &NpuBurnDevice, _dtype: FloatDType) -> FloatTensor<Self> {
        NpuFloatTensor::zeros(shape.dims.to_vec())
    }

    fn float_ones(shape: Shape, _device: &NpuBurnDevice, _dtype: FloatDType) -> FloatTensor<Self> {
        NpuFloatTensor::ones(shape.dims.to_vec())
    }

    fn float_full(
        shape: Shape,
        fill_value: f32,
        _device: &NpuBurnDevice,
        _dtype: FloatDType,
    ) -> FloatTensor<Self> {
        NpuFloatTensor::full(shape.dims.to_vec(), fill_value)
    }

    fn float_device(_tensor: &FloatTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }

    fn float_to_device(tensor: FloatTensor<Self>, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        tensor
    }

    fn float_empty(shape: Shape, device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        Self::float_zeros(shape, device, dtype)
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        let shape = burn_tensor::TensorMetadata::shape(&tensor);
        Ok(TensorData::new(tensor.data, shape))
    }

    // ── Matmul ──────────────────────────────────────────────────────────

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        // Intel: try OpenVINO NPU for large matmuls
        #[cfg(feature = "intel")]
        {
            if let Ok(result) = crate::backends::intel::openvino_matmul(&lhs, &rhs) {
                return result;
            }
            return crate::backends::intel::cpu_matmul(&lhs, &rhs);
        }

        // Qualcomm: CPU matmul (TODO: QNN HTP dispatch)
        #[cfg(feature = "qualcomm")]
        {
            return crate::backends::qualcomm::cpu_matmul(&lhs, &rhs);
        }
    }

    fn float_cross(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_cross(nd_lhs, nd_rhs, dim);
        ndarray_to_npu(&result)
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> IntTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        <Nd as FloatTensorOps<Nd>>::float_into_int(nd)
    }

    // ── Arithmetic ──────────────────────────────────────────────────────

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_add(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_add_scalar(nd, rhs);
        ndarray_to_npu(&result)
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_sub(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_sub_scalar(nd, rhs);
        ndarray_to_npu(&result)
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_mul(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_mul_scalar(nd, rhs);
        ndarray_to_npu(&result)
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_div(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_div_scalar(nd, rhs);
        ndarray_to_npu(&result)
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_remainder(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_remainder_scalar(nd, rhs);
        ndarray_to_npu(&result)
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_recip(nd);
        ndarray_to_npu(&result)
    }

    // ── Shape / layout ──────────────────────────────────────────────────

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_swap_dims(nd, dim1, dim2);
        ndarray_to_npu(&result)
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_permute(nd, axes);
        ndarray_to_npu(&result)
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_flip(nd, axes);
        ndarray_to_npu(&result)
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_reshape(nd, shape);
        ndarray_to_npu(&result)
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_expand(nd, shape);
        ndarray_to_npu(&result)
    }

    // ── Gather / scatter / select ───────────────────────────────────────

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_gather(dim, nd, indices);
        ndarray_to_npu(&result)
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result = <Nd as FloatTensorOps<Nd>>::float_scatter_add(dim, nd_tensor, indices, nd_value);
        ndarray_to_npu(&result)
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_select(nd, dim, indices);
        ndarray_to_npu(&result)
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result = <Nd as FloatTensorOps<Nd>>::float_select_add(nd_tensor, dim, indices, nd_value);
        ndarray_to_npu(&result)
    }

    // ── Slice ───────────────────────────────────────────────────────────

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_slice(nd, slices);
        ndarray_to_npu(&result)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result = <Nd as FloatTensorOps<Nd>>::float_slice_assign(nd_tensor, slices, nd_value);
        ndarray_to_npu(&result)
    }

    // ── Mask ────────────────────────────────────────────────────────────

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result = <Nd as FloatTensorOps<Nd>>::float_mask_where(nd_tensor, mask, nd_value);
        ndarray_to_npu(&result)
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: f32,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_mask_fill(nd_tensor, mask, value);
        ndarray_to_npu(&result)
    }

    // ── Comparison ──────────────────────────────────────────────────────

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        <Nd as FloatTensorOps<Nd>>::float_equal(nd_lhs, nd_rhs)
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        <Nd as FloatTensorOps<Nd>>::float_equal_elem(nd, rhs)
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        <Nd as FloatTensorOps<Nd>>::float_greater(nd_lhs, nd_rhs)
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        <Nd as FloatTensorOps<Nd>>::float_greater_elem(nd, rhs)
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        <Nd as FloatTensorOps<Nd>>::float_greater_equal(nd_lhs, nd_rhs)
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        <Nd as FloatTensorOps<Nd>>::float_greater_equal_elem(nd, rhs)
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        <Nd as FloatTensorOps<Nd>>::float_lower(nd_lhs, nd_rhs)
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        <Nd as FloatTensorOps<Nd>>::float_lower_elem(nd, rhs)
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        <Nd as FloatTensorOps<Nd>>::float_lower_equal(nd_lhs, nd_rhs)
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        <Nd as FloatTensorOps<Nd>>::float_lower_equal_elem(nd, rhs)
    }

    // ── Reductions ──────────────────────────────────────────────────────

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sum(nd);
        ndarray_to_npu(&result)
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sum_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_mean(nd);
        ndarray_to_npu(&result)
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_mean_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_prod(nd);
        ndarray_to_npu(&result)
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_prod_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cumsum(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cumprod(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cummin(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cummax(nd, dim);
        ndarray_to_npu(&result)
    }

    // ── Argmax / Argmin ─────────────────────────────────────────────────

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        <Nd as FloatTensorOps<Nd>>::float_argmax(nd, dim)
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        <Nd as FloatTensorOps<Nd>>::float_argmin(nd, dim)
    }

    // ── Max / Min ───────────────────────────────────────────────────────

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_max(nd);
        ndarray_to_npu(&result)
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_max_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_min(nd);
        ndarray_to_npu(&result)
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_min_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    // ── Unary math ──────────────────────────────────────────────────────

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_exp(nd);
        ndarray_to_npu(&result)
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_log(nd);
        ndarray_to_npu(&result)
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_log1p(nd);
        ndarray_to_npu(&result)
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_powf(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_powf_scalar_impl(nd, value);
        ndarray_to_npu(&result)
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sqrt(nd);
        ndarray_to_npu(&result)
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_abs(nd);
        ndarray_to_npu(&result)
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cos(nd);
        ndarray_to_npu(&result)
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sin(nd);
        ndarray_to_npu(&result)
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_tanh(nd);
        ndarray_to_npu(&result)
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_erf(nd);
        ndarray_to_npu(&result)
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_floor(nd);
        ndarray_to_npu(&result)
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_ceil(nd);
        ndarray_to_npu(&result)
    }

    fn float_neg(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_neg(nd);
        ndarray_to_npu(&result)
    }

    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_tan(nd);
        ndarray_to_npu(&result)
    }

    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cosh(nd);
        ndarray_to_npu(&result)
    }

    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sinh(nd);
        ndarray_to_npu(&result)
    }

    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_acos(nd);
        ndarray_to_npu(&result)
    }

    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_acosh(nd);
        ndarray_to_npu(&result)
    }

    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_asin(nd);
        ndarray_to_npu(&result)
    }

    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_asinh(nd);
        ndarray_to_npu(&result)
    }

    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_atan(nd);
        ndarray_to_npu(&result)
    }

    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_atanh(nd);
        ndarray_to_npu(&result)
    }

    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_atan2(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_round(nd);
        ndarray_to_npu(&result)
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_trunc(nd);
        ndarray_to_npu(&result)
    }

    // ── Clamp ───────────────────────────────────────────────────────────

    fn float_clamp_min(tensor: FloatTensor<Self>, min: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_clamp_min(nd, min);
        ndarray_to_npu(&result)
    }

    fn float_clamp_max(tensor: FloatTensor<Self>, max: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_clamp_max(nd, max);
        ndarray_to_npu(&result)
    }

    fn float_clamp(tensor: FloatTensor<Self>, min: f32, max: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_clamp(nd, min, max);
        ndarray_to_npu(&result)
    }

    // ── Cat ─────────────────────────────────────────────────────────────

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        let nd_tensors: Vec<_> = tensors.iter().map(npu_to_ndarray).collect();
        let result = <Nd as FloatTensorOps<Nd>>::float_cat(nd_tensors, dim);
        ndarray_to_npu(&result)
    }

    // ── Sign ────────────────────────────────────────────────────────────

    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sign(nd);
        ndarray_to_npu(&result)
    }

    // ── Cast ────────────────────────────────────────────────────────────

    fn float_cast(tensor: FloatTensor<Self>, _dtype: FloatDType) -> FloatTensor<Self> {
        // Only f32 supported; casting is a no-op.
        tensor
    }

    // ── Grid sample ─────────────────────────────────────────────────────

    fn float_grid_sample_2d(
        tensor: FloatTensor<Self>,
        grid: FloatTensor<Self>,
        options: GridSampleOptions,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_grid = npu_to_ndarray(&grid);
        let result = <Nd as FloatTensorOps<Nd>>::float_grid_sample_2d(nd_tensor, nd_grid, options);
        ndarray_to_npu(&result)
    }

    // ── Unfold ──────────────────────────────────────────────────────────

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_unfold(nd, dim, size, step);
        ndarray_to_npu(&result)
    }
}

// ===========================================================================
// IntTensorOps — intel/qualcomm: int_into_float and int_powf bridge NpuFloatTensor
// ===========================================================================
#[cfg(any(feature = "intel", feature = "qualcomm"))]
impl IntTensorOps<Self> for NpuBurnBackend {
    fn int_from_data(data: TensorData, _device: &NpuBurnDevice) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_from_data(data, &nd_dev())
    }
    async fn int_into_data(tensor: IntTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as IntTensorOps<Nd>>::int_into_data(tensor).await
    }
    fn int_device(_tensor: &IntTensor<Self>) -> NpuBurnDevice { NpuBurnDevice::Default }
    fn int_to_device(tensor: IntTensor<Self>, _device: &NpuBurnDevice) -> IntTensor<Self> { tensor }
    fn int_empty(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_empty(shape, &nd_dev(), dtype)
    }
    fn int_zeros(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_zeros(shape, &nd_dev(), dtype)
    }
    fn int_ones(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_ones(shape, &nd_dev(), dtype)
    }
    fn int_full(shape: Shape, fill_value: i64, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_full(shape, fill_value, &nd_dev(), dtype)
    }
    fn int_random(shape: Shape, distribution: Distribution, _device: &NpuBurnDevice) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_random(shape, distribution, &nd_dev())
    }
    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_reshape(tensor, shape)
    }
    fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_slice(tensor, slices)
    }
    fn int_slice_assign(tensor: IntTensor<Self>, slices: &[Slice], value: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_slice_assign(tensor, slices, value)
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        // Convert NdArrayTensor<i64> -> NdArrayTensor<f32> -> NpuFloatTensor
        let nd_float = <Nd as IntTensorOps<Nd>>::int_into_float(tensor);
        ndarray_to_npu(&nd_float)
    }

    fn int_mask_where(tensor: IntTensor<Self>, mask: BoolTensor<Self>, source: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mask_where(tensor, mask, source)
    }
    fn int_mask_fill(tensor: IntTensor<Self>, mask: BoolTensor<Self>, value: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mask_fill(tensor, mask, value)
    }
    fn int_gather(dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_gather(dim, tensor, indices)
    }
    fn int_scatter_add(dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>, value: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_scatter_add(dim, tensor, indices, value)
    }
    fn int_select(tensor: IntTensor<Self>, dim: usize, indices: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_select(tensor, dim, indices)
    }
    fn int_select_add(tensor: IntTensor<Self>, dim: usize, indices: IntTensor<Self>, value: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_select_add(tensor, dim, indices, value)
    }
    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cat(tensors, dim)
    }
    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_equal(lhs, rhs)
    }
    fn int_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_equal_elem(lhs, rhs)
    }
    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater(lhs, rhs)
    }
    fn int_greater_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_elem(lhs, rhs)
    }
    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_equal(lhs, rhs)
    }
    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_equal_elem(lhs, rhs)
    }
    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower(lhs, rhs)
    }
    fn int_lower_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_elem(lhs, rhs)
    }
    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_equal(lhs, rhs)
    }
    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_equal_elem(lhs, rhs)
    }
    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_add(lhs, rhs)
    }
    fn int_add_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_add_scalar(lhs, rhs)
    }
    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sub(lhs, rhs)
    }
    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sub_scalar(lhs, rhs)
    }
    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mul(lhs, rhs)
    }
    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mul_scalar(lhs, rhs)
    }
    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_div(lhs, rhs)
    }
    fn int_div_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_div_scalar(lhs, rhs)
    }
    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_remainder(lhs, rhs)
    }
    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_remainder_scalar(lhs, rhs)
    }
    fn int_neg(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_neg(tensor)
    }
    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sum(tensor)
    }
    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sum_dim(tensor, dim)
    }
    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_prod(tensor)
    }
    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_prod_dim(tensor, dim)
    }
    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mean(tensor)
    }
    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mean_dim(tensor, dim)
    }
    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_max(tensor)
    }
    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_min(tensor)
    }
    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cumsum(tensor, dim)
    }
    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cumprod(tensor, dim)
    }
    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cummin(tensor, dim)
    }
    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cummax(tensor, dim)
    }
    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_matmul(lhs, rhs)
    }
    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_argmax(tensor, dim)
    }
    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_argmin(tensor, dim)
    }
    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_abs(tensor)
    }
    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_swap_dims(tensor, dim1, dim2)
    }
    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_permute(tensor, axes)
    }
    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_flip(tensor, axes)
    }
    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_expand(tensor, shape)
    }
    fn int_sign(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sign(tensor)
    }
    fn int_powi(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_powi(lhs, rhs)
    }
    fn int_powf(lhs: IntTensor<Self>, rhs: FloatTensor<Self>) -> IntTensor<Self> {
        // rhs is NpuFloatTensor, convert to NdArrayTensor for NdArray
        let nd_rhs = npu_to_ndarray(&rhs);
        <Nd as IntTensorOps<Nd>>::int_powf(lhs, nd_rhs)
    }
    fn int_powf_scalar_impl(lhs: IntTensor<Self>, rhs: f32) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_powf_scalar_impl(lhs, rhs)
    }
    fn int_clamp_min(tensor: IntTensor<Self>, min: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp_min(tensor, min)
    }
    fn int_clamp_max(tensor: IntTensor<Self>, max: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp_max(tensor, max)
    }
    fn int_clamp(tensor: IntTensor<Self>, min: i64, max: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp(tensor, min, max)
    }
    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_and(lhs, rhs)
    }
    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_and_scalar(lhs, rhs)
    }
    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_or(lhs, rhs)
    }
    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_or_scalar(lhs, rhs)
    }
    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_xor(lhs, rhs)
    }
    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_xor_scalar(lhs, rhs)
    }
    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_not(tensor)
    }
    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_left_shift(lhs, rhs)
    }
    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_left_shift_scalar(lhs, rhs)
    }
    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_right_shift(lhs, rhs)
    }
    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_right_shift_scalar(lhs, rhs)
    }
    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cast(tensor, dtype)
    }
    fn int_unfold(tensor: IntTensor<Self>, dim: usize, size: usize, step: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_unfold(tensor, dim, size, step)
    }
}

// ===========================================================================
// BoolTensorOps — intel/qualcomm: bool_into_float bridges to NpuFloatTensor
// ===========================================================================
#[cfg(any(feature = "intel", feature = "qualcomm"))]
impl BoolTensorOps<Self> for NpuBurnBackend {
    fn bool_from_data(data: TensorData, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_from_data(data, &nd_dev())
    }
    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as BoolTensorOps<Nd>>::bool_into_data(tensor).await
    }
    fn bool_device(_tensor: &BoolTensor<Self>) -> NpuBurnDevice { NpuBurnDevice::Default }
    fn bool_to_device(tensor: BoolTensor<Self>, _device: &NpuBurnDevice) -> BoolTensor<Self> { tensor }
    fn bool_empty(shape: Shape, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_empty(shape, &nd_dev())
    }
    fn bool_zeros(shape: Shape, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_zeros(shape, &nd_dev())
    }
    fn bool_ones(shape: Shape, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_ones(shape, &nd_dev())
    }
    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_into_int(tensor)
    }

    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        // Convert NdArrayTensor<bool> -> NdArrayTensor<f32> -> NpuFloatTensor
        let nd_float = <Nd as BoolTensorOps<Nd>>::bool_into_float(tensor);
        ndarray_to_npu(&nd_float)
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_reshape(tensor, shape)
    }
    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_slice(tensor, slices)
    }
    fn bool_slice_assign(tensor: BoolTensor<Self>, slices: &[Slice], value: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_slice_assign(tensor, slices, value)
    }
    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_equal(lhs, rhs)
    }
    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_not(tensor)
    }
    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_and(lhs, rhs)
    }
    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_or(lhs, rhs)
    }
    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_swap_dims(tensor, dim1, dim2)
    }
    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_permute(tensor, axes)
    }
    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_flip(tensor, axes)
    }
    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_expand(tensor, shape)
    }
    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_cat(tensors, dim)
    }
    fn bool_select(tensor: BoolTensor<Self>, dim: usize, indices: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_select(tensor, dim, indices)
    }
    fn bool_select_or(tensor: BoolTensor<Self>, dim: usize, indices: IntTensor<Self>, value: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_select_or(tensor, dim, indices, value)
    }
    fn bool_unfold(tensor: BoolTensor<Self>, dim: usize, size: usize, step: usize) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_unfold(tensor, dim, size, step)
    }
    fn bool_mask_where(tensor: BoolTensor<Self>, mask: BoolTensor<Self>, value: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_mask_where(tensor, mask, value)
    }
    fn bool_mask_fill(tensor: BoolTensor<Self>, mask: BoolTensor<Self>, value: bool) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_mask_fill(tensor, mask, value)
    }
    fn bool_gather(dim: usize, tensor: BoolTensor<Self>, indices: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_gather(dim, tensor, indices)
    }
    fn bool_scatter_or(dim: usize, tensor: BoolTensor<Self>, indices: IntTensor<Self>, value: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_scatter_or(dim, tensor, indices, value)
    }
    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: bool) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_equal_elem(lhs, rhs)
    }
    fn bool_any(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_any(tensor)
    }
    fn bool_all(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Nd as BoolTensorOps<Nd>>::bool_all(tensor)
    }
}

// ===========================================================================
// ModuleOps — intel/qualcomm: round-trip through NdArray
// ===========================================================================
#[cfg(any(feature = "intel", feature = "qualcomm"))]
impl ModuleOps<Self> for NpuBurnBackend {
    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv2d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_off = npu_to_ndarray(&offset);
        let nd_w = npu_to_ndarray(&weight);
        let nd_m = mask.as_ref().map(npu_to_ndarray);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::deform_conv2d(nd_x, nd_off, nd_w, nd_m, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn deform_conv2d_backward(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_off = npu_to_ndarray(&offset);
        let nd_w = npu_to_ndarray(&weight);
        let nd_m = mask.as_ref().map(npu_to_ndarray);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let nd_g = npu_to_ndarray(&output_grad);
        let r = <Nd as ModuleOps<Nd>>::deform_conv2d_backward(
            nd_x, nd_off, nd_w, nd_m, nd_b, nd_g, options,
        );
        DeformConv2dBackward::new(
            ndarray_to_npu(&r.x_grad),
            ndarray_to_npu(&r.offset_grad),
            ndarray_to_npu(&r.weight_grad),
            r.mask_grad.map(|g| ndarray_to_npu(&g)),
            r.bias_grad.map(|g| ndarray_to_npu(&g)),
        )
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv3d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv_transpose2d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv_transpose3d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::avg_pool2d(nd_x, kernel_size, stride, padding, count_include_pad, ceil_mode);
        ndarray_to_npu(&result)
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&grad);
        let result = <Nd as ModuleOps<Nd>>::avg_pool2d_backward(
            nd_x, nd_g, kernel_size, stride, padding, count_include_pad, ceil_mode,
        );
        ndarray_to_npu(&result)
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::adaptive_avg_pool2d(nd_x, output_size);
        ndarray_to_npu(&result)
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&grad);
        let result = <Nd as ModuleOps<Nd>>::adaptive_avg_pool2d_backward(nd_x, nd_g);
        ndarray_to_npu(&result)
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::max_pool2d(nd_x, kernel_size, stride, padding, dilation, ceil_mode);
        ndarray_to_npu(&result)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::max_pool2d_with_indices(
            nd_x, kernel_size, stride, padding, dilation, ceil_mode,
        );
        MaxPool2dWithIndices::new(ndarray_to_npu(&result.output), result.indices)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool2dBackward<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&output_grad);
        let result = <Nd as ModuleOps<Nd>>::max_pool2d_with_indices_backward(
            nd_x, kernel_size, stride, padding, dilation, ceil_mode, nd_g, indices,
        );
        MaxPool2dBackward::new(ndarray_to_npu(&result.x_grad))
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::interpolate(nd_x, output_size, options);
        ndarray_to_npu(&result)
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&grad);
        let result = <Nd as ModuleOps<Nd>>::interpolate_backward(nd_x, nd_g, output_size, options);
        ndarray_to_npu(&result)
    }
}

// ===========================================================================
// QTensorOps — intel/qualcomm: quantize/dequantize bridge NpuFloatTensor
// ===========================================================================
#[cfg(any(feature = "intel", feature = "qualcomm"))]
impl QTensorOps<Self> for NpuBurnBackend {
    fn q_from_data(data: TensorData, _device: &NpuBurnDevice) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_from_data(data, &nd_dev())
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_scales = npu_to_ndarray(&qparams.scales);
        let nd_qparams = QuantizationParametersPrimitive::<Nd> {
            scales: nd_scales,
        };
        <Nd as QTensorOps<Nd>>::quantize(nd_tensor, scheme, nd_qparams)
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        let nd_result = <Nd as QTensorOps<Nd>>::dequantize(tensor);
        ndarray_to_npu(&nd_result)
    }

    fn q_device(_tensor: &QuantizedTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, _device: &NpuBurnDevice) -> QuantizedTensor<Self> {
        tensor
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as QTensorOps<Nd>>::q_into_data(tensor).await
    }

    fn q_swap_dims(tensor: QuantizedTensor<Self>, dim1: usize, dim2: usize) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_permute(tensor, axes)
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_flip(tensor, axes)
    }

    fn q_gather(dim: usize, tensor: QuantizedTensor<Self>, indices: IntTensor<Self>) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_gather(dim, tensor, indices)
    }

    fn q_select(tensor: QuantizedTensor<Self>, dim: usize, indices: IntTensor<Self>) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_select(tensor, dim, indices)
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_slice(tensor, slices)
    }

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_argmax(tensor, dim)
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_argmin(tensor, dim)
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_expand(tensor, shape)
    }
}
