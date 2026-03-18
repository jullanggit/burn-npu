/// Apple NPU backend via MLTensor (macOS 15+).
///
/// MLTensor dispatches tensor ops to ANE/GPU/CPU automatically.
/// No model compilation needed. No subprocess. Direct FFI.
#[cfg(target_os = "macos")]
mod inner {
    extern "C" {
        fn npu_create_tensor(shape: *const i32, dims: i32, data: *const f32, len: i32) -> i32;
        fn npu_free_tensor(id: i32);
        fn npu_matmul(a: i32, b: i32) -> i32;
        fn npu_softmax(x: i32, axis: i32) -> i32;
        fn npu_add(a: i32, b: i32) -> i32;
        fn npu_sub(a: i32, b: i32) -> i32;
        fn npu_mul(a: i32, b: i32) -> i32;
        fn npu_div(a: i32, b: i32) -> i32;
        fn npu_sqrt(x: i32) -> i32;
        fn npu_mul_scalar(a: i32, scalar: f32) -> i32;
        fn npu_mean(x: i32, axis: i32) -> i32;
        fn npu_tanh(x: i32) -> i32;
        fn npu_clamp_min(x: i32, min_val: f32) -> i32;
        fn npu_transpose(x: i32, dim0: i32, dim1: i32) -> i32;
        fn npu_reshape(x: i32, shape: *const i32, dims: i32) -> i32;
        fn npu_narrow(x: i32, dim: i32, start: i32, length: i32) -> i32;
        fn npu_index_select(x: i32, indices: *const i32, len: i32) -> i32;
        fn npu_get_shape(id: i32, out: *mut i32, max_dims: i32) -> i32;
        fn npu_get_data(id: i32, out: *mut f32, max_len: i32) -> i32;
        fn npu_scalar_tensor(value: f32) -> i32;
    }

    /// Handle to a tensor living on the NPU (MLTensor).
    /// Not Clone/Copy — drop frees the native tensor.
    pub struct NpuTensor(i32);

    impl NpuTensor {
        pub fn from_data(shape: &[usize], data: &[f32]) -> Self {
            let s: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
            let id = unsafe { npu_create_tensor(s.as_ptr(), s.len() as i32, data.as_ptr(), data.len() as i32) };
            Self(id)
        }

        pub fn matmul(&self, other: &NpuTensor) -> NpuTensor {
            NpuTensor(unsafe { npu_matmul(self.0, other.0) })
        }

        pub fn softmax(&self, axis: i32) -> NpuTensor {
            NpuTensor(unsafe { npu_softmax(self.0, axis) })
        }

        pub fn add(&self, other: &NpuTensor) -> NpuTensor {
            NpuTensor(unsafe { npu_add(self.0, other.0) })
        }

        pub fn sub(&self, other: &NpuTensor) -> NpuTensor {
            NpuTensor(unsafe { npu_sub(self.0, other.0) })
        }

        pub fn mul(&self, other: &NpuTensor) -> NpuTensor {
            NpuTensor(unsafe { npu_mul(self.0, other.0) })
        }

        pub fn div(&self, other: &NpuTensor) -> NpuTensor {
            NpuTensor(unsafe { npu_div(self.0, other.0) })
        }

        pub fn sqrt(&self) -> NpuTensor {
            NpuTensor(unsafe { npu_sqrt(self.0) })
        }

        pub fn scale(&self, s: f32) -> NpuTensor {
            NpuTensor(unsafe { npu_mul_scalar(self.0, s) })
        }

        pub fn mean(&self, axis: i32) -> NpuTensor {
            NpuTensor(unsafe { npu_mean(self.0, axis) })
        }

        pub fn relu(&self) -> NpuTensor {
            NpuTensor(unsafe { npu_clamp_min(self.0, 0.0) })
        }

        pub fn tanh(&self) -> NpuTensor {
            NpuTensor(unsafe { npu_tanh(self.0) })
        }

        /// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        pub fn gelu(&self) -> NpuTensor {
            let x3 = self.mul(self).mul(self);
            let inner = self.add(&x3.scale(0.044715)).scale(0.7978845608); // sqrt(2/pi)
            let tanh_val = inner.tanh();
            let one = NpuTensor::scalar(1.0);
            self.mul(&tanh_val.add(&one)).scale(0.5)
        }

        pub fn transpose(&self, d0: i32, d1: i32) -> NpuTensor {
            NpuTensor(unsafe { npu_transpose(self.0, d0, d1) })
        }

        pub fn reshape(&self, shape: &[usize]) -> NpuTensor {
            let s: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
            NpuTensor(unsafe { npu_reshape(self.0, s.as_ptr(), s.len() as i32) })
        }

        pub fn narrow(&self, dim: i32, start: usize, len: usize) -> NpuTensor {
            NpuTensor(unsafe { npu_narrow(self.0, dim, start as i32, len as i32) })
        }

        pub fn index_select(&self, indices: &[i32]) -> NpuTensor {
            NpuTensor(unsafe { npu_index_select(self.0, indices.as_ptr(), indices.len() as i32) })
        }

        pub fn to_vec(&self) -> Vec<f32> {
            let mut shape = [0i32; 8];
            let ndim = unsafe { npu_get_shape(self.0, shape.as_mut_ptr(), 8) };
            let total: i32 = shape[..ndim as usize].iter().product();
            let mut out = vec![0.0f32; total as usize];
            unsafe { npu_get_data(self.0, out.as_mut_ptr(), total) };
            out
        }

        pub fn scalar(v: f32) -> NpuTensor {
            NpuTensor(unsafe { npu_scalar_tensor(v) })
        }

        /// Layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta
        pub fn layernorm(&self, gamma: &NpuTensor, beta: &NpuTensor) -> NpuTensor {
            // mean(-1) drops the last dim. We need keepdim for broadcasting.
            // Get shape, compute keepdim shape, reshape.
            let mut shape_buf = [0i32; 8];
            let ndim = unsafe { npu_get_shape(self.0, shape_buf.as_mut_ptr(), 8) } as usize;
            let mut keepdim_shape: Vec<usize> = shape_buf[..ndim].iter().map(|&d| d as usize).collect();
            if let Some(last) = keepdim_shape.last_mut() { *last = 1; }

            let mean = self.mean(-1).reshape(&keepdim_shape);
            let diff = self.sub(&mean);
            let sq = diff.mul(&diff);
            let var_ = sq.mean(-1).reshape(&keepdim_shape);
            let eps = NpuTensor::scalar(1e-5);
            let std = var_.add(&eps).sqrt();
            let normed = diff.div(&std);
            normed.mul(gamma).add(beta)
        }
    }

    impl Drop for NpuTensor {
        fn drop(&mut self) {
            unsafe { npu_free_tensor(self.0) };
        }
    }


    pub struct AppleNpuBackend;

    #[derive(Debug)]
    pub enum AppleNpuError {
        OpFailed(String),
    }
    impl std::fmt::Display for AppleNpuError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self { Self::OpFailed(m) => write!(f, "Apple NPU: {m}") }
        }
    }
    impl std::error::Error for AppleNpuError {}

    impl AppleNpuBackend {
        pub fn new() -> Result<Self, AppleNpuError> { Ok(Self) }

        pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, AppleNpuError> {
            let ta = NpuTensor::from_data(&[m, k], a);
            let tb = NpuTensor::from_data(&[k, n], b);
            Ok(ta.matmul(&tb).to_vec())
        }

        pub fn relu(&self, x: &[f32], shape: &[usize]) -> Result<Vec<f32>, AppleNpuError> {
            let t = NpuTensor::from_data(shape, x);
            Ok(t.relu().to_vec())
        }

        pub fn softmax(&self, x: &[f32], shape: &[usize]) -> Result<Vec<f32>, AppleNpuError> {
            let t = NpuTensor::from_data(shape, x);
            Ok(t.softmax(-1).to_vec())
        }

        pub fn layernorm(&self, x: &[f32], shape: &[usize], gamma: &[f32], beta: &[f32]) -> Result<Vec<f32>, AppleNpuError> {
            let last = *shape.last().unwrap();
            let tx = NpuTensor::from_data(shape, x);
            let tg = NpuTensor::from_data(&[last], gamma);
            let tb = NpuTensor::from_data(&[last], beta);
            Ok(tx.layernorm(&tg, &tb).to_vec())
        }
    }
}

#[cfg(target_os = "macos")]
pub use inner::{AppleNpuBackend, AppleNpuError, NpuTensor};

#[cfg(not(target_os = "macos"))]
pub use stub::*;

#[cfg(not(target_os = "macos"))]
mod stub {
    use std::path::Path;
    #[derive(Debug)]
    pub enum AppleNpuError { NotSupported }
    impl std::fmt::Display for AppleNpuError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "Apple NPU: macOS only") }
    }
    impl std::error::Error for AppleNpuError {}
    pub struct AppleNpuBackend;
    impl AppleNpuBackend {
        pub fn new() -> Result<Self, AppleNpuError> { Err(AppleNpuError::NotSupported) }
        pub fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Result<Vec<f32>, AppleNpuError> { Err(AppleNpuError::NotSupported) }
        pub fn relu(&self, _: &[f32], _: &[usize]) -> Result<Vec<f32>, AppleNpuError> { Err(AppleNpuError::NotSupported) }
        pub fn softmax(&self, _: &[f32], _: &[usize]) -> Result<Vec<f32>, AppleNpuError> { Err(AppleNpuError::NotSupported) }
        pub fn layernorm(&self, _: &[f32], _: &[usize], _: &[f32], _: &[f32]) -> Result<Vec<f32>, AppleNpuError> { Err(AppleNpuError::NotSupported) }
    }
    #[derive(Clone, Copy)]
    pub struct NpuTensor(i32);
}
