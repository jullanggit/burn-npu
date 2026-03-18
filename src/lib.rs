//! # burn-npu
//!
//! NPU backend for [Burn](https://burn.dev). Drop-in replacement that runs on
//! Apple Neural Engine, Intel NPU, or Qualcomm Hexagon.
//!
//! ```rust
//! use burn::tensor::Tensor;
//! use burn_npu::{NpuBurnBackend, NpuBurnDevice};
//!
//! type B = NpuBurnBackend;
//! let device = NpuBurnDevice::Default;
//! let a = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
//! let b = Tensor::<B, 2>::from_floats([[5.0, 6.0], [7.0, 8.0]], &device);
//! let c = a.matmul(b);
//! ```

pub mod backends;
pub mod burn_backend;
pub mod detect;
pub mod info;

pub use burn_backend::{NpuBurnBackend, NpuBurnDevice};
pub use info::{NpuInfo, NpuVendor, Precision};
