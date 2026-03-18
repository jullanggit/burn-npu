/// Precision levels supported by NPU hardware.
///
/// NPUs typically operate at reduced precision for higher throughput.
/// The [`NpuInfo::max_precision`] field indicates the highest precision
/// the hardware supports for ML operations.
#[derive(Debug, Clone, PartialEq)]
pub enum Precision {
    /// 32-bit floating point.
    FP32,
    /// 16-bit floating point (most common for Apple ANE).
    FP16,
    /// 8-bit integer quantization (common for Intel NPU).
    INT8,
    /// 4-bit integer quantization.
    INT4,
}

/// NPU hardware vendor.
///
/// Used by [`NpuInfo`] to identify the detected hardware and select
/// the appropriate backend.
#[derive(Debug, Clone, PartialEq)]
pub enum NpuVendor {
    /// Intel Core Ultra NPU, via OpenVINO.
    Intel,
    /// Apple Neural Engine (M1/M2/M3/M4), via Core ML / Accelerate.
    Apple,
    /// Qualcomm Hexagon DSP (Snapdragon), via QNN SDK.
    Qualcomm,
    /// Unknown or unsupported vendor.
    Unknown,
}

/// Information about a detected NPU device.
///
/// Information about detected NPU hardware.
#[derive(Debug, Clone)]
pub struct NpuInfo {
    /// Vendor of the NPU hardware.
    pub vendor: NpuVendor,

    /// Peak AI performance in Tera Operations Per Second.
    ///
    /// Typical values: M1 = 11, M2 = 15.8, M3 = 18, M4 = 38,
    /// Intel Core Ultra = 11.
    pub tops: f32,

    /// Maximum precision supported for ML operations.
    pub max_precision: Precision,

    /// Human-readable description, e.g. `"Apple Neural Engine (M2) -- Apple M2 Pro"`.
    pub description: String,
}
