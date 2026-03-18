use crate::info::{NpuInfo, NpuVendor, Precision};

/// Probe the current hardware and return [`NpuInfo`] if an NPU is present.
///
/// Detection is platform-specific:
/// - **macOS**: checks for Apple Silicon via `sysctl hw.optional.arm64`
/// - **Linux**: checks `/proc/cpuinfo` for Intel + `/dev/accel*` for NPU device
/// - **Windows**: checks for OpenVINO DLL in standard install paths
///
/// Returns `None` if no NPU is detected or on unsupported platforms.
pub fn detect() -> Option<NpuInfo> {
    #[cfg(target_os = "macos")]
    {
        if let Some(info) = detect_apple() {
            return Some(info);
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Some(info) = detect_intel_linux() {
            return Some(info);
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Some(info) = detect_intel_windows() {
            return Some(info);
        }
    }

    None
}

// ── macOS: Apple Silicon Neural Engine ──────────────────────────────────

#[cfg(target_os = "macos")]
fn detect_apple() -> Option<NpuInfo> {
    use std::process::Command;

    let output = Command::new("sysctl")
        .args(["-n", "hw.optional.arm64"])
        .output()
        .ok()?;
    let arm64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u32>()
        .unwrap_or(0);
    if arm64 != 1 {
        return None;
    }

    let brand_output = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()?;
    let brand = String::from_utf8_lossy(&brand_output.stdout)
        .trim()
        .to_string();

    let (tops, desc) = estimate_apple_tops(&brand);

    Some(NpuInfo {
        vendor: NpuVendor::Apple,
        tops,
        max_precision: Precision::FP16,
        description: desc,
    })
}

#[cfg(target_os = "macos")]
fn estimate_apple_tops(brand: &str) -> (f32, String) {
    let lower = brand.to_lowercase();
    if lower.contains("m4") {
        (38.0, format!("Apple Neural Engine (M4) -- {brand}"))
    } else if lower.contains("m3") {
        (18.0, format!("Apple Neural Engine (M3) -- {brand}"))
    } else if lower.contains("m2") {
        (15.8, format!("Apple Neural Engine (M2) -- {brand}"))
    } else if lower.contains("m1") {
        (11.0, format!("Apple Neural Engine (M1) -- {brand}"))
    } else {
        (11.0, format!("Apple Neural Engine (unknown) -- {brand}"))
    }
}

// ── Linux: Intel OpenVINO NPU ──────────────────────────────────────────

#[cfg(target_os = "linux")]
fn detect_intel_linux() -> Option<NpuInfo> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    if !cpuinfo.contains("GenuineIntel") {
        return None;
    }

    let model_name = cpuinfo
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "Intel CPU".to_string());

    let has_npu_device = std::path::Path::new("/dev/accel").exists()
        || std::fs::read_dir("/dev")
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .any(|e| {
                        e.file_name()
                            .to_string_lossy()
                            .starts_with("accel")
                    })
            })
            .unwrap_or(false);

    if !has_npu_device {
        return None;
    }

    Some(NpuInfo {
        vendor: NpuVendor::Intel,
        tops: 11.0,
        max_precision: Precision::INT8,
        description: format!("Intel NPU -- {model_name}"),
    })
}

// ── Windows: Intel OpenVINO NPU ────────────────────────────────────────

#[cfg(target_os = "windows")]
fn detect_intel_windows() -> Option<NpuInfo> {
    let openvino_paths = [
        r"C:\Program Files (x86)\Intel\openvino\runtime\bin\intel64\Release\openvino.dll",
        r"C:\Program Files\Intel\openvino\runtime\bin\intel64\Release\openvino.dll",
    ];
    let has_openvino = openvino_paths.iter().any(|p| std::path::Path::new(p).exists());
    if !has_openvino {
        return None;
    }

    Some(NpuInfo {
        vendor: NpuVendor::Intel,
        tops: 11.0,
        max_precision: Precision::INT8,
        description: "Intel NPU (OpenVINO detected)".to_string(),
    })
}
