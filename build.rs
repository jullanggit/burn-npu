fn main() {
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=framework=Accelerate");

    #[cfg(feature = "apple")]
    build_npu_sys();
}

#[cfg(feature = "apple")]
fn build_npu_sys() {
    use std::process::Command;

    let npu_dir = std::path::Path::new("npu-sys");
    let src = npu_dir.join("Sources/npu_sys.swift");
    let lib = npu_dir.join("libnpu_sys.a");

    println!("cargo:rerun-if-changed={}", src.display());

    if !src.exists() {
        println!("cargo:warning=npu-sys/Sources/npu_sys.swift not found");
        return;
    }

    // Compile Swift to static library
    let status = Command::new("swiftc")
        .args([
            "-parse-as-library", "-emit-library", "-static", "-O",
            "-module-name", "npu_sys",
            &src.to_string_lossy(),
            "-o", &lib.to_string_lossy(),
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            // Link the static library
            println!("cargo:rustc-link-search=native={}", npu_dir.canonicalize().unwrap().display());
            println!("cargo:rustc-link-lib=static=npu_sys");

            // Link Swift runtime (system) and CoreML
            println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
            println!("cargo:rustc-link-lib=framework=CoreML");
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
        _ => {
            println!("cargo:warning=Failed to compile npu-sys Swift library");
        }
    }
}
