# burn-npu

NPU backend for [Burn](https://burn.dev). Drop-in replacement for `burn-wgpu` or `burn-ndarray` that runs on Apple Neural Engine, Intel NPU, or Qualcomm Hexagon.

```rust
use burn::tensor::Tensor;
use burn_npu::{NpuBurnBackend, NpuBurnDevice};

type B = NpuBurnBackend;  // swap this line — that's it

let device = NpuBurnDevice::Default;
let a = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
let b = Tensor::<B, 2>::from_floats([[5.0, 6.0], [7.0, 8.0]], &device);
let c = a.matmul(b);
```

Any Burn model works. No code changes needed.

## Benchmark

GPT-2 124M forward pass, seq=32, FP32, Apple M2 Pro.

| Backend | Latency | Throughput |
|---|---|---|
| **burn-npu (Apple NPU)** | **29 ms** | **34.8 tok/s** |
| burn-wgpu (Metal GPU) | 37 ms | 27.1 tok/s |
| burn-ndarray (CPU) | 107 ms | 9.4 tok/s |

```bash
cargo run --release --example bench --features apple
```

## Installation

```toml
[dependencies]
burn-npu = { version = "0.3", features = ["apple"] }
```

| Feature | Hardware | Status | Requires |
|---|---|---|---|
| `apple` | Apple Neural Engine (M1/M2/M3/M4) | tested, working | macOS 15+, Xcode |
| `intel` | Intel Core Ultra NPU | implemented, needs hardware validation | OpenVINO runtime |
| `qualcomm` | Qualcomm Hexagon (Snapdragon) | implemented, needs hardware + QNN SDK | QNN SDK |

Enable one feature at a time. Without any feature, falls back to burn-ndarray (CPU).

## How It Works

burn-npu implements Burn's `Backend` trait. Float tensors live on NPU hardware as native handles. No data copies between operations — only `into_data()` reads back to CPU.

| Platform | Tensor type | NPU dispatch |
|---|---|---|
| Apple | MLTensor handle | ANE / GPU / CPU via Core ML |
| Intel | OpenVINO tensor | NPU / GPU / CPU via OpenVINO |
| Qualcomm | QNN tensor (planned) | CPU fallback (QNN SDK integration ready) |

## Operations

All Burn `Backend` trait operations are implemented. On Apple, 37 float ops run natively on NPU (matmul, add, sub, mul, div, exp, log, sqrt, abs, neg, sin, cos, tanh, erf, floor, ceil, pow, clamp, softmax, sum, mean, max, min, argmax, argmin, reshape, transpose, narrow, cat, select, gather, slice). On Intel, matmul is NPU-accelerated via OpenVINO. Remaining ops and int/bool tensors delegate to burn-ndarray.

## Inference Only

NPUs do not support automatic differentiation. burn-npu accelerates the forward pass only. For training, use `burn-wgpu` or `burn-cuda`.

## Background

This project was motivated by [this discussion](https://github.com/tracel-ai/burn/discussions/4245) in the Burn repo, where NPU support was considered difficult because NPUs "are often not programmable chips" with no common API. On Apple Silicon, MLTensor (macOS 15+) solves this — it's a tensor API that Core ML routes to ANE/GPU/CPU automatically. On Intel, OpenVINO provides a similar abstraction. burn-npu wraps these platform APIs behind Burn's `Backend` trait.

## Contributing

burn-npu needs help with:
- **Intel hardware testing** — if you have a Core Ultra laptop, run `cargo test --features intel` and open an issue with results
- **Qualcomm hardware testing** — if you have a Snapdragon X Elite device or Rubik Pi 3, help integrate the QNN SDK
- **More NPU-accelerated ops** — move remaining float ops from NdArray delegation to native NPU execution
- **Thread safety hardening** — the Apple MLTensor handle table uses NSLock but needs stress testing

## License

MIT OR Apache-2.0
