//! GPT-2 inference benchmark: NpuBurnBackend (NPU) vs Wgpu (Metal GPU) vs NdArray (CPU)
//!
//!   cargo run --release --example bench --features apple

use burn::nn;
use burn::prelude::*;
use burn::tensor::activation;
use std::time::Instant;

// ── GPT-2 architecture in Burn ──────────────────────────────────────

const VOCAB: usize = 50257;
const HIDDEN: usize = 768;
const HEADS: usize = 12;
const LAYERS: usize = 12;
const FFN: usize = 3072;
const MAX_SEQ: usize = 1024;

#[derive(Module, Debug)]
struct Gpt2Attention<B: Backend> {
    c_attn: nn::Linear<B>,
    c_proj: nn::Linear<B>,
}

#[derive(Module, Debug)]
struct Gpt2Mlp<B: Backend> {
    c_fc: nn::Linear<B>,
    c_proj: nn::Linear<B>,
}

#[derive(Module, Debug)]
struct Gpt2Block<B: Backend> {
    ln_1: nn::LayerNorm<B>,
    attn: Gpt2Attention<B>,
    ln_2: nn::LayerNorm<B>,
    mlp: Gpt2Mlp<B>,
}

#[derive(Module, Debug)]
struct Gpt2<B: Backend> {
    wte: nn::Embedding<B>,
    wpe: nn::Embedding<B>,
    blocks: Vec<Gpt2Block<B>>,
    ln_f: nn::LayerNorm<B>,
}

impl<B: Backend> Gpt2<B> {
    fn init(device: &B::Device) -> Self {
        let wte = nn::EmbeddingConfig::new(VOCAB, HIDDEN).init(device);
        let wpe = nn::EmbeddingConfig::new(MAX_SEQ, HIDDEN).init(device);
        let ln_f = nn::LayerNormConfig::new(HIDDEN).init(device);
        let blocks = (0..LAYERS)
            .map(|_| Gpt2Block {
                ln_1: nn::LayerNormConfig::new(HIDDEN).init(device),
                attn: Gpt2Attention {
                    c_attn: nn::LinearConfig::new(HIDDEN, 3 * HIDDEN).init(device),
                    c_proj: nn::LinearConfig::new(HIDDEN, HIDDEN).init(device),
                },
                ln_2: nn::LayerNormConfig::new(HIDDEN).init(device),
                mlp: Gpt2Mlp {
                    c_fc: nn::LinearConfig::new(HIDDEN, FFN).init(device),
                    c_proj: nn::LinearConfig::new(FFN, HIDDEN).init(device),
                },
            })
            .collect();
        Self { wte, wpe, blocks, ln_f }
    }

    /// Forward pass. Returns logits [batch, seq, vocab] (last token only for efficiency).
    fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let [_batch, seq_len] = tokens.dims();
        let device = tokens.device();

        // Embeddings
        let positions = Tensor::arange(0..seq_len as i64, &device).unsqueeze::<2>();
        let mut x = self.wte.forward(tokens) + self.wpe.forward(positions);

        // Transformer blocks
        for block in &self.blocks {
            x = self.block_forward(block, x);
        }

        // Final LN — return hidden states (skip LM head for benchmark simplicity)
        let x = self.ln_f.forward(x); // [batch, seq, hidden]
        // Take last token: [batch, hidden]
        x.narrow(1, seq_len - 1, 1).reshape([_batch, HIDDEN])
    }

    fn block_forward(&self, block: &Gpt2Block<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, _hidden] = x.dims();
        let device = x.device();

        // Attention
        let residual = x.clone();
        let h = block.ln_1.forward(x);
        let qkv = block.attn.c_attn.forward(h); // [batch, seq, 3*hidden]
        let q = qkv.clone().narrow(2, 0, HIDDEN);
        let k = qkv.clone().narrow(2, HIDDEN, HIDDEN);
        let v = qkv.narrow(2, 2 * HIDDEN, HIDDEN);

        // Multi-head: [batch, seq, hidden] -> [batch, heads, seq, head_dim]
        let head_dim = HIDDEN / HEADS;
        let q = q.reshape([batch, seq, HEADS, head_dim]).swap_dims(1, 2);
        let k = k.reshape([batch, seq, HEADS, head_dim]).swap_dims(1, 2);
        let v = v.reshape([batch, seq, HEADS, head_dim]).swap_dims(1, 2);

        // Attention scores + causal mask
        let scale = (head_dim as f32).sqrt();
        let scores = q.matmul(k.transpose()) / scale;

        // Causal mask: -inf for future positions [1, 1, seq, seq]
        let mut mask_data = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in i + 1..seq {
                mask_data[i * seq + j] = -1e9;
            }
        }
        let mask = Tensor::<B, 4>::from_data(
            burn::tensor::TensorData::new(mask_data, [1, 1, seq, seq]),
            &device,
        );
        let scores = scores + mask;

        let probs = activation::softmax(scores, 3);
        let attn_out = probs.matmul(v); // [batch, heads, seq, head_dim]
        let attn_out = attn_out.swap_dims(1, 2).reshape([batch, seq, HIDDEN]);

        let x = block.attn.c_proj.forward(attn_out) + residual;

        // FFN
        let residual = x.clone();
        let h = block.ln_2.forward(x);
        let h = block.mlp.c_fc.forward(h);
        let h = activation::gelu(h);
        let h = block.mlp.c_proj.forward(h);
        h + residual
    }
}

// ── Benchmark runner ────────────────────────────────────────────────

fn run_bench<B: Backend>(name: &str, iters: usize) {
    let device = B::Device::default();
    let model = Gpt2::<B>::init(&device);

    // Dummy input: batch=1, seq=32
    let tokens = Tensor::<B, 2, Int>::zeros([1, 32], &device);

    // Warmup
    let _ = model.forward(tokens.clone());
    let _ = B::sync(&device);

    let start = Instant::now();
    for _ in 0..iters {
        let _ = model.forward(tokens.clone());
        let _ = B::sync(&device);
    }
    let elapsed = start.elapsed();
    let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
    let tok_s = 1000.0 / ms;

    println!("{:<30} {:>8.1} ms/fwd   {:>6.1} tok/s", name, ms, tok_s);
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    println!("burn-npu benchmark — GPT-2 forward pass (random weights, seq=32)\n");
    println!("{:<30} {:>12}   {:>9}", "Backend", "Latency", "Throughput");
    println!("{}", "─".repeat(58));

    // NdArray (CPU)
    run_bench::<burn_ndarray::NdArray>("burn-ndarray (CPU)", 10);

    // Wgpu (Metal GPU)
    run_bench::<burn_wgpu::Wgpu>("burn-wgpu (Metal GPU)", 10);

    // NpuBurnBackend (NPU via MLTensor)
    #[cfg(feature = "apple")]
    run_bench::<burn_npu::NpuBurnBackend>("burn-npu (Apple NPU)", 10);

    println!("{}", "─".repeat(58));

    // candle
    run_candle("candle (CPU)", &candle_core::Device::Cpu, 10);
    if let Ok(dev) = candle_core::Device::new_metal(0) {
        run_candle("candle (Metal GPU)", &dev, 10);
    }

    // llama.cpp
    println!();
    run_llamacpp();
}

// ── Candle GPT-2 (same architecture, candle tensors) ────────────────

fn run_candle(name: &str, device: &candle_core::Device, iters: usize) {
    use candle_core::{DType, Tensor};

    // Build GPT-2 with random weights in candle
    let h = HIDDEN;
    let wte = Tensor::randn(0f32, 1.0, &[VOCAB, h], device).unwrap();
    let wpe = Tensor::randn(0f32, 1.0, &[MAX_SEQ, h], device).unwrap();

    struct CLayer { ln1w: Tensor, ln1b: Tensor, aw: Tensor, ab: Tensor, pw: Tensor, pb: Tensor,
                    ln2w: Tensor, ln2b: Tensor, fw: Tensor, fb: Tensor, mw: Tensor, mb: Tensor }
    let layers: Vec<CLayer> = (0..LAYERS).map(|_| CLayer {
        ln1w: Tensor::ones(&[h], DType::F32, device).unwrap(),
        ln1b: Tensor::zeros(&[h], DType::F32, device).unwrap(),
        aw: Tensor::randn(0f32, 0.02, &[h, 3*h], device).unwrap(),
        ab: Tensor::zeros(&[3*h], DType::F32, device).unwrap(),
        pw: Tensor::randn(0f32, 0.02, &[h, h], device).unwrap(),
        pb: Tensor::zeros(&[h], DType::F32, device).unwrap(),
        ln2w: Tensor::ones(&[h], DType::F32, device).unwrap(),
        ln2b: Tensor::zeros(&[h], DType::F32, device).unwrap(),
        fw: Tensor::randn(0f32, 0.02, &[h, FFN], device).unwrap(),
        fb: Tensor::zeros(&[FFN], DType::F32, device).unwrap(),
        mw: Tensor::randn(0f32, 0.02, &[FFN, h], device).unwrap(),
        mb: Tensor::zeros(&[h], DType::F32, device).unwrap(),
    }).collect();
    let lnfw = Tensor::ones(&[h], DType::F32, device).unwrap();
    let lnfb = Tensor::zeros(&[h], DType::F32, device).unwrap();

    let candle_forward = |tokens: &Tensor| -> Tensor {
        let (batch, seq) = (tokens.dim(0).unwrap(), tokens.dim(1).unwrap());
        let positions = Tensor::arange(0u32, seq as u32, device).unwrap();
        let tok_emb = wte.index_select(tokens.flatten_all().unwrap().to_dtype(DType::U32).as_ref().unwrap(), 0)
            .unwrap().reshape(&[batch, seq, h]).unwrap();
        let pos_emb = wpe.index_select(&positions, 0).unwrap().unsqueeze(0).unwrap();
        let mut x = (tok_emb + pos_emb).unwrap();

        for l in &layers {
            let residual = x.clone();
            // LayerNorm
            let mean = x.mean_keepdim(2).unwrap();
            let diff = x.broadcast_sub(&mean).unwrap();
            let var = diff.sqr().unwrap().mean_keepdim(2).unwrap();
            let std = (var + 1e-5).unwrap().sqrt().unwrap();
            let n = diff.broadcast_div(&std).unwrap().broadcast_mul(&l.ln1w).unwrap().broadcast_add(&l.ln1b).unwrap();

            // QKV + attention
            let n2d = n.reshape(&[batch*seq, h]).unwrap();
            let qkv = n2d.matmul(&l.aw).unwrap().reshape(&[batch, seq, 3*h]).unwrap().broadcast_add(&l.ab).unwrap();
            let q = qkv.narrow(2, 0, h).unwrap().reshape(&[batch, seq, HEADS, h/HEADS]).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
            let k = qkv.narrow(2, h, h).unwrap().reshape(&[batch, seq, HEADS, h/HEADS]).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
            let v = qkv.narrow(2, 2*h, h).unwrap().reshape(&[batch, seq, HEADS, h/HEADS]).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
            let scale = ((h/HEADS) as f64).sqrt();
            let scores = (q.matmul(&k.transpose(2, 3).unwrap().contiguous().unwrap()).unwrap() / scale).unwrap();
            let mut mask_data = vec![0f32; seq*seq];
            for i in 0..seq { for j in i+1..seq { mask_data[i*seq+j] = -1e9; } }
            let mask = Tensor::from_vec(mask_data, &[1, 1, seq, seq], device).unwrap().expand(&[batch, HEADS, seq, seq]).unwrap();
            let scores = (scores + mask).unwrap();
            // Manual softmax (candle Metal may not have kernel)
            let max = scores.max_keepdim(3).unwrap();
            let exp = scores.broadcast_sub(&max).unwrap().exp().unwrap();
            let sum = exp.sum_keepdim(3).unwrap();
            let probs = exp.broadcast_div(&sum).unwrap();
            let attn = probs.matmul(&v).unwrap().transpose(1, 2).unwrap().contiguous().unwrap().reshape(&[batch, seq, h]).unwrap();
            x = (attn.reshape(&[batch*seq, h]).unwrap().matmul(&l.pw).unwrap().reshape(&[batch, seq, h]).unwrap().broadcast_add(&l.pb).unwrap() + residual).unwrap();

            // FFN
            let residual = x.clone();
            let mean = x.mean_keepdim(2).unwrap();
            let diff = x.broadcast_sub(&mean).unwrap();
            let var = diff.sqr().unwrap().mean_keepdim(2).unwrap();
            let std = (var + 1e-5).unwrap().sqrt().unwrap();
            let n = diff.broadcast_div(&std).unwrap().broadcast_mul(&l.ln2w).unwrap().broadcast_add(&l.ln2b).unwrap();
            let h_ff = n.reshape(&[batch*seq, h]).unwrap().matmul(&l.fw).unwrap().reshape(&[batch, seq, FFN]).unwrap().broadcast_add(&l.fb).unwrap();
            // GELU approx
            let c = (2.0f64 / std::f64::consts::PI).sqrt();
            let x3 = h_ff.powf(3.0).unwrap();
            let inner = ((h_ff.clone() + (x3 * 0.044715).unwrap()).unwrap() * c).unwrap();
            let gelu = (h_ff * (inner.tanh().unwrap() + 1.0).unwrap()).unwrap() * 0.5;
            x = (gelu.unwrap().reshape(&[batch*seq, FFN]).unwrap().matmul(&l.mw).unwrap().reshape(&[batch, seq, h]).unwrap().broadcast_add(&l.mb).unwrap() + residual).unwrap();
        }

        // Final LN
        let mean = x.mean_keepdim(2).unwrap();
        let diff = x.broadcast_sub(&mean).unwrap();
        let var = diff.sqr().unwrap().mean_keepdim(2).unwrap();
        let std = (var + 1e-5).unwrap().sqrt().unwrap();
        diff.broadcast_div(&std).unwrap().broadcast_mul(&lnfw).unwrap().broadcast_add(&lnfb).unwrap()
            .narrow(1, seq - 1, 1).unwrap().squeeze(1).unwrap()
    };

    // Warmup
    let tokens = Tensor::zeros(&[1, 32], DType::U32, device).unwrap();
    let _ = candle_forward(&tokens);

    let start = Instant::now();
    for _ in 0..iters {
        let _ = candle_forward(&tokens);
    }
    let elapsed = start.elapsed();
    let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
    let tok_s = 1000.0 / ms;
    println!("{:<30} {:>8.1} ms/fwd   {:>6.1} tok/s", name, ms, tok_s);
}

fn run_llamacpp() {
    use std::process::Command;

    // Find llama-completion
    let cli = ["llama-completion", "llama-cli"].iter()
        .find(|name| Command::new("which").arg(name).output().map(|o| o.status.success()).unwrap_or(false));
    let cli = match cli {
        Some(c) => *c,
        None => { println!("llama.cpp: not installed (brew install llama.cpp)"); return; }
    };

    let gguf = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("burn-npu/models/gpt2.Q4_0.gguf");

    if !gguf.exists() {
        println!("llama.cpp: downloading GPT-2 Q4_0 GGUF...");
        let url = "https://huggingface.co/QuantFactory/gpt2-GGUF/resolve/main/gpt2.Q4_0.gguf";
        let mut resp = reqwest::blocking::get(url).unwrap();
        std::fs::create_dir_all(gguf.parent().unwrap()).ok();
        let mut f = std::fs::File::create(&gguf).unwrap();
        std::io::copy(&mut resp, &mut f).unwrap();
    }

    // Run 50 tokens
    let output = Command::new(cli)
        .args(["-m", &gguf.to_string_lossy(), "-p", "The key insight of distributed machine learning is", "-n", "50", "-s", "42"])
        .output();

    match output {
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            // Parse eval tok/s
            for line in stderr.lines() {
                if line.contains("eval") && line.contains("tokens per second") && !line.contains("prompt") {
                    if let Some(before) = line.split("tokens per second").next() {
                        if let Some(tps) = before.split_whitespace().last() {
                            if let Ok(tps) = tps.parse::<f64>() {
                                let ms = 1000.0 / tps;
                                println!("\nExternal comparison:");
                                println!("{:<30} {:>8.1} ms/fwd   {:>6.1} tok/s  (Q4_0 quantized)", "llama.cpp (Metal)", ms, tps);
                                return;
                            }
                        }
                    }
                }
            }
            println!("llama.cpp: ran but couldn't parse timing");
        }
        Err(e) => println!("llama.cpp: failed to run: {e}"),
    }
}
