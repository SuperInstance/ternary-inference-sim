# ternary-inference-sim

Simulated ternary neural network inference. Ternary weight packing, batch inference, conservation verification.

## Why This Matters

# ternary-inference-sim
Simulates ternary neural network inference on GPU.
Ternary weights packed 16-per-u32, matmul via element-wise Z₃,
batch inference with conservation verification.

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub struct TritPack
pub fn new
pub fn get
pub fn unpack
pub fn ternary_matvec
pub fn ternary_matmul
pub fn ternary_sign
pub struct TernaryLayer
pub fn new
pub fn forward
pub fn forward_batch
pub struct TernaryNetwork
```

## Usage

```toml
[dependencies]
ternary-inference-sim = "0.1.0"
```

```rust
use ternary_inference_sim::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/ternary-inference-sim.git
cd ternary-inference-sim
cargo test    # 8 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 8 |
| Lines of Rust | 250 |
| Public API | 22 items |

## License

Apache-2.0
