# ternary-inference-sim

Simulated ternary neural network inference — packed weights, Z₃ matmul, and conservation verification.

## Why This Exists

Before deploying ternary weights to a real GPU, you need to verify the arithmetic is correct. This crate simulates the full ternary inference pipeline: pack weights 16-per-u32, run matrix-vector and matrix-matrix multiply using Z₃ addition, verify that ternary conservation laws hold (sum of outputs should be predictable from sum of inputs), and benchmark the theoretical throughput. It's a reference implementation and test oracle for GPU ternary kernels.

## Architecture

### Core Types

- **`TritPack(u32)`** — 16 ternary values packed into one u32 register.
- **`TernaryLayer`** — One neural network layer: packed weight matrix + dimensions.
- **`TernaryNetwork`** — Multi-layer inference pipeline.
- **`ConservationResult`** — Verifies input/output ternary sum conservation.
- **`InferenceBenchmark`** — Tracks throughput (inferences/sec, effective TFLOPS).

### Key Functions

- `ternary_matvec`: Matrix-vector multiply in Z₃ arithmetic.
- `ternary_matmul`: Matrix-matrix multiply.
- `ternary_sign`: Sign activation function.
- `verify_conservation`: Check that ternary mass is conserved across a layer.

## Usage

```rust
use ternary_inference_sim::{TritPack, TernaryLayer, TernaryNetwork, verify_conservation};

// Pack some weights
let weights: Vec<TritPack> = vec![
    TritPack::new(&[1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1]),
    TritPack::new(&[-1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1]),
];

// Create a layer
let layer = TernaryLayer::new("hidden_0", &weights, 16, 2);
let input = TritPack::new(&[1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]);
let output = layer.forward(&input);

// Verify conservation
let conservation = verify_conservation(&input, &output);
println!("Input sum: {}, Output sum: {}", conservation.input_sum, conservation.output_sum);

// Multi-layer network
let network = TernaryNetwork::new(vec![layer]);
let batch_result = network.batch_inference(&[input]);
```

## API Reference

| Method | Returns | Description |
|--------|---------|-------------|
| `TritPack::new(trits)` | `TritPack` | Pack 16 trits into u32 |
| `pack.get(i)` | `i8` | Read one trit |
| `ternary_matvec(w, v, rows)` | `Vec<i8>` | Matrix-vector multiply |
| `ternary_matmul(a, b, r, c)` | `Vec<TritPack>` | Matrix-matrix multiply |
| `TernaryLayer::forward(&self, input)` | `Vec<i8>` | Single inference |
| `TernaryLayer::forward_batch(&self, inputs)` | `Vec<Vec<i8>>` | Batch inference |
| `TernaryNetwork::inference(&self, input)` | `Vec<i8>` | Multi-layer forward pass |
| `verify_conservation(input, output)` | `ConservationResult` | Conservation check |

## The Deeper Idea

Ternary conservation is the analogue of **charge conservation in physics**. In a ternary system, the sum of all values (the "charge") should be traceable through every operation. If you put in a total charge of +5 and get out +12, something went wrong — either the arithmetic isn't in Z₃ or the packing is corrupt. Conservation verification is cheap (one sum per layer) and catches an entire class of packing and arithmetic bugs that unit tests with small values might miss.

## Related Crates

- **ternary-pack** — bit-packing trits into u32 registers
- **ternary-hotswap-inference** — live model swapping with ternary tensors
- **ternary-cortex** — hierarchical ternary processing layers
