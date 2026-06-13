# Ternary Inference Sim — Simulated Ternary Neural Network Inference on GPU

**Ternary Inference Sim** simulates ternary neural network inference: weights are packed 16 per u32, matrix multiplication uses Z₃ arithmetic (conditional add/subtract/skip), and batch inference includes conservation verification. It provides the exact computational kernel that would run on ternary GPU hardware.

## Why It Matters

Building ternary GPU kernels requires understanding the exact arithmetic at the bit level. This simulator provides that understanding without requiring actual ternary hardware. Every operation — packing, unpacking, Z₃ multiply, Z₃ add, matvec, matmul, activation — is implemented in pure Rust, serving as both executable specification and reference implementation. The conservation verification (checking that γ + η = C after each layer) ensures correctness: if a bug causes the conservation law to break, the simulator catches it immediately.

## How It Works

### Trit Packing

16 trits pack into a single u32, 2 bits per trit:
- `-1` → `0b11`
- `0` → `0b00`  
- `+1` → `0b01`

`TritPack::new(&[i8])` packs; `get(i)` extracts; `unpack()` returns all 16. O(16) = O(1) per pack.

### Z₃ Arithmetic

**Ternary multiply** (tmul):
```
(-1)(-1) = +1, (-1)(+1) = -1, (+1)(-1) = -1, (+1)(+1) = +1
anything × 0 = 0
```

**Ternary add** (tadd):
```
(-1)+(-1) = +1 (wraps), (-1)+0 = -1, (-1)+(+1) = 0
0+0 = 0, 0+(+1) = +1
(+1)+(+1) = -1 (wraps)
```

Both are O(1) table lookups.

### Matrix-Vector Product

`ternary_matvec(weight, vector, rows)`: for each row, compute the dot product using Z₃ arithmetic:

```
output[r] = Z₃Σ weight[r][c] × vector[c]  for c in 0..16
```

O(rows × 16) = O(16·R) per operation.

### Matrix-Matrix Multiply

`ternary_matmul(a, b, rows, cols)`: batch dot products. O(R × C × 16).

### Ternary Sign Activation

```
sign(x) = +1 if x > 0, -1 if x < 0, 0 if x == 0
```

Standard sign function on the ternary domain.

### Ternary Layer

A `TernaryLayer` combines a weight matrix with sign activation: `output = sign(matvec(weights, input))`.

## Quick Start

```rust
use ternary_inference_sim::{TritPack, ternary_matvec, ternary_sign};

// Pack weights and input
let weights = vec![TritPack::new(&[1, -1, 0, 1, 1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1])];
let input = TritPack::new(&[1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, -1, 0, 1, -1, 1]);

// Matrix-vector product
let output = ternary_matvec(&weights, &input, 1);
let activated: Vec<i8> = output.iter().map(|&v| ternary_sign(v)).collect();
```

```bash
cargo add ternary-inference-sim
```

## API

| Type / Function | Description |
|---|---|
| `TritPack(u32)` | 16 packed trits: `new()`, `get(i)`, `unpack()` |
| `ternary_matvec(w, v, rows)` | Weight × vector in Z₃ |
| `ternary_matmul(a, b, r, c)` | Batch matrix multiply in Z₃ |
| `ternary_sign(i8)` | Activation function |

## Architecture Notes

This is the reference implementation for all ternary computation in **SuperInstance**. Fleet GPU kernels implement these exact operations in CUDA/PTX. The γ + η = C conservation is verified per layer: non-zero outputs contribute γ, zero outputs contribute η. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Li, Feng et al. "Ternary Weight Networks," *arXiv:1605.04711*, 2016.
- Rastegari, Mohammad et al. "XNOR-Net," *ECCV*, 2016 — binary/ternary network inference.
- Zhu, Chenzhuo et al. "Trained Ternary Quantization," *ICLR*, 2017.

## License

MIT
