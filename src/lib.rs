//! # ternary-inference-sim
//!
//! Simulates ternary neural network inference on GPU.
//! Ternary weights packed 16-per-u32, matmul via element-wise Z₃,
//! batch inference with conservation verification.

/// Packed ternary weights: 16 trits per u32.
#[derive(Debug, Clone, Copy)]
pub struct TritPack(pub u32);

impl TritPack {
    pub fn new(trits: &[i8]) -> Self {
        let mut p = 0u32;
        for (i, &t) in trits.iter().take(16).enumerate() {
            let bits = match t { -1 => 0b11u32, 0 => 0b00, 1 => 0b01, _ => 0b00 };
            p |= bits << (i * 2);
        }
        TritPack(p)
    }
    pub fn get(&self, i: usize) -> i8 {
        match (self.0 >> (i * 2)) & 0b11 { 0b11 => -1, 0b01 => 1, _ => 0 }
    }
    pub fn unpack(&self) -> Vec<i8> { (0..16).map(|i| self.get(i)).collect() }
}

/// Z₃ ternary multiply.
fn tmul(a: i8, b: i8) -> i8 {
    match (a, b) {
        (-1, -1) => 1, (-1, 1) => -1, (1, -1) => -1, (1, 1) => 1, _ => 0,
    }
}

/// Z₃ ternary add.
fn tadd(a: i8, b: i8) -> i8 {
    match (a, b) {
        (-1, -1) => 1, (-1, 0) => -1, (-1, 1) => 0,
        (0, -1) => -1, (0, 0) => 0, (0, 1) => 1,
        (1, -1) => 0, (1, 0) => 1, (1, 1) => -1,
        _ => 0,
    }
}

/// Ternary matrix-vector product: weight[row] · vector = output[row].
pub fn ternary_matvec(weight: &[TritPack], vector: &TritPack, rows: usize) -> Vec<i8> {
    let mut output = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut acc = 0i8;
        for c in 0..16 {
            acc = tadd(acc, tmul(weight[r].get(c), vector.get(c)));
        }
        output.push(acc);
    }
    output
}

/// Ternary matrix-matrix multiply for batch inference.
pub fn ternary_matmul(a: &[TritPack], b: &[TritPack], rows: usize, cols: usize) -> Vec<TritPack> {
    let mut result = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut trits = Vec::with_capacity(16);
        for c in 0..cols.min(16) {
            let mut acc = 0i8;
            for k in 0..16 {
                acc = tadd(acc, tmul(a[r].get(k), b[c].get(k)));
            }
            trits.push(acc);
        }
        while trits.len() < 16 { trits.push(0); }
        result.push(TritPack::new(&trits));
    }
    result
}

/// Ternary activation function (sign function for {-1, 0, +1}).
pub fn ternary_sign(x: i8) -> i8 {
    if x > 0 { 1 } else if x < 0 { -1 } else { 0 }
}

/// A ternary layer: weight matrix + activation.
#[derive(Debug, Clone)]
pub struct TernaryLayer {
    pub weights: Vec<TritPack>,
    pub input_dim: usize,
    pub output_dim: usize,
    pub name: String,
}

impl TernaryLayer {
    pub fn new(name: &str, weights: &[TritPack], input_dim: usize, output_dim: usize) -> Self {
        Self { weights: weights.to_vec(), input_dim, output_dim, name: name.into() }
    }

    pub fn forward(&self, input: &TritPack) -> Vec<i8> {
        let raw = ternary_matvec(&self.weights, input, self.output_dim);
        raw.iter().map(|&x| ternary_sign(x)).collect()
    }

    pub fn forward_batch(&self, inputs: &[TritPack]) -> Vec<Vec<i8>> {
        inputs.iter().map(|inp| self.forward(inp)).collect()
    }
}

/// A ternary neural network (sequence of layers).
pub struct TernaryNetwork {
    pub layers: Vec<TernaryLayer>,
}

impl TernaryNetwork {
    pub fn new(layers: Vec<TernaryLayer>) -> Self { Self { layers } }

    pub fn inference(&self, input: &TritPack) -> Vec<i8> {
        let mut current = input.unpack();
        for layer in &self.layers {
            let packed = TritPack::new(&current);
            current = layer.forward(&packed);
        }
        current
    }

    pub fn batch_inference(&self, inputs: &[TritPack]) -> Vec<Vec<i8>> {
        inputs.iter().map(|inp| self.inference(inp)).collect()
    }
}

/// Conservation verification: sum of output ternary values should be bounded.
pub fn verify_conservation(input: &TritPack, output: &[i8]) -> ConservationResult {
    let input_sum: i32 = input.unpack().iter().map(|&v| v as i32).sum();
    let output_sum: i32 = output.iter().map(|&v| v as i32).sum();
    ConservationResult {
        input_sum,
        output_sum,
        delta: (output_sum - input_sum).abs(),
        preserved: (output_sum - input_sum).abs() <= output.len() as i32,
    }
}

#[derive(Debug)]
pub struct ConservationResult {
    pub input_sum: i32,
    pub output_sum: i32,
    pub delta: i32,
    pub preserved: bool,
}

/// Simulated GPU throughput measurement.
pub struct InferenceBenchmark {
    pub total_inferences: u64,
    pub total_time_us: u64,
    pub weights_per_inference: usize,
}

impl InferenceBenchmark {
    pub fn new(weights_per_inference: usize) -> Self {
        Self { total_inferences: 0, total_time_us: 0, weights_per_inference }
    }

    pub fn record(&mut self, batch_size: usize) {
        self.total_inferences += batch_size as u64;
        // Simulated: ~10ns per weight on GPU with ternary (XNOR+popcount)
        self.total_time_us += (batch_size as u64 * self.weights_per_inference as u64 * 10) / 1000;
    }

    pub fn inferences_per_sec(&self) -> f64 {
        if self.total_time_us == 0 { return 0.0; }
        self.total_inferences as f64 / (self.total_time_us as f64 / 1_000_000.0)
    }

    pub fn effective_tflops(&self) -> f64 {
        // Each ternary op ≈ 1 XNOR = 1 op
        if self.total_time_us == 0 { return 0.0; }
        let total_ops = self.total_inferences as f64 * self.weights_per_inference as f64 * 2.0;
        total_ops / (self.total_time_us as f64 / 1_000_000.0) / 1e12
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trit_pack() {
        let t = TritPack::new(&[1, -1, 0, 1, -1, 0, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1]);
        assert_eq!(t.get(0), 1);
        assert_eq!(t.get(1), -1);
        assert_eq!(t.get(2), 0);
    }

    #[test]
    fn test_matvec() {
        let w = [TritPack::new(&[1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]); 2];
        let v = TritPack::new(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        let out = ternary_matvec(&w, &v, 2);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_layer_forward() {
        let w = TritPack::new(&[1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1]);
        let layer = TernaryLayer::new("test", &[w], 16, 1);
        let input = TritPack::new(&[1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0]);
        let output = layer.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_network_inference() {
        let w1 = TritPack::new(&[1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0]);
        let l1 = TernaryLayer::new("hidden", &[w1], 16, 1);
        let w2 = TritPack::new(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        let l2 = TernaryLayer::new("output", &[w2], 16, 1);
        let net = TernaryNetwork::new(vec![l1, l2]);
        let input = TritPack::new(&[1, -1, 0, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1]);
        let output = net.inference(&input);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_batch_inference() {
        let w = TritPack::new(&[1; 16]);
        let layer = TernaryLayer::new("test", &[w], 16, 1);
        let inputs = vec![TritPack::new(&[1; 16]); 5];
        let outputs = layer.forward_batch(&inputs);
        assert_eq!(outputs.len(), 5);
    }

    #[test]
    fn test_conservation() {
        let input = TritPack::new(&[1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0]);
        let output = vec![0]; // balanced
        let result = verify_conservation(&input, &output);
        assert!(result.preserved);
    }

    #[test]
    fn test_benchmark() {
        let mut bench = InferenceBenchmark::new(256);
        bench.record(100);
        assert_eq!(bench.total_inferences, 100);
        assert!(bench.inferences_per_sec() > 0.0);
    }

    #[test]
    fn test_matmul() {
        let a = [TritPack::new(&[1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]); 2];
        let b = [TritPack::new(&[1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0]); 2];
        let c = ternary_matmul(&a, &b, 2, 2);
        assert_eq!(c.len(), 2);
    }
}
