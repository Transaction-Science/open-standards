//! End-to-end Phase 1.1 demo.
//!
//! Builds a graph that exercises every one of the eight math primitives,
//! compiles on the reference backend, runs it twice with identical inputs,
//! and verifies the outputs are byte-identical (the determinism oracle in
//! action).

use jouleclaw_core::graph::GraphBuilder;
use jouleclaw_core::op::{ActivationKind, NormKind};
use jouleclaw_core::tensor::{Dtype, LifetimeTier, Tensor, TensorMeta};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;

fn main() {
    let seq = 4usize;
    let d = 16usize;
    let dff = 32usize;

    // Graph: a small block exercising MatMul, Norm, Activation, Mul,
    // Softmax, Add. Lookup and Sample are exercised in the unit tests.
    let graph = {
        let mut g = GraphBuilder::new();
        let x = g.input("x", TensorMeta::new(Dtype::F32, &[seq, d]));
        let w_norm = g.input("w_norm",
            TensorMeta::new(Dtype::F32, &[d]).with_tier(LifetimeTier::Cold));
        let w_proj = g.input("w_proj",
            TensorMeta::new(Dtype::F32, &[d, dff]).with_tier(LifetimeTier::Cold));
        let w_gate = g.input("w_gate",
            TensorMeta::new(Dtype::F32, &[d, dff]).with_tier(LifetimeTier::Cold));
        let w_down = g.input("w_down",
            TensorMeta::new(Dtype::F32, &[dff, d]).with_tier(LifetimeTier::Cold));

        let xn = g.norm(x, w_norm, NormKind::Rms, 1e-6);
        let proj = g.matmul(xn, w_proj);
        let gate = g.matmul(xn, w_gate);
        let gate = g.activation(gate, ActivationKind::SiLU);
        let hidden = g.mul(gate, proj);
        let hidden = g.softmax(hidden, -1);
        let y = g.matmul(hidden, w_down);
        let out = g.add(x, y);
        g.output("y", out);
        g.build()
    };
    println!("Graph: {} nodes", graph.nodes.len());

    // Build runtime with reference backend, compile.
    let runtime = Runtime::boot();
    println!("Registered {} kernels", runtime.kernels.iter().count());
    let compiled = compile(graph, &runtime.kernels).expect("compile failed");
    println!("Compiled: {} plan entries\n", compiled.plan.len());

    // Bind inputs from a deterministic PRNG.
    fn det_random(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = (s >> 40) as u32;
            (bits as f32) * (1.0 / (1u32 << 24) as f32) - 0.5
        }).collect()
    }
    fn bind(name: &str, shape: &[usize], seed: u64) -> (String, Tensor) {
        let n = shape.iter().product::<usize>();
        let data = det_random(n, seed);
        (name.to_string(), Tensor::from_f32(TensorMeta::new(Dtype::F32, shape), &data))
    }

    let mut inputs = HashMap::new();
    let (k, v) = bind("x", &[seq, d], 1); inputs.insert(k, v);
    let (k, v) = bind("w_norm", &[d], 2); inputs.insert(k, v);
    let (k, v) = bind("w_proj", &[d, dff], 3); inputs.insert(k, v);
    let (k, v) = bind("w_gate", &[d, dff], 4); inputs.insert(k, v);
    let (k, v) = bind("w_down", &[dff, d], 5); inputs.insert(k, v);

    // Run twice with identical inputs.
    let res1 = execute(&compiled, inputs.clone(), ExecutionOptions::default()).expect("exec1");
    let res2 = execute(&compiled, inputs.clone(), ExecutionOptions::default()).expect("exec2");

    println!("--- Run 1 ---");
    println!("graph_hash:  {}", hex(&res1.trace.graph_hash.0));
    println!("output_hash: {}", hex(&res1.trace.output_hashes[0].0));
    println!("joules:      {:.6e}", res1.trace.joule_accounting.total_joules);
    println!("wall_clock:  {:?}", res1.trace.wall_clock);

    println!("\n--- Run 2 ---");
    println!("graph_hash:  {}", hex(&res2.trace.graph_hash.0));
    println!("output_hash: {}", hex(&res2.trace.output_hashes[0].0));
    println!("joules:      {:.6e}", res2.trace.joule_accounting.total_joules);
    println!("wall_clock:  {:?}", res2.trace.wall_clock);

    // Determinism check.
    assert_eq!(res1.trace.graph_hash.0, res2.trace.graph_hash.0,
        "graph hash must be stable across runs");
    assert_eq!(res1.trace.output_hashes, res2.trace.output_hashes,
        "outputs must be byte-identical across runs");

    let y1 = res1.outputs.get("y").unwrap();
    let y2 = res2.outputs.get("y").unwrap();
    assert_eq!(y1.meta.shape, vec![seq, d]);
    assert_eq!(y1.storage.bytes, y2.storage.bytes);

    let y1f = y1.as_f32_vec();
    println!("\nOutput[0..8] = {:?}", &y1f[0..8.min(y1f.len())]);

    println!("\n--- Joule accounting (Run 1) ---");
    let mut by_op: Vec<_> = res1.trace.joule_accounting.by_op_kind.iter().collect();
    by_op.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (op, j) in by_op {
        println!("  {:>10?}: {:.6e} J", op, j);
    }

    println!("\nOK: graph and outputs are bit-identical across two runs.");
    println!("The determinism oracle is in place; other backends verify against this baseline.");
}

fn hex(b: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(b.len() * 2);
    for x in b { write!(s, "{:02x}", x).unwrap(); }
    s
}
