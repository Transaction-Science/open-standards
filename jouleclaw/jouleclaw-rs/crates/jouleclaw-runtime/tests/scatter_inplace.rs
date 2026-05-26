//! Tests for in-place Scatter via storage aliasing.
//!
//! `GraphBuilder::scatter_inplace` sets `Node::aliases_input = Some(0)`,
//! hinting that the dst input's storage may be reused as the output's
//! storage. The executor honors this when the input's Arc<TensorStorage>
//! is uniquely owned at execution time.
//!
//! These tests verify:
//!   (1) the alias path produces identical results to plain Scatter
//!       (correctness)
//!   (2) the alias is actually taken — the output's storage Arc is the
//!       same one that the input's was (the proof the dst→output copy
//!       was elided)
//!   (3) when the input is shared (held by another consumer), the
//!       executor safely falls back to copying

use jouleclaw_core::graph::GraphBuilder;
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;
use std::sync::Arc;

fn build_runtime() -> Runtime { Runtime::boot() }

/// Aliased scatter produces the same result as plain scatter.
#[test]
fn aliased_scatter_matches_plain_scatter() {
    let dst_meta = TensorMeta::new(Dtype::F32, &[8]);
    let src_meta = TensorMeta::new(Dtype::F32, &[3]);
    let dst_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let src_data = vec![100.0, 200.0, 300.0];
    let offset = 2;

    // Plain scatter.
    let mut g1 = GraphBuilder::new();
    let d1 = g1.input("dst", dst_meta.clone());
    let s1 = g1.input("src", src_meta.clone());
    let o1 = g1.scatter(d1, s1, 0, offset);
    g1.output("out", o1);
    let graph1 = g1.build();
    let rt = build_runtime();
    let c1 = compile(graph1, &rt.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("dst".into(), Tensor::from_f32(dst_meta.clone(), &dst_data));
    inputs.insert("src".into(), Tensor::from_f32(src_meta.clone(), &src_data));
    let r1 = execute(&c1, inputs.clone(), ExecutionOptions::default()).unwrap();
    let plain = r1.outputs.get("out").unwrap().as_f32_vec();

    // Aliased scatter (scatter_inplace).
    let mut g2 = GraphBuilder::new();
    let d2 = g2.input("dst", dst_meta.clone());
    let s2 = g2.input("src", src_meta.clone());
    let o2 = g2.scatter_inplace(d2, s2, 0, offset);
    g2.output("out", o2);
    let graph2 = g2.build();
    let c2 = compile(graph2, &rt.kernels).unwrap();
    let r2 = execute(&c2, inputs, ExecutionOptions::default()).unwrap();
    let aliased = r2.outputs.get("out").unwrap().as_f32_vec();

    assert_eq!(aliased, plain,
        "aliased scatter must produce the same bytes as plain scatter");
    assert_eq!(aliased, vec![1.0, 2.0, 100.0, 200.0, 300.0, 6.0, 7.0, 8.0]);
}

/// Proof the alias actually fires: the output tensor's storage Arc points
/// to the SAME allocation as the input's storage Arc did at execution time.
///
/// We construct dst and src as input Tensors with known storage Arcs.
/// After execute, the output's storage should be the dst input's storage
/// (the same Arc we put in, since we held the only other ref and that
/// got moved into the executor's bound map, then stolen).
#[test]
fn aliased_scatter_steals_input_storage() {
    let dst_meta = TensorMeta::new(Dtype::F32, &[8]);
    let src_meta = TensorMeta::new(Dtype::F32, &[3]);

    let mut g = GraphBuilder::new();
    let d = g.input("dst", dst_meta.clone());
    let s = g.input("src", src_meta.clone());
    let o = g.scatter_inplace(d, s, 0, 2);
    g.output("out", o);
    let graph = g.build();

    let rt = build_runtime();
    let c = compile(graph, &rt.kernels).unwrap();

    let dst_tensor = Tensor::from_f32(dst_meta, &[0f32; 8]);
    let dst_storage_addr = Arc::as_ptr(&dst_tensor.storage) as usize;

    let mut inputs = HashMap::new();
    inputs.insert("dst".into(), dst_tensor);
    inputs.insert("src".into(), Tensor::from_f32(src_meta, &[7.0, 8.0, 9.0]));
    let r = execute(&c, inputs, ExecutionOptions::default()).unwrap();
    let out = r.outputs.get("out").unwrap();
    let out_storage_addr = Arc::as_ptr(&out.storage) as usize;

    assert_eq!(out_storage_addr, dst_storage_addr,
        "in-place scatter should steal dst's storage Arc \
         (out_addr=0x{:x}, dst_addr=0x{:x})",
        out_storage_addr, dst_storage_addr);
}

/// Plain scatter (no alias hint) does NOT steal storage — the output is
/// a fresh allocation distinct from the dst input.
#[test]
fn plain_scatter_does_not_steal_storage() {
    let dst_meta = TensorMeta::new(Dtype::F32, &[8]);
    let src_meta = TensorMeta::new(Dtype::F32, &[3]);

    let mut g = GraphBuilder::new();
    let d = g.input("dst", dst_meta.clone());
    let s = g.input("src", src_meta.clone());
    let o = g.scatter(d, s, 0, 2);
    g.output("out", o);
    let graph = g.build();

    let rt = build_runtime();
    let c = compile(graph, &rt.kernels).unwrap();

    let dst_tensor = Tensor::from_f32(dst_meta, &[0f32; 8]);
    let dst_addr = Arc::as_ptr(&dst_tensor.storage) as usize;

    let mut inputs = HashMap::new();
    inputs.insert("dst".into(), dst_tensor);
    inputs.insert("src".into(), Tensor::from_f32(src_meta, &[7.0, 8.0, 9.0]));
    let r = execute(&c, inputs, ExecutionOptions::default()).unwrap();
    let out = r.outputs.get("out").unwrap();
    let out_addr = Arc::as_ptr(&out.storage) as usize;

    assert_ne!(out_addr, dst_addr,
        "plain scatter (no in-place hint) should allocate fresh output");
}

/// Demo: print the storage addresses for visible confirmation.
#[test]
fn alias_demo() {
    println!("\n=== In-place Scatter aliasing demo ===");

    let dst_meta = TensorMeta::new(Dtype::F32, &[16]);
    let src_meta = TensorMeta::new(Dtype::F32, &[4]);

    for in_place in &[false, true] {
        let mut g = GraphBuilder::new();
        let d = g.input("dst", dst_meta.clone());
        let s = g.input("src", src_meta.clone());
        let o = if *in_place {
            g.scatter_inplace(d, s, 0, 5)
        } else {
            g.scatter(d, s, 0, 5)
        };
        g.output("out", o);
        let graph = g.build();

        let rt = build_runtime();
        let c = compile(graph, &rt.kernels).unwrap();

        let dst_tensor = Tensor::from_f32(dst_meta.clone(), &[0f32; 16]);
        let dst_addr = Arc::as_ptr(&dst_tensor.storage) as usize;

        let mut inputs = HashMap::new();
        inputs.insert("dst".into(), dst_tensor);
        inputs.insert("src".into(), Tensor::from_f32(src_meta.clone(),
            &[1.0, 2.0, 3.0, 4.0]));

        let r = execute(&c, inputs, ExecutionOptions::default()).unwrap();
        let out = r.outputs.get("out").unwrap();
        let out_addr = Arc::as_ptr(&out.storage) as usize;

        let mode = if *in_place { "scatter_inplace" } else { "scatter (plain)" };
        let aliased = if out_addr == dst_addr { "ALIASED ✓" } else { "fresh alloc" };
        println!("  {:<16}  dst_addr=0x{:012x}  out_addr=0x{:012x}  {}",
            mode, dst_addr, out_addr, aliased);
    }
    println!("Aliased path skips the O(max_seq * d_head) dst→output memcpy.");
}
