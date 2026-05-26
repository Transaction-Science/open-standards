//! Energy accounting comparison: aliased Scatter vs plain Scatter.
//!
//! The executor's Scatter alias path skips the dst→output memcpy and
//! tracks only the `src_bytes` write traffic. Plain Scatter copies the
//! full dst (`dst_bytes`) plus overlays src. We verify the accounting
//! reflects this difference: at a fixed src size, increasing dst size
//! does NOT increase tracked write traffic for the aliased path, while
//! plain Scatter's write traffic grows linearly with dst.

use jouleclaw_core::graph::GraphBuilder;
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta};
use jouleclaw_core::op::OpKind;
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;

fn run_and_get_scatter_joules(
    dst_shape: &[usize], src_shape: &[usize],
    axis: i32, offset: usize, in_place: bool,
) -> f64 {
    let mut g = GraphBuilder::new();
    let dst_meta = TensorMeta::new(Dtype::F32, dst_shape);
    let src_meta = TensorMeta::new(Dtype::F32, src_shape);
    let dst = g.input("dst", dst_meta.clone());
    let src = g.input("src", src_meta.clone());
    let out = if in_place {
        g.scatter_inplace(dst, src, axis, offset)
    } else {
        g.scatter(dst, src, axis, offset)
    };
    g.output("out", out);
    let graph = g.build();

    let rt = Runtime::boot();
    let compiled = compile(graph, &rt.kernels).unwrap();

    let dst_elems: usize = dst_shape.iter().product();
    let src_elems: usize = src_shape.iter().product();
    let mut inputs = HashMap::new();
    inputs.insert("dst".into(), Tensor::from_f32(dst_meta, &vec![0.0; dst_elems]));
    inputs.insert("src".into(), Tensor::from_f32(src_meta, &vec![1.0; src_elems]));
    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    *res.trace.joule_accounting.by_op_kind.get(&OpKind::Scatter).unwrap_or(&0.0)
}

/// Plain Scatter's tracked energy scales with dst size.
#[test]
fn plain_scatter_energy_scales_with_dst() {
    let src_shape = &[4usize];
    let small_dst = run_and_get_scatter_joules(&[16], src_shape, 0, 0, false);
    let large_dst = run_and_get_scatter_joules(&[1024], src_shape, 0, 0, false);

    // Larger dst → strictly more energy.
    assert!(large_dst > small_dst,
        "plain scatter energy should grow with dst: small={}, large={}",
        small_dst, large_dst);
    // Ratio approximately matches the size ratio (1024/16 = 64).
    let ratio = large_dst / small_dst;
    assert!(ratio > 32.0,
        "plain scatter dst-size scaling: expected ratio > 32, got {}", ratio);
}

/// Aliased Scatter's tracked energy is independent of dst size.
#[test]
fn aliased_scatter_energy_independent_of_dst() {
    let src_shape = &[4usize];
    let small_dst = run_and_get_scatter_joules(&[16], src_shape, 0, 0, true);
    let large_dst = run_and_get_scatter_joules(&[1024], src_shape, 0, 0, true);

    // Aliased path tracks only src_bytes, which is the same in both cases.
    assert_eq!(small_dst, large_dst,
        "aliased scatter energy should be independent of dst size: \
         small={}, large={}", small_dst, large_dst);
}

/// Visible demo of the energy difference.
#[test]
fn energy_accounting_demo() {
    println!("\n=== Energy accounting: aliased vs plain Scatter ===");
    println!("Fixed src size = 4 elems = 16 bytes.");
    println!();
    println!("                  | plain         | aliased");
    println!("                  |---------------|---------------");
    for &dst_len in &[16usize, 64, 256, 1024, 4096] {
        let plain = run_and_get_scatter_joules(&[dst_len], &[4], 0, 0, false);
        let aliased = run_and_get_scatter_joules(&[dst_len], &[4], 0, 0, true);
        let ratio = if aliased > 0.0 { plain / aliased } else { 0.0 };
        println!("  dst = {:5} F32 | {:11.2e} J | {:11.2e} J  ({:>6.0}× cheaper)",
            dst_len, plain, aliased, ratio);
    }
    println!();
    println!("Aliased Scatter writes only src_bytes; plain writes dst_bytes + src_bytes.");
    println!("This is the real cost of NOT doing in-place: a copy of the entire");
    println!("preallocated buffer on every decode step.");
}
