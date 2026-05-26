//! Execution engine.
//!
//! Walks a `CompiledGraph` in topological order, invoking each kernel with
//! the appropriate input/output tensor views. Accumulates joule accounting
//! and emits an `ExecutionTrace`.

use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::{
    DeterminismMode, ExecutionTrace, PlanHash, TopologyId,
};
use jouleclaw_core::energy::JouleAccounting;
use jouleclaw_core::error::{Error, ExecutionError, Result};
use jouleclaw_core::graph::{NodeId, NodeKind};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::hash::{hash_graph, hash_tensor};
use jouleclaw_core::kernel::ExecutionContext;
use jouleclaw_core::tensor::{Tensor, TensorView, TensorViewMut};
use std::collections::HashMap;
use std::time::Instant;

use crate::compile::CompiledGraph;

#[derive(Debug, Clone)]
pub struct ExecutionOptions {
    pub determinism: DeterminismMode,
    pub seed: Option<u64>,
}

impl Default for ExecutionOptions {
    fn default() -> Self {
        Self { determinism: DeterminismMode::Strict, seed: None }
    }
}

pub struct ExecutionResult {
    pub outputs: HashMap<String, Tensor>,
    pub trace: ExecutionTrace,
}

pub fn execute(
    compiled: &CompiledGraph,
    inputs: HashMap<String, Tensor>,
    opts: ExecutionOptions,
) -> Result<ExecutionResult> {
    let wall_start = Instant::now();

    // Bind tensors per node.
    let mut tensors: HashMap<NodeId, Tensor> = HashMap::new();
    let mut input_hashes = Vec::new();

    // First pass: hash inputs (uses borrows; doesn't consume).
    for &nid in &compiled.graph.inputs {
        let node = compiled.graph.node(nid);
        if let NodeKind::Input { name, .. } = &node.kind {
            let t = inputs.get(name).ok_or_else(|| {
                Error::Execution(ExecutionError::KernelFailed {
                    op: jouleclaw_core::op::OpKind::Tokenize,
                    backend: BackendId::Custom(0),
                    reason: format!("missing input: {}", name),
                })
            })?;
            input_hashes.push(hash_tensor(t));
        }
    }

    // Second pass: consume the inputs map to move tensors into the bound
    // map without cloning. This keeps each input tensor's storage Arc at
    // strong_count=1, enabling the in-place aliasing path in Scatter.
    let mut inputs = inputs;
    for &nid in &compiled.graph.inputs {
        let node = compiled.graph.node(nid);
        if let NodeKind::Input { name, .. } = &node.kind {
            let t = inputs.remove(name).expect("input was present in first pass");
            tensors.insert(nid, t);
        }
    }
    drop(inputs);

    // Bind constants directly.
    for node in &compiled.graph.nodes {
        if let NodeKind::Constant { tensor } = &node.kind {
            tensors.insert(node.id, Tensor {
                meta: tensor.meta.clone(),
                storage: tensor.storage.clone(),
            });
        }
    }

    // Execute nodes in topological order.
    let mut accounting = JouleAccounting::empty();
    let mut kernel_selections = Vec::new();
    let mut scratch = vec![0u8; 64 * 1024];

    let plan_by_node: HashMap<NodeId, &crate::compile::NodePlan> =
        compiled.plan.iter().map(|p| (p.node_id, p)).collect();

    for &nid in &compiled.topo_order {
        let node = compiled.graph.node(nid);
        match &node.kind {
            NodeKind::Op { op, .. } => {
                let plan = plan_by_node.get(&nid).ok_or_else(|| {
                    Error::Execution(ExecutionError::KernelFailed {
                        op: *op, backend: BackendId::Custom(0),
                        reason: "no plan entry for op node".into(),
                    })
                })?;

                if opts.determinism == DeterminismMode::Strict {
                    use jouleclaw_core::determinism::DeterminismClass::*;
                    match plan.kernel.determinism() {
                        Stochastic => {
                            return Err(Error::Determinism(
                                jouleclaw_core::error::DeterminismError::NonDeterministicKernelSelected {
                                    op: *op, backend: plan.kernel.backend(),
                                }));
                        }
                        _ => {}
                    }
                }

                // Determine whether we can run this node in-place.
                // The Node::aliases_input hint says "input i's storage may
                // be reused as the output's storage". Safe iff:
                //   (1) the hint is set,
                //   (2) the aliased input tensor's storage Arc is uniquely
                //       owned (strong_count == 1) — no other node holds it.
                // When safe, we steal the input's storage rather than
                // allocating fresh bytes, and the kernel runs over the
                // remaining inputs with the output pre-loaded.
                let alias_idx = node.aliases_input.map(|i| i as usize);
                let in_place_ok = alias_idx
                    .and_then(|i| node.inputs.get(i).copied())
                    .map(|in_id| {
                        let t = tensors.get(&in_id).expect("input not bound");
                        std::sync::Arc::strong_count(&t.storage) == 1
                            && t.meta.size_bytes() == node.output_meta[0].size_bytes()
                    })
                    .unwrap_or(false);

                let mut output = if in_place_ok {
                    // Move the input's tensor out of the bound map. This
                    // gives us sole ownership of its Arc<TensorStorage>.
                    let in_id = node.inputs[alias_idx.unwrap()];
                    let stolen = tensors.remove(&in_id)
                        .expect("checked above");
                    // Build the output with the stolen storage and the
                    // node's declared output meta.
                    Tensor {
                        meta: node.output_meta[0].clone(),
                        storage: stolen.storage,
                    }
                } else {
                    let out_meta = node.output_meta[0].clone();
                    Tensor::zeros(out_meta)
                };

                // Build input views and run.
                {
                    // When aliasing, input `alias_idx` is no longer in the
                    // bound map (we stole it). The kernel sees inputs in
                    // their declared positions; we synthesize a view of
                    // the aliased input from the output's current bytes
                    // (which is the input's pre-write content — exactly
                    // what the kernel would see if we hadn't aliased).
                    let in_tensors: Vec<Option<&Tensor>> = node.inputs.iter().enumerate()
                        .map(|(i, in_id)| {
                            if Some(i) == alias_idx && in_place_ok {
                                None  // sourced from output below
                            } else {
                                Some(tensors.get(in_id).expect("input not bound"))
                            }
                        })
                        .collect();

                    // Storage handle for the output's bytes. After this,
                    // we can build the output mutable view and the
                    // synthesized "aliased input" view at the same time
                    // by splitting borrows is impossible; instead we tell
                    // the kernel via the alias index which input is the
                    // output, and the kernel reads from the output bytes
                    // when it needs the "old dst" content.
                    //
                    // For Scatter specifically: when aliasing, the kernel's
                    // dst→output copy is a no-op (same bytes), so we can
                    // pass an EMPTY view for input 0 and update the
                    // Scatter kernel to detect that case. But changing
                    // the kernel ABI is invasive.
                    //
                    // Cleanest compromise: when aliasing, the executor
                    // handles Scatter ITSELF inline (just overlay src
                    // bytes at the offset; output already holds dst's
                    // old content). For other ops, aliasing isn't useful
                    // in the same way; we don't set aliases_input for them.
                    if in_place_ok && *op == OpKind::Scatter {
                        let (axis_raw, mut offset) = match &plan.attrs {
                            OpAttrs::Scatter { axis, offset } => (*axis, *offset),
                            _ => return Err(Error::Execution(
                                ExecutionError::KernelFailed {
                                    op: *op, backend: plan.kernel.backend(),
                                    reason: "Scatter expects OpAttrs::Scatter".into(),
                                })),
                        };
                        // scatter_inplace_dyn: a 3rd input (I32 [1]
                        // `pos`) overrides the static offset. The inline
                        // alias fast-path must honour it too, else the
                        // const decode graph would scatter every step's
                        // K/V at slot 0.
                        if in_tensors.len() >= 3 {
                            if let Some(p) = in_tensors[2] {
                                let b = &p.storage.bytes;
                                if b.len() >= 4 {
                                    offset = i32::from_le_bytes(
                                        [b[0], b[1], b[2], b[3]]).max(0) as usize;
                                }
                            }
                        }
                        let src = in_tensors[1].expect("src not aliased");
                        let storage = std::sync::Arc::get_mut(&mut output.storage)
                            .expect("output storage just freshly owned");
                        let rank = output.meta.shape.len();
                        let axis = if axis_raw < 0 {
                            (rank as i32 + axis_raw) as usize
                        } else { axis_raw as usize };
                        let elem_size = output.meta.dtype.size_bytes();
                        let outer: usize = output.meta.shape.iter().take(axis).product::<usize>().max(1);
                        let inner: usize = output.meta.shape.iter().skip(axis + 1).product::<usize>().max(1);
                        let dst_axis = output.meta.shape[axis];
                        let src_axis = src.meta.shape[axis];
                        let dst_row_bytes = dst_axis * inner * elem_size;
                        let src_row_bytes = src_axis * inner * elem_size;
                        let off_bytes = offset * inner * elem_size;
                        for o in 0..outer {
                            let dst_off = o * dst_row_bytes + off_bytes;
                            let src_off = o * src_row_bytes;
                            storage.bytes[dst_off..dst_off + src_row_bytes]
                                .copy_from_slice(&src.storage.bytes[src_off..src_off + src_row_bytes]);
                        }
                        // Accounting: just src_bytes worth of writes.
                        let src_bytes = src.storage.bytes.len();
                        let joules = (src_bytes as f64) * 1e-12;
                        accounting.total_joules += joules;
                        *accounting.by_op_kind.entry(*op).or_insert(0.0) += joules;
                        *accounting.by_backend.entry(plan.kernel.backend()).or_insert(0.0) += joules;
                        accounting.deterministic_joules += joules;
                    } else {
                        let in_views: Vec<TensorView<'_>> =
                            in_tensors.iter().map(|t| {
                                t.expect("non-aliased inputs are bound").view()
                            }).collect();

                        let storage = std::sync::Arc::get_mut(&mut output.storage)
                            .expect("fresh tensor storage must be uniquely owned");
                        let mut out_view = TensorViewMut {
                            meta: &output.meta,
                            bytes: &mut storage.bytes,
                        };

                        let mut ctx = ExecutionContext {
                            backend: plan.kernel.backend(),
                            deterministic: opts.determinism == DeterminismMode::Strict,
                            seed: opts.seed,
                            scratch: &mut scratch,
                        };
                        let result = plan.kernel.execute(
                            &mut ctx,
                            &plan.attrs,
                            &in_views,
                            std::slice::from_mut(&mut out_view),
                        )?;

                        // Accounting.
                        accounting.total_joules += result.joules.joules;
                        *accounting.by_op_kind.entry(*op).or_insert(0.0) += result.joules.joules;
                        *accounting.by_backend.entry(plan.kernel.backend()).or_insert(0.0)
                            += result.joules.joules;
                        match plan.kernel.determinism() {
                            jouleclaw_core::determinism::DeterminismClass::Deterministic => {
                                accounting.deterministic_joules += result.joules.joules;
                            }
                            _ => accounting.stochastic_joules += result.joules.joules,
                        }
                    }
                }

                kernel_selections.push((nid, jouleclaw_core::kernel::KernelId(0)));
                tensors.insert(nid, output);
            }
            NodeKind::Output { name } => {
                // The Output node forwards its single input. We resolve at the end.
                let _ = name;
            }
            _ => {}
        }
    }

    // Collect outputs.
    let mut outputs = HashMap::new();
    let mut output_hashes = Vec::new();
    for &nid in &compiled.graph.outputs {
        let node = compiled.graph.node(nid);
        if let NodeKind::Output { name } = &node.kind {
            let source = node.inputs[0];
            let t = tensors.get(&source).expect("output source not bound").clone();
            output_hashes.push(hash_tensor(&t));
            outputs.insert(name.clone(), t);
        }
    }

    let trace = ExecutionTrace {
        graph_hash: hash_graph(&compiled.graph),
        input_hashes,
        output_hashes,
        topology_id: TopologyId(0),
        kernel_selections,
        memory_plan_hash: PlanHash([0u8; 32]),
        joule_accounting: accounting,
        wall_clock: wall_start.elapsed(),
        determinism_mode: opts.determinism,
    };

    Ok(ExecutionResult { outputs, trace })
}
