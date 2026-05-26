//! Graph compilation: topo-sort, kernel selection, output-shape inference.
//!
//! Phase 1.1 compiler is intentionally minimal. It:
//! 1. Topologically sorts the graph nodes.
//! 2. For each `Op` node, picks the first registered kernel matching the
//!    `(OpKind, BackendId)` pair, preferring the reference backend.
//! 3. Records output `TensorMeta` for each node.
//!
//! Phase 2+ adds: real placement plan, memory plan, fusion, layout planning.

use jouleclaw_core::backend::BackendId;
use jouleclaw_core::error::{Error, ExecutionError, Result};
use jouleclaw_core::graph::{Graph, NodeId, NodeKind};
use jouleclaw_core::kernel::Kernel;
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::TensorMeta;
use std::sync::Arc;

use crate::KernelRegistry;

/// A compiled, executable graph.
pub struct CompiledGraph {
    pub graph: Graph,
    pub topo_order: Vec<NodeId>,
    /// For each `Op` node, the kernel chosen and the attrs to pass.
    pub plan: Vec<NodePlan>,
    /// Output metadata for every node (indexed by NodeId).
    pub node_outputs: Vec<TensorMeta>,
}

pub struct NodePlan {
    pub node_id: NodeId,
    pub kernel: Arc<dyn Kernel>,
    pub attrs: OpAttrs,
}

pub fn compile(graph: Graph, registry: &KernelRegistry) -> Result<CompiledGraph> {
    let topo_order = topo_sort(&graph)?;
    let mut plan = Vec::new();
    let mut node_outputs: Vec<TensorMeta> = Vec::with_capacity(graph.nodes.len());
    // Initialize with placeholder; we rebuild from the actual nodes below.
    for node in &graph.nodes {
        node_outputs.push(node.output_meta.first().cloned().unwrap_or_else(|| {
            TensorMeta::new(jouleclaw_core::tensor::Dtype::F32, &[1])
        }));
    }

    for nid in &topo_order {
        let node = graph.node(*nid);
        if let NodeKind::Op { op, attrs } = &node.kind {
            // Gather input metas so kernels can express shape preferences.
            let input_metas: Vec<&jouleclaw_core::tensor::TensorMeta> = node.inputs
                .iter()
                .map(|in_id| &graph.node(*in_id).output_meta[0])
                .collect();
            let kernel = pick_kernel(registry, *op, attrs, &input_metas)
                .ok_or_else(|| {
                    Error::Execution(ExecutionError::KernelFailed {
                        op: *op,
                        backend: BackendId::Custom(0),
                        reason: "no kernel registered for op".into(),
                    })
                })?;
            plan.push(NodePlan { node_id: *nid, kernel, attrs: attrs.clone() });
        }
    }

    Ok(CompiledGraph { graph, topo_order, plan, node_outputs })
}

/// Pick a kernel for this op invocation. Each candidate's `prefers`
/// returns a [`KernelPreference`] based on the actual op attrs + input
/// shapes; the picker takes the highest-preference candidate and
/// breaks ties by preferring non-reference backends (vendor accel).
/// Refused candidates are excluded entirely.
fn pick_kernel(
    registry: &KernelRegistry,
    op: OpKind,
    attrs: &jouleclaw_core::op::OpAttrs,
    input_metas: &[&jouleclaw_core::tensor::TensorMeta],
) -> Option<Arc<dyn Kernel>> {
    let ref_backend = jouleclaw_backend_reference::BACKEND_ID;
    let mut best: Option<(Arc<dyn Kernel>, jouleclaw_core::kernel::KernelPreference, bool)> = None;
    for k in registry.iter() {
        if k.op_kind() != op { continue; }
        let pref = k.prefers(attrs, input_metas);
        if pref == jouleclaw_core::kernel::KernelPreference::Refuse { continue; }
        let is_accel = k.backend() != ref_backend;
        // Higher preference wins; tie-break to non-reference backend.
        let replace = match &best {
            None => true,
            Some((_, b_pref, b_accel)) =>
                pref > *b_pref || (pref == *b_pref && is_accel && !*b_accel),
        };
        if replace {
            best = Some((Arc::clone(k), pref, is_accel));
        }
    }
    best.map(|(k, _, _)| k)
}

fn topo_sort(graph: &Graph) -> Result<Vec<NodeId>> {
    // Simple Kahn's algorithm. Graphs are small in Phase 1.
    let n = graph.nodes.len();
    let mut indeg = vec![0u32; n];
    for node in &graph.nodes {
        for _ in &node.inputs { indeg[node.id.0 as usize] += 1; }
    }
    // Wait: we computed indegree per *target*. Actually, edges go from
    // node.inputs -> node, so node has indegree == node.inputs.len().
    // That's what the loop above does. Good.

    let mut queue: Vec<NodeId> = indeg.iter().enumerate()
        .filter(|(_, d)| **d == 0)
        .map(|(i, _)| NodeId(i as u32))
        .collect();
    let mut order = Vec::with_capacity(n);
    while let Some(nid) = queue.pop() {
        order.push(nid);
        // Find all successors and decrement.
        for other in &graph.nodes {
            if other.inputs.iter().any(|i| *i == nid) {
                indeg[other.id.0 as usize] -= 1;
                if indeg[other.id.0 as usize] == 0 {
                    queue.push(other.id);
                }
            }
        }
    }
    if order.len() != n {
        return Err(Error::Execution(ExecutionError::TopologyMismatch {
            reason: "graph has a cycle".into(),
        }));
    }
    Ok(order)
}
