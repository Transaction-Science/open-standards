//! Content hashing for graphs and tensors.
//!
//! Phase 1 uses a deterministic 256-bit hash built from FNV-1a 64-bit and
//! mixing. Cryptographically weak but fast and deterministic; sufficient for
//! cache addressing and trace comparison. Phase 3+ will swap in BLAKE3 once
//! we accept third-party deps.

use crate::determinism::{GraphHash, TensorHash};
use crate::graph::{Graph, NodeKind};
use crate::tensor::Tensor;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Streaming FNV-1a hasher with four parallel lanes for 256-bit output.
pub struct Hasher256 {
    lanes: [u64; 4],
}

impl Hasher256 {
    pub fn new() -> Self {
        Self {
            lanes: [
                FNV_OFFSET,
                FNV_OFFSET ^ 0xa5a5_a5a5_a5a5_a5a5,
                FNV_OFFSET ^ 0x5a5a_5a5a_5a5a_5a5a,
                FNV_OFFSET ^ 0xc3c3_c3c3_c3c3_c3c3,
            ],
        }
    }

    pub fn update(&mut self, bytes: &[u8]) {
        for (i, &b) in bytes.iter().enumerate() {
            let lane = i & 3;
            self.lanes[lane] ^= b as u64;
            self.lanes[lane] = self.lanes[lane].wrapping_mul(FNV_PRIME);
        }
    }

    pub fn finalize(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, lane) in self.lanes.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&lane.to_le_bytes());
        }
        out
    }
}

impl Default for Hasher256 {
    fn default() -> Self { Self::new() }
}

/// Hash a graph by its structure: node kinds, op kinds, attrs (debug-formatted),
/// edge structure, and constant-tensor hashes. Phase 0 of hashing.
pub fn hash_graph(graph: &Graph) -> GraphHash {
    let mut h = Hasher256::new();
    for node in &graph.nodes {
        h.update(&node.id.0.to_le_bytes());
        match &node.kind {
            NodeKind::Input { name, meta } => {
                h.update(b"INPUT");
                h.update(name.as_bytes());
                h.update(&format!("{:?}", meta).as_bytes());
            }
            NodeKind::Output { name } => {
                h.update(b"OUTPUT");
                h.update(name.as_bytes());
            }
            NodeKind::Constant { tensor } => {
                h.update(b"CONST");
                h.update(&format!("{:?}", tensor.meta).as_bytes());
                h.update(tensor.storage.view_bytes());
            }
            NodeKind::Op { op, attrs } => {
                h.update(b"OP");
                h.update(format!("{:?}", op).as_bytes());
                h.update(format!("{:?}", attrs).as_bytes());
            }
        }
        for input in &node.inputs {
            h.update(&input.0.to_le_bytes());
        }
    }
    GraphHash(h.finalize())
}

/// Hash a tensor's contents (and metadata).
pub fn hash_tensor(tensor: &Tensor) -> TensorHash {
    let mut h = Hasher256::new();
    h.update(format!("{:?}", tensor.meta).as_bytes());
    h.update(tensor.storage.view_bytes());
    TensorHash(h.finalize())
}
