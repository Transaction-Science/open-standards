//! 3D generation modality handler.
//!
//! Handles image-to-3D with:
//! - Feed-forward generation (no optimization loop)
//! - Gaussian Splatting for real-time rendering
//! - Arbitrary camera control

use super::{CacheStrategy, ModalityHandler, PrefetchPattern};
use crate::core::{Modality, Result};
use crate::tensor::Tensor;
use alloc::vec::Vec;
use std::collections::HashMap;

// ============================================================================
// Marching Cubes Lookup Tables (Paul Bourke / Lorensen & Cline)
// ============================================================================

/// Edge table: for each of the 256 cube configurations, which of the 12 edges
/// are intersected by the isosurface. Bit i set means edge i is intersected.
const EDGE_TABLE: [u16; 256] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
];

/// Triangle table: for each of 256 cube configs, list of edge indices forming
/// triangles. Each row has up to 16 entries, terminated by -1.
const TRI_TABLE: [[i8; 16]; 256] = [
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1],
    [3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1],
    [3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1],
    [3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1],
    [9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1],
    [9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1],
    [2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1],
    [8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1],
    [9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1],
    [4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1],
    [3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1],
    [1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1],
    [4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1],
    [4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1],
    [5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1],
    [2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1],
    [9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1],
    [0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1],
    [2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1],
    [10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1],
    [4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1],
    [5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1],
    [5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1],
    [9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1],
    [0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1],
    [1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1],
    [10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1],
    [8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1],
    [2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1],
    [7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1],
    [9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1],
    [2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1],
    [11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1],
    [9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1],
    [5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1],
    [11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1],
    [11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1],
    [1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1],
    [9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1],
    [5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1],
    [2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1],
    [6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1],
    [0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1],
    [3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1],
    [6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1],
    [10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1],
    [6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1],
    [1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1],
    [8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1],
    [7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1],
    [3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1],
    [0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1],
    [9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1],
    [8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1],
    [5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1],
    [0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1],
    [6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1],
    [10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1],
    [10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1],
    [8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1],
    [1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1],
    [0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1],
    [10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1],
    [0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1],
    [3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1],
    [6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1],
    [9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1],
    [8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1],
    [3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1],
    [6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1],
    [0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1],
    [10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1],
    [10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1],
    [1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1],
    [2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1],
    [7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1],
    [7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1],
    [2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1],
    [1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1],
    [11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1],
    [8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1],
    [0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1],
    [7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1],
    [10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1],
    [2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1],
    [6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1],
    [7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1],
    [2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1],
    [1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1],
    [10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1],
    [10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1],
    [0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1],
    [7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1],
    [6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1],
    [8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1],
    [9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1],
    [6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1],
    [4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1],
    [10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1],
    [8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1],
    [0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1],
    [1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1],
    [8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1],
    [10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1],
    [4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1],
    [10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1],
    [5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1],
    [11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1],
    [9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1],
    [6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1],
    [7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1],
    [3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1],
    [7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1],
    [3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1],
    [6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1],
    [9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1],
    [1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1],
    [4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1],
    [7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1],
    [6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1],
    [3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1],
    [0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1],
    [6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1],
    [0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1],
    [11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1],
    [6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1],
    [5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1],
    [9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1],
    [1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1],
    [1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1],
    [10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1],
    [0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1],
    [5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1],
    [10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1],
    [11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1],
    [9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1],
    [7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1],
    [2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1],
    [8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1],
    [9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1],
    [9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1],
    [1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1],
    [9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1],
    [9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1],
    [5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1],
    [0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1],
    [10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1],
    [2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1],
    [0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1],
    [0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1],
    [9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1],
    [5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1],
    [3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1],
    [5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1],
    [8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1],
    [0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1],
    [9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1],
    [1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1],
    [3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1],
    [4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1],
    [9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1],
    [11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1],
    [11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1],
    [2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1],
    [9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1],
    [3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1],
    [1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1],
    [4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1],
    [4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1],
    [0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1],
    [3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1],
    [3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1],
    [0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1],
    [9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1],
    [1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
];

/// Cube edge vertex indices: each edge connects two of the 8 cube corners.
const EDGE_VERTICES: [[usize; 2]; 12] = [
    [0, 1], [1, 2], [2, 3], [3, 0],  // Bottom face edges
    [4, 5], [5, 6], [6, 7], [7, 4],  // Top face edges
    [0, 4], [1, 5], [2, 6], [3, 7],  // Vertical edges
];

/// Cube corner offsets: (dx, dy, dz) for each of the 8 corners.
const CORNER_OFFSETS: [[usize; 3]; 8] = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];

/// Mesh extraction method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshExtractionMethod {
    /// Marching cubes: smooth isosurface, good for organic shapes.
    MarchingCubes,
    /// Dual contouring: preserves sharp features (edges, corners).
    DualContouring,
}

/// Extract an isosurface mesh from a 3D density grid using marching cubes.
///
/// Uses the full 256-entry lookup table for correct triangle generation,
/// linear interpolation for edge vertices, and vertex welding via quantized
/// position hashing to produce a watertight mesh with per-vertex normals.
pub fn marching_cubes(
    grid: &[f32],
    res: [usize; 3],
    threshold: f32,
    bounds_min: Vec3,
    bounds_max: Vec3,
) -> Result<TriangleMesh> {
    let [rx, ry, rz] = res;
    if rx < 2 || ry < 2 || rz < 2 {
        return empty_mesh();
    }

    let step = Vec3::new(
        (bounds_max.x - bounds_min.x) / (rx - 1) as f32,
        (bounds_max.y - bounds_min.y) / (ry - 1) as f32,
        (bounds_max.z - bounds_min.z) / (rz - 1) as f32,
    );

    let grid_val = |ix: usize, iy: usize, iz: usize| -> f32 {
        grid.get(iz * ry * rx + iy * rx + ix).copied().unwrap_or(0.0)
    };

    let mut vertices: Vec<f32> = Vec::new();
    let mut faces: Vec<u32> = Vec::new();
    // Vertex welding: quantized position → vertex index
    let mut weld_map: HashMap<(i32, i32, i32), u32> = HashMap::new();
    const WELD_SCALE: f32 = 10000.0;

    let get_or_insert_vertex = |pos: Vec3, verts: &mut Vec<f32>, wmap: &mut HashMap<(i32, i32, i32), u32>| -> u32 {
        let key = (
            (pos.x * WELD_SCALE) as i32,
            (pos.y * WELD_SCALE) as i32,
            (pos.z * WELD_SCALE) as i32,
        );
        if let Some(&idx) = wmap.get(&key) {
            idx
        } else {
            let idx = (verts.len() / 3) as u32;
            verts.push(pos.x);
            verts.push(pos.y);
            verts.push(pos.z);
            wmap.insert(key, idx);
            idx
        }
    };

    // Process each cube
    for iz in 0..rz - 1 {
        for iy in 0..ry - 1 {
            for ix in 0..rx - 1 {
                // Get corner values
                let mut corner_vals = [0.0f32; 8];
                for (c, off) in CORNER_OFFSETS.iter().enumerate() {
                    corner_vals[c] = grid_val(ix + off[0], iy + off[1], iz + off[2]);
                }

                // Compute cube index
                let mut cube_index: u8 = 0;
                for c in 0..8 {
                    if corner_vals[c] > threshold {
                        cube_index |= 1 << c;
                    }
                }

                let edges = EDGE_TABLE[cube_index as usize];
                if edges == 0 {
                    continue;
                }

                // Interpolate vertices on active edges
                let mut edge_verts = [0u32; 12];
                for e in 0..12 {
                    if edges & (1 << e) != 0 {
                        let [c0, c1] = EDGE_VERTICES[e];
                        let v0 = corner_vals[c0];
                        let v1 = corner_vals[c1];
                        let t = if (v1 - v0).abs() > 1e-10 {
                            (threshold - v0) / (v1 - v0)
                        } else {
                            0.5
                        };
                        let t = t.clamp(0.0, 1.0);

                        let p0 = Vec3::new(
                            bounds_min.x + (ix + CORNER_OFFSETS[c0][0]) as f32 * step.x,
                            bounds_min.y + (iy + CORNER_OFFSETS[c0][1]) as f32 * step.y,
                            bounds_min.z + (iz + CORNER_OFFSETS[c0][2]) as f32 * step.z,
                        );
                        let p1 = Vec3::new(
                            bounds_min.x + (ix + CORNER_OFFSETS[c1][0]) as f32 * step.x,
                            bounds_min.y + (iy + CORNER_OFFSETS[c1][1]) as f32 * step.y,
                            bounds_min.z + (iz + CORNER_OFFSETS[c1][2]) as f32 * step.z,
                        );

                        let pos = p0.lerp(&p1, t);
                        edge_verts[e] = get_or_insert_vertex(pos, &mut vertices, &mut weld_map);
                    }
                }

                // Emit triangles
                let row = &TRI_TABLE[cube_index as usize];
                let mut i = 0;
                while i < 16 && row[i] >= 0 {
                    faces.push(edge_verts[row[i] as usize]);
                    faces.push(edge_verts[row[i + 1] as usize]);
                    faces.push(edge_verts[row[i + 2] as usize]);
                    i += 3;
                }
            }
        }
    }

    let num_verts = vertices.len() / 3;
    let num_faces = faces.len() / 3;

    if num_verts == 0 {
        return empty_mesh();
    }

    // Compute face-weighted vertex normals
    let normals = compute_vertex_normals(&vertices, &faces, num_verts);

    let device = crate::hal::DeviceId::cpu();
    let faces_f32: Vec<f32> = faces.iter().map(|&i| i as f32).collect();

    Ok(TriangleMesh {
        vertices: Tensor::from_slice(&vertices, crate::core::Shape::from([num_verts, 3]), crate::tensor::DType::F32, device)?,
        faces: Tensor::from_slice(&faces_f32, crate::core::Shape::from([num_faces, 3]), crate::tensor::DType::F32, device)?,
        normals: Some(Tensor::from_slice(&normals, crate::core::Shape::from([num_verts, 3]), crate::tensor::DType::F32, device)?),
        uvs: None,
        texture: None,
    })
}

/// Compute face-weighted vertex normals by accumulating face normals.
fn compute_vertex_normals(vertices: &[f32], faces: &[u32], num_verts: usize) -> Vec<f32> {
    let mut normals = vec![0.0f32; num_verts * 3];
    let num_faces = faces.len() / 3;

    for f in 0..num_faces {
        let i0 = faces[f * 3] as usize;
        let i1 = faces[f * 3 + 1] as usize;
        let i2 = faces[f * 3 + 2] as usize;

        let v0 = Vec3::new(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
        let v1 = Vec3::new(vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
        let v2 = Vec3::new(vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

        let e1 = Vec3::new(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        let e2 = Vec3::new(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
        let n = e1.cross(&e2); // Area-weighted (not normalized)

        for &idx in &[i0, i1, i2] {
            normals[idx * 3] += n.x;
            normals[idx * 3 + 1] += n.y;
            normals[idx * 3 + 2] += n.z;
        }
    }

    // Normalize
    for i in 0..num_verts {
        let nx = normals[i * 3];
        let ny = normals[i * 3 + 1];
        let nz = normals[i * 3 + 2];
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len > 1e-10 {
            normals[i * 3] /= len;
            normals[i * 3 + 1] /= len;
            normals[i * 3 + 2] /= len;
        }
    }

    normals
}

/// Create an empty mesh.
fn empty_mesh() -> Result<TriangleMesh> {
    let device = crate::hal::DeviceId::cpu();
    let verts = vec![0.0f32; 3];
    let fs = vec![0.0f32; 3];
    Ok(TriangleMesh {
        vertices: Tensor::from_slice(&verts, crate::core::Shape::from([1, 3]), crate::tensor::DType::F32, device)?,
        faces: Tensor::from_slice(&fs, crate::core::Shape::from([1, 3]), crate::tensor::DType::F32, device)?,
        normals: None,
        uvs: None,
        texture: None,
    })
}

// ============================================================================
// Dual Contouring + QEF Solver
// ============================================================================

/// Quadric Error Function solver for optimal vertex placement in dual contouring.
///
/// Accumulates plane equations (point + normal) and finds the point that
/// minimizes the sum of squared distances to all planes.
struct QefSolver {
    ata: [[f64; 3]; 3],
    atb: [f64; 3],
    mass_point: [f64; 3],
    num_points: u32,
}

impl QefSolver {
    fn new() -> Self {
        Self {
            ata: [[0.0; 3]; 3],
            atb: [0.0; 3],
            mass_point: [0.0; 3],
            num_points: 0,
        }
    }

    /// Add a plane equation defined by a point on the plane and its normal.
    fn add(&mut self, position: [f64; 3], normal: [f64; 3]) {
        // A^T A += n * n^T
        for i in 0..3 {
            for j in 0..3 {
                self.ata[i][j] += normal[i] * normal[j];
            }
        }
        // A^T b += n * (n . p)
        let d = normal[0] * position[0] + normal[1] * position[1] + normal[2] * position[2];
        for i in 0..3 {
            self.atb[i] += normal[i] * d;
        }
        // Mass point accumulation
        for i in 0..3 {
            self.mass_point[i] += position[i];
        }
        self.num_points += 1;
    }

    /// Solve using Cramer's rule with Tikhonov regularization.
    /// Falls back to mass point if the system is degenerate.
    fn solve(&self, cell_min: [f64; 3], cell_max: [f64; 3]) -> [f64; 3] {
        if self.num_points == 0 {
            return [
                (cell_min[0] + cell_max[0]) * 0.5,
                (cell_min[1] + cell_max[1]) * 0.5,
                (cell_min[2] + cell_max[2]) * 0.5,
            ];
        }

        let mp = [
            self.mass_point[0] / self.num_points as f64,
            self.mass_point[1] / self.num_points as f64,
            self.mass_point[2] / self.num_points as f64,
        ];

        // Tikhonov regularization: λ = 0.1 * trace(A^T A) / 3
        let trace = self.ata[0][0] + self.ata[1][1] + self.ata[2][2];
        let lambda = 0.1 * trace / 3.0;

        let mut a = self.ata;
        let mut b = self.atb;

        // Add regularization toward mass point: (A^T A + λI) x = A^T b + λ * mp
        for i in 0..3 {
            a[i][i] += lambda;
            b[i] += lambda * mp[i];
        }

        // Cramer's rule for 3x3
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

        if det.abs() < 1e-12 {
            return [
                mp[0].clamp(cell_min[0], cell_max[0]),
                mp[1].clamp(cell_min[1], cell_max[1]),
                mp[2].clamp(cell_min[2], cell_max[2]),
            ];
        }

        let inv_det = 1.0 / det;
        let x = (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
               - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
               + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2])) * inv_det;
        let y = (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
               - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
               + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0])) * inv_det;
        let z = (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
               - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
               + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])) * inv_det;

        // Clamp to cell bounds
        [
            x.clamp(cell_min[0], cell_max[0]),
            y.clamp(cell_min[1], cell_max[1]),
            z.clamp(cell_min[2], cell_max[2]),
        ]
    }
}

/// Extract an isosurface mesh using dual contouring with QEF vertex placement.
///
/// Produces meshes that preserve sharp features (edges, corners) better than
/// marching cubes, at the cost of potentially non-manifold output.
pub fn dual_contouring(
    grid: &[f32],
    res: [usize; 3],
    threshold: f32,
    bounds_min: Vec3,
    bounds_max: Vec3,
) -> Result<TriangleMesh> {
    let [rx, ry, rz] = res;
    if rx < 2 || ry < 2 || rz < 2 {
        return empty_mesh();
    }

    let step = [
        (bounds_max.x - bounds_min.x) / (rx - 1) as f32,
        (bounds_max.y - bounds_min.y) / (ry - 1) as f32,
        (bounds_max.z - bounds_min.z) / (rz - 1) as f32,
    ];

    let grid_val = |ix: usize, iy: usize, iz: usize| -> f32 {
        grid.get(iz * ry * rx + iy * rx + ix).copied().unwrap_or(0.0)
    };

    // Numerical gradient at a grid point
    let gradient = |ix: usize, iy: usize, iz: usize| -> [f64; 3] {
        let gx = if ix + 1 < rx && ix > 0 {
            (grid_val(ix + 1, iy, iz) - grid_val(ix - 1, iy, iz)) as f64 / (2.0 * step[0] as f64)
        } else if ix + 1 < rx {
            (grid_val(ix + 1, iy, iz) - grid_val(ix, iy, iz)) as f64 / (step[0] as f64)
        } else {
            (grid_val(ix, iy, iz) - grid_val(ix - 1, iy, iz)) as f64 / (step[0] as f64)
        };
        let gy = if iy + 1 < ry && iy > 0 {
            (grid_val(ix, iy + 1, iz) - grid_val(ix, iy - 1, iz)) as f64 / (2.0 * step[1] as f64)
        } else if iy + 1 < ry {
            (grid_val(ix, iy + 1, iz) - grid_val(ix, iy, iz)) as f64 / (step[1] as f64)
        } else {
            (grid_val(ix, iy, iz) - grid_val(ix, iy - 1, iz)) as f64 / (step[1] as f64)
        };
        let gz = if iz + 1 < rz && iz > 0 {
            (grid_val(ix, iy, iz + 1) - grid_val(ix, iy, iz - 1)) as f64 / (2.0 * step[2] as f64)
        } else if iz + 1 < rz {
            (grid_val(ix, iy, iz + 1) - grid_val(ix, iy, iz)) as f64 / (step[2] as f64)
        } else {
            (grid_val(ix, iy, iz) - grid_val(ix, iy, iz - 1)) as f64 / (step[2] as f64)
        };
        let len = (gx * gx + gy * gy + gz * gz).sqrt();
        if len > 1e-10 { [gx / len, gy / len, gz / len] } else { [0.0, 0.0, 1.0] }
    };

    // Phase 1: Compute one vertex per cell using QEF
    let cx = rx - 1;
    let cy = ry - 1;
    let cz = rz - 1;
    let mut cell_vertices: Vec<Option<u32>> = vec![None; cx * cy * cz];
    let mut vertices: Vec<f32> = Vec::new();

    // Edge directions: (axis, d0, d1) for the 3 edge types per cell
    // X-edges: (ix, iy, iz) -> (ix+1, iy, iz)
    // Y-edges: (ix, iy, iz) -> (ix, iy+1, iz)
    // Z-edges: (ix, iy, iz) -> (ix, iy, iz+1)

    for iz in 0..cz {
        for iy in 0..cy {
            for ix in 0..cx {
                let mut qef = QefSolver::new();
                let mut has_sign_change = false;

                // Check all 12 edges of this cell for sign changes
                for e in 0..12 {
                    let [c0, c1] = EDGE_VERTICES[e];
                    let p0 = [ix + CORNER_OFFSETS[c0][0], iy + CORNER_OFFSETS[c0][1], iz + CORNER_OFFSETS[c0][2]];
                    let p1 = [ix + CORNER_OFFSETS[c1][0], iy + CORNER_OFFSETS[c1][1], iz + CORNER_OFFSETS[c1][2]];

                    if p0[0] >= rx || p0[1] >= ry || p0[2] >= rz || p1[0] >= rx || p1[1] >= ry || p1[2] >= rz {
                        continue;
                    }

                    let v0 = grid_val(p0[0], p0[1], p0[2]);
                    let v1 = grid_val(p1[0], p1[1], p1[2]);

                    if (v0 > threshold) != (v1 > threshold) {
                        has_sign_change = true;
                        let t = if (v1 - v0).abs() > 1e-10 { (threshold - v0) / (v1 - v0) } else { 0.5 };
                        let t = t.clamp(0.0, 1.0) as f64;

                        let pos = [
                            (bounds_min.x + p0[0] as f32 * step[0]) as f64 * (1.0 - t) + (bounds_min.x + p1[0] as f32 * step[0]) as f64 * t,
                            (bounds_min.y + p0[1] as f32 * step[1]) as f64 * (1.0 - t) + (bounds_min.y + p1[1] as f32 * step[1]) as f64 * t,
                            (bounds_min.z + p0[2] as f32 * step[2]) as f64 * (1.0 - t) + (bounds_min.z + p1[2] as f32 * step[2]) as f64 * t,
                        ];

                        // Interpolate gradient
                        let g0 = gradient(p0[0], p0[1], p0[2]);
                        let g1 = gradient(p1[0], p1[1], p1[2]);
                        let n = [
                            g0[0] * (1.0 - t) + g1[0] * t,
                            g0[1] * (1.0 - t) + g1[1] * t,
                            g0[2] * (1.0 - t) + g1[2] * t,
                        ];
                        let nlen = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                        let n = if nlen > 1e-10 { [n[0] / nlen, n[1] / nlen, n[2] / nlen] } else { [0.0, 0.0, 1.0] };

                        qef.add(pos, n);
                    }
                }

                if has_sign_change {
                    let cell_min = [
                        (bounds_min.x + ix as f32 * step[0]) as f64,
                        (bounds_min.y + iy as f32 * step[1]) as f64,
                        (bounds_min.z + iz as f32 * step[2]) as f64,
                    ];
                    let cell_max = [
                        (bounds_min.x + (ix + 1) as f32 * step[0]) as f64,
                        (bounds_min.y + (iy + 1) as f32 * step[1]) as f64,
                        (bounds_min.z + (iz + 1) as f32 * step[2]) as f64,
                    ];

                    let pos = qef.solve(cell_min, cell_max);
                    let idx = (vertices.len() / 3) as u32;
                    vertices.push(pos[0] as f32);
                    vertices.push(pos[1] as f32);
                    vertices.push(pos[2] as f32);
                    cell_vertices[iz * cy * cx + iy * cx + ix] = Some(idx);
                }
            }
        }
    }

    // Phase 2: Emit quads for each sign-changing edge between adjacent cells
    let mut faces: Vec<u32> = Vec::new();

    let cell_idx = |cx_: usize, cy_: usize, cz_: usize| -> Option<u32> {
        if cx_ < cx && cy_ < cy && cz_ < cz {
            cell_vertices[cz_ * cy * cx + cy_ * cx + cx_]
        } else {
            None
        }
    };

    // X-edges: connect 4 cells sharing the edge
    for iz in 0..rz {
        for iy in 0..ry {
            for ix in 0..rx - 1 {
                let v0 = grid_val(ix, iy, iz);
                let v1 = grid_val(ix + 1, iy, iz);
                if (v0 > threshold) != (v1 > threshold) {
                    // 4 cells sharing this X-edge (if they exist and have vertices)
                    if iy > 0 && iz > 0 {
                        let c0 = cell_idx(ix, iy - 1, iz - 1);
                        let c1 = cell_idx(ix, iy, iz - 1);
                        let c2 = cell_idx(ix, iy, iz);
                        let c3 = cell_idx(ix, iy - 1, iz);
                        if let (Some(a), Some(b), Some(c), Some(d)) = (c0, c1, c2, c3) {
                            if v0 > threshold {
                                faces.extend_from_slice(&[a, b, c, a, c, d]);
                            } else {
                                faces.extend_from_slice(&[a, c, b, a, d, c]);
                            }
                        }
                    }
                }
            }
        }
    }

    // Y-edges
    for iz in 0..rz {
        for iy in 0..ry - 1 {
            for ix in 0..rx {
                let v0 = grid_val(ix, iy, iz);
                let v1 = grid_val(ix, iy + 1, iz);
                if (v0 > threshold) != (v1 > threshold) {
                    if ix > 0 && iz > 0 {
                        let c0 = cell_idx(ix - 1, iy, iz - 1);
                        let c1 = cell_idx(ix, iy, iz - 1);
                        let c2 = cell_idx(ix, iy, iz);
                        let c3 = cell_idx(ix - 1, iy, iz);
                        if let (Some(a), Some(b), Some(c), Some(d)) = (c0, c1, c2, c3) {
                            if v0 > threshold {
                                faces.extend_from_slice(&[a, c, b, a, d, c]);
                            } else {
                                faces.extend_from_slice(&[a, b, c, a, c, d]);
                            }
                        }
                    }
                }
            }
        }
    }

    // Z-edges
    for iz in 0..rz - 1 {
        for iy in 0..ry {
            for ix in 0..rx {
                let v0 = grid_val(ix, iy, iz);
                let v1 = grid_val(ix, iy, iz + 1);
                if (v0 > threshold) != (v1 > threshold) {
                    if ix > 0 && iy > 0 {
                        let c0 = cell_idx(ix - 1, iy - 1, iz);
                        let c1 = cell_idx(ix, iy - 1, iz);
                        let c2 = cell_idx(ix, iy, iz);
                        let c3 = cell_idx(ix - 1, iy, iz);
                        if let (Some(a), Some(b), Some(c), Some(d)) = (c0, c1, c2, c3) {
                            if v0 > threshold {
                                faces.extend_from_slice(&[a, b, c, a, c, d]);
                            } else {
                                faces.extend_from_slice(&[a, c, b, a, d, c]);
                            }
                        }
                    }
                }
            }
        }
    }

    let num_verts = vertices.len() / 3;
    let num_faces = faces.len() / 3;

    if num_verts == 0 || num_faces == 0 {
        return empty_mesh();
    }

    let normals = compute_vertex_normals(&vertices, &faces, num_verts);
    let device = crate::hal::DeviceId::cpu();
    let faces_f32: Vec<f32> = faces.iter().map(|&i| i as f32).collect();

    Ok(TriangleMesh {
        vertices: Tensor::from_slice(&vertices, crate::core::Shape::from([num_verts, 3]), crate::tensor::DType::F32, device)?,
        faces: Tensor::from_slice(&faces_f32, crate::core::Shape::from([num_faces, 3]), crate::tensor::DType::F32, device)?,
        normals: Some(Tensor::from_slice(&normals, crate::core::Shape::from([num_verts, 3]), crate::tensor::DType::F32, device)?),
        uvs: None,
        texture: None,
    })
}

// ============================================================================
// QEM Mesh Simplification (Garland-Heckbert)
// ============================================================================

/// Quadric error matrix for mesh simplification.
/// Represents the sum of squared distances to a set of planes.
#[derive(Clone)]
struct Quadric {
    a: [[f64; 3]; 3],
    b: [f64; 3],
    c: f64,
}

impl Quadric {
    fn zero() -> Self {
        Self { a: [[0.0; 3]; 3], b: [0.0; 3], c: 0.0 }
    }

    /// Create from a plane equation (normal * x = d).
    fn from_plane(normal: [f64; 3], d: f64) -> Self {
        let n = normal;
        Self {
            a: [
                [n[0] * n[0], n[0] * n[1], n[0] * n[2]],
                [n[1] * n[0], n[1] * n[1], n[1] * n[2]],
                [n[2] * n[0], n[2] * n[1], n[2] * n[2]],
            ],
            b: [n[0] * d, n[1] * d, n[2] * d],
            c: d * d,
        }
    }

    fn add(&mut self, other: &Quadric) {
        for i in 0..3 {
            for j in 0..3 {
                self.a[i][j] += other.a[i][j];
            }
            self.b[i] += other.b[i];
        }
        self.c += other.c;
    }

    /// Evaluate error at a point: v^T A v - 2 b^T v + c
    fn error(&self, v: [f64; 3]) -> f64 {
        let mut result = self.c;
        for i in 0..3 {
            result -= 2.0 * self.b[i] * v[i];
            for j in 0..3 {
                result += v[i] * self.a[i][j] * v[j];
            }
        }
        result
    }

    /// Find optimal point minimizing error via 3x3 solve. Returns None if degenerate.
    fn optimal_point(&self) -> Option<[f64; 3]> {
        let a = &self.a;
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

        if det.abs() < 1e-12 {
            return None;
        }

        let inv_det = 1.0 / det;
        let b = &self.b;
        let x = (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
               - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
               + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2])) * inv_det;
        let y = (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
               - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
               + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0])) * inv_det;
        let z = (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
               - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
               + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])) * inv_det;

        Some([x, y, z])
    }
}

/// Union-find with path compression and union by rank.
struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u32>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            self.parent[x as usize] = self.parent[self.parent[x as usize] as usize];
            x = self.parent[x as usize];
        }
        x
    }

    fn union(&mut self, x: u32, y: u32) -> u32 {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return rx;
        }
        if self.rank[rx as usize] < self.rank[ry as usize] {
            self.parent[rx as usize] = ry;
            ry
        } else if self.rank[rx as usize] > self.rank[ry as usize] {
            self.parent[ry as usize] = rx;
            rx
        } else {
            self.parent[ry as usize] = rx;
            self.rank[rx as usize] += 1;
            rx
        }
    }
}

/// Simplify a triangle mesh using Garland-Heckbert quadric error metrics.
///
/// Iteratively collapses the lowest-cost edge until the face count reaches
/// `target_ratio` of the original. Uses a priority queue with generation
/// counters for stale-candidate detection.
pub fn simplify_mesh(mesh: &TriangleMesh, target_ratio: f32) -> Result<TriangleMesh> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let vertices_data: Vec<f32> = mesh.vertices.to_vec()?;
    let faces_data: Vec<f32> = mesh.faces.to_vec()?;

    let num_verts = vertices_data.len() / 3;
    let num_faces = faces_data.len() / 3;

    if num_verts < 4 || num_faces < 4 {
        return Ok(TriangleMesh {
            vertices: mesh.vertices.clone(),
            faces: mesh.faces.clone(),
            normals: mesh.normals.clone(),
            uvs: None,
            texture: None,
        });
    }

    let target_faces = ((num_faces as f32 * target_ratio).ceil() as usize).max(4);

    // Build vertex positions (f64 for precision)
    let mut positions: Vec<[f64; 3]> = (0..num_verts)
        .map(|i| [
            vertices_data[i * 3] as f64,
            vertices_data[i * 3 + 1] as f64,
            vertices_data[i * 3 + 2] as f64,
        ])
        .collect();

    // Build face index list
    let mut faces: Vec<[u32; 3]> = (0..num_faces)
        .map(|f| [
            faces_data[f * 3] as u32,
            faces_data[f * 3 + 1] as u32,
            faces_data[f * 3 + 2] as u32,
        ])
        .collect();

    // Build per-vertex quadrics
    let mut quadrics: Vec<Quadric> = vec![Quadric::zero(); num_verts];
    for face in &faces {
        let v0 = positions[face[0] as usize];
        let v1 = positions[face[1] as usize];
        let v2 = positions[face[2] as usize];

        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len < 1e-15 {
            continue;
        }
        let n = [n[0] / len, n[1] / len, n[2] / len];
        let d = n[0] * v0[0] + n[1] * v0[1] + n[2] * v0[2];
        let q = Quadric::from_plane(n, d);

        for &vi in face {
            quadrics[vi as usize].add(&q);
        }
    }

    // Build edge set
    let mut edge_set: HashMap<(u32, u32), ()> = HashMap::new();
    for face in &faces {
        for k in 0..3 {
            let a = face[k];
            let b = face[(k + 1) % 3];
            let key = if a < b { (a, b) } else { (b, a) };
            edge_set.insert(key, ());
        }
    }

    // Generation counters for stale detection
    let mut generations: Vec<u32> = vec![0; num_verts];
    let mut uf = UnionFind::new(num_verts);

    // Collapse candidate
    #[derive(PartialEq)]
    struct Candidate {
        cost: f64,
        v0: u32,
        v1: u32,
        gen0: u32,
        gen1: u32,
    }
    impl Eq for Candidate {}
    impl PartialOrd for Candidate {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Candidate {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.cost.partial_cmp(&other.cost).unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    let compute_cost = |v0: u32, v1: u32, quadrics: &[Quadric], positions: &[[f64; 3]]| -> (f64, [f64; 3]) {
        let mut combined = quadrics[v0 as usize].clone();
        combined.add(&quadrics[v1 as usize]);

        let optimal = combined.optimal_point().unwrap_or_else(|| {
            // Fallback: midpoint
            let p0 = positions[v0 as usize];
            let p1 = positions[v1 as usize];
            [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5, (p0[2] + p1[2]) * 0.5]
        });

        let cost = combined.error(optimal).max(0.0);
        (cost, optimal)
    };

    // Build priority queue
    let mut heap: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
    for &(v0, v1) in edge_set.keys() {
        let (cost, _) = compute_cost(v0, v1, &quadrics, &positions);
        heap.push(Reverse(Candidate {
            cost,
            v0,
            v1,
            gen0: generations[v0 as usize],
            gen1: generations[v1 as usize],
        }));
    }

    // Collapse loop
    let mut live_faces = num_faces;
    let mut face_alive: Vec<bool> = vec![true; num_faces];

    while live_faces > target_faces {
        let candidate = match heap.pop() {
            Some(Reverse(c)) => c,
            None => break,
        };

        // Check staleness
        let r0 = uf.find(candidate.v0);
        let r1 = uf.find(candidate.v1);
        if r0 == r1 {
            continue;
        }
        if generations[r0 as usize] != candidate.gen0 || generations[r1 as usize] != candidate.gen1 {
            continue;
        }

        // Compute collapse position
        let (_, optimal) = compute_cost(r0, r1, &quadrics, &positions);

        // Merge quadrics
        let q1_clone = quadrics[r1 as usize].clone();
        quadrics[r0 as usize].add(&q1_clone);

        // Union and update position
        let root = uf.union(r0, r1);
        positions[root as usize] = optimal;
        quadrics[root as usize] = quadrics[r0 as usize].clone();
        if root != r0 {
            quadrics[root as usize] = quadrics[r0 as usize].clone();
        }
        generations[root as usize] += 1;

        // Update faces: remap collapsed vertex, remove degenerate faces
        for (fi, face) in faces.iter_mut().enumerate() {
            if !face_alive[fi] {
                continue;
            }
            for v in face.iter_mut() {
                let rv = uf.find(*v);
                *v = rv;
            }
            if face[0] == face[1] || face[1] == face[2] || face[0] == face[2] {
                face_alive[fi] = false;
                live_faces -= 1;
            }
        }

        // Re-insert edges touching the new vertex
        let mut neighbors: Vec<u32> = Vec::new();
        for (fi, face) in faces.iter().enumerate() {
            if !face_alive[fi] {
                continue;
            }
            for &v in face {
                if v == root {
                    for &other in face {
                        if other != root && !neighbors.contains(&other) {
                            neighbors.push(other);
                        }
                    }
                }
            }
        }

        for &nb in &neighbors {
            let (cost, _) = compute_cost(root, nb, &quadrics, &positions);
            heap.push(Reverse(Candidate {
                cost,
                v0: root,
                v1: nb,
                gen0: generations[root as usize],
                gen1: generations[nb as usize],
            }));
        }
    }

    // Compact: build new vertex/face arrays
    let mut new_vertex_map: HashMap<u32, u32> = HashMap::new();
    let mut new_vertices: Vec<f32> = Vec::new();
    let mut new_faces: Vec<f32> = Vec::new();

    for (fi, face) in faces.iter().enumerate() {
        if !face_alive[fi] {
            continue;
        }
        let mut mapped = [0u32; 3];
        for (k, &v) in face.iter().enumerate() {
            let root = uf.find(v);
            let new_idx = *new_vertex_map.entry(root).or_insert_with(|| {
                let idx = (new_vertices.len() / 3) as u32;
                new_vertices.push(positions[root as usize][0] as f32);
                new_vertices.push(positions[root as usize][1] as f32);
                new_vertices.push(positions[root as usize][2] as f32);
                idx
            });
            mapped[k] = new_idx;
        }
        if mapped[0] != mapped[1] && mapped[1] != mapped[2] && mapped[0] != mapped[2] {
            new_faces.push(mapped[0] as f32);
            new_faces.push(mapped[1] as f32);
            new_faces.push(mapped[2] as f32);
        }
    }

    let final_verts = new_vertices.len() / 3;
    let final_faces = new_faces.len() / 3;
    let device = crate::hal::DeviceId::cpu();

    if final_verts == 0 || final_faces == 0 {
        return empty_mesh();
    }

    // Recompute normals
    let faces_u32: Vec<u32> = new_faces.iter().map(|&f| f as u32).collect();
    let normals = compute_vertex_normals(&new_vertices, &faces_u32, final_verts);

    Ok(TriangleMesh {
        vertices: Tensor::from_slice(&new_vertices, crate::core::Shape::from([final_verts, 3]), crate::tensor::DType::F32, device)?,
        faces: Tensor::from_slice(&new_faces, crate::core::Shape::from([final_faces, 3]), crate::tensor::DType::F32, device)?,
        normals: Some(Tensor::from_slice(&normals, crate::core::Shape::from([final_verts, 3]), crate::tensor::DType::F32, device)?),
        uvs: None,
        texture: None,
    })
}

/// 3D input configuration.
#[derive(Debug, Clone)]
pub struct ThreeDInput {
    /// Input source
    pub source: ThreeDSource,
    /// Output representation
    pub representation: Representation3D,
}

/// 3D source.
#[derive(Debug, Clone)]
pub enum ThreeDSource {
    /// Single image to 3D
    SingleImage(Tensor),
    /// Multiple view images
    MultiView(Vec<(Tensor, Camera3D)>),
    /// Text to 3D
    Text(alloc::string::String),
}

/// 3D representation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Representation3D {
    /// Gaussian splats (fast rendering)
    GaussianSplats,
    /// Triangle mesh
    Mesh,
    /// Triplane features (generation intermediate)
    Triplane,
    /// Neural radiance field
    NeRF,
}

impl Default for Representation3D {
    fn default() -> Self {
        Self::GaussianSplats
    }
}

/// 3D output.
#[derive(Debug, Default)]
pub struct ThreeDOutput {
    /// Generated 3D object
    pub object: Option<Object3D>,
    /// Statistics
    pub stats: ThreeDStats,
}

/// 3D generation statistics.
#[derive(Debug, Default, Clone)]
pub struct ThreeDStats {
    /// Encoding time (ms)
    pub encoding_time_ms: f32,
    /// Generation time (ms)
    pub generation_time_ms: f32,
    /// First render time (ms)
    pub first_render_ms: f32,
}

/// 3D handler.
#[derive(Debug)]
pub struct ThreeDHandler {
    /// Default representation
    default_representation: Representation3D,
}

impl ThreeDHandler {
    /// Create a new 3D handler.
    pub fn new() -> Self {
        Self {
            default_representation: Representation3D::GaussianSplats,
        }
    }

    /// Generate a 3D object from input.
    ///
    /// Pipeline:
    /// 1. Encode input (single image, multi-view images, or text)
    /// 2. Generate triplane features via feed-forward network
    /// 3. Convert triplane to target representation (Gaussian splats or mesh)
    /// 4. Return the 3D object with bounding box
    pub async fn generate(&self, input: ThreeDInput) -> Result<ThreeDOutput> {
        let start = std::time::Instant::now();

        // Step 1: Encode input to feature embedding
        let encoding_start = std::time::Instant::now();
        let features = match &input.source {
            ThreeDSource::SingleImage(image) => self.encode_image(image)?,
            ThreeDSource::MultiView(views) => self.encode_multi_view(views)?,
            ThreeDSource::Text(text) => self.encode_text(text)?,
        };
        let encoding_time = encoding_start.elapsed().as_secs_f32() * 1000.0;

        // Step 2: Generate triplane features from encoding
        let gen_start = std::time::Instant::now();
        let triplane_res = 64usize;
        let triplane_channels = 32usize;
        let triplane = self.generate_triplane(&features, triplane_res, triplane_channels)?;
        let generation_time = gen_start.elapsed().as_secs_f32() * 1000.0;

        // Step 3: Convert to target representation
        let representation = input.representation;
        let object = match representation {
            Representation3D::GaussianSplats => {
                let cloud = self.triplane_to_gaussians(&triplane, triplane_res, triplane_channels)?;
                let bounds = self.compute_gaussian_bounds(&cloud);
                let center = bounds.center();
                Object3D {
                    representation: Object3DData::GaussianSplats(cloud),
                    bounds,
                    center,
                }
            }
            Representation3D::Mesh => {
                let mesh = self.triplane_to_mesh(&triplane, triplane_res, triplane_channels)?;
                let bounds = self.compute_mesh_bounds(&mesh);
                let center = bounds.center();
                Object3D {
                    representation: Object3DData::Mesh(mesh),
                    bounds,
                    center,
                }
            }
            Representation3D::Triplane => {
                Object3D {
                    representation: Object3DData::Triplane(triplane),
                    bounds: BoundingBox {
                        min: Vec3::new(-1.0, -1.0, -1.0),
                        max: Vec3::new(1.0, 1.0, 1.0),
                    },
                    center: Vec3::ZERO,
                }
            }
            Representation3D::NeRF => {
                // NeRF uses triplane as backing representation
                Object3D {
                    representation: Object3DData::Triplane(triplane),
                    bounds: BoundingBox {
                        min: Vec3::new(-1.0, -1.0, -1.0),
                        max: Vec3::new(1.0, 1.0, 1.0),
                    },
                    center: Vec3::ZERO,
                }
            }
        };

        // Step 4: First render
        let render_start = std::time::Instant::now();
        let _preview = self.render_view(&object, &Camera3D::default());
        let first_render = render_start.elapsed().as_secs_f32() * 1000.0;

        Ok(ThreeDOutput {
            object: Some(object),
            stats: ThreeDStats {
                encoding_time_ms: encoding_time,
                generation_time_ms: generation_time,
                first_render_ms: first_render,
            },
        })
    }

    /// Encode a single image to feature vector.
    ///
    /// Vision encoder requires loaded DINOv2/CLIP weights for proper feature extraction.
    /// Without weights, uses global average pooling as a feature extraction fallback
    /// that preserves coarse spatial statistics of the input.
    fn encode_image(&self, image: &Tensor) -> Result<Vec<f32>> {
        let image_data: Vec<f32> = image.to_vec()?;
        let feature_dim = 512;
        let mut features = vec![0.0f32; feature_dim];

        if image_data.is_empty() {
            return Ok(features);
        }

        // Global average pooling over spatial dimensions
        let pool_size = image_data.len() / feature_dim.max(1);
        for i in 0..feature_dim {
            let start = i * pool_size;
            let end = (start + pool_size).min(image_data.len());
            if start < end {
                let sum: f32 = image_data[start..end].iter().sum();
                features[i] = sum / (end - start) as f32;
            }
        }

        // Normalize
        let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for v in features.iter_mut() {
            *v /= norm;
        }

        Ok(features)
    }

    /// Encode multiple views to feature vector.
    fn encode_multi_view(&self, views: &[(Tensor, Camera3D)]) -> Result<Vec<f32>> {
        let feature_dim = 512;
        let mut combined = vec![0.0f32; feature_dim];

        for (image, _camera) in views {
            let view_features = self.encode_image(image)?;
            for (i, &f) in view_features.iter().enumerate() {
                combined[i] += f;
            }
        }

        // Average
        let n = views.len().max(1) as f32;
        for v in combined.iter_mut() {
            *v /= n;
        }

        Ok(combined)
    }

    /// Encode text prompt to feature vector.
    ///
    /// Text encoder requires loaded CLIP weights for proper text-to-embedding conversion.
    /// Without weights, uses a bag-of-characters embedding as fallback, where each
    /// character contributes to all feature dimensions via sinusoidal frequency weighting.
    /// This preserves some character-level information but lacks semantic understanding.
    fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        let feature_dim = 512;
        let mut features = vec![0.0f32; feature_dim];

        for (_i, ch) in text.chars().enumerate() {
            let char_val = (ch as u32 as f32) / 128.0 - 0.5;
            for d in 0..feature_dim {
                let freq = ((d as f32 + 1.0) / feature_dim as f32 * std::f32::consts::PI).sin();
                features[d] += char_val * freq / (text.len() as f32).sqrt();
            }
        }

        // Normalize
        let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for v in features.iter_mut() {
            *v /= norm;
        }

        Ok(features)
    }

    /// Generate triplane features from conditioning.
    ///
    /// Triplane generation requires loaded neural network weights (e.g., from
    /// a feed-forward 3D generator like InstantMesh or TripoSR).
    /// Without weights, projects features onto planes using smooth spatial
    /// variation that produces a plausible density distribution.
    fn generate_triplane(
        &self,
        features: &[f32],
        resolution: usize,
        channels: usize,
    ) -> Result<TriplaneFeatures> {
        let plane_size = resolution * resolution * channels;
        let device = crate::hal::DeviceId::cpu();
        let plane_shape = crate::core::Shape::from([channels, resolution, resolution]);

        let mut xy_data = vec![0.0f32; plane_size];
        let mut xz_data = vec![0.0f32; plane_size];
        let mut yz_data = vec![0.0f32; plane_size];

        // Feature projection onto each plane
        for y in 0..resolution {
            for x in 0..resolution {
                for c in 0..channels {
                    let feat_idx = c % features.len().max(1);
                    let feat = features.get(feat_idx).copied().unwrap_or(0.0);

                    let u = x as f32 / resolution as f32;
                    let v = y as f32 / resolution as f32;
                    let idx = c * resolution * resolution + y * resolution + x;

                    // Smooth spatial variation from features
                    xy_data[idx] = feat * (1.0 - (u - 0.5).abs() * 2.0) * (1.0 - (v - 0.5).abs() * 2.0);
                    xz_data[idx] = feat * (1.0 - (u - 0.5).abs() * 2.0);
                    yz_data[idx] = feat * (1.0 - (v - 0.5).abs() * 2.0);
                }
            }
        }

        Ok(TriplaneFeatures {
            xy: Tensor::from_slice(&xy_data, plane_shape.clone(), crate::tensor::DType::F32, device)?,
            xz: Tensor::from_slice(&xz_data, plane_shape.clone(), crate::tensor::DType::F32, device)?,
            yz: Tensor::from_slice(&yz_data, plane_shape, crate::tensor::DType::F32, device)?,
        })
    }

    /// Convert triplane features to Gaussian splat cloud.
    fn triplane_to_gaussians(
        &self,
        triplane: &TriplaneFeatures,
        resolution: usize,
        channels: usize,
    ) -> Result<GaussianCloud> {
        let xy_data: Vec<f32> = triplane.xy.to_vec()?;
        let xz_data: Vec<f32> = triplane.xz.to_vec()?;
        let yz_data: Vec<f32> = triplane.yz.to_vec()?;

        // Sample 3D points from the triplane
        let sample_res = resolution / 2; // Subsample for efficiency
        let mut positions = Vec::new();
        let mut colors = Vec::new();
        let mut opacities = Vec::new();

        for zi in 0..sample_res {
            for yi in 0..sample_res {
                for xi in 0..sample_res {
                    let x_coord = (xi as f32 / sample_res as f32) * 2.0 - 1.0;
                    let y_coord = (yi as f32 / sample_res as f32) * 2.0 - 1.0;
                    let z_coord = (zi as f32 / sample_res as f32) * 2.0 - 1.0;

                    // Query triplane features at this 3D point
                    let xi_plane = (((x_coord + 1.0) * 0.5) * (resolution - 1) as f32) as usize;
                    let yi_plane = (((y_coord + 1.0) * 0.5) * (resolution - 1) as f32) as usize;
                    let zi_plane = (((z_coord + 1.0) * 0.5) * (resolution - 1) as f32) as usize;

                    let xi_plane = xi_plane.min(resolution - 1);
                    let yi_plane = yi_plane.min(resolution - 1);
                    let zi_plane = zi_plane.min(resolution - 1);

                    // Aggregate features from all three planes
                    let mut density = 0.0f32;
                    for c in 0..channels.min(4) {
                        let xy_feat = xy_data.get(c * resolution * resolution + yi_plane * resolution + xi_plane).copied().unwrap_or(0.0);
                        let xz_feat = xz_data.get(c * resolution * resolution + zi_plane * resolution + xi_plane).copied().unwrap_or(0.0);
                        let yz_feat = yz_data.get(c * resolution * resolution + zi_plane * resolution + yi_plane).copied().unwrap_or(0.0);
                        density += (xy_feat + xz_feat + yz_feat) / 3.0;
                    }
                    density /= channels.min(4) as f32;

                    // Threshold: only keep points with significant density
                    if density.abs() > 0.15 {
                        positions.extend_from_slice(&[x_coord, y_coord, z_coord]);

                        // Color from triplane features
                        let r = xy_data.get(yi_plane * resolution + xi_plane).copied().unwrap_or(0.5).abs().clamp(0.0, 1.0);
                        let g = xz_data.get(zi_plane * resolution + xi_plane).copied().unwrap_or(0.5).abs().clamp(0.0, 1.0);
                        let b = yz_data.get(zi_plane * resolution + yi_plane).copied().unwrap_or(0.5).abs().clamp(0.0, 1.0);
                        colors.extend_from_slice(&[r, g, b]);

                        opacities.push(density.abs().clamp(0.1, 1.0));
                    }
                }
            }
        }

        let count = opacities.len();
        let device = crate::hal::DeviceId::cpu();

        if count == 0 {
            return GaussianCloud::empty();
        }

        // Create covariance data (identity-like for isotropic Gaussians)
        let covariances: Vec<f32> = (0..count)
            .flat_map(|_| vec![0.01, 0.0, 0.0, 0.01, 0.0, 0.01]) // Upper triangle of 3x3
            .collect();

        Ok(GaussianCloud {
            positions: Tensor::from_slice(&positions, crate::core::Shape::from([count, 3]), crate::tensor::DType::F32, device)?,
            covariances: Tensor::from_slice(&covariances, crate::core::Shape::from([count, 6]), crate::tensor::DType::F32, device)?,
            colors: Tensor::from_slice(&colors, crate::core::Shape::from([count, 3]), crate::tensor::DType::F32, device)?,
            opacities: Tensor::from_slice(&opacities, crate::core::Shape::from([count, 1]), crate::tensor::DType::F32, device)?,
            count,
        })
    }

    /// Convert triplane features to triangle mesh using marching cubes.
    fn triplane_to_mesh(
        &self,
        triplane: &TriplaneFeatures,
        resolution: usize,
        channels: usize,
    ) -> Result<TriangleMesh> {
        let xy_data: Vec<f32> = triplane.xy.to_vec()?;
        let xz_data: Vec<f32> = triplane.xz.to_vec()?;
        let yz_data: Vec<f32> = triplane.yz.to_vec()?;

        // Sample density field
        let grid_res = resolution / 2;
        let mut density_grid = vec![0.0f32; grid_res * grid_res * grid_res];

        for zi in 0..grid_res {
            for yi in 0..grid_res {
                for xi in 0..grid_res {
                    let xi_plane = (xi * (resolution - 1) / grid_res.max(1)).min(resolution - 1);
                    let yi_plane = (yi * (resolution - 1) / grid_res.max(1)).min(resolution - 1);
                    let zi_plane = (zi * (resolution - 1) / grid_res.max(1)).min(resolution - 1);

                    let mut density = 0.0f32;
                    for c in 0..channels.min(4) {
                        let xy_feat = xy_data.get(c * resolution * resolution + yi_plane * resolution + xi_plane).copied().unwrap_or(0.0);
                        let xz_feat = xz_data.get(c * resolution * resolution + zi_plane * resolution + xi_plane).copied().unwrap_or(0.0);
                        let yz_feat = yz_data.get(c * resolution * resolution + zi_plane * resolution + yi_plane).copied().unwrap_or(0.0);
                        density += (xy_feat + xz_feat + yz_feat) / 3.0;
                    }
                    density /= channels.min(4) as f32;
                    density_grid[zi * grid_res * grid_res + yi * grid_res + xi] = density;
                }
            }
        }

        // Full marching cubes with lookup tables
        marching_cubes(
            &density_grid,
            [grid_res, grid_res, grid_res],
            0.15,
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
        )
    }

    /// Compute bounding box from Gaussian positions.
    fn compute_gaussian_bounds(&self, cloud: &GaussianCloud) -> BoundingBox {
        if cloud.count == 0 {
            return BoundingBox::default();
        }

        let positions: Vec<f32> = cloud.positions.to_vec().unwrap_or_default();
        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        for i in 0..cloud.count {
            let x = positions.get(i * 3).copied().unwrap_or(0.0);
            let y = positions.get(i * 3 + 1).copied().unwrap_or(0.0);
            let z = positions.get(i * 3 + 2).copied().unwrap_or(0.0);

            min.x = min.x.min(x);
            min.y = min.y.min(y);
            min.z = min.z.min(z);
            max.x = max.x.max(x);
            max.y = max.y.max(y);
            max.z = max.z.max(z);
        }

        BoundingBox { min, max }
    }

    /// Compute bounding box from mesh vertices.
    fn compute_mesh_bounds(&self, mesh: &TriangleMesh) -> BoundingBox {
        let vertices: Vec<f32> = mesh.vertices.to_vec().unwrap_or_default();
        let num_verts = vertices.len() / 3;

        if num_verts == 0 {
            return BoundingBox::default();
        }

        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        for i in 0..num_verts {
            let x = vertices.get(i * 3).copied().unwrap_or(0.0);
            let y = vertices.get(i * 3 + 1).copied().unwrap_or(0.0);
            let z = vertices.get(i * 3 + 2).copied().unwrap_or(0.0);

            min.x = min.x.min(x);
            min.y = min.y.min(y);
            min.z = min.z.min(z);
            max.x = max.x.max(x);
            max.y = max.y.max(y);
            max.z = max.z.max(z);
        }

        BoundingBox { min, max }
    }

    /// Generate with progressive detail.
    ///
    /// Produces 3D output at increasing detail levels (Coarse -> Medium -> Fine),
    /// streaming each stage as it completes. This allows clients to display a
    /// low-resolution preview while higher-detail generation continues.
    pub fn generate_progressive(
        &self,
        input: ThreeDInput,
    ) -> crate::runtime::StreamingOutput<Progressive3D> {
        let (output, sender) = crate::runtime::stream::StreamBuilder::new()
            .buffer_size(4)
            .build();

        let default_representation = self.default_representation;

        tokio::spawn(async move {
            // Generate at increasing triplane resolutions for progressive detail
            let levels = [
                (DetailLevel::Coarse, 16usize, 8usize),
                (DetailLevel::Medium, 32usize, 16usize),
                (DetailLevel::Fine, 64usize, 32usize),
            ];

            // Create a temporary handler for generation
            let handler = ThreeDHandler {
                default_representation,
            };

            for (level, triplane_res, triplane_channels) in &levels {
                if sender.is_cancelled() {
                    break;
                }

                // Encode input features
                let features = match &input.source {
                    ThreeDSource::SingleImage(image) => match handler.encode_image(image) {
                        Ok(f) => f,
                        Err(e) => { let _ = sender.send_error(e).await; return; }
                    },
                    ThreeDSource::MultiView(views) => match handler.encode_multi_view(views) {
                        Ok(f) => f,
                        Err(e) => { let _ = sender.send_error(e).await; return; }
                    },
                    ThreeDSource::Text(text) => match handler.encode_text(text) {
                        Ok(f) => f,
                        Err(e) => { let _ = sender.send_error(e).await; return; }
                    },
                };

                // Generate triplane at current resolution
                let triplane = match handler.generate_triplane(&features, *triplane_res, *triplane_channels) {
                    Ok(t) => t,
                    Err(e) => { let _ = sender.send_error(e).await; return; }
                };

                // Convert to target representation
                let object = match input.representation {
                    Representation3D::GaussianSplats => {
                        match handler.triplane_to_gaussians(&triplane, *triplane_res, *triplane_channels) {
                            Ok(cloud) => {
                                let bounds = handler.compute_gaussian_bounds(&cloud);
                                let center = bounds.center();
                                Object3D {
                                    representation: Object3DData::GaussianSplats(cloud),
                                    bounds,
                                    center,
                                }
                            }
                            Err(e) => { let _ = sender.send_error(e).await; return; }
                        }
                    }
                    Representation3D::Mesh => {
                        match handler.triplane_to_mesh(&triplane, *triplane_res, *triplane_channels) {
                            Ok(mesh) => {
                                let bounds = handler.compute_mesh_bounds(&mesh);
                                let center = bounds.center();
                                Object3D {
                                    representation: Object3DData::Mesh(mesh),
                                    bounds,
                                    center,
                                }
                            }
                            Err(e) => { let _ = sender.send_error(e).await; return; }
                        }
                    }
                    Representation3D::Triplane | Representation3D::NeRF => {
                        Object3D {
                            representation: Object3DData::Triplane(triplane),
                            bounds: BoundingBox {
                                min: Vec3::new(-1.0, -1.0, -1.0),
                                max: Vec3::new(1.0, 1.0, 1.0),
                            },
                            center: Vec3::ZERO,
                        }
                    }
                };

                let progressive = Progressive3D {
                    level: *level,
                    object,
                };

                if sender.send(progressive).await.is_err() {
                    break;
                }
            }

            sender.complete();
        });

        output
    }

    /// Render a view from the generated object.
    pub fn render_view(&self, object: &Object3D, camera: &Camera3D) -> Result<Tensor> {
        match &object.representation {
            Object3DData::GaussianSplats(splats) => {
                self.render_gaussians(splats, camera)
            }
            Object3DData::Mesh(mesh) => {
                self.render_mesh(mesh, camera)
            }
            Object3DData::Triplane(_) => {
                Err(crate::core::Error::unsupported("direct triplane rendering"))
            }
        }
    }

    /// Stream views along a camera path.
    pub fn stream_camera_path(
        &self,
        _object: &Object3D,
        _path: CameraPath,
    ) -> crate::runtime::StreamingOutput<Tensor> {
        let (output, sender) = crate::runtime::stream::StreamBuilder::new()
            .buffer_size(8)
            .build();

        tokio::spawn(async move {
            // Camera path rendering not yet implemented; complete the stream
            // so consumers don't block indefinitely.
            sender.complete();
        });

        output
    }

    /// CPU-based Gaussian splatting renderer.
    ///
    /// Renders a set of 3D Gaussians to an image by:
    /// 1. Projecting each Gaussian center to screen space
    /// 2. Sorting by depth (back to front)
    /// 3. Rasterizing each Gaussian as a 2D splat with alpha blending
    fn render_gaussians(&self, splats: &GaussianCloud, camera: &Camera3D) -> Result<Tensor> {
        if splats.count == 0 {
            // Return black image
            return Tensor::zeros(
                crate::core::Shape::from([1, 3, 512, 512]),
                crate::tensor::DType::F32,
            );
        }

        let width = 512usize;
        let height = 512usize;
        let mut image_data = vec![0.0f32; 3 * height * width];

        // Get position data
        let positions: Vec<f32> = splats.positions.to_vec()?;
        let colors: Vec<f32> = splats.colors.to_vec()?;
        let opacities: Vec<f32> = splats.opacities.to_vec()?;

        // View and projection matrices
        let view = camera.view_matrix();
        let proj = camera.projection_matrix();

        // Project and sort Gaussians
        let mut projected: Vec<(usize, f32)> = Vec::with_capacity(splats.count); // (index, depth)

        for i in 0..splats.count {
            let px = positions.get(i * 3).copied().unwrap_or(0.0);
            let py = positions.get(i * 3 + 1).copied().unwrap_or(0.0);
            let pz = positions.get(i * 3 + 2).copied().unwrap_or(0.0);

            // Transform to view space
            let vx = view.data[0][0] * px + view.data[1][0] * py + view.data[2][0] * pz + view.data[3][0];
            let vy = view.data[0][1] * px + view.data[1][1] * py + view.data[2][1] * pz + view.data[3][1];
            let vz = view.data[0][2] * px + view.data[1][2] * py + view.data[2][2] * pz + view.data[3][2];

            // Skip if behind camera
            if vz > -camera.near {
                continue;
            }

            projected.push((i, vz));
        }

        // Sort back to front (larger negative z = farther away)
        projected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

        // Rasterize each Gaussian
        for &(idx, depth) in &projected {
            let px = positions.get(idx * 3).copied().unwrap_or(0.0);
            let py = positions.get(idx * 3 + 1).copied().unwrap_or(0.0);
            let pz = positions.get(idx * 3 + 2).copied().unwrap_or(0.0);

            // Project to screen coordinates
            let vx = view.data[0][0] * px + view.data[1][0] * py + view.data[2][0] * pz + view.data[3][0];
            let vy = view.data[0][1] * px + view.data[1][1] * py + view.data[2][1] * pz + view.data[3][1];
            let vz = view.data[0][2] * px + view.data[1][2] * py + view.data[2][2] * pz + view.data[3][2];

            let clip_x = proj.data[0][0] * vx + proj.data[2][0] * vz;
            let clip_y = proj.data[1][1] * vy + proj.data[2][1] * vz;
            let clip_w = proj.data[2][3] * vz + proj.data[3][3];

            if clip_w.abs() < 1e-6 {
                continue;
            }

            let ndc_x = clip_x / clip_w;
            let ndc_y = clip_y / clip_w;

            // NDC to pixel
            let screen_x = ((ndc_x + 1.0) * 0.5 * width as f32) as i32;
            let screen_y = ((1.0 - ndc_y) * 0.5 * height as f32) as i32;

            // Get color and opacity
            let r = colors.get(idx * 3).copied().unwrap_or(0.5);
            let g = colors.get(idx * 3 + 1).copied().unwrap_or(0.5);
            let b = colors.get(idx * 3 + 2).copied().unwrap_or(0.5);
            let alpha = opacities.get(idx).copied().unwrap_or(0.5).clamp(0.0, 1.0);

            // Splat radius based on depth
            let radius = (3.0 / (-depth).max(0.1)) as i32;
            let radius = radius.clamp(1, 20);

            // Rasterize Gaussian splat
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let px = screen_x + dx;
                    let py = screen_y + dy;

                    if px < 0 || px >= width as i32 || py < 0 || py >= height as i32 {
                        continue;
                    }

                    // Gaussian falloff
                    let dist_sq = (dx * dx + dy * dy) as f32;
                    let sigma_sq = (radius as f32 * 0.5).powi(2);
                    let weight = alpha * (-dist_sq / (2.0 * sigma_sq)).exp();

                    if weight < 0.001 {
                        continue;
                    }

                    let pixel_idx = py as usize * width + px as usize;

                    // Alpha blend (front-to-back would be more efficient but this works)
                    let r_idx = pixel_idx;
                    let g_idx = height * width + pixel_idx;
                    let b_idx = 2 * height * width + pixel_idx;

                    image_data[r_idx] = image_data[r_idx] * (1.0 - weight) + r * weight;
                    image_data[g_idx] = image_data[g_idx] * (1.0 - weight) + g * weight;
                    image_data[b_idx] = image_data[b_idx] * (1.0 - weight) + b * weight;
                }
            }
        }

        Tensor::from_slice(
            &image_data,
            crate::core::Shape::from([1, 3, height, width]),
            crate::tensor::DType::F32,
            crate::hal::DeviceId::cpu(),
        )
    }

    /// CPU-based mesh rasterizer.
    ///
    /// Simple scanline rasterizer that:
    /// 1. Projects triangle vertices to screen space
    /// 2. Rasterizes each triangle using barycentric coordinates
    /// 3. Z-buffer for depth testing
    fn render_mesh(&self, mesh: &TriangleMesh, camera: &Camera3D) -> Result<Tensor> {
        let width = 512usize;
        let height = 512usize;
        let mut image_data = vec![0.5f32; 3 * height * width]; // Gray background
        let mut z_buffer = vec![f32::INFINITY; height * width];

        let vertices: Vec<f32> = mesh.vertices.to_vec()?;
        let faces: Vec<f32> = mesh.faces.to_vec()?;

        let view = camera.view_matrix();
        let proj = camera.projection_matrix();

        let num_faces = faces.len() / 3;

        for f in 0..num_faces {
            let i0 = faces.get(f * 3).copied().unwrap_or(0.0) as usize;
            let i1 = faces.get(f * 3 + 1).copied().unwrap_or(0.0) as usize;
            let i2 = faces.get(f * 3 + 2).copied().unwrap_or(0.0) as usize;

            // Get vertices
            let get_vertex = |i: usize| -> (f32, f32, f32) {
                (
                    vertices.get(i * 3).copied().unwrap_or(0.0),
                    vertices.get(i * 3 + 1).copied().unwrap_or(0.0),
                    vertices.get(i * 3 + 2).copied().unwrap_or(0.0),
                )
            };

            let v0 = get_vertex(i0);
            let v1 = get_vertex(i1);
            let v2 = get_vertex(i2);

            // Project vertices
            let project = |p: (f32, f32, f32)| -> (f32, f32, f32) {
                let vx = view.data[0][0] * p.0 + view.data[1][0] * p.1 + view.data[2][0] * p.2 + view.data[3][0];
                let vy = view.data[0][1] * p.0 + view.data[1][1] * p.1 + view.data[2][1] * p.2 + view.data[3][1];
                let vz = view.data[0][2] * p.0 + view.data[1][2] * p.1 + view.data[2][2] * p.2 + view.data[3][2];

                let clip_x = proj.data[0][0] * vx + proj.data[2][0] * vz;
                let clip_y = proj.data[1][1] * vy + proj.data[2][1] * vz;
                let clip_w = proj.data[2][3] * vz + proj.data[3][3];

                if clip_w.abs() < 1e-6 {
                    return (0.0, 0.0, f32::INFINITY);
                }

                let ndc_x = clip_x / clip_w;
                let ndc_y = clip_y / clip_w;

                let sx = (ndc_x + 1.0) * 0.5 * width as f32;
                let sy = (1.0 - ndc_y) * 0.5 * height as f32;

                (sx, sy, -vz) // depth is positive distance from camera
            };

            let p0 = project(v0);
            let p1 = project(v1);
            let p2 = project(v2);

            // Skip degenerate or behind-camera triangles
            if p0.2 == f32::INFINITY || p1.2 == f32::INFINITY || p2.2 == f32::INFINITY {
                continue;
            }

            // Bounding box
            let min_x = (p0.0.min(p1.0).min(p2.0).floor() as i32).max(0);
            let max_x = (p0.0.max(p1.0).max(p2.0).ceil() as i32).min(width as i32 - 1);
            let min_y = (p0.1.min(p1.1).min(p2.1).floor() as i32).max(0);
            let max_y = (p0.1.max(p1.1).max(p2.1).ceil() as i32).min(height as i32 - 1);

            // Simple face normal for shading
            let e1 = (v1.0 - v0.0, v1.1 - v0.1, v1.2 - v0.2);
            let e2 = (v2.0 - v0.0, v2.1 - v0.1, v2.2 - v0.2);
            let normal = Vec3::new(
                e1.1 * e2.2 - e1.2 * e2.1,
                e1.2 * e2.0 - e1.0 * e2.2,
                e1.0 * e2.1 - e1.1 * e2.0,
            ).normalize();

            // Simple diffuse lighting
            let light_dir = Vec3::new(0.5, 0.8, 0.3).normalize();
            let diffuse = normal.dot(&light_dir).abs().clamp(0.1, 1.0);

            // Rasterize with barycentric coordinates
            let area = edge_function(p0.0, p0.1, p1.0, p1.1, p2.0, p2.1);
            if area.abs() < 1e-6 {
                continue;
            }

            for py in min_y..=max_y {
                for px in min_x..=max_x {
                    let fx = px as f32 + 0.5;
                    let fy = py as f32 + 0.5;

                    let w0 = edge_function(p1.0, p1.1, p2.0, p2.1, fx, fy) / area;
                    let w1 = edge_function(p2.0, p2.1, p0.0, p0.1, fx, fy) / area;
                    let w2 = 1.0 - w0 - w1;

                    if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                        let depth = w0 * p0.2 + w1 * p1.2 + w2 * p2.2;
                        let pixel_idx = py as usize * width + px as usize;

                        if depth < z_buffer[pixel_idx] {
                            z_buffer[pixel_idx] = depth;

                            // Shade pixel
                            let r_idx = pixel_idx;
                            let g_idx = height * width + pixel_idx;
                            let b_idx = 2 * height * width + pixel_idx;

                            image_data[r_idx] = diffuse * 0.8;
                            image_data[g_idx] = diffuse * 0.7;
                            image_data[b_idx] = diffuse * 0.6;
                        }
                    }
                }
            }
        }

        Tensor::from_slice(
            &image_data,
            crate::core::Shape::from([1, 3, height, width]),
            crate::tensor::DType::F32,
            crate::hal::DeviceId::cpu(),
        )
    }

    /// Simplify a mesh using QEM (Garland-Heckbert quadric error metrics).
    ///
    /// `target_ratio` is the fraction of faces to keep (e.g., 0.5 = reduce to 50%).
    pub fn simplify(&self, mesh: &TriangleMesh, target_ratio: f32) -> Result<TriangleMesh> {
        simplify_mesh(mesh, target_ratio)
    }
}

impl Default for ThreeDHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl ModalityHandler for ThreeDHandler {
    fn modality(&self) -> Modality {
        Modality::ThreeD
    }

    fn optimal_chunk_size(&self, _available_memory: usize) -> usize {
        1 // 3D objects are typically processed as whole
    }

    fn supports_streaming(&self) -> bool {
        true // Progressive detail levels
    }

    fn prefetch_pattern(&self) -> PrefetchPattern {
        PrefetchPattern::Spatial // Similar camera angles share data
    }

    fn cache_strategy(&self) -> CacheStrategy {
        CacheStrategy::Adaptive
    }
}

/// A 3D object.
#[derive(Debug)]
pub struct Object3D {
    /// Object representation
    pub representation: Object3DData,
    /// Bounding box
    pub bounds: BoundingBox,
    /// Center point
    pub center: Vec3,
}

/// 3D object data.
#[derive(Debug)]
pub enum Object3DData {
    /// Gaussian splat cloud
    GaussianSplats(GaussianCloud),
    /// Triangle mesh
    Mesh(TriangleMesh),
    /// Triplane features
    Triplane(TriplaneFeatures),
}

/// Gaussian splat cloud.
#[derive(Debug)]
pub struct GaussianCloud {
    /// Positions [N, 3]
    pub positions: Tensor,
    /// Covariances [N, 6] (upper triangle)
    pub covariances: Tensor,
    /// Colors [N, 3] or [N, spherical_harmonics]
    pub colors: Tensor,
    /// Opacities [N, 1]
    pub opacities: Tensor,
    /// Number of gaussians
    pub count: usize,
}

impl GaussianCloud {
    /// Create an empty cloud.
    ///
    /// # Errors
    /// Returns an error if tensor allocation fails.
    pub fn empty() -> Result<Self> {
        Ok(Self {
            positions: Tensor::zeros([0, 3], crate::tensor::DType::F32)?,
            covariances: Tensor::zeros([0, 6], crate::tensor::DType::F32)?,
            colors: Tensor::zeros([0, 3], crate::tensor::DType::F32)?,
            opacities: Tensor::zeros([0, 1], crate::tensor::DType::F32)?,
            count: 0,
        })
    }
}

/// Triangle mesh.
#[derive(Debug)]
pub struct TriangleMesh {
    /// Vertices [N, 3]
    pub vertices: Tensor,
    /// Faces [M, 3] (indices)
    pub faces: Tensor,
    /// Normals [N, 3]
    pub normals: Option<Tensor>,
    /// UV coordinates [N, 2]
    pub uvs: Option<Tensor>,
    /// Texture
    pub texture: Option<Tensor>,
}

/// Triplane features.
#[derive(Debug)]
pub struct TriplaneFeatures {
    /// XY plane [C, H, W]
    pub xy: Tensor,
    /// XZ plane [C, H, W]
    pub xz: Tensor,
    /// YZ plane [C, H, W]
    pub yz: Tensor,
}

/// 3D camera.
#[derive(Debug, Clone)]
pub struct Camera3D {
    /// Position in world space
    pub position: Vec3,
    /// Look-at target
    pub target: Vec3,
    /// Up vector
    pub up: Vec3,
    /// Field of view (degrees)
    pub fov: f32,
    /// Aspect ratio
    pub aspect: f32,
    /// Near clip plane
    pub near: f32,
    /// Far clip plane
    pub far: f32,
}

impl Camera3D {
    /// Create a camera orbiting around a point.
    pub fn orbit(center: Vec3, radius: f32, theta: f32, phi: f32) -> Self {
        let x = radius * phi.cos() * theta.cos();
        let y = radius * phi.sin();
        let z = radius * phi.cos() * theta.sin();

        Self {
            position: Vec3 {
                x: center.x + x,
                y: center.y + y,
                z: center.z + z,
            },
            target: center,
            up: Vec3::Y,
            fov: 45.0,
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
        }
    }

    /// Interpolate between two cameras.
    pub fn lerp(a: &Camera3D, b: &Camera3D, t: f32) -> Self {
        Self {
            position: a.position.lerp(&b.position, t),
            target: a.target.lerp(&b.target, t),
            up: a.up.lerp(&b.up, t).normalize(),
            fov: a.fov + (b.fov - a.fov) * t,
            aspect: a.aspect + (b.aspect - a.aspect) * t,
            near: a.near + (b.near - a.near) * t,
            far: a.far + (b.far - a.far) * t,
        }
    }

    /// Compute view matrix.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at(&self.position, &self.target, &self.up)
    }

    /// Compute projection matrix.
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective(self.fov.to_radians(), self.aspect, self.near, self.far)
    }
}

impl Default for Camera3D {
    fn default() -> Self {
        Self::orbit(Vec3::ZERO, 2.0, 0.0, 0.3)
    }
}

/// Camera path for animations.
#[derive(Debug, Clone)]
pub struct CameraPath {
    /// Keyframe cameras
    pub keyframes: Vec<(f32, Camera3D)>, // (time, camera)
}

impl CameraPath {
    /// Create an orbit path.
    pub fn orbit(center: Vec3, radius: f32, phi: f32, duration: f32) -> Self {
        let keyframes = (0..=4)
            .map(|i| {
                let t = i as f32 / 4.0;
                let theta = t * std::f32::consts::TAU;
                (t * duration, Camera3D::orbit(center, radius, theta, phi))
            })
            .collect();

        Self { keyframes }
    }

    /// Sample camera at time t.
    pub fn sample(&self, t: f32) -> Camera3D {
        if self.keyframes.is_empty() {
            return Camera3D::default();
        }

        // Find surrounding keyframes
        let mut prev = &self.keyframes[0];
        let mut next = &self.keyframes[0];

        for kf in &self.keyframes {
            if kf.0 <= t {
                prev = kf;
            }
            if kf.0 >= t {
                next = kf;
                break;
            }
        }

        if prev.0 == next.0 {
            return prev.1.clone();
        }

        let local_t = (t - prev.0) / (next.0 - prev.0);
        Camera3D::lerp(&prev.1, &next.1, local_t)
    }
}

/// Progressive 3D output.
#[derive(Debug)]
pub struct Progressive3D {
    /// Detail level
    pub level: DetailLevel,
    /// Current object state
    pub object: Object3D,
}

/// Detail levels for progressive generation.
#[derive(Debug, Clone, Copy)]
pub enum DetailLevel {
    /// Coarse preview
    Coarse,
    /// Medium detail
    Medium,
    /// Full detail
    Fine,
}

/// 3D vector.
#[derive(Debug, Clone, Copy, Default)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// Zero vector.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    /// Unit X vector.
    pub const X: Self = Self { x: 1.0, y: 0.0, z: 0.0 };
    /// Unit Y vector.
    pub const Y: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    /// Unit Z vector.
    pub const Z: Self = Self { x: 0.0, y: 0.0, z: 1.0 };

    /// Create a new vector.
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Linear interpolation between two vectors.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }

    /// Compute vector length.
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize to unit length.
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len == 0.0 {
            return *self;
        }
        Self {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    /// Cross product.
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Dot product.
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

/// 4x4 matrix.
#[derive(Debug, Clone, Copy)]
pub struct Mat4 {
    /// Matrix data in row-major order.
    pub data: [[f32; 4]; 4],
}

impl Mat4 {
    /// Create a look-at view matrix.
    pub fn look_at(eye: &Vec3, target: &Vec3, up: &Vec3) -> Self {
        let f = Vec3::new(target.x - eye.x, target.y - eye.y, target.z - eye.z).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(&f);

        Self {
            data: [
                [s.x, u.x, -f.x, 0.0],
                [s.y, u.y, -f.y, 0.0],
                [s.z, u.z, -f.z, 0.0],
                [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
            ],
        }
    }

    /// Create a perspective projection matrix.
    pub fn perspective(fov_radians: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_radians / 2.0).tan();
        let nf = 1.0 / (near - far);

        Self {
            data: [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (far + near) * nf, -1.0],
                [0.0, 0.0, 2.0 * far * near * nf, 0.0],
            ],
        }
    }
}

/// Edge function for triangle rasterization.
///
/// Returns a positive value if the point (px, py) is on the left side of the edge
/// from (x0, y0) to (x1, y1), negative if on the right, and zero if on the edge.
#[inline]
fn edge_function(x0: f32, y0: f32, x1: f32, y1: f32, px: f32, py: f32) -> f32 {
    (px - x0) * (y1 - y0) - (py - y0) * (x1 - x0)
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoundingBox {
    /// Minimum corner.
    pub min: Vec3,
    /// Maximum corner.
    pub max: Vec3,
}

impl BoundingBox {
    /// Compute the center point.
    pub fn center(&self) -> Vec3 {
        Vec3 {
            x: (self.min.x + self.max.x) / 2.0,
            y: (self.min.y + self.max.y) / 2.0,
            z: (self.min.z + self.max.z) / 2.0,
        }
    }

    /// Compute the size along each axis.
    pub fn size(&self) -> Vec3 {
        Vec3 {
            x: self.max.x - self.min.x,
            y: self.max.y - self.min.y,
            z: self.max.z - self.min.z,
        }
    }
}
