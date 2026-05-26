//! Graph k-coloring: assign one of k colors to each vertex so no edge
//! connects same-colored vertices.
//!
//! Energy = number of monochromatic edges; zero ⇒ a proper coloring.
//! Solved by backtracking over vertices with forward-checking.
//!
//! Input grammar (`color` query payload):
//!
//!   "<k> : <edge> <edge> ..."   where edge ::= "<u>-<v>"  (0-indexed vertices)
//!
//! e.g. `3 : 0-1 1-2 0-2` is K3 (a triangle) with 3 colors — solvable.
//!      `2 : 0-1 1-2 0-2` is K3 with 2 colors — UNSAT (odd cycle).

use crate::EnergyFunction;

#[derive(Debug, Clone, PartialEq)]
pub enum ColoringError {
    BadFormat(String),
    BadEdge(String),
    ZeroColors,
}

impl std::fmt::Display for ColoringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadFormat(s) => write!(f, "coloring: expected '<k> : <u>-<v> ...', got {:?}", s),
            Self::BadEdge(s) => write!(f, "coloring: bad edge {:?}", s),
            Self::ZeroColors => write!(f, "coloring: k must be ≥ 1"),
        }
    }
}

impl std::error::Error for ColoringError {}

#[derive(Debug, Clone)]
pub struct GraphColoring {
    pub k: usize,
    pub n_vertices: usize,
    /// adjacency[v] = sorted neighbours of v.
    pub adjacency: Vec<Vec<usize>>,
}

/// `colors[v]` ∈ 0..k.
pub type Coloring = Vec<usize>;

impl GraphColoring {
    pub fn parse(input: &str) -> Result<Self, ColoringError> {
        let (k_str, edges_str) = input
            .split_once(':')
            .ok_or_else(|| ColoringError::BadFormat(input.to_string()))?;
        let k: usize = k_str
            .trim()
            .parse()
            .map_err(|_| ColoringError::BadFormat(input.to_string()))?;
        if k == 0 {
            return Err(ColoringError::ZeroColors);
        }
        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut max_v = 0usize;
        for tok in edges_str.split_whitespace() {
            let (u, v) = tok
                .split_once('-')
                .ok_or_else(|| ColoringError::BadEdge(tok.to_string()))?;
            let u: usize = u
                .parse()
                .map_err(|_| ColoringError::BadEdge(tok.to_string()))?;
            let v: usize = v
                .parse()
                .map_err(|_| ColoringError::BadEdge(tok.to_string()))?;
            max_v = max_v.max(u).max(v);
            edges.push((u, v));
        }
        let n_vertices = if edges.is_empty() { 0 } else { max_v + 1 };
        let mut adjacency = vec![Vec::new(); n_vertices];
        for (u, v) in edges {
            adjacency[u].push(v);
            adjacency[v].push(u);
        }
        for adj in &mut adjacency {
            adj.sort_unstable();
            adj.dedup();
        }
        Ok(GraphColoring { k, n_vertices, adjacency })
    }

    fn safe(&self, colors: &[usize], v: usize, c: usize) -> bool {
        self.adjacency[v]
            .iter()
            .all(|&nb| colors[nb] == usize::MAX || colors[nb] != c)
    }

    pub fn solve(&self, max_steps: usize) -> Option<Coloring> {
        if self.n_vertices == 0 {
            return Some(vec![]);
        }
        let mut colors = vec![usize::MAX; self.n_vertices];
        let mut steps = 0usize;
        if self.assign(&mut colors, 0, &mut steps, max_steps) {
            Some(colors)
        } else {
            None
        }
    }

    fn assign(
        &self,
        colors: &mut [usize],
        v: usize,
        steps: &mut usize,
        max_steps: usize,
    ) -> bool {
        if v == self.n_vertices {
            return true;
        }
        for c in 0..self.k {
            *steps += 1;
            if *steps > max_steps {
                return false;
            }
            if self.safe(colors, v, c) {
                colors[v] = c;
                if self.assign(colors, v + 1, steps, max_steps) {
                    return true;
                }
                colors[v] = usize::MAX;
            }
        }
        false
    }

    pub fn render(&self, coloring: &Coloring) -> String {
        let mut s = format!(
            "graph {}-coloring ({} vertices):\n",
            self.k, self.n_vertices
        );
        for (v, &c) in coloring.iter().enumerate() {
            s.push_str(&format!("  v{} → color {}\n", v, c));
        }
        s
    }
}

/// Energy = monochromatic edges (0 ⇒ proper coloring).
impl EnergyFunction for GraphColoring {
    type State = Coloring;
    fn energy(&self, state: &Self::State) -> f64 {
        let mut bad = 0usize;
        for (u, neighbours) in self.adjacency.iter().enumerate() {
            for &w in neighbours {
                if w > u && state[u] == state[w] {
                    bad += 1;
                }
            }
        }
        bad as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_triangle() {
        let g = GraphColoring::parse("3 : 0-1 1-2 0-2").unwrap();
        assert_eq!(g.k, 3);
        assert_eq!(g.n_vertices, 3);
        assert_eq!(g.adjacency[0], vec![1, 2]);
    }

    #[test]
    fn parse_rejects_bad_format() {
        assert!(matches!(
            GraphColoring::parse("no colon here"),
            Err(ColoringError::BadFormat(_))
        ));
        assert!(matches!(
            GraphColoring::parse("0 : 0-1"),
            Err(ColoringError::ZeroColors)
        ));
    }

    #[test]
    fn triangle_3colorable() {
        let g = GraphColoring::parse("3 : 0-1 1-2 0-2").unwrap();
        let sol = g.solve(100_000).unwrap();
        assert_eq!(g.energy(&sol), 0.0);
        // All three must differ in a triangle.
        assert!(sol[0] != sol[1] && sol[1] != sol[2] && sol[0] != sol[2]);
    }

    #[test]
    fn triangle_not_2colorable() {
        // Odd cycle K3 needs 3 colors.
        let g = GraphColoring::parse("2 : 0-1 1-2 0-2").unwrap();
        assert!(g.solve(100_000).is_none());
    }

    #[test]
    fn bipartite_is_2colorable() {
        // 4-cycle 0-1-2-3-0 is bipartite → 2-colorable.
        let g = GraphColoring::parse("2 : 0-1 1-2 2-3 3-0").unwrap();
        let sol = g.solve(100_000).unwrap();
        assert_eq!(g.energy(&sol), 0.0);
    }

    #[test]
    fn solution_is_deterministic() {
        let g = GraphColoring::parse("3 : 0-1 1-2 0-2 2-3 3-0").unwrap();
        assert_eq!(g.solve(100_000), g.solve(100_000));
    }

    #[test]
    fn energy_counts_monochromatic_edges() {
        let g = GraphColoring::parse("3 : 0-1 1-2 0-2").unwrap();
        // All same color → 3 bad edges.
        assert_eq!(g.energy(&vec![0, 0, 0]), 3.0);
    }
}
