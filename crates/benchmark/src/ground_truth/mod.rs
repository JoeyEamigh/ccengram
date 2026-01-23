//! Ground truth for benchmark validation.
//!
//! Three approaches:
//! 1. Call graph (automatic) - verify reachability, score navigation hints
//! 2. Noise patterns (automatic) - detect test code, internal symbols
//! 3. Annotations (manual) - per-scenario critical files/symbols

mod annotations;
mod call_graph;
mod patterns;

pub use annotations::{load_scenario_annotations, Annotations, ExplorationPath};
pub use call_graph::CallGraph;
pub use patterns::NoisePatterns;
