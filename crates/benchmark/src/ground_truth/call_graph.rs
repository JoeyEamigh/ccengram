//! Call graph analysis using petgraph.
//!
//! Builds a graph from indexed calls and symbols to:
//! - Verify reachability between symbols
//! - Score navigation hints (are callers/callees in the graph?)

use petgraph::algo::dijkstra;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

/// Call graph for analyzing symbol relationships.
#[derive(Debug)]
pub struct CallGraph {
    graph: DiGraph<String, ()>,
    symbol_to_node: HashMap<String, NodeIndex>,
}

impl CallGraph {
    /// Create a new empty call graph.
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            symbol_to_node: HashMap::new(),
        }
    }

    /// Add a symbol to the graph.
    pub fn add_symbol(&mut self, symbol: &str) -> NodeIndex {
        if let Some(&idx) = self.symbol_to_node.get(symbol) {
            idx
        } else {
            let idx = self.graph.add_node(symbol.to_string());
            self.symbol_to_node.insert(symbol.to_string(), idx);
            idx
        }
    }

    /// Add a call edge (caller -> callee).
    pub fn add_call(&mut self, caller: &str, callee: &str) {
        let caller_idx = self.add_symbol(caller);
        let callee_idx = self.add_symbol(callee);
        self.graph.add_edge(caller_idx, callee_idx, ());
    }

    /// Build from a list of (caller, callee) pairs.
    pub fn from_calls(calls: impl IntoIterator<Item = (String, String)>) -> Self {
        let mut graph = Self::new();
        for (caller, callee) in calls {
            graph.add_call(&caller, &callee);
        }
        graph
    }

    /// Check if there's a path from source to target.
    pub fn is_reachable(&self, source: &str, target: &str) -> bool {
        let Some(&source_idx) = self.symbol_to_node.get(source) else {
            return false;
        };
        let Some(&target_idx) = self.symbol_to_node.get(target) else {
            return false;
        };

        // Use Dijkstra with unit weights to find shortest path
        let distances = dijkstra(&self.graph, source_idx, Some(target_idx), |_| 1);
        distances.contains_key(&target_idx)
    }

    /// Get the shortest path length from source to target (number of hops).
    pub fn path_length(&self, source: &str, target: &str) -> Option<usize> {
        let source_idx = *self.symbol_to_node.get(source)?;
        let target_idx = *self.symbol_to_node.get(target)?;

        let distances = dijkstra(&self.graph, source_idx, Some(target_idx), |_| 1usize);
        distances.get(&target_idx).copied()
    }

    /// Check if reachable within N hops.
    pub fn is_reachable_within(&self, source: &str, target: &str, max_hops: usize) -> bool {
        self.path_length(source, target)
            .is_some_and(|len| len <= max_hops)
    }

    /// Get all symbols that call the given symbol (direct callers).
    pub fn callers(&self, symbol: &str) -> Vec<String> {
        let Some(&idx) = self.symbol_to_node.get(symbol) else {
            return vec![];
        };

        self.graph
            .neighbors_directed(idx, petgraph::Direction::Incoming)
            .map(|n| self.graph[n].clone())
            .collect()
    }

    /// Get all symbols called by the given symbol (direct callees).
    pub fn callees(&self, symbol: &str) -> Vec<String> {
        let Some(&idx) = self.symbol_to_node.get(symbol) else {
            return vec![];
        };

        self.graph
            .neighbors_directed(idx, petgraph::Direction::Outgoing)
            .map(|n| self.graph[n].clone())
            .collect()
    }

    /// Check if a symbol exists in the graph.
    pub fn contains(&self, symbol: &str) -> bool {
        self.symbol_to_node.contains_key(symbol)
    }

    /// Get the number of symbols in the graph.
    pub fn symbol_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of call edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Score navigation hints by checking if they exist in the graph.
    /// Returns (found_count, total_count).
    pub fn score_hints(&self, hints: &[String]) -> (usize, usize) {
        let found = hints.iter().filter(|h| self.contains(h)).count();
        (found, hints.len())
    }

    /// Get all symbols in the graph.
    pub fn symbols(&self) -> Vec<String> {
        self.symbol_to_node.keys().cloned().collect()
    }
}

impl Default for CallGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> CallGraph {
        CallGraph::from_calls([
            ("main".to_string(), "run".to_string()),
            ("run".to_string(), "setup".to_string()),
            ("run".to_string(), "execute".to_string()),
            ("execute".to_string(), "Task::new".to_string()),
            ("setup".to_string(), "Config::load".to_string()),
        ])
    }

    #[test]
    fn test_add_symbol() {
        let mut graph = CallGraph::new();
        let idx1 = graph.add_symbol("foo");
        let idx2 = graph.add_symbol("foo");
        assert_eq!(idx1, idx2); // Same symbol returns same index
    }

    #[test]
    fn test_is_reachable() {
        let graph = sample_graph();
        assert!(graph.is_reachable("main", "Task::new"));
        assert!(graph.is_reachable("run", "Config::load"));
        assert!(!graph.is_reachable("Task::new", "main")); // Reverse direction
    }

    #[test]
    fn test_path_length() {
        let graph = sample_graph();
        assert_eq!(graph.path_length("main", "run"), Some(1));
        assert_eq!(graph.path_length("main", "Task::new"), Some(3));
        assert_eq!(graph.path_length("main", "nonexistent"), None);
    }

    #[test]
    fn test_is_reachable_within() {
        let graph = sample_graph();
        assert!(graph.is_reachable_within("main", "Task::new", 3));
        assert!(!graph.is_reachable_within("main", "Task::new", 2));
    }

    #[test]
    fn test_callers_callees() {
        let graph = sample_graph();

        let callers = graph.callers("execute");
        assert_eq!(callers, vec!["run"]);

        let callees = graph.callees("run");
        assert!(callees.contains(&"setup".to_string()));
        assert!(callees.contains(&"execute".to_string()));
    }

    #[test]
    fn test_score_hints() {
        let graph = sample_graph();
        let hints = vec![
            "main".to_string(),
            "run".to_string(),
            "nonexistent".to_string(),
        ];
        let (found, total) = graph.score_hints(&hints);
        assert_eq!(found, 2);
        assert_eq!(total, 3);
    }

    #[test]
    fn test_counts() {
        let graph = sample_graph();
        assert_eq!(graph.symbol_count(), 6);
        assert_eq!(graph.edge_count(), 5);
    }
}
