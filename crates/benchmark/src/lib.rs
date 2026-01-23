//! E2E Benchmark Harness for CCEngram Explore/Context Tools
//!
//! This crate provides comprehensive benchmarking for testing the exploration
//! capabilities of CCEngram's `explore` and `context` tools against large
//! real-world codebases (Zed, VSCode).
//!
//! ## Key Concepts
//!
//! - **Scenarios**: TOML-defined multi-step exploration tasks
//! - **Metrics**: Performance (latency, throughput) and accuracy (recall, noise ratio)
//! - **Ground Truth**: Call graph analysis, noise patterns, optional annotations
//! - **Reports**: JSON (machine-readable) and Markdown (human-readable)

pub mod ground_truth;
pub mod metrics;
pub mod repos;
pub mod reports;
pub mod scenarios;
pub mod session;

pub use ground_truth::{Annotations, CallGraph, NoisePatterns};
pub use metrics::{AccuracyMetrics, PerformanceMetrics};
pub use repos::{RepoConfig, RepoRegistry};
pub use reports::{BenchmarkReport, ComparisonReport};
pub use scenarios::{Scenario, ScenarioRunner};
pub use session::ExplorationSession;

use thiserror::Error;

/// Benchmark-specific errors
#[derive(Debug, Error)]
pub enum BenchmarkError {
    #[error("Repository error: {0}")]
    Repo(String),

    #[error("Scenario error: {0}")]
    Scenario(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Database error: {0}")]
    Db(#[from] db::DbError),
}

pub type Result<T> = std::result::Result<T, BenchmarkError>;
