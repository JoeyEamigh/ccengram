//! Shared utilities for the handler system.
//!
//! This module provides reusable utilities that eliminate common boilerplate
//! patterns across handlers:
//!
//! - `error` - Unified error types for service operations
//! - `resolve` - Generic ID/prefix resolution for all entity types
//! - `filter` - SQL-injection-safe filter builder
//! - `search` - Vector search with text fallback pattern
//! - `format` - Response formatting for human-readable output

mod error;
mod filter;
pub mod fusion;
mod resolve;

pub use error::ServiceError;
pub use filter::FilterBuilder;
pub use resolve::Resolver;
