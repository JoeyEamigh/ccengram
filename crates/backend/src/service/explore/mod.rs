//! Explore service layer.
//!
//! This module provides the business logic for unified exploration across
//! code, memories, and documents.
//!
//! ## Design Principles
//!
//! - **Parallel search**: Uses `tokio::join!` for concurrent cross-domain search
//! - **Static methods**: Services are stateless; all dependencies passed as parameters
//! - **Service errors**: Operations return `Result<T, ServiceError>` for clean error handling
//!
//! ## Available Operations
//!
//! - [`search`] - Unified search across code, memories, and documents
//! - [`get_context`] - Get comprehensive context for an explore result

pub mod context;
mod search;
mod types;
mod util;

pub use context::get_context;
pub use search::search;
pub use types::*;
