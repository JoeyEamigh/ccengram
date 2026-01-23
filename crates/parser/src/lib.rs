//! Tree-sitter based code parsing for CCEngram
//!
//! This crate provides accurate extraction of:
//! - Import statements
//! - Function/method calls
//! - Symbol definitions
//!
//! # Example
//! ```ignore
//! use ccengram_parser::TreeSitterParser;
//! use engram_core::Language;
//!
//! let mut parser = TreeSitterParser::new();
//! let imports = parser.extract_imports(code, Language::Rust);
//! let calls = parser.extract_calls(code, Language::Rust);
//! ```

mod error;
mod parser;
mod queries;

pub use error::ParseError;
pub use parser::{Definition, DefinitionKind, TreeSitterParser};

// Re-export for convenience
pub use engram_core::Language;
