//! Error types for the parser crate

use thiserror::Error;

/// Errors that can occur during parsing
#[derive(Error, Debug)]
pub enum ParseError {
  #[error("Unsupported language: {0:?}")]
  UnsupportedLanguage(engram_core::Language),

  #[error("Failed to parse code: {0}")]
  ParseFailed(String),

  #[error("Query compilation failed: {0}")]
  QueryFailed(String),
}
