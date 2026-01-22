use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
  #[error("Not found: {entity} {id}")]
  NotFound { entity: &'static str, id: String },

  #[error("Validation: {0}")]
  Validation(String),

  #[error("Database: {0}")]
  Database(String),

  #[error("Embedding: {0}")]
  Embedding(String),

  #[error("Index: {0}")]
  Index(String),

  #[error("IO: {0}")]
  Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
