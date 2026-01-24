use thiserror::Error;

#[derive(Debug, Error)]
pub enum IpcError {
  #[error("Serialization error: {0}")]
  Serialization(#[from] serde_json::Error),

  #[error("RPC error {code}: {message}")]
  Rpc { code: i32, message: String },

  #[error("No result in response")]
  NoResult,

  #[error("Connection error: {0}")]
  Connection(String),
}
