use crate::Method;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request<P = serde_json::Value> {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub id: Option<u64>,
  pub method: Method,
  #[serde(default)]
  pub params: P,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response<R = serde_json::Value> {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub id: Option<u64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub result: Option<R>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub error: Option<RpcError>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub progress: Option<IndexProgress>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
  pub code: i32,
  pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexProgress {
  pub phase: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub total_files: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub processed_files: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub chunks_created: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub current_file: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub bytes_processed: Option<u64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub total_bytes: Option<u64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub message: Option<String>,
}

impl<R: Serialize> Response<R> {
  pub fn success(id: Option<u64>, result: R) -> Self {
    Self {
      id,
      result: Some(result),
      error: None,
      progress: None,
    }
  }
}

impl Response<()> {
  pub fn error(id: Option<u64>, code: i32, message: &str) -> Self {
    Self {
      id,
      result: None,
      error: Some(RpcError {
        code,
        message: message.to_string(),
      }),
      progress: None,
    }
  }

  pub fn progress_update(id: Option<u64>, progress: IndexProgress) -> Self {
    Self {
      id,
      result: None,
      error: None,
      progress: Some(progress),
    }
  }
}

impl IndexProgress {
  pub fn scanning(scanned: u32, current_file: Option<String>) -> Self {
    Self {
      phase: "scanning".to_string(),
      total_files: None,
      processed_files: Some(scanned),
      chunks_created: None,
      current_file,
      bytes_processed: None,
      total_bytes: None,
      message: Some(format!("Scanning... {} files found", scanned)),
    }
  }

  pub fn indexing(
    processed: u32,
    total: u32,
    chunks: u32,
    current_file: Option<String>,
    bytes_processed: u64,
    total_bytes: u64,
  ) -> Self {
    let percent = if total > 0 { (processed * 100) / total } else { 0 };
    Self {
      phase: "indexing".to_string(),
      total_files: Some(total),
      processed_files: Some(processed),
      chunks_created: Some(chunks),
      current_file,
      bytes_processed: Some(bytes_processed),
      total_bytes: Some(total_bytes),
      message: Some(format!("Indexing... {}% ({}/{})", percent, processed, total)),
    }
  }

  pub fn complete(files: u32, chunks: u32) -> Self {
    Self {
      phase: "complete".to_string(),
      total_files: Some(files),
      processed_files: Some(files),
      chunks_created: Some(chunks),
      current_file: None,
      bytes_processed: None,
      total_bytes: None,
      message: Some(format!("Complete: {} files, {} chunks", files, chunks)),
    }
  }
}
