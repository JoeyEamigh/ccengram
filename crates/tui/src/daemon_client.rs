use anyhow::{Context, Result};
use daemon::{Client, Response};
use serde_json::Value;
use std::path::PathBuf;
use tracing::debug;

/// Wrapper around daemon client for TUI-specific operations
pub struct DaemonClient {
  client: Client,
  project_path: PathBuf,
}

impl DaemonClient {
  pub async fn connect(project_path: PathBuf) -> Result<Self> {
    let client = Client::connect().await.context("Failed to connect to daemon")?;

    Ok(Self { client, project_path })
  }

  /// Get the cwd parameter for requests
  fn cwd(&self) -> String {
    self.project_path.to_string_lossy().to_string()
  }

  /// Make a request to the daemon
  async fn call(&mut self, method: &str, mut params: Value) -> Result<Response> {
    // Add cwd to params if it's an object
    if let Some(obj) = params.as_object_mut()
      && !obj.contains_key("cwd")
    {
      obj.insert("cwd".to_string(), serde_json::json!(self.cwd()));
    }

    debug!("Calling daemon method: {}", method);
    self
      .client
      .call(method, params)
      .await
      .map_err(|e| anyhow::anyhow!("{}", e))
  }

  /// Ping the daemon to check if it's responsive
  pub async fn ping(&mut self) -> Result<bool> {
    match self.call("ping", serde_json::json!({})).await {
      Ok(response) => Ok(response.error.is_none()),
      Err(_) => Ok(false),
    }
  }

  /// Get daemon status
  pub async fn status(&mut self) -> Result<Value> {
    let response = self.call("status", serde_json::json!({})).await?;
    response.result.ok_or_else(|| {
      let msg = response
        .error
        .map(|e| e.message)
        .unwrap_or_else(|| "Unknown error".to_string());
      anyhow::anyhow!("{}", msg)
    })
  }

  /// Get project statistics
  pub async fn project_stats(&mut self) -> Result<Value> {
    let response = self.call("project_stats", serde_json::json!({})).await?;
    response.result.ok_or_else(|| {
      let msg = response
        .error
        .map(|e| e.message)
        .unwrap_or_else(|| "Unknown error".to_string());
      anyhow::anyhow!("{}", msg)
    })
  }

  /// Get health check
  pub async fn health_check(&mut self) -> Result<Value> {
    let response = self.call("health_check", serde_json::json!({})).await?;
    response.result.ok_or_else(|| {
      let msg = response
        .error
        .map(|e| e.message)
        .unwrap_or_else(|| "Unknown error".to_string());
      anyhow::anyhow!("{}", msg)
    })
  }

  /// Search memories
  pub async fn memory_search(&mut self, query: &str, limit: usize) -> Result<Vec<Value>> {
    let response = self
      .call(
        "memory_search",
        serde_json::json!({
            "query": query,
            "limit": limit,
        }),
      )
      .await?;

    if let Some(err) = response.error {
      return Err(anyhow::anyhow!("{}", err.message));
    }

    response
      .result
      .and_then(|v| v.as_array().cloned())
      .ok_or_else(|| anyhow::anyhow!("Invalid response format"))
  }

  /// List memories
  pub async fn memory_list(&mut self, limit: usize, offset: usize) -> Result<Vec<Value>> {
    let response = self
      .call(
        "memory_list",
        serde_json::json!({
            "limit": limit,
            "offset": offset,
        }),
      )
      .await?;

    if let Some(err) = response.error {
      return Err(anyhow::anyhow!("{}", err.message));
    }

    response
      .result
      .and_then(|v| v.as_array().cloned())
      .ok_or_else(|| anyhow::anyhow!("Invalid response format"))
  }

  /// Get a single memory by ID
  pub async fn memory_get(&mut self, memory_id: &str) -> Result<Value> {
    let response = self
      .call(
        "memory_get",
        serde_json::json!({
            "memory_id": memory_id,
            "include_related": true,
        }),
      )
      .await?;

    if let Some(err) = response.error {
      return Err(anyhow::anyhow!("{}", err.message));
    }

    response.result.ok_or_else(|| anyhow::anyhow!("Memory not found"))
  }

  /// Reinforce a memory
  pub async fn memory_reinforce(&mut self, memory_id: &str) -> Result<()> {
    let response = self
      .call(
        "memory_reinforce",
        serde_json::json!({
            "memory_id": memory_id,
            "amount": 0.2,
        }),
      )
      .await?;

    if let Some(err) = response.error {
      return Err(anyhow::anyhow!("{}", err.message));
    }

    Ok(())
  }

  /// Deemphasize a memory
  pub async fn memory_deemphasize(&mut self, memory_id: &str) -> Result<()> {
    let response = self
      .call(
        "memory_deemphasize",
        serde_json::json!({
            "memory_id": memory_id,
            "amount": 0.2,
        }),
      )
      .await?;

    if let Some(err) = response.error {
      return Err(anyhow::anyhow!("{}", err.message));
    }

    Ok(())
  }

  /// Search code
  pub async fn code_search(&mut self, query: &str, limit: usize) -> Result<Vec<Value>> {
    let response = self
      .call(
        "code_search",
        serde_json::json!({
            "query": query,
            "limit": limit,
        }),
      )
      .await?;

    if let Some(err) = response.error {
      return Err(anyhow::anyhow!("{}", err.message));
    }

    response
      .result
      .and_then(|v| v.as_array().cloned())
      .ok_or_else(|| anyhow::anyhow!("Invalid response format"))
  }

  /// List code chunks
  pub async fn code_list(&mut self) -> Result<Vec<Value>> {
    let response = self.call("code_list", serde_json::json!({})).await?;

    if let Some(err) = response.error {
      return Err(anyhow::anyhow!("{}", err.message));
    }

    response
      .result
      .and_then(|v| v.as_array().cloned())
      .ok_or_else(|| anyhow::anyhow!("Invalid response format"))
  }

  /// Get code stats
  pub async fn code_stats(&mut self) -> Result<Value> {
    let response = self.call("code_stats", serde_json::json!({})).await?;

    if let Some(err) = response.error {
      return Err(anyhow::anyhow!("{}", err.message));
    }

    response
      .result
      .ok_or_else(|| anyhow::anyhow!("Invalid response format"))
  }

  /// Search documents
  pub async fn docs_search(&mut self, query: &str, limit: usize) -> Result<Vec<Value>> {
    let response = self
      .call(
        "docs_search",
        serde_json::json!({
            "query": query,
            "limit": limit,
        }),
      )
      .await?;

    if let Some(err) = response.error {
      return Err(anyhow::anyhow!("{}", err.message));
    }

    response
      .result
      .and_then(|v| v.as_array().cloned())
      .ok_or_else(|| anyhow::anyhow!("Invalid response format"))
  }

  /// List entities
  pub async fn entity_list(&mut self, limit: usize) -> Result<Vec<Value>> {
    let response = self
      .call(
        "entity_list",
        serde_json::json!({
            "limit": limit,
        }),
      )
      .await?;

    if let Some(err) = response.error {
      return Err(anyhow::anyhow!("{}", err.message));
    }

    response
      .result
      .and_then(|v| v.as_array().cloned())
      .ok_or_else(|| anyhow::anyhow!("Invalid response format"))
  }

  /// Get top entities
  pub async fn entity_top(&mut self, limit: usize) -> Result<Vec<Value>> {
    let response = self
      .call(
        "entity_top",
        serde_json::json!({
            "limit": limit,
        }),
      )
      .await?;

    if let Some(err) = response.error {
      return Err(anyhow::anyhow!("{}", err.message));
    }

    response
      .result
      .and_then(|v| v.as_array().cloned())
      .ok_or_else(|| anyhow::anyhow!("Invalid response format"))
  }

  /// Get entity details
  pub async fn entity_get(&mut self, entity_id: &str) -> Result<Value> {
    let response = self
      .call(
        "entity_get",
        serde_json::json!({
            "entity_id": entity_id,
        }),
      )
      .await?;

    if let Some(err) = response.error {
      return Err(anyhow::anyhow!("{}", err.message));
    }

    response.result.ok_or_else(|| anyhow::anyhow!("Entity not found"))
  }

  /// List relationships for a memory
  pub async fn relationship_list(&mut self, memory_id: &str) -> Result<Vec<Value>> {
    let response = self
      .call(
        "relationship_list",
        serde_json::json!({
            "memory_id": memory_id,
        }),
      )
      .await?;

    if let Some(err) = response.error {
      return Err(anyhow::anyhow!("{}", err.message));
    }

    response
      .result
      .and_then(|v| v.as_array().cloned())
      .ok_or_else(|| anyhow::anyhow!("Invalid response format"))
  }

  /// Get memory timeline
  pub async fn memory_timeline(&mut self, limit: usize) -> Result<Vec<Value>> {
    let response = self
      .call(
        "memory_timeline",
        serde_json::json!({
            "limit": limit,
        }),
      )
      .await?;

    if let Some(err) = response.error {
      return Err(anyhow::anyhow!("{}", err.message));
    }

    response
      .result
      .and_then(|v| v.as_array().cloned())
      .ok_or_else(|| anyhow::anyhow!("Invalid response format"))
  }

  /// Shutdown the daemon
  pub async fn shutdown(&mut self) -> Result<()> {
    debug!("Sending shutdown request to daemon");
    let response = self.call("shutdown", serde_json::json!({})).await?;

    if let Some(err) = response.error {
      return Err(anyhow::anyhow!("{}", err.message));
    }

    Ok(())
  }
}
