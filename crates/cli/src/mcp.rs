//! MCP (Model Context Protocol) server for Claude Code integration

use anyhow::{Context, Result};
use cli::to_daemon_request;
use daemon::connect_or_start;
use ipc::{Method, Request};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

#[derive(Debug, Deserialize)]
struct McpRequest {
  #[serde(rename = "jsonrpc")]
  _jsonrpc: String,
  id: Option<serde_json::Value>,
  method: String,
  #[serde(default)]
  params: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct McpResponse {
  jsonrpc: &'static str,
  #[serde(skip_serializing_if = "Option::is_none")]
  id: Option<serde_json::Value>,
  #[serde(skip_serializing_if = "Option::is_none")]
  result: Option<serde_json::Value>,
  #[serde(skip_serializing_if = "Option::is_none")]
  error: Option<McpError>,
}

#[derive(Debug, Serialize)]
struct McpError {
  code: i32,
  message: String,
}

// MCP protocol response structures
#[derive(Serialize)]
struct InitializeResult {
  #[serde(rename = "protocolVersion")]
  protocol_version: &'static str,
  capabilities: McpCapabilities,
  #[serde(rename = "serverInfo")]
  server_info: McpServerInfo,
}

#[derive(Serialize)]
struct McpCapabilities {
  tools: McpToolsCapability,
}

#[derive(Serialize)]
struct McpToolsCapability {}

#[derive(Serialize)]
struct McpServerInfo {
  name: &'static str,
  version: &'static str,
}

#[derive(Serialize)]
struct ToolsListResult {
  tools: serde_json::Value,
}

#[derive(Serialize)]
struct McpContent {
  #[serde(rename = "type")]
  content_type: &'static str,
  text: String,
}

#[derive(Serialize)]
struct McpToolResult {
  content: Vec<McpContent>,
  #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
  is_error: Option<bool>,
}

fn mcp_success(id: Option<serde_json::Value>, result: serde_json::Value) -> McpResponse {
  McpResponse {
    jsonrpc: "2.0",
    id,
    result: Some(result),
    error: None,
  }
}

fn mcp_error(id: Option<serde_json::Value>, code: i32, message: &str) -> McpResponse {
  McpResponse {
    jsonrpc: "2.0",
    id,
    result: None,
    error: Some(McpError {
      code,
      message: message.to_string(),
    }),
  }
}

/// MCP stdio server - implements the Model Context Protocol for Claude Code
pub async fn cmd_mcp() -> Result<()> {
  // Tool definitions are loaded from cli::tools and filtered based on config

  // Use async IO for proper non-blocking behavior with MCP
  let stdin = tokio::io::stdin();
  let mut stdout = tokio::io::stdout();
  let reader = tokio::io::BufReader::new(stdin);
  let mut lines = reader.lines();

  // Process MCP requests
  while let Some(line) = lines.next_line().await.context("Failed to read line from stdin")? {
    if line.trim().is_empty() {
      continue;
    }

    let mcp_request: McpRequest = match serde_json::from_str(&line) {
      Ok(r) => r,
      Err(e) => {
        let response = mcp_error(None, -32700, &format!("Parse error: {}", e));
        let out = serde_json::to_string(&response)? + "\n";
        stdout.write_all(out.as_bytes()).await?;
        stdout.flush().await?;
        continue;
      }
    };

    let response = match mcp_request.method.as_str() {
      // MCP protocol methods
      "initialize" => mcp_success(
        mcp_request.id,
        serde_json::to_value(InitializeResult {
          protocol_version: "2024-11-05",
          capabilities: McpCapabilities {
            tools: McpToolsCapability {},
          },
          server_info: McpServerInfo {
            name: "ccengram",
            version: env!("CARGO_PKG_VERSION"),
          },
        })
        .unwrap_or_default(),
      ),
      "notifications/initialized" => {
        // No response needed for notification
        continue;
      }
      "tools/list" => mcp_success(
        mcp_request.id,
        serde_json::to_value(ToolsListResult {
          tools: cli::get_tool_definitions_for_cwd(),
        })
        .unwrap_or_default(),
      ),
      "tools/call" => {
        // Extract tool name and arguments
        let tool_name = mcp_request.params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let arguments = mcp_request
          .params
          .get("arguments")
          .cloned()
          .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

        // Add cwd to arguments for project context
        let mut args = arguments;
        if let Some(obj) = args.as_object_mut()
          && !obj.contains_key("cwd")
          && let Ok(cwd) = std::env::current_dir()
        {
          obj.insert(
            "cwd".to_string(),
            serde_json::Value::String(cwd.to_string_lossy().to_string()),
          );
        }

        // Parse the tool name into a Method enum via serde deserialization
        let method: Method = match serde_json::from_value(serde_json::Value::String(tool_name.to_string())) {
          Ok(m) => m,
          Err(_) => {
            // Return error for unknown tool
            let resp = mcp_error(mcp_request.id, -32601, &format!("Unknown tool: {}", tool_name));
            let out = serde_json::to_string(&resp)? + "\n";
            stdout.write_all(out.as_bytes()).await?;
            stdout.flush().await?;
            continue;
          }
        };

        // Connect to daemon (auto-starts if not running)
        match connect_or_start().await {
          Ok(mut client) => {
            let request = Request {
              id: Some(1),
              method,
              params: args,
            };

            match client.request(to_daemon_request(request)).await {
              Ok(daemon_response) => {
                if let Some(err) = daemon_response.error {
                  // Return error as text content (MCP style)
                  mcp_success(
                    mcp_request.id,
                    serde_json::to_value(McpToolResult {
                      content: vec![McpContent {
                        content_type: "text",
                        text: format!("Error: {}", err.message),
                      }],
                      is_error: Some(true),
                    })
                    .unwrap_or_default(),
                  )
                } else if let Some(result) = daemon_response.result {
                  // Format result as text content
                  let text = serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string());
                  mcp_success(
                    mcp_request.id,
                    serde_json::to_value(McpToolResult {
                      content: vec![McpContent {
                        content_type: "text",
                        text,
                      }],
                      is_error: None,
                    })
                    .unwrap_or_default(),
                  )
                } else {
                  mcp_success(
                    mcp_request.id,
                    serde_json::to_value(McpToolResult {
                      content: vec![McpContent {
                        content_type: "text",
                        text: "Success".to_string(),
                      }],
                      is_error: None,
                    })
                    .unwrap_or_default(),
                  )
                }
              }
              Err(e) => mcp_error(mcp_request.id, -32000, &format!("Daemon error: {}", e)),
            }
          }
          Err(e) => mcp_error(mcp_request.id, -32000, &format!("Failed to start daemon: {}", e)),
        }
      }
      // Unknown method
      _ => mcp_error(
        mcp_request.id,
        -32601,
        &format!("Method not found: {}", mcp_request.method),
      ),
    };

    let out = serde_json::to_string(&response)? + "\n";
    stdout.write_all(out.as_bytes()).await?;
    stdout.flush().await?;
  }

  Ok(())
}
