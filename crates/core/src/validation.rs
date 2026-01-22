//! Input validation utilities
//!
//! Provides centralized validation functions with consistent error messages.

use std::fmt;
use thiserror::Error;

/// A validation error with field information
#[derive(Debug, Clone, Error)]
pub struct ValidationError {
  pub field: String,
  pub message: String,
}

impl fmt::Display for ValidationError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}: {}", self.field, self.message)
  }
}

impl ValidationError {
  pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
    Self {
      field: field.into(),
      message: message.into(),
    }
  }

  /// Create error for missing required field
  pub fn missing(field: impl Into<String>) -> Self {
    let field = field.into();
    Self {
      message: format!("{} is required", field),
      field,
    }
  }

  /// Create error for invalid type
  pub fn invalid_type(field: impl Into<String>, expected: &str) -> Self {
    Self {
      field: field.into(),
      message: format!("expected {}", expected),
    }
  }

  /// Create error for out of range value
  pub fn out_of_range(field: impl Into<String>, min: impl fmt::Display, max: impl fmt::Display) -> Self {
    Self {
      field: field.into(),
      message: format!("must be between {} and {}", min, max),
    }
  }

  /// Create error for too short string
  pub fn too_short(field: impl Into<String>, min_len: usize) -> Self {
    Self {
      field: field.into(),
      message: format!("must be at least {} characters", min_len),
    }
  }

  /// Create error for too long string
  pub fn too_long(field: impl Into<String>, max_len: usize) -> Self {
    Self {
      field: field.into(),
      message: format!("must be at most {} characters", max_len),
    }
  }

  /// Create error for invalid enum value
  pub fn invalid_enum(field: impl Into<String>, valid_values: &[&str]) -> Self {
    Self {
      field: field.into(),
      message: format!("must be one of: {}", valid_values.join(", ")),
    }
  }
}

/// Result type for validation
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Validate a required string field
pub fn require_string(value: Option<&serde_json::Value>, field: &str) -> ValidationResult<String> {
  match value {
    Some(v) => v
      .as_str()
      .map(String::from)
      .ok_or_else(|| ValidationError::invalid_type(field, "string")),
    None => Err(ValidationError::missing(field)),
  }
}

/// Validate a required string field with minimum length
pub fn require_string_min(value: Option<&serde_json::Value>, field: &str, min_len: usize) -> ValidationResult<String> {
  let s = require_string(value, field)?;
  if s.len() < min_len {
    return Err(ValidationError::too_short(field, min_len));
  }
  Ok(s)
}

/// Validate a required string field with length constraints
pub fn require_string_range(
  value: Option<&serde_json::Value>,
  field: &str,
  min_len: usize,
  max_len: usize,
) -> ValidationResult<String> {
  let s = require_string(value, field)?;
  if s.len() < min_len {
    return Err(ValidationError::too_short(field, min_len));
  }
  if s.len() > max_len {
    return Err(ValidationError::too_long(field, max_len));
  }
  Ok(s)
}

/// Validate an optional string field
pub fn optional_string(value: Option<&serde_json::Value>, field: &str) -> ValidationResult<Option<String>> {
  match value {
    Some(v) if v.is_null() => Ok(None),
    Some(v) => v
      .as_str()
      .map(|s| Some(s.to_string()))
      .ok_or_else(|| ValidationError::invalid_type(field, "string")),
    None => Ok(None),
  }
}

/// Validate an optional string field with minimum length (if present)
pub fn optional_string_min(
  value: Option<&serde_json::Value>,
  field: &str,
  min_len: usize,
) -> ValidationResult<Option<String>> {
  match optional_string(value, field)? {
    Some(s) if s.len() < min_len => Err(ValidationError::too_short(field, min_len)),
    other => Ok(other),
  }
}

/// Validate a required integer field
pub fn require_i64(value: Option<&serde_json::Value>, field: &str) -> ValidationResult<i64> {
  match value {
    Some(v) => v
      .as_i64()
      .ok_or_else(|| ValidationError::invalid_type(field, "integer")),
    None => Err(ValidationError::missing(field)),
  }
}

/// Validate a required integer field with range constraints
pub fn require_i64_range(value: Option<&serde_json::Value>, field: &str, min: i64, max: i64) -> ValidationResult<i64> {
  let n = require_i64(value, field)?;
  if n < min || n > max {
    return Err(ValidationError::out_of_range(field, min, max));
  }
  Ok(n)
}

/// Validate an optional integer field
pub fn optional_i64(value: Option<&serde_json::Value>, field: &str) -> ValidationResult<Option<i64>> {
  match value {
    Some(v) if v.is_null() => Ok(None),
    Some(v) => v
      .as_i64()
      .map(Some)
      .ok_or_else(|| ValidationError::invalid_type(field, "integer")),
    None => Ok(None),
  }
}

/// Validate an optional integer field with range (if present)
pub fn optional_i64_range(
  value: Option<&serde_json::Value>,
  field: &str,
  min: i64,
  max: i64,
) -> ValidationResult<Option<i64>> {
  match optional_i64(value, field)? {
    Some(n) if n < min || n > max => Err(ValidationError::out_of_range(field, min, max)),
    other => Ok(other),
  }
}

/// Validate a required unsigned integer field
pub fn require_u64(value: Option<&serde_json::Value>, field: &str) -> ValidationResult<u64> {
  match value {
    Some(v) => v
      .as_u64()
      .ok_or_else(|| ValidationError::invalid_type(field, "non-negative integer")),
    None => Err(ValidationError::missing(field)),
  }
}

/// Validate an optional unsigned integer field
pub fn optional_u64(value: Option<&serde_json::Value>, field: &str) -> ValidationResult<Option<u64>> {
  match value {
    Some(v) if v.is_null() => Ok(None),
    Some(v) => v
      .as_u64()
      .map(Some)
      .ok_or_else(|| ValidationError::invalid_type(field, "non-negative integer")),
    None => Ok(None),
  }
}

/// Validate a required float field
pub fn require_f64(value: Option<&serde_json::Value>, field: &str) -> ValidationResult<f64> {
  match value {
    Some(v) => v.as_f64().ok_or_else(|| ValidationError::invalid_type(field, "number")),
    None => Err(ValidationError::missing(field)),
  }
}

/// Validate a required float field with range constraints
pub fn require_f64_range(value: Option<&serde_json::Value>, field: &str, min: f64, max: f64) -> ValidationResult<f64> {
  let n = require_f64(value, field)?;
  if n < min || n > max {
    return Err(ValidationError::out_of_range(field, min, max));
  }
  Ok(n)
}

/// Validate an optional float field
pub fn optional_f64(value: Option<&serde_json::Value>, field: &str) -> ValidationResult<Option<f64>> {
  match value {
    Some(v) if v.is_null() => Ok(None),
    Some(v) => v
      .as_f64()
      .map(Some)
      .ok_or_else(|| ValidationError::invalid_type(field, "number")),
    None => Ok(None),
  }
}

/// Validate an optional float field with range (if present)
pub fn optional_f64_range(
  value: Option<&serde_json::Value>,
  field: &str,
  min: f64,
  max: f64,
) -> ValidationResult<Option<f64>> {
  match optional_f64(value, field)? {
    Some(n) if n < min || n > max => Err(ValidationError::out_of_range(field, min, max)),
    other => Ok(other),
  }
}

/// Validate a required boolean field
pub fn require_bool(value: Option<&serde_json::Value>, field: &str) -> ValidationResult<bool> {
  match value {
    Some(v) => v
      .as_bool()
      .ok_or_else(|| ValidationError::invalid_type(field, "boolean")),
    None => Err(ValidationError::missing(field)),
  }
}

/// Validate an optional boolean field
pub fn optional_bool(value: Option<&serde_json::Value>, field: &str) -> ValidationResult<Option<bool>> {
  match value {
    Some(v) if v.is_null() => Ok(None),
    Some(v) => v
      .as_bool()
      .map(Some)
      .ok_or_else(|| ValidationError::invalid_type(field, "boolean")),
    None => Ok(None),
  }
}

/// Validate a required array field
pub fn require_array(value: Option<&serde_json::Value>, field: &str) -> ValidationResult<Vec<serde_json::Value>> {
  match value {
    Some(v) => v
      .as_array()
      .cloned()
      .ok_or_else(|| ValidationError::invalid_type(field, "array")),
    None => Err(ValidationError::missing(field)),
  }
}

/// Validate an optional array field
pub fn optional_array(
  value: Option<&serde_json::Value>,
  field: &str,
) -> ValidationResult<Option<Vec<serde_json::Value>>> {
  match value {
    Some(v) if v.is_null() => Ok(None),
    Some(v) => v
      .as_array()
      .map(|a| Some(a.clone()))
      .ok_or_else(|| ValidationError::invalid_type(field, "array")),
    None => Ok(None),
  }
}

/// Validate a required array of strings
pub fn require_string_array(value: Option<&serde_json::Value>, field: &str) -> ValidationResult<Vec<String>> {
  let arr = require_array(value, field)?;
  arr
    .into_iter()
    .enumerate()
    .map(|(i, v)| {
      v.as_str()
        .map(String::from)
        .ok_or_else(|| ValidationError::invalid_type(format!("{}[{}]", field, i), "string"))
    })
    .collect()
}

/// Validate an optional array of strings
pub fn optional_string_array(value: Option<&serde_json::Value>, field: &str) -> ValidationResult<Option<Vec<String>>> {
  match optional_array(value, field)? {
    Some(arr) => {
      let result: ValidationResult<Vec<String>> = arr
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
          v.as_str()
            .map(String::from)
            .ok_or_else(|| ValidationError::invalid_type(format!("{}[{}]", field, i), "string"))
        })
        .collect();
      result.map(Some)
    }
    None => Ok(None),
  }
}

/// Validate an enum value (string must match one of the valid values)
pub fn require_enum<'a>(
  value: Option<&serde_json::Value>,
  field: &str,
  valid_values: &[&'a str],
) -> ValidationResult<&'a str> {
  let s = require_string(value, field)?;
  valid_values
    .iter()
    .find(|&&v| v.eq_ignore_ascii_case(&s))
    .copied()
    .ok_or_else(|| ValidationError::invalid_enum(field, valid_values))
}

/// Validate an optional enum value
pub fn optional_enum<'a>(
  value: Option<&serde_json::Value>,
  field: &str,
  valid_values: &[&'a str],
) -> ValidationResult<Option<&'a str>> {
  match optional_string(value, field)? {
    Some(s) => valid_values
      .iter()
      .find(|&&v| v.eq_ignore_ascii_case(&s))
      .copied()
      .map(Some)
      .ok_or_else(|| ValidationError::invalid_enum(field, valid_values)),
    None => Ok(None),
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use serde_json::json;

  #[test]
  fn test_require_string() {
    let obj = json!({"name": "hello"});
    assert_eq!(require_string(obj.get("name"), "name").unwrap(), "hello");
    assert!(require_string(obj.get("missing"), "missing").is_err());

    let obj = json!({"name": 123});
    assert!(require_string(obj.get("name"), "name").is_err());
  }

  #[test]
  fn test_require_string_min() {
    let obj = json!({"content": "hi"});
    let result = require_string_min(obj.get("content"), "content", 5);
    assert!(result.is_err());
    assert!(result.unwrap_err().message.contains("at least 5"));

    let obj = json!({"content": "hello world"});
    assert!(require_string_min(obj.get("content"), "content", 5).is_ok());
  }

  #[test]
  fn test_optional_string() {
    let obj = json!({"name": "hello"});
    assert_eq!(
      optional_string(obj.get("name"), "name").unwrap(),
      Some("hello".to_string())
    );
    assert_eq!(optional_string(obj.get("missing"), "missing").unwrap(), None);

    let obj = json!({"name": null});
    assert_eq!(optional_string(obj.get("name"), "name").unwrap(), None);
  }

  #[test]
  fn test_require_i64() {
    let obj = json!({"count": 42});
    assert_eq!(require_i64(obj.get("count"), "count").unwrap(), 42);

    let obj = json!({"count": "not a number"});
    assert!(require_i64(obj.get("count"), "count").is_err());
  }

  #[test]
  fn test_require_i64_range() {
    let obj = json!({"limit": 10});
    assert!(require_i64_range(obj.get("limit"), "limit", 1, 100).is_ok());

    let obj = json!({"limit": 0});
    let result = require_i64_range(obj.get("limit"), "limit", 1, 100);
    assert!(result.is_err());
    assert!(result.unwrap_err().message.contains("between 1 and 100"));
  }

  #[test]
  fn test_require_f64_range() {
    let obj = json!({"salience": 0.5});
    assert!(require_f64_range(obj.get("salience"), "salience", 0.0, 1.0).is_ok());

    let obj = json!({"salience": 1.5});
    let result = require_f64_range(obj.get("salience"), "salience", 0.0, 1.0);
    assert!(result.is_err());
  }

  #[test]
  fn test_require_bool() {
    let obj = json!({"active": true});
    assert!(require_bool(obj.get("active"), "active").unwrap());

    let obj = json!({"active": "not a bool"});
    assert!(require_bool(obj.get("active"), "active").is_err());
  }

  #[test]
  fn test_require_string_array() {
    let obj = json!({"tags": ["a", "b", "c"]});
    let tags = require_string_array(obj.get("tags"), "tags").unwrap();
    assert_eq!(tags, vec!["a", "b", "c"]);

    let obj = json!({"tags": [1, 2, 3]});
    let result = require_string_array(obj.get("tags"), "tags");
    assert!(result.is_err());
    assert!(result.unwrap_err().field.contains("[0]"));
  }

  #[test]
  fn test_optional_string_array() {
    let obj = json!({});
    assert_eq!(optional_string_array(obj.get("tags"), "tags").unwrap(), None);

    let obj = json!({"tags": null});
    assert_eq!(optional_string_array(obj.get("tags"), "tags").unwrap(), None);

    let obj = json!({"tags": ["a", "b"]});
    assert_eq!(
      optional_string_array(obj.get("tags"), "tags").unwrap(),
      Some(vec!["a".to_string(), "b".to_string()])
    );
  }

  #[test]
  fn test_require_enum() {
    let obj = json!({"sector": "semantic"});
    assert_eq!(
      require_enum(
        obj.get("sector"),
        "sector",
        &["semantic", "episodic", "procedural", "emotional"]
      )
      .unwrap(),
      "semantic"
    );

    // Case insensitive
    let obj = json!({"sector": "SEMANTIC"});
    assert_eq!(
      require_enum(obj.get("sector"), "sector", &["semantic", "episodic"]).unwrap(),
      "semantic"
    );

    // Invalid value
    let obj = json!({"sector": "invalid"});
    let result = require_enum(obj.get("sector"), "sector", &["semantic", "episodic"]);
    assert!(result.is_err());
    assert!(result.unwrap_err().message.contains("one of: semantic, episodic"));
  }

  #[test]
  fn test_optional_enum() {
    let obj = json!({});
    assert_eq!(
      optional_enum(obj.get("sector"), "sector", &["semantic", "episodic"]).unwrap(),
      None
    );

    let obj = json!({"sector": "semantic"});
    assert_eq!(
      optional_enum(obj.get("sector"), "sector", &["semantic", "episodic"]).unwrap(),
      Some("semantic")
    );
  }

  #[test]
  fn test_validation_error_constructors() {
    let err = ValidationError::missing("content");
    assert!(err.message.contains("is required"));

    let err = ValidationError::too_short("content", 10);
    assert!(err.message.contains("at least 10"));

    let err = ValidationError::too_long("content", 1000);
    assert!(err.message.contains("at most 1000"));

    let err = ValidationError::invalid_type("count", "integer");
    assert!(err.message.contains("expected integer"));

    let err = ValidationError::out_of_range("limit", 1, 100);
    assert!(err.message.contains("between 1 and 100"));
  }
}
