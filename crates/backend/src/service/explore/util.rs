//! Utility functions for the explore service.

use crate::domain::code::CodeChunk;

/// Truncate content to a preview length.
pub fn truncate_preview(content: &str, max_len: usize) -> String {
  let content = content.trim();
  if content.len() <= max_len {
    content.to_string()
  } else {
    let mut truncate_at = max_len.saturating_sub(3);
    // Walk back to find a valid UTF-8 char boundary
    while truncate_at > 0 && !content.is_char_boundary(truncate_at) {
      truncate_at -= 1;
    }
    format!("{}...", &content[..truncate_at])
  }
}

/// Create a semantic preview for a code chunk.
///
/// Prioritizes information useful for relevance evaluation:
/// 1. Signature (most semantically meaningful for functions/classes)
/// 2. First line of docstring (natural language description)
/// 3. Body code (as space permits)
///
/// This helps LLMs quickly evaluate "is this relevant?" without
/// needing to fetch full content.
pub fn semantic_code_preview(chunk: &CodeChunk, max_len: usize) -> String {
  let mut preview = String::with_capacity(max_len);

  // Start with signature if available (most meaningful for definitions)
  if let Some(ref sig) = chunk.signature {
    // Clean up multi-line signatures to single line
    let clean_sig: String = sig.lines().map(|l| l.trim()).collect::<Vec<_>>().join(" ");
    preview.push_str(&clean_sig);
  }

  // Add truncated docstring if we have room
  if let Some(ref doc) = chunk.docstring {
    if !preview.is_empty() {
      preview.push_str(" // ");
    }
    // Take first meaningful line of docstring
    let first_line = doc
      .lines()
      .map(|l| l.trim().trim_start_matches("///").trim_start_matches("//").trim())
      .find(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with('*'))
      .unwrap_or("");
    if !first_line.is_empty() {
      let remaining = max_len.saturating_sub(preview.len()).saturating_sub(3);
      if remaining > 20 {
        preview.push_str(&truncate_preview(first_line, remaining));
      }
    }
  }

  // If we still have room and preview is short, add some body code
  if preview.len() < max_len / 2 {
    if !preview.is_empty() {
      preview.push_str(" | ");
    }
    // Skip past signature/docstring in content to get body
    let body_start = chunk.content.find('{').unwrap_or(0);
    let body = &chunk.content[body_start..];
    let remaining = max_len.saturating_sub(preview.len());
    if remaining > 20 {
      preview.push_str(&truncate_preview(body, remaining));
    }
  }

  // Fallback: if nothing useful, just use truncated content
  if preview.is_empty() {
    return truncate_preview(&chunk.content, max_len);
  }

  // Final truncation to ensure we're within limits
  truncate_preview(&preview, max_len)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_truncate_preview() {
    assert_eq!(truncate_preview("short", 10), "short");
    assert_eq!(truncate_preview("this is a longer string", 10), "this is...");
  }

  #[test]
  fn test_truncate_preview_exact_length() {
    assert_eq!(truncate_preview("exactly10!", 10), "exactly10!");
  }

  #[test]
  fn test_truncate_preview_whitespace() {
    assert_eq!(truncate_preview("  trimmed  ", 20), "trimmed");
  }

  #[test]
  fn test_truncate_preview_empty() {
    assert_eq!(truncate_preview("", 10), "");
  }

  #[test]
  fn test_truncate_preview_multibyte_chars() {
    // 'ˇ' is 2 bytes, ensure we don't slice in the middle
    let content = "abcdefˇghij";
    let result = truncate_preview(content, 10);
    // Should not panic and should produce valid UTF-8
    assert!(
      result.is_ascii() || result.chars().count() > 0,
      "result should be valid UTF-8"
    );
  }
}
