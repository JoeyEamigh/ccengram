/// Splits code identifiers into searchable tokens for FTS indexing.
///
/// Handles camelCase, snake_case, SCREAMING_CASE, file paths, and mixed patterns.
/// Preserves original tokens alongside splits so exact matches still work.
/// Lowercases everything and filters common code stop words.
///
/// Examples:
/// - "camelCase" -> "camelcase camel case"
/// - "snake_case" -> "snake_case snake case"
/// - "HTTPServer" -> "httpserver http server"
/// - "src/auth/handler.rs" -> "src auth handler rs"
pub fn tokenize_code(text: &str) -> String {
  let mut tokens: Vec<String> = Vec::new();

  for word in text.split_whitespace() {
    // Split on path separators and dots
    let path_parts: Vec<&str> = word.split(['/', '\\', '.']).collect();

    for part in path_parts {
      if part.is_empty() {
        continue;
      }

      // Split on underscores
      let underscore_parts: Vec<&str> = part.split('_').filter(|s| !s.is_empty()).collect();

      if underscore_parts.len() > 1 {
        // Preserve original (e.g., "snake_case")
        let original = part.to_lowercase();
        if !is_code_stop_word(&original) {
          tokens.push(original);
        }
        // Add sub-parts
        for sub in &underscore_parts {
          add_camel_split(&mut tokens, sub);
        }
      } else {
        // No underscores: try camelCase split
        add_camel_split(&mut tokens, part);
      }
    }
  }

  tokens.join(" ")
}

fn add_camel_split(tokens: &mut Vec<String>, word: &str) {
  let lower = word.to_lowercase();

  // Split on case transitions
  let parts = split_camel_case(word);

  if parts.len() > 1 {
    // Preserve original lowercased
    if !is_code_stop_word(&lower) {
      tokens.push(lower);
    }
    // Add each sub-token
    for part in parts {
      let p = part.to_lowercase();
      if !p.is_empty() && !is_code_stop_word(&p) {
        tokens.push(p);
      }
    }
  } else if !lower.is_empty() && !is_code_stop_word(&lower) {
    tokens.push(lower);
  }
}

fn split_camel_case(s: &str) -> Vec<&str> {
  let mut parts = Vec::new();
  let bytes = s.as_bytes();
  let len = bytes.len();

  if len == 0 {
    return parts;
  }

  let mut start = 0;

  for i in 1..len {
    let prev = bytes[i - 1];
    let curr = bytes[i];

    let should_split = if prev.is_ascii_lowercase() && curr.is_ascii_uppercase() {
      // aB -> split between a and B
      true
    } else if i + 1 < len && prev.is_ascii_uppercase() && curr.is_ascii_uppercase() && bytes[i + 1].is_ascii_lowercase()
    {
      // ABc -> split between A and Bc (handles HTTPServer -> HTTP + Server)
      true
    } else {
      false
    };

    if should_split {
      if start < i {
        parts.push(&s[start..i]);
      }
      start = i;
    }
  }

  if start < len {
    parts.push(&s[start..]);
  }

  parts
}

fn is_code_stop_word(word: &str) -> bool {
  matches!(
    word,
    "fn"
      | "pub"
      | "struct"
      | "impl"
      | "def"
      | "class"
      | "function"
      | "return"
      | "self"
      | "this"
      | "let"
      | "const"
      | "var"
      | "import"
      | "use"
      | "for"
      | "while"
      | "if"
      | "else"
      | "match"
      | "where"
      | "type"
      | "trait"
      | "enum"
      | "mod"
  )
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_camel_case_splitting() {
    let result = tokenize_code("camelCase");
    assert!(result.contains("camelcase"), "should have original: {result}");
    assert!(result.contains("camel"), "should have 'camel': {result}");
    assert!(result.contains("case"), "should have 'case': {result}");
  }

  #[test]
  fn test_snake_case_splitting() {
    let result = tokenize_code("snake_case");
    assert!(result.contains("snake_case"), "should preserve original: {result}");
    assert!(result.contains("snake"), "should have 'snake': {result}");
    assert!(result.contains("case"), "should have 'case': {result}");
  }

  #[test]
  fn test_screaming_case() {
    let result = tokenize_code("HTTPServer");
    assert!(result.contains("httpserver"), "should have original: {result}");
    assert!(result.contains("http"), "should have 'http': {result}");
    assert!(result.contains("server"), "should have 'server': {result}");
  }

  #[test]
  fn test_file_path_splitting() {
    let result = tokenize_code("src/auth/handler.rs");
    assert!(result.contains("src"), "should have 'src': {result}");
    assert!(result.contains("auth"), "should have 'auth': {result}");
    assert!(result.contains("handler"), "should have 'handler': {result}");
    assert!(result.contains("rs"), "should have 'rs': {result}");
  }

  #[test]
  fn test_stop_word_filtering() {
    let result = tokenize_code("fn authenticate_user impl UserService");
    assert!(!result.contains(" fn "), "should filter 'fn': {result}");
    assert!(!result.contains(" impl "), "should filter 'impl': {result}");
    assert!(result.contains("authenticate"), "should keep 'authenticate': {result}");
    assert!(result.contains("user"), "should keep 'user': {result}");
    assert!(result.contains("userservice"), "should keep 'userservice': {result}");
  }

  #[test]
  fn test_mixed_identifier_splitting() {
    let result = tokenize_code("authenticate_user");
    assert!(
      result.contains("authenticate_user"),
      "should preserve original: {result}"
    );
    assert!(result.contains("authenticate"), "should have 'authenticate': {result}");
    assert!(result.contains("user"), "should have 'user': {result}");
  }

  #[test]
  fn test_complex_embedding_text() {
    let input = "[DEFINITION] Function: authenticate_user\n[FILE] src/auth/handler.rs\n[SIGNATURE] pub fn authenticate_user(credentials: &Credentials) -> Result<User>";
    let result = tokenize_code(input);
    assert!(result.contains("authenticate"), "should find 'authenticate': {result}");
    assert!(result.contains("credentials"), "should find 'credentials': {result}");
    assert!(result.contains("handler"), "should find 'handler': {result}");
    assert!(result.contains("auth"), "should find 'auth': {result}");
  }

  #[test]
  fn test_empty_and_whitespace() {
    assert_eq!(tokenize_code(""), "");
    assert_eq!(tokenize_code("   "), "");
  }

  #[test]
  fn test_consecutive_uppercase_acronyms() {
    // XMLParser -> XML + Parser
    let result = tokenize_code("XMLParser");
    assert!(result.contains("xml"), "should split XML from XMLParser: {result}");
    assert!(
      result.contains("parser"),
      "should split Parser from XMLParser: {result}"
    );

    // JSONAPIClient -> JSONAPI + Client (limited by the look-ahead algorithm)
    // The algorithm splits at uppercase->lowercase transitions with lookbehind
    let result = tokenize_code("JSONAPIClient");
    assert!(result.contains("client"), "should extract Client: {result}");
  }

  #[test]
  fn test_mixed_snake_and_camel() {
    // snake_case containing camelCase sub-parts
    let result = tokenize_code("get_HTTPResponse");
    assert!(result.contains("get"), "should split 'get': {result}");
    assert!(result.contains("http"), "should extract HTTP: {result}");
    assert!(result.contains("response"), "should extract Response: {result}");
  }

  #[test]
  fn test_deeply_nested_path() {
    let result = tokenize_code("src/service/memory/ranking.rs");
    assert!(result.contains("service"), "should have 'service': {result}");
    assert!(result.contains("memory"), "should have 'memory': {result}");
    assert!(result.contains("ranking"), "should have 'ranking': {result}");
    assert!(result.contains("rs"), "should have 'rs': {result}");
    assert!(
      !result.contains("src/service"),
      "should NOT have unsplit path: {result}"
    );
  }

  #[test]
  fn test_windows_path_splitting() {
    let result = tokenize_code("src\\auth\\handler.rs");
    assert!(result.contains("auth"), "backslash paths should split: {result}");
    assert!(result.contains("handler"), "backslash paths should split: {result}");
  }

  #[test]
  fn test_all_stop_words_filtered() {
    // Every token is a stop word
    let result = tokenize_code("fn pub struct impl def class");
    assert_eq!(
      result.trim(),
      "",
      "all stop words should produce empty output: '{result}'"
    );
  }

  #[test]
  fn test_stop_words_in_compound_identifiers_preserved() {
    // "for_each" contains "for" as a stop word, but "for_each" itself is NOT a stop word
    let result = tokenize_code("for_each");
    assert!(
      result.contains("for_each"),
      "compound identifier should be preserved: {result}"
    );
    assert!(result.contains("each"), "should have 'each': {result}");
  }

  #[test]
  fn test_multiple_underscores() {
    let result = tokenize_code("__private__method__");
    // Should handle leading/trailing underscores gracefully
    assert!(!result.is_empty(), "double underscores should not crash: {result}");
  }

  #[test]
  fn test_single_character_tokens() {
    let result = tokenize_code("a_b_c");
    assert!(result.contains("a"), "single char tokens should be kept: {result}");
    assert!(result.contains("b"), "single char tokens should be kept: {result}");
    assert!(result.contains("c"), "single char tokens should be kept: {result}");
  }

  #[test]
  fn test_numeric_identifiers() {
    let result = tokenize_code("handler404 error500");
    assert!(
      result.contains("handler404"),
      "numeric suffixes should be kept: {result}"
    );
    assert!(result.contains("error500"), "numeric suffixes should be kept: {result}");
  }

  #[test]
  fn test_real_world_rust_signature() {
    let input = "pub async fn search_code_chunks(query_vec: &[f32], limit: usize) -> Result<Vec<CodeChunk>, DbError>";
    let result = tokenize_code(input);
    assert!(result.contains("search"), "should extract 'search': {result}");
    assert!(result.contains("code"), "should extract 'code': {result}");
    assert!(result.contains("chunks"), "should extract 'chunks': {result}");
    assert!(result.contains("query"), "should extract 'query': {result}");
    assert!(result.contains("vec"), "should extract 'vec': {result}");
    // "fn" and "pub" are stop words
    assert!(!result.starts_with("pub "), "should filter pub stop word: {result}");
  }
}
