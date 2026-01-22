use engram_core::Sector;

/// Pattern definitions for sector classification
struct PatternSet {
  patterns: Vec<&'static str>,
  weight: f32,
}

impl PatternSet {
  /// Score against pre-lowercased text (avoids repeated to_lowercase calls)
  fn score_lowercase(&self, lower: &str) -> f32 {
    self.patterns.iter().filter(|p| lower.contains(*p)).count() as f32 * self.weight
  }
}

/// Episodic patterns - session events, temporal references
fn episodic_patterns() -> PatternSet {
  PatternSet {
    patterns: vec![
      "asked",
      "mentioned",
      "user wanted",
      "during this session",
      "during session",
      "today",
      "just now",
      "earlier",
      "recently",
      "we discussed",
      "you said",
      "i mentioned",
      "this morning",
      "this afternoon",
      "right now",
    ],
    weight: 1.0,
  }
}

/// Semantic patterns - facts, codebase knowledge, structure
fn semantic_patterns() -> PatternSet {
  PatternSet {
    patterns: vec![
      "located in",
      "file",
      "module",
      "contains",
      "defined",
      "architecture",
      "structure",
      "component",
      "system",
      "implements",
      "function",
      "class",
      "interface",
      "database",
      "api",
      "endpoint",
      "directory",
      "package",
      "crate",
    ],
    weight: 1.0,
  }
}

/// Procedural patterns - how-to, workflows, instructions
fn procedural_patterns() -> PatternSet {
  PatternSet {
    patterns: vec![
      "how to",
      "steps to",
      "first",
      "then",
      "to do this",
      "process",
      "workflow",
      "procedure",
      "recipe",
      "run",
      "execute",
      "command",
      "install",
      "configure",
      "setup",
      "build",
      "deploy",
      "step 1",
      "step 2",
      "finally",
      "next",
    ],
    weight: 1.0,
  }
}

/// Emotional patterns - preferences, frustrations
fn emotional_patterns() -> PatternSet {
  PatternSet {
    patterns: vec![
      "frustrated",
      "prefer",
      "hate",
      "love",
      "pain point",
      "annoying",
      "want",
      "need",
      "like",
      "dislike",
      "important",
      "always",
      "never",
      "favorite",
      "best",
      "worst",
      "must",
      "should always",
      "should never",
    ],
    weight: 1.5, // Higher weight for emotional patterns
  }
}

/// Reflective patterns - insights, summaries, lessons
fn reflective_patterns() -> PatternSet {
  PatternSet {
    patterns: vec![
      "learned",
      "realized",
      "should have",
      "insight",
      "pattern",
      "conclusion",
      "summary",
      "takeaway",
      "lesson",
      "key point",
      "in retrospect",
      "looking back",
      "overall",
      "in general",
      "the main thing",
      "remember that",
      "note to self",
    ],
    weight: 1.3, // Higher weight for reflective patterns
  }
}

/// Classify content into a memory sector
pub fn classify_sector(content: &str) -> Sector {
  // Convert to lowercase once, then score all patterns
  let lower = content.to_lowercase();
  let scores = [
    (Sector::Emotional, emotional_patterns().score_lowercase(&lower)),
    (Sector::Reflective, reflective_patterns().score_lowercase(&lower)),
    (Sector::Procedural, procedural_patterns().score_lowercase(&lower)),
    (Sector::Semantic, semantic_patterns().score_lowercase(&lower)),
    (Sector::Episodic, episodic_patterns().score_lowercase(&lower)),
  ];

  scores
    .into_iter()
    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    .map(|(sector, _)| sector)
    .unwrap_or(Sector::Semantic)
}

/// Get all sector scores for debugging/analysis
pub fn sector_scores(content: &str) -> Vec<(Sector, f32)> {
  // Convert to lowercase once, then score all patterns
  let lower = content.to_lowercase();
  vec![
    (Sector::Emotional, emotional_patterns().score_lowercase(&lower)),
    (Sector::Reflective, reflective_patterns().score_lowercase(&lower)),
    (Sector::Procedural, procedural_patterns().score_lowercase(&lower)),
    (Sector::Semantic, semantic_patterns().score_lowercase(&lower)),
    (Sector::Episodic, episodic_patterns().score_lowercase(&lower)),
  ]
}

/// Extract concepts from memory content
pub fn extract_concepts(content: &str) -> Vec<String> {
  let mut concepts = Vec::new();

  // Backtick strings (code references)
  for cap in find_backtick_content(content) {
    if cap.len() >= 2 && cap.len() <= 100 {
      concepts.push(cap);
    }
  }

  // CamelCase identifiers
  for word in content.split_whitespace() {
    if is_camel_case(word) && word.len() >= 3 && word.len() <= 50 {
      concepts.push(word.to_string());
    }
  }

  // snake_case identifiers
  for word in content.split_whitespace() {
    let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
    if is_snake_case(cleaned) && cleaned.len() >= 3 && cleaned.len() <= 50 {
      concepts.push(cleaned.to_string());
    }
  }

  // File paths
  for word in content.split_whitespace() {
    if looks_like_file_path(word) {
      concepts.push(
        word
          .trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '/' && c != '_' && c != '-')
          .to_string(),
      );
    }
  }

  // Deduplicate and sort
  concepts.sort();
  concepts.dedup();
  concepts
}

/// Extract file references from content
pub fn extract_files(content: &str) -> Vec<String> {
  let mut files = Vec::new();

  for word in content.split_whitespace() {
    let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '/' && c != '_' && c != '-');
    if looks_like_file_path(cleaned) {
      files.push(cleaned.to_string());
    }
  }

  // Also check for backtick-wrapped paths
  for cap in find_backtick_content(content) {
    if looks_like_file_path(&cap) {
      files.push(cap);
    }
  }

  files.sort();
  files.dedup();
  files
}

fn find_backtick_content(content: &str) -> Vec<String> {
  let mut results = Vec::new();
  let mut in_backtick = false;
  let mut current = String::new();

  for ch in content.chars() {
    if ch == '`' {
      if in_backtick {
        if !current.is_empty() {
          results.push(current.clone());
        }
        current.clear();
      }
      in_backtick = !in_backtick;
    } else if in_backtick {
      current.push(ch);
    }
  }

  results
}

fn is_camel_case(s: &str) -> bool {
  if s.is_empty() {
    return false;
  }

  let chars: Vec<char> = s.chars().collect();

  // Must start with uppercase
  if !chars[0].is_uppercase() {
    return false;
  }

  // Must have at least one lowercase and one more uppercase (after first)
  let has_lower = chars.iter().any(|c| c.is_lowercase());
  let has_internal_upper = chars.iter().skip(1).any(|c| c.is_uppercase());

  has_lower && has_internal_upper && s.chars().all(|c| c.is_alphanumeric())
}

fn is_snake_case(s: &str) -> bool {
  if s.is_empty() || !s.contains('_') {
    return false;
  }

  // All lowercase letters, digits, and underscores
  s.chars().all(|c| c.is_lowercase() || c.is_ascii_digit() || c == '_')
    && !s.starts_with('_')
    && !s.ends_with('_')
    && !s.contains("__")
}

fn looks_like_file_path(s: &str) -> bool {
  if s.is_empty() || s.len() < 3 {
    return false;
  }

  // Must contain a dot for extension or a slash for path
  let has_extension = s.contains('.') && {
    let parts: Vec<&str> = s.split('.').collect();
    if let Some(ext) = parts.last() {
      !ext.is_empty() && ext.len() <= 4 && ext.chars().all(|c| c.is_alphabetic())
    } else {
      false
    }
  };

  let has_path_sep = s.contains('/');

  (has_extension || has_path_sep)
    && s
      .chars()
      .all(|c| c.is_alphanumeric() || c == '.' || c == '/' || c == '_' || c == '-')
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_classify_episodic() {
    let content = "The user asked about authentication earlier today during this session";
    assert_eq!(classify_sector(content), Sector::Episodic);
  }

  #[test]
  fn test_classify_semantic() {
    let content = "The authentication module is located in src/auth/ and contains the login function";
    assert_eq!(classify_sector(content), Sector::Semantic);
  }

  #[test]
  fn test_classify_procedural() {
    let content = "To deploy the application, first run the build command, then execute the deploy script";
    assert_eq!(classify_sector(content), Sector::Procedural);
  }

  #[test]
  fn test_classify_emotional() {
    let content = "The user hates dealing with CSS and prefers using Tailwind. This is very important to them.";
    assert_eq!(classify_sector(content), Sector::Emotional);
  }

  #[test]
  fn test_classify_reflective() {
    let content = "Looking back, the main takeaway is that we should have tested more. Lesson learned.";
    assert_eq!(classify_sector(content), Sector::Reflective);
  }

  #[test]
  fn test_extract_concepts_backticks() {
    let content = "The `UserService` handles `user_authentication` in the `auth.ts` file";
    let concepts = extract_concepts(content);

    assert!(concepts.contains(&"UserService".to_string()));
    assert!(concepts.contains(&"user_authentication".to_string()));
    assert!(concepts.contains(&"auth.ts".to_string()));
  }

  #[test]
  fn test_extract_concepts_camel_case() {
    let content = "The AuthController uses the UserRepository to fetch data";
    let concepts = extract_concepts(content);

    assert!(concepts.contains(&"AuthController".to_string()));
    assert!(concepts.contains(&"UserRepository".to_string()));
  }

  #[test]
  fn test_extract_concepts_snake_case() {
    let content = "Call the get_user_by_id function to retrieve user_data";
    let concepts = extract_concepts(content);

    assert!(concepts.contains(&"get_user_by_id".to_string()));
    assert!(concepts.contains(&"user_data".to_string()));
  }

  #[test]
  fn test_extract_files() {
    let content = "Check the src/auth/login.ts file and the `config/app.json` configuration";
    let files = extract_files(content);

    assert!(files.contains(&"src/auth/login.ts".to_string()));
    assert!(files.contains(&"config/app.json".to_string()));
  }

  #[test]
  fn test_is_camel_case() {
    assert!(is_camel_case("UserService"));
    assert!(is_camel_case("HttpResponse"));
    assert!(!is_camel_case("userservice"));
    assert!(!is_camel_case("USERSERVICE"));
    assert!(!is_camel_case("user_service"));
  }

  #[test]
  fn test_is_snake_case() {
    assert!(is_snake_case("user_service"));
    assert!(is_snake_case("get_user_by_id"));
    assert!(!is_snake_case("userService"));
    assert!(!is_snake_case("_private"));
    assert!(!is_snake_case("trailing_"));
    assert!(!is_snake_case("double__underscore"));
  }

  #[test]
  fn test_looks_like_file_path() {
    assert!(looks_like_file_path("src/auth/login.ts"));
    assert!(looks_like_file_path("config.json"));
    assert!(looks_like_file_path("README.md"));
    assert!(!looks_like_file_path("hello"));
    assert!(!looks_like_file_path("a.toolongextension"));
  }

  #[test]
  fn test_sector_scores() {
    let content = "The user prefers TypeScript and hates JavaScript";
    let scores = sector_scores(content);

    // Emotional should have highest score
    let emotional_score = scores
      .iter()
      .find(|(s, _)| *s == Sector::Emotional)
      .map(|(_, score)| *score)
      .unwrap_or(0.0);

    assert!(emotional_score > 0.0);
  }
}
