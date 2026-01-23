//! Noise detection patterns.
//!
//! Identifies test code, internal symbols, and other results
//! that add noise to exploration without providing value.

use glob::Pattern;
use serde::{Deserialize, Serialize};

/// Patterns for detecting noise in search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoisePatterns {
    /// File path patterns (glob format)
    #[serde(default = "default_file_patterns")]
    pub file_patterns: Vec<String>,
    /// Symbol name patterns (glob format)
    #[serde(default = "default_symbol_patterns")]
    pub symbol_patterns: Vec<String>,
    /// Content patterns (literal substrings)
    #[serde(default = "default_content_patterns")]
    pub content_patterns: Vec<String>,
}

fn default_file_patterns() -> Vec<String> {
    vec![
        "**/tests/**".to_string(),
        "**/test/**".to_string(),
        "**/benches/**".to_string(),
        "**/benchmark/**".to_string(),
        "**/examples/**".to_string(),
        "**/*_test.rs".to_string(),
        "**/*_test.ts".to_string(),
        "**/*.test.ts".to_string(),
        "**/*.test.tsx".to_string(),
        "**/*.spec.ts".to_string(),
        "**/*.spec.tsx".to_string(),
        "**/mock/**".to_string(),
        "**/mocks/**".to_string(),
        "**/fixtures/**".to_string(),
        "**/testdata/**".to_string(),
        "**/__tests__/**".to_string(),
        "**/__mocks__/**".to_string(),
    ]
}

fn default_symbol_patterns() -> Vec<String> {
    vec![
        "test_*".to_string(),
        "*_test".to_string(),
        "Test*".to_string(),
        "*Test".to_string(),
        "Mock*".to_string(),
        "*Mock".to_string(),
        "Fake*".to_string(),
        "*Fake".to_string(),
        "Stub*".to_string(),
        "*Stub".to_string(),
        "_*".to_string(),           // Internal/private
        "__*".to_string(),          // Dunder methods
        "assert_*".to_string(),
        "expect_*".to_string(),
        "should_*".to_string(),
    ]
}

fn default_content_patterns() -> Vec<String> {
    vec![
        "#[cfg(test)]".to_string(),
        "#[test]".to_string(),
        "#[ignore]".to_string(),
        "#[bench]".to_string(),
        "mod tests".to_string(),
        "@test".to_string(),
        "@Test".to_string(),
        "describe(".to_string(),
        "it(".to_string(),
        "test(".to_string(),
        "expect(".to_string(),
        "assert!".to_string(),
        "assert_eq!".to_string(),
        "assert_ne!".to_string(),
        "debug_assert!".to_string(),
    ]
}

impl Default for NoisePatterns {
    fn default() -> Self {
        Self {
            file_patterns: default_file_patterns(),
            symbol_patterns: default_symbol_patterns(),
            content_patterns: default_content_patterns(),
        }
    }
}

impl NoisePatterns {
    /// Create empty patterns (nothing is noise).
    pub fn none() -> Self {
        Self {
            file_patterns: vec![],
            symbol_patterns: vec![],
            content_patterns: vec![],
        }
    }

    /// Create with custom file patterns only.
    pub fn with_file_patterns(patterns: Vec<String>) -> Self {
        Self {
            file_patterns: patterns,
            symbol_patterns: vec![],
            content_patterns: vec![],
        }
    }

    /// Add custom patterns.
    pub fn add_file_pattern(&mut self, pattern: &str) {
        self.file_patterns.push(pattern.to_string());
    }

    pub fn add_symbol_pattern(&mut self, pattern: &str) {
        self.symbol_patterns.push(pattern.to_string());
    }

    pub fn add_content_pattern(&mut self, pattern: &str) {
        self.content_patterns.push(pattern.to_string());
    }

    /// Check if a file path is noise.
    pub fn is_noise_file(&self, path: &str) -> bool {
        for pattern in &self.file_patterns {
            if let Ok(p) = Pattern::new(pattern)
                && p.matches(path) {
                    return true;
                }
        }
        false
    }

    /// Check if a symbol name is noise.
    pub fn is_noise_symbol(&self, symbol: &str) -> bool {
        for pattern in &self.symbol_patterns {
            if let Ok(p) = Pattern::new(pattern)
                && p.matches(symbol) {
                    return true;
                }
        }
        false
    }

    /// Check if content contains noise patterns.
    pub fn has_noise_content(&self, content: &str) -> bool {
        for pattern in &self.content_patterns {
            if content.contains(pattern) {
                return true;
            }
        }
        false
    }

    /// Check if any aspect is noise.
    pub fn is_noise(&self, file: Option<&str>, symbol: Option<&str>, content: Option<&str>) -> bool {
        file.is_some_and(|f| self.is_noise_file(f))
            || symbol.is_some_and(|s| self.is_noise_symbol(s))
            || content.is_some_and(|c| self.has_noise_content(c))
    }

    /// Count noise items in a list.
    pub fn count_noise_files(&self, files: &[String]) -> usize {
        files.iter().filter(|f| self.is_noise_file(f)).count()
    }

    /// Count noise symbols in a list.
    pub fn count_noise_symbols(&self, symbols: &[String]) -> usize {
        symbols.iter().filter(|s| self.is_noise_symbol(s)).count()
    }

    /// Calculate noise ratio for a set of results.
    pub fn noise_ratio(&self, files: &[String], symbols: &[String]) -> f64 {
        let total = files.len() + symbols.len();
        if total == 0 {
            return 0.0;
        }

        let noise = self.count_noise_files(files) + self.count_noise_symbols(symbols);
        noise as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_noise_file() {
        let patterns = NoisePatterns::default();

        assert!(patterns.is_noise_file("src/tests/test_main.rs"));
        assert!(patterns.is_noise_file("crates/core/benches/bench.rs"));
        assert!(patterns.is_noise_file("src/__tests__/foo.ts"));
        assert!(patterns.is_noise_file("src/component.test.tsx"));

        assert!(!patterns.is_noise_file("src/main.rs"));
        assert!(!patterns.is_noise_file("crates/core/src/lib.rs"));
    }

    #[test]
    fn test_is_noise_symbol() {
        let patterns = NoisePatterns::default();

        assert!(patterns.is_noise_symbol("test_something"));
        assert!(patterns.is_noise_symbol("MockDatabase"));
        assert!(patterns.is_noise_symbol("_internal"));
        assert!(patterns.is_noise_symbol("FakeClient"));

        assert!(!patterns.is_noise_symbol("Database"));
        assert!(!patterns.is_noise_symbol("run_server"));
    }

    #[test]
    fn test_has_noise_content() {
        let patterns = NoisePatterns::default();

        assert!(patterns.has_noise_content("#[cfg(test)]\nmod tests {"));
        assert!(patterns.has_noise_content("fn test() { assert!(true); }"));
        assert!(patterns.has_noise_content("describe('component', () => {"));

        assert!(!patterns.has_noise_content("fn main() { run(); }"));
    }

    #[test]
    fn test_is_noise_combined() {
        let patterns = NoisePatterns::default();

        assert!(patterns.is_noise(Some("src/tests/foo.rs"), None, None));
        assert!(patterns.is_noise(None, Some("test_foo"), None));
        assert!(patterns.is_noise(None, None, Some("#[test]")));
        assert!(patterns.is_noise(Some("src/tests/foo.rs"), Some("test_foo"), Some("#[test]")));

        assert!(!patterns.is_noise(Some("src/main.rs"), Some("main"), Some("fn main() {}")));
    }

    #[test]
    fn test_noise_ratio() {
        let patterns = NoisePatterns::default();

        let files = vec![
            "src/main.rs".to_string(),
            "src/tests/test.rs".to_string(),
            "src/lib.rs".to_string(),
        ];
        let symbols = vec![
            "main".to_string(),
            "test_foo".to_string(),
            "run".to_string(),
        ];

        let ratio = patterns.noise_ratio(&files, &symbols);
        // 1 noise file + 1 noise symbol out of 6 total
        assert!((ratio - 2.0 / 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_patterns() {
        let patterns = NoisePatterns::none();

        assert!(!patterns.is_noise_file("src/tests/test.rs"));
        assert!(!patterns.is_noise_symbol("test_foo"));
        assert!(!patterns.has_noise_content("#[test]"));
    }

    #[test]
    fn test_custom_patterns() {
        let mut patterns = NoisePatterns::none();
        patterns.add_file_pattern("**/vendor/**");

        assert!(patterns.is_noise_file("vendor/lib/foo.rs"));
        assert!(!patterns.is_noise_file("src/main.rs"));
    }
}
