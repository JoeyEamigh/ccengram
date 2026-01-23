//! TreeSitterParser implementation

use std::collections::{HashMap, HashSet};
use tree_sitter::{Language as TsLanguage, Parser, Query, QueryCursor, StreamingIterator};

use crate::queries;
use engram_core::Language;

/// Holds the queries for a specific language
pub struct LanguageQueries {
  pub imports: Option<Query>,
  pub calls: Option<Query>,
  pub definitions: Option<Query>,
}

/// A definition extracted from code
#[derive(Debug, Clone)]
pub struct Definition {
  pub name: String,
  pub kind: DefinitionKind,
  pub start_line: u32,
  pub end_line: u32,
}

/// The kind of definition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefinitionKind {
  Function,
  Method,
  Class,
  Struct,
  Interface,
  Trait,
  Enum,
  Const,
  Type,
  Module,
}

/// Tree-sitter based code parser
///
/// Lazily loads parsers and queries for each language as needed.
pub struct TreeSitterParser {
  parsers: HashMap<Language, Parser>,
  queries: HashMap<Language, LanguageQueries>,
}

impl TreeSitterParser {
  /// Create a new TreeSitterParser
  pub fn new() -> Self {
    Self {
      parsers: HashMap::new(),
      queries: HashMap::new(),
    }
  }

  /// Check if a language is supported for parsing
  pub fn supports_language(&self, lang: Language) -> bool {
    self.get_grammar(lang).is_some()
  }

  /// Extract import statements from code
  pub fn extract_imports(&mut self, content: &str, lang: Language) -> Vec<String> {
    self.run_query(content, lang, |q| &q.imports)
  }

  /// Extract function/method calls from code
  pub fn extract_calls(&mut self, content: &str, lang: Language) -> Vec<String> {
    self.run_query(content, lang, |q| &q.calls)
  }

  /// Extract symbol definitions from code
  pub fn extract_definitions(&mut self, content: &str, lang: Language) -> Vec<Definition> {
    self.ensure_loaded(lang);

    let Some(parser) = self.parsers.get_mut(&lang) else {
      return Vec::new();
    };

    let Some(tree) = parser.parse(content, None) else {
      return Vec::new();
    };

    let Some(queries) = self.queries.get(&lang) else {
      return Vec::new();
    };

    let Some(query) = &queries.definitions else {
      return Vec::new();
    };

    let mut cursor = QueryCursor::new();
    let mut definitions = Vec::new();

    // Use StreamingIterator's .next() method
    let mut matches = cursor.matches(query, tree.root_node(), content.as_bytes());

    while let Some(match_) = matches.next() {
      // Look for name and kind captures
      let mut name: Option<String> = None;
      let mut start_line: Option<u32> = None;
      let mut end_line: Option<u32> = None;
      let mut kind = DefinitionKind::Function; // default

      for cap in match_.captures {
        let cap_name = &query.capture_names()[cap.index as usize];
        let node = cap.node;

        match *cap_name {
          "name" => {
            if let Ok(text) = node.utf8_text(content.as_bytes()) {
              name = Some(text.to_string());
            }
          }
          "definition.function" | "function" => {
            kind = DefinitionKind::Function;
            start_line = Some(node.start_position().row as u32 + 1);
            end_line = Some(node.end_position().row as u32 + 1);
          }
          "definition.method" | "method" => {
            kind = DefinitionKind::Method;
            start_line = Some(node.start_position().row as u32 + 1);
            end_line = Some(node.end_position().row as u32 + 1);
          }
          "definition.class" | "class" => {
            kind = DefinitionKind::Class;
            start_line = Some(node.start_position().row as u32 + 1);
            end_line = Some(node.end_position().row as u32 + 1);
          }
          "definition.struct" | "struct" => {
            kind = DefinitionKind::Struct;
            start_line = Some(node.start_position().row as u32 + 1);
            end_line = Some(node.end_position().row as u32 + 1);
          }
          "definition.interface" | "interface" => {
            kind = DefinitionKind::Interface;
            start_line = Some(node.start_position().row as u32 + 1);
            end_line = Some(node.end_position().row as u32 + 1);
          }
          "definition.trait" | "trait" => {
            kind = DefinitionKind::Trait;
            start_line = Some(node.start_position().row as u32 + 1);
            end_line = Some(node.end_position().row as u32 + 1);
          }
          "definition.enum" | "enum" => {
            kind = DefinitionKind::Enum;
            start_line = Some(node.start_position().row as u32 + 1);
            end_line = Some(node.end_position().row as u32 + 1);
          }
          "definition.module" | "module" => {
            kind = DefinitionKind::Module;
            start_line = Some(node.start_position().row as u32 + 1);
            end_line = Some(node.end_position().row as u32 + 1);
          }
          "definition.const" | "const" => {
            kind = DefinitionKind::Const;
            start_line = Some(node.start_position().row as u32 + 1);
            end_line = Some(node.end_position().row as u32 + 1);
          }
          "definition.type" | "type" => {
            kind = DefinitionKind::Type;
            start_line = Some(node.start_position().row as u32 + 1);
            end_line = Some(node.end_position().row as u32 + 1);
          }
          _ => {}
        }
      }

      if let (Some(n), Some(sl), Some(el)) = (name, start_line, end_line) {
        definitions.push(Definition {
          name: n,
          kind,
          start_line: sl,
          end_line: el,
        });
      }
    }

    definitions
  }

  fn run_query<F>(&mut self, content: &str, lang: Language, get_query: F) -> Vec<String>
  where
    F: Fn(&LanguageQueries) -> &Option<Query>,
  {
    // Ensure parser and queries are loaded for this language
    self.ensure_loaded(lang);

    let Some(parser) = self.parsers.get_mut(&lang) else {
      return Vec::new();
    };

    let Some(tree) = parser.parse(content, None) else {
      return Vec::new();
    };

    let Some(queries) = self.queries.get(&lang) else {
      return Vec::new();
    };

    let Some(query) = get_query(queries) else {
      return Vec::new();
    };

    let mut cursor = QueryCursor::new();
    let mut results: Vec<String> = Vec::new();

    // Use StreamingIterator's .next() method
    let mut matches = cursor.matches(query, tree.root_node(), content.as_bytes());

    while let Some(match_) = matches.next() {
      for cap in match_.captures {
        if let Ok(text) = cap.node.utf8_text(content.as_bytes()) {
          // Clean up the string (remove quotes and angle brackets for imports, etc.)
          let cleaned = text.trim_matches(|c: char| c == '"' || c == '\'' || c == '`' || c == '<' || c == '>');
          if !cleaned.is_empty() {
            results.push(cleaned.to_string());
          }
        }
      }
    }

    // Deduplicate while preserving order
    let mut seen: HashSet<String> = HashSet::new();
    results.retain(|s| seen.insert(s.clone()));

    results
  }

  fn ensure_loaded(&mut self, lang: Language) {
    if self.parsers.contains_key(&lang) {
      return;
    }

    if let Some(grammar) = self.get_grammar(lang) {
      let mut parser = Parser::new();
      if parser.set_language(&grammar).is_ok() {
        self.parsers.insert(lang, parser);
        self.queries.insert(lang, queries::load_queries(lang, &grammar));
      }
    }
  }

  fn get_grammar(&self, lang: Language) -> Option<TsLanguage> {
    match lang {
      Language::Rust => Some(tree_sitter_rust::LANGUAGE.into()),
      Language::Python => Some(tree_sitter_python::LANGUAGE.into()),
      Language::JavaScript | Language::Jsx => Some(tree_sitter_javascript::LANGUAGE.into()),
      Language::TypeScript => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
      Language::Tsx => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
      Language::Go => Some(tree_sitter_go::LANGUAGE.into()),
      Language::Java => Some(tree_sitter_java::LANGUAGE.into()),
      Language::C => Some(tree_sitter_c::LANGUAGE.into()),
      Language::Cpp => Some(tree_sitter_cpp::LANGUAGE.into()),

      // Tier 2 (feature-gated)
      #[cfg(feature = "tier2")]
      Language::Ruby => Some(tree_sitter_ruby::LANGUAGE.into()),
      #[cfg(feature = "tier2")]
      Language::Php => Some(tree_sitter_php::LANGUAGE_PHP.into()),
      #[cfg(feature = "tier2")]
      Language::CSharp => Some(tree_sitter_c_sharp::LANGUAGE.into()),
      #[cfg(feature = "tier2")]
      Language::Kotlin => Some(tree_sitter_kotlin::LANGUAGE.into()),
      #[cfg(feature = "tier2")]
      Language::Shell => Some(tree_sitter_bash::LANGUAGE.into()),

      // Unsupported or not compiled
      _ => None,
    }
  }
}

impl Default for TreeSitterParser {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_supports_tier1_languages() {
    let parser = TreeSitterParser::new();

    assert!(parser.supports_language(Language::Rust));
    assert!(parser.supports_language(Language::Python));
    assert!(parser.supports_language(Language::JavaScript));
    assert!(parser.supports_language(Language::TypeScript));
    assert!(parser.supports_language(Language::Go));
    assert!(parser.supports_language(Language::Java));
    assert!(parser.supports_language(Language::C));
    assert!(parser.supports_language(Language::Cpp));
  }

  #[test]
  fn test_unsupported_language_returns_empty() {
    let mut parser = TreeSitterParser::new();

    // Markdown has no import/call queries
    let imports = parser.extract_imports("# Header", Language::Markdown);
    assert!(imports.is_empty());
  }
}
