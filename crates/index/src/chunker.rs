use ccengram_parser::TreeSitterParser;
use chrono::Utc;
use engram_core::{CHARS_PER_TOKEN, ChunkType, CodeChunk, Language};
use uuid::Uuid;

/// Configuration for the chunker
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
  /// Target number of lines per chunk
  pub target_lines: usize,
  /// Minimum lines per chunk
  pub min_lines: usize,
  /// Maximum lines per chunk
  pub max_lines: usize,
}

impl Default for ChunkerConfig {
  fn default() -> Self {
    Self {
      target_lines: 50,
      min_lines: 10,
      max_lines: 100,
    }
  }
}

/// Line-based code chunker with tree-sitter integration for import/call extraction
pub struct Chunker {
  config: ChunkerConfig,
  ts_parser: TreeSitterParser,
}

impl Default for Chunker {
  fn default() -> Self {
    Self::new(ChunkerConfig::default())
  }
}

impl Chunker {
  pub fn new(config: ChunkerConfig) -> Self {
    Self {
      config,
      ts_parser: TreeSitterParser::new(),
    }
  }

  /// Chunk source code into semantic pieces
  ///
  /// Uses tree-sitter to extract imports and function calls for supported languages.
  pub fn chunk(&mut self, source: &str, file_path: &str, language: Language, file_hash: &str) -> Vec<CodeChunk> {
    let lines: Vec<&str> = source.lines().collect();
    let total_lines = lines.len();

    // Small files: single chunk
    if total_lines <= self.config.max_lines {
      let chunk_type = self.determine_chunk_type(source, language);
      // Extract imports and calls using tree-sitter
      let imports = self.ts_parser.extract_imports(source, language);
      let calls = self.ts_parser.extract_calls(source, language);

      return vec![CodeChunk {
        id: Uuid::now_v7(),
        file_path: file_path.to_string(),
        content: source.to_string(),
        language,
        chunk_type,
        symbols: self.extract_symbols(source, language),
        imports,
        calls,
        start_line: 1,
        end_line: total_lines as u32,
        file_hash: file_hash.to_string(),
        indexed_at: Utc::now(),
        tokens_estimate: (source.len() / CHARS_PER_TOKEN) as u32,
      }];
    }

    // Find semantic boundaries
    let boundaries = self.find_boundaries(&lines, language);
    let mut chunks = Vec::new();
    let mut current_start = 0usize;

    for boundary in boundaries {
      let chunk_lines = boundary - current_start;

      // Accumulate until we hit target
      if chunk_lines >= self.config.target_lines {
        let content = lines[current_start..boundary].join("\n");
        let chunk_type = self.determine_chunk_type(&content, language);
        let tokens_estimate = (content.len() / CHARS_PER_TOKEN) as u32;
        // Extract imports and calls using tree-sitter
        let imports = self.ts_parser.extract_imports(&content, language);
        let calls = self.ts_parser.extract_calls(&content, language);

        chunks.push(CodeChunk {
          id: Uuid::now_v7(),
          file_path: file_path.to_string(),
          content,
          language,
          chunk_type,
          symbols: self.extract_symbols_in_range(&lines, current_start, boundary, language),
          imports,
          calls,
          start_line: (current_start + 1) as u32,
          end_line: boundary as u32,
          file_hash: file_hash.to_string(),
          indexed_at: Utc::now(),
          tokens_estimate,
        });

        current_start = boundary;
      }
    }

    // Final chunk
    if current_start < total_lines {
      let content = lines[current_start..].join("\n");
      let chunk_type = self.determine_chunk_type(&content, language);
      let tokens_estimate = (content.len() / CHARS_PER_TOKEN) as u32;
      // Extract imports and calls using tree-sitter
      let imports = self.ts_parser.extract_imports(&content, language);
      let calls = self.ts_parser.extract_calls(&content, language);

      chunks.push(CodeChunk {
        id: Uuid::now_v7(),
        file_path: file_path.to_string(),
        content,
        language,
        chunk_type,
        symbols: self.extract_symbols_in_range(&lines, current_start, total_lines, language),
        imports,
        calls,
        start_line: (current_start + 1) as u32,
        end_line: total_lines as u32,
        file_hash: file_hash.to_string(),
        indexed_at: Utc::now(),
        tokens_estimate,
      });
    }

    // If no chunks were created (no boundaries found), create one big chunk or split evenly
    if chunks.is_empty() {
      self.split_evenly(&lines, file_path, language, file_hash)
    } else {
      chunks
    }
  }

  /// Find semantic boundaries in the code
  fn find_boundaries(&self, lines: &[&str], language: Language) -> Vec<usize> {
    let mut boundaries = Vec::new();

    for (i, line) in lines.iter().enumerate() {
      let trimmed = line.trim();

      // Skip empty lines and comments
      if trimmed.is_empty() {
        continue;
      }

      // Look for function/class definitions based on language
      let is_boundary = match language {
        Language::Rust => {
          trimmed.starts_with("pub fn ")
            || trimmed.starts_with("fn ")
            || trimmed.starts_with("pub struct ")
            || trimmed.starts_with("struct ")
            || trimmed.starts_with("pub enum ")
            || trimmed.starts_with("enum ")
            || trimmed.starts_with("impl ")
            || trimmed.starts_with("pub trait ")
            || trimmed.starts_with("trait ")
            || trimmed.starts_with("pub mod ")
            || trimmed.starts_with("mod ")
        }
        Language::Python => {
          trimmed.starts_with("def ") || trimmed.starts_with("async def ") || trimmed.starts_with("class ")
        }
        Language::TypeScript | Language::JavaScript | Language::Tsx | Language::Jsx => {
          trimmed.starts_with("function ")
            || trimmed.starts_with("async function ")
            || trimmed.starts_with("export function ")
            || trimmed.starts_with("export async function ")
            || trimmed.starts_with("export default function ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("export class ")
            || trimmed.starts_with("export default class ")
            || trimmed.starts_with("interface ")
            || trimmed.starts_with("export interface ")
            || trimmed.starts_with("type ")
            || trimmed.starts_with("export type ")
            || trimmed.contains("const ") && trimmed.contains(" = (")
            || trimmed.contains("const ") && trimmed.contains(" = async (")
        }
        Language::Go => {
          trimmed.starts_with("func ")
            || trimmed.starts_with("type ") && trimmed.contains("struct")
            || trimmed.starts_with("type ") && trimmed.contains("interface")
        }
        _ => false,
      };

      if is_boundary {
        boundaries.push(i);
      }
    }

    boundaries
  }

  /// Determine chunk type from content
  fn determine_chunk_type(&self, content: &str, language: Language) -> ChunkType {
    let trimmed = content.trim();

    match language {
      Language::Rust => {
        if trimmed.contains("fn ") {
          ChunkType::Function
        } else if trimmed.contains("struct ") || trimmed.contains("impl ") {
          ChunkType::Class
        } else if trimmed.starts_with("use ") {
          ChunkType::Import
        } else {
          ChunkType::Block
        }
      }
      Language::Python => {
        if trimmed.contains("def ") {
          ChunkType::Function
        } else if trimmed.contains("class ") {
          ChunkType::Class
        } else if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
          ChunkType::Import
        } else {
          ChunkType::Block
        }
      }
      Language::TypeScript | Language::JavaScript | Language::Tsx | Language::Jsx => {
        if trimmed.contains("function ")
          || (trimmed.contains("const ") && trimmed.contains(" = ("))
          || (trimmed.contains("const ") && trimmed.contains(" => "))
        {
          ChunkType::Function
        } else if trimmed.contains("class ") || trimmed.contains("interface ") {
          ChunkType::Class
        } else if trimmed.starts_with("import ") {
          ChunkType::Import
        } else {
          ChunkType::Block
        }
      }
      Language::Go => {
        if trimmed.contains("func ") {
          ChunkType::Function
        } else if trimmed.contains("type ") && trimmed.contains("struct") {
          ChunkType::Class
        } else if trimmed.starts_with("import ") {
          ChunkType::Import
        } else {
          ChunkType::Block
        }
      }
      _ => ChunkType::Block,
    }
  }

  /// Extract symbol names from content
  fn extract_symbols(&self, content: &str, language: Language) -> Vec<String> {
    let mut symbols = Vec::new();

    for line in content.lines() {
      if let Some(symbol) = self.extract_symbol_from_line(line, language) {
        symbols.push(symbol);
      }
    }

    symbols
  }

  /// Extract symbols from a range of lines
  fn extract_symbols_in_range(&self, lines: &[&str], start: usize, end: usize, language: Language) -> Vec<String> {
    let mut symbols = Vec::new();

    for line in &lines[start..end.min(lines.len())] {
      if let Some(symbol) = self.extract_symbol_from_line(line, language) {
        symbols.push(symbol);
      }
    }

    symbols
  }

  /// Extract a symbol name from a single line
  fn extract_symbol_from_line(&self, line: &str, language: Language) -> Option<String> {
    let trimmed = line.trim();

    match language {
      Language::Rust => {
        // fn name(...
        if let Some(rest) = trimmed.strip_prefix("pub fn ").or(trimmed.strip_prefix("fn ")) {
          return rest.split('(').next().map(|s| s.trim().to_string());
        }
        // struct Name
        if let Some(rest) = trimmed.strip_prefix("pub struct ").or(trimmed.strip_prefix("struct ")) {
          return rest.split([' ', '<', '{']).next().map(|s| s.trim().to_string());
        }
        // impl Name
        if let Some(rest) = trimmed.strip_prefix("impl ") {
          let rest = rest.strip_prefix('<').unwrap_or(rest);
          return rest.split([' ', '<', '{']).next().map(|s| s.trim().to_string());
        }
      }
      Language::Python => {
        if let Some(rest) = trimmed.strip_prefix("def ").or(trimmed.strip_prefix("async def ")) {
          return rest.split('(').next().map(|s| s.trim().to_string());
        }
        if let Some(rest) = trimmed.strip_prefix("class ") {
          return rest.split(['(', ':']).next().map(|s| s.trim().to_string());
        }
      }
      Language::TypeScript | Language::JavaScript | Language::Tsx | Language::Jsx => {
        if let Some(rest) = trimmed
          .strip_prefix("function ")
          .or(trimmed.strip_prefix("async function "))
        {
          return rest.split('(').next().map(|s| s.trim().to_string());
        }
        if let Some(rest) = trimmed
          .strip_prefix("export function ")
          .or(trimmed.strip_prefix("export async function "))
        {
          return rest.split('(').next().map(|s| s.trim().to_string());
        }
        if let Some(rest) = trimmed.strip_prefix("class ").or(trimmed.strip_prefix("export class ")) {
          return rest.split([' ', '{', '<']).next().map(|s| s.trim().to_string());
        }
        if let Some(rest) = trimmed
          .strip_prefix("interface ")
          .or(trimmed.strip_prefix("export interface "))
        {
          return rest.split([' ', '{', '<']).next().map(|s| s.trim().to_string());
        }
        // const name = (
        if trimmed.starts_with("const ") || trimmed.starts_with("export const ") {
          let start = if trimmed.starts_with("export ") {
            "export const "
          } else {
            "const "
          };
          if let Some(rest) = trimmed.strip_prefix(start)
            && (rest.contains(" = (") || rest.contains(" = async ("))
          {
            return rest.split('=').next().map(|s| s.trim().to_string());
          }
        }
      }
      Language::Go => {
        if let Some(rest) = trimmed.strip_prefix("func ") {
          // Skip receiver if present: func (r *Receiver) Name(...
          let rest = if rest.starts_with('(') {
            rest.split(')').nth(1).unwrap_or(rest).trim()
          } else {
            rest
          };
          return rest.split('(').next().map(|s| s.trim().to_string());
        }
        if let Some(rest) = trimmed.strip_prefix("type ") {
          return rest.split_whitespace().next().map(|s| s.to_string());
        }
      }
      _ => {}
    }

    None
  }

  /// Split into evenly-sized chunks when no semantic boundaries found
  fn split_evenly(&mut self, lines: &[&str], file_path: &str, language: Language, file_hash: &str) -> Vec<CodeChunk> {
    let total_lines = lines.len();
    let chunk_count = (total_lines / self.config.target_lines).max(1);
    let chunk_size = total_lines / chunk_count;

    let mut chunks = Vec::new();

    for i in 0..chunk_count {
      let start = i * chunk_size;
      let end = if i == chunk_count - 1 {
        total_lines
      } else {
        (i + 1) * chunk_size
      };

      let content = lines[start..end].join("\n");
      let chunk_type = self.determine_chunk_type(&content, language);
      let tokens_estimate = (content.len() / CHARS_PER_TOKEN) as u32;
      // Extract imports and calls using tree-sitter
      let imports = self.ts_parser.extract_imports(&content, language);
      let calls = self.ts_parser.extract_calls(&content, language);

      chunks.push(CodeChunk {
        id: Uuid::now_v7(),
        file_path: file_path.to_string(),
        content,
        language,
        chunk_type,
        symbols: self.extract_symbols_in_range(lines, start, end, language),
        imports,
        calls,
        start_line: (start + 1) as u32,
        end_line: end as u32,
        file_hash: file_hash.to_string(),
        indexed_at: Utc::now(),
        tokens_estimate,
      });
    }

    chunks
  }

  /// Find the best break point near a boundary
  ///
  /// Looks for natural boundaries (empty lines, closing braces) to avoid
  /// breaking in the middle of a function/block.
  #[allow(dead_code)] // Will be used when integrating into chunk()
  fn find_best_break_point(&self, lines: &[&str], _start: usize, target_end: usize) -> usize {
    // Look within a window of 5 lines before/after the target
    let window = 5;
    let search_start = target_end.saturating_sub(window);
    let search_end = (target_end + window).min(lines.len());

    // First, look for empty lines (natural paragraph breaks)
    for i in (search_start..target_end).rev() {
      if lines[i].trim().is_empty() {
        return i + 1; // Break after the empty line
      }
    }

    // Then look for closing braces/brackets at end of line
    for i in (search_start..target_end).rev() {
      let trimmed = lines[i].trim();
      if trimmed == "}" || trimmed == "};" || trimmed == "end" || trimmed.ends_with("};") {
        return i + 1; // Break after the closing brace
      }
    }

    // Look forward if we didn't find anything backward
    for (i, line) in lines.iter().enumerate().take(search_end).skip(target_end) {
      if line.trim().is_empty() {
        return i + 1;
      }
    }

    // No good break point found, use the original target
    target_end.min(lines.len())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_chunk_small_file() {
    let source = "fn main() {\n    println!(\"Hello\");\n}";
    let mut chunker = Chunker::default();

    let chunks = chunker.chunk(source, "main.rs", Language::Rust, "hash123");

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].chunk_type, ChunkType::Function);
    assert!(chunks[0].symbols.contains(&"main".to_string()));
  }

  #[test]
  fn test_chunk_large_file() {
    // Generate 200-line file
    let source = (0..200)
      .map(|i| format!("fn func{}() {{}}", i))
      .collect::<Vec<_>>()
      .join("\n");

    let mut chunker = Chunker::default();
    let chunks = chunker.chunk(&source, "large.rs", Language::Rust, "hash123");

    // Should have multiple chunks
    assert!(chunks.len() > 1);

    // All chunks should be within limits
    for chunk in &chunks {
      let lines = chunk.end_line - chunk.start_line + 1;
      assert!(lines <= 100);
    }
  }

  #[test]
  fn test_extract_rust_symbols() {
    let source = r#"
pub fn my_function() {}
struct MyStruct {}
impl MyStruct {
    fn method(&self) {}
}
"#;
    let chunker = Chunker::default();
    let symbols = chunker.extract_symbols(source, Language::Rust);

    assert!(symbols.contains(&"my_function".to_string()));
    assert!(symbols.contains(&"MyStruct".to_string()));
    assert!(symbols.contains(&"method".to_string()));
  }

  #[test]
  fn test_extract_python_symbols() {
    let source = r#"
def my_function():
    pass

class MyClass:
    def method(self):
        pass
"#;
    let chunker = Chunker::default();
    let symbols = chunker.extract_symbols(source, Language::Python);

    assert!(symbols.contains(&"my_function".to_string()));
    assert!(symbols.contains(&"MyClass".to_string()));
    assert!(symbols.contains(&"method".to_string()));
  }

  #[test]
  fn test_extract_typescript_symbols() {
    let source = r#"
export function myFunction() {}
export const arrowFunc = () => {}
export class MyClass {}
export interface MyInterface {}
"#;
    let chunker = Chunker::default();
    let symbols = chunker.extract_symbols(source, Language::TypeScript);

    assert!(symbols.contains(&"myFunction".to_string()));
    assert!(symbols.contains(&"MyClass".to_string()));
    assert!(symbols.contains(&"MyInterface".to_string()));
  }

  #[test]
  fn test_determine_chunk_type() {
    let chunker = Chunker::default();

    assert_eq!(
      chunker.determine_chunk_type("fn main() {}", Language::Rust),
      ChunkType::Function
    );
    assert_eq!(
      chunker.determine_chunk_type("struct Foo {}", Language::Rust),
      ChunkType::Class
    );
    assert_eq!(
      chunker.determine_chunk_type("use std::io;", Language::Rust),
      ChunkType::Import
    );
  }

  #[test]
  fn test_chunk_extracts_imports_rust() {
    let source = r#"
use std::collections::HashMap;
use std::io::{Read, Write};

pub fn main() {
    let map = HashMap::new();
    println!("hello");
}
"#;
    let mut chunker = Chunker::default();
    let chunks = chunker.chunk(source, "main.rs", Language::Rust, "hash123");

    assert_eq!(chunks.len(), 1);
    // Verify imports were extracted via tree-sitter
    assert!(
      !chunks[0].imports.is_empty(),
      "imports should be populated: {:?}",
      chunks[0].imports
    );
    assert!(
      chunks[0].imports.iter().any(|i| i.contains("HashMap")),
      "should find HashMap import: {:?}",
      chunks[0].imports
    );
  }

  #[test]
  fn test_chunk_extracts_calls_rust() {
    let source = r#"
pub fn main() {
    let result = helper_function(42);
    println!("Result: {}", result);
    result.unwrap();
}
"#;
    let mut chunker = Chunker::default();
    let chunks = chunker.chunk(source, "main.rs", Language::Rust, "hash123");

    assert_eq!(chunks.len(), 1);
    // Verify calls were extracted via tree-sitter
    assert!(
      !chunks[0].calls.is_empty(),
      "calls should be populated: {:?}",
      chunks[0].calls
    );
    assert!(
      chunks[0].calls.contains(&"helper_function".to_string()),
      "should find helper_function call: {:?}",
      chunks[0].calls
    );
    assert!(
      chunks[0].calls.contains(&"println".to_string()),
      "should find println macro: {:?}",
      chunks[0].calls
    );
  }

  #[test]
  fn test_chunk_extracts_imports_typescript() {
    let source = r#"
import { useState, useEffect } from 'react';
import axios from 'axios';

export function App() {
    const [data, setData] = useState(null);
    return <div>{data}</div>;
}
"#;
    let mut chunker = Chunker::default();
    let chunks = chunker.chunk(source, "app.tsx", Language::Tsx, "hash123");

    assert_eq!(chunks.len(), 1);
    // Verify imports were extracted via tree-sitter
    assert!(
      !chunks[0].imports.is_empty(),
      "imports should be populated: {:?}",
      chunks[0].imports
    );
    assert!(
      chunks[0].imports.contains(&"react".to_string()),
      "should find react import: {:?}",
      chunks[0].imports
    );
    assert!(
      chunks[0].imports.contains(&"axios".to_string()),
      "should find axios import: {:?}",
      chunks[0].imports
    );
  }

  #[test]
  fn test_chunk_extracts_calls_tsx() {
    let source = r#"
import React from 'react';

export function Counter() {
    const [count, setCount] = React.useState(0);

    return (
        <div>
            <Button onClick={() => setCount(count + 1)}>
                Count: {count}
            </Button>
        </div>
    );
}
"#;
    let mut chunker = Chunker::default();
    let chunks = chunker.chunk(source, "counter.tsx", Language::Tsx, "hash123");

    assert_eq!(chunks.len(), 1);
    // Verify calls were extracted via tree-sitter
    assert!(
      !chunks[0].calls.is_empty(),
      "calls should be populated: {:?}",
      chunks[0].calls
    );
    // Should find regular function calls
    assert!(
      chunks[0].calls.contains(&"useState".to_string()),
      "should find useState call: {:?}",
      chunks[0].calls
    );
    // Should find JSX components as calls
    assert!(
      chunks[0].calls.contains(&"Button".to_string()),
      "should find Button JSX component: {:?}",
      chunks[0].calls
    );
  }
}
