use std::cell::RefCell;

use chrono::Utc;
use sha2::{Digest, Sha256};
use tracing::{debug, trace, warn};
use uuid::Uuid;

use super::parser::{Definition, DefinitionKind, TreeSitterParser};
use crate::{
  config::CHARS_PER_TOKEN,
  domain::code::{ChunkType, CodeChunk, Language},
};

/// Compute a content hash for differential re-indexing
///
/// Uses SHA-256 truncated to 16 hex chars for compact storage
/// while still having negligible collision probability.
fn compute_content_hash(content: &str) -> String {
  let result = Sha256::digest(content.as_bytes());
  // Take first 8 bytes (16 hex chars) for compact storage
  format!("{:016x}", u64::from_be_bytes(result[0..8].try_into().unwrap()))
}

// Thread-local parser pool to avoid creating new parsers for each Chunker instance.
// TreeSitterParser is not thread-safe (uses internal mutable state), so thread_local!
// is the appropriate pattern for reusing parsers across multiple chunking operations
// on the same thread.
thread_local! {
  static PARSER_POOL: RefCell<TreeSitterParser> = RefCell::new(TreeSitterParser::new());
}

/// Configuration for the chunker
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
  /// Target number of lines per chunk (for fallback line-based chunking)
  pub target_lines: usize,
  /// Maximum lines per chunk - definitions larger than this get split
  pub max_lines: usize,
  /// Whether to use AST-level chunking (true) or line-based (false)
  pub use_ast_chunking: bool,
}

impl Default for ChunkerConfig {
  fn default() -> Self {
    Self {
      target_lines: 50,
      max_lines: 150, // Increased for AST chunking - allow larger definitions
      use_ast_chunking: true,
    }
  }
}

/// AST-aware code chunker
///
/// Chunks code by semantic definitions (functions, classes, structs) using tree-sitter.
/// Falls back to line-based chunking for unsupported languages.
///
/// Uses a thread-local parser pool to avoid creating new parsers for each Chunker instance,
/// significantly reducing memory and initialization overhead when processing many files.
pub struct Chunker {
  config: ChunkerConfig,
  /// Whether to use the thread-local parser pool (default: true)
  /// Set to false for testing or when an isolated parser is needed
  use_parser_pool: bool,
  /// Fallback owned parser (only used when use_parser_pool is false)
  owned_parser: Option<TreeSitterParser>,
}

impl Default for Chunker {
  fn default() -> Self {
    Self::new(ChunkerConfig::default())
  }
}

impl Clone for Chunker {
  fn clone(&self) -> Self {
    // Create a new Chunker with the same config.
    // If the original uses an owned parser, create a new owned parser.
    // Otherwise, use the thread-local pool (default).
    Self {
      config: self.config.clone(),
      use_parser_pool: self.use_parser_pool,
      owned_parser: if self.use_parser_pool {
        None
      } else {
        Some(TreeSitterParser::new())
      },
    }
  }
}

impl Chunker {
  pub fn new(config: ChunkerConfig) -> Self {
    Self {
      config,
      use_parser_pool: true,
      owned_parser: None,
    }
  }

  #[cfg(test)]
  /// Create a chunker with an isolated parser (useful for testing)
  pub fn with_owned_parser(config: ChunkerConfig) -> Self {
    Self {
      config,
      use_parser_pool: false,
      owned_parser: Some(TreeSitterParser::new()),
    }
  }

  /// Execute a function with access to the parser.
  /// Uses thread-local pool or owned parser based on configuration.
  fn with_parser<F, R>(&mut self, f: F) -> R
  where
    F: FnOnce(&mut TreeSitterParser) -> R,
  {
    if self.use_parser_pool {
      PARSER_POOL.with(|pool| {
        let mut parser = pool.borrow_mut();
        f(&mut parser)
      })
    } else {
      f(self
        .owned_parser
        .as_mut()
        .expect("owned_parser should be set when use_parser_pool is false"))
    }
  }

  /// Check if the parser supports a language
  fn supports_language(&self, lang: Language) -> bool {
    if self.use_parser_pool {
      PARSER_POOL.with(|pool| pool.borrow().supports_language(lang))
    } else {
      self.owned_parser.as_ref().is_some_and(|p| p.supports_language(lang))
    }
  }

  /// Chunk source code with incremental parsing support.
  ///
  /// When `old_content` is provided, uses incremental tree-sitter parsing which
  /// is significantly faster for small edits (O(log n) vs O(n) for full reparse).
  ///
  /// The parser computes the diff between old and new content and only reparses
  /// the affected portions of the syntax tree. Falls back to full parse if:
  /// - No old content is provided
  /// - Changes affect more than 50% of the file (full parse is faster in that case)
  ///
  /// # Arguments
  /// * `source` - The new file content to parse
  /// * `file_path` - Relative path to the file
  /// * `language` - Programming language
  /// * `file_hash` - Hash of the file for cache invalidation
  /// * `old_content` - Optional previous content for incremental parsing
  #[tracing::instrument(level = "trace", skip(self, source, old_content), fields(
    file = %file_path,
    language = ?language,
    size = source.len(),
    incremental = old_content.is_some()
  ))]
  pub fn chunk(
    &mut self,
    source: &str,
    file_path: &str,
    language: Language,
    file_hash: &str,
    old_content: Option<&str>,
  ) -> Vec<CodeChunk> {
    debug!(
      file = %file_path,
      language = ?language,
      size_bytes = source.len(),
      incremental = old_content.is_some(),
      "Starting file chunking"
    );

    // Skip files with very few newlines - likely minified code, data files, or assets.
    // These are expensive to embed and produce poor quality chunks.
    // Heuristic: >10KB and >500 chars per line = probably minified/data/asset
    let line_count = source.lines().count();
    let chars_per_line = if line_count > 0 {
      source.len() / line_count
    } else {
      source.len()
    };
    if source.len() > 10_000 && chars_per_line > 500 {
      debug!(
        file = %file_path,
        size_bytes = source.len(),
        line_count,
        chars_per_line,
        "Skipping file with very few newlines (likely minified code or data)"
      );
      return Vec::new();
    }

    // Skip highly repetitive files - likely data dumps (JSON fixtures, CSV-like data, etc.)
    // These produce poor quality embeddings and waste resources.
    // Only check files >20KB to avoid overhead on small files.
    if source.len() > 20_000 {
      use std::collections::HashSet;

      let mut unique_lines = HashSet::new();
      let mut total_lines_checked = 0usize;

      for line in source.lines() {
        total_lines_checked += 1;
        unique_lines.insert(line);

        // Early exit: after 500 lines, if <20% unique, it's repetitive data
        if total_lines_checked >= 500 && unique_lines.len() * 5 < total_lines_checked {
          debug!(
            file = %file_path,
            size_bytes = source.len(),
            lines_checked = total_lines_checked,
            unique_lines = unique_lines.len(),
            "Skipping highly repetitive file (likely data dump)"
          );
          return Vec::new();
        }
      }

      // Final check for files with 100-500 lines
      if total_lines_checked > 100 && unique_lines.len() * 5 < total_lines_checked {
        debug!(
          file = %file_path,
          size_bytes = source.len(),
          total_lines = total_lines_checked,
          unique_lines = unique_lines.len(),
          "Skipping highly repetitive file (likely data dump)"
        );
        return Vec::new();
      }
    }

    // Warn about large files that may take longer to process
    if source.len() > 100_000 {
      debug!(
        file = %file_path,
        size_bytes = source.len(),
        "Processing large file (>100KB)"
      );
    }

    // Don't clear tree cache when using incremental parsing with old content
    // This allows the parser to reuse the previous tree
    if old_content.is_none() {
      self.with_parser(|p| p.clear_cache());
    }

    let lines: Vec<&str> = source.lines().collect();
    let total_lines = lines.len();

    // Try AST-level chunking if enabled and language is supported
    if self.config.use_ast_chunking && self.supports_language(language) {
      trace!(file = %file_path, "Attempting AST-level chunking");
      let chunks = self.chunk_by_definitions_incremental(source, &lines, file_path, language, file_hash, old_content);
      if !chunks.is_empty() {
        trace!(
          file = %file_path,
          chunks = chunks.len(),
          "AST chunking produced chunks"
        );
        return chunks;
      }
      trace!(file = %file_path, "No definitions found, falling back to line-based chunking");
      // Fall through to line-based if no definitions found
    }

    // Fallback: line-based chunking
    trace!(file = %file_path, total_lines = total_lines, "Using line-based chunking");
    self.chunk_by_lines(source, &lines, file_path, language, file_hash, total_lines)
  }

  /// Chunk code by AST definitions with incremental parsing support
  #[tracing::instrument(level = "trace", skip(self, source, lines, old_content), fields(file = %file_path, lines = lines.len()))]
  fn chunk_by_definitions_incremental(
    &mut self,
    source: &str,
    lines: &[&str],
    file_path: &str,
    language: Language,
    file_hash: &str,
    old_content: Option<&str>,
  ) -> Vec<CodeChunk> {
    // Use incremental parsing if old content is available
    let parsed = self.with_parser(|p| {
      if old_content.is_some() {
        p.parse_file_incremental(source, language, None)
      } else {
        p.parse_file(source, language)
      }
    });

    if !parsed {
      // Debug level, not warn - some files may legitimately fail to parse
      // (e.g., partial files, generated code with syntax errors)
      debug!(file = %file_path, language = ?language, "Parse failed, falling back to line-based chunking");
      return Vec::new();
    }

    // Extract definitions using cached tree
    let definitions = self.with_parser(|p| p.extract_definitions(source, language));

    if definitions.is_empty() {
      trace!(file = %file_path, "No definitions extracted from AST");
      return Vec::new();
    }

    trace!(
      file = %file_path,
      definitions = definitions.len(),
      "Extracted definitions from AST"
    );

    // Extract file-level imports using the cached tree
    let file_imports = self.with_parser(|p| p.extract_imports(source, language));

    // Sort definitions by start line
    let mut defs: Vec<_> = definitions.into_iter().collect();
    defs.sort_by_key(|d| d.start_line);

    let mut chunks = Vec::new();
    let total_lines = lines.len();

    // Use a bitmap (Vec<bool>) instead of HashSet for O(1) line coverage tracking.
    // Index 0 is unused since lines are 1-indexed; index i represents line i.
    // This avoids HashSet allocations and provides faster insert/lookup.
    let mut covered_lines: Vec<bool> = vec![false; total_lines + 1];

    for def in &defs {
      // Skip if this definition is entirely contained within already-processed lines
      // (handles nested definitions - we keep the outer one)
      let start = def.start_line as usize;
      let end = def.end_line as usize;
      let is_subset = (start..=end.min(total_lines)).all(|i| covered_lines[i]);
      if is_subset {
        continue;
      }

      let chunk = self.create_definition_chunk(def, source, lines, file_path, language, file_hash, &file_imports);

      // Mark these lines as covered (O(1) per line)
      for line in covered_lines.iter_mut().take(end.min(total_lines) + 1).skip(start) {
        *line = true;
      }

      chunks.push(chunk);
    }

    // Handle any remaining code not covered by definitions (imports, constants, etc.)
    let total_lines = total_lines as u32;
    let uncovered: Vec<u32> = (1..=total_lines).filter(|&l| !covered_lines[l as usize]).collect();

    if !uncovered.is_empty() {
      // Group contiguous uncovered regions
      let regions = self.find_contiguous_regions(&uncovered);
      for (start, end) in regions {
        // Only create chunk if region is meaningful (not just whitespace)
        let region_content: String = lines[(start - 1) as usize..end as usize].join("\n");
        if region_content.trim().is_empty() {
          continue;
        }

        // Skip regions that are mostly structural characters (braces, newlines, etc.)
        // These have no semantic value for embedding/search
        let meaningful_chars = region_content
          .chars()
          .filter(|c| c.is_alphanumeric() || *c == '_')
          .count();
        if meaningful_chars < 10 {
          trace!(
            file = %file_path,
            start_line = start,
            end_line = end,
            meaningful_chars,
            "Skipping region with insufficient meaningful content"
          );
          continue;
        }

        // Check if it's primarily imports
        let is_imports = region_content.lines().all(|l| {
          let t = l.trim();
          t.is_empty()
            || t.starts_with("use ")
            || t.starts_with("import ")
            || t.starts_with("from ")
            || t.starts_with("//")
            || t.starts_with("#")
        });

        if is_imports && region_content.lines().filter(|l| !l.trim().is_empty()).count() < 3 {
          // Skip tiny import-only regions - they'll be included via file_imports context
          continue;
        }

        let chunk = self.create_region_chunk(source, &region_content, start, end, file_path, language, file_hash);
        chunks.push(chunk);
      }
    }

    // Sort chunks by start line for consistent ordering
    chunks.sort_by_key(|c| c.start_line);

    trace!(
      file = %file_path,
      chunks = chunks.len(),
      "File chunking complete"
    );

    chunks
  }

  #[allow(clippy::too_many_arguments)]
  /// Create a chunk from a definition
  fn create_definition_chunk(
    &mut self,
    def: &Definition,
    source: &str,
    lines: &[&str],
    file_path: &str,
    language: Language,
    file_hash: &str,
    file_imports: &[String],
  ) -> CodeChunk {
    let start_idx = (def.start_line - 1) as usize;
    let end_idx = (def.end_line as usize).min(lines.len());

    // Look for docstring/comments preceding the definition
    let (docstring, doc_start_line) = self.extract_docstring(lines, start_idx, language);

    // Adjust start to include docstring
    let actual_start = doc_start_line.unwrap_or(start_idx);
    let content = lines[actual_start..end_idx].join("\n");

    // Extract the signature (first line of definition, possibly multi-line)
    let signature = self.extract_signature(lines, start_idx, language);

    // Extract visibility
    let visibility = self.extract_visibility(&signature, language);

    // Extract imports and calls for this chunk using cached tree (no re-parsing)
    let (chunk_imports, calls) = self
      .with_parser(|p| p.extract_imports_and_calls_in_range(source, language, (actual_start + 1) as u32, def.end_line));

    // Combine chunk-level imports with file-level imports for relationship tracking
    // This ensures that functions can be linked via the imports used in their file
    let mut combined_imports: Vec<String> = chunk_imports.clone();
    for imp in file_imports {
      if !combined_imports.contains(imp) {
        combined_imports.push(imp.clone());
      }
    }

    // Determine chunk type from definition kind
    let chunk_type = match def.kind {
      DefinitionKind::Function | DefinitionKind::Method => ChunkType::Function,
      DefinitionKind::Class | DefinitionKind::Struct | DefinitionKind::Interface | DefinitionKind::Trait => {
        ChunkType::Class
      }
      DefinitionKind::Module => ChunkType::Module,
      _ => ChunkType::Block,
    };

    // Create enriched embedding text
    let embedding_text = self.create_embedding_text(
      &def.name,
      &def.kind,
      def.parent.as_deref(),
      signature.as_deref(),
      docstring.as_deref(),
      &chunk_imports,
      file_imports,
      &calls,
      file_path,
      &content,
    );

    let tokens_estimate = (content.len() / CHARS_PER_TOKEN) as u32;

    let content_hash = compute_content_hash(&content);

    CodeChunk {
      id: Uuid::now_v7(),
      file_path: file_path.to_string(),
      content,
      language,
      chunk_type,
      symbols: vec![def.name.clone()],
      imports: combined_imports,
      calls,
      start_line: (actual_start + 1) as u32,
      end_line: def.end_line,
      file_hash: file_hash.to_string(),
      indexed_at: Utc::now(),
      tokens_estimate,
      definition_kind: Some(format!("{:?}", def.kind).to_lowercase()),
      definition_name: Some(def.name.clone()),
      visibility,
      signature,
      docstring,
      parent_definition: def.parent.clone(),
      embedding_text: Some(embedding_text),
      content_hash: Some(content_hash),
      caller_count: 0, // Computed during reference extraction
      callee_count: 0, // Computed during reference extraction
    }
  }

  #[allow(clippy::too_many_arguments)]
  /// Create a chunk from a non-definition region (imports, constants, etc.)
  fn create_region_chunk(
    &mut self,
    source: &str,
    content: &str,
    start_line: u32,
    end_line: u32,
    file_path: &str,
    language: Language,
    file_hash: &str,
  ) -> CodeChunk {
    let chunk_type = self.determine_chunk_type(content, language);
    // Use cached tree with line range for efficiency
    let (imports, calls) =
      self.with_parser(|p| p.extract_imports_and_calls_in_range(source, language, start_line, end_line));
    let symbols = self.extract_symbols(content, language);
    let tokens_estimate = (content.len() / CHARS_PER_TOKEN) as u32;
    let content_hash = compute_content_hash(content);

    CodeChunk {
      id: Uuid::now_v7(),
      file_path: file_path.to_string(),
      content: content.to_string(),
      language,
      chunk_type,
      symbols,
      imports,
      calls,
      start_line,
      end_line,
      file_hash: file_hash.to_string(),
      indexed_at: Utc::now(),
      tokens_estimate,
      definition_kind: None,
      definition_name: None,
      visibility: None,
      signature: None,
      docstring: None,
      parent_definition: None,
      embedding_text: None,
      content_hash: Some(content_hash),
      caller_count: 0,
      callee_count: 0,
    }
  }

  /// Extract docstring/comments preceding a definition
  fn extract_docstring(&self, lines: &[&str], def_start: usize, language: Language) -> (Option<String>, Option<usize>) {
    if def_start == 0 {
      return (None, None);
    }

    let mut doc_lines = Vec::new();
    let mut i = def_start - 1;

    // Look backwards for doc comments
    loop {
      let line = lines[i].trim();

      let is_doc_comment = match language {
        Language::Rust => line.starts_with("///") || line.starts_with("//!") || line.starts_with("#["),
        Language::Python => {
          // Python docstrings are inside the function, but we can catch decorators and comments
          line.starts_with('#') || line.starts_with('@')
        }
        Language::TypeScript | Language::JavaScript | Language::Tsx | Language::Jsx => {
          line.starts_with("/**") || line.starts_with("*") || line.starts_with("//") || line.starts_with("@")
        }
        Language::Go => line.starts_with("//"),
        Language::Java => {
          line.starts_with("/**") || line.starts_with("*") || line.starts_with("//") || line.starts_with("@")
        }
        _ => line.starts_with("//") || line.starts_with("#"),
      };

      // Also accept empty lines within the doc block
      let is_empty = line.is_empty();

      if is_doc_comment {
        doc_lines.push(lines[i]);
      } else if is_empty && !doc_lines.is_empty() {
        // Empty line in the middle of docs
        doc_lines.push(lines[i]);
      } else if !is_empty {
        // Hit non-doc content
        break;
      }

      if i == 0 {
        break;
      }
      i -= 1;
    }

    if doc_lines.is_empty() {
      return (None, None);
    }

    doc_lines.reverse();

    // Trim trailing empty lines
    while doc_lines.last().is_some_and(|l| l.trim().is_empty()) {
      doc_lines.pop();
    }

    if doc_lines.is_empty() {
      return (None, None);
    }

    let doc_start = def_start - doc_lines.len();
    let docstring = doc_lines.join("\n");

    (Some(docstring), Some(doc_start))
  }

  /// Extract the function/class signature
  fn extract_signature(&self, lines: &[&str], def_start: usize, language: Language) -> Option<String> {
    if def_start >= lines.len() {
      return None;
    }

    let first_line = lines[def_start];

    // For single-line signatures, just return the first line
    if self.is_complete_signature(first_line, language) {
      return Some(first_line.to_string());
    }

    // For multi-line signatures, collect until we find the body start
    let mut signature_lines = vec![first_line];
    for line in lines.iter().skip(def_start + 1) {
      signature_lines.push(*line);

      // Check if we've reached the body
      let trimmed = line.trim();
      if trimmed.ends_with('{') || trimmed.ends_with(':') || trimmed == "{" {
        break;
      }

      // Don't go too far
      if signature_lines.len() > 10 {
        break;
      }
    }

    Some(signature_lines.join("\n"))
  }

  /// Check if a line contains a complete signature
  fn is_complete_signature(&self, line: &str, language: Language) -> bool {
    let trimmed = line.trim();
    match language {
      Language::Rust => trimmed.ends_with('{') || trimmed.ends_with(';'),
      Language::Python => trimmed.ends_with(':'),
      Language::Go => trimmed.ends_with('{'),
      Language::TypeScript | Language::JavaScript | Language::Tsx | Language::Jsx => {
        trimmed.ends_with('{') || trimmed.ends_with(';')
      }
      _ => trimmed.ends_with('{') || trimmed.ends_with(':'),
    }
  }

  /// Extract visibility modifier from signature
  fn extract_visibility(&self, signature: &Option<String>, language: Language) -> Option<String> {
    let sig = signature.as_ref()?;
    let trimmed = sig.trim();

    match language {
      Language::Rust => {
        if trimmed.starts_with("pub(crate)") {
          Some("pub(crate)".to_string())
        } else if trimmed.starts_with("pub(super)") {
          Some("pub(super)".to_string())
        } else if trimmed.starts_with("pub ") {
          Some("pub".to_string())
        } else {
          Some("private".to_string())
        }
      }
      Language::TypeScript | Language::JavaScript | Language::Tsx | Language::Jsx => {
        if trimmed.starts_with("export default") {
          Some("export default".to_string())
        } else if trimmed.starts_with("export") {
          Some("export".to_string())
        } else if trimmed.contains("private ") {
          Some("private".to_string())
        } else if trimmed.contains("protected ") {
          Some("protected".to_string())
        } else if trimmed.contains("public ") {
          Some("public".to_string())
        } else {
          None
        }
      }
      Language::Python => {
        // Python uses naming convention
        let first_word = trimmed.split_whitespace().nth(1)?;
        if first_word.starts_with("__") && !first_word.ends_with("__") {
          Some("private".to_string())
        } else if first_word.starts_with('_') {
          Some("protected".to_string())
        } else {
          Some("public".to_string())
        }
      }
      Language::Go => {
        // Go uses capitalization
        let name = trimmed
          .split_whitespace()
          .find(|w| w.chars().next().is_some_and(|c| c.is_alphabetic()))?;
        if name.chars().next()?.is_uppercase() {
          Some("public".to_string())
        } else {
          Some("private".to_string())
        }
      }
      Language::Java => {
        if trimmed.starts_with("public ") {
          Some("public".to_string())
        } else if trimmed.starts_with("private ") {
          Some("private".to_string())
        } else if trimmed.starts_with("protected ") {
          Some("protected".to_string())
        } else {
          Some("package-private".to_string())
        }
      }
      _ => None,
    }
  }

  #[allow(clippy::too_many_arguments)]
  /// Create enriched text for embedding.
  ///
  /// Uses pre-allocated String with write!() to minimize allocations.
  /// Previously used 8-10 format!() calls per chunk; now uses a single allocation.
  fn create_embedding_text(
    &self,
    name: &str,
    kind: &DefinitionKind,
    parent: Option<&str>,
    signature: Option<&str>,
    docstring: Option<&str>,
    chunk_imports: &[String],
    file_imports: &[String],
    calls: &[String],
    file_path: &str,
    code: &str,
  ) -> String {
    use std::fmt::Write;

    // Estimate capacity: headers + signature + doc preview + imports + calls + code
    // This avoids reallocations during construction
    let estimated_size = 100 // headers and labels
      + name.len()
      + file_path.len()
      + parent.map_or(0, |p| p.len() + 12) // [PARENT] + name + newline
      + signature.map_or(0, |s| s.len())
      + docstring.map_or(0, |d| d.len().min(500)) // first 5 lines ~ 500 chars max
      + chunk_imports.iter().take(10).map(|s| s.len() + 2).sum::<usize>()
      + file_imports.iter().take(10).map(|s| s.len() + 2).sum::<usize>()
      + calls.iter().take(15).map(|s| s.len() + 2).sum::<usize>()
      + code.len()
      + 50; // newlines and separators

    let mut result = String::with_capacity(estimated_size);

    // Definition header
    let _ = writeln!(result, "[DEFINITION] {:?}: {}", kind, name);

    // Parent definition for nested items (methods inside class/impl)
    if let Some(p) = parent {
      let _ = writeln!(result, "[PARENT] {}", p);
    }

    // File path for context
    let _ = writeln!(result, "[FILE] {}", file_path);

    // Signature
    if let Some(sig) = signature {
      result.push_str("[SIGNATURE] ");
      // Clean up multi-line signatures inline
      let mut first = true;
      for line in sig.lines() {
        if !first {
          result.push(' ');
        }
        result.push_str(line.trim());
        first = false;
      }
      result.push('\n');
    }

    // Docstring (truncated to first 5 lines)
    if let Some(doc) = docstring {
      result.push_str("[DOC] ");
      let mut first = true;
      for line in doc.lines().take(5) {
        if !first {
          result.push(' ');
        }
        result.push_str(line);
        first = false;
      }
      result.push('\n');
    }

    // Imports (deduplicate via HashSet, take first 10)
    let all_imports: std::collections::HashSet<&str> = chunk_imports
      .iter()
      .chain(file_imports.iter())
      .map(|s| s.as_str())
      .collect();
    if !all_imports.is_empty() {
      result.push_str("[IMPORTS] ");
      let mut first = true;
      for import in all_imports.iter().take(10) {
        if !first {
          result.push_str(", ");
        }
        result.push_str(import);
        first = false;
      }
      result.push('\n');
    }

    // Calls (take first 15)
    if !calls.is_empty() {
      result.push_str("[CALLS] ");
      let mut first = true;
      for call in calls.iter().take(15) {
        if !first {
          result.push_str(", ");
        }
        result.push_str(call);
        first = false;
      }
      result.push('\n');
    }

    // Separator before code
    result.push_str("---\n");

    // The actual code
    result.push_str(code);

    result
  }

  /// Find contiguous regions from a list of line numbers
  fn find_contiguous_regions(&self, lines: &[u32]) -> Vec<(u32, u32)> {
    if lines.is_empty() {
      return Vec::new();
    }

    let mut regions = Vec::new();
    let mut start = lines[0];
    let mut end = lines[0];

    for &line in lines.iter().skip(1) {
      if line == end + 1 {
        end = line;
      } else {
        regions.push((start, end));
        start = line;
        end = line;
      }
    }
    regions.push((start, end));

    regions
  }

  /// Fallback: Line-based chunking
  #[tracing::instrument(level = "trace", skip(self, source, lines), fields(file = %file_path, total_lines = total_lines))]
  fn chunk_by_lines(
    &mut self,
    source: &str,
    lines: &[&str],
    file_path: &str,
    language: Language,
    file_hash: &str,
    total_lines: usize,
  ) -> Vec<CodeChunk> {
    trace!(
      file = %file_path,
      total_lines = total_lines,
      target_lines = self.config.target_lines,
      "Using line-based chunking strategy"
    );

    // Small files: single chunk
    if total_lines <= self.config.max_lines {
      trace!(
        file = %file_path,
        total_lines = total_lines,
        max_lines = self.config.max_lines,
        "Small file, creating single chunk"
      );
      let chunk_type = self.determine_chunk_type(source, language);
      let (imports, calls) = self.with_parser(|p| p.extract_imports_and_calls(source, language));
      let symbols = self.extract_symbols(source, language);
      let content_hash = compute_content_hash(source);

      return vec![CodeChunk {
        id: Uuid::now_v7(),
        file_path: file_path.to_string(),
        content: source.to_string(),
        language,
        chunk_type,
        symbols,
        imports,
        calls,
        start_line: 1,
        end_line: total_lines as u32,
        file_hash: file_hash.to_string(),
        indexed_at: Utc::now(),
        tokens_estimate: (source.len() / CHARS_PER_TOKEN) as u32,
        definition_kind: None,
        definition_name: None,
        visibility: None,
        signature: None,
        docstring: None,
        parent_definition: None,
        embedding_text: None,
        content_hash: Some(content_hash),
        caller_count: 0,
        callee_count: 0,
      }];
    }

    // Find semantic boundaries
    let boundaries = self.find_boundaries(lines, language);
    let mut chunks = Vec::new();
    let mut current_start = 0usize;

    for boundary in boundaries {
      let chunk_lines = boundary - current_start;

      // Accumulate until we hit target
      if chunk_lines >= self.config.target_lines {
        let content = lines[current_start..boundary].join("\n");
        let chunk_type = self.determine_chunk_type(&content, language);
        let tokens_estimate = (content.len() / CHARS_PER_TOKEN) as u32;
        let (imports, calls) = self.with_parser(|p| p.extract_imports_and_calls(&content, language));
        let content_hash = compute_content_hash(&content);

        chunks.push(CodeChunk {
          id: Uuid::now_v7(),
          file_path: file_path.to_string(),
          content,
          language,
          chunk_type,
          symbols: self.extract_symbols_in_range(lines, current_start, boundary, language),
          imports,
          calls,
          start_line: (current_start + 1) as u32,
          end_line: boundary as u32,
          file_hash: file_hash.to_string(),
          indexed_at: Utc::now(),
          tokens_estimate,
          definition_kind: None,
          definition_name: None,
          visibility: None,
          signature: None,
          docstring: None,
          parent_definition: None,
          embedding_text: None,
          content_hash: Some(content_hash),
          caller_count: 0,
          callee_count: 0,
        });

        current_start = boundary;
      }
    }

    // Final chunk
    if current_start < total_lines {
      let content = lines[current_start..].join("\n");
      let chunk_type = self.determine_chunk_type(&content, language);
      let tokens_estimate = (content.len() / CHARS_PER_TOKEN) as u32;
      let (imports, calls) = self.with_parser(|p| p.extract_imports_and_calls(&content, language));
      let content_hash = compute_content_hash(&content);

      chunks.push(CodeChunk {
        id: Uuid::now_v7(),
        file_path: file_path.to_string(),
        content,
        language,
        chunk_type,
        symbols: self.extract_symbols_in_range(lines, current_start, total_lines, language),
        imports,
        calls,
        start_line: (current_start + 1) as u32,
        end_line: total_lines as u32,
        file_hash: file_hash.to_string(),
        indexed_at: Utc::now(),
        tokens_estimate,
        definition_kind: None,
        definition_name: None,
        visibility: None,
        signature: None,
        docstring: None,
        parent_definition: None,
        embedding_text: None,
        content_hash: Some(content_hash),
        caller_count: 0,
        callee_count: 0,
      });
    }

    // If no chunks were created, split evenly
    if chunks.is_empty() {
      trace!(file = %file_path, "No semantic boundaries found, splitting evenly");
      self.split_evenly(lines, file_path, language, file_hash)
    } else {
      trace!(
        file = %file_path,
        chunks = chunks.len(),
        "Line-based chunking complete"
      );
      chunks
    }
  }

  /// Find semantic boundaries in the code (for line-based fallback)
  fn find_boundaries(&self, lines: &[&str], language: Language) -> Vec<usize> {
    let mut boundaries = Vec::new();

    for (i, line) in lines.iter().enumerate() {
      let trimmed = line.trim();

      if trimmed.is_empty() {
        continue;
      }

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
        if let Some(rest) = trimmed.strip_prefix("pub fn ").or(trimmed.strip_prefix("fn ")) {
          return rest.split('(').next().map(|s| s.trim().to_string());
        }
        if let Some(rest) = trimmed.strip_prefix("pub struct ").or(trimmed.strip_prefix("struct ")) {
          return rest.split([' ', '<', '{']).next().map(|s| s.trim().to_string());
        }
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
      let (imports, calls) = self.with_parser(|p| p.extract_imports_and_calls(&content, language));
      let content_hash = compute_content_hash(&content);

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
        definition_kind: None,
        definition_name: None,
        visibility: None,
        signature: None,
        docstring: None,
        parent_definition: None,
        embedding_text: None,
        content_hash: Some(content_hash),
        caller_count: 0,
        callee_count: 0,
      });
    }

    chunks
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_chunk_small_file() {
    let source = "fn main() {\n    println!(\"Hello\");\n}";
    let mut chunker = Chunker::default();

    let chunks = chunker.chunk(source, "main.rs", Language::Rust, "hash123", None);

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].chunk_type, ChunkType::Function);
    assert!(chunks[0].symbols.contains(&"main".to_string()));
  }

  #[test]
  fn test_ast_chunking_extracts_definitions() {
    let source = r#"
use std::io;

/// A helper function that does something important
pub fn helper_function() {
    println!("hello");
}

/// Another function
fn private_function(x: i32) -> i32 {
    x * 2
}

pub struct MyStruct {
    field: i32,
}
"#;
    let mut chunker = Chunker::default();
    let chunks = chunker.chunk(source, "lib.rs", Language::Rust, "hash123", None);

    // Should have chunks for each definition
    assert!(chunks.len() >= 2, "Expected at least 2 chunks, got {}", chunks.len());

    // Check that we have function chunks with metadata
    let helper_chunk = chunks
      .iter()
      .find(|c| c.symbols.contains(&"helper_function".to_string()));
    assert!(helper_chunk.is_some(), "Should find helper_function chunk");

    let helper = helper_chunk.unwrap();
    assert_eq!(helper.definition_kind.as_deref(), Some("function"));
    assert_eq!(helper.definition_name.as_deref(), Some("helper_function"));
    assert!(helper.docstring.is_some(), "Should extract docstring");
    assert!(helper.embedding_text.is_some(), "Should have enriched embedding text");
  }

  #[test]
  fn test_ast_chunking_typescript() {
    let source = r#"
import { useState } from 'react';

/**
 * A counter component
 */
export function Counter() {
    const [count, setCount] = useState(0);
    return <div>{count}</div>;
}

interface Config {
    enabled: boolean;
}
"#;
    let mut chunker = Chunker::default();
    let chunks = chunker.chunk(source, "Counter.tsx", Language::Tsx, "hash123", None);

    let counter_chunk = chunks.iter().find(|c| c.symbols.contains(&"Counter".to_string()));
    assert!(counter_chunk.is_some(), "Should find Counter chunk");

    let counter = counter_chunk.unwrap();
    assert!(
      counter.visibility.as_deref() == Some("export"),
      "Should detect export visibility"
    );
  }

  #[test]
  fn test_enriched_embedding_text() {
    let source = r#"
use std::collections::HashMap;

/// Calculates the total price of items
pub fn calculate_total(items: Vec<Item>) -> f64 {
    items.iter().map(|i| i.price).sum()
}
"#;
    let mut chunker = Chunker::default();
    let chunks = chunker.chunk(source, "pricing.rs", Language::Rust, "hash123", None);

    let calc_chunk = chunks
      .iter()
      .find(|c| c.symbols.contains(&"calculate_total".to_string()));
    assert!(calc_chunk.is_some());

    let embedding_text = calc_chunk.unwrap().embedding_text.as_ref().unwrap();

    // Check that embedding text contains structured information
    assert!(embedding_text.contains("[DEFINITION]"), "Should have definition header");
    assert!(embedding_text.contains("[FILE]"), "Should have file path");
    assert!(
      embedding_text.contains("calculate_total"),
      "Should contain function name"
    );
    assert!(embedding_text.contains("---"), "Should have separator before code");
  }

  #[test]
  fn test_parent_definition_in_chunk() {
    let source = r#"
struct UserRepo {
    db: Database,
}

impl UserRepo {
    /// Saves a user to the database
    pub fn save(&self, user: &User) -> Result<()> {
        self.db.insert(user)
    }

    pub fn load(&self, id: u64) -> Option<User> {
        self.db.get(id)
    }
}
"#;
    let mut chunker = Chunker::default();
    let chunks = chunker.chunk(source, "repo.rs", Language::Rust, "hash123", None);

    // Find the 'save' method chunk
    let save_chunk = chunks.iter().find(|c| c.symbols.contains(&"save".to_string()));
    assert!(
      save_chunk.is_some(),
      "should find save method chunk, chunks: {:?}",
      chunks.iter().map(|c| &c.symbols).collect::<Vec<_>>()
    );

    let save = save_chunk.unwrap();

    // Verify parent_definition is set correctly
    assert_eq!(
      save.parent_definition.as_deref(),
      Some("UserRepo"),
      "save method should have UserRepo as parent_definition"
    );

    // Verify embedding text includes [PARENT]
    let embedding_text = save.embedding_text.as_ref().unwrap();
    assert!(
      embedding_text.contains("[PARENT] UserRepo"),
      "embedding text should contain [PARENT] UserRepo, got: {}",
      embedding_text
    );

    // Verify the struct itself has no parent
    let struct_chunk = chunks.iter().find(|c| c.symbols.contains(&"UserRepo".to_string()));
    assert!(struct_chunk.is_some(), "should find UserRepo struct chunk");
    assert_eq!(
      struct_chunk.unwrap().parent_definition,
      None,
      "struct should have no parent"
    );
  }

  #[test]
  fn test_chunk_large_file() {
    // Generate 200-line file
    let source = (0..200)
      .map(|i| format!("fn func{}() {{}}", i))
      .collect::<Vec<_>>()
      .join("\n");

    let mut chunker = Chunker::default();
    let chunks = chunker.chunk(&source, "large.rs", Language::Rust, "hash123", None);

    // Should have multiple chunks (one per function with AST chunking)
    assert!(chunks.len() > 1);
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
    let chunks = chunker.chunk(source, "main.rs", Language::Rust, "hash123", None);

    // Find the main function chunk
    let main_chunk = chunks.iter().find(|c| c.symbols.contains(&"main".to_string()));
    assert!(main_chunk.is_some(), "Should find main chunk");
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
    let chunks = chunker.chunk(source, "main.rs", Language::Rust, "hash123", None);

    let main_chunk = chunks.iter().find(|c| c.symbols.contains(&"main".to_string())).unwrap();

    assert!(
      !main_chunk.calls.is_empty(),
      "calls should be populated: {:?}",
      main_chunk.calls
    );
    assert!(
      main_chunk.calls.contains(&"helper_function".to_string()),
      "should find helper_function call: {:?}",
      main_chunk.calls
    );
  }

  #[test]
  fn test_skip_repetitive_data_files() {
    // Simulate a large repetitive JSON-like data file (>20KB, <20% unique lines)
    // This should be skipped by the repetition detection heuristic
    let repeated_line = r#"  {"id": 1, "name": "test", "value": 12345},"#;
    let unique_header = r#"{"data": ["#;
    let unique_footer = r#"]}"#;

    // Create ~25KB of repetitive content (~600 identical lines)
    let mut source = String::with_capacity(30_000);
    source.push_str(unique_header);
    source.push('\n');
    for _ in 0..600 {
      source.push_str(repeated_line);
      source.push('\n');
    }
    source.push_str(unique_footer);

    assert!(source.len() > 20_000, "test file should be >20KB");

    let mut chunker = Chunker::default();
    let chunks = chunker.chunk(&source, "data.json", Language::Json, "hash123", None);

    assert!(
      chunks.is_empty(),
      "highly repetitive files should produce no chunks, got {} chunks",
      chunks.len()
    );
  }

  #[test]
  fn test_allow_non_repetitive_large_files() {
    // A large file with diverse content should NOT be skipped
    // Generate 600 unique lines (>20KB)
    let mut source = String::with_capacity(30_000);
    for i in 0..600 {
      source.push_str(&format!("pub fn function_{}() {{ println!(\"{}\"); }}\n", i, i * 7));
    }

    assert!(source.len() > 20_000, "test file should be >20KB");

    let mut chunker = Chunker::default();
    let chunks = chunker.chunk(&source, "large.rs", Language::Rust, "hash123", None);

    assert!(!chunks.is_empty(), "non-repetitive large files should produce chunks");
  }
}
