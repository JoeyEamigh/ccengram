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

  // ============================================================================
  // LANGUAGE SUPPORT TESTS
  // ============================================================================

  #[test]
  fn test_supports_tier1_languages() {
    let parser = TreeSitterParser::new();

    assert!(parser.supports_language(Language::Rust));
    assert!(parser.supports_language(Language::Python));
    assert!(parser.supports_language(Language::JavaScript));
    assert!(parser.supports_language(Language::TypeScript));
    assert!(parser.supports_language(Language::Tsx));
    assert!(parser.supports_language(Language::Jsx));
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

    let calls = parser.extract_calls("# Header", Language::Markdown);
    assert!(calls.is_empty());

    let defs = parser.extract_definitions("# Header", Language::Markdown);
    assert!(defs.is_empty());
  }

  // ============================================================================
  // ERROR HANDLING TESTS
  // ============================================================================

  #[test]
  fn test_invalid_syntax_returns_partial_results() {
    let mut parser = TreeSitterParser::new();

    // Invalid Rust syntax - parser should still work with partial results
    let content = r#"
use std::collections::HashMap;
fn broken( { // syntax error
    let x = helper_fn();
}
use chrono::Utc;
"#;
    let imports = parser.extract_imports(content, Language::Rust);
    // Should still extract valid imports despite syntax errors
    assert!(
      imports.contains(&"std::collections::HashMap".to_string()),
      "imports: {:?}",
      imports
    );
    assert!(imports.contains(&"chrono::Utc".to_string()), "imports: {:?}", imports);
  }

  #[test]
  fn test_empty_content() {
    let mut parser = TreeSitterParser::new();

    let imports = parser.extract_imports("", Language::Rust);
    assert!(imports.is_empty());

    let calls = parser.extract_calls("", Language::Rust);
    assert!(calls.is_empty());

    let defs = parser.extract_definitions("", Language::Rust);
    assert!(defs.is_empty());
  }

  #[test]
  fn test_whitespace_only_content() {
    let mut parser = TreeSitterParser::new();

    let content = "   \n\t\n   ";
    let imports = parser.extract_imports(content, Language::Rust);
    assert!(imports.is_empty());

    let calls = parser.extract_calls(content, Language::Python);
    assert!(calls.is_empty());
  }

  // ============================================================================
  // EDGE CASES: RUST
  // ============================================================================

  #[test]
  fn test_rust_nested_use_lists() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
use std::collections::{HashMap, HashSet, BTreeMap};
use std::{
    io::{self, Read, Write},
    fs::File,
};
use crate::foo::bar::{baz, qux as quux};
"#;
    let imports = parser.extract_imports(content, Language::Rust);

    // The parser should extract individual items from use lists
    assert!(imports.contains(&"HashMap".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"HashSet".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"BTreeMap".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"baz".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"qux".to_string()), "imports: {:?}", imports);
  }

  #[test]
  fn test_rust_reexports() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
pub use crate::module::Type;
pub use self::internal::Helper;
pub(crate) use super::parent::Item;
"#;
    let imports = parser.extract_imports(content, Language::Rust);

    assert!(
      imports.contains(&"crate::module::Type".to_string()),
      "imports: {:?}",
      imports
    );
    assert!(
      imports.contains(&"self::internal::Helper".to_string()),
      "imports: {:?}",
      imports
    );
    assert!(
      imports.contains(&"super::parent::Item".to_string()),
      "imports: {:?}",
      imports
    );
  }

  #[test]
  fn test_rust_async_calls() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
async fn example() {
    let result = fetch_data().await;
    let processed = process(result).await?;
    tokio::spawn(async move {
        task_fn().await;
    });
}
"#;
    let calls = parser.extract_calls(content, Language::Rust);

    assert!(calls.contains(&"fetch_data".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"process".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"spawn".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"task_fn".to_string()), "calls: {:?}", calls);
  }

  #[test]
  fn test_rust_closure_calls() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
fn example() {
    let items = vec![1, 2, 3];
    items.iter().map(|x| transform(x)).filter(|y| validate(y)).collect();
    let closure = |a, b| compute(a, b);
    closure(1, 2);
}
"#;
    let calls = parser.extract_calls(content, Language::Rust);

    assert!(calls.contains(&"iter".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"map".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"transform".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"filter".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"validate".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"collect".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"compute".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"closure".to_string()), "calls: {:?}", calls);
  }

  #[test]
  fn test_rust_generic_calls() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
fn example() {
    let map: HashMap<String, i32> = HashMap::new();
    let result = parse::<MyType>(data);
    Vec::<u8>::with_capacity(100);
}
"#;
    let calls = parser.extract_calls(content, Language::Rust);

    // Parser should handle turbofish syntax for generic calls
    assert!(calls.contains(&"new".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"parse".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"with_capacity".to_string()), "calls: {:?}", calls);
  }

  // ============================================================================
  // EDGE CASES: PYTHON
  // ============================================================================

  #[test]
  fn test_python_relative_imports() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
from . import sibling_module
from .. import parent_module
from ...grandparent import specific
from .utils import helper, another
"#;
    let imports = parser.extract_imports(content, Language::Python);

    // Check we capture relative import patterns
    assert!(
      imports.iter().any(|i| i.starts_with('.')),
      "should have relative imports: {:?}",
      imports
    );
  }

  #[test]
  fn test_python_star_import() {
    let mut parser = TreeSitterParser::new();
    let content = "from module import *";
    let imports = parser.extract_imports(content, Language::Python);

    assert!(imports.contains(&"module".to_string()), "imports: {:?}", imports);
  }

  #[test]
  fn test_python_decorated_function_calls() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
@decorator
@another_decorator(arg)
def my_function():
    helper()

@property
def computed(self):
    return calculate()
"#;
    let calls = parser.extract_calls(content, Language::Python);

    // Parser should extract decorators as calls (they're invocations)
    assert!(calls.contains(&"decorator".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"another_decorator".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"helper".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"property".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"calculate".to_string()), "calls: {:?}", calls);
  }

  #[test]
  fn test_python_comprehension_calls() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
result = [transform(x) for x in items if validate(x)]
gen = (process(i) for i in range(10))
dict_comp = {key(k): value(v) for k, v in pairs}
"#;
    let calls = parser.extract_calls(content, Language::Python);

    assert!(calls.contains(&"transform".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"validate".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"process".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"range".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"key".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"value".to_string()), "calls: {:?}", calls);
  }

  // ============================================================================
  // EDGE CASES: TYPESCRIPT/JAVASCRIPT
  // ============================================================================

  #[test]
  fn test_typescript_type_only_imports() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
// Type-only imports
import type { User, Post } from './types';
import type * as Types from './all-types';

// Mixed: type-only and regular imports in same statement
import { type Config, useConfig } from './config';
import { useState, type Dispatch, type SetStateAction } from 'react';

// Regular imports for comparison
import axios from 'axios';
import { helper } from './utils';
"#;
    let imports = parser.extract_imports(content, Language::TypeScript);

    // Type-only imports should be captured
    assert!(imports.contains(&"./types".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"./all-types".to_string()), "imports: {:?}", imports);

    // Mixed imports should be captured
    assert!(imports.contains(&"./config".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"react".to_string()), "imports: {:?}", imports);

    // Regular imports
    assert!(imports.contains(&"axios".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"./utils".to_string()), "imports: {:?}", imports);
  }

  #[test]
  fn test_typescript_dynamic_imports() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
const module = await import('./dynamic');
const { default: Component } = await import(`./components/${name}`);
"#;
    let imports = parser.extract_imports(content, Language::TypeScript);

    assert!(imports.contains(&"./dynamic".to_string()), "imports: {:?}", imports);
  }

  #[test]
  fn test_jsx_fragment_and_components() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
function App() {
    return (
        <>
            <Header />
            <Main>
                <Sidebar items={items} />
                <Content />
            </Main>
            <Footer />
        </>
    );
}
"#;
    let calls = parser.extract_calls(content, Language::Jsx);

    assert!(calls.contains(&"Header".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"Main".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"Sidebar".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"Content".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"Footer".to_string()), "calls: {:?}", calls);
  }

  #[test]
  fn test_typescript_optional_chaining_calls() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
const result = obj?.method();
const nested = a?.b?.c();
const arr = items?.[0]?.transform();
"#;
    let calls = parser.extract_calls(content, Language::TypeScript);

    assert!(calls.contains(&"method".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"c".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"transform".to_string()), "calls: {:?}", calls);
  }

  /// Test NodeNext/Node16 module resolution style imports
  /// Uses .js extension for TypeScript files, explicit extensions required
  #[test]
  fn test_typescript_nodenext_imports() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
// NodeNext/Node16: Must use .js extension for .ts files
import { helper } from './utils.js';           // Actual file: ./utils.ts
import { Button } from './components/Button.js'; // Actual file: ./components/Button.tsx
import type { Config } from './config.js';

// ESM extensions
import { logger } from './logging.mjs';
import { data } from './data.cjs';

// JSON imports with assertion (Node16+)
import config from './config.json' assert { type: 'json' };

// External packages (no extension needed)
import express from 'express';
import { z } from 'zod';
"#;
    let imports = parser.extract_imports(content, Language::TypeScript);

    // NodeNext style with .js extension
    assert!(imports.contains(&"./utils.js".to_string()), "imports: {:?}", imports);
    assert!(
      imports.contains(&"./components/Button.js".to_string()),
      "imports: {:?}",
      imports
    );
    assert!(imports.contains(&"./config.js".to_string()), "imports: {:?}", imports);

    // ESM extensions
    assert!(imports.contains(&"./logging.mjs".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"./data.cjs".to_string()), "imports: {:?}", imports);

    // JSON import
    assert!(imports.contains(&"./config.json".to_string()), "imports: {:?}", imports);

    // External packages
    assert!(imports.contains(&"express".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"zod".to_string()), "imports: {:?}", imports);
  }

  /// Test Bundler module resolution style imports
  /// No extensions needed, bundler handles resolution
  #[test]
  fn test_typescript_bundler_imports() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
// Bundler style: No extensions needed
import { helper } from './utils';
import { Button } from './components/Button';
import type { Config } from './config';

// Index file imports (bundler resolves)
import { routes } from './routes';         // Resolves to ./routes/index.ts
import * as models from './models';        // Resolves to ./models/index.ts

// Alias imports (bundler resolves via paths config)
import { api } from '@/api';
import { useAuth } from '@hooks/useAuth';
import { Button } from '~/components/Button';

// External packages
import React, { useState, useEffect } from 'react';
import { clsx } from 'clsx';
"#;
    let imports = parser.extract_imports(content, Language::TypeScript);

    // Bundler style without extensions
    assert!(imports.contains(&"./utils".to_string()), "imports: {:?}", imports);
    assert!(
      imports.contains(&"./components/Button".to_string()),
      "imports: {:?}",
      imports
    );
    assert!(imports.contains(&"./config".to_string()), "imports: {:?}", imports);

    // Index file imports
    assert!(imports.contains(&"./routes".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"./models".to_string()), "imports: {:?}", imports);

    // Alias imports
    assert!(imports.contains(&"@/api".to_string()), "imports: {:?}", imports);
    assert!(
      imports.contains(&"@hooks/useAuth".to_string()),
      "imports: {:?}",
      imports
    );
    assert!(
      imports.contains(&"~/components/Button".to_string()),
      "imports: {:?}",
      imports
    );

    // External packages
    assert!(imports.contains(&"react".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"clsx".to_string()), "imports: {:?}", imports);
  }

  /// Test Classic/Node10 module resolution style imports
  /// .ts extension allowed, node_modules resolution
  #[test]
  fn test_typescript_classic_imports() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
// Classic/Node10: Can use .ts extension directly (less common)
import { helper } from './utils.ts';
import { Button } from './components/Button.tsx';

// Also supports extensionless
import { config } from './config';

// Package imports
import lodash from 'lodash';
import { map, filter } from 'lodash/fp';

// Namespace import
import * as fs from 'fs';
import * as path from 'path';

// Side effect imports
import 'reflect-metadata';
import './polyfills';
"#;
    let imports = parser.extract_imports(content, Language::TypeScript);

    // Classic style with .ts extension
    assert!(imports.contains(&"./utils.ts".to_string()), "imports: {:?}", imports);
    assert!(
      imports.contains(&"./components/Button.tsx".to_string()),
      "imports: {:?}",
      imports
    );

    // Extensionless
    assert!(imports.contains(&"./config".to_string()), "imports: {:?}", imports);

    // Package imports
    assert!(imports.contains(&"lodash".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"lodash/fp".to_string()), "imports: {:?}", imports);

    // Built-in modules
    assert!(imports.contains(&"fs".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"path".to_string()), "imports: {:?}", imports);

    // Side effect imports
    assert!(
      imports.contains(&"reflect-metadata".to_string()),
      "imports: {:?}",
      imports
    );
    assert!(imports.contains(&"./polyfills".to_string()), "imports: {:?}", imports);
  }

  /// Test require() and dynamic imports for CommonJS interop
  #[test]
  fn test_typescript_require_imports() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
// CommonJS require (for CJS modules)
const fs = require('fs');
const { join } = require('path');
const config = require('./config.json');

// Dynamic import (ESM)
const module = await import('./dynamic.js');
const { default: Component } = await import('./Component');

// Re-exports
export { helper } from './utils';
export * from './constants';
export * as utils from './utils';
"#;
    let imports = parser.extract_imports(content, Language::TypeScript);

    // require() calls
    assert!(imports.contains(&"fs".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"path".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"./config.json".to_string()), "imports: {:?}", imports);

    // Dynamic imports
    assert!(imports.contains(&"./dynamic.js".to_string()), "imports: {:?}", imports);

    // Re-exports count as imports (source dependency)
    assert!(imports.contains(&"./utils".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"./constants".to_string()), "imports: {:?}", imports);
  }

  // ============================================================================
  // EDGE CASES: GO
  // ============================================================================

  #[test]
  fn test_go_blank_identifier_import() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
import (
    _ "github.com/lib/pq"        // side-effect import
    . "fmt"                       // dot import
    alias "long/package/name"
)
"#;
    let imports = parser.extract_imports(content, Language::Go);

    assert!(
      imports.contains(&"github.com/lib/pq".to_string()),
      "imports: {:?}",
      imports
    );
    assert!(imports.contains(&"fmt".to_string()), "imports: {:?}", imports);
    assert!(
      imports.contains(&"long/package/name".to_string()),
      "imports: {:?}",
      imports
    );
  }

  #[test]
  fn test_go_deferred_calls() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
func example() {
    defer file.Close()
    defer func() { cleanup() }()
    go processAsync()
}
"#;
    let calls = parser.extract_calls(content, Language::Go);

    assert!(calls.contains(&"Close".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"cleanup".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"processAsync".to_string()), "calls: {:?}", calls);
  }

  // ============================================================================
  // EDGE CASES: JAVA
  // ============================================================================

  #[test]
  fn test_java_static_imports() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
import static java.lang.Math.PI;
import static java.util.Collections.*;
import static org.junit.Assert.assertEquals;
"#;
    let imports = parser.extract_imports(content, Language::Java);

    // Static imports should be captured
    assert!(imports.iter().any(|i| i.contains("Math")), "imports: {:?}", imports);
  }

  #[test]
  fn test_java_anonymous_class_calls() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
public class Example {
    public void test() {
        Runnable r = new Runnable() {
            @Override
            public void run() {
                execute();
            }
        };
        list.forEach(item -> process(item));
    }
}
"#;
    let calls = parser.extract_calls(content, Language::Java);

    assert!(calls.contains(&"execute".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"forEach".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"process".to_string()), "calls: {:?}", calls);
  }

  // ============================================================================
  // EDGE CASES: C/C++
  // ============================================================================

  #[test]
  fn test_c_macro_calls() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
#include <assert.h>
#define MAX(a, b) ((a) > (b) ? (a) : (b))

int main() {
    assert(condition);
    int x = MAX(1, 2);
    printf("result: %d\n", x);
}
"#;
    let calls = parser.extract_calls(content, Language::C);

    assert!(calls.contains(&"assert".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"MAX".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"printf".to_string()), "calls: {:?}", calls);
  }

  #[test]
  fn test_cpp_template_calls() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
#include <memory>
#include <vector>

void example() {
    auto ptr = std::make_shared<MyClass>(args);
    auto vec = std::vector<int>{1, 2, 3};
    std::sort(vec.begin(), vec.end());
}
"#;
    let calls = parser.extract_calls(content, Language::Cpp);

    assert!(calls.contains(&"sort".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"begin".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"end".to_string()), "calls: {:?}", calls);
  }

  #[test]
  fn test_cpp_namespace_using() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
using namespace std;
using std::cout;
using std::endl;

namespace myns {
    void func() {}
}
"#;
    let imports = parser.extract_imports(content, Language::Cpp);

    // Check that using declarations are captured
    assert!(!imports.is_empty(), "should have using declarations: {:?}", imports);
  }

  // ============================================================================
  // DEFINITION EXTRACTION TESTS
  // ============================================================================

  #[test]
  fn test_definition_line_numbers() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
fn first_fn() {
    // line 2-4
}

fn second_fn() {
    // line 6-8
}
"#;
    let defs = parser.extract_definitions(content, Language::Rust);

    let first = defs.iter().find(|d| d.name == "first_fn");
    let second = defs.iter().find(|d| d.name == "second_fn");

    assert!(first.is_some(), "should find first_fn");
    assert!(second.is_some(), "should find second_fn");

    let first = first.unwrap();
    let second = second.unwrap();

    assert!(first.start_line < second.start_line, "first should be before second");
    assert_eq!(first.kind, DefinitionKind::Function);
    assert_eq!(second.kind, DefinitionKind::Function);
  }

  #[test]
  fn test_definition_kinds() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
fn my_function() {}
struct MyStruct {}
enum MyEnum { A, B }
trait MyTrait {}
mod my_module {}
const MY_CONST: i32 = 42;
type MyType = String;
"#;
    let defs = parser.extract_definitions(content, Language::Rust);

    let fn_def = defs.iter().find(|d| d.name == "my_function");
    let struct_def = defs.iter().find(|d| d.name == "MyStruct");
    let enum_def = defs.iter().find(|d| d.name == "MyEnum");
    let trait_def = defs.iter().find(|d| d.name == "MyTrait");
    let mod_def = defs.iter().find(|d| d.name == "my_module");

    assert!(
      matches!(fn_def.map(|d| d.kind), Some(DefinitionKind::Function)),
      "fn kind"
    );
    assert!(
      matches!(struct_def.map(|d| d.kind), Some(DefinitionKind::Struct)),
      "struct kind"
    );
    assert!(
      matches!(enum_def.map(|d| d.kind), Some(DefinitionKind::Enum)),
      "enum kind"
    );
    assert!(
      matches!(trait_def.map(|d| d.kind), Some(DefinitionKind::Trait)),
      "trait kind"
    );
    assert!(
      matches!(mod_def.map(|d| d.kind), Some(DefinitionKind::Module)),
      "module kind"
    );
  }

  // ============================================================================
  // DEDUPLICATION TESTS
  // ============================================================================

  #[test]
  fn test_deduplication() {
    let mut parser = TreeSitterParser::new();
    let content = r#"
fn example() {
    helper();
    helper();
    helper();
}
"#;
    let calls = parser.extract_calls(content, Language::Rust);

    // Should only have one entry for "helper"
    let helper_count = calls.iter().filter(|c| *c == "helper").count();
    assert_eq!(helper_count, 1, "helper should appear only once: {:?}", calls);
  }

  // ============================================================================
  // PARSER REUSE TESTS
  // ============================================================================

  #[test]
  fn test_parser_reuse_across_files() {
    let mut parser = TreeSitterParser::new();

    // Parse multiple files of the same language
    let rust1 = "use std::io; fn a() { helper1(); }";
    let rust2 = "use std::fs; fn b() { helper2(); }";

    let imports1 = parser.extract_imports(rust1, Language::Rust);
    let calls1 = parser.extract_calls(rust1, Language::Rust);

    let imports2 = parser.extract_imports(rust2, Language::Rust);
    let calls2 = parser.extract_calls(rust2, Language::Rust);

    assert!(imports1.contains(&"std::io".to_string()));
    assert!(calls1.contains(&"helper1".to_string()));
    assert!(imports2.contains(&"std::fs".to_string()));
    assert!(calls2.contains(&"helper2".to_string()));
  }

  #[test]
  fn test_parser_multiple_languages() {
    let mut parser = TreeSitterParser::new();

    let rust_code = "use std::io; fn main() { println!(\"hello\"); }";
    let python_code = "import os\nprint('hello')";
    let js_code = "import fs from 'fs'; console.log('hello');";

    let rust_imports = parser.extract_imports(rust_code, Language::Rust);
    let python_imports = parser.extract_imports(python_code, Language::Python);
    let js_imports = parser.extract_imports(js_code, Language::JavaScript);

    assert!(rust_imports.contains(&"std::io".to_string()));
    assert!(python_imports.contains(&"os".to_string()));
    assert!(js_imports.contains(&"fs".to_string()));
  }
}
