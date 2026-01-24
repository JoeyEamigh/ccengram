//! Tree-sitter query loading and compilation

use tree_sitter::{Language as TsLanguage, Query};

use crate::parser::LanguageQueries;
use engram_core::Language;

mod c;
mod cpp;
mod go;
mod java;
mod python;
mod rust;
mod typescript;

/// Load all queries for a language
pub fn load_queries(lang: Language, grammar: &TsLanguage) -> LanguageQueries {
  match lang {
    Language::Rust => rust::queries(grammar),
    Language::Python => python::queries(grammar),
    Language::JavaScript | Language::Jsx | Language::TypeScript | Language::Tsx => {
      // Use variant-aware loader for JS/TS family
      typescript::queries_for_variant(lang, grammar)
    }
    Language::Go => go::queries(grammar),
    Language::Java => java::queries(grammar),
    Language::C => c::queries(grammar),
    Language::Cpp => cpp::queries(grammar),
    _ => LanguageQueries {
      imports: None,
      calls: None,
      definitions: None,
    },
  }
}

/// Helper to compile a query, returning None on failure
pub fn compile_query(grammar: &TsLanguage, source: &str) -> Option<Query> {
  match Query::new(grammar, source) {
    Ok(q) => Some(q),
    Err(e) => {
      eprintln!("Query compilation error: {:?}", e);
      None
    }
  }
}
