use engram_core::Language;
use std::path::Path;

/// Simple language detection from file extension.
/// This module provides a lightweight alternative to full tree-sitter parsing
/// for cases where we only need basic language detection.
///
/// Detect language from file path.
pub fn detect_language(path: &Path) -> Option<Language> {
  let ext = path.extension()?.to_str()?;
  Language::from_extension(ext)
}

/// Check if a file should be indexed based on its language
pub fn is_indexable(path: &Path) -> bool {
  detect_language(path).is_some()
}

/// Get all supported extensions
pub fn supported_extensions() -> &'static [&'static str] {
  &[
    "rs",
    "py",
    "ts",
    "tsx",
    "js",
    "jsx",
    "go",
    "java",
    "c",
    "cpp",
    "h",
    "hpp",
    "cs",
    "rb",
    "php",
    "swift",
    "kt",
    "scala",
    "r",
    "jl",
    "lua",
    "pl",
    "sh",
    "bash",
    "zsh",
    "fish",
    "ps1",
    "sql",
    "graphql",
    "proto",
    "toml",
    "yaml",
    "yml",
    "json",
    "xml",
    "html",
    "css",
    "scss",
    "sass",
    "less",
    "vue",
    "svelte",
    "astro",
    "md",
    "mdx",
    "rst",
    "tex",
    "dockerfile",
    "makefile",
    "cmake",
    "gradle",
    "zig",
    "nim",
    "elm",
    "clj",
    "cljs",
    "ex",
    "exs",
    "erl",
    "hrl",
    "hs",
    "ml",
    "mli",
    "fs",
    "fsi",
    "v",
    "sv",
    "vhdl",
  ]
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_detect_language() {
    assert_eq!(detect_language(Path::new("main.rs")), Some(Language::Rust));
    assert_eq!(detect_language(Path::new("app.py")), Some(Language::Python));
    assert_eq!(detect_language(Path::new("index.ts")), Some(Language::TypeScript));
    assert_eq!(detect_language(Path::new("readme.txt")), None);
  }

  #[test]
  fn test_is_indexable() {
    assert!(is_indexable(Path::new("lib.rs")));
    assert!(is_indexable(Path::new("script.py")));
    assert!(!is_indexable(Path::new("image.png")));
    assert!(!is_indexable(Path::new("no_extension")));
  }
}
