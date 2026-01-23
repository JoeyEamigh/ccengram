//! Import path resolution for TypeScript/JavaScript module systems
//!
//! Handles the various module resolution strategies:
//! - **NodeNext/Node16**: Imports use `.js` extension for `.ts` files
//! - **Bundler**: Extensionless imports, index file resolution
//! - **Classic/Node10**: Can use `.ts` extension directly
//!
//! This module provides utilities to normalize import paths and match them
//! against actual file paths in the index.

use std::path::Path;

/// Extension mappings for NodeNext resolution
/// When importing `./foo.js`, the actual file could be any of these
const JS_TO_TS_EXTENSIONS: &[(&str, &[&str])] = &[
  (".js", &[".ts", ".tsx", ".js", ".jsx"]),
  (".jsx", &[".tsx", ".jsx", ".ts", ".js"]),
  (".mjs", &[".mts", ".mjs"]),
  (".cjs", &[".cts", ".cjs"]),
];

/// Index file names to try when importing a directory
const INDEX_FILES: &[&str] = &[
  "index.ts",
  "index.tsx",
  "index.js",
  "index.jsx",
  "index.mts",
  "index.mjs",
];

/// Normalize an import path for matching
///
/// Removes quotes, normalizes path separators, and handles relative paths.
pub fn normalize_import(import: &str) -> String {
  // Remove surrounding quotes if present
  let import = import.trim_matches(|c| c == '"' || c == '\'');

  // Normalize path separators
  import.replace('\\', "/")
}

/// Generate all possible file paths that an import could resolve to
///
/// For NodeNext-style imports like `./utils.js`, returns:
/// - `utils.js`
/// - `utils.ts`
/// - `utils.tsx`
/// - etc.
pub fn possible_resolutions(import_path: &str) -> Vec<String> {
  let normalized = normalize_import(import_path);
  let mut results = Vec::new();

  // Skip external packages (no relative path prefix)
  if !normalized.starts_with('.') && !normalized.starts_with('/') {
    // External package - just return as-is
    results.push(normalized);
    return results;
  }

  // Remove leading ./ for path matching
  let path_part = normalized.trim_start_matches("./").trim_start_matches("../");

  // Check if this has an extension we should map
  for (from_ext, to_exts) in JS_TO_TS_EXTENSIONS {
    if let Some(base) = path_part.strip_suffix(from_ext) {
      for to_ext in *to_exts {
        results.push(format!("{}{}", base, to_ext));
      }
      // Also try the original
      if !results.contains(&path_part.to_string()) {
        results.push(path_part.to_string());
      }
      return results;
    }
  }

  // No extension or unrecognized extension - try adding common extensions
  // This handles bundler-style extensionless imports
  if !path_part.contains('.') || path_part.ends_with('/') {
    // Extensionless import - could be a file or directory
    let base = path_part.trim_end_matches('/');

    // Try as file with various extensions
    for ext in &[".ts", ".tsx", ".js", ".jsx", ".mts", ".mjs"] {
      results.push(format!("{}{}", base, ext));
    }

    // Try as directory with index file
    for index in INDEX_FILES {
      results.push(format!("{}/{}", base, index));
    }
  }

  // Always include the original path
  if !results.contains(&path_part.to_string()) {
    results.push(path_part.to_string());
  }

  results
}

/// Check if an import path could resolve to a given file path
///
/// Returns true if the import `./utils.js` could resolve to `src/utils.ts`
pub fn import_matches_file(import_path: &str, file_path: &str) -> bool {
  let possible = possible_resolutions(import_path);
  let file_normalized = file_path.replace('\\', "/");

  for candidate in possible {
    // Check if file_path ends with the candidate
    if file_normalized.ends_with(&candidate) {
      return true;
    }

    // Also check without extension for the file
    let file_stem = Path::new(&file_normalized)
      .file_stem()
      .and_then(|s| s.to_str())
      .unwrap_or("");
    let candidate_stem = Path::new(&candidate).file_stem().and_then(|s| s.to_str()).unwrap_or("");

    if !file_stem.is_empty() && !candidate_stem.is_empty() && file_stem == candidate_stem {
      // Same base name - check if directories match
      let file_dir = Path::new(&file_normalized).parent().and_then(|p| p.to_str());
      let candidate_dir = Path::new(&candidate).parent().and_then(|p| p.to_str());

      if file_dir == candidate_dir
        || candidate_dir.is_none()
        || candidate_dir == Some("")
        || file_dir
          .map(|d| d.ends_with(candidate_dir.unwrap_or("")))
          .unwrap_or(false)
      {
        return true;
      }
    }
  }

  false
}

/// Generate SQL LIKE patterns for matching imports in the database
///
/// For an import like `./utils.js`, generates patterns that will match:
/// - Files named `utils.ts`, `utils.tsx`, `utils.js`, etc.
pub fn import_to_file_patterns(import_path: &str) -> Vec<String> {
  let possible = possible_resolutions(import_path);
  possible.into_iter().map(|p| format!("%{}", p)).collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_normalize_import() {
    assert_eq!(normalize_import("'./utils'"), "./utils");
    assert_eq!(normalize_import("\"./utils.js\""), "./utils.js");
    assert_eq!(normalize_import("./utils"), "./utils");
  }

  #[test]
  fn test_nodenext_resolution() {
    let possible = possible_resolutions("./utils.js");
    assert!(possible.contains(&"utils.ts".to_string()), "possible: {:?}", possible);
    assert!(possible.contains(&"utils.tsx".to_string()), "possible: {:?}", possible);
    assert!(possible.contains(&"utils.js".to_string()), "possible: {:?}", possible);
  }

  #[test]
  fn test_mjs_resolution() {
    let possible = possible_resolutions("./utils.mjs");
    assert!(possible.contains(&"utils.mts".to_string()), "possible: {:?}", possible);
    assert!(possible.contains(&"utils.mjs".to_string()), "possible: {:?}", possible);
  }

  #[test]
  fn test_bundler_extensionless() {
    let possible = possible_resolutions("./utils");
    assert!(possible.contains(&"utils.ts".to_string()), "possible: {:?}", possible);
    assert!(possible.contains(&"utils.tsx".to_string()), "possible: {:?}", possible);
    assert!(
      possible.contains(&"utils/index.ts".to_string()),
      "possible: {:?}",
      possible
    );
  }

  #[test]
  fn test_import_matches_file() {
    // NodeNext: ./utils.js should match utils.ts
    assert!(import_matches_file("./utils.js", "src/utils.ts"));
    assert!(import_matches_file("./utils.js", "utils.ts"));
    assert!(import_matches_file("./utils.js", "utils.js"));

    // Bundler: ./utils should match utils.ts
    assert!(import_matches_file("./utils", "src/utils.ts"));
    assert!(import_matches_file("./utils", "utils.tsx"));

    // Directory import should match index files
    assert!(import_matches_file("./components", "components/index.ts"));
    assert!(import_matches_file("./components", "src/components/index.tsx"));
  }

  #[test]
  fn test_import_matches_file_with_path() {
    // Import with directory should match
    assert!(import_matches_file(
      "./components/Button.js",
      "src/components/Button.ts"
    ));
    assert!(import_matches_file("../utils/helper.js", "utils/helper.ts"));
  }

  #[test]
  fn test_external_packages() {
    let possible = possible_resolutions("react");
    assert_eq!(possible, vec!["react".to_string()]);

    let possible = possible_resolutions("@scope/package");
    assert_eq!(possible, vec!["@scope/package".to_string()]);
  }

  #[test]
  fn test_import_to_file_patterns() {
    let patterns = import_to_file_patterns("./utils.js");
    assert!(
      patterns.iter().any(|p| p.contains("utils.ts")),
      "patterns: {:?}",
      patterns
    );
    assert!(
      patterns.iter().any(|p| p.contains("utils.tsx")),
      "patterns: {:?}",
      patterns
    );
  }
}
