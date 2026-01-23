//! Python tree-sitter queries

use tree_sitter::Language as TsLanguage;

use super::compile_query;
use crate::parser::LanguageQueries;

/// Import extraction query for Python
const IMPORTS_QUERY: &str = r#"
; import foo
(import_statement
  name: (dotted_name) @import)

; import foo, bar
(import_statement
  name: (dotted_name) @import)

; from foo import bar
(import_from_statement
  module_name: (dotted_name) @import)

; from foo import bar, baz
(import_from_statement
  module_name: (dotted_name) @import)

; from . import foo (relative imports)
(import_from_statement
  module_name: (relative_import) @import)
"#;

/// Call extraction query for Python
const CALLS_QUERY: &str = r#"
; Direct function calls: foo()
(call
  function: (identifier) @call)

; Method/attribute calls: obj.method()
(call
  function: (attribute
    attribute: (identifier) @call))

; Chained calls: obj.foo().bar()
(call
  function: (attribute
    object: (call)
    attribute: (identifier) @call))
"#;

/// Definition extraction query for Python
const DEFINITIONS_QUERY: &str = r#"
; Functions
(function_definition
  name: (identifier) @name) @definition.function

; Async functions
(function_definition
  name: (identifier) @name) @definition.function

; Classes
(class_definition
  name: (identifier) @name) @definition.class

; Methods (inside class)
(class_definition
  body: (block
    (function_definition
      name: (identifier) @name) @definition.method))
"#;

pub fn queries(grammar: &TsLanguage) -> LanguageQueries {
  LanguageQueries {
    imports: compile_query(grammar, IMPORTS_QUERY),
    calls: compile_query(grammar, CALLS_QUERY),
    definitions: compile_query(grammar, DEFINITIONS_QUERY),
  }
}

#[cfg(test)]
mod tests {

  use crate::TreeSitterParser;
  use engram_core::Language;

  #[test]
  fn test_python_imports() {
    let content = r#"
import os
import sys
from pathlib import Path
from typing import Optional, List
from . import sibling
from ..parent import module
"#;
    let mut parser = TreeSitterParser::new();
    let imports = parser.extract_imports(content, Language::Python);

    assert!(imports.contains(&"os".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"sys".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"pathlib".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"typing".to_string()), "imports: {:?}", imports);
  }

  #[test]
  fn test_python_calls() {
    let content = r#"
def example():
    result = helper_fn()
    obj.method_call()
    data = json.loads(text)
    chain.foo().bar()
    print("hello")
"#;
    let mut parser = TreeSitterParser::new();
    let calls = parser.extract_calls(content, Language::Python);

    assert!(calls.contains(&"helper_fn".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"method_call".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"loads".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"foo".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"bar".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"print".to_string()), "calls: {:?}", calls);
  }

  #[test]
  fn test_python_definitions() {
    let content = r#"
def my_function():
    pass

async def async_function():
    pass

class MyClass:
    def method(self):
        pass

    async def async_method(self):
        pass
"#;
    let mut parser = TreeSitterParser::new();
    let defs = parser.extract_definitions(content, Language::Python);

    let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"my_function"), "defs: {:?}", names);
    assert!(names.contains(&"async_function"), "defs: {:?}", names);
    assert!(names.contains(&"MyClass"), "defs: {:?}", names);
    assert!(names.contains(&"method"), "defs: {:?}", names);
  }
}
