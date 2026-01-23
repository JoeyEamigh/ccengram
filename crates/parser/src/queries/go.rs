//! Go tree-sitter queries

use tree_sitter::Language as TsLanguage;

use super::compile_query;
use crate::parser::LanguageQueries;

/// Import extraction query for Go
const IMPORTS_QUERY: &str = r#"
; Single import: import "fmt"
(import_declaration
  (import_spec
    path: (interpreted_string_literal) @import))

; Import with alias: import f "fmt"
(import_declaration
  (import_spec
    path: (interpreted_string_literal) @import))

; Import block: import ( "fmt" "os" )
(import_declaration
  (import_spec_list
    (import_spec
      path: (interpreted_string_literal) @import)))
"#;

/// Call extraction query for Go
const CALLS_QUERY: &str = r#"
; Direct function calls: foo()
(call_expression
  function: (identifier) @call)

; Package function calls: fmt.Println()
(call_expression
  function: (selector_expression
    field: (field_identifier) @call))

; Method calls on variables: obj.Method()
(call_expression
  function: (selector_expression
    field: (field_identifier) @call))

; Chained calls: obj.Foo().Bar()
(call_expression
  function: (selector_expression
    operand: (call_expression)
    field: (field_identifier) @call))
"#;

/// Definition extraction query for Go
const DEFINITIONS_QUERY: &str = r#"
; Function declarations
(function_declaration
  name: (identifier) @name) @definition.function

; Method declarations
(method_declaration
  name: (field_identifier) @name) @definition.method

; Type declarations (struct)
(type_declaration
  (type_spec
    name: (type_identifier) @name
    type: (struct_type))) @definition.struct

; Type declarations (interface)
(type_declaration
  (type_spec
    name: (type_identifier) @name
    type: (interface_type))) @definition.interface

; Type alias
(type_declaration
  (type_spec
    name: (type_identifier) @name)) @definition.type

; Const declarations
(const_declaration
  (const_spec
    name: (identifier) @name)) @definition.const
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
  fn test_go_imports() {
    let content = r#"
package main

import "fmt"
import "os"

import (
    "encoding/json"
    "net/http"
    myalias "github.com/example/pkg"
)
"#;
    let mut parser = TreeSitterParser::new();
    let imports = parser.extract_imports(content, Language::Go);

    assert!(imports.contains(&"fmt".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"os".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"encoding/json".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"net/http".to_string()), "imports: {:?}", imports);
    assert!(
      imports.contains(&"github.com/example/pkg".to_string()),
      "imports: {:?}",
      imports
    );
  }

  #[test]
  fn test_go_calls() {
    let content = r#"
package main

func example() {
    result := helper()
    fmt.Println("hello")
    data := json.Marshal(obj)
    client.Do(req).Body.Close()
}
"#;
    let mut parser = TreeSitterParser::new();
    let calls = parser.extract_calls(content, Language::Go);

    assert!(calls.contains(&"helper".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"Println".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"Marshal".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"Do".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"Close".to_string()), "calls: {:?}", calls);
  }

  #[test]
  fn test_go_definitions() {
    let content = r#"
package main

func myFunction() {}

func (r *Receiver) myMethod() {}

type MyStruct struct {
    Field string
}

type MyInterface interface {
    Method() error
}

const MyConst = 42
"#;
    let mut parser = TreeSitterParser::new();
    let defs = parser.extract_definitions(content, Language::Go);

    let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"myFunction"), "defs: {:?}", names);
    assert!(names.contains(&"myMethod"), "defs: {:?}", names);
    assert!(names.contains(&"MyStruct"), "defs: {:?}", names);
    assert!(names.contains(&"MyInterface"), "defs: {:?}", names);
    assert!(names.contains(&"MyConst"), "defs: {:?}", names);
  }
}
