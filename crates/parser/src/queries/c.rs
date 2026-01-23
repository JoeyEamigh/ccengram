//! C tree-sitter queries

use tree_sitter::Language as TsLanguage;

use super::compile_query;
use crate::parser::LanguageQueries;

/// Import extraction query for C
const IMPORTS_QUERY: &str = r#"
; #include <stdio.h>
(preproc_include
  path: (system_lib_string) @import)

; #include "myheader.h"
(preproc_include
  path: (string_literal) @import)
"#;

/// Call extraction query for C
const CALLS_QUERY: &str = r#"
; Direct function calls: foo()
(call_expression
  function: (identifier) @call)

; Function pointer calls via field: obj->func()
(call_expression
  function: (field_expression
    field: (field_identifier) @call))

; Macro-style calls (macros look like function calls)
(call_expression
  function: (identifier) @call)
"#;

/// Definition extraction query for C
const DEFINITIONS_QUERY: &str = r#"
; Function definitions
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name)) @definition.function

; Function definitions with pointer return type
(function_definition
  declarator: (pointer_declarator
    declarator: (function_declarator
      declarator: (identifier) @name))) @definition.function

; Struct definitions
(struct_specifier
  name: (type_identifier) @name) @definition.struct

; Enum definitions
(enum_specifier
  name: (type_identifier) @name) @definition.enum

; Typedef
(type_definition
  declarator: (type_identifier) @name) @definition.type
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
  fn test_c_imports() {
    let content = r#"
#include <stdio.h>
#include <stdlib.h>
#include "myheader.h"
#include "utils/helper.h"
"#;
    let mut parser = TreeSitterParser::new();
    let imports = parser.extract_imports(content, Language::C);

    assert!(imports.contains(&"stdio.h".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"stdlib.h".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"myheader.h".to_string()), "imports: {:?}", imports);
  }

  #[test]
  fn test_c_calls() {
    let content = r#"
int main() {
    printf("hello");
    int result = helper();
    obj->callback();
    malloc(100);
    free(ptr);
}
"#;
    let mut parser = TreeSitterParser::new();
    let calls = parser.extract_calls(content, Language::C);

    assert!(calls.contains(&"printf".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"helper".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"callback".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"malloc".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"free".to_string()), "calls: {:?}", calls);
  }

  #[test]
  fn test_c_definitions() {
    let content = r#"
void my_function(int arg) {
    // body
}

int* pointer_return_function(void) {
    return NULL;
}

struct MyStruct {
    int field;
};

enum MyEnum {
    VALUE_A,
    VALUE_B
};

typedef struct MyStruct MyType;
"#;
    let mut parser = TreeSitterParser::new();
    let defs = parser.extract_definitions(content, Language::C);

    let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"my_function"), "defs: {:?}", names);
    assert!(names.contains(&"pointer_return_function"), "defs: {:?}", names);
    assert!(names.contains(&"MyStruct"), "defs: {:?}", names);
    assert!(names.contains(&"MyEnum"), "defs: {:?}", names);
    assert!(names.contains(&"MyType"), "defs: {:?}", names);
  }
}
