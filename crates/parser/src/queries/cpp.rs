//! C++ tree-sitter queries

use tree_sitter::Language as TsLanguage;

use super::compile_query;
use crate::parser::LanguageQueries;

/// Import extraction query for C++
const IMPORTS_QUERY: &str = r#"
; #include <iostream>
(preproc_include
  path: (system_lib_string) @import)

; #include "myheader.hpp"
(preproc_include
  path: (string_literal) @import)

; using namespace std;
(using_declaration
  (qualified_identifier) @import)

; using std::cout;
(using_declaration
  (qualified_identifier) @import)
"#;

/// Call extraction query for C++
const CALLS_QUERY: &str = r#"
; Direct function calls: foo()
(call_expression
  function: (identifier) @call)

; Method calls: obj.method()
(call_expression
  function: (field_expression
    field: (field_identifier) @call))

; Pointer method calls: ptr->method()
(call_expression
  function: (field_expression
    field: (field_identifier) @call))

; Namespaced calls: std::cout
(call_expression
  function: (qualified_identifier
    name: (identifier) @call))

; Template function calls: make_shared<T>()
(call_expression
  function: (template_function
    name: (identifier) @call))

; Constructor calls: MyClass()
(call_expression
  function: (identifier) @call)

; Operator calls (like operator<<)
(call_expression
  function: (field_expression
    field: (field_identifier) @call))
"#;

/// Definition extraction query for C++
const DEFINITIONS_QUERY: &str = r#"
; Function definitions
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name)) @definition.function

; Method definitions outside class (MyClass::method)
(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier
      name: (identifier) @name))) @definition.method

; Class definitions
(class_specifier
  name: (type_identifier) @name) @definition.class

; Struct definitions
(struct_specifier
  name: (type_identifier) @name) @definition.struct

; Enum definitions
(enum_specifier
  name: (type_identifier) @name) @definition.enum

; Namespace definitions (C++ uses namespace_identifier)
(namespace_definition
  name: (namespace_identifier) @name) @definition.module
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
  fn test_cpp_imports() {
    let content = r#"
#include <iostream>
#include <vector>
#include "myheader.hpp"
#include "utils/helper.h"

using namespace std;
using std::cout;
"#;
    let mut parser = TreeSitterParser::new();
    let imports = parser.extract_imports(content, Language::Cpp);

    assert!(imports.contains(&"iostream".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"vector".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"myheader.hpp".to_string()), "imports: {:?}", imports);
  }

  #[test]
  fn test_cpp_calls() {
    let content = r#"
int main() {
    std::cout << "hello";
    helper();
    obj.method();
    ptr->callback();
    auto p = std::make_shared<MyClass>();
    vec.push_back(1);
}
"#;
    let mut parser = TreeSitterParser::new();
    let calls = parser.extract_calls(content, Language::Cpp);

    assert!(calls.contains(&"helper".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"method".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"callback".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"push_back".to_string()), "calls: {:?}", calls);
  }

  #[test]
  fn test_cpp_definitions() {
    let content = r#"
void my_function(int arg) {
    // body
}

class MyClass {
public:
    void method();
};

void MyClass::method() {
    // implementation
}

struct MyStruct {
    int field;
};

namespace MyNamespace {
    void namespaced_func() {}
}

template<typename T>
class TemplateClass {
};

template<typename T>
T template_function(T arg) {
    return arg;
}

enum class MyEnum {
    VALUE_A,
    VALUE_B
};
"#;
    let mut parser = TreeSitterParser::new();
    let defs = parser.extract_definitions(content, Language::Cpp);

    let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"my_function"), "defs: {:?}", names);
    assert!(names.contains(&"MyClass"), "defs: {:?}", names);
    assert!(names.contains(&"method"), "defs: {:?}", names);
    assert!(names.contains(&"MyStruct"), "defs: {:?}", names);
    assert!(names.contains(&"MyNamespace"), "defs: {:?}", names);
    assert!(names.contains(&"TemplateClass"), "defs: {:?}", names);
    assert!(names.contains(&"template_function"), "defs: {:?}", names);
    assert!(names.contains(&"MyEnum"), "defs: {:?}", names);
  }
}
