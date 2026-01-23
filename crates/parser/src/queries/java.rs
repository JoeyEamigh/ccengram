//! Java tree-sitter queries

use tree_sitter::Language as TsLanguage;

use super::compile_query;
use crate::parser::LanguageQueries;

/// Import extraction query for Java
const IMPORTS_QUERY: &str = r#"
; import java.util.List;
(import_declaration
  (scoped_identifier) @import)

; import java.util.*;
(import_declaration
  (scoped_identifier) @import)

; import static java.util.Collections.sort;
(import_declaration
  (scoped_identifier) @import)
"#;

/// Call extraction query for Java
const CALLS_QUERY: &str = r#"
; Direct method calls: foo()
(method_invocation
  name: (identifier) @call)

; Object method calls: obj.method()
(method_invocation
  name: (identifier) @call)

; Chained calls: obj.foo().bar()
(method_invocation
  object: (method_invocation)
  name: (identifier) @call)

; Static method calls: Class.method()
(method_invocation
  name: (identifier) @call)

; Constructor calls: new MyClass()
(object_creation_expression
  type: (type_identifier) @call)
"#;

/// Definition extraction query for Java
const DEFINITIONS_QUERY: &str = r#"
; Method declarations
(method_declaration
  name: (identifier) @name) @definition.method

; Constructor declarations
(constructor_declaration
  name: (identifier) @name) @definition.method

; Class declarations
(class_declaration
  name: (identifier) @name) @definition.class

; Interface declarations
(interface_declaration
  name: (identifier) @name) @definition.interface

; Enum declarations
(enum_declaration
  name: (identifier) @name) @definition.enum

; Field declarations (constants)
(field_declaration
  declarator: (variable_declarator
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
  fn test_java_imports() {
    let content = r#"
package com.example;

import java.util.List;
import java.util.Map;
import java.io.*;
import static java.util.Collections.sort;
"#;
    let mut parser = TreeSitterParser::new();
    let imports = parser.extract_imports(content, Language::Java);

    assert!(
      imports.contains(&"java.util.List".to_string()),
      "imports: {:?}",
      imports
    );
    assert!(imports.contains(&"java.util.Map".to_string()), "imports: {:?}", imports);
  }

  #[test]
  fn test_java_calls() {
    let content = r#"
public class Example {
    public void example() {
        helper();
        obj.methodCall();
        list.stream().filter().map();
        System.out.println("hello");
        MyClass instance = new MyClass();
    }
}
"#;
    let mut parser = TreeSitterParser::new();
    let calls = parser.extract_calls(content, Language::Java);

    assert!(calls.contains(&"helper".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"methodCall".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"stream".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"filter".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"map".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"println".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"MyClass".to_string()), "calls: {:?}", calls);
  }

  #[test]
  fn test_java_definitions() {
    let content = r#"
public class MyClass {
    private String field;

    public MyClass() {}

    public void myMethod() {}

    private static void staticMethod() {}
}

public interface MyInterface {
    void interfaceMethod();
}

public enum MyEnum {
    VALUE_A,
    VALUE_B
}
"#;
    let mut parser = TreeSitterParser::new();
    let defs = parser.extract_definitions(content, Language::Java);

    let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"MyClass"), "defs: {:?}", names);
    assert!(names.contains(&"myMethod"), "defs: {:?}", names);
    assert!(names.contains(&"staticMethod"), "defs: {:?}", names);
    assert!(names.contains(&"MyInterface"), "defs: {:?}", names);
    assert!(names.contains(&"MyEnum"), "defs: {:?}", names);
  }
}
