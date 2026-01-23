//! TypeScript/JavaScript tree-sitter queries
//!
//! Handles four variants:
//! - JavaScript: tree-sitter-javascript grammar (includes JSX)
//! - JSX: same as JavaScript
//! - TypeScript: tree-sitter-typescript grammar (no JSX, uses type_identifier)
//! - TSX: tree-sitter-typescript TSX grammar (JSX + type_identifier)

use tree_sitter::Language as TsLanguage;

use super::compile_query;
use crate::parser::LanguageQueries;
use engram_core::Language;

/// Import extraction query for TypeScript/JavaScript (same for all variants)
const IMPORTS_QUERY: &str = r#"
; import { foo } from 'module'
(import_statement
  source: (string) @import)

; import foo from 'module'
(import_statement
  source: (string) @import)

; import * as foo from 'module'
(import_statement
  source: (string) @import)

; const foo = require('module')
(call_expression
  function: (identifier) @_require
  arguments: (arguments (string) @import)
  (#eq? @_require "require"))

; Dynamic imports: import('module')
(call_expression
  function: (import)
  arguments: (arguments (string) @import))

; export { foo } from 'module'
(export_statement
  source: (string) @import)
"#;

/// Base call extraction query (works for all variants)
const BASE_CALLS_QUERY: &str = r#"
; Direct function calls: foo()
(call_expression
  function: (identifier) @call)

; Method calls: obj.method()
(call_expression
  function: (member_expression
    property: (property_identifier) @call))

; Chained method calls
(call_expression
  function: (member_expression
    object: (call_expression)
    property: (property_identifier) @call))

; Optional chaining: obj?.method()
(call_expression
  function: (member_expression
    property: (property_identifier) @call))

; new Constructor()
(new_expression
  constructor: (identifier) @call)

; new module.Constructor()
(new_expression
  constructor: (member_expression
    property: (property_identifier) @call))
"#;

/// JSX-specific call patterns (only for JSX/TSX grammars)
const JSX_CALLS_QUERY: &str = r#"
; JSX self-closing element: <Component />
(jsx_self_closing_element
  name: (identifier) @call)

; JSX opening element: <Component>...</Component>
(jsx_opening_element
  name: (identifier) @call)

; JSX member expression: <Module.Component />
(jsx_self_closing_element
  name: (member_expression
    property: (property_identifier) @call))

; JSX member expression: <Module.Component>...</Module.Component>
(jsx_opening_element
  name: (member_expression
    property: (property_identifier) @call))
"#;

/// Definition extraction query for JavaScript/JSX (uses identifier for class names)
const JS_DEFINITIONS_QUERY: &str = r#"
; function declarations
(function_declaration
  name: (identifier) @name) @definition.function

; arrow functions assigned to const/let
(variable_declarator
  name: (identifier) @name
  value: (arrow_function) @definition.function)

; class declarations (JavaScript uses identifier)
(class_declaration
  name: (identifier) @name) @definition.class

; method definitions
(method_definition
  name: (property_identifier) @name) @definition.method
"#;

/// Definition extraction query for TypeScript (uses type_identifier for class names)
const TS_DEFINITIONS_QUERY: &str = r#"
; function declarations
(function_declaration
  name: (identifier) @name) @definition.function

; arrow functions assigned to const/let
(variable_declarator
  name: (identifier) @name
  value: (arrow_function) @definition.function)

; class declarations (TypeScript uses type_identifier)
(class_declaration
  name: (type_identifier) @name) @definition.class

; interface declarations (TypeScript)
(interface_declaration
  name: (type_identifier) @name) @definition.interface

; type alias declarations (TypeScript)
(type_alias_declaration
  name: (type_identifier) @name) @definition.type

; method definitions
(method_definition
  name: (property_identifier) @name) @definition.method
"#;

/// Load queries for a specific JS/TS variant
pub fn queries_for_variant(lang: Language, grammar: &TsLanguage) -> LanguageQueries {
  match lang {
    Language::JavaScript | Language::Jsx => {
      // JavaScript grammar includes JSX support
      let calls_query = format!("{}\n{}", BASE_CALLS_QUERY, JSX_CALLS_QUERY);
      LanguageQueries {
        imports: compile_query(grammar, IMPORTS_QUERY),
        calls: compile_query(grammar, &calls_query),
        definitions: compile_query(grammar, JS_DEFINITIONS_QUERY),
      }
    }
    Language::TypeScript => {
      // TypeScript grammar does NOT include JSX
      LanguageQueries {
        imports: compile_query(grammar, IMPORTS_QUERY),
        calls: compile_query(grammar, BASE_CALLS_QUERY),
        definitions: compile_query(grammar, TS_DEFINITIONS_QUERY),
      }
    }
    Language::Tsx => {
      // TSX grammar includes JSX support
      let calls_query = format!("{}\n{}", BASE_CALLS_QUERY, JSX_CALLS_QUERY);
      LanguageQueries {
        imports: compile_query(grammar, IMPORTS_QUERY),
        calls: compile_query(grammar, &calls_query),
        definitions: compile_query(grammar, TS_DEFINITIONS_QUERY),
      }
    }
    _ => LanguageQueries {
      imports: None,
      calls: None,
      definitions: None,
    },
  }
}

#[cfg(test)]
mod tests {
  use crate::TreeSitterParser;
  use engram_core::Language;

  /// NOTE: Node.js "nodeNext" / "node16" module resolution
  ///
  /// In TypeScript with `moduleResolution: "nodeNext"`, imports use the output
  /// extension (.js) even though source files are .ts:
  ///
  /// ```typescript
  /// // In src/app.ts
  /// import { helper } from './utils.js';  // Actual file is ./utils.ts
  /// ```
  ///
  /// The parser extracts the import path as-written (e.g., "./utils.js").
  /// File resolution (mapping .js → .ts) should be handled at a higher level,
  /// as it depends on tsconfig.json settings and file system state.
  #[test]
  fn test_typescript_nodenext_resolution() {
    let content = r#"
// nodeNext style imports - uses .js extension for .ts files
import { readData } from './utils.js';
import { Schema } from '../models/schema.js';
import type { Config } from './config.js';

// ESM with .mjs/.mts
import { logger } from './logging.mjs';

// JSON imports (allowed in node16/nodenext)
import data from './data.json' assert { type: 'json' };
"#;
    let mut parser = TreeSitterParser::new();
    let imports = parser.extract_imports(content, Language::TypeScript);

    // Parser extracts paths as-written; .js → .ts mapping is caller's responsibility
    assert!(
      imports.contains(&"./utils.js".to_string()),
      "should preserve .js extension: {:?}",
      imports
    );
    assert!(
      imports.contains(&"../models/schema.js".to_string()),
      "should preserve relative path with .js: {:?}",
      imports
    );
    assert!(
      imports.contains(&"./config.js".to_string()),
      "should include type-only imports: {:?}",
      imports
    );
    assert!(
      imports.contains(&"./logging.mjs".to_string()),
      "should handle .mjs extension: {:?}",
      imports
    );
    assert!(
      imports.contains(&"./data.json".to_string()),
      "should handle JSON imports: {:?}",
      imports
    );
  }

  #[test]
  fn test_tsx_nodenext_resolution() {
    let content = r#"
import React from 'react';
import { Button } from './components/Button.js';  // Actual file: Button.tsx
import { useAuth } from '../hooks/useAuth.js';    // Actual file: useAuth.ts
import type { User } from './types/User.js';

const App = () => {
    const { user } = useAuth();
    return <Button>Hello {user?.name}</Button>;
};
"#;
    let mut parser = TreeSitterParser::new();
    let imports = parser.extract_imports(content, Language::Tsx);

    assert!(
      imports.contains(&"./components/Button.js".to_string()),
      "should preserve .js for .tsx files: {:?}",
      imports
    );
    assert!(
      imports.contains(&"../hooks/useAuth.js".to_string()),
      "should preserve .js for .ts files: {:?}",
      imports
    );
    assert!(
      imports.contains(&"./types/User.js".to_string()),
      "should include type-only imports: {:?}",
      imports
    );
  }

  #[test]
  fn test_typescript_imports() {
    let content = r#"
import { foo, bar } from './module';
import * as utils from '../utils';
import defaultExport from 'package';
const legacy = require('old-package');
export { something } from './other';
"#;
    let mut parser = TreeSitterParser::new();
    let imports = parser.extract_imports(content, Language::TypeScript);

    // Note: strings come with quotes, we strip them in run_query
    assert!(imports.contains(&"./module".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"../utils".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"package".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"old-package".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"./other".to_string()), "imports: {:?}", imports);
  }

  #[test]
  fn test_typescript_calls() {
    let content = r#"
function example() {
    const x = helperFn();
    obj.methodCall();
    data.map().filter().reduce();
    console.log("hello");
    const instance = new MyClass();
}
"#;
    let mut parser = TreeSitterParser::new();
    let calls = parser.extract_calls(content, Language::TypeScript);

    assert!(calls.contains(&"helperFn".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"methodCall".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"map".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"filter".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"reduce".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"log".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"MyClass".to_string()), "calls: {:?}", calls);
  }

  #[test]
  fn test_typescript_definitions() {
    let content = r#"
function myFunction() {}

const arrowFunc = () => {};

class MyClass {
    method() {}
}

interface MyInterface {
    field: string;
}

type MyType = string | number;

export function exportedFunc() {}
export class ExportedClass {}
"#;
    let mut parser = TreeSitterParser::new();
    let defs = parser.extract_definitions(content, Language::TypeScript);

    let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"myFunction"), "defs: {:?}", names);
    assert!(names.contains(&"arrowFunc"), "defs: {:?}", names);
    assert!(names.contains(&"MyClass"), "defs: {:?}", names);
    assert!(names.contains(&"method"), "defs: {:?}", names);
    assert!(names.contains(&"MyInterface"), "defs: {:?}", names);
    assert!(names.contains(&"MyType"), "defs: {:?}", names);
    assert!(names.contains(&"exportedFunc"), "defs: {:?}", names);
    assert!(names.contains(&"ExportedClass"), "defs: {:?}", names);
  }

  #[test]
  fn test_javascript_imports() {
    let content = r#"
import React from 'react';
const fs = require('fs');
import('./dynamic-module');
"#;
    let mut parser = TreeSitterParser::new();
    let imports = parser.extract_imports(content, Language::JavaScript);

    assert!(imports.contains(&"react".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"fs".to_string()), "imports: {:?}", imports);
  }

  #[test]
  fn test_jsx_imports() {
    let content = r#"
import React from 'react';
import { useState, useEffect } from 'react';
import Button from './components/Button';
import * as Icons from './icons';
"#;
    let mut parser = TreeSitterParser::new();
    let imports = parser.extract_imports(content, Language::Jsx);

    assert!(imports.contains(&"react".to_string()), "imports: {:?}", imports);
    assert!(
      imports.contains(&"./components/Button".to_string()),
      "imports: {:?}",
      imports
    );
    assert!(imports.contains(&"./icons".to_string()), "imports: {:?}", imports);
  }

  #[test]
  fn test_jsx_calls() {
    let content = r#"
import React from 'react';

function App() {
    const [count, setCount] = useState(0);

    useEffect(() => {
        console.log('mounted');
    }, []);

    return (
        <div className="app">
            <Header title="Welcome" />
            <Button onClick={() => setCount(count + 1)}>
                Click me
            </Button>
            <Icons.Star size={24} />
            <footer>Copyright</footer>
        </div>
    );
}
"#;
    let mut parser = TreeSitterParser::new();
    let calls = parser.extract_calls(content, Language::Jsx);

    // Regular function calls
    assert!(calls.contains(&"useState".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"useEffect".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"log".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"setCount".to_string()), "calls: {:?}", calls);

    // JSX components (uppercase = component)
    assert!(calls.contains(&"Header".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"Button".to_string()), "calls: {:?}", calls);

    // JSX member expression
    assert!(calls.contains(&"Star".to_string()), "calls: {:?}", calls);
  }

  #[test]
  fn test_jsx_definitions() {
    let content = r#"
import React from 'react';

function FunctionalComponent({ name }) {
    return <div>Hello {name}</div>;
}

const ArrowComponent = ({ children }) => {
    return <span>{children}</span>;
};

class ClassComponent extends React.Component {
    render() {
        return <div>Class</div>;
    }
}

export default function DefaultExport() {
    return <main>Main</main>;
}
"#;
    let mut parser = TreeSitterParser::new();
    let defs = parser.extract_definitions(content, Language::Jsx);

    let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"FunctionalComponent"), "defs: {:?}", names);
    assert!(names.contains(&"ArrowComponent"), "defs: {:?}", names);
    assert!(names.contains(&"ClassComponent"), "defs: {:?}", names);
    assert!(names.contains(&"render"), "defs: {:?}", names);
    assert!(names.contains(&"DefaultExport"), "defs: {:?}", names);
  }

  #[test]
  fn test_tsx_imports() {
    let content = r#"
import React, { FC, useState, useCallback } from 'react';
import type { User, Post } from './types';
import { Button, type ButtonProps } from './components';
import axios from 'axios';
"#;
    let mut parser = TreeSitterParser::new();
    let imports = parser.extract_imports(content, Language::Tsx);

    assert!(imports.contains(&"react".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"./types".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"./components".to_string()), "imports: {:?}", imports);
    assert!(imports.contains(&"axios".to_string()), "imports: {:?}", imports);
  }

  #[test]
  fn test_tsx_calls() {
    let content = r#"
import React, { FC, useState, useEffect } from 'react';

interface Props {
    initialCount: number;
}

const Counter: FC<Props> = ({ initialCount }) => {
    const [count, setCount] = useState<number>(initialCount);
    const [name, setName] = useState<string>('');

    const handleClick = useCallback(() => {
        setCount(prev => prev + 1);
        console.log('clicked');
    }, []);

    return (
        <div>
            <span>{count}</span>
            <Button<number> value={count} onClick={handleClick}>
                Increment
            </Button>
            <Input.Text value={name} onChange={setName} />
            <Tooltip content="Help">
                <Icon name="help" />
            </Tooltip>
        </div>
    );
};

export default Counter;
"#;
    let mut parser = TreeSitterParser::new();
    let calls = parser.extract_calls(content, Language::Tsx);

    // Regular function calls with generics
    assert!(calls.contains(&"useState".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"useCallback".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"setCount".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"log".to_string()), "calls: {:?}", calls);

    // JSX components
    assert!(calls.contains(&"Button".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"Tooltip".to_string()), "calls: {:?}", calls);
    assert!(calls.contains(&"Icon".to_string()), "calls: {:?}", calls);

    // JSX member expression (Input.Text)
    assert!(calls.contains(&"Text".to_string()), "calls: {:?}", calls);
  }

  #[test]
  fn test_tsx_definitions() {
    let content = r#"
import React, { FC, Component } from 'react';

interface UserProps {
    name: string;
    age: number;
}

type Status = 'active' | 'inactive';

function UserCard({ name, age }: UserProps): JSX.Element {
    return <div>{name}</div>;
}

const UserAvatar: FC<{ src: string }> = ({ src }) => {
    return <img src={src} />;
};

class UserProfile extends Component<UserProps> {
    componentDidMount() {
        console.log('mounted');
    }

    render() {
        return <div>Profile</div>;
    }
}

export const useUser = (id: number) => {
    return { name: 'test' };
};
"#;
    let mut parser = TreeSitterParser::new();
    let defs = parser.extract_definitions(content, Language::Tsx);

    let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();

    // Function components
    assert!(names.contains(&"UserCard"), "defs: {:?}", names);
    assert!(names.contains(&"UserAvatar"), "defs: {:?}", names);

    // Class component and methods
    assert!(names.contains(&"UserProfile"), "defs: {:?}", names);
    assert!(names.contains(&"componentDidMount"), "defs: {:?}", names);
    assert!(names.contains(&"render"), "defs: {:?}", names);

    // TypeScript types
    assert!(names.contains(&"UserProps"), "defs: {:?}", names);
    assert!(names.contains(&"Status"), "defs: {:?}", names);

    // Custom hook
    assert!(names.contains(&"useUser"), "defs: {:?}", names);
  }
}
