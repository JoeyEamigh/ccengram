//! Benchmarks for code chunking operations
//!
//! Run with: cargo bench

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use engram_core::Language;
use index::Chunker;

fn generate_rust_code(lines: usize) -> String {
  let mut code = String::new();
  code.push_str("//! Module documentation\n\n");

  for i in 0..(lines / 20) {
    code.push_str(&format!(
      r#"
/// Function {} documentation
pub fn function_{}(arg: i32) -> Result<i32, Error> {{
    let result = arg * 2;
    if result > 100 {{
        return Err(Error::TooLarge);
    }}
    Ok(result)
}}

#[derive(Debug, Clone)]
pub struct Struct_{} {{
    field_a: String,
    field_b: i32,
    field_c: Option<Vec<u8>>,
}}

impl Struct_{} {{
    pub fn new() -> Self {{
        Self {{
            field_a: String::new(),
            field_b: 0,
            field_c: None,
        }}
    }}
}}
"#,
      i, i, i, i
    ));
  }

  code
}

fn generate_typescript_code(lines: usize) -> String {
  let mut code = String::new();
  code.push_str("// TypeScript module\n\n");

  for i in 0..(lines / 20) {
    code.push_str(&format!(
      r#"
/**
 * Function {} documentation
 */
export function function_{}(arg: number): number {{
    const result = arg * 2;
    if (result > 100) {{
        throw new Error('Too large');
    }}
    return result;
}}

interface Interface_{} {{
    fieldA: string;
    fieldB: number;
    fieldC?: Array<number>;
}}

export class Class_{} implements Interface_{} {{
    fieldA: string = '';
    fieldB: number = 0;
    fieldC?: Array<number>;

    constructor() {{
        this.fieldA = 'default';
    }}
}}
"#,
      i, i, i, i, i
    ));
  }

  code
}

fn bench_chunk_rust(c: &mut Criterion) {
  let mut group = c.benchmark_group("chunk_rust");
  let chunker = Chunker::default();

  for size in [100, 500, 1000, 2000].iter() {
    let code = generate_rust_code(*size);
    group.throughput(Throughput::Bytes(code.len() as u64));
    group.bench_with_input(BenchmarkId::from_parameter(size), &code, |b, code| {
      b.iter(|| chunker.chunk(black_box(code), "test.rs", Language::Rust, "checksum123"));
    });
  }

  group.finish();
}

fn bench_chunk_typescript(c: &mut Criterion) {
  let mut group = c.benchmark_group("chunk_typescript");
  let chunker = Chunker::default();

  for size in [100, 500, 1000, 2000].iter() {
    let code = generate_typescript_code(*size);
    group.throughput(Throughput::Bytes(code.len() as u64));
    group.bench_with_input(BenchmarkId::from_parameter(size), &code, |b, code| {
      b.iter(|| chunker.chunk(black_box(code), "test.ts", Language::TypeScript, "checksum123"));
    });
  }

  group.finish();
}

fn bench_symbol_extraction(c: &mut Criterion) {
  let mut group = c.benchmark_group("symbol_extraction");
  let chunker = Chunker::default();

  // Large file for symbol extraction
  let rust_code = generate_rust_code(1000);

  group.bench_function("rust_1000_lines", |b| {
    b.iter(|| {
      let chunks = chunker.chunk(black_box(&rust_code), "test.rs", Language::Rust, "checksum");
      // Force evaluation of symbols
      let total_symbols: usize = chunks.iter().map(|c| c.symbols.len()).sum();
      black_box(total_symbols)
    });
  });

  let ts_code = generate_typescript_code(1000);
  group.bench_function("typescript_1000_lines", |b| {
    b.iter(|| {
      let chunks = chunker.chunk(black_box(&ts_code), "test.ts", Language::TypeScript, "checksum");
      let total_symbols: usize = chunks.iter().map(|c| c.symbols.len()).sum();
      black_box(total_symbols)
    });
  });

  group.finish();
}

criterion_group!(
  benches,
  bench_chunk_rust,
  bench_chunk_typescript,
  bench_symbol_extraction
);
criterion_main!(benches);
