//! Benchmarks for file scanning operations
//!
//! Run with: cargo bench -p index --bench scanner_bench

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use index::Scanner;
use std::fs;
use tempfile::TempDir;

fn create_test_repo(file_count: usize, avg_lines: usize) -> TempDir {
  let dir = TempDir::new().unwrap();

  // Create .git directory so gitignore is respected
  fs::create_dir(dir.path().join(".git")).unwrap();

  // Create .gitignore
  fs::write(dir.path().join(".gitignore"), "target/\nnode_modules/\n*.log").unwrap();

  // Create source files
  let src_dir = dir.path().join("src");
  fs::create_dir(&src_dir).unwrap();

  for i in 0..file_count {
    let content = generate_rust_file(avg_lines, i);
    let filename = format!("module_{}.rs", i);
    fs::write(src_dir.join(&filename), content).unwrap();
  }

  // Create some TypeScript files
  let ts_dir = dir.path().join("frontend");
  fs::create_dir(&ts_dir).unwrap();

  for i in 0..(file_count / 3) {
    let content = generate_ts_file(avg_lines, i);
    let filename = format!("component_{}.ts", i);
    fs::write(ts_dir.join(&filename), content).unwrap();
  }

  // Create some files that should be ignored
  let target_dir = dir.path().join("target");
  fs::create_dir(&target_dir).unwrap();
  for i in 0..10 {
    fs::write(target_dir.join(format!("build_{}.rs", i)), "ignored").unwrap();
  }

  dir
}

fn generate_rust_file(lines: usize, seed: usize) -> String {
  let mut content = String::new();
  content.push_str("//! Module documentation\n\n");
  content.push_str("use std::collections::HashMap;\n\n");

  let funcs_needed = lines / 10;
  for i in 0..funcs_needed {
    content.push_str(&format!(
      r#"
/// Function {} documentation
pub fn function_{}_{}(arg: i32) -> i32 {{
    let x = arg * 2;
    let y = x + {};
    y
}}
"#,
      i,
      seed,
      i,
      i + seed
    ));
  }

  content
}

fn generate_ts_file(lines: usize, seed: usize) -> String {
  let mut content = String::new();
  content.push_str("// TypeScript component\n\n");
  content.push_str("import { useState } from 'react';\n\n");

  let funcs_needed = lines / 10;
  for i in 0..funcs_needed {
    content.push_str(&format!(
      r#"
/**
 * Function {} documentation
 */
export function function_{}_{}(arg: number): number {{
    const x = arg * 2;
    const y = x + {};
    return y;
}}
"#,
      i,
      seed,
      i,
      i + seed
    ));
  }

  content
}

fn bench_scan_small_repo(c: &mut Criterion) {
  let mut group = c.benchmark_group("scan_small_repo");
  let scanner = Scanner::new();

  // 20 files, 50 lines each
  let dir = create_test_repo(20, 50);

  group.bench_function("20_files", |b| {
    b.iter(|| {
      scanner.scan(black_box(dir.path()), |_| {});
    });
  });

  group.finish();
}

fn bench_scan_medium_repo(c: &mut Criterion) {
  let mut group = c.benchmark_group("scan_medium_repo");
  let scanner = Scanner::new();

  // 100 files, 100 lines each
  let dir = create_test_repo(100, 100);

  group.bench_function("100_files", |b| {
    b.iter(|| {
      scanner.scan(black_box(dir.path()), |_| {});
    });
  });

  group.finish();
}

fn bench_scan_with_file_sizes(c: &mut Criterion) {
  let mut group = c.benchmark_group("scan_file_sizes");
  let scanner = Scanner::new();

  for lines in [50, 200, 500].iter() {
    let dir = create_test_repo(30, *lines);
    group.bench_with_input(BenchmarkId::from_parameter(lines), lines, |b, _| {
      b.iter(|| {
        scanner.scan(black_box(dir.path()), |_| {});
      });
    });
  }

  group.finish();
}

fn bench_scan_single_file(c: &mut Criterion) {
  let mut group = c.benchmark_group("scan_single_file");
  let scanner = Scanner::new();

  let dir = TempDir::new().unwrap();
  let file_path = dir.path().join("test.rs");
  fs::write(&file_path, generate_rust_file(200, 0)).unwrap();

  group.bench_function("200_lines", |b| {
    b.iter(|| {
      scanner.scan_file(black_box(&file_path), black_box(dir.path()));
    });
  });

  group.finish();
}

criterion_group!(
  benches,
  bench_scan_small_repo,
  bench_scan_medium_repo,
  bench_scan_with_file_sizes,
  bench_scan_single_file
);
criterion_main!(benches);
