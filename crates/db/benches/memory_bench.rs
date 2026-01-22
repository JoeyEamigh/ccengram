//! Benchmarks for memory database operations
//!
//! NOTE: These benchmarks use synthetic vectors to isolate database performance
//! from embedding service latency. This allows consistent benchmarking of:
//! - Memory insertion performance
//! - Vector search performance
//! - Memory listing/filtering
//!
//! For end-to-end benchmarks including embedding generation, see the daemon crate
//! integration tests or run manual benchmarks with `ccengram` CLI.
//!
//! Run with: cargo bench -p db

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use db::ProjectDb;
use engram_core::{Memory, Sector};
use std::path::Path;
use tempfile::TempDir;
use uuid::Uuid;

fn create_test_memory(project_id: Uuid, idx: usize) -> Memory {
  let mut memory = Memory::new(
    project_id,
    format!(
      "Test memory content #{} with some additional text for embedding. \
             This simulates a realistic memory about code patterns, decisions, or preferences. \
             The user prefers to use async/await over callbacks for handling asynchronous operations.",
      idx
    ),
    Sector::Semantic,
  );
  memory.content_hash = format!("hash_{}", idx);
  memory.tags = vec!["test".to_string(), "benchmark".to_string()];
  memory.concepts = vec!["async".to_string(), "patterns".to_string()];
  memory
}

fn bench_memory_add(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();

  let mut group = c.benchmark_group("memory_add");
  group.throughput(Throughput::Elements(1));

  group.bench_function("single", |b| {
    b.iter(|| {
      rt.block_on(async {
        let temp_dir = TempDir::new().unwrap();
        let project_id = engram_core::ProjectId::from_path(Path::new("/bench"));
        let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 768)
          .await
          .unwrap();

        let memory = create_test_memory(Uuid::new_v4(), 0);
        let vector: Vec<f32> = (0..768).map(|i| (i as f32 * 0.001).sin()).collect();
        db.add_memory(black_box(&memory), Some(&vector)).await.unwrap();
      });
    });
  });

  group.finish();
}

fn bench_memory_batch_add(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();

  let mut group = c.benchmark_group("memory_batch_add");

  for size in [10, 50, 100].iter() {
    group.throughput(Throughput::Elements(*size as u64));
    group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
      b.iter(|| {
        rt.block_on(async {
          let temp_dir = TempDir::new().unwrap();
          let project_id = engram_core::ProjectId::from_path(Path::new("/bench"));
          let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 768)
            .await
            .unwrap();

          let project_uuid = Uuid::new_v4();
          for i in 0..size {
            let memory = create_test_memory(project_uuid, i);
            let vector: Vec<f32> = (0..768).map(|j| ((i + j) as f32 * 0.001).sin()).collect();
            db.add_memory(&memory, Some(&vector)).await.unwrap();
          }
        });
      });
    });
  }

  group.finish();
}

fn bench_memory_search(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();

  // Setup: create a db with 100 memories
  let (db, _temp_dir) = rt.block_on(async {
    let temp_dir = TempDir::new().unwrap();
    let project_id = engram_core::ProjectId::from_path(Path::new("/bench"));
    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 768)
      .await
      .unwrap();

    let project_uuid = Uuid::new_v4();
    for i in 0..100 {
      let memory = create_test_memory(project_uuid, i);
      let vector: Vec<f32> = (0..768).map(|j| ((i + j) as f32 * 0.001).sin()).collect();
      db.add_memory(&memory, Some(&vector)).await.unwrap();
    }

    (db, temp_dir)
  });

  let mut group = c.benchmark_group("memory_search");

  for limit in [5, 10, 20].iter() {
    group.bench_with_input(BenchmarkId::from_parameter(limit), limit, |b, &limit| {
      let query_vec: Vec<f32> = (0..768).map(|i| (i as f32 * 0.002).cos()).collect();
      b.iter(|| {
        rt.block_on(async {
          db.search_memories(black_box(&query_vec), black_box(limit), None)
            .await
            .unwrap()
        });
      });
    });
  }

  group.finish();
}

fn bench_memory_list(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();

  // Setup: create a db with memories
  let (db, _temp_dir) = rt.block_on(async {
    let temp_dir = TempDir::new().unwrap();
    let project_id = engram_core::ProjectId::from_path(Path::new("/bench"));
    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 768)
      .await
      .unwrap();

    let project_uuid = Uuid::new_v4();
    for i in 0..200 {
      let memory = create_test_memory(project_uuid, i);
      let vector: Vec<f32> = (0..768).map(|j| ((i + j) as f32 * 0.001).sin()).collect();
      db.add_memory(&memory, Some(&vector)).await.unwrap();
    }

    (db, temp_dir)
  });

  let mut group = c.benchmark_group("memory_list");

  // Benchmark listing with different filters
  group.bench_function("no_filter", |b| {
    b.iter(|| {
      rt.block_on(async { db.list_memories(None, Some(50)).await.unwrap() });
    });
  });

  group.bench_function("with_filter", |b| {
    b.iter(|| {
      rt.block_on(async { db.list_memories(Some("sector = 'semantic'"), Some(50)).await.unwrap() });
    });
  });

  group.finish();
}

criterion_group!(
  benches,
  bench_memory_add,
  bench_memory_batch_add,
  bench_memory_search,
  bench_memory_list
);
criterion_main!(benches);
