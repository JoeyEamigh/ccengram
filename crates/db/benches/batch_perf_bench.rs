//! Benchmarks comparing single vs batch insert performance
//!
//! These benchmarks establish baselines for the P0 optimization:
//! converting watcher from per-chunk to batch inserts.
//!
//! Run with: cargo bench -p db --bench batch_perf_bench
//!
//! Run specific groups:
//!   cargo bench -p db --bench batch_perf_bench -- "single_vs_batch"
//!   cargo bench -p db --bench batch_perf_bench -- "scaling"

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use db::ProjectDb;
use engram_core::{ChunkType, CodeChunk, Language, ProjectId};
use std::hint::black_box;
use std::path::Path;
use tempfile::TempDir;
use uuid::Uuid;

fn create_test_chunk(idx: usize) -> CodeChunk {
  let content = format!(
    r#"/// Function {} documentation
pub fn function_{}(arg: i32) -> Result<i32, Error> {{
    let result = arg * 2;
    if result > 100 {{
        return Err(Error::TooLarge);
    }}
    Ok(result)
}}"#,
    idx, idx
  );

  let tokens_estimate = CodeChunk::estimate_tokens(&content);
  CodeChunk {
    id: Uuid::new_v4(),
    file_path: format!("src/module_{}.rs", idx / 10), // 10 chunks per "file"
    content,
    language: Language::Rust,
    chunk_type: ChunkType::Function,
    start_line: (idx % 10) as u32 * 10 + 1,
    end_line: (idx % 10) as u32 * 10 + 8,
    symbols: vec![format!("function_{}", idx)],
    imports: Vec::new(),
    calls: Vec::new(),
    file_hash: format!("hash_{}", idx / 10),
    indexed_at: chrono::Utc::now(),
    tokens_estimate,
    definition_kind: Some("function".to_string()),
    definition_name: Some(format!("function_{}", idx)),
    visibility: Some("pub".to_string()),
    signature: Some(format!("fn function_{}(arg: i32) -> Result<i32, Error>", idx)),
    docstring: Some(format!("Function {} documentation", idx)),
    parent_definition: None,
    embedding_text: None,
    content_hash: None,
    caller_count: 0,
    callee_count: 0,
  }
}

fn create_test_vector(seed: usize) -> Vec<f32> {
  (0..4096).map(|i| ((i + seed) as f32 * 0.001).sin()).collect()
}

/// Benchmark: Single inserts vs batch insert
///
/// This directly measures the overhead of per-chunk inserts
/// that the watcher currently uses.
fn bench_single_vs_batch(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();
  let mut group = c.benchmark_group("single_vs_batch");

  for count in [10, 50, 100, 200].iter() {
    group.throughput(Throughput::Elements(*count as u64));

    // Prepare chunks and vectors
    let chunks_and_vectors: Vec<(CodeChunk, Vec<f32>)> = (0..*count)
      .map(|i| (create_test_chunk(i), create_test_vector(i)))
      .collect();

    // Benchmark: Single inserts (current watcher behavior)
    group.bench_with_input(BenchmarkId::new("single_inserts", count), count, |b, &count| {
      b.iter(|| {
        rt.block_on(async {
          let temp_dir = TempDir::new().unwrap();
          let project_id = ProjectId::from_path(Path::new("/bench"));
          let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 4096)
            .await
            .unwrap();

          // Simulate watcher: insert one chunk at a time
          for (chunk, vector) in chunks_and_vectors.iter().take(count) {
            db.add_code_chunk(black_box(chunk), Some(vector)).await.unwrap();
          }
        });
      });
    });

    // Benchmark: Batch insert (target behavior)
    group.bench_with_input(
      BenchmarkId::new("batch_insert", count),
      &chunks_and_vectors,
      |b, chunks| {
        b.iter(|| {
          rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let project_id = ProjectId::from_path(Path::new("/bench"));
            let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 4096)
              .await
              .unwrap();

            db.add_code_chunks(black_box(chunks)).await.unwrap();
          });
        });
      },
    );
  }

  group.finish();
}

/// Benchmark: Scaling characteristics for batch sizes
///
/// Helps determine optimal batch size for the watcher.
fn bench_batch_scaling(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();
  let mut group = c.benchmark_group("batch_scaling");
  group.sample_size(20); // Fewer samples for larger batches

  for count in [50, 100, 250, 500, 1000].iter() {
    group.throughput(Throughput::Elements(*count as u64));

    let chunks_and_vectors: Vec<(CodeChunk, Vec<f32>)> = (0..*count)
      .map(|i| (create_test_chunk(i), create_test_vector(i)))
      .collect();

    group.bench_with_input(BenchmarkId::from_parameter(count), &chunks_and_vectors, |b, chunks| {
      b.iter(|| {
        rt.block_on(async {
          let temp_dir = TempDir::new().unwrap();
          let project_id = ProjectId::from_path(Path::new("/bench"));
          let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 4096)
            .await
            .unwrap();

          db.add_code_chunks(black_box(chunks)).await.unwrap();
        });
      });
    });
  }

  group.finish();
}

/// Benchmark: Delete + insert pattern (current update behavior)
///
/// The watcher deletes all chunks for a file, then re-inserts.
/// This measures that pattern vs alternatives.
fn bench_delete_reinsert(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();
  let mut group = c.benchmark_group("delete_reinsert");

  // Setup: pre-populate database with chunks
  let chunks_per_file = 10;
  let num_files = 20;

  for files_to_update in [1, 5, 10].iter() {
    group.throughput(Throughput::Elements((*files_to_update * chunks_per_file) as u64));

    group.bench_with_input(
      BenchmarkId::new("sequential", files_to_update),
      files_to_update,
      |b, &files_to_update| {
        b.iter(|| {
          rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let project_id = ProjectId::from_path(Path::new("/bench"));
            let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 4096)
              .await
              .unwrap();

            // Pre-populate
            let initial_chunks: Vec<(CodeChunk, Vec<f32>)> = (0..(num_files * chunks_per_file))
              .map(|i| (create_test_chunk(i), create_test_vector(i)))
              .collect();
            db.add_code_chunks(&initial_chunks).await.unwrap();

            // Simulate watcher: delete + reinsert for each file (sequential)
            for file_idx in 0..files_to_update {
              let file_path = format!("src/module_{}.rs", file_idx);
              db.delete_chunks_for_file(&file_path).await.unwrap();

              // Reinsert chunks for this file
              let start = file_idx * chunks_per_file;
              let end = start + chunks_per_file;
              for i in start..end {
                let chunk = create_test_chunk(i);
                let vector = create_test_vector(i);
                db.add_code_chunk(&chunk, Some(&vector)).await.unwrap();
              }
            }
          });
        });
      },
    );

    // Batched version: collect all chunks, single batch insert
    group.bench_with_input(
      BenchmarkId::new("batched", files_to_update),
      files_to_update,
      |b, &files_to_update| {
        b.iter(|| {
          rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let project_id = ProjectId::from_path(Path::new("/bench"));
            let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 4096)
              .await
              .unwrap();

            // Pre-populate
            let initial_chunks: Vec<(CodeChunk, Vec<f32>)> = (0..(num_files * chunks_per_file))
              .map(|i| (create_test_chunk(i), create_test_vector(i)))
              .collect();
            db.add_code_chunks(&initial_chunks).await.unwrap();

            // Delete all affected files first
            for file_idx in 0..files_to_update {
              let file_path = format!("src/module_{}.rs", file_idx);
              db.delete_chunks_for_file(&file_path).await.unwrap();
            }

            // Batch insert all new chunks
            let new_chunks: Vec<(CodeChunk, Vec<f32>)> = (0..(files_to_update * chunks_per_file))
              .map(|i| (create_test_chunk(i), create_test_vector(i)))
              .collect();
            db.add_code_chunks(&new_chunks).await.unwrap();
          });
        });
      },
    );
  }

  group.finish();
}

/// Benchmark: Search performance with varying database sizes
///
/// Ensures optimizations don't regress search performance.
fn bench_search_after_batch(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();
  let mut group = c.benchmark_group("search_after_batch");

  for db_size in [100, 500, 1000].iter() {
    // Setup database with chunks
    let (db, _temp_dir) = rt.block_on(async {
      let temp_dir = TempDir::new().unwrap();
      let project_id = ProjectId::from_path(Path::new("/bench"));
      let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 4096)
        .await
        .unwrap();

      let chunks: Vec<(CodeChunk, Vec<f32>)> = (0..*db_size)
        .map(|i| (create_test_chunk(i), create_test_vector(i)))
        .collect();

      db.add_code_chunks(&chunks).await.unwrap();
      (db, temp_dir)
    });

    let query_vec = create_test_vector(42); // Arbitrary query

    group.bench_with_input(BenchmarkId::from_parameter(db_size), &query_vec, |b, query| {
      b.iter(|| {
        rt.block_on(async { db.search_code_chunks(black_box(query), 10, None).await.unwrap() });
      });
    });
  }

  group.finish();
}

criterion_group!(
  benches,
  bench_single_vs_batch,
  bench_batch_scaling,
  bench_delete_reinsert,
  bench_search_after_batch
);
criterion_main!(benches);
