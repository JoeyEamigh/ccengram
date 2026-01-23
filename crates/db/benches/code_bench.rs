//! Benchmarks for code chunk operations
//!
//! Run with: cargo bench -p db --bench code_bench

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use db::ProjectDb;
use engram_core::{ChunkType, CodeChunk, Language};
use std::hint::black_box;
use std::path::Path;
use tempfile::TempDir;
use uuid::Uuid;

fn create_test_chunk(idx: usize, language: Language) -> CodeChunk {
  let (content, ext) = match language {
    Language::Rust => (
      format!(
        r#"/// Function {} documentation
pub fn function_{}(arg: i32) -> Result<i32, Error> {{
    let result = arg * 2;
    if result > 100 {{
        return Err(Error::TooLarge);
    }}
    Ok(result)
}}"#,
        idx, idx
      ),
      "rs",
    ),
    Language::TypeScript => (
      format!(
        r#"/**
 * Function {} documentation
 */
export function function_{}(arg: number): number {{
    const result = arg * 2;
    if (result > 100) {{
        throw new Error('Too large');
    }}
    return result;
}}"#,
        idx, idx
      ),
      "ts",
    ),
    _ => (format!("// Code chunk {}", idx), "txt"),
  };

  let chunk_id = Uuid::new_v4();
  let tokens_estimate = CodeChunk::estimate_tokens(&content);
  CodeChunk {
    id: chunk_id,
    file_path: format!("src/module_{}.{}", idx, ext),
    content,
    language,
    chunk_type: ChunkType::Function,
    start_line: 1,
    end_line: 10,
    symbols: vec![format!("function_{}", idx)],
    imports: Vec::new(),
    calls: Vec::new(),
    file_hash: format!("hash_{}", idx),
    indexed_at: chrono::Utc::now(),
    tokens_estimate,
    definition_kind: None,
    definition_name: None,
    visibility: None,
    signature: None,
    docstring: None,
    parent_definition: None,
    embedding_text: None,
  }
}

fn bench_code_chunk_add(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();

  let mut group = c.benchmark_group("code_chunk_add");
  group.throughput(Throughput::Elements(1));

  group.bench_function("single", |b| {
    b.iter(|| {
      rt.block_on(async {
        let temp_dir = TempDir::new().unwrap();
        let project_id = engram_core::ProjectId::from_path(Path::new("/bench"));
        let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 4096)
          .await
          .unwrap();

        let chunk = create_test_chunk(0, Language::Rust);
        let vector: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.001).sin()).collect();
        db.add_code_chunk(black_box(&chunk), Some(&vector)).await.unwrap();
      });
    });
  });

  group.finish();
}

fn bench_code_chunk_batch_add(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();

  let mut group = c.benchmark_group("code_chunk_batch_add");

  for size in [10, 50, 100, 200].iter() {
    group.throughput(Throughput::Elements(*size as u64));
    group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
      b.iter(|| {
        rt.block_on(async {
          let temp_dir = TempDir::new().unwrap();
          let project_id = engram_core::ProjectId::from_path(Path::new("/bench"));
          let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 4096)
            .await
            .unwrap();

          let chunks: Vec<(CodeChunk, Vec<f32>)> = (0..size)
            .map(|i| {
              let chunk = create_test_chunk(
                i,
                if i % 2 == 0 {
                  Language::Rust
                } else {
                  Language::TypeScript
                },
              );
              let vector: Vec<f32> = (0..4096).map(|j| ((i + j) as f32 * 0.001).sin()).collect();
              (chunk, vector)
            })
            .collect();

          db.add_code_chunks(&chunks).await.unwrap();
        });
      });
    });
  }

  group.finish();
}

fn bench_code_chunk_search(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();

  // Setup: create a db with code chunks
  let (db, _temp_dir) = rt.block_on(async {
    let temp_dir = TempDir::new().unwrap();
    let project_id = engram_core::ProjectId::from_path(Path::new("/bench"));
    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 4096)
      .await
      .unwrap();

    let chunks: Vec<(CodeChunk, Vec<f32>)> = (0..200)
      .map(|i| {
        let chunk = create_test_chunk(
          i,
          if i % 3 == 0 {
            Language::Rust
          } else {
            Language::TypeScript
          },
        );
        let vector: Vec<f32> = (0..4096).map(|j| ((i + j) as f32 * 0.001).sin()).collect();
        (chunk, vector)
      })
      .collect();

    db.add_code_chunks(&chunks).await.unwrap();

    (db, temp_dir)
  });

  let mut group = c.benchmark_group("code_chunk_search");

  for limit in [5, 10, 20].iter() {
    group.bench_with_input(BenchmarkId::from_parameter(limit), limit, |b, &limit| {
      let query_vec: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.002).cos()).collect();
      b.iter(|| {
        rt.block_on(async {
          db.search_code_chunks(black_box(&query_vec), black_box(limit), None)
            .await
            .unwrap()
        });
      });
    });
  }

  group.finish();
}

fn bench_code_chunk_search_filtered(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();

  // Setup: create a db with code chunks
  let (db, _temp_dir) = rt.block_on(async {
    let temp_dir = TempDir::new().unwrap();
    let project_id = engram_core::ProjectId::from_path(Path::new("/bench"));
    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 4096)
      .await
      .unwrap();

    let chunks: Vec<(CodeChunk, Vec<f32>)> = (0..200)
      .map(|i| {
        let chunk = create_test_chunk(
          i,
          if i % 3 == 0 {
            Language::Rust
          } else {
            Language::TypeScript
          },
        );
        let vector: Vec<f32> = (0..4096).map(|j| ((i + j) as f32 * 0.001).sin()).collect();
        (chunk, vector)
      })
      .collect();

    db.add_code_chunks(&chunks).await.unwrap();

    (db, temp_dir)
  });

  let mut group = c.benchmark_group("code_chunk_search_filtered");

  // Filter by language
  group.bench_function("by_language", |b| {
    let query_vec: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.002).cos()).collect();
    b.iter(|| {
      rt.block_on(async {
        db.search_code_chunks(black_box(&query_vec), 10, Some("language = 'rust'"))
          .await
          .unwrap()
      });
    });
  });

  // Filter by file path pattern
  group.bench_function("by_file_path", |b| {
    let query_vec: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.002).cos()).collect();
    b.iter(|| {
      rt.block_on(async {
        db.search_code_chunks(black_box(&query_vec), 10, Some("file_path LIKE 'src/module_1%'"))
          .await
          .unwrap()
      });
    });
  });

  group.finish();
}

fn bench_code_chunk_get(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();

  // Setup: create a db with code chunks
  let (db, chunk_ids, _temp_dir) = rt.block_on(async {
    let temp_dir = TempDir::new().unwrap();
    let project_id = engram_core::ProjectId::from_path(Path::new("/bench"));
    let db = ProjectDb::open_at_path(project_id, temp_dir.path().join("test.lancedb"), 4096)
      .await
      .unwrap();

    let mut chunk_ids = Vec::new();
    for i in 0..100 {
      let chunk = create_test_chunk(i, Language::Rust);
      chunk_ids.push(chunk.id);
      let vector: Vec<f32> = (0..4096).map(|j| ((i + j) as f32 * 0.001).sin()).collect();
      db.add_code_chunk(&chunk, Some(&vector)).await.unwrap();
    }

    (db, chunk_ids, temp_dir)
  });

  let mut group = c.benchmark_group("code_chunk_get");

  group.bench_function("by_id", |b| {
    let test_id = chunk_ids[50];
    b.iter(|| {
      rt.block_on(async { db.get_code_chunk(black_box(&test_id)).await.unwrap() });
    });
  });

  group.finish();
}

criterion_group!(
  benches,
  bench_code_chunk_add,
  bench_code_chunk_batch_add,
  bench_code_chunk_search,
  bench_code_chunk_search_filtered,
  bench_code_chunk_get
);
criterion_main!(benches);
