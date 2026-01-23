//! Benchmarks for deduplication and decay operations
//!
//! Run with: cargo bench -p extract

use chrono::Utc;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use engram_core::{Memory, Sector};
use extract::{decay, dedup};
use std::hint::black_box;
use uuid::Uuid;

fn generate_memory_content(idx: usize, variation: usize) -> String {
  let base = format!(
    "The user prefers to use async/await patterns in TypeScript. \
         This is memory number {} about code patterns and best practices. \
         We should always use proper error handling with try/catch blocks.",
    idx
  );
  if variation > 0 {
    format!("{} Additional variation text #{}", base, variation)
  } else {
    base
  }
}

fn create_test_memories(count: usize) -> Vec<Memory> {
  let project_id = Uuid::new_v4();
  (0..count)
    .map(|i| {
      let mut memory = Memory::new(project_id, generate_memory_content(i, 0), Sector::Semantic);
      memory.importance = 0.5 + (i as f32 * 0.01);
      memory.salience = 1.0;
      memory
    })
    .collect()
}

fn bench_simhash(c: &mut Criterion) {
  let mut group = c.benchmark_group("simhash");

  for size in [100, 500, 1000, 2000].iter() {
    let content = "a ".repeat(*size);
    group.throughput(Throughput::Bytes(*size as u64 * 2));
    group.bench_with_input(BenchmarkId::from_parameter(size), &content, |b, content| {
      b.iter(|| dedup::simhash(black_box(content)));
    });
  }

  group.finish();
}

fn bench_hamming_distance(c: &mut Criterion) {
  let mut group = c.benchmark_group("hamming_distance");

  let hash1 = 0xDEADBEEFu64;
  let hash2 = 0xCAFEBABEu64;

  group.bench_function("compute", |b| {
    b.iter(|| dedup::hamming_distance(black_box(hash1), black_box(hash2)));
  });

  group.finish();
}

fn bench_content_hash(c: &mut Criterion) {
  let mut group = c.benchmark_group("content_hash");

  for size in [100, 500, 1000, 5000].iter() {
    let content = "test content ".repeat(*size / 13 + 1);
    group.throughput(Throughput::Bytes(content.len() as u64));
    group.bench_with_input(BenchmarkId::from_parameter(size), &content, |b, content| {
      b.iter(|| dedup::content_hash(black_box(content)));
    });
  }

  group.finish();
}

fn bench_jaccard_similarity(c: &mut Criterion) {
  let mut group = c.benchmark_group("jaccard_similarity");

  let content1 = "The user prefers async/await patterns for handling asynchronous operations in TypeScript";
  let content2 = "The user likes to use async/await instead of callbacks for async operations";

  group.bench_function("similar_texts", |b| {
    b.iter(|| dedup::jaccard_similarity(black_box(content1), black_box(content2)));
  });

  let different1 = "Configuration for database connection pooling";
  let different2 = "User interface styling guidelines and best practices";

  group.bench_function("different_texts", |b| {
    b.iter(|| dedup::jaccard_similarity(black_box(different1), black_box(different2)));
  });

  group.finish();
}

fn bench_duplicate_checker(c: &mut Criterion) {
  let mut group = c.benchmark_group("duplicate_checker");

  // Create checker with existing content
  for size in [10, 50, 100, 200].iter() {
    group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
      let checker = dedup::DuplicateChecker::new();
      let memories: Vec<Memory> = (0..size)
        .map(|i| {
          let content = generate_memory_content(i, 0);
          let (hash, sh) = dedup::compute_hashes(&content);
          let project_id = Uuid::new_v4();
          let mut memory = Memory::new(project_id, content, Sector::Semantic);
          memory.content_hash = hash;
          memory.simhash = sh;
          memory
        })
        .collect();

      let new_content = generate_memory_content(999, 0);
      b.iter(|| checker.find_duplicate(black_box(&new_content), black_box(&memories)));
    });
  }

  group.finish();
}

fn bench_decay_apply(c: &mut Criterion) {
  let mut group = c.benchmark_group("decay_apply");
  let config = decay::DecayConfig::default();
  let future = Utc::now() + chrono::Duration::days(30);

  let memories = create_test_memories(100);

  group.bench_function("batch_100", |b| {
    b.iter(|| {
      let mut mems = memories.clone();
      decay::apply_decay_batch(black_box(&mut mems), future, &config);
    });
  });

  let large_memories = create_test_memories(500);
  group.bench_function("batch_500", |b| {
    b.iter(|| {
      let mut mems = large_memories.clone();
      decay::apply_decay_batch(black_box(&mut mems), future, &config);
    });
  });

  group.finish();
}

fn bench_decay_predict(c: &mut Criterion) {
  let mut group = c.benchmark_group("decay_predict");

  group.bench_function("predict_salience", |b| {
    b.iter(|| {
      decay::predict_salience(
        black_box(1.0),
        black_box(0.7),
        black_box(Sector::Semantic),
        black_box(5),
        black_box(7.0),
      )
    });
  });

  group.bench_function("days_until_salience", |b| {
    b.iter(|| {
      decay::days_until_salience(
        black_box(1.0),
        black_box(0.3),
        black_box(0.7),
        black_box(Sector::Semantic),
        black_box(5),
      )
    });
  });

  group.finish();
}

fn bench_decay_stats(c: &mut Criterion) {
  let mut group = c.benchmark_group("decay_stats");
  let config = decay::DecayConfig::default();
  let future = Utc::now() + chrono::Duration::days(30);

  let memories = create_test_memories(100);
  let mut mems = memories.clone();
  let results = decay::apply_decay_batch(&mut mems, future, &config);

  group.bench_function("compute_100", |b| {
    b.iter(|| decay::DecayStats::from_results(black_box(&results)));
  });

  let large_memories = create_test_memories(500);
  let mut large_mems = large_memories.clone();
  let large_results = decay::apply_decay_batch(&mut large_mems, future, &config);

  group.bench_function("compute_500", |b| {
    b.iter(|| decay::DecayStats::from_results(black_box(&large_results)));
  });

  group.finish();
}

criterion_group!(
  benches,
  bench_simhash,
  bench_hamming_distance,
  bench_content_hash,
  bench_jaccard_similarity,
  bench_duplicate_checker,
  bench_decay_apply,
  bench_decay_predict,
  bench_decay_stats
);
criterion_main!(benches);
