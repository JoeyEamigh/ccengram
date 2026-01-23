//! Benchmarks for embedding batch performance
//!
//! These benchmarks measure:
//! 1. Sequential embed() calls (current watcher behavior)
//! 2. embed_batch() with semaphore parallelism (current batch impl)
//! 3. True batch API (proposed /api/embed endpoint)
//!
//! REQUIRES: Ollama running locally with qwen3-embedding model
//!
//! Run with: cargo bench -p embedding --bench batch_embed_bench
//!
//! Run specific groups:
//!   cargo bench -p embedding --bench batch_embed_bench -- "sequential_vs_batch"
//!   cargo bench -p embedding --bench batch_embed_bench -- "concurrency"
//!
//! Note: These benchmarks are marked with sample_size(10) due to network latency.
//! Results will vary based on Ollama server load and GPU utilization.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use embedding::{EmbeddingProvider, OllamaProvider};
use std::hint::black_box;

fn generate_code_chunks(count: usize) -> Vec<String> {
  (0..count)
    .map(|i| {
      format!(
        r#"/// Function {} documentation
pub fn function_{}(arg: i32) -> Result<i32, Error> {{
    let result = arg * 2;
    if result > 100 {{
        return Err(Error::TooLarge);
    }}
    Ok(result)
}}"#,
        i, i
      )
    })
    .collect()
}

/// Benchmark: Sequential embed() calls vs embed_batch()
///
/// This measures the overhead of calling embed() in a loop
/// vs using the batched interface.
fn bench_sequential_vs_batch(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();
  let provider = OllamaProvider::new();

  // Check if Ollama is available
  let available = rt.block_on(provider.is_available());
  if !available {
    eprintln!("Ollama not available - skipping embedding benchmarks");
    eprintln!("Start Ollama with: ollama serve");
    eprintln!("Pull model with: ollama pull qwen3-embedding");
    return;
  }

  let mut group = c.benchmark_group("sequential_vs_batch");
  group.sample_size(10); // Fewer samples due to network latency

  for count in [5, 10, 20].iter() {
    let chunks = generate_code_chunks(*count);
    group.throughput(Throughput::Elements(*count as u64));

    // Sequential: call embed() for each chunk (current watcher behavior)
    group.bench_with_input(BenchmarkId::new("sequential", count), &chunks, |b, chunks| {
      b.iter(|| {
        rt.block_on(async {
          let mut results = Vec::with_capacity(chunks.len());
          for chunk in chunks {
            let embedding = provider.embed(black_box(chunk)).await.unwrap();
            results.push(embedding);
          }
          results
        })
      });
    });

    // Batch: use embed_batch() (parallel with semaphore)
    group.bench_with_input(BenchmarkId::new("batch_parallel", count), &chunks, |b, chunks| {
      let refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
      b.iter(|| rt.block_on(async { provider.embed_batch(black_box(&refs)).await.unwrap() }));
    });
  }

  group.finish();
}

/// Benchmark: Different batch sizes for throughput optimization
///
/// Helps determine optimal batch size for watcher processing.
fn bench_batch_sizes(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();
  let provider = OllamaProvider::new();

  let available = rt.block_on(provider.is_available());
  if !available {
    eprintln!("Ollama not available - skipping batch size benchmarks");
    return;
  }

  let mut group = c.benchmark_group("batch_sizes");
  group.sample_size(10);

  // Test different batch sizes to find optimal
  for count in [10, 25, 50, 100].iter() {
    let chunks = generate_code_chunks(*count);
    let refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();

    group.throughput(Throughput::Elements(*count as u64));

    group.bench_with_input(BenchmarkId::from_parameter(count), &refs, |b, refs| {
      b.iter(|| rt.block_on(async { provider.embed_batch(black_box(refs)).await.unwrap() }));
    });
  }

  group.finish();
}

/// Benchmark: Single embedding latency
///
/// Baseline measurement for individual embed() calls.
fn bench_single_embed_latency(c: &mut Criterion) {
  let rt = tokio::runtime::Runtime::new().unwrap();
  let provider = OllamaProvider::new();

  let available = rt.block_on(provider.is_available());
  if !available {
    eprintln!("Ollama not available - skipping latency benchmark");
    return;
  }

  let mut group = c.benchmark_group("single_embed_latency");
  group.sample_size(20);

  // Different content sizes
  let short_content = "fn foo() {}";
  let medium_content = generate_code_chunks(1).pop().unwrap();
  let long_content = (0..10)
    .map(|i| {
      format!(
        "pub fn function_{}(x: i32) -> i32 {{ x * {} }}\n",
        i,
        i + 1
      )
    })
    .collect::<String>();

  group.bench_function("short", |b| {
    b.iter(|| rt.block_on(async { provider.embed(black_box(short_content)).await.unwrap() }));
  });

  group.bench_function("medium", |b| {
    b.iter(|| rt.block_on(async { provider.embed(black_box(&medium_content)).await.unwrap() }));
  });

  group.bench_function("long", |b| {
    b.iter(|| rt.block_on(async { provider.embed(black_box(&long_content)).await.unwrap() }));
  });

  group.finish();
}

criterion_group!(benches, bench_sequential_vs_batch, bench_batch_sizes, bench_single_embed_latency);
criterion_main!(benches);
