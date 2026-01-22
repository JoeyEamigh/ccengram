//! Benchmarks for document chunking operations
//!
//! Run with: cargo bench -p core

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use engram_core::document::{ChunkParams, chunk_text};

/// Generate a realistic document with paragraphs and sentences
fn generate_document(paragraphs: usize, sentences_per_para: usize) -> String {
  let mut doc = String::new();

  for p in 0..paragraphs {
    for s in 0..sentences_per_para {
      doc.push_str(&format!(
        "This is sentence {} in paragraph {}. It contains some text about software development, \
                 including topics like memory management, async programming, and system design. ",
        s + 1,
        p + 1
      ));
    }
    doc.push_str("\n\n");
  }

  doc
}

/// Generate a document with longer technical content
fn generate_technical_document(sections: usize) -> String {
  let mut doc = String::new();

  for s in 0..sections {
    doc.push_str(&format!("## Section {}: Technical Overview\n\n", s + 1));

    doc.push_str(
      "The implementation uses a layered architecture with clear separation of concerns. \
             Each layer has well-defined responsibilities and communicates through interfaces. \
             This design promotes testability and maintainability. \
             Error handling is centralized and follows the fail-fast principle. \
             All public APIs are documented with examples.\n\n",
    );

    doc.push_str(
      "Performance considerations include:\n\
             - Lazy initialization of expensive resources\n\
             - Connection pooling for database access\n\
             - Caching of frequently accessed data\n\
             - Batch processing for bulk operations\n\n",
    );

    doc.push_str(
      "Security measures implemented:\n\
             - Input validation at all boundaries\n\
             - Parameterized queries to prevent injection\n\
             - Rate limiting for API endpoints\n\
             - Encryption for sensitive data at rest\n\n",
    );
  }

  doc
}

fn bench_chunk_text_small(c: &mut Criterion) {
  let mut group = c.benchmark_group("chunk_text_small");
  let params = ChunkParams::default(); // 1000 chars, 200 overlap

  // Small documents that fit in a single chunk
  for size in [100, 500, 900].iter() {
    let doc = "a".repeat(*size);
    group.throughput(Throughput::Bytes(doc.len() as u64));
    group.bench_with_input(BenchmarkId::from_parameter(size), &doc, |b, doc| {
      b.iter(|| chunk_text(black_box(doc), &params));
    });
  }

  group.finish();
}

fn bench_chunk_text_medium(c: &mut Criterion) {
  let mut group = c.benchmark_group("chunk_text_medium");
  let params = ChunkParams::default();

  // Medium documents with paragraphs (2-5 chunks)
  for paras in [5, 10, 20].iter() {
    let doc = generate_document(*paras, 5);
    group.throughput(Throughput::Bytes(doc.len() as u64));
    group.bench_with_input(BenchmarkId::from_parameter(paras), &doc, |b, doc| {
      b.iter(|| chunk_text(black_box(doc), &params));
    });
  }

  group.finish();
}

fn bench_chunk_text_large(c: &mut Criterion) {
  let mut group = c.benchmark_group("chunk_text_large");
  let params = ChunkParams::default();

  // Large documents (many chunks)
  for sections in [10, 25, 50].iter() {
    let doc = generate_technical_document(*sections);
    group.throughput(Throughput::Bytes(doc.len() as u64));
    group.bench_with_input(BenchmarkId::from_parameter(sections), &doc, |b, doc| {
      b.iter(|| chunk_text(black_box(doc), &params));
    });
  }

  group.finish();
}

fn bench_chunk_different_sizes(c: &mut Criterion) {
  let mut group = c.benchmark_group("chunk_params_comparison");

  let doc = generate_technical_document(20);

  // Compare different chunk sizes
  for (chunk_size, overlap) in [(500, 100), (1000, 200), (2000, 400)].iter() {
    let params = ChunkParams {
      chunk_size: *chunk_size,
      overlap: *overlap,
    };

    group.bench_with_input(BenchmarkId::new("chunk_size", chunk_size), &doc, |b, doc| {
      b.iter(|| chunk_text(black_box(doc), &params));
    });
  }

  group.finish();
}

fn bench_sentence_heavy_document(c: &mut Criterion) {
  let mut group = c.benchmark_group("sentence_heavy");

  // Documents with many short sentences (stress tests sentence splitting)
  let short_sentences = (0..500)
    .map(|i| format!("This is sentence {}. ", i))
    .collect::<String>();

  let params = ChunkParams::default();

  group.throughput(Throughput::Bytes(short_sentences.len() as u64));
  group.bench_function("500_short_sentences", |b| {
    b.iter(|| chunk_text(black_box(&short_sentences), &params));
  });

  // Long sentences with complex punctuation
  let complex = (0..100)
    .map(|i| {
      format!(
        "This is a complex sentence #{} with nested clauses (like this one), \
                 multiple sub-parts; various punctuation! And it goes on? Yes, it does. ",
        i
      )
    })
    .collect::<String>();

  group.throughput(Throughput::Bytes(complex.len() as u64));
  group.bench_function("100_complex_sentences", |b| {
    b.iter(|| chunk_text(black_box(&complex), &params));
  });

  group.finish();
}

criterion_group!(
  benches,
  bench_chunk_text_small,
  bench_chunk_text_medium,
  bench_chunk_text_large,
  bench_chunk_different_sizes,
  bench_sentence_heavy_document
);
criterion_main!(benches);
