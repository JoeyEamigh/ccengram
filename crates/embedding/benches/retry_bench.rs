//! Benchmarks for retry and backoff calculation
//!
//! Run with: cargo bench -p embedding --bench retry_bench

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use embedding::{EmbeddingError, RetryConfig, is_retryable_error};
use std::time::Duration;

fn bench_backoff_calculation(c: &mut Criterion) {
  let mut group = c.benchmark_group("backoff_calculation");

  // Test with different configs
  for (name, config) in [
    ("default", RetryConfig::default()),
    ("local", RetryConfig::for_local()),
    ("cloud", RetryConfig::for_cloud()),
  ] {
    group.bench_with_input(BenchmarkId::new("config", name), &config, |b, config| {
      b.iter(|| {
        // Calculate backoff for attempts 0-5
        for attempt in 0..6 {
          black_box(config.backoff_for_attempt(attempt));
        }
      });
    });
  }

  group.finish();
}

fn bench_backoff_with_jitter(c: &mut Criterion) {
  let mut group = c.benchmark_group("backoff_jitter");

  let config_no_jitter = RetryConfig {
    add_jitter: false,
    ..Default::default()
  };

  let config_with_jitter = RetryConfig {
    add_jitter: true,
    ..Default::default()
  };

  group.bench_function("no_jitter", |b| {
    b.iter(|| {
      for attempt in 0..5 {
        black_box(config_no_jitter.backoff_for_attempt(attempt));
      }
    });
  });

  group.bench_function("with_jitter", |b| {
    b.iter(|| {
      for attempt in 0..5 {
        black_box(config_with_jitter.backoff_for_attempt(attempt));
      }
    });
  });

  group.finish();
}

fn bench_is_retryable_error(c: &mut Criterion) {
  let mut group = c.benchmark_group("is_retryable_error");

  // Prepare different error types
  let errors = [
    ("network", EmbeddingError::Network("connection reset".to_string())),
    ("timeout", EmbeddingError::Timeout),
    (
      "rate_limited",
      EmbeddingError::ProviderError("Status 429 Too Many Requests".to_string()),
    ),
    (
      "bad_gateway",
      EmbeddingError::ProviderError("Got 502 Bad Gateway".to_string()),
    ),
    (
      "service_unavail",
      EmbeddingError::ProviderError("Service returned 503".to_string()),
    ),
    (
      "not_retryable",
      EmbeddingError::ProviderError("Invalid input format".to_string()),
    ),
  ];

  for (name, error) in errors.iter() {
    group.bench_with_input(BenchmarkId::from_parameter(name), error, |b, error| {
      b.iter(|| is_retryable_error(black_box(error)));
    });
  }

  group.finish();
}

fn bench_retry_config_creation(c: &mut Criterion) {
  let mut group = c.benchmark_group("retry_config_creation");

  group.bench_function("default", |b| {
    b.iter(|| black_box(RetryConfig::default()));
  });

  group.bench_function("for_local", |b| {
    b.iter(|| black_box(RetryConfig::for_local()));
  });

  group.bench_function("for_cloud", |b| {
    b.iter(|| black_box(RetryConfig::for_cloud()));
  });

  group.bench_function("custom", |b| {
    b.iter(|| {
      black_box(RetryConfig {
        max_retries: 5,
        initial_backoff: Duration::from_millis(100),
        max_backoff: Duration::from_secs(10),
        backoff_multiplier: 1.5,
        add_jitter: true,
        request_timeout: Duration::from_secs(45),
      })
    });
  });

  group.finish();
}

criterion_group!(
  benches,
  bench_backoff_calculation,
  bench_backoff_with_jitter,
  bench_is_retryable_error,
  bench_retry_config_creation
);
criterion_main!(benches);
