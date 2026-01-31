//! E2E Benchmark Harness for CCEngram Explore/Context Tools
//!
//! This crate provides comprehensive benchmarking for testing the exploration
//! capabilities of CCEngram's `explore` and `context` tools against large
//! real-world codebases (Zed, VSCode).
//!
//! ## Key Concepts
//!
//! - **Scenarios**: TOML-defined multi-step exploration tasks
//! - **Metrics**: Performance (latency, throughput) and accuracy (recall, noise ratio)
//! - **Ground Truth**: Call graph analysis, noise patterns, optional annotations
//! - **Reports**: JSON (machine-readable) and Markdown (human-readable)

use std::path::PathBuf;

use ccengram::ipc::Client;
use clap::{Parser, Subcommand};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tracing::{info, warn};
use tracing_subscriber::{EnvFilter, fmt};

use self::{
  fixtures::FixtureGenerator,
  indexing::{IncrementalBenchConfig, IncrementalBenchmark, IndexingBenchmark, IndexingComparison, IndexingReport},
  reports::{ComparisonReport, generate_reports},
  repos::{RepoCache, RepoRegistry, TargetRepo, default_cache_dir, prepare_repo},
  scenarios::{Scenario, ScenarioRunner, filter_scenarios, load_scenarios_from_dir, run_scenarios_parallel},
  watcher::{WatcherBenchConfig, WatcherBenchmark, WatcherTestType},
};

mod fixtures;
mod ground_truth;
mod indexing;
mod llm_judge;
mod metrics;
mod reports;
mod repos;
mod scenarios;
mod session;
mod watcher;

/// Benchmark-specific errors
#[derive(Debug, thiserror::Error)]
enum BenchmarkError {
  #[error("Repository error: {0}")]
  Repo(String),
  #[error("Scenario error: {0}")]
  Scenario(String),
  #[error("Execution error: {0}")]
  Execution(String),
  #[error("IO error: {0}")]
  Io(#[from] std::io::Error),
  #[error("TOML parse error: {0}")]
  Toml(#[from] toml::de::Error),
  #[error("JSON error: {0}")]
  Json(#[from] serde_json::Error),
  #[error("HTTP error: {0}")]
  Http(#[from] reqwest::Error),
  #[error("IPC error: {0}")]
  Ipc(#[from] ccengram::ipc::IpcError),
}

type Result<T> = std::result::Result<T, BenchmarkError>;

#[derive(Parser)]
#[command(name = "ccengram-bench")]
#[command(about = "Benchmark harness for CCEngram explore/context tools")]
#[command(version)]
struct Cli {
  /// Enable verbose logging
  #[arg(short, long, global = true)]
  verbose: bool,

  #[command(subcommand)]
  command: Commands,
}

#[derive(Subcommand)]
enum Commands {
  /// Run benchmark scenarios
  Run {
    /// Output directory for results
    #[arg(short, long, default_value = "./benchmark-results")]
    output: PathBuf,

    /// Filter scenarios by pattern (supports glob wildcards)
    #[arg(short, long)]
    scenarios: Option<String>,

    /// Enable LLM-as-judge evaluation (expensive)
    #[arg(long)]
    llm_judge: bool,

    /// Scenarios directory
    #[arg(long)]
    scenarios_dir: Option<PathBuf>,

    /// Run scenarios in parallel
    #[arg(long)]
    parallel: bool,

    /// Name for this benchmark run
    #[arg(long)]
    name: Option<String>,
  },

  /// Compare two benchmark results for regressions
  Compare {
    /// Baseline results file (JSON)
    baseline: PathBuf,

    /// Current results file (JSON)
    current: PathBuf,

    /// Regression threshold percentage
    #[arg(short, long, default_value = "10")]
    threshold: f64,

    /// Output comparison report
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Type of comparison: scenario (default) or indexing
    #[arg(long, default_value = "scenario")]
    compare_type: String,
  },

  /// Download repositories
  Download {
    /// Repositories to download (comma-separated: zed,vscode or 'all')
    #[arg(short, long, default_value = "all")]
    repos: String,

    /// Force re-download
    #[arg(long)]
    force: bool,

    /// Cache directory
    #[arg(long)]
    cache_dir: Option<PathBuf>,
  },

  /// Index repositories (code and docs) via daemon
  Index {
    /// Repositories to index (comma-separated: zed,vscode or 'all')
    #[arg(short, long, default_value = "all")]
    repos: String,

    /// Force re-index even if already indexed
    #[arg(long)]
    force: bool,

    /// Cache directory for repositories
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Embedding provider to use (ollama or openrouter)
    #[arg(long, default_value = "ollama")]
    embedding_provider: String,

    /// OpenRouter API key (required if --embedding-provider=openrouter)
    /// Falls back to OPENROUTER_API_KEY environment variable if not provided
    #[arg(long)]
    openrouter_api_key: Option<String>,
  },

  /// List available scenarios
  List {
    /// Scenarios directory
    #[arg(long)]
    scenarios_dir: Option<PathBuf>,

    /// Show detailed information
    #[arg(short, long)]
    detailed: bool,
  },

  /// Clean cached data
  Clean {
    /// Clean all cached data (repos and databases)
    #[arg(long)]
    all: bool,

    /// Specific repository to clean
    #[arg(long)]
    repo: Option<String>,

    /// Clean only repository caches (downloaded source code)
    #[arg(long)]
    repos_only: bool,

    /// Clean only LanceDB databases (indexed data)
    #[arg(long)]
    db_only: bool,

    /// Cache directory for repos
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Data directory for databases (defaults to ~/.local/share/ccengram)
    #[arg(long)]
    data_dir: Option<PathBuf>,
  },

  /// Benchmark indexing performance
  IndexPerf {
    /// Repositories to benchmark (comma-separated: zed,vscode or 'all')
    #[arg(short, long, default_value = "all")]
    repos: String,

    /// Number of iterations per repository
    #[arg(short, long, default_value = "3")]
    iterations: usize,

    /// Output directory for results
    #[arg(short, long, default_value = "./benchmark-results")]
    output: PathBuf,

    /// Force cold start (clear index before each iteration)
    #[arg(long)]
    cold: bool,

    /// Cache directory for repositories
    #[arg(long)]
    cache_dir: Option<PathBuf>,
  },

  /// Benchmark incremental indexing performance
  IncrementalPerf {
    /// Repositories to benchmark (comma-separated: zed,vscode or 'all')
    #[arg(short, long, default_value = "all")]
    repos: String,

    /// Number of files to modify per iteration
    #[arg(short, long, default_value = "10")]
    files_per_iter: usize,

    /// Number of iterations per repository
    #[arg(short, long, default_value = "3")]
    iterations: usize,

    /// Output directory for results
    #[arg(short, long, default_value = "./benchmark-results")]
    output: PathBuf,

    /// Cache directory for repositories
    #[arg(long)]
    cache_dir: Option<PathBuf>,
  },

  /// Benchmark file watcher performance
  WatcherPerf {
    /// Repository to benchmark
    #[arg(short, long, default_value = "zed")]
    repo: String,

    /// Number of iterations per test
    #[arg(short, long, default_value = "5")]
    iterations: usize,

    /// Output directory for results
    #[arg(short, long, default_value = "./benchmark-results")]
    output: PathBuf,

    /// Cache directory for repositories
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Specific test to run (lifecycle, single, batch, operations, gitignore)
    #[arg(long)]
    test: Option<String>,
  },

  /// Test large file handling
  LargeFilePerf {
    /// Output directory for results
    #[arg(short, long, default_value = "./benchmark-results")]
    output: PathBuf,

    /// File sizes to test in MB (comma-separated)
    #[arg(long, default_value = "1,5,10,50")]
    sizes_mb: String,

    /// Repository to use for testing
    #[arg(short, long, default_value = "zed")]
    repo: String,

    /// Cache directory for repositories
    #[arg(long)]
    cache_dir: Option<PathBuf>,
  },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  let cli = Cli::parse();

  // Setup logging - respect RUST_LOG if set, otherwise use -v flag
  let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
    let level = if cli.verbose { "debug" } else { "info" };
    EnvFilter::new(level)
  });
  let subscriber = fmt::Subscriber::builder()
    .with_env_filter(filter)
    .with_target(false)
    .finish();
  tracing::subscriber::set_global_default(subscriber)?;

  match cli.command {
    Commands::Run {
      output,
      scenarios,
      llm_judge,
      scenarios_dir,
      parallel,
      name,
    } => run_benchmarks(output, scenarios, llm_judge, scenarios_dir, parallel, name).await,
    Commands::Compare {
      baseline,
      current,
      threshold,
      output,
      compare_type,
    } => compare_results(baseline, current, threshold, output, compare_type).await,
    Commands::Download {
      repos,
      force,
      cache_dir,
    } => download_repos(repos, force, cache_dir).await,
    Commands::Index {
      repos,
      force,
      cache_dir,
      embedding_provider,
      openrouter_api_key,
    } => index_repos_streaming(repos, force, cache_dir, embedding_provider, openrouter_api_key).await,
    Commands::List {
      scenarios_dir,
      detailed,
    } => list_scenarios(scenarios_dir, detailed).await,
    Commands::Clean {
      all,
      repo,
      repos_only,
      db_only,
      cache_dir,
      data_dir,
    } => clean_cache(all, repo, repos_only, db_only, cache_dir, data_dir).await,
    Commands::IndexPerf {
      repos,
      iterations,
      output,
      cold,
      cache_dir,
    } => run_indexing_benchmark(repos, iterations, output, cold, cache_dir).await,
    Commands::IncrementalPerf {
      repos,
      files_per_iter,
      iterations,
      output,
      cache_dir,
    } => run_incremental_benchmark(repos, files_per_iter, iterations, output, cache_dir).await,
    Commands::WatcherPerf {
      repo,
      iterations,
      output,
      cache_dir,
      test,
    } => run_watcher_benchmark(repo, iterations, output, cache_dir, test).await,
    Commands::LargeFilePerf {
      output,
      sizes_mb,
      repo,
      cache_dir,
    } => run_large_file_benchmark(output, sizes_mb, repo, cache_dir).await,
  }
}

async fn run_benchmarks(
  output: PathBuf,
  scenario_filter: Option<String>,
  llm_judge: bool,
  scenarios_dir: Option<PathBuf>,
  parallel: bool,
  run_name: Option<String>,
) -> anyhow::Result<()> {
  use std::collections::HashMap;

  let socket_path = ScenarioRunner::default_socket_path();

  // Load scenarios
  let scenarios_dir = scenarios_dir.unwrap_or_else(|| PathBuf::from("crates/benchmark/scenarios"));
  info!("Loading scenarios from: {}", scenarios_dir.display());

  let all_scenarios = load_scenarios_from_dir(&scenarios_dir).await?;
  if all_scenarios.is_empty() {
    warn!("No scenarios found in {}", scenarios_dir.display());
    return Ok(());
  }

  // Filter scenarios
  let scenarios: Vec<Scenario> = if let Some(pattern) = &scenario_filter {
    filter_scenarios(&all_scenarios, pattern).into_iter().cloned().collect()
  } else {
    all_scenarios.clone()
  };

  if scenarios.is_empty() {
    warn!(
      "No scenarios match filter: {}",
      scenario_filter.as_deref().unwrap_or("*")
    );
    return Ok(());
  }

  info!("Running {} scenarios", scenarios.len());

  let annotations_dir = scenarios_dir.parent().map(|p| p.join("annotations"));

  // Group scenarios by repo
  let mut scenarios_by_repo: HashMap<TargetRepo, Vec<&Scenario>> = HashMap::new();
  for scenario in &scenarios {
    scenarios_by_repo
      .entry(scenario.metadata.repo)
      .or_default()
      .push(scenario);
  }

  // Prepare repos - verify they're downloaded and indexed
  let mut repo_paths: HashMap<TargetRepo, PathBuf> = HashMap::new();
  for repo in scenarios_by_repo.keys() {
    // Ensure repo is downloaded
    let repo_path = match prepare_repo(*repo, None).await {
      Ok(path) => path,
      Err(e) => {
        anyhow::bail!(
          "Repository {} not available. Run:\n  cargo run -p benchmark -- download --repos {}\nError: {}",
          repo,
          repo,
          e
        );
      }
    };

    // Check if repo is indexed (quick stats check)
    if let Err(e) = check_repo_indexed(&socket_path, &repo_path).await {
      anyhow::bail!(
        "Repository {} not indexed. Run:\n  cargo run -p benchmark -- index --repos {}\nError: {}",
        repo,
        repo,
        e
      );
    }

    repo_paths.insert(*repo, repo_path.to_path_buf());
  }

  // Run scenarios grouped by repo
  let mut results = Vec::new();

  // Progress bar for sequential execution
  let pb = if !parallel {
    let pb = ProgressBar::new(scenarios.len() as u64);
    pb.set_style(
      ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
        .unwrap()
        .progress_chars("#>-"),
    );
    Some(pb)
  } else {
    None
  };

  for (repo, repo_scenarios) in &scenarios_by_repo {
    let repo_path = repo_paths.get(repo).unwrap();

    // Create runner for this repo
    let client = Client::connect(repo_path.clone()).await?;
    let runner = ScenarioRunner::new(client, annotations_dir.clone());
    let runner = if llm_judge { runner.with_llm_judge()? } else { runner };

    // Check daemon
    if !runner.check_daemon().await {
      anyhow::bail!(
        "CCEngram daemon is not running. Start it with: ccengram daemon\n\
               Socket: {}",
        socket_path
      );
    }

    if parallel {
      info!("Running {} scenarios for {} in parallel", repo_scenarios.len(), repo);
      // Clone scenarios for parallel execution
      let scenarios_owned: Vec<Scenario> = repo_scenarios.iter().map(|s| (*s).clone()).collect();
      let repo_results = run_scenarios_parallel(&runner, &scenarios_owned).await;
      results.extend(repo_results);
    } else {
      for scenario in repo_scenarios {
        if let Some(ref pb) = pb {
          pb.set_message(scenario.metadata.id.clone());
        }

        match runner.run(scenario).await {
          Ok(result) => {
            let status = if result.passed { "✓" } else { "✗" };
            info!("{} {} ({}ms)", status, scenario.metadata.id, result.total_duration_ms);

            // Export discovered annotations for passed scenarios (useful for building ground truth)
            if result.passed && std::env::var("CCENGRAM_EXPORT_ANNOTATIONS").is_ok() {
              let export_dir = output.join("discovered_annotations");
              let export_path = export_dir.join(format!("{}.json", scenario.metadata.id));
              if let Err(e) = tokio::fs::create_dir_all(&export_dir).await {
                warn!("Failed to create export dir: {}", e);
              } else if let Err(e) = runner
                .export_annotations(
                  &scenario.metadata.id,
                  &result.accuracy.files_found,
                  &result.accuracy.symbols_found,
                  &export_path,
                )
                .await
              {
                warn!("Failed to export annotations: {}", e);
              }
            }

            // Build call graph from discovered callers/callees for analysis
            let mut discovered_calls: Vec<(String, String)> = Vec::new();
            for step in &result.steps {
              // Build call relationships from step callers -> step symbols
              for caller in &step.callers {
                for symbol in &step.symbols_found {
                  discovered_calls.push((caller.clone(), symbol.clone()));
                }
              }
              // Build call relationships from step symbols -> step callees
              for symbol in &step.symbols_found {
                for callee in &step.callees {
                  discovered_calls.push((symbol.clone(), callee.clone()));
                }
              }
            }

            if !discovered_calls.is_empty() {
              let call_graph = runner.build_call_graph_from_results(discovered_calls);
              tracing::debug!(
                "Scenario {} discovered call graph: {} symbols, {} edges",
                scenario.metadata.id,
                call_graph.symbol_count(),
                call_graph.edge_count()
              );
            }

            results.push(result);
          }
          Err(e) => {
            warn!("Failed to run {}: {}", scenario.metadata.id, e);
          }
        }

        if let Some(ref pb) = pb {
          pb.inc(1);
        }
      }
    }
  }

  if let Some(pb) = pb {
    pb.finish_with_message("done");
  }

  // Generate reports
  info!("Generating reports in: {}", output.display());
  generate_reports(&results, &output, run_name.as_deref()).await?;

  // Print summary
  let passed = results.iter().filter(|r| r.passed).count();
  let failed = results.len() - passed;
  println!("\n{} passed, {} failed", passed, failed);

  if failed > 0 {
    std::process::exit(1);
  }

  Ok(())
}

/// Check if a repo is indexed by querying code stats.
async fn check_repo_indexed(_socket_path: &str, repo_path: &std::path::Path) -> anyhow::Result<()> {
  use ccengram::ipc::code::CodeStatsParams;

  let client = Client::connect(repo_path.to_path_buf()).await?;
  let result = client.call(CodeStatsParams).await?;

  let chunks = result.total_chunks;
  if chunks == 0 {
    anyhow::bail!("No code indexed (0 chunks)");
  }
  info!(
    "  {} has {} chunks indexed",
    repo_path.file_name().unwrap_or_default().to_string_lossy(),
    chunks
  );

  Ok(())
}

/// Start the daemon with a specific embedding provider.
async fn start_daemon_with_provider(provider: &str, api_key: Option<&str>) -> anyhow::Result<()> {
  use std::process::Stdio;

  use tokio::process::Command;

  let mut cmd = Command::new("ccengram");
  cmd.arg("daemon");
  cmd.arg("--background");
  cmd.arg("--embedding-provider");
  cmd.arg(provider);

  if let Some(key) = api_key {
    cmd.arg("--openrouter-api-key");
    cmd.arg(key);
  } else if provider == "openrouter" {
    // Try to get from env
    if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
      cmd.arg("--openrouter-api-key");
      cmd.arg(key);
    }
  }

  cmd.stdin(Stdio::null());
  cmd.stdout(Stdio::null());
  cmd.stderr(Stdio::null());

  let child = cmd.spawn()?;

  // Detach the child process
  drop(child);

  Ok(())
}

/// Index repositories (code and docs) with streaming progress display.
async fn index_repos_streaming(
  repos: String,
  force: bool,
  cache_dir: Option<PathBuf>,
  embedding_provider: String,
  openrouter_api_key: Option<String>,
) -> anyhow::Result<()> {
  use tokio::net::UnixStream;

  // Validate embedding provider settings
  let provider = embedding_provider.to_lowercase();
  match provider.as_str() {
    "openrouter" => {
      let has_key = openrouter_api_key.is_some() || std::env::var("OPENROUTER_API_KEY").is_ok();
      if !has_key {
        anyhow::bail!("OpenRouter API key required. Provide --openrouter-api-key or set OPENROUTER_API_KEY");
      }
      info!("Using OpenRouter embedding provider");
    }
    "ollama" => {
      info!("Using Ollama embedding provider");
    }
    other => {
      anyhow::bail!("Unknown embedding provider: {}. Use 'ollama' or 'openrouter'", other);
    }
  }

  let targets: Vec<TargetRepo> = if repos == "all" {
    TargetRepo::all().to_vec()
  } else {
    repos
      .split(',')
      .filter_map(|s| TargetRepo::from_name(s.trim()))
      .collect()
  };

  if targets.is_empty() {
    anyhow::bail!("No valid repositories specified. Use: zed, vscode, or 'all'");
  }

  let socket_path = ScenarioRunner::default_socket_path();

  // Check if daemon is running, start it if not
  if UnixStream::connect(&socket_path).await.is_err() {
    info!("Daemon not running, starting with {} provider...", provider);
    start_daemon_with_provider(&provider, openrouter_api_key.as_deref()).await?;

    // Wait for daemon to be ready
    for _ in 0..30 {
      tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
      if UnixStream::connect(&socket_path).await.is_ok() {
        break;
      }
    }

    if UnixStream::connect(&socket_path).await.is_err() {
      anyhow::bail!("Failed to start daemon");
    }
  }

  for repo in targets {
    // Ensure repo is downloaded first
    let repo_path = match prepare_repo(repo, cache_dir.clone()).await {
      Ok(path) => path,
      Err(e) => {
        warn!("Repository {} not downloaded: {}", repo, e);
        info!("Run: cargo run -p benchmark -- download --repos {}", repo);
        continue;
      }
    };

    let repo_config = RepoRegistry::get(repo);

    // Index code
    info!("Indexing code for {} at {}", repo, repo_path.display());
    index_code_for_repo(&socket_path, &repo_path, force).await?;

    // Index docs if docs_dir is configured
    if let Some(ref docs_dir) = repo_config.docs_dir {
      let docs_path = repo_path.join(docs_dir);
      if docs_path.exists() {
        info!("Indexing docs for {} at {}", repo, docs_path.display());
        index_docs_for_repo(&socket_path, &repo_path, &docs_path).await?;
      } else {
        info!("No docs directory found at {}", docs_path.display());
      }
    }
  }

  Ok(())
}

/// Index code for a single repository.
async fn index_code_for_repo(_socket_path: &str, repo_path: &std::path::Path, force: bool) -> anyhow::Result<()> {
  use std::collections::HashMap;

  use ccengram::ipc::{StreamUpdate, code::CodeIndexParams};

  let client = Client::connect(repo_path.to_path_buf()).await?;
  let mut rx = client.call_streaming(CodeIndexParams { force, stream: true }).await?;

  let mp = MultiProgress::new();
  let mut stage_bars: HashMap<String, ProgressBar> = HashMap::new();
  let stage_style = ProgressStyle::default_bar()
    .template("{prefix:>12.cyan} [{bar:30.cyan/blue}] {pos:>5}/{len:>5} {msg}")
    .unwrap()
    .progress_chars("=>-");

  while let Some(update) = rx.recv().await {
    match update {
      StreamUpdate::Progress {
        message: _,
        percent: _,
        stage,
      } => {
        if let (Some(stage_name), Some(processed), Some(total)) = (&stage.stage, stage.processed, stage.total) {
          let pb = stage_bars.entry(stage_name.clone()).or_insert_with(|| {
            let pb = mp.add(ProgressBar::new(total as u64));
            pb.set_style(stage_style.clone());
            pb.set_prefix(stage_name.clone());
            pb
          });
          pb.set_length(total as u64);
          pb.set_position(processed as u64);
          if let Some(ref file) = stage.current_file {
            let msg = if file.len() > 40 {
              format!("...{}", &file[file.len() - 37..])
            } else {
              file.clone()
            };
            pb.set_message(msg);
          }
        }
      }
      StreamUpdate::Done(result) => {
        for (_, pb) in stage_bars {
          pb.finish_and_clear();
        }
        let result = result?;
        println!(
          "  Code: {} files indexed, {} chunks created",
          result.files_indexed, result.chunks_created
        );
        break;
      }
    }
  }

  Ok(())
}

/// Index docs for a single repository.
async fn index_docs_for_repo(
  _socket_path: &str,
  repo_path: &std::path::Path,
  docs_path: &std::path::Path,
) -> anyhow::Result<()> {
  use std::collections::HashMap;

  use ccengram::ipc::{StreamUpdate, docs::DocsIngestParams};

  let client = Client::connect(repo_path.to_path_buf()).await?;
  let mut rx = client
    .call_streaming(DocsIngestParams {
      directory: Some(docs_path.to_string_lossy().to_string()),
      file: None,
      stream: true,
    })
    .await?;

  let mp = MultiProgress::new();
  let mut stage_bars: HashMap<String, ProgressBar> = HashMap::new();
  let stage_style = ProgressStyle::default_bar()
    .template("{prefix:>12.cyan} [{bar:30.cyan/blue}] {pos:>5}/{len:>5} {msg}")
    .unwrap()
    .progress_chars("=>-");

  while let Some(update) = rx.recv().await {
    match update {
      StreamUpdate::Progress {
        message: _,
        percent: _,
        stage,
      } => {
        if let (Some(stage_name), Some(processed), Some(total)) = (&stage.stage, stage.processed, stage.total) {
          let pb = stage_bars.entry(stage_name.clone()).or_insert_with(|| {
            let pb = mp.add(ProgressBar::new(total as u64));
            pb.set_style(stage_style.clone());
            pb.set_prefix(stage_name.clone());
            pb
          });
          pb.set_length(total as u64);
          pb.set_position(processed as u64);
          if let Some(ref file) = stage.current_file {
            let msg = if file.len() > 40 {
              format!("...{}", &file[file.len() - 37..])
            } else {
              file.clone()
            };
            pb.set_message(msg);
          }
        }
      }
      StreamUpdate::Done(result) => {
        for (_, pb) in stage_bars {
          pb.finish_and_clear();
        }
        let result = result?;
        println!(
          "  Docs: {} files ingested, {} chunks created",
          result.files_ingested, result.chunks_created
        );
        break;
      }
    }
  }

  Ok(())
}

async fn compare_results(
  baseline: PathBuf,
  current: PathBuf,
  threshold: f64,
  output: Option<PathBuf>,
  compare_type: String,
) -> anyhow::Result<()> {
  info!(
    "Comparing {} vs {} (threshold: {:.0}%, type: {})",
    baseline.display(),
    current.display(),
    threshold,
    compare_type
  );

  match compare_type.as_str() {
    "indexing" => {
      // Load indexing reports
      let baseline_content = tokio::fs::read_to_string(&baseline).await?;
      let current_content = tokio::fs::read_to_string(&current).await?;

      let baseline_report: IndexingReport = serde_json::from_str(&baseline_content)?;
      let current_report: IndexingReport = serde_json::from_str(&current_content)?;

      let comparison = IndexingComparison::compare(&baseline_report, &current_report, threshold);

      // Print markdown summary
      println!("{}", comparison.to_markdown());

      // Save if output specified
      if let Some(output) = output {
        let json = serde_json::to_string_pretty(&comparison)?;
        tokio::fs::write(&output, json).await?;
        info!("Comparison saved to: {}", output.display());
      }

      if !comparison.passes {
        std::process::exit(1);
      }
    }
    _ => {
      // Default: scenario comparison
      let comparison = ComparisonReport::from_files(&baseline, &current, threshold).await?;

      // Print markdown summary
      println!("{}", comparison.to_markdown());

      // Save if output specified
      if let Some(output) = output {
        comparison.save(&output).await?;
        info!("Comparison saved to: {}", output.display());
      }

      if !comparison.summary.passes {
        std::process::exit(1);
      }
    }
  }

  Ok(())
}

async fn download_repos(repos: String, force: bool, cache_dir: Option<PathBuf>) -> anyhow::Result<()> {
  let cache_dir = cache_dir.unwrap_or_else(default_cache_dir);
  let cache = RepoCache::new(cache_dir.clone());

  let targets: Vec<TargetRepo> = if repos == "all" {
    TargetRepo::all().to_vec()
  } else {
    repos
      .split(',')
      .filter_map(|s| TargetRepo::from_name(s.trim()))
      .collect()
  };

  if targets.is_empty() {
    anyhow::bail!("No valid repositories specified. Use: zed, vscode, or 'all'");
  }

  for repo in targets {
    let config = RepoRegistry::get(repo);
    info!("Downloading {} ({})", repo, config.release_tag);

    if force {
      info!("Removing existing cache for {}", repo);
      cache.remove(repo).await?;
    }

    match prepare_repo(repo, Some(cache_dir.clone())).await {
      Ok(path) => {
        info!("Repository downloaded to: {}", path.display());
      }
      Err(e) => {
        warn!("Failed to download {}: {}", repo, e);
      }
    }
  }

  Ok(())
}

async fn list_scenarios(scenarios_dir: Option<PathBuf>, detailed: bool) -> anyhow::Result<()> {
  let scenarios_dir = scenarios_dir.unwrap_or_else(|| PathBuf::from("crates/benchmark/scenarios"));

  let scenarios = load_scenarios_from_dir(&scenarios_dir).await?;

  if scenarios.is_empty() {
    println!("No scenarios found in {}", scenarios_dir.display());
    return Ok(());
  }

  println!("Available scenarios ({}):\n", scenarios.len());

  for scenario in &scenarios {
    if detailed {
      println!("  {} - {}", scenario.metadata.id, scenario.metadata.name);
      println!("    Repo: {}", scenario.metadata.repo);
      println!("    Difficulty: {:?}", scenario.metadata.difficulty);
      println!("    Steps: {}", scenario.steps.len());
      println!("    Expected files: {}", scenario.expected.must_find_files.len());
      println!("    Expected symbols: {}", scenario.expected.must_find_symbols.len());
      println!();
    } else {
      println!(
        "  {:30} {:10} {:10} ({} steps)",
        scenario.metadata.id,
        format!("{}", scenario.metadata.repo),
        format!("{:?}", scenario.metadata.difficulty).to_lowercase(),
        scenario.steps.len()
      );
    }
  }

  Ok(())
}

async fn run_indexing_benchmark(
  repos: String,
  iterations: usize,
  output: PathBuf,
  cold: bool,
  cache_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
  let socket_path = ScenarioRunner::default_socket_path();

  let targets: Vec<TargetRepo> = if repos == "all" {
    TargetRepo::all().to_vec()
  } else {
    repos
      .split(',')
      .filter_map(|s| TargetRepo::from_name(s.trim()))
      .collect()
  };

  if targets.is_empty() {
    anyhow::bail!("No valid repositories specified. Use: zed, vscode, or 'all'");
  }

  info!(
    "Running indexing benchmark: {} repos, {} iterations, cold={}",
    targets.len(),
    iterations,
    cold
  );

  // Create benchmark runner
  let client = Client::connect(cache_dir.clone().unwrap_or_else(default_cache_dir)).await?;
  let mut benchmark = IndexingBenchmark::new(client, cache_dir);

  // Check daemon
  if !benchmark.check_daemon().await {
    anyhow::bail!(
      "CCEngram daemon is not running. Start it with: ccengram daemon\n\
             Socket: {}",
      socket_path
    );
  }

  // Run benchmark
  let report = benchmark.run(&targets, iterations, cold).await?;

  // Save reports
  report.save(&output).await?;

  // Print summary
  println!("\n{}", report.to_markdown());

  Ok(())
}

async fn clean_cache(
  all: bool,
  repo: Option<String>,
  repos_only: bool,
  db_only: bool,
  cache_dir: Option<PathBuf>,
  data_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
  use ccengram::dirs::default_data_dir;

  let cache_dir = cache_dir.unwrap_or_else(default_cache_dir);
  let data_dir = data_dir.unwrap_or_else(default_data_dir);
  let cache = RepoCache::new(cache_dir.clone());

  // Determine what to clean
  let clean_repos = !db_only;
  let clean_dbs = !repos_only;

  if all {
    info!("Cleaning all cached data");

    if clean_repos {
      cache.clean_all().await?;
      println!("Repository cache cleaned");
    }

    if clean_dbs {
      // Clean databases for all known benchmark repos
      for target in TargetRepo::all() {
        if let Some(project_dir) = get_project_data_dir(&cache, *target, &data_dir)
          && project_dir.exists()
        {
          tokio::fs::remove_dir_all(&project_dir).await?;
          info!("Removed database for {}: {}", target, project_dir.display());
          println!("Database cleaned for {}", target);
        }
      }
    }

    println!("All benchmark data cleaned");
  } else if let Some(repo_name) = repo {
    if let Some(target) = TargetRepo::from_name(&repo_name) {
      info!("Cleaning data for {}", target);

      if clean_repos {
        cache.remove(target).await?;
        println!("Repository cache cleaned for {}", target);
      }

      if clean_dbs {
        if let Some(project_dir) = get_project_data_dir(&cache, target, &data_dir) {
          if project_dir.exists() {
            tokio::fs::remove_dir_all(&project_dir).await?;
            info!("Removed database: {}", project_dir.display());
            println!("Database cleaned for {}", target);
          } else {
            println!("No database found for {}", target);
          }
        } else {
          println!("Repository {} not cached, cannot determine database location", target);
        }
      }
    } else {
      anyhow::bail!("Unknown repository: {}", repo_name);
    }
  } else {
    println!("Usage: clean [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --all         Clean all repos and databases");
    println!("  --repo <name> Clean specific repo (zed, vscode)");
    println!("  --repos-only  Only clean repository caches");
    println!("  --db-only     Only clean LanceDB databases");
    println!();
    println!("Examples:");
    println!("  clean --all              # Clean everything");
    println!("  clean --all --db-only    # Clean all databases only");
    println!("  clean --repo zed         # Clean zed repo and database");
    println!("  clean --repo zed --db-only  # Clean only zed's database");
  }

  Ok(())
}

/// Get the project data directory for a benchmark repo.
fn get_project_data_dir(cache: &RepoCache, repo: TargetRepo, data_dir: &std::path::Path) -> Option<PathBuf> {
  use ccengram::project::ProjectId;

  let repo_path = cache.repo_path(repo);
  if !repo_path.exists() {
    return None;
  }

  // Canonicalize the path to match how ProjectId computes the hash
  let canonical = repo_path.canonicalize().ok()?;
  let project_id = ProjectId::from_path_exact(&canonical);
  Some(project_id.data_dir(data_dir))
}

async fn run_incremental_benchmark(
  repos: String,
  files_per_iter: usize,
  iterations: usize,
  output: PathBuf,
  cache_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
  let targets: Vec<TargetRepo> = if repos == "all" {
    TargetRepo::all().to_vec()
  } else {
    repos
      .split(',')
      .filter_map(|s| TargetRepo::from_name(s.trim()))
      .collect()
  };

  if targets.is_empty() {
    anyhow::bail!("No valid repositories specified. Use: zed, vscode, or 'all'");
  }

  info!(
    "Running incremental indexing benchmark: {} repos, {} files/iter, {} iterations",
    targets.len(),
    files_per_iter,
    iterations
  );

  // Check if daemon is running before connecting
  let socket_path = ScenarioRunner::default_socket_path();
  if !std::path::Path::new(&socket_path).exists() {
    anyhow::bail!(
      "CCEngram daemon is not running (socket not found). Start it with: ccengram daemon\n\
             Expected socket: {}",
      socket_path
    );
  }

  // Create benchmark runner
  let client = match Client::connect(cache_dir.clone().unwrap_or_else(default_cache_dir)).await {
    Ok(c) => c,
    Err(e) => {
      anyhow::bail!(
        "Failed to connect to CCEngram daemon: {}\n\
               Start it with: ccengram daemon\n\
               Socket: {}",
        e,
        socket_path
      );
    }
  };
  let config = IncrementalBenchConfig {
    files_per_iteration: files_per_iter,
    iterations,
    threshold_ms_per_file: 200,
  };
  let benchmark = IncrementalBenchmark::new(client, cache_dir).with_config(config);

  // Verify daemon is healthy
  if !benchmark.check_daemon().await {
    anyhow::bail!(
      "CCEngram daemon is not responding. Restart it with: ccengram daemon\n\
             Socket: {}",
      socket_path
    );
  }

  // Run benchmark
  let mut benchmark = benchmark; // rebind as mutable for run()
  let report = benchmark.run(&targets).await?;

  // Save reports
  report.save(&output).await?;

  // Print summary
  println!("\n{}", report.to_markdown());

  if !report.summary.passes {
    std::process::exit(1);
  }

  Ok(())
}

async fn run_watcher_benchmark(
  repo: String,
  iterations: usize,
  output: PathBuf,
  cache_dir: Option<PathBuf>,
  test_filter: Option<String>,
) -> anyhow::Result<()> {
  let target = TargetRepo::from_name(&repo).ok_or_else(|| anyhow::anyhow!("Unknown repository: {}", repo))?;

  let test_type = test_filter.and_then(|t| WatcherTestType::from_str(&t));

  info!(
    "Running watcher benchmark: {}, {} iterations, test: {:?}",
    target, iterations, test_type
  );

  // Create benchmark runner
  let client = Client::connect(cache_dir.clone().unwrap_or_else(default_cache_dir)).await?;
  let config = WatcherBenchConfig {
    iterations,
    test_filter: test_type,
    ..Default::default()
  };
  let mut benchmark = WatcherBenchmark::new(client, cache_dir).with_config(config);

  // Run benchmark
  let report = benchmark.run(target).await?;

  // Save reports
  report.save(&output).await?;

  // Print summary
  println!("\n{}", report.to_markdown());

  if !report.summary.passes {
    std::process::exit(1);
  }

  Ok(())
}

async fn run_large_file_benchmark(
  output: PathBuf,
  sizes_mb: String,
  repo: String,
  cache_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
  use ccengram::ipc::code::{CodeIndexParams, CodeSearchParams, CodeStatsParams};

  let target = TargetRepo::from_name(&repo).ok_or_else(|| anyhow::anyhow!("Unknown repository: {}", repo))?;

  let sizes: Vec<u64> = sizes_mb
    .split(',')
    .filter_map(|s| s.trim().parse::<u64>().ok())
    .collect();

  if sizes.is_empty() {
    anyhow::bail!("No valid sizes specified. Use comma-separated MB values like: 1,5,10,50");
  }

  info!("Running large file benchmark: repo={}, sizes={:?} MB", target, sizes);

  let repo_path = prepare_repo(target, cache_dir.clone()).await?;
  let client = Client::connect(repo_path.clone()).await?;

  // Get daemon PID for resource monitoring
  use ccengram::ipc::system::StatusParams;
  let daemon_status = client.call(StatusParams).await?;
  let daemon_pid = daemon_status.pid;

  let mut results = Vec::new();
  let mut monitor = metrics::ResourceMonitor::new(daemon_pid);

  for size_mb in sizes {
    let size_bytes = size_mb * 1024 * 1024;
    info!("Testing large file: {} MB", size_mb);

    let mut fixtures = FixtureGenerator::new(&repo_path).await?;

    // Get initial stats
    let initial_stats = client.call(CodeStatsParams).await?;
    let initial_chunks = initial_stats.total_chunks;

    // Create large file
    let (path, marker) = fixtures.create_large_file(size_bytes).await?;
    let actual_size = tokio::fs::metadata(&path).await?.len();

    monitor.snapshot();
    let start = std::time::Instant::now();

    // Trigger indexing
    let _ = client
      .call(CodeIndexParams {
        force: false,
        stream: false,
      })
      .await?;

    let elapsed = start.elapsed();
    monitor.snapshot();

    // Check if file was indexed
    let search_result = client
      .call(CodeSearchParams {
        query: marker.clone(),
        limit: Some(1),
        ..Default::default()
      })
      .await?;

    let final_stats = client.call(CodeStatsParams).await?;
    let chunks_added = final_stats.total_chunks.saturating_sub(initial_chunks);

    let indexed = !search_result.chunks.is_empty();
    let skip_reason = if !indexed {
      Some("File may be too large or skipped by indexer".to_string())
    } else {
      None
    };

    results.push(metrics::LargeFileBenchResult {
      file_size_bytes: actual_size,
      indexed,
      chunks_created: if indexed { Some(chunks_added) } else { None },
      processing_time_ms: elapsed.as_millis() as u64,
      peak_memory_bytes: monitor.peak_memory(),
      skip_reason,
    });

    // Cleanup
    fixtures.cleanup().await?;
  }

  // Generate report
  let report = metrics::IncrementalReport {
    timestamp: chrono::Utc::now().to_rfc3339(),
    version: env!("CARGO_PKG_VERSION").to_string(),
    results: Vec::new(),
    large_file_results: results.clone(),
    summary: metrics::IncrementalSummary {
      avg_time_per_file_ms: 0.0,
      max_time_per_file_ms: 0.0,
      detection_accuracy: 1.0,
      false_positive_rate: 0.0,
      max_indexed_file_bytes: results
        .iter()
        .filter(|r| r.indexed)
        .map(|r| r.file_size_bytes)
        .max()
        .unwrap_or(0),
      passes: true,
    },
  };

  // Save
  tokio::fs::create_dir_all(&output).await?;

  let json_path = output.join("large_file.json");
  let json = serde_json::to_string_pretty(&report)?;
  tokio::fs::write(&json_path, json).await?;
  info!("Saved JSON report: {}", json_path.display());

  // Print results
  println!("\n# Large File Benchmark Results\n");
  println!("| Size | Indexed | Chunks | Time | Memory |");
  println!("|------|---------|--------|------|--------|");

  for result in &results {
    let size_mb = result.file_size_bytes as f64 / (1024.0 * 1024.0);
    let chunks = result
      .chunks_created
      .map(|c| c.to_string())
      .unwrap_or_else(|| "-".to_string());
    let indexed = if result.indexed { "Yes" } else { "No" };
    let memory_mb = result.peak_memory_bytes as f64 / (1024.0 * 1024.0);

    println!(
      "| {:.1} MB | {} | {} | {} ms | {:.1} MB |",
      size_mb, indexed, chunks, result.processing_time_ms, memory_mb,
    );
  }

  Ok(())
}
