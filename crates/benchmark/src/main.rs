//! CCEngram Benchmark CLI
//!
//! Run benchmarks against the explore/context tools using real codebases.

use anyhow::Result;
use benchmark::{
    repos::{default_cache_dir, prepare_repo, RepoCache, RepoRegistry, TargetRepo},
    reports::{generate_reports, ComparisonReport},
    scenarios::{
        filter_scenarios, load_scenarios_from_dir, run_scenarios_parallel, Scenario,
        ScenarioRunner,
    },
};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

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
    },

    /// Prepare repositories (download and index)
    Index {
        /// Repositories to prepare (comma-separated: zed,vscode or 'all')
        #[arg(short, long, default_value = "all")]
        repos: String,

        /// Force re-download
        #[arg(long)]
        force: bool,

        /// Cache directory
        #[arg(long)]
        cache_dir: Option<PathBuf>,
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
        /// Clean all cached data
        #[arg(long)]
        all: bool,

        /// Specific repository to clean
        #[arg(long)]
        repo: Option<String>,

        /// Cache directory
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let level = if cli.verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
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
        } => {
            run_benchmarks(output, scenarios, llm_judge, scenarios_dir, parallel, name).await
        }
        Commands::Compare {
            baseline,
            current,
            threshold,
            output,
        } => compare_results(baseline, current, threshold, output),
        Commands::Index {
            repos,
            force,
            cache_dir,
        } => index_repos(repos, force, cache_dir).await,
        Commands::List {
            scenarios_dir,
            detailed,
        } => list_scenarios(scenarios_dir, detailed),
        Commands::Clean {
            all,
            repo,
            cache_dir,
        } => clean_cache(all, repo, cache_dir),
    }
}

async fn run_benchmarks(
    output: PathBuf,
    scenario_filter: Option<String>,
    llm_judge: bool,
    scenarios_dir: Option<PathBuf>,
    parallel: bool,
    run_name: Option<String>,
) -> Result<()> {
    if llm_judge {
        warn!("LLM-as-judge evaluation is not yet implemented");
    }

    // Load scenarios
    let scenarios_dir = scenarios_dir.unwrap_or_else(|| PathBuf::from("crates/benchmark/scenarios"));
    info!("Loading scenarios from: {}", scenarios_dir.display());

    let all_scenarios = load_scenarios_from_dir(&scenarios_dir)?;
    if all_scenarios.is_empty() {
        warn!("No scenarios found in {}", scenarios_dir.display());
        return Ok(());
    }

    // Filter scenarios
    let scenarios: Vec<Scenario> = if let Some(pattern) = &scenario_filter {
        filter_scenarios(&all_scenarios, pattern)
            .into_iter()
            .cloned()
            .collect()
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

    // Create runner
    let socket_path = ScenarioRunner::default_socket_path();
    let project_path = std::env::current_dir()?.to_string_lossy().to_string();

    // Determine annotations directory (sibling to scenarios dir)
    let annotations_dir = scenarios_dir.parent().map(|p| p.join("annotations"));
    let runner = ScenarioRunner::new(&socket_path, &project_path, annotations_dir);

    // Check daemon
    if !runner.check_daemon().await {
        anyhow::bail!(
            "CCEngram daemon is not running. Start it with: ccengram daemon\n\
             Socket: {}",
            socket_path
        );
    }

    // Run scenarios (parallel or sequential)
    let results = if parallel {
        info!("Running scenarios in parallel");
        run_scenarios_parallel(&runner, &scenarios).await
    } else {
        // Progress bar for sequential execution
        let pb = ProgressBar::new(scenarios.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut results = Vec::new();
        for scenario in &scenarios {
            pb.set_message(scenario.metadata.id.clone());

            match runner.run(scenario).await {
                Ok(result) => {
                    let status = if result.passed { "✓" } else { "✗" };
                    info!(
                        "{} {} ({}ms)",
                        status, scenario.metadata.id, result.total_duration_ms
                    );
                    results.push(result);
                }
                Err(e) => {
                    warn!("Failed to run {}: {}", scenario.metadata.id, e);
                }
            }

            pb.inc(1);
        }

        pb.finish_with_message("done");
        results
    };

    // Generate reports
    info!("Generating reports in: {}", output.display());
    generate_reports(&results, &output, run_name.as_deref())?;

    // Print summary
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.len() - passed;
    println!("\n{} passed, {} failed", passed, failed);

    if failed > 0 {
        std::process::exit(1);
    }

    Ok(())
}

fn compare_results(
    baseline: PathBuf,
    current: PathBuf,
    threshold: f64,
    output: Option<PathBuf>,
) -> Result<()> {
    info!(
        "Comparing {} vs {} (threshold: {:.0}%)",
        baseline.display(),
        current.display(),
        threshold
    );

    let comparison = ComparisonReport::from_files(&baseline, &current, threshold)?;

    // Print markdown summary
    println!("{}", comparison.to_markdown());

    // Save if output specified
    if let Some(output) = output {
        comparison.save(&output)?;
        info!("Comparison saved to: {}", output.display());
    }

    if !comparison.summary.passes {
        std::process::exit(1);
    }

    Ok(())
}

async fn index_repos(repos: String, force: bool, cache_dir: Option<PathBuf>) -> Result<()> {
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
        info!("Preparing {} ({})", repo, config.release_tag);

        if force {
            info!("Removing existing cache for {}", repo);
            cache.remove(repo)?;
        }

        match prepare_repo(repo, Some(cache_dir.clone())).await {
            Ok(path) => {
                info!("Repository ready at: {}", path.display());
            }
            Err(e) => {
                warn!("Failed to prepare {}: {}", repo, e);
            }
        }
    }

    Ok(())
}

fn list_scenarios(scenarios_dir: Option<PathBuf>, detailed: bool) -> Result<()> {
    let scenarios_dir = scenarios_dir.unwrap_or_else(|| PathBuf::from("crates/benchmark/scenarios"));

    let scenarios = load_scenarios_from_dir(&scenarios_dir)?;

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
            println!(
                "    Expected files: {}",
                scenario.expected.must_find_files.len()
            );
            println!(
                "    Expected symbols: {}",
                scenario.expected.must_find_symbols.len()
            );
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

fn clean_cache(all: bool, repo: Option<String>, cache_dir: Option<PathBuf>) -> Result<()> {
    let cache_dir = cache_dir.unwrap_or_else(default_cache_dir);
    let cache = RepoCache::new(cache_dir);

    if all {
        info!("Cleaning all cached data");
        cache.clean_all()?;
        println!("Cache cleaned");
    } else if let Some(repo_name) = repo {
        if let Some(target) = TargetRepo::from_name(&repo_name) {
            info!("Cleaning cache for {}", target);
            cache.remove(target)?;
            println!("Cleaned {}", target);
        } else {
            anyhow::bail!("Unknown repository: {}", repo_name);
        }
    } else {
        println!("Specify --all or --repo <name>");
    }

    Ok(())
}
