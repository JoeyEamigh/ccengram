# CCEngram Benchmark Harness

Comprehensive benchmarking for testing the exploration capabilities of CCEngram's `explore` and `context` tools against large real-world codebases.

## Overview

The benchmark harness tests how well CCEngram helps agents discover important code without prior context. Unlike search benchmarks that test finding known items, this tests **exploration** - the ability to navigate unfamiliar codebases and find architecturally significant code.

## Quick Start

```bash
# Build the benchmark tool
cargo build -p benchmark

# Download and cache test repositories
cargo run -p benchmark -- index --repos zed,vscode

# List available scenarios
cargo run -p benchmark -- list --detailed

# Run all benchmarks
cargo run -p benchmark -- run --output ./results

# Run specific scenarios (glob patterns supported)
cargo run -p benchmark -- run --scenarios "zed*" --output ./results

# Run in parallel (faster, less detailed progress)
cargo run -p benchmark -- run --parallel --output ./results

# Compare two runs for regressions
cargo run -p benchmark -- compare baseline.json current.json --threshold 10
```

## CLI Commands

### `run` - Execute Benchmark Scenarios

```bash
ccengram-bench run [OPTIONS]

Options:
  -o, --output <DIR>         Output directory for results [default: ./benchmark-results]
  -s, --scenarios <PATTERN>  Filter scenarios by glob pattern
      --parallel             Run scenarios concurrently
      --llm-judge            Enable LLM-as-judge evaluation (not yet implemented)
      --scenarios-dir <DIR>  Custom scenarios directory
      --name <NAME>          Name for this benchmark run
```

### `compare` - Regression Detection

```bash
ccengram-bench compare <BASELINE> <CURRENT> [OPTIONS]

Arguments:
  <BASELINE>  Baseline results JSON file
  <CURRENT>   Current results JSON file

Options:
  -t, --threshold <PCT>  Regression threshold percentage [default: 10]
  -o, --output <FILE>    Save comparison report
```

### `index` - Prepare Repositories

```bash
ccengram-bench index [OPTIONS]

Options:
  -r, --repos <LIST>      Repositories to prepare: zed, vscode, or 'all' [default: all]
      --force             Force re-download even if cached
      --cache-dir <DIR>   Custom cache directory
```

### `list` - Show Available Scenarios

```bash
ccengram-bench list [OPTIONS]

Options:
  -d, --detailed          Show full scenario details
      --scenarios-dir     Custom scenarios directory
```

### `clean` - Remove Cached Data

```bash
ccengram-bench clean [OPTIONS]

Options:
  --all           Clean all cached data
  --repo <NAME>   Clean specific repository cache
```

## Target Repositories

| Repository | Language | Size | Use Case |
|------------|----------|------|----------|
| **Zed** | Rust | ~1M LOC | Editor architecture, commands, LSP integration |
| **VSCode** | TypeScript | ~1M LOC | Large codebase stress test, extension system |

Both are editor codebases with complex architectural discovery scenarios.

## Scenario Definition Format

Scenarios are defined in TOML files in `crates/benchmark/scenarios/`:

```toml
[scenario]
id = "zed-command-system"
name = "Understanding Zed Command Architecture"
repo = "zed"
difficulty = "hard"  # easy, medium, hard

[task]
prompt = "How does Zed handle editor commands?"
intent = "architectural_discovery"  # or: symbol_lookup, flow_tracing, bug_investigation

[expected]
must_find_files = [
    "**/commands.rs",
    "**/actions.rs",
    "**/keymap/**",
]
must_find_symbols = ["Action", "actions", "dispatch", "Keymap", "KeyBinding"]
noise_patterns = ["**/tests/**", "test_*", "Mock*"]

[[steps]]
query = "How does Zed handle editor commands and actions?"
expected_results = 5
max_noise_ratio = 0.3
scope = "code"  # code, memory, docs, all

[[steps]]
query = "What is the Action type and how is it dispatched?"
depends_on_previous = true

[success]
min_discovery_score = 0.7   # File recall target
max_noise_ratio = 0.25      # Maximum acceptable noise
max_steps_to_core = 3       # Steps to find first core result
```

## Metrics

### Performance Metrics

| Metric | Description |
|--------|-------------|
| **Search Latency** | p50/p95/p99 latency for explore queries |
| **Context Latency** | p50/p95/p99 latency for context fetches |
| **Total Time** | End-to-end scenario execution time |
| **Peak Memory** | Maximum memory usage during execution |
| **Avg CPU** | Average CPU utilization |

### Accuracy Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **File Recall** | % of must-find files discovered | >= 70% |
| **Symbol Recall** | % of must-find symbols discovered | >= 70% |
| **Steps to Core** | Queries needed to find first core result | <= 3 |
| **MRR** | Mean reciprocal rank of first correct result | >= 0.5 |
| **Noise Ratio** | % of results matching noise patterns | <= 25% |
| **Top-3 Noise** | Noise in top 3 results | <= 10% |
| **Hint Utility** | % of callers/callees that are relevant | >= 60% |
| **Suggestion Quality** | % of suggestions leading to useful results | >= 50% |

## Ground Truth

The benchmark uses a hybrid approach for validation:

### 1. Noise Pattern Detection (Automatic)

Default patterns that identify test/mock code:

**File Patterns:**
- `**/tests/**`, `**/test/**`, `**/__tests__/**`
- `*_test.rs`, `*_test.go`, `*.test.ts`
- `**/fixtures/**`, `**/mocks/**`

**Symbol Patterns:**
- `test_*`, `Test*`, `Mock*`, `Stub*`, `Fake*`
- `_*` (internal/private symbols)

**Content Patterns:**
- `#[test]`, `#[cfg(test)]`
- `describe(`, `it(`, `expect(`

### 2. Manual Annotations (Optional)

JSON files in `crates/benchmark/annotations/<repo>/`:

```json
{
  "scenario_id": "zed-command-system",
  "critical_files": ["crates/gpui/src/action.rs"],
  "critical_symbols": ["Action", "ActionRegistry"],
  "key_locations": ["crates/gpui/src/action.rs:42"],
  "exploration_paths": [
    {
      "start": "Action",
      "through": ["ActionRegistry"],
      "target": "dispatch",
      "max_hops": 3
    }
  ],
  "notes": ["The Action trait is the core abstraction"]
}
```

### 3. Call Graph Analysis

Petgraph-based analysis for:
- Verifying reachability between symbols
- Scoring navigation hints (callers/callees)
- Measuring path lengths

## Reports

### JSON Report

Machine-readable format for CI integration:

```json
{
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "0.1.0",
    "git_commit": "abc123",
    "hostname": "benchmark-runner",
    "total_scenarios": 4
  },
  "summary": {
    "passed": 3,
    "failed": 1,
    "pass_rate": 0.75,
    "performance": {
      "avg_search_latency_p50_ms": 45,
      "avg_search_latency_p95_ms": 120
    },
    "accuracy": {
      "avg_file_recall": 0.82,
      "avg_symbol_recall": 0.78,
      "avg_noise_ratio": 0.18
    }
  },
  "scenarios": [...]
}
```

### Markdown Report

Human-readable summary with pass/fail indicators:

```markdown
# Benchmark Results

**Run:** 2024-01-15 10:30:00
**Pass Rate:** 75% (3/4)

## Summary

| Scenario | Status | File Recall | Noise | Latency p50 |
|----------|--------|-------------|-------|-------------|
| zed-command-system | PASS | 85% | 15% | 42ms |
| zed-lsp-integration | PASS | 78% | 22% | 51ms |
| vscode-extensions | FAIL | 45% | 35% | 38ms |
| vscode-editor-core | PASS | 72% | 18% | 55ms |
```

### Comparison Report

Regression detection between runs:

```markdown
# Comparison: baseline vs current

## Regressions (threshold: 10%)

| Scenario | Metric | Baseline | Current | Change |
|----------|--------|----------|---------|--------|
| vscode-extensions | file_recall | 0.65 | 0.45 | -30.8% |

## Improvements

| Scenario | Metric | Baseline | Current | Change |
|----------|--------|----------|---------|--------|
| zed-commands | latency_p50 | 65ms | 42ms | -35.4% |
```

## Architecture

```
crates/benchmark/
├── Cargo.toml
├── src/
│   ├── lib.rs                # Public API
│   ├── main.rs               # CLI (ccengram-bench)
│   ├── session.rs            # Multi-step exploration state
│   ├── repos/
│   │   ├── mod.rs            # Repository management
│   │   ├── registry.rs       # Zed/VSCode configs
│   │   └── clone.rs          # Tarball download & caching
│   ├── scenarios/
│   │   ├── mod.rs            # Scenario loader
│   │   ├── definition.rs     # TOML schema types
│   │   └── runner.rs         # Daemon communication
│   ├── metrics/
│   │   ├── mod.rs            # Metric types
│   │   ├── performance.rs    # Latency, memory, CPU
│   │   └── accuracy.rs       # Recall, noise, MRR
│   ├── ground_truth/
│   │   ├── mod.rs            # Ground truth API
│   │   ├── call_graph.rs     # Petgraph analysis
│   │   ├── patterns.rs       # Noise detection
│   │   └── annotations.rs    # Manual annotations
│   └── reports/
│       ├── mod.rs            # Report generation
│       ├── json.rs           # Machine-readable
│       ├── markdown.rs       # Human-readable
│       └── comparison.rs     # Regression detection
├── scenarios/                # Built-in scenarios
│   ├── zed_commands.toml
│   ├── zed_lsp.toml
│   ├── vscode_extensions.toml
│   └── vscode_editor.toml
└── annotations/              # Optional ground truth
    ├── zed/
    └── vscode/
```

## CI Integration

Example GitHub Actions workflow:

```yaml
name: Benchmark

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-action@stable

      - name: Cache repositories
        uses: actions/cache@v4
        with:
          path: ~/.cache/ccengram-bench
          key: bench-repos-${{ hashFiles('crates/benchmark/src/repos/registry.rs') }}

      - name: Download repos
        run: cargo run -p benchmark -- index --repos all

      - name: Start daemon
        run: |
          cargo run --release -- daemon &
          sleep 5

      - name: Run benchmarks
        run: cargo run -p benchmark -- run --output ./results

      - name: Check for regressions
        if: github.event_name == 'pull_request'
        run: |
          # Download baseline from main branch
          cargo run -p benchmark -- compare baseline.json results/benchmark.json --threshold 10

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: results/
```

## Writing New Scenarios

1. **Identify the exploration goal**: What should an agent discover?

2. **Define expected outcomes**: Which files/symbols are critical?

3. **Create multi-step queries**: How would an agent naturally explore?

4. **Set realistic thresholds**: Based on difficulty level

Example scenario creation:

```toml
# scenarios/my_new_scenario.toml

[scenario]
id = "my-new-scenario"
name = "Exploring Feature X"
repo = "zed"
difficulty = "medium"

[task]
prompt = "How does feature X work?"
intent = "architectural_discovery"

[expected]
must_find_files = ["**/feature_x.rs", "**/feature_x/**"]
must_find_symbols = ["FeatureX", "init_feature_x"]
noise_patterns = ["**/tests/**"]

[[steps]]
query = "Where is feature X implemented?"
expected_results = 3

[[steps]]
query = "How is FeatureX initialized?"
depends_on_previous = true

[success]
min_discovery_score = 0.6
max_noise_ratio = 0.3
max_steps_to_core = 2
```

## Troubleshooting

### "Daemon not running" error

```bash
# Start the daemon first
ccengram daemon

# Then run benchmarks
cargo run -p benchmark -- run
```

### Repository download fails

```bash
# Check network connectivity
curl -I https://github.com/zed-industries/zed/archive/refs/tags/v0.220.3.tar.gz

# Force re-download
cargo run -p benchmark -- index --repos zed --force
```

### Low recall scores

1. Check if expected files use correct glob patterns
2. Verify the daemon has indexed the repository
3. Review noise patterns - may be too aggressive

### High noise ratio

1. Add more specific noise patterns to scenario
2. Check if test files are being returned as top results
3. Consider adjusting ranking weights in the daemon
