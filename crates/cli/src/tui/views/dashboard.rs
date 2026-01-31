use std::time::Duration;

use ccengram::ipc::{
  code::CodeStatsResult,
  project::ProjectStatsResult,
  system::{HealthCheckResult, MetricsResult},
  watch::WatchStatusResult,
};
use ratatui::{
  buffer::Buffer,
  layout::{Constraint, Direction, Layout, Rect},
  style::Style,
  widgets::{Block, Borders, Widget},
};

use crate::tui::{theme::Theme, widgets::SalienceBar};

/// Dashboard view state
#[derive(Debug, Default)]
pub struct DashboardState {
  pub stats: Option<ProjectStatsResult>,
  pub health: Option<HealthCheckResult>,
  pub loading: bool,
  pub error: Option<String>,

  // Watcher status
  pub watcher_running: bool,
  pub watcher_scanning: bool,
  pub watcher_pending_changes: usize,
  pub watcher_scan_progress: Option<(usize, usize)>, // (processed, total)

  // Index quality (from code_stats)
  pub index_health_score: u32,
  pub index_total_lines: u64,
  pub index_total_files: usize,
  pub language_breakdown: Vec<(String, usize)>,

  // Daemon metrics
  pub daemon_uptime_seconds: u64,
  pub daemon_requests_per_second: f64,
  pub daemon_memory_kb: Option<u64>,
  pub daemon_active_sessions: usize,
}

impl DashboardState {
  pub fn new() -> Self {
    Self::default()
  }

  /// Update stats from daemon response
  pub fn set_stats(&mut self, stats: ProjectStatsResult) {
    self.stats = Some(stats);
  }

  /// Update health from daemon response
  pub fn set_health(&mut self, health: HealthCheckResult) {
    self.health = Some(health);
  }

  /// Get memory count
  pub fn memory_count(&self) -> u64 {
    self.stats.as_ref().map(|s| s.memories as u64).unwrap_or(0)
  }

  /// Get memories by sector (top 3, sorted by count)
  pub fn memories_by_sector(&self) -> Vec<(String, u64)> {
    self
      .stats
      .as_ref()
      .and_then(|s| s.memories_by_sector.as_ref())
      .map(|sectors| {
        let mut sorted: Vec<(String, u64)> = sectors.iter().map(|(k, v)| (k.clone(), *v as u64)).collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.into_iter().take(3).collect()
      })
      .unwrap_or_default()
  }

  /// Get average salience
  pub fn average_salience(&self) -> f32 {
    self.stats.as_ref().and_then(|s| s.average_salience).unwrap_or(0.5)
  }

  /// Get code stats
  pub fn code_files(&self) -> u64 {
    self.index_total_files as u64
  }

  pub fn code_chunks(&self) -> u64 {
    self.stats.as_ref().map(|s| s.code_chunks as u64).unwrap_or(0)
  }

  /// Get top languages (top 3)
  pub fn top_languages(&self) -> Vec<(String, u64)> {
    self
      .language_breakdown
      .iter()
      .take(3)
      .map(|(lang, count)| (lang.clone(), *count as u64))
      .collect()
  }

  /// Check if daemon is healthy
  pub fn is_daemon_healthy(&self) -> bool {
    self.health.is_some()
  }

  /// Check if embedding is available
  pub fn is_embedding_available(&self) -> bool {
    // HealthCheckResult has a checks field with individual health checks
    // Look for an embedding check with status "ok"
    self
      .health
      .as_ref()
      .map(|h| {
        h.checks
          .iter()
          .any(|c| c.name.to_lowercase().contains("embedding") && c.status == "ok")
      })
      .unwrap_or(false)
  }

  /// Update watch status from daemon response
  pub fn set_watch_status(&mut self, status: WatchStatusResult) {
    self.watcher_running = status.running;
    self.watcher_scanning = status.scanning;
    self.watcher_pending_changes = status.pending_changes;

    // Parse scan progress if available
    if let Some(progress) = status.scan_progress {
      let processed = progress[0];
      let total = progress[1];
      if total > 0 {
        self.watcher_scan_progress = Some((processed, total));
      } else {
        self.watcher_scan_progress = None;
      }
    } else {
      self.watcher_scan_progress = None;
    }
  }

  /// Update code stats (extracts health score, total lines, files, and language breakdown)
  pub fn set_code_stats(&mut self, stats: CodeStatsResult) {
    self.index_health_score = stats.index_health_score;
    self.index_total_lines = stats.total_lines;
    self.index_total_files = stats.total_files;

    // Convert language breakdown from HashMap to sorted Vec
    let mut langs: Vec<(String, usize)> = stats.language_breakdown.into_iter().collect();
    langs.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by count descending
    self.language_breakdown = langs;
  }

  /// Update daemon metrics from daemon response
  pub fn set_daemon_metrics(&mut self, metrics: MetricsResult) {
    self.daemon_uptime_seconds = metrics.daemon.uptime_seconds;
    self.daemon_requests_per_second = metrics.requests.per_second;
    self.daemon_memory_kb = metrics.memory.rss_kb;
    self.daemon_active_sessions = metrics.sessions.active;
  }

  /// Get suggested refresh interval based on current state
  pub fn suggested_refresh_interval(&self) -> Duration {
    if self.watcher_scanning {
      Duration::from_secs(2)
    } else if self.watcher_pending_changes > 0 {
      Duration::from_secs(5)
    } else {
      Duration::from_secs(30)
    }
  }
}

/// Dashboard view widget
pub struct DashboardView<'a> {
  state: &'a DashboardState,
}

impl<'a> DashboardView<'a> {
  pub fn new(state: &'a DashboardState) -> Self {
    Self { state }
  }
}

impl Widget for DashboardView<'_> {
  fn render(self, area: Rect, buf: &mut Buffer) {
    if area.width < 20 || area.height < 10 {
      let msg = "Terminal too small";
      buf.set_string(area.x, area.y, msg, Style::default().fg(Theme::ERROR));
      return;
    }

    // Main layout: two stats rows
    let chunks = Layout::default()
      .direction(Direction::Vertical)
      .constraints([
        Constraint::Length(7), // Row 1: existing cards
        Constraint::Length(7), // Row 2: new cards
        Constraint::Min(1),    // Spacer
      ])
      .split(area);

    // Row 1: Memories, Code Index, Health
    let row1_chunks = Layout::default()
      .direction(Direction::Horizontal)
      .constraints([
        Constraint::Percentage(33),
        Constraint::Percentage(33),
        Constraint::Percentage(34),
      ])
      .split(chunks[0]);

    self.render_memory_card(row1_chunks[0], buf);
    self.render_code_card(row1_chunks[1], buf);
    self.render_health_card(row1_chunks[2], buf);

    // Row 2: File Watcher, Index Quality, Daemon
    let row2_chunks = Layout::default()
      .direction(Direction::Horizontal)
      .constraints([
        Constraint::Percentage(33),
        Constraint::Percentage(33),
        Constraint::Percentage(34),
      ])
      .split(chunks[1]);

    self.render_watcher_card(row2_chunks[0], buf);
    self.render_index_quality_card(row2_chunks[1], buf);
    self.render_daemon_card(row2_chunks[2], buf);
  }
}

impl DashboardView<'_> {
  fn render_memory_card(&self, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
      .title("MEMORIES")
      .title_style(Style::default().fg(Theme::SEMANTIC).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::OVERLAY));

    let inner = block.inner(area);
    block.render(area, buf);

    let mut y = inner.y;

    // Total count
    let total = self.state.memory_count();
    let line = format!("Total: {}", total);
    buf.set_string(inner.x, y, &line, Style::default().fg(Theme::TEXT));
    y += 1;

    // By sector (top 2)
    let sectors = self.state.memories_by_sector();
    for (sector, count) in sectors.iter().take(2) {
      if y >= inner.y + inner.height {
        break;
      }
      let color = Theme::sector_color(sector);
      let line = format!("{}: {}", capitalize(sector), count);
      buf.set_string(inner.x, y, &line, Style::default().fg(color));
      y += 1;
    }

    // Salience bar
    if y + 1 < inner.y + inner.height {
      y += 1;
      buf.set_string(inner.x, y, "Salience:", Style::default().fg(Theme::SUBTEXT));
      y += 1;
      if y < inner.y + inner.height {
        let salience = self.state.average_salience();
        let bar_area = Rect::new(inner.x, y, inner.width.min(15), 1);
        SalienceBar::new(salience).width(10).render(bar_area, buf);
      }
    }
  }

  fn render_code_card(&self, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
      .title("CODE INDEX")
      .title_style(Style::default().fg(Theme::PROCEDURAL).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::OVERLAY));

    let inner = block.inner(area);
    block.render(area, buf);

    let mut y = inner.y;

    // File count
    let files = self.state.code_files();
    let line = format!("Files: {}", format_number(files));
    buf.set_string(inner.x, y, &line, Style::default().fg(Theme::TEXT));
    y += 1;

    // Chunk count
    let chunks = self.state.code_chunks();
    let line = format!("Chunks: {}", format_number(chunks));
    buf.set_string(inner.x, y, &line, Style::default().fg(Theme::TEXT));
    y += 1;

    // Top languages
    let langs = self.state.top_languages();
    for (lang, count) in langs.iter().take(2) {
      if y >= inner.y + inner.height {
        break;
      }
      let color = Theme::language_color(lang);
      let line = format!("{}: {}", capitalize(lang), count);
      buf.set_string(inner.x, y, &line, Style::default().fg(color));
      y += 1;
    }
  }

  fn render_health_card(&self, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
      .title("HEALTH")
      .title_style(Style::default().fg(Theme::INFO).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::OVERLAY));

    let inner = block.inner(area);
    block.render(area, buf);

    let mut y = inner.y;

    // Daemon status
    let daemon_ok = self.state.is_daemon_healthy();
    let (indicator, text, color) = if daemon_ok {
      ("●", "Running", Theme::SUCCESS)
    } else {
      ("○", "Stopped", Theme::ERROR)
    };
    buf.set_string(inner.x, y, "Daemon: ", Style::default().fg(Theme::TEXT));
    buf.set_string(
      inner.x + 8,
      y,
      format!("{} {}", indicator, text),
      Style::default().fg(color),
    );
    y += 1;

    // Embedding status
    let embed_ok = self.state.is_embedding_available();
    let (indicator, text, color) = if embed_ok {
      ("●", "OK", Theme::SUCCESS)
    } else {
      ("○", "N/A", Theme::WARNING)
    };
    buf.set_string(inner.x, y, "Embedding: ", Style::default().fg(Theme::TEXT));
    buf.set_string(
      inner.x + 11,
      y,
      format!("{} {}", indicator, text),
      Style::default().fg(color),
    );
    y += 1;

    // Loading indicator
    if self.state.loading && y < inner.y + inner.height {
      buf.set_string(inner.x, y, "Loading...", Style::default().fg(Theme::MUTED));
    }

    // Error message
    if let Some(ref error) = self.state.error
      && y < inner.y + inner.height
    {
      let err_msg = if error.len() > inner.width as usize - 2 {
        format!("{}...", &error[..inner.width as usize - 5])
      } else {
        error.clone()
      };
      buf.set_string(inner.x, y, &err_msg, Style::default().fg(Theme::ERROR));
    }
  }

  fn render_watcher_card(&self, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
      .title("FILE WATCHER")
      .title_style(Style::default().fg(Theme::EPISODIC).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::OVERLAY));

    let inner = block.inner(area);
    block.render(area, buf);

    let mut y = inner.y;

    // Status indicator
    let (indicator, status_text, color) = if self.state.watcher_scanning {
      ("◐", "Scanning", Theme::WARNING)
    } else if self.state.watcher_running {
      ("●", "Running", Theme::SUCCESS)
    } else {
      ("○", "Stopped", Theme::MUTED)
    };

    buf.set_string(inner.x, y, "Status: ", Style::default().fg(Theme::TEXT));
    buf.set_string(
      inner.x + 8,
      y,
      format!("{} {}", indicator, status_text),
      Style::default().fg(color),
    );
    y += 1;

    // Scan progress bar (when scanning)
    if let Some((processed, total)) = self.state.watcher_scan_progress
      && total > 0
      && y < inner.y + inner.height
    {
      let pct = (processed as f32 / total as f32 * 100.0).min(100.0);
      let progress_text = format!("Progress: {:.0}%", pct);
      buf.set_string(inner.x, y, &progress_text, Style::default().fg(Theme::TEXT));
      y += 1;

      // Simple progress bar
      if y < inner.y + inner.height {
        let bar_width = inner.width.min(15) as usize;
        let filled = (pct / 100.0 * bar_width as f32) as usize;
        let bar: String = "█".repeat(filled) + &"░".repeat(bar_width.saturating_sub(filled));
        buf.set_string(inner.x, y, &bar, Style::default().fg(Theme::ACCENT));
        y += 1;
      }
    }

    // Pending changes
    if y < inner.y + inner.height {
      let pending = self.state.watcher_pending_changes;
      let pending_color = if pending > 0 { Theme::WARNING } else { Theme::TEXT };
      let line = format!("Pending: {}", pending);
      buf.set_string(inner.x, y, &line, Style::default().fg(pending_color));
    }
  }

  fn render_index_quality_card(&self, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
      .title("INDEX QUALITY")
      .title_style(Style::default().fg(Theme::REFLECTIVE).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::OVERLAY));

    let inner = block.inner(area);
    block.render(area, buf);

    let mut y = inner.y;

    // Health score with color coding
    let score = self.state.index_health_score;
    let score_color = if score >= 80 {
      Theme::SUCCESS
    } else if score >= 50 {
      Theme::WARNING
    } else {
      Theme::ERROR
    };

    buf.set_string(inner.x, y, "Health: ", Style::default().fg(Theme::TEXT));
    buf.set_string(inner.x + 8, y, format!("{}%", score), Style::default().fg(score_color));
    y += 1;

    // Health bar
    if y < inner.y + inner.height {
      let bar_width = inner.width.min(15) as usize;
      let filled = (score as f32 / 100.0 * bar_width as f32) as usize;
      let bar: String = "█".repeat(filled) + &"░".repeat(bar_width.saturating_sub(filled));
      buf.set_string(inner.x, y, &bar, Style::default().fg(score_color));
      y += 1;
    }

    // Total lines
    if y < inner.y + inner.height {
      let lines = self.state.index_total_lines;
      let line = format!("Lines: {}", format_number(lines));
      buf.set_string(inner.x, y, &line, Style::default().fg(Theme::TEXT));
    }
  }

  fn render_daemon_card(&self, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
      .title("DAEMON")
      .title_style(Style::default().fg(Theme::INFO).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::OVERLAY));

    let inner = block.inner(area);
    block.render(area, buf);

    let mut y = inner.y;

    // Uptime
    let uptime = format_duration(self.state.daemon_uptime_seconds);
    buf.set_string(inner.x, y, "Uptime: ", Style::default().fg(Theme::TEXT));
    buf.set_string(inner.x + 8, y, &uptime, Style::default().fg(Theme::SUCCESS));
    y += 1;

    // Requests per second
    if y < inner.y + inner.height {
      let rps = format!("{:.1}/s", self.state.daemon_requests_per_second);
      buf.set_string(inner.x, y, "Req/s: ", Style::default().fg(Theme::TEXT));
      buf.set_string(inner.x + 7, y, &rps, Style::default().fg(Theme::TEXT));
      y += 1;
    }

    // Active sessions
    if y < inner.y + inner.height {
      let sessions = self.state.daemon_active_sessions;
      buf.set_string(inner.x, y, "Sessions: ", Style::default().fg(Theme::TEXT));
      buf.set_string(inner.x + 10, y, sessions.to_string(), Style::default().fg(Theme::TEXT));
      y += 1;
    }

    // Memory usage (if available)
    if let Some(mem_kb) = self.state.daemon_memory_kb
      && y < inner.y + inner.height
    {
      let mem_str = if mem_kb >= 1024 {
        format!("{:.1} MB", mem_kb as f64 / 1024.0)
      } else {
        format!("{} KB", mem_kb)
      };
      buf.set_string(inner.x, y, "Memory: ", Style::default().fg(Theme::TEXT));
      buf.set_string(inner.x + 8, y, &mem_str, Style::default().fg(Theme::TEXT));
    }
  }
}

fn capitalize(s: &str) -> String {
  let mut chars = s.chars();
  match chars.next() {
    None => String::new(),
    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
  }
}

fn format_number(n: u64) -> String {
  if n >= 1_000_000 {
    format!("{:.1}M", n as f64 / 1_000_000.0)
  } else if n >= 1_000 {
    format!("{:.1}K", n as f64 / 1_000.0)
  } else {
    n.to_string()
  }
}

fn format_duration(seconds: u64) -> String {
  let hours = seconds / 3600;
  let minutes = (seconds % 3600) / 60;

  if hours > 0 {
    format!("{}h {}m", hours, minutes)
  } else if minutes > 0 {
    format!("{}m", minutes)
  } else {
    format!("{}s", seconds)
  }
}
