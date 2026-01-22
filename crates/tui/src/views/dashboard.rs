use crate::theme::Theme;
use crate::widgets::SalienceBar;
use ratatui::{
  buffer::Buffer,
  layout::{Constraint, Direction, Layout, Rect},
  style::{Style, Stylize},
  widgets::{Block, Borders, Widget},
};
use serde_json::Value;

/// Dashboard view state
#[derive(Debug, Default)]
pub struct DashboardState {
  pub stats: Option<Value>,
  pub health: Option<Value>,
  pub recent_activity: Vec<ActivityItem>,
  pub loading: bool,
  pub error: Option<String>,
}

/// A recent activity item
#[derive(Debug, Clone)]
pub struct ActivityItem {
  pub time_ago: String,
  pub description: String,
  pub item_type: ActivityType,
}

#[derive(Debug, Clone, Copy)]
pub enum ActivityType {
  Memory,
  Code,
  Session,
  Document,
}

impl DashboardState {
  pub fn new() -> Self {
    Self::default()
  }

  /// Update stats from daemon response
  pub fn set_stats(&mut self, stats: Value) {
    self.stats = Some(stats);
  }

  /// Update health from daemon response
  pub fn set_health(&mut self, health: Value) {
    self.health = Some(health);
  }

  /// Get memory count
  pub fn memory_count(&self) -> u64 {
    self
      .stats
      .as_ref()
      .and_then(|s| s.get("memories"))
      .and_then(|m| m.get("total"))
      .and_then(|t| t.as_u64())
      .unwrap_or(0)
  }

  /// Get memories by sector
  pub fn memories_by_sector(&self) -> Vec<(String, u64)> {
    self
      .stats
      .as_ref()
      .and_then(|s| s.get("memories"))
      .and_then(|m| m.get("by_sector"))
      .and_then(|bs| bs.as_object())
      .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0))).collect())
      .unwrap_or_default()
  }

  /// Get average salience
  pub fn average_salience(&self) -> f32 {
    self
      .stats
      .as_ref()
      .and_then(|s| s.get("memories"))
      .and_then(|m| m.get("average_salience"))
      .and_then(|a| a.as_f64())
      .map(|v| v as f32)
      .unwrap_or(0.0)
  }

  /// Get code stats
  pub fn code_files(&self) -> u64 {
    self
      .stats
      .as_ref()
      .and_then(|s| s.get("code"))
      .and_then(|c| c.get("total_files"))
      .and_then(|f| f.as_u64())
      .unwrap_or(0)
  }

  pub fn code_chunks(&self) -> u64 {
    self
      .stats
      .as_ref()
      .and_then(|s| s.get("code"))
      .and_then(|c| c.get("total_chunks"))
      .and_then(|f| f.as_u64())
      .unwrap_or(0)
  }

  /// Get top languages
  pub fn top_languages(&self) -> Vec<(String, u64)> {
    self
      .stats
      .as_ref()
      .and_then(|s| s.get("code"))
      .and_then(|c| c.get("by_language"))
      .and_then(|bl| bl.as_object())
      .map(|obj| {
        let mut langs: Vec<_> = obj.iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0))).collect();
        langs.sort_by(|a, b| b.1.cmp(&a.1));
        langs.truncate(3);
        langs
      })
      .unwrap_or_default()
  }

  /// Check if daemon is healthy
  pub fn is_daemon_healthy(&self) -> bool {
    self.health.is_some()
  }

  /// Check if embedding is available
  pub fn is_embedding_available(&self) -> bool {
    self
      .health
      .as_ref()
      .and_then(|h| h.get("embedding"))
      .and_then(|e| e.get("available"))
      .and_then(|a| a.as_bool())
      .unwrap_or(false)
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

    // Main layout: stats row + activity section
    let chunks = Layout::default()
      .direction(Direction::Vertical)
      .constraints([Constraint::Length(8), Constraint::Min(5)])
      .split(area);

    // Stats row: three stat cards
    let stat_chunks = Layout::default()
      .direction(Direction::Horizontal)
      .constraints([
        Constraint::Percentage(33),
        Constraint::Percentage(33),
        Constraint::Percentage(34),
      ])
      .split(chunks[0]);

    // Memory stats card
    self.render_memory_card(stat_chunks[0], buf);

    // Code stats card
    self.render_code_card(stat_chunks[1], buf);

    // Health card
    self.render_health_card(stat_chunks[2], buf);

    // Recent activity section
    self.render_activity(chunks[1], buf);
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

  fn render_activity(&self, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
      .title("RECENT ACTIVITY")
      .title_style(Style::default().fg(Theme::ACCENT).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::OVERLAY));

    let inner = block.inner(area);
    block.render(area, buf);

    if self.state.recent_activity.is_empty() {
      let msg = "No recent activity";
      buf.set_string(inner.x, inner.y, msg, Style::default().fg(Theme::MUTED));
      return;
    }

    for (i, activity) in self.state.recent_activity.iter().enumerate() {
      let y = inner.y + i as u16;
      if y >= inner.y + inner.height {
        break;
      }

      let type_color = match activity.item_type {
        ActivityType::Memory => Theme::SEMANTIC,
        ActivityType::Code => Theme::PROCEDURAL,
        ActivityType::Session => Theme::EPISODIC,
        ActivityType::Document => Theme::REFLECTIVE,
      };

      // Time ago
      let time_width = 8;
      let time_str = if activity.time_ago.len() > time_width {
        &activity.time_ago[..time_width]
      } else {
        &activity.time_ago
      };
      buf.set_string(inner.x, y, time_str, Style::default().fg(Theme::MUTED));

      // Description
      let desc_x = inner.x + time_width as u16 + 1;
      let max_desc_len = inner.width.saturating_sub(time_width as u16 + 1) as usize;
      let desc = if activity.description.len() > max_desc_len {
        format!("{}...", &activity.description[..max_desc_len.saturating_sub(3)])
      } else {
        activity.description.clone()
      };
      buf.set_string(desc_x, y, &desc, Style::default().fg(type_color));
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
