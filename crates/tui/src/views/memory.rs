use crate::theme::Theme;
use crate::widgets::SalienceBar;
use ratatui::{
  buffer::Buffer,
  layout::{Constraint, Direction, Layout, Rect},
  style::{Style, Stylize},
  widgets::{Block, Borders, ListState, Widget},
};
use serde_json::Value;

/// Sort order for memories
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum MemorySortBy {
  #[default]
  Salience,
  Date,
  Sector,
}

impl MemorySortBy {
  pub fn next(self) -> Self {
    match self {
      Self::Salience => Self::Date,
      Self::Date => Self::Sector,
      Self::Sector => Self::Salience,
    }
  }

  pub fn label(self) -> &'static str {
    match self {
      Self::Salience => "Salience",
      Self::Date => "Date",
      Self::Sector => "Sector",
    }
  }
}

/// Memory browser view state
#[derive(Debug, Default)]
pub struct MemoryState {
  pub memories: Vec<Value>,
  pub selected: usize,
  pub list_state: ListState,
  pub search_query: String,
  pub filter_sector: Option<String>,
  pub sort_by: MemorySortBy,
  pub detail_scroll: u16,
  pub loading: bool,
  pub error: Option<String>,
}

impl MemoryState {
  pub fn new() -> Self {
    let mut state = Self::default();
    state.list_state.select(Some(0));
    state
  }

  pub fn set_memories(&mut self, memories: Vec<Value>) {
    self.memories = memories;
    if self.selected >= self.memories.len() && !self.memories.is_empty() {
      self.selected = self.memories.len() - 1;
    }
    self.list_state.select(Some(self.selected));
  }

  pub fn selected_memory(&self) -> Option<&Value> {
    self.memories.get(self.selected)
  }

  pub fn select_next(&mut self) {
    if self.memories.is_empty() {
      return;
    }
    self.selected = (self.selected + 1).min(self.memories.len() - 1);
    self.list_state.select(Some(self.selected));
    self.detail_scroll = 0;
  }

  pub fn select_prev(&mut self) {
    if self.memories.is_empty() {
      return;
    }
    self.selected = self.selected.saturating_sub(1);
    self.list_state.select(Some(self.selected));
    self.detail_scroll = 0;
  }

  pub fn scroll_detail_down(&mut self) {
    self.detail_scroll = self.detail_scroll.saturating_add(1);
  }

  pub fn scroll_detail_up(&mut self) {
    self.detail_scroll = self.detail_scroll.saturating_sub(1);
  }

  pub fn selected_id(&self) -> Option<String> {
    self
      .selected_memory()
      .and_then(|m| m.get("id"))
      .and_then(|id| id.as_str())
      .map(|s| s.to_string())
  }

  /// Cycle to next sort order and re-sort
  pub fn cycle_sort(&mut self) {
    self.sort_by = self.sort_by.next();
    self.apply_sort();
  }

  /// Apply current sort order to memories
  pub fn apply_sort(&mut self) {
    match self.sort_by {
      MemorySortBy::Salience => {
        self.memories.sort_by(|a, b| {
          let sa = a.get("salience").and_then(|s| s.as_f64()).unwrap_or(0.0);
          let sb = b.get("salience").and_then(|s| s.as_f64()).unwrap_or(0.0);
          sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
      }
      MemorySortBy::Date => {
        self.memories.sort_by(|a, b| {
          let da = a.get("created_at").and_then(|d| d.as_str()).unwrap_or("");
          let db = b.get("created_at").and_then(|d| d.as_str()).unwrap_or("");
          db.cmp(da) // Newest first
        });
      }
      MemorySortBy::Sector => {
        self.memories.sort_by(|a, b| {
          let sa = a.get("sector").and_then(|s| s.as_str()).unwrap_or("");
          let sb = b.get("sector").and_then(|s| s.as_str()).unwrap_or("");
          sa.cmp(sb)
        });
      }
    }
    // Keep selection valid
    if self.selected >= self.memories.len() && !self.memories.is_empty() {
      self.selected = self.memories.len() - 1;
    }
    self.list_state.select(Some(self.selected));
  }
}

/// Memory browser view widget
pub struct MemoryView<'a> {
  state: &'a MemoryState,
  focused: bool,
}

impl<'a> MemoryView<'a> {
  pub fn new(state: &'a MemoryState) -> Self {
    Self { state, focused: true }
  }

  pub fn focused(mut self, focused: bool) -> Self {
    self.focused = focused;
    self
  }
}

impl Widget for MemoryView<'_> {
  fn render(self, area: Rect, buf: &mut Buffer) {
    // Split into list and detail panels
    let chunks = Layout::default()
      .direction(Direction::Horizontal)
      .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
      .split(area);

    // Render list panel
    self.render_list(chunks[0], buf);

    // Render detail panel
    self.render_detail(chunks[1], buf);
  }
}

impl MemoryView<'_> {
  fn render_list(&self, area: Rect, buf: &mut Buffer) {
    let sort_label = self.state.sort_by.label();
    let title = if !self.state.search_query.is_empty() {
      format!(
        "MEMORIES ({}) - Search: {} [{}]",
        self.state.memories.len(),
        self.state.search_query,
        sort_label
      )
    } else if let Some(ref sector) = self.state.filter_sector {
      format!(
        "MEMORIES ({}) - Filter: {} [{}]",
        self.state.memories.len(),
        sector,
        sort_label
      )
    } else {
      format!("MEMORIES ({}) [{}]", self.state.memories.len(), sort_label)
    };

    let border_color = if self.focused { Theme::ACCENT } else { Theme::OVERLAY };

    let block = Block::default()
      .title(title)
      .title_style(Style::default().fg(Theme::ACCENT).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    block.render(area, buf);

    if self.state.memories.is_empty() {
      let msg = if self.state.loading {
        "Loading..."
      } else if let Some(ref err) = self.state.error {
        err
      } else {
        "No memories found"
      };
      buf.set_string(inner.x, inner.y, msg, Style::default().fg(Theme::MUTED));
      return;
    }

    // Render memory list items
    let visible_height = inner.height as usize;
    let start = if self.state.selected >= visible_height {
      self.state.selected - visible_height + 1
    } else {
      0
    };

    for (i, memory) in self.state.memories.iter().enumerate().skip(start).take(visible_height) {
      let y = inner.y + (i - start) as u16;
      if y >= inner.y + inner.height {
        break;
      }

      let is_selected = i == self.state.selected;
      self.render_memory_item(memory, inner.x, y, inner.width, is_selected, buf);
    }
  }

  fn render_memory_item(&self, memory: &Value, x: u16, y: u16, width: u16, selected: bool, buf: &mut Buffer) {
    let sector = memory.get("sector").and_then(|s| s.as_str()).unwrap_or("unknown");
    let salience = memory.get("salience").and_then(|s| s.as_f64()).unwrap_or(0.0) as f32;
    let content = memory.get("content").and_then(|c| c.as_str()).unwrap_or("");

    let bg = if selected { Theme::SURFACE } else { Theme::BG };
    let fg = if selected { Theme::TEXT } else { Theme::SUBTEXT };

    // Clear line with background
    for i in 0..width {
      buf[(x + i, y)].set_bg(bg);
    }

    // Selection indicator
    let indicator = if selected { "▶ " } else { "  " };
    buf.set_string(x, y, indicator, Style::default().fg(Theme::ACCENT));

    // Sector badge
    let sector_badge = format!("[{}] ", &sector[..3.min(sector.len())].to_uppercase());
    let sector_color = Theme::sector_color(sector);
    buf.set_string(x + 2, y, &sector_badge, Style::default().fg(sector_color).bold());

    // Content preview
    let content_start = x + 2 + sector_badge.len() as u16;
    let content_width = width.saturating_sub(content_start - x + 12) as usize; // Leave room for salience
    let preview = content.lines().next().unwrap_or("").trim();
    let preview = if preview.len() > content_width {
      format!("{}...", &preview[..content_width.saturating_sub(3)])
    } else {
      preview.to_string()
    };
    buf.set_string(content_start, y, &preview, Style::default().fg(fg));

    // Salience bar at end
    let salience_x = x + width.saturating_sub(10);
    let salience_color = Theme::salience_color(salience);
    let filled = (salience * 5.0).round() as usize;
    let empty = 5 - filled;
    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
    buf.set_string(salience_x, y, &bar, Style::default().fg(salience_color));
  }

  fn render_detail(&self, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
      .title("DETAIL")
      .title_style(Style::default().fg(Theme::ACCENT).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::OVERLAY));

    let inner = block.inner(area);
    block.render(area, buf);

    let Some(memory) = self.state.selected_memory() else {
      buf.set_string(
        inner.x,
        inner.y,
        "Select a memory to view details",
        Style::default().fg(Theme::MUTED),
      );
      return;
    };

    let mut y = inner.y;

    // ID
    if let Some(id) = memory.get("id").and_then(|i| i.as_str()) {
      let short_id = if id.len() > 8 { &id[..8] } else { id };
      buf.set_string(inner.x, y, "ID: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(inner.x + 4, y, short_id, Style::default().fg(Theme::TEXT));
      buf.set_string(
        inner.x + 4 + short_id.len() as u16,
        y,
        "...",
        Style::default().fg(Theme::MUTED),
      );
      y += 1;
    }

    // Sector
    if let Some(sector) = memory.get("sector").and_then(|s| s.as_str()) {
      buf.set_string(inner.x, y, "Sector: ", Style::default().fg(Theme::SUBTEXT));
      let sector_color = Theme::sector_color(sector);
      buf.set_string(
        inner.x + 8,
        y,
        capitalize(sector),
        Style::default().fg(sector_color).bold(),
      );
      y += 1;
    }

    // Type
    if let Some(mem_type) = memory.get("memory_type").and_then(|t| t.as_str()) {
      buf.set_string(inner.x, y, "Type: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(inner.x + 6, y, capitalize(mem_type), Style::default().fg(Theme::TEXT));
      y += 1;
    }

    // Salience
    if let Some(salience) = memory.get("salience").and_then(|s| s.as_f64()) {
      buf.set_string(inner.x, y, "Salience: ", Style::default().fg(Theme::SUBTEXT));
      let bar_area = Rect::new(inner.x + 10, y, inner.width.saturating_sub(10).min(15), 1);
      SalienceBar::new(salience as f32).width(10).render(bar_area, buf);
      y += 1;
    }

    // Importance
    if let Some(importance) = memory.get("importance").and_then(|i| i.as_f64()) {
      buf.set_string(inner.x, y, "Importance: ", Style::default().fg(Theme::SUBTEXT));
      let bar_area = Rect::new(inner.x + 12, y, inner.width.saturating_sub(12).min(15), 1);
      SalienceBar::new(importance as f32).width(10).render(bar_area, buf);
      y += 1;
    }

    // Timestamps
    if let Some(created) = memory.get("created_at").and_then(|c| c.as_str()) {
      buf.set_string(inner.x, y, "Created: ", Style::default().fg(Theme::SUBTEXT));
      let date = parse_date_friendly(created).unwrap_or_else(|| created.to_string());
      buf.set_string(inner.x + 9, y, &date, Style::default().fg(Theme::TEXT));
      y += 1;
    }

    if let Some(accessed) = memory.get("last_accessed").and_then(|a| a.as_str()) {
      buf.set_string(inner.x, y, "Accessed: ", Style::default().fg(Theme::SUBTEXT));
      let date = parse_date_friendly(accessed).unwrap_or_else(|| accessed.to_string());
      buf.set_string(inner.x + 10, y, &date, Style::default().fg(Theme::TEXT));
      y += 1;
    }

    y += 1; // Separator

    // Content header
    buf.set_string(inner.x, y, "CONTENT", Style::default().fg(Theme::ACCENT).bold());
    y += 1;

    // Content
    if let Some(content) = memory.get("content").and_then(|c| c.as_str()) {
      let lines: Vec<&str> = content.lines().collect();
      let scroll = self.state.detail_scroll as usize;

      for line in lines.iter().skip(scroll) {
        if y >= inner.y + inner.height {
          break;
        }
        let display_line = if line.len() > inner.width as usize {
          format!("{}...", &line[..inner.width as usize - 3])
        } else {
          line.to_string()
        };
        buf.set_string(inner.x, y, &display_line, Style::default().fg(Theme::TEXT));
        y += 1;
      }
    }

    // Relationships
    if let Some(relationships) = memory.get("relationships").and_then(|r| r.as_array())
      && !relationships.is_empty()
      && y + 2 < inner.y + inner.height
    {
      y += 1;
      buf.set_string(inner.x, y, "RELATIONSHIPS", Style::default().fg(Theme::ACCENT).bold());
      y += 1;

      for rel in relationships.iter().take(3) {
        if y >= inner.y + inner.height {
          break;
        }
        let rel_type = rel
          .get("relationship_type")
          .and_then(|t| t.as_str())
          .unwrap_or("related");
        let target = rel.get("to_memory_id").and_then(|t| t.as_str()).unwrap_or("?");
        let target_short = if target.len() > 8 { &target[..8] } else { target };
        let line = format!("└─{}: {}...", rel_type, target_short);
        buf.set_string(inner.x, y, &line, Style::default().fg(Theme::SUBTEXT));
        y += 1;
      }
    }
  }
}

fn capitalize(s: &str) -> String {
  let s = s.replace('_', " ");
  let mut chars = s.chars();
  match chars.next() {
    None => String::new(),
    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
  }
}

fn parse_date_friendly(s: &str) -> Option<String> {
  // Parse ISO 8601 and return friendly format
  // Example: "2024-01-15T10:30:00Z" -> "Jan 15, 2024 10:30"
  let parts: Vec<&str> = s.split('T').collect();
  if parts.len() >= 2 {
    let date_parts: Vec<&str> = parts[0].split('-').collect();
    let time_parts: Vec<&str> = parts[1].split(':').collect();
    if date_parts.len() >= 3 && time_parts.len() >= 2 {
      let month = match date_parts[1] {
        "01" => "Jan",
        "02" => "Feb",
        "03" => "Mar",
        "04" => "Apr",
        "05" => "May",
        "06" => "Jun",
        "07" => "Jul",
        "08" => "Aug",
        "09" => "Sep",
        "10" => "Oct",
        "11" => "Nov",
        "12" => "Dec",
        _ => return None,
      };
      let day = date_parts[2].trim_start_matches('0');
      return Some(format!(
        "{} {}, {} {}:{}",
        month, day, date_parts[0], time_parts[0], time_parts[1]
      ));
    }
  }
  None
}
