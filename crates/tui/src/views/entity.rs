use crate::theme::Theme;
use ratatui::{
  buffer::Buffer,
  layout::{Constraint, Direction, Layout, Rect},
  style::Style,
  widgets::{Block, Borders, Widget},
};
use serde_json::Value;

/// Entity network view state
#[derive(Debug, Default)]
pub struct EntityState {
  pub entities: Vec<Value>,
  pub selected: usize,
  pub linked_memories: Vec<Value>,
  pub search_query: String,
  pub filter_type: Option<String>,
  pub loading: bool,
  pub error: Option<String>,
}

impl EntityState {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn set_entities(&mut self, entities: Vec<Value>) {
    self.entities = entities;
    if self.selected >= self.entities.len() && !self.entities.is_empty() {
      self.selected = self.entities.len() - 1;
    }
  }

  pub fn set_linked_memories(&mut self, memories: Vec<Value>) {
    self.linked_memories = memories;
  }

  pub fn selected_entity(&self) -> Option<&Value> {
    self.entities.get(self.selected)
  }

  pub fn select_next(&mut self) {
    if self.entities.is_empty() {
      return;
    }
    self.selected = (self.selected + 1).min(self.entities.len() - 1);
  }

  pub fn select_prev(&mut self) {
    if self.entities.is_empty() {
      return;
    }
    self.selected = self.selected.saturating_sub(1);
  }

  pub fn selected_id(&self) -> Option<String> {
    self
      .selected_entity()
      .and_then(|e| e.get("id"))
      .and_then(|id| id.as_str())
      .map(|s| s.to_string())
  }

  /// Get max mention count for bar scaling
  pub fn max_mention_count(&self) -> u64 {
    self
      .entities
      .iter()
      .filter_map(|e| e.get("mention_count").and_then(|c| c.as_u64()))
      .max()
      .unwrap_or(1)
  }
}

/// Entity network view widget
pub struct EntityView<'a> {
  state: &'a EntityState,
  focused: bool,
}

impl<'a> EntityView<'a> {
  pub fn new(state: &'a EntityState) -> Self {
    Self { state, focused: true }
  }

  pub fn focused(mut self, focused: bool) -> Self {
    self.focused = focused;
    self
  }
}

impl Widget for EntityView<'_> {
  fn render(self, area: Rect, buf: &mut Buffer) {
    // Split into entity list and detail panels
    let chunks = Layout::default()
      .direction(Direction::Horizontal)
      .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
      .split(area);

    // Render entity list
    self.render_list(chunks[0], buf);

    // Render entity detail with linked memories
    self.render_detail(chunks[1], buf);
  }
}

impl EntityView<'_> {
  fn render_list(&self, area: Rect, buf: &mut Buffer) {
    let title = if !self.state.search_query.is_empty() {
      format!(
        "ENTITIES ({}) - Search: {}",
        self.state.entities.len(),
        self.state.search_query
      )
    } else if let Some(ref etype) = self.state.filter_type {
      format!("ENTITIES ({}) - Type: {}", self.state.entities.len(), etype)
    } else {
      format!("ENTITIES ({})", self.state.entities.len())
    };

    let border_color = if self.focused { Theme::ACCENT } else { Theme::OVERLAY };

    let block = Block::default()
      .title(title)
      .title_style(Style::default().fg(Theme::INFO).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    block.render(area, buf);

    if self.state.entities.is_empty() {
      let msg = if self.state.loading {
        "Loading..."
      } else if let Some(ref err) = self.state.error {
        err
      } else {
        "No entities found"
      };
      buf.set_string(inner.x, inner.y, msg, Style::default().fg(Theme::MUTED));
      return;
    }

    let max_count = self.state.max_mention_count();

    // Render entity list items
    let visible_height = inner.height as usize;
    let start = if self.state.selected >= visible_height {
      self.state.selected - visible_height + 1
    } else {
      0
    };

    for (i, entity) in self.state.entities.iter().enumerate().skip(start).take(visible_height) {
      let y = inner.y + (i - start) as u16;
      if y >= inner.y + inner.height {
        break;
      }

      let is_selected = i == self.state.selected;
      self.render_entity_item(entity, inner.x, y, inner.width, is_selected, max_count, buf);
    }
  }

  #[allow(clippy::too_many_arguments)]
  fn render_entity_item(
    &self,
    entity: &Value,
    x: u16,
    y: u16,
    width: u16,
    selected: bool,
    max_count: u64,
    buf: &mut Buffer,
  ) {
    let name = entity.get("name").and_then(|n| n.as_str()).unwrap_or("Unknown");
    let entity_type = entity.get("entity_type").and_then(|t| t.as_str()).unwrap_or("other");
    let mention_count = entity.get("mention_count").and_then(|c| c.as_u64()).unwrap_or(0);

    let bg = if selected { Theme::SURFACE } else { Theme::BG };
    let fg = if selected { Theme::TEXT } else { Theme::SUBTEXT };

    // Clear line with background
    for i in 0..width {
      buf[(x + i, y)].set_bg(bg);
    }

    // Selection indicator
    let indicator = if selected { "▶ " } else { "  " };
    buf.set_string(x, y, indicator, Style::default().fg(Theme::ACCENT));

    // Entity type icon
    let (icon, color) = match entity_type {
      "person" => ("👤", Theme::EMOTIONAL),
      "project" => ("📁", Theme::PROCEDURAL),
      "technology" => ("⚙️", Theme::INFO),
      "organization" => ("🏢", Theme::REFLECTIVE),
      "concept" => ("💡", Theme::SEMANTIC),
      "file" => ("📄", Theme::PROCEDURAL),
      "symbol" => ("🔣", Theme::INFO),
      _ => ("•", Theme::MUTED),
    };
    buf.set_string(x + 2, y, icon, Style::default());

    // Name
    let name_start = x + 5;
    let bar_width = 8;
    let name_width = width.saturating_sub(name_start - x + bar_width + 3) as usize;
    let display_name = if name.len() > name_width {
      format!("{}...", &name[..name_width.saturating_sub(3)])
    } else {
      name.to_string()
    };
    buf.set_string(name_start, y, &display_name, Style::default().fg(fg));

    // Mention count bar
    let bar_x = x + width.saturating_sub(bar_width + 2);
    let pct = if max_count > 0 {
      mention_count as f32 / max_count as f32
    } else {
      0.0
    };
    let filled = (pct * (bar_width - 2) as f32).round() as usize;
    let empty = (bar_width as usize - 2).saturating_sub(filled);

    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
    buf.set_string(bar_x, y, &bar, Style::default().fg(color));

    // Count
    let count_str = format!("{}", mention_count);
    buf.set_string(bar_x + bar_width, y, &count_str, Style::default().fg(Theme::MUTED));
  }

  fn render_detail(&self, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
      .title("ENTITY DETAIL")
      .title_style(Style::default().fg(Theme::ACCENT).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::OVERLAY));

    let inner = block.inner(area);
    block.render(area, buf);

    let Some(entity) = self.state.selected_entity() else {
      buf.set_string(
        inner.x,
        inner.y,
        "Select an entity to view details",
        Style::default().fg(Theme::MUTED),
      );
      return;
    };

    let mut y = inner.y;

    // Name
    if let Some(name) = entity.get("name").and_then(|n| n.as_str()) {
      buf.set_string(inner.x, y, "Name: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(inner.x + 6, y, name, Style::default().fg(Theme::TEXT).bold());
      y += 1;
    }

    // ID
    if let Some(id) = entity.get("id").and_then(|i| i.as_str()) {
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

    // Type
    if let Some(entity_type) = entity.get("entity_type").and_then(|t| t.as_str()) {
      buf.set_string(inner.x, y, "Type: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(
        inner.x + 6,
        y,
        capitalize(entity_type),
        Style::default().fg(Theme::INFO),
      );
      y += 1;
    }

    // Mention count
    if let Some(count) = entity.get("mention_count").and_then(|c| c.as_u64()) {
      buf.set_string(inner.x, y, "Mentions: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(inner.x + 10, y, count.to_string(), Style::default().fg(Theme::TEXT));
      y += 1;
    }

    // First/last seen
    if let Some(first_seen) = entity.get("first_seen_at").and_then(|f| f.as_str()) {
      buf.set_string(inner.x, y, "First seen: ", Style::default().fg(Theme::SUBTEXT));
      let date = parse_date_short(first_seen).unwrap_or_else(|| first_seen.to_string());
      buf.set_string(inner.x + 12, y, &date, Style::default().fg(Theme::TEXT));
      y += 1;
    }

    if let Some(last_seen) = entity.get("last_seen_at").and_then(|l| l.as_str()) {
      buf.set_string(inner.x, y, "Last seen: ", Style::default().fg(Theme::SUBTEXT));
      let date = parse_date_short(last_seen).unwrap_or_else(|| last_seen.to_string());
      buf.set_string(inner.x + 11, y, &date, Style::default().fg(Theme::TEXT));
      y += 1;
    }

    // Summary
    if let Some(summary) = entity.get("summary").and_then(|s| s.as_str()) {
      y += 1;
      buf.set_string(inner.x, y, "Summary: ", Style::default().fg(Theme::SUBTEXT));
      y += 1;
      let max_len = inner.width as usize;
      let summary_display = if summary.len() > max_len {
        format!("{}...", &summary[..max_len - 3])
      } else {
        summary.to_string()
      };
      buf.set_string(inner.x, y, &summary_display, Style::default().fg(Theme::TEXT));
      y += 1;
    }

    // Aliases
    if let Some(aliases) = entity.get("aliases").and_then(|a| a.as_array())
      && !aliases.is_empty()
    {
      y += 1;
      buf.set_string(inner.x, y, "Aliases: ", Style::default().fg(Theme::SUBTEXT));
      let aliases_str = aliases.iter().filter_map(|a| a.as_str()).collect::<Vec<_>>().join(", ");
      let max_len = inner.width as usize - 9;
      let aliases_display = if aliases_str.len() > max_len {
        format!("{}...", &aliases_str[..max_len - 3])
      } else {
        aliases_str
      };
      buf.set_string(inner.x + 9, y, &aliases_display, Style::default().fg(Theme::TEXT));
      y += 1;
    }

    // Linked memories
    y += 1;
    buf.set_string(inner.x, y, "LINKED MEMORIES", Style::default().fg(Theme::ACCENT).bold());
    y += 1;

    if self.state.linked_memories.is_empty() {
      buf.set_string(inner.x, y, "No linked memories", Style::default().fg(Theme::MUTED));
    } else {
      for memory in self.state.linked_memories.iter().take(5) {
        if y >= inner.y + inner.height {
          break;
        }

        let sector = memory.get("sector").and_then(|s| s.as_str()).unwrap_or("unknown");
        let content = memory.get("content").and_then(|c| c.as_str()).unwrap_or("");
        let preview = content.lines().next().unwrap_or("").trim();
        let max_len = inner.width as usize - 8;
        let preview = if preview.len() > max_len {
          format!("{}...", &preview[..max_len - 3])
        } else {
          preview.to_string()
        };

        let sector_short = &sector[..3.min(sector.len())].to_uppercase();
        let sector_color = Theme::sector_color(sector);

        buf.set_string(inner.x, y, "└─", Style::default().fg(Theme::MUTED));
        buf.set_string(
          inner.x + 2,
          y,
          format!("[{}] ", sector_short),
          Style::default().fg(sector_color),
        );
        buf.set_string(inner.x + 8, y, &preview, Style::default().fg(Theme::SUBTEXT));
        y += 1;
      }

      if self.state.linked_memories.len() > 5 && y < inner.y + inner.height {
        let more = format!("... {} more", self.state.linked_memories.len() - 5);
        buf.set_string(inner.x + 2, y, &more, Style::default().fg(Theme::MUTED));
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

fn parse_date_short(s: &str) -> Option<String> {
  let parts: Vec<&str> = s.split('T').collect();
  if let Some(date) = parts.first() {
    let date_parts: Vec<&str> = date.split('-').collect();
    if date_parts.len() >= 3 {
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
      return Some(format!("{} {}", month, day));
    }
  }
  None
}
