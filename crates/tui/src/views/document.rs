use crate::theme::Theme;
use ratatui::{
  buffer::Buffer,
  layout::{Constraint, Direction, Layout, Rect},
  style::{Style, Stylize},
  widgets::{Block, Borders, Widget},
};
use serde_json::Value;

/// Document browser view state
#[derive(Debug, Default)]
pub struct DocumentState {
  pub documents: Vec<Value>,
  pub selected: usize,
  pub search_query: String,
  pub detail_scroll: u16,
  pub loading: bool,
  pub error: Option<String>,
}

impl DocumentState {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn set_documents(&mut self, documents: Vec<Value>) {
    self.documents = documents;
    if self.selected >= self.documents.len() && !self.documents.is_empty() {
      self.selected = self.documents.len() - 1;
    }
  }

  pub fn selected_document(&self) -> Option<&Value> {
    self.documents.get(self.selected)
  }

  pub fn select_next(&mut self) {
    if self.documents.is_empty() {
      return;
    }
    self.selected = (self.selected + 1).min(self.documents.len() - 1);
    self.detail_scroll = 0;
  }

  pub fn select_prev(&mut self) {
    if self.documents.is_empty() {
      return;
    }
    self.selected = self.selected.saturating_sub(1);
    self.detail_scroll = 0;
  }

  pub fn scroll_detail_down(&mut self) {
    self.detail_scroll = self.detail_scroll.saturating_add(1);
  }

  pub fn scroll_detail_up(&mut self) {
    self.detail_scroll = self.detail_scroll.saturating_sub(1);
  }
}

/// Document browser view widget
pub struct DocumentView<'a> {
  state: &'a DocumentState,
  focused: bool,
}

impl<'a> DocumentView<'a> {
  pub fn new(state: &'a DocumentState) -> Self {
    Self { state, focused: true }
  }

  pub fn focused(mut self, focused: bool) -> Self {
    self.focused = focused;
    self
  }
}

impl Widget for DocumentView<'_> {
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

impl DocumentView<'_> {
  fn render_list(&self, area: Rect, buf: &mut Buffer) {
    let title = if !self.state.search_query.is_empty() {
      format!(
        "DOCUMENTS ({}) - Search: {}",
        self.state.documents.len(),
        self.state.search_query
      )
    } else {
      format!("DOCUMENTS ({})", self.state.documents.len())
    };

    let border_color = if self.focused { Theme::ACCENT } else { Theme::OVERLAY };

    let block = Block::default()
      .title(title)
      .title_style(Style::default().fg(Theme::REFLECTIVE).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    block.render(area, buf);

    if self.state.documents.is_empty() {
      let msg = if self.state.loading {
        "Loading..."
      } else if let Some(ref err) = self.state.error {
        err
      } else {
        "No documents found"
      };
      buf.set_string(inner.x, inner.y, msg, Style::default().fg(Theme::MUTED));
      return;
    }

    // Render document list items
    let visible_height = inner.height as usize;
    let start = if self.state.selected >= visible_height {
      self.state.selected - visible_height + 1
    } else {
      0
    };

    for (i, doc) in self.state.documents.iter().enumerate().skip(start).take(visible_height) {
      let y = inner.y + (i - start) as u16;
      if y >= inner.y + inner.height {
        break;
      }

      let is_selected = i == self.state.selected;
      self.render_document_item(doc, inner.x, y, inner.width, is_selected, buf);
    }
  }

  fn render_document_item(&self, doc: &Value, x: u16, y: u16, width: u16, selected: bool, buf: &mut Buffer) {
    let title = doc.get("title").and_then(|t| t.as_str()).unwrap_or("Untitled");
    let source_type = doc.get("source_type").and_then(|s| s.as_str()).unwrap_or("file");
    let chunk_count = doc.get("chunk_count").and_then(|c| c.as_u64()).unwrap_or(0);

    let bg = if selected { Theme::SURFACE } else { Theme::BG };
    let fg = if selected { Theme::TEXT } else { Theme::SUBTEXT };

    // Clear line with background
    for i in 0..width {
      buf[(x + i, y)].set_bg(bg);
    }

    // Selection indicator
    let indicator = if selected { "▶ " } else { "  " };
    buf.set_string(x, y, indicator, Style::default().fg(Theme::ACCENT));

    // Source type icon
    let icon = match source_type {
      "url" => "🌐 ",
      "file" => "📄 ",
      _ => "📝 ",
    };
    buf.set_string(x + 2, y, icon, Style::default());

    // Title
    let title_start = x + 5;
    let title_width = width.saturating_sub(title_start - x + 8) as usize;
    let display_title = if title.len() > title_width {
      format!("{}...", &title[..title_width.saturating_sub(3)])
    } else {
      title.to_string()
    };
    buf.set_string(title_start, y, &display_title, Style::default().fg(fg));

    // Chunk count
    let count = format!(" ({} chunks)", chunk_count);
    let count_x = x + width.saturating_sub(count.len() as u16 + 1);
    buf.set_string(count_x, y, &count, Style::default().fg(Theme::MUTED));
  }

  fn render_detail(&self, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
      .title("DOCUMENT DETAIL")
      .title_style(Style::default().fg(Theme::ACCENT).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::OVERLAY));

    let inner = block.inner(area);
    block.render(area, buf);

    let Some(doc) = self.state.selected_document() else {
      buf.set_string(
        inner.x,
        inner.y,
        "Select a document to view details",
        Style::default().fg(Theme::MUTED),
      );
      return;
    };

    let mut y = inner.y;

    // Title
    if let Some(title) = doc.get("title").and_then(|t| t.as_str()) {
      buf.set_string(inner.x, y, "Title: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(inner.x + 7, y, title, Style::default().fg(Theme::TEXT).bold());
      y += 1;
    }

    // ID
    if let Some(id) = doc.get("id").and_then(|i| i.as_str()) {
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

    // Source
    if let Some(source) = doc.get("source").and_then(|s| s.as_str()) {
      buf.set_string(inner.x, y, "Source: ", Style::default().fg(Theme::SUBTEXT));
      let max_len = inner.width as usize - 8;
      let display_source = if source.len() > max_len {
        format!("{}...", &source[..max_len - 3])
      } else {
        source.to_string()
      };
      buf.set_string(inner.x + 8, y, &display_source, Style::default().fg(Theme::INFO));
      y += 1;
    }

    // Source type
    if let Some(source_type) = doc.get("source_type").and_then(|s| s.as_str()) {
      buf.set_string(inner.x, y, "Type: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(
        inner.x + 6,
        y,
        capitalize(source_type),
        Style::default().fg(Theme::TEXT),
      );
      y += 1;
    }

    // Stats
    if let Some(char_count) = doc.get("char_count").and_then(|c| c.as_u64()) {
      buf.set_string(inner.x, y, "Characters: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(
        inner.x + 12,
        y,
        format_number(char_count),
        Style::default().fg(Theme::TEXT),
      );
      y += 1;
    }

    if let Some(chunk_count) = doc.get("chunk_count").and_then(|c| c.as_u64()) {
      buf.set_string(inner.x, y, "Chunks: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(
        inner.x + 8,
        y,
        chunk_count.to_string(),
        Style::default().fg(Theme::TEXT),
      );
      y += 1;
    }

    // Timestamps
    if let Some(created) = doc.get("created_at").and_then(|c| c.as_str()) {
      buf.set_string(inner.x, y, "Created: ", Style::default().fg(Theme::SUBTEXT));
      let date = parse_date_friendly(created).unwrap_or_else(|| created.to_string());
      buf.set_string(inner.x + 9, y, &date, Style::default().fg(Theme::TEXT));
      y += 1;
    }

    y += 1;

    // Content preview
    if let Some(content) = doc.get("full_content").and_then(|c| c.as_str()) {
      buf.set_string(inner.x, y, "CONTENT PREVIEW", Style::default().fg(Theme::ACCENT).bold());
      y += 1;

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

fn parse_date_friendly(s: &str) -> Option<String> {
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
