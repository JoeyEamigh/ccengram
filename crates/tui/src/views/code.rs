use crate::theme::Theme;
use ratatui::{
  buffer::Buffer,
  layout::{Constraint, Direction, Layout, Rect},
  style::{Style, Stylize},
  widgets::{Block, Borders, Widget},
};
use serde_json::Value;
use std::collections::HashMap;

/// Code index view state
#[derive(Debug, Default)]
pub struct CodeState {
  pub chunks: Vec<Value>,
  pub stats: Option<Value>,
  pub selected: usize,
  pub search_query: String,
  pub filter_language: Option<String>,
  pub tree_expanded: HashMap<String, bool>,
  pub loading: bool,
  pub error: Option<String>,
}

impl CodeState {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn set_chunks(&mut self, chunks: Vec<Value>) {
    self.chunks = chunks;
    if self.selected >= self.chunks.len() && !self.chunks.is_empty() {
      self.selected = self.chunks.len() - 1;
    }
  }

  pub fn set_stats(&mut self, stats: Value) {
    self.stats = Some(stats);
  }

  pub fn selected_chunk(&self) -> Option<&Value> {
    self.chunks.get(self.selected)
  }

  pub fn select_next(&mut self) {
    if self.chunks.is_empty() {
      return;
    }
    self.selected = (self.selected + 1).min(self.chunks.len() - 1);
  }

  pub fn select_prev(&mut self) {
    if self.chunks.is_empty() {
      return;
    }
    self.selected = self.selected.saturating_sub(1);
  }

  /// Group chunks by file path
  pub fn files(&self) -> Vec<(&str, Vec<&Value>)> {
    let mut file_map: HashMap<&str, Vec<&Value>> = HashMap::new();
    for chunk in &self.chunks {
      if let Some(path) = chunk.get("file_path").and_then(|p| p.as_str()) {
        file_map.entry(path).or_default().push(chunk);
      }
    }
    let mut files: Vec<_> = file_map.into_iter().collect();
    files.sort_by(|a, b| a.0.cmp(b.0));
    files
  }

  /// Get language breakdown from stats
  pub fn language_breakdown(&self) -> Vec<(String, u64)> {
    self
      .stats
      .as_ref()
      .and_then(|s| s.get("language_breakdown"))
      .and_then(|lb| lb.as_object())
      .map(|obj| {
        let mut langs: Vec<_> = obj.iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0))).collect();
        langs.sort_by(|a, b| b.1.cmp(&a.1));
        langs
      })
      .unwrap_or_default()
  }
}

/// Code index view widget
pub struct CodeView<'a> {
  state: &'a CodeState,
  focused: bool,
}

impl<'a> CodeView<'a> {
  pub fn new(state: &'a CodeState) -> Self {
    Self { state, focused: true }
  }

  pub fn focused(mut self, focused: bool) -> Self {
    self.focused = focused;
    self
  }
}

impl Widget for CodeView<'_> {
  fn render(self, area: Rect, buf: &mut Buffer) {
    // Split into file tree, code preview, and stats
    let main_chunks = Layout::default()
      .direction(Direction::Vertical)
      .constraints([Constraint::Min(10), Constraint::Length(6)])
      .split(area);

    let content_chunks = Layout::default()
      .direction(Direction::Horizontal)
      .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
      .split(main_chunks[0]);

    // File tree
    self.render_file_tree(content_chunks[0], buf);

    // Code preview
    self.render_code_preview(content_chunks[1], buf);

    // Stats bar
    self.render_stats(main_chunks[1], buf);
  }
}

impl CodeView<'_> {
  fn render_file_tree(&self, area: Rect, buf: &mut Buffer) {
    let border_color = if self.focused { Theme::ACCENT } else { Theme::OVERLAY };

    let block = Block::default()
      .title(format!("FILES ({})", self.state.chunks.len()))
      .title_style(Style::default().fg(Theme::PROCEDURAL).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    block.render(area, buf);

    if self.state.chunks.is_empty() {
      let msg = if self.state.loading {
        "Loading..."
      } else if let Some(ref err) = self.state.error {
        err
      } else {
        "No code indexed"
      };
      buf.set_string(inner.x, inner.y, msg, Style::default().fg(Theme::MUTED));
      return;
    }

    let files = self.state.files();
    let mut y = inner.y;

    for (file_path, chunks) in files.iter() {
      if y >= inner.y + inner.height {
        break;
      }

      // Determine language from first chunk
      let language = chunks
        .first()
        .and_then(|c| c.get("language"))
        .and_then(|l| l.as_str())
        .unwrap_or("unknown");

      let lang_color = Theme::language_color(language);

      // Check if any chunk in this file is selected
      let file_selected = chunks.iter().any(|c| {
        self
          .state
          .chunks
          .get(self.state.selected)
          .map(|sel| std::ptr::eq(*c, sel))
          .unwrap_or(false)
      });

      let bg = if file_selected { Theme::SURFACE } else { Theme::BG };

      // Clear line
      for i in 0..inner.width {
        buf[(inner.x + i, y)].set_bg(bg);
      }

      // File icon/indicator
      let icon = if file_selected { "▶ " } else { "  " };
      buf.set_string(inner.x, y, icon, Style::default().fg(Theme::ACCENT));

      // Shortened file path
      let display_path = shorten_path(file_path, inner.width as usize - 10);
      buf.set_string(inner.x + 2, y, &display_path, Style::default().fg(lang_color));

      // Chunk count
      let count = format!(" ({})", chunks.len());
      let count_x = inner.x + inner.width.saturating_sub(count.len() as u16 + 1);
      buf.set_string(count_x, y, &count, Style::default().fg(Theme::MUTED));

      y += 1;

      // If expanded, show chunks
      if file_selected {
        for chunk in chunks.iter().take(3) {
          if y >= inner.y + inner.height {
            break;
          }

          let chunk_type = chunk.get("chunk_type").and_then(|t| t.as_str()).unwrap_or("block");
          let start_line = chunk.get("start_line").and_then(|l| l.as_u64()).unwrap_or(0);
          let end_line = chunk.get("end_line").and_then(|l| l.as_u64()).unwrap_or(0);

          let symbols = chunk
            .get("symbols")
            .and_then(|s| s.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(", "))
            .unwrap_or_default();

          let chunk_info = format!(
            "  └ {}:{}-{} {}",
            chunk_type,
            start_line,
            end_line,
            if symbols.len() > 20 {
              format!("{}...", &symbols[..20])
            } else {
              symbols
            }
          );

          let max_len = inner.width as usize;
          let chunk_info = if chunk_info.len() > max_len {
            format!("{}...", &chunk_info[..max_len - 3])
          } else {
            chunk_info
          };

          buf.set_string(inner.x, y, &chunk_info, Style::default().fg(Theme::SUBTEXT));
          y += 1;
        }

        if chunks.len() > 3 && y < inner.y + inner.height {
          let more = format!("  ... {} more", chunks.len() - 3);
          buf.set_string(inner.x, y, &more, Style::default().fg(Theme::MUTED));
          y += 1;
        }
      }
    }
  }

  fn render_code_preview(&self, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
      .title("CODE PREVIEW")
      .title_style(Style::default().fg(Theme::ACCENT).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::OVERLAY));

    let inner = block.inner(area);
    block.render(area, buf);

    let Some(chunk) = self.state.selected_chunk() else {
      buf.set_string(
        inner.x,
        inner.y,
        "Select a file to preview",
        Style::default().fg(Theme::MUTED),
      );
      return;
    };

    let mut y = inner.y;

    // Header info
    if let Some(file_path) = chunk.get("file_path").and_then(|p| p.as_str()) {
      buf.set_string(inner.x, y, "File: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(inner.x + 6, y, file_path, Style::default().fg(Theme::TEXT));
      y += 1;
    }

    if let (Some(start), Some(end)) = (
      chunk.get("start_line").and_then(|l| l.as_u64()),
      chunk.get("end_line").and_then(|l| l.as_u64()),
    ) {
      buf.set_string(inner.x, y, "Lines: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(
        inner.x + 7,
        y,
        format!("{}-{}", start, end),
        Style::default().fg(Theme::TEXT),
      );
      y += 1;
    }

    if let Some(language) = chunk.get("language").and_then(|l| l.as_str()) {
      buf.set_string(inner.x, y, "Language: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(
        inner.x + 10,
        y,
        capitalize(language),
        Style::default().fg(Theme::language_color(language)),
      );
      y += 1;
    }

    if let Some(symbols) = chunk.get("symbols").and_then(|s| s.as_array())
      && !symbols.is_empty()
    {
      buf.set_string(inner.x, y, "Symbols: ", Style::default().fg(Theme::SUBTEXT));
      let symbols_str = symbols.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(", ");
      let max_len = inner.width as usize - 9;
      let symbols_str = if symbols_str.len() > max_len {
        format!("{}...", &symbols_str[..max_len - 3])
      } else {
        symbols_str
      };
      buf.set_string(inner.x + 9, y, &symbols_str, Style::default().fg(Theme::INFO));
      y += 1;
    }

    y += 1; // Separator

    // Code content
    if let Some(content) = chunk.get("content").and_then(|c| c.as_str()) {
      buf.set_string(inner.x, y, "Content:", Style::default().fg(Theme::ACCENT).bold());
      y += 1;

      for line in content.lines() {
        if y >= inner.y + inner.height {
          break;
        }

        let display_line = if line.len() > inner.width as usize {
          format!("{}...", &line[..inner.width as usize - 3])
        } else {
          line.to_string()
        };

        // Simple syntax highlighting based on language
        let style = Style::default().fg(Theme::TEXT);
        buf.set_string(inner.x, y, &display_line, style);
        y += 1;
      }
    }
  }

  fn render_stats(&self, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
      .title("LANGUAGE BREAKDOWN")
      .title_style(Style::default().fg(Theme::REFLECTIVE).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::OVERLAY));

    let inner = block.inner(area);
    block.render(area, buf);

    let langs = self.state.language_breakdown();
    if langs.is_empty() {
      buf.set_string(
        inner.x,
        inner.y,
        "No statistics available",
        Style::default().fg(Theme::MUTED),
      );
      return;
    }

    let total: u64 = langs.iter().map(|(_, c)| c).sum();
    if total == 0 {
      return;
    }

    // Render horizontal bar chart
    let bar_width = inner.width.saturating_sub(2);
    let y = inner.y;

    let mut x = inner.x;
    for (lang, count) in langs.iter().take(6) {
      let pct = *count as f32 / total as f32;
      let segment_width = ((pct * bar_width as f32).round() as u16).max(1);

      if x + segment_width > inner.x + bar_width {
        break;
      }

      let color = Theme::language_color(lang);
      let bar_str: String = "█".repeat(segment_width as usize);
      buf.set_string(x, y, &bar_str, Style::default().fg(color));

      x += segment_width;
    }

    // Legend below
    let mut legend_x = inner.x;
    let legend_y = inner.y + 2;
    if legend_y < inner.y + inner.height {
      for (lang, count) in langs.iter().take(6) {
        let color = Theme::language_color(lang);
        let label = format!("● {} ({}) ", lang, count);
        if legend_x + label.len() as u16 > inner.x + inner.width {
          break;
        }
        buf.set_string(legend_x, legend_y, &label, Style::default().fg(color));
        legend_x += label.len() as u16;
      }
    }
  }
}

fn shorten_path(path: &str, max_len: usize) -> String {
  if path.len() <= max_len {
    return path.to_string();
  }

  let parts: Vec<&str> = path.split('/').collect();
  if parts.len() <= 2 {
    return format!("...{}", &path[path.len().saturating_sub(max_len - 3)..]);
  }

  // Try to keep first and last parts
  let last = parts.last().unwrap_or(&"");
  let first = parts.first().unwrap_or(&"");

  if first.len() + last.len() + 5 <= max_len {
    format!("{}/.../{}", first, last)
  } else if last.len() + 4 <= max_len {
    format!(".../{}", last)
  } else {
    format!("...{}", &last[last.len().saturating_sub(max_len - 3)..])
  }
}

fn capitalize(s: &str) -> String {
  let mut chars = s.chars();
  match chars.next() {
    None => String::new(),
    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
  }
}
