use crate::theme::Theme;
use ratatui::{
  buffer::Buffer,
  layout::{Constraint, Direction, Layout, Rect},
  style::Style,
  widgets::{Block, Borders, Widget},
};
use serde_json::Value;

/// Search result type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchResultType {
  Memory,
  Code,
  Document,
}

/// A unified search result
#[derive(Debug, Clone)]
pub struct SearchResult {
  pub result_type: SearchResultType,
  pub data: Value,
  pub similarity: f32,
}

/// Unified search view state
#[derive(Debug, Default)]
pub struct SearchState {
  pub query: String,
  pub results: Vec<SearchResult>,
  pub filtered_results: Vec<SearchResult>,
  pub selected: usize,
  pub search_memories: bool,
  pub search_code: bool,
  pub search_documents: bool,
  pub input_active: bool,
  pub filter_input_active: bool,
  pub loading: bool,
  pub error: Option<String>,
  pub filter_text: String,
  pub filter_active: bool,
}

impl SearchState {
  pub fn new() -> Self {
    Self {
      search_memories: true,
      search_code: true,
      search_documents: true,
      filter_text: String::new(),
      filter_active: false,
      filtered_results: Vec::new(),
      ..Default::default()
    }
  }

  pub fn set_results(&mut self, results: Vec<SearchResult>) {
    self.results = results;
    // Re-apply filter if active
    if self.filter_active {
      self.apply_filter();
    } else {
      self.filtered_results = self.results.clone();
    }
    // Bounds check selection
    let display_len = self.display_results().len();
    if self.selected >= display_len && display_len > 0 {
      self.selected = display_len - 1;
    }
  }

  pub fn selected_result(&self) -> Option<&SearchResult> {
    self.display_results().get(self.selected)
  }

  pub fn select_next(&mut self) {
    let display = self.display_results();
    if display.is_empty() {
      return;
    }
    self.selected = (self.selected + 1).min(display.len() - 1);
  }

  pub fn select_prev(&mut self) {
    let display = self.display_results();
    if display.is_empty() {
      return;
    }
    self.selected = self.selected.saturating_sub(1);
  }

  pub fn clear(&mut self) {
    self.query.clear();
    self.results.clear();
    self.filtered_results.clear();
    self.selected = 0;
    self.filter_text.clear();
    self.filter_active = false;
  }

  pub fn toggle_memories(&mut self) -> bool {
    // Prevent disabling if it's the only enabled scope
    if self.search_memories && !self.search_code && !self.search_documents {
      return false;
    }
    self.search_memories = !self.search_memories;
    true
  }

  pub fn toggle_code(&mut self) -> bool {
    // Prevent disabling if it's the only enabled scope
    if self.search_code && !self.search_memories && !self.search_documents {
      return false;
    }
    self.search_code = !self.search_code;
    true
  }

  pub fn toggle_documents(&mut self) -> bool {
    // Prevent disabling if it's the only enabled scope
    if self.search_documents && !self.search_memories && !self.search_code {
      return false;
    }
    self.search_documents = !self.search_documents;
    true
  }

  /// Apply filter to results
  pub fn apply_filter(&mut self) {
    if self.filter_text.is_empty() {
      self.filter_active = false;
      self.filtered_results = self.results.clone();
    } else {
      self.filter_active = true;
      let filter_lower = self.filter_text.to_lowercase();
      self.filtered_results = self
        .results
        .iter()
        .filter(|r| {
          // Filter by preview content
          let preview = self.get_result_filter_text(r);
          preview.to_lowercase().contains(&filter_lower)
        })
        .cloned()
        .collect();
    }
    // Reset selection if out of bounds
    if self.selected >= self.filtered_results.len() {
      self.selected = 0;
    }
  }

  /// Get text to match against for filtering
  fn get_result_filter_text(&self, result: &SearchResult) -> String {
    match result.result_type {
      SearchResultType::Memory => {
        let content = result.data.get("content").and_then(|c| c.as_str()).unwrap_or("");
        let sector = result.data.get("sector").and_then(|s| s.as_str()).unwrap_or("");
        format!("{} {}", sector, content)
      }
      SearchResultType::Code => {
        let file = result.data.get("file_path").and_then(|f| f.as_str()).unwrap_or("");
        let content = result.data.get("content").and_then(|c| c.as_str()).unwrap_or("");
        let symbols = result
          .data
          .get("symbols")
          .and_then(|s| s.as_array())
          .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(" "))
          .unwrap_or_default();
        format!("{} {} {}", file, symbols, content)
      }
      SearchResultType::Document => {
        let title = result.data.get("title").and_then(|t| t.as_str()).unwrap_or("");
        let content = result.data.get("content").and_then(|c| c.as_str()).unwrap_or("");
        format!("{} {}", title, content)
      }
    }
  }

  /// Clear filter
  pub fn clear_filter(&mut self) {
    self.filter_text.clear();
    self.filter_active = false;
    self.filtered_results = self.results.clone();
    self.selected = 0;
  }

  /// Get displayable results (filtered or all)
  pub fn display_results(&self) -> &[SearchResult] {
    if self.filter_active || !self.filter_text.is_empty() {
      &self.filtered_results
    } else {
      &self.results
    }
  }

  /// Count results by type (from display results)
  pub fn count_by_type(&self) -> (usize, usize, usize) {
    let mut memories = 0;
    let mut code = 0;
    let mut documents = 0;
    for result in self.display_results() {
      match result.result_type {
        SearchResultType::Memory => memories += 1,
        SearchResultType::Code => code += 1,
        SearchResultType::Document => documents += 1,
      }
    }
    (memories, code, documents)
  }
}

/// Unified search view widget
pub struct SearchView<'a> {
  state: &'a SearchState,
  focused: bool,
}

impl<'a> SearchView<'a> {
  pub fn new(state: &'a SearchState) -> Self {
    Self { state, focused: true }
  }

  pub fn focused(mut self, focused: bool) -> Self {
    self.focused = focused;
    self
  }
}

impl Widget for SearchView<'_> {
  fn render(self, area: Rect, buf: &mut Buffer) {
    // Layout: search bar, filter bar (when active), scope toggles, results list, detail panel
    let constraints = if self.state.filter_input_active {
      vec![
        Constraint::Length(3), // Search bar
        Constraint::Length(3), // Filter bar
        Constraint::Length(2), // Scope toggles
        Constraint::Min(10),   // Results
      ]
    } else {
      vec![
        Constraint::Length(3), // Search bar
        Constraint::Length(2), // Scope toggles
        Constraint::Min(10),   // Results
      ]
    };

    let chunks = Layout::default()
      .direction(Direction::Vertical)
      .constraints(constraints)
      .split(area);

    // Search bar
    self.render_search_bar(chunks[0], buf);

    // Filter bar and results based on whether filter input is active
    let (scope_chunk, results_chunk) = if self.state.filter_input_active {
      self.render_filter_bar(chunks[1], buf);
      (chunks[2], chunks[3])
    } else {
      (chunks[1], chunks[2])
    };

    // Scope toggles
    self.render_scope_toggles(scope_chunk, buf);

    // Results split into list and detail
    let result_chunks = Layout::default()
      .direction(Direction::Horizontal)
      .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
      .split(results_chunk);

    self.render_results_list(result_chunks[0], buf);
    self.render_result_detail(result_chunks[1], buf);
  }
}

impl SearchView<'_> {
  fn render_search_bar(&self, area: Rect, buf: &mut Buffer) {
    let border_color = if self.state.input_active {
      Theme::ACCENT
    } else {
      Theme::OVERLAY
    };

    let block = Block::default()
      .title("SEARCH")
      .title_style(Style::default().fg(Theme::ACCENT).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    block.render(area, buf);

    // Search prompt
    buf.set_string(inner.x, inner.y, "> ", Style::default().fg(Theme::ACCENT));

    // Query text
    let query_display = if self.state.query.len() > inner.width as usize - 4 {
      format!(
        "...{}",
        &self.state.query[self.state.query.len().saturating_sub(inner.width as usize - 7)..]
      )
    } else {
      self.state.query.clone()
    };
    buf.set_string(inner.x + 2, inner.y, &query_display, Style::default().fg(Theme::TEXT));

    // Cursor
    if self.state.input_active {
      let cursor_x = inner.x + 2 + query_display.len() as u16;
      if cursor_x < inner.x + inner.width {
        buf.set_string(cursor_x, inner.y, "▌", Style::default().fg(Theme::ACCENT));
      }
    }
  }

  fn render_filter_bar(&self, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
      .title("FILTER")
      .title_style(Style::default().fg(Theme::ACCENT).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::ACCENT));

    let inner = block.inner(area);
    block.render(area, buf);

    // Filter prompt
    buf.set_string(inner.x, inner.y, "> ", Style::default().fg(Theme::ACCENT));

    // Filter text
    let filter_display = if self.state.filter_text.len() > inner.width as usize - 4 {
      format!(
        "...{}",
        &self.state.filter_text[self.state.filter_text.len().saturating_sub(inner.width as usize - 7)..]
      )
    } else {
      self.state.filter_text.clone()
    };
    buf.set_string(inner.x + 2, inner.y, &filter_display, Style::default().fg(Theme::TEXT));

    // Cursor
    let cursor_x = inner.x + 2 + filter_display.len() as u16;
    if cursor_x < inner.x + inner.width {
      buf.set_string(cursor_x, inner.y, "▌", Style::default().fg(Theme::ACCENT));
    }
  }

  fn render_scope_toggles(&self, area: Rect, buf: &mut Buffer) {
    let mut x = area.x + 2;
    let y = area.y;

    // Memories toggle with keybinding hint
    let memories_style = if self.state.search_memories {
      Style::default().fg(Theme::SEMANTIC).bold()
    } else {
      Style::default().fg(Theme::MUTED)
    };
    let memories_label = if self.state.search_memories {
      "[✓] Memories (m)"
    } else {
      "[ ] Memories (m)"
    };
    buf.set_string(x, y, memories_label, memories_style);
    x += memories_label.len() as u16 + 2;

    // Code toggle with keybinding hint
    let code_style = if self.state.search_code {
      Style::default().fg(Theme::PROCEDURAL).bold()
    } else {
      Style::default().fg(Theme::MUTED)
    };
    let code_label = if self.state.search_code {
      "[✓] Code (c)"
    } else {
      "[ ] Code (c)"
    };
    buf.set_string(x, y, code_label, code_style);
    x += code_label.len() as u16 + 2;

    // Documents toggle with keybinding hint
    let docs_style = if self.state.search_documents {
      Style::default().fg(Theme::REFLECTIVE).bold()
    } else {
      Style::default().fg(Theme::MUTED)
    };
    let docs_label = if self.state.search_documents {
      "[✓] Docs (d)"
    } else {
      "[ ] Docs (d)"
    };
    buf.set_string(x, y, docs_label, docs_style);
    x += docs_label.len() as u16 + 3;

    // Filter indicator
    if self.state.filter_active {
      let filter_label = format!("│ Filter: \"{}\"", self.state.filter_text);
      buf.set_string(x, y, &filter_label, Style::default().fg(Theme::ACCENT));
    }
  }

  fn render_results_list(&self, area: Rect, buf: &mut Buffer) {
    let display_results = self.state.display_results();
    let (mem_count, code_count, doc_count) = self.state.count_by_type();

    // Build title with filter info if active
    let title = if self.state.filter_active {
      format!(
        "RESULTS ({}/{} filtered: {} mem, {} code, {} doc)",
        display_results.len(),
        self.state.results.len(),
        mem_count,
        code_count,
        doc_count
      )
    } else {
      format!(
        "RESULTS ({} total: {} mem, {} code, {} doc)",
        display_results.len(),
        mem_count,
        code_count,
        doc_count
      )
    };

    let border_color = if self.focused && !self.state.input_active {
      Theme::ACCENT
    } else {
      Theme::OVERLAY
    };

    let block = Block::default()
      .title(title)
      .title_style(Style::default().fg(Theme::ACCENT).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    block.render(area, buf);

    if display_results.is_empty() {
      let msg = if self.state.loading {
        "Searching..."
      } else if let Some(ref err) = self.state.error {
        err
      } else if self.state.query.is_empty() {
        "Enter a search query"
      } else if self.state.filter_active {
        "No results match filter"
      } else {
        "No results found"
      };
      buf.set_string(inner.x, inner.y, msg, Style::default().fg(Theme::MUTED));
      return;
    }

    // Render grouped results using filtered/display results
    let visible_height = inner.height as usize;
    let start = if self.state.selected >= visible_height {
      self.state.selected - visible_height + 1
    } else {
      0
    };

    for (i, result) in display_results.iter().enumerate().skip(start).take(visible_height) {
      let y = inner.y + (i - start) as u16;
      if y >= inner.y + inner.height {
        break;
      }

      let is_selected = i == self.state.selected;
      self.render_result_item(result, inner.x, y, inner.width, is_selected, buf);
    }
  }

  fn render_result_item(&self, result: &SearchResult, x: u16, y: u16, width: u16, selected: bool, buf: &mut Buffer) {
    let (type_label, type_color) = match result.result_type {
      SearchResultType::Memory => ("MEM", Theme::SEMANTIC),
      SearchResultType::Code => ("COD", Theme::PROCEDURAL),
      SearchResultType::Document => ("DOC", Theme::REFLECTIVE),
    };

    let bg = if selected { Theme::SURFACE } else { Theme::BG };
    let fg = if selected { Theme::TEXT } else { Theme::SUBTEXT };

    // Clear line
    for i in 0..width {
      buf[(x + i, y)].set_bg(bg);
    }

    // Selection indicator
    let indicator = if selected { "▶ " } else { "  " };
    buf.set_string(x, y, indicator, Style::default().fg(Theme::ACCENT));

    // Type badge
    buf.set_string(
      x + 2,
      y,
      format!("[{}] ", type_label),
      Style::default().fg(type_color).bold(),
    );

    // Content preview
    let preview = self.get_result_preview(result);
    let preview_start = x + 8;
    let preview_width = width.saturating_sub(preview_start - x + 6) as usize;
    let preview = if preview.len() > preview_width {
      format!("{}...", &preview[..preview_width.saturating_sub(3)])
    } else {
      preview
    };
    buf.set_string(preview_start, y, &preview, Style::default().fg(fg));

    // Similarity score
    let sim_str = format!("{:.0}%", result.similarity * 100.0);
    let sim_x = x + width.saturating_sub(sim_str.len() as u16 + 1);
    buf.set_string(sim_x, y, &sim_str, Style::default().fg(Theme::MUTED));
  }

  fn get_result_preview(&self, result: &SearchResult) -> String {
    match result.result_type {
      SearchResultType::Memory => result
        .data
        .get("content")
        .and_then(|c| c.as_str())
        .map(|s| s.lines().next().unwrap_or("").trim().to_string())
        .unwrap_or_default(),
      SearchResultType::Code => {
        let file = result.data.get("file_path").and_then(|f| f.as_str()).unwrap_or("");
        let symbols = result
          .data
          .get("symbols")
          .and_then(|s| s.as_array())
          .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(", "))
          .unwrap_or_default();
        if !symbols.is_empty() {
          format!("{}: {}", shorten_path(file), symbols)
        } else {
          shorten_path(file).to_string()
        }
      }
      SearchResultType::Document => result
        .data
        .get("title")
        .and_then(|t| t.as_str())
        .unwrap_or("Untitled")
        .to_string(),
    }
  }

  fn render_result_detail(&self, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
      .title("DETAIL")
      .title_style(Style::default().fg(Theme::ACCENT).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::OVERLAY));

    let inner = block.inner(area);
    block.render(area, buf);

    let Some(result) = self.state.selected_result() else {
      buf.set_string(
        inner.x,
        inner.y,
        "Select a result to view details",
        Style::default().fg(Theme::MUTED),
      );
      return;
    };

    match result.result_type {
      SearchResultType::Memory => self.render_memory_detail(&result.data, inner, buf),
      SearchResultType::Code => self.render_code_detail(&result.data, inner, buf),
      SearchResultType::Document => self.render_document_detail(&result.data, inner, buf),
    }
  }

  fn render_memory_detail(&self, data: &Value, area: Rect, buf: &mut Buffer) {
    let mut y = area.y;

    // ID (full ID for copy/paste)
    if let Some(id) = data.get("id").and_then(|i| i.as_str()) {
      buf.set_string(area.x, y, "ID: ", Style::default().fg(Theme::SUBTEXT));
      let max_id_width = area.width.saturating_sub(4) as usize;
      let display_id = if id.len() > max_id_width {
        format!("{}...", &id[..max_id_width.saturating_sub(3)])
      } else {
        id.to_string()
      };
      buf.set_string(area.x + 4, y, &display_id, Style::default().fg(Theme::TEXT));
      y += 1;
    }

    // Sector
    if let Some(sector) = data.get("sector").and_then(|s| s.as_str()) {
      buf.set_string(area.x, y, "Sector: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(
        area.x + 8,
        y,
        capitalize(sector),
        Style::default().fg(Theme::sector_color(sector)).bold(),
      );
      y += 1;
    }

    // Salience
    if let Some(salience) = data.get("salience").and_then(|s| s.as_f64()) {
      buf.set_string(area.x, y, "Salience: ", Style::default().fg(Theme::SUBTEXT));
      let pct = (salience * 100.0) as u32;
      buf.set_string(
        area.x + 10,
        y,
        format!("{}%", pct),
        Style::default().fg(Theme::salience_color(salience as f32)),
      );
      y += 1;
    }

    y += 1;

    // Content
    if let Some(content) = data.get("content").and_then(|c| c.as_str()) {
      buf.set_string(area.x, y, "Content:", Style::default().fg(Theme::ACCENT).bold());
      y += 1;

      for line in content.lines() {
        if y >= area.y + area.height {
          break;
        }
        let display_line = if line.len() > area.width as usize {
          format!("{}...", &line[..area.width as usize - 3])
        } else {
          line.to_string()
        };
        buf.set_string(area.x, y, &display_line, Style::default().fg(Theme::TEXT));
        y += 1;
      }
    }
  }

  fn render_code_detail(&self, data: &Value, area: Rect, buf: &mut Buffer) {
    let mut y = area.y;

    // File path
    if let Some(file) = data.get("file_path").and_then(|f| f.as_str()) {
      buf.set_string(area.x, y, "File: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(area.x + 6, y, file, Style::default().fg(Theme::TEXT));
      y += 1;
    }

    // Lines
    if let (Some(start), Some(end)) = (
      data.get("start_line").and_then(|l| l.as_u64()),
      data.get("end_line").and_then(|l| l.as_u64()),
    ) {
      buf.set_string(area.x, y, "Lines: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(
        area.x + 7,
        y,
        format!("{}-{}", start, end),
        Style::default().fg(Theme::TEXT),
      );
      y += 1;
    }

    // Language
    if let Some(lang) = data.get("language").and_then(|l| l.as_str()) {
      buf.set_string(area.x, y, "Language: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(
        area.x + 10,
        y,
        capitalize(lang),
        Style::default().fg(Theme::language_color(lang)),
      );
      y += 1;
    }

    // Symbols
    if let Some(symbols) = data.get("symbols").and_then(|s| s.as_array())
      && !symbols.is_empty()
    {
      buf.set_string(area.x, y, "Symbols: ", Style::default().fg(Theme::SUBTEXT));
      let symbols_str = symbols.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(", ");
      let max_len = area.width as usize - 9;
      let symbols_str = if symbols_str.len() > max_len {
        format!("{}...", &symbols_str[..max_len - 3])
      } else {
        symbols_str
      };
      buf.set_string(area.x + 9, y, &symbols_str, Style::default().fg(Theme::INFO));
      y += 1;
    }

    y += 1;

    // Code content
    if let Some(content) = data.get("content").and_then(|c| c.as_str()) {
      buf.set_string(area.x, y, "Code:", Style::default().fg(Theme::ACCENT).bold());
      y += 1;

      for line in content.lines() {
        if y >= area.y + area.height {
          break;
        }
        let display_line = if line.len() > area.width as usize {
          format!("{}...", &line[..area.width as usize - 3])
        } else {
          line.to_string()
        };
        buf.set_string(area.x, y, &display_line, Style::default().fg(Theme::TEXT));
        y += 1;
      }
    }
  }

  fn render_document_detail(&self, data: &Value, area: Rect, buf: &mut Buffer) {
    let mut y = area.y;

    // Document ID (full ID for copy/paste)
    if let Some(id) = data.get("document_id").and_then(|i| i.as_str()) {
      buf.set_string(area.x, y, "ID: ", Style::default().fg(Theme::SUBTEXT));
      let max_id_width = area.width.saturating_sub(4) as usize;
      let display_id = if id.len() > max_id_width {
        format!("{}...", &id[..max_id_width.saturating_sub(3)])
      } else {
        id.to_string()
      };
      buf.set_string(area.x + 4, y, &display_id, Style::default().fg(Theme::TEXT));
      y += 1;
    }

    // Title
    if let Some(title) = data.get("title").and_then(|t| t.as_str()) {
      buf.set_string(area.x, y, "Title: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(area.x + 7, y, title, Style::default().fg(Theme::TEXT).bold());
      y += 1;
    }

    // Source
    if let Some(source) = data.get("source").and_then(|s| s.as_str()) {
      buf.set_string(area.x, y, "Source: ", Style::default().fg(Theme::SUBTEXT));
      let max_len = area.width as usize - 8;
      let source_display = if source.len() > max_len {
        format!("{}...", &source[..max_len - 3])
      } else {
        source.to_string()
      };
      buf.set_string(area.x + 8, y, &source_display, Style::default().fg(Theme::INFO));
      y += 1;
    }

    y += 1;

    // Content preview
    if let Some(content) = data.get("content").and_then(|c| c.as_str()) {
      buf.set_string(area.x, y, "Preview:", Style::default().fg(Theme::ACCENT).bold());
      y += 1;

      for line in content.lines() {
        if y >= area.y + area.height {
          break;
        }
        let display_line = if line.len() > area.width as usize {
          format!("{}...", &line[..area.width as usize - 3])
        } else {
          line.to_string()
        };
        buf.set_string(area.x, y, &display_line, Style::default().fg(Theme::TEXT));
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

fn shorten_path(path: &str) -> &str {
  path.rsplit('/').next().unwrap_or(path)
}
