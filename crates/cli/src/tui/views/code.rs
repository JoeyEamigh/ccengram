use std::collections::{HashMap, HashSet};

use ccengram::ipc::code::{CodeContextResponse, CodeItem, CodeStatsResult};
use ratatui::{
  buffer::Buffer,
  layout::{Constraint, Direction, Layout, Rect},
  style::Style,
  widgets::{Block, Borders, Widget},
};

use crate::tui::theme::Theme;

/// Which panel is focused
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Panel {
  #[default]
  Left,
  Right,
}

#[allow(clippy::large_enum_variant)]
/// A tree item in the left panel - either a file or a chunk
#[derive(Debug, Clone)]
pub enum TreeItem {
  File {
    path: String,
    chunk_count: usize,
    language: Option<String>,
  },
  Chunk {
    #[allow(dead_code)]
    file_path: String,
    chunk: CodeItem,
  },
}

/// Code index view state
#[derive(Debug, Default)]
pub struct CodeState {
  pub chunks: Vec<CodeItem>,
  pub stats: Option<CodeStatsResult>,
  /// Which panel is focused
  pub focus: Panel,
  /// Expanded files in the tree (by file path)
  pub expanded_files: HashSet<String>,
  /// Flattened tree items for navigation
  pub tree_items: Vec<TreeItem>,
  /// Selected index in tree
  pub tree_selected: usize,
  /// Scroll position for right panel
  pub detail_scroll: usize,
  pub loading: bool,
  pub error: Option<String>,
  /// Expanded context for the currently selected chunk
  pub expanded_context: Option<CodeContextResponse>,
}

impl CodeState {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn set_chunks(&mut self, chunks: Vec<CodeItem>) {
    self.chunks = chunks;
    self.expanded_context = None;
    self.detail_scroll = 0;
    self.rebuild_tree();
  }

  pub fn set_stats(&mut self, stats: CodeStatsResult) {
    self.stats = Some(stats);
  }

  /// Rebuild the flattened tree from chunks
  fn rebuild_tree(&mut self) {
    let mut file_map: HashMap<&str, Vec<&CodeItem>> = HashMap::new();
    for chunk in &self.chunks {
      file_map.entry(&chunk.file_path).or_default().push(chunk);
    }

    let mut files: Vec<_> = file_map.into_iter().collect();
    files.sort_by(|a, b| a.0.cmp(b.0));

    self.tree_items.clear();
    for (path, chunks) in files {
      let language = chunks.first().and_then(|c| c.language.clone());
      self.tree_items.push(TreeItem::File {
        path: path.to_string(),
        chunk_count: chunks.len(),
        language,
      });

      // If expanded, add chunks
      if self.expanded_files.contains(path) {
        for chunk in chunks {
          self.tree_items.push(TreeItem::Chunk {
            file_path: path.to_string(),
            chunk: chunk.clone(),
          });
        }
      }
    }

    // Bounds check selection
    if self.tree_selected >= self.tree_items.len() && !self.tree_items.is_empty() {
      self.tree_selected = self.tree_items.len() - 1;
    }
  }

  /// Get the currently selected tree item
  pub fn selected_tree_item(&self) -> Option<&TreeItem> {
    self.tree_items.get(self.tree_selected)
  }

  /// Get the currently selected chunk (if a chunk is selected)
  pub fn selected_chunk(&self) -> Option<&CodeItem> {
    match self.selected_tree_item()? {
      TreeItem::Chunk { chunk, .. } => Some(chunk),
      TreeItem::File { path, .. } => {
        // If file is selected but expanded, return first chunk
        if self.expanded_files.contains(path) {
          self.chunks.iter().find(|c| &c.file_path == path)
        } else {
          None
        }
      }
    }
  }

  /// Toggle focus between panels
  pub fn toggle_focus(&mut self) {
    self.focus = match self.focus {
      Panel::Left => Panel::Right,
      Panel::Right => Panel::Left,
    };
  }

  /// Navigate down in current panel
  pub fn select_next(&mut self) {
    match self.focus {
      Panel::Left => {
        if !self.tree_items.is_empty() {
          let new_selected = (self.tree_selected + 1).min(self.tree_items.len() - 1);
          if new_selected != self.tree_selected {
            self.tree_selected = new_selected;
            self.expanded_context = None;
            self.detail_scroll = 0;
          }
        }
      }
      Panel::Right => {
        // Scroll down in right panel
        self.detail_scroll = self.detail_scroll.saturating_add(1);
      }
    }
  }

  /// Navigate up in current panel
  pub fn select_prev(&mut self) {
    match self.focus {
      Panel::Left => {
        if !self.tree_items.is_empty() {
          let new_selected = self.tree_selected.saturating_sub(1);
          if new_selected != self.tree_selected {
            self.tree_selected = new_selected;
            self.expanded_context = None;
            self.detail_scroll = 0;
          }
        }
      }
      Panel::Right => {
        // Scroll up in right panel
        self.detail_scroll = self.detail_scroll.saturating_sub(1);
      }
    }
  }

  /// Handle Enter key - expand/collapse in left panel, context in right
  /// Returns true if context should be fetched (right panel, chunk selected)
  pub fn handle_enter(&mut self) -> bool {
    match self.focus {
      Panel::Left => {
        // Expand/collapse file, or select chunk
        if let Some(item) = self.selected_tree_item().cloned() {
          match item {
            TreeItem::File { path, .. } => {
              if self.expanded_files.contains(&path) {
                self.expanded_files.remove(&path);
              } else {
                self.expanded_files.insert(path);
              }
              self.rebuild_tree();
            }
            TreeItem::Chunk { .. } => {
              // Switch focus to right panel when selecting a chunk
              self.focus = Panel::Right;
            }
          }
        }
        false
      }
      Panel::Right => {
        // Toggle context expansion
        if self.expanded_context.is_some() {
          self.expanded_context = None;
          false
        } else {
          // Signal to fetch context
          self.selected_chunk().is_some()
        }
      }
    }
  }

  /// Set expanded context for the current selection
  pub fn set_expanded_context(&mut self, context: CodeContextResponse) {
    self.expanded_context = Some(context);
    self.detail_scroll = 0;
  }

  /// Clear expanded context
  #[allow(dead_code)]
  pub fn clear_expanded_context(&mut self) {
    self.expanded_context = None;
  }

  /// Check if we have expanded context
  #[allow(dead_code)]
  pub fn has_expanded_context(&self) -> bool {
    self.expanded_context.is_some()
  }

  /// Get language breakdown from stats
  pub fn language_breakdown(&self) -> Vec<(String, u64)> {
    self
      .stats
      .as_ref()
      .map(|s| {
        let mut langs: Vec<_> = s
          .language_breakdown
          .iter()
          .map(|(k, v)| (k.clone(), *v as u64))
          .collect();
        langs.sort_by(|a, b| b.1.cmp(&a.1));
        langs
      })
      .unwrap_or_default()
  }
}

/// Code index view widget
pub struct CodeView<'a> {
  state: &'a CodeState,
}

impl<'a> CodeView<'a> {
  pub fn new(state: &'a CodeState) -> Self {
    Self { state }
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
    let is_focused = self.state.focus == Panel::Left;
    let border_color = if is_focused { Theme::ACCENT } else { Theme::OVERLAY };

    let file_count = self
      .state
      .tree_items
      .iter()
      .filter(|i| matches!(i, TreeItem::File { .. }))
      .count();
    let title = format!("FILES ({}) [Tab to switch]", file_count);

    let block = Block::default()
      .title(title)
      .title_style(Style::default().fg(Theme::PROCEDURAL).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    block.render(area, buf);

    if self.state.tree_items.is_empty() {
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

    // Render tree items with scroll
    let visible_height = inner.height as usize;
    let start = if self.state.tree_selected >= visible_height {
      self.state.tree_selected - visible_height + 1
    } else {
      0
    };

    for (i, item) in self
      .state
      .tree_items
      .iter()
      .enumerate()
      .skip(start)
      .take(visible_height)
    {
      let y = inner.y + (i - start) as u16;
      if y >= inner.y + inner.height {
        break;
      }

      let is_selected = i == self.state.tree_selected && is_focused;
      self.render_tree_item(
        item,
        inner.x,
        y,
        inner.width,
        is_selected,
        i == self.state.tree_selected,
        buf,
      );
    }
  }

  #[allow(clippy::too_many_arguments)]
  fn render_tree_item(
    &self,
    item: &TreeItem,
    x: u16,
    y: u16,
    width: u16,
    selected: bool,
    is_current: bool,
    buf: &mut Buffer,
  ) {
    let bg = if selected || is_current {
      Theme::SURFACE
    } else {
      Theme::BG
    };
    let fg = if selected || is_current {
      Theme::TEXT
    } else {
      Theme::SUBTEXT
    };

    // Clear line
    for i in 0..width {
      buf[(x + i, y)].set_bg(bg);
    }

    match item {
      TreeItem::File {
        path,
        chunk_count,
        language,
      } => {
        let is_expanded = self.state.expanded_files.contains(path);
        let icon = if is_expanded { "▼ " } else { "▶ " };
        let indicator = if selected { "» " } else { "  " };

        buf.set_string(x, y, indicator, Style::default().fg(Theme::ACCENT));
        buf.set_string(x + 2, y, icon, Style::default().fg(Theme::MUTED));

        let lang_color = language.as_deref().map(Theme::language_color).unwrap_or(Theme::TEXT);
        let display_path = shorten_path(path, width as usize - 15);
        buf.set_string(x + 4, y, &display_path, Style::default().fg(lang_color));

        let count = format!(" ({})", chunk_count);
        let count_x = x + width.saturating_sub(count.len() as u16 + 1);
        buf.set_string(count_x, y, &count, Style::default().fg(Theme::MUTED));
      }
      TreeItem::Chunk { chunk, .. } => {
        let indicator = if selected { "» " } else { "  " };
        buf.set_string(x, y, indicator, Style::default().fg(Theme::ACCENT));
        buf.set_string(x + 2, y, "  └ ", Style::default().fg(Theme::MUTED));

        let chunk_type = chunk.chunk_type.as_deref().unwrap_or("block");
        let symbols = if !chunk.symbols.is_empty() {
          chunk.symbols.join(", ")
        } else {
          String::new()
        };

        let info = format!("{}:{}-{}", chunk_type, chunk.start_line, chunk.end_line);
        buf.set_string(x + 6, y, &info, Style::default().fg(fg));

        if !symbols.is_empty() {
          let sym_x = x + 6 + info.len() as u16 + 1;
          let max_sym_len = width.saturating_sub(sym_x - x + 1) as usize;
          let sym_display = if symbols.len() > max_sym_len && max_sym_len > 3 {
            format!("{}...", &symbols[..max_sym_len - 3])
          } else {
            symbols
          };
          buf.set_string(sym_x, y, &sym_display, Style::default().fg(Theme::INFO));
        }
      }
    }
  }

  fn render_code_preview(&self, area: Rect, buf: &mut Buffer) {
    let is_focused = self.state.focus == Panel::Right;
    let border_color = if is_focused { Theme::ACCENT } else { Theme::OVERLAY };

    // Check if we have expanded context
    let title = if self.state.expanded_context.is_some() {
      "CONTEXT (Enter to collapse)"
    } else if is_focused {
      "CODE PREVIEW (Enter for context)"
    } else {
      "CODE PREVIEW"
    };

    let block = Block::default()
      .title(title)
      .title_style(Style::default().fg(Theme::ACCENT).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    block.render(area, buf);

    // If we have expanded context, render that
    if let Some(ref ctx) = self.state.expanded_context {
      self.render_code_context(ctx, inner, buf);
      return;
    }

    let Some(chunk) = self.state.selected_chunk() else {
      buf.set_string(
        inner.x,
        inner.y,
        "Select a chunk to preview (Enter on file to expand)",
        Style::default().fg(Theme::MUTED),
      );
      return;
    };

    self.render_chunk_preview(chunk, inner, buf);
  }

  fn render_chunk_preview(&self, chunk: &CodeItem, area: Rect, buf: &mut Buffer) {
    let mut lines: Vec<(String, Style)> = Vec::new();

    // Header info
    lines.push((format!("File: {}", chunk.file_path), Style::default().fg(Theme::TEXT)));
    lines.push((
      format!("Lines: {}-{}", chunk.start_line, chunk.end_line),
      Style::default().fg(Theme::TEXT),
    ));

    if let Some(ref language) = chunk.language {
      lines.push((
        format!("Language: {}", capitalize(language)),
        Style::default().fg(Theme::language_color(language)),
      ));
    }

    // Chunk type and definition kind
    if let Some(ref chunk_type) = chunk.chunk_type {
      let kind_info = if let Some(ref def_kind) = chunk.definition_kind {
        format!("Type: {} ({})", capitalize(chunk_type), def_kind)
      } else {
        format!("Type: {}", capitalize(chunk_type))
      };
      lines.push((kind_info, Style::default().fg(Theme::PROCEDURAL)));
    }

    // Visibility
    if let Some(ref visibility) = chunk.visibility {
      lines.push((format!("Visibility: {}", visibility), Style::default().fg(Theme::MUTED)));
    }

    // Symbol name and parent
    if let Some(ref symbol_name) = chunk.symbol_name {
      let name_info = if let Some(ref parent) = chunk.parent_definition {
        format!("Symbol: {}::{}", parent, symbol_name)
      } else {
        format!("Symbol: {}", symbol_name)
      };
      lines.push((name_info, Style::default().fg(Theme::ACCENT)));
    } else if !chunk.symbols.is_empty() {
      let symbols_str = chunk.symbols.join(", ");
      lines.push((format!("Symbols: {}", symbols_str), Style::default().fg(Theme::INFO)));
    }

    // Signature
    if let Some(ref signature) = chunk.signature {
      lines.push((String::new(), Style::default()));
      lines.push(("Signature:".to_string(), Style::default().fg(Theme::REFLECTIVE).bold()));
      lines.push((format!("  {}", signature), Style::default().fg(Theme::TEXT)));
    }

    // Docstring
    if let Some(ref docstring) = chunk.docstring {
      lines.push((String::new(), Style::default()));
      lines.push((
        "Documentation:".to_string(),
        Style::default().fg(Theme::REFLECTIVE).bold(),
      ));
      for doc_line in docstring.lines().take(5) {
        lines.push((format!("  {}", doc_line), Style::default().fg(Theme::SUBTEXT)));
      }
      let doc_line_count = docstring.lines().count();
      if doc_line_count > 5 {
        lines.push((
          format!("  ... ({} more lines)", doc_line_count - 5),
          Style::default().fg(Theme::MUTED),
        ));
      }
    }

    // Imports and calls
    if !chunk.imports.is_empty() {
      lines.push((String::new(), Style::default()));
      lines.push((
        format!("Imports ({}):", chunk.imports.len()),
        Style::default().fg(Theme::INFO),
      ));
      for imp in chunk.imports.iter().take(5) {
        lines.push((format!("  {}", imp), Style::default().fg(Theme::SUBTEXT)));
      }
      if chunk.imports.len() > 5 {
        lines.push((
          format!("  ... ({} more)", chunk.imports.len() - 5),
          Style::default().fg(Theme::MUTED),
        ));
      }
    }

    if !chunk.calls.is_empty() {
      lines.push((String::new(), Style::default()));
      lines.push((
        format!("Calls ({}):", chunk.calls.len()),
        Style::default().fg(Theme::INFO),
      ));
      for call in chunk.calls.iter().take(5) {
        lines.push((format!("  {}", call), Style::default().fg(Theme::SUBTEXT)));
      }
      if chunk.calls.len() > 5 {
        lines.push((
          format!("  ... ({} more)", chunk.calls.len() - 5),
          Style::default().fg(Theme::MUTED),
        ));
      }
    }

    // Relationship counts
    let has_callers = chunk.caller_count.map(|c| c > 0).unwrap_or(false);
    let has_callees = chunk.callee_count.map(|c| c > 0).unwrap_or(false);
    if has_callers || has_callees {
      lines.push((String::new(), Style::default()));
      let mut rel_parts = Vec::new();
      if let Some(callers) = chunk.caller_count
        && callers > 0
      {
        rel_parts.push(format!("{} callers", callers));
      }
      if let Some(callees) = chunk.callee_count
        && callees > 0
      {
        rel_parts.push(format!("{} callees", callees));
      }
      lines.push((
        format!("Relationships: {}", rel_parts.join(", ")),
        Style::default().fg(Theme::MUTED),
      ));
    }

    lines.push((String::new(), Style::default()));
    lines.push(("Content:".to_string(), Style::default().fg(Theme::ACCENT).bold()));

    for line in chunk.content.lines() {
      lines.push((line.to_string(), Style::default().fg(Theme::TEXT)));
    }

    // Render with scroll
    let scroll = self.state.detail_scroll;
    for (i, (line, style)) in lines.iter().enumerate().skip(scroll).take(area.height as usize) {
      let y = area.y + (i - scroll) as u16;
      if y >= area.y + area.height {
        break;
      }
      let display_line = truncate_line(line, area.width as usize);
      buf.set_string(area.x, y, &display_line, *style);
    }

    // Scroll indicator
    if scroll > 0 || lines.len() > area.height as usize + scroll {
      let indicator = format!(
        "[{}/{}]",
        scroll + 1,
        lines.len().saturating_sub(area.height as usize) + 1
      );
      let ind_x = area.x + area.width.saturating_sub(indicator.len() as u16);
      buf.set_string(ind_x, area.y, &indicator, Style::default().fg(Theme::MUTED));
    }
  }

  fn render_code_context(&self, ctx: &CodeContextResponse, area: Rect, buf: &mut Buffer) {
    let mut lines: Vec<(String, Style)> = Vec::new();

    // File info
    lines.push((format!("File: {}", ctx.file_path), Style::default().fg(Theme::TEXT)));
    lines.push((
      format!("Language: {}", capitalize(&ctx.language)),
      Style::default().fg(Theme::language_color(&ctx.language)),
    ));

    if let Some(ref warning) = ctx.warning {
      lines.push((format!("Warning: {}", warning), Style::default().fg(Theme::ERROR)));
    }

    lines.push((String::new(), Style::default()));

    // Before section
    if !ctx.context.before.content.is_empty() {
      lines.push((
        format!("--- Before (line {}) ---", ctx.context.before.start_line),
        Style::default().fg(Theme::MUTED),
      ));
      for line in ctx.context.before.content.lines() {
        lines.push((line.to_string(), Style::default().fg(Theme::SUBTEXT)));
      }
      lines.push((String::new(), Style::default()));
    }

    // Target section
    lines.push((
      format!(
        ">>> Target (lines {}-{}) <<<",
        ctx.context.target.start_line, ctx.context.target.end_line
      ),
      Style::default().fg(Theme::ACCENT).bold(),
    ));
    for line in ctx.context.target.content.lines() {
      lines.push((line.to_string(), Style::default().fg(Theme::TEXT)));
    }
    lines.push((String::new(), Style::default()));

    // After section
    if !ctx.context.after.content.is_empty() {
      lines.push((
        format!("--- After (line {}) ---", ctx.context.after.start_line),
        Style::default().fg(Theme::MUTED),
      ));
      for line in ctx.context.after.content.lines() {
        lines.push((line.to_string(), Style::default().fg(Theme::SUBTEXT)));
      }
    }

    // Render with scroll
    let scroll = self.state.detail_scroll;
    for (i, (line, style)) in lines.iter().enumerate().skip(scroll).take(area.height as usize) {
      let y = area.y + (i - scroll) as u16;
      if y >= area.y + area.height {
        break;
      }
      let display_line = truncate_line(line, area.width as usize);
      buf.set_string(area.x, y, &display_line, *style);
    }

    // Scroll indicator
    if scroll > 0 || lines.len() > area.height as usize + scroll {
      let indicator = format!(
        "[{}/{}]",
        scroll + 1,
        lines.len().saturating_sub(area.height as usize) + 1
      );
      let ind_x = area.x + area.width.saturating_sub(indicator.len() as u16);
      buf.set_string(ind_x, area.y, &indicator, Style::default().fg(Theme::MUTED));
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

fn truncate_line(line: &str, max_len: usize) -> String {
  if line.len() > max_len && max_len > 3 {
    format!("{}...", &line[..max_len - 3])
  } else {
    line.to_string()
  }
}
