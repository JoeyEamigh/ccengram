use std::collections::{HashMap, HashSet};

use ccengram::ipc::docs::{DocContextResult, DocSearchItem};
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

/// A tree item in the left panel - either a document or a chunk
#[derive(Debug, Clone)]
pub enum TreeItem {
  Document {
    id: String,
    title: String,
    source: String,
    chunk_count: usize,
  },
  Chunk {
    #[allow(dead_code)]
    doc_id: String,
    chunk: DocSearchItem,
  },
}

/// Document browser view state
#[derive(Debug, Default)]
pub struct DocumentState {
  pub documents: Vec<DocSearchItem>,
  /// Which panel is focused
  pub focus: Panel,
  /// Expanded documents in the tree (by document ID)
  pub expanded_docs: HashSet<String>,
  /// Flattened tree items for navigation
  pub tree_items: Vec<TreeItem>,
  /// Selected index in tree
  pub tree_selected: usize,
  /// Scroll position for right panel
  pub detail_scroll: usize,
  pub search_query: String,
  pub loading: bool,
  pub error: Option<String>,
  /// Expanded context for the currently selected chunk
  pub expanded_context: Option<DocContextResult>,
}

impl DocumentState {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn set_documents(&mut self, documents: Vec<DocSearchItem>) {
    self.documents = documents;
    self.expanded_context = None;
    self.detail_scroll = 0;
    self.rebuild_tree();
  }

  /// Rebuild the flattened tree from documents
  fn rebuild_tree(&mut self) {
    // Group by document_id
    let mut doc_map: HashMap<&str, Vec<&DocSearchItem>> = HashMap::new();
    for doc in &self.documents {
      doc_map.entry(&doc.document_id).or_default().push(doc);
    }

    // Sort chunks within each document by chunk_index
    for chunks in doc_map.values_mut() {
      chunks.sort_by_key(|c| c.chunk_index);
    }

    // Sort documents by title
    let mut docs: Vec<_> = doc_map.into_iter().collect();
    docs.sort_by(|a, b| {
      let title_a = a.1.first().map(|c| c.title.as_str()).unwrap_or("");
      let title_b = b.1.first().map(|c| c.title.as_str()).unwrap_or("");
      title_a.cmp(title_b)
    });

    self.tree_items.clear();
    for (doc_id, chunks) in docs {
      let first_chunk = chunks.first();
      let title = first_chunk.map(|c| c.title.clone()).unwrap_or_default();
      let source = first_chunk.map(|c| c.source.clone()).unwrap_or_default();
      let chunk_count = first_chunk.map(|c| c.total_chunks).unwrap_or(chunks.len());

      self.tree_items.push(TreeItem::Document {
        id: doc_id.to_string(),
        title,
        source,
        chunk_count,
      });

      // If expanded, add chunks
      if self.expanded_docs.contains(doc_id) {
        for chunk in chunks {
          self.tree_items.push(TreeItem::Chunk {
            doc_id: doc_id.to_string(),
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

  /// Get the currently selected document chunk (if a chunk is selected)
  pub fn selected_document(&self) -> Option<&DocSearchItem> {
    match self.selected_tree_item()? {
      TreeItem::Chunk { chunk, .. } => Some(chunk),
      TreeItem::Document { id, .. } => {
        // If document is selected but expanded, return first chunk
        if self.expanded_docs.contains(id) {
          self.documents.iter().find(|c| &c.document_id == id)
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
        // Expand/collapse document, or select chunk
        if let Some(item) = self.selected_tree_item().cloned() {
          match item {
            TreeItem::Document { id, .. } => {
              if self.expanded_docs.contains(&id) {
                self.expanded_docs.remove(&id);
              } else {
                self.expanded_docs.insert(id);
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
          self.selected_document().is_some()
        }
      }
    }
  }

  pub fn scroll_detail_down(&mut self) {
    self.detail_scroll = self.detail_scroll.saturating_add(1);
  }

  pub fn scroll_detail_up(&mut self) {
    self.detail_scroll = self.detail_scroll.saturating_sub(1);
  }

  /// Set expanded context for the current selection
  pub fn set_expanded_context(&mut self, context: DocContextResult) {
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
}

/// Document browser view widget
pub struct DocumentView<'a> {
  state: &'a DocumentState,
}

impl<'a> DocumentView<'a> {
  pub fn new(state: &'a DocumentState) -> Self {
    Self { state }
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
    let is_focused = self.state.focus == Panel::Left;
    let border_color = if is_focused { Theme::ACCENT } else { Theme::OVERLAY };

    let doc_count = self
      .state
      .tree_items
      .iter()
      .filter(|i| matches!(i, TreeItem::Document { .. }))
      .count();

    let title = if !self.state.search_query.is_empty() {
      format!("DOCUMENTS ({}) - Search: {} [Tab]", doc_count, self.state.search_query)
    } else {
      format!("DOCUMENTS ({}) [Tab to switch]", doc_count)
    };

    let block = Block::default()
      .title(title)
      .title_style(Style::default().fg(Theme::REFLECTIVE).bold())
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
        "No documents found"
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
      TreeItem::Document {
        id,
        title,
        source,
        chunk_count,
      } => {
        let is_expanded = self.state.expanded_docs.contains(id);
        let icon = if is_expanded { "▼ " } else { "▶ " };
        let indicator = if selected { "» " } else { "  " };

        // Source type icon
        let source_icon = if source.starts_with("http") { "🌐" } else { "📄" };

        buf.set_string(x, y, indicator, Style::default().fg(Theme::ACCENT));
        buf.set_string(x + 2, y, icon, Style::default().fg(Theme::MUTED));
        buf.set_string(x + 4, y, source_icon, Style::default());

        let title_start = x + 7;
        let title_width = width.saturating_sub(title_start - x + 12) as usize;
        let display_title = if title.len() > title_width && title_width > 3 {
          format!("{}...", &title[..title_width - 3])
        } else {
          title.clone()
        };
        buf.set_string(title_start, y, &display_title, Style::default().fg(fg));

        let count = format!(" ({} chunks)", chunk_count);
        let count_x = x + width.saturating_sub(count.len() as u16 + 1);
        buf.set_string(count_x, y, &count, Style::default().fg(Theme::MUTED));
      }
      TreeItem::Chunk { chunk, .. } => {
        let indicator = if selected { "» " } else { "  " };
        buf.set_string(x, y, indicator, Style::default().fg(Theme::ACCENT));
        buf.set_string(x + 2, y, "  └ ", Style::default().fg(Theme::MUTED));

        let info = format!("Chunk {}/{}", chunk.chunk_index + 1, chunk.total_chunks);
        buf.set_string(x + 6, y, &info, Style::default().fg(fg));

        // Preview of content
        let preview_start = x + 6 + info.len() as u16 + 2;
        let preview_width = width.saturating_sub(preview_start - x + 1) as usize;
        if preview_width > 5 {
          let preview = chunk.content.lines().next().unwrap_or("").trim();
          let display_preview = if preview.len() > preview_width && preview_width > 3 {
            format!("{}...", &preview[..preview_width - 3])
          } else {
            preview.to_string()
          };
          buf.set_string(preview_start, y, &display_preview, Style::default().fg(Theme::SUBTEXT));
        }
      }
    }
  }

  fn render_detail(&self, area: Rect, buf: &mut Buffer) {
    let is_focused = self.state.focus == Panel::Right;
    let border_color = if is_focused { Theme::ACCENT } else { Theme::OVERLAY };

    // Check if we have expanded context
    let title = if self.state.expanded_context.is_some() {
      "CONTEXT (Enter to collapse)"
    } else if is_focused {
      "DOCUMENT DETAIL (Enter for context)"
    } else {
      "DOCUMENT DETAIL"
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
      self.render_doc_context(ctx, inner, buf);
      return;
    }

    let Some(doc) = self.state.selected_document() else {
      buf.set_string(
        inner.x,
        inner.y,
        "Select a chunk to preview (Enter on doc to expand)",
        Style::default().fg(Theme::MUTED),
      );
      return;
    };

    self.render_doc_preview(doc, inner, buf);
  }

  fn render_doc_preview(&self, doc: &DocSearchItem, area: Rect, buf: &mut Buffer) {
    let mut lines: Vec<(String, Style)> = Vec::new();

    // Header info
    lines.push((format!("Title: {}", doc.title), Style::default().fg(Theme::TEXT).bold()));
    lines.push((
      format!("ID: {}...", &doc.id[..8.min(doc.id.len())]),
      Style::default().fg(Theme::MUTED),
    ));
    lines.push((format!("Source: {}", doc.source), Style::default().fg(Theme::INFO)));

    // Use the actual source_type from the data
    let source_type_display = match doc.source_type.as_str() {
      "url" => "URL",
      "file" => "File",
      "content" => "Direct Content",
      other => other,
    };
    lines.push((
      format!("Type: {}", source_type_display),
      Style::default().fg(Theme::PROCEDURAL),
    ));

    // Chunk position info
    lines.push((
      format!("Chunk: {}/{}", doc.chunk_index + 1, doc.total_chunks),
      Style::default().fg(Theme::TEXT),
    ));

    // Character offset if available
    if let Some(offset) = doc.char_offset {
      lines.push((format!("Offset: {} chars", offset), Style::default().fg(Theme::MUTED)));
    }

    // Similarity score if available
    if let Some(sim) = doc.similarity {
      lines.push((
        format!("Similarity: {:.1}%", sim * 100.0),
        Style::default().fg(Theme::REFLECTIVE),
      ));
    }

    lines.push((String::new(), Style::default()));
    lines.push((
      "Content Preview:".to_string(),
      Style::default().fg(Theme::ACCENT).bold(),
    ));

    for line in doc.content.lines() {
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

  fn render_doc_context(&self, ctx: &DocContextResult, area: Rect, buf: &mut Buffer) {
    let mut lines: Vec<(String, Style)> = Vec::new();

    // Document info
    lines.push((format!("Title: {}", ctx.title), Style::default().fg(Theme::TEXT).bold()));
    lines.push((format!("Source: {}", ctx.source), Style::default().fg(Theme::INFO)));
    lines.push((
      format!("Chunk {} of {}", ctx.context.target.chunk_index + 1, ctx.total_chunks),
      Style::default().fg(Theme::MUTED),
    ));

    lines.push((String::new(), Style::default()));

    // Before chunks
    for chunk in &ctx.context.before {
      lines.push((
        format!("--- Chunk {} ---", chunk.chunk_index + 1),
        Style::default().fg(Theme::MUTED),
      ));
      for line in chunk.content.lines() {
        lines.push((line.to_string(), Style::default().fg(Theme::SUBTEXT)));
      }
      lines.push((String::new(), Style::default()));
    }

    // Target chunk
    lines.push((
      format!(">>> Chunk {} (target) <<<", ctx.context.target.chunk_index + 1),
      Style::default().fg(Theme::ACCENT).bold(),
    ));
    for line in ctx.context.target.content.lines() {
      lines.push((line.to_string(), Style::default().fg(Theme::TEXT)));
    }
    lines.push((String::new(), Style::default()));

    // After chunks
    for chunk in &ctx.context.after {
      lines.push((
        format!("--- Chunk {} ---", chunk.chunk_index + 1),
        Style::default().fg(Theme::MUTED),
      ));
      for line in chunk.content.lines() {
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
}

fn truncate_line(line: &str, max_len: usize) -> String {
  if line.len() > max_len && max_len > 3 {
    format!("{}...", &line[..max_len - 3])
  } else {
    line.to_string()
  }
}
