use crate::theme::Theme;
use ratatui::{
  buffer::Buffer,
  layout::{Constraint, Direction, Layout, Rect},
  style::Style,
  widgets::{Block, Borders, Widget},
};
use serde_json::Value;

/// Session timeline view state
#[derive(Debug, Default)]
pub struct SessionState {
  pub sessions: Vec<Value>,
  pub selected: usize,
  pub expanded_session: Option<usize>,
  pub session_memories: Vec<Value>,
  pub loading: bool,
  pub error: Option<String>,
}

impl SessionState {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn set_sessions(&mut self, sessions: Vec<Value>) {
    self.sessions = sessions;
    if self.selected >= self.sessions.len() && !self.sessions.is_empty() {
      self.selected = self.sessions.len() - 1;
    }
  }

  pub fn set_session_memories(&mut self, memories: Vec<Value>) {
    self.session_memories = memories;
  }

  pub fn selected_session(&self) -> Option<&Value> {
    self.sessions.get(self.selected)
  }

  pub fn select_next(&mut self) {
    if self.sessions.is_empty() {
      return;
    }
    self.selected = (self.selected + 1).min(self.sessions.len() - 1);
  }

  pub fn select_prev(&mut self) {
    if self.sessions.is_empty() {
      return;
    }
    self.selected = self.selected.saturating_sub(1);
  }

  pub fn toggle_expand(&mut self) {
    if self.expanded_session == Some(self.selected) {
      self.expanded_session = None;
    } else {
      self.expanded_session = Some(self.selected);
    }
  }

  pub fn selected_id(&self) -> Option<String> {
    self
      .selected_session()
      .and_then(|s| s.get("id"))
      .and_then(|id| id.as_str())
      .map(|s| s.to_string())
  }
}

/// Session timeline view widget
pub struct SessionView<'a> {
  state: &'a SessionState,
  focused: bool,
}

impl<'a> SessionView<'a> {
  pub fn new(state: &'a SessionState) -> Self {
    Self { state, focused: true }
  }

  pub fn focused(mut self, focused: bool) -> Self {
    self.focused = focused;
    self
  }
}

impl Widget for SessionView<'_> {
  fn render(self, area: Rect, buf: &mut Buffer) {
    // Split into timeline and detail panels
    let chunks = Layout::default()
      .direction(Direction::Horizontal)
      .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
      .split(area);

    // Render timeline
    self.render_timeline(chunks[0], buf);

    // Render session detail
    self.render_detail(chunks[1], buf);
  }
}

impl SessionView<'_> {
  fn render_timeline(&self, area: Rect, buf: &mut Buffer) {
    let border_color = if self.focused { Theme::ACCENT } else { Theme::OVERLAY };

    let block = Block::default()
      .title(format!("SESSION TIMELINE ({})", self.state.sessions.len()))
      .title_style(Style::default().fg(Theme::EPISODIC).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    block.render(area, buf);

    if self.state.sessions.is_empty() {
      let msg = if self.state.loading {
        "Loading..."
      } else if let Some(ref err) = self.state.error {
        err
      } else {
        "No sessions found"
      };
      buf.set_string(inner.x, inner.y, msg, Style::default().fg(Theme::MUTED));
      return;
    }

    // Render vertical timeline
    let visible_height = inner.height as usize;
    let start = if self.state.selected >= visible_height {
      self.state.selected - visible_height + 1
    } else {
      0
    };

    let timeline_x = inner.x + 1;

    for (i, session) in self.state.sessions.iter().enumerate().skip(start).take(visible_height) {
      let y = inner.y + (i - start) as u16;
      if y >= inner.y + inner.height {
        break;
      }

      let is_selected = i == self.state.selected;
      let is_expanded = self.state.expanded_session == Some(i);

      self.render_session_card(session, timeline_x, y, inner.width - 2, is_selected, is_expanded, buf);
    }
  }

  #[allow(clippy::too_many_arguments)]
  fn render_session_card(
    &self,
    session: &Value,
    x: u16,
    y: u16,
    width: u16,
    selected: bool,
    expanded: bool,
    buf: &mut Buffer,
  ) {
    let summary = session.get("summary").and_then(|s| s.as_str()).unwrap_or("Session");
    let started_at = session
      .get("started_at")
      .and_then(|s| s.as_str())
      .and_then(parse_time_ago)
      .unwrap_or_default();
    let ended_at = session.get("ended_at").and_then(|e| e.as_str());

    let bg = if selected { Theme::SURFACE } else { Theme::BG };
    let fg = if selected { Theme::TEXT } else { Theme::SUBTEXT };

    // Clear line with background
    for i in 0..width {
      buf[(x + i, y)].set_bg(bg);
    }

    // Timeline marker
    let marker = if selected {
      if expanded { "▼ " } else { "▶ " }
    } else {
      "○ "
    };
    let marker_color = if ended_at.is_some() {
      Theme::MUTED
    } else {
      Theme::SUCCESS
    };
    buf.set_string(x, y, marker, Style::default().fg(marker_color));

    // Time ago
    let time_str = format!("{:>8}", started_at);
    buf.set_string(x + 2, y, &time_str, Style::default().fg(Theme::MUTED));

    // Status indicator
    let status = if ended_at.is_some() { "●" } else { "◐" };
    let status_color = if ended_at.is_some() {
      Theme::MUTED
    } else {
      Theme::SUCCESS
    };
    buf.set_string(x + 11, y, status, Style::default().fg(status_color));

    // Summary
    let summary_start = x + 13;
    let summary_width = width.saturating_sub(summary_start - x) as usize;
    let display_summary = if summary.len() > summary_width {
      format!("{}...", &summary[..summary_width.saturating_sub(3)])
    } else {
      summary.to_string()
    };
    buf.set_string(summary_start, y, &display_summary, Style::default().fg(fg));
  }

  fn render_detail(&self, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
      .title("SESSION DETAIL")
      .title_style(Style::default().fg(Theme::ACCENT).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::OVERLAY));

    let inner = block.inner(area);
    block.render(area, buf);

    let Some(session) = self.state.selected_session() else {
      buf.set_string(
        inner.x,
        inner.y,
        "Select a session to view details",
        Style::default().fg(Theme::MUTED),
      );
      return;
    };

    let mut y = inner.y;

    // ID
    if let Some(id) = session.get("id").and_then(|i| i.as_str()) {
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

    // Started at
    if let Some(started) = session.get("started_at").and_then(|s| s.as_str()) {
      buf.set_string(inner.x, y, "Started: ", Style::default().fg(Theme::SUBTEXT));
      let date = parse_date_friendly(started).unwrap_or_else(|| started.to_string());
      buf.set_string(inner.x + 9, y, &date, Style::default().fg(Theme::TEXT));
      y += 1;
    }

    // Ended at
    if let Some(ended) = session.get("ended_at").and_then(|e| e.as_str()) {
      buf.set_string(inner.x, y, "Ended: ", Style::default().fg(Theme::SUBTEXT));
      let date = parse_date_friendly(ended).unwrap_or_else(|| ended.to_string());
      buf.set_string(inner.x + 7, y, &date, Style::default().fg(Theme::TEXT));
      y += 1;
    } else {
      buf.set_string(inner.x, y, "Status: ", Style::default().fg(Theme::SUBTEXT));
      buf.set_string(inner.x + 8, y, "● Active", Style::default().fg(Theme::SUCCESS));
      y += 1;
    }

    // Summary
    if let Some(summary) = session.get("summary").and_then(|s| s.as_str()) {
      y += 1;
      buf.set_string(inner.x, y, "SUMMARY", Style::default().fg(Theme::ACCENT).bold());
      y += 1;

      for line in summary.lines() {
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

    // User prompt
    if let Some(prompt) = session.get("user_prompt").and_then(|p| p.as_str()) {
      y += 1;
      if y < inner.y + inner.height {
        buf.set_string(inner.x, y, "USER PROMPT", Style::default().fg(Theme::ACCENT).bold());
        y += 1;
      }

      let max_len = inner.width as usize;
      let prompt_display = if prompt.len() > max_len * 2 {
        format!("{}...", &prompt[..max_len * 2 - 3])
      } else {
        prompt.to_string()
      };

      for line in prompt_display.lines().take(2) {
        if y >= inner.y + inner.height {
          break;
        }
        buf.set_string(inner.x, y, line, Style::default().fg(Theme::SUBTEXT));
        y += 1;
      }
    }

    // Memories created/recalled
    y += 1;
    if y + 2 < inner.y + inner.height {
      buf.set_string(
        inner.x,
        y,
        "SESSION MEMORIES",
        Style::default().fg(Theme::ACCENT).bold(),
      );
      y += 1;

      if self.state.session_memories.is_empty() {
        buf.set_string(
          inner.x,
          y,
          "No memories in this session",
          Style::default().fg(Theme::MUTED),
        );
      } else {
        for memory in self.state.session_memories.iter().take(5) {
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

        if self.state.session_memories.len() > 5 && y < inner.y + inner.height {
          let more = format!("... {} more", self.state.session_memories.len() - 5);
          buf.set_string(inner.x + 2, y, &more, Style::default().fg(Theme::MUTED));
        }
      }
    }
  }
}

fn parse_time_ago(s: &str) -> Option<String> {
  // This is a simplified version - in a real app, you'd calculate from now
  // For now, just return a shortened date
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
