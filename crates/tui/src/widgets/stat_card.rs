use crate::theme::Theme;
use ratatui::{
  buffer::Buffer,
  layout::Rect,
  style::{Style, Stylize},
  widgets::{Block, Borders, Widget},
};

/// A bordered card displaying a statistic
/// Example:
/// ┌─MEMORIES──────┐
/// │ Total: 245    │
/// │ Semantic: 89  │
/// │ ████████░░72% │
/// └───────────────┘
pub struct StatCard<'a> {
  title: &'a str,
  lines: Vec<(&'a str, String)>,
  bar: Option<(f32, &'a str)>,
  width: u16,
}

impl<'a> StatCard<'a> {
  pub fn new(title: &'a str) -> Self {
    Self {
      title,
      lines: Vec::new(),
      bar: None,
      width: 15,
    }
  }

  pub fn line(mut self, label: &'a str, value: impl ToString) -> Self {
    self.lines.push((label, value.to_string()));
    self
  }

  pub fn bar(mut self, value: f32, label: &'a str) -> Self {
    self.bar = Some((value, label));
    self
  }

  pub fn width(mut self, width: u16) -> Self {
    self.width = width;
    self
  }
}

impl Widget for StatCard<'_> {
  fn render(self, area: Rect, buf: &mut Buffer) {
    if area.width < 5 || area.height < 3 {
      return;
    }

    // Create the bordered block
    let block = Block::default()
      .title(self.title)
      .title_style(Style::default().fg(Theme::ACCENT).bold())
      .borders(Borders::ALL)
      .border_style(Style::default().fg(Theme::OVERLAY));

    let inner = block.inner(area);
    block.render(area, buf);

    // Render lines
    let mut y = inner.y;
    for (label, value) in &self.lines {
      if y >= inner.y + inner.height {
        break;
      }

      let text = format!("{}: {}", label, value);
      let text = if text.len() > inner.width as usize {
        format!("{}...", &text[..inner.width.saturating_sub(3) as usize])
      } else {
        text
      };

      buf.set_string(inner.x, y, &text, Style::default().fg(Theme::TEXT));
      y += 1;
    }

    // Render bar if present
    if let Some((value, label)) = self.bar
      && y < inner.y + inner.height
    {
      let bar_width = inner.width.saturating_sub(label.len() as u16 + 1);
      let filled = ((value * bar_width as f32).round() as u16).min(bar_width);
      let empty = bar_width.saturating_sub(filled);

      let color = Theme::salience_color(value);

      // Label
      buf.set_string(inner.x, y, label, Style::default().fg(Theme::SUBTEXT));

      // Bar
      let bar_x = inner.x + label.len() as u16 + 1;
      let filled_str: String = "█".repeat(filled as usize);
      let empty_str: String = "░".repeat(empty as usize);

      buf.set_string(bar_x, y, &filled_str, Style::default().fg(color));
      buf.set_string(bar_x + filled, y, &empty_str, Style::default().fg(Theme::MUTED));
    }
  }
}

/// A simple status indicator
/// Example: "Daemon: ● Running" or "Daemon: ○ Stopped"
pub struct StatusIndicator<'a> {
  label: &'a str,
  status: bool,
  on_text: &'a str,
  off_text: &'a str,
}

impl<'a> StatusIndicator<'a> {
  pub fn new(label: &'a str, status: bool) -> Self {
    Self {
      label,
      status,
      on_text: "Running",
      off_text: "Stopped",
    }
  }

  pub fn texts(mut self, on: &'a str, off: &'a str) -> Self {
    self.on_text = on;
    self.off_text = off;
    self
  }
}

impl Widget for StatusIndicator<'_> {
  fn render(self, area: Rect, buf: &mut Buffer) {
    if area.width == 0 || area.height == 0 {
      return;
    }

    let (indicator, text, color) = if self.status {
      ("●", self.on_text, Theme::SUCCESS)
    } else {
      ("○", self.off_text, Theme::ERROR)
    };

    let full_text = format!("{}: {} {}", self.label, indicator, text);
    let full_text = if full_text.len() > area.width as usize {
      format!("{}...", &full_text[..area.width.saturating_sub(3) as usize])
    } else {
      full_text
    };

    // We need to render with multiple colors
    let label_end = self.label.len() + 2; // "Label: "
    let indicator_end = label_end + 2; // "● "

    // Render label
    if area.width > 0 {
      let label_part = &full_text[..label_end.min(full_text.len())];
      buf.set_string(area.x, area.y, label_part, Style::default().fg(Theme::TEXT));
    }

    // Render indicator
    if area.width as usize > label_end {
      buf.set_string(area.x + label_end as u16, area.y, indicator, Style::default().fg(color));
    }

    // Render status text
    if area.width as usize > indicator_end {
      buf.set_string(area.x + indicator_end as u16, area.y, text, Style::default().fg(color));
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_stat_card_builder() {
    let card = StatCard::new("Test")
      .line("Total", 100)
      .line("Active", 50)
      .bar(0.5, "Progress");

    assert_eq!(card.title, "Test");
    assert_eq!(card.lines.len(), 2);
    assert!(card.bar.is_some());
  }
}
