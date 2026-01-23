use crate::theme::Theme;
use ratatui::{buffer::Buffer, layout::Rect, style::Style, widgets::Widget};

/// A colored badge displaying a memory sector
/// Example: [Semantic] or [Episodic]
pub struct SectorBadge<'a> {
  sector: &'a str,
  short: bool,
}

impl<'a> SectorBadge<'a> {
  pub fn new(sector: &'a str) -> Self {
    Self { sector, short: false }
  }

  /// Use short form (first 3 characters)
  pub fn short(mut self) -> Self {
    self.short = true;
    self
  }

  fn display_text(&self) -> String {
    if self.short {
      let s = self.sector.to_lowercase();
      format!("[{}]", &s[..3.min(s.len())].to_uppercase())
    } else {
      format!("[{}]", self.capitalize_first())
    }
  }

  fn capitalize_first(&self) -> String {
    let mut chars = self.sector.chars();
    match chars.next() {
      None => String::new(),
      Some(first) => first.to_uppercase().collect::<String>() + chars.as_str().to_lowercase().as_str(),
    }
  }
}

impl Widget for SectorBadge<'_> {
  fn render(self, area: Rect, buf: &mut Buffer) {
    if area.width == 0 || area.height == 0 {
      return;
    }

    let color = Theme::sector_color(self.sector);
    let text = self.display_text();
    let text = if text.len() > area.width as usize {
      &text[..area.width as usize]
    } else {
      &text
    };

    buf.set_string(area.x, area.y, text, Style::default().fg(color).bold());
  }
}

/// A memory type badge
/// Example: [Preference] or [Gotcha]
pub struct TypeBadge<'a> {
  memory_type: &'a str,
}

impl<'a> TypeBadge<'a> {
  pub fn new(memory_type: &'a str) -> Self {
    Self { memory_type }
  }

  fn display_text(&self) -> String {
    let formatted = self.memory_type.replace('_', " ");
    let capitalized: String = formatted
      .split_whitespace()
      .map(|word| {
        let mut chars = word.chars();
        match chars.next() {
          None => String::new(),
          Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        }
      })
      .collect::<Vec<_>>()
      .join(" ");
    format!("({})", capitalized)
  }
}

impl Widget for TypeBadge<'_> {
  fn render(self, area: Rect, buf: &mut Buffer) {
    if area.width == 0 || area.height == 0 {
      return;
    }

    let text = self.display_text();
    let text = if text.len() > area.width as usize {
      &text[..area.width as usize]
    } else {
      &text
    };

    buf.set_string(area.x, area.y, text, Style::default().fg(Theme::SUBTEXT));
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_sector_badge_display() {
    let badge = SectorBadge::new("semantic");
    assert_eq!(badge.display_text(), "[Semantic]");
  }

  #[test]
  fn test_sector_badge_short() {
    let badge = SectorBadge::new("semantic").short();
    assert_eq!(badge.display_text(), "[SEM]");
  }

  #[test]
  fn test_type_badge_display() {
    let badge = TypeBadge::new("turn_summary");
    assert_eq!(badge.display_text(), "(Turn Summary)");
  }
}
