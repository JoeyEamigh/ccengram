use ratatui::style::Color;

/// Catppuccin-inspired theme colors
pub struct Theme;

impl Theme {
  // Base colors
  pub const BG: Color = Color::Rgb(30, 30, 46); // Base
  pub const SURFACE: Color = Color::Rgb(49, 50, 68); // Surface0
  pub const OVERLAY: Color = Color::Rgb(69, 71, 90); // Surface1
  pub const TEXT: Color = Color::Rgb(205, 214, 244); // Text
  pub const SUBTEXT: Color = Color::Rgb(166, 173, 200); // Subtext0
  pub const MUTED: Color = Color::Rgb(108, 112, 134); // Overlay1

  // Accent colors
  pub const ACCENT: Color = Color::Rgb(137, 180, 250); // Blue
  pub const SUCCESS: Color = Color::Rgb(166, 227, 161); // Green
  pub const WARNING: Color = Color::Rgb(249, 226, 175); // Yellow
  pub const ERROR: Color = Color::Rgb(243, 139, 168); // Red
  pub const INFO: Color = Color::Rgb(148, 226, 213); // Teal

  // Sector colors (memory types)
  pub const SEMANTIC: Color = Color::Rgb(137, 180, 250); // Blue
  pub const EPISODIC: Color = Color::Rgb(203, 166, 247); // Purple/Mauve
  pub const PROCEDURAL: Color = Color::Rgb(166, 227, 161); // Green
  pub const EMOTIONAL: Color = Color::Rgb(243, 139, 168); // Red
  pub const REFLECTIVE: Color = Color::Rgb(249, 226, 175); // Yellow

  /// Get color for a memory sector
  pub fn sector_color(sector: &str) -> Color {
    match sector.to_lowercase().as_str() {
      "semantic" => Self::SEMANTIC,
      "episodic" => Self::EPISODIC,
      "procedural" => Self::PROCEDURAL,
      "emotional" => Self::EMOTIONAL,
      "reflective" => Self::REFLECTIVE,
      _ => Self::TEXT,
    }
  }

  /// Get color for salience level (0.0 - 1.0)
  pub fn salience_color(salience: f32) -> Color {
    if salience >= 0.7 {
      Self::SUCCESS // Green for high
    } else if salience >= 0.4 {
      Self::WARNING // Yellow for medium
    } else if salience >= 0.2 {
      Color::Rgb(250, 179, 135) // Orange/Peach for low
    } else {
      Self::ERROR // Red for very low
    }
  }

  /// Get color for health status
  pub fn health_color(healthy: bool) -> Color {
    if healthy { Self::SUCCESS } else { Self::ERROR }
  }

  /// Get color for language type
  pub fn language_color(language: &str) -> Color {
    match language.to_lowercase().as_str() {
      "rust" => Color::Rgb(250, 179, 135),                      // Peach
      "python" => Color::Rgb(249, 226, 175),                    // Yellow
      "typescript" | "javascript" => Color::Rgb(249, 226, 175), // Yellow
      "go" => Color::Rgb(148, 226, 213),                        // Teal
      "java" | "kotlin" => Color::Rgb(250, 179, 135),           // Peach
      "c" | "cpp" | "c++" => Color::Rgb(137, 180, 250),         // Blue
      "ruby" => Color::Rgb(243, 139, 168),                      // Red
      "shell" | "bash" | "sh" => Color::Rgb(166, 227, 161),     // Green
      "markdown" | "md" => Color::Rgb(203, 166, 247),           // Mauve
      "json" | "yaml" | "toml" => Color::Rgb(148, 226, 213),    // Teal
      _ => Self::TEXT,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_sector_colors() {
    assert_eq!(Theme::sector_color("semantic"), Theme::SEMANTIC);
    assert_eq!(Theme::sector_color("SEMANTIC"), Theme::SEMANTIC);
    assert_eq!(Theme::sector_color("unknown"), Theme::TEXT);
  }

  #[test]
  fn test_salience_colors() {
    assert_eq!(Theme::salience_color(0.9), Theme::SUCCESS);
    assert_eq!(Theme::salience_color(0.5), Theme::WARNING);
    assert_eq!(Theme::salience_color(0.1), Theme::ERROR);
  }
}
