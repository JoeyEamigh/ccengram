pub mod code;
pub mod dashboard;
pub mod document;
pub mod entity;
pub mod memory;
pub mod search;
pub mod session;

pub use code::CodeView;
pub use dashboard::DashboardView;
pub use document::DocumentView;
pub use entity::EntityView;
pub use memory::MemoryView;
pub use search::SearchView;
pub use session::SessionView;

use ratatui::{buffer::Buffer, layout::Rect};

/// Trait for all views to implement
pub trait ViewWidget {
  fn render(&self, area: Rect, buf: &mut Buffer);
}
