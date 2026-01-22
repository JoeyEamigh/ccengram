use crossterm::event::{Event as CrosstermEvent, KeyCode, KeyEvent, KeyModifiers};

/// Input events for the TUI
#[derive(Debug, Clone)]
pub enum Event {
  /// Terminal key press
  Key(KeyEvent),
  /// Terminal resize
  Resize(u16, u16),
  /// Timer tick for refresh
  Tick,
}

/// Actions that can be performed in the TUI
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
  /// Quit the application
  Quit,
  /// Switch to a specific view
  SwitchView(usize),
  /// Navigate up in a list
  NavigateUp,
  /// Navigate down in a list
  NavigateDown,
  /// Navigate left (e.g., collapse tree node)
  NavigateLeft,
  /// Navigate right (e.g., expand tree node)
  NavigateRight,
  /// Select/Enter on current item
  Select,
  /// Go back (Escape)
  Back,
  /// Open search
  OpenSearch,
  /// Open filter
  OpenFilter,
  /// Toggle help overlay
  ToggleHelp,
  /// Reinforce selected memory
  Reinforce,
  /// Deemphasize selected memory
  Deemphasize,
  /// Cycle sort order
  CycleSort,
  /// Submit search/filter input
  Submit,
  /// Character input for search/filter
  Input(char),
  /// Delete character in input
  DeleteChar,
  /// Page up
  PageUp,
  /// Page down
  PageDown,
  /// Go to top of list
  GoToTop,
  /// Go to bottom of list
  GoToBottom,
  /// Tab to next panel
  NextPanel,
  /// Refresh current view
  Refresh,
  /// No action
  None,
}

impl From<CrosstermEvent> for Event {
  fn from(event: CrosstermEvent) -> Self {
    match event {
      CrosstermEvent::Key(key) => Event::Key(key),
      CrosstermEvent::Resize(w, h) => Event::Resize(w, h),
      _ => Event::Tick,
    }
  }
}

/// Convert a key event to an action based on the current input mode
pub fn key_to_action(key: KeyEvent, in_input_mode: bool) -> Action {
  if in_input_mode {
    // In input mode, most keys are text input
    match key.code {
      KeyCode::Esc => Action::Back,
      KeyCode::Enter => Action::Submit,
      KeyCode::Backspace => Action::DeleteChar,
      KeyCode::Char(c) => Action::Input(c),
      _ => Action::None,
    }
  } else {
    // Normal mode keybindings (vim-style)
    match key.code {
      // Quit
      KeyCode::Char('q') => Action::Quit,
      KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => Action::Quit,

      // View switching (1-7)
      KeyCode::Char('1') => Action::SwitchView(0),
      KeyCode::Char('2') => Action::SwitchView(1),
      KeyCode::Char('3') => Action::SwitchView(2),
      KeyCode::Char('4') => Action::SwitchView(3),
      KeyCode::Char('5') => Action::SwitchView(4),
      KeyCode::Char('6') => Action::SwitchView(5),
      KeyCode::Char('7') => Action::SwitchView(6),

      // Navigation
      KeyCode::Char('j') | KeyCode::Down => Action::NavigateDown,
      KeyCode::Char('k') | KeyCode::Up => Action::NavigateUp,
      KeyCode::Char('h') | KeyCode::Left => Action::NavigateLeft,
      KeyCode::Char('l') | KeyCode::Right => Action::NavigateRight,
      KeyCode::Enter => Action::Select,
      KeyCode::Esc => Action::Back,
      KeyCode::Tab => Action::NextPanel,

      // Page navigation
      KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => Action::PageDown,
      KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => Action::PageUp,
      KeyCode::PageDown => Action::PageDown,
      KeyCode::PageUp => Action::PageUp,
      KeyCode::Char('g') => Action::GoToTop,
      KeyCode::Char('G') => Action::GoToBottom,

      // Actions
      KeyCode::Char('/') => Action::OpenSearch,
      KeyCode::Char('f') => Action::OpenFilter,
      KeyCode::Char('s') => Action::CycleSort,
      KeyCode::Char('?') => Action::ToggleHelp,
      KeyCode::Char('r') => Action::Reinforce,
      KeyCode::Char('d') => Action::Deemphasize,
      KeyCode::Char('R') => Action::Refresh,

      _ => Action::None,
    }
  }
}
