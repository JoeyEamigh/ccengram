use crate::daemon_client::DaemonClient;
use crate::event::{Action, key_to_action};
use crate::theme::Theme;
use crate::views::{
  CodeView, DashboardView, DocumentView, EntityView, MemoryView, SearchView, SessionView,
  code::CodeState,
  dashboard::DashboardState,
  document::DocumentState,
  entity::EntityState,
  memory::MemoryState,
  search::{SearchResult, SearchResultType, SearchState},
  session::SessionState,
};
use anyhow::{Context, Result};
use crossterm::{
  event::{self, Event as CrosstermEvent, KeyEventKind},
  execute,
  terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use daemon::{default_socket_path, is_running};
use ratatui::{
  Terminal,
  backend::CrosstermBackend,
  buffer::Buffer,
  layout::{Constraint, Direction, Layout, Rect},
  style::{Style, Stylize},
  widgets::{Block, Borders, Clear, Widget},
};
use std::io;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::time::interval;
use tracing::{info, warn};

/// The current view being displayed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum View {
  #[default]
  Dashboard,
  Memory,
  Code,
  Document,
  Entity,
  Session,
  Search,
}

impl View {
  pub fn name(&self) -> &'static str {
    match self {
      View::Dashboard => "Dashboard",
      View::Memory => "Memories",
      View::Code => "Code",
      View::Document => "Docs",
      View::Entity => "Entities",
      View::Session => "Sessions",
      View::Search => "Search",
    }
  }

  pub fn index(&self) -> usize {
    match self {
      View::Dashboard => 0,
      View::Memory => 1,
      View::Code => 2,
      View::Document => 3,
      View::Entity => 4,
      View::Session => 5,
      View::Search => 6,
    }
  }

  pub fn from_index(index: usize) -> Self {
    match index {
      0 => View::Dashboard,
      1 => View::Memory,
      2 => View::Code,
      3 => View::Document,
      4 => View::Entity,
      5 => View::Session,
      6 => View::Search,
      _ => View::Dashboard,
    }
  }
}

/// Input mode for the application
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputMode {
  #[default]
  Normal,
  Search,
  Filter,
}

/// Main application state
pub struct App {
  pub current_view: View,
  pub daemon_client: Option<DaemonClient>,
  pub daemon_started_by_tui: bool,
  pub input_mode: InputMode,
  pub should_quit: bool,
  pub show_help: bool,
  pub project_path: PathBuf,

  // View states
  pub dashboard: DashboardState,
  pub memory: MemoryState,
  pub code: CodeState,
  pub document: DocumentState,
  pub entity: EntityState,
  pub session: SessionState,
  pub search: SearchState,
}

impl App {
  pub async fn new(project_path: PathBuf) -> Result<Self> {
    let socket_path = default_socket_path();
    let daemon_started_by_tui = if !is_running(&socket_path) {
      info!("Daemon not running, starting...");
      // Start daemon as background process
      let exe_path = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("ccengram"));
      Command::new(&exe_path)
        .arg("daemon")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to start daemon")?;

      // Wait for ready (poll up to 5 seconds)
      for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if is_running(&socket_path) {
          info!("Daemon started successfully");
          break;
        }
      }

      if !is_running(&socket_path) {
        warn!("Daemon did not start in time, continuing without connection");
      }
      true
    } else {
      info!("Connecting to existing daemon");
      false
    };

    let daemon_client = match DaemonClient::connect(project_path.clone()).await {
      Ok(client) => Some(client),
      Err(e) => {
        warn!("Failed to connect to daemon: {}", e);
        None
      }
    };

    Ok(Self {
      current_view: View::Dashboard,
      daemon_client,
      daemon_started_by_tui,
      input_mode: InputMode::Normal,
      should_quit: false,
      show_help: false,
      project_path,
      dashboard: DashboardState::new(),
      memory: MemoryState::new(),
      code: CodeState::new(),
      document: DocumentState::new(),
      entity: EntityState::new(),
      session: SessionState::new(),
      search: SearchState::new(),
    })
  }

  pub async fn refresh_current_view(&mut self) {
    let Some(ref mut client) = self.daemon_client else {
      return;
    };

    match self.current_view {
      View::Dashboard => {
        self.dashboard.loading = true;
        if let Ok(stats) = client.project_stats().await {
          self.dashboard.set_stats(stats);
        }
        if let Ok(health) = client.health_check().await {
          self.dashboard.set_health(health);
        }
        self.dashboard.loading = false;
      }
      View::Memory => {
        self.memory.loading = true;
        match client.memory_list(100, 0).await {
          Ok(memories) => {
            self.memory.set_memories(memories);
            self.memory.error = None;
          }
          Err(e) => {
            self.memory.error = Some(format!("{}", e));
          }
        }
        self.memory.loading = false;
      }
      View::Code => {
        self.code.loading = true;
        match client.code_list().await {
          Ok(chunks) => {
            self.code.set_chunks(chunks);
            self.code.error = None;
          }
          Err(e) => {
            self.code.error = Some(format!("{}", e));
          }
        }
        if let Ok(stats) = client.code_stats().await {
          self.code.set_stats(stats);
        }
        self.code.loading = false;
      }
      View::Document => {
        self.document.loading = true;
        // Documents are fetched via docs_search with empty query to list all
        match client.docs_search("", 100).await {
          Ok(docs) => {
            self.document.set_documents(docs);
            self.document.error = None;
          }
          Err(e) => {
            self.document.error = Some(format!("{}", e));
          }
        }
        self.document.loading = false;
      }
      View::Entity => {
        self.entity.loading = true;
        match client.entity_top(100).await {
          Ok(entities) => {
            self.entity.set_entities(entities);
            self.entity.error = None;
          }
          Err(e) => {
            self.entity.error = Some(format!("{}", e));
          }
        }
        self.entity.loading = false;
      }
      View::Session => {
        self.session.loading = true;
        match client.memory_timeline(50).await {
          Ok(timeline) => {
            // Timeline includes session info
            self.session.set_sessions(timeline);
            self.session.error = None;
          }
          Err(e) => {
            self.session.error = Some(format!("{}", e));
          }
        }
        self.session.loading = false;
      }
      View::Search => {
        // Search is triggered explicitly, not on refresh
      }
    }
  }

  pub async fn handle_action(&mut self, action: Action) {
    match action {
      Action::Quit => self.should_quit = true,
      Action::SwitchView(index) => {
        self.current_view = View::from_index(index);
        self.input_mode = InputMode::Normal;
        self.refresh_current_view().await;
      }
      Action::NavigateUp => self.navigate_up(),
      Action::NavigateDown => self.navigate_down(),
      Action::NavigateLeft => self.navigate_left(),
      Action::NavigateRight => self.navigate_right(),
      Action::Select => self.select().await,
      Action::Back => self.back(),
      Action::OpenSearch => self.open_search(),
      Action::OpenFilter => self.open_filter(),
      Action::ToggleHelp => self.show_help = !self.show_help,
      Action::Reinforce => self.reinforce().await,
      Action::Deemphasize => self.deemphasize().await,
      Action::Submit => self.submit().await,
      Action::Input(c) => self.input_char(c),
      Action::DeleteChar => self.delete_char(),
      Action::PageUp => self.page_up(),
      Action::PageDown => self.page_down(),
      Action::GoToTop => self.go_to_top(),
      Action::GoToBottom => self.go_to_bottom(),
      Action::NextPanel => self.next_panel(),
      Action::Refresh => self.refresh_current_view().await,
      Action::CycleSort => self.cycle_sort(),
      Action::None => {}
    }
  }

  fn navigate_up(&mut self) {
    match self.current_view {
      View::Memory => self.memory.select_prev(),
      View::Code => self.code.select_prev(),
      View::Document => self.document.select_prev(),
      View::Entity => self.entity.select_prev(),
      View::Session => self.session.select_prev(),
      View::Search => self.search.select_prev(),
      _ => {}
    }
  }

  fn navigate_down(&mut self) {
    match self.current_view {
      View::Memory => self.memory.select_next(),
      View::Code => self.code.select_next(),
      View::Document => self.document.select_next(),
      View::Entity => self.entity.select_next(),
      View::Session => self.session.select_next(),
      View::Search => self.search.select_next(),
      _ => {}
    }
  }

  fn navigate_left(&mut self) {
    match self.current_view {
      View::Memory => self.memory.scroll_detail_up(),
      View::Document => self.document.scroll_detail_up(),
      _ => {}
    }
  }

  fn navigate_right(&mut self) {
    match self.current_view {
      View::Memory => self.memory.scroll_detail_down(),
      View::Document => self.document.scroll_detail_down(),
      _ => {}
    }
  }

  async fn select(&mut self) {
    match self.current_view {
      View::Session => self.session.toggle_expand(),
      View::Search => {
        if self.search.input_active {
          self.execute_search().await;
        }
      }
      _ => {}
    }
  }

  fn back(&mut self) {
    match self.input_mode {
      InputMode::Search | InputMode::Filter => {
        self.input_mode = InputMode::Normal;
        if self.current_view == View::Search {
          self.search.input_active = false;
        }
      }
      InputMode::Normal => {
        if self.show_help {
          self.show_help = false;
        } else {
          self.current_view = View::Dashboard;
        }
      }
    }
  }

  fn open_search(&mut self) {
    self.input_mode = InputMode::Search;
    if self.current_view == View::Search {
      self.search.input_active = true;
    } else {
      // Switch to search view
      self.current_view = View::Search;
      self.search.input_active = true;
    }
  }

  fn open_filter(&mut self) {
    self.input_mode = InputMode::Filter;
  }

  async fn reinforce(&mut self) {
    if self.current_view != View::Memory {
      return;
    }
    let Some(id) = self.memory.selected_id() else {
      return;
    };
    let Some(ref mut client) = self.daemon_client else {
      return;
    };

    if let Err(e) = client.memory_reinforce(&id).await {
      self.memory.error = Some(format!("Reinforce failed: {}", e));
    } else {
      // Refresh to show updated salience
      self.refresh_current_view().await;
    }
  }

  async fn deemphasize(&mut self) {
    if self.current_view != View::Memory {
      return;
    }
    let Some(id) = self.memory.selected_id() else {
      return;
    };
    let Some(ref mut client) = self.daemon_client else {
      return;
    };

    if let Err(e) = client.memory_deemphasize(&id).await {
      self.memory.error = Some(format!("Deemphasize failed: {}", e));
    } else {
      self.refresh_current_view().await;
    }
  }

  async fn submit(&mut self) {
    match self.input_mode {
      InputMode::Search => {
        self.execute_search().await;
        self.input_mode = InputMode::Normal;
        self.search.input_active = false;
      }
      InputMode::Filter => {
        // Apply filter
        self.input_mode = InputMode::Normal;
      }
      InputMode::Normal => {}
    }
  }

  fn input_char(&mut self, c: char) {
    match self.input_mode {
      InputMode::Search => {
        if self.current_view == View::Search {
          self.search.query.push(c);
        } else {
          self.memory.search_query.push(c);
        }
      }
      InputMode::Filter => {
        // Add to filter string
      }
      InputMode::Normal => {}
    }
  }

  fn delete_char(&mut self) {
    match self.input_mode {
      InputMode::Search => {
        if self.current_view == View::Search {
          self.search.query.pop();
        } else {
          self.memory.search_query.pop();
        }
      }
      InputMode::Filter => {
        // Remove from filter string
      }
      InputMode::Normal => {}
    }
  }

  fn page_up(&mut self) {
    for _ in 0..10 {
      self.navigate_up();
    }
  }

  fn page_down(&mut self) {
    for _ in 0..10 {
      self.navigate_down();
    }
  }

  fn go_to_top(&mut self) {
    match self.current_view {
      View::Memory => self.memory.selected = 0,
      View::Code => self.code.selected = 0,
      View::Document => self.document.selected = 0,
      View::Entity => self.entity.selected = 0,
      View::Session => self.session.selected = 0,
      View::Search => self.search.selected = 0,
      _ => {}
    }
  }

  fn go_to_bottom(&mut self) {
    match self.current_view {
      View::Memory => {
        if !self.memory.memories.is_empty() {
          self.memory.selected = self.memory.memories.len() - 1;
        }
      }
      View::Code => {
        if !self.code.chunks.is_empty() {
          self.code.selected = self.code.chunks.len() - 1;
        }
      }
      View::Document => {
        if !self.document.documents.is_empty() {
          self.document.selected = self.document.documents.len() - 1;
        }
      }
      View::Entity => {
        if !self.entity.entities.is_empty() {
          self.entity.selected = self.entity.entities.len() - 1;
        }
      }
      View::Session => {
        if !self.session.sessions.is_empty() {
          self.session.selected = self.session.sessions.len() - 1;
        }
      }
      View::Search => {
        if !self.search.results.is_empty() {
          self.search.selected = self.search.results.len() - 1;
        }
      }
      _ => {}
    }
  }

  fn next_panel(&mut self) {
    // Cycle through views
    let next = (self.current_view.index() + 1) % 7;
    self.current_view = View::from_index(next);
  }

  fn cycle_sort(&mut self) {
    // Only memory view supports sorting currently
    if self.current_view == View::Memory {
      self.memory.cycle_sort();
    }
  }

  async fn execute_search(&mut self) {
    let Some(ref mut client) = self.daemon_client else {
      self.search.error = Some("Not connected to daemon".to_string());
      return;
    };

    if self.search.query.is_empty() {
      return;
    }

    self.search.loading = true;
    let mut results = Vec::new();

    // Search memories
    if self.search.search_memories
      && let Ok(memories) = client.memory_search(&self.search.query, 20).await
    {
      for memory in memories {
        let similarity = memory.get("similarity").and_then(|s| s.as_f64()).unwrap_or(0.0) as f32;
        results.push(SearchResult {
          result_type: SearchResultType::Memory,
          data: memory,
          similarity,
        });
      }
    }

    // Search code
    if self.search.search_code
      && let Ok(chunks) = client.code_search(&self.search.query, 20).await
    {
      for chunk in chunks {
        let similarity = chunk.get("similarity").and_then(|s| s.as_f64()).unwrap_or(0.0) as f32;
        results.push(SearchResult {
          result_type: SearchResultType::Code,
          data: chunk,
          similarity,
        });
      }
    }

    // Search documents
    if self.search.search_documents
      && let Ok(docs) = client.docs_search(&self.search.query, 20).await
    {
      for doc in docs {
        let similarity = doc.get("similarity").and_then(|s| s.as_f64()).unwrap_or(0.0) as f32;
        results.push(SearchResult {
          result_type: SearchResultType::Document,
          data: doc,
          similarity,
        });
      }
    }

    // Sort by similarity
    results.sort_by(|a, b| {
      b.similarity
        .partial_cmp(&a.similarity)
        .unwrap_or(std::cmp::Ordering::Equal)
    });

    self.search.set_results(results);
    self.search.loading = false;
  }

  pub async fn cleanup(&mut self) -> Result<()> {
    if self.daemon_started_by_tui {
      info!("TUI started daemon, sending shutdown request");
      if let Some(ref mut client) = self.daemon_client
        && let Err(e) = client.shutdown().await
      {
        warn!("Failed to shutdown daemon: {}", e);
      }
    }
    Ok(())
  }
}

/// Run the TUI application
pub async fn run(project_path: PathBuf) -> Result<()> {
  // Setup terminal
  enable_raw_mode()?;
  let mut stdout = io::stdout();
  execute!(stdout, EnterAlternateScreen)?;
  let backend = CrosstermBackend::new(stdout);
  let mut terminal = Terminal::new(backend)?;

  // Create app
  let mut app = App::new(project_path).await?;

  // Initial data load
  app.refresh_current_view().await;

  // Event loop
  let mut refresh_interval = interval(Duration::from_secs(30));

  loop {
    // Draw
    terminal.draw(|f| {
      render_app(&app, f.area(), f.buffer_mut());
    })?;

    // Handle events
    tokio::select! {
        _ = refresh_interval.tick() => {
            app.refresh_current_view().await;
        }
        result = tokio::task::spawn_blocking(|| {
            if event::poll(Duration::from_millis(100)).ok()? {
                event::read().ok()
            } else {
                None
            }
        }) => {
            if let Ok(Some(event)) = result {
                match event {
                    CrosstermEvent::Key(key) if key.kind == KeyEventKind::Press => {
                        let action = key_to_action(key, app.input_mode != InputMode::Normal);
                        app.handle_action(action).await;
                    }
                    CrosstermEvent::Resize(_, _) => {
                        // Terminal will redraw on next loop
                    }
                    _ => {}
                }
            }
        }
    }

    if app.should_quit {
      break;
    }
  }

  // Cleanup
  app.cleanup().await?;
  disable_raw_mode()?;
  execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

  Ok(())
}

/// Render the application
fn render_app(app: &App, area: Rect, buf: &mut Buffer) {
  // Clear with background
  Clear.render(area, buf);
  for y in area.y..area.y + area.height {
    for x in area.x..area.x + area.width {
      buf[(x, y)].set_bg(Theme::BG);
    }
  }

  // Layout: header + content + footer
  let chunks = Layout::default()
    .direction(Direction::Vertical)
    .constraints([
      Constraint::Length(2), // Header with tabs
      Constraint::Min(10),   // Content
      Constraint::Length(1), // Footer with keybindings
    ])
    .split(area);

  // Render header with tabs
  render_header(app, chunks[0], buf);

  // Render current view
  match app.current_view {
    View::Dashboard => DashboardView::new(&app.dashboard).render(chunks[1], buf),
    View::Memory => MemoryView::new(&app.memory).render(chunks[1], buf),
    View::Code => CodeView::new(&app.code).render(chunks[1], buf),
    View::Document => DocumentView::new(&app.document).render(chunks[1], buf),
    View::Entity => EntityView::new(&app.entity).render(chunks[1], buf),
    View::Session => SessionView::new(&app.session).render(chunks[1], buf),
    View::Search => SearchView::new(&app.search).render(chunks[1], buf),
  }

  // Render footer
  render_footer(app, chunks[2], buf);

  // Render help overlay if active
  if app.show_help {
    render_help_overlay(area, buf);
  }
}

fn render_header(app: &App, area: Rect, buf: &mut Buffer) {
  // Title
  let title = "CCEngram TUI";
  buf.set_string(area.x + 1, area.y, title, Style::default().fg(Theme::ACCENT).bold());

  // Tabs
  let tabs_x = area.x + title.len() as u16 + 3;
  let views = [
    View::Dashboard,
    View::Memory,
    View::Code,
    View::Document,
    View::Entity,
    View::Session,
    View::Search,
  ];

  let mut x = tabs_x;
  for (i, view) in views.iter().enumerate() {
    let is_selected = *view == app.current_view;
    let label = format!("[{}]{} ", i + 1, view.name());

    let style = if is_selected {
      Style::default().fg(Theme::ACCENT).bold()
    } else {
      Style::default().fg(Theme::SUBTEXT)
    };

    buf.set_string(x, area.y, &label, style);
    x += label.len() as u16;
  }

  // Connection status
  let conn_status = if app.daemon_client.is_some() {
    "● Connected"
  } else {
    "○ Disconnected"
  };
  let conn_color = if app.daemon_client.is_some() {
    Theme::SUCCESS
  } else {
    Theme::ERROR
  };
  let conn_x = area.x + area.width.saturating_sub(conn_status.len() as u16 + 2);
  buf.set_string(conn_x, area.y, conn_status, Style::default().fg(conn_color));

  // Separator
  for x in area.x..area.x + area.width {
    buf[(x, area.y + 1)].set_char('─').set_fg(Theme::OVERLAY);
  }
}

fn render_footer(app: &App, area: Rect, buf: &mut Buffer) {
  let keybindings = match app.input_mode {
    InputMode::Normal => {
      if app.current_view == View::Memory {
        "q:Quit  1-7:Views  j/k:Nav  /:Search  f:Filter  s:Sort  ?:Help  r/d:Salience"
      } else {
        "q:Quit  1-7:Views  j/k:Nav  /:Search  ?:Help  r:Reinforce  d:Deemphasize  R:Refresh"
      }
    }
    InputMode::Search => "Enter:Search  Esc:Cancel",
    InputMode::Filter => "Enter:Apply  Esc:Cancel",
  };

  buf.set_string(area.x + 1, area.y, keybindings, Style::default().fg(Theme::MUTED));

  // Project path on right
  let path_display = app.project_path.file_name().and_then(|n| n.to_str()).unwrap_or(".");
  let path_x = area.x + area.width.saturating_sub(path_display.len() as u16 + 2);
  buf.set_string(path_x, area.y, path_display, Style::default().fg(Theme::SUBTEXT));
}

fn render_help_overlay(area: Rect, buf: &mut Buffer) {
  // Center the help box
  let help_width = 50;
  let help_height = 18;
  let x = area.x + (area.width.saturating_sub(help_width)) / 2;
  let y = area.y + (area.height.saturating_sub(help_height)) / 2;

  let help_area = Rect::new(x, y, help_width, help_height);

  // Clear and draw border
  for hy in help_area.y..help_area.y + help_area.height {
    for hx in help_area.x..help_area.x + help_area.width {
      buf[(hx, hy)].set_bg(Theme::SURFACE).set_char(' ');
    }
  }

  let block = Block::default()
    .title("Help")
    .title_style(Style::default().fg(Theme::ACCENT).bold())
    .borders(Borders::ALL)
    .border_style(Style::default().fg(Theme::ACCENT));
  let inner = block.inner(help_area);
  block.render(help_area, buf);

  let help_text = [
    "NAVIGATION",
    "  1-7      Switch views",
    "  Tab      Cycle views",
    "  j/k      Navigate up/down",
    "  h/l      Scroll detail left/right",
    "  Enter    Select/expand",
    "  Esc      Back/cancel",
    "",
    "ACTIONS",
    "  /        Open search",
    "  f        Open filter",
    "  s        Cycle sort (Memory view)",
    "  r        Reinforce memory",
    "  d        Deemphasize memory",
    "  R        Refresh view",
    "  q        Quit",
    "  ?        Toggle help",
  ];

  for (i, line) in help_text.iter().enumerate() {
    if i as u16 >= inner.height {
      break;
    }
    let style = if line.starts_with(|c: char| c.is_uppercase()) {
      Style::default().fg(Theme::ACCENT).bold()
    } else {
      Style::default().fg(Theme::TEXT)
    };
    buf.set_string(inner.x, inner.y + i as u16, line, style);
  }
}
