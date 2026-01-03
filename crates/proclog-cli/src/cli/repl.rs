use crate::repl::{EngineResponse, ReplEngine, ReplStats, ResponseKind};
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
#[cfg(test)]
use ratatui::backend::TestBackend;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    prelude::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use std::{fs, io::stdout, path::Path, time::Duration};

#[derive(Clone)]
struct HistoryEntry {
    text: String,
    style: Style,
}

impl HistoryEntry {
    fn new(text: impl Into<String>, style: Style) -> Self {
        Self {
            text: text.into(),
            style,
        }
    }

    fn to_line(&self) -> Line<'static> {
        Line::from(Span::styled(self.text.clone(), self.style))
    }
}

fn plain_line(text: impl Into<String>) -> Line<'static> {
    Line::from(Span::raw(text.into()))
}

fn styled_line(text: impl Into<String>, style: Style) -> Line<'static> {
    Line::from(Span::styled(text.into(), style))
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ActivePane {
    History,
    Database,
    Stats,
    Input,
}

#[derive(Copy, Clone, Debug)]
enum FocusDirection {
    Left,
    Right,
    Up,
    Down,
}

pub(crate) struct ReplApp {
    engine: ReplEngine,
    input: String,
    history: Vec<HistoryEntry>,
    history_scroll: usize,
    follow_history: bool,
    db_scroll: usize,
    db_visible_lines: usize,
    db_total_lines: usize,
    last_sample: Option<Vec<String>>,
    active_pane: ActivePane,
    pending_focus: bool,
    should_quit: bool,
}

impl ReplApp {
    pub fn new(engine: ReplEngine) -> Self {
        Self {
            engine,
            input: String::new(),
            history: Vec::new(),
            history_scroll: 0,
            follow_history: true,
            db_scroll: 0,
            db_visible_lines: 1,
            db_total_lines: 0,
            last_sample: None,
            active_pane: ActivePane::Input,
            pending_focus: false,
            should_quit: false,
        }
    }

    fn push_history_entry(&mut self, entry: HistoryEntry) {
        const MAX_HISTORY: usize = 200;
        self.history.push(entry);
        if self.history.len() > MAX_HISTORY {
            let excess = self.history.len() - MAX_HISTORY;
            self.history.drain(0..excess);
        }
        self.follow_history = true;
    }

    fn push_history_plain(&mut self, text: impl Into<String>) {
        self.push_history_entry(HistoryEntry::new(text, Style::default()));
    }

    fn push_history_colored(&mut self, text: impl Into<String>, color: Color) {
        self.push_history_entry(HistoryEntry::new(text, Style::default().fg(color)));
    }

    fn push_history_bold(&mut self, text: impl Into<String>, color: Color) {
        self.push_history_entry(HistoryEntry::new(
            text,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ));
    }

    fn record_response(&mut self, response: EngineResponse) {
        let EngineResponse { kind, lines } = response;
        if lines.is_empty() {
            return;
        }

        if let Some(first) = lines.first() {
            if first.starts_with("Sampled ") {
                self.last_sample = Some(lines.clone());
            }
        }

        match kind {
            ResponseKind::Info => {
                let mut iter = lines.into_iter();
                if let Some(first) = iter.next() {
                    self.push_history_colored(first, Color::Cyan);
                }
                for line in iter {
                    self.push_history_plain(line);
                }
            }
            ResponseKind::Output => {
                let mut iter = lines.into_iter();
                if let Some(first) = iter.next() {
                    let color = if first == "true." {
                        Color::Green
                    } else if first == "false." || first.starts_with("No results.") {
                        Color::Red
                    } else {
                        Color::Yellow
                    };
                    self.push_history_colored(first, color);
                }
                for line in iter {
                    self.push_history_plain(line);
                }
            }
            ResponseKind::Error => {
                for line in lines {
                    self.push_history_colored(line, Color::Red);
                }
            }
        }
    }

    fn info_lines(&self) -> Vec<Line<'static>> {
        let stats: ReplStats = self.engine.stats();
        let mut lines = Vec::new();

        lines.push(styled_line(
            "Program",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
        lines.push(plain_line(format!("  Facts: {}", stats.fact_count)));
        lines.push(plain_line(format!("  Rules: {}", stats.rule_count)));
        lines.push(plain_line(format!(
            "  Constraints: {}",
            stats.constraint_count
        )));
        lines.push(plain_line(format!(
            "  Choice rules: {}",
            stats.choice_rule_count
        )));
        lines.push(plain_line(format!("  Constants: {}", stats.constant_count)));

        lines.push(Line::default());
        lines.push(styled_line(
            "Solver",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
        lines.push(plain_line(format!(
            "  Current: {}",
            if stats.use_sat_solver {
                "sat"
            } else {
                "native"
            }
        )));

        lines.push(Line::default());
        lines.push(styled_line(
            "Sampling",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
        lines.push(plain_line(format!(
            "  Default samples: {}",
            stats.sample_count
        )));
        lines.push(plain_line(format!("  Default seed: {}", stats.sample_seed)));
        let cache_line = match stats.cached_answer_sets {
            Some(count) => format!("  Cached answer sets: {}", count),
            None => "  Cached answer sets: n/a".to_string(),
        };
        lines.push(plain_line(cache_line));

        if let Some(sample_lines) = &self.last_sample {
            lines.push(Line::default());
            lines.push(styled_line(
                "Last :sample output",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ));
            const MAX_SAMPLE_LINES: usize = 12;
            for line in sample_lines.iter().take(MAX_SAMPLE_LINES) {
                lines.push(plain_line(line.clone()));
            }
            if sample_lines.len() > MAX_SAMPLE_LINES {
                lines.push(plain_line(format!(
                    "  ... {} more line(s)",
                    sample_lines.len() - MAX_SAMPLE_LINES
                )));
            }
        }

        lines.push(Line::default());
        lines.push(styled_line(
            "Commands",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
        lines.push(plain_line("  :sample [count] [seed]  Sample answer sets"));
        lines.push(plain_line("  :solver <native|sat>   Switch solver backend"));
        lines.push(plain_line("  :help / :h             Show help"));
        lines.push(plain_line("  :quit / :q             Exit"));

        lines
    }

    fn database_lines(&mut self) -> Vec<Line<'static>> {
        let lines = self.engine.database_view();
        if lines.is_empty() {
            vec![plain_line("(no facts)")]
        } else {
            lines.into_iter().map(plain_line).collect()
        }
    }

    pub fn draw(&mut self, frame: &mut Frame<'_>) {
        let size = frame.size();
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(5), Constraint::Length(3)])
            .split(size);

        let main_area = layout[0];
        let input_area = layout[1];

        let main_split = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(main_area);

        let history_area = main_split[0];
        let right_split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(main_split[1]);

        let database_area = right_split[0];
        let stats_area = right_split[1];

        // History rendering
        let history_lines: Vec<Line<'static>> =
            self.history.iter().map(HistoryEntry::to_line).collect();
        let history_visible = history_area.height.saturating_sub(2) as usize;
        let max_history_scroll = history_lines.len().saturating_sub(history_visible.max(1));
        if self.follow_history {
            self.history_scroll = max_history_scroll;
        } else {
            self.history_scroll = self.history_scroll.min(max_history_scroll);
        }
        render_history(
            frame,
            history_area,
            &history_lines,
            self.history_scroll,
            self.active_pane == ActivePane::History,
        );

        // Database rendering
        let database_lines = self.database_lines();
        let db_visible = database_area.height.saturating_sub(2) as usize;
        self.db_visible_lines = db_visible.max(1);
        self.db_total_lines = database_lines.len();
        let max_db_scroll = self
            .db_total_lines
            .saturating_sub(self.db_visible_lines.max(1));
        self.db_scroll = self.db_scroll.min(max_db_scroll);
        render_database(
            frame,
            database_area,
            &database_lines,
            self.db_scroll,
            self.active_pane == ActivePane::Database,
        );

        // Stats rendering
        let stats_lines = self.info_lines();
        render_stats(
            frame,
            stats_area,
            &stats_lines,
            self.active_pane == ActivePane::Stats,
        );

        render_input(
            frame,
            input_area,
            &self.input,
            self.active_pane == ActivePane::Input,
        );
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        if self.pending_focus {
            if let Some(direction) = focus_direction_from_key(key) {
                self.move_focus(direction);
                self.pending_focus = false;
                return;
            } else if key.code == KeyCode::Char('w')
                && key.modifiers.contains(KeyModifiers::CONTROL)
            {
                // Allow repeating Ctrl+W
                self.pending_focus = true;
                return;
            } else {
                self.pending_focus = false;
            }
        }

        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.pending_focus = true;
            }
            KeyCode::Char(ch) => {
                if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                    if self.active_pane != ActivePane::Input {
                        self.active_pane = ActivePane::Input;
                    }
                    self.input.push(ch);
                }
            }
            KeyCode::Backspace => {
                if self.active_pane != ActivePane::Input {
                    self.active_pane = ActivePane::Input;
                }
                self.input.pop();
            }
            KeyCode::Enter => {
                self.submit_input();
            }
            KeyCode::Up => {
                if self.active_pane == ActivePane::Database && self.db_scroll > 0 {
                    self.db_scroll -= 1;
                }
            }
            KeyCode::Down => {
                if self.active_pane == ActivePane::Database {
                    let max_scroll = self
                        .db_total_lines
                        .saturating_sub(self.db_visible_lines.max(1));
                    if self.db_scroll < max_scroll {
                        self.db_scroll += 1;
                    }
                }
            }
            KeyCode::Esc => {
                self.should_quit = true;
            }
            _ => {}
        }
    }

    fn submit_input(&mut self) {
        let trimmed = self.input.trim().to_string();
        if trimmed.is_empty() {
            self.input.clear();
            return;
        }

        self.push_history_bold(format!("> {}", trimmed), Color::White);
        match trimmed.as_str() {
            ":quit" | ":q" => {
                self.push_history_plain("Goodbye!");
                self.should_quit = true;
            }
            ":help" | ":h" => {
                let help_items: &[(Style, &str)] = &[
                    (
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                        "Commands:",
                    ),
                    (Style::default(), "  :help / :h  Show this help message"),
                    (
                        Style::default(),
                        "  :sample [count] [seed]  Sample answer sets",
                    ),
                    (
                        Style::default(),
                        "  :solver <native|sat>  Switch solver backend",
                    ),
                    (Style::default(), "  :quit / :q  Exit the REPL"),
                    (
                        Style::default(),
                        "Enter facts, rules, constraints, and #const declarations to extend the program.",
                    ),
                    (
                        Style::default(),
                        "Queries start with '?-' and evaluate against the current program.",
                    ),
                ];
                for (style, text) in help_items {
                    self.push_history_entry(HistoryEntry::new(*text, *style));
                }
            }
            _ => {
                let response = self.engine.process_line(trimmed.as_str());
                self.record_response(response);
            }
        }

        self.input.clear();
        self.active_pane = ActivePane::Input;
        self.follow_history = true;
        self.pending_focus = false;
    }

    fn move_focus(&mut self, direction: FocusDirection) {
        use ActivePane::*;
        let new_focus = match (self.active_pane, direction) {
            (Input, FocusDirection::Left | FocusDirection::Up) => History,
            (Input, FocusDirection::Right) => Database,
            (History, FocusDirection::Right) => Database,
            (History, FocusDirection::Down) => Input,
            (Database, FocusDirection::Left) => History,
            (Database, FocusDirection::Down) => Stats,
            (Database, FocusDirection::Up) => History,
            (Stats, FocusDirection::Left) => History,
            (Stats, FocusDirection::Up) => Database,
            (Stats, FocusDirection::Down) => Input,
            (pane, _) => pane,
        };

        if new_focus != self.active_pane {
            self.active_pane = new_focus;
            if new_focus == ActivePane::History {
                self.follow_history = false;
            }
        }
    }

    pub fn should_quit(&self) -> bool {
        self.should_quit
    }

    #[cfg(test)]
    pub(crate) fn active_pane(&self) -> ActivePane {
        self.active_pane
    }

    #[cfg(test)]
    pub(crate) fn db_scroll(&self) -> usize {
        self.db_scroll
    }
}

fn focus_direction_from_key(key: KeyEvent) -> Option<FocusDirection> {
    match key.code {
        KeyCode::Char('h') => Some(FocusDirection::Left),
        KeyCode::Char('j') => Some(FocusDirection::Down),
        KeyCode::Char('k') => Some(FocusDirection::Up),
        KeyCode::Char('l') => Some(FocusDirection::Right),
        KeyCode::Left => Some(FocusDirection::Left),
        KeyCode::Down => Some(FocusDirection::Down),
        KeyCode::Up => Some(FocusDirection::Up),
        KeyCode::Right => Some(FocusDirection::Right),
        _ => None,
    }
}

pub fn run(initial_file: Option<&Path>) -> std::io::Result<()> {
    let engine = ReplEngine::new();

    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.show_cursor()?;

    let mut app = ReplApp::new(engine);
    app.push_history_entry(HistoryEntry::new(
        "Welcome to the ProcLog REPL! Type :help for commands.",
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    ));

    if let Some(path) = initial_file {
        let display_name = path.to_string_lossy().to_string();
        match fs::read_to_string(path) {
            Ok(content) => {
                app.push_history_colored(format!("Loading file: {}", display_name), Color::Cyan);
                let response = app.engine.process_line(&content);
                app.record_response(response);
            }
            Err(err) => {
                app.push_history_colored(
                    format!("Failed to read '{}': {}", display_name, err),
                    Color::Red,
                );
            }
        }
    }

    let res = loop {
        terminal.draw(|f| app.draw(f))?;

        if app.should_quit() {
            break Ok(());
        }

        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => app.handle_key(key),
                Event::Resize(_, _) => {
                    app.follow_history = true;
                }
                Event::FocusLost | Event::FocusGained | Event::Mouse(_) => {}
                _ => {}
            }
        }
    };

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    res
}

fn render_history(
    frame: &mut Frame<'_>,
    area: Rect,
    lines: &[Line<'static>],
    scroll: usize,
    focused: bool,
) {
    let border_style = if focused {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default()
    };
    let paragraph = Paragraph::new(lines.to_vec())
        .block(
            Block::default()
                .title("History")
                .borders(Borders::ALL)
                .border_style(border_style),
        )
        .wrap(Wrap { trim: false })
        .scroll((scroll.min(u16::MAX as usize) as u16, 0));
    frame.render_widget(paragraph, area);
}

fn render_database(
    frame: &mut Frame<'_>,
    area: Rect,
    lines: &[Line<'static>],
    scroll: usize,
    focused: bool,
) {
    let border_style = if focused {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default()
    };
    let paragraph = Paragraph::new(lines.to_vec())
        .block(
            Block::default()
                .title("Database")
                .borders(Borders::ALL)
                .border_style(border_style),
        )
        .wrap(Wrap { trim: false })
        .scroll((scroll.min(u16::MAX as usize) as u16, 0));
    frame.render_widget(paragraph, area);
}

fn render_stats(frame: &mut Frame<'_>, area: Rect, lines: &[Line<'static>], focused: bool) {
    let border_style = if focused {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default()
    };
    let paragraph = Paragraph::new(lines.to_vec())
        .block(
            Block::default()
                .title("Stats")
                .borders(Borders::ALL)
                .border_style(border_style),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn render_input(frame: &mut Frame<'_>, area: Rect, input: &str, focused: bool) {
    let border_style = if focused {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default()
    };
    let paragraph = Paragraph::new(Line::from(format!("> {}", input)))
        .block(
            Block::default()
                .title("Input")
                .borders(Borders::ALL)
                .border_style(border_style),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);

    let cursor_x = area.x.saturating_add(2 + input.len() as u16);
    let cursor_y = area.y.saturating_add(1);
    frame.set_cursor(cursor_x, cursor_y);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repl::ReplEngine;

    fn ctrl_w() -> KeyEvent {
        KeyEvent::new(KeyCode::Char('w'), KeyModifiers::CONTROL)
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn ctrl_w_moves_focus_between_panes() {
        let mut app = ReplApp::new(ReplEngine::new());
        app.handle_key(ctrl_w());
        app.handle_key(key(KeyCode::Char('h')));
        assert_eq!(app.active_pane(), ActivePane::History);

        app.handle_key(ctrl_w());
        app.handle_key(key(KeyCode::Char('l')));
        assert_eq!(app.active_pane(), ActivePane::Database);

        app.handle_key(ctrl_w());
        app.handle_key(key(KeyCode::Char('j')));
        assert_eq!(app.active_pane(), ActivePane::Stats);
    }

    #[test]
    fn database_view_scrolls_with_arrow_keys() {
        let mut app = ReplApp::new(ReplEngine::new());
        // Populate with many facts
        let response = app.engine.process_line("value(1..50).");
        app.record_response(response);

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.draw(f)).unwrap();

        app.handle_key(ctrl_w());
        app.handle_key(key(KeyCode::Char('l')));
        assert_eq!(app.active_pane(), ActivePane::Database);

        let initial_scroll = app.db_scroll();
        app.handle_key(key(KeyCode::Down));
        let after_scroll = app.db_scroll();
        assert!(
            after_scroll >= initial_scroll,
            "scroll should not decrease when pressing down"
        );
    }

    #[test]
    fn ctrl_w_with_arrow_keys_moves_focus() {
        let mut app = ReplApp::new(ReplEngine::new());
        app.handle_key(ctrl_w());
        app.handle_key(key(KeyCode::Right));
        assert_eq!(app.active_pane(), ActivePane::Database);

        app.handle_key(ctrl_w());
        app.handle_key(key(KeyCode::Down));
        assert_eq!(app.active_pane(), ActivePane::Stats);

        app.handle_key(ctrl_w());
        app.handle_key(key(KeyCode::Left));
        assert_eq!(app.active_pane(), ActivePane::History);
    }
}
