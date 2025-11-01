use crate::repl::{EngineResponse, ReplEngine, ReplStats, ResponseKind};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
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

fn plain_line(text: impl Into<String>) -> Line<'static> {
    Line::from(text.into())
}

fn styled_line(text: impl Into<String>, style: Style) -> Line<'static> {
    Line::from(vec![Span::styled(text.into(), style)])
}

fn colored_line(text: impl Into<String>, color: Color) -> Line<'static> {
    styled_line(text, Style::default().fg(color))
}

pub fn run(initial_file: Option<&Path>) -> std::io::Result<()> {
    struct ReplState {
        input: String,
        history: Vec<Line<'static>>,
        should_quit: bool,
        engine: ReplEngine,
        last_sample: Option<Vec<Line<'static>>>,
    }

    impl ReplState {
        fn new(engine: ReplEngine) -> Self {
            Self {
                input: String::new(),
                history: Vec::new(),
                should_quit: false,
                engine,
                last_sample: None,
            }
        }

        fn push_history_line(&mut self, line: Line<'static>) {
            const MAX_LEN: usize = 200;
            self.history.push(line);
            if self.history.len() > MAX_LEN {
                let excess = self.history.len() - MAX_LEN;
                self.history.drain(0..excess);
            }
        }

        fn record_response(&mut self, response: EngineResponse) {
            let EngineResponse { kind, lines } = response;
            if lines.is_empty() {
                return;
            }

            if lines
                .first()
                .map(|line| line.starts_with("Sampled "))
                .unwrap_or(false)
            {
                self.last_sample = Some(
                    lines
                        .iter()
                        .map(|line| plain_line(line.clone()))
                        .collect::<Vec<_>>(),
                );
            }

            match kind {
                ResponseKind::Info => {
                    let mut iter = lines.into_iter();
                    if let Some(first) = iter.next() {
                        self.push_history_line(colored_line(first, Color::Cyan));
                    }
                    for line in iter {
                        self.push_history_line(plain_line(line));
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
                        self.push_history_line(colored_line(first, color));
                    }
                    for line in iter {
                        self.push_history_line(plain_line(line));
                    }
                }
                ResponseKind::Error => {
                    for line in lines {
                        self.push_history_line(colored_line(line, Color::Red));
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

            if let Some(sample) = &self.last_sample {
                lines.push(Line::default());
                lines.push(styled_line(
                    "Last :sample output",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ));
                const MAX_SAMPLE_LINES: usize = 12;
                for line in sample.iter().take(MAX_SAMPLE_LINES) {
                    lines.push(line.clone());
                }
                if sample.len() > MAX_SAMPLE_LINES {
                    lines.push(plain_line("  ..."));
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
            lines.push(plain_line("  :help / :h             Show help"));
            lines.push(plain_line("  :quit / :q             Exit"));

            lines
        }
    }

    enum PreloadEntry {
        Message(Line<'static>),
        Response(EngineResponse),
    }

    let mut engine = ReplEngine::new();
    let mut preload = vec![PreloadEntry::Message(styled_line(
        "Welcome to the ProcLog REPL! Type :help for commands.",
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    ))];

    if let Some(path) = initial_file {
        let display_name = path.to_string_lossy().to_string();
        match fs::read_to_string(path) {
            Ok(content) => {
                preload.push(PreloadEntry::Message(colored_line(
                    format!("Loading file: {}", display_name),
                    Color::Cyan,
                )));
                let response = engine.process_line(&content);
                preload.push(PreloadEntry::Response(response));
            }
            Err(err) => {
                preload.push(PreloadEntry::Message(colored_line(
                    format!("Failed to read '{}': {}", display_name, err),
                    Color::Red,
                )));
            }
        }
    }

    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.show_cursor()?;

    let mut state = ReplState::new(engine);
    for entry in preload {
        match entry {
            PreloadEntry::Message(line) => state.push_history_line(line),
            PreloadEntry::Response(response) => state.record_response(response),
        }
    }

    let res = loop {
        terminal.draw(|frame| {
            let size = frame.size();
            let layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(5), Constraint::Length(3)])
                .split(size);

            let main_area = layout[0];
            let input_area = layout[1];

            let top_layout = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
                .split(main_area);

            let history_area = top_layout[0];
            let info_area = top_layout[1];

            render_history(frame, history_area, &state.history);
            let info_lines = state.info_lines();
            render_info(frame, info_area, &info_lines);
            render_input(frame, input_area, &state.input);
        })?;

        if state.should_quit {
            break Ok(());
        }

        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        state.should_quit = true;
                    }
                    KeyCode::Char(ch) => {
                        if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                            state.input.push(ch);
                        }
                    }
                    KeyCode::Backspace => {
                        state.input.pop();
                    }
                    KeyCode::Esc => {
                        state.should_quit = true;
                    }
                    KeyCode::Enter => {
                        let trimmed = state.input.trim().to_string();
                        if !trimmed.is_empty() {
                            state.push_history_line(styled_line(
                                format!("> {}", trimmed),
                                Style::default()
                                    .fg(Color::White)
                                    .add_modifier(Modifier::BOLD),
                            ));
                            match trimmed.as_str() {
                                ":quit" | ":q" => {
                                    state.push_history_line(plain_line("Goodbye!"));
                                    state.should_quit = true;
                                }
                                ":help" | ":h" => {
                                    let help_lines = [
                                        styled_line(
                                            "Commands:",
                                            Style::default()
                                                .fg(Color::Cyan)
                                                .add_modifier(Modifier::BOLD),
                                        ),
                                        plain_line("  :help / :h  Show this help message"),
                                        plain_line("  :quit / :q  Exit the REPL"),
                                        plain_line(
                                            "  :sample [count] [seed]  Sample answer sets",
                                        ),
                                        plain_line(
                                            "Enter facts, rules, constraints, and #const declarations to extend the program.",
                                        ),
                                        plain_line(
                                            "Enter queries starting with '?-' to evaluate them.",
                                        ),
                                    ];
                                    for line in help_lines {
                                        state.push_history_line(line);
                                    }
                                }
                                _ => {
                                    let response = state.engine.process_line(trimmed.as_str());
                                    state.record_response(response);
                                }
                            }
                        }
                        state.input.clear();
                    }
                    _ => {}
                },
                Event::Resize(_, _) | Event::FocusLost | Event::FocusGained | Event::Mouse(_) => {}
                _ => {}
            }
        }
    };

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    res
}

fn render_history(frame: &mut Frame<'_>, area: Rect, history: &[Line<'static>]) {
    let inner_height = area.height.saturating_sub(2) as usize;
    let scroll_offset = if inner_height == 0 {
        0
    } else {
        history.len().saturating_sub(inner_height)
    };
    let scroll_y = scroll_offset.min(u16::MAX as usize) as u16;

    let paragraph = Paragraph::new(history.to_vec())
        .block(Block::default().title("History").borders(Borders::ALL))
        .wrap(Wrap { trim: false })
        .scroll((scroll_y, 0));
    frame.render_widget(paragraph, area);
}

fn render_info(frame: &mut Frame<'_>, area: Rect, lines: &[Line<'static>]) {
    let paragraph = Paragraph::new(lines.to_vec())
        .block(Block::default().title("Info").borders(Borders::ALL))
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn render_input(frame: &mut Frame<'_>, area: Rect, input: &str) {
    let paragraph = Paragraph::new(Line::from(format!("> {}", input)))
        .block(Block::default().title("Input").borders(Borders::ALL))
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);

    let cursor_x = area.x.saturating_add(2 + input.len() as u16);
    let cursor_y = area.y.saturating_add(1);
    frame.set_cursor(cursor_x, cursor_y);
}
