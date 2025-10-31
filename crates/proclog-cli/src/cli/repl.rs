use crate::repl::{EngineResponse, ReplEngine, ResponseKind};
use crate::{COLOR_CYAN, COLOR_GREEN, COLOR_RED, COLOR_RESET, COLOR_YELLOW};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use crate::ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    text::Line,
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};
use std::{fs, io::stdout, path::Path, time::Duration};

pub fn run(initial_file: Option<&Path>) -> std::io::Result<()> {
    struct ReplState {
        input: String,
        history: Vec<String>,
        should_quit: bool,
        engine: ReplEngine,
    }

    impl ReplState {
        fn new(engine: ReplEngine) -> Self {
            Self {
                input: String::new(),
                history: Vec::new(),
                should_quit: false,
                engine,
            }
        }

        fn push_history(&mut self, line: impl Into<String>) {
            const MAX_LEN: usize = 200;
            self.history.push(line.into());
            if self.history.len() > MAX_LEN {
                let excess = self.history.len() - MAX_LEN;
                self.history.drain(0..excess);
            }
        }

        fn record_response(&mut self, response: EngineResponse) {
            if response.lines.is_empty() {
                return;
            }
            match response.kind {
                ResponseKind::Info => {
                    self.push_history(format!(
                        "{}{}{}",
                        COLOR_CYAN, response.lines[0], COLOR_RESET
                    ));
                    for line in response.lines.iter().skip(1) {
                        self.push_history(line.clone());
                    }
                }
                ResponseKind::Output => {
                    let first = response.lines[0].clone();
                    let color = if first == "true." {
                        COLOR_GREEN
                    } else if first == "false." || first.starts_with("No results.") {
                        COLOR_RED
                    } else {
                        COLOR_YELLOW
                    };
                    self.push_history(format!("{}{}{}", color, first, COLOR_RESET));
                    for line in response.lines.iter().skip(1) {
                        self.push_history(line.clone());
                    }
                }
                ResponseKind::Error => {
                    for line in response.lines {
                        self.push_history(format!("{}{}{}", COLOR_RED, line, COLOR_RESET));
                    }
                }
            }
        }
    }

    enum PreloadEntry {
        Message(String),
        Response(EngineResponse),
    }

    let mut engine = ReplEngine::new();
    let mut preload = vec![PreloadEntry::Message(format!(
        "{}Welcome to the ProcLog REPL! Type :help for commands.{}",
        COLOR_CYAN, COLOR_RESET
    ))];

    if let Some(path) = initial_file {
        let display_name = path.to_string_lossy().to_string();
        match fs::read_to_string(path) {
            Ok(content) => {
                preload.push(PreloadEntry::Message(format!(
                    "{}Loading file: {}{}",
                    COLOR_CYAN, display_name, COLOR_RESET
                )));
                let response = engine.process_line(&content);
                preload.push(PreloadEntry::Response(response));
            }
            Err(err) => {
                preload.push(PreloadEntry::Message(format!(
                    "{}Failed to read '{}': {}{}",
                    COLOR_RED, display_name, err, COLOR_RESET
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
            PreloadEntry::Message(line) => state.push_history(line),
            PreloadEntry::Response(response) => state.record_response(response),
        }
    }

    let res = loop {
        terminal.draw(|frame| {
            let size = frame.size();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(3), Constraint::Length(3)].as_ref())
                .split(size);

            let history_height = chunks[0].height.saturating_sub(2) as usize;
            let start_index = state.history.len().saturating_sub(history_height.max(1));
            let visible_history = state
                .history
                .iter()
                .skip(start_index)
                .take(history_height.max(1));
            let history_text = visible_history
                .map(|s| Line::from(s.as_str()))
                .collect::<Vec<_>>();

            let history = Paragraph::new(history_text)
                .block(Block::default().title("History").borders(Borders::ALL))
                .wrap(Wrap { trim: false });
            frame.render_widget(history, chunks[0]);

            let input_display = format!("> {}", state.input);
            let input_area = chunks[1];

            let input = Paragraph::new(vec![Line::from(input_display.as_str())])
                .wrap(Wrap { trim: false })
                .style(Style::default().bg(Color::Rgb(40, 40, 40)));
            frame.render_widget(input, input_area);

            let _ = frame.set_cursor(chunks[1].x + 2 + state.input.len() as u16, chunks[1].y + 1);
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
                            state.push_history(format!("> {}", trimmed));
                            match trimmed.as_str() {
                                ":quit" | ":q" => {
                                    state.push_history("Goodbye!");
                                    state.should_quit = true;
                                }
                                ":help" | ":h" => {
                                    let help_lines = [
                                        format!("{}Commands:{}", COLOR_CYAN, COLOR_RESET),
                                        "  :help / :h  Show this help message".to_string(),
                                        "  :quit / :q  Exit the REPL".to_string(),
                                        "Enter facts, rules, constraints, and #const declarations to extend the program.".to_string(),
                                        "Enter queries starting with '?-' to evaluate them against the current program.".to_string(),
                                    ];
                                    for line in help_lines {
                                        state.push_history(line);
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
