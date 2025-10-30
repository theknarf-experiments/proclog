use crate::repl::{ReplEngine, ResponseKind};
use crate::{COLOR_CYAN, COLOR_GREEN, COLOR_RED, COLOR_RESET, COLOR_YELLOW};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    text::Line,
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};
use std::{io::stdout, time::Duration};

pub fn run() -> std::io::Result<()> {
    struct ReplState {
        input: String,
        history: Vec<String>,
        should_quit: bool,
        engine: ReplEngine,
    }

    impl Default for ReplState {
        fn default() -> Self {
            Self {
                input: String::new(),
                history: Vec::new(),
                should_quit: false,
                engine: ReplEngine::new(),
            }
        }
    }

    impl ReplState {
        fn push_history(&mut self, line: impl Into<String>) {
            const MAX_LEN: usize = 200;
            self.history.push(line.into());
            if self.history.len() > MAX_LEN {
                let excess = self.history.len() - MAX_LEN;
                self.history.drain(0..excess);
            }
        }
    }

    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.show_cursor()?;

    let mut state = ReplState::default();
    state.push_history("Welcome to the ProcLog REPL! Type :help for commands.");

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
            let input = Paragraph::new(vec![Line::from(input_display.as_str())])
                .block(Block::default().title("Input").borders(Borders::ALL))
                .wrap(Wrap { trim: false });
            frame.render_widget(input, chunks[1]);

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
                                    if response.lines.is_empty() {
                                        // Nothing to report
                                    } else {
                                        match response.kind {
                                            ResponseKind::Info => {
                                                state.push_history(format!(
                                                    "{}{}{}",
                                                    COLOR_CYAN, response.lines[0], COLOR_RESET
                                                ));
                                                for line in response.lines.iter().skip(1) {
                                                    state.push_history(line.clone());
                                                }
                                            }
                                            ResponseKind::Output => {
                                                let first = response.lines[0].clone();
                                                let color = if first == "true." {
                                                    COLOR_GREEN
                                                } else if first == "false."
                                                    || first.starts_with("No results.")
                                                {
                                                    COLOR_RED
                                                } else {
                                                    COLOR_YELLOW
                                                };
                                                state.push_history(format!(
                                                    "{}{}{}",
                                                    color, first, COLOR_RESET
                                                ));
                                                for line in response.lines.iter().skip(1) {
                                                    state.push_history(line.clone());
                                                }
                                            }
                                            ResponseKind::Error => {
                                                for line in response.lines {
                                                    state.push_history(format!(
                                                        "{}{}{}",
                                                        COLOR_RED, line, COLOR_RESET
                                                    ));
                                                }
                                            }
                                        }
                                    }
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
