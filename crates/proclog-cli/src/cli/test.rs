use crate::test_runner::{self, TestResultExt};
use crate::{COLOR_CYAN, COLOR_GREEN, COLOR_RED, COLOR_RESET, COLOR_YELLOW};
use chumsky::error::{Simple, SimpleReason};
use notify::{recommended_watcher, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use proclog::{ast, parser};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub fn run(files: &[PathBuf], watch: bool, use_sat_solver: bool) {
    if files.is_empty() {
        eprintln!("No test files provided.");
        std::process::exit(1);
    }

    #[derive(Clone)]
    struct InputFile {
        canonical: PathBuf,
        display: String,
    }

    let mut seen = HashSet::new();
    let mut inputs = Vec::new();

    for file in files {
        let path = file.clone();
        if !path.exists() {
            eprintln!("File '{}' not found.", path.display());
            std::process::exit(1);
        }

        let canonical = path.canonicalize().unwrap_or(path.clone());
        if seen.insert(canonical.clone()) {
            inputs.push(InputFile {
                canonical,
                display: path.display().to_string(),
            });
        }
    }

    if inputs.is_empty() {
        eprintln!("No valid test files provided.");
        std::process::exit(1);
    }

    let mut overall_success = true;
    for (idx, file) in inputs.iter().enumerate() {
        let success = run_tests_from_file(&file.canonical, &file.display, use_sat_solver);
        overall_success &= success;
        if idx < inputs.len() - 1 {
            println!();
        }
    }

    if watch {
        let file_list = inputs
            .iter()
            .map(|f| f.display.clone())
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "{}Watching {} for changes... Press Ctrl+C to stop.{}",
            COLOR_CYAN, file_list, COLOR_RESET
        );

        let (tx, rx) = mpsc::channel();
        let mut watcher: RecommendedWatcher = match recommended_watcher(move |res| {
            let _ = tx.send(res);
        }) {
            Ok(w) => w,
            Err(err) => {
                eprintln!("Failed to start file watcher: {}", err);
                std::process::exit(1);
            }
        };

        for file in &inputs {
            if let Err(err) = watcher.watch(&file.canonical, RecursiveMode::NonRecursive) {
                eprintln!("Failed to watch '{}': {}", file.display, err);
            }
        }

        let mut last_event = Instant::now();
        while let Ok(event) = rx.recv() {
            match event {
                Ok(event) => {
                    if should_trigger(&event) && last_event.elapsed() >= Duration::from_millis(100)
                    {
                        last_event = Instant::now();
                        println!(
                            "
{}Detected change, re-running tests...{}",
                            COLOR_CYAN, COLOR_RESET
                        );
                        for (idx, file) in inputs.iter().enumerate() {
                            let _ = run_tests_from_file(&file.canonical, &file.display, use_sat_solver);
                            if idx < inputs.len() - 1 {
                                println!();
                            }
                        }
                    }
                }
                Err(err) => {
                    eprintln!("Watch error: {}", err);
                }
            }
        }
    } else if !overall_success {
        std::process::exit(1);
    }
}

fn run_tests_from_file(path: &Path, display_name: &str, use_sat_solver: bool) -> bool {
    let identifier = display_name;

    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", identifier, e);
            return false;
        }
    };

    let program = match parser::parse_program(&content) {
        Ok(p) => p,
        Err(errors) => {
            eprintln!(
                "{}error:{} Failed to parse '{}'.",
                COLOR_RED, COLOR_RESET, identifier
            );
            for error in errors {
                eprintln!("{}", format_parse_error(identifier, &content, &error));
            }
            return false;
        }
    };

    let mut test_blocks = Vec::new();
    let mut base_statements = Vec::new();
    for statement in program.statements {
        match statement {
            ast::Statement::Test(test_block) => test_blocks.push(test_block),
            other => base_statements.push(other),
        }
    }

    if test_blocks.is_empty() {
        println!(
            "{}No test blocks found in '{}'{}",
            COLOR_YELLOW, identifier, COLOR_RESET
        );
        return true;
    }

    println!(
        "{}Running {} test blocks from '{}'...{}",
        COLOR_CYAN,
        test_blocks.len(),
        identifier,
        COLOR_RESET
    );

    let mut total_passed = 0usize;
    let mut total_failed = 0usize;

    for test_block in &test_blocks {
        let result = test_runner::run_test_block(&base_statements, test_block, use_sat_solver);
        let color = if result.passed {
            COLOR_GREEN
        } else {
            COLOR_RED
        };
        println!("{}{}{}", color, result.summary(), COLOR_RESET);

        if result.passed {
            total_passed += 1;
        } else {
            total_failed += 1;

            for case_result in &result.case_results {
                if !case_result.passed {
                    println!("  {}{}", COLOR_RED, case_result.message);
                    println!("{}", COLOR_RESET);
                }
            }
        }
    }

    let total = total_passed + total_failed;
    let status_color = if total_failed == 0 {
        COLOR_GREEN
    } else {
        COLOR_RED
    };
    println!(
        "{}Summary:{} passed={}, failed={}, total={} | Status: {}{}{}",
        COLOR_YELLOW,
        COLOR_RESET,
        total_passed,
        total_failed,
        total,
        status_color,
        if total_failed == 0 { "PASS" } else { "FAIL" },
        COLOR_RESET
    );

    total_failed == 0
}

fn format_parse_error(file_name: &str, source: &str, error: &Simple<char>) -> String {
    let span = error.span();
    let total_chars = source.chars().count();

    let start_char = span.start.min(total_chars);
    let end_char = span.end.min(total_chars);

    let start_byte = char_idx_to_byte(source, start_char);
    let end_byte = char_idx_to_byte(source, end_char);

    let (line, column, line_start_byte, line_end_byte) = line_info(source, start_byte);

    let line_end_byte = line_end_byte.min(source.len());
    let highlight_end_byte = end_byte.min(line_end_byte);
    let highlight_end_byte = highlight_end_byte.max(start_byte);

    let highlight_text = &source[start_byte..highlight_end_byte];

    let highlighted_line = format!(
        "{}{}{}{}{}",
        &source[line_start_byte..start_byte],
        COLOR_RED,
        highlight_text,
        COLOR_RESET,
        &source[highlight_end_byte..line_end_byte]
    );

    let pointer_prefix: String = source[line_start_byte..start_byte]
        .chars()
        .map(|ch| if ch == '\t' { '\t' } else { ' ' })
        .collect();

    let highlight_width = std::cmp::max(highlight_text.chars().count(), 1);
    let pointer_line = format!(
        "{}{}{}{}",
        pointer_prefix,
        COLOR_RED,
        "^".repeat(highlight_width),
        COLOR_RESET
    );

    let mut message = match error.reason() {
        SimpleReason::Unexpected => match error.found() {
            Some(found) => format!("Unexpected character '{}'", found),
            None => "Unexpected end of input".to_string(),
        },
        SimpleReason::Unclosed { span: _, delimiter } => {
            format!("Unclosed delimiter '{}'", delimiter)
        }
        SimpleReason::Custom(msg) => msg.to_string(),
    };

    if let Some(label) = error.label() {
        if !label.is_empty() {
            message = format!("{} ({})", message.trim_end_matches('.'), label);
        }
    }

    let expected: Vec<String> = error
        .expected()
        .filter_map(|expected| match expected {
            Some(ch) => Some(format!("'{}'", ch)),
            None => Some("end of input".to_string()),
        })
        .collect();

    if !expected.is_empty() {
        let expected_list = expected.join(", ");
        message = format!(
            "{}. Expected one of: {}",
            message.trim_end_matches('.'),
            expected_list
        );
    } else if !message.ends_with('.') {
        message.push('.');
    }

    format!(
        " --> {}:{}:{}\n  |\n{:>3} | {}\n  | {}\n{}{}{}",
        file_name,
        line,
        column,
        line,
        highlighted_line,
        pointer_line,
        COLOR_YELLOW,
        message,
        COLOR_RESET
    )
}

fn char_idx_to_byte(input: &str, char_idx: usize) -> usize {
    if char_idx == 0 {
        return 0;
    }

    input
        .char_indices()
        .nth(char_idx)
        .map(|(idx, _)| idx)
        .unwrap_or_else(|| input.len())
}

fn line_info(input: &str, byte_idx: usize) -> (usize, usize, usize, usize) {
    let mut line = 1;
    let mut line_start = 0;

    for (idx, ch) in input.char_indices() {
        if idx >= byte_idx {
            break;
        }
        if ch == '\n' {
            line += 1;
            line_start = idx + ch.len_utf8();
        }
    }

    let line_end = input[line_start..]
        .find('\n')
        .map(|offset| line_start + offset)
        .unwrap_or_else(|| input.len());

    let column = input[line_start..byte_idx]
        .chars()
        .count()
        .saturating_add(1);

    (line, column, line_start, line_end)
}

fn should_trigger(event: &Event) -> bool {
    matches!(
        event.kind,
        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_) | EventKind::Any
    )
}
