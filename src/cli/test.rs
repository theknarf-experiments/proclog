use crate::{ast, parser, test_runner};
use crate::{COLOR_CYAN, COLOR_GREEN, COLOR_RED, COLOR_RESET, COLOR_YELLOW};
use notify::{recommended_watcher, Event, EventKind, RecursiveMode, Watcher};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub fn run(filename: &str, watch: bool) {
    let path = PathBuf::from(filename);

    if !path.exists() {
        eprintln!("File '{}' not found.", filename);
        std::process::exit(1);
    }

    let last_success = run_tests_from_file(&path);

    if watch {
        println!(
            "{}Watching '{}' for changes... Press Ctrl+C to stop.{}",
            COLOR_CYAN,
            path.display(),
            COLOR_RESET
        );

        let (tx, rx) = mpsc::channel();
        let mut watcher = match recommended_watcher(move |res| {
            let _ = tx.send(res);
        }) {
            Ok(w) => w,
            Err(err) => {
                eprintln!("Failed to start file watcher: {}", err);
                std::process::exit(1);
            }
        };

        if let Err(err) = watcher.watch(&path, RecursiveMode::NonRecursive) {
            eprintln!("Failed to watch '{}': {}", path.display(), err);
            std::process::exit(1);
        }

        let mut last_event = Instant::now();
        while let Ok(event) = rx.recv() {
            match event {
                Ok(event) => {
                    if should_trigger(&event) && last_event.elapsed() >= Duration::from_millis(100)
                    {
                        last_event = Instant::now();
                        println!(
                            "\n{}Detected change, re-running tests...{}",
                            COLOR_CYAN, COLOR_RESET
                        );
                        let _ = run_tests_from_file(&path);
                    }
                }
                Err(err) => {
                    eprintln!("Watch error: {}", err);
                }
            }
        }
    } else if !last_success {
        std::process::exit(1);
    }
}

fn run_tests_from_file(path: &Path) -> bool {
    let filename = path.display().to_string();

    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", filename, e);
            return false;
        }
    };

    let program = match parser::parse_program(&content) {
        Ok(p) => p,
        Err(errors) => {
            eprintln!("Parse errors in '{}':", filename);
            for error in errors {
                eprintln!("  {:?}", error);
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
            COLOR_YELLOW, filename, COLOR_RESET
        );
        return true;
    }

    println!(
        "{}Running {} test blocks from '{}'...{}",
        COLOR_CYAN,
        test_blocks.len(),
        filename,
        COLOR_RESET
    );

    let mut total_passed = 0usize;
    let mut total_failed = 0usize;

    for test_block in &test_blocks {
        let result = test_runner::run_test_block(&base_statements, test_block);
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

fn should_trigger(event: &Event) -> bool {
    matches!(
        event.kind,
        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_) | EventKind::Any
    )
}
