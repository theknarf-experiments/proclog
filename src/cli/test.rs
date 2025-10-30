use crate::{ast, parser, test_runner};
use crate::{COLOR_CYAN, COLOR_GREEN, COLOR_RED, COLOR_RESET, COLOR_YELLOW};
use std::fs;

pub fn run(filename: &str) {
    let content = match fs::read_to_string(filename) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", filename, e);
            std::process::exit(1);
        }
    };

    let program = match parser::parse_program(&content) {
        Ok(p) => p,
        Err(errors) => {
            eprintln!("Parse errors in '{}':", filename);
            for error in errors {
                eprintln!("  {:?}", error);
            }
            std::process::exit(1);
        }
    };

    let mut test_blocks = Vec::new();
    for statement in &program.statements {
        if let ast::Statement::Test(test_block) = statement {
            test_blocks.push(test_block);
        }
    }

    if test_blocks.is_empty() {
        println!(
            "{}No test blocks found in '{}'{}",
            COLOR_YELLOW, filename, COLOR_RESET
        );
        return;
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

    for test_block in test_blocks {
        let result = test_runner::run_test_block(test_block);
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

    if total_failed > 0 {
        std::process::exit(1);
    }
}
