//! WebAssembly bindings for ProcLog
//!
//! This crate provides WASM bindings for the ProcLog logic programming language,
//! allowing it to be used from JavaScript/TypeScript in Node.js and browsers.

use proclog::{asp, parser, test_runner};
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Initialize the WASM module.
/// This sets up better error messages in the browser console.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Parse a ProcLog program from source code.
/// Returns the parsed AST as a JavaScript object.
#[wasm_bindgen]
pub fn parse_program(source: &str) -> Result<JsValue, JsValue> {
    parser::parse_program(source)
        .map(|program| serde_wasm_bindgen::to_value(&program))
        .map_err(|errors| JsValue::from_str(&format!("Parse error: {:?}", errors)))?
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Evaluate a ProcLog program and return answer sets.
/// Returns a JavaScript object containing answer sets.
#[wasm_bindgen]
pub fn evaluate_program(source: &str) -> Result<JsValue, JsValue> {
    // Parse the program
    let program = parser::parse_program(source)
        .map_err(|errors| JsValue::from_str(&format!("Parse error: {:?}", errors)))?;

    // Evaluate using ASP
    let answer_sets = asp::asp_evaluation(&program);

    // Create result structure
    #[derive(Serialize)]
    struct EvaluationResult {
        #[serde(rename = "answerSets")]
        answer_sets: Vec<AnswerSetWrapper>,
    }

    #[derive(Serialize)]
    struct AnswerSetWrapper {
        atoms: Vec<proclog::ast::Atom>,
    }

    let result = EvaluationResult {
        answer_sets: answer_sets
            .into_iter()
            .map(|as_set| AnswerSetWrapper {
                atoms: as_set.atoms.into_iter().collect(),
            })
            .collect(),
    };

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Run test blocks in a ProcLog program.
/// Returns test results as a JavaScript object.
#[wasm_bindgen]
pub fn run_tests(source: &str, use_sat_solver: bool) -> Result<JsValue, JsValue> {
    // Parse the program
    let program = parser::parse_program(source)
        .map_err(|errors| JsValue::from_str(&format!("Parse error: {:?}", errors)))?;

    // Separate test blocks from base statements
    let mut test_blocks = Vec::new();
    let mut base_statements = Vec::new();

    for statement in program.statements {
        match statement {
            proclog::ast::Statement::Test(test_block) => test_blocks.push(test_block),
            other => base_statements.push(other),
        }
    }

    // Run each test block
    let mut results = Vec::new();
    for test_block in &test_blocks {
        let result = test_runner::run_test_block(&base_statements, test_block, use_sat_solver);
        results.push(result);
    }

    // Calculate summary
    let total_tests = results.len();
    let passed_tests = results.iter().filter(|r| r.passed).count();
    let failed_tests = total_tests - passed_tests;

    // Create result structure
    #[derive(Serialize)]
    struct TestRunResult {
        results: Vec<test_runner::TestResult>,
        #[serde(rename = "totalTests")]
        total_tests: usize,
        #[serde(rename = "passedTests")]
        passed_tests: usize,
        #[serde(rename = "failedTests")]
        failed_tests: usize,
    }

    let result = TestRunResult {
        results,
        total_tests,
        passed_tests,
        failed_tests,
    };

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_program() {
        let result = parse_program("parent(alice, bob).");
        assert!(result.is_ok());
    }

    #[test]
    fn test_evaluate_program() {
        let result = evaluate_program("parent(alice, bob).");
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_tests() {
        let source = r#"
            parent(alice, bob).

            #test "simple test" {
                ?- parent(alice, bob).
            }
        "#;
        let result = run_tests(source, false);
        assert!(result.is_ok());
    }
}
