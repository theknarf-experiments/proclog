//! CLI test runner for ProcLog
//!
//! This module provides CLI-specific formatting and presentation for test results.
//! The core testing logic lives in the proclog crate's test_runner module.

// Re-export the core types and functions from proclog
pub use proclog::test_runner::{run_test_block, TestResult};

/// Extension trait for CLI-specific formatting of test results
pub trait TestResultExt {
    /// Get a formatted summary message with CLI styling
    fn summary(&self) -> String;
}

impl TestResultExt for TestResult {
    fn summary(&self) -> String {
        if self.passed {
            format!(
                "✓ Test '{}': {} / {} cases passed",
                self.test_name, self.passed_cases, self.total_cases
            )
        } else {
            format!(
                "✗ Test '{}': {} / {} cases passed",
                self.test_name, self.passed_cases, self.total_cases
            )
        }
    }
}
