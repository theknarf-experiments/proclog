//! Test runner for ProcLog test blocks
//!
//! This module implements functionality to run ProcLog tests defined with `#test` blocks.
//! Each test block contains facts, rules, and queries with assertions to verify behavior.

use crate::ast::{Atom, Literal, Rule, Statement, TestBlock, TestCase};
use crate::constants::ConstantEnv;
use crate::database::FactDatabase;
use crate::evaluation::semi_naive_evaluation;
use crate::query::evaluate_query;
use crate::unification::Substitution;

/// Result of running a single test case
#[derive(Debug, Clone)]
pub struct TestCaseResult {
    pub passed: bool,
    pub query_text: String,
    pub message: String,
    pub positive_failures: Vec<Atom>, // Positive assertions that weren't found
    pub negative_failures: Vec<Atom>, // Negative assertions that were found
}

/// Result of running an entire test block
#[derive(Debug, Clone)]
pub struct TestResult {
    pub test_name: String,
    pub passed: bool,
    pub total_cases: usize,
    pub passed_cases: usize,
    pub case_results: Vec<TestCaseResult>,
}

impl TestResult {
    /// Check if all test cases passed
    pub fn all_passed(&self) -> bool {
        self.passed
    }

    /// Get a summary message
    pub fn summary(&self) -> String {
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

/// Run a test block and return results
pub fn run_test_block(test_block: &TestBlock) -> TestResult {
    // Extract constants from test statements
    let const_env = ConstantEnv::from_statements(&test_block.statements);

    // Build initial database and rules from test statements
    let mut initial_facts = FactDatabase::new();
    let mut rules = Vec::new();

    for statement in &test_block.statements {
        match statement {
            Statement::Fact(fact) => {
                let substituted = const_env.substitute_atom(&fact.atom);
                initial_facts.insert(substituted);
            }
            Statement::Rule(rule) => {
                // Substitute constants in rule
                let substituted_head = const_env.substitute_atom(&rule.head);
                let substituted_body: Vec<_> = rule
                    .body
                    .iter()
                    .map(|lit| match lit {
                        Literal::Positive(atom) => {
                            Literal::Positive(const_env.substitute_atom(atom))
                        }
                        Literal::Negative(atom) => {
                            Literal::Negative(const_env.substitute_atom(atom))
                        }
                    })
                    .collect();

                rules.push(Rule {
                    head: substituted_head,
                    body: substituted_body,
                });
            }
            Statement::ConstDecl(_) => {
                // Already handled by const_env
            }
            _ => {
                // Ignore other statement types in tests
            }
        }
    }

    // Evaluate rules to get complete database
    let db = if rules.is_empty() {
        initial_facts
    } else {
        semi_naive_evaluation(&rules, initial_facts)
    };

    // Run each test case
    let mut case_results = Vec::new();
    let mut passed_count = 0;

    for test_case in &test_block.test_cases {
        let result = run_test_case(test_case, &db, &const_env);
        if result.passed {
            passed_count += 1;
        }
        case_results.push(result);
    }

    let all_passed = passed_count == test_block.test_cases.len();

    TestResult {
        test_name: test_block.name.clone(),
        passed: all_passed,
        total_cases: test_block.test_cases.len(),
        passed_cases: passed_count,
        case_results,
    }
}

/// Run a single test case
fn run_test_case(
    test_case: &TestCase,
    db: &FactDatabase,
    const_env: &ConstantEnv,
) -> TestCaseResult {
    // Substitute constants in query
    let query_body: Vec<_> = test_case
        .query
        .body
        .iter()
        .map(|lit| match lit {
            Literal::Positive(atom) => Literal::Positive(const_env.substitute_atom(atom)),
            Literal::Negative(atom) => Literal::Negative(const_env.substitute_atom(atom)),
        })
        .collect();

    let query = crate::ast::Query { body: query_body };

    // Run the query
    let results = evaluate_query(&query, db);

    // Build query text for display
    let query_text = format_query(&test_case.query);

    // Check assertions
    let (positive_failures, negative_failures) =
        check_assertions(test_case, &results, const_env);

    let passed = positive_failures.is_empty() && negative_failures.is_empty();

    let message = if passed {
        format!("✓ Query succeeded: {}", query_text)
    } else {
        let mut msg = format!("✗ Query failed: {}\n", query_text);
        if !positive_failures.is_empty() {
            msg.push_str(&format!(
                "  Missing expected results: {}\n",
                positive_failures
                    .iter()
                    .map(|a| format!("{:?}", a))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if !negative_failures.is_empty() {
            msg.push_str(&format!(
                "  Found unexpected results: {}\n",
                negative_failures
                    .iter()
                    .map(|a| format!("{:?}", a))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        msg
    };

    TestCaseResult {
        passed,
        query_text,
        message,
        positive_failures,
        negative_failures,
    }
}

/// Check positive and negative assertions against query results
fn check_assertions(
    test_case: &TestCase,
    results: &[Substitution],
    const_env: &ConstantEnv,
) -> (Vec<Atom>, Vec<Atom>) {
    let mut positive_failures = Vec::new();
    let mut negative_failures = Vec::new();

    // Check positive assertions (should be in results)
    for assertion in &test_case.positive_assertions {
        let substituted = const_env.substitute_atom(assertion);

        // Special case: "true" atom means the query should have at least one result
        if assertion.predicate.as_ref() == "true" && assertion.terms.is_empty() {
            if results.is_empty() {
                positive_failures.push(assertion.clone());
            }
        } else {
            // Check if this atom matches any result
            let matches = results.iter().any(|subst| {
                let instantiated = subst.apply_atom(&test_case.query.body[0].atom());
                atoms_match(&instantiated, &substituted)
            });

            if !matches {
                positive_failures.push(assertion.clone());
            }
        }
    }

    // Check negative assertions (should NOT be in results)
    for assertion in &test_case.negative_assertions {
        let substituted = const_env.substitute_atom(assertion);

        // Check if this atom matches any result
        let matches = results.iter().any(|subst| {
            let instantiated = subst.apply_atom(&test_case.query.body[0].atom());
            atoms_match(&instantiated, &substituted)
        });

        if matches {
            negative_failures.push(assertion.clone());
        }
    }

    (positive_failures, negative_failures)
}

/// Check if two atoms match (same predicate and terms)
fn atoms_match(a: &Atom, b: &Atom) -> bool {
    a == b
}

/// Format a query for display
fn format_query(query: &crate::ast::Query) -> String {
    let literals: Vec<String> = query
        .body
        .iter()
        .map(|lit| match lit {
            Literal::Positive(atom) => format!("{:?}", atom.predicate),
            Literal::Negative(atom) => format!("not {:?}", atom.predicate),
        })
        .collect();
    format!("?- {}.", literals.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use crate::parser::parse_program;

    #[test]
    fn test_simple_passing_test() {
        let input = r#"
            #test "basic test" {
                parent(john, mary).

                ?- parent(john, mary).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(test_block);
        assert!(result.passed, "Test should pass");
        assert_eq!(result.passed_cases, 1);
    }

    #[test]
    fn test_simple_failing_test() {
        let input = r#"
            #test "should fail" {
                parent(john, mary).

                ?- parent(alice, bob).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(test_block);
        assert!(!result.passed, "Test should fail");
        assert_eq!(result.passed_cases, 0);
    }

    #[test]
    fn test_with_rules() {
        let input = r#"
            #test "transitive closure" {
                edge(a, b).
                edge(b, c).
                path(X, Y) :- edge(X, Y).
                path(X, Z) :- path(X, Y), edge(Y, Z).

                ?- path(a, c).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(test_block);
        assert!(result.passed, "Test should pass: {:?}", result.case_results);
    }
}
