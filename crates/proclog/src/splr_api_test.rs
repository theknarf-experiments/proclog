//! Tests to understand splr's API

use splr::*;
use std::convert::TryFrom;

#[test]
fn test_splr_from_cnf() {
    // Create a simple CNF: (a OR b) AND (NOT a OR c)
    // Variable 1 = a, Variable 2 = b, Variable 3 = c
    let config = Config::default();
    let clauses: Vec<Vec<i32>> = vec![
        vec![1, 2],  // a OR b
        vec![-1, 3], // NOT a OR c
    ];

    let mut solver = Solver::try_from((config, clauses.as_ref())).expect("Failed to create solver");

    println!("Created solver with 2 clauses");

    // Solve
    match solver.solve() {
        Ok(Certificate::SAT(model)) => {
            println!("SAT! Model: {:?}", model);
            assert!(model.len() >= 3);
        }
        Ok(Certificate::UNSAT) => panic!("Should be SAT"),
        Err(e) => panic!("Error: {:?}", e),
    }
}

#[test]
fn test_splr_incremental_simple() {
    // Start with empty clauses, try adding them incrementally
    let config = Config::default();
    let clauses: Vec<Vec<i32>> = vec![
        vec![1, 2],  // a OR b
        vec![-1, 3], // NOT a OR c
        vec![1],     // a is true
    ];

    let mut solver = Solver::try_from((config, clauses.as_ref())).expect("Failed to create solver");

    println!("Created solver with 3 clauses");

    match solver.solve() {
        Ok(Certificate::SAT(model)) => {
            println!("SAT! Model: {:?}", model);
            // a must be true (variable 1 should be positive)
            assert!(model.len() >= 3);
            println!("Variable 1 (a): {}", model[0]);
        }
        Ok(Certificate::UNSAT) => panic!("Should be SAT"),
        Err(e) => panic!("Error: {:?}", e),
    }
}

#[test]
fn test_splr_add_clause_after_init() {
    // Start with some clauses that define all variables
    let config = Config::default();
    let clauses: Vec<Vec<i32>> = vec![
        vec![1, 2, 3], // Declare all 3 variables upfront
    ];

    let mut solver = Solver::try_from((config, clauses.as_ref())).expect("Failed to create solver");

    println!("Created solver with 1 clause (defining vars 1-3)");

    // Now try adding another clause with existing variables
    match solver.add_clause(vec![-1, 3]) {
        Ok(_) => println!("Successfully added clause"),
        Err(e) => panic!("Error adding clause: {:?}", e),
    }

    match solver.solve() {
        Ok(Certificate::SAT(model)) => {
            println!("SAT! Model: {:?}", model);
        }
        Ok(Certificate::UNSAT) => panic!("Should be SAT"),
        Err(e) => panic!("Error: {:?}", e),
    }
}

#[test]
fn test_splr_needs_variables_upfront() {
    // Verify that we can't use variables not declared in initial clauses
    let config = Config::default();
    let clauses: Vec<Vec<i32>> = vec![
        vec![1, 2], // Only declare vars 1, 2
    ];

    let mut solver = Solver::try_from((config, clauses.as_ref())).expect("Failed to create solver");

    // This should fail because var 3 wasn't in initial clauses
    match solver.add_clause(vec![-1, 3]) {
        Ok(_) => panic!("Should have failed - variable 3 not declared"),
        Err(SolverError::InvalidLiteral) => {
            println!("As expected: can't use undeclared variables");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

#[test]
fn test_splr_unsat_example() {
    // Test the exact same clauses as our constraint test
    // This demonstrates that splr detects UNSAT during TryFrom, not during solve()
    let config = Config::default();
    let clauses: Vec<Vec<i32>> = vec![
        vec![1],      // a is true
        vec![2],      // b is true
        vec![-1, -2], // NOT a OR NOT b (cannot have both)
    ];

    match Solver::try_from((config, clauses.as_ref())) {
        Ok(mut solver) => {
            // If solver creation succeeds, it should still be UNSAT when solved
            match solver.solve() {
                Ok(Certificate::SAT(model)) => {
                    panic!("Should be UNSAT, got SAT: {:?}", model);
                }
                Ok(Certificate::UNSAT) => {
                    println!("Correctly identified as UNSAT during solve");
                }
                Err(e) => panic!("Error during solve: {:?}", e),
            }
        }
        Err(Ok(Certificate::UNSAT)) => {
            println!("Correctly identified as UNSAT during preprocessing");
        }
        Err(Err(SolverError::EmptyClause)) => {
            println!("Correctly detected empty clause (UNSAT) during preprocessing");
        }
        Err(e) => panic!("Unexpected error during TryFrom: {:?}", e),
    }
}
