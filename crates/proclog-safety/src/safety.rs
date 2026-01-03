//! Safety checking for Datalog rules
//!
//! This module checks that all variables in a rule are "safe" - i.e., they appear
//! in positive literals in the body. This ensures finite grounding.
//!
//! # Safety Rules
//!
//! A rule is safe if:
//! 1. All variables in the head appear in positive body literals
//! 2. All variables in negative literals appear in positive body literals
//! 3. All variables in built-in predicates appear in positive body literals
//!
//! # Why Safety Matters
//!
//! Unsafe rules can have infinite groundings, making evaluation impossible.
//!
//! # Example
//!
//! ```ignore
//! // Safe: ancestor(X, Y) :- parent(X, Y).
//! // Unsafe: bad(X) :- not good(X).  // X appears only in negation
//! ```

use proclog_ast::{Atom, Literal, Rule, Symbol, Term};
use std::collections::HashSet;

/// Error indicating a rule is unsafe
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafetyError {
    /// Variable appears only in negated literals
    UnsafeNegation {
        rule: String,
        variables: Vec<Symbol>,
    },
}

impl std::fmt::Display for SafetyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafetyError::UnsafeNegation { rule, variables } => {
                write!(
                    f,
                    "Unsafe negation in rule '{}': variables {:?} appear only in negated literals",
                    rule, variables
                )
            }
        }
    }
}

impl std::error::Error for SafetyError {}

/// Check if a rule is safe
/// A rule is safe if every variable that appears in:
/// - The head
/// - A negated literal
///   also appears in at least one positive literal in the body
pub fn check_rule_safety(rule: &Rule) -> Result<(), SafetyError> {
    // Collect all variables from positive literals
    let mut positive_vars = HashSet::new();
    for literal in &rule.body {
        if let Literal::Positive(atom) = literal {
            collect_vars_from_atom(atom, &mut positive_vars);
        }
    }

    // Check head variables - all must appear in positive literals
    let mut head_vars = HashSet::new();
    collect_vars_from_atom(&rule.head, &mut head_vars);

    let unsafe_head_vars: Vec<Symbol> = head_vars
        .iter()
        .filter(|v| !positive_vars.contains(v))
        .cloned()
        .collect();

    if !unsafe_head_vars.is_empty() {
        return Err(SafetyError::UnsafeNegation {
            rule: format_rule(rule),
            variables: unsafe_head_vars,
        });
    }

    // Check negated literal variables - all must appear in positive literals
    for literal in &rule.body {
        if let Literal::Negative(atom) = literal {
            let mut neg_vars = HashSet::new();
            collect_vars_from_atom(atom, &mut neg_vars);

            let unsafe_vars: Vec<Symbol> = neg_vars
                .iter()
                .filter(|v| !positive_vars.contains(v))
                .cloned()
                .collect();

            if !unsafe_vars.is_empty() {
                return Err(SafetyError::UnsafeNegation {
                    rule: format_rule(rule),
                    variables: unsafe_vars,
                });
            }
        }
    }

    Ok(())
}

/// Check if all rules in a program are safe
pub fn check_program_safety(rules: &[Rule]) -> Result<(), SafetyError> {
    for rule in rules {
        check_rule_safety(rule)?;
    }
    Ok(())
}

/// Collect all variables from an atom (including in compound terms)
fn collect_vars_from_atom(atom: &Atom, vars: &mut HashSet<Symbol>) {
    for term in &atom.terms {
        collect_vars_from_term(term, vars);
    }
}

/// Collect all variables from a term recursively
fn collect_vars_from_term(term: &Term, vars: &mut HashSet<Symbol>) {
    match term {
        Term::Variable(name) => {
            vars.insert(*name);
        }
        Term::Constant(_) => {
            // Constants don't have variables
        }
        Term::Range(_, _) => {
            // Ranges don't have variables (yet - could support constant names later)
        }
        Term::Compound(_functor, args) => {
            for arg in args {
                collect_vars_from_term(arg, vars);
            }
        }
    }
}

/// Format a rule as a string for error messages
fn format_rule(rule: &Rule) -> String {
    format!("{:?}", rule.head.predicate)
}

#[cfg(test)]
#[path = "../tests/unit/safety_tests.rs"]
mod tests;
