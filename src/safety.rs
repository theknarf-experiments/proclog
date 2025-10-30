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

use crate::ast::{Atom, Literal, Rule, Symbol, Term};
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
/// also appears in at least one positive literal in the body
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
            vars.insert(name.clone());
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
mod tests {
    use super::*;
    use crate::ast::Value;
    use internment::Intern;

    // Helper functions
    #[allow(dead_code)]
    fn atom_const(name: &str) -> Term {
        Term::Constant(Value::Atom(Intern::new(name.to_string())))
    }

    fn var(name: &str) -> Term {
        Term::Variable(Intern::new(name.to_string()))
    }

    fn make_atom(predicate: &str, terms: Vec<Term>) -> Atom {
        Atom {
            predicate: Intern::new(predicate.to_string()),
            terms,
        }
    }

    fn make_rule(head: Atom, body: Vec<Literal>) -> Rule {
        Rule { head, body }
    }

    #[test]
    fn test_safe_rule_all_positive() {
        // p(X) :- q(X).
        let rule = make_rule(
            make_atom("p", vec![var("X")]),
            vec![Literal::Positive(make_atom("q", vec![var("X")]))],
        );

        assert!(check_rule_safety(&rule).is_ok());
    }

    #[test]
    fn test_safe_rule_with_negation() {
        // p(X) :- q(X), not r(X).
        let rule = make_rule(
            make_atom("p", vec![var("X")]),
            vec![
                Literal::Positive(make_atom("q", vec![var("X")])),
                Literal::Negative(make_atom("r", vec![var("X")])),
            ],
        );

        assert!(check_rule_safety(&rule).is_ok());
    }

    #[test]
    fn test_unsafe_negation_only() {
        // UNSAFE: bad(X) :- not good(X).
        let rule = make_rule(
            make_atom("bad", vec![var("X")]),
            vec![Literal::Negative(make_atom("good", vec![var("X")]))],
        );

        let result = check_rule_safety(&rule);
        assert!(result.is_err());

        if let Err(SafetyError::UnsafeNegation { variables, .. }) = result {
            assert_eq!(variables.len(), 1);
            assert_eq!(variables[0], Intern::new("X".to_string()));
        }
    }

    #[test]
    fn test_unsafe_head_variable() {
        // UNSAFE: p(X, Y) :- q(X).
        // Y appears in head but not in any positive body literal
        let rule = make_rule(
            make_atom("p", vec![var("X"), var("Y")]),
            vec![Literal::Positive(make_atom("q", vec![var("X")]))],
        );

        let result = check_rule_safety(&rule);
        assert!(result.is_err());

        if let Err(SafetyError::UnsafeNegation { variables, .. }) = result {
            assert!(variables.contains(&Intern::new("Y".to_string())));
        }
    }

    #[test]
    fn test_safe_with_multiple_positive_literals() {
        // p(X, Y) :- q(X), r(Y), not s(X, Y).
        let rule = make_rule(
            make_atom("p", vec![var("X"), var("Y")]),
            vec![
                Literal::Positive(make_atom("q", vec![var("X")])),
                Literal::Positive(make_atom("r", vec![var("Y")])),
                Literal::Negative(make_atom("s", vec![var("X"), var("Y")])),
            ],
        );

        assert!(check_rule_safety(&rule).is_ok());
    }

    #[test]
    fn test_unsafe_variable_in_negation_only() {
        // UNSAFE: p(X) :- q(X), not r(X, Y).
        // Y only appears in negated literal
        let rule = make_rule(
            make_atom("p", vec![var("X")]),
            vec![
                Literal::Positive(make_atom("q", vec![var("X")])),
                Literal::Negative(make_atom("r", vec![var("X"), var("Y")])),
            ],
        );

        let result = check_rule_safety(&rule);
        assert!(result.is_err());

        if let Err(SafetyError::UnsafeNegation { variables, .. }) = result {
            assert!(variables.contains(&Intern::new("Y".to_string())));
        }
    }

    #[test]
    fn test_safe_with_compound_terms() {
        // p(X) :- q(item(X, Y)), not r(item(X, Y)).
        let rule = make_rule(
            make_atom("p", vec![var("X")]),
            vec![
                Literal::Positive(make_atom(
                    "q",
                    vec![Term::Compound(
                        Intern::new("item".to_string()),
                        vec![var("X"), var("Y")],
                    )],
                )),
                Literal::Negative(make_atom(
                    "r",
                    vec![Term::Compound(
                        Intern::new("item".to_string()),
                        vec![var("X"), var("Y")],
                    )],
                )),
            ],
        );

        assert!(check_rule_safety(&rule).is_ok());
    }

    #[test]
    fn test_unsafe_compound_term_variable() {
        // UNSAFE: p(X) :- q(X), not r(item(Y)).
        // Y only appears in negated literal within compound term
        let rule = make_rule(
            make_atom("p", vec![var("X")]),
            vec![
                Literal::Positive(make_atom("q", vec![var("X")])),
                Literal::Negative(make_atom(
                    "r",
                    vec![Term::Compound(
                        Intern::new("item".to_string()),
                        vec![var("Y")],
                    )],
                )),
            ],
        );

        let result = check_rule_safety(&rule);
        assert!(result.is_err());
    }

    #[test]
    fn test_program_safety_all_safe() {
        let rules = vec![
            make_rule(
                make_atom("p", vec![var("X")]),
                vec![Literal::Positive(make_atom("q", vec![var("X")]))],
            ),
            make_rule(
                make_atom("r", vec![var("X")]),
                vec![
                    Literal::Positive(make_atom("s", vec![var("X")])),
                    Literal::Negative(make_atom("t", vec![var("X")])),
                ],
            ),
        ];

        assert!(check_program_safety(&rules).is_ok());
    }

    #[test]
    fn test_program_safety_one_unsafe() {
        let rules = vec![
            make_rule(
                make_atom("p", vec![var("X")]),
                vec![Literal::Positive(make_atom("q", vec![var("X")]))],
            ),
            // This one is unsafe
            make_rule(
                make_atom("bad", vec![var("X")]),
                vec![Literal::Negative(make_atom("good", vec![var("X")]))],
            ),
        ];

        let result = check_program_safety(&rules);
        assert!(result.is_err());
    }
}
