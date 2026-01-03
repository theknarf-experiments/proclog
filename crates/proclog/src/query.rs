//! Query evaluation
//!
//! This module implements query evaluation against a fact database.
//! Queries find all substitutions that satisfy the query body.
//!
//! # Query Types
//!
//! - **Ground queries**: No variables, returns true/false
//! - **Variable queries**: With variables, returns all matching substitutions
//!
//! # Example
//!
//! ```ignore
//! // Query: ?- parent(X, mary).
//! let results = evaluate_query(&query, &db);
//! // Returns: [{X -> john}, {X -> alice}]
//! ```

use crate::ast::{Query, Symbol};
use crate::database::FactDatabase;
use crate::grounding::satisfy_body;
use crate::unification::Substitution;
use std::collections::HashSet;

/// Result of query evaluation - list of substitutions that satisfy the query
pub type QueryResult = Vec<Substitution>;

/// Evaluate a query against a fact database
///
/// Returns all substitutions that satisfy the query body.
/// For ground queries (no variables), returns either:
/// - `vec![Substitution::new()]` if the query is true
/// - `vec![]` if the query is false
///
/// For queries with variables, returns all matching substitutions.
pub fn evaluate_query(query: &Query, db: &FactDatabase) -> QueryResult {
    // Use satisfy_body from grounding module to find all substitutions
    satisfy_body(&query.body, db)
}

/// Extract variable bindings from a substitution for a set of variables
///
/// Returns a mapping from variable names to their bound values (as strings)
#[cfg_attr(not(test), allow(dead_code))]
pub fn extract_bindings(
    subst: &Substitution,
    variables: &HashSet<Symbol>,
) -> Vec<(Symbol, String)> {
    variables
        .iter()
        .filter_map(|var| subst.get(var).map(|term| (*var, format!("{:?}", term))))
        .collect()
}

/// Extract all variables from a query
#[cfg_attr(not(test), allow(dead_code))]
pub fn query_variables(query: &Query) -> HashSet<Symbol> {
    use crate::ast::Term;

    let mut vars = HashSet::new();

    fn collect_term_vars(term: &Term, vars: &mut HashSet<Symbol>) {
        match term {
            Term::Variable(v) => {
                vars.insert(*v);
            }
            Term::Compound(_, args) => {
                for arg in args {
                    collect_term_vars(arg, vars);
                }
            }
            _ => {}
        }
    }

    for literal in &query.body {
        if let Some(atom) = literal.atom() {
            for term in &atom.terms {
                collect_term_vars(term, &mut vars);
            }
        }
        // For aggregates, variables are tracked separately in the aggregate structure
    }

    vars
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Atom, Literal, Query, Term, Value};
    use internment::Intern;

    // Helper functions
    fn make_atom(pred: &str, terms: Vec<Term>) -> Atom {
        Atom {
            predicate: Intern::new(pred.to_string()),
            terms,
        }
    }

    fn atom_term(name: &str) -> Term {
        Term::Constant(Value::Atom(Intern::new(name.to_string())))
    }

    fn var(name: &str) -> Term {
        Term::Variable(Intern::new(name.to_string()))
    }

    #[test]
    fn test_query_ground_true() {
        // Setup database
        let mut db = FactDatabase::new();
        let fact = make_atom("parent", vec![atom_term("john"), atom_term("mary")]);
        db.insert(fact.clone()).unwrap();

        // Query: ?- parent(john, mary).
        let query = Query {
            body: vec![Literal::Positive(fact)],
        };

        let results = evaluate_query(&query, &db);
        assert_eq!(results.len(), 1, "Ground query should return one result");
    }

    #[test]
    fn test_query_ground_false() {
        // Setup database
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![atom_term("john"), atom_term("mary")],
        ))
        .unwrap();

        // Query: ?- parent(alice, bob). (not in database)
        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![atom_term("alice"), atom_term("bob")],
            ))],
        };

        let results = evaluate_query(&query, &db);
        assert_eq!(results.len(), 0, "False query should return empty result");
    }

    #[test]
    fn test_query_with_one_variable() {
        // Setup database
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![atom_term("john"), atom_term("mary")],
        ))
        .unwrap();
        db.insert(make_atom(
            "parent",
            vec![atom_term("alice"), atom_term("mary")],
        ))
        .unwrap();
        db.insert(make_atom(
            "parent",
            vec![atom_term("bob"), atom_term("sue")],
        ))
        .unwrap();

        // Query: ?- parent(X, mary).
        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var("X"), atom_term("mary")],
            ))],
        };

        let results = evaluate_query(&query, &db);
        assert_eq!(
            results.len(),
            2,
            "Should find 2 parents of mary: {:?}",
            results
        );

        // Verify the bindings
        let vars = query_variables(&query);
        for subst in &results {
            let bindings = extract_bindings(subst, &vars);
            assert_eq!(bindings.len(), 1);
            // X should be bound to either john or alice
            let x_binding = &bindings[0].1;
            assert!(
                x_binding.contains("john") || x_binding.contains("alice"),
                "X should be john or alice, got: {}",
                x_binding
            );
        }
    }

    #[test]
    fn test_query_with_multiple_variables() {
        // Setup database
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![atom_term("john"), atom_term("mary")],
        ))
        .unwrap();
        db.insert(make_atom(
            "parent",
            vec![atom_term("alice"), atom_term("bob")],
        ))
        .unwrap();

        // Query: ?- parent(X, Y).
        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var("X"), var("Y")],
            ))],
        };

        let results = evaluate_query(&query, &db);
        assert_eq!(results.len(), 2, "Should find 2 parent relationships");
    }

    #[test]
    fn test_query_with_join() {
        // Setup database
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![atom_term("john"), atom_term("mary")],
        ))
        .unwrap();
        db.insert(make_atom(
            "parent",
            vec![atom_term("mary"), atom_term("sue")],
        ))
        .unwrap();

        // Query: ?- parent(X, Y), parent(Y, Z).
        // Should find: X=john, Y=mary, Z=sue
        let query = Query {
            body: vec![
                Literal::Positive(make_atom("parent", vec![var("X"), var("Y")])),
                Literal::Positive(make_atom("parent", vec![var("Y"), var("Z")])),
            ],
        };

        let results = evaluate_query(&query, &db);
        assert_eq!(results.len(), 1, "Should find 1 grandparent relationship");
    }

    #[test]
    fn test_query_with_negation() {
        // Setup database
        let mut db = FactDatabase::new();
        db.insert(make_atom("person", vec![atom_term("john")]))
            .unwrap();
        db.insert(make_atom("person", vec![atom_term("mary")]))
            .unwrap();
        db.insert(make_atom("dead", vec![atom_term("john")]))
            .unwrap();

        // Query: ?- person(X), not dead(X).
        // Should find only mary
        let query = Query {
            body: vec![
                Literal::Positive(make_atom("person", vec![var("X")])),
                Literal::Negative(make_atom("dead", vec![var("X")])),
            ],
        };

        let results = evaluate_query(&query, &db);
        assert_eq!(results.len(), 1, "Should find 1 living person");
    }

    #[test]
    fn test_query_variables_extraction() {
        let query = Query {
            body: vec![
                Literal::Positive(make_atom("parent", vec![var("X"), var("Y")])),
                Literal::Positive(make_atom("age", vec![var("Y"), var("A")])),
            ],
        };

        let vars = query_variables(&query);
        assert_eq!(vars.len(), 3);
        assert!(vars.contains(&Intern::new("X".to_string())));
        assert!(vars.contains(&Intern::new("Y".to_string())));
        assert!(vars.contains(&Intern::new("A".to_string())));
    }
}
