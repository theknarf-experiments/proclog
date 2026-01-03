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

use proclog_ast::{Query, Symbol};
use proclog_core::{FactDatabase, Substitution};
use proclog_grounding::satisfy_body;
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
    use proclog_ast::Term;

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
#[path = "../tests/unit/query_tests.rs"]
mod tests;
