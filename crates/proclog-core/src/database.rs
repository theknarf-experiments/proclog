//! Fact database with efficient indexing and querying
//!
//! This module provides a database for storing and querying ground facts.
//! Facts are indexed by predicate name for efficient lookup.
//!
//! # Features
//!
//! - **Indexing**: Facts are indexed by predicate for O(1) lookup
//! - **Ground facts only**: Only fully ground atoms (no variables) can be stored
//! - **Unification-based querying**: Query with patterns containing variables
//! - **Set semantics**: Duplicate facts are automatically deduplicated
//!
//! # Example
//!
//! ```ignore
//! let mut db = FactDatabase::new();
//! db.insert(ground_fact).unwrap();
//! let results = db.query(&pattern_with_variables);
//! ```

use crate::unification::{unify_atoms, Substitution};
use proclog_ast::{Atom, Symbol, Term};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

#[cfg(any(test, feature = "test-utils"))]
use std::cell::Cell;

/// A database of ground facts with efficient indexing
#[derive(Debug, Clone)]
pub struct FactDatabase {
    // Index: predicate -> set of ground atoms
    facts_by_predicate: HashMap<Symbol, HashSet<Atom>>,
}

impl Default for FactDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Error type for failed fact insertions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertError {
    /// Attempted to insert an atom containing variables
    NonGroundAtom(Atom),
}

impl fmt::Display for InsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InsertError::NonGroundAtom(atom) => {
                write!(f, "cannot insert non-ground atom: {:?}", atom)
            }
        }
    }
}

impl Error for InsertError {}

#[cfg(any(test, feature = "test-utils"))]
thread_local! {
    static TRACK_GROUND_QUERIES: Cell<bool> = const { Cell::new(false) };
    static GROUND_QUERY_COUNT: Cell<usize> = const { Cell::new(0) };
}

impl FactDatabase {
    pub fn new() -> Self {
        FactDatabase {
            facts_by_predicate: HashMap::new(),
        }
    }

    /// Insert a ground fact into the database
    pub fn insert(&mut self, atom: Atom) -> Result<bool, InsertError> {
        // Check that atom is ground (no variables)
        if !is_ground(&atom) {
            return Err(InsertError::NonGroundAtom(atom));
        }

        Ok(self
            .facts_by_predicate
            .entry(atom.predicate)
            .or_default()
            .insert(atom))
    }

    /// Merge another fact database into this one, consuming the other database.
    pub fn absorb(&mut self, mut other: FactDatabase) {
        for (predicate, mut facts) in other.facts_by_predicate.drain() {
            self.facts_by_predicate
                .entry(predicate)
                .or_default()
                .extend(facts.drain());
        }
    }

    /// Check if a fact exists in the database
    pub fn contains(&self, atom: &Atom) -> bool {
        if let Some(facts) = self.facts_by_predicate.get(&atom.predicate) {
            facts.contains(atom)
        } else {
            false
        }
    }

    /// Query for facts matching a pattern (may contain variables)
    /// Returns all substitutions that make the pattern match facts in the database
    pub fn query(&self, pattern: &Atom) -> Vec<Substitution> {
        #[cfg(any(test, feature = "test-utils"))]
        TRACK_GROUND_QUERIES.with(|flag| {
            if flag.get() && is_ground(pattern) {
                GROUND_QUERY_COUNT.with(|count| count.set(count.get() + 1));
            }
        });

        let mut results = Vec::new();

        // Get all facts with the same predicate
        if let Some(facts) = self.facts_by_predicate.get(&pattern.predicate) {
            for fact in facts {
                let mut subst = Substitution::new();
                if unify_atoms(pattern, fact, &mut subst) {
                    results.push(subst);
                }
            }
        }

        results
    }

    /// Get all facts with a specific predicate
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn get_by_predicate(&self, predicate: &Symbol) -> Vec<&Atom> {
        if let Some(facts) = self.facts_by_predicate.get(predicate) {
            facts.iter().collect()
        } else {
            vec![]
        }
    }

    /// Get all facts in the database
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn all_facts(&self) -> Vec<&Atom> {
        self.facts_by_predicate
            .values()
            .flat_map(|facts| facts.iter())
            .collect()
    }

    /// Count total number of facts
    pub fn len(&self) -> usize {
        self.facts_by_predicate
            .values()
            .map(|facts| facts.len())
            .sum()
    }

    /// Check if database is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub fn track_ground_queries() -> GroundQueryTracker {
        GroundQueryTracker::new()
    }
}

#[cfg(any(test, feature = "test-utils"))]
pub struct GroundQueryTracker {
    _private: (),
}

#[cfg(any(test, feature = "test-utils"))]
impl GroundQueryTracker {
    fn new() -> Self {
        TRACK_GROUND_QUERIES.with(|flag| flag.set(true));
        GROUND_QUERY_COUNT.with(|count| count.set(0));
        GroundQueryTracker { _private: () }
    }

    pub fn count(&self) -> usize {
        GROUND_QUERY_COUNT.with(|count| count.get())
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl Drop for GroundQueryTracker {
    fn drop(&mut self) {
        TRACK_GROUND_QUERIES.with(|flag| flag.set(false));
    }
}

/// Check if an atom is ground (contains no variables)
fn is_ground(atom: &Atom) -> bool {
    atom.terms.iter().all(is_ground_term)
}

/// Check if a term is ground (contains no variables)
fn is_ground_term(term: &Term) -> bool {
    match term {
        Term::Variable(_) => false,
        Term::Constant(_) => true,
        Term::Range(_, _) => true, // Ranges are ground (no variables)
        Term::Compound(_, args) => args.iter().all(is_ground_term),
    }
}

#[cfg(test)]
#[path = "../tests/unit/database_tests.rs"]
mod tests;
