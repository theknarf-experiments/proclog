use std::collections::{HashMap, HashSet};
use internment::Intern;
use crate::ast::{Atom, Term, Value, Symbol};
use crate::unification::{Substitution, unify_atoms};

/// A database of ground facts with efficient indexing
#[derive(Debug, Clone)]
pub struct FactDatabase {
    // Index: predicate -> set of ground atoms
    facts_by_predicate: HashMap<Symbol, HashSet<Atom>>,
}

impl FactDatabase {
    pub fn new() -> Self {
        FactDatabase {
            facts_by_predicate: HashMap::new(),
        }
    }

    /// Insert a ground fact into the database
    pub fn insert(&mut self, atom: Atom) -> bool {
        // Check that atom is ground (no variables)
        if !is_ground(&atom) {
            return false;
        }

        self.facts_by_predicate
            .entry(atom.predicate.clone())
            .or_insert_with(HashSet::new)
            .insert(atom)
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
    pub fn get_by_predicate(&self, predicate: &Symbol) -> Vec<&Atom> {
        if let Some(facts) = self.facts_by_predicate.get(predicate) {
            facts.iter().collect()
        } else {
            vec![]
        }
    }

    /// Get all facts in the database
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
        Term::Compound(_, args) => args.iter().all(is_ground_term),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper functions
    fn atom_const(name: &str) -> Term {
        Term::Constant(Value::Atom(Intern::new(name.to_string())))
    }

    fn int(n: i64) -> Term {
        Term::Constant(Value::Integer(n))
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

    // Basic tests
    #[test]
    fn test_database_new() {
        let db = FactDatabase::new();
        assert_eq!(db.len(), 0);
        assert!(db.is_empty());
    }

    #[test]
    fn test_insert_ground_fact() {
        let mut db = FactDatabase::new();
        let fact = make_atom("parent", vec![atom_const("john"), atom_const("mary")]);

        assert!(db.insert(fact.clone()));
        assert_eq!(db.len(), 1);
        assert!(db.contains(&fact));
    }

    #[test]
    fn test_insert_non_ground_fact_fails() {
        let mut db = FactDatabase::new();
        let fact = make_atom("parent", vec![var("X"), atom_const("mary")]);

        assert!(!db.insert(fact.clone()));
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_insert_duplicate() {
        let mut db = FactDatabase::new();
        let fact = make_atom("parent", vec![atom_const("john"), atom_const("mary")]);

        assert!(db.insert(fact.clone()));
        assert!(!db.insert(fact.clone())); // Duplicate returns false
        assert_eq!(db.len(), 1); // Still only one fact
    }

    #[test]
    fn test_insert_multiple_facts() {
        let mut db = FactDatabase::new();

        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));
        db.insert(make_atom("parent", vec![atom_const("mary"), atom_const("alice")]));
        db.insert(make_atom("age", vec![atom_const("john"), int(42)]));

        assert_eq!(db.len(), 3);
    }

    // Query tests
    #[test]
    fn test_query_exact_match() {
        let mut db = FactDatabase::new();
        let fact = make_atom("parent", vec![atom_const("john"), atom_const("mary")]);
        db.insert(fact.clone());

        let results = db.query(&fact);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 0); // No variables, so empty substitution
    }

    #[test]
    fn test_query_with_one_variable() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));

        let pattern = make_atom("parent", vec![atom_const("john"), var("X")]);
        let results = db.query(&pattern);

        assert_eq!(results.len(), 1);
        let x = Intern::new("X".to_string());
        assert_eq!(results[0].get(&x), Some(&atom_const("mary")));
    }

    #[test]
    fn test_query_with_multiple_variables() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));

        let pattern = make_atom("parent", vec![var("X"), var("Y")]);
        let results = db.query(&pattern);

        assert_eq!(results.len(), 1);
        let x = Intern::new("X".to_string());
        let y = Intern::new("Y".to_string());
        assert_eq!(results[0].get(&x), Some(&atom_const("john")));
        assert_eq!(results[0].get(&y), Some(&atom_const("mary")));
    }

    #[test]
    fn test_query_multiple_matches() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("bob")]));
        db.insert(make_atom("parent", vec![atom_const("alice"), atom_const("charlie")]));

        let pattern = make_atom("parent", vec![atom_const("john"), var("X")]);
        let results = db.query(&pattern);

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_no_matches() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));

        let pattern = make_atom("parent", vec![atom_const("alice"), var("X")]);
        let results = db.query(&pattern);

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_query_wrong_predicate() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));

        let pattern = make_atom("child", vec![atom_const("john"), var("X")]);
        let results = db.query(&pattern);

        assert_eq!(results.len(), 0);
    }

    // get_by_predicate tests
    #[test]
    fn test_get_by_predicate() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));
        db.insert(make_atom("parent", vec![atom_const("bob"), atom_const("alice")]));
        db.insert(make_atom("age", vec![atom_const("john"), int(42)]));

        let parent_pred = Intern::new("parent".to_string());
        let facts = db.get_by_predicate(&parent_pred);

        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn test_get_by_predicate_empty() {
        let db = FactDatabase::new();
        let pred = Intern::new("nonexistent".to_string());
        let facts = db.get_by_predicate(&pred);

        assert_eq!(facts.len(), 0);
    }

    // all_facts test
    #[test]
    fn test_all_facts() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));
        db.insert(make_atom("age", vec![atom_const("john"), int(42)]));

        let all = db.all_facts();
        assert_eq!(all.len(), 2);
    }

    // is_ground tests
    #[test]
    fn test_is_ground_term() {
        assert!(is_ground_term(&atom_const("john")));
        assert!(is_ground_term(&int(42)));
        assert!(!is_ground_term(&var("X")));

        let compound = Term::Compound(
            Intern::new("f".to_string()),
            vec![atom_const("a"), int(1)]
        );
        assert!(is_ground_term(&compound));

        let compound_with_var = Term::Compound(
            Intern::new("f".to_string()),
            vec![var("X"), int(1)]
        );
        assert!(!is_ground_term(&compound_with_var));
    }
}
