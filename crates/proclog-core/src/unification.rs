//! Unification algorithm (Robinson's unification)
//!
//! This module implements first-order unification, which finds substitutions
//! that make two terms equal. This is a core operation in logic programming.
//!
//! # Algorithm
//!
//! Implements Robinson's unification algorithm with occurs check to prevent
//! infinite structures.
//!
//! # Example
//!
//! ```ignore
//! // Unify parent(X, mary) with parent(john, Y)
//! // Result: X=john, Y=mary
//! let sub = unify_atoms(&pattern1, &pattern2);
//! ```

use proclog_ast::{Atom, Symbol, Term};
use std::collections::HashMap;

/// A substitution maps variables to terms
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Substitution {
    bindings: HashMap<Symbol, Term>,
}

impl Default for Substitution {
    fn default() -> Self {
        Self::new()
    }
}

impl Substitution {
    pub fn new() -> Self {
        Substitution {
            bindings: HashMap::new(),
        }
    }

    /// Bind a variable to a term
    pub fn bind(&mut self, var: Symbol, term: Term) {
        self.bindings.insert(var, term);
    }

    /// Get the binding for a variable
    pub fn get(&self, var: &Symbol) -> Option<&Term> {
        self.bindings.get(var)
    }

    /// Check if a variable is bound
    #[allow(dead_code)]
    pub fn contains(&self, var: &Symbol) -> bool {
        self.bindings.contains_key(var)
    }

    /// Get the number of bindings
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Check if substitution is empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Iterate over bindings
    pub fn iter(&self) -> impl Iterator<Item = (&Symbol, &Term)> {
        self.bindings.iter()
    }

    /// Apply substitution to a term
    pub fn apply(&self, term: &Term) -> Term {
        match term {
            Term::Variable(var) => {
                if let Some(bound_term) = self.get(var) {
                    // Recursively apply in case bound term contains variables
                    self.apply(bound_term)
                } else {
                    term.clone()
                }
            }
            Term::Constant(_) => term.clone(),
            Term::Range(_, _) => {
                // Ranges should be expanded before unification
                // For now, just return the range as-is
                term.clone()
            }
            Term::Compound(functor, args) => {
                let new_args = args.iter().map(|arg| self.apply(arg)).collect();
                Term::Compound(*functor, new_args)
            }
        }
    }

    /// Apply substitution to an atom
    pub fn apply_atom(&self, atom: &Atom) -> Atom {
        Atom {
            predicate: atom.predicate,
            terms: atom.terms.iter().map(|t| self.apply(t)).collect(),
        }
    }
}

/// Unify two terms, returning a substitution if successful
/// This implements Robinson's Unification Algorithm
pub fn unify(term1: &Term, term2: &Term, subst: &mut Substitution) -> bool {
    // Apply current substitution to both terms
    let t1 = subst.apply(term1);
    let t2 = subst.apply(term2);

    match (&t1, &t2) {
        // Two identical terms unify trivially
        (Term::Constant(v1), Term::Constant(v2)) if v1 == v2 => true,

        // Anonymous variable "_" unifies with anything without binding
        (Term::Variable(var), _) if var.as_ref() == "_" => true,
        (_, Term::Variable(var)) if var.as_ref() == "_" => true,

        // Variable unifies with anything (occurs check handled)
        (Term::Variable(var), t) | (t, Term::Variable(var)) => {
            // Occurs check: make sure var doesn't occur in t
            if occurs_check(var, t) {
                false
            } else {
                subst.bind(*var, t.clone());
                true
            }
        }

        // Compound terms unify if functors match and all arguments unify
        (Term::Compound(f1, args1), Term::Compound(f2, args2)) => {
            if f1 != f2 || args1.len() != args2.len() {
                false
            } else {
                // Unify all arguments in order
                for (arg1, arg2) in args1.iter().zip(args2.iter()) {
                    if !unify(arg1, arg2, subst) {
                        return false;
                    }
                }
                true
            }
        }

        // Everything else fails to unify
        _ => false,
    }
}

/// Occurs check: does variable occur in term?
fn occurs_check(var: &Symbol, term: &Term) -> bool {
    match term {
        Term::Variable(v) => v == var,
        Term::Constant(_) => false,
        Term::Range(_, _) => false, // Ranges don't contain variables
        Term::Compound(_, args) => args.iter().any(|arg| occurs_check(var, arg)),
    }
}

/// Unify two atoms
pub fn unify_atoms(atom1: &Atom, atom2: &Atom, subst: &mut Substitution) -> bool {
    if atom1.predicate != atom2.predicate || atom1.terms.len() != atom2.terms.len() {
        return false;
    }

    for (t1, t2) in atom1.terms.iter().zip(atom2.terms.iter()) {
        if !unify(t1, t2, subst) {
            return false;
        }
    }

    true
}

#[cfg(test)]
#[path = "../tests/unit/unification_tests.rs"]
mod tests;
