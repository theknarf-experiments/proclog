use std::collections::HashMap;
use internment::Intern;
use crate::ast::{Term, Value, Atom, Symbol};

/// A substitution maps variables to terms
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Substitution {
    bindings: HashMap<Symbol, Term>,
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
    pub fn contains(&self, var: &Symbol) -> bool {
        self.bindings.contains_key(var)
    }

    /// Get the number of bindings
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Check if substitution is empty
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
                Term::Compound(functor.clone(), new_args)
            }
        }
    }

    /// Apply substitution to an atom
    pub fn apply_atom(&self, atom: &Atom) -> Atom {
        Atom {
            predicate: atom.predicate.clone(),
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

        // Variable unifies with anything (occurs check handled)
        (Term::Variable(var), t) | (t, Term::Variable(var)) => {
            // Occurs check: make sure var doesn't occur in t
            if occurs_check(var, t) {
                false
            } else {
                subst.bind(var.clone(), t.clone());
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
        Term::Range(_, _) => false,  // Ranges don't contain variables
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
mod tests {
    use super::*;

    // Helper functions for creating terms in tests
    fn var(name: &str) -> Term {
        Term::Variable(Intern::new(name.to_string()))
    }

    fn int(n: i64) -> Term {
        Term::Constant(Value::Integer(n))
    }

    fn atom(name: &str) -> Term {
        Term::Constant(Value::Atom(Intern::new(name.to_string())))
    }

    fn compound(functor: &str, args: Vec<Term>) -> Term {
        Term::Compound(Intern::new(functor.to_string()), args)
    }

    // Substitution tests
    #[test]
    fn test_substitution_new() {
        let subst = Substitution::new();
        assert_eq!(subst.bindings.len(), 0);
    }

    #[test]
    fn test_substitution_bind_and_get() {
        let mut subst = Substitution::new();
        let x = Intern::new("X".to_string());

        subst.bind(x.clone(), int(42));
        assert!(subst.contains(&x));
        assert_eq!(subst.get(&x), Some(&int(42)));
    }

    #[test]
    fn test_substitution_apply_variable() {
        let mut subst = Substitution::new();
        let x = Intern::new("X".to_string());

        subst.bind(x.clone(), int(42));

        let result = subst.apply(&var("X"));
        assert_eq!(result, int(42));
    }

    #[test]
    fn test_substitution_apply_unbound_variable() {
        let subst = Substitution::new();
        let result = subst.apply(&var("Y"));
        assert_eq!(result, var("Y"));
    }

    #[test]
    fn test_substitution_apply_compound() {
        let mut subst = Substitution::new();
        let x = Intern::new("X".to_string());

        subst.bind(x.clone(), int(42));

        let term = compound("f", vec![var("X"), atom("a")]);
        let result = subst.apply(&term);

        assert_eq!(result, compound("f", vec![int(42), atom("a")]));
    }

    #[test]
    fn test_substitution_apply_transitive() {
        let mut subst = Substitution::new();
        let x = Intern::new("X".to_string());
        let y = Intern::new("Y".to_string());

        subst.bind(x.clone(), var("Y"));
        subst.bind(y.clone(), int(42));

        let result = subst.apply(&var("X"));
        assert_eq!(result, int(42));
    }

    // Unification tests - Constants
    #[test]
    fn test_unify_identical_constants() {
        let mut subst = Substitution::new();
        assert!(unify(&int(42), &int(42), &mut subst));
        assert_eq!(subst.bindings.len(), 0);
    }

    #[test]
    fn test_unify_different_constants() {
        let mut subst = Substitution::new();
        assert!(!unify(&int(42), &int(43), &mut subst));
    }

    #[test]
    fn test_unify_different_types() {
        let mut subst = Substitution::new();
        assert!(!unify(&int(42), &atom("john"), &mut subst));
    }

    // Unification tests - Variables
    #[test]
    fn test_unify_variable_with_constant() {
        let mut subst = Substitution::new();
        assert!(unify(&var("X"), &int(42), &mut subst));

        let x = Intern::new("X".to_string());
        assert_eq!(subst.get(&x), Some(&int(42)));
    }

    #[test]
    fn test_unify_constant_with_variable() {
        let mut subst = Substitution::new();
        assert!(unify(&int(42), &var("X"), &mut subst));

        let x = Intern::new("X".to_string());
        assert_eq!(subst.get(&x), Some(&int(42)));
    }

    #[test]
    fn test_unify_two_variables() {
        let mut subst = Substitution::new();
        assert!(unify(&var("X"), &var("Y"), &mut subst));

        // One should be bound to the other
        assert_eq!(subst.bindings.len(), 1);
    }

    #[test]
    fn test_unify_variable_already_bound() {
        let mut subst = Substitution::new();
        let x = Intern::new("X".to_string());

        subst.bind(x.clone(), int(42));

        // Unifying X with 42 should succeed (already bound to 42)
        assert!(unify(&var("X"), &int(42), &mut subst));

        // Unifying X with 43 should fail (bound to different value)
        assert!(!unify(&var("X"), &int(43), &mut subst));
    }

    // Unification tests - Compound terms
    #[test]
    fn test_unify_identical_compound() {
        let mut subst = Substitution::new();
        let term = compound("f", vec![atom("a"), atom("b")]);
        assert!(unify(&term, &term, &mut subst));
    }

    #[test]
    fn test_unify_compound_different_functors() {
        let mut subst = Substitution::new();
        let t1 = compound("f", vec![atom("a")]);
        let t2 = compound("g", vec![atom("a")]);
        assert!(!unify(&t1, &t2, &mut subst));
    }

    #[test]
    fn test_unify_compound_different_arity() {
        let mut subst = Substitution::new();
        let t1 = compound("f", vec![atom("a")]);
        let t2 = compound("f", vec![atom("a"), atom("b")]);
        assert!(!unify(&t1, &t2, &mut subst));
    }

    #[test]
    fn test_unify_compound_with_variables() {
        let mut subst = Substitution::new();
        let t1 = compound("f", vec![var("X"), atom("b")]);
        let t2 = compound("f", vec![atom("a"), var("Y")]);

        assert!(unify(&t1, &t2, &mut subst));

        let x = Intern::new("X".to_string());
        let y = Intern::new("Y".to_string());

        assert_eq!(subst.get(&x), Some(&atom("a")));
        assert_eq!(subst.get(&y), Some(&atom("b")));
    }

    #[test]
    fn test_unify_nested_compound() {
        let mut subst = Substitution::new();
        let t1 = compound("f", vec![compound("g", vec![var("X")])]);
        let t2 = compound("f", vec![compound("g", vec![int(42)])]);

        assert!(unify(&t1, &t2, &mut subst));

        let x = Intern::new("X".to_string());
        assert_eq!(subst.get(&x), Some(&int(42)));
    }

    // Occurs check tests
    #[test]
    fn test_occurs_check_simple() {
        let mut subst = Substitution::new();
        // X = f(X) should fail (infinite structure)
        assert!(!unify(&var("X"), &compound("f", vec![var("X")]), &mut subst));
    }

    #[test]
    fn test_occurs_check_nested() {
        let mut subst = Substitution::new();
        // X = f(g(X)) should fail
        let term = compound("f", vec![compound("g", vec![var("X")])]);
        assert!(!unify(&var("X"), &term, &mut subst));
    }

    // Atom unification tests
    #[test]
    fn test_unify_atoms_simple() {
        use crate::ast::Atom;

        let mut subst = Substitution::new();
        let atom1 = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![atom("john"), var("X")],
        };
        let atom2 = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![var("Y"), atom("mary")],
        };

        assert!(unify_atoms(&atom1, &atom2, &mut subst));

        let x = Intern::new("X".to_string());
        let y = Intern::new("Y".to_string());

        assert_eq!(subst.get(&x), Some(&atom("mary")));
        assert_eq!(subst.get(&y), Some(&atom("john")));
    }

    #[test]
    fn test_unify_atoms_different_predicates() {
        use crate::ast::Atom;

        let mut subst = Substitution::new();
        let atom1 = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![atom("john")],
        };
        let atom2 = Atom {
            predicate: Intern::new("child".to_string()),
            terms: vec![atom("john")],
        };

        assert!(!unify_atoms(&atom1, &atom2, &mut subst));
    }

    #[test]
    fn test_unify_atoms_different_arity() {
        use crate::ast::Atom;

        let mut subst = Substitution::new();
        let atom1 = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![atom("john")],
        };
        let atom2 = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![atom("john"), atom("mary")],
        };

        assert!(!unify_atoms(&atom1, &atom2, &mut subst));
    }
}
