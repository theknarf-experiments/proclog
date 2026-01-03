use super::*;
use internment::Intern;
use proclog_ast::Value;

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

    subst.bind(x, int(42));
    assert!(subst.contains(&x));
    assert_eq!(subst.get(&x), Some(&int(42)));
}

#[test]
fn test_substitution_apply_variable() {
    let mut subst = Substitution::new();
    let x = Intern::new("X".to_string());

    subst.bind(x, int(42));

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

    subst.bind(x, int(42));

    let term = compound("f", vec![var("X"), atom("a")]);
    let result = subst.apply(&term);

    assert_eq!(result, compound("f", vec![int(42), atom("a")]));
}

#[test]
fn test_substitution_apply_transitive() {
    let mut subst = Substitution::new();
    let x = Intern::new("X".to_string());
    let y = Intern::new("Y".to_string());

    subst.bind(x, var("Y"));
    subst.bind(y, int(42));

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

    subst.bind(x, int(42));

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
    assert!(!unify(
        &var("X"),
        &compound("f", vec![var("X")]),
        &mut subst
    ));
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
    use proclog_ast::Atom;

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
    use proclog_ast::Atom;

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
    use proclog_ast::Atom;

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
