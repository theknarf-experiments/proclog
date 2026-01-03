use super::*;
use internment::Intern;
use proclog_ast::Value;

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

    assert!(db.insert(fact.clone()).unwrap());
    assert_eq!(db.len(), 1);
    assert!(db.contains(&fact));
}

#[test]
fn test_insert_non_ground_fact_fails() {
    let mut db = FactDatabase::new();
    let fact = make_atom("parent", vec![var("X"), atom_const("mary")]);

    let err = db.insert(fact.clone()).unwrap_err();
    match &err {
        InsertError::NonGroundAtom(atom) => assert_eq!(atom, &fact),
    }
    assert!(format!("{}", err).contains("non-ground"));
    assert_eq!(db.len(), 0);
}

#[test]
fn test_insert_duplicate() {
    let mut db = FactDatabase::new();
    let fact = make_atom("parent", vec![atom_const("john"), atom_const("mary")]);

    assert!(db.insert(fact.clone()).unwrap());
    assert!(!db.insert(fact.clone()).unwrap()); // Duplicate returns false
    assert_eq!(db.len(), 1); // Still only one fact
}

#[test]
fn test_insert_multiple_facts() {
    let mut db = FactDatabase::new();

    db.insert(make_atom(
        "parent",
        vec![atom_const("john"), atom_const("mary")],
    ))
    .unwrap();
    db.insert(make_atom(
        "parent",
        vec![atom_const("mary"), atom_const("alice")],
    ))
    .unwrap();
    db.insert(make_atom("age", vec![atom_const("john"), int(42)]))
        .unwrap();

    assert_eq!(db.len(), 3);
}

// Query tests
#[test]
fn test_query_exact_match() {
    let mut db = FactDatabase::new();
    let fact = make_atom("parent", vec![atom_const("john"), atom_const("mary")]);
    db.insert(fact.clone()).unwrap();

    let results = db.query(&fact);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].len(), 0); // No variables, so empty substitution
}

#[test]
fn test_query_with_one_variable() {
    let mut db = FactDatabase::new();
    db.insert(make_atom(
        "parent",
        vec![atom_const("john"), atom_const("mary")],
    ))
    .unwrap();

    let pattern = make_atom("parent", vec![atom_const("john"), var("X")]);
    let results = db.query(&pattern);

    assert_eq!(results.len(), 1);
    let x = Intern::new("X".to_string());
    assert_eq!(results[0].get(&x), Some(&atom_const("mary")));
}

#[test]
fn test_query_with_multiple_variables() {
    let mut db = FactDatabase::new();
    db.insert(make_atom(
        "parent",
        vec![atom_const("john"), atom_const("mary")],
    ))
    .unwrap();

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
    db.insert(make_atom(
        "parent",
        vec![atom_const("john"), atom_const("mary")],
    ))
    .unwrap();
    db.insert(make_atom(
        "parent",
        vec![atom_const("john"), atom_const("bob")],
    ))
    .unwrap();
    db.insert(make_atom(
        "parent",
        vec![atom_const("alice"), atom_const("charlie")],
    ))
    .unwrap();

    let pattern = make_atom("parent", vec![atom_const("john"), var("X")]);
    let results = db.query(&pattern);

    assert_eq!(results.len(), 2);
}

#[test]
fn test_query_no_matches() {
    let mut db = FactDatabase::new();
    db.insert(make_atom(
        "parent",
        vec![atom_const("john"), atom_const("mary")],
    ))
    .unwrap();

    let pattern = make_atom("parent", vec![atom_const("alice"), var("X")]);
    let results = db.query(&pattern);

    assert_eq!(results.len(), 0);
}

#[test]
fn test_query_wrong_predicate() {
    let mut db = FactDatabase::new();
    db.insert(make_atom(
        "parent",
        vec![atom_const("john"), atom_const("mary")],
    ))
    .unwrap();

    let pattern = make_atom("child", vec![atom_const("john"), var("X")]);
    let results = db.query(&pattern);

    assert_eq!(results.len(), 0);
}

// get_by_predicate tests
#[test]
fn test_get_by_predicate() {
    let mut db = FactDatabase::new();
    db.insert(make_atom(
        "parent",
        vec![atom_const("john"), atom_const("mary")],
    ))
    .unwrap();
    db.insert(make_atom(
        "parent",
        vec![atom_const("bob"), atom_const("alice")],
    ))
    .unwrap();
    db.insert(make_atom("age", vec![atom_const("john"), int(42)]))
        .unwrap();

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
    db.insert(make_atom(
        "parent",
        vec![atom_const("john"), atom_const("mary")],
    ))
    .unwrap();
    db.insert(make_atom("age", vec![atom_const("john"), int(42)]))
        .unwrap();

    let all = db.all_facts();
    assert_eq!(all.len(), 2);
}

// is_ground tests
#[test]
fn test_is_ground_term() {
    assert!(is_ground_term(&atom_const("john")));
    assert!(is_ground_term(&int(42)));
    assert!(!is_ground_term(&var("X")));

    let compound = Term::Compound(Intern::new("f".to_string()), vec![atom_const("a"), int(1)]);
    assert!(is_ground_term(&compound));

    let compound_with_var =
        Term::Compound(Intern::new("f".to_string()), vec![var("X"), int(1)]);
    assert!(!is_ground_term(&compound_with_var));
}
