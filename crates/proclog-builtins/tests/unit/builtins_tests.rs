use super::*;
use internment::Intern;
use proclog_ast::Term;

fn int(n: i64) -> Term {
    Term::Constant(Value::Integer(n))
}

fn float(f: f64) -> Term {
    Term::Constant(Value::Float(f))
}

fn var(name: &str) -> Term {
    Term::Variable(Intern::new(name.to_string()))
}

fn arith(op: &str, left: Term, right: Term) -> Term {
    Term::Compound(Intern::new(op.to_string()), vec![left, right])
}

fn assert_int(result: Option<Value>, expected: i64) {
    match result {
        Some(Value::Integer(v)) => assert_eq!(v, expected),
        other => panic!("expected integer {expected}, got {:?}", other),
    }
}

fn assert_float(result: Option<Value>, expected: f64) {
    match result {
        Some(Value::Float(v)) => {
            assert!((v - expected).abs() < 1e-9, "expected {expected}, got {v}")
        }
        other => panic!("expected float {expected}, got {:?}", other),
    }
}

#[test]
fn test_eval_arith_constants() {
    let subst = Substitution::new();

    assert_int(eval_arith(&int(42), &subst), 42);
    assert_int(eval_arith(&int(-10), &subst), -10);
    assert_int(eval_arith(&int(0), &subst), 0);
}

#[test]
fn test_eval_arith_addition() {
    let subst = Substitution::new();
    let expr = arith("+", int(3), int(4));

    assert_int(eval_arith(&expr, &subst), 7);
}

#[test]
fn test_eval_arith_subtraction() {
    let subst = Substitution::new();
    let expr = arith("-", int(10), int(3));

    assert_int(eval_arith(&expr, &subst), 7);
}

#[test]
fn test_eval_arith_multiplication() {
    let subst = Substitution::new();
    let expr = arith("*", int(6), int(7));

    assert_int(eval_arith(&expr, &subst), 42);
}

#[test]
fn test_eval_arith_division() {
    let subst = Substitution::new();
    let expr = arith("/", int(20), int(4));

    assert_int(eval_arith(&expr, &subst), 5);
}

#[test]
fn test_eval_arith_division_by_zero() {
    let subst = Substitution::new();
    let expr = arith("/", int(10), int(0));

    assert_eq!(eval_arith(&expr, &subst), None);
}

#[test]
fn test_eval_arith_modulo() {
    let subst = Substitution::new();
    let expr = arith("mod", int(17), int(5));

    assert_int(eval_arith(&expr, &subst), 2);
}

#[test]
fn test_eval_arith_nested() {
    let subst = Substitution::new();
    // (3 + 4) * 2 = 14
    let inner = arith("+", int(3), int(4));
    let expr = arith("*", inner, int(2));

    assert_int(eval_arith(&expr, &subst), 14);
}

#[test]
fn test_eval_arith_with_variables() {
    let mut subst = Substitution::new();
    subst.bind(Intern::new("X".to_string()), int(5));
    subst.bind(Intern::new("Y".to_string()), int(3));

    let expr = arith("+", var("X"), var("Y"));
    assert_int(eval_arith(&expr, &subst), 8);
}

#[test]
fn test_eval_arith_unbound_variable() {
    let subst = Substitution::new();
    let expr = arith("+", var("X"), int(5));

    assert_eq!(eval_arith(&expr, &subst), None);
}

#[test]
fn test_eval_arith_float_constants() {
    let subst = Substitution::new();

    assert_float(eval_arith(&float(3.5), &subst), 3.5);
}

#[test]
fn test_eval_arith_mixed_addition() {
    let subst = Substitution::new();
    let expr = arith("+", int(2), float(0.5));

    assert_float(eval_arith(&expr, &subst), 2.5);
}

#[test]
fn test_eval_arith_float_division() {
    let subst = Substitution::new();
    let expr = arith("/", float(7.5), int(2));

    assert_float(eval_arith(&expr, &subst), 3.75);
}

#[test]
fn test_eval_arith_division_by_zero_float() {
    let subst = Substitution::new();
    let expr = arith("/", float(3.0), float(0.0));

    assert_eq!(eval_arith(&expr, &subst), None);
}

#[test]
fn test_eval_arith_modulo_float_zero() {
    let subst = Substitution::new();
    let expr = arith("mod", float(3.0), float(0.0));

    assert_eq!(eval_arith(&expr, &subst), None);
}

#[test]
fn test_eval_arith_float_modulo() {
    let subst = Substitution::new();
    let expr = arith("mod", float(5.5), int(2));

    assert_float(eval_arith(&expr, &subst), 1.5);
}

#[test]
fn test_eval_comparison_eq() {
    let subst = Substitution::new();

    assert_eq!(
        eval_comparison(&CompOp::Eq, &int(5), &int(5), &subst),
        Some(true)
    );
    assert_eq!(
        eval_comparison(&CompOp::Eq, &int(5), &int(3), &subst),
        Some(false)
    );
}

#[test]
fn test_eval_comparison_lt() {
    let subst = Substitution::new();

    assert_eq!(
        eval_comparison(&CompOp::Lt, &int(3), &int(5), &subst),
        Some(true)
    );
    assert_eq!(
        eval_comparison(&CompOp::Lt, &int(5), &int(3), &subst),
        Some(false)
    );
    assert_eq!(
        eval_comparison(&CompOp::Lt, &int(5), &int(5), &subst),
        Some(false)
    );
}

#[test]
fn test_eval_comparison_with_arithmetic() {
    let subst = Substitution::new();
    // 3 + 4 = 7
    let left = arith("+", int(3), int(4));
    let right = int(7);

    assert_eq!(
        eval_comparison(&CompOp::Eq, &left, &right, &subst),
        Some(true)
    );
}

#[test]
fn test_eval_comparison_float_semantics() {
    let subst = Substitution::new();

    assert_eq!(
        eval_comparison(&CompOp::Eq, &float(2.5), &float(2.5), &subst),
        Some(true)
    );
    assert_eq!(
        eval_comparison(&CompOp::Gt, &float(3.1), &float(3.0), &subst),
        Some(true)
    );
    assert_eq!(
        eval_comparison(&CompOp::Lt, &float(3.1), &float(3.0), &subst),
        Some(false)
    );
}

#[test]
fn test_eval_comparison_mixed_numeric_types() {
    let subst = Substitution::new();

    assert_eq!(
        eval_comparison(&CompOp::Eq, &int(5), &float(5.0), &subst),
        Some(true)
    );
    assert_eq!(
        eval_comparison(&CompOp::Lt, &int(4), &float(4.1), &subst),
        Some(true)
    );
    assert_eq!(
        eval_comparison(&CompOp::Gt, &float(6.2), &int(6), &subst),
        Some(true)
    );
}

#[test]
fn test_parse_builtin_comparison() {
    let atom = Atom {
        predicate: Intern::new("=".to_string()),
        terms: vec![int(5), int(5)],
    };

    assert!(parse_builtin(&atom).is_some());
}

#[test]
fn test_parse_builtin_not_builtin() {
    let atom = Atom {
        predicate: Intern::new("parent".to_string()),
        terms: vec![
            Term::Constant(Value::Atom(Intern::new("john".to_string()))),
            Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
        ],
    };

    assert!(parse_builtin(&atom).is_none());
}

#[test]
fn test_parse_and_eval_ground_comparison() {
    // Test that we can parse and evaluate a ground comparison like "5 > 3"
    let atom = Atom {
        predicate: Intern::new(">".to_string()),
        terms: vec![int(5), int(3)],
    };

    let builtin = parse_builtin(&atom).expect("Should parse as built-in");
    let subst = Substitution::new();
    let result = eval_builtin(&builtin, &subst);

    assert_eq!(result, Some(true));
}

#[test]
fn test_parse_and_eval_ground_comparison_false() {
    // Test that we can parse and evaluate a ground comparison like "3 > 5"
    let atom = Atom {
        predicate: Intern::new(">".to_string()),
        terms: vec![int(3), int(5)],
    };

    let builtin = parse_builtin(&atom).expect("Should parse as built-in");
    let subst = Substitution::new();
    let result = eval_builtin(&builtin, &subst);

    assert_eq!(result, Some(false));
}
