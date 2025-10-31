//! Built-in predicates and arithmetic evaluation
//!
//! This module handles built-in predicates like arithmetic operations and comparisons.
//! Built-ins are special predicates evaluated during grounding rather than via rules.
//!
//! # Supported Built-ins
//!
//! - **Arithmetic**: `+`, `-`, `*`, `/`, `mod`
//! - **Comparisons**: `=`, `!=`, `\=`, `<`, `>`, `<=`, `>=`
//!
//! # Usage
//!
//! Built-ins can appear in rule bodies:
//! ```ignore
//! // age(X, A), A > 18, Result = A + 1
//! ```
//!
//! During grounding, built-ins are evaluated to filter or compute values.

use crate::ast::{Atom, Term, Value};
use crate::unification::Substitution;

/// Comparison operators
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompOp {
    Eq,  // =
    Neq, // !=
    Lt,  // <
    Gt,  // >
    Lte, // <=
    Gte, // >=
}

/// Built-in predicates that can be evaluated directly
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BuiltIn {
    /// Arithmetic comparison: X = Y + Z, X < Y, etc.
    Comparison(CompOp, Term, Term),
    /// True (always succeeds)
    True,
    /// Fail (always fails)
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Numeric {
    Int(i64),
    Float(f64),
}

impl Numeric {
    fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Integer(i) => Some(Numeric::Int(*i)),
            Value::Float(f) => Some(Numeric::Float(*f)),
            _ => None,
        }
    }

    fn to_value(self) -> Value {
        match self {
            Numeric::Int(i) => Value::Integer(i),
            Numeric::Float(f) => Value::Float(f),
        }
    }

    fn promote(lhs: Numeric, rhs: Numeric) -> (Numeric, Numeric) {
        match (lhs, rhs) {
            (Numeric::Int(l), Numeric::Int(r)) => (Numeric::Int(l), Numeric::Int(r)),
            (Numeric::Int(l), Numeric::Float(r)) => (Numeric::Float(l as f64), Numeric::Float(r)),
            (Numeric::Float(l), Numeric::Int(r)) => (Numeric::Float(l), Numeric::Float(r as f64)),
            (Numeric::Float(l), Numeric::Float(r)) => (Numeric::Float(l), Numeric::Float(r)),
        }
    }

    fn add(self, other: Numeric) -> Numeric {
        match (self, other) {
            (Numeric::Int(l), Numeric::Int(r)) => Numeric::Int(l + r),
            (l, r) => Numeric::Float(l.to_f64() + r.to_f64()),
        }
    }

    fn sub(self, other: Numeric) -> Numeric {
        match (self, other) {
            (Numeric::Int(l), Numeric::Int(r)) => Numeric::Int(l - r),
            (l, r) => Numeric::Float(l.to_f64() - r.to_f64()),
        }
    }

    fn mul(self, other: Numeric) -> Numeric {
        match (self, other) {
            (Numeric::Int(l), Numeric::Int(r)) => Numeric::Int(l * r),
            (l, r) => Numeric::Float(l.to_f64() * r.to_f64()),
        }
    }

    fn div(self, other: Numeric) -> Option<Numeric> {
        match (self, other) {
            (Numeric::Int(_), Numeric::Int(0)) => None,
            (Numeric::Int(l), Numeric::Int(r)) => Some(Numeric::Int(l / r)),
            (l, r) => {
                let divisor = r.to_f64();
                if divisor == 0.0 {
                    None
                } else {
                    Some(Numeric::Float(l.to_f64() / divisor))
                }
            }
        }
    }

    fn modulo(self, other: Numeric) -> Option<Numeric> {
        match (self, other) {
            (Numeric::Int(_), Numeric::Int(0)) => None,
            (Numeric::Int(l), Numeric::Int(r)) => Some(Numeric::Int(l % r)),
            (l, r) => {
                let divisor = r.to_f64();
                if divisor == 0.0 {
                    None
                } else {
                    Some(Numeric::Float(l.to_f64() % divisor))
                }
            }
        }
    }

    fn to_f64(self) -> f64 {
        match self {
            Numeric::Int(i) => i as f64,
            Numeric::Float(f) => f,
        }
    }
}

/// Evaluate an arithmetic expression to a numeric value
/// Returns None if the expression cannot be evaluated (contains unbound variables)
pub fn eval_arith(term: &Term, subst: &Substitution) -> Option<Value> {
    match term {
        Term::Constant(value) => Numeric::from_value(value).map(Numeric::to_value),

        Term::Variable(var) => {
            // Look up variable in substitution
            if let Some(bound_term) = subst.get(var) {
                eval_arith(bound_term, subst)
            } else {
                None // Unbound variable
            }
        }

        Term::Compound(functor, args) => {
            // Check if this is an arithmetic operator
            match functor.as_ref().as_str() {
                "+" if args.len() == 2 => {
                    let left_val = eval_arith(&args[0], subst)?;
                    let right_val = eval_arith(&args[1], subst)?;
                    let left = Numeric::from_value(&left_val)?;
                    let right = Numeric::from_value(&right_val)?;
                    Some(left.add(right).to_value())
                }
                "-" if args.len() == 2 => {
                    let left_val = eval_arith(&args[0], subst)?;
                    let right_val = eval_arith(&args[1], subst)?;
                    let left = Numeric::from_value(&left_val)?;
                    let right = Numeric::from_value(&right_val)?;
                    Some(left.sub(right).to_value())
                }
                "*" if args.len() == 2 => {
                    let left_val = eval_arith(&args[0], subst)?;
                    let right_val = eval_arith(&args[1], subst)?;
                    let left = Numeric::from_value(&left_val)?;
                    let right = Numeric::from_value(&right_val)?;
                    Some(left.mul(right).to_value())
                }
                "/" if args.len() == 2 => {
                    let left_val = eval_arith(&args[0], subst)?;
                    let right_val = eval_arith(&args[1], subst)?;
                    let left = Numeric::from_value(&left_val)?;
                    let right = Numeric::from_value(&right_val)?;
                    left.div(right).map(Numeric::to_value)
                }
                "mod" if args.len() == 2 => {
                    let left_val = eval_arith(&args[0], subst)?;
                    let right_val = eval_arith(&args[1], subst)?;
                    let left = Numeric::from_value(&left_val)?;
                    let right = Numeric::from_value(&right_val)?;
                    left.modulo(right).map(Numeric::to_value)
                }
                _ => None, // Not an arithmetic operator
            }
        }

        _ => None, // Cannot evaluate this type of term arithmetically
    }
}

/// Evaluate a comparison built-in predicate
/// Returns true if the comparison holds, false otherwise
/// Returns None if terms cannot be evaluated
pub fn eval_comparison(
    op: &CompOp,
    left: &Term,
    right: &Term,
    subst: &Substitution,
) -> Option<bool> {
    // Try to evaluate both sides as arithmetic expressions
    let left_val = eval_arith(left, subst)?;
    let right_val = eval_arith(right, subst)?;
    let left_num = Numeric::from_value(&left_val)?;
    let right_num = Numeric::from_value(&right_val)?;
    let (left_num, right_num) = Numeric::promote(left_num, right_num);

    let result = match op {
        CompOp::Eq => match (left_num, right_num) {
            (Numeric::Int(l), Numeric::Int(r)) => l == r,
            (l, r) => l.to_f64() == r.to_f64(),
        },
        CompOp::Neq => match (left_num, right_num) {
            (Numeric::Int(l), Numeric::Int(r)) => l != r,
            (l, r) => l.to_f64() != r.to_f64(),
        },
        CompOp::Lt => match (left_num, right_num) {
            (Numeric::Int(l), Numeric::Int(r)) => l < r,
            (l, r) => l.to_f64() < r.to_f64(),
        },
        CompOp::Gt => match (left_num, right_num) {
            (Numeric::Int(l), Numeric::Int(r)) => l > r,
            (l, r) => l.to_f64() > r.to_f64(),
        },
        CompOp::Lte => match (left_num, right_num) {
            (Numeric::Int(l), Numeric::Int(r)) => l <= r,
            (l, r) => l.to_f64() <= r.to_f64(),
        },
        CompOp::Gte => match (left_num, right_num) {
            (Numeric::Int(l), Numeric::Int(r)) => l >= r,
            (l, r) => l.to_f64() >= r.to_f64(),
        },
    };

    Some(result)
}

/// Evaluate a built-in predicate
/// Returns true if it succeeds, false if it fails, None if it cannot be evaluated
pub fn eval_builtin(builtin: &BuiltIn, subst: &Substitution) -> Option<bool> {
    match builtin {
        BuiltIn::Comparison(op, left, right) => eval_comparison(op, left, right, subst),
        BuiltIn::True => Some(true),
        BuiltIn::Fail => Some(false),
    }
}

/// Parse an atom as a potential built-in predicate
/// Returns Some(BuiltIn) if this is a built-in, None otherwise
pub fn parse_builtin(atom: &Atom) -> Option<BuiltIn> {
    let pred = atom.predicate.as_ref().as_str();

    match pred {
        "=" if atom.terms.len() == 2 => Some(BuiltIn::Comparison(
            CompOp::Eq,
            atom.terms[0].clone(),
            atom.terms[1].clone(),
        )),
        "!=" | "\\=" if atom.terms.len() == 2 => Some(BuiltIn::Comparison(
            CompOp::Neq,
            atom.terms[0].clone(),
            atom.terms[1].clone(),
        )),
        "<" if atom.terms.len() == 2 => Some(BuiltIn::Comparison(
            CompOp::Lt,
            atom.terms[0].clone(),
            atom.terms[1].clone(),
        )),
        ">" if atom.terms.len() == 2 => Some(BuiltIn::Comparison(
            CompOp::Gt,
            atom.terms[0].clone(),
            atom.terms[1].clone(),
        )),
        "<=" | "=<" if atom.terms.len() == 2 => Some(BuiltIn::Comparison(
            CompOp::Lte,
            atom.terms[0].clone(),
            atom.terms[1].clone(),
        )),
        ">=" if atom.terms.len() == 2 => Some(BuiltIn::Comparison(
            CompOp::Gte,
            atom.terms[0].clone(),
            atom.terms[1].clone(),
        )),
        "true" if atom.terms.is_empty() => Some(BuiltIn::True),
        "fail" | "false" if atom.terms.is_empty() => Some(BuiltIn::Fail),
        _ => None, // Not a built-in
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Term;
    use internment::Intern;

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
}
