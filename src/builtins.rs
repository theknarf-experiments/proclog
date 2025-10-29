// Built-in predicates and arithmetic evaluation

use crate::ast::{Atom, Term, Value};
use crate::unification::Substitution;
use internment::Intern;

/// Arithmetic operators
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArithOp {
    Add, // +
    Sub, // -
    Mul, // *
    Div, // /
    Mod, // mod
}

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

/// Evaluate an arithmetic expression to an integer value
/// Returns None if the expression cannot be evaluated (contains unbound variables)
pub fn eval_arith(term: &Term, subst: &Substitution) -> Option<i64> {
    match term {
        Term::Constant(Value::Integer(n)) => Some(*n),

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
                    let left = eval_arith(&args[0], subst)?;
                    let right = eval_arith(&args[1], subst)?;
                    Some(left + right)
                }
                "-" if args.len() == 2 => {
                    let left = eval_arith(&args[0], subst)?;
                    let right = eval_arith(&args[1], subst)?;
                    Some(left - right)
                }
                "*" if args.len() == 2 => {
                    let left = eval_arith(&args[0], subst)?;
                    let right = eval_arith(&args[1], subst)?;
                    Some(left * right)
                }
                "/" if args.len() == 2 => {
                    let left = eval_arith(&args[0], subst)?;
                    let right = eval_arith(&args[1], subst)?;
                    if right == 0 {
                        None // Division by zero
                    } else {
                        Some(left / right)
                    }
                }
                "mod" if args.len() == 2 => {
                    let left = eval_arith(&args[0], subst)?;
                    let right = eval_arith(&args[1], subst)?;
                    if right == 0 {
                        None // Modulo by zero
                    } else {
                        Some(left % right)
                    }
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
    let left_val = eval_arith(left, subst);
    let right_val = eval_arith(right, subst);

    match (left_val, right_val) {
        (Some(l), Some(r)) => {
            let result = match op {
                CompOp::Eq => l == r,
                CompOp::Neq => l != r,
                CompOp::Lt => l < r,
                CompOp::Gt => l > r,
                CompOp::Lte => l <= r,
                CompOp::Gte => l >= r,
            };
            Some(result)
        }
        _ => None, // Cannot evaluate comparison
    }
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

    fn int(n: i64) -> Term {
        Term::Constant(Value::Integer(n))
    }

    fn var(name: &str) -> Term {
        Term::Variable(Intern::new(name.to_string()))
    }

    fn arith(op: &str, left: Term, right: Term) -> Term {
        Term::Compound(Intern::new(op.to_string()), vec![left, right])
    }

    #[test]
    fn test_eval_arith_constants() {
        let subst = Substitution::new();

        assert_eq!(eval_arith(&int(42), &subst), Some(42));
        assert_eq!(eval_arith(&int(-10), &subst), Some(-10));
        assert_eq!(eval_arith(&int(0), &subst), Some(0));
    }

    #[test]
    fn test_eval_arith_addition() {
        let subst = Substitution::new();
        let expr = arith("+", int(3), int(4));

        assert_eq!(eval_arith(&expr, &subst), Some(7));
    }

    #[test]
    fn test_eval_arith_subtraction() {
        let subst = Substitution::new();
        let expr = arith("-", int(10), int(3));

        assert_eq!(eval_arith(&expr, &subst), Some(7));
    }

    #[test]
    fn test_eval_arith_multiplication() {
        let subst = Substitution::new();
        let expr = arith("*", int(6), int(7));

        assert_eq!(eval_arith(&expr, &subst), Some(42));
    }

    #[test]
    fn test_eval_arith_division() {
        let subst = Substitution::new();
        let expr = arith("/", int(20), int(4));

        assert_eq!(eval_arith(&expr, &subst), Some(5));
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

        assert_eq!(eval_arith(&expr, &subst), Some(2));
    }

    #[test]
    fn test_eval_arith_nested() {
        let subst = Substitution::new();
        // (3 + 4) * 2 = 14
        let inner = arith("+", int(3), int(4));
        let expr = arith("*", inner, int(2));

        assert_eq!(eval_arith(&expr, &subst), Some(14));
    }

    #[test]
    fn test_eval_arith_with_variables() {
        let mut subst = Substitution::new();
        subst.bind(Intern::new("X".to_string()), int(5));
        subst.bind(Intern::new("Y".to_string()), int(3));

        let expr = arith("+", var("X"), var("Y"));
        assert_eq!(eval_arith(&expr, &subst), Some(8));
    }

    #[test]
    fn test_eval_arith_unbound_variable() {
        let subst = Substitution::new();
        let expr = arith("+", var("X"), int(5));

        assert_eq!(eval_arith(&expr, &subst), None);
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
