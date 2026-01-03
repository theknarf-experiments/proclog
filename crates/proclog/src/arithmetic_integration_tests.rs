//! Integration tests for arithmetic and built-in predicates
//!
//! These tests verify that arithmetic expressions and comparison operators
//! are parsed correctly.
//!
//! Note: Grounding and evaluation tests with built-ins are in src/grounding.rs as:
//! - test_satisfy_body_with_builtin_comparison
//! - test_ground_rule_with_builtin_comparison

#[cfg(test)]
mod arithmetic_integration_tests {
    use crate::ast::{Literal, Statement, Term};
    use crate::parser::{ParseError, SrcId};
    use internment::Intern;

    fn parse_program(input: &str) -> Result<crate::ast::Program, Vec<ParseError>> {
        crate::parser::parse_program(input, SrcId::empty())
    }

    #[test]
    fn test_parse_rule_with_comparison() {
        let program_text = "large(X) :- number(X), X > 5.";
        let result = parse_program(program_text);

        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);

        match &program.statements[0] {
            Statement::Rule(rule) => {
                assert_eq!(rule.body.len(), 2);

                // Second literal should be X > 5
                match &rule.body[1] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new(">".to_string()));
                        assert_eq!(atom.terms.len(), 2);
                    }
                    _ => panic!("Expected positive literal"),
                }
            }
            _ => panic!("Expected rule"),
        }
    }

    #[test]
    fn test_parse_rule_with_arithmetic() {
        let program_text = "double(X, Y) :- number(X), Y = X * 2.";
        let result = parse_program(program_text);

        assert!(result.is_ok());

        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Rule(rule) => {
                assert_eq!(rule.body.len(), 2);

                // Second literal should be Y = X * 2
                match &rule.body[1] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new("=".to_string()));
                        // Right side should be X * 2 (a compound term)
                        match &atom.terms[1] {
                            Term::Compound(op, args) => {
                                assert_eq!(*op, Intern::new("*".to_string()));
                                assert_eq!(args.len(), 2);
                            }
                            _ => panic!("Expected compound term for multiplication"),
                        }
                    }
                    _ => panic!("Expected positive literal"),
                }
            }
            _ => panic!("Expected rule"),
        }
    }

    #[test]
    fn test_parse_multiple_comparisons() {
        let program_text = "in_range(X) :- val(X), X >= 3, X <= 7.";
        let result = parse_program(program_text);

        assert!(result.is_ok());

        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Rule(rule) => {
                assert_eq!(rule.body.len(), 3);

                // Check that both comparisons are parsed
                match &rule.body[1] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new(">=".to_string()));
                    }
                    _ => panic!("Expected comparison"),
                }

                match &rule.body[2] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new("<=".to_string()));
                    }
                    _ => panic!("Expected comparison"),
                }
            }
            _ => panic!("Expected rule"),
        }
    }
}
