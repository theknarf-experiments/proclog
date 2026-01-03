    use super::*;

    fn parse_with<T>(
        parser: impl Parser<Token, T, Error = ParserError>,
        input: &str,
    ) -> Result<T, Vec<ParseError>> {
        super::parse_with(parser, input, SrcId::empty())
    }

    fn parse_program(input: &str) -> Result<Program, Vec<ParseError>> {
        super::parse_program(input, SrcId::empty())
    }

    fn parse_query(input: &str) -> Result<Query, Vec<ParseError>> {
        super::parse_query(input, SrcId::empty())
    }

    // Helper functions for tests
    fn int_term(n: i64) -> Term {
        Term::Constant(Value::Integer(n))
    }

    fn atom_term(name: &str) -> Term {
        Term::Constant(Value::Atom(Intern::new(name.to_string())))
    }

    fn range(start: i64, end: i64) -> Term {
        Term::Range(Box::new(int_term(start)), Box::new(int_term(end)))
    }

    fn range_const(start: i64, end_name: &str) -> Term {
        Term::Range(Box::new(int_term(start)), Box::new(atom_term(end_name)))
    }

    #[test]
    fn test_string_literal_escape_sequences() {
        let input = "\"escaped\\\" newline\\n tab\\t backslash\\\\\"";
        let result = lex(input);
        assert!(
            result.is_ok(),
            "Failed to parse escapes: {:?}",
            result.err()
        );
        let tokens = result.unwrap();
        assert_eq!(tokens.len(), 1);
        match &tokens[0].0 {
            Token::String(parsed) => {
                assert_eq!(parsed, "escaped\" newline\n tab\t backslash\\");
            }
            other => panic!("Expected string token, got {:?}", other),
        }
    }

    #[test]
    fn test_string_literal_rejects_raw_newline() {
        let input = "\"line\nnext\"";
        let result = lex(input);
        assert!(
            result.is_err(),
            "String literal with raw newline should be rejected"
        );
    }

    #[test]
    fn test_string_literal_rejects_invalid_escape() {
        let input = "\"invalid\\xescape\"";
        let result = lex(input);
        assert!(
            result.is_err(),
            "String literal with invalid escape should be rejected"
        );
    }

    #[test]
    fn test_string_literal_in_ast_has_interpreted_value() {
        let input = "value(\"line\\nnext\").";
        let result = parse_program(input);
        assert!(
            result.is_ok(),
            "Failed to parse program: {:?}",
            result.err()
        );
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.terms.len(), 1);
                match &fact.atom.terms[0] {
                    Term::Constant(Value::String(s)) => {
                        assert_eq!(s.as_ref(), "line\nnext");
                    }
                    other => panic!("Expected string constant, got {:?}", other),
                }
            }
            other => panic!("Expected fact statement, got {:?}", other),
        }
    }

    // Term parsing tests - Datatypes

    // Integers
    #[test]
    fn test_parse_positive_integer() {
        let result = parse_with(term(), "42");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Integer(42)));
    }

    #[test]
    fn test_parse_negative_integer() {
        let result = parse_with(term(), "-42");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Integer(-42)));
    }

    #[test]
    fn test_parse_zero() {
        let result = parse_with(term(), "0");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Integer(0)));
    }

    // Ranges
    #[test]
    fn test_parse_range_simple() {
        let result = parse_with(term(), "1..10");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), range(1, 10));
    }

    #[test]
    fn test_parse_range_zero_start() {
        let result = parse_with(term(), "0..5");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), range(0, 5));
    }

    #[test]
    fn test_parse_range_large() {
        let result = parse_with(term(), "1..100");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), range(1, 100));
    }

    #[test]
    fn test_parse_range_same_values() {
        let result = parse_with(term(), "5..5");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), range(5, 5));
    }

    #[test]
    fn test_parse_range_with_constant() {
        let result = parse_with(term(), "1..width");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), range_const(1, "width"));
    }

    #[test]
    fn test_parse_range_both_constants() {
        let result = parse_with(term(), "min..max");
        assert!(result.is_ok());
        let expected = Term::Range(Box::new(atom_term("min")), Box::new(atom_term("max")));
        assert_eq!(result.unwrap(), expected);
    }

    // Floats
    #[test]
    #[allow(clippy::approx_constant)]
    fn test_parse_positive_float() {
        let result = parse_with(term(), "3.14");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Float(3.14)));
    }

    #[test]
    fn test_parse_negative_float() {
        let result = parse_with(term(), "-2.5");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Float(-2.5)));
    }

    #[test]
    fn test_parse_float_with_zero_decimal() {
        let result = parse_with(term(), "10.0");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Float(10.0)));
    }

    // Booleans
    #[test]
    fn test_parse_boolean_true() {
        let result = parse_with(term(), "true");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Boolean(true)));
    }

    #[test]
    fn test_parse_boolean_false() {
        let result = parse_with(term(), "false");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Boolean(false)));
    }

    // Variables
    #[test]
    fn test_parse_uppercase_variable() {
        let result = parse_with(term(), "X");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Variable(Intern::new("X".to_string()))
        );
    }

    #[test]
    fn test_parse_multichar_variable() {
        let result = parse_with(term(), "Player");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Variable(Intern::new("Player".to_string()))
        );
    }

    #[test]
    fn test_parse_underscore_variable() {
        let result = parse_with(term(), "_tmp");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Variable(Intern::new("_tmp".to_string()))
        );
    }

    // Strings
    #[test]
    fn test_parse_string_constant() {
        let result = parse_with(term(), "\"hello\"");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Constant(Value::String(Intern::new("hello".to_string())))
        );
    }

    #[test]
    fn test_parse_empty_string() {
        let result = parse_with(term(), "\"\"");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Constant(Value::String(Intern::new("".to_string())))
        );
    }

    // Atoms
    #[test]
    fn test_parse_atom_constant() {
        let result = parse_with(term(), "john");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Constant(Value::Atom(Intern::new("john".to_string())))
        );
    }

    #[test]
    fn test_parse_compound_term() {
        let result = parse_with(term(), "f(a, b)");
        assert!(result.is_ok());
        let expected = Term::Compound(
            Intern::new("f".to_string()),
            vec![
                Term::Constant(Value::Atom(Intern::new("a".to_string()))),
                Term::Constant(Value::Atom(Intern::new("b".to_string()))),
            ],
        );
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_parse_range_term() {
        let result = parse_with(term(), "1..10");
        assert!(result.is_ok());
        match result.unwrap() {
            Term::Range(start, end) => {
                assert_eq!(*start, Term::Constant(Value::Integer(1)));
                assert_eq!(*end, Term::Constant(Value::Integer(10)));
            }
            other => panic!("Expected range term, got {:?}", other),
        }
    }

    #[test]
    fn test_term_arithmetic_precedence_basic() {
        let result = parse_with(term(), "1 + 2 * 3");
        assert!(result.is_ok());
        let expected = Term::Compound(
            Intern::new("+".to_string()),
            vec![
                Term::Constant(Value::Integer(1)),
                Term::Compound(
                    Intern::new("*".to_string()),
                    vec![
                        Term::Constant(Value::Integer(2)),
                        Term::Constant(Value::Integer(3)),
                    ],
                ),
            ],
        );
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_parse_mixed_term_arguments() {
        let result = parse_with(term(), "foo(X, 1..3, Y + 4)");
        assert!(result.is_ok());
        let expected = Term::Compound(
            Intern::new("foo".to_string()),
            vec![
                Term::Variable(Intern::new("X".to_string())),
                Term::Range(
                    Box::new(Term::Constant(Value::Integer(1))),
                    Box::new(Term::Constant(Value::Integer(3))),
                ),
                Term::Compound(
                    Intern::new("+".to_string()),
                    vec![
                        Term::Variable(Intern::new("Y".to_string())),
                        Term::Constant(Value::Integer(4)),
                    ],
                ),
            ],
        );
        assert_eq!(result.unwrap(), expected);
    }

    // Atom parsing tests
    #[test]
    fn test_parse_atom_no_args() {
        let result = parse_with(atom(), "foo");
        assert!(result.is_ok());
        let expected = Atom {
            predicate: Intern::new("foo".to_string()),
            terms: vec![],
        };
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_parse_atom_with_args() {
        let result = parse_with(atom(), "parent(john, mary)");
        assert!(result.is_ok());
        let expected = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![
                Term::Constant(Value::Atom(Intern::new("john".to_string()))),
                Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
            ],
        };
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_parse_atom_with_variables() {
        let result = parse_with(atom(), "parent(X, Y)");
        assert!(result.is_ok());
        let expected = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![
                Term::Variable(Intern::new("X".to_string())),
                Term::Variable(Intern::new("Y".to_string())),
            ],
        };
        assert_eq!(result.unwrap(), expected);
    }

    // Literal parsing tests
    #[test]
    fn test_parse_positive_literal() {
        let result = parse_with(literal(), "parent(X, Y)");
        assert!(result.is_ok());
        match result.unwrap() {
            Literal::Positive(atom) => {
                assert_eq!(atom.predicate, Intern::new("parent".to_string()));
                assert_eq!(atom.terms.len(), 2);
            }
            Literal::Negative(_) => panic!("Expected positive literal"),
            Literal::Aggregate(_) => panic!("Expected positive literal, got aggregate"),
        }
    }

    #[test]
    fn test_parse_negative_literal() {
        let result = parse_with(literal(), "not parent(X, Y)");
        assert!(result.is_ok());
        match result.unwrap() {
            Literal::Negative(atom) => {
                assert_eq!(atom.predicate, Intern::new("parent".to_string()));
                assert_eq!(atom.terms.len(), 2);
            }
            Literal::Positive(_) => panic!("Expected negative literal"),
            Literal::Aggregate(_) => panic!("Expected negative literal, got aggregate"),
        }
    }

    // Aggregate parsing tests
    #[test]
    fn test_parse_count_aggregate_simple() {
        let result = parse_with(literal(), "count { X : selected(X) } > 2");
        assert!(result.is_ok());
        match result.unwrap() {
            Literal::Aggregate(agg) => {
                assert_eq!(agg.function, AggregateFunction::Count);
                assert_eq!(agg.variables.len(), 1);
                assert_eq!(agg.variables[0], Intern::new("X".to_string()));
                assert_eq!(agg.elements.len(), 1);
                assert_eq!(agg.comparison, ComparisonOp::GreaterThan);
                match &agg.value {
                    Term::Constant(Value::Integer(n)) => assert_eq!(*n, 2),
                    _ => panic!("Expected integer value"),
                }
            }
            _ => panic!("Expected aggregate literal"),
        }
    }

    #[test]
    fn test_parse_count_aggregate_equality() {
        let result = parse_with(literal(), "count { X : item(X) } = 5");
        assert!(result.is_ok());
        match result.unwrap() {
            Literal::Aggregate(agg) => {
                assert_eq!(agg.comparison, ComparisonOp::Equal);
                match &agg.value {
                    Term::Constant(Value::Integer(n)) => assert_eq!(*n, 5),
                    _ => panic!("Expected integer value"),
                }
            }
            _ => panic!("Expected aggregate literal"),
        }
    }

    #[test]
    fn test_parse_count_aggregate_less_equal() {
        let result = parse_with(literal(), "count { Y : member(Y) } <= 10");
        assert!(result.is_ok());
        match result.unwrap() {
            Literal::Aggregate(agg) => {
                assert_eq!(agg.comparison, ComparisonOp::LessOrEqual);
                assert_eq!(agg.variables[0], Intern::new("Y".to_string()));
            }
            _ => panic!("Expected aggregate literal"),
        }
    }

    #[test]
    fn test_parse_count_aggregate_multiple_conditions() {
        let result = parse_with(literal(), "count { X : item(X), heavy(X) } < 3");
        assert!(result.is_ok());
        match result.unwrap() {
            Literal::Aggregate(agg) => {
                assert_eq!(agg.elements.len(), 2);
                assert_eq!(agg.comparison, ComparisonOp::LessThan);
            }
            _ => panic!("Expected aggregate literal"),
        }
    }

    #[test]
    fn test_parse_constraint_with_count_aggregate() {
        let result = parse_program(":- count { X : selected(X) } > 2.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        match &result.unwrap().statements[0] {
            Statement::Constraint(constraint) => {
                assert_eq!(constraint.body.len(), 1);
                match &constraint.body[0] {
                    Literal::Aggregate(agg) => {
                        assert_eq!(agg.function, AggregateFunction::Count);
                        assert_eq!(agg.comparison, ComparisonOp::GreaterThan);
                    }
                    _ => panic!("Expected aggregate in constraint body"),
                }
            }
            _ => panic!("Expected constraint"),
        }
    }

    #[test]
    fn test_parse_rule_with_count_aggregate() {
        let result = parse_program("valid :- count { X : item(X) } <= 5.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        match &result.unwrap().statements[0] {
            Statement::Rule(rule) => {
                assert_eq!(rule.head.predicate, Intern::new("valid".to_string()));
                assert_eq!(rule.body.len(), 1);
                match &rule.body[0] {
                    Literal::Aggregate(agg) => {
                        assert_eq!(agg.function, AggregateFunction::Count);
                        assert_eq!(agg.comparison, ComparisonOp::LessOrEqual);
                    }
                    _ => panic!("Expected aggregate in rule body"),
                }
            }
            _ => panic!("Expected rule"),
        }
    }

    // Statement parsing tests
    #[test]
    fn test_parse_fact() {
        let result = parse_program("parent(john, mary).");
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.predicate, Intern::new("parent".to_string()));
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_rule() {
        let result = parse_program("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).");
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            Statement::Rule(rule) => {
                assert_eq!(rule.head.predicate, Intern::new("ancestor".to_string()));
                assert_eq!(rule.body.len(), 2);
            }
            _ => panic!("Expected rule"),
        }
    }

    #[test]
    fn test_parse_constraint() {
        let result = parse_program(":- unsafe(X).");
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            Statement::Constraint(constraint) => {
                assert_eq!(constraint.body.len(), 1);
            }
            _ => panic!("Expected constraint"),
        }
    }

    #[test]
    fn test_parse_const_decl() {
        let result = parse_program("#const width = 10.");
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("width".to_string()));
                assert_eq!(const_decl.value, Value::Integer(10));
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    #[test]
    fn test_parse_const_decl_negative() {
        let result = parse_program("#const min_temp = -5.");
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("min_temp".to_string()));
                assert_eq!(const_decl.value, Value::Integer(-5));
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    // Constant declarations with different data types
    #[test]
    #[allow(clippy::approx_constant)]
    fn test_parse_const_decl_float() {
        let result = parse_program("#const pi = 3.14.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("pi".to_string()));
                match const_decl.value {
                    Value::Float(f) => assert_eq!(f, 3.14),
                    _ => panic!("Expected float value, got {:?}", const_decl.value),
                }
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    #[test]
    fn test_parse_const_decl_negative_float() {
        let result = parse_program("#const neg = -2.5.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("neg".to_string()));
                match const_decl.value {
                    Value::Float(f) => assert_eq!(f, -2.5),
                    _ => panic!("Expected float value"),
                }
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    #[test]
    fn test_parse_const_decl_boolean_true() {
        let result = parse_program("#const enabled = true.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("enabled".to_string()));
                match const_decl.value {
                    Value::Boolean(true) => {}
                    _ => panic!("Expected true boolean value"),
                }
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    #[test]
    fn test_parse_const_decl_boolean_false() {
        let result = parse_program("#const disabled = false.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("disabled".to_string()));
                match const_decl.value {
                    Value::Boolean(false) => {}
                    _ => panic!("Expected false boolean value"),
                }
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    #[test]
    fn test_parse_const_decl_string() {
        let result = parse_program("#const message = \"hello world\".");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("message".to_string()));
                match &const_decl.value {
                    Value::String(s) => assert_eq!(*s, Intern::new("hello world".to_string())),
                    _ => panic!("Expected string value"),
                }
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    #[test]
    fn test_parse_const_decl_atom() {
        let result = parse_program("#const default_color = red.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("default_color".to_string()));
                match &const_decl.value {
                    Value::Atom(a) => assert_eq!(*a, Intern::new("red".to_string())),
                    _ => panic!("Expected atom value"),
                }
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    #[test]
    fn test_parse_multiple_const_decls() {
        let input = r#"
            #const width = 10.
            #const height = 20.
            #const max_enemies = 5.
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 3);

        match &program.statements[0] {
            Statement::ConstDecl(c) => {
                assert_eq!(c.name, Intern::new("width".to_string()));
                assert_eq!(c.value, Value::Integer(10));
            }
            _ => panic!("Expected const"),
        }

        match &program.statements[1] {
            Statement::ConstDecl(c) => {
                assert_eq!(c.name, Intern::new("height".to_string()));
                assert_eq!(c.value, Value::Integer(20));
            }
            _ => panic!("Expected const"),
        }
    }

    // Choice rules
    #[test]
    fn test_parse_simple_choice() {
        let input = "{ solid(1, 2); solid(2, 3) }.";
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);

        match &program.statements[0] {
            Statement::ChoiceRule(choice) => {
                assert_eq!(choice.lower_bound, None);
                assert_eq!(choice.upper_bound, None);
                assert_eq!(choice.elements.len(), 2);
                assert_eq!(choice.body.len(), 0);
                assert_eq!(
                    choice.elements[0].atom.predicate,
                    Intern::new("solid".to_string())
                );
                assert_eq!(choice.elements[0].condition.len(), 0);
            }
            _ => panic!("Expected choice rule"),
        }
    }

    #[test]
    fn test_parse_choice_with_bounds() {
        let input = "1 { item(a); item(b); item(c) } 2.";
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::ChoiceRule(choice) => {
                assert_eq!(choice.lower_bound, Some(Term::Constant(Value::Integer(1))));
                assert_eq!(choice.upper_bound, Some(Term::Constant(Value::Integer(2))));
                assert_eq!(choice.elements.len(), 3);
            }
            _ => panic!("Expected choice rule"),
        }
    }

    #[test]
    fn test_parse_choice_with_lower_bound() {
        let input = "2 { selected(X) }.";
        let result = parse_program(input);
        assert!(result.is_ok());

        match &result.unwrap().statements[0] {
            Statement::ChoiceRule(choice) => {
                assert_eq!(choice.lower_bound, Some(Term::Constant(Value::Integer(2))));
                assert_eq!(choice.upper_bound, None);
            }
            _ => panic!("Expected choice rule"),
        }
    }

    #[test]
    fn test_parse_choice_with_condition() {
        let input = "{ selected(X) : item(X) }.";
        let result = parse_program(input);
        assert!(result.is_ok());

        match &result.unwrap().statements[0] {
            Statement::ChoiceRule(choice) => {
                assert_eq!(choice.elements.len(), 1);
                assert_eq!(choice.elements[0].condition.len(), 1);
                match &choice.elements[0].condition[0] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new("item".to_string()));
                    }
                    _ => panic!("Expected positive literal"),
                }
            }
            _ => panic!("Expected choice rule"),
        }
    }

    #[test]
    fn test_parse_choice_with_body() {
        let input = "{ selected(X) : item(X) } :- room(R).";
        let result = parse_program(input);
        assert!(result.is_ok());

        match &result.unwrap().statements[0] {
            Statement::ChoiceRule(choice) => {
                assert_eq!(choice.body.len(), 1);
                match &choice.body[0] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new("room".to_string()));
                    }
                    _ => panic!("Expected positive literal"),
                }
            }
            _ => panic!("Expected choice rule"),
        }
    }

    // Full program parsing tests
    #[test]
    fn test_parse_empty_program() {
        let result = parse_program("");
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 0);
    }

    #[test]
    fn test_parse_program_with_multiple_statements() {
        let input = r#"
            parent(john, mary).
            parent(mary, alice).
            ancestor(X, Y) :- parent(X, Y).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 3);
    }

    #[test]
    fn test_parse_program_with_comments() {
        let input = r#"
            % This is a comment
            parent(john, mary). % Inline comment
            % Another comment
            parent(mary, alice).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 2);
    }

    // Block comment tests
    #[test]
    fn test_parse_simple_block_comment() {
        let input = "/* This is a block comment */ parent(john, mary).";
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
    }

    #[test]
    fn test_parse_multiline_block_comment() {
        let input = r#"
            /* This is a
               multi-line
               block comment */
            parent(john, mary).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
    }

    #[test]
    fn test_parse_block_comment_between_statements() {
        let input = r#"
            parent(john, mary).
            /* Block comment in the middle */
            parent(mary, alice).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 2);
    }

    #[test]
    fn test_parse_mixed_comments() {
        let input = r#"
            % Line comment
            /* Block comment */
            parent(john, mary).
            /* Another block
               comment */
            % Another line comment
            parent(mary, alice).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 2);
    }

    #[test]
    fn test_parse_block_comment_with_special_chars() {
        let input = r#"
            /* Comment with special chars: %, :-, ::, (). */
            parent(john, mary).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
    }

    #[test]
    fn test_parse_empty_block_comment() {
        let input = "/**/ parent(john, mary).";
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
    }

    // Comprehensive datatype test
    #[test]
    fn test_parse_all_datatypes_in_program() {
        let input = r#"
            % All datatypes example
            data(42).                    /* integer */
            data(-10).                   /* negative integer */
            data(3.14).                  /* float */
            data(-2.5).                  /* negative float */
            data(true).                  /* boolean true */
            data(false).                 /* boolean false */
            data("hello").               /* string */
            data(atom).                  /* atom */
            process(X, Y, _tmp).         /* variables */
            mixed(42, 3.14, true, "hi", atom, X).  /* all together */
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 10);
    }

    // Range parsing tests
    #[test]
    fn test_parse_fact_with_range() {
        let result = parse_program("dim(1..10).");
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);

        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.predicate, Intern::new("dim".to_string()));
                assert_eq!(fact.atom.terms.len(), 1);
                assert_eq!(fact.atom.terms[0], range(1, 10));
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_fact_with_multiple_ranges() {
        let result = parse_program("cell(1..5, 1..10).");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.predicate, Intern::new("cell".to_string()));
                assert_eq!(fact.atom.terms.len(), 2);
                assert_eq!(fact.atom.terms[0], range(1, 5));
                assert_eq!(fact.atom.terms[1], range(1, 10));
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_fact_with_mixed_terms_and_range() {
        let result = parse_program("edge(a, 1..10, b).");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.terms.len(), 3);
                assert_eq!(
                    fact.atom.terms[0],
                    Term::Constant(Value::Atom(Intern::new("a".to_string())))
                );
                assert_eq!(fact.atom.terms[1], range(1, 10));
                assert_eq!(
                    fact.atom.terms[2],
                    Term::Constant(Value::Atom(Intern::new("b".to_string())))
                );
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_range_in_compound_term() {
        let result = parse_program("data(item(1..5)).");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.terms.len(), 1);
                match &fact.atom.terms[0] {
                    Term::Compound(functor, args) => {
                        assert_eq!(*functor, Intern::new("item".to_string()));
                        assert_eq!(args.len(), 1);
                        assert_eq!(args[0], range(1, 5));
                    }
                    _ => panic!("Expected compound term"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_program_with_consts_and_ranges() {
        let input = r#"
            #const width = 10.
            #const height = 5.

            dim(1..10).
            cell(1..10, 1..5).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 4);

        // First two should be const decls
        match &program.statements[0] {
            Statement::ConstDecl(c) => {
                assert_eq!(c.name, Intern::new("width".to_string()));
            }
            _ => panic!("Expected const"),
        }

        // Third should be dim(1..width) - note: width won't be substituted yet
        match &program.statements[2] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.predicate, Intern::new("dim".to_string()));
                // Range still contains literal values until expansion
                assert_eq!(fact.atom.terms[0], range(1, 10));
            }
            _ => panic!("Expected fact"),
        }
    }

    // Operator and arithmetic parsing tests
    #[test]
    fn test_parse_operator_as_predicate() {
        let result = parse_program(">(X, 3).");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.predicate, Intern::new(">".to_string()));
                assert_eq!(fact.atom.terms.len(), 2);
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_infix_comparison() {
        let result = parse_program("test :- X > 3.");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Rule(rule) => {
                assert_eq!(rule.body.len(), 1);
                match &rule.body[0] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new(">".to_string()));
                        assert_eq!(atom.terms.len(), 2);
                        // Check that terms are correct
                        match &atom.terms[0] {
                            Term::Variable(v) => assert_eq!(*v, Intern::new("X".to_string())),
                            _ => panic!("Expected variable X"),
                        }
                        match &atom.terms[1] {
                            Term::Constant(Value::Integer(3)) => {}
                            _ => panic!("Expected integer 3, got {:?}", atom.terms[1]),
                        }
                    }
                    _ => panic!("Expected positive literal"),
                }
            }
            _ => panic!("Expected rule"),
        }
    }

    #[test]
    fn test_parse_arithmetic_addition() {
        // Note: X + Y is parsed as a compound term +(X, Y)
        let result = parse_program("test(X + Y).");
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.predicate, Intern::new("test".to_string()));
                assert_eq!(fact.atom.terms.len(), 1);
                match &fact.atom.terms[0] {
                    Term::Compound(functor, args) => {
                        assert_eq!(*functor, Intern::new("+".to_string()));
                        assert_eq!(args.len(), 2);
                    }
                    _ => panic!("Expected compound term for addition"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_arithmetic_precedence() {
        // X + Y * 2 should parse as +(X, *(Y, 2))
        let result = parse_program("test(X + Y * 2).");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Fact(fact) => {
                match &fact.atom.terms[0] {
                    Term::Compound(plus, args) => {
                        assert_eq!(*plus, Intern::new("+".to_string()));
                        assert_eq!(args.len(), 2);
                        // Right side should be *(Y, 2)
                        match &args[1] {
                            Term::Compound(mult, mult_args) => {
                                assert_eq!(*mult, Intern::new("*".to_string()));
                                assert_eq!(mult_args.len(), 2);
                            }
                            _ => panic!("Expected multiplication on right side"),
                        }
                    }
                    _ => panic!("Expected compound term"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_parenthesized_arithmetic() {
        // (X + Y) * 2 should parse as *((+(X, Y)), 2)
        let result = parse_program("test((X + Y) * 2).");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Fact(fact) => {
                match &fact.atom.terms[0] {
                    Term::Compound(mult, args) => {
                        assert_eq!(*mult, Intern::new("*".to_string()));
                        assert_eq!(args.len(), 2);
                        // Left side should be +(X, Y)
                        match &args[0] {
                            Term::Compound(plus, plus_args) => {
                                assert_eq!(*plus, Intern::new("+".to_string()));
                                assert_eq!(plus_args.len(), 2);
                            }
                            _ => panic!("Expected addition on left side"),
                        }
                    }
                    _ => panic!("Expected compound term"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_comparison_operators() {
        let operators = vec!["<", ">", "<=", ">=", "=", "!="];
        for op in operators {
            let program = format!("test :- X {} 5.", op);
            let result = parse_program(&program);
            assert!(result.is_ok(), "Failed to parse operator: {}", op);

            match &result.unwrap().statements[0] {
                Statement::Rule(rule) => match &rule.body[0] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new(op.to_string()));
                    }
                    _ => panic!("Expected positive literal"),
                },
                _ => panic!("Expected rule"),
            }
        }
    }

    #[test]
    fn test_parse_arithmetic_in_rule_body() {
        let result = parse_program("result(Z) :- X > 0, Y = X + 1, Z = Y * 2.");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Rule(rule) => {
                assert_eq!(rule.body.len(), 3);

                // First: X > 0
                match &rule.body[0] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new(">".to_string()));
                    }
                    _ => panic!("Expected comparison"),
                }

                // Second: Y = X + 1
                match &rule.body[1] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new("=".to_string()));
                        match &atom.terms[1] {
                            Term::Compound(plus, _) => {
                                assert_eq!(*plus, Intern::new("+".to_string()));
                            }
                            _ => panic!("Expected addition"),
                        }
                    }
                    _ => panic!("Expected comparison"),
                }
            }
            _ => panic!("Expected rule"),
        }
    }

    #[test]
    fn test_parse_negated_comparison() {
        let result = parse_program("test :- not X > 10.");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Rule(rule) => {
                assert_eq!(rule.body.len(), 1);
                match &rule.body[0] {
                    Literal::Negative(atom) => {
                        assert_eq!(atom.predicate, Intern::new(">".to_string()));
                    }
                    _ => panic!("Expected negative literal"),
                }
            }
            _ => panic!("Expected rule"),
        }
    }

    // Constant names in cardinality bounds tests
    #[test]
    fn test_parse_choice_with_constant_lower_bound() {
        let input = "min { selected(X) : item(X) }.";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        match &result.unwrap().statements[0] {
            Statement::ChoiceRule(choice) => {
                // Lower bound should be parsed as a constant name
                assert!(choice.lower_bound.is_some());
                match &choice.lower_bound {
                    Some(Term::Constant(Value::Atom(name))) => {
                        assert_eq!(*name, Intern::new("min".to_string()));
                    }
                    _ => panic!(
                        "Expected atom constant for lower bound, got {:?}",
                        choice.lower_bound
                    ),
                }
                assert!(choice.upper_bound.is_none());
            }
            _ => panic!("Expected choice rule"),
        }
    }

    #[test]
    fn test_parse_choice_with_constant_upper_bound() {
        let input = "{ selected(X) : item(X) } max.";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        match &result.unwrap().statements[0] {
            Statement::ChoiceRule(choice) => {
                assert!(choice.lower_bound.is_none());
                assert!(choice.upper_bound.is_some());
                match &choice.upper_bound {
                    Some(Term::Constant(Value::Atom(name))) => {
                        assert_eq!(*name, Intern::new("max".to_string()));
                    }
                    _ => panic!(
                        "Expected atom constant for upper bound, got {:?}",
                        choice.upper_bound
                    ),
                }
            }
            _ => panic!("Expected choice rule"),
        }
    }

    #[test]
    fn test_parse_choice_with_constant_both_bounds() {
        let input = "min { selected(X) : item(X) } max.";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        match &result.unwrap().statements[0] {
            Statement::ChoiceRule(choice) => {
                assert!(choice.lower_bound.is_some());
                assert!(choice.upper_bound.is_some());
                match &choice.lower_bound {
                    Some(Term::Constant(Value::Atom(name))) => {
                        assert_eq!(*name, Intern::new("min".to_string()));
                    }
                    _ => panic!("Expected atom constant for lower bound"),
                }
                match &choice.upper_bound {
                    Some(Term::Constant(Value::Atom(name))) => {
                        assert_eq!(*name, Intern::new("max".to_string()));
                    }
                    _ => panic!("Expected atom constant for upper bound"),
                }
            }
            _ => panic!("Expected choice rule"),
        }
    }

    #[test]
    fn test_parse_choice_with_mixed_bounds() {
        // Integer lower, constant upper
        let input = "2 { selected(X) : item(X) } max.";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        match &result.unwrap().statements[0] {
            Statement::ChoiceRule(choice) => {
                match &choice.lower_bound {
                    Some(Term::Constant(Value::Integer(2))) => {}
                    _ => panic!(
                        "Expected integer 2 for lower bound, got {:?}",
                        choice.lower_bound
                    ),
                }
                match &choice.upper_bound {
                    Some(Term::Constant(Value::Atom(name))) => {
                        assert_eq!(*name, Intern::new("max".to_string()));
                    }
                    _ => panic!("Expected atom constant for upper bound"),
                }
            }
            _ => panic!("Expected choice rule"),
        }
    }

    // Query parsing tests
    #[test]
    fn test_parse_query_ground() {
        let result = parse_query("?- parent(john, mary).");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let query = result.unwrap();
        assert_eq!(query.body.len(), 1);
        match &query.body[0] {
            Literal::Positive(atom) => {
                assert_eq!(atom.predicate, Intern::new("parent".to_string()));
                assert_eq!(atom.terms.len(), 2);
            }
            _ => panic!("Expected positive literal"),
        }
    }

    #[test]
    fn test_parse_query_with_variable() {
        let result = parse_query("?- parent(X, mary).");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let query = result.unwrap();
        assert_eq!(query.body.len(), 1);
        match &query.body[0] {
            Literal::Positive(atom) => {
                assert_eq!(atom.predicate, Intern::new("parent".to_string()));
                match &atom.terms[0] {
                    Term::Variable(v) => assert_eq!(*v, Intern::new("X".to_string())),
                    _ => panic!("Expected variable"),
                }
            }
            _ => panic!("Expected positive literal"),
        }
    }

    #[test]
    fn test_parse_query_multiple_literals() {
        let result = parse_query("?- parent(X, Y), parent(Y, Z).");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let query = result.unwrap();
        assert_eq!(query.body.len(), 2);
    }

    #[test]
    fn test_parse_query_with_negation() {
        let result = parse_query("?- parent(X, Y), not dead(X).");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let query = result.unwrap();
        assert_eq!(query.body.len(), 2);
        assert!(query.body[0].is_positive());
        assert!(query.body[1].is_negative());
    }

    #[test]
    fn test_parse_query_complex() {
        let result = parse_query("?- ancestor(X, Z), parent(X, Y), parent(Y, Z), not dead(Z).");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let query = result.unwrap();
        assert_eq!(query.body.len(), 4);
    }

    #[test]
    fn test_parse_query_with_builtin() {
        let result = parse_query("?- age(X, A), A > 18.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let query = result.unwrap();
        assert_eq!(query.body.len(), 2);
    }

    #[test]
    fn test_parse_query_zero_arity() {
        let result = parse_query("?- running.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let query = result.unwrap();
        assert_eq!(query.body.len(), 1);
    }

    #[test]
    fn test_parse_simple_test_block() {
        let input = r#"
            #test "basic test" {
                parent(john, mary).

                ?- parent(john, mary).
                + true.
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);

        match &program.statements[0] {
            Statement::Test(test_block) => {
                assert_eq!(test_block.name, "basic test");
                assert_eq!(test_block.statements.len(), 1);
                assert_eq!(test_block.test_cases.len(), 1);

                // Check test case
                let test_case = &test_block.test_cases[0];
                assert_eq!(test_case.query.body.len(), 1);
                assert_eq!(test_case.positive_assertions.len(), 1);
                assert_eq!(test_case.negative_assertions.len(), 0);
            }
            _ => panic!("Expected test block"),
        }
    }

    #[test]
    fn test_parse_test_block_with_rules() {
        let input = r#"
            #test "transitive closure" {
                edge(a, b).
                edge(b, c).
                path(X, Y) :- edge(X, Y).
                path(X, Z) :- path(X, Y), edge(Y, Z).

                ?- path(a, c).
                + path(a, c).
                - path(c, a).
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Test(test_block) => {
                assert_eq!(test_block.name, "transitive closure");
                assert_eq!(test_block.statements.len(), 4); // 2 facts + 2 rules
                assert_eq!(test_block.test_cases.len(), 1);

                let test_case = &test_block.test_cases[0];
                assert_eq!(test_case.positive_assertions.len(), 1);
                assert_eq!(test_case.negative_assertions.len(), 1);
            }
            _ => panic!("Expected test block"),
        }
    }

    #[test]
    fn test_parse_test_block_multiple_queries() {
        let input = r#"
            #test "multiple queries" {
                value(1).
                value(2).

                ?- value(1).
                + true.

                ?- value(X).
                + value(1).
                + value(2).
                - value(3).
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Test(test_block) => {
                assert_eq!(test_block.test_cases.len(), 2);

                // First query
                assert_eq!(test_block.test_cases[0].positive_assertions.len(), 1);

                // Second query
                assert_eq!(test_block.test_cases[1].positive_assertions.len(), 2);
                assert_eq!(test_block.test_cases[1].negative_assertions.len(), 1);
            }
            _ => panic!("Expected test block"),
        }
    }

    #[test]
    fn test_parse_test_block_with_comments() {
        let input = r#"
            #test "comments ok" {
                parent(john, mary).

                % inline comment inside test block
                ?- parent(john, mary).
                + true.
            }
        "#;
        let result = parse_program(input);
        assert!(
            result.is_ok(),
            "Parser should allow comments in test blocks, but got: {:?}",
            result.err()
        );

        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);

        match &program.statements[0] {
            Statement::Test(test_block) => {
                assert_eq!(test_block.name, "comments ok");
                assert_eq!(test_block.statements.len(), 1);
                assert_eq!(test_block.test_cases.len(), 1);
            }
            _ => panic!("Expected test block"),
        }
    }

    #[test]
    fn test_parse_test_block_with_constants() {
        let input = r#"
            #test "with constants" {
                #const max_val = 10.
                limit(max_val).

                ?- limit(10).
                + true.
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Test(test_block) => {
                assert_eq!(test_block.statements.len(), 2); // const decl + fact
            }
            _ => panic!("Expected test block"),
        }
    }

    #[test]
    fn test_parse_multiple_test_blocks() {
        let input = r#"
            #test "test 1" {
                a(1).
                ?- a(1).
                + true.
            }

            #test "test 2" {
                b(2).
                ?- b(2).
                + true.
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        let program = result.unwrap();
        assert_eq!(program.statements.len(), 2);

        match (&program.statements[0], &program.statements[1]) {
            (Statement::Test(test1), Statement::Test(test2)) => {
                assert_eq!(test1.name, "test 1");
                assert_eq!(test2.name, "test 2");
            }
            _ => panic!("Expected two test blocks"),
        }
    }

    #[test]
    fn test_parse_minimize_simple() {
        // Test: #minimize { X : cost(X) }.
        let result = parse_with(program(), "#minimize { X : cost(X) }.");
        assert!(
            result.is_ok(),
            "Failed to parse minimize: {:?}",
            result.err()
        );

        let prog = result.unwrap();
        assert_eq!(prog.statements.len(), 1);

        match &prog.statements[0] {
            Statement::Optimize(opt) => {
                assert_eq!(opt.direction, OptimizeDirection::Minimize);
                assert_eq!(opt.terms.len(), 1);
                // Should have one term with cost(X) condition
            }
            _ => panic!("Expected Optimize statement"),
        }
    }

    #[test]
    fn test_parse_maximize_simple() {
        // Test: #maximize { X : value(X) }.
        let result = parse_with(program(), "#maximize { X : value(X) }.");
        assert!(
            result.is_ok(),
            "Failed to parse maximize: {:?}",
            result.err()
        );

        let prog = result.unwrap();
        assert_eq!(prog.statements.len(), 1);

        match &prog.statements[0] {
            Statement::Optimize(opt) => {
                assert_eq!(opt.direction, OptimizeDirection::Maximize);
                assert_eq!(opt.terms.len(), 1);
            }
            _ => panic!("Expected Optimize statement"),
        }
    }

    #[test]
    fn test_parse_minimize_with_weight() {
        // Test: #minimize { C*X : cost(X, C) }.
        let result = parse_with(program(), "#minimize { C*X : cost(X, C) }.");
        assert!(
            result.is_ok(),
            "Failed to parse weighted minimize: {:?}",
            result.err()
        );

        let prog = result.unwrap();
        match &prog.statements[0] {
            Statement::Optimize(opt) => {
                assert_eq!(opt.direction, OptimizeDirection::Minimize);
                // Should have weighted term
            }
            _ => panic!("Expected Optimize statement"),
        }
    }

    #[test]
    fn test_parse_minimize_integer() {
        // Test: #minimize { 5 }.
        let result = parse_with(program(), "#minimize { 5 }.");
        assert!(
            result.is_ok(),
            "Failed to parse integer minimize: {:?}",
            result.err()
        );

        let prog = result.unwrap();
        match &prog.statements[0] {
            Statement::Optimize(opt) => {
                assert_eq!(opt.direction, OptimizeDirection::Minimize);
                assert_eq!(opt.terms.len(), 1);
            }
            _ => panic!("Expected Optimize statement"),
        }
    }
