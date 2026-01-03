    use super::*;
    use internment::Intern;
    use proclog_ast::Value;

    // Helper functions
    #[allow(dead_code)]
    fn atom_const(name: &str) -> Term {
        Term::Constant(Value::Atom(Intern::new(name.to_string())))
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

    fn make_rule(head: Atom, body: Vec<Literal>) -> Rule {
        Rule { head, body }
    }

    #[test]
    fn test_safe_rule_all_positive() {
        // p(X) :- q(X).
        let rule = make_rule(
            make_atom("p", vec![var("X")]),
            vec![Literal::Positive(make_atom("q", vec![var("X")]))],
        );

        assert!(check_rule_safety(&rule).is_ok());
    }

    #[test]
    fn test_safe_rule_with_negation() {
        // p(X) :- q(X), not r(X).
        let rule = make_rule(
            make_atom("p", vec![var("X")]),
            vec![
                Literal::Positive(make_atom("q", vec![var("X")])),
                Literal::Negative(make_atom("r", vec![var("X")])),
            ],
        );

        assert!(check_rule_safety(&rule).is_ok());
    }

    #[test]
    fn test_unsafe_negation_only() {
        // UNSAFE: bad(X) :- not good(X).
        let rule = make_rule(
            make_atom("bad", vec![var("X")]),
            vec![Literal::Negative(make_atom("good", vec![var("X")]))],
        );

        let result = check_rule_safety(&rule);
        assert!(result.is_err());

        if let Err(SafetyError::UnsafeNegation { variables, .. }) = result {
            assert_eq!(variables.len(), 1);
            assert_eq!(variables[0], Intern::new("X".to_string()));
        }
    }

    #[test]
    fn test_unsafe_head_variable() {
        // UNSAFE: p(X, Y) :- q(X).
        // Y appears in head but not in any positive body literal
        let rule = make_rule(
            make_atom("p", vec![var("X"), var("Y")]),
            vec![Literal::Positive(make_atom("q", vec![var("X")]))],
        );

        let result = check_rule_safety(&rule);
        assert!(result.is_err());

        if let Err(SafetyError::UnsafeNegation { variables, .. }) = result {
            assert!(variables.contains(&Intern::new("Y".to_string())));
        }
    }

    #[test]
    fn test_safe_with_multiple_positive_literals() {
        // p(X, Y) :- q(X), r(Y), not s(X, Y).
        let rule = make_rule(
            make_atom("p", vec![var("X"), var("Y")]),
            vec![
                Literal::Positive(make_atom("q", vec![var("X")])),
                Literal::Positive(make_atom("r", vec![var("Y")])),
                Literal::Negative(make_atom("s", vec![var("X"), var("Y")])),
            ],
        );

        assert!(check_rule_safety(&rule).is_ok());
    }

    #[test]
    fn test_unsafe_variable_in_negation_only() {
        // UNSAFE: p(X) :- q(X), not r(X, Y).
        // Y only appears in negated literal
        let rule = make_rule(
            make_atom("p", vec![var("X")]),
            vec![
                Literal::Positive(make_atom("q", vec![var("X")])),
                Literal::Negative(make_atom("r", vec![var("X"), var("Y")])),
            ],
        );

        let result = check_rule_safety(&rule);
        assert!(result.is_err());

        if let Err(SafetyError::UnsafeNegation { variables, .. }) = result {
            assert!(variables.contains(&Intern::new("Y".to_string())));
        }
    }

    #[test]
    fn test_safe_with_compound_terms() {
        // p(X) :- q(item(X, Y)), not r(item(X, Y)).
        let rule = make_rule(
            make_atom("p", vec![var("X")]),
            vec![
                Literal::Positive(make_atom(
                    "q",
                    vec![Term::Compound(
                        Intern::new("item".to_string()),
                        vec![var("X"), var("Y")],
                    )],
                )),
                Literal::Negative(make_atom(
                    "r",
                    vec![Term::Compound(
                        Intern::new("item".to_string()),
                        vec![var("X"), var("Y")],
                    )],
                )),
            ],
        );

        assert!(check_rule_safety(&rule).is_ok());
    }

    #[test]
    fn test_unsafe_compound_term_variable() {
        // UNSAFE: p(X) :- q(X), not r(item(Y)).
        // Y only appears in negated literal within compound term
        let rule = make_rule(
            make_atom("p", vec![var("X")]),
            vec![
                Literal::Positive(make_atom("q", vec![var("X")])),
                Literal::Negative(make_atom(
                    "r",
                    vec![Term::Compound(
                        Intern::new("item".to_string()),
                        vec![var("Y")],
                    )],
                )),
            ],
        );

        let result = check_rule_safety(&rule);
        assert!(result.is_err());
    }

    #[test]
    fn test_program_safety_all_safe() {
        let rules = vec![
            make_rule(
                make_atom("p", vec![var("X")]),
                vec![Literal::Positive(make_atom("q", vec![var("X")]))],
            ),
            make_rule(
                make_atom("r", vec![var("X")]),
                vec![
                    Literal::Positive(make_atom("s", vec![var("X")])),
                    Literal::Negative(make_atom("t", vec![var("X")])),
                ],
            ),
        ];

        assert!(check_program_safety(&rules).is_ok());
    }

    #[test]
    fn test_program_safety_one_unsafe() {
        let rules = vec![
            make_rule(
                make_atom("p", vec![var("X")]),
                vec![Literal::Positive(make_atom("q", vec![var("X")]))],
            ),
            // This one is unsafe
            make_rule(
                make_atom("bad", vec![var("X")]),
                vec![Literal::Negative(make_atom("good", vec![var("X")]))],
            ),
        ];

        let result = check_program_safety(&rules);
        assert!(result.is_err());
    }
