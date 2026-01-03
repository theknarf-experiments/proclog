    use super::*;
    use internment::Intern;
    use proclog_ast::{Atom, Term, Value};

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

    fn pred(name: &str) -> Symbol {
        Intern::new(name.to_string())
    }

    // Basic stratification tests
    #[test]
    fn test_stratify_empty_program() {
        let rules = vec![];
        let result = stratify(&rules).unwrap();

        assert_eq!(result.num_strata, 0);
        assert_eq!(result.rules_by_stratum.len(), 0);
    }

    #[test]
    fn test_stratify_single_rule_no_negation() {
        // p(X) :- q(X).
        let rules = vec![make_rule(
            make_atom("p", vec![var("X")]),
            vec![Literal::Positive(make_atom("q", vec![var("X")]))],
        )];

        let result = stratify(&rules).unwrap();

        assert_eq!(result.num_strata, 1);
        assert_eq!(result.predicate_strata.get(&pred("p")), Some(&0));
        assert_eq!(result.predicate_strata.get(&pred("q")), Some(&0));
    }

    #[test]
    fn test_stratify_negation_simple() {
        // p(X) :- q(X), not r(X).
        let rules = vec![make_rule(
            make_atom("p", vec![var("X")]),
            vec![
                Literal::Positive(make_atom("q", vec![var("X")])),
                Literal::Negative(make_atom("r", vec![var("X")])),
            ],
        )];

        let result = stratify(&rules).unwrap();

        // q and r should be in stratum 0, p in stratum 1 (after r)
        assert_eq!(result.predicate_strata.get(&pred("q")), Some(&0));
        assert_eq!(result.predicate_strata.get(&pred("r")), Some(&0));
        assert_eq!(result.predicate_strata.get(&pred("p")), Some(&1));
        assert_eq!(result.num_strata, 2);
    }

    #[test]
    fn test_stratify_chain_of_negations() {
        // p(X) :- not q(X).
        // q(X) :- not r(X).
        // r(X) :- s(X).
        let rules = vec![
            make_rule(
                make_atom("p", vec![var("X")]),
                vec![Literal::Negative(make_atom("q", vec![var("X")]))],
            ),
            make_rule(
                make_atom("q", vec![var("X")]),
                vec![Literal::Negative(make_atom("r", vec![var("X")]))],
            ),
            make_rule(
                make_atom("r", vec![var("X")]),
                vec![Literal::Positive(make_atom("s", vec![var("X")]))],
            ),
        ];

        let result = stratify(&rules).unwrap();

        // s and r should be in stratum 0
        // q should be in stratum 1 (after r due to negation)
        // p should be in stratum 2 (after q due to negation)
        assert_eq!(result.predicate_strata.get(&pred("s")), Some(&0));
        assert_eq!(result.predicate_strata.get(&pred("r")), Some(&0));
        assert_eq!(result.predicate_strata.get(&pred("q")), Some(&1));
        assert_eq!(result.predicate_strata.get(&pred("p")), Some(&2));
        assert_eq!(result.num_strata, 3);
    }

    #[test]
    fn test_stratify_multiple_rules_same_predicate() {
        // p(X) :- q(X).
        // p(X) :- not r(X).
        let rules = vec![
            make_rule(
                make_atom("p", vec![var("X")]),
                vec![Literal::Positive(make_atom("q", vec![var("X")]))],
            ),
            make_rule(
                make_atom("p", vec![var("X")]),
                vec![Literal::Negative(make_atom("r", vec![var("X")]))],
            ),
        ];

        let result = stratify(&rules).unwrap();

        // q and r in stratum 0, p in stratum 1
        assert_eq!(result.predicate_strata.get(&pred("q")), Some(&0));
        assert_eq!(result.predicate_strata.get(&pred("r")), Some(&0));
        assert_eq!(result.predicate_strata.get(&pred("p")), Some(&1));
    }

    // Cycle detection tests
    #[test]
    fn test_detect_direct_negative_cycle() {
        // p(X) :- not p(X).  [Illegal!]
        let rules = vec![make_rule(
            make_atom("p", vec![var("X")]),
            vec![Literal::Negative(make_atom("p", vec![var("X")]))],
        )];

        let result = stratify(&rules);
        let err = result.unwrap_err();
        assert_eq!(err.to_string(), "Cycle through negation detected: p -> p");

        match err {
            StratificationError::CycleThroughNegation(cycle) => {
                assert_eq!(cycle, vec![pred("p")]);
            }
        }
    }

    #[test]
    fn test_detect_indirect_negative_cycle() {
        // p(X) :- not q(X).
        // q(X) :- p(X).  [Creates cycle through negation!]
        let rules = vec![
            make_rule(
                make_atom("p", vec![var("X")]),
                vec![Literal::Negative(make_atom("q", vec![var("X")]))],
            ),
            make_rule(
                make_atom("q", vec![var("X")]),
                vec![Literal::Positive(make_atom("p", vec![var("X")]))],
            ),
        ];

        let result = stratify(&rules);
        assert!(result.is_err());
    }

    #[test]
    fn test_positive_cycle_is_ok() {
        // p(X) :- p(X).  [Recursive, but no negation - OK!]
        let rules = vec![make_rule(
            make_atom("p", vec![var("X")]),
            vec![Literal::Positive(make_atom("p", vec![var("X")]))],
        )];

        let result = stratify(&rules);
        assert!(result.is_ok());

        // Should all be in stratum 0
        assert_eq!(result.unwrap().num_strata, 1);
    }

    #[test]
    fn test_transitive_closure_is_ok() {
        // path(X, Y) :- edge(X, Y).
        // path(X, Z) :- path(X, Y), edge(Y, Z).
        let rules = vec![
            make_rule(
                make_atom("path", vec![var("X"), var("Y")]),
                vec![Literal::Positive(make_atom(
                    "edge",
                    vec![var("X"), var("Y")],
                ))],
            ),
            make_rule(
                make_atom("path", vec![var("X"), var("Z")]),
                vec![
                    Literal::Positive(make_atom("path", vec![var("X"), var("Y")])),
                    Literal::Positive(make_atom("edge", vec![var("Y"), var("Z")])),
                ],
            ),
        ];

        let result = stratify(&rules).unwrap();

        // All in same stratum (positive recursion is fine)
        assert_eq!(result.num_strata, 1);
    }

    #[test]
    fn test_rules_organized_by_stratum() {
        // r0(X) :- base(X).
        // r1(X) :- not r0(X).
        // r2(X) :- not r1(X).
        let rules = vec![
            make_rule(
                make_atom("r0", vec![var("X")]),
                vec![Literal::Positive(make_atom("base", vec![var("X")]))],
            ),
            make_rule(
                make_atom("r1", vec![var("X")]),
                vec![Literal::Negative(make_atom("r0", vec![var("X")]))],
            ),
            make_rule(
                make_atom("r2", vec![var("X")]),
                vec![Literal::Negative(make_atom("r1", vec![var("X")]))],
            ),
        ];

        let result = stratify(&rules).unwrap();

        assert_eq!(result.num_strata, 3);
        assert_eq!(result.rules_by_stratum[0].len(), 1); // r0 rule
        assert_eq!(result.rules_by_stratum[1].len(), 1); // r1 rule
        assert_eq!(result.rules_by_stratum[2].len(), 1); // r2 rule
    }
