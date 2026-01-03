    use super::*;
    use internment::Intern;
    use proclog_ast::Program;
    use proclog_parser::{ParseError, SrcId};

    fn parse_program(input: &str) -> Result<Program, Vec<ParseError>> {
        proclog_parser::parse_program(input, SrcId::empty())
    }

    fn satisfy_body_naive(body: &[Literal], db: &FactDatabase) -> Vec<Substitution> {
        if body.is_empty() {
            return vec![Substitution::new()];
        }

        let first_literal = &body[0];
        let rest = &body[1..];

        match first_literal {
            Literal::Positive(atom) => {
                if let Some(builtin) = builtins::parse_builtin(atom) {
                    if rest.is_empty() {
                        let empty_subst = Substitution::new();
                        match builtins::eval_builtin(&builtin, &empty_subst) {
                            Some(true) => vec![empty_subst],
                            _ => vec![],
                        }
                    } else {
                        let rest_substs = satisfy_body_naive(rest, db);
                        let mut result = Vec::new();

                        for subst in rest_substs {
                            let applied_builtin = apply_subst_to_builtin(&subst, &builtin);
                            if let Some(true) = builtins::eval_builtin(&applied_builtin, &subst) {
                                result.push(subst);
                            }
                        }

                        result
                    }
                } else {
                    let initial_substs = db.query(atom);

                    if rest.is_empty() {
                        initial_substs
                    } else {
                        let mut all_substs = Vec::new();

                        for subst in initial_substs {
                            let applied_rest: Vec<Literal> = rest
                                .iter()
                                .map(|lit| apply_subst_to_literal(&subst, lit))
                                .collect();

                            let rest_substs = satisfy_body_naive(&applied_rest, db);

                            for rest_subst in rest_substs {
                                if let Some(combined) = combine_substs(&subst, &rest_subst) {
                                    all_substs.push(combined);
                                }
                            }
                        }

                        all_substs
                    }
                }
            }
            Literal::Negative(atom) => {
                if rest.is_empty() {
                    if database_has_match(db, atom) {
                        vec![]
                    } else {
                        vec![Substitution::new()]
                    }
                } else {
                    let rest_substs = satisfy_body_naive(rest, db);
                    let mut result = Vec::new();

                    for subst in rest_substs {
                        let applied_atom = subst.apply_atom(atom);

                        if !database_has_match(db, &applied_atom) {
                            result.push(subst);
                        }
                    }

                    result
                }
            }
            Literal::Aggregate(_) => {
                // Aggregates are evaluated as filters after grounding
                // Process rest of body first
                satisfy_body_naive(rest, db)
            }
        }
    }

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

    fn make_rule(head: Atom, body: Vec<Literal>) -> Rule {
        Rule { head, body }
    }

    fn normalize_substitutions(substs: Vec<Substitution>) -> Vec<Vec<(String, String)>> {
        let mut normalized: Vec<_> = substs
            .into_iter()
            .map(|subst| {
                let mut bindings: Vec<_> = subst
                    .iter()
                    .map(|(symbol, term)| (symbol.as_ref().clone(), format!("{:?}", term)))
                    .collect();
                bindings.sort();
                bindings
            })
            .collect();
        normalized.sort();
        normalized
    }

    #[test]
    fn test_combine_substs_conflicting_bindings_are_filtered() {
        let mut s1 = Substitution::new();
        s1.bind(Intern::new("X".to_string()), atom_const("a"));

        let mut s2 = Substitution::new();
        s2.bind(Intern::new("X".to_string()), atom_const("b"));

        assert!(combine_substs(&s1, &s2).is_none());
    }

    #[test]
    fn test_combine_substs_indirect_conflict_is_filtered() {
        let mut s1 = Substitution::new();
        s1.bind(Intern::new("X".to_string()), var("Y"));

        let mut s2 = Substitution::new();
        s2.bind(Intern::new("Y".to_string()), atom_const("a"));
        s2.bind(Intern::new("X".to_string()), atom_const("b"));

        assert!(combine_substs(&s1, &s2).is_none());
    }

    #[test]
    fn test_combine_substs_indirect_resolution_succeeds() {
        let mut s1 = Substitution::new();
        s1.bind(Intern::new("X".to_string()), var("Y"));

        let mut s2 = Substitution::new();
        s2.bind(Intern::new("Y".to_string()), atom_const("a"));
        s2.bind(Intern::new("X".to_string()), atom_const("a"));

        let combined = combine_substs(&s1, &s2).expect("substitutions should merge");

        assert_eq!(combined.apply(&var("X")), atom_const("a"));
        assert_eq!(combined.apply(&var("Y")), atom_const("a"));
    }

    // Basic grounding tests
    #[test]
    fn test_ground_rule_no_variables() {
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![atom_const("john"), atom_const("mary")],
        ))
        .unwrap();

        // Rule: ancestor(john, mary) :- parent(john, mary).
        let rule = make_rule(
            make_atom("ancestor", vec![atom_const("john"), atom_const("mary")]),
            vec![Literal::Positive(make_atom(
                "parent",
                vec![atom_const("john"), atom_const("mary")],
            ))],
        );

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].predicate, Intern::new("ancestor".to_string()));
    }

    #[test]
    fn test_ground_rule_single_variable() {
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

        // Rule: child(X) :- parent(john, X).
        let rule = make_rule(
            make_atom("child", vec![var("X")]),
            vec![Literal::Positive(make_atom(
                "parent",
                vec![atom_const("john"), var("X")],
            ))],
        );

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 2);

        // Both child(mary) and child(bob) should be generated
        let predicates: Vec<_> = results.iter().map(|a| &a.predicate).collect();
        assert!(predicates
            .iter()
            .all(|p| **p == Intern::new("child".to_string())));
    }

    #[test]
    fn test_ground_rule_multiple_variables() {
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

        // Rule: ancestor(X, Y) :- parent(X, Y).
        let rule = make_rule(
            make_atom("ancestor", vec![var("X"), var("Y")]),
            vec![Literal::Positive(make_atom(
                "parent",
                vec![var("X"), var("Y")],
            ))],
        );

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ground_rule_join_two_literals() {
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
        db.insert(make_atom(
            "parent",
            vec![atom_const("bob"), atom_const("charlie")],
        ))
        .unwrap();

        // Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        let rule = make_rule(
            make_atom("grandparent", vec![var("X"), var("Z")]),
            vec![
                Literal::Positive(make_atom("parent", vec![var("X"), var("Y")])),
                Literal::Positive(make_atom("parent", vec![var("Y"), var("Z")])),
            ],
        );

        let results = ground_rule(&rule, &db);

        // Only john -> mary -> alice forms a valid chain
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_const("john"));
        assert_eq!(results[0].terms[1], atom_const("alice"));
    }

    #[test]
    fn test_ground_rule_multiple_chains() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("a"), atom_const("b")]))
            .unwrap();
        db.insert(make_atom("parent", vec![atom_const("b"), atom_const("c")]))
            .unwrap();
        db.insert(make_atom("parent", vec![atom_const("b"), atom_const("d")]))
            .unwrap();

        // Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        let rule = make_rule(
            make_atom("grandparent", vec![var("X"), var("Z")]),
            vec![
                Literal::Positive(make_atom("parent", vec![var("X"), var("Y")])),
                Literal::Positive(make_atom("parent", vec![var("Y"), var("Z")])),
            ],
        );

        let results = ground_rule(&rule, &db);

        // a -> b -> c and a -> b -> d
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ground_rule_no_matches() {
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![atom_const("john"), atom_const("mary")],
        ))
        .unwrap();

        // Rule: child(X) :- parent(alice, X).
        // No facts match parent(alice, X)
        let rule = make_rule(
            make_atom("child", vec![var("X")]),
            vec![Literal::Positive(make_atom(
                "parent",
                vec![atom_const("alice"), var("X")],
            ))],
        );

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_ground_rule_empty_body() {
        let db = FactDatabase::new();

        // Rule: fact(a) :- .
        // (A rule with no body is always true)
        let rule = make_rule(make_atom("fact", vec![atom_const("a")]), vec![]);

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], make_atom("fact", vec![atom_const("a")]));
    }

    #[test]
    fn test_ground_rule_three_literals() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("edge", vec![atom_const("a"), atom_const("b")]))
            .unwrap();
        db.insert(make_atom("edge", vec![atom_const("b"), atom_const("c")]))
            .unwrap();
        db.insert(make_atom("edge", vec![atom_const("c"), atom_const("d")]))
            .unwrap();

        // Rule: path3(X, W) :- edge(X, Y), edge(Y, Z), edge(Z, W).
        let rule = make_rule(
            make_atom("path3", vec![var("X"), var("W")]),
            vec![
                Literal::Positive(make_atom("edge", vec![var("X"), var("Y")])),
                Literal::Positive(make_atom("edge", vec![var("Y"), var("Z")])),
                Literal::Positive(make_atom("edge", vec![var("Z"), var("W")])),
            ],
        );

        let results = ground_rule(&rule, &db);

        // Only a -> b -> c -> d
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_const("a"));
        assert_eq!(results[0].terms[1], atom_const("d"));
    }

    // Integration test: Parse → Load Facts → Ground Rules → Query
    // Negation tests
    #[test]
    fn test_ground_rule_simple_negation() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("bird", vec![atom_const("tweety")]))
            .unwrap();
        db.insert(make_atom("bird", vec![atom_const("polly")]))
            .unwrap();
        db.insert(make_atom("penguin", vec![atom_const("polly")]))
            .unwrap();

        // Rule: flies(X) :- bird(X), not penguin(X).
        let rule = make_rule(
            make_atom("flies", vec![var("X")]),
            vec![
                Literal::Positive(make_atom("bird", vec![var("X")])),
                Literal::Negative(make_atom("penguin", vec![var("X")])),
            ],
        );

        let results = ground_rule(&rule, &db);

        // Only tweety should fly (polly is a penguin)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_const("tweety"));
    }

    #[test]
    fn test_ground_rule_negation_no_match() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("bird", vec![atom_const("tweety")]))
            .unwrap();

        // Rule: mammal(X) :- not bird(X).
        // Since only tweety exists and is a bird, no mammals
        let rule = make_rule(
            make_atom("mammal", vec![var("X")]),
            vec![Literal::Negative(make_atom("bird", vec![var("X")]))],
        );

        let results = ground_rule(&rule, &db);

        // No results - we can't prove something is NOT a bird
        // unless we have a closed world assumption with a finite domain
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_ground_rule_negation_with_ground_term() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("bird", vec![atom_const("tweety")]))
            .unwrap();

        // Rule: not_bird_polly :- not bird(polly).
        let rule = make_rule(
            make_atom("not_bird_polly", vec![]),
            vec![Literal::Negative(make_atom(
                "bird",
                vec![atom_const("polly")],
            ))],
        );

        let results = ground_rule(&rule, &db);

        // polly is not a bird, so this succeeds
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_negated_ground_literal_avoids_query_allocations() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("p", vec![atom_const("a")])).unwrap();

        let body = vec![
            Literal::Positive(make_atom("p", vec![var("X")])),
            Literal::Negative(make_atom("q", vec![var("X")])),
        ];

        let tracker = FactDatabase::track_ground_queries();
        let substitutions = satisfy_body(&body, &db);

        assert_eq!(substitutions.len(), 1);
        assert_eq!(substitutions[0].apply(&var("X")), atom_const("a"));
        assert_eq!(tracker.count(), 0);
    }

    #[test]
    fn test_ground_rule_multiple_negations() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("a", vec![atom_const("x")])).unwrap();
        db.insert(make_atom("b", vec![atom_const("y")])).unwrap();
        db.insert(make_atom("c", vec![atom_const("z")])).unwrap();

        // Rule: result :- not a(y), not b(x), not c(w).
        let rule = make_rule(
            make_atom("result", vec![]),
            vec![
                Literal::Negative(make_atom("a", vec![atom_const("y")])),
                Literal::Negative(make_atom("b", vec![atom_const("x")])),
                Literal::Negative(make_atom("c", vec![atom_const("w")])),
            ],
        );

        let results = ground_rule(&rule, &db);

        // All negations succeed
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_large_rule_allocation_regression() {
        let mut db = FactDatabase::new();
        let chain_len = 30;

        for i in 0..=chain_len {
            db.insert(make_atom("link", vec![int(i as i64), int((i + 1) as i64)]))
                .unwrap();
        }

        let mut body = Vec::new();
        body.push(Literal::Positive(make_atom(
            "link",
            vec![int(0), var("X1")],
        )));

        for i in 1..=chain_len {
            let current = format!("X{i}");
            let next = format!("X{}", i + 1);
            body.push(Literal::Positive(make_atom(
                "link",
                vec![var(&current), var(&next)],
            )));
        }

        allocation_tracker::reset();
        let naive_results = satisfy_body_naive(&body, &db);
        let naive_allocs = allocation_tracker::allocations();

        allocation_tracker::reset();
        let optimized_results = satisfy_body(&body, &db);
        let optimized_allocs = allocation_tracker::allocations();

        assert_eq!(naive_results, optimized_results);
        assert!(optimized_allocs < naive_allocs);
    }

    #[test]
    fn test_integration_parse_ground_query() {
        use proclog_ast::Statement;

        // A complete ProcLog program with facts and rules
        let program_text = r#"
            % Facts about family relationships
            parent(john, mary).
            parent(mary, alice).
            parent(bob, charlie).
            parent(charlie, dave).

            % Rule: grandparent relationship
            grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        "#;

        // Step 1: Parse the program
        let program = parse_program(program_text).expect("Should parse successfully");

        // Step 2: Load facts into database
        let mut db = FactDatabase::new();
        let mut rules = Vec::new();

        for statement in program.statements {
            match statement {
                Statement::Fact(fact) => {
                    db.insert(fact.atom).unwrap();
                }
                Statement::Rule(rule) => {
                    rules.push(rule);
                }
                _ => {} // Ignore other statement types for this test
            }
        }

        // Verify we loaded 4 facts and 1 rule
        assert_eq!(db.len(), 4);
        assert_eq!(rules.len(), 1);

        // Step 3: Ground the rule
        let grandparent_rule = &rules[0];
        let derived_facts = ground_rule(grandparent_rule, &db);

        // Should derive two grandparent facts:
        // grandparent(john, alice) from parent(john, mary) + parent(mary, alice)
        // grandparent(bob, dave) from parent(bob, charlie) + parent(charlie, dave)
        assert_eq!(derived_facts.len(), 2);

        // Step 4: Add derived facts to database
        for fact in derived_facts {
            db.insert(fact).unwrap();
        }

        // Step 5: Query for grandparents
        let grandparent_pred = Intern::new("grandparent".to_string());
        let grandparent_facts = db.get_by_predicate(&grandparent_pred);

        assert_eq!(grandparent_facts.len(), 2);

        // Verify we can query with patterns
        let query_pattern = make_atom("grandparent", vec![atom_const("john"), var("Z")]);
        let results = db.query(&query_pattern);

        assert_eq!(results.len(), 1);
        let z = Intern::new("Z".to_string());
        assert_eq!(results[0].get(&z), Some(&atom_const("alice")));

        // Query for all grandparent relationships
        let all_grandparents = make_atom("grandparent", vec![var("X"), var("Z")]);
        let all_results = db.query(&all_grandparents);

        assert_eq!(all_results.len(), 2);
    }

    // Range expansion tests
    #[test]
    fn test_expand_range_simple() {
        let const_env = ConstantEnv::new();
        let start = Term::Constant(Value::Integer(1));
        let end = Term::Constant(Value::Integer(3));

        let result = expand_range(&start, &end, &const_env);
        assert_eq!(result, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_expand_range_with_constants() {
        let mut const_env = ConstantEnv::new();
        const_env.define(Intern::new("max".to_string()), Value::Integer(5));

        let start = Term::Constant(Value::Integer(3));
        let end = Term::Constant(Value::Atom(Intern::new("max".to_string())));

        let result = expand_range(&start, &end, &const_env);
        assert_eq!(result, Some(vec![3, 4, 5]));
    }

    #[test]
    fn test_expand_atom_ranges_single_range() {
        let const_env = ConstantEnv::new();
        let atom = make_atom(
            "cell",
            vec![
                Term::Range(
                    Box::new(Term::Constant(Value::Integer(1))),
                    Box::new(Term::Constant(Value::Integer(3))),
                ),
                atom_const("a"),
            ],
        );

        let result = expand_atom_ranges(&atom, &const_env);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], make_atom("cell", vec![int(1), atom_const("a")]));
        assert_eq!(result[1], make_atom("cell", vec![int(2), atom_const("a")]));
        assert_eq!(result[2], make_atom("cell", vec![int(3), atom_const("a")]));
    }

    #[test]
    fn test_expand_atom_ranges_multiple_ranges() {
        let const_env = ConstantEnv::new();
        let atom = make_atom(
            "cell",
            vec![
                Term::Range(
                    Box::new(Term::Constant(Value::Integer(1))),
                    Box::new(Term::Constant(Value::Integer(2))),
                ),
                Term::Range(
                    Box::new(Term::Constant(Value::Integer(10))),
                    Box::new(Term::Constant(Value::Integer(11))),
                ),
            ],
        );

        let result = expand_atom_ranges(&atom, &const_env);
        assert_eq!(result.len(), 4); // 2 * 2 = 4 combinations
        assert_eq!(result[0], make_atom("cell", vec![int(1), int(10)]));
        assert_eq!(result[1], make_atom("cell", vec![int(1), int(11)]));
        assert_eq!(result[2], make_atom("cell", vec![int(2), int(10)]));
        assert_eq!(result[3], make_atom("cell", vec![int(2), int(11)]));
    }

    #[test]
    fn test_ground_choice_element_no_condition() {
        let db = FactDatabase::new();
        let const_env = ConstantEnv::new();

        let element = ChoiceElement {
            atom: make_atom("selected", vec![atom_const("a")]),
            condition: vec![],
        };

        let result = ground_choice_element(&element, &db, &const_env);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], make_atom("selected", vec![atom_const("a")]));
    }

    #[test]
    fn test_ground_choice_element_with_condition() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("item", vec![atom_const("a")])).unwrap();
        db.insert(make_atom("item", vec![atom_const("b")])).unwrap();

        let const_env = ConstantEnv::new();

        let element = ChoiceElement {
            atom: make_atom("selected", vec![var("X")]),
            condition: vec![Literal::Positive(make_atom("item", vec![var("X")]))],
        };

        let result = ground_choice_element(&element, &db, &const_env);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&make_atom("selected", vec![atom_const("a")])));
        assert!(result.contains(&make_atom("selected", vec![atom_const("b")])));
    }

    #[test]
    fn test_ground_choice_element_with_range() {
        let db = FactDatabase::new();
        let const_env = ConstantEnv::new();

        let element = ChoiceElement {
            atom: make_atom(
                "cell",
                vec![
                    Term::Range(
                        Box::new(Term::Constant(Value::Integer(1))),
                        Box::new(Term::Constant(Value::Integer(2))),
                    ),
                    atom_const("solid"),
                ],
            ),
            condition: vec![],
        };

        let result = ground_choice_element(&element, &db, &const_env);
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0],
            make_atom("cell", vec![int(1), atom_const("solid")])
        );
        assert_eq!(
            result[1],
            make_atom("cell", vec![int(2), atom_const("solid")])
        );
    }

    // Edge case tests for range expansion

    #[test]
    fn test_expand_range_backwards() {
        // Backwards range should produce empty result
        let const_env = ConstantEnv::new();
        let start = Term::Constant(Value::Integer(5));
        let end = Term::Constant(Value::Integer(1));

        let result = expand_range(&start, &end, &const_env);
        // Backwards range produces empty vector
        assert_eq!(result, Some(vec![]));
    }

    #[test]
    fn test_expand_range_single_element() {
        let const_env = ConstantEnv::new();
        let start = Term::Constant(Value::Integer(5));
        let end = Term::Constant(Value::Integer(5));

        let result = expand_range(&start, &end, &const_env);
        assert_eq!(result, Some(vec![5]));
    }

    #[test]
    fn test_expand_range_negative_values() {
        let const_env = ConstantEnv::new();
        let start = Term::Constant(Value::Integer(-3));
        let end = Term::Constant(Value::Integer(-1));

        let result = expand_range(&start, &end, &const_env);
        assert_eq!(result, Some(vec![-3, -2, -1]));
    }

    #[test]
    fn test_expand_range_crossing_zero() {
        let const_env = ConstantEnv::new();
        let start = Term::Constant(Value::Integer(-2));
        let end = Term::Constant(Value::Integer(2));

        let result = expand_range(&start, &end, &const_env);
        assert_eq!(result, Some(vec![-2, -1, 0, 1, 2]));
    }

    #[test]
    fn test_expand_range_with_undefined_constant() {
        let const_env = ConstantEnv::new();
        let start = Term::Constant(Value::Integer(1));
        let end = Term::Constant(Value::Atom(Intern::new("undefined".to_string())));

        let result = expand_range(&start, &end, &const_env);
        // Undefined constant can't be expanded
        assert_eq!(result, None);
    }

    #[test]
    fn test_expand_range_with_non_integer_constant() {
        let const_env = ConstantEnv::new();
        let start = Term::Constant(Value::Atom(Intern::new("not_a_number".to_string())));
        let end = Term::Constant(Value::Integer(5));

        let result = expand_range(&start, &end, &const_env);
        assert_eq!(result, None);
    }

    #[test]
    fn test_expand_atom_ranges_no_ranges() {
        let const_env = ConstantEnv::new();
        let atom = make_atom("cell", vec![atom_const("a"), atom_const("b")]);

        let result = expand_atom_ranges(&atom, &const_env);
        // No ranges, should return single atom unchanged
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], atom);
    }

    #[test]
    fn test_expand_atom_ranges_empty_range() {
        let const_env = ConstantEnv::new();
        // 5..1 is a backwards range (empty)
        let atom = make_atom(
            "cell",
            vec![
                Term::Range(
                    Box::new(Term::Constant(Value::Integer(5))),
                    Box::new(Term::Constant(Value::Integer(1))),
                ),
                atom_const("x"),
            ],
        );

        let result = expand_atom_ranges(&atom, &const_env);
        // Empty range produces no atoms
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_expand_atom_ranges_mixed_terms() {
        let const_env = ConstantEnv::new();
        let atom = make_atom(
            "cell",
            vec![
                atom_const("prefix"),
                Term::Range(
                    Box::new(Term::Constant(Value::Integer(1))),
                    Box::new(Term::Constant(Value::Integer(2))),
                ),
                atom_const("suffix"),
            ],
        );

        let result = expand_atom_ranges(&atom, &const_env);
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0],
            make_atom(
                "cell",
                vec![atom_const("prefix"), int(1), atom_const("suffix")]
            )
        );
        assert_eq!(
            result[1],
            make_atom(
                "cell",
                vec![atom_const("prefix"), int(2), atom_const("suffix")]
            )
        );
    }

    #[test]
    fn test_ground_choice_element_empty_condition_result() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("item", vec![atom_const("a")])).unwrap();

        let const_env = ConstantEnv::new();

        // Condition that won't match anything
        let element = ChoiceElement {
            atom: make_atom("selected", vec![var("X")]),
            condition: vec![Literal::Positive(make_atom("nonexistent", vec![var("X")]))],
        };

        let result = ground_choice_element(&element, &db, &const_env);
        // No items match the condition
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_ground_choice_element_multiple_conditions() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("item", vec![atom_const("a")])).unwrap();
        db.insert(make_atom("item", vec![atom_const("b")])).unwrap();
        db.insert(make_atom("valid", vec![atom_const("a")]))
            .unwrap();
        db.insert(make_atom("safe", vec![atom_const("a")])).unwrap();

        let const_env = ConstantEnv::new();

        // Multiple conditions: both must be satisfied
        let element = ChoiceElement {
            atom: make_atom("selected", vec![var("X")]),
            condition: vec![
                Literal::Positive(make_atom("item", vec![var("X")])),
                Literal::Positive(make_atom("valid", vec![var("X")])),
                Literal::Positive(make_atom("safe", vec![var("X")])),
            ],
        };

        let result = ground_choice_element(&element, &db, &const_env);
        // Only 'a' satisfies all three conditions
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], make_atom("selected", vec![atom_const("a")]));
    }

    #[test]
    fn test_ground_choice_rule_with_body_variables() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("room", vec![atom_const("r1")]))
            .unwrap();
        db.insert(make_atom("item", vec![atom_const("sword")]))
            .unwrap();
        db.insert(make_atom("item", vec![atom_const("shield")]))
            .unwrap();

        let const_env = ConstantEnv::new();

        // Choice rule with body
        let choice = ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: make_atom("place", vec![var("I"), var("R")]),
                condition: vec![Literal::Positive(make_atom("item", vec![var("I")]))],
            }],
            body: vec![Literal::Positive(make_atom("room", vec![var("R")]))],
        };

        let result = ground_choice_rule(&choice, &db, &const_env);
        // 1 room * 2 items = 2 possible placements
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_ground_choice_rule_empty_body() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("item", vec![atom_const("a")])).unwrap();

        let const_env = ConstantEnv::new();

        let choice = ChoiceRule {
            lower_bound: Some(Term::Constant(Value::Integer(1))),
            upper_bound: Some(Term::Constant(Value::Integer(1))),
            elements: vec![ChoiceElement {
                atom: make_atom("selected", vec![var("X")]),
                condition: vec![Literal::Positive(make_atom("item", vec![var("X")]))],
            }],
            body: vec![],
        };

        let result = ground_choice_rule(&choice, &db, &const_env);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], make_atom("selected", vec![atom_const("a")]));
    }

    #[test]
    fn test_satisfy_body_with_builtin_comparison() {
        // Test: number(X), X > 5 with number(7) in the database
        let mut db = FactDatabase::new();
        db.insert(make_atom("number", vec![Term::Constant(Value::Integer(7))]))
            .unwrap();
        db.insert(make_atom("number", vec![Term::Constant(Value::Integer(3))]))
            .unwrap();

        let body = vec![
            Literal::Positive(make_atom("number", vec![var("X")])),
            Literal::Positive(Atom {
                predicate: Intern::new(">".to_string()),
                terms: vec![var("X"), Term::Constant(Value::Integer(5))],
            }),
        ];

        let result = satisfy_body(&body, &db);

        // Should have exactly one substitution: {X -> 7}
        assert_eq!(
            result.len(),
            1,
            "Expected 1 substitution, got {}: {:#?}",
            result.len(),
            result
        );

        let x_var = Intern::new("X".to_string());
        let bound_value = result[0].get(&x_var).expect("X should be bound");
        match bound_value {
            Term::Constant(Value::Integer(7)) => {}
            _ => panic!("Expected X to be bound to 7, got {:?}", bound_value),
        }
    }

    #[test]
    fn test_builtin_filters_identical_in_pure_and_mixed() {
        let mut full_db = FactDatabase::new();
        full_db
            .insert(make_atom("number", vec![Term::Constant(Value::Integer(7))]))
            .unwrap();
        full_db
            .insert(make_atom("number", vec![Term::Constant(Value::Integer(3))]))
            .unwrap();

        let mut delta = FactDatabase::new();
        delta
            .insert(make_atom("number", vec![Term::Constant(Value::Integer(7))]))
            .unwrap();

        let body = vec![
            Literal::Positive(make_atom("number", vec![var("X")])),
            Literal::Positive(Atom {
                predicate: Intern::new(">".to_string()),
                terms: vec![var("X"), Term::Constant(Value::Integer(5))],
            }),
        ];

        let pure = normalize_substitutions(satisfy_body(&body, &full_db));

        for delta_pos in 0..body.len() {
            let mixed =
                normalize_substitutions(satisfy_body_mixed(&body, &delta, &full_db, delta_pos));
            assert_eq!(
                mixed, pure,
                "mixed results differed at delta_pos {}",
                delta_pos
            );
        }
    }

    #[test]
    fn test_ground_rule_with_builtin_comparison() {
        // Test: large(X) :- number(X), X > 5
        let mut db = FactDatabase::new();
        db.insert(make_atom("number", vec![Term::Constant(Value::Integer(7))]))
            .unwrap();
        db.insert(make_atom("number", vec![Term::Constant(Value::Integer(3))]))
            .unwrap();

        let rule = Rule {
            head: make_atom("large", vec![var("X")]),
            body: vec![
                Literal::Positive(make_atom("number", vec![var("X")])),
                Literal::Positive(Atom {
                    predicate: Intern::new(">".to_string()),
                    terms: vec![var("X"), Term::Constant(Value::Integer(5))],
                }),
            ],
        };

        let result = ground_rule(&rule, &db);

        // Should derive large(7)
        assert_eq!(
            result.len(),
            1,
            "Expected 1 derived fact, got {}: {:#?}",
            result.len(),
            result
        );

        match &result[0].terms[0] {
            Term::Constant(Value::Integer(7)) => {}
            other => panic!("Expected large(7), got large({:?})", other),
        }
    }

    #[test]
    fn test_negation_identical_in_pure_and_mixed() {
        let mut full_db = FactDatabase::new();
        full_db
            .insert(make_atom("person", vec![atom_const("alice")]))
            .unwrap();
        full_db
            .insert(make_atom("person", vec![atom_const("bob")]))
            .unwrap();
        full_db
            .insert(make_atom("enemy", vec![atom_const("bob")]))
            .unwrap();

        let mut delta = FactDatabase::new();
        delta
            .insert(make_atom("person", vec![atom_const("alice")]))
            .unwrap();

        let body = vec![
            Literal::Positive(make_atom("person", vec![var("X")])),
            Literal::Negative(make_atom("enemy", vec![var("X")])),
        ];

        let pure = normalize_substitutions(satisfy_body(&body, &full_db));

        for delta_pos in 0..body.len() {
            let mixed =
                normalize_substitutions(satisfy_body_mixed(&body, &delta, &full_db, delta_pos));
            assert_eq!(
                mixed, pure,
                "mixed results differed at delta_pos {}",
                delta_pos
            );
        }
    }

    // Tests for choice rule body splitting
    #[test]
    fn test_ground_choice_rule_no_body_returns_single_group() {
        // Choice rule with no body: { selected(X) : item(X) }.
        // Should return ONE group containing all grounded atoms
        let mut db = FactDatabase::new();
        db.insert(make_atom("item", vec![atom_const("a")])).unwrap();
        db.insert(make_atom("item", vec![atom_const("b")])).unwrap();

        let const_env = ConstantEnv::new();

        let choice = ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: make_atom("selected", vec![var("X")]),
                condition: vec![Literal::Positive(make_atom("item", vec![var("X")]))],
            }],
            body: vec![], // NO BODY
        };

        // Current function returns Vec<Atom>, but we need Vec<Vec<Atom>>
        // For now, test what we have
        let result = ground_choice_rule(&choice, &db, &const_env);

        // Should have 2 atoms: selected(a), selected(b)
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_ground_choice_rule_with_body_should_split() {
        // Choice rule with body: 1 { has_weapon(P, W) : weapon(W) } 1 :- player(P).
        // Should return TWO groups (one per player)
        let mut db = FactDatabase::new();
        db.insert(make_atom("player", vec![atom_const("alice")]))
            .unwrap();
        db.insert(make_atom("player", vec![atom_const("bob")]))
            .unwrap();
        db.insert(make_atom("weapon", vec![atom_const("sword")]))
            .unwrap();
        db.insert(make_atom("weapon", vec![atom_const("bow")]))
            .unwrap();

        let const_env = ConstantEnv::new();

        let choice = ChoiceRule {
            lower_bound: Some(Term::Constant(Value::Integer(1))),
            upper_bound: Some(Term::Constant(Value::Integer(1))),
            elements: vec![ChoiceElement {
                atom: make_atom("has_weapon", vec![var("P"), var("W")]),
                condition: vec![Literal::Positive(make_atom("weapon", vec![var("W")]))],
            }],
            body: vec![Literal::Positive(make_atom("player", vec![var("P")]))],
        };

        // Current implementation - returns flat list of 4 atoms
        let result = ground_choice_rule(&choice, &db, &const_env);

        // Currently returns 4 atoms (all combined)
        assert_eq!(result.len(), 4);

        // But we WANT 2 groups of 2 atoms each
        // This test documents the CURRENT (incorrect) behavior
        // We'll add a new function that returns Vec<Vec<Atom>>
    }

    #[test]
    fn test_ground_choice_rule_split_no_body() {
        // NEW FUNCTION: ground_choice_rule_split
        // With no body, should return single group
        let mut db = FactDatabase::new();
        db.insert(make_atom("item", vec![atom_const("a")])).unwrap();
        db.insert(make_atom("item", vec![atom_const("b")])).unwrap();

        let const_env = ConstantEnv::new();

        let choice = ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: make_atom("selected", vec![var("X")]),
                condition: vec![Literal::Positive(make_atom("item", vec![var("X")]))],
            }],
            body: vec![],
        };

        let result = ground_choice_rule_split(&choice, &db, &const_env);

        // Should return 1 group with 2 atoms
        assert_eq!(result.len(), 1, "Expected 1 group");
        assert_eq!(result[0].len(), 2, "Expected 2 atoms in the group");

        // Verify atoms
        let atoms: Vec<String> = result[0]
            .iter()
            .map(|a| format!("{:?}", a.predicate))
            .collect();
        assert!(atoms.iter().all(|s| s.contains("selected")));
    }

    #[test]
    fn test_ground_choice_rule_split_with_body() {
        // NEW FUNCTION: ground_choice_rule_split
        // With body that grounds to 2 substitutions, should return 2 groups
        let mut db = FactDatabase::new();
        db.insert(make_atom("player", vec![atom_const("alice")]))
            .unwrap();
        db.insert(make_atom("player", vec![atom_const("bob")]))
            .unwrap();
        db.insert(make_atom("weapon", vec![atom_const("sword")]))
            .unwrap();
        db.insert(make_atom("weapon", vec![atom_const("bow")]))
            .unwrap();

        let const_env = ConstantEnv::new();

        let choice = ChoiceRule {
            lower_bound: Some(Term::Constant(Value::Integer(1))),
            upper_bound: Some(Term::Constant(Value::Integer(1))),
            elements: vec![ChoiceElement {
                atom: make_atom("has_weapon", vec![var("P"), var("W")]),
                condition: vec![Literal::Positive(make_atom("weapon", vec![var("W")]))],
            }],
            body: vec![Literal::Positive(make_atom("player", vec![var("P")]))],
        };

        let result = ground_choice_rule_split(&choice, &db, &const_env);

        // Should return 2 groups (one per player)
        assert_eq!(result.len(), 2, "Expected 2 groups (one per player)");

        // Each group should have 2 atoms (one per weapon)
        assert_eq!(result[0].len(), 2, "Expected 2 atoms in first group");
        assert_eq!(result[1].len(), 2, "Expected 2 atoms in second group");

        // Verify first group is for one player (either alice or bob)
        let first_group_player = &result[0][0].terms[0];
        assert!(
            result[0]
                .iter()
                .all(|atom| &atom.terms[0] == first_group_player),
            "All atoms in first group should be for same player"
        );

        // Verify second group is for the other player
        let second_group_player = &result[1][0].terms[0];
        assert!(
            result[1]
                .iter()
                .all(|atom| &atom.terms[0] == second_group_player),
            "All atoms in second group should be for same player"
        );

        // Verify different players
        assert_ne!(
            first_group_player, second_group_player,
            "Each group should be for a different player"
        );
    }
