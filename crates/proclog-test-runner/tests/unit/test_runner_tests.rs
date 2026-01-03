    use super::*;
    use internment::Intern;
    use proclog_ast::{Literal, Program, Rule, Statement, Term, Value};
    use proclog_parser::{ParseError, SrcId};

    fn parse_program(input: &str) -> Result<Program, Vec<ParseError>> {
        proclog_parser::parse_program(input, SrcId::empty())
    }

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

    #[test]
    fn prepare_environment_collects_statements_and_constraints() {
        let input = r#"
            #test "prep constraint" {
                parent(alice, bob).
                :- parent(alice, carol).
                ancestor(X, Y) :- parent(X, Y).

                ?- parent(alice, bob).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let prepared =
            prepare_test_environment(&[], test_block).expect("Environment preparation failed");

        assert_eq!(prepared.initial_facts.len(), 1);
        assert_eq!(prepared.rules.len(), 1);
        assert_eq!(prepared.constraints.len(), 1);
        assert!(!prepared.has_asp_statements());
    }

    #[test]
    fn evaluate_prepared_env_runs_stratified_for_constraints() {
        let input = r#"
            #test "stratified" {
                parent(alice, bob).
                :- parent(alice, carol).
                ancestor(X, Y) :- parent(X, Y).

                ?- parent(alice, bob).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let prepared =
            prepare_test_environment(&[], test_block).expect("Environment preparation failed");

        let answer_sets = evaluate_prepared_env(&prepared, false).expect("Evaluation failed");
        assert_eq!(
            answer_sets.len(),
            1,
            "Stratified evaluation should return single answer set"
        );

        let parent_atom = make_atom("parent", vec![atom_const("alice"), atom_const("bob")]);
        assert!(
            answer_sets[0].atoms.contains(&parent_atom),
            "Expected answer set to contain derived parent fact"
        );
    }

    #[test]
    fn evaluate_prepared_env_runs_asp_for_choice_rules() {
        let input = r#"
            #test "asp" {
                option(a).
                option(b).
                1 { choose(X) : option(X) } 1.

                ?- choose(X).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let prepared =
            prepare_test_environment(&[], test_block).expect("Environment preparation failed");
        assert!(
            prepared.has_asp_statements(),
            "Choice rule should enable ASP path"
        );

        let answer_sets = evaluate_prepared_env(&prepared, false).expect("Evaluation failed");
        assert_eq!(
            answer_sets.len(),
            2,
            "Choice rule should yield two answer sets for two options",
        );
    }

    #[test]
    fn rule_fallback_uses_library_grounding() {
        let mut db = FactDatabase::new();
        let parent_fact = make_atom("parent", vec![atom_const("alice")]);
        db.insert(parent_fact).unwrap();

        let rules = vec![Rule {
            head: make_atom("person", vec![var("X")]),
            body: vec![Literal::Positive(make_atom("parent", vec![var("X")]))],
        }];

        let query = proclog_ast::Query {
            body: vec![Literal::Positive(make_atom(
                "person",
                vec![atom_const("alice")],
            ))],
        };

        assert!(evaluate_query(&query, &db).is_empty());
        assert!(should_use_rule_fallback(&query, &rules));

        let results = evaluate_query_with_rules(&query, &db, &rules);
        assert_eq!(results.len(), 1, "Fallback should derive person(alice)");
    }

    #[test]
    fn rule_fallback_handles_builtin_comparison() {
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "item",
            vec![
                atom_const("item1"),
                atom_const("sword"),
                atom_const("common"),
                Term::Constant(Value::Integer(1)),
                Term::Constant(Value::Integer(10)),
            ],
        ))
        .unwrap();
        db.insert(make_atom(
            "item",
            vec![
                atom_const("item2"),
                atom_const("sword"),
                atom_const("rare"),
                Term::Constant(Value::Integer(2)),
                Term::Constant(Value::Integer(20)),
            ],
        ))
        .unwrap();

        let rules = vec![Rule {
            head: make_atom("better_than_common", vec![var("Item")]),
            body: vec![
                Literal::Positive(make_atom(
                    "item",
                    vec![
                        var("Item"),
                        var("Type"),
                        var("Rarity"),
                        var("RequiredLevel"),
                        var("Price"),
                    ],
                )),
                Literal::Positive(make_atom("=", vec![var("Rarity"), atom_const("rare")])),
            ],
        }];

        let query = proclog_ast::Query {
            body: vec![Literal::Positive(make_atom(
                "better_than_common",
                vec![atom_const("item2")],
            ))],
        };

        assert!(evaluate_query(&query, &db).is_empty());

        let derived = ground_rule(&rules[0], &db);
        assert!(
            derived.is_empty(),
            "Expected library grounding to skip equality without numeric terms: {:?}",
            derived
        );

        let results = evaluate_query_with_rules(&query, &db, &rules);
        assert!(
            !results.is_empty(),
            "Expected fallback to derive better_than_common(item2)"
        );
    }

    #[test]
    fn test_simple_passing_test() {
        let input = r#"
            #test "basic test" {
                parent(john, mary).

                ?- parent(john, mary).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block, false);
        assert!(result.passed, "Test should pass");
        assert_eq!(result.passed_cases, 1);
    }

    #[test]
    fn test_simple_failing_test() {
        let input = r#"
            #test "should fail" {
                parent(john, mary).

                ?- parent(alice, bob).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block, false);
        assert!(!result.passed, "Test should fail");
        assert_eq!(result.passed_cases, 0);
    }

    #[test]
    fn test_constraint_violation_fails_test_case() {
        let input = r#"
            #test "constraint failure" {
                parent(alice, bob).
                :- parent(alice, bob).

                ?- parent(alice, bob).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block, false);
        assert!(
            !result.passed,
            "Test should fail due to constraint violation"
        );
        assert_eq!(result.passed_cases, 0);
        assert_eq!(result.case_results.len(), 1);
        assert!(
            result.case_results[0].message.contains("Constraint"),
            "Expected constraint failure message, got: {}",
            result.case_results[0].message
        );
    }

    #[test]
    fn test_constraint_constant_substitution() {
        let input = r#"
            #test "constraint constant" {
                #const banned = bob.
                parent(bob).
                :- parent(banned).

                ?- parent(bob).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block, false);
        assert!(
            !result.passed,
            "Test should fail due to constraint with substituted constant: {:?}",
            result.case_results
        );
    }

    #[test]
    fn test_choice_rule_constant_substitution() {
        let input = r#"
            #test "choice rule constants" {
                #const required = 1.
                #const flag_value = ready.

                option(a).
                guard(ready, a).
                state(ready).

                required { choose(X) : option(X), guard(flag_value, X) } required :- state(flag_value).

                ?- choose(X).
                + choose(a).
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block, false);
        assert!(
            result.passed,
            "Choice rule with constant substitution should succeed: {:?}",
            result.case_results
        );
    }

    #[test]
    fn test_query_constant_substitution() {
        let input = r#"
            #test "query constant" {
                #const target = bob.
                parent(bob).

                ?- parent(target).
                + parent(bob).
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block, false);
        assert!(result.passed, "Query constants should be substituted");
        assert_eq!(result.passed_cases, 1);
    }

    #[test]
    fn test_with_rules() {
        let input = r#"
            #test "transitive closure" {
                edge(a, b).
                edge(b, c).
                path(X, Y) :- edge(X, Y).
                path(X, Z) :- path(X, Y), edge(Y, Z).

                ?- path(a, c).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block, false);
        assert!(result.passed, "Test should pass: {:?}", result.case_results);
    }

    #[test]
    fn test_with_comments_between_queries() {
        let input = r#"
            #test "health status checks" {
                #const healthy_threshold = 50.
                #const critical_health = 20.

                character(warrior, 85, 15, 500).
                character(mage, 15, 45, 1200).
                character(rogue, 60, 20, 800).

                healthy(Name) :- character(Name, Health, _, _), Health >= healthy_threshold.
                critical(Name) :- character(Name, Health, _, _), Health <= critical_health.

                % Warrior is healthy
                ?- healthy(warrior).
                + true.

                % Mage is critical
                ?- critical(mage).
                + true.

                % Rogue is healthy
                ?- healthy(rogue).
                + true.

                % Warrior is not critical
                ?- critical(warrior).
                - true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        assert_eq!(test_block.statements.len(), 7);
        assert_eq!(test_block.test_cases.len(), 4);

        use proclog_core::ConstantEnv;
        use proclog_eval::semi_naive_evaluation;

        let const_env = ConstantEnv::from_statements(&test_block.statements);
        let mut initial_facts = FactDatabase::new();
        let mut rules_vec = Vec::new();

        for statement in &test_block.statements {
            match statement {
                Statement::Fact(fact) => {
                    initial_facts
                        .insert(const_env.substitute_atom(&fact.atom))
                        .unwrap();
                }
                Statement::Rule(rule) => {
                    let head = const_env.substitute_atom(&rule.head);
                    let body = rule
                        .body
                        .iter()
                        .map(|lit| match lit {
                            Literal::Positive(atom) => {
                                Literal::Positive(const_env.substitute_atom(atom))
                            }
                            Literal::Negative(atom) => {
                                Literal::Negative(const_env.substitute_atom(atom))
                            }
                            Literal::Aggregate(agg) => Literal::Aggregate(agg.clone()),
                        })
                        .collect();
                    rules_vec.push(Rule { head, body });
                }
                Statement::ConstDecl(_) => {}
                other => panic!("Unexpected statement in test block: {:?}", other),
            }
        }

        let db = if rules_vec.is_empty() {
            initial_facts.clone()
        } else {
            semi_naive_evaluation(&rules_vec, initial_facts.clone())
                .expect("semi-naive evaluation should succeed")
        };

        let healthy_pred = Intern::new("healthy".to_string());
        assert!(
            db.get_by_predicate(&healthy_pred)
                .iter()
                .any(|atom| matches!(
                    atom.terms.as_slice(),
                    [Term::Constant(Value::Atom(name))] if name.as_ref() == "warrior"
                )),
            "Derived database should contain healthy(warrior)"
        );

        let critical_pred = Intern::new("critical".to_string());
        assert!(
            db.get_by_predicate(&critical_pred)
                .iter()
                .any(|atom| matches!(
                    atom.terms.as_slice(),
                    [Term::Constant(Value::Atom(name))] if name.as_ref() == "mage"
                )),
            "Derived database should contain critical(mage)"
        );

        let result = run_test_block(&[], test_block, false);
        assert!(
            result.passed,
            "Test block should pass: {:?}",
            result.case_results
        );
        assert_eq!(result.passed_cases, test_block.test_cases.len());
    }

    #[test]
    fn test_choice_rule_body_constant_substitution() {
        let input = r#"
            #test "choice body constants" {
                #const chosen = hero.

                available(hero).
                trigger(chosen).

                1 { select(X) : available(X) } 1 :- trigger(chosen).

                ?- select(hero).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block, false);
        assert!(
            result.passed,
            "Choice rule body constants should be substituted: {:?}",
            result.case_results
        );
        assert_eq!(result.passed_cases, 1);
    }

    #[test]
    fn test_non_ground_fact_in_test_block_reports_error() {
        let input = r#"
            #test "non-ground fact" {
                parent(X, mary).

                ?- parent(john, mary).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block, false);
        assert!(
            !result.passed,
            "Test should fail due to non-ground fact: {:?}",
            result.case_results
        );
        assert_eq!(result.passed_cases, 0);
        assert!(!result.case_results.is_empty());
        let message = &result.case_results[0].message;
        assert!(
            message.contains("Failed to prepare test inputs"),
            "Expected preparation failure message, got: {}",
            message
        );
        assert!(
            message.contains("non-ground fact parent(X, mary)"),
            "Expected non-ground fact details in message, got: {}",
            message
        );
    }

    #[test]
    fn test_non_ground_fact_in_base_program_reports_error() {
        let input = r#"
            parent(X, mary).

            #test "non-ground base" {
                ?- parent(john, mary).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let mut base_statements = Vec::new();
        let mut test_block = None;

        for statement in &program.statements {
            match statement {
                Statement::Test(tb) => {
                    test_block = Some(tb.clone());
                }
                other => base_statements.push(other.clone()),
            }
        }

        let test_block = test_block.expect("Expected test block");
        let result = run_test_block(&base_statements, &test_block, false);

        assert!(
            !result.passed,
            "Test should fail due to non-ground base fact: {:?}",
            result.case_results
        );
        assert_eq!(result.passed_cases, 0);
        assert!(!result.case_results.is_empty());
        let message = &result.case_results[0].message;
        assert!(
            message.contains("non-ground fact parent(X, mary)"),
            "Expected non-ground fact details in message, got: {}",
            message
        );
    }

    #[test]
    fn test_available_at_level_rule_fallback() {
        let input = r#"
            #test "quest level requirements" {
                quest(gather_herbs).
                quest_level(gather_herbs, 1).

                available_at_level(Quest, Level) :-
                    quest(Quest),
                    quest_level(Quest, Required),
                    Level >= Required.

                ?- available_at_level(gather_herbs, 1).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block, false);
        assert!(result.passed, "Fallback should allow rule evaluation");
        assert_eq!(result.passed_cases, 1);
    }

    #[test]
    fn test_level_appropriate_rule_fallback() {
        let input = r#"
            #test "item level requirements" {
                item(iron_sword, sword, common, 1, 10).
                item(steel_sword, sword, rare, 5, 50).

                level_appropriate(Item, CharLevel) :-
                    item(Item, _, _, ItemLevel, _),
                    CharLevel >= ItemLevel.

                ?- level_appropriate(steel_sword, 5).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block, false);
        assert!(result.passed, "Fallback should handle item level checks");
        assert_eq!(result.passed_cases, 1);
    }

    #[test]
    fn test_better_than_common_rule_fallback() {
        let input = r#"
            #test "rarity check" {
                item(item1, sword, common, 1, 10).
                item(item2, sword, rare, 2, 20).

                better_than_common(Item) :- item(Item, _, Rarity, _, _), Rarity = rare.
                better_than_common(Item) :- item(Item, _, Rarity, _, _), Rarity = legendary.

                ?- better_than_common(item2).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block, false);
        assert!(result.passed, "Fallback should handle rarity checks");
        assert_eq!(result.passed_cases, 1);
    }
