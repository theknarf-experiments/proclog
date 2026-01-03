//! Integration tests for query evaluation
//!
//! These tests demonstrate complete workflows:
//! 1. Parse a program
//! 2. Evaluate it (derive facts)
//! 3. Query the results

#[cfg(test)]
mod query_integration_tests {
    use crate::evaluation::semi_naive_evaluation;
    use crate::parser::{ParseError, SrcId};
    use crate::query::{evaluate_query, extract_bindings, query_variables};

    fn parse_program(input: &str) -> Result<crate::ast::Program, Vec<ParseError>> {
        crate::parser::parse_program(input, SrcId::empty())
    }

    fn parse_query(input: &str) -> Result<crate::ast::Query, Vec<ParseError>> {
        crate::parser::parse_query(input, SrcId::empty())
    }

    #[test]
    fn test_query_after_evaluation() {
        // Program with rules
        let program_text = r#"
            parent(john, mary).
            parent(mary, sue).
            parent(alice, bob).

            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");

        // Extract facts and rules
        let mut facts = crate::database::FactDatabase::new();
        let mut rules = vec![];

        for statement in &program.statements {
            match statement {
                crate::ast::Statement::Fact(fact) => {
                    facts.insert(fact.atom.clone()).unwrap();
                }
                crate::ast::Statement::Rule(rule) => {
                    rules.push(rule.clone());
                }
                _ => {}
            }
        }

        // Evaluate to derive all facts
        let result_db =
            semi_naive_evaluation(&rules, facts).expect("semi-naive evaluation should succeed");

        // Query 1: Ground query - is john an ancestor of sue?
        let query1 = parse_query("?- ancestor(john, sue).").expect("Failed to parse query");
        let results1 = evaluate_query(&query1, &result_db);
        assert_eq!(
            results1.len(),
            1,
            "John should be an ancestor of Sue (via Mary)"
        );

        // Query 2: Variable query - who are john's descendants?
        let query2 = parse_query("?- ancestor(john, X).").expect("Failed to parse query");
        let results2 = evaluate_query(&query2, &result_db);
        assert_eq!(
            results2.len(),
            2,
            "John should have 2 descendants: Mary and Sue"
        );

        // Query 3: Multiple variables - all ancestor relationships
        let query3 = parse_query("?- ancestor(X, Y).").expect("Failed to parse query");
        let results3 = evaluate_query(&query3, &result_db);
        // Direct: john->mary, mary->sue, alice->bob = 3
        // Transitive: john->sue = 1
        // Total = 4
        assert_eq!(
            results3.len(),
            4,
            "Should have 4 ancestor relationships (3 direct + 1 transitive), got {}",
            results3.len()
        );
    }

    #[test]
    fn test_query_with_negation_after_evaluation() {
        let program_text = r#"
            person(john).
            person(mary).
            person(alice).

            dead(john).

            alive(X) :- person(X), not dead(X).
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");

        let mut facts = crate::database::FactDatabase::new();
        let mut rules = vec![];

        for statement in &program.statements {
            match statement {
                crate::ast::Statement::Fact(fact) => {
                    facts.insert(fact.atom.clone()).unwrap();
                }
                crate::ast::Statement::Rule(rule) => {
                    rules.push(rule.clone());
                }
                _ => {}
            }
        }

        let result_db =
            semi_naive_evaluation(&rules, facts).expect("semi-naive evaluation should succeed");

        // Query for alive people
        let query = parse_query("?- alive(X).").expect("Failed to parse query");
        let results = evaluate_query(&query, &result_db);
        assert_eq!(
            results.len(),
            2,
            "Should find 2 alive people: Mary and Alice"
        );
    }

    #[test]
    fn test_query_variable_extraction() {
        let program_text = r#"
            parent(john, mary).
            parent(alice, bob).
            age(mary, 25).
            age(bob, 5).
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");

        let mut facts = crate::database::FactDatabase::new();
        for statement in &program.statements {
            if let crate::ast::Statement::Fact(fact) = statement {
                facts.insert(fact.atom.clone()).unwrap();
            }
        }

        // Query: ?- parent(P, C), age(C, A).
        let query = parse_query("?- parent(P, C), age(C, A).").expect("Failed to parse query");
        let results = evaluate_query(&query, &facts);

        assert_eq!(
            results.len(),
            2,
            "Should find 2 parent-child pairs with ages"
        );

        // Extract variable bindings
        let vars = query_variables(&query);
        assert_eq!(vars.len(), 3); // P, C, A

        for subst in &results {
            let bindings = extract_bindings(subst, &vars);
            assert_eq!(bindings.len(), 3, "Should have 3 variable bindings");
        }
    }

    #[test]
    fn test_query_empty_result() {
        let program_text = r#"
            parent(john, mary).
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");

        let mut facts = crate::database::FactDatabase::new();
        for statement in &program.statements {
            if let crate::ast::Statement::Fact(fact) = statement {
                facts.insert(fact.atom.clone()).unwrap();
            }
        }

        // Query for non-existent relationship
        let query = parse_query("?- parent(alice, bob).").expect("Failed to parse query");
        let results = evaluate_query(&query, &facts);

        assert_eq!(results.len(), 0, "Should return empty result");
    }

    #[test]
    fn test_query_with_constants_and_ranges() {
        let program_text = r#"
            #const max = 3.

            item(1).
            item(2).
            item(3).

            valid(X) :- item(X).
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");

        let mut facts = crate::database::FactDatabase::new();
        let mut rules = vec![];

        for statement in &program.statements {
            match statement {
                crate::ast::Statement::Fact(fact) => {
                    facts.insert(fact.atom.clone()).unwrap();
                }
                crate::ast::Statement::Rule(rule) => {
                    rules.push(rule.clone());
                }
                _ => {}
            }
        }

        let result_db =
            semi_naive_evaluation(&rules, facts).expect("semi-naive evaluation should succeed");

        // Query for all valid items
        let query = parse_query("?- valid(X).").expect("Failed to parse query");
        let results = evaluate_query(&query, &result_db);

        assert_eq!(results.len(), 3, "Should find 3 valid items");
    }

    #[test]
    fn test_complex_query_with_multiple_joins() {
        let program_text = r#"
            parent(john, mary).
            parent(mary, sue).
            parent(sue, anna).

            gender(john, male).
            gender(mary, female).
            gender(sue, female).
            gender(anna, female).

            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

            female_ancestor(X, Y) :- ancestor(X, Y), gender(X, female).
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");

        let mut facts = crate::database::FactDatabase::new();
        let mut rules = vec![];

        for statement in &program.statements {
            match statement {
                crate::ast::Statement::Fact(fact) => {
                    facts.insert(fact.atom.clone()).unwrap();
                }
                crate::ast::Statement::Rule(rule) => {
                    rules.push(rule.clone());
                }
                _ => {}
            }
        }

        let result_db =
            semi_naive_evaluation(&rules, facts).expect("semi-naive evaluation should succeed");

        // Query for female ancestors of anna
        let query = parse_query("?- female_ancestor(X, anna).").expect("Failed to parse query");
        let results = evaluate_query(&query, &result_db);

        assert_eq!(
            results.len(),
            2,
            "Should find 2 female ancestors of Anna: Mary and Sue"
        );
    }

    #[test]
    fn test_query_with_integer_constant() {
        let program_text = r#"
            #const max_items = 5.

            item(1).
            item(2).
            item(3).
            item(4).
            item(5).

            limit(max_items).
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");

        let const_env = crate::constants::ConstantEnv::from_program(&program);
        let mut facts = crate::database::FactDatabase::new();

        for statement in &program.statements {
            if let crate::ast::Statement::Fact(fact) = statement {
                let substituted = const_env.substitute_atom(&fact.atom);
                facts.insert(substituted).unwrap();
            }
        }

        // Query: ?- limit(5). (should match because max_items=5)
        let query = parse_query("?- limit(5).").expect("Failed to parse query");
        let results = evaluate_query(&query, &facts);
        assert_eq!(results.len(), 1, "Should find limit(5)");

        // Query: ?- limit(X).
        let query2 = parse_query("?- limit(X).").expect("Failed to parse query");
        let results2 = evaluate_query(&query2, &facts);
        assert_eq!(results2.len(), 1, "Should find one limit value");
    }

    #[test]
    fn test_query_with_float_constant() {
        let program_text = r#"
            #const gravity = 9.81.
            #const pi = 3.14.

            physics_constant(gravity).
            math_constant(pi).
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");

        let const_env = crate::constants::ConstantEnv::from_program(&program);
        let mut facts = crate::database::FactDatabase::new();

        for statement in &program.statements {
            if let crate::ast::Statement::Fact(fact) = statement {
                let substituted = const_env.substitute_atom(&fact.atom);
                facts.insert(substituted).unwrap();
            }
        }

        // Query: ?- physics_constant(9.81).
        let query = parse_query("?- physics_constant(9.81).").expect("Failed to parse query");
        let results = evaluate_query(&query, &facts);
        assert_eq!(results.len(), 1, "Should find physics_constant(9.81)");

        // Query: ?- math_constant(X).
        let query2 = parse_query("?- math_constant(X).").expect("Failed to parse query");
        let results2 = evaluate_query(&query2, &facts);
        assert_eq!(results2.len(), 1, "Should find one math constant");
    }

    #[test]
    fn test_query_with_boolean_constant() {
        let program_text = r#"
            #const debug_enabled = true.
            #const production_mode = false.

            setting(debug, debug_enabled).
            setting(production, production_mode).
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");

        let const_env = crate::constants::ConstantEnv::from_program(&program);
        let mut facts = crate::database::FactDatabase::new();

        for statement in &program.statements {
            if let crate::ast::Statement::Fact(fact) = statement {
                let substituted = const_env.substitute_atom(&fact.atom);
                facts.insert(substituted).unwrap();
            }
        }

        // Query: ?- setting(debug, true).
        let query = parse_query("?- setting(debug, true).").expect("Failed to parse query");
        let results = evaluate_query(&query, &facts);
        assert_eq!(results.len(), 1, "Should find setting(debug, true)");

        // Query: ?- setting(production, false).
        let query2 = parse_query("?- setting(production, false).").expect("Failed to parse query");
        let results2 = evaluate_query(&query2, &facts);
        assert_eq!(results2.len(), 1, "Should find setting(production, false)");

        // Query: ?- setting(X, true).
        let query3 = parse_query("?- setting(X, true).").expect("Failed to parse query");
        let results3 = evaluate_query(&query3, &facts);
        assert_eq!(results3.len(), 1, "Should find one setting with true");
    }

    #[test]
    fn test_query_with_string_constant() {
        let program_text = r#"
            #const player_name = "Alice".
            #const game_title = "Adventure Quest".

            player(player_name).
            game(game_title).
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");

        let const_env = crate::constants::ConstantEnv::from_program(&program);
        let mut facts = crate::database::FactDatabase::new();

        for statement in &program.statements {
            if let crate::ast::Statement::Fact(fact) = statement {
                let substituted = const_env.substitute_atom(&fact.atom);
                facts.insert(substituted).unwrap();
            }
        }

        // Query: ?- player("Alice").
        let query = parse_query(r#"?- player("Alice")."#).expect("Failed to parse query");
        let results = evaluate_query(&query, &facts);
        assert_eq!(results.len(), 1, "Should find player(\"Alice\")");

        // Query: ?- game(X).
        let query2 = parse_query("?- game(X).").expect("Failed to parse query");
        let results2 = evaluate_query(&query2, &facts);
        assert_eq!(results2.len(), 1, "Should find one game");
    }

    #[test]
    fn test_query_with_atom_constant() {
        let program_text = r#"
            #const default_color = red.
            #const default_weapon = sword.

            color(item1, default_color).
            weapon(player1, default_weapon).
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");

        let const_env = crate::constants::ConstantEnv::from_program(&program);
        let mut facts = crate::database::FactDatabase::new();

        for statement in &program.statements {
            if let crate::ast::Statement::Fact(fact) = statement {
                let substituted = const_env.substitute_atom(&fact.atom);
                facts.insert(substituted).unwrap();
            }
        }

        // Query: ?- color(item1, red).
        let query = parse_query("?- color(item1, red).").expect("Failed to parse query");
        let results = evaluate_query(&query, &facts);
        assert_eq!(results.len(), 1, "Should find color(item1, red)");

        // Query: ?- weapon(X, sword).
        let query2 = parse_query("?- weapon(X, sword).").expect("Failed to parse query");
        let results2 = evaluate_query(&query2, &facts);
        assert_eq!(results2.len(), 1, "Should find one weapon");
    }

    #[test]
    fn test_query_with_mixed_constants() {
        let program_text = r#"
            #const max_health = 100.
            #const default_name = "Player".
            #const is_active = true.
            #const starting_level = 1.

            character(default_name, max_health, is_active, starting_level).
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");

        let const_env = crate::constants::ConstantEnv::from_program(&program);
        let mut facts = crate::database::FactDatabase::new();

        for statement in &program.statements {
            if let crate::ast::Statement::Fact(fact) = statement {
                let substituted = const_env.substitute_atom(&fact.atom);
                facts.insert(substituted).unwrap();
            }
        }

        // Query: ?- character("Player", 100, true, 1).
        let query =
            parse_query(r#"?- character("Player", 100, true, 1)."#).expect("Failed to parse query");
        let results = evaluate_query(&query, &facts);
        assert_eq!(
            results.len(),
            1,
            "Should find character with all substituted constants"
        );

        // Query: ?- character(Name, Health, Active, Level).
        let query2 = parse_query("?- character(Name, Health, Active, Level).")
            .expect("Failed to parse query");
        let results2 = evaluate_query(&query2, &facts);
        assert_eq!(results2.len(), 1, "Should find one character");

        // Verify variable extraction
        let vars = query_variables(&query2);
        assert_eq!(vars.len(), 4, "Should have 4 variables");

        for subst in &results2 {
            let bindings = extract_bindings(subst, &vars);
            assert_eq!(bindings.len(), 4, "Should have 4 bindings");
        }
    }

    #[test]
    fn test_query_constants_with_simple_rule() {
        // Test with a simple rule that has no built-ins
        let program_text = r#"
            #const max_score = 100.

            score(alice, 85).
            score(bob, 100).

            perfect(X) :- score(X, max_score).
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");

        let const_env = crate::constants::ConstantEnv::from_program(&program);
        let mut facts = crate::database::FactDatabase::new();
        let mut rules = vec![];

        for statement in &program.statements {
            match statement {
                crate::ast::Statement::Fact(fact) => {
                    let substituted = const_env.substitute_atom(&fact.atom);
                    facts.insert(substituted).unwrap();
                }
                crate::ast::Statement::Rule(rule) => {
                    let substituted_head = const_env.substitute_atom(&rule.head);
                    let substituted_body: Vec<_> = rule
                        .body
                        .iter()
                        .map(|lit| match lit {
                            crate::ast::Literal::Positive(atom) => {
                                crate::ast::Literal::Positive(const_env.substitute_atom(atom))
                            }
                            crate::ast::Literal::Negative(atom) => {
                                crate::ast::Literal::Negative(const_env.substitute_atom(atom))
                            }
                            crate::ast::Literal::Aggregate(agg) => {
                                crate::ast::Literal::Aggregate(agg.clone())
                            }
                        })
                        .collect();

                    rules.push(crate::ast::Rule {
                        head: substituted_head,
                        body: substituted_body,
                    });
                }
                _ => {}
            }
        }

        let result_db =
            semi_naive_evaluation(&rules, facts).expect("semi-naive evaluation should succeed");

        // Query: ?- perfect(X).
        let query = parse_query("?- perfect(X).").expect("Failed to parse query");
        let results = evaluate_query(&query, &result_db);
        assert_eq!(
            results.len(),
            1,
            "Should find 1 person with perfect score (bob)"
        );
    }

    #[test]
    fn test_query_constants_with_rules() {
        let program_text = r#"
            #const max_score = 100.
            #const passing_grade = 60.

            score(alice, 85).
            score(bob, 55).
            score(charlie, 95).

            passed(X) :- score(X, S), S >= passing_grade.
            perfect(X) :- score(X, max_score).
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");

        let const_env = crate::constants::ConstantEnv::from_program(&program);
        let mut facts = crate::database::FactDatabase::new();
        let mut rules = vec![];

        for statement in &program.statements {
            match statement {
                crate::ast::Statement::Fact(fact) => {
                    let substituted = const_env.substitute_atom(&fact.atom);
                    facts.insert(substituted).unwrap();
                }
                crate::ast::Statement::Rule(rule) => {
                    // Substitute constants in rule
                    let substituted_head = const_env.substitute_atom(&rule.head);
                    let substituted_body: Vec<_> = rule
                        .body
                        .iter()
                        .map(|lit| match lit {
                            crate::ast::Literal::Positive(atom) => {
                                crate::ast::Literal::Positive(const_env.substitute_atom(atom))
                            }
                            crate::ast::Literal::Negative(atom) => {
                                crate::ast::Literal::Negative(const_env.substitute_atom(atom))
                            }
                            crate::ast::Literal::Aggregate(agg) => {
                                crate::ast::Literal::Aggregate(agg.clone())
                            }
                        })
                        .collect();

                    rules.push(crate::ast::Rule {
                        head: substituted_head,
                        body: substituted_body,
                    });
                }
                _ => {}
            }
        }

        let result_db =
            semi_naive_evaluation(&rules, facts).expect("semi-naive evaluation should succeed");

        // Query: ?- passed(X).
        let query = parse_query("?- passed(X).").expect("Failed to parse query");
        let results = evaluate_query(&query, &result_db);
        assert_eq!(
            results.len(),
            2,
            "Should find 2 students who passed (alice and charlie)"
        );

        // Query: ?- perfect(X).
        let query2 = parse_query("?- perfect(X).").expect("Failed to parse query");
        let results2 = evaluate_query(&query2, &result_db);
        assert_eq!(results2.len(), 0, "No one has a perfect score of 100");
    }
}
