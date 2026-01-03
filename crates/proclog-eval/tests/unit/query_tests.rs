    use super::*;
    use internment::Intern;
    use proclog_ast::{Atom, Literal, Query, Term, Value};

    // Helper functions
    fn make_atom(pred: &str, terms: Vec<Term>) -> Atom {
        Atom {
            predicate: Intern::new(pred.to_string()),
            terms,
        }
    }

    fn atom_term(name: &str) -> Term {
        Term::Constant(Value::Atom(Intern::new(name.to_string())))
    }

    fn var(name: &str) -> Term {
        Term::Variable(Intern::new(name.to_string()))
    }

    #[test]
    fn test_query_ground_true() {
        // Setup database
        let mut db = FactDatabase::new();
        let fact = make_atom("parent", vec![atom_term("john"), atom_term("mary")]);
        db.insert(fact.clone()).unwrap();

        // Query: ?- parent(john, mary).
        let query = Query {
            body: vec![Literal::Positive(fact)],
        };

        let results = evaluate_query(&query, &db);
        assert_eq!(results.len(), 1, "Ground query should return one result");
    }

    #[test]
    fn test_query_ground_false() {
        // Setup database
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![atom_term("john"), atom_term("mary")],
        ))
        .unwrap();

        // Query: ?- parent(alice, bob). (not in database)
        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![atom_term("alice"), atom_term("bob")],
            ))],
        };

        let results = evaluate_query(&query, &db);
        assert_eq!(results.len(), 0, "False query should return empty result");
    }

    #[test]
    fn test_query_with_one_variable() {
        // Setup database
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![atom_term("john"), atom_term("mary")],
        ))
        .unwrap();
        db.insert(make_atom(
            "parent",
            vec![atom_term("alice"), atom_term("mary")],
        ))
        .unwrap();
        db.insert(make_atom(
            "parent",
            vec![atom_term("bob"), atom_term("sue")],
        ))
        .unwrap();

        // Query: ?- parent(X, mary).
        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var("X"), atom_term("mary")],
            ))],
        };

        let results = evaluate_query(&query, &db);
        assert_eq!(
            results.len(),
            2,
            "Should find 2 parents of mary: {:?}",
            results
        );

        // Verify the bindings
        let vars = query_variables(&query);
        for subst in &results {
            let bindings = extract_bindings(subst, &vars);
            assert_eq!(bindings.len(), 1);
            // X should be bound to either john or alice
            let x_binding = &bindings[0].1;
            assert!(
                x_binding.contains("john") || x_binding.contains("alice"),
                "X should be john or alice, got: {}",
                x_binding
            );
        }
    }

    #[test]
    fn test_query_with_multiple_variables() {
        // Setup database
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![atom_term("john"), atom_term("mary")],
        ))
        .unwrap();
        db.insert(make_atom(
            "parent",
            vec![atom_term("alice"), atom_term("bob")],
        ))
        .unwrap();

        // Query: ?- parent(X, Y).
        let query = Query {
            body: vec![Literal::Positive(make_atom(
                "parent",
                vec![var("X"), var("Y")],
            ))],
        };

        let results = evaluate_query(&query, &db);
        assert_eq!(results.len(), 2, "Should find 2 parent relationships");
    }

    #[test]
    fn test_query_with_join() {
        // Setup database
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![atom_term("john"), atom_term("mary")],
        ))
        .unwrap();
        db.insert(make_atom(
            "parent",
            vec![atom_term("mary"), atom_term("sue")],
        ))
        .unwrap();

        // Query: ?- parent(X, Y), parent(Y, Z).
        // Should find: X=john, Y=mary, Z=sue
        let query = Query {
            body: vec![
                Literal::Positive(make_atom("parent", vec![var("X"), var("Y")])),
                Literal::Positive(make_atom("parent", vec![var("Y"), var("Z")])),
            ],
        };

        let results = evaluate_query(&query, &db);
        assert_eq!(results.len(), 1, "Should find 1 grandparent relationship");
    }

    #[test]
    fn test_query_with_negation() {
        // Setup database
        let mut db = FactDatabase::new();
        db.insert(make_atom("person", vec![atom_term("john")]))
            .unwrap();
        db.insert(make_atom("person", vec![atom_term("mary")]))
            .unwrap();
        db.insert(make_atom("dead", vec![atom_term("john")]))
            .unwrap();

        // Query: ?- person(X), not dead(X).
        // Should find only mary
        let query = Query {
            body: vec![
                Literal::Positive(make_atom("person", vec![var("X")])),
                Literal::Negative(make_atom("dead", vec![var("X")])),
            ],
        };

        let results = evaluate_query(&query, &db);
        assert_eq!(results.len(), 1, "Should find 1 living person");
    }

    #[test]
    fn test_query_variables_extraction() {
        let query = Query {
            body: vec![
                Literal::Positive(make_atom("parent", vec![var("X"), var("Y")])),
                Literal::Positive(make_atom("age", vec![var("Y"), var("A")])),
            ],
        };

        let vars = query_variables(&query);
        assert_eq!(vars.len(), 3);
        assert!(vars.contains(&Intern::new("X".to_string())));
        assert!(vars.contains(&Intern::new("Y".to_string())));
        assert!(vars.contains(&Intern::new("A".to_string())));
    }
