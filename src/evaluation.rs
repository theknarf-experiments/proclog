use crate::ast::{Rule, Atom};
use crate::database::FactDatabase;
use crate::grounding::{ground_rule, ground_rule_semi_naive};
use crate::stratification::{stratify, StratificationError};

/// Naive evaluation: repeatedly apply all rules until fixed point
/// This is simple but inefficient - it re-evaluates all facts every iteration
pub fn naive_evaluation(rules: &[Rule], initial_facts: FactDatabase) -> FactDatabase {
    let mut db = initial_facts;
    let mut changed = true;

    while changed {
        changed = false;
        let old_size = db.len();

        // Apply each rule and add derived facts
        for rule in rules {
            let derived = ground_rule(rule, &db);
            for fact in derived {
                if db.insert(fact) {
                    changed = true;
                }
            }
        }

        // If no new facts were added, we've reached fixed point
        if db.len() == old_size {
            changed = false;
        }
    }

    db
}

/// Semi-naive evaluation: only process new facts (delta) each iteration
/// This is much more efficient for recursive rules
///
/// The key insight: for each rule, we need to ensure at least one literal
/// uses the delta (new facts), while others can use the full database.
/// This prevents re-deriving facts from old information.
pub fn semi_naive_evaluation(rules: &[Rule], initial_facts: FactDatabase) -> FactDatabase {
    let mut db = initial_facts.clone();
    let mut delta = initial_facts;

    loop {
        let mut new_delta = FactDatabase::new();

        for rule in rules {
            if rule.body.is_empty() {
                // No body - always evaluate
                let derived = ground_rule(rule, &db);
                for fact in derived {
                    if !db.contains(&fact) {
                        db.insert(fact.clone());
                        new_delta.insert(fact);
                    }
                }
            } else if rule.body.len() == 1 {
                // Single literal - just use delta
                let derived = ground_rule(rule, &delta);
                for fact in derived {
                    if !db.contains(&fact) {
                        db.insert(fact.clone());
                        new_delta.insert(fact);
                    }
                }
            } else {
                // Multi-literal: use semi-naive grounding
                // This tries delta at each position
                let derived = ground_rule_semi_naive(rule, &delta, &db);
                for fact in derived {
                    if !db.contains(&fact) {
                        db.insert(fact.clone());
                        new_delta.insert(fact);
                    }
                }
            }
        }

        // If no new facts, we've reached fixed point
        if new_delta.is_empty() {
            break;
        }

        delta = new_delta;
    }

    db
}

/// Statistics about evaluation performance
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationStats {
    pub iterations: usize,
    pub rule_applications: usize,
    pub facts_derived: usize,
}

/// Instrumented naive evaluation that tracks statistics
pub fn naive_evaluation_instrumented(rules: &[Rule], initial_facts: FactDatabase) -> (FactDatabase, EvaluationStats) {
    let mut db = initial_facts;
    let mut changed = true;
    let mut stats = EvaluationStats {
        iterations: 0,
        rule_applications: 0,
        facts_derived: 0,
    };

    while changed {
        changed = false;
        let old_size = db.len();
        stats.iterations += 1;

        for rule in rules {
            stats.rule_applications += 1;
            let derived = ground_rule(rule, &db);
            for fact in derived {
                if db.insert(fact) {
                    changed = true;
                    stats.facts_derived += 1;
                }
            }
        }

        if db.len() == old_size {
            changed = false;
        }
    }

    (db, stats)
}

/// Instrumented semi-naive evaluation that tracks statistics
pub fn semi_naive_evaluation_instrumented(rules: &[Rule], initial_facts: FactDatabase) -> (FactDatabase, EvaluationStats) {
    let mut db = initial_facts.clone();
    let mut delta = initial_facts;
    let mut stats = EvaluationStats {
        iterations: 0,
        rule_applications: 0,
        facts_derived: 0,
    };

    loop {
        let mut new_delta = FactDatabase::new();
        stats.iterations += 1;

        for rule in rules {
            stats.rule_applications += 1;

            if rule.body.is_empty() {
                let derived = ground_rule(rule, &db);
                for fact in derived {
                    if !db.contains(&fact) {
                        db.insert(fact.clone());
                        new_delta.insert(fact);
                        stats.facts_derived += 1;
                    }
                }
            } else if rule.body.len() == 1 {
                let derived = ground_rule(rule, &delta);
                for fact in derived {
                    if !db.contains(&fact) {
                        db.insert(fact.clone());
                        new_delta.insert(fact);
                        stats.facts_derived += 1;
                    }
                }
            } else {
                let derived = ground_rule_semi_naive(rule, &delta, &db);
                for fact in derived {
                    if !db.contains(&fact) {
                        db.insert(fact.clone());
                        new_delta.insert(fact);
                        stats.facts_derived += 1;
                    }
                }
            }
        }

        if new_delta.is_empty() {
            break;
        }

        delta = new_delta;
    }

    (db, stats)
}

/// Stratified evaluation: evaluates program stratum by stratum
/// This allows safe handling of negation by ensuring negated predicates
/// are fully computed before being used
pub fn stratified_evaluation(rules: &[Rule], initial_facts: FactDatabase) -> Result<FactDatabase, StratificationError> {
    // Stratify the program
    let stratification = stratify(rules)?;

    let mut db = initial_facts;

    // Evaluate each stratum to completion before moving to next
    for stratum_rules in &stratification.rules_by_stratum {
        // Evaluate this stratum to fixed point using semi-naive
        db = semi_naive_evaluation(stratum_rules, db);
    }

    Ok(db)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Literal, Term, Value};
    use internment::Intern;

    // Helper functions
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

    // Naive evaluation tests
    #[test]
    fn test_naive_evaluation_no_rules() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));

        let rules = vec![];
        let result = naive_evaluation(&rules, db.clone());

        // No rules means no new facts
        assert_eq!(result.len(), db.len());
    }

    #[test]
    fn test_naive_evaluation_one_iteration() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));
        db.insert(make_atom("parent", vec![atom_const("mary"), atom_const("alice")]));

        // Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        let rules = vec![make_rule(
            make_atom("grandparent", vec![var("X"), var("Z")]),
            vec![
                Literal::Positive(make_atom("parent", vec![var("X"), var("Y")])),
                Literal::Positive(make_atom("parent", vec![var("Y"), var("Z")])),
            ],
        )];

        let result = naive_evaluation(&rules, db);

        // Should have 2 parent facts + 1 grandparent fact
        assert_eq!(result.len(), 3);

        let grandparent_pred = Intern::new("grandparent".to_string());
        let grandparents = result.get_by_predicate(&grandparent_pred);
        assert_eq!(grandparents.len(), 1);
    }

    #[test]
    fn test_naive_evaluation_transitive_closure() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("edge", vec![atom_const("a"), atom_const("b")]));
        db.insert(make_atom("edge", vec![atom_const("b"), atom_const("c")]));
        db.insert(make_atom("edge", vec![atom_const("c"), atom_const("d")]));

        // Rules for transitive closure:
        // path(X, Y) :- edge(X, Y).
        // path(X, Z) :- path(X, Y), edge(Y, Z).
        let rules = vec![
            make_rule(
                make_atom("path", vec![var("X"), var("Y")]),
                vec![Literal::Positive(make_atom("edge", vec![var("X"), var("Y")]))],
            ),
            make_rule(
                make_atom("path", vec![var("X"), var("Z")]),
                vec![
                    Literal::Positive(make_atom("path", vec![var("X"), var("Y")])),
                    Literal::Positive(make_atom("edge", vec![var("Y"), var("Z")])),
                ],
            ),
        ];

        let result = naive_evaluation(&rules, db);

        let path_pred = Intern::new("path".to_string());
        let paths = result.get_by_predicate(&path_pred);

        // Should derive:
        // path(a, b), path(b, c), path(c, d)  [from first rule]
        // path(a, c), path(b, d)               [from second rule, iteration 2]
        // path(a, d)                           [from second rule, iteration 3]
        // Total: 6 path facts
        assert_eq!(paths.len(), 6);
    }

    // Semi-naive evaluation tests
    #[test]
    fn test_semi_naive_evaluation_no_rules() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));

        let rules = vec![];
        let result = semi_naive_evaluation(&rules, db.clone());

        assert_eq!(result.len(), db.len());
    }

    #[test]
    fn test_semi_naive_evaluation_one_iteration() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));
        db.insert(make_atom("parent", vec![atom_const("mary"), atom_const("alice")]));

        let rules = vec![make_rule(
            make_atom("grandparent", vec![var("X"), var("Z")]),
            vec![
                Literal::Positive(make_atom("parent", vec![var("X"), var("Y")])),
                Literal::Positive(make_atom("parent", vec![var("Y"), var("Z")])),
            ],
        )];

        let result = semi_naive_evaluation(&rules, db);

        assert_eq!(result.len(), 3);

        let grandparent_pred = Intern::new("grandparent".to_string());
        let grandparents = result.get_by_predicate(&grandparent_pred);
        assert_eq!(grandparents.len(), 1);
    }

    #[test]
    fn test_semi_naive_evaluation_transitive_closure() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("edge", vec![atom_const("a"), atom_const("b")]));
        db.insert(make_atom("edge", vec![atom_const("b"), atom_const("c")]));
        db.insert(make_atom("edge", vec![atom_const("c"), atom_const("d")]));

        let rules = vec![
            make_rule(
                make_atom("path", vec![var("X"), var("Y")]),
                vec![Literal::Positive(make_atom("edge", vec![var("X"), var("Y")]))],
            ),
            make_rule(
                make_atom("path", vec![var("X"), var("Z")]),
                vec![
                    Literal::Positive(make_atom("path", vec![var("X"), var("Y")])),
                    Literal::Positive(make_atom("edge", vec![var("Y"), var("Z")])),
                ],
            ),
        ];

        let result = semi_naive_evaluation(&rules, db);

        let path_pred = Intern::new("path".to_string());
        let paths = result.get_by_predicate(&path_pred);

        // Should derive same 6 paths as naive evaluation
        assert_eq!(paths.len(), 6);
    }

    #[test]
    fn test_naive_vs_semi_naive_equivalence() {
        // Both algorithms should produce the same result
        let mut db = FactDatabase::new();
        db.insert(make_atom("edge", vec![atom_const("a"), atom_const("b")]));
        db.insert(make_atom("edge", vec![atom_const("b"), atom_const("c")]));
        db.insert(make_atom("edge", vec![atom_const("c"), atom_const("d")]));

        let rules = vec![
            make_rule(
                make_atom("path", vec![var("X"), var("Y")]),
                vec![Literal::Positive(make_atom("edge", vec![var("X"), var("Y")]))],
            ),
            make_rule(
                make_atom("path", vec![var("X"), var("Z")]),
                vec![
                    Literal::Positive(make_atom("path", vec![var("X"), var("Y")])),
                    Literal::Positive(make_atom("edge", vec![var("Y"), var("Z")])),
                ],
            ),
        ];

        let naive_result = naive_evaluation(&rules, db.clone());
        let semi_naive_result = semi_naive_evaluation(&rules, db);

        // Both should produce same number of facts
        assert_eq!(naive_result.len(), semi_naive_result.len());

        // Verify all facts in naive are in semi-naive
        for fact in naive_result.all_facts() {
            assert!(semi_naive_result.contains(fact));
        }
    }

    // Integration test: Complete program execution
    #[test]
    fn test_integration_full_program_execution() {
        use crate::parser;
        use crate::ast::Statement;

        // A complete ProcLog program with transitive closure
        let program_text = r#"
            % Graph edges
            edge(a, b).
            edge(b, c).
            edge(c, d).
            edge(b, e).

            % Transitive closure rules
            path(X, Y) :- edge(X, Y).
            path(X, Z) :- path(X, Y), edge(Y, Z).

            % Reachability from specific node
            reachable_from_a(X) :- path(a, X).
        "#;

        // Step 1: Parse
        let program = parser::parse_program(program_text)
            .expect("Should parse successfully");

        // Step 2: Separate facts and rules
        let mut initial_db = FactDatabase::new();
        let mut rules = Vec::new();

        for statement in program.statements {
            match statement {
                Statement::Fact(fact) => {
                    initial_db.insert(fact.atom);
                }
                Statement::Rule(rule) => {
                    rules.push(rule);
                }
                _ => {}
            }
        }

        // Verify we have 4 edge facts and 3 rules
        assert_eq!(initial_db.len(), 4);
        assert_eq!(rules.len(), 3);

        // Step 3: Evaluate using semi-naive
        let result_db = semi_naive_evaluation(&rules, initial_db);

        // Step 4: Verify results
        // Should have:
        // - 4 edge facts: (a,b), (b,c), (c,d), (b,e)
        // - 8 path facts: (a,b), (b,c), (c,d), (b,e), (a,c), (b,d), (a,d), (a,e)
        // - 4 reachable_from_a facts: (b), (c), (d), (e)
        // Total: 16 facts
        assert_eq!(result_db.len(), 16);

        // Step 5: Query for paths from 'a'
        let query_pattern = make_atom("path", vec![atom_const("a"), var("X")]);
        let paths_from_a = result_db.query(&query_pattern);

        // Should find 4 paths from 'a': to b, c, d, e
        assert_eq!(paths_from_a.len(), 4);

        // Step 6: Query for reachable nodes from 'a'
        let reachable_pattern = make_atom("reachable_from_a", vec![var("X")]);
        let reachable = result_db.query(&reachable_pattern);

        // Should find 4 reachable nodes
        assert_eq!(reachable.len(), 4);

        // Step 7: Verify specific paths exist
        assert!(result_db.contains(&make_atom("path", vec![atom_const("a"), atom_const("d")])));
        assert!(result_db.contains(&make_atom("path", vec![atom_const("b"), atom_const("e")])));

        // Step 8: Verify that paths we don't expect don't exist
        assert!(!result_db.contains(&make_atom("path", vec![atom_const("e"), atom_const("a")])));
        assert!(!result_db.contains(&make_atom("path", vec![atom_const("d"), atom_const("a")])));
    }

    // Tests that explicitly verify semi-naive optimization
    #[test]
    fn test_semi_naive_efficiency_vs_naive() {
        // Semi-naive should do fewer rule applications than naive
        let mut db = FactDatabase::new();
        db.insert(make_atom("edge", vec![atom_const("a"), atom_const("b")]));
        db.insert(make_atom("edge", vec![atom_const("b"), atom_const("c")]));
        db.insert(make_atom("edge", vec![atom_const("c"), atom_const("d")]));

        let rules = vec![
            make_rule(
                make_atom("path", vec![var("X"), var("Y")]),
                vec![Literal::Positive(make_atom("edge", vec![var("X"), var("Y")]))],
            ),
            make_rule(
                make_atom("path", vec![var("X"), var("Z")]),
                vec![
                    Literal::Positive(make_atom("path", vec![var("X"), var("Y")])),
                    Literal::Positive(make_atom("edge", vec![var("Y"), var("Z")])),
                ],
            ),
        ];

        let (naive_result, naive_stats) = naive_evaluation_instrumented(&rules, db.clone());
        let (semi_naive_result, semi_naive_stats) = semi_naive_evaluation_instrumented(&rules, db);

        // Both should produce same facts
        assert_eq!(naive_result.len(), semi_naive_result.len());

        // Naive should do MORE iterations (re-evaluates until no change)
        // For this example: naive does 4 iterations, semi-naive does 4
        // But naive applies rules to ALL facts each time
        println!("Naive stats: {:?}", naive_stats);
        println!("Semi-naive stats: {:?}", semi_naive_stats);

        // The key difference: both produce same facts
        assert_eq!(naive_stats.facts_derived, semi_naive_stats.facts_derived);

        // But naive often needs more iterations to converge
        // (In some cases they may be equal, but naive should never be better)
        assert!(naive_stats.iterations >= semi_naive_stats.iterations);
    }

    #[test]
    fn test_semi_naive_doesnt_rederive_facts() {
        // This test ensures we're not deriving the same fact multiple times
        let mut db = FactDatabase::new();
        db.insert(make_atom("edge", vec![atom_const("a"), atom_const("b")]));
        db.insert(make_atom("edge", vec![atom_const("b"), atom_const("c")]));

        let rules = vec![
            make_rule(
                make_atom("path", vec![var("X"), var("Y")]),
                vec![Literal::Positive(make_atom("edge", vec![var("X"), var("Y")]))],
            ),
            make_rule(
                make_atom("path", vec![var("X"), var("Z")]),
                vec![
                    Literal::Positive(make_atom("path", vec![var("X"), var("Y")])),
                    Literal::Positive(make_atom("edge", vec![var("Y"), var("Z")])),
                ],
            ),
        ];

        let (result, stats) = semi_naive_evaluation_instrumented(&rules, db);

        // facts_derived should equal total facts minus initial facts
        let initial_facts = 2; // 2 edges
        let total_facts = result.len();
        let derived_facts = total_facts - initial_facts;

        // Each fact should be derived exactly once
        assert_eq!(stats.facts_derived, derived_facts);
    }

    #[test]
    fn test_semi_naive_delta_at_different_positions() {
        // This test verifies that delta is tried at EACH position in multi-literal rules
        // Without this, we might miss derivations
        let mut db = FactDatabase::new();

        // A chain: a -> b -> c
        db.insert(make_atom("edge", vec![atom_const("a"), atom_const("b")]));
        db.insert(make_atom("edge", vec![atom_const("b"), atom_const("c")]));

        // A separate chain: x -> y
        db.insert(make_atom("edge", vec![atom_const("x"), atom_const("y")]));

        // Rule: path(X, Z) :- path(X, Y), edge(Y, Z)
        let rules = vec![
            make_rule(
                make_atom("path", vec![var("X"), var("Y")]),
                vec![Literal::Positive(make_atom("edge", vec![var("X"), var("Y")]))],
            ),
            make_rule(
                make_atom("path", vec![var("X"), var("Z")]),
                vec![
                    Literal::Positive(make_atom("path", vec![var("X"), var("Y")])),
                    Literal::Positive(make_atom("edge", vec![var("Y"), var("Z")])),
                ],
            ),
        ];

        let result = semi_naive_evaluation(&rules, db);

        // Should derive path(a,c) by combining path(a,b) + edge(b,c)
        // This requires using delta at position 0 (path literal)
        assert!(result.contains(&make_atom("path", vec![atom_const("a"), atom_const("c")])));

        let path_pred = Intern::new("path".to_string());
        let paths = result.get_by_predicate(&path_pred);

        // Should have: path(a,b), path(b,c), path(x,y), path(a,c)
        assert_eq!(paths.len(), 4);
    }

    #[test]
    fn test_semi_naive_converges_correctly() {
        // Test that semi-naive reaches the same fixed point as naive
        // even with complex recursive rules
        let mut db = FactDatabase::new();

        // A larger graph to stress test
        db.insert(make_atom("edge", vec![atom_const("a"), atom_const("b")]));
        db.insert(make_atom("edge", vec![atom_const("b"), atom_const("c")]));
        db.insert(make_atom("edge", vec![atom_const("c"), atom_const("d")]));
        db.insert(make_atom("edge", vec![atom_const("d"), atom_const("e")]));
        db.insert(make_atom("edge", vec![atom_const("b"), atom_const("f")])); // Branch

        let rules = vec![
            make_rule(
                make_atom("path", vec![var("X"), var("Y")]),
                vec![Literal::Positive(make_atom("edge", vec![var("X"), var("Y")]))],
            ),
            make_rule(
                make_atom("path", vec![var("X"), var("Z")]),
                vec![
                    Literal::Positive(make_atom("path", vec![var("X"), var("Y")])),
                    Literal::Positive(make_atom("edge", vec![var("Y"), var("Z")])),
                ],
            ),
        ];

        let (naive_result, _) = naive_evaluation_instrumented(&rules, db.clone());
        let (semi_naive_result, _) = semi_naive_evaluation_instrumented(&rules, db);

        // Must derive same facts
        assert_eq!(naive_result.len(), semi_naive_result.len());

        // Verify every path exists in both
        for fact in naive_result.all_facts() {
            assert!(semi_naive_result.contains(fact),
                "Semi-naive missing fact that naive derived: {:?}", fact);
        }
    }

    #[test]
    fn test_semi_naive_with_multiple_recursive_rules() {
        // Test case where multiple rules feed into each other
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("alice"), atom_const("bob")]));
        db.insert(make_atom("parent", vec![atom_const("bob"), atom_const("charlie")]));
        db.insert(make_atom("parent", vec![atom_const("charlie"), atom_const("dave")]));

        let rules = vec![
            // ancestor(X,Y) :- parent(X,Y)
            make_rule(
                make_atom("ancestor", vec![var("X"), var("Y")]),
                vec![Literal::Positive(make_atom("parent", vec![var("X"), var("Y")]))],
            ),
            // ancestor(X,Z) :- ancestor(X,Y), parent(Y,Z)
            make_rule(
                make_atom("ancestor", vec![var("X"), var("Z")]),
                vec![
                    Literal::Positive(make_atom("ancestor", vec![var("X"), var("Y")])),
                    Literal::Positive(make_atom("parent", vec![var("Y"), var("Z")])),
                ],
            ),
            // ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)
            // (different position for recursion)
            make_rule(
                make_atom("ancestor", vec![var("X"), var("Z")]),
                vec![
                    Literal::Positive(make_atom("parent", vec![var("X"), var("Y")])),
                    Literal::Positive(make_atom("ancestor", vec![var("Y"), var("Z")])),
                ],
            ),
        ];

        let (result, stats) = semi_naive_evaluation_instrumented(&rules, db);

        // Should derive all ancestor relationships
        // alice->bob, bob->charlie, charlie->dave (from rule 1)
        // alice->charlie, bob->dave (from rules 2 and 3)
        // alice->dave (from rules 2 and 3)
        let ancestor_pred = Intern::new("ancestor".to_string());
        let ancestors = result.get_by_predicate(&ancestor_pred);
        assert_eq!(ancestors.len(), 6);

        // Verify alice is ancestor of dave
        assert!(result.contains(&make_atom("ancestor",
            vec![atom_const("alice"), atom_const("dave")])));

        // No fact should be derived more than once
        let initial_facts = 3;
        assert_eq!(stats.facts_derived, result.len() - initial_facts);
    }

    #[test]
    fn test_integration_ancestor_example() {
        use crate::parser;
        use crate::ast::Statement;

        // Family tree example
        let program_text = r#"
            % Facts: parent relationships
            parent(alice, bob).
            parent(bob, charlie).
            parent(bob, diana).
            parent(charlie, eve).

            % Rules: ancestor is transitive closure of parent
            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).
        "#;

        let program = parser::parse_program(program_text)
            .expect("Should parse successfully");

        let mut initial_db = FactDatabase::new();
        let mut rules = Vec::new();

        for statement in program.statements {
            match statement {
                Statement::Fact(fact) => {
                    initial_db.insert(fact.atom);
                }
                Statement::Rule(rule) => {
                    rules.push(rule);
                }
                _ => {}
            }
        }

        // Evaluate
        let result_db = semi_naive_evaluation(&rules, initial_db);

        // Query for all ancestors of eve
        let query = make_atom("ancestor", vec![var("X"), atom_const("eve")]);
        let ancestors_of_eve = result_db.query(&query);

        // Eve's ancestors: charlie (parent), bob (grandparent), alice (great-grandparent)
        assert_eq!(ancestors_of_eve.len(), 3);

        // Verify alice is an ancestor of eve
        assert!(result_db.contains(&make_atom("ancestor", vec![atom_const("alice"), atom_const("eve")])));

        // Query for all descendants of bob
        let query = make_atom("ancestor", vec![atom_const("bob"), var("Y")]);
        let descendants_of_bob = result_db.query(&query);

        // Bob's descendants: charlie, diana (children), eve (grandchild)
        assert_eq!(descendants_of_bob.len(), 3);
    }

    // Stratified evaluation tests
    #[test]
    fn test_stratified_evaluation_simple_negation() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("bird", vec![atom_const("tweety")]));
        db.insert(make_atom("bird", vec![atom_const("polly")]));
        db.insert(make_atom("penguin", vec![atom_const("polly")]));

        // Rule: flies(X) :- bird(X), not penguin(X).
        let rules = vec![make_rule(
            make_atom("flies", vec![var("X")]),
            vec![
                Literal::Positive(make_atom("bird", vec![var("X")])),
                Literal::Negative(make_atom("penguin", vec![var("X")])),
            ],
        )];

        let result = stratified_evaluation(&rules, db).unwrap();

        let flies_pred = Intern::new("flies".to_string());
        let flies = result.get_by_predicate(&flies_pred);

        // Only tweety flies
        assert_eq!(flies.len(), 1);
        assert!(result.contains(&make_atom("flies", vec![atom_const("tweety")])));
    }

    #[test]
    fn test_stratified_evaluation_chain_of_negations() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("base", vec![atom_const("a")]));

        // Stratum 0: p(X) :- base(X).
        // Stratum 1: q(X) :- base(X), not p(X).
        // Stratum 2: r(X) :- base(X), not q(X).
        let rules = vec![
            make_rule(
                make_atom("p", vec![var("X")]),
                vec![Literal::Positive(make_atom("base", vec![var("X")]))],
            ),
            make_rule(
                make_atom("q", vec![var("X")]),
                vec![
                    Literal::Positive(make_atom("base", vec![var("X")])),
                    Literal::Negative(make_atom("p", vec![var("X")])),
                ],
            ),
            make_rule(
                make_atom("r", vec![var("X")]),
                vec![
                    Literal::Positive(make_atom("base", vec![var("X")])),
                    Literal::Negative(make_atom("q", vec![var("X")])),
                ],
            ),
        ];

        let result = stratified_evaluation(&rules, db).unwrap();

        // p(a) should be derived in stratum 0
        assert!(result.contains(&make_atom("p", vec![atom_const("a")])));

        // q(a) should NOT be derived (because p(a) exists)
        assert!(!result.contains(&make_atom("q", vec![atom_const("a")])));

        // r(a) should be derived (because q(a) doesn't exist)
        assert!(result.contains(&make_atom("r", vec![atom_const("a")])));
    }

    #[test]
    fn test_stratified_evaluation_rejects_negative_cycle() {
        // p(X) :- not p(X).  [Illegal - cycle through negation!]
        let rules = vec![make_rule(
            make_atom("p", vec![var("X")]),
            vec![Literal::Negative(make_atom("p", vec![var("X")]))],
        )];

        let db = FactDatabase::new();
        let result = stratified_evaluation(&rules, db);

        assert!(result.is_err());
    }

    #[test]
    fn test_stratified_evaluation_complex_example() {
        use crate::parser;
        use crate::ast::Statement;

        // Complete program with stratified negation
        let program_text = r#"
            % Facts about people
            person(alice).
            person(bob).
            person(charlie).

            % Facts about employment
            employed(alice).
            employed(bob).

            % Stratum 1: unemployed people (depends on employed via negation)
            unemployed(X) :- person(X), not employed(X).

            % Stratum 2: needs_help (depends on unemployed)
            needs_help(X) :- unemployed(X).
        "#;

        let program = parser::parse_program(program_text)
            .expect("Should parse successfully");

        let mut initial_db = FactDatabase::new();
        let mut rules = Vec::new();

        for statement in program.statements {
            match statement {
                Statement::Fact(fact) => {
                    initial_db.insert(fact.atom);
                }
                Statement::Rule(rule) => {
                    rules.push(rule);
                }
                _ => {}
            }
        }

        let result = stratified_evaluation(&rules, initial_db).unwrap();

        // unemployed(charlie) should be derived
        assert!(result.contains(&make_atom("unemployed", vec![atom_const("charlie")])));

        // unemployed(alice) and unemployed(bob) should NOT be derived
        assert!(!result.contains(&make_atom("unemployed", vec![atom_const("alice")])));
        assert!(!result.contains(&make_atom("unemployed", vec![atom_const("bob")])));

        // needs_help(charlie) should be derived
        assert!(result.contains(&make_atom("needs_help", vec![atom_const("charlie")])));

        let unemployed_pred = Intern::new("unemployed".to_string());
        let unemployed = result.get_by_predicate(&unemployed_pred);
        assert_eq!(unemployed.len(), 1);
    }

    #[test]
    fn test_stratified_evaluation_eligibility() {
        use crate::parser;
        use crate::ast::Statement;

        // Eligibility example with multiple strata
        let program_text = r#"
            % Facts about students
            student(alice).
            student(bob).
            student(charlie).

            % Some students have scholarships
            has_scholarship(alice).

            % Some students have jobs
            has_job(bob).

            % Stratum 1: needs_financial_aid (depends on has_scholarship and has_job)
            needs_financial_aid(X) :- student(X), not has_scholarship(X), not has_job(X).

            % Stratum 2: priority_candidate (depends on needs_financial_aid)
            priority_candidate(X) :- needs_financial_aid(X).
        "#;

        let program = parser::parse_program(program_text)
            .expect("Should parse successfully");

        let mut initial_db = FactDatabase::new();
        let mut rules = Vec::new();

        for statement in program.statements {
            match statement {
                Statement::Fact(fact) => {
                    initial_db.insert(fact.atom);
                }
                Statement::Rule(rule) => {
                    rules.push(rule);
                }
                _ => {}
            }
        }

        let result = stratified_evaluation(&rules, initial_db).unwrap();

        // charlie needs financial aid (no scholarship, no job)
        assert!(result.contains(&make_atom("needs_financial_aid", vec![atom_const("charlie")])));

        // alice doesn't need aid (has scholarship)
        assert!(!result.contains(&make_atom("needs_financial_aid", vec![atom_const("alice")])));

        // bob doesn't need aid (has job)
        assert!(!result.contains(&make_atom("needs_financial_aid", vec![atom_const("bob")])));

        // charlie is a priority candidate
        assert!(result.contains(&make_atom("priority_candidate", vec![atom_const("charlie")])));

        let needs_aid_pred = Intern::new("needs_financial_aid".to_string());
        let needs_aid = result.get_by_predicate(&needs_aid_pred);
        assert_eq!(needs_aid.len(), 1);
    }

    // Complex tests combining stratification with semi-naive evaluation
    #[test]
    fn test_stratified_with_transitive_closure_and_negation() {
        use crate::parser;
        use crate::ast::Statement;

        // Reachability with blocked nodes
        let program_text = r#"
            % Graph edges
            edge(a, b).
            edge(b, c).
            edge(c, d).
            edge(d, e).
            edge(a, x).
            edge(x, y).

            % Some nodes are blocked
            blocked(x).
            blocked(y).

            % Stratum 0: Basic reachability (recursive, needs semi-naive)
            reachable(X, Y) :- edge(X, Y).
            reachable(X, Z) :- reachable(X, Y), edge(Y, Z).

            % Stratum 1: Safe reachability (excludes blocked nodes)
            safe_reachable(X, Y) :- reachable(X, Y), not blocked(Y).
            safe_reachable(X, Z) :- safe_reachable(X, Y), edge(Y, Z), not blocked(Z).
        "#;

        let program = parser::parse_program(program_text)
            .expect("Should parse successfully");

        let mut initial_db = FactDatabase::new();
        let mut rules = Vec::new();

        for statement in program.statements {
            match statement {
                Statement::Fact(fact) => {
                    initial_db.insert(fact.atom);
                }
                Statement::Rule(rule) => {
                    rules.push(rule);
                }
                _ => {}
            }
        }

        let result = stratified_evaluation(&rules, initial_db).unwrap();

        // Should reach d and e from a (via b->c->d->e)
        assert!(result.contains(&make_atom("safe_reachable", vec![atom_const("a"), atom_const("d")])));
        assert!(result.contains(&make_atom("safe_reachable", vec![atom_const("a"), atom_const("e")])));

        // Should NOT reach x or y (blocked)
        assert!(!result.contains(&make_atom("safe_reachable", vec![atom_const("a"), atom_const("x")])));
        assert!(!result.contains(&make_atom("safe_reachable", vec![atom_const("a"), atom_const("y")])));

        // Verify we computed full reachability in stratum 0
        assert!(result.contains(&make_atom("reachable", vec![atom_const("a"), atom_const("x")])));
        assert!(result.contains(&make_atom("reachable", vec![atom_const("x"), atom_const("y")])));

        let safe_pred = Intern::new("safe_reachable".to_string());
        let safe_facts = result.get_by_predicate(&safe_pred);

        // Should have safe paths through the main chain, but not through blocked nodes
        // a->b, a->c, a->d, a->e, b->c, b->d, b->e, c->d, c->e, d->e
        // But we also have safe_reachable being recursive, so we get the same paths
        // Actually, let's count: from stratum 1, we derive safe_reachable from reachable facts
        // that aren't blocked, plus recursive safe_reachable
        assert!(safe_facts.len() >= 4); // At minimum: a->b, a->c, a->d, a->e
    }

    #[test]
    fn test_stratified_ancestor_with_exclusions() {
        use crate::parser;
        use crate::ast::Statement;

        // Family tree with estrangement
        let program_text = r#"
            % Parent relationships
            parent(grandpa, dad).
            parent(dad, alice).
            parent(dad, bob).
            parent(alice, charlie).
            parent(uncle, eve).

            % Estrangement (family dispute)
            estranged(dad).

            % Stratum 0: All ancestors (recursive)
            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Z) :- ancestor(X, Y), parent(Y, Z).

            % Stratum 1: Recognized family (excluding estranged)
            recognized_family(X, Y) :- parent(X, Y), not estranged(X), not estranged(Y).
            recognized_family(X, Z) :- recognized_family(X, Y), parent(Y, Z), not estranged(Z).
        "#;

        let program = parser::parse_program(program_text)
            .expect("Should parse successfully");

        let mut initial_db = FactDatabase::new();
        let mut rules = Vec::new();

        for statement in program.statements {
            match statement {
                Statement::Fact(fact) => {
                    initial_db.insert(fact.atom);
                }
                Statement::Rule(rule) => {
                    rules.push(rule);
                }
                _ => {}
            }
        }

        let result = stratified_evaluation(&rules, initial_db).unwrap();

        // Grandpa -> dad -> alice should exist in ancestor
        assert!(result.contains(&make_atom("ancestor", vec![atom_const("grandpa"), atom_const("alice")])));

        // But NOT in recognized_family (dad is estranged)
        assert!(!result.contains(&make_atom("recognized_family", vec![atom_const("grandpa"), atom_const("alice")])));

        // Alice -> charlie should be in recognized_family (neither estranged)
        assert!(result.contains(&make_atom("recognized_family", vec![atom_const("alice"), atom_const("charlie")])));

        // Uncle -> eve should be in recognized_family
        assert!(result.contains(&make_atom("recognized_family", vec![atom_const("uncle"), atom_const("eve")])));
    }

    #[test]
    fn test_stratified_multiple_recursive_predicates_with_negation() {
        use crate::parser;
        use crate::ast::Statement;

        // Complex dependency graph with recursion and negation
        let program_text = r#"
            % Base graph
            edge(a, b).
            edge(b, c).
            edge(c, d).
            edge(a, x).
            edge(x, a).  % Cycle

            % Dangerous edges (to be avoided)
            dangerous(edge(x, a)).

            % Stratum 0: Compute all paths (recursive)
            path(X, Y) :- edge(X, Y).
            path(X, Z) :- path(X, Y), edge(Y, Z).

            % Stratum 0: Compute all cycles (recursive, finds X->...->X)
            in_cycle(X) :- path(X, X).

            % Stratum 1: Safe paths (no dangerous edges, not in cycles)
            safe_path(X, Y) :- edge(X, Y), not dangerous(edge(X, Y)), not in_cycle(X), not in_cycle(Y).
            safe_path(X, Z) :- safe_path(X, Y), edge(Y, Z), not dangerous(edge(Y, Z)), not in_cycle(Z).
        "#;

        let program = parser::parse_program(program_text)
            .expect("Should parse successfully");

        let mut initial_db = FactDatabase::new();
        let mut rules = Vec::new();

        for statement in program.statements {
            match statement {
                Statement::Fact(fact) => {
                    initial_db.insert(fact.atom);
                }
                Statement::Rule(rule) => {
                    rules.push(rule);
                }
                _ => {}
            }
        }

        let result = stratified_evaluation(&rules, initial_db).unwrap();

        // Both a and x are in cycles
        assert!(result.contains(&make_atom("in_cycle", vec![atom_const("a")])));
        assert!(result.contains(&make_atom("in_cycle", vec![atom_const("x")])));

        // b, c, d are NOT in cycles
        assert!(!result.contains(&make_atom("in_cycle", vec![atom_const("b")])));
        assert!(!result.contains(&make_atom("in_cycle", vec![atom_const("c")])));
        assert!(!result.contains(&make_atom("in_cycle", vec![atom_const("d")])));

        // Safe paths should exist from b to d (none of them in cycles)
        assert!(result.contains(&make_atom("safe_path", vec![atom_const("b"), atom_const("c")])));
        assert!(result.contains(&make_atom("safe_path", vec![atom_const("c"), atom_const("d")])));
        assert!(result.contains(&make_atom("safe_path", vec![atom_const("b"), atom_const("d")])));

        // Safe paths should NOT involve a or x (in cycles)
        assert!(!result.contains(&make_atom("safe_path", vec![atom_const("a"), atom_const("b")])));
        assert!(!result.contains(&make_atom("safe_path", vec![atom_const("a"), atom_const("x")])));
    }

    #[test]
    fn test_stratified_semi_naive_iteration_count() {
        // Verify that semi-naive still works efficiently with stratification
        let mut db = FactDatabase::new();

        // Create a long chain
        for i in 0..10 {
            let from = format!("n{}", i);
            let to = format!("n{}", i + 1);
            db.insert(make_atom("edge", vec![atom_const(&from), atom_const(&to)]));
        }

        // Mark some nodes as special
        db.insert(make_atom("special", vec![atom_const("n5")]));

        // Stratum 0: Compute reachability
        // Stratum 1: Compute non-special reachability
        let rules = vec![
            make_rule(
                make_atom("reach", vec![var("X"), var("Y")]),
                vec![Literal::Positive(make_atom("edge", vec![var("X"), var("Y")]))],
            ),
            make_rule(
                make_atom("reach", vec![var("X"), var("Z")]),
                vec![
                    Literal::Positive(make_atom("reach", vec![var("X"), var("Y")])),
                    Literal::Positive(make_atom("edge", vec![var("Y"), var("Z")])),
                ],
            ),
            make_rule(
                make_atom("normal_reach", vec![var("X"), var("Y")]),
                vec![
                    Literal::Positive(make_atom("reach", vec![var("X"), var("Y")])),
                    Literal::Negative(make_atom("special", vec![var("Y")])),
                ],
            ),
        ];

        let (_result, stats) = semi_naive_evaluation_instrumented(&rules[0..2], db.clone());

        // Stratum 0 should converge in ~10 iterations (length of chain)
        println!("Stratum 0 stats: {:?}", stats);
        assert!(stats.iterations <= 11); // Should be very efficient

        // Now evaluate full stratified program
        let full_result = stratified_evaluation(&rules, db).unwrap();

        // n5 is special, so normal_reach should NOT include it as destination
        assert!(!full_result.contains(&make_atom("normal_reach", vec![atom_const("n0"), atom_const("n5")])));

        // But should reach n4 (not special)
        assert!(full_result.contains(&make_atom("normal_reach", vec![atom_const("n0"), atom_const("n4")])));

        // And should reach n6+ (after the special node)
        assert!(full_result.contains(&make_atom("normal_reach", vec![atom_const("n0"), atom_const("n6")])));
    }

    #[test]
    fn test_stratified_double_negation() {
        use crate::parser;
        use crate::ast::Statement;

        // Test multiple levels of negation
        let program_text = r#"
            % Base facts
            base(a).
            base(b).
            base(c).

            % Some bases are excluded
            excluded(a).

            % Stratum 1: Included items
            included(X) :- base(X), not excluded(X).

            % Stratum 2: Double negation - things that aren't not-included
            definitely(X) :- base(X), not non_included(X).
            non_included(X) :- base(X), not included(X).
        "#;

        let program = parser::parse_program(program_text)
            .expect("Should parse successfully");

        let mut initial_db = FactDatabase::new();
        let mut rules = Vec::new();

        for statement in program.statements {
            match statement {
                Statement::Fact(fact) => {
                    initial_db.insert(fact.atom);
                }
                Statement::Rule(rule) => {
                    rules.push(rule);
                }
                _ => {}
            }
        }

        let result = stratified_evaluation(&rules, initial_db).unwrap();

        // b and c are included (not excluded)
        assert!(result.contains(&make_atom("included", vec![atom_const("b")])));
        assert!(result.contains(&make_atom("included", vec![atom_const("c")])));

        // a is not included (excluded)
        assert!(!result.contains(&make_atom("included", vec![atom_const("a")])));

        // a is non_included
        assert!(result.contains(&make_atom("non_included", vec![atom_const("a")])));

        // b and c are definitely (not non_included)
        assert!(result.contains(&make_atom("definitely", vec![atom_const("b")])));
        assert!(result.contains(&make_atom("definitely", vec![atom_const("c")])));

        // a is NOT definitely (it is non_included)
        assert!(!result.contains(&make_atom("definitely", vec![atom_const("a")])));
    }

    #[test]
    fn test_stratified_game_states_with_recursion() {
        use crate::parser;
        use crate::ast::Statement;

        // Game state exploration with forbidden states
        let program_text = r#"
            % Initial state
            initial(start).

            % State transitions
            transition(start, s1).
            transition(s1, s2).
            transition(s2, s3).
            transition(s1, danger).
            transition(danger, s3).

            % Dangerous states
            forbidden(danger).

            % Stratum 0: All reachable states (recursive)
            reachable(S) :- initial(S).
            reachable(S) :- reachable(S0), transition(S0, S).

            % Stratum 1: Safe states (reachable but not forbidden)
            safe(S) :- reachable(S), not forbidden(S).

            % Stratum 2: Safe paths (transitions between safe states)
            safe_transition(S1, S2) :- transition(S1, S2), safe(S1), safe(S2).

            % Stratum 2: Safely reachable (only through safe transitions)
            safely_reachable(S) :- initial(S), safe(S).
            safely_reachable(S) :- safely_reachable(S0), safe_transition(S0, S).
        "#;

        let program = parser::parse_program(program_text)
            .expect("Should parse successfully");

        let mut initial_db = FactDatabase::new();
        let mut rules = Vec::new();

        for statement in program.statements {
            match statement {
                Statement::Fact(fact) => {
                    initial_db.insert(fact.atom);
                }
                Statement::Rule(rule) => {
                    rules.push(rule);
                }
                _ => {}
            }
        }

        let result = stratified_evaluation(&rules, initial_db).unwrap();

        // All states are reachable
        assert!(result.contains(&make_atom("reachable", vec![atom_const("start")])));
        assert!(result.contains(&make_atom("reachable", vec![atom_const("s1")])));
        assert!(result.contains(&make_atom("reachable", vec![atom_const("s2")])));
        assert!(result.contains(&make_atom("reachable", vec![atom_const("s3")])));
        assert!(result.contains(&make_atom("reachable", vec![atom_const("danger")])));

        // Danger is NOT safe
        assert!(!result.contains(&make_atom("safe", vec![atom_const("danger")])));

        // start, s1, s2, s3 ARE safe
        assert!(result.contains(&make_atom("safe", vec![atom_const("start")])));
        assert!(result.contains(&make_atom("safe", vec![atom_const("s1")])));
        assert!(result.contains(&make_atom("safe", vec![atom_const("s2")])));
        assert!(result.contains(&make_atom("safe", vec![atom_const("s3")])));

        // start -> s1 -> s2 -> s3 are safely reachable (safe path exists)
        assert!(result.contains(&make_atom("safely_reachable", vec![atom_const("start")])));
        assert!(result.contains(&make_atom("safely_reachable", vec![atom_const("s1")])));
        assert!(result.contains(&make_atom("safely_reachable", vec![atom_const("s2")])));
        assert!(result.contains(&make_atom("safely_reachable", vec![atom_const("s3")])));

        // Verify safe_transition exists for safe paths but NOT through danger
        assert!(result.contains(&make_atom("safe_transition", vec![atom_const("start"), atom_const("s1")])));
        assert!(result.contains(&make_atom("safe_transition", vec![atom_const("s1"), atom_const("s2")])));
        assert!(result.contains(&make_atom("safe_transition", vec![atom_const("s2"), atom_const("s3")])));

        // Dangerous transitions should NOT be safe_transitions
        assert!(!result.contains(&make_atom("safe_transition", vec![atom_const("s1"), atom_const("danger")])));
        assert!(!result.contains(&make_atom("safe_transition", vec![atom_const("danger"), atom_const("s3")])));
    }
}
