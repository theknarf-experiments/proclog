use crate::ast::{Rule, Atom, Literal, Term, Value};
use crate::database::FactDatabase;
use crate::unification::Substitution;
use internment::Intern;

/// Ground a rule: generate all ground instances by substituting variables
/// For a rule like `ancestor(X, Z) :- parent(X, Y), parent(Y, Z)`
/// This finds all ways to satisfy the body and applies those substitutions to the head
pub fn ground_rule(rule: &Rule, db: &FactDatabase) -> Vec<Atom> {
    let mut results = Vec::new();

    // Get all substitutions that satisfy the entire body
    let substitutions = satisfy_body(&rule.body, db);

    // Apply each substitution to the head to get ground facts
    for subst in substitutions {
        let ground_head = subst.apply_atom(&rule.head);
        results.push(ground_head);
    }

    results
}

/// Find all substitutions that satisfy a conjunction of literals
fn satisfy_body(body: &[Literal], db: &FactDatabase) -> Vec<Substitution> {
    if body.is_empty() {
        // Empty body is trivially satisfied with empty substitution
        return vec![Substitution::new()];
    }

    // Process first literal
    let first_literal = &body[0];
    let rest = &body[1..];

    match first_literal {
        Literal::Positive(atom) => {
            // Query the database for matches
            let initial_substs = db.query(atom);

            if rest.is_empty() {
                // No more literals to process
                initial_substs
            } else {
                // For each substitution from the first literal,
                // apply it to the rest and continue
                let mut all_substs = Vec::new();

                for subst in initial_substs {
                    // Apply current substitution to remaining literals
                    let applied_rest: Vec<Literal> = rest
                        .iter()
                        .map(|lit| apply_subst_to_literal(&subst, lit))
                        .collect();

                    // Recursively satisfy the rest
                    let rest_substs = satisfy_body(&applied_rest, db);

                    // Combine substitutions
                    for rest_subst in rest_substs {
                        let combined = combine_substs(&subst, &rest_subst);
                        all_substs.push(combined);
                    }
                }

                all_substs
            }
        }
        Literal::Negative(atom) => {
            // Negation as failure: the atom must NOT unify with any fact in the database
            // For each substitution from the rest of the body,
            // check that the atom (with substitution applied) doesn't match any fact

            if rest.is_empty() {
                // No more literals - check if the negated atom is NOT in the database
                let matches = db.query(atom);
                if matches.is_empty() {
                    // Atom is not in the database - negation succeeds with empty substitution
                    vec![Substitution::new()]
                } else {
                    // Atom is in the database - negation fails
                    vec![]
                }
            } else {
                // Process the rest first, then filter by negation
                let rest_substs = satisfy_body(rest, db);
                let mut result = Vec::new();

                for subst in rest_substs {
                    // Apply substitution to the negated atom
                    let ground_atom = subst.apply_atom(atom);

                    // Check if this ground atom exists in the database
                    let matches = db.query(&ground_atom);

                    if matches.is_empty() {
                        // Atom doesn't exist - negation succeeds
                        result.push(subst);
                    }
                    // If atom exists, negation fails - don't include this substitution
                }

                result
            }
        }
    }
}

/// Apply substitution to a literal
fn apply_subst_to_literal(subst: &Substitution, literal: &Literal) -> Literal {
    match literal {
        Literal::Positive(atom) => Literal::Positive(subst.apply_atom(atom)),
        Literal::Negative(atom) => Literal::Negative(subst.apply_atom(atom)),
    }
}

/// Combine two substitutions
fn combine_substs(s1: &Substitution, s2: &Substitution) -> Substitution {
    let mut combined = s1.clone();

    // Add bindings from s2, applying s1 to them first
    for (var, term) in s2.iter() {
        let applied_term = s1.apply(term);
        combined.bind(var.clone(), applied_term);
    }

    combined
}

/// Ground a rule using semi-naive evaluation
/// For multi-literal rules, this tries using delta at each position
/// and the full database for other positions
pub fn ground_rule_semi_naive(rule: &Rule, delta: &FactDatabase, full_db: &FactDatabase) -> Vec<Atom> {
    let mut results = Vec::new();

    if rule.body.is_empty() {
        // No body - just return the head
        return vec![rule.head.clone()];
    }

    // For each position i in the body, use delta for position i and full_db for others
    for delta_pos in 0..rule.body.len() {
        let substs = satisfy_body_mixed(&rule.body, delta, full_db, delta_pos);
        for subst in substs {
            let ground_head = subst.apply_atom(&rule.head);
            results.push(ground_head);
        }
    }

    results
}

/// Satisfy body literals using delta for one position and full DB for others
fn satisfy_body_mixed(
    body: &[Literal],
    delta: &FactDatabase,
    full_db: &FactDatabase,
    delta_pos: usize,
) -> Vec<Substitution> {
    satisfy_body_mixed_recursive(body, delta, full_db, delta_pos, 0)
}

/// Recursive helper for satisfy_body_mixed
fn satisfy_body_mixed_recursive(
    body: &[Literal],
    delta: &FactDatabase,
    full_db: &FactDatabase,
    delta_pos: usize,
    current_pos: usize,
) -> Vec<Substitution> {
    if body.is_empty() {
        return vec![Substitution::new()];
    }

    let first_literal = &body[0];
    let rest = &body[1..];

    match first_literal {
        Literal::Positive(atom) => {
            // Use delta if current position is delta_pos, otherwise use full_db
            let db = if current_pos == delta_pos { delta } else { full_db };
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

                    let rest_substs = satisfy_body_mixed_recursive(
                        &applied_rest,
                        delta,
                        full_db,
                        delta_pos,
                        current_pos + 1,
                    );

                    for rest_subst in rest_substs {
                        let combined = combine_substs(&subst, &rest_subst);
                        all_substs.push(combined);
                    }
                }

                all_substs
            }
        }
        Literal::Negative(atom) => {
            // Negation uses full_db (not delta) - we need the complete view
            if rest.is_empty() {
                let matches = full_db.query(atom);
                if matches.is_empty() {
                    vec![Substitution::new()]
                } else {
                    vec![]
                }
            } else {
                let rest_substs = satisfy_body_mixed_recursive(
                    rest,
                    delta,
                    full_db,
                    delta_pos,
                    current_pos + 1,
                );
                let mut result = Vec::new();

                for subst in rest_substs {
                    let ground_atom = subst.apply_atom(atom);
                    let matches = full_db.query(&ground_atom);

                    if matches.is_empty() {
                        result.push(subst);
                    }
                }

                result
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // Basic grounding tests
    #[test]
    fn test_ground_rule_no_variables() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));

        // Rule: ancestor(john, mary) :- parent(john, mary).
        let rule = make_rule(
            make_atom("ancestor", vec![atom_const("john"), atom_const("mary")]),
            vec![Literal::Positive(make_atom("parent", vec![atom_const("john"), atom_const("mary")]))],
        );

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].predicate, Intern::new("ancestor".to_string()));
    }

    #[test]
    fn test_ground_rule_single_variable() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("bob")]));

        // Rule: child(X) :- parent(john, X).
        let rule = make_rule(
            make_atom("child", vec![var("X")]),
            vec![Literal::Positive(make_atom("parent", vec![atom_const("john"), var("X")]))],
        );

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 2);

        // Both child(mary) and child(bob) should be generated
        let predicates: Vec<_> = results.iter().map(|a| &a.predicate).collect();
        assert!(predicates.iter().all(|p| **p == Intern::new("child".to_string())));
    }

    #[test]
    fn test_ground_rule_multiple_variables() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));
        db.insert(make_atom("parent", vec![atom_const("bob"), atom_const("alice")]));

        // Rule: ancestor(X, Y) :- parent(X, Y).
        let rule = make_rule(
            make_atom("ancestor", vec![var("X"), var("Y")]),
            vec![Literal::Positive(make_atom("parent", vec![var("X"), var("Y")]))],
        );

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ground_rule_join_two_literals() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));
        db.insert(make_atom("parent", vec![atom_const("mary"), atom_const("alice")]));
        db.insert(make_atom("parent", vec![atom_const("bob"), atom_const("charlie")]));

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
        db.insert(make_atom("parent", vec![atom_const("a"), atom_const("b")]));
        db.insert(make_atom("parent", vec![atom_const("b"), atom_const("c")]));
        db.insert(make_atom("parent", vec![atom_const("b"), atom_const("d")]));

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
        db.insert(make_atom("parent", vec![atom_const("john"), atom_const("mary")]));

        // Rule: child(X) :- parent(alice, X).
        // No facts match parent(alice, X)
        let rule = make_rule(
            make_atom("child", vec![var("X")]),
            vec![Literal::Positive(make_atom("parent", vec![atom_const("alice"), var("X")]))],
        );

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_ground_rule_empty_body() {
        let db = FactDatabase::new();

        // Rule: fact(a) :- .
        // (A rule with no body is always true)
        let rule = make_rule(
            make_atom("fact", vec![atom_const("a")]),
            vec![],
        );

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], make_atom("fact", vec![atom_const("a")]));
    }

    #[test]
    fn test_ground_rule_three_literals() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("edge", vec![atom_const("a"), atom_const("b")]));
        db.insert(make_atom("edge", vec![atom_const("b"), atom_const("c")]));
        db.insert(make_atom("edge", vec![atom_const("c"), atom_const("d")]));

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
        db.insert(make_atom("bird", vec![atom_const("tweety")]));
        db.insert(make_atom("bird", vec![atom_const("polly")]));
        db.insert(make_atom("penguin", vec![atom_const("polly")]));

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
        db.insert(make_atom("bird", vec![atom_const("tweety")]));

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
        db.insert(make_atom("bird", vec![atom_const("tweety")]));

        // Rule: not_bird_polly :- not bird(polly).
        let rule = make_rule(
            make_atom("not_bird_polly", vec![]),
            vec![Literal::Negative(make_atom("bird", vec![atom_const("polly")]))],
        );

        let results = ground_rule(&rule, &db);

        // polly is not a bird, so this succeeds
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_ground_rule_multiple_negations() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("a", vec![atom_const("x")]));
        db.insert(make_atom("b", vec![atom_const("y")]));
        db.insert(make_atom("c", vec![atom_const("z")]));

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
    fn test_integration_parse_ground_query() {
        use crate::parser;
        use crate::ast::Statement;

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
        let program = parser::parse_program(program_text)
            .expect("Should parse successfully");

        // Step 2: Load facts into database
        let mut db = FactDatabase::new();
        let mut rules = Vec::new();

        for statement in program.statements {
            match statement {
                Statement::Fact(fact) => {
                    db.insert(fact.atom);
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
            db.insert(fact);
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
}
