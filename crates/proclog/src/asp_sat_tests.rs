//! Tests for ASP evaluation using SAT solver backend
//!
//! Following TDD: Start with simple tests, implement features to make them pass

use crate::ast::*;
use crate::asp_sat::*;
use internment::Intern;

#[cfg(test)]
mod basic_sat_tests {
    use super::*;

    // Helper to create an atom
    fn make_atom(pred: &str, terms: Vec<&str>) -> Atom {
        Atom {
            predicate: Intern::new(pred.to_string()),
            terms: terms
                .into_iter()
                .map(|t| Term::Constant(Value::Atom(Intern::new(t.to_string()))))
                .collect(),
        }
    }

    #[test]
    fn test_sat_simple_facts() {
        // Test: Simple facts should be in the answer set
        // Program:
        //   a.
        //   b.
        let mut program = Program::new();
        program.add_statement(Statement::Fact(Fact {
            atom: make_atom("a", vec![]),
        }));
        program.add_statement(Statement::Fact(Fact {
            atom: make_atom("b", vec![]),
        }));

        let answer_sets = asp_sat_evaluation(&program);

        assert_eq!(answer_sets.len(), 1, "Should have exactly one answer set");

        let answer_set = &answer_sets[0];
        assert!(answer_set.atoms.contains(&make_atom("a", vec![])));
        assert!(answer_set.atoms.contains(&make_atom("b", vec![])));
    }

    #[test]
    fn test_sat_simple_rule() {
        // Test: Simple rule derivation
        // Program:
        //   a.
        //   b :- a.
        let mut program = Program::new();
        program.add_statement(Statement::Fact(Fact {
            atom: make_atom("a", vec![]),
        }));
        program.add_statement(Statement::Rule(Rule {
            head: make_atom("b", vec![]),
            body: vec![Literal::Positive(make_atom("a", vec![]))],
        }));

        let answer_sets = asp_sat_evaluation(&program);

        assert_eq!(answer_sets.len(), 1, "Should have exactly one answer set");

        let answer_set = &answer_sets[0];
        assert!(answer_set.atoms.contains(&make_atom("a", vec![])), "Should derive a");
        assert!(answer_set.atoms.contains(&make_atom("b", vec![])), "Should derive b from rule");
    }

    #[test]
    fn test_sat_constraint() {
        // Test: Constraint filtering
        // Program:
        //   a.
        //   b.
        //   :- a, b.
        let mut program = Program::new();
        program.add_statement(Statement::Fact(Fact {
            atom: make_atom("a", vec![]),
        }));
        program.add_statement(Statement::Fact(Fact {
            atom: make_atom("b", vec![]),
        }));
        program.add_statement(Statement::Constraint(Constraint {
            body: vec![
                Literal::Positive(make_atom("a", vec![])),
                Literal::Positive(make_atom("b", vec![])),
            ],
        }));

        let answer_sets = asp_sat_evaluation(&program);

        assert_eq!(answer_sets.len(), 0, "Should have no answer sets due to constraint");
    }

    #[test]
    fn test_sat_choice_rule() {
        // Test: Simple choice rule
        // Program:
        //   { a }.
        let mut program = Program::new();
        program.add_statement(Statement::ChoiceRule(ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: make_atom("a", vec![]),
                condition: vec![],
            }],
            body: vec![],
        }));

        let answer_sets = asp_sat_evaluation(&program);

        // Should have 2 answer sets: one with a, one without
        assert_eq!(answer_sets.len(), 2, "Should have 2 answer sets");

        let has_a = answer_sets.iter().any(|as_set| as_set.atoms.contains(&make_atom("a", vec![])));
        let has_empty = answer_sets.iter().any(|as_set| !as_set.atoms.contains(&make_atom("a", vec![])));

        assert!(has_a, "Should have answer set with a");
        assert!(has_empty, "Should have answer set without a");
    }

    #[test]
    fn test_sat_negation() {
        // Test: Negation as failure
        // Program:
        //   a.
        //   b :- not c.
        let mut program = Program::new();
        program.add_statement(Statement::Fact(Fact {
            atom: make_atom("a", vec![]),
        }));
        program.add_statement(Statement::Rule(Rule {
            head: make_atom("b", vec![]),
            body: vec![Literal::Negative(make_atom("c", vec![]))],
        }));

        let answer_sets = asp_sat_evaluation(&program);

        assert_eq!(answer_sets.len(), 1, "Should have exactly one answer set");

        let answer_set = &answer_sets[0];
        assert!(answer_set.atoms.contains(&make_atom("a", vec![])), "Should have a");
        assert!(answer_set.atoms.contains(&make_atom("b", vec![])), "Should derive b (c is false)");
        assert!(!answer_set.atoms.contains(&make_atom("c", vec![])), "Should not have c");
    }

    #[test]
    fn test_sat_multiple_choice_rules() {
        // Test: Multiple independent choice rules
        // Program:
        //   { a }.
        //   { b }.
        let mut program = Program::new();
        program.add_statement(Statement::ChoiceRule(ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: make_atom("a", vec![]),
                condition: vec![],
            }],
            body: vec![],
        }));
        program.add_statement(Statement::ChoiceRule(ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: make_atom("b", vec![]),
                condition: vec![],
            }],
            body: vec![],
        }));

        let answer_sets = asp_sat_evaluation(&program);

        // Should have 4 answer sets: {}, {a}, {b}, {a, b}
        assert_eq!(answer_sets.len(), 4, "Should have 4 answer sets");
    }

    #[test]
    fn test_sat_choice_with_derivation() {
        // Test: Choice rule with derived consequences
        // Program:
        //   { a }.
        //   b :- a.
        let mut program = Program::new();
        program.add_statement(Statement::ChoiceRule(ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: make_atom("a", vec![]),
                condition: vec![],
            }],
            body: vec![],
        }));
        program.add_statement(Statement::Rule(Rule {
            head: make_atom("b", vec![]),
            body: vec![Literal::Positive(make_atom("a", vec![]))],
        }));

        let answer_sets = asp_sat_evaluation(&program);

        // Should have 2 answer sets: {} and {a, b}
        assert_eq!(answer_sets.len(), 2, "Should have 2 answer sets");

        let has_both = answer_sets
            .iter()
            .any(|as_set| as_set.atoms.contains(&make_atom("a", vec![])) && as_set.atoms.contains(&make_atom("b", vec![])));
        let has_empty = answer_sets.iter().any(|as_set| as_set.atoms.is_empty());

        assert!(has_both, "Should have answer set with both a and b");
        assert!(has_empty, "Should have empty answer set");
    }

    #[test]
    fn test_sat_choice_with_constraint() {
        // Test: Choice rule with constraint
        // Program:
        //   { a }.
        //   { b }.
        //   :- a, b.
        let mut program = Program::new();
        program.add_statement(Statement::ChoiceRule(ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: make_atom("a", vec![]),
                condition: vec![],
            }],
            body: vec![],
        }));
        program.add_statement(Statement::ChoiceRule(ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: make_atom("b", vec![]),
                condition: vec![],
            }],
            body: vec![],
        }));
        program.add_statement(Statement::Constraint(Constraint {
            body: vec![
                Literal::Positive(make_atom("a", vec![])),
                Literal::Positive(make_atom("b", vec![])),
            ],
        }));

        let answer_sets = asp_sat_evaluation(&program);

        // Should have 3 answer sets: {}, {a}, {b} (not {a, b} due to constraint)
        assert_eq!(answer_sets.len(), 3, "Should have 3 answer sets");

        let has_both = answer_sets
            .iter()
            .any(|as_set| as_set.atoms.contains(&make_atom("a", vec![])) && as_set.atoms.contains(&make_atom("b", vec![])));

        assert!(!has_both, "Should not have answer set with both a and b");
    }

    #[test]
    fn test_sat_complex_negation() {
        // Test: Complex negation scenario
        // Program:
        //   { a }.
        //   b :- not a.
        //   c :- b.
        let mut program = Program::new();
        program.add_statement(Statement::ChoiceRule(ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: make_atom("a", vec![]),
                condition: vec![],
            }],
            body: vec![],
        }));
        program.add_statement(Statement::Rule(Rule {
            head: make_atom("b", vec![]),
            body: vec![Literal::Negative(make_atom("a", vec![]))],
        }));
        program.add_statement(Statement::Rule(Rule {
            head: make_atom("c", vec![]),
            body: vec![Literal::Positive(make_atom("b", vec![]))],
        }));

        let answer_sets = asp_sat_evaluation(&program);

        // Should have 2 answer sets: {a} and {b, c}
        assert_eq!(answer_sets.len(), 2, "Should have 2 answer sets");

        let has_a_only = answer_sets.iter().any(|as_set| {
            as_set.atoms.contains(&make_atom("a", vec![]))
                && !as_set.atoms.contains(&make_atom("b", vec![]))
                && !as_set.atoms.contains(&make_atom("c", vec![]))
        });

        let has_bc = answer_sets.iter().any(|as_set| {
            !as_set.atoms.contains(&make_atom("a", vec![]))
                && as_set.atoms.contains(&make_atom("b", vec![]))
                && as_set.atoms.contains(&make_atom("c", vec![]))
        });

        assert!(has_a_only, "Should have answer set with only a");
        assert!(has_bc, "Should have answer set with b and c");
    }
}

#[cfg(test)]
mod grounding_integration_tests {
    use super::*;
    use internment::Intern;

    // Helper to create an atom with string terms
    fn make_atom_terms(pred: &str, terms: Vec<&str>) -> Atom {
        Atom {
            predicate: Intern::new(pred.to_string()),
            terms: terms
                .into_iter()
                .map(|t| Term::Constant(Value::Atom(Intern::new(t.to_string()))))
                .collect(),
        }
    }

    #[test]
    fn test_sat_with_grounded_choice() {
        // Test: Choice rule with variables, grounded manually
        // Program (conceptually):
        //   node(a). node(b).
        //   { edge(X, Y) : node(X), node(Y) }.
        //
        // After grounding:
        //   node(a). node(b).
        //   { edge(a, a) }. { edge(a, b) }. { edge(b, a) }. { edge(b, b) }.

        let mut program = Program::new();

        // Add ground facts
        program.add_statement(Statement::Fact(Fact {
            atom: make_atom_terms("node", vec!["a"]),
        }));
        program.add_statement(Statement::Fact(Fact {
            atom: make_atom_terms("node", vec!["b"]),
        }));

        // Add ground choice rules (simulating what grounding would produce)
        for x in &["a", "b"] {
            for y in &["a", "b"] {
                program.add_statement(Statement::ChoiceRule(ChoiceRule {
                    lower_bound: None,
                    upper_bound: None,
                    elements: vec![ChoiceElement {
                        atom: make_atom_terms("edge", vec![x, y]),
                        condition: vec![],
                    }],
                    body: vec![],
                }));
            }
        }

        let answer_sets = asp_sat_evaluation(&program);

        // With 4 independent choice atoms, should have 2^4 = 16 answer sets
        // (including the node facts in all of them)
        assert_eq!(answer_sets.len(), 16, "Should have 16 answer sets (2^4 edges)");

        // All answer sets should contain node facts
        for answer_set in &answer_sets {
            assert!(answer_set.atoms.contains(&make_atom_terms("node", vec!["a"])));
            assert!(answer_set.atoms.contains(&make_atom_terms("node", vec!["b"])));
        }
    }

    #[test]
    fn test_sat_with_grounded_rule() {
        // Test: Rule with variables, grounded manually
        // Program (conceptually):
        //   node(a). node(b).
        //   { edge(X, Y) : node(X), node(Y) }.
        //   path(X, Y) :- edge(X, Y).
        //
        // After grounding:
        //   node(a). node(b).
        //   { edge(a, a) }. { edge(a, b) }. { edge(b, a) }. { edge(b, b) }.
        //   path(a, a) :- edge(a, a).
        //   path(a, b) :- edge(a, b).
        //   path(b, a) :- edge(b, a).
        //   path(b, b) :- edge(b, b).

        let mut program = Program::new();

        // Add facts
        program.add_statement(Statement::Fact(Fact {
            atom: make_atom_terms("node", vec!["a"]),
        }));
        program.add_statement(Statement::Fact(Fact {
            atom: make_atom_terms("node", vec!["b"]),
        }));

        // Add choice rules
        for x in &["a", "b"] {
            for y in &["a", "b"] {
                program.add_statement(Statement::ChoiceRule(ChoiceRule {
                    lower_bound: None,
                    upper_bound: None,
                    elements: vec![ChoiceElement {
                        atom: make_atom_terms("edge", vec![x, y]),
                        condition: vec![],
                    }],
                    body: vec![],
                }));

                // Add grounded rule: path(X, Y) :- edge(X, Y)
                program.add_statement(Statement::Rule(Rule {
                    head: make_atom_terms("path", vec![x, y]),
                    body: vec![Literal::Positive(make_atom_terms("edge", vec![x, y]))],
                }));
            }
        }

        let answer_sets = asp_sat_evaluation(&program);

        // Should still have 16 answer sets
        assert_eq!(answer_sets.len(), 16);

        // In each answer set, if edge(X, Y) is present, path(X, Y) should also be present
        for answer_set in &answer_sets {
            for x in &["a", "b"] {
                for y in &["a", "b"] {
                    let edge = make_atom_terms("edge", vec![x, y]);
                    let path = make_atom_terms("path", vec![x, y]);

                    if answer_set.atoms.contains(&edge) {
                        assert!(
                            answer_set.atoms.contains(&path),
                            "If edge({}, {}) is in answer set, path({}, {}) should also be present",
                            x, y, x, y
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_grounding_integration_simple() {
        // Test: Simple non-ground program with grounding
        // Program:
        //   node(a). node(b).
        //   { selected(X) : node(X) }.
        //
        // After grounding:
        //   node(a). node(b).
        //   { selected(a) }. { selected(b) }.

        use crate::asp_sat::asp_sat_evaluation_with_grounding;
        use internment::Intern;

        let mut program = Program::new();

        // Add facts
        program.add_statement(Statement::Fact(Fact {
            atom: Atom {
                predicate: Intern::new("node".to_string()),
                terms: vec![Term::Constant(Value::Atom(Intern::new("a".to_string())))],
            },
        }));
        program.add_statement(Statement::Fact(Fact {
            atom: Atom {
                predicate: Intern::new("node".to_string()),
                terms: vec![Term::Constant(Value::Atom(Intern::new("b".to_string())))],
            },
        }));

        // Add choice rule with variable and condition
        program.add_statement(Statement::ChoiceRule(ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: Atom {
                    predicate: Intern::new("selected".to_string()),
                    terms: vec![Term::Variable(Intern::new("X".to_string()))],
                },
                condition: vec![Literal::Positive(Atom {
                    predicate: Intern::new("node".to_string()),
                    terms: vec![Term::Variable(Intern::new("X".to_string()))],
                })],
            }],
            body: vec![],
        }));

        let answer_sets = asp_sat_evaluation_with_grounding(&program);

        // Should have 4 answer sets: {}, {selected(a)}, {selected(b)}, {selected(a), selected(b)}
        // Plus node facts in all of them
        assert_eq!(answer_sets.len(), 4, "Should have 4 answer sets");

        // All answer sets should contain node facts
        for answer_set in &answer_sets {
            assert!(answer_set.atoms.iter().any(|a| a.predicate.as_ref() == "node"));
        }
    }

    #[test]
    fn test_grounding_with_rules() {
        // Test: Non-ground program with rules that derive from facts
        // Program:
        //   edge(a, b). edge(b, c).
        //   path(X, Y) :- edge(X, Y).
        //   { avoid(X) : path(X, Y) }.
        //
        // The rule derives path facts before grounding the choice
        // Then choice is grounded based on path facts

        use crate::asp_sat::asp_sat_evaluation_with_grounding;
        use internment::Intern;

        let mut program = Program::new();

        // Add edge facts
        program.add_statement(Statement::Fact(Fact {
            atom: Atom {
                predicate: Intern::new("edge".to_string()),
                terms: vec![
                    Term::Constant(Value::Atom(Intern::new("a".to_string()))),
                    Term::Constant(Value::Atom(Intern::new("b".to_string()))),
                ],
            },
        }));
        program.add_statement(Statement::Fact(Fact {
            atom: Atom {
                predicate: Intern::new("edge".to_string()),
                terms: vec![
                    Term::Constant(Value::Atom(Intern::new("b".to_string()))),
                    Term::Constant(Value::Atom(Intern::new("c".to_string()))),
                ],
            },
        }));

        // Add rule: path(X, Y) :- edge(X, Y)
        program.add_statement(Statement::Rule(Rule {
            head: Atom {
                predicate: Intern::new("path".to_string()),
                terms: vec![
                    Term::Variable(Intern::new("X".to_string())),
                    Term::Variable(Intern::new("Y".to_string())),
                ],
            },
            body: vec![Literal::Positive(Atom {
                predicate: Intern::new("edge".to_string()),
                terms: vec![
                    Term::Variable(Intern::new("X".to_string())),
                    Term::Variable(Intern::new("Y".to_string())),
                ],
            })],
        }));

        // Add choice rule: { avoid(X) : path(X, _) }
        program.add_statement(Statement::ChoiceRule(ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: Atom {
                    predicate: Intern::new("avoid".to_string()),
                    terms: vec![Term::Variable(Intern::new("X".to_string()))],
                },
                condition: vec![Literal::Positive(Atom {
                    predicate: Intern::new("path".to_string()),
                    terms: vec![
                        Term::Variable(Intern::new("X".to_string())),
                        Term::Variable(Intern::new("_".to_string())),
                    ],
                })],
            }],
            body: vec![],
        }));

        let answer_sets = asp_sat_evaluation_with_grounding(&program);

        // The rule derives path(a, b) and path(b, c)
        // So we can choose avoid(a), avoid(b), both, or neither
        // That's 4 answer sets
        assert_eq!(answer_sets.len(), 4, "Should have 4 answer sets");

        // All answer sets should contain the path facts (derived from rules)
        for answer_set in &answer_sets {
            let has_path_ab = answer_set.atoms.iter().any(|a| {
                a.predicate.as_ref() == "path"
                    && a.terms.len() == 2
                    && matches!(&a.terms[0], Term::Constant(Value::Atom(ref s)) if s.as_ref() == "a")
                    && matches!(&a.terms[1], Term::Constant(Value::Atom(ref s)) if s.as_ref() == "b")
            });
            let has_path_bc = answer_set.atoms.iter().any(|a| {
                a.predicate.as_ref() == "path"
                    && a.terms.len() == 2
                    && matches!(&a.terms[0], Term::Constant(Value::Atom(ref s)) if s.as_ref() == "a")
                    && matches!(&a.terms[1], Term::Constant(Value::Atom(ref s)) if s.as_ref() == "c")
            });
            assert!(has_path_ab || has_path_bc, "Should have at least one path fact");
        }
    }
}

#[cfg(test)]
mod aggregate_integration_tests {
    use super::*;
    use crate::asp_sat::asp_sat_evaluation_with_grounding;
    use internment::Intern;

    #[test]
    fn test_sat_simple_count_constraint() {
        // Test: Count aggregate in constraint
        // Program:
        //   item(a). item(b). item(c).
        //   { selected(X) : item(X) }.
        //   :- count { X : selected(X) } > 2.
        //
        // Should eliminate answer sets with 3 selected items

        let mut program = Program::new();

        // Add facts
        for item in &["a", "b", "c"] {
            program.add_statement(Statement::Fact(Fact {
                atom: Atom {
                    predicate: Intern::new("item".to_string()),
                    terms: vec![Term::Constant(Value::Atom(Intern::new(item.to_string())))],
                },
            }));
        }

        // Add choice rule: { selected(X) : item(X) }
        program.add_statement(Statement::ChoiceRule(ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: Atom {
                    predicate: Intern::new("selected".to_string()),
                    terms: vec![Term::Variable(Intern::new("X".to_string()))],
                },
                condition: vec![Literal::Positive(Atom {
                    predicate: Intern::new("item".to_string()),
                    terms: vec![Term::Variable(Intern::new("X".to_string()))],
                })],
            }],
            body: vec![],
        }));

        // Add constraint: :- count { X : selected(X) } > 2
        program.add_statement(Statement::Constraint(Constraint {
            body: vec![Literal::Aggregate(AggregateAtom {
                function: AggregateFunction::Count,
                variables: vec![Intern::new("X".to_string())],
                elements: vec![Literal::Positive(Atom {
                    predicate: Intern::new("selected".to_string()),
                    terms: vec![Term::Variable(Intern::new("X".to_string()))],
                })],
                comparison: ComparisonOp::GreaterThan,
                value: Term::Constant(Value::Integer(2)),
            })],
        }));

        let answer_sets = asp_sat_evaluation_with_grounding(&program);

        // Without constraint: 2^3 = 8 answer sets
        // With constraint: eliminate those with 3 items = 7 answer sets
        assert_eq!(
            answer_sets.len(),
            7,
            "Should have 7 answer sets (all except the one with 3 selected)"
        );

        // Verify no answer set has more than 2 selected items
        for answer_set in &answer_sets {
            let selected_count = answer_set
                .atoms
                .iter()
                .filter(|a| a.predicate.as_ref() == "selected")
                .count();
            assert!(
                selected_count <= 2,
                "Answer set should not have more than 2 selected items, found {}",
                selected_count
            );
        }
    }

    #[test]
    fn test_sat_exact_count_constraint() {
        // Test: Exact count requirement
        // Program:
        //   color(red). color(blue). color(green).
        //   { picked(C) : color(C) }.
        //   :- count { C : picked(C) } != 2.
        //
        // Should only allow answer sets with exactly 2 colors

        let mut program = Program::new();

        // Add facts
        for color in &["red", "blue", "green"] {
            program.add_statement(Statement::Fact(Fact {
                atom: Atom {
                    predicate: Intern::new("color".to_string()),
                    terms: vec![Term::Constant(Value::Atom(Intern::new(color.to_string())))],
                },
            }));
        }

        // Add choice rule
        program.add_statement(Statement::ChoiceRule(ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: Atom {
                    predicate: Intern::new("picked".to_string()),
                    terms: vec![Term::Variable(Intern::new("C".to_string()))],
                },
                condition: vec![Literal::Positive(Atom {
                    predicate: Intern::new("color".to_string()),
                    terms: vec![Term::Variable(Intern::new("C".to_string()))],
                })],
            }],
            body: vec![],
        }));

        // Add constraint: must pick exactly 2
        program.add_statement(Statement::Constraint(Constraint {
            body: vec![Literal::Aggregate(AggregateAtom {
                function: AggregateFunction::Count,
                variables: vec![Intern::new("C".to_string())],
                elements: vec![Literal::Positive(Atom {
                    predicate: Intern::new("picked".to_string()),
                    terms: vec![Term::Variable(Intern::new("C".to_string()))],
                })],
                comparison: ComparisonOp::NotEqual,
                value: Term::Constant(Value::Integer(2)),
            })],
        }));

        let answer_sets = asp_sat_evaluation_with_grounding(&program);

        // Should have exactly C(3,2) = 3 answer sets
        assert_eq!(
            answer_sets.len(),
            3,
            "Should have exactly 3 answer sets (choose 2 from 3 colors)"
        );

        // Verify all answer sets have exactly 2 picked colors
        for answer_set in &answer_sets {
            let picked_count = answer_set
                .atoms
                .iter()
                .filter(|a| a.predicate.as_ref() == "picked")
                .count();
            assert_eq!(
                picked_count, 2,
                "Each answer set should have exactly 2 picked colors"
            );
        }
    }

    #[test]
    fn test_sat_count_with_multiple_conditions() {
        // Test: Count with multiple conditions in aggregate
        // Program:
        //   weapon(sword). weapon(axe). weapon(bow).
        //   heavy(sword). heavy(axe).
        //   { carry(W) : weapon(W) }.
        //   :- count { W : carry(W), heavy(W) } > 1.
        //
        // Cannot carry more than 1 heavy weapon

        let mut program = Program::new();

        // Add weapon facts
        for weapon in &["sword", "axe", "bow"] {
            program.add_statement(Statement::Fact(Fact {
                atom: Atom {
                    predicate: Intern::new("weapon".to_string()),
                    terms: vec![Term::Constant(Value::Atom(Intern::new(weapon.to_string())))],
                },
            }));
        }

        // Add heavy facts
        for heavy_weapon in &["sword", "axe"] {
            program.add_statement(Statement::Fact(Fact {
                atom: Atom {
                    predicate: Intern::new("heavy".to_string()),
                    terms: vec![Term::Constant(Value::Atom(Intern::new(heavy_weapon.to_string())))],
                },
            }));
        }

        // Add choice rule
        program.add_statement(Statement::ChoiceRule(ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: Atom {
                    predicate: Intern::new("carry".to_string()),
                    terms: vec![Term::Variable(Intern::new("W".to_string()))],
                },
                condition: vec![Literal::Positive(Atom {
                    predicate: Intern::new("weapon".to_string()),
                    terms: vec![Term::Variable(Intern::new("W".to_string()))],
                })],
            }],
            body: vec![],
        }));

        // Add constraint: cannot carry more than 1 heavy weapon
        program.add_statement(Statement::Constraint(Constraint {
            body: vec![Literal::Aggregate(AggregateAtom {
                function: AggregateFunction::Count,
                variables: vec![Intern::new("W".to_string())],
                elements: vec![
                    Literal::Positive(Atom {
                        predicate: Intern::new("carry".to_string()),
                        terms: vec![Term::Variable(Intern::new("W".to_string()))],
                    }),
                    Literal::Positive(Atom {
                        predicate: Intern::new("heavy".to_string()),
                        terms: vec![Term::Variable(Intern::new("W".to_string()))],
                    }),
                ],
                comparison: ComparisonOp::GreaterThan,
                value: Term::Constant(Value::Integer(1)),
            })],
        }));

        let answer_sets = asp_sat_evaluation_with_grounding(&program);

        // Should eliminate answer sets where both sword and axe are carried
        // Without constraint: 2^3 = 8 answer sets
        // With constraint: eliminate {sword, axe}, {sword, axe, bow} = 6 answer sets
        assert_eq!(
            answer_sets.len(),
            6,
            "Should have 6 answer sets (8 minus 2 with both heavy weapons)"
        );

        // Verify no answer set has both heavy weapons
        for answer_set in &answer_sets {
            let has_sword = answer_set.atoms.iter().any(|a| {
                a.predicate.as_ref() == "carry"
                    && a.terms.len() == 1
                    && matches!(&a.terms[0], Term::Constant(Value::Atom(s)) if s.as_ref() == "sword")
            });
            let has_axe = answer_set.atoms.iter().any(|a| {
                a.predicate.as_ref() == "carry"
                    && a.terms.len() == 1
                    && matches!(&a.terms[0], Term::Constant(Value::Atom(s)) if s.as_ref() == "axe")
            });
            assert!(
                !(has_sword && has_axe),
                "Should not carry both sword and axe"
            );
        }
    }
}
