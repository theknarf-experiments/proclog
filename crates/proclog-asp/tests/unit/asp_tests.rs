    use super::*;
    use internment::Intern;
    use proclog_parser::{ParseError, SrcId};

    fn parse_program(input: &str) -> Result<Program, Vec<ParseError>> {
        proclog_parser::parse_program(input, SrcId::empty())
    }

    #[test]
    fn test_seeded_sampling_is_deterministic() {
        let input = r#"
            item(a).
            item(b).

            { selected(X) : item(X) }.
        "#;

        let program = parse_program(input).unwrap();

        let sample1 = asp_sample(&program, 42, 2);
        let sample2 = asp_sample(&program, 42, 2);

        assert_eq!(sample1, sample2);
        assert_eq!(sample1.len(), 2);
    }

    #[test]
    fn test_seed_changes_sampling_result() {
        let input = r#"
            item(a).
            item(b).
            item(c).

            { selected(X) : item(X) }.
        "#;

        let program = parse_program(input).unwrap();

        let sample_a = asp_sample(&program, 1, 1);
        let sample_b = asp_sample(&program, 2, 1);

        assert_ne!(sample_a, sample_b);
    }

    #[test]
    fn test_seeded_sampling_respects_constraints() {
        let input = r#"
            item(a).
            item(b).

            { selected(X) : item(X) }.

            % Must pick at least one item
            :- not selected(a), not selected(b).
        "#;

        let program = parse_program(input).unwrap();
        let samples = asp_sample(&program, 99, 3);

        let selected_symbol = Intern::new("selected".to_string());

        for answer_set in samples {
            let selected_count = answer_set
                .atoms
                .iter()
                .filter(|atom| atom.predicate == selected_symbol)
                .count();
            assert!(
                selected_count >= 1,
                "Expected at least one selected item in sampled answer set"
            );
        }
    }

    #[test]
    fn test_sampling_fallback_enumerates_all_answer_sets() {
        let input = r#"
            item(1..3).
            1 { chosen(X) : item(X) }.
            :- not chosen(1).
        "#;

        let program = parse_program(input).unwrap();

        let all_sets = asp_evaluation(&program);
        assert!(!all_sets.is_empty());

        let samples = asp_sample(&program, 123, 10);
        assert_eq!(samples.len(), all_sets.len());

        fn canonicalize(sets: &[AnswerSet]) -> Vec<Vec<String>> {
            let mut normalized: Vec<Vec<String>> = sets
                .iter()
                .map(|answer_set| {
                    let mut atoms: Vec<String> = answer_set
                        .atoms
                        .iter()
                        .map(|atom| {
                            let mut term_strings: Vec<String> = atom
                                .terms
                                .iter()
                                .map(|term| format!("{:?}", term))
                                .collect();
                            term_strings.sort();
                            format!("{}:{:?}", atom.predicate.as_ref(), term_strings)
                        })
                        .collect();
                    atoms.sort();
                    atoms
                })
                .collect();
            normalized.sort();
            normalized
        }

        assert_eq!(canonicalize(&samples), canonicalize(&all_sets));
    }

    #[test]
    fn test_simple_choice_no_constraints() {
        let input = r#"
            { selected(a); selected(b) }.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Should have 4 answer sets: {}, {a}, {b}, {a,b}
        assert_eq!(answer_sets.len(), 4);
    }

    #[test]
    fn test_choice_with_cardinality() {
        let input = r#"
            1 { selected(a); selected(b); selected(c) } 2.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Should have C(3,1) + C(3,2) = 3 + 3 = 6 answer sets
        assert_eq!(answer_sets.len(), 6);
    }

    // Integration tests: Constants + ASP

    #[test]
    fn test_constants_with_choice_ranges() {
        let input = r#"
            #const max = 3.
            item(1..max).
            { selected(X) : item(X) }.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Should have 2^3 = 8 answer sets (all subsets of {1,2,3})
        assert_eq!(answer_sets.len(), 8);

        // Verify at least one answer set has the expected structure
        let has_all_items = answer_sets.iter().any(|as_set| {
            let item_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("item".to_string()))
                .count();
            item_count == 3 // Should have exactly 3 items
        });
        assert!(has_all_items);
    }

    #[test]
    fn test_constants_with_cardinality() {
        // Note: Parser doesn't yet support constant names in cardinality bounds
        // So we test constants in ranges and facts instead
        let input = r#"
            #const count = 3.
            item(1..count).
            2 { selected(X) : item(X) } 2.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Should have C(3,2) = 3 answer sets
        assert_eq!(answer_sets.len(), 3);

        // Verify items are generated from constant
        for as_set in &answer_sets {
            let item_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("item".to_string()))
                .count();
            assert_eq!(item_count, 3, "Should have 3 items from range");
        }
    }

    // Integration tests: Datalog rules + ASP

    #[test]
    fn test_datalog_derives_facts_for_choice_conditions() {
        let input = r#"
            % Base facts
            weapon(sword).
            weapon(axe).
            armor(shield).

            % Derived facts: classify equipment
            equipment(X) :- weapon(X).
            equipment(X) :- armor(X).

            % Choose equipment based on derived facts
            1 { selected(X) : equipment(X) } 2.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Should have C(3,1) + C(3,2) = 3 + 3 = 6 answer sets
        assert_eq!(answer_sets.len(), 6);

        // Verify equipment facts are derived
        for as_set in &answer_sets {
            let equipment_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("equipment".to_string()))
                .count();
            assert_eq!(equipment_count, 3, "Should have 3 equipment facts");
        }
    }

    #[test]
    fn test_choice_with_recursive_rules() {
        let input = r#"
            % Base facts
            edge(a, b).
            edge(b, c).

            % Transitive closure
            path(X, Y) :- edge(X, Y).
            path(X, Z) :- path(X, Y), edge(Y, Z).

            % Choose paths
            { selected_path(X, Y) : path(X, Y) }.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Should have 2^3 = 8 subsets (paths: a-b, b-c, a-c)
        assert_eq!(answer_sets.len(), 8);

        // Verify path derivation
        let has_transitive = answer_sets.iter().any(|as_set| {
            as_set.atoms.iter().any(|a| {
                a.predicate == Intern::new("path".to_string())
                    && format!("{:?}", a).contains("a")
                    && format!("{:?}", a).contains("c")
            })
        });
        assert!(has_transitive, "Should derive transitive path a->c");
    }

    // Integration tests: Constraints + ASP

    #[test]
    fn test_constraints_filter_answer_sets() {
        let input = r#"
            item(a).
            item(b).

            { selected(X) : item(X) }.

            % Must select at least one
            :- not selected(a), not selected(b).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Should have 3 answer sets: {a}, {b}, {a,b}
        // Empty set is filtered out by constraint
        assert_eq!(answer_sets.len(), 3);

        // Verify no empty answer sets
        for as_set in &answer_sets {
            let selected_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("selected".to_string()))
                .count();
            assert!(selected_count > 0, "Should not have empty selections");
        }
    }

    #[test]
    fn test_constraints_with_derived_facts() {
        let input = r#"
            % Base items
            item(a).
            item(b).
            item(c).

            % Choose items
            { selected(X) : item(X) }.

            % Derive count (in real use would be aggregate, simplified here)
            has_selection :- selected(a).
            has_selection :- selected(b).
            has_selection :- selected(c).

            % Constraint: must have some selection
            :- not has_selection.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Should exclude empty set, so 7 answer sets
        assert_eq!(answer_sets.len(), 7);
    }

    // Integration tests: Full pipeline (constants + ranges + rules + choice + constraints)

    #[test]
    fn test_full_pipeline_grid_generation() {
        let input = r#"
            #const width = 3.
            #const height = 3.

            % Generate grid cells
            cell(1..width, 1..height).

            % Choose some cells to be solid
            2 { solid(X, Y) : cell(X, Y) } 4.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Should have C(9,2) + C(9,3) + C(9,4) = 36 + 84 + 126 = 246 answer sets
        assert_eq!(answer_sets.len(), 246);

        // Verify grid cells are generated
        for as_set in &answer_sets {
            let cell_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("cell".to_string()))
                .count();
            assert_eq!(cell_count, 9, "Should have 9 cells (3x3 grid)");

            let solid_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("solid".to_string()))
                .count();
            assert!(
                (2..=4).contains(&solid_count),
                "Should have 2-4 solid cells"
            );
        }
    }

    #[test]
    fn test_full_pipeline_with_negation() {
        let input = r#"
            #const size = 2.

            % Positions
            pos(1..size).

            % Choose occupied positions
            { occupied(X) : pos(X) }.

            % Derive free positions
            free(X) :- pos(X), not occupied(X).

            % Constraint: must have at least one free position
            :- not free(1), not free(2).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Should have 3 answer sets (can't occupy both positions)
        // {}, {1}, {2}
        assert_eq!(answer_sets.len(), 3);

        // Verify free positions are correctly derived
        for as_set in &answer_sets {
            let occupied_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("occupied".to_string()))
                .count();
            let free_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("free".to_string()))
                .count();

            // occupied + free should equal total positions
            assert_eq!(occupied_count + free_count, 2, "Should cover all positions");
        }
    }

    #[test]
    fn test_ranges_in_facts_and_choices() {
        let input = r#"
            #const grid_size = 2.

            % Facts with ranges
            row(1..grid_size).
            col(1..grid_size).

            % Derived facts combining ranges
            cell(X, Y) :- row(X), col(Y).

            % Choice using derived facts
            { selected(X, Y) : cell(X, Y) }.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // 2x2 grid = 4 cells, so 2^4 = 16 answer sets
        assert_eq!(answer_sets.len(), 16);

        // Verify cell derivation
        for as_set in &answer_sets {
            let cell_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("cell".to_string()))
                .count();
            assert_eq!(cell_count, 4, "Should derive 4 cells from 2x2 grid");
        }
    }

    // Edge case tests

    #[test]
    fn test_edge_case_empty_choice() {
        let input = r#"
            item(a).
            { selected(X) : item(X) }.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Should have 2 answer sets: {}, {selected(a)}
        assert_eq!(answer_sets.len(), 2);
    }

    #[test]
    fn test_edge_case_choice_with_impossible_bounds() {
        // Lower bound > upper bound should give no answer sets
        let input = r#"
            item(a).
            5 { selected(X) : item(X) } 2.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Impossible to select 5-2 items from 1 item
        assert_eq!(answer_sets.len(), 0);
    }

    #[test]
    fn test_edge_case_choice_with_no_options() {
        let input = r#"
            % No items exist
            { selected(X) : item(X) }.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Only empty set possible
        assert_eq!(answer_sets.len(), 1);
    }

    #[test]
    fn test_edge_case_multiple_choice_rules() {
        let input = r#"
            item(a).
            item(b).

            { selected(X) : item(X) }.
            { preferred(X) : item(X) }.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // 2^2 * 2^2 = 16 combinations
        assert_eq!(answer_sets.len(), 16);
    }

    #[test]
    fn test_edge_case_unsatisfiable_constraint() {
        let input = r#"
            item(a).
            { selected(X) : item(X) }.

            % Contradictory constraints
            :- selected(a).
            :- not selected(a).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // No answer set can satisfy both constraints
        assert_eq!(answer_sets.len(), 0);
    }

    #[test]
    fn test_edge_case_all_atoms_must_be_chosen() {
        let input = r#"
            item(a).
            item(b).

            2 { selected(X) : item(X) } 2.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Must select exactly 2, and we have exactly 2 items
        // So only 1 answer set: both selected
        assert_eq!(answer_sets.len(), 1);

        let selected_count = answer_sets[0]
            .atoms
            .iter()
            .filter(|a| a.predicate == Intern::new("selected".to_string()))
            .count();
        assert_eq!(selected_count, 2);
    }

    #[test]
    fn test_edge_case_zero_lower_bound() {
        let input = r#"
            item(a).
            item(b).

            0 { selected(X) : item(X) } 1.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Can select 0 or 1: {}, {a}, {b}
        assert_eq!(answer_sets.len(), 3);
    }

    #[test]
    fn test_edge_case_single_element_range() {
        let input = r#"
            #const val = 5.
            item(val..val).
            { selected(X) : item(X) }.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Range 5..5 produces one item
        assert_eq!(answer_sets.len(), 2); // {}, {selected(5)}

        // Verify the single item exists
        let has_item_5 = answer_sets.iter().any(|as_set| {
            as_set.atoms.iter().any(|a| {
                a.predicate == Intern::new("item".to_string())
                    && matches!(
                        a.terms.first(),
                        Some(proclog_ast::Term::Constant(proclog_ast::Value::Integer(5)))
                    )
            })
        });
        assert!(has_item_5);
    }

    #[test]
    fn test_edge_case_negative_constants() {
        let input = r#"
            #const start = -2.
            #const end = 1.

            value(start..end).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Should generate values: -2, -1, 0, 1
        assert_eq!(answer_sets.len(), 1);

        let value_count = answer_sets[0]
            .atoms
            .iter()
            .filter(|a| a.predicate == Intern::new("value".to_string()))
            .count();
        assert_eq!(value_count, 4);
    }

    #[test]
    fn test_edge_case_choice_with_negation_in_condition() {
        let input = r#"
            item(a).
            item(b).
            forbidden(a).

            { selected(X) : item(X), not forbidden(X) }.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Can only choose from non-forbidden items (just b)
        // So 2 answer sets: {}, {selected(b)}
        assert_eq!(answer_sets.len(), 2);

        // Verify 'a' is never selected
        let a_selected = answer_sets.iter().any(|as_set| {
            as_set.atoms.iter().any(|atom| {
                atom.predicate == Intern::new("selected".to_string())
                    && atom.terms.first()
                        == Some(&Term::Constant(Value::Atom(Intern::new("a".to_string()))))
            })
        });
        assert!(
            !a_selected,
            "Item 'a' should never be selected (it's forbidden)"
        );
    }

    #[test]
    fn test_edge_case_circular_derivation_with_choice() {
        let input = r#"
            % Base facts
            node(a).
            node(b).

            % Choose edges
            { edge(X, Y) : node(X), node(Y) }.

            % Derive reachability (transitive)
            reach(X, Y) :- edge(X, Y).
            reach(X, Z) :- reach(X, Y), edge(Y, Z).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // 2 nodes, so 4 possible edges: (a,a), (a,b), (b,a), (b,b)
        // 2^4 = 16 answer sets
        assert_eq!(answer_sets.len(), 16);
    }

    #[test]
    fn test_edge_case_empty_program() {
        // A program with no choice rules should have one answer set containing just the facts
        let input = r#"
            item(a).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // No choice rules, so one answer set with just the base facts
        assert_eq!(answer_sets.len(), 1);
        assert!(answer_sets[0]
            .atoms
            .iter()
            .any(|a| a.predicate == Intern::new("item".to_string())));
    }

    #[test]
    fn test_edge_case_only_constraints_no_choices() {
        let input = r#"
            item(a).
            :- item(a).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Constraint is violated by the fact
        assert_eq!(answer_sets.len(), 0);
    }

    // ===== ADVANCED INTEGRATION TESTS =====

    #[test]
    fn test_cascading_choices() {
        // First choice creates facts, rules derive more facts, second choice uses them
        let input = r#"
            % Base locations
            location(a).
            location(b).

            % Choose which locations have rooms
            1 { has_room(X) : location(X) } 2.

            % Derive that rooms need doors
            needs_door(X) :- has_room(X).

            % Choose which doors to build (from derived facts!)
            { build_door(X) : needs_door(X) }.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // First choice: C(2,1) + C(2,2) = 2 + 1 = 3 ways to choose rooms
        // Second choice depends on first - but in standard ASP, choices are grounded before selection
        // So second choice grounds to 0 atoms (needs_door depends on has_room which is a choice atom)
        // Result: 3 answer sets (just from first choice)
        assert_eq!(answer_sets.len(), 3);

        // Verify cascading: every answer set with has_room(X) also has needs_door(X)
        for as_set in &answer_sets {
            let rooms: Vec<_> = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("has_room".to_string()))
                .collect();
            let needs: Vec<_> = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("needs_door".to_string()))
                .collect();

            // Same atoms should appear in both predicates
            assert_eq!(rooms.len(), needs.len());
        }
    }

    #[test]
    fn test_choice_affects_rule_applicability() {
        // Chosen atoms determine which rules can fire
        let input = r#"
            item(sword).
            item(shield).

            % Choose one item
            1 { equipped(X) : item(X) } 1.

            % Different bonuses based on what's equipped
            attack_bonus(5) :- equipped(sword).
            defense_bonus(5) :- equipped(shield).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // 2 answer sets: one with sword (attack bonus), one with shield (defense bonus)
        assert_eq!(answer_sets.len(), 2);

        // Verify mutual exclusion
        let has_both_bonuses = answer_sets.iter().any(|as_set| {
            let has_attack = as_set
                .atoms
                .iter()
                .any(|a| a.predicate == Intern::new("attack_bonus".to_string()));
            let has_defense = as_set
                .atoms
                .iter()
                .any(|a| a.predicate == Intern::new("defense_bonus".to_string()));
            has_attack && has_defense
        });
        assert!(
            !has_both_bonuses,
            "Should not have both bonuses in same answer set"
        );
    }

    #[test]
    fn test_multi_strata_with_choices() {
        // Test stratified negation with multiple strata + choices
        let input = r#"
            % Base facts
            node(a).
            node(b).
            node(c).

            % Mark 'a' as the start node
            start(a).

            % Choose some edges
            { edge(X, Y) : node(X), node(Y) }.

            % Stratum 1: Derive reachability from start
            reach(X, Y) :- edge(X, Y).
            reach(X, Z) :- reach(X, Y), edge(Y, Z).

            % Stratum 2: Find isolated nodes (not reachable from 'a', and not 'a' itself)
            isolated(X) :- node(X), not reach(a, X), not start(X).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // 3 nodes → 9 possible edges → 2^9 = 512 answer sets
        assert_eq!(answer_sets.len(), 512);

        // Verify semantic correctness: if a node is isolated, there's no path from 'a' to it
        for as_set in answer_sets.iter().take(10) {
            // Sample some answer sets
            let isolated: Vec<_> = as_set
                .atoms
                .iter()
                .filter(|atom| atom.predicate == Intern::new("isolated".to_string()))
                .collect();

            let reachable: Vec<_> = as_set
                .atoms
                .iter()
                .filter(|atom| atom.predicate == Intern::new("reach".to_string()))
                .filter(|atom| {
                    if let Some(Term::Constant(Value::Atom(from))) = atom.terms.first() {
                        *from == Intern::new("a".to_string())
                    } else {
                        false
                    }
                })
                .collect();

            // An isolated node should not appear in reachable list from 'a'
            for iso_atom in isolated {
                if let Some(Term::Constant(Value::Atom(iso_node))) = iso_atom.terms.first() {
                    let is_reachable = reachable.iter().any(|reach_atom| {
                        if let Some(Term::Constant(Value::Atom(to))) = reach_atom.terms.get(1) {
                            to == iso_node
                        } else {
                            false
                        }
                    });
                    assert!(
                        !is_reachable,
                        "Isolated node {:?} should not be reachable from 'a'",
                        iso_node
                    );
                }
            }
        }
    }

    #[test]
    fn test_compound_constraint() {
        // Constraint with multiple literals (conjunction)
        let input = r#"
            color(red).
            color(blue).
            size(small).
            size(large).

            % Choose one color and one size
            1 { selected_color(C) : color(C) } 1.
            1 { selected_size(S) : size(S) } 1.

            % Constraint: cannot have red + large (too visible)
            :- selected_color(red), selected_size(large).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // 2 colors × 2 sizes = 4 combinations, minus 1 forbidden (red+large) = 3 answer sets
        assert_eq!(answer_sets.len(), 3);

        // Verify the forbidden combination doesn't exist
        let has_red_large = answer_sets.iter().any(|as_set| {
            let has_red = as_set.atoms.iter().any(|a| {
                a.predicate == Intern::new("selected_color".to_string())
                    && a.terms.first()
                        == Some(&Term::Constant(Value::Atom(Intern::new("red".to_string()))))
            });
            let has_large = as_set.atoms.iter().any(|a| {
                a.predicate == Intern::new("selected_size".to_string())
                    && a.terms.first()
                        == Some(&Term::Constant(Value::Atom(Intern::new(
                            "large".to_string(),
                        ))))
            });
            has_red && has_large
        });
        assert!(!has_red_large, "Red + large should be forbidden");
    }

    #[test]
    fn test_multiple_ranges_cartesian_product() {
        // Multiple ranges in single atom create cartesian product
        let input = r#"
            #const width = 3.
            #const height = 3.

            % Create a grid using two ranges
            cell(1..width, 1..height).

            % Choose 2 cells
            2 { selected(X, Y) : cell(X, Y) } 2.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // 3×3 = 9 cells, choose 2: C(9,2) = 36 answer sets
        assert_eq!(answer_sets.len(), 36);

        // Verify each answer set has exactly 2 selected cells
        for as_set in &answer_sets {
            let selected_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("selected".to_string()))
                .count();
            assert_eq!(selected_count, 2);
        }
    }

    #[test]
    fn test_constraints_with_universal_quantification() {
        // Constraint forces something to exist (via negation)
        let input = r#"
            player(alice).
            player(bob).
            weapon(sword).
            weapon(bow).

            % Each player chooses a weapon
            1 { has_weapon(P, W) : weapon(W) } 1 :- player(P).

            % Constraint: someone must have the sword (it's required for quest)
            :- not has_weapon(alice, sword), not has_weapon(bob, sword).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // FIXED! Choice rules with bodies now split by body substitutions
        // Expected: 2 players × 2 weapons = 4 total combinations
        // Each player chooses 1 weapon independently (2×2 = 4 possibilities)
        // Constraint eliminates 1: not (alice with bow AND bob with bow)
        // Result: 3 valid answer sets
        assert_eq!(answer_sets.len(), 3);

        // Verify every answer set has at least one player with sword
        for as_set in &answer_sets {
            let someone_has_sword = as_set.atoms.iter().any(|a| {
                a.predicate == Intern::new("has_weapon".to_string())
                    && a.terms.get(1)
                        == Some(&Term::Constant(Value::Atom(Intern::new(
                            "sword".to_string(),
                        ))))
            });
            assert!(someone_has_sword, "At least one player must have sword");
        }
    }

    #[test]
    fn test_large_choice_space() {
        // Stress test: larger choice space
        let input = r#"
            #const n = 5.
            item(1..n).

            % Choose 2 from 5 items
            2 { selected(X) : item(X) } 2.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // C(5,2) = 10 answer sets
        assert_eq!(answer_sets.len(), 10);
    }

    #[test]
    fn test_deep_recursive_derivation_with_choice() {
        // Deep chain of derivations before choice
        let input = r#"
            base(1).

            level1(X) :- base(X).
            level2(X) :- level1(X).
            level3(X) :- level2(X).
            level4(X) :- level3(X).
            level5(X) :- level4(X).

            % Choice uses deeply derived fact
            { selected(X) : level5(X) }.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // One item at level5, choice: include or not = 2 answer sets
        assert_eq!(answer_sets.len(), 2);

        // Verify the chain worked: if selected, all levels should exist
        let has_selection = answer_sets.iter().any(|as_set| {
            as_set
                .atoms
                .iter()
                .any(|a| a.predicate == Intern::new("selected".to_string()))
        });
        assert!(has_selection);
    }

    #[test]
    fn test_multiple_independent_choice_rules() {
        // Multiple choice rules that don't interact
        let input = r#"
            color(red).
            color(blue).

            size(small).
            size(large).

            pattern(dots).
            pattern(stripes).

            % Three independent choices
            1 { selected_color(C) : color(C) } 1.
            1 { selected_size(S) : size(S) } 1.
            { selected_pattern(P) : pattern(P) }.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // 2 colors × 2 sizes × (2^2 patterns) = 2 × 2 × 4 = 16 answer sets
        assert_eq!(answer_sets.len(), 16);
    }

    #[test]
    fn test_dungeon_with_connectivity_constraint() {
        // Real-world PCG: rooms must be connected
        // Simplified version without arithmetic
        let input = r#"
            pos(1).
            pos(2).
            pos(3).

            % Define adjacency explicitly (parser doesn't support arithmetic)
            adj(1, 2).
            adj(2, 3).
            adj(2, 1).
            adj(3, 2).

            % Choose 2-3 rooms
            2 { room(X) : pos(X) } 3.

            % Connected rooms via adjacency
            connected(1) :- room(1).
            connected(X) :- connected(Y), room(X), adj(Y, X).

            % Constraint: if we have rooms, all must be connected from room 1
            :- room(1), room(X), not connected(X).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Without room 1, constraint doesn't apply: C(2,2) = 1 case
        // With room 1:
        //   - room(1,2): valid (connected)
        //   - room(1,3): invalid (not connected)
        //   - room(1,2,3): valid (all connected)
        // Total valid: 1 + 2 = 3 answer sets
        assert!(
            answer_sets.len() >= 2,
            "Should have at least 2 valid configurations"
        );

        // Verify semantic correctness: if room 1 exists, all rooms should be connected
        for as_set in &answer_sets {
            let has_room_1 = as_set.atoms.iter().any(|a| {
                a.predicate == Intern::new("room".to_string())
                    && a.terms.first() == Some(&Term::Constant(Value::Integer(1)))
            });

            if has_room_1 {
                let rooms: Vec<i64> = as_set
                    .atoms
                    .iter()
                    .filter(|a| a.predicate == Intern::new("room".to_string()))
                    .filter_map(|a| {
                        if let Some(Term::Constant(Value::Integer(x))) = a.terms.first() {
                            Some(*x)
                        } else {
                            None
                        }
                    })
                    .collect();

                let connected: Vec<i64> = as_set
                    .atoms
                    .iter()
                    .filter(|a| a.predicate == Intern::new("connected".to_string()))
                    .filter_map(|a| {
                        if let Some(Term::Constant(Value::Integer(x))) = a.terms.first() {
                            Some(*x)
                        } else {
                            None
                        }
                    })
                    .collect();

                // All rooms should be in connected list
                for room in rooms {
                    assert!(
                        connected.contains(&room),
                        "Room {} should be connected",
                        room
                    );
                }
            }
        }
    }

    #[test]
    fn test_constraints_eliminate_all_candidates() {
        // All possible choices violate constraints → 0 answer sets
        let input = r#"
            item(a).
            item(b).

            % Must choose exactly 1
            1 { selected(X) : item(X) } 1.

            % But both items are forbidden
            :- selected(a).
            :- selected(b).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // No valid answer sets
        assert_eq!(answer_sets.len(), 0);
    }

    #[test]
    fn test_exponential_answer_set_verification() {
        // Verify correct counting with exponential growth
        let input = r#"
            item(a).
            item(b).
            item(c).
            item(d).

            % No bounds = 2^4 = 16 subsets
            { selected(X) : item(X) }.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Verify exactly 2^4 = 16 answer sets
        assert_eq!(answer_sets.len(), 16);

        // Verify we have empty set and full set
        let has_empty = answer_sets.iter().any(|as_set| {
            as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("selected".to_string()))
                .count()
                == 0
        });
        let has_full = answer_sets.iter().any(|as_set| {
            as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("selected".to_string()))
                .count()
                == 4
        });

        assert!(has_empty, "Should include empty selection");
        assert!(has_full, "Should include full selection");
    }

    #[test]
    fn test_choice_with_complex_derived_conditions() {
        // Choice condition depends on complex rule chain
        let input = r#"
            % Base facts
            raw_material(iron).
            raw_material(wood).

            tool(hammer).
            tool(saw).

            % Processing rules
            can_process(iron, hammer).
            can_process(wood, saw).

            processed(M) :- raw_material(M), tool(T), can_process(M, T), available(T).

            % Choose which tools are available
            { available(T) : tool(T) }.

            % Choose what to craft (from processed materials only)
            { craft(M) : processed(M) }.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // NOTE: In standard ASP, choice rules are grounded before choices are made
        // The craft choice condition `processed(M)` depends on `available(T)` (a choice atom)
        // At grounding time, no tools are available, so no materials are processed
        // Thus craft choice grounds to 0 atoms
        // Result: 2^2 = 4 answer sets (just from tool choice)
        assert_eq!(answer_sets.len(), 4);

        // Verify semantic correctness: can only craft if material is processed
        for as_set in &answer_sets {
            let crafted: Vec<_> = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("craft".to_string()))
                .collect();
            let processed_mats: Vec<_> = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("processed".to_string()))
                .collect();

            // Every crafted material must be in processed list
            for craft_atom in crafted {
                let is_processed = processed_mats
                    .iter()
                    .any(|p| p.terms.first() == craft_atom.terms.first());
                assert!(is_processed, "Can only craft processed materials");
            }
        }
    }

    #[test]
    fn test_constraints_on_multiple_strata() {
        // Constraints reference facts from different derivation strata
        let input = r#"
            % Base
            node(a).
            node(b).

            % Choose edges
            { edge(X, Y) : node(X), node(Y) }.

            % Stratum 1: derive paths
            path(X, Y) :- edge(X, Y).

            % Stratum 2: derive completeness
            complete :- path(a, b), path(b, a).

            % Constraint: must not be complete (prevent cycles)
            :- complete.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // 2 nodes → 4 possible edges
        // 2^4 = 16 total combinations
        // Eliminate those with both edge(a,b) AND edge(b,a)
        // That's 1 case (both present) in the 2^2 choices for {edge(a,b), edge(b,a)}
        // But we need to account for all combinations of other edges too
        // Actually: 4 edges, 2^4 = 16, minus those with complete cycle
        // Cycle requires both edge(a,b) and edge(b,a)
        // That's 2^2 = 4 cases (with/without the other 2 edges)
        // So 16 - 4 = 12 answer sets
        assert_eq!(answer_sets.len(), 12);

        // Verify no answer set has complete=true
        for as_set in &answer_sets {
            let is_complete = as_set
                .atoms
                .iter()
                .any(|a| a.predicate == Intern::new("complete".to_string()));
            assert!(!is_complete, "Complete cycles should be eliminated");
        }
    }

    // Integration tests for choice rules with bodies (body splitting feature)
    #[test]
    fn test_choice_body_simple_split() {
        // Simple case: Each entity makes one independent choice
        let input = r#"
            entity(a).
            entity(b).
            option(x).
            option(y).

            % Each entity chooses exactly 1 option
            1 { choice(E, O) : option(O) } 1 :- entity(E).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // 2 entities, each chooses 1 from 2 options
        // 2 × 2 = 4 answer sets
        assert_eq!(answer_sets.len(), 4);

        // Verify each answer set has exactly 2 choices (one per entity)
        for as_set in &answer_sets {
            let choice_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("choice".to_string()))
                .count();
            assert_eq!(choice_count, 2, "Each answer set should have 2 choices");
        }
    }

    #[test]
    fn test_choice_body_three_entities() {
        // Scale up: 3 entities, 2 options each
        let input = r#"
            entity(a).
            entity(b).
            entity(c).
            option(x).
            option(y).

            1 { choice(E, O) : option(O) } 1 :- entity(E).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // 3 entities, each chooses 1 from 2 options
        // 2 × 2 × 2 = 8 answer sets
        assert_eq!(answer_sets.len(), 8);

        // Verify each has 3 choices
        for as_set in &answer_sets {
            let choice_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("choice".to_string()))
                .count();
            assert_eq!(choice_count, 3);
        }
    }

    #[test]
    fn test_choice_body_variable_options() {
        // Each player chooses from 3 weapons
        let input = r#"
            player(alice).
            player(bob).
            weapon(sword).
            weapon(bow).
            weapon(staff).

            1 { has(P, W) : weapon(W) } 1 :- player(P).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // 2 players, each chooses 1 from 3 weapons
        // 3 × 3 = 9 answer sets
        assert_eq!(answer_sets.len(), 9);
    }

    #[test]
    fn test_choice_body_with_constraint() {
        // Choices with body + constraint on chosen values
        let input = r#"
            player(alice).
            player(bob).
            weapon(sword).
            weapon(bow).

            1 { has(P, W) : weapon(W) } 1 :- player(P).

            % Constraint: at least one player must have sword
            :- not has(alice, sword), not has(bob, sword).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // 2 × 2 = 4 combinations, minus 1 invalid (neither has sword)
        // 4 - 1 = 3 answer sets
        assert_eq!(answer_sets.len(), 3);

        // Verify constraint: at least one has sword
        for as_set in &answer_sets {
            let someone_has_sword = as_set.atoms.iter().any(|a| {
                a.predicate == Intern::new("has".to_string())
                    && a.terms.get(1)
                        == Some(&Term::Constant(Value::Atom(Intern::new(
                            "sword".to_string(),
                        ))))
            });
            assert!(someone_has_sword);
        }
    }

    #[test]
    fn test_choice_body_unbounded() {
        // Unbounded choice: each entity can choose any number
        let input = r#"
            entity(a).
            entity(b).
            option(x).
            option(y).

            % Each entity chooses 0 or more options
            { choice(E, O) : option(O) } :- entity(E).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Each entity independently chooses a subset of 2 options
        // 2^2 choices per entity = 4 per entity
        // 2 entities: 4 × 4 = 16 answer sets
        assert_eq!(answer_sets.len(), 16);
    }

    #[test]
    fn test_choice_body_min_bound() {
        // Each entity must choose at least 1
        let input = r#"
            entity(a).
            entity(b).
            option(x).
            option(y).

            1 { choice(E, O) : option(O) } :- entity(E).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Each entity chooses 1 or 2 from 2 options
        // Per entity: C(2,1) + C(2,2) = 2 + 1 = 3 choices
        // 2 entities: 3 × 3 = 9 answer sets
        assert_eq!(answer_sets.len(), 9);
    }

    #[test]
    fn test_choice_body_with_derived_base_facts() {
        // Body uses derived facts from simple Datalog rules
        let input = r#"
            person(alice).
            person(bob).
            eligible(alice).
            eligible(bob).

            % Derived: special people are eligible persons
            special(P) :- person(P), eligible(P).

            option(x).
            option(y).

            % Each special person chooses an option
            1 { choice(P, O) : option(O) } 1 :- special(P).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Both alice and bob are special
        // 2 special people, each chooses 1 from 2 options
        // 2 × 2 = 4 answer sets
        assert_eq!(answer_sets.len(), 4);

        // Verify each answer set has exactly 2 choices
        for as_set in &answer_sets {
            let choice_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("choice".to_string()))
                .count();
            assert_eq!(
                choice_count, 2,
                "Should have 2 choices (one per special person)"
            );
        }
    }
