//! Integration tests for constant names in choice rule cardinality bounds
//!
//! These tests verify that constant names can be used in place of integer literals
//! for the lower and upper bounds of choice rules, and that they are properly
//! resolved during evaluation.

#[cfg(test)]
mod choice_constant_bounds_tests {
    use crate::asp::asp_evaluation;
    use crate::parser::parse_program;

    #[test]
    fn test_constant_lower_bound() {
        let program_text = r#"
            #const min_items = 2.

            item(a).
            item(b).
            item(c).

            min_items { selected(X) : item(X) }.
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");
        let answer_sets = asp_evaluation(&program);

        // Should generate answer sets with at least 2 items
        assert!(answer_sets.len() > 0, "Should generate answer sets");

        // All answer sets should have at least 2 selected items
        for answer_set in &answer_sets {
            let selected_count = answer_set
                .atoms
                .iter()
                .filter(|atom| atom.predicate.as_ref() == "selected")
                .count();
            assert!(
                selected_count >= 2,
                "Should have at least 2 selected items, got {}",
                selected_count
            );
        }
    }

    #[test]
    fn test_constant_upper_bound() {
        let program_text = r#"
            #const max_items = 2.

            item(a).
            item(b).
            item(c).

            { selected(X) : item(X) } max_items.
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");
        let answer_sets = asp_evaluation(&program);

        // Should generate answer sets with at most 2 items
        assert!(answer_sets.len() > 0, "Should generate answer sets");

        // All answer sets should have at most 2 selected items
        for answer_set in &answer_sets {
            let selected_count = answer_set
                .atoms
                .iter()
                .filter(|atom| atom.predicate.as_ref() == "selected")
                .count();
            assert!(
                selected_count <= 2,
                "Should have at most 2 selected items, got {}",
                selected_count
            );
        }
    }

    #[test]
    fn test_constant_both_bounds() {
        let program_text = r#"
            #const min_sel = 1.
            #const max_sel = 2.

            option(x).
            option(y).
            option(z).

            min_sel { chosen(O) : option(O) } max_sel.
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");
        let answer_sets = asp_evaluation(&program);

        // Should generate answer sets with 1 or 2 items
        assert!(answer_sets.len() > 0, "Should generate answer sets");

        // All answer sets should have between 1 and 2 chosen items
        for answer_set in &answer_sets {
            let chosen_count = answer_set
                .atoms
                .iter()
                .filter(|atom| atom.predicate.as_ref() == "chosen")
                .count();
            assert!(
                chosen_count >= 1 && chosen_count <= 2,
                "Should have 1-2 chosen items, got {}",
                chosen_count
            );
        }

        // Should have exactly 6 answer sets: C(3,1) + C(3,2) = 3 + 3 = 6
        assert_eq!(answer_sets.len(), 6, "Should have 6 answer sets");
    }

    #[test]
    fn test_mixed_constant_and_integer_bounds() {
        let program_text = r#"
            #const max_weapons = 2.

            weapon(sword).
            weapon(bow).
            weapon(axe).

            1 { has(W) : weapon(W) } max_weapons.
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");
        let answer_sets = asp_evaluation(&program);

        // Should generate answer sets with 1 or 2 weapons
        assert!(answer_sets.len() > 0, "Should generate answer sets");

        for answer_set in &answer_sets {
            let weapon_count = answer_set
                .atoms
                .iter()
                .filter(|atom| atom.predicate.as_ref() == "has")
                .count();
            assert!(
                weapon_count >= 1 && weapon_count <= 2,
                "Should have 1-2 weapons, got {}",
                weapon_count
            );
        }

        // Should have 6 answer sets: C(3,1) + C(3,2) = 3 + 3 = 6
        assert_eq!(answer_sets.len(), 6);
    }

    #[test]
    fn test_constant_exact_bound() {
        let program_text = r#"
            #const exactly = 2.

            color(red).
            color(blue).
            color(green).

            exactly { pick(C) : color(C) } exactly.
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");
        let answer_sets = asp_evaluation(&program);

        // Should have exactly 2 colors in each answer set
        for answer_set in &answer_sets {
            let color_count = answer_set
                .atoms
                .iter()
                .filter(|atom| atom.predicate.as_ref() == "pick")
                .count();
            assert_eq!(
                color_count, 2,
                "Should have exactly 2 colors, got {}",
                color_count
            );
        }

        // Should have C(3,2) = 3 answer sets
        assert_eq!(answer_sets.len(), 3);
    }

    #[test]
    fn test_constant_with_choice_body() {
        let program_text = r#"
            #const weapons_per_player = 1.

            player(alice).
            player(bob).

            weapon(sword).
            weapon(bow).

            weapons_per_player { has_weapon(P, W) : weapon(W) } weapons_per_player :- player(P).
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");
        let answer_sets = asp_evaluation(&program);

        // Each player should have exactly 1 weapon
        for answer_set in &answer_sets {
            let alice_weapons = answer_set
                .atoms
                .iter()
                .filter(|atom| {
                    atom.predicate.as_ref() == "has_weapon"
                        && atom.terms.len() >= 1
                        && format!("{:?}", atom.terms[0]).contains("alice")
                })
                .count();
            let bob_weapons = answer_set
                .atoms
                .iter()
                .filter(|atom| {
                    atom.predicate.as_ref() == "has_weapon"
                        && atom.terms.len() >= 1
                        && format!("{:?}", atom.terms[0]).contains("bob")
                })
                .count();

            assert_eq!(alice_weapons, 1, "Alice should have exactly 1 weapon");
            assert_eq!(bob_weapons, 1, "Bob should have exactly 1 weapon");
        }

        // Should have 2×2 = 4 answer sets (each player independently chooses 1 weapon)
        assert_eq!(answer_sets.len(), 4);
    }
}
