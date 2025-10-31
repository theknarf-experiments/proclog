// Tests for multiple independent choice rules (cartesian product behavior)

use crate::asp::asp_evaluation;
use crate::ast::{Term, Value};
use crate::parser::parse_program;
use internment::Intern;

#[cfg(test)]
mod multiple_choice_tests {
    use super::*;

    #[test]
    fn test_two_independent_choices_simple() {
        // Two independent choice rules should produce cartesian product
        let input = r#"
            color(red).
            color(blue).

            size(small).
            size(large).

            % Two independent choices - each must select exactly 1
            1 { selected_color(C) : color(C) } 1.
            1 { selected_size(S) : size(S) } 1.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Expected: 2 colors × 2 sizes = 4 answer sets
        assert_eq!(
            answer_sets.len(),
            4,
            "Should have 4 answer sets (2×2 cartesian product)"
        );

        // Verify all combinations exist
        let combinations = [
            ("red", "small"),
            ("red", "large"),
            ("blue", "small"),
            ("blue", "large"),
        ];

        for (expected_color, expected_size) in &combinations {
            let found = answer_sets.iter().any(|as_set| {
                let has_color = as_set.atoms.iter().any(|a| {
                    a.predicate == Intern::new("selected_color".to_string())
                        && a.terms.get(0)
                            == Some(&Term::Constant(Value::Atom(Intern::new(
                                expected_color.to_string(),
                            ))))
                });
                let has_size = as_set.atoms.iter().any(|a| {
                    a.predicate == Intern::new("selected_size".to_string())
                        && a.terms.get(0)
                            == Some(&Term::Constant(Value::Atom(Intern::new(
                                expected_size.to_string(),
                            ))))
                });
                has_color && has_size
            });
            assert!(
                found,
                "Should have combination ({}, {})",
                expected_color, expected_size
            );
        }
    }

    #[test]
    fn test_three_independent_choices() {
        // Three independent choice rules
        let input = r#"
            color(red).
            color(blue).

            size(small).
            size(large).

            material(wood).
            material(metal).

            1 { c(C) : color(C) } 1.
            1 { s(S) : size(S) } 1.
            1 { m(M) : material(M) } 1.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // Expected: 2 × 2 × 2 = 8 answer sets
        assert_eq!(answer_sets.len(), 8, "Should have 8 answer sets (2×2×2)");

        // Each answer set should have exactly 1 of each choice
        for as_set in &answer_sets {
            let color_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("c".to_string()))
                .count();
            let size_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("s".to_string()))
                .count();
            let material_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("m".to_string()))
                .count();

            assert_eq!(
                color_count, 1,
                "Each answer set should select exactly 1 color"
            );
            assert_eq!(
                size_count, 1,
                "Each answer set should select exactly 1 size"
            );
            assert_eq!(
                material_count, 1,
                "Each answer set should select exactly 1 material"
            );
        }
    }

    #[test]
    fn test_choice_with_different_cardinalities() {
        // Choice rules with different cardinality bounds
        let input = r#"
            item(a).
            item(b).
            item(c).

            bonus(x).
            bonus(y).

            % Choose exactly 2 items
            2 { selected_item(I) : item(I) } 2.

            % Choose 0 or 1 bonus
            0 { selected_bonus(B) : bonus(B) } 1.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // First choice: C(3,2) = 3 ways
        // Second choice: 0 or 1 from 2 items = 1 (empty) + 2 (single) = 3 ways
        // Total: 3 × 3 = 9 answer sets
        assert_eq!(answer_sets.len(), 9, "Should have 9 answer sets (3×3)");
    }

    #[test]
    fn test_unbounded_choice_with_bounded_choice() {
        // Mix of unbounded and bounded choices
        let input = r#"
            tag(fast).
            tag(slow).

            mode(easy).
            mode(hard).

            % Unbounded: any subset of tags (2^2 = 4)
            { has_tag(T) : tag(T) }.

            % Bounded: exactly 1 mode
            1 { has_mode(M) : mode(M) } 1.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // First choice: 2^2 = 4 subsets (including empty)
        // Second choice: 2 modes
        // Total: 4 × 2 = 8 answer sets
        assert_eq!(answer_sets.len(), 8, "Should have 8 answer sets (4×2)");
    }

    #[test]
    fn test_choice_rules_with_constraints() {
        // Multiple choices with constraint filtering
        let input = r#"
            color(red).
            color(blue).

            size(small).
            size(large).

            1 { selected_color(C) : color(C) } 1.
            1 { selected_size(S) : size(S) } 1.

            % Constraint: cannot have red + large
            :- selected_color(red), selected_size(large).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // 2 × 2 = 4 combinations, minus 1 forbidden = 3 answer sets
        assert_eq!(answer_sets.len(), 3, "Should have 3 answer sets (4-1)");

        // Verify red+large doesn't exist
        let has_red_large = answer_sets.iter().any(|as_set| {
            let has_red = as_set.atoms.iter().any(|a| {
                a.predicate == Intern::new("selected_color".to_string())
                    && a.terms.get(0)
                        == Some(&Term::Constant(Value::Atom(Intern::new("red".to_string()))))
            });
            let has_large = as_set.atoms.iter().any(|a| {
                a.predicate == Intern::new("selected_size".to_string())
                    && a.terms.get(0)
                        == Some(&Term::Constant(Value::Atom(Intern::new(
                            "large".to_string(),
                        ))))
            });
            has_red && has_large
        });
        assert!(!has_red_large, "Red+large should be filtered out");
    }

    #[test]
    fn test_cascading_multiple_choices() {
        // Multiple independent choices with derived facts between them
        // Note: In standard ASP, choice rules are grounded before choices are made
        // So we need base facts for both choices
        let input = r#"
            location(a).
            location(b).

            item(sword).
            item(shield).

            % First choice: select locations with rooms
            1 { has_room(L) : location(L) } 2.

            % Second choice: select items (independent)
            { has_item(I) : item(I) }.

            % Rules derive facts based on choices
            room_count(2) :- has_room(a), has_room(b).
            room_count(1) :- has_room(a), not has_room(b).
            room_count(1) :- has_room(b), not has_room(a).
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // First choice: C(2,1) + C(2,2) = 2 + 1 = 3 ways
        // Second choice: 2^2 = 4 ways (empty, {sword}, {shield}, {both})
        // Total: 3 × 4 = 12 answer sets
        assert_eq!(answer_sets.len(), 12, "Should have 12 answer sets (3×4)");

        // Verify semantic correctness: room_count is derived correctly
        for as_set in &answer_sets {
            let rooms: Vec<_> = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("has_room".to_string()))
                .collect();
            let room_count: Option<i64> = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("room_count".to_string()))
                .filter_map(|a| {
                    if let Some(Term::Constant(Value::Integer(n))) = a.terms.get(0) {
                        Some(*n)
                    } else {
                        None
                    }
                })
                .next();

            assert_eq!(
                room_count,
                Some(rooms.len() as i64),
                "room_count should match actual room count"
            );
        }
    }

    #[test]
    fn test_single_choice_rule() {
        // Edge case: single choice rule (no cartesian product needed)
        let input = r#"
            item(a).
            item(b).
            item(c).

            1 { selected(I) : item(I) } 2.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // C(3,1) + C(3,2) = 3 + 3 = 6 answer sets
        assert_eq!(answer_sets.len(), 6, "Should have 6 answer sets");
    }

    #[test]
    fn test_empty_choice_in_cartesian_product() {
        // One choice rule produces empty selection
        let input = r#"
            item(a).

            tag(x).
            tag(y).

            % Must select the item
            1 { selected(I) : item(I) } 1.

            % Optional tags (can select none)
            { has_tag(T) : tag(T) }.
        "#;

        let program = parse_program(input).unwrap();
        let answer_sets = asp_evaluation(&program);

        // First choice: 1 way (must select 'a')
        // Second choice: 2^2 = 4 ways (empty, {x}, {y}, {x,y})
        // Total: 1 × 4 = 4 answer sets
        assert_eq!(answer_sets.len(), 4, "Should have 4 answer sets (1×4)");

        // Verify one has just selected(a) with no tags
        let has_empty_tags = answer_sets.iter().any(|as_set| {
            let has_selected = as_set
                .atoms
                .iter()
                .any(|a| a.predicate == Intern::new("selected".to_string()));
            let tag_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("has_tag".to_string()))
                .count();
            has_selected && tag_count == 0
        });
        assert!(has_empty_tags, "Should have answer set with no tags");
    }
}
