//! Tests for optimization statements (#minimize and #maximize)
//!
//! Following TDD: Write tests first, see them fail, then implement

use crate::asp::asp_evaluation;
use crate::asp_sat::asp_sat_evaluation;
use crate::ast::{Term, Value};
use crate::parser::{ParseError, SrcId};

fn parse_program(input: &str) -> Result<crate::ast::Program, Vec<ParseError>> {
    crate::parser::parse_program(input, SrcId::empty())
}

#[cfg(test)]
mod native_optimization_tests {
    use super::*;

    #[test]
    fn test_minimize_simple_cost() {
        // Test: Find answer set that minimizes cost
        // Items have different costs, choose the cheapest
        let program_text = r#"
            % Available items with costs
            item(a, 10).
            item(b, 5).
            item(c, 15).

            % Choose exactly one item
            1 { selected(X) : item(X, C) } 1.

            % Minimize the cost
            #minimize { C : selected(X), item(X, C) }.
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");
        let answer_sets = asp_evaluation(&program);

        // Should have exactly 1 optimal answer set
        assert_eq!(
            answer_sets.len(),
            1,
            "Expected 1 optimal answer set, got {}",
            answer_sets.len()
        );

        // The optimal answer set should select item b (cost 5)
        let optimal = &answer_sets[0];
        let has_b = optimal.atoms.iter().any(|atom| {
            atom.predicate.as_str() == "selected"
                && atom.terms.len() == 1
                && matches!(&atom.terms[0], Term::Constant(Value::Atom(name)) if name.as_str() == "b")
        });
        assert!(has_b, "Expected optimal answer set to select item b");
    }

    #[test]
    fn test_maximize_simple_value() {
        // Test: Find answer set that maximizes value
        let program_text = r#"
            % Available items with values
            item(a, 10).
            item(b, 25).
            item(c, 15).

            % Choose exactly one item
            1 { selected(X) : item(X, V) } 1.

            % Maximize the value
            #maximize { V : selected(X), item(X, V) }.
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");
        let answer_sets = asp_evaluation(&program);

        // Should have exactly 1 optimal answer set
        assert_eq!(
            answer_sets.len(),
            1,
            "Expected 1 optimal answer set, got {}",
            answer_sets.len()
        );

        // The optimal answer set should select item b (value 25)
        let optimal = &answer_sets[0];
        let has_b = optimal.atoms.iter().any(|atom| {
            atom.predicate.as_str() == "selected"
                && atom.terms.len() == 1
                && matches!(&atom.terms[0], Term::Constant(Value::Atom(name)) if name.as_str() == "b")
        });
        assert!(has_b, "Expected optimal answer set to select item b");
    }

    #[test]
    fn test_minimize_with_multiple_terms() {
        // Test: Minimize sum of multiple selected items
        let program_text = r#"
            % Available items with costs
            cost(a, 10).
            cost(b, 5).
            cost(c, 8).

            % Select 0 to 3 items
            { picked(X) : cost(X, C) }.

            % Minimize total cost
            #minimize { C : picked(X), cost(X, C) }.
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");
        let answer_sets = asp_evaluation(&program);

        // Should have exactly 1 optimal answer set
        assert_eq!(
            answer_sets.len(),
            1,
            "Expected 1 optimal answer set, got {}",
            answer_sets.len()
        );

        // The optimal answer set should select nothing (cost 0)
        let optimal = &answer_sets[0];
        let picked_count = optimal
            .atoms
            .iter()
            .filter(|atom| atom.predicate.as_str() == "picked")
            .count();
        assert_eq!(
            picked_count, 0,
            "Expected optimal answer set to pick nothing (minimize cost)"
        );
    }

    #[test]
    fn test_minimize_weighted() {
        // Test: Minimize weighted sum (quantity * cost)
        let program_text = r#"
            % Item quantities and costs
            item(a, 2, 10).  % 2 items at cost 10 each = 20 total
            item(b, 1, 5).   % 1 item at cost 5 = 5 total
            item(c, 3, 8).   % 3 items at cost 8 each = 24 total

            % Choose exactly one item type
            1 { selected(X, Q, C) : item(X, Q, C) } 1.

            % Minimize total cost (quantity * cost)
            #minimize { Q*C : selected(X, Q, C), item(X, Q, C) }.
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");
        let answer_sets = asp_evaluation(&program);

        // Should have exactly 1 optimal answer set
        assert_eq!(
            answer_sets.len(),
            1,
            "Expected 1 optimal answer set, got {}",
            answer_sets.len()
        );

        // The optimal should select b (1*5 = 5, vs a=20, c=24)
        let optimal = &answer_sets[0];
        let has_b = optimal.atoms.iter().any(|atom| {
            atom.predicate.as_str() == "selected"
                && atom.terms.len() == 3
                && matches!(&atom.terms[0], Term::Constant(Value::Atom(name)) if name.as_str() == "b")
        });
        assert!(has_b, "Expected optimal answer set to select item b");
    }

    #[test]
    fn test_multiple_optimization_criteria() {
        // Test: Lexicographic optimization (minimize cost, then maximize quality)
        let program_text = r#"
            % Items with cost and quality
            item(a, 10, 5).   % cost 10, quality 5
            item(b, 10, 8).   % cost 10, quality 8 (same cost as a, better quality)
            item(c, 5, 3).    % cost 5, quality 3 (lowest cost)

            % Choose exactly one item
            1 { selected(X, Cost, Quality) : item(X, Cost, Quality) } 1.

            % Primary: minimize cost
            #minimize { Cost : selected(X, Cost, Quality), item(X, Cost, Quality) }.
            % Secondary: maximize quality (among equal-cost items)
            #maximize { Quality : selected(X, Cost, Quality), item(X, Cost, Quality) }.
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");
        let answer_sets = asp_evaluation(&program);

        // Should have exactly 1 optimal answer set
        assert_eq!(
            answer_sets.len(),
            1,
            "Expected 1 optimal answer set, got {}",
            answer_sets.len()
        );

        // The optimal should select c (lowest cost=5, even though quality is worse)
        // Lexicographic means cost is optimized first, quality second
        let optimal = &answer_sets[0];
        let has_c = optimal.atoms.iter().any(|atom| {
            atom.predicate.as_str() == "selected"
                && atom.terms.len() == 3
                && matches!(&atom.terms[0], Term::Constant(Value::Atom(name)) if name.as_str() == "c")
        });
        assert!(
            has_c,
            "Expected optimal answer set to select item c (lowest cost)"
        );
    }

    #[test]
    fn test_minimize_with_constraint() {
        // Test: Minimize while satisfying constraints
        let program_text = r#"
            % Available items with costs
            cost(a, 10).
            cost(b, 5).
            cost(c, 8).

            % Select items
            { picked(X) : cost(X, C) }.

            % Must pick at least 2 items
            :- count { X : picked(X) } < 2.

            % Minimize total cost
            #minimize { C : picked(X), cost(X, C) }.
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");
        let answer_sets = asp_evaluation(&program);

        // Should have exactly 1 optimal answer set
        assert_eq!(
            answer_sets.len(),
            1,
            "Expected 1 optimal answer set, got {}",
            answer_sets.len()
        );

        // The optimal answer set should pick b and c (cost 5+8=13)
        let optimal = &answer_sets[0];
        let has_b = optimal.atoms.iter().any(|atom| {
            atom.predicate.as_str() == "picked"
                && atom.terms.len() == 1
                && matches!(&atom.terms[0], Term::Constant(Value::Atom(name)) if name.as_str() == "b")
        });
        let has_c = optimal.atoms.iter().any(|atom| {
            atom.predicate.as_str() == "picked"
                && atom.terms.len() == 1
                && matches!(&atom.terms[0], Term::Constant(Value::Atom(name)) if name.as_str() == "c")
        });
        assert!(has_b && has_c, "Expected optimal answer set to pick b and c");
    }
}

#[cfg(test)]
mod sat_optimization_tests {
    use super::*;

    #[test]
    fn test_sat_minimize_simple_cost() {
        // Test: Find answer set that minimizes cost (using SAT solver)
        // Using ground atoms to avoid grounding issues
        let program_text = r#"
            % Choose exactly one item (ground choice rule)
            1 { picked_a; picked_b; picked_c } 1.

            % Minimize the cost (explicit costs for each choice)
            #minimize { 10 : picked_a; 5 : picked_b; 15 : picked_c }.
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");
        let answer_sets = asp_sat_evaluation(&program);

        // Should have exactly 1 optimal answer set
        assert_eq!(
            answer_sets.len(),
            1,
            "Expected 1 optimal answer set, got {}",
            answer_sets.len()
        );

        // The optimal answer set should pick b (cost 5)
        let optimal = &answer_sets[0];
        let has_b = optimal.atoms.iter().any(|atom| atom.predicate.as_str() == "picked_b");
        assert!(has_b, "Expected optimal answer set to pick_b");
    }

    #[test]
    fn test_sat_maximize_simple_value() {
        // Test: Find answer set that maximizes value (using SAT solver)
        // Using ground atoms to avoid grounding issues
        let program_text = r#"
            % Choose exactly one item (ground choice rule)
            1 { picked_a; picked_b; picked_c } 1.

            % Maximize the value (explicit values for each choice)
            #maximize { 10 : picked_a; 25 : picked_b; 15 : picked_c }.
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");
        let answer_sets = asp_sat_evaluation(&program);

        // Should have exactly 1 optimal answer set
        assert_eq!(
            answer_sets.len(),
            1,
            "Expected 1 optimal answer set, got {}",
            answer_sets.len()
        );

        // The optimal answer set should pick b (value 25)
        let optimal = &answer_sets[0];
        let has_b = optimal.atoms.iter().any(|atom| atom.predicate.as_str() == "picked_b");
        assert!(has_b, "Expected optimal answer set to pick_b");
    }

    #[test]
    fn test_sat_minimize_with_multiple_items() {
        // Test: Minimize sum of multiple selected items (using SAT solver)
        // Using ground atoms to avoid grounding issues
        let program_text = r#"
            % Select 0 to 3 items (ground choice rule)
            { picked_a; picked_b; picked_c }.

            % Minimize total cost (sum of all picked items)
            #minimize { 10 : picked_a; 5 : picked_b; 8 : picked_c }.
        "#;

        let program = parse_program(program_text).expect("Failed to parse program");
        let answer_sets = asp_sat_evaluation(&program);

        // Should have exactly 1 optimal answer set
        assert_eq!(
            answer_sets.len(),
            1,
            "Expected 1 optimal answer set, got {}",
            answer_sets.len()
        );

        // The optimal answer set should pick nothing (cost 0)
        let optimal = &answer_sets[0];
        let picked_count = optimal
            .atoms
            .iter()
            .filter(|atom| atom.predicate.as_str().starts_with("picked_"))
            .count();
        assert_eq!(
            picked_count, 0,
            "Expected optimal answer set to pick nothing (minimize cost)"
        );
    }
}
