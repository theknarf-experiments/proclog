//! Tests for aggregate functions (count, sum, min, max)
//!
//! Following TDD: Start with one test, get it passing, then add more

use internment::Intern;
use proclog_ast::*;

#[cfg(test)]
mod count_aggregate_tests {
    use super::*;
    use proclog_parser::{ParseError, SrcId};

    fn parse_program(input: &str) -> Result<Program, Vec<ParseError>> {
        proclog_parser::parse_program(input, SrcId::empty())
    }

    // Helper to create aggregate atom: count { X : predicate(X) } op value
    fn make_count_aggregate(
        variables: Vec<&str>,
        condition: Vec<Literal>,
        op: ComparisonOp,
        value: i64,
    ) -> Literal {
        Literal::Aggregate(AggregateAtom {
            function: AggregateFunction::Count,
            variables: variables
                .into_iter()
                .map(|v| Intern::new(v.to_string()))
                .collect(),
            elements: condition,
            comparison: op,
            value: Term::Constant(Value::Integer(value)),
        })
    }

    // Helper to create an atom
    fn make_atom(pred: &str, terms: Vec<Term>) -> Atom {
        Atom {
            predicate: Intern::new(pred.to_string()),
            terms,
        }
    }

    #[test]
    fn test_count_constraint_parsed_from_text() {
        // Test: Parse and evaluate count aggregate in constraint
        let program_text = r#"
            item(a).
            item(b).
            item(c).

            { selected(X) : item(X) }.

            :- count { X : selected(X) } > 2.
        "#;

        let program =
            parse_program(program_text).expect("Failed to parse program with count aggregate");
        let answer_sets = crate::asp_evaluation(&program);

        // Without constraint: 2^3 = 8 answer sets
        // With constraint: eliminate those with 3 items = 7 answer sets
        assert_eq!(
            answer_sets.len(),
            7,
            "Should have 7 answer sets (all except the one with 3 selected)"
        );

        // Verify no answer set has more than 2 selected items
        for as_set in &answer_sets {
            let selected_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("selected".to_string()))
                .count();
            assert!(
                selected_count <= 2,
                "No answer set should have more than 2 selected items"
            );
        }
    }

    #[test]
    fn test_count_constraint_filters_answer_sets() {
        // Test: count aggregate in constraint
        // Count number of selected items, reject if > 2

        // Manually build program (alternative to parsing)
        let mut program = Program::new();

        // Facts: item(a), item(b), item(c)
        for item in ["a", "b", "c"] {
            program.add_statement(Statement::Fact(Fact {
                atom: make_atom(
                    "item",
                    vec![Term::Constant(Value::Atom(Intern::new(item.to_string())))],
                ),
            }));
        }

        // Choice rule: { selected(X) : item(X) }
        program.add_statement(Statement::ChoiceRule(ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: make_atom(
                    "selected",
                    vec![Term::Variable(Intern::new("X".to_string()))],
                ),
                condition: vec![Literal::Positive(make_atom(
                    "item",
                    vec![Term::Variable(Intern::new("X".to_string()))],
                ))],
            }],
            body: vec![],
        }));

        // Constraint: :- count { X : selected(X) } > 2
        let selected_x = make_atom(
            "selected",
            vec![Term::Variable(Intern::new("X".to_string()))],
        );
        let count_agg = make_count_aggregate(
            vec!["X"],
            vec![Literal::Positive(selected_x)],
            ComparisonOp::GreaterThan,
            2,
        );
        program.add_statement(Statement::Constraint(Constraint {
            body: vec![count_agg],
        }));

        let answer_sets = crate::asp_evaluation(&program);

        // Without constraint: 2^3 = 8 answer sets
        // With constraint: eliminate those with 3 items = 7 answer sets
        assert_eq!(
            answer_sets.len(),
            7,
            "Should have 7 answer sets (all except the one with 3 selected)"
        );

        // Verify no answer set has more than 2 selected items
        for as_set in &answer_sets {
            let selected_count = as_set
                .atoms
                .iter()
                .filter(|a| a.predicate == Intern::new("selected".to_string()))
                .count();
            assert!(
                selected_count <= 2,
                "No answer set should have more than 2 selected items"
            );
        }
    }
}
