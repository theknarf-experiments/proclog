    use super::*;

    #[test]
    fn test_add_fact_and_query_variable() {
        let mut engine = ReplEngine::new();
        let response = engine.process_line("parent(john, mary).");
        assert_eq!(response.kind, ResponseKind::Info);
        assert_eq!(response.lines, vec!["Added 1 fact.".to_string()]);

        let response = engine.process_line("parent(alice, mary).");
        assert_eq!(response.kind, ResponseKind::Info);
        assert_eq!(response.lines, vec!["Added 1 fact.".to_string()]);

        let response = engine.process_line("?- parent(X, mary).");
        assert_eq!(response.kind, ResponseKind::Output);
        assert_eq!(response.lines[0], "Found 2 results.".to_string());
        assert_eq!(
            &response.lines[1..],
            &["X = alice".to_string(), "X = john".to_string()]
        );
    }

    #[test]
    fn test_repl_choice_rules() {
        let mut engine = ReplEngine::new();

        // Add facts
        let response = engine.process_line("race(human).");
        assert_eq!(response.kind, ResponseKind::Info);
        assert!(response.lines[0].contains("Added 1 fact"));

        let response = engine.process_line("race(elf).");
        assert_eq!(response.kind, ResponseKind::Info);

        // Add choice rule
        let response = engine.process_line("1 { character_race(R) : race(R) } 1.");
        assert_eq!(response.kind, ResponseKind::Info);
        assert!(response.lines[0].contains("Added 1 choice rule"));

        // Query should work and show results from answer sets
        let response = engine.process_line("?- character_race(X).");
        assert_eq!(response.kind, ResponseKind::Output);
        // Should contain results from ASP evaluation
        assert!(!response.lines.is_empty());
    }

    #[test]
    fn test_sample_command_outputs_samples() {
        let mut engine = ReplEngine::new();
        engine.process_line("item(a).");
        engine.process_line("item(b).");
        engine.process_line("{ selected(X) : item(X) }.");

        let response = engine.process_line(":sample 2 42");
        assert_eq!(response.kind, ResponseKind::Output);
        assert!(
            response.lines[0].contains("Sampled 2 answer set(s)")
                || response.lines[0].contains("Sampled 1 answer set(s)")
        );
        assert!(
            response
                .lines
                .iter()
                .skip(1)
                .any(|line| line.contains("selected(")),
            "Expected sampled answer sets to be listed"
        );
    }

    #[test]
    fn test_sample_command_invalid_arguments() {
        let mut engine = ReplEngine::new();
        let response = engine.process_line(":sample abc");
        assert_eq!(response.kind, ResponseKind::Error);
        assert!(response
            .lines
            .iter()
            .any(|line| line.contains("Invalid sample count")));
    }

    #[test]
    fn test_ground_query_true_false() {
        let mut engine = ReplEngine::new();
        engine.process_line("parent(john, mary).");

        let response = engine.process_line("?- parent(john, mary).");
        assert_eq!(response.kind, ResponseKind::Output);
        assert_eq!(response.lines, vec!["true.".to_string()]);

        let response = engine.process_line("?- parent(alice, mary).");
        assert_eq!(response.kind, ResponseKind::Output);
        assert_eq!(response.lines, vec!["false.".to_string()]);
    }

    #[test]
    fn test_rule_derivation() {
        let mut engine = ReplEngine::new();
        engine.process_line("parent(john, mary).");
        engine.process_line("parent(mary, alice).");
        engine.process_line("ancestor(X, Y) :- parent(X, Y).");
        engine.process_line("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).");

        let response = engine.process_line("?- ancestor(john, alice).");
        assert_eq!(response.kind, ResponseKind::Output);
        assert_eq!(response.lines, vec!["true.".to_string()]);

        let response = engine.process_line("?- ancestor(X, alice).");
        assert_eq!(response.kind, ResponseKind::Output);
        assert_eq!(response.lines[0], "Found 2 results.".to_string());
        assert_eq!(
            &response.lines[1..],
            &["X = john".to_string(), "X = mary".to_string()]
        );
    }

    #[test]
    fn test_constants_substitution() {
        let mut engine = ReplEngine::new();
        engine.process_line("#const width = 10.");
        engine.process_line("dimension(width).");

        let response = engine.process_line("?- dimension(10).");
        assert_eq!(response.kind, ResponseKind::Output);
        assert_eq!(response.lines, vec!["true.".to_string()]);
    }

    #[test]
    fn test_parse_error_reported() {
        let mut engine = ReplEngine::new();
        let response = engine.process_line("parent(john, mary)");
        assert_eq!(response.kind, ResponseKind::Error);
        assert_eq!(
            response.lines,
            vec![
                "Parse error at line 1, column 19: expected '.', ':-', found end of input"
                    .to_string(),
                "  parent(john, mary)".to_string(),
                "                    ^".to_string(),
            ]
        );
    }

    #[test]
    fn test_parse_error_reports_correct_line_and_column() {
        let mut engine = ReplEngine::new();
        let response = engine.process_line("parent(john, mary).\nchild(alice, bob)");
        assert_eq!(response.kind, ResponseKind::Error);
        assert!(
            response.lines[0].contains("line 2, column 18"),
            "Unexpected location: {}",
            response.lines[0]
        );
        assert_eq!(response.lines[1], "  child(alice, bob)");
        assert_eq!(response.lines[2], format!("  {}^", " ".repeat(17)));
    }
