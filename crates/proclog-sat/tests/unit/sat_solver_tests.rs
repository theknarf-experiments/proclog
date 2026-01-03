    use super::*;
    use internment::Intern;

    fn make_atom(pred: &str, args: Vec<&str>) -> Atom {
        Atom {
            predicate: Intern::new(pred.to_string()),
            terms: args
                .into_iter()
                .map(|a| Term::Constant(Value::Atom(Intern::new(a.to_string()))))
                .collect(),
        }
    }

    #[test]
    fn test_sat_solver_basic() {
        let mut solver = AspSatSolver::new();

        // Add fact: a
        let a = make_atom("a", vec![]);
        solver.add_fact(&a);

        // Solve
        let model = solver.solve().unwrap().expect("Should be SAT");

        // Check that a is true
        assert_eq!(model.get(&a), Some(&true));
    }

    #[test]
    fn test_sat_solver_constraint() {
        let mut solver = AspSatSolver::new();

        // Add facts: a, b
        let a = make_atom("a", vec![]);
        let b = make_atom("b", vec![]);
        solver.add_fact(&a);
        solver.add_fact(&b);

        // Add constraint: :- a, b (cannot have both a and b)
        solver.add_constraint(&[a.clone(), b.clone()]);

        // Solve - should be UNSAT
        let result = solver.solve().unwrap();
        assert!(result.is_none(), "Should be UNSAT due to constraint");
    }

    #[test]
    fn test_sat_solver_rule() {
        let mut solver = AspSatSolver::new();

        // Rule: c :- a, b
        let a = make_atom("a", vec![]);
        let b = make_atom("b", vec![]);
        let c = make_atom("c", vec![]);

        solver.add_fact(&a);
        solver.add_fact(&b);
        solver.add_rule(&c, &[a.clone(), b.clone()]);

        // Solve
        let model = solver.solve().unwrap().expect("Should be SAT");

        // All should be true
        assert_eq!(model.get(&a), Some(&true));
        assert_eq!(model.get(&b), Some(&true));
        assert_eq!(model.get(&c), Some(&true));
    }
