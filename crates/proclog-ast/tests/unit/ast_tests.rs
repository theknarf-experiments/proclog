use super::*;

#[test]
fn test_atom_equality() {
    let atom1 = Atom {
        predicate: Intern::new("parent".to_string()),
        terms: vec![
            Term::Constant(Value::Atom(Intern::new("john".to_string()))),
            Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
        ],
    };

    let atom2 = Atom {
        predicate: Intern::new("parent".to_string()),
        terms: vec![
            Term::Constant(Value::Atom(Intern::new("john".to_string()))),
            Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
        ],
    };

    assert_eq!(atom1, atom2);
}

#[test]
fn test_literal_methods() {
    let atom = Atom {
        predicate: Intern::new("test".to_string()),
        terms: vec![],
    };

    let pos = Literal::Positive(atom.clone());
    let neg = Literal::Negative(atom.clone());

    assert!(pos.is_positive());
    assert!(!pos.is_negative());
    assert!(!neg.is_positive());
    assert!(neg.is_negative());
    assert_eq!(pos.atom(), Some(&atom));
    assert_eq!(neg.atom(), Some(&atom));
}

#[test]
fn test_program_construction() {
    let mut program = Program::new();
    assert_eq!(program.statements.len(), 0);

    let fact = Statement::Fact(Fact {
        atom: Atom {
            predicate: Intern::new("test".to_string()),
            terms: vec![],
        },
    });

    program.add_statement(fact);
    assert_eq!(program.statements.len(), 1);
}
