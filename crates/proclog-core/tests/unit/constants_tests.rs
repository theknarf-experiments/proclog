use super::*;
use internment::Intern;
use proclog_ast::ConstDecl;

fn make_const_decl(name: &str, value: i64) -> ConstDecl {
    ConstDecl {
        name: Intern::new(name.to_string()),
        value: Value::Integer(value),
    }
}

fn atom_term(name: &str) -> Term {
    Term::Constant(Value::Atom(Intern::new(name.to_string())))
}

fn int_term(value: i64) -> Term {
    Term::Constant(Value::Integer(value))
}

#[test]
fn test_constant_env_new() {
    let env = ConstantEnv::new();
    assert!(env.get(&Intern::new("width".to_string())).is_none());
}

#[test]
fn test_constant_env_define_and_get() {
    let mut env = ConstantEnv::new();
    env.define(Intern::new("width".to_string()), Value::Integer(10));
    env.define(Intern::new("height".to_string()), Value::Integer(20));

    assert_eq!(env.get_int(&Intern::new("width".to_string())), Some(10));
    assert_eq!(env.get_int(&Intern::new("height".to_string())), Some(20));
    assert_eq!(env.get_int(&Intern::new("depth".to_string())), None);
}

#[test]
fn test_constant_env_from_program() {
    let program = Program {
        statements: vec![
            Statement::ConstDecl(make_const_decl("width", 10)),
            Statement::ConstDecl(make_const_decl("height", 20)),
        ],
    };

    let env = ConstantEnv::from_program(&program);
    assert_eq!(env.get_int(&Intern::new("width".to_string())), Some(10));
    assert_eq!(env.get_int(&Intern::new("height".to_string())), Some(20));
}

#[test]
fn test_substitute_term_atom_to_int() {
    let mut env = ConstantEnv::new();
    env.define(Intern::new("max_enemies".to_string()), Value::Integer(5));

    let term = atom_term("max_enemies");
    let result = env.substitute_term(&term);

    assert_eq!(result, int_term(5));
}

#[test]
fn test_substitute_term_no_match() {
    let env = ConstantEnv::new();

    let term = atom_term("regular_atom");
    let result = env.substitute_term(&term);

    assert_eq!(result, atom_term("regular_atom"));
}

#[test]
fn test_substitute_term_in_compound() {
    let mut env = ConstantEnv::new();
    env.define(Intern::new("width".to_string()), Value::Integer(10));

    let term = Term::Compound(
        Intern::new("dims".to_string()),
        vec![atom_term("width"), int_term(20)],
    );

    let result = env.substitute_term(&term);

    let expected = Term::Compound(
        Intern::new("dims".to_string()),
        vec![int_term(10), int_term(20)],
    );

    assert_eq!(result, expected);
}

#[test]
fn test_substitute_atom() {
    let mut env = ConstantEnv::new();
    env.define(Intern::new("max_count".to_string()), Value::Integer(100));

    let atom = Atom {
        predicate: Intern::new("count".to_string()),
        terms: vec![atom_term("max_count")],
    };

    let result = env.substitute_atom(&atom);

    let expected = Atom {
        predicate: Intern::new("count".to_string()),
        terms: vec![int_term(100)],
    };

    assert_eq!(result, expected);
}

// Tests for non-integer constant types
#[test]
#[allow(clippy::approx_constant)]
fn test_substitute_term_with_float() {
    let mut env = ConstantEnv::new();
    env.define(Intern::new("pi".to_string()), Value::Float(3.14));

    let term = atom_term("pi");
    let result = env.substitute_term(&term);

    assert_eq!(result, Term::Constant(Value::Float(3.14)));
}

#[test]
fn test_substitute_term_with_boolean() {
    let mut env = ConstantEnv::new();
    env.define(Intern::new("enabled".to_string()), Value::Boolean(true));

    let term = atom_term("enabled");
    let result = env.substitute_term(&term);

    assert_eq!(result, Term::Constant(Value::Boolean(true)));
}

#[test]
fn test_substitute_term_with_string() {
    let mut env = ConstantEnv::new();
    env.define(
        Intern::new("message".to_string()),
        Value::String(Intern::new("hello".to_string())),
    );

    let term = atom_term("message");
    let result = env.substitute_term(&term);

    assert_eq!(
        result,
        Term::Constant(Value::String(Intern::new("hello".to_string())))
    );
}

#[test]
fn test_substitute_term_with_atom_value() {
    let mut env = ConstantEnv::new();
    env.define(
        Intern::new("color".to_string()),
        Value::Atom(Intern::new("red".to_string())),
    );

    let term = atom_term("color");
    let result = env.substitute_term(&term);

    assert_eq!(
        result,
        Term::Constant(Value::Atom(Intern::new("red".to_string())))
    );
}

// Integration tests for constants with different types
#[test]
fn test_integration_multiple_const_types_in_atoms() {
    let mut env = ConstantEnv::new();
    env.define(Intern::new("max_health".to_string()), Value::Integer(100));
    env.define(
        Intern::new("damage_multiplier".to_string()),
        Value::Float(1.5),
    );
    env.define(Intern::new("is_enabled".to_string()), Value::Boolean(true));
    env.define(
        Intern::new("default_weapon".to_string()),
        Value::Atom(Intern::new("sword".to_string())),
    );

    // Create an atom with multiple constant references
    let atom = Atom {
        predicate: Intern::new("config".to_string()),
        terms: vec![
            atom_term("max_health"),
            atom_term("damage_multiplier"),
            atom_term("is_enabled"),
            atom_term("default_weapon"),
        ],
    };

    let result = env.substitute_atom(&atom);

    let expected = Atom {
        predicate: Intern::new("config".to_string()),
        terms: vec![
            int_term(100),
            Term::Constant(Value::Float(1.5)),
            Term::Constant(Value::Boolean(true)),
            Term::Constant(Value::Atom(Intern::new("sword".to_string()))),
        ],
    };

    assert_eq!(result, expected);
}

#[test]
fn test_integration_constants_in_nested_compounds() {
    let mut env = ConstantEnv::new();
    env.define(Intern::new("width".to_string()), Value::Integer(10));
    env.define(Intern::new("height".to_string()), Value::Integer(20));
    env.define(Intern::new("scale".to_string()), Value::Float(2.0));

    // Create a nested compound term with constants
    let term = Term::Compound(
        Intern::new("rectangle".to_string()),
        vec![
            Term::Compound(
                Intern::new("dimensions".to_string()),
                vec![atom_term("width"), atom_term("height")],
            ),
            atom_term("scale"),
        ],
    );

    let result = env.substitute_term(&term);

    let expected = Term::Compound(
        Intern::new("rectangle".to_string()),
        vec![
            Term::Compound(
                Intern::new("dimensions".to_string()),
                vec![int_term(10), int_term(20)],
            ),
            Term::Constant(Value::Float(2.0)),
        ],
    );

    assert_eq!(result, expected);
}

#[test]
fn test_integration_mixed_constants_and_regular_values() {
    let mut env = ConstantEnv::new();
    env.define(
        Intern::new("default_port".to_string()),
        Value::Integer(8080),
    );
    env.define(
        Intern::new("default_host".to_string()),
        Value::String(Intern::new("localhost".to_string())),
    );

    // Create an atom mixing constants and regular values
    let atom = Atom {
        predicate: Intern::new("server".to_string()),
        terms: vec![
            atom_term("default_host"),
            atom_term("default_port"),
            Term::Constant(Value::Boolean(true)), // Regular value, not a constant
            Term::Constant(Value::Atom(Intern::new("http".to_string()))), // Regular atom
        ],
    };

    let result = env.substitute_atom(&atom);

    let expected = Atom {
        predicate: Intern::new("server".to_string()),
        terms: vec![
            Term::Constant(Value::String(Intern::new("localhost".to_string()))),
            int_term(8080),
            Term::Constant(Value::Boolean(true)),
            Term::Constant(Value::Atom(Intern::new("http".to_string()))),
        ],
    };

    assert_eq!(result, expected);
}

#[test]
fn test_integration_from_program_with_all_types() {
    let program = Program {
        statements: vec![
            Statement::ConstDecl(ConstDecl {
                name: Intern::new("max_players".to_string()),
                value: Value::Integer(4),
            }),
            Statement::ConstDecl(ConstDecl {
                name: Intern::new("gravity".to_string()),
                value: Value::Float(9.81),
            }),
            Statement::ConstDecl(ConstDecl {
                name: Intern::new("debug_mode".to_string()),
                value: Value::Boolean(false),
            }),
            Statement::ConstDecl(ConstDecl {
                name: Intern::new("game_name".to_string()),
                value: Value::String(Intern::new("Adventure".to_string())),
            }),
            Statement::ConstDecl(ConstDecl {
                name: Intern::new("difficulty".to_string()),
                value: Value::Atom(Intern::new("hard".to_string())),
            }),
        ],
    };

    let env = ConstantEnv::from_program(&program);

    assert_eq!(
        env.get_int(&Intern::new("max_players".to_string())),
        Some(4)
    );
    assert_eq!(
        env.get(&Intern::new("gravity".to_string())),
        Some(&Value::Float(9.81))
    );
    assert_eq!(
        env.get(&Intern::new("debug_mode".to_string())),
        Some(&Value::Boolean(false))
    );
    assert_eq!(
        env.get(&Intern::new("game_name".to_string())),
        Some(&Value::String(Intern::new("Adventure".to_string())))
    );
    assert_eq!(
        env.get(&Intern::new("difficulty".to_string())),
        Some(&Value::Atom(Intern::new("hard".to_string())))
    );
}
