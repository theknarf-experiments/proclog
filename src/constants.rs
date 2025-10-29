use crate::ast::{Atom, Program, Statement, Symbol, Term, Value};
use internment::Intern;
use std::collections::HashMap;

/// Environment storing constant declarations
#[derive(Debug, Clone)]
pub struct ConstantEnv {
    constants: HashMap<Symbol, i64>,
}

impl ConstantEnv {
    pub fn new() -> Self {
        ConstantEnv {
            constants: HashMap::new(),
        }
    }

    /// Add a constant to the environment
    pub fn define(&mut self, name: Symbol, value: i64) {
        self.constants.insert(name, value);
    }

    /// Get the value of a constant
    pub fn get(&self, name: &Symbol) -> Option<i64> {
        self.constants.get(name).copied()
    }

    /// Build a constant environment from a program's #const declarations
    pub fn from_program(program: &Program) -> Self {
        let mut env = ConstantEnv::new();

        for statement in &program.statements {
            if let Statement::ConstDecl(const_decl) = statement {
                env.define(const_decl.name.clone(), const_decl.value);
            }
        }

        env
    }

    /// Substitute constants in a term
    /// If a term is an atom (constant) that matches a const name, replace with integer
    pub fn substitute_term(&self, term: &Term) -> Term {
        match term {
            Term::Constant(Value::Atom(name)) => {
                // Check if this atom is actually a constant name
                if let Some(value) = self.get(name) {
                    Term::Constant(Value::Integer(value))
                } else {
                    term.clone()
                }
            }
            Term::Compound(functor, args) => {
                // Recursively substitute in compound terms
                let new_args: Vec<Term> =
                    args.iter().map(|arg| self.substitute_term(arg)).collect();
                Term::Compound(functor.clone(), new_args)
            }
            _ => term.clone(),
        }
    }

    /// Substitute constants in an atom
    pub fn substitute_atom(&self, atom: &Atom) -> Atom {
        Atom {
            predicate: atom.predicate.clone(),
            terms: atom.terms.iter().map(|t| self.substitute_term(t)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ConstDecl;

    fn make_const_decl(name: &str, value: i64) -> ConstDecl {
        ConstDecl {
            name: Intern::new(name.to_string()),
            value,
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
        env.define(Intern::new("width".to_string()), 10);
        env.define(Intern::new("height".to_string()), 20);

        assert_eq!(env.get(&Intern::new("width".to_string())), Some(10));
        assert_eq!(env.get(&Intern::new("height".to_string())), Some(20));
        assert_eq!(env.get(&Intern::new("depth".to_string())), None);
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
        assert_eq!(env.get(&Intern::new("width".to_string())), Some(10));
        assert_eq!(env.get(&Intern::new("height".to_string())), Some(20));
    }

    #[test]
    fn test_substitute_term_atom_to_int() {
        let mut env = ConstantEnv::new();
        env.define(Intern::new("max_enemies".to_string()), 5);

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
        env.define(Intern::new("width".to_string()), 10);

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
        env.define(Intern::new("max_count".to_string()), 100);

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
}
