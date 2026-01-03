use proclog_ast::{Atom, Program, Statement, Symbol, Term, Value};
use std::collections::HashMap;

/// Environment storing constant declarations
#[derive(Debug, Clone)]
pub struct ConstantEnv {
    constants: HashMap<Symbol, Value>,
}

impl Default for ConstantEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstantEnv {
    pub fn new() -> Self {
        ConstantEnv {
            constants: HashMap::new(),
        }
    }

    /// Add a constant to the environment
    pub fn define(&mut self, name: Symbol, value: Value) {
        self.constants.insert(name, value);
    }

    /// Get the value of a constant
    pub fn get(&self, name: &Symbol) -> Option<&Value> {
        self.constants.get(name)
    }

    /// Get an integer value from a constant (for backwards compatibility with code expecting integers)
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn get_int(&self, name: &Symbol) -> Option<i64> {
        self.constants.get(name).and_then(|v| match v {
            Value::Integer(n) => Some(*n),
            _ => None,
        })
    }

    /// Build a constant environment from a program's #const declarations
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn from_program(program: &Program) -> Self {
        let mut env = ConstantEnv::new();

        for statement in &program.statements {
            if let Statement::ConstDecl(const_decl) = statement {
                env.define(const_decl.name, const_decl.value.clone());
            }
        }

        env
    }

    /// Build a constant environment from a list of statements
    pub fn from_statements(statements: &[Statement]) -> Self {
        let mut env = ConstantEnv::new();

        for statement in statements {
            if let Statement::ConstDecl(const_decl) = statement {
                env.define(const_decl.name, const_decl.value.clone());
            }
        }

        env
    }

    /// Substitute constants in a term
    /// If a term is an atom (constant) that matches a const name, replace with its value
    pub fn substitute_term(&self, term: &Term) -> Term {
        match term {
            Term::Constant(Value::Atom(name)) => {
                // Check if this atom is actually a constant name
                if let Some(value) = self.get(name) {
                    Term::Constant(value.clone())
                } else {
                    term.clone()
                }
            }
            Term::Compound(functor, args) => {
                // Recursively substitute in compound terms
                let new_args: Vec<Term> =
                    args.iter().map(|arg| self.substitute_term(arg)).collect();
                Term::Compound(*functor, new_args)
            }
            _ => term.clone(),
        }
    }

    /// Substitute constants in an atom
    #[allow(dead_code)]
    pub fn substitute_atom(&self, atom: &Atom) -> Atom {
        Atom {
            predicate: atom.predicate,
            terms: atom.terms.iter().map(|t| self.substitute_term(t)).collect(),
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/constants_tests.rs"]
mod tests;
