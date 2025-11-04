// Abstract Syntax Tree (AST) definitions for ProcLog
//!
//! This module defines the core data structures representing a ProcLog program.
//! ProcLog combines features from Datalog, Answer Set Programming (ASP), and declarative design.
//!
//! # Key Components
//!
//! - **Program**: A collection of statements (facts, rules, constraints, etc.)
//! - **Statement**: Top-level constructs (facts, rules, constraints, const decls, choice rules)
//! - **Atom**: Predicate applied to terms (e.g., `parent(john, mary)`)
//! - **Term**: Variables, constants, compound terms, or ranges
//! - **Value**: Constant values (integers, floats, booleans, strings, atoms)
//! - **Literal**: Positive or negative atoms
//!
//! # Example
//!
//! ```ignore
//! // A simple fact: parent(john, mary).
//! let fact = Fact {
//!     atom: Atom {
//!         predicate: Intern::new("parent".to_string()),
//!         terms: vec![
//!             Term::Constant(Value::Atom(Intern::new("john".to_string()))),
//!             Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
//!         ],
//!     },
//! };
//! ```

use internment::Intern;

/// Interned string for efficient storage and comparison
pub type Symbol = Intern<String>;

/// A ProcLog program consists of facts, rules, constraints, and other statements
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Program {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Statement {
    Fact(Fact),
    Rule(Rule),
    Constraint(Constraint),
    ConstDecl(ConstDecl),
    ChoiceRule(ChoiceRule),
    Test(TestBlock),
    Optimize(OptimizeStatement),
}

/// A fact is simply an atom: `parent(john, mary).`
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fact {
    pub atom: Atom,
}

/// A rule has a head and a body: `ancestor(X, Y) :- parent(X, Y).`
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rule {
    pub head: Atom,
    pub body: Vec<Literal>,
}

/// A constraint has no head, only a body: `:- unsafe(X).`
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Constraint {
    pub body: Vec<Literal>,
}

/// A constant declaration: `#const width = 10.` or `#const pi = 3.14.` or `#const name = foo.`
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstDecl {
    pub name: Symbol,
    pub value: Value,
}

/// A choice rule: `{ atom1; atom2 }` or `1 { atom1; atom2 } 3`
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChoiceRule {
    pub lower_bound: Option<Term>, // None means 0, can be integer or constant name
    pub upper_bound: Option<Term>, // None means infinity, can be integer or constant name
    pub elements: Vec<ChoiceElement>,
    pub body: Vec<Literal>, // Optional body like regular rules
}

/// An element in a choice rule
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChoiceElement {
    pub atom: Atom,
    pub condition: Vec<Literal>, // Optional condition after ':'
}

/// A query: `?- parent(X, mary).` or `?- ancestor(X, Y), parent(Y, mary).`
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Query {
    pub body: Vec<Literal>,
}

/// A test block: `#test "name" { facts, rules, queries with assertions }`
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TestBlock {
    pub name: String,
    pub statements: Vec<Statement>, // Facts, rules, const decls for this test
    pub test_cases: Vec<TestCase>,
}

/// A test case: query with positive and negative assertions
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TestCase {
    pub query: Query,
    pub positive_assertions: Vec<Atom>, // These should be in results
    pub negative_assertions: Vec<Atom>, // These should NOT be in results
}

/// An optimization statement: `#minimize { expression }` or `#maximize { expression }`
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptimizeStatement {
    pub direction: OptimizeDirection,
    pub terms: Vec<OptimizeTerm>, // Terms to optimize (can be weighted)
}

/// Optimization direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OptimizeDirection {
    Minimize,
    Maximize,
}

/// A term in an optimization statement with optional weight
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptimizeTerm {
    pub weight: Option<Term>, // Optional weight (defaults to 1)
    pub term: Term,           // The term being optimized
    pub condition: Vec<Literal>, // Optional condition
}

/// A literal is either a positive or negative atom, or an aggregate
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Literal {
    Positive(Atom),
    Negative(Atom),
    Aggregate(AggregateAtom),
}

/// An aggregate literal: count { X : predicate(X) } >= 2
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AggregateAtom {
    pub function: AggregateFunction,
    pub variables: Vec<Symbol>,        // Variables used in aggregate (X, Y, etc.)
    pub elements: Vec<Literal>,        // Condition literals
    pub comparison: ComparisonOp,
    pub value: Term,                   // Right-hand side value to compare against
}

/// Aggregate functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AggregateFunction {
    Count,
    // Future: Sum, Min, Max
}

/// Comparison operators for aggregates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ComparisonOp {
    Equal,         // =, ==
    NotEqual,      // !=
    LessThan,      // <
    LessOrEqual,   // <=
    GreaterThan,   // >
    GreaterOrEqual, // >=
}

/// An atom is a predicate applied to terms: `parent(john, mary)`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Atom {
    pub predicate: Symbol,
    pub terms: Vec<Term>,
}

/// A term can be a variable, constant, compound term, or range
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Term {
    /// Variable: uppercase or starts with underscore (X, Y, _tmp)
    Variable(Symbol),
    /// Constant: integer, string, or lowercase identifier
    Constant(Value),
    /// Compound term: functor with arguments (f(a, b))
    Compound(Symbol, Vec<Term>),
    /// Range: generates sequence of integers (1..10, 1..width)
    Range(Box<Term>, Box<Term>),
}

/// Constant values
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Value {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(Symbol),
    Atom(Symbol),
}

// Implement PartialEq for Value
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Atom(a), Value::Atom(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Value {}

// Implement Hash for Value
impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::Integer(i) => {
                0u8.hash(state);
                i.hash(state);
            }
            Value::Float(f) => {
                1u8.hash(state);
                // Hash the bit representation of the float for consistency
                f.to_bits().hash(state);
            }
            Value::Boolean(b) => {
                2u8.hash(state);
                b.hash(state);
            }
            Value::String(s) => {
                3u8.hash(state);
                s.hash(state);
            }
            Value::Atom(a) => {
                4u8.hash(state);
                a.hash(state);
            }
        }
    }
}

impl Program {
    /// Create a new empty program
    #[allow(dead_code)]
    pub fn new() -> Self {
        Program {
            statements: Vec::new(),
        }
    }

    /// Add a statement to the program
    #[allow(dead_code)]
    pub fn add_statement(&mut self, statement: Statement) {
        self.statements.push(statement);
    }
}

impl Literal {
    /// Get the underlying atom from a literal (None for aggregates)
    #[allow(dead_code)]
    pub fn atom(&self) -> Option<&Atom> {
        match self {
            Literal::Positive(atom) | Literal::Negative(atom) => Some(atom),
            Literal::Aggregate(_) => None,
        }
    }

    /// Check if the literal is positive
    #[allow(dead_code)]
    pub fn is_positive(&self) -> bool {
        matches!(self, Literal::Positive(_))
    }

    /// Check if the literal is negative
    #[allow(dead_code)]
    pub fn is_negative(&self) -> bool {
        matches!(self, Literal::Negative(_))
    }

    /// Check if the literal is an aggregate
    #[allow(dead_code)]
    pub fn is_aggregate(&self) -> bool {
        matches!(self, Literal::Aggregate(_))
    }
}

#[cfg(test)]
mod tests {
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
}
