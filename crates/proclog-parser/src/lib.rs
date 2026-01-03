//! Parser for ProcLog programs
//!
//! This crate implements a parser combinator-based parser using the Chumsky library.
//! It parses ProcLog programs from text into an AST (Abstract Syntax Tree).
//!
//! # Supported Syntax
//!
//! - **Facts**: `parent(john, mary).`
//! - **Rules**: `ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).`
//! - **Constraints**: `:- unsafe(X).`
//! - **Choice Rules**: `{ selected(X) : item(X) } 2.`
//! - **Constants**: `#const max_items = 10.`
//! - **Ranges**: `cell(1..width, 1..height).`
//! - **Built-ins**: Arithmetic (`X + Y = Z`) and comparisons (`X > 5`)
//!
//! # Example
//!
//! ```ignore
//! use proclog_parser::{parse_program, SrcId};
//!
//! let program_text = "parent(john, mary). ancestor(X, Z) :- parent(X, Z).";
//! let program = parse_program(program_text, SrcId::empty()).expect("Parse error");
//! ```

mod parser;
mod span;
mod src;
mod token;

pub use parser::{parse_program, parse_query, ParseError};
pub use span::Span;
pub use src::SrcId;
pub use token::{Keyword, LexError, Token};

#[cfg(test)]
#[path = "../tests/unit/arithmetic_integration_tests.rs"]
mod arithmetic_integration_tests;
