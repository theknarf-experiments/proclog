//! Parser implementation details.

use chumsky::prelude::*;
use chumsky::stream::Stream;
use internment::Intern;

use crate::token::{lexer, Keyword, LexError, SpannedToken, Token};
use crate::{Span, SrcId};
use proclog_ast::*;

type ParserError = Simple<Token, Span>;

#[derive(Debug, Clone)]
pub enum ParseError {
    Lex(LexError),
    Parse(ParserError),
}

fn ident_token() -> impl Parser<Token, String, Error = ParserError> + Clone {
    select! {
        Token::Ident(ident) => ident,
        Token::Keyword(Keyword::Const) => "const".to_string(),
        Token::Keyword(Keyword::Test) => "test".to_string(),
        Token::Keyword(Keyword::Minimize) => "minimize".to_string(),
        Token::Keyword(Keyword::Maximize) => "maximize".to_string(),
    }
    .labelled("identifier")
}

fn ident_value(expected: &'static str) -> impl Parser<Token, String, Error = ParserError> + Clone {
    select! { Token::Ident(ident) if ident == expected => ident }
}

fn variable_token() -> impl Parser<Token, String, Error = ParserError> + Clone {
    select! {
        Token::Variable(ident) => ident,
        Token::Ident(ident) if ident.starts_with('_') => ident,
    }
    .labelled("variable")
}

fn string_token() -> impl Parser<Token, String, Error = ParserError> + Clone {
    select! { Token::String(value) => value }.labelled("string")
}

fn number_token() -> impl Parser<Token, Value, Error = ParserError> + Clone {
    select! { Token::Number(number) => number }
        .try_map(|value: String, span| {
            if value.contains('.') {
                value
                    .parse::<f64>()
                    .map(Value::Float)
                    .map_err(|_| ParserError::custom(span, "invalid float"))
            } else {
                value
                    .parse::<i64>()
                    .map(Value::Integer)
                    .map_err(|_| ParserError::custom(span, "invalid integer"))
            }
        })
        .labelled("number")
}

fn signed_number_token() -> impl Parser<Token, Value, Error = ParserError> + Clone {
    operator_token("-")
        .ignore_then(number_token())
        .map(|value| match value {
            Value::Integer(n) => Value::Integer(-n),
            Value::Float(n) => Value::Float(-n),
            other => other,
        })
        .or(number_token())
        .labelled("number")
}

fn keyword_token(keyword: Keyword) -> impl Parser<Token, Keyword, Error = ParserError> + Clone {
    just(Token::Keyword(keyword)).to(keyword)
}

fn operator_token(op: &'static str) -> impl Parser<Token, String, Error = ParserError> + Clone {
    select! { Token::Operator(value) if value == op => value }
}

fn token(kind: Token) -> impl Parser<Token, Token, Error = ParserError> + Clone {
    just(kind)
}

fn lex_with_src(input: &str, src: SrcId) -> Result<Vec<SpannedToken>, Vec<ParseError>> {
    let len = input.chars().count();
    let eoi = Span::new(src, len..len);
    let stream = Stream::from_iter(
        eoi,
        input
            .chars()
            .enumerate()
            .map(|(idx, ch)| (ch, Span::new(src, idx..idx + 1))),
    );
    lexer()
        .parse(stream)
        .map_err(|errors| errors.into_iter().map(ParseError::Lex).collect())
}

#[cfg(test)]
fn lex(input: &str) -> Result<Vec<SpannedToken>, Vec<ParseError>> {
    lex_with_src(input, SrcId::empty())
}

fn parse_with<T>(
    parser: impl Parser<Token, T, Error = ParserError>,
    input: &str,
    src: SrcId,
) -> Result<T, Vec<ParseError>> {
    let tokens = lex_with_src(input, src)?;
    let end = input.chars().count();
    let eoi = Span::new(src, end..end);
    let stream = Stream::from_iter(eoi, tokens.into_iter());
    parser
        .parse(stream)
        .map_err(|errors| errors.into_iter().map(ParseError::Parse).collect())
}

// This is now handled inline within term() to avoid circular dependency issues

fn factor_parser<'a>(
    term: Recursive<'a, Token, Term, ParserError>,
) -> impl Parser<Token, Term, Error = ParserError> + Clone + 'a {
    let variable = variable_token().map(|s| Term::Variable(Intern::new(s)));

    let number_const = signed_number_token().map(Term::Constant);

    let string_const = string_token().map(|s| Term::Constant(Value::String(Intern::new(s))));

    let parens = term
        .clone()
        .delimited_by(token(Token::LParen), token(Token::RParen));

    let compound_or_atom = ident_token()
        .then(
            term.clone()
                .separated_by(token(Token::Comma))
                .delimited_by(token(Token::LParen), token(Token::RParen))
                .or_not(),
        )
        .map(|(name, args)| {
            if let Some(args) = args {
                Term::Compound(Intern::new(name), args)
            } else {
                match name.as_str() {
                    "true" => Term::Constant(Value::Boolean(true)),
                    "false" => Term::Constant(Value::Boolean(false)),
                    _ => Term::Constant(Value::Atom(Intern::new(name))),
                }
            }
        });

    choice((variable, number_const, string_const, parens, compound_or_atom))
}

fn range_parser() -> impl Parser<Token, Term, Error = ParserError> + Clone {
    let range_bound = choice((
        select! { Token::Number(value) => value }.try_map(|value: String, span| {
            value
                .parse::<i64>()
                .map(|n| Term::Constant(Value::Integer(n)))
                .map_err(|_| ParserError::custom(span, "invalid integer"))
        }),
        ident_token().map(|s| Term::Constant(Value::Atom(Intern::new(s)))),
    ));

    range_bound
        .clone()
        .then_ignore(token(Token::RangeDots))
        .then(range_bound)
        .map(|(start, end)| Term::Range(Box::new(start), Box::new(end)))
}

fn arithmetic_parser<'a>(
    factor: impl Parser<Token, Term, Error = ParserError> + Clone + 'a,
) -> impl Parser<Token, Term, Error = ParserError> + Clone + 'a {
    let mul_div = factor
        .clone()
        .then(
            choice((
                operator_token("*").to("*"),
                operator_token("/").to("/"),
                ident_value("mod").to("mod"),
            ))
            .then(factor.clone())
                .repeated(),
        )
        .foldl(|left, (op, right)| Term::Compound(Intern::new(op.to_string()), vec![left, right]));

    mul_div
        .clone()
        .then(
            choice((operator_token("+").to("+"), operator_token("-").to("-")))
                .then(mul_div)
                .repeated(),
        )
        .foldl(|left, (op, right)| Term::Compound(Intern::new(op.to_string()), vec![left, right]))
}

/// Parse a term (variable, constant, or compound) with arithmetic operator precedence
fn term() -> impl Parser<Token, Term, Error = ParserError> + Clone {
    recursive(|term| {
        let factor = factor_parser(term.clone());
        let range = range_parser();
        let arithmetic = arithmetic_parser(factor);

        range.or(arithmetic)
    })
    .labelled("term")
}

/// Parse an atom
fn atom() -> impl Parser<Token, Atom, Error = ParserError> + Clone {
    // Parse either a regular identifier or an operator as the predicate
    let predicate = choice((ident_token(), select! { Token::Operator(op) => op }));

    predicate
        .then(
            term()
                .separated_by(token(Token::Comma))
                .delimited_by(token(Token::LParen), token(Token::RParen))
                .or_not(),
        )
        .map(|(predicate, terms)| Atom {
            predicate: Intern::new(predicate),
            terms: terms.unwrap_or_default(),
        })
        .labelled("atom")
}

/// Parse an infix comparison as an atom (e.g., X > 3, Y = 5)
fn infix_comparison() -> impl Parser<Token, Atom, Error = ParserError> + Clone {
    term()
        .then(
            choice((
                operator_token("<=").to("<="),
                operator_token(">=").to(">="),
                operator_token("!=").to("!="),
                operator_token("\\=").to("\\="),
                operator_token("=<").to("=<"),
                operator_token("=").to("="),
                operator_token("<").to("<"),
                operator_token(">").to(">"),
            )),
        )
        .then(term())
        .map(|((left, op), right)| Atom {
            predicate: Intern::new(op.to_string()),
            terms: vec![left, right],
        })
}

/// Parse a comparison operator and return the ComparisonOp enum
fn comparison_operator() -> impl Parser<Token, ComparisonOp, Error = ParserError> + Clone {
    choice((
        operator_token("<=").to(ComparisonOp::LessOrEqual),
        operator_token(">=").to(ComparisonOp::GreaterOrEqual),
        operator_token("!=").to(ComparisonOp::NotEqual),
        operator_token("=").to(ComparisonOp::Equal),
        operator_token("<").to(ComparisonOp::LessThan),
        operator_token(">").to(ComparisonOp::GreaterThan),
    ))
}

/// Parse a count aggregate: count { Variables : Conditions } Comparison Value
fn count_aggregate() -> impl Parser<Token, Literal, Error = ParserError> + Clone {
    ident_value("count")
        .ignore_then(token(Token::LBrace))
        .ignore_then(
            variable_token()
                .separated_by(token(Token::Comma))
                .at_least(1)
                .map(|vars: Vec<String>| vars.into_iter().map(Intern::new).collect::<Vec<_>>()),
        )
        .then_ignore(token(Token::Colon))
        .then(
            choice((infix_comparison(), atom()))
                .map(Literal::Positive)
                .separated_by(token(Token::Comma))
                .at_least(1),
        )
        .then_ignore(token(Token::RBrace))
        .then(comparison_operator())
        .then(term())
        .map(|(((variables, elements), comparison), value)| {
            Literal::Aggregate(AggregateAtom {
                function: AggregateFunction::Count,
                variables,
                elements,
                comparison,
                value,
            })
        })
        .labelled("count aggregate")
}

/// Parse a literal (positive, negative, or aggregate)
fn literal() -> impl Parser<Token, Literal, Error = ParserError> + Clone {
    let aggregate = count_aggregate();

    let negated = keyword_token(Keyword::Not)
        .ignore_then(choice((infix_comparison(), atom())))
        .map(Literal::Negative);

    let positive = choice((infix_comparison(), atom())).map(Literal::Positive);

    choice((aggregate, negated, positive)).labelled("literal")
}

/// Parse an optimization statement: #minimize { ... } or #maximize { ... }
fn optimize_statement() -> impl Parser<Token, Statement, Error = ParserError> + Clone {
    use proclog_ast::{OptimizeDirection, OptimizeStatement, OptimizeTerm};

    // Parse optimization direction (#minimize or #maximize)
    let direction = token(Token::Hash).ignore_then(choice((
        keyword_token(Keyword::Minimize).to(OptimizeDirection::Minimize),
        keyword_token(Keyword::Maximize).to(OptimizeDirection::Maximize),
    )));

    // Parse an optimization term: term[:condition]
    // The term itself can be a simple term or compound (e.g., C*X)
    let optimize_term = term()
        .then(
            token(Token::Colon)
                .ignore_then(literal().separated_by(token(Token::Comma)).at_least(1))
                .or_not(),
        )
        .map(|(t, condition)| OptimizeTerm {
            weight: None, // Weight extraction from compound terms can be added later
            term: t,
            condition: condition.unwrap_or_default(),
        });

    // Elements inside braces, separated by semicolons
    let elements = optimize_term
        .separated_by(token(Token::Semicolon))
        .at_least(1)
        .delimited_by(token(Token::LBrace), token(Token::RBrace));

    direction
        .then(elements)
        .then_ignore(token(Token::Dot))
        .map(|(direction, terms)| Statement::Optimize(OptimizeStatement { direction, terms }))
        .labelled("optimization statement")
}

fn non_test_statement() -> impl Parser<Token, Statement, Error = ParserError> + Clone {
    choice((
        optimize_statement(), // Try optimization statements first (starts with #)
        const_decl(),
        choice_rule(),
        constraint(),
        rule(),
        fact(),
    ))
}

/// Parse a fact
fn fact() -> impl Parser<Token, Statement, Error = ParserError> + Clone {
    atom()
        .then_ignore(token(Token::Dot))
        .map(|atom| Statement::Fact(Fact { atom }))
        .labelled("fact")
}

/// Parse a rule
fn rule() -> impl Parser<Token, Statement, Error = ParserError> + Clone {
    atom()
        .then_ignore(token(Token::RuleSep))
        .then(
            literal()
                .separated_by(token(Token::Comma))
                .at_least(1),
        )
        .then_ignore(token(Token::Dot))
        .map(|(head, body)| Statement::Rule(Rule { head, body }))
        .labelled("rule")
}

/// Parse a constraint
fn constraint() -> impl Parser<Token, Statement, Error = ParserError> + Clone {
    token(Token::RuleSep)
        .ignore_then(literal().separated_by(token(Token::Comma)).at_least(1))
        .then_ignore(token(Token::Dot))
        .map(|body| Statement::Constraint(Constraint { body }))
        .labelled("constraint")
}

/// Parse a constant declaration: #const name = value.
/// Supports all value types: integers, floats, booleans, strings, atoms
fn const_decl() -> impl Parser<Token, Statement, Error = ParserError> + Clone {
    token(Token::Hash)
        .then_ignore(keyword_token(Keyword::Const))
        .ignore_then(ident_token())
        .then_ignore(operator_token("="))
        .then(choice((
            // Try number (int or float) first
            signed_number_token(),
            // Try string
            string_token().map(|s| Value::String(Intern::new(s))),
            // Try boolean or atom (both are lowercase identifiers)
            ident_token().map(|s| match s.as_str() {
                "true" => Value::Boolean(true),
                "false" => Value::Boolean(false),
                _ => Value::Atom(Intern::new(s)),
            }),
        )))
        .then_ignore(token(Token::Dot))
        .map(|(name, value)| {
            Statement::ConstDecl(ConstDecl {
                name: Intern::new(name),
                value,
            })
        })
        .labelled("constant declaration")
}

/// Parse a test assertion: + atom. or - atom.
fn test_assertion() -> impl Parser<Token, (bool, Atom), Error = ParserError> + Clone {
    choice((
        operator_token("+").to(true),
        operator_token("-").to(false),
    ))
    .then(atom())
    .then_ignore(token(Token::Dot))
        .labelled("test assertion")
}

/// Parse a test case: query followed by assertions
fn test_case() -> impl Parser<Token, TestCase, Error = ParserError> + Clone {
    // Parse query: ?- body.
    let query_body = token(Token::Question)
        .then_ignore(operator_token("-"))
        .ignore_then(literal().separated_by(token(Token::Comma)).at_least(1))
        .then_ignore(token(Token::Dot));

    query_body
        .then(test_assertion().repeated().collect::<Vec<_>>())
        .map(|(body, assertions)| {
            let mut positive_assertions = Vec::new();
            let mut negative_assertions = Vec::new();

            for (is_positive, atom) in assertions {
                if is_positive {
                    positive_assertions.push(atom);
                } else {
                    negative_assertions.push(atom);
                }
            }

            TestCase {
                query: Query { body },
                positive_assertions,
                negative_assertions,
            }
        })
        .labelled("test case")
}

/// Parse a test block: #test "name" { statements and test cases }
fn test_block() -> impl Parser<Token, Statement, Error = ParserError> + Clone {
    // Test block content item can be either a statement or a test case
    let test_item = choice((
        // Try test case first (starts with ?-)
        test_case().map(|tc| (None, Some(tc))),
        // Then try regular statements (but not other test blocks)
        non_test_statement().map(|stmt| (Some(stmt), None)),
    ));

    token(Token::Hash)
        .then_ignore(keyword_token(Keyword::Test))
        .ignore_then(string_token())
        .then(
            token(Token::LBrace)
                .ignore_then(test_item.repeated().collect::<Vec<_>>())
                .then_ignore(token(Token::RBrace)),
        )
        .map(|(name, items)| {
            let mut statements = Vec::new();
            let mut test_cases = Vec::new();

            for (stmt_opt, test_opt) in items {
                if let Some(stmt) = stmt_opt {
                    statements.push(stmt);
                }
                if let Some(test) = test_opt {
                    test_cases.push(test);
                }
            }

            Statement::Test(TestBlock {
                name,
                statements,
                test_cases,
            })
        })
        .labelled("test block")
}

/// Parse a choice rule: { atom1; atom2 } or 1 { atom1; atom2 } 2 :- body.
fn choice_rule() -> impl Parser<Token, Statement, Error = ParserError> + Clone {
    // Optional lower bound - can be integer or identifier (constant name)
    let lower_bound = choice((
        select! { Token::Number(value) => value }.try_map(|value: String, span| {
            value
                .parse::<i64>()
                .map(|n| Term::Constant(Value::Integer(n)))
                .map_err(|_| ParserError::custom(span, "invalid integer"))
        }),
        ident_token().map(|s| Term::Constant(Value::Atom(Intern::new(s)))),
    ))
    .or_not();

    // Choice element: atom or atom : condition1, condition2
    let choice_element = atom()
        .then(
            token(Token::Colon)
                .ignore_then(literal().separated_by(token(Token::Comma)).at_least(1))
                .or_not(),
        )
        .map(|(atom, condition)| ChoiceElement {
            atom,
            condition: condition.unwrap_or_default(),
        });

    // Elements inside braces, separated by semicolons
    let elements = choice_element
        .separated_by(token(Token::Semicolon))
        .at_least(1)
        .delimited_by(token(Token::LBrace), token(Token::RBrace));

    // Optional upper bound - can be integer or identifier (constant name)
    let upper_bound = choice((
        select! { Token::Number(value) => value }.try_map(|value: String, span| {
            value
                .parse::<i64>()
                .map(|n| Term::Constant(Value::Integer(n)))
                .map_err(|_| ParserError::custom(span, "invalid integer"))
        }),
        ident_token().map(|s| Term::Constant(Value::Atom(Intern::new(s)))),
    ))
    .or_not();

    // Optional body after :-
    let body = token(Token::RuleSep)
        .ignore_then(literal().separated_by(token(Token::Comma)).at_least(1))
        .or_not();

    lower_bound
        .then(elements)
        .then(upper_bound)
        .then(body)
        .then_ignore(token(Token::Dot))
        .map(|(((lower, elements), upper), body)| {
            Statement::ChoiceRule(ChoiceRule {
                lower_bound: lower,
                upper_bound: upper,
                elements,
                body: body.unwrap_or_default(),
            })
        })
        .labelled("choice rule")
}

/// Parse a statement
fn statement() -> impl Parser<Token, Statement, Error = ParserError> + Clone {
    choice((
        test_block(), // Try #test first (distinctive prefix)
        non_test_statement(),
    ))
    .labelled("statement")
}

/// Parse a program
pub fn program() -> impl Parser<Token, Program, Error = ParserError> + Clone {
    statement()
        .repeated()
        .map(|statements| Program { statements })
        .then_ignore(end())
        .labelled("program")
}

/// Helper function to parse with better error handling
pub fn parse_program(input: &str, src: SrcId) -> Result<Program, Vec<ParseError>> {
    parse_with(program(), input, src)
}

/// Parse a query: ?- literal1, literal2, ..., literalN.
#[cfg_attr(not(test), allow(dead_code))]
fn query() -> impl Parser<Token, Query, Error = ParserError> + Clone {
    token(Token::Question)
        .then_ignore(operator_token("-"))
        .ignore_then(literal().separated_by(token(Token::Comma)).at_least(1))
        .then_ignore(token(Token::Dot))
        .map(|body| Query { body })
        .labelled("query")
}

/// Helper function to parse a query
#[cfg_attr(not(test), allow(dead_code))]
pub fn parse_query(input: &str, src: SrcId) -> Result<Query, Vec<ParseError>> {
    parse_with(query(), input, src)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_with<T>(
        parser: impl Parser<Token, T, Error = ParserError>,
        input: &str,
    ) -> Result<T, Vec<ParseError>> {
        super::parse_with(parser, input, SrcId::empty())
    }

    fn parse_program(input: &str) -> Result<Program, Vec<ParseError>> {
        super::parse_program(input, SrcId::empty())
    }

    fn parse_query(input: &str) -> Result<Query, Vec<ParseError>> {
        super::parse_query(input, SrcId::empty())
    }

    // Helper functions for tests
    fn int_term(n: i64) -> Term {
        Term::Constant(Value::Integer(n))
    }

    fn atom_term(name: &str) -> Term {
        Term::Constant(Value::Atom(Intern::new(name.to_string())))
    }

    fn range(start: i64, end: i64) -> Term {
        Term::Range(Box::new(int_term(start)), Box::new(int_term(end)))
    }

    fn range_const(start: i64, end_name: &str) -> Term {
        Term::Range(Box::new(int_term(start)), Box::new(atom_term(end_name)))
    }

    #[test]
    fn test_string_literal_escape_sequences() {
        let input = "\"escaped\\\" newline\\n tab\\t backslash\\\\\"";
        let result = lex(input);
        assert!(
            result.is_ok(),
            "Failed to parse escapes: {:?}",
            result.err()
        );
        let tokens = result.unwrap();
        assert_eq!(tokens.len(), 1);
        match &tokens[0].0 {
            Token::String(parsed) => {
                assert_eq!(parsed, "escaped\" newline\n tab\t backslash\\");
            }
            other => panic!("Expected string token, got {:?}", other),
        }
    }

    #[test]
    fn test_string_literal_rejects_raw_newline() {
        let input = "\"line\nnext\"";
        let result = lex(input);
        assert!(
            result.is_err(),
            "String literal with raw newline should be rejected"
        );
    }

    #[test]
    fn test_string_literal_rejects_invalid_escape() {
        let input = "\"invalid\\xescape\"";
        let result = lex(input);
        assert!(
            result.is_err(),
            "String literal with invalid escape should be rejected"
        );
    }

    #[test]
    fn test_string_literal_in_ast_has_interpreted_value() {
        let input = "value(\"line\\nnext\").";
        let result = parse_program(input);
        assert!(
            result.is_ok(),
            "Failed to parse program: {:?}",
            result.err()
        );
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.terms.len(), 1);
                match &fact.atom.terms[0] {
                    Term::Constant(Value::String(s)) => {
                        assert_eq!(s.as_ref(), "line\nnext");
                    }
                    other => panic!("Expected string constant, got {:?}", other),
                }
            }
            other => panic!("Expected fact statement, got {:?}", other),
        }
    }

    // Term parsing tests - Datatypes

    // Integers
    #[test]
    fn test_parse_positive_integer() {
        let result = parse_with(term(), "42");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Integer(42)));
    }

    #[test]
    fn test_parse_negative_integer() {
        let result = parse_with(term(), "-42");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Integer(-42)));
    }

    #[test]
    fn test_parse_zero() {
        let result = parse_with(term(), "0");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Integer(0)));
    }

    // Ranges
    #[test]
    fn test_parse_range_simple() {
        let result = parse_with(term(), "1..10");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), range(1, 10));
    }

    #[test]
    fn test_parse_range_zero_start() {
        let result = parse_with(term(), "0..5");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), range(0, 5));
    }

    #[test]
    fn test_parse_range_large() {
        let result = parse_with(term(), "1..100");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), range(1, 100));
    }

    #[test]
    fn test_parse_range_same_values() {
        let result = parse_with(term(), "5..5");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), range(5, 5));
    }

    #[test]
    fn test_parse_range_with_constant() {
        let result = parse_with(term(), "1..width");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), range_const(1, "width"));
    }

    #[test]
    fn test_parse_range_both_constants() {
        let result = parse_with(term(), "min..max");
        assert!(result.is_ok());
        let expected = Term::Range(Box::new(atom_term("min")), Box::new(atom_term("max")));
        assert_eq!(result.unwrap(), expected);
    }

    // Floats
    #[test]
    fn test_parse_positive_float() {
        let result = parse_with(term(), "3.14");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Float(3.14)));
    }

    #[test]
    fn test_parse_negative_float() {
        let result = parse_with(term(), "-2.5");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Float(-2.5)));
    }

    #[test]
    fn test_parse_float_with_zero_decimal() {
        let result = parse_with(term(), "10.0");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Float(10.0)));
    }

    // Booleans
    #[test]
    fn test_parse_boolean_true() {
        let result = parse_with(term(), "true");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Boolean(true)));
    }

    #[test]
    fn test_parse_boolean_false() {
        let result = parse_with(term(), "false");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Boolean(false)));
    }

    // Variables
    #[test]
    fn test_parse_uppercase_variable() {
        let result = parse_with(term(), "X");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Variable(Intern::new("X".to_string()))
        );
    }

    #[test]
    fn test_parse_multichar_variable() {
        let result = parse_with(term(), "Player");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Variable(Intern::new("Player".to_string()))
        );
    }

    #[test]
    fn test_parse_underscore_variable() {
        let result = parse_with(term(), "_tmp");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Variable(Intern::new("_tmp".to_string()))
        );
    }

    // Strings
    #[test]
    fn test_parse_string_constant() {
        let result = parse_with(term(), "\"hello\"");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Constant(Value::String(Intern::new("hello".to_string())))
        );
    }

    #[test]
    fn test_parse_empty_string() {
        let result = parse_with(term(), "\"\"");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Constant(Value::String(Intern::new("".to_string())))
        );
    }

    // Atoms
    #[test]
    fn test_parse_atom_constant() {
        let result = parse_with(term(), "john");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Constant(Value::Atom(Intern::new("john".to_string())))
        );
    }

    #[test]
    fn test_parse_compound_term() {
        let result = parse_with(term(), "f(a, b)");
        assert!(result.is_ok());
        let expected = Term::Compound(
            Intern::new("f".to_string()),
            vec![
                Term::Constant(Value::Atom(Intern::new("a".to_string()))),
                Term::Constant(Value::Atom(Intern::new("b".to_string()))),
            ],
        );
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_parse_range_term() {
        let result = parse_with(term(), "1..10");
        assert!(result.is_ok());
        match result.unwrap() {
            Term::Range(start, end) => {
                assert_eq!(*start, Term::Constant(Value::Integer(1)));
                assert_eq!(*end, Term::Constant(Value::Integer(10)));
            }
            other => panic!("Expected range term, got {:?}", other),
        }
    }

    #[test]
    fn test_term_arithmetic_precedence_basic() {
        let result = parse_with(term(), "1 + 2 * 3");
        assert!(result.is_ok());
        let expected = Term::Compound(
            Intern::new("+".to_string()),
            vec![
                Term::Constant(Value::Integer(1)),
                Term::Compound(
                    Intern::new("*".to_string()),
                    vec![
                        Term::Constant(Value::Integer(2)),
                        Term::Constant(Value::Integer(3)),
                    ],
                ),
            ],
        );
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_parse_mixed_term_arguments() {
        let result = parse_with(term(), "foo(X, 1..3, Y + 4)");
        assert!(result.is_ok());
        let expected = Term::Compound(
            Intern::new("foo".to_string()),
            vec![
                Term::Variable(Intern::new("X".to_string())),
                Term::Range(
                    Box::new(Term::Constant(Value::Integer(1))),
                    Box::new(Term::Constant(Value::Integer(3))),
                ),
                Term::Compound(
                    Intern::new("+".to_string()),
                    vec![
                        Term::Variable(Intern::new("Y".to_string())),
                        Term::Constant(Value::Integer(4)),
                    ],
                ),
            ],
        );
        assert_eq!(result.unwrap(), expected);
    }

    // Atom parsing tests
    #[test]
    fn test_parse_atom_no_args() {
        let result = parse_with(atom(), "foo");
        assert!(result.is_ok());
        let expected = Atom {
            predicate: Intern::new("foo".to_string()),
            terms: vec![],
        };
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_parse_atom_with_args() {
        let result = parse_with(atom(), "parent(john, mary)");
        assert!(result.is_ok());
        let expected = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![
                Term::Constant(Value::Atom(Intern::new("john".to_string()))),
                Term::Constant(Value::Atom(Intern::new("mary".to_string()))),
            ],
        };
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_parse_atom_with_variables() {
        let result = parse_with(atom(), "parent(X, Y)");
        assert!(result.is_ok());
        let expected = Atom {
            predicate: Intern::new("parent".to_string()),
            terms: vec![
                Term::Variable(Intern::new("X".to_string())),
                Term::Variable(Intern::new("Y".to_string())),
            ],
        };
        assert_eq!(result.unwrap(), expected);
    }

    // Literal parsing tests
    #[test]
    fn test_parse_positive_literal() {
        let result = parse_with(literal(), "parent(X, Y)");
        assert!(result.is_ok());
        match result.unwrap() {
            Literal::Positive(atom) => {
                assert_eq!(atom.predicate, Intern::new("parent".to_string()));
                assert_eq!(atom.terms.len(), 2);
            }
            Literal::Negative(_) => panic!("Expected positive literal"),
            Literal::Aggregate(_) => panic!("Expected positive literal, got aggregate"),
        }
    }

    #[test]
    fn test_parse_negative_literal() {
        let result = parse_with(literal(), "not parent(X, Y)");
        assert!(result.is_ok());
        match result.unwrap() {
            Literal::Negative(atom) => {
                assert_eq!(atom.predicate, Intern::new("parent".to_string()));
                assert_eq!(atom.terms.len(), 2);
            }
            Literal::Positive(_) => panic!("Expected negative literal"),
            Literal::Aggregate(_) => panic!("Expected negative literal, got aggregate"),
        }
    }

    // Aggregate parsing tests
    #[test]
    fn test_parse_count_aggregate_simple() {
        let result = parse_with(literal(), "count { X : selected(X) } > 2");
        assert!(result.is_ok());
        match result.unwrap() {
            Literal::Aggregate(agg) => {
                assert_eq!(agg.function, AggregateFunction::Count);
                assert_eq!(agg.variables.len(), 1);
                assert_eq!(agg.variables[0], Intern::new("X".to_string()));
                assert_eq!(agg.elements.len(), 1);
                assert_eq!(agg.comparison, ComparisonOp::GreaterThan);
                match &agg.value {
                    Term::Constant(Value::Integer(n)) => assert_eq!(*n, 2),
                    _ => panic!("Expected integer value"),
                }
            }
            _ => panic!("Expected aggregate literal"),
        }
    }

    #[test]
    fn test_parse_count_aggregate_equality() {
        let result = parse_with(literal(), "count { X : item(X) } = 5");
        assert!(result.is_ok());
        match result.unwrap() {
            Literal::Aggregate(agg) => {
                assert_eq!(agg.comparison, ComparisonOp::Equal);
                match &agg.value {
                    Term::Constant(Value::Integer(n)) => assert_eq!(*n, 5),
                    _ => panic!("Expected integer value"),
                }
            }
            _ => panic!("Expected aggregate literal"),
        }
    }

    #[test]
    fn test_parse_count_aggregate_less_equal() {
        let result = parse_with(literal(), "count { Y : member(Y) } <= 10");
        assert!(result.is_ok());
        match result.unwrap() {
            Literal::Aggregate(agg) => {
                assert_eq!(agg.comparison, ComparisonOp::LessOrEqual);
                assert_eq!(agg.variables[0], Intern::new("Y".to_string()));
            }
            _ => panic!("Expected aggregate literal"),
        }
    }

    #[test]
    fn test_parse_count_aggregate_multiple_conditions() {
        let result = parse_with(literal(), "count { X : item(X), heavy(X) } < 3");
        assert!(result.is_ok());
        match result.unwrap() {
            Literal::Aggregate(agg) => {
                assert_eq!(agg.elements.len(), 2);
                assert_eq!(agg.comparison, ComparisonOp::LessThan);
            }
            _ => panic!("Expected aggregate literal"),
        }
    }

    #[test]
    fn test_parse_constraint_with_count_aggregate() {
        let result = parse_program(":- count { X : selected(X) } > 2.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        match &result.unwrap().statements[0] {
            Statement::Constraint(constraint) => {
                assert_eq!(constraint.body.len(), 1);
                match &constraint.body[0] {
                    Literal::Aggregate(agg) => {
                        assert_eq!(agg.function, AggregateFunction::Count);
                        assert_eq!(agg.comparison, ComparisonOp::GreaterThan);
                    }
                    _ => panic!("Expected aggregate in constraint body"),
                }
            }
            _ => panic!("Expected constraint"),
        }
    }

    #[test]
    fn test_parse_rule_with_count_aggregate() {
        let result = parse_program("valid :- count { X : item(X) } <= 5.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        match &result.unwrap().statements[0] {
            Statement::Rule(rule) => {
                assert_eq!(rule.head.predicate, Intern::new("valid".to_string()));
                assert_eq!(rule.body.len(), 1);
                match &rule.body[0] {
                    Literal::Aggregate(agg) => {
                        assert_eq!(agg.function, AggregateFunction::Count);
                        assert_eq!(agg.comparison, ComparisonOp::LessOrEqual);
                    }
                    _ => panic!("Expected aggregate in rule body"),
                }
            }
            _ => panic!("Expected rule"),
        }
    }

    // Statement parsing tests
    #[test]
    fn test_parse_fact() {
        let result = parse_program("parent(john, mary).");
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.predicate, Intern::new("parent".to_string()));
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_rule() {
        let result = parse_program("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).");
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            Statement::Rule(rule) => {
                assert_eq!(rule.head.predicate, Intern::new("ancestor".to_string()));
                assert_eq!(rule.body.len(), 2);
            }
            _ => panic!("Expected rule"),
        }
    }

    #[test]
    fn test_parse_constraint() {
        let result = parse_program(":- unsafe(X).");
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            Statement::Constraint(constraint) => {
                assert_eq!(constraint.body.len(), 1);
            }
            _ => panic!("Expected constraint"),
        }
    }

    #[test]
    fn test_parse_const_decl() {
        let result = parse_program("#const width = 10.");
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("width".to_string()));
                assert_eq!(const_decl.value, Value::Integer(10));
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    #[test]
    fn test_parse_const_decl_negative() {
        let result = parse_program("#const min_temp = -5.");
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("min_temp".to_string()));
                assert_eq!(const_decl.value, Value::Integer(-5));
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    // Constant declarations with different data types
    #[test]
    fn test_parse_const_decl_float() {
        let result = parse_program("#const pi = 3.14.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("pi".to_string()));
                match const_decl.value {
                    Value::Float(f) => assert_eq!(f, 3.14),
                    _ => panic!("Expected float value, got {:?}", const_decl.value),
                }
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    #[test]
    fn test_parse_const_decl_negative_float() {
        let result = parse_program("#const neg = -2.5.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("neg".to_string()));
                match const_decl.value {
                    Value::Float(f) => assert_eq!(f, -2.5),
                    _ => panic!("Expected float value"),
                }
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    #[test]
    fn test_parse_const_decl_boolean_true() {
        let result = parse_program("#const enabled = true.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("enabled".to_string()));
                match const_decl.value {
                    Value::Boolean(true) => {}
                    _ => panic!("Expected true boolean value"),
                }
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    #[test]
    fn test_parse_const_decl_boolean_false() {
        let result = parse_program("#const disabled = false.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("disabled".to_string()));
                match const_decl.value {
                    Value::Boolean(false) => {}
                    _ => panic!("Expected false boolean value"),
                }
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    #[test]
    fn test_parse_const_decl_string() {
        let result = parse_program("#const message = \"hello world\".");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("message".to_string()));
                match &const_decl.value {
                    Value::String(s) => assert_eq!(*s, Intern::new("hello world".to_string())),
                    _ => panic!("Expected string value"),
                }
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    #[test]
    fn test_parse_const_decl_atom() {
        let result = parse_program("#const default_color = red.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let program = result.unwrap();
        match &program.statements[0] {
            Statement::ConstDecl(const_decl) => {
                assert_eq!(const_decl.name, Intern::new("default_color".to_string()));
                match &const_decl.value {
                    Value::Atom(a) => assert_eq!(*a, Intern::new("red".to_string())),
                    _ => panic!("Expected atom value"),
                }
            }
            _ => panic!("Expected constant declaration"),
        }
    }

    #[test]
    fn test_parse_multiple_const_decls() {
        let input = r#"
            #const width = 10.
            #const height = 20.
            #const max_enemies = 5.
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 3);

        match &program.statements[0] {
            Statement::ConstDecl(c) => {
                assert_eq!(c.name, Intern::new("width".to_string()));
                assert_eq!(c.value, Value::Integer(10));
            }
            _ => panic!("Expected const"),
        }

        match &program.statements[1] {
            Statement::ConstDecl(c) => {
                assert_eq!(c.name, Intern::new("height".to_string()));
                assert_eq!(c.value, Value::Integer(20));
            }
            _ => panic!("Expected const"),
        }
    }

    // Choice rules
    #[test]
    fn test_parse_simple_choice() {
        let input = "{ solid(1, 2); solid(2, 3) }.";
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);

        match &program.statements[0] {
            Statement::ChoiceRule(choice) => {
                assert_eq!(choice.lower_bound, None);
                assert_eq!(choice.upper_bound, None);
                assert_eq!(choice.elements.len(), 2);
                assert_eq!(choice.body.len(), 0);
                assert_eq!(
                    choice.elements[0].atom.predicate,
                    Intern::new("solid".to_string())
                );
                assert_eq!(choice.elements[0].condition.len(), 0);
            }
            _ => panic!("Expected choice rule"),
        }
    }

    #[test]
    fn test_parse_choice_with_bounds() {
        let input = "1 { item(a); item(b); item(c) } 2.";
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::ChoiceRule(choice) => {
                assert_eq!(choice.lower_bound, Some(Term::Constant(Value::Integer(1))));
                assert_eq!(choice.upper_bound, Some(Term::Constant(Value::Integer(2))));
                assert_eq!(choice.elements.len(), 3);
            }
            _ => panic!("Expected choice rule"),
        }
    }

    #[test]
    fn test_parse_choice_with_lower_bound() {
        let input = "2 { selected(X) }.";
        let result = parse_program(input);
        assert!(result.is_ok());

        match &result.unwrap().statements[0] {
            Statement::ChoiceRule(choice) => {
                assert_eq!(choice.lower_bound, Some(Term::Constant(Value::Integer(2))));
                assert_eq!(choice.upper_bound, None);
            }
            _ => panic!("Expected choice rule"),
        }
    }

    #[test]
    fn test_parse_choice_with_condition() {
        let input = "{ selected(X) : item(X) }.";
        let result = parse_program(input);
        assert!(result.is_ok());

        match &result.unwrap().statements[0] {
            Statement::ChoiceRule(choice) => {
                assert_eq!(choice.elements.len(), 1);
                assert_eq!(choice.elements[0].condition.len(), 1);
                match &choice.elements[0].condition[0] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new("item".to_string()));
                    }
                    _ => panic!("Expected positive literal"),
                }
            }
            _ => panic!("Expected choice rule"),
        }
    }

    #[test]
    fn test_parse_choice_with_body() {
        let input = "{ selected(X) : item(X) } :- room(R).";
        let result = parse_program(input);
        assert!(result.is_ok());

        match &result.unwrap().statements[0] {
            Statement::ChoiceRule(choice) => {
                assert_eq!(choice.body.len(), 1);
                match &choice.body[0] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new("room".to_string()));
                    }
                    _ => panic!("Expected positive literal"),
                }
            }
            _ => panic!("Expected choice rule"),
        }
    }

    // Full program parsing tests
    #[test]
    fn test_parse_empty_program() {
        let result = parse_program("");
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 0);
    }

    #[test]
    fn test_parse_program_with_multiple_statements() {
        let input = r#"
            parent(john, mary).
            parent(mary, alice).
            ancestor(X, Y) :- parent(X, Y).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 3);
    }

    #[test]
    fn test_parse_program_with_comments() {
        let input = r#"
            % This is a comment
            parent(john, mary). % Inline comment
            % Another comment
            parent(mary, alice).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 2);
    }

    // Block comment tests
    #[test]
    fn test_parse_simple_block_comment() {
        let input = "/* This is a block comment */ parent(john, mary).";
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
    }

    #[test]
    fn test_parse_multiline_block_comment() {
        let input = r#"
            /* This is a
               multi-line
               block comment */
            parent(john, mary).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
    }

    #[test]
    fn test_parse_block_comment_between_statements() {
        let input = r#"
            parent(john, mary).
            /* Block comment in the middle */
            parent(mary, alice).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 2);
    }

    #[test]
    fn test_parse_mixed_comments() {
        let input = r#"
            % Line comment
            /* Block comment */
            parent(john, mary).
            /* Another block
               comment */
            % Another line comment
            parent(mary, alice).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 2);
    }

    #[test]
    fn test_parse_block_comment_with_special_chars() {
        let input = r#"
            /* Comment with special chars: %, :-, ::, (). */
            parent(john, mary).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
    }

    #[test]
    fn test_parse_empty_block_comment() {
        let input = "/**/ parent(john, mary).";
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
    }

    // Comprehensive datatype test
    #[test]
    fn test_parse_all_datatypes_in_program() {
        let input = r#"
            % All datatypes example
            data(42).                    /* integer */
            data(-10).                   /* negative integer */
            data(3.14).                  /* float */
            data(-2.5).                  /* negative float */
            data(true).                  /* boolean true */
            data(false).                 /* boolean false */
            data("hello").               /* string */
            data(atom).                  /* atom */
            process(X, Y, _tmp).         /* variables */
            mixed(42, 3.14, true, "hi", atom, X).  /* all together */
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 10);
    }

    // Range parsing tests
    #[test]
    fn test_parse_fact_with_range() {
        let result = parse_program("dim(1..10).");
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);

        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.predicate, Intern::new("dim".to_string()));
                assert_eq!(fact.atom.terms.len(), 1);
                assert_eq!(fact.atom.terms[0], range(1, 10));
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_fact_with_multiple_ranges() {
        let result = parse_program("cell(1..5, 1..10).");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.predicate, Intern::new("cell".to_string()));
                assert_eq!(fact.atom.terms.len(), 2);
                assert_eq!(fact.atom.terms[0], range(1, 5));
                assert_eq!(fact.atom.terms[1], range(1, 10));
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_fact_with_mixed_terms_and_range() {
        let result = parse_program("edge(a, 1..10, b).");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.terms.len(), 3);
                assert_eq!(
                    fact.atom.terms[0],
                    Term::Constant(Value::Atom(Intern::new("a".to_string())))
                );
                assert_eq!(fact.atom.terms[1], range(1, 10));
                assert_eq!(
                    fact.atom.terms[2],
                    Term::Constant(Value::Atom(Intern::new("b".to_string())))
                );
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_range_in_compound_term() {
        let result = parse_program("data(item(1..5)).");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.terms.len(), 1);
                match &fact.atom.terms[0] {
                    Term::Compound(functor, args) => {
                        assert_eq!(*functor, Intern::new("item".to_string()));
                        assert_eq!(args.len(), 1);
                        assert_eq!(args[0], range(1, 5));
                    }
                    _ => panic!("Expected compound term"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_program_with_consts_and_ranges() {
        let input = r#"
            #const width = 10.
            #const height = 5.

            dim(1..10).
            cell(1..10, 1..5).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 4);

        // First two should be const decls
        match &program.statements[0] {
            Statement::ConstDecl(c) => {
                assert_eq!(c.name, Intern::new("width".to_string()));
            }
            _ => panic!("Expected const"),
        }

        // Third should be dim(1..width) - note: width won't be substituted yet
        match &program.statements[2] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.predicate, Intern::new("dim".to_string()));
                // Range still contains literal values until expansion
                assert_eq!(fact.atom.terms[0], range(1, 10));
            }
            _ => panic!("Expected fact"),
        }
    }

    // Operator and arithmetic parsing tests
    #[test]
    fn test_parse_operator_as_predicate() {
        let result = parse_program(">(X, 3).");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.predicate, Intern::new(">".to_string()));
                assert_eq!(fact.atom.terms.len(), 2);
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_infix_comparison() {
        let result = parse_program("test :- X > 3.");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Rule(rule) => {
                assert_eq!(rule.body.len(), 1);
                match &rule.body[0] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new(">".to_string()));
                        assert_eq!(atom.terms.len(), 2);
                        // Check that terms are correct
                        match &atom.terms[0] {
                            Term::Variable(v) => assert_eq!(*v, Intern::new("X".to_string())),
                            _ => panic!("Expected variable X"),
                        }
                        match &atom.terms[1] {
                            Term::Constant(Value::Integer(3)) => {}
                            _ => panic!("Expected integer 3, got {:?}", atom.terms[1]),
                        }
                    }
                    _ => panic!("Expected positive literal"),
                }
            }
            _ => panic!("Expected rule"),
        }
    }

    #[test]
    fn test_parse_arithmetic_addition() {
        // Note: X + Y is parsed as a compound term +(X, Y)
        let result = parse_program("test(X + Y).");
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Fact(fact) => {
                assert_eq!(fact.atom.predicate, Intern::new("test".to_string()));
                assert_eq!(fact.atom.terms.len(), 1);
                match &fact.atom.terms[0] {
                    Term::Compound(functor, args) => {
                        assert_eq!(*functor, Intern::new("+".to_string()));
                        assert_eq!(args.len(), 2);
                    }
                    _ => panic!("Expected compound term for addition"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_arithmetic_precedence() {
        // X + Y * 2 should parse as +(X, *(Y, 2))
        let result = parse_program("test(X + Y * 2).");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Fact(fact) => {
                match &fact.atom.terms[0] {
                    Term::Compound(plus, args) => {
                        assert_eq!(*plus, Intern::new("+".to_string()));
                        assert_eq!(args.len(), 2);
                        // Right side should be *(Y, 2)
                        match &args[1] {
                            Term::Compound(mult, mult_args) => {
                                assert_eq!(*mult, Intern::new("*".to_string()));
                                assert_eq!(mult_args.len(), 2);
                            }
                            _ => panic!("Expected multiplication on right side"),
                        }
                    }
                    _ => panic!("Expected compound term"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_parenthesized_arithmetic() {
        // (X + Y) * 2 should parse as *((+(X, Y)), 2)
        let result = parse_program("test((X + Y) * 2).");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Fact(fact) => {
                match &fact.atom.terms[0] {
                    Term::Compound(mult, args) => {
                        assert_eq!(*mult, Intern::new("*".to_string()));
                        assert_eq!(args.len(), 2);
                        // Left side should be +(X, Y)
                        match &args[0] {
                            Term::Compound(plus, plus_args) => {
                                assert_eq!(*plus, Intern::new("+".to_string()));
                                assert_eq!(plus_args.len(), 2);
                            }
                            _ => panic!("Expected addition on left side"),
                        }
                    }
                    _ => panic!("Expected compound term"),
                }
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_comparison_operators() {
        let operators = vec!["<", ">", "<=", ">=", "=", "!="];
        for op in operators {
            let program = format!("test :- X {} 5.", op);
            let result = parse_program(&program);
            assert!(result.is_ok(), "Failed to parse operator: {}", op);

            match &result.unwrap().statements[0] {
                Statement::Rule(rule) => match &rule.body[0] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new(op.to_string()));
                    }
                    _ => panic!("Expected positive literal"),
                },
                _ => panic!("Expected rule"),
            }
        }
    }

    #[test]
    fn test_parse_arithmetic_in_rule_body() {
        let result = parse_program("result(Z) :- X > 0, Y = X + 1, Z = Y * 2.");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Rule(rule) => {
                assert_eq!(rule.body.len(), 3);

                // First: X > 0
                match &rule.body[0] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new(">".to_string()));
                    }
                    _ => panic!("Expected comparison"),
                }

                // Second: Y = X + 1
                match &rule.body[1] {
                    Literal::Positive(atom) => {
                        assert_eq!(atom.predicate, Intern::new("=".to_string()));
                        match &atom.terms[1] {
                            Term::Compound(plus, _) => {
                                assert_eq!(*plus, Intern::new("+".to_string()));
                            }
                            _ => panic!("Expected addition"),
                        }
                    }
                    _ => panic!("Expected comparison"),
                }
            }
            _ => panic!("Expected rule"),
        }
    }

    #[test]
    fn test_parse_negated_comparison() {
        let result = parse_program("test :- not X > 10.");
        assert!(result.is_ok());
        let program = result.unwrap();

        match &program.statements[0] {
            Statement::Rule(rule) => {
                assert_eq!(rule.body.len(), 1);
                match &rule.body[0] {
                    Literal::Negative(atom) => {
                        assert_eq!(atom.predicate, Intern::new(">".to_string()));
                    }
                    _ => panic!("Expected negative literal"),
                }
            }
            _ => panic!("Expected rule"),
        }
    }

    // Constant names in cardinality bounds tests
    #[test]
    fn test_parse_choice_with_constant_lower_bound() {
        let input = "min { selected(X) : item(X) }.";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        match &result.unwrap().statements[0] {
            Statement::ChoiceRule(choice) => {
                // Lower bound should be parsed as a constant name
                assert!(choice.lower_bound.is_some());
                match &choice.lower_bound {
                    Some(Term::Constant(Value::Atom(name))) => {
                        assert_eq!(*name, Intern::new("min".to_string()));
                    }
                    _ => panic!(
                        "Expected atom constant for lower bound, got {:?}",
                        choice.lower_bound
                    ),
                }
                assert!(choice.upper_bound.is_none());
            }
            _ => panic!("Expected choice rule"),
        }
    }

    #[test]
    fn test_parse_choice_with_constant_upper_bound() {
        let input = "{ selected(X) : item(X) } max.";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        match &result.unwrap().statements[0] {
            Statement::ChoiceRule(choice) => {
                assert!(choice.lower_bound.is_none());
                assert!(choice.upper_bound.is_some());
                match &choice.upper_bound {
                    Some(Term::Constant(Value::Atom(name))) => {
                        assert_eq!(*name, Intern::new("max".to_string()));
                    }
                    _ => panic!(
                        "Expected atom constant for upper bound, got {:?}",
                        choice.upper_bound
                    ),
                }
            }
            _ => panic!("Expected choice rule"),
        }
    }

    #[test]
    fn test_parse_choice_with_constant_both_bounds() {
        let input = "min { selected(X) : item(X) } max.";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        match &result.unwrap().statements[0] {
            Statement::ChoiceRule(choice) => {
                assert!(choice.lower_bound.is_some());
                assert!(choice.upper_bound.is_some());
                match &choice.lower_bound {
                    Some(Term::Constant(Value::Atom(name))) => {
                        assert_eq!(*name, Intern::new("min".to_string()));
                    }
                    _ => panic!("Expected atom constant for lower bound"),
                }
                match &choice.upper_bound {
                    Some(Term::Constant(Value::Atom(name))) => {
                        assert_eq!(*name, Intern::new("max".to_string()));
                    }
                    _ => panic!("Expected atom constant for upper bound"),
                }
            }
            _ => panic!("Expected choice rule"),
        }
    }

    #[test]
    fn test_parse_choice_with_mixed_bounds() {
        // Integer lower, constant upper
        let input = "2 { selected(X) : item(X) } max.";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        match &result.unwrap().statements[0] {
            Statement::ChoiceRule(choice) => {
                match &choice.lower_bound {
                    Some(Term::Constant(Value::Integer(2))) => {}
                    _ => panic!(
                        "Expected integer 2 for lower bound, got {:?}",
                        choice.lower_bound
                    ),
                }
                match &choice.upper_bound {
                    Some(Term::Constant(Value::Atom(name))) => {
                        assert_eq!(*name, Intern::new("max".to_string()));
                    }
                    _ => panic!("Expected atom constant for upper bound"),
                }
            }
            _ => panic!("Expected choice rule"),
        }
    }

    // Query parsing tests
    #[test]
    fn test_parse_query_ground() {
        let result = parse_query("?- parent(john, mary).");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let query = result.unwrap();
        assert_eq!(query.body.len(), 1);
        match &query.body[0] {
            Literal::Positive(atom) => {
                assert_eq!(atom.predicate, Intern::new("parent".to_string()));
                assert_eq!(atom.terms.len(), 2);
            }
            _ => panic!("Expected positive literal"),
        }
    }

    #[test]
    fn test_parse_query_with_variable() {
        let result = parse_query("?- parent(X, mary).");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let query = result.unwrap();
        assert_eq!(query.body.len(), 1);
        match &query.body[0] {
            Literal::Positive(atom) => {
                assert_eq!(atom.predicate, Intern::new("parent".to_string()));
                match &atom.terms[0] {
                    Term::Variable(v) => assert_eq!(*v, Intern::new("X".to_string())),
                    _ => panic!("Expected variable"),
                }
            }
            _ => panic!("Expected positive literal"),
        }
    }

    #[test]
    fn test_parse_query_multiple_literals() {
        let result = parse_query("?- parent(X, Y), parent(Y, Z).");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let query = result.unwrap();
        assert_eq!(query.body.len(), 2);
    }

    #[test]
    fn test_parse_query_with_negation() {
        let result = parse_query("?- parent(X, Y), not dead(X).");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let query = result.unwrap();
        assert_eq!(query.body.len(), 2);
        assert!(query.body[0].is_positive());
        assert!(query.body[1].is_negative());
    }

    #[test]
    fn test_parse_query_complex() {
        let result = parse_query("?- ancestor(X, Z), parent(X, Y), parent(Y, Z), not dead(Z).");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let query = result.unwrap();
        assert_eq!(query.body.len(), 4);
    }

    #[test]
    fn test_parse_query_with_builtin() {
        let result = parse_query("?- age(X, A), A > 18.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let query = result.unwrap();
        assert_eq!(query.body.len(), 2);
    }

    #[test]
    fn test_parse_query_zero_arity() {
        let result = parse_query("?- running.");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let query = result.unwrap();
        assert_eq!(query.body.len(), 1);
    }

    #[test]
    fn test_parse_simple_test_block() {
        let input = r#"
            #test "basic test" {
                parent(john, mary).

                ?- parent(john, mary).
                + true.
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);

        match &program.statements[0] {
            Statement::Test(test_block) => {
                assert_eq!(test_block.name, "basic test");
                assert_eq!(test_block.statements.len(), 1);
                assert_eq!(test_block.test_cases.len(), 1);

                // Check test case
                let test_case = &test_block.test_cases[0];
                assert_eq!(test_case.query.body.len(), 1);
                assert_eq!(test_case.positive_assertions.len(), 1);
                assert_eq!(test_case.negative_assertions.len(), 0);
            }
            _ => panic!("Expected test block"),
        }
    }

    #[test]
    fn test_parse_test_block_with_rules() {
        let input = r#"
            #test "transitive closure" {
                edge(a, b).
                edge(b, c).
                path(X, Y) :- edge(X, Y).
                path(X, Z) :- path(X, Y), edge(Y, Z).

                ?- path(a, c).
                + path(a, c).
                - path(c, a).
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Test(test_block) => {
                assert_eq!(test_block.name, "transitive closure");
                assert_eq!(test_block.statements.len(), 4); // 2 facts + 2 rules
                assert_eq!(test_block.test_cases.len(), 1);

                let test_case = &test_block.test_cases[0];
                assert_eq!(test_case.positive_assertions.len(), 1);
                assert_eq!(test_case.negative_assertions.len(), 1);
            }
            _ => panic!("Expected test block"),
        }
    }

    #[test]
    fn test_parse_test_block_multiple_queries() {
        let input = r#"
            #test "multiple queries" {
                value(1).
                value(2).

                ?- value(1).
                + true.

                ?- value(X).
                + value(1).
                + value(2).
                - value(3).
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Test(test_block) => {
                assert_eq!(test_block.test_cases.len(), 2);

                // First query
                assert_eq!(test_block.test_cases[0].positive_assertions.len(), 1);

                // Second query
                assert_eq!(test_block.test_cases[1].positive_assertions.len(), 2);
                assert_eq!(test_block.test_cases[1].negative_assertions.len(), 1);
            }
            _ => panic!("Expected test block"),
        }
    }

    #[test]
    fn test_parse_test_block_with_comments() {
        let input = r#"
            #test "comments ok" {
                parent(john, mary).

                % inline comment inside test block
                ?- parent(john, mary).
                + true.
            }
        "#;
        let result = parse_program(input);
        assert!(
            result.is_ok(),
            "Parser should allow comments in test blocks, but got: {:?}",
            result.err()
        );

        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);

        match &program.statements[0] {
            Statement::Test(test_block) => {
                assert_eq!(test_block.name, "comments ok");
                assert_eq!(test_block.statements.len(), 1);
                assert_eq!(test_block.test_cases.len(), 1);
            }
            _ => panic!("Expected test block"),
        }
    }

    #[test]
    fn test_parse_test_block_with_constants() {
        let input = r#"
            #test "with constants" {
                #const max_val = 10.
                limit(max_val).

                ?- limit(10).
                + true.
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        let program = result.unwrap();
        match &program.statements[0] {
            Statement::Test(test_block) => {
                assert_eq!(test_block.statements.len(), 2); // const decl + fact
            }
            _ => panic!("Expected test block"),
        }
    }

    #[test]
    fn test_parse_multiple_test_blocks() {
        let input = r#"
            #test "test 1" {
                a(1).
                ?- a(1).
                + true.
            }

            #test "test 2" {
                b(2).
                ?- b(2).
                + true.
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        let program = result.unwrap();
        assert_eq!(program.statements.len(), 2);

        match (&program.statements[0], &program.statements[1]) {
            (Statement::Test(test1), Statement::Test(test2)) => {
                assert_eq!(test1.name, "test 1");
                assert_eq!(test2.name, "test 2");
            }
            _ => panic!("Expected two test blocks"),
        }
    }

    #[test]
    fn test_parse_minimize_simple() {
        // Test: #minimize { X : cost(X) }.
        let result = parse_with(program(), "#minimize { X : cost(X) }.");
        assert!(result.is_ok(), "Failed to parse minimize: {:?}", result.err());

        let prog = result.unwrap();
        assert_eq!(prog.statements.len(), 1);

        match &prog.statements[0] {
            Statement::Optimize(opt) => {
                assert_eq!(opt.direction, OptimizeDirection::Minimize);
                assert_eq!(opt.terms.len(), 1);
                // Should have one term with cost(X) condition
            }
            _ => panic!("Expected Optimize statement"),
        }
    }

    #[test]
    fn test_parse_maximize_simple() {
        // Test: #maximize { X : value(X) }.
        let result = parse_with(program(), "#maximize { X : value(X) }.");
        assert!(result.is_ok(), "Failed to parse maximize: {:?}", result.err());

        let prog = result.unwrap();
        assert_eq!(prog.statements.len(), 1);

        match &prog.statements[0] {
            Statement::Optimize(opt) => {
                assert_eq!(opt.direction, OptimizeDirection::Maximize);
                assert_eq!(opt.terms.len(), 1);
            }
            _ => panic!("Expected Optimize statement"),
        }
    }

    #[test]
    fn test_parse_minimize_with_weight() {
        // Test: #minimize { C*X : cost(X, C) }.
        let result = parse_with(program(), "#minimize { C*X : cost(X, C) }.");
        assert!(result.is_ok(), "Failed to parse weighted minimize: {:?}", result.err());

        let prog = result.unwrap();
        match &prog.statements[0] {
            Statement::Optimize(opt) => {
                assert_eq!(opt.direction, OptimizeDirection::Minimize);
                // Should have weighted term
            }
            _ => panic!("Expected Optimize statement"),
        }
    }

    #[test]
    fn test_parse_minimize_integer() {
        // Test: #minimize { 5 }.
        let result = parse_with(program(), "#minimize { 5 }.");
        assert!(result.is_ok(), "Failed to parse integer minimize: {:?}", result.err());

        let prog = result.unwrap();
        match &prog.statements[0] {
            Statement::Optimize(opt) => {
                assert_eq!(opt.direction, OptimizeDirection::Minimize);
                assert_eq!(opt.terms.len(), 1);
            }
            _ => panic!("Expected Optimize statement"),
        }
    }
}
