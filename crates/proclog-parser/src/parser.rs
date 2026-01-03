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

    choice((
        variable,
        number_const,
        string_const,
        parens,
        compound_or_atom,
    ))
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
        .then(choice((
            operator_token("<=").to("<="),
            operator_token(">=").to(">="),
            operator_token("!=").to("!="),
            operator_token("\\=").to("\\="),
            operator_token("=<").to("=<"),
            operator_token("=").to("="),
            operator_token("<").to("<"),
            operator_token(">").to(">"),
        )))
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
        .then(literal().separated_by(token(Token::Comma)).at_least(1))
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
    choice((operator_token("+").to(true), operator_token("-").to(false)))
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
#[path = "../tests/unit/parser_tests.rs"]
mod tests;
