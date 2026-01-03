use chumsky::prelude::*;
use std::fmt;

use crate::Span;

pub type SpannedToken = (Token, Span);
pub type LexError = Simple<char, Span>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Keyword {
    Const,
    Test,
    Not,
    Minimize,
    Maximize,
}

impl fmt::Display for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Keyword::Const => "const",
            Keyword::Test => "test",
            Keyword::Not => "not",
            Keyword::Minimize => "minimize",
            Keyword::Maximize => "maximize",
        };
        write!(f, "{}", text)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Token {
    Ident(String),
    Variable(String),
    Number(String),
    String(String),
    Operator(String),
    Keyword(Keyword),
    RuleSep,
    RangeDots,
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Dot,
    Colon,
    Semicolon,
    Question,
    Hash,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Ident(text) => write!(f, "{}", text),
            Token::Variable(text) => write!(f, "{}", text),
            Token::Number(text) => write!(f, "{}", text),
            Token::String(text) => write!(f, "\"{}\"", text),
            Token::Operator(text) => write!(f, "{}", text),
            Token::Keyword(keyword) => write!(f, "{}", keyword),
            Token::RuleSep => write!(f, ":-"),
            Token::RangeDots => write!(f, ".."),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::Comma => write!(f, ","),
            Token::Dot => write!(f, "."),
            Token::Colon => write!(f, ":"),
            Token::Semicolon => write!(f, ";"),
            Token::Question => write!(f, "?"),
            Token::Hash => write!(f, "#"),
        }
    }
}

fn string_literal() -> impl Parser<char, String, Error = LexError> + Clone {
    let escape_sequence = just('\\').ignore_then(choice((
        just('"').to('"'),
        just('n').to('\n'),
        just('t').to('\t'),
        just('\\').to('\\'),
    )));

    let string_char = choice((
        escape_sequence,
        filter(|c| *c != '"' && *c != '\\' && *c != '\n'),
    ));

    just('"')
        .ignore_then(string_char.repeated())
        .then_ignore(just('"'))
        .collect::<String>()
        .labelled("string")
}

fn number_literal() -> impl Parser<char, String, Error = LexError> + Clone {
    let digits = text::int(10);

    digits
        .then(just('.').ignore_then(text::digits(10)).or_not())
        .map(|(whole, frac)| {
            if let Some(frac) = frac {
                format!("{}.{}", whole, frac)
            } else {
                whole
            }
        })
        .labelled("number")
}

fn identifier() -> impl Parser<char, Token, Error = LexError> + Clone {
    text::ident()
        .map(|ident: String| match ident.as_str() {
            "const" => Token::Keyword(Keyword::Const),
            "test" => Token::Keyword(Keyword::Test),
            "not" => Token::Keyword(Keyword::Not),
            "minimize" => Token::Keyword(Keyword::Minimize),
            "maximize" => Token::Keyword(Keyword::Maximize),
            _ => {
                let first = ident.chars().next().unwrap();
                if first.is_uppercase() {
                    Token::Variable(ident)
                } else {
                    Token::Ident(ident)
                }
            }
        })
        .labelled("identifier")
}

fn line_comment() -> impl Parser<char, (), Error = LexError> + Clone {
    just('%')
        .then(filter(|c| *c != '\n').repeated())
        .ignored()
        .labelled("line comment")
}

fn block_comment() -> impl Parser<char, (), Error = LexError> + Clone {
    just("/*")
        .then(
            choice((
                filter(|c| *c != '*').ignored(),
                just('*').then(filter(|c| *c != '/')).ignored(),
            ))
            .repeated()
            .then(just("*/")),
        )
        .ignored()
        .labelled("block comment")
}

fn comment() -> impl Parser<char, (), Error = LexError> + Clone {
    block_comment().or(line_comment()).labelled("comment")
}

fn spacing() -> impl Parser<char, (), Error = LexError> + Clone {
    comment()
        .or(text::whitespace().at_least(1).ignored())
        .repeated()
        .ignored()
}

pub fn lexer() -> impl Parser<char, Vec<SpannedToken>, Error = LexError> + Clone {
    let punct = choice((
        just(":-").to(Token::RuleSep),
        just("..").to(Token::RangeDots),
        just("<=").to(Token::Operator("<=".to_string())),
        just(">=").to(Token::Operator(">=".to_string())),
        just("!=").to(Token::Operator("!=".to_string())),
        just("\\=").to(Token::Operator("\\=".to_string())),
        just("=<").to(Token::Operator("=<".to_string())),
        just("=").to(Token::Operator("=".to_string())),
        just("<").to(Token::Operator("<".to_string())),
        just(">").to(Token::Operator(">".to_string())),
        just("+").to(Token::Operator("+".to_string())),
        just("-").to(Token::Operator("-".to_string())),
        just("*").to(Token::Operator("*".to_string())),
        just("/").to(Token::Operator("/".to_string())),
        just('(').to(Token::LParen),
        just(')').to(Token::RParen),
        just('{').to(Token::LBrace),
        just('}').to(Token::RBrace),
        just(',').to(Token::Comma),
        just('.').to(Token::Dot),
        just(':').to(Token::Colon),
        just(';').to(Token::Semicolon),
        just('?').to(Token::Question),
        just('#').to(Token::Hash),
    ));

    let token = choice((
        string_literal().map(Token::String),
        number_literal().map(Token::Number),
        identifier(),
        punct,
    ))
    .map_with_span(|token, span| (token, span))
    .padded_by(spacing());

    token.repeated().then_ignore(end())
}
