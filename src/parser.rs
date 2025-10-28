use chumsky::prelude::*;
use internment::Intern;

use crate::ast::*;

type ParseError = Simple<char>;

/// Parse a number (integer or float)
fn number() -> impl Parser<char, Value, Error = ParseError> + Clone {
    let sign = just('-').or_not();

    let digits = text::int(10);

    sign.then(digits)
        .then(just('.').ignore_then(text::digits(10)).or_not())
        .try_map(|((sign, whole), frac), span| {
            let sign_str = if sign.is_some() { "-" } else { "" };

            if let Some(frac) = frac {
                // It's a float
                let num_str = format!("{}{}.{}", sign_str, whole, frac);
                num_str.parse::<f64>()
                    .map(Value::Float)
                    .map_err(|_| ParseError::custom(span, "invalid float"))
            } else {
                // It's an integer
                let num_str = format!("{}{}", sign_str, whole);
                num_str.parse::<i64>()
                    .map(Value::Integer)
                    .map_err(|_| ParseError::custom(span, "invalid integer"))
            }
        })
        .labelled("number")
}

/// Parse a lowercase identifier (starts with lowercase, not underscore)
fn lowercase_ident() -> impl Parser<char, String, Error = ParseError> + Clone {
    text::ident().try_map(|s: String, span| {
        if s.chars().next().unwrap().is_lowercase() {
            Ok(s)
        } else {
            Err(ParseError::custom(span, "expected lowercase identifier"))
        }
    })
    .labelled("lowercase identifier")
}

/// Parse an uppercase identifier (variable - starts with uppercase or underscore)
fn uppercase_ident() -> impl Parser<char, String, Error = ParseError> + Clone {
    text::ident().try_map(|s: String, span| {
        let first = s.chars().next().unwrap();
        if first.is_uppercase() || first == '_' {
            Ok(s)
        } else {
            Err(ParseError::custom(span, "expected uppercase identifier or underscore (variable)"))
        }
    })
    .labelled("variable")
}

/// Parse a string literal
fn string_literal() -> impl Parser<char, String, Error = ParseError> + Clone {
    just('"')
        .ignore_then(filter(|c| *c != '"').repeated())
        .then_ignore(just('"'))
        .collect::<String>()
        .labelled("string")
}

/// Parse a term (variable, constant, or compound)
fn term() -> impl Parser<char, Term, Error = ParseError> + Clone {
    recursive(|term| {
        let variable = uppercase_ident()
            .map(|s| Term::Variable(Intern::new(s)));

        // Numbers (int or float, positive or negative)
        let number_const = number()
            .map(|v| Term::Constant(v));

        let string_const = string_literal()
            .map(|s| Term::Constant(Value::String(Intern::new(s))));

        // Boolean, atom, or compound term
        let bool_atom_or_compound = lowercase_ident()
            .then(
                term.clone()
                    .separated_by(just(',').padded())
                    .delimited_by(just('('), just(')'))
                    .or_not()
            )
            .map(|(name, args)| {
                if let Some(args) = args {
                    // It's a compound term
                    Term::Compound(Intern::new(name), args)
                } else {
                    // Check if it's a boolean keyword
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
            bool_atom_or_compound,
        ))
        .padded()
    })
    .labelled("term")
}

/// Parse an atom
fn atom() -> impl Parser<char, Atom, Error = ParseError> + Clone {
    lowercase_ident()
        .then(
            term()
                .separated_by(just(',').padded())
                .delimited_by(just('('), just(')'))
                .or_not()
        )
        .map(|(predicate, terms)| Atom {
            predicate: Intern::new(predicate),
            terms: terms.unwrap_or_default(),
        })
        .labelled("atom")
}

/// Parse a literal (positive or negative atom)
fn literal() -> impl Parser<char, Literal, Error = ParseError> + Clone {
    just("not")
        .padded()
        .ignore_then(atom())
        .map(Literal::Negative)
        .or(atom().map(Literal::Positive))
        .labelled("literal")
}

/// Parse a line comment (starts with % and goes to end of line)
fn line_comment() -> impl Parser<char, (), Error = ParseError> + Clone {
    just('%')
        .then(filter(|c| *c != '\n').repeated())
        .ignored()
        .labelled("line comment")
}

/// Parse a block comment (/* ... */)
fn block_comment() -> impl Parser<char, (), Error = ParseError> + Clone {
    just("/*")
        .then(
            // Match characters until we find */
            // We match either:
            // - Any character that is not '*'
            // - Or '*' followed by a character that is not '/'
            choice((
                filter(|c| *c != '*').ignored(),
                just('*').then(filter(|c| *c != '/')).ignored(),
            ))
            .repeated()
            .then(just("*/"))
        )
        .ignored()
        .labelled("block comment")
}

/// Parse any kind of comment (line or block)
fn comment() -> impl Parser<char, (), Error = ParseError> + Clone {
    block_comment().or(line_comment()).labelled("comment")
}

/// Parse a fact
fn fact() -> impl Parser<char, Statement, Error = ParseError> + Clone {
    atom()
        .then_ignore(just('.').padded())
        .map(|atom| Statement::Fact(Fact { atom }))
        .labelled("fact")
}

/// Parse a rule
fn rule() -> impl Parser<char, Statement, Error = ParseError> + Clone {
    atom()
        .then_ignore(just(":-").padded())
        .then(
            literal()
                .separated_by(just(',').padded())
                .at_least(1)
        )
        .then_ignore(just('.').padded())
        .map(|(head, body)| Statement::Rule(Rule { head, body }))
        .labelled("rule")
}

/// Parse a constraint
fn constraint() -> impl Parser<char, Statement, Error = ParseError> + Clone {
    just(":-")
        .padded()
        .ignore_then(
            literal()
                .separated_by(just(',').padded())
                .at_least(1)
        )
        .then_ignore(just('.').padded())
        .map(|body| Statement::Constraint(Constraint { body }))
        .labelled("constraint")
}

/// Parse a probabilistic fact
fn prob_fact() -> impl Parser<char, Statement, Error = ParseError> + Clone {
    // Parse a float probability (0.0 to 1.0)
    text::digits(10)
        .then_ignore(just('.'))
        .then(text::digits(10))
        .try_map(|(whole, frac), span| {
            let prob_str = format!("{}.{}", whole, frac);
            match prob_str.parse::<f64>() {
                Ok(p) if (0.0..=1.0).contains(&p) => Ok(p),
                Ok(_) => Err(ParseError::custom(span, "probability must be between 0.0 and 1.0")),
                Err(_) => Err(ParseError::custom(span, "invalid probability")),
            }
        })
        .then_ignore(just("::").padded())
        .then(atom())
        .then_ignore(just('.').padded())
        .map(|(probability, atom)| Statement::ProbFact(ProbFact { probability, atom }))
        .labelled("probabilistic fact")
}

/// Parse a statement
fn statement() -> impl Parser<char, Statement, Error = ParseError> + Clone {
    choice((
        prob_fact(),
        constraint(),
        rule(),
        fact(),
    ))
    .labelled("statement")
}

/// Parse a program
pub fn program() -> impl Parser<char, Program, Error = ParseError> + Clone {
    statement()
        .padded_by(comment().or(text::whitespace().at_least(1).ignored()).repeated())
        .repeated()
        .map(|statements| Program { statements })
        .then_ignore(end())
        .padded()
        .labelled("program")
}

/// Helper function to parse with better error handling
pub fn parse_program(input: &str) -> Result<Program, Vec<ParseError>> {
    program().parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Term parsing tests - Datatypes

    // Integers
    #[test]
    fn test_parse_positive_integer() {
        let result = term().parse("42");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Integer(42)));
    }

    #[test]
    fn test_parse_negative_integer() {
        let result = term().parse("-42");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Integer(-42)));
    }

    #[test]
    fn test_parse_zero() {
        let result = term().parse("0");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Integer(0)));
    }

    // Floats
    #[test]
    fn test_parse_positive_float() {
        let result = term().parse("3.14");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Float(3.14)));
    }

    #[test]
    fn test_parse_negative_float() {
        let result = term().parse("-2.5");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Float(-2.5)));
    }

    #[test]
    fn test_parse_float_with_zero_decimal() {
        let result = term().parse("10.0");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Float(10.0)));
    }

    // Booleans
    #[test]
    fn test_parse_boolean_true() {
        let result = term().parse("true");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Boolean(true)));
    }

    #[test]
    fn test_parse_boolean_false() {
        let result = term().parse("false");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Constant(Value::Boolean(false)));
    }

    // Variables
    #[test]
    fn test_parse_uppercase_variable() {
        let result = term().parse("X");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Variable(Intern::new("X".to_string())));
    }

    #[test]
    fn test_parse_multichar_variable() {
        let result = term().parse("Player");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Variable(Intern::new("Player".to_string())));
    }

    #[test]
    fn test_parse_underscore_variable() {
        let result = term().parse("_tmp");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Variable(Intern::new("_tmp".to_string())));
    }

    // Strings
    #[test]
    fn test_parse_string_constant() {
        let result = term().parse("\"hello\"");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Constant(Value::String(Intern::new("hello".to_string())))
        );
    }

    #[test]
    fn test_parse_empty_string() {
        let result = term().parse("\"\"");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Constant(Value::String(Intern::new("".to_string())))
        );
    }

    // Atoms
    #[test]
    fn test_parse_atom_constant() {
        let result = term().parse("john");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Constant(Value::Atom(Intern::new("john".to_string())))
        );
    }

    #[test]
    fn test_parse_compound_term() {
        let result = term().parse("f(a, b)");
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

    // Atom parsing tests
    #[test]
    fn test_parse_atom_no_args() {
        let result = atom().parse("foo");
        assert!(result.is_ok());
        let expected = Atom {
            predicate: Intern::new("foo".to_string()),
            terms: vec![],
        };
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_parse_atom_with_args() {
        let result = atom().parse("parent(john, mary)");
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
        let result = atom().parse("parent(X, Y)");
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
        let result = literal().parse("parent(X, Y)");
        assert!(result.is_ok());
        match result.unwrap() {
            Literal::Positive(atom) => {
                assert_eq!(atom.predicate, Intern::new("parent".to_string()));
                assert_eq!(atom.terms.len(), 2);
            }
            Literal::Negative(_) => panic!("Expected positive literal"),
        }
    }

    #[test]
    fn test_parse_negative_literal() {
        let result = literal().parse("not parent(X, Y)");
        assert!(result.is_ok());
        match result.unwrap() {
            Literal::Negative(atom) => {
                assert_eq!(atom.predicate, Intern::new("parent".to_string()));
                assert_eq!(atom.terms.len(), 2);
            }
            Literal::Positive(_) => panic!("Expected negative literal"),
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
    fn test_parse_prob_fact() {
        let result = parse_program("0.7 :: treasure(X).");
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            Statement::ProbFact(prob_fact) => {
                assert_eq!(prob_fact.probability, 0.7);
                assert_eq!(prob_fact.atom.predicate, Intern::new("treasure".to_string()));
            }
            _ => panic!("Expected probabilistic fact"),
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
}
