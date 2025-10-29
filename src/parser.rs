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
                num_str
                    .parse::<f64>()
                    .map(Value::Float)
                    .map_err(|_| ParseError::custom(span, "invalid float"))
            } else {
                // It's an integer
                let num_str = format!("{}{}", sign_str, whole);
                num_str
                    .parse::<i64>()
                    .map(Value::Integer)
                    .map_err(|_| ParseError::custom(span, "invalid integer"))
            }
        })
        .labelled("number")
}

/// Parse a lowercase identifier (starts with lowercase, not underscore)
fn lowercase_ident() -> impl Parser<char, String, Error = ParseError> + Clone {
    text::ident()
        .try_map(|s: String, span| {
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
    text::ident()
        .try_map(|s: String, span| {
            let first = s.chars().next().unwrap();
            if first.is_uppercase() || first == '_' {
                Ok(s)
            } else {
                Err(ParseError::custom(
                    span,
                    "expected uppercase identifier or underscore (variable)",
                ))
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

/// Parse an operator (comparison or arithmetic)
fn operator() -> impl Parser<char, String, Error = ParseError> + Clone {
    choice((
        just("<=").to("<="),
        just(">=").to(">="),
        just("!=").to("!="),
        just("\\=").to("\\="),
        just("=<").to("=<"), // Alternative syntax for <=
        just("=").to("="),
        just("<").to("<"),
        just(">").to(">"),
        just("+").to("+"),
        just("-").to("-"),
        just("*").to("*"),
        just("/").to("/"),
        just("mod").to("mod"),
    ))
    .map(|s| s.to_string())
    .labelled("operator")
}

// This is now handled inline within term() to avoid circular dependency issues

/// Parse a term (variable, constant, or compound) with arithmetic operator precedence
fn term() -> impl Parser<char, Term, Error = ParseError> + Clone {
    recursive(|term| {
        // Base factors: variables, numbers, strings, atoms, compound terms, parentheses
        let factor = {
            let variable = uppercase_ident().map(|s| Term::Variable(Intern::new(s)));

            let number_const = number().map(|v| Term::Constant(v));

            let string_const =
                string_literal().map(|s| Term::Constant(Value::String(Intern::new(s))));

            // Parenthesized term
            let parens = term
                .clone()
                .delimited_by(just('(').padded(), just(')').padded());

            // Compound term or atom (lowercase identifier with optional args)
            let compound_or_atom = lowercase_ident()
                .then(
                    term.clone()
                        .separated_by(just(',').padded())
                        .delimited_by(just('(').padded(), just(')').padded())
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
            .padded()
        };

        // Check if we have a range (term..term) - must check before arithmetic
        // Ranges can only have integer or atom constants as bounds
        let range_bound = choice((
            text::int(10).try_map(|s: String, span: std::ops::Range<usize>| {
                s.parse::<i64>()
                    .map(|n| Term::Constant(Value::Integer(n)))
                    .map_err(|_| ParseError::custom(span, "invalid integer"))
            }),
            lowercase_ident().map(|s| Term::Constant(Value::Atom(Intern::new(s)))),
        ));

        let range = range_bound
            .clone()
            .then_ignore(just("..").padded())
            .then(range_bound)
            .map(|(start, end)| Term::Range(Box::new(start), Box::new(end)));

        // Try range first, then arithmetic with precedence
        choice((range, {
            // Middle precedence: *, /, mod
            let mul_div = factor
                .clone()
                .then(
                    choice((just('*').to("*"), just('/').to("/"), just("mod").to("mod")))
                        .padded()
                        .then(factor.clone())
                        .repeated(),
                )
                .foldl(|left, (op, right)| {
                    Term::Compound(Intern::new(op.to_string()), vec![left, right])
                });

            // Lowest precedence: +, -
            mul_div
                .clone()
                .then(
                    choice((just('+').to("+"), just('-').to("-")))
                        .padded()
                        .then(mul_div)
                        .repeated(),
                )
                .foldl(|left, (op, right)| {
                    Term::Compound(Intern::new(op.to_string()), vec![left, right])
                })
        }))
        .padded()
    })
    .labelled("term")
}

/// Parse an atom
fn atom() -> impl Parser<char, Atom, Error = ParseError> + Clone {
    // Parse either a regular identifier or an operator as the predicate
    let predicate = choice((lowercase_ident(), operator()));

    predicate
        .then(
            term()
                .separated_by(just(',').padded())
                .delimited_by(just('('), just(')'))
                .or_not(),
        )
        .map(|(predicate, terms)| Atom {
            predicate: Intern::new(predicate),
            terms: terms.unwrap_or_default(),
        })
        .labelled("atom")
}

/// Parse an infix comparison as an atom (e.g., X > 3, Y = 5)
fn infix_comparison() -> impl Parser<char, Atom, Error = ParseError> + Clone {
    term()
        .then(
            choice((
                just("<=").to("<="),
                just(">=").to(">="),
                just("!=").to("!="),
                just("\\=").to("\\="),
                just("=<").to("=<"),
                just("=").to("="),
                just("<").to("<"),
                just(">").to(">"),
            ))
            .padded(),
        )
        .then(term())
        .map(|((left, op), right)| Atom {
            predicate: Intern::new(op.to_string()),
            terms: vec![left, right],
        })
}

/// Parse a literal (positive or negative atom)
fn literal() -> impl Parser<char, Literal, Error = ParseError> + Clone {
    let negated = just("not")
        .padded()
        .ignore_then(choice((infix_comparison(), atom())))
        .map(Literal::Negative);

    let positive = choice((infix_comparison(), atom())).map(Literal::Positive);

    negated.or(positive).labelled("literal")
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
            .then(just("*/")),
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
        .then(literal().separated_by(just(',').padded()).at_least(1))
        .then_ignore(just('.').padded())
        .map(|(head, body)| Statement::Rule(Rule { head, body }))
        .labelled("rule")
}

/// Parse a constraint
fn constraint() -> impl Parser<char, Statement, Error = ParseError> + Clone {
    just(":-")
        .padded()
        .ignore_then(literal().separated_by(just(',').padded()).at_least(1))
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
                Ok(_) => Err(ParseError::custom(
                    span,
                    "probability must be between 0.0 and 1.0",
                )),
                Err(_) => Err(ParseError::custom(span, "invalid probability")),
            }
        })
        .then_ignore(just("::").padded())
        .then(atom())
        .then_ignore(just('.').padded())
        .map(|(probability, atom)| Statement::ProbFact(ProbFact { probability, atom }))
        .labelled("probabilistic fact")
}

/// Parse a constant declaration: #const name = value.
fn const_decl() -> impl Parser<char, Statement, Error = ParseError> + Clone {
    just('#')
        .then_ignore(text::keyword("const").padded())
        .ignore_then(lowercase_ident())
        .then_ignore(just('=').padded())
        .then(
            just('-')
                .or_not()
                .then(text::int(10))
                .try_map(|(sign, num_str), span| {
                    let num: i64 = num_str
                        .parse()
                        .map_err(|_| ParseError::custom(span, "invalid integer"))?;
                    Ok(if sign.is_some() { -num } else { num })
                }),
        )
        .then_ignore(just('.').padded())
        .map(|(name, value)| {
            Statement::ConstDecl(ConstDecl {
                name: Intern::new(name),
                value,
            })
        })
        .labelled("constant declaration")
}

/// Parse a choice rule: { atom1; atom2 } or 1 { atom1; atom2 } 2 :- body.
fn choice_rule() -> impl Parser<char, Statement, Error = ParseError> + Clone {
    // Optional lower bound - can be integer or identifier (constant name)
    let lower_bound = choice((
        text::int(10).try_map(|s: String, span: std::ops::Range<usize>| {
            s.parse::<i64>()
                .map(|n| Term::Constant(Value::Integer(n)))
                .map_err(|_| ParseError::custom(span, "invalid integer"))
        }),
        lowercase_ident().map(|s| Term::Constant(Value::Atom(Intern::new(s)))),
    ))
    .padded()
    .or_not();

    // Choice element: atom or atom : condition1, condition2
    let choice_element = atom()
        .then(
            just(':')
                .padded()
                .ignore_then(literal().separated_by(just(',').padded()).at_least(1))
                .or_not(),
        )
        .map(|(atom, condition)| ChoiceElement {
            atom,
            condition: condition.unwrap_or_default(),
        });

    // Elements inside braces, separated by semicolons
    let elements = choice_element
        .separated_by(just(';').padded())
        .at_least(1)
        .delimited_by(just('{').padded(), just('}').padded());

    // Optional upper bound - can be integer or identifier (constant name)
    let upper_bound = choice((
        text::int(10).try_map(|s: String, span: std::ops::Range<usize>| {
            s.parse::<i64>()
                .map(|n| Term::Constant(Value::Integer(n)))
                .map_err(|_| ParseError::custom(span, "invalid integer"))
        }),
        lowercase_ident().map(|s| Term::Constant(Value::Atom(Intern::new(s)))),
    ))
    .padded()
    .or_not();

    // Optional body after :-
    let body = just(":-")
        .padded()
        .ignore_then(literal().separated_by(just(',').padded()).at_least(1))
        .or_not();

    lower_bound
        .then(elements)
        .then(upper_bound)
        .then(body)
        .then_ignore(just('.').padded())
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
fn statement() -> impl Parser<char, Statement, Error = ParseError> + Clone {
    choice((
        const_decl(), // Try #const first (distinctive prefix)
        prob_fact(),
        choice_rule(), // Try choice rules (distinctive { prefix)
        constraint(),
        rule(),
        fact(),
    ))
    .labelled("statement")
}

/// Parse a program
pub fn program() -> impl Parser<char, Program, Error = ParseError> + Clone {
    statement()
        .padded_by(
            comment()
                .or(text::whitespace().at_least(1).ignored())
                .repeated(),
        )
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

    // Ranges
    #[test]
    fn test_parse_range_simple() {
        let result = term().parse("1..10");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), range(1, 10));
    }

    #[test]
    fn test_parse_range_zero_start() {
        let result = term().parse("0..5");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), range(0, 5));
    }

    #[test]
    fn test_parse_range_large() {
        let result = term().parse("1..100");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), range(1, 100));
    }

    #[test]
    fn test_parse_range_same_values() {
        let result = term().parse("5..5");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), range(5, 5));
    }

    #[test]
    fn test_parse_range_with_constant() {
        let result = term().parse("1..width");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), range_const(1, "width"));
    }

    #[test]
    fn test_parse_range_both_constants() {
        let result = term().parse("min..max");
        assert!(result.is_ok());
        let expected = Term::Range(Box::new(atom_term("min")), Box::new(atom_term("max")));
        assert_eq!(result.unwrap(), expected);
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
        assert_eq!(
            result.unwrap(),
            Term::Variable(Intern::new("X".to_string()))
        );
    }

    #[test]
    fn test_parse_multichar_variable() {
        let result = term().parse("Player");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Variable(Intern::new("Player".to_string()))
        );
    }

    #[test]
    fn test_parse_underscore_variable() {
        let result = term().parse("_tmp");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Term::Variable(Intern::new("_tmp".to_string()))
        );
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
                assert_eq!(
                    prob_fact.atom.predicate,
                    Intern::new("treasure".to_string())
                );
            }
            _ => panic!("Expected probabilistic fact"),
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
                assert_eq!(const_decl.value, 10);
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
                assert_eq!(const_decl.value, -5);
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
                assert_eq!(c.value, 10);
            }
            _ => panic!("Expected const"),
        }

        match &program.statements[1] {
            Statement::ConstDecl(c) => {
                assert_eq!(c.name, Intern::new("height".to_string()));
                assert_eq!(c.value, 20);
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
        assert!(result.is_ok());
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
}
