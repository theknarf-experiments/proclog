mod ast;
mod parser;
mod unification;
mod database;
mod grounding;
mod evaluation;

fn main() {
    println!("ProcLog - Datalog for Procedural Generation\n");
    println!("============================================\n");

    demo_evaluation();
    println!("\n");
    demo_datatypes();
}

fn demo_evaluation() {
    use ast::Statement;
    use database::FactDatabase;
    use evaluation::semi_naive_evaluation;

    println!("Demo 1: Transitive Closure (Semi-Naive Evaluation)");
    println!("---------------------------------------------------");

    let program_text = r#"
        % Graph edges
        edge(a, b).
        edge(b, c).
        edge(c, d).
        edge(b, e).

        % Transitive closure
        path(X, Y) :- edge(X, Y).
        path(X, Z) :- path(X, Y), edge(Y, Z).
    "#;

    match parser::parse_program(program_text) {
        Ok(program) => {
            let mut initial_db = FactDatabase::new();
            let mut rules = Vec::new();

            for statement in program.statements {
                match statement {
                    Statement::Fact(fact) => {
                        initial_db.insert(fact.atom);
                    }
                    Statement::Rule(rule) => {
                        rules.push(rule);
                    }
                    _ => {}
                }
            }

            println!("Initial facts: {} edges", initial_db.len());
            println!("Rules: {} (edge to path, transitive path)", rules.len());

            let result_db = semi_naive_evaluation(&rules, initial_db);

            println!("\nAfter evaluation: {} total facts", result_db.len());

            let path_pred = internment::Intern::new("path".to_string());
            let paths = result_db.get_by_predicate(&path_pred);
            println!("Derived {} path facts:", paths.len());

            for path in paths {
                if let (Some(ast::Term::Constant(ast::Value::Atom(from))),
                        Some(ast::Term::Constant(ast::Value::Atom(to)))) =
                    (path.terms.get(0), path.terms.get(1))
                {
                    println!("  path({}, {})", from, to);
                }
            }
        }
        Err(errors) => {
            println!("Parse errors: {:?}", errors);
        }
    }
}

fn demo_datatypes() {
    println!("Demo 2: Datatype Support");
    println!("------------------------");

    let program_text = r#"
        /* ProcLog Demo - All Supported Datatypes */

        % Integers (positive, negative, zero)
        health(player1, 100).
        damage(enemy, -25).
        score(0).

        % Floats (positive, negative, with decimals)
        position(player1, 3.14, -2.5).
        speed(10.0).

        % Booleans
        is_alive(player1, true).
        has_key(player2, false).

        % Strings
        name(player1, "Alice").
        message("Hello, World!").

        % Atoms (identifiers)
        type(player1, warrior).
        state(game, running).

        % Variables (uppercase or starting with underscore)
        can_attack(X, Y) :- is_alive(X, true), position(X, _x, _y).

        % Compound terms
        inventory(player1, item(sword, 10, 5.5)).

        % Mixed datatypes in one rule
        valid_move(Player, X, Y, Distance) :-
            is_alive(Player, true),
            position(Player, X, Y),
            speed(Distance).

        % Probabilistic fact
        0.7 :: random_encounter(X).

        % Constraint - no positive damage (damage should be negative)
        :- damage(X, D), not negative(D).
    "#;

    match parser::parse_program(program_text) {
        Ok(program) => {
            println!("✓ Successfully parsed {} statements:\n", program.statements.len());

            for (i, statement) in program.statements.iter().enumerate() {
                match statement {
                    ast::Statement::Fact(fact) => {
                        println!("  {}. Fact: {:?}", i + 1, fact.atom.predicate);
                    }
                    ast::Statement::Rule(rule) => {
                        println!("  {}. Rule: {:?} with {} body literals",
                                 i + 1, rule.head.predicate, rule.body.len());
                    }
                    ast::Statement::Constraint(constraint) => {
                        println!("  {}. Constraint with {} conditions",
                                 i + 1, constraint.body.len());
                    }
                    ast::Statement::ProbFact(prob_fact) => {
                        println!("  {}. Probabilistic Fact: {} :: {:?}",
                                 i + 1, prob_fact.probability, prob_fact.atom.predicate);
                    }
                }
            }
        }
        Err(errors) => {
            println!("✗ Parse errors:");
            for error in errors {
                println!("  - {:?}", error);
            }
        }
    }
}
