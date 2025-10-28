mod ast;
mod parser;

fn main() {
    println!("ProcLog - Datalog for Procedural Generation\n");

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
