mod asp;
mod ast;
mod builtins;
mod constants;
mod database;
mod evaluation;
mod grounding;
mod parser;
mod query;
mod safety;
mod stratification;
mod test_runner;
mod unification;

#[cfg(test)]
mod asp_multiple_choice_tests;

#[cfg(test)]
mod arithmetic_integration_tests;

#[cfg(test)]
mod choice_constant_bounds_tests;

#[cfg(test)]
mod query_integration_tests;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "test" => {
                if args.len() < 3 {
                    eprintln!("Usage: {} test <filename>", args[0]);
                    std::process::exit(1);
                }
                let filename = &args[2];
                run_tests(filename);
            }
            "--help" | "-h" => {
                print_help(&args[0]);
            }
            _ => {
                eprintln!("Unknown command: {}", args[1]);
                eprintln!("Run {} --help for usage information", args[0]);
                std::process::exit(1);
            }
        }
    } else {
        // No arguments - run demos
        println!("ProcLog - Datalog for Procedural Generation\n");
        println!("============================================\n");

        demo_evaluation();
        println!("\n");
        demo_asp();
        println!("\n");
        demo_dungeon_generator();
        println!("\n");
        demo_datatypes();
    }
}

fn print_help(program_name: &str) {
    println!("ProcLog - Datalog for Procedural Generation\n");
    println!("Usage:");
    println!("  {}              Run demo programs", program_name);
    println!("  {} test <file>  Run tests from a ProcLog file", program_name);
    println!("  {} --help       Show this help message", program_name);
}

fn run_tests(filename: &str) {
    use std::fs;

    // Read the file
    let content = match fs::read_to_string(filename) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", filename, e);
            std::process::exit(1);
        }
    };

    // Parse the file
    let program = match parser::parse_program(&content) {
        Ok(p) => p,
        Err(errors) => {
            eprintln!("Parse errors in '{}':", filename);
            for error in errors {
                eprintln!("  {:?}", error);
            }
            std::process::exit(1);
        }
    };

    // Extract test blocks
    let mut test_blocks = Vec::new();
    for statement in &program.statements {
        if let ast::Statement::Test(test_block) = statement {
            test_blocks.push(test_block);
        }
    }

    if test_blocks.is_empty() {
        println!("No test blocks found in '{}'", filename);
        return;
    }

    println!("Running {} test blocks from '{}'...\n", test_blocks.len(), filename);

    // Run each test block
    let mut total_passed = 0;
    let mut total_failed = 0;

    for test_block in test_blocks {
        let result = test_runner::run_test_block(test_block);

        // Print results
        println!("{}", result.summary());

        if result.passed {
            total_passed += 1;
        } else {
            total_failed += 1;

            // Print details of failures
            for case_result in &result.case_results {
                if !case_result.passed {
                    println!("  {}", case_result.message);
                }
            }
        }
        println!();
    }

    // Print overall summary
    println!("=====================================");
    println!("Test Summary:");
    println!("  Passed: {}", total_passed);
    println!("  Failed: {}", total_failed);
    println!("  Total:  {}", total_passed + total_failed);

    if total_failed > 0 {
        std::process::exit(1);
    }
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
                if let (
                    Some(ast::Term::Constant(ast::Value::Atom(from))),
                    Some(ast::Term::Constant(ast::Value::Atom(to))),
                ) = (path.terms.get(0), path.terms.get(1))
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

fn demo_asp() {
    println!("Demo 2: Answer Set Programming (Choice Rules)");
    println!("----------------------------------------------");

    let program_text = r#"
        % Define possible items
        item(sword).
        item(shield).
        item(potion).

        % Choose exactly 2 items
        2 { selected(X) : item(X) } 2.
    "#;

    match parser::parse_program(program_text) {
        Ok(program) => {
            println!("Program parsed successfully!");
            println!("Generating answer sets...\n");

            let answer_sets = asp::asp_evaluation(&program);

            println!("Found {} answer sets:\n", answer_sets.len());

            for (i, answer_set) in answer_sets.iter().enumerate() {
                println!("Answer Set {}:", i + 1);
                let mut selected: Vec<_> = answer_set
                    .atoms
                    .iter()
                    .filter(|atom| {
                        atom.predicate == internment::Intern::new("selected".to_string())
                    })
                    .collect();
                selected.sort_by_key(|atom| format!("{:?}", atom));

                for atom in selected {
                    if let Some(ast::Term::Constant(ast::Value::Atom(name))) = atom.terms.get(0) {
                        println!("  - selected({})", name);
                    }
                }
                println!();
            }
        }
        Err(errors) => {
            println!("Parse errors: {:?}", errors);
        }
    }
}

fn demo_dungeon_generator() {
    println!("Demo 3: Dungeon Generator (ASP for PCG)");
    println!("----------------------------------------");

    let program_text = r#"
        % Grid boundaries
        #const size = 4.

        % Define all possible cells using ranges
        cell(1..size, 1..size).

        % Choose 3-5 rooms from available cells
        3 { room(X, Y) : cell(X, Y) } 5.
    "#;

    match parser::parse_program(program_text) {
        Ok(program) => {
            println!("Dungeon generator program loaded!");
            println!("Grid size: 4x4");
            println!("Generating dungeons with 3-5 rooms...\n");

            let answer_sets = asp::asp_evaluation(&program);

            println!("Generated {} valid dungeon layouts:\n", answer_sets.len());

            // Show first few dungeons
            for (i, answer_set) in answer_sets.iter().take(5).enumerate() {
                println!("Layout {}:", i + 1);

                // Extract room positions
                let mut rooms: Vec<(i64, i64)> = Vec::new();
                for atom in &answer_set.atoms {
                    if atom.predicate == internment::Intern::new("room".to_string()) {
                        if let (
                            Some(ast::Term::Constant(ast::Value::Integer(x))),
                            Some(ast::Term::Constant(ast::Value::Integer(y))),
                        ) = (atom.terms.get(0), atom.terms.get(1))
                        {
                            rooms.push((*x, *y));
                        }
                    }
                }

                rooms.sort();

                // Draw the dungeon
                print!("  ");
                for y in 1..=4 {
                    for x in 1..=4 {
                        if rooms.contains(&(x, y)) {
                            print!("█");
                        } else {
                            print!("·");
                        }
                    }
                    if y < 4 {
                        print!("\n  ");
                    }
                }
                println!(" ({} rooms)\n", rooms.len());
            }

            if answer_sets.len() > 5 {
                println!("... and {} more layouts!", answer_sets.len() - 5);
            }
        }
        Err(errors) => {
            println!("Parse errors: {:?}", errors);
        }
    }
}

fn demo_datatypes() {
    println!("Demo 4: Datatype Support");
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
            println!(
                "✓ Successfully parsed {} statements:\n",
                program.statements.len()
            );

            for (i, statement) in program.statements.iter().enumerate() {
                match statement {
                    ast::Statement::Fact(fact) => {
                        println!("  {}. Fact: {:?}", i + 1, fact.atom.predicate);
                    }
                    ast::Statement::Rule(rule) => {
                        println!(
                            "  {}. Rule: {:?} with {} body literals",
                            i + 1,
                            rule.head.predicate,
                            rule.body.len()
                        );
                    }
                    ast::Statement::Constraint(constraint) => {
                        println!(
                            "  {}. Constraint with {} conditions",
                            i + 1,
                            constraint.body.len()
                        );
                    }
                    ast::Statement::ProbFact(prob_fact) => {
                        println!(
                            "  {}. Probabilistic Fact: {} :: {:?}",
                            i + 1,
                            prob_fact.probability,
                            prob_fact.atom.predicate
                        );
                    }
                    ast::Statement::ConstDecl(const_decl) => {
                        let value_str = match &const_decl.value {
                            ast::Value::Integer(n) => n.to_string(),
                            ast::Value::Float(f) => f.to_string(),
                            ast::Value::Boolean(b) => b.to_string(),
                            ast::Value::String(s) => format!("\"{}\"", s),
                            ast::Value::Atom(a) => a.to_string(),
                        };
                        println!("  {}. Constant: {} = {}", i + 1, const_decl.name, value_str);
                    }
                    ast::Statement::ChoiceRule(choice) => {
                        // Format bound terms for display
                        let format_bound = |term: &ast::Term| -> String {
                            match term {
                                ast::Term::Constant(ast::Value::Integer(n)) => n.to_string(),
                                ast::Term::Constant(ast::Value::Atom(a)) => a.to_string(),
                                _ => "?".to_string(),
                            }
                        };

                        let bounds = match (&choice.lower_bound, &choice.upper_bound) {
                            (None, None) => String::new(),
                            (Some(l), None) => format!("{} ", format_bound(l)),
                            (None, Some(u)) => format!(" {}", format_bound(u)),
                            (Some(l), Some(u)) => {
                                format!("{} ... {}", format_bound(l), format_bound(u))
                            }
                        };
                        println!(
                            "  {}. Choice Rule: {}{{ {} elements }}{}",
                            i + 1,
                            bounds,
                            choice.elements.len(),
                            if choice.body.is_empty() {
                                ""
                            } else {
                                " with body"
                            }
                        );
                    }
                    ast::Statement::Test(test_block) => {
                        println!(
                            "  {}. Test Block: \"{}\" with {} statements and {} test cases",
                            i + 1,
                            test_block.name,
                            test_block.statements.len(),
                            test_block.test_cases.len()
                        );
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
