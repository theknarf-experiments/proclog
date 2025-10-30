//! Test runner for ProcLog test blocks
//!
//! This module implements functionality to run ProcLog tests defined with `#test` blocks.
//! Each test block contains facts, rules, and queries with assertions to verify behavior.

use crate::ast::{Atom, Literal, Rule, Statement, Symbol, Term, TestBlock, TestCase};
use crate::builtins;
use crate::constants::ConstantEnv;
use crate::database::FactDatabase;
use crate::evaluation::semi_naive_evaluation;
use crate::query::evaluate_query;
use crate::unification::{unify_atoms, Substitution};
use internment::Intern;
use std::collections::{HashMap, HashSet};

/// Result of running a single test case
#[derive(Debug, Clone)]
pub struct TestCaseResult {
    pub passed: bool,
    pub message: String,
}

/// Result of running an entire test block
#[derive(Debug, Clone)]
pub struct TestResult {
    pub test_name: String,
    pub passed: bool,
    pub total_cases: usize,
    pub passed_cases: usize,
    pub case_results: Vec<TestCaseResult>,
}

impl TestResult {
    /// Get a summary message
    pub fn summary(&self) -> String {
        if self.passed {
            format!(
                "✓ Test '{}': {} / {} cases passed",
                self.test_name, self.passed_cases, self.total_cases
            )
        } else {
            format!(
                "✗ Test '{}': {} / {} cases passed",
                self.test_name, self.passed_cases, self.total_cases
            )
        }
    }
}

/// Run a test block and return results
pub fn run_test_block(base_statements: &[Statement], test_block: &TestBlock) -> TestResult {
    // Build constant environment from base program + test statements
    let mut const_env = ConstantEnv::new();
    for statement in base_statements.iter().chain(test_block.statements.iter()) {
        if let Statement::ConstDecl(const_decl) = statement {
            const_env.define(const_decl.name.clone(), const_decl.value.clone());
        }
    }

    // Build initial database and rules from base statements + test statements
    let mut initial_facts = FactDatabase::new();
    let mut rules = Vec::new();

    let mut process_statement = |statement: &Statement| match statement {
        Statement::Fact(fact) => {
            let substituted = const_env.substitute_atom(&fact.atom);
            initial_facts.insert(substituted);
        }
        Statement::Rule(rule) => {
            // Substitute constants in rule
            let substituted_head = const_env.substitute_atom(&rule.head);
            let substituted_body: Vec<_> = rule
                .body
                .iter()
                .map(|lit| match lit {
                    Literal::Positive(atom) => Literal::Positive(const_env.substitute_atom(atom)),
                    Literal::Negative(atom) => Literal::Negative(const_env.substitute_atom(atom)),
                })
                .collect();

            rules.push(Rule {
                head: substituted_head,
                body: substituted_body,
            });
        }
        Statement::ConstDecl(_) => {
            // Already handled when building const_env
        }
        Statement::Test(_) => {
            // Ignore embedded test blocks when preparing execution environment
        }
        _ => {
            // Ignore other statement types in tests
        }
    };

    for statement in base_statements {
        process_statement(statement);
    }

    for statement in &test_block.statements {
        match statement {
            Statement::Test(_) => {
                // Nested tests inside a test block are ignored
            }
            other => process_statement(other),
        }
    }

    // Evaluate rules to get complete database
    let db = if rules.is_empty() {
        initial_facts
    } else {
        semi_naive_evaluation(&rules, initial_facts)
    };

    // Run each test case
    let mut case_results = Vec::new();
    let mut passed_count = 0;

    for test_case in &test_block.test_cases {
        let result = run_test_case(test_case, &db, &rules, &const_env);
        if result.passed {
            passed_count += 1;
        }
        case_results.push(result);
    }

    let all_passed = passed_count == test_block.test_cases.len();

    TestResult {
        test_name: test_block.name.clone(),
        passed: all_passed,
        total_cases: test_block.test_cases.len(),
        passed_cases: passed_count,
        case_results,
    }
}

/// Run a single test case
fn run_test_case(
    test_case: &TestCase,
    db: &FactDatabase,
    rules: &[Rule],
    const_env: &ConstantEnv,
) -> TestCaseResult {
    // Substitute constants in query
    let query_body: Vec<_> = test_case
        .query
        .body
        .iter()
        .map(|lit| match lit {
            Literal::Positive(atom) => Literal::Positive(const_env.substitute_atom(atom)),
            Literal::Negative(atom) => Literal::Negative(const_env.substitute_atom(atom)),
        })
        .collect();

    let query = crate::ast::Query { body: query_body };

    // Run the query against derived facts first
    let mut results = evaluate_query(&query, db);

    // If nothing matched, try a rule-based evaluation for simple cases
    if results.is_empty() && should_use_rule_fallback(&query, rules) {
        results = evaluate_query_with_rules(&query, db, rules);
    }

    // Build query text for display
    let query_text = format_query(&test_case.query);

    // Check assertions
    let (positive_failures, negative_failures) = check_assertions(test_case, &results, const_env);

    let passed = positive_failures.is_empty() && negative_failures.is_empty();

    let message = if passed {
        format!("✓ Query succeeded: {}", query_text)
    } else {
        let mut msg = format!("✗ Query failed: {}", query_text);
        if !positive_failures.is_empty() {
            msg.push_str(&format!(
                "\n  Missing expected results: {}",
                positive_failures
                    .iter()
                    .map(|a| format!("{:?}", a))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if !negative_failures.is_empty() {
            msg.push_str(&format!(
                "\n  Found unexpected results: {}",
                negative_failures
                    .iter()
                    .map(|a| format!("{:?}", a))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        msg
    };

    TestCaseResult { passed, message }
}

/// Evaluate a query against both derived facts and local rules
fn evaluate_query_with_rules(
    query: &crate::ast::Query,
    db: &FactDatabase,
    rules: &[Rule],
) -> Vec<Substitution> {
    let mut visited = HashSet::new();
    let mut gensym = 0usize;
    satisfy_with_rules(&query.body, db, rules, &mut visited, &mut gensym)
}

/// Recursively satisfy literals using available facts and rules
fn satisfy_with_rules(
    literals: &[Literal],
    db: &FactDatabase,
    rules: &[Rule],
    visited: &mut HashSet<Atom>,
    gensym: &mut usize,
) -> Vec<Substitution> {
    if literals.is_empty() {
        return vec![Substitution::new()];
    }

    let first = &literals[0];
    let rest = &literals[1..];

    match first {
        Literal::Positive(atom) => {
            if let Some(builtin) = builtins::parse_builtin(atom) {
                let mut result = Vec::new();

                if rest.is_empty() {
                    let empty = Substitution::new();
                    if matches!(evaluate_builtin_with_subst(&empty, &builtin), Some(true)) {
                        result.push(empty);
                    }
                } else {
                    let rest_substs = satisfy_with_rules(rest, db, rules, visited, gensym);
                    for subst in rest_substs {
                        if matches!(evaluate_builtin_with_subst(&subst, &builtin), Some(true)) {
                            result.push(subst);
                        }
                    }
                }

                return result;
            }

            let guard_atom = canonicalize_atom(atom);
            if !visited.insert(guard_atom.clone()) {
                return Vec::new();
            }

            let mut results = Vec::new();

            // Matches provided by existing facts
            let fact_matches = db.query(atom);
            results.extend(extend_with_rest(
                fact_matches,
                rest,
                db,
                rules,
                visited,
                gensym,
            ));

            // Matches provided via rules
            for rule in rules.iter().filter(|r| r.head.predicate == atom.predicate) {
                let fresh_rule = freshen_rule(rule, gensym);

                let mut head_subst = Substitution::new();
                if unify_atoms(&fresh_rule.head, atom, &mut head_subst) {
                    let rule_body: Vec<Literal> = fresh_rule
                        .body
                        .iter()
                        .map(|lit| apply_subst_to_literal(&head_subst, lit))
                        .collect();

                    let body_substs = satisfy_with_rules(&rule_body, db, rules, visited, gensym);
                    let combined_substs: Vec<Substitution> = body_substs
                        .into_iter()
                        .map(|body_subst| combine_substitutions(&head_subst, &body_subst))
                        .collect();

                    results.extend(extend_with_rest(
                        combined_substs,
                        rest,
                        db,
                        rules,
                        visited,
                        gensym,
                    ));
                }
            }

            visited.remove(&guard_atom);
            results
        }
        Literal::Negative(atom) => {
            let rest_substs = satisfy_with_rules(rest, db, rules, visited, gensym);
            rest_substs
                .into_iter()
                .filter(|subst| {
                    let grounded = subst.apply_atom(atom);
                    satisfy_with_rules(&[Literal::Positive(grounded)], db, rules, visited, gensym)
                        .is_empty()
                })
                .collect()
        }
    }
}

/// Extend substitutions by satisfying the remaining literals
fn extend_with_rest(
    matches: Vec<Substitution>,
    rest: &[Literal],
    db: &FactDatabase,
    rules: &[Rule],
    visited: &mut HashSet<Atom>,
    gensym: &mut usize,
) -> Vec<Substitution> {
    if rest.is_empty() {
        return matches;
    }

    let mut combined = Vec::new();

    for subst in matches {
        let applied_rest: Vec<Literal> = rest
            .iter()
            .map(|lit| apply_subst_to_literal(&subst, lit))
            .collect();
        let rest_substs = satisfy_with_rules(&applied_rest, db, rules, visited, gensym);
        for rest_subst in rest_substs {
            combined.push(combine_substitutions(&subst, &rest_subst));
        }
    }

    combined
}

/// Apply substitution to a literal
fn apply_subst_to_literal(subst: &Substitution, literal: &Literal) -> Literal {
    match literal {
        Literal::Positive(atom) => Literal::Positive(subst.apply_atom(atom)),
        Literal::Negative(atom) => Literal::Negative(subst.apply_atom(atom)),
    }
}

/// Combine two substitutions, applying the first to the second's bindings
fn combine_substitutions(first: &Substitution, second: &Substitution) -> Substitution {
    let mut combined = first.clone();
    for (var, term) in second.iter() {
        let applied = first.apply(term);
        combined.bind(var.clone(), applied);
    }
    combined
}

fn should_use_rule_fallback(query: &crate::ast::Query, rules: &[Rule]) -> bool {
    let mut found_relevant_rule = false;

    for literal in &query.body {
        let predicate = match literal {
            Literal::Positive(atom) => atom.predicate.clone(),
            Literal::Negative(_) => continue,
        };

        let relevant_rules: Vec<&Rule> = rules
            .iter()
            .filter(|rule| rule.head.predicate == predicate)
            .collect();

        if relevant_rules.is_empty() {
            continue;
        }

        if relevant_rules.iter().any(|rule| rule_is_recursive(rule)) {
            return false;
        }

        found_relevant_rule = true;
    }

    found_relevant_rule
}

fn rule_is_recursive(rule: &Rule) -> bool {
    rule.body.iter().any(|lit| match lit {
        Literal::Positive(atom) => atom.predicate == rule.head.predicate,
        Literal::Negative(_) => false,
    })
}

/// Create a fresh copy of a rule with uniquely named variables
fn freshen_rule(rule: &Rule, gensym: &mut usize) -> Rule {
    let mut mapping = HashMap::new();
    Rule {
        head: freshen_atom(&rule.head, &mut mapping, gensym),
        body: rule
            .body
            .iter()
            .map(|lit| freshen_literal(lit, &mut mapping, gensym))
            .collect(),
    }
}

fn freshen_literal(
    literal: &Literal,
    mapping: &mut HashMap<Symbol, Symbol>,
    gensym: &mut usize,
) -> Literal {
    match literal {
        Literal::Positive(atom) => Literal::Positive(freshen_atom(atom, mapping, gensym)),
        Literal::Negative(atom) => Literal::Negative(freshen_atom(atom, mapping, gensym)),
    }
}

fn freshen_atom(atom: &Atom, mapping: &mut HashMap<Symbol, Symbol>, gensym: &mut usize) -> Atom {
    Atom {
        predicate: atom.predicate.clone(),
        terms: atom
            .terms
            .iter()
            .map(|term| freshen_term(term, mapping, gensym))
            .collect(),
    }
}

fn freshen_term(term: &Term, mapping: &mut HashMap<Symbol, Symbol>, gensym: &mut usize) -> Term {
    match term {
        Term::Variable(var) => {
            if var.as_ref() == "_" {
                let name = format!("_G{}", *gensym);
                *gensym += 1;
                Term::Variable(Intern::new(name))
            } else {
                let entry = mapping.entry(var.clone()).or_insert_with(|| {
                    let name = format!("_G{}", *gensym);
                    *gensym += 1;
                    Intern::new(name)
                });
                Term::Variable(entry.clone())
            }
        }
        Term::Constant(_) => term.clone(),
        Term::Compound(functor, args) => Term::Compound(
            functor.clone(),
            args.iter()
                .map(|arg| freshen_term(arg, mapping, gensym))
                .collect(),
        ),
        Term::Range(start, end) => Term::Range(
            Box::new(freshen_term(start, mapping, gensym)),
            Box::new(freshen_term(end, mapping, gensym)),
        ),
    }
}

fn canonicalize_atom(atom: &Atom) -> Atom {
    let mut mapping = HashMap::new();
    let mut counter = 0usize;
    Atom {
        predicate: atom.predicate.clone(),
        terms: atom
            .terms
            .iter()
            .map(|term| canonicalize_term(term, &mut mapping, &mut counter))
            .collect(),
    }
}

fn canonicalize_term(
    term: &Term,
    mapping: &mut HashMap<Symbol, Symbol>,
    counter: &mut usize,
) -> Term {
    match term {
        Term::Variable(var) => {
            let entry = mapping.entry(var.clone()).or_insert_with(|| {
                let name = format!("_C{}", *counter);
                *counter += 1;
                Intern::new(name)
            });
            Term::Variable(entry.clone())
        }
        Term::Constant(_) => term.clone(),
        Term::Compound(functor, args) => Term::Compound(
            functor.clone(),
            args.iter()
                .map(|arg| canonicalize_term(arg, mapping, counter))
                .collect(),
        ),
        Term::Range(start, end) => Term::Range(
            Box::new(canonicalize_term(start, mapping, counter)),
            Box::new(canonicalize_term(end, mapping, counter)),
        ),
    }
}

fn apply_builtin_with_subst(
    subst: &Substitution,
    builtin: &builtins::BuiltIn,
) -> builtins::BuiltIn {
    match builtin {
        builtins::BuiltIn::Comparison(op, left, right) => {
            builtins::BuiltIn::Comparison(op.clone(), subst.apply(left), subst.apply(right))
        }
        builtins::BuiltIn::True => builtins::BuiltIn::True,
        builtins::BuiltIn::Fail => builtins::BuiltIn::Fail,
    }
}

fn evaluate_builtin_with_subst(subst: &Substitution, builtin: &builtins::BuiltIn) -> Option<bool> {
    let applied = apply_builtin_with_subst(subst, builtin);
    builtins::eval_builtin(&applied, subst).or_else(|| match &applied {
        builtins::BuiltIn::Comparison(op, left, right) => match op {
            builtins::CompOp::Eq => Some(left == right),
            builtins::CompOp::Neq => Some(left != right),
            _ => None,
        },
        builtins::BuiltIn::True => Some(true),
        builtins::BuiltIn::Fail => Some(false),
    })
}

/// Check positive and negative assertions against query results
fn check_assertions(
    test_case: &TestCase,
    results: &[Substitution],
    const_env: &ConstantEnv,
) -> (Vec<Atom>, Vec<Atom>) {
    let mut positive_failures = Vec::new();
    let mut negative_failures = Vec::new();

    // Check positive assertions (should be in results)
    for assertion in &test_case.positive_assertions {
        let substituted = const_env.substitute_atom(assertion);

        // Special case: "true" atom means the query should have at least one result
        if assertion.predicate.as_ref() == "true" && assertion.terms.is_empty() {
            if results.is_empty() {
                positive_failures.push(assertion.clone());
            }
        } else {
            // Check if this atom matches any result
            let matches = results.iter().any(|subst| {
                let instantiated = subst.apply_atom(&test_case.query.body[0].atom());
                atoms_match(&instantiated, &substituted)
            });

            if !matches {
                positive_failures.push(assertion.clone());
            }
        }
    }

    // Check negative assertions (should NOT be in results)
    for assertion in &test_case.negative_assertions {
        let substituted = const_env.substitute_atom(assertion);

        // Check if this atom matches any result
        let matches = results.iter().any(|subst| {
            let instantiated = subst.apply_atom(&test_case.query.body[0].atom());
            atoms_match(&instantiated, &substituted)
        });

        if matches {
            negative_failures.push(assertion.clone());
        }
    }

    (positive_failures, negative_failures)
}

/// Check if two atoms match (same predicate and terms)
fn atoms_match(a: &Atom, b: &Atom) -> bool {
    a == b
}

/// Format a query for display
fn format_query(query: &crate::ast::Query) -> String {
    let literals: Vec<String> = query
        .body
        .iter()
        .map(|lit| match lit {
            Literal::Positive(atom) => format!("{:?}", atom.predicate),
            Literal::Negative(atom) => format!("not {:?}", atom.predicate),
        })
        .collect();
    format!("?- {}.", literals.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use crate::parser::parse_program;

    #[test]
    fn test_simple_passing_test() {
        let input = r#"
            #test "basic test" {
                parent(john, mary).

                ?- parent(john, mary).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block);
        assert!(result.passed, "Test should pass");
        assert_eq!(result.passed_cases, 1);
    }

    #[test]
    fn test_simple_failing_test() {
        let input = r#"
            #test "should fail" {
                parent(john, mary).

                ?- parent(alice, bob).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block);
        assert!(!result.passed, "Test should fail");
        assert_eq!(result.passed_cases, 0);
    }

    #[test]
    fn test_with_rules() {
        let input = r#"
            #test "transitive closure" {
                edge(a, b).
                edge(b, c).
                path(X, Y) :- edge(X, Y).
                path(X, Z) :- path(X, Y), edge(Y, Z).

                ?- path(a, c).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block);
        assert!(result.passed, "Test should pass: {:?}", result.case_results);
    }

    #[test]
    fn test_with_comments_between_queries() {
        let input = r#"
            #test "health status checks" {
                #const healthy_threshold = 50.
                #const critical_health = 20.

                character(warrior, 85, 15, 500).
                character(mage, 15, 45, 1200).
                character(rogue, 60, 20, 800).

                healthy(Name) :- character(Name, Health, _, _), Health >= healthy_threshold.
                critical(Name) :- character(Name, Health, _, _), Health <= critical_health.

                % Warrior is healthy
                ?- healthy(warrior).
                + true.

                % Mage is critical
                ?- critical(mage).
                + true.

                % Rogue is healthy
                ?- healthy(rogue).
                + true.

                % Warrior is not critical
                ?- critical(warrior).
                - true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        assert_eq!(test_block.statements.len(), 7);
        assert_eq!(test_block.test_cases.len(), 4);

        use crate::constants::ConstantEnv;
        use crate::evaluation::semi_naive_evaluation;
        use internment::Intern;

        let const_env = ConstantEnv::from_statements(&test_block.statements);
        let mut initial_facts = FactDatabase::new();
        let mut rules_vec = Vec::new();

        for statement in &test_block.statements {
            match statement {
                Statement::Fact(fact) => {
                    initial_facts.insert(const_env.substitute_atom(&fact.atom));
                }
                Statement::Rule(rule) => {
                    let head = const_env.substitute_atom(&rule.head);
                    let body = rule
                        .body
                        .iter()
                        .map(|lit| match lit {
                            Literal::Positive(atom) => {
                                Literal::Positive(const_env.substitute_atom(atom))
                            }
                            Literal::Negative(atom) => {
                                Literal::Negative(const_env.substitute_atom(atom))
                            }
                        })
                        .collect();
                    rules_vec.push(Rule { head, body });
                }
                Statement::ConstDecl(_) => {}
                other => panic!("Unexpected statement in test block: {:?}", other),
            }
        }

        let db = if rules_vec.is_empty() {
            initial_facts.clone()
        } else {
            semi_naive_evaluation(&rules_vec, initial_facts.clone())
        };

        let healthy_pred = Intern::new("healthy".to_string());
        assert!(
            db.get_by_predicate(&healthy_pred)
                .iter()
                .any(|atom| matches!(
                    atom.terms.as_slice(),
                    [Term::Constant(Value::Atom(name))] if name.as_ref() == "warrior"
                )),
            "Derived database should contain healthy(warrior)"
        );

        let critical_pred = Intern::new("critical".to_string());
        assert!(
            db.get_by_predicate(&critical_pred)
                .iter()
                .any(|atom| matches!(
                    atom.terms.as_slice(),
                    [Term::Constant(Value::Atom(name))] if name.as_ref() == "mage"
                )),
            "Derived database should contain critical(mage)"
        );

        let result = run_test_block(&[], test_block);
        assert!(
            result.passed,
            "Test block should pass: {:?}",
            result.case_results
        );
        assert_eq!(result.passed_cases, test_block.test_cases.len());
    }

    #[test]
    fn test_available_at_level_rule_fallback() {
        let input = r#"
            #test "quest level requirements" {
                quest(gather_herbs).
                quest_level(gather_herbs, 1).

                available_at_level(Quest, Level) :-
                    quest(Quest),
                    quest_level(Quest, Required),
                    Level >= Required.

                ?- available_at_level(gather_herbs, 1).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block);
        assert!(result.passed, "Fallback should allow rule evaluation");
        assert_eq!(result.passed_cases, 1);
    }

    #[test]
    fn test_level_appropriate_rule_fallback() {
        let input = r#"
            #test "item level requirements" {
                item(iron_sword, sword, common, 1, 10).
                item(steel_sword, sword, rare, 5, 50).

                level_appropriate(Item, CharLevel) :-
                    item(Item, _, _, ItemLevel, _),
                    CharLevel >= ItemLevel.

                ?- level_appropriate(steel_sword, 5).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block);
        assert!(result.passed, "Fallback should handle item level checks");
        assert_eq!(result.passed_cases, 1);
    }

    #[test]
    fn test_better_than_common_rule_fallback() {
        let input = r#"
            #test "rarity check" {
                item(item1, sword, common, 1, 10).
                item(item2, sword, rare, 2, 20).

                better_than_common(Item) :- item(Item, _, Rarity, _, _), Rarity = rare.
                better_than_common(Item) :- item(Item, _, Rarity, _, _), Rarity = legendary.

                ?- better_than_common(item2).
                + true.
            }
        "#;

        let program = parse_program(input).expect("Parse failed");
        let test_block = match &program.statements[0] {
            Statement::Test(tb) => tb,
            _ => panic!("Expected test block"),
        };

        let result = run_test_block(&[], test_block);
        assert!(result.passed, "Fallback should handle rarity checks");
        assert_eq!(result.passed_cases, 1);
    }
}
