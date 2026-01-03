//! Test runner for ProcLog test blocks
//!
//! This module implements the core testing logic for ProcLog test blocks.
//! Each test block contains facts, rules, and queries with assertions to verify behavior.
//!
//! This is a language feature - the CLI uses this module for presentation.

use internment::Intern;
use proclog_asp::{asp_evaluation, AnswerSet};
use proclog_ast::{
    Atom, ChoiceElement, ChoiceRule, Constraint, Literal, Rule, Statement, Symbol, Term, TestBlock,
    TestCase, Value,
};
use proclog_builtins as builtins;
use proclog_core::{unify_atoms, ConstantEnv, FactDatabase, InsertError, Substitution};
use proclog_eval::{evaluate_query, stratified_evaluation_with_constraints, EvaluationError};
use proclog_grounding::{ground_rule, satisfy_body};
use proclog_sat::asp_sat_evaluation_with_grounding;
use std::collections::{HashMap, HashSet, VecDeque};

/// Result of running a single test case
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TestCaseResult {
    pub passed: bool,
    pub message: String,
}

/// Result of running an entire test block
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TestResult {
    pub test_name: String,
    pub passed: bool,
    pub total_cases: usize,
    pub passed_cases: usize,
    pub case_results: Vec<TestCaseResult>,
}

#[derive(Debug)]
enum PreparationError {
    NonGroundFact { atom: Atom },
}

impl PreparationError {
    fn message(&self) -> String {
        match self {
            PreparationError::NonGroundFact { atom } => format!(
                "Encountered non-ground fact {}. Facts in test inputs must be ground.",
                format_atom_pretty(atom)
            ),
        }
    }
}

#[derive(Debug, Clone)]
struct PreparedTestEnv {
    const_env: ConstantEnv,
    initial_facts: FactDatabase,
    rules: Vec<Rule>,
    constraints: Vec<Constraint>,
    choice_rules: Vec<ChoiceRule>,
}

impl PreparedTestEnv {
    fn has_asp_statements(&self) -> bool {
        !self.choice_rules.is_empty()
    }
}

fn substitute_literal(const_env: &ConstantEnv, literal: &Literal) -> Literal {
    match literal {
        Literal::Positive(atom) => Literal::Positive(const_env.substitute_atom(atom)),
        Literal::Negative(atom) => Literal::Negative(const_env.substitute_atom(atom)),
        Literal::Aggregate(agg) => Literal::Aggregate(agg.clone()),
    }
}

fn substitute_literals(const_env: &ConstantEnv, literals: &[Literal]) -> Vec<Literal> {
    literals
        .iter()
        .map(|literal| substitute_literal(const_env, literal))
        .collect()
}

fn substitute_constraint(const_env: &ConstantEnv, constraint: &Constraint) -> Constraint {
    Constraint {
        body: substitute_literals(const_env, &constraint.body),
    }
}

fn substitute_choice_rule(const_env: &ConstantEnv, choice_rule: &ChoiceRule) -> ChoiceRule {
    ChoiceRule {
        lower_bound: choice_rule
            .lower_bound
            .as_ref()
            .map(|term| const_env.substitute_term(term)),
        upper_bound: choice_rule
            .upper_bound
            .as_ref()
            .map(|term| const_env.substitute_term(term)),
        elements: choice_rule
            .elements
            .iter()
            .map(|element| substitute_choice_element(const_env, element))
            .collect(),
        body: substitute_literals(const_env, &choice_rule.body),
    }
}

fn substitute_choice_element(const_env: &ConstantEnv, element: &ChoiceElement) -> ChoiceElement {
    ChoiceElement {
        atom: const_env.substitute_atom(&element.atom),
        condition: substitute_literals(const_env, &element.condition),
    }
}

/// Run a test block and return results
pub fn run_test_block(
    base_statements: &[Statement],
    test_block: &TestBlock,
    use_sat_solver: bool,
) -> TestResult {
    let prepared = match prepare_test_environment(base_statements, test_block) {
        Ok(env) => env,
        Err(err) => {
            return preparation_failure_result(test_block, err.message());
        }
    };

    let evaluation_result = evaluate_prepared_env(&prepared, use_sat_solver);

    let answer_sets = match evaluation_result {
        Ok(answer_sets) => answer_sets,
        Err(err) => {
            let mut case_results = Vec::new();
            for test_case in &test_block.test_cases {
                let query_text = format_query(&test_case.query);
                let message = format!("✗ Evaluation failed before running {}: {}", query_text, err);
                case_results.push(TestCaseResult {
                    passed: false,
                    message,
                });
            }

            return TestResult {
                test_name: test_block.name.clone(),
                passed: false,
                total_cases: test_block.test_cases.len(),
                passed_cases: 0,
                case_results,
            };
        }
    };

    // Run each test case
    let mut case_results = Vec::new();
    let mut passed_count = 0;

    for test_case in &test_block.test_cases {
        let result = run_test_case_asp(
            test_case,
            &answer_sets,
            &prepared.rules,
            &prepared.const_env,
        );
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

fn prepare_test_environment(
    base_statements: &[Statement],
    test_block: &TestBlock,
) -> Result<PreparedTestEnv, PreparationError> {
    // Build constant environment from base program + test statements
    let mut const_env = ConstantEnv::new();
    for statement in base_statements.iter().chain(test_block.statements.iter()) {
        if let Statement::ConstDecl(const_decl) = statement {
            const_env.define(const_decl.name, const_decl.value.clone());
        }
    }

    // Build initial database, rules, and ASP components from base statements + test statements
    let mut initial_facts = FactDatabase::new();
    let mut rules = Vec::new();
    let mut constraints = Vec::new();
    let mut choice_rules = Vec::new();

    let mut process_statement = |statement: &Statement| -> Result<(), PreparationError> {
        match statement {
            Statement::Fact(fact) => {
                let substituted = const_env.substitute_atom(&fact.atom);
                initial_facts
                    .insert(substituted)
                    .map(|_| ())
                    .map_err(|err| match err {
                        InsertError::NonGroundAtom(atom) => {
                            PreparationError::NonGroundFact { atom }
                        }
                    })
            }
            Statement::Rule(rule) => {
                // Substitute constants in rule
                let substituted_head = const_env.substitute_atom(&rule.head);
                let substituted_body = substitute_literals(&const_env, &rule.body);

                rules.push(Rule {
                    head: substituted_head,
                    body: substituted_body,
                });
                Ok(())
            }
            Statement::Constraint(constraint) => {
                // Substitute constants in constraint body
                constraints.push(substitute_constraint(&const_env, constraint));
                Ok(())
            }
            Statement::ChoiceRule(choice) => {
                // Substitute constants in choice rule bounds and elements
                choice_rules.push(substitute_choice_rule(&const_env, choice));
                Ok(())
            }
            Statement::ConstDecl(_) => {
                // Already handled when building const_env
                Ok(())
            }
            Statement::Test(_) => {
                // Ignore embedded test blocks when preparing execution environment
                Ok(())
            }
            Statement::Optimize(_) => {
                // Optimization statements not yet supported in test runner
                Ok(())
            }
        }
    };

    // Check if test block defines its own choice rules
    let test_has_choice_rules = test_block
        .statements
        .iter()
        .any(|stmt| matches!(stmt, Statement::ChoiceRule(_)));

    for statement in base_statements {
        // Skip choice rules from base program if test block defines its own
        // This prevents conflicts when test overrides base choice rules
        if test_has_choice_rules && matches!(statement, Statement::ChoiceRule(_)) {
            continue;
        }
        process_statement(statement)?;
    }

    for statement in &test_block.statements {
        if let Statement::Test(_) = statement {
            // Nested tests inside a test block are ignored
            continue;
        }

        process_statement(statement)?;
    }

    Ok(PreparedTestEnv {
        const_env,
        initial_facts,
        rules,
        constraints,
        choice_rules,
    })
}

fn evaluate_prepared_env(
    prepared: &PreparedTestEnv,
    use_sat_solver: bool,
) -> Result<Vec<AnswerSet>, EvaluationError> {
    if prepared.has_asp_statements() {
        // Use ASP evaluation - construct program from collected components
        let mut statements = Vec::new();

        // Add facts
        for atom in prepared.initial_facts.all_facts() {
            statements.push(Statement::Fact(proclog_ast::Fact { atom: atom.clone() }));
        }

        // Add rules
        for rule in &prepared.rules {
            statements.push(Statement::Rule(rule.clone()));
        }

        // Add constraints
        for constraint in &prepared.constraints {
            statements.push(Statement::Constraint(constraint.clone()));
        }

        // Add choice rules
        for choice in &prepared.choice_rules {
            statements.push(Statement::ChoiceRule(choice.clone()));
        }

        let program = proclog_ast::Program { statements };

        if use_sat_solver {
            // Use SAT solver backend
            let sat_answer_sets = asp_sat_evaluation_with_grounding(&program);
            // Convert from asp_sat::AnswerSet to asp::AnswerSet
            Ok(sat_answer_sets
                .into_iter()
                .map(|as_set| AnswerSet {
                    atoms: as_set.atoms,
                })
                .collect())
        } else {
            // Use native ASP solver
            Ok(asp_evaluation(&program))
        }
    } else {
        // Use regular Datalog evaluation with constraint checking
        stratified_evaluation_with_constraints(
            &prepared.rules,
            &prepared.constraints,
            prepared.initial_facts.clone(),
        )
        .map(|db| AnswerSet {
            atoms: db.all_facts().into_iter().cloned().collect(),
        })
        .map(|answer_set| vec![answer_set])
    }
}

fn preparation_failure_result(test_block: &TestBlock, error_message: String) -> TestResult {
    let case_results: Vec<TestCaseResult> = test_block
        .test_cases
        .iter()
        .map(|test_case| {
            let query_text = format_query(&test_case.query);
            TestCaseResult {
                passed: false,
                message: format!(
                    "✗ Failed to prepare test inputs before running {}: {}",
                    query_text, error_message
                ),
            }
        })
        .collect();

    TestResult {
        test_name: test_block.name.clone(),
        passed: false,
        total_cases: test_block.test_cases.len(),
        passed_cases: 0,
        case_results,
    }
}

/// Run a single test case against answer sets (ASP-aware)
fn run_test_case_asp(
    test_case: &TestCase,
    answer_sets: &[AnswerSet],
    rules: &[Rule],
    const_env: &ConstantEnv,
) -> TestCaseResult {
    // Substitute constants in query
    let query_body = substitute_literals(const_env, &test_case.query.body);

    let query = proclog_ast::Query { body: query_body };

    // For ASP, a query succeeds if it succeeds in at least one answer set
    let mut all_results = Vec::new();
    let mut query_succeeded = false;

    for answer_set in answer_sets {
        // Convert answer set to fact database for query evaluation
        let mut db = FactDatabase::new();
        for atom in &answer_set.atoms {
            db.insert(atom.clone()).unwrap();
        }

        let results = evaluate_query(&query, &db);

        // If nothing matched, try rule-based evaluation for simple cases
        let final_results = if results.is_empty() && should_use_rule_fallback(&query, rules) {
            evaluate_query_with_rules(&query, &db, rules)
        } else {
            results
        };

        if !final_results.is_empty() {
            query_succeeded = true;
            all_results.extend(final_results);
        }
    }

    // Build query text for display
    let query_text = format_query(&test_case.query);

    // For ASP, handle "true" assertions specially
    let mut special_true_handled = false;
    let mut adjusted_positive = Vec::new();
    let mut adjusted_negative = Vec::new();

    for assertion in &test_case.positive_assertions {
        if assertion.predicate.as_ref() == "true" && assertion.terms.is_empty() {
            // + true means "query should succeed"
            if !query_succeeded {
                adjusted_positive.push(assertion.clone());
            }
            special_true_handled = true;
        } else {
            adjusted_positive.push(assertion.clone());
        }
    }

    for assertion in &test_case.negative_assertions {
        if assertion.predicate.as_ref() == "true" && assertion.terms.is_empty() {
            // - true means "query should fail"
            if query_succeeded {
                adjusted_negative.push(assertion.clone());
            }
            special_true_handled = true;
        } else {
            adjusted_negative.push(assertion.clone());
        }
    }

    // Create adjusted test case for regular assertion checking
    let adjusted_test_case = if special_true_handled {
        TestCase {
            query: test_case.query.clone(),
            positive_assertions: adjusted_positive,
            negative_assertions: adjusted_negative,
        }
    } else {
        test_case.clone()
    };

    // Check remaining assertions against the union of all results
    let (positive_failures, negative_failures) =
        check_assertions(&adjusted_test_case, &all_results, const_env);

    let passed = positive_failures.is_empty() && negative_failures.is_empty();

    let message = if passed {
        format!("✓ Query succeeded: {}", query_text)
    } else {
        let mut msg = format!("✗ Query failed: {}", query_text);
        if !query_succeeded {
            msg.push_str("\n  Query failed in all answer sets");
        }
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
    query: &proclog_ast::Query,
    db: &FactDatabase,
    rules: &[Rule],
) -> Vec<Substitution> {
    let relevant_rules = collect_relevant_rules(query, rules);

    if relevant_rules.is_empty() {
        return Vec::new();
    }

    let mut augmented_db = db.clone();
    let mut visited = HashSet::new();

    for literal in &query.body {
        if let Literal::Positive(atom) = literal {
            if builtins::parse_builtin(atom).is_none() {
                derive_specialized_rule_facts(
                    atom,
                    &relevant_rules,
                    &mut augmented_db,
                    &mut visited,
                );
            }
        }
    }

    evaluate_query(query, &augmented_db)
}

fn derive_specialized_rule_facts(
    atom: &Atom,
    rules: &[Rule],
    db: &mut FactDatabase,
    visited: &mut HashSet<Atom>,
) {
    let canonical = canonicalize_atom(atom);
    if !visited.insert(canonical) {
        return;
    }

    for rule in rules.iter().filter(|r| r.head.predicate == atom.predicate) {
        let mut head_subst = Substitution::new();

        if !unify_atoms(&rule.head, atom, &mut head_subst) {
            continue;
        }

        let specialized_head = head_subst.apply_atom(&rule.head);
        let specialized_body: Vec<Literal> = rule
            .body
            .iter()
            .map(|literal| match literal {
                Literal::Positive(inner) => Literal::Positive(head_subst.apply_atom(inner)),
                Literal::Negative(inner) => Literal::Negative(head_subst.apply_atom(inner)),
                Literal::Aggregate(agg) => Literal::Aggregate(agg.clone()),
            })
            .collect();

        for body_literal in &specialized_body {
            if let Literal::Positive(body_atom) = body_literal {
                if builtins::parse_builtin(body_atom).is_none() {
                    derive_specialized_rule_facts(body_atom, rules, db, visited);
                }
            }
        }

        let specialized_rule = Rule {
            head: specialized_head.clone(),
            body: specialized_body.clone(),
        };

        for fact in ground_rule(&specialized_rule, db) {
            let _ = db.insert(fact);
        }

        handle_equality_builtins(&specialized_head, &specialized_body, db);
    }
}

fn canonicalize_atom(atom: &Atom) -> Atom {
    let mut mapping = HashMap::new();
    Atom {
        predicate: atom.predicate,
        terms: atom
            .terms
            .iter()
            .map(|term| canonicalize_term(term, &mut mapping))
            .collect(),
    }
}

fn canonicalize_term(term: &Term, mapping: &mut HashMap<Symbol, Symbol>) -> Term {
    match term {
        Term::Variable(var) => {
            let symbol = if let Some(existing) = mapping.get(var) {
                *existing
            } else {
                let name = format!("_G{}", mapping.len());
                let interned = Intern::new(name);
                mapping.insert(*var, interned);
                interned
            };
            Term::Variable(symbol)
        }
        Term::Compound(functor, args) => Term::Compound(
            *functor,
            args.iter()
                .map(|arg| canonicalize_term(arg, mapping))
                .collect(),
        ),
        Term::Range(start, end) => Term::Range(
            Box::new(canonicalize_term(start, mapping)),
            Box::new(canonicalize_term(end, mapping)),
        ),
        Term::Constant(_) => term.clone(),
    }
}

fn handle_equality_builtins(head: &Atom, body: &[Literal], db: &mut FactDatabase) {
    if !body.iter().any(is_equality_builtin_literal) {
        return;
    }

    let (filtered_body, equality_literals): (Vec<Literal>, Vec<Literal>) = body
        .iter()
        .cloned()
        .partition(|literal| !is_equality_builtin_literal(literal));

    let substitutions = satisfy_body(&filtered_body, db);

    for subst in substitutions {
        if equality_literals
            .iter()
            .all(|literal| builtin_holds(&subst, literal))
        {
            let head_atom = subst.apply_atom(head);
            let _ = db.insert(head_atom);
        }
    }
}

fn is_equality_builtin_literal(literal: &Literal) -> bool {
    matches!(
        literal,
        Literal::Positive(atom)
            if matches!(
                builtins::parse_builtin(atom),
                Some(builtins::BuiltIn::Comparison(builtins::CompOp::Eq, _, _))
                    | Some(builtins::BuiltIn::Comparison(builtins::CompOp::Neq, _, _))
            )
    )
}

fn builtin_holds(subst: &Substitution, literal: &Literal) -> bool {
    let atom = match literal {
        Literal::Positive(atom) => atom,
        Literal::Negative(_) => return false,
        Literal::Aggregate(_) => return false,
    };

    let builtin = match builtins::parse_builtin(atom) {
        Some(builtin) => builtin,
        None => return true,
    };

    let applied = match builtin {
        builtins::BuiltIn::Comparison(op, left, right) => {
            builtins::BuiltIn::Comparison(op.clone(), subst.apply(&left), subst.apply(&right))
        }
        other => other,
    };

    match builtins::eval_builtin(&applied, subst) {
        Some(result) => result,
        None => match applied {
            builtins::BuiltIn::Comparison(builtins::CompOp::Eq, ref left, ref right) => {
                left == right
            }
            builtins::BuiltIn::Comparison(builtins::CompOp::Neq, ref left, ref right) => {
                left != right
            }
            _ => false,
        },
    }
}

fn collect_relevant_rules(query: &proclog_ast::Query, rules: &[Rule]) -> Vec<Rule> {
    let mut queue = VecDeque::new();
    let mut seen = HashSet::new();
    let mut collected = Vec::new();

    for literal in &query.body {
        if let Literal::Positive(atom) = literal {
            if builtins::parse_builtin(atom).is_none() {
                queue.push_back(atom.predicate);
            }
        }
    }

    while let Some(predicate) = queue.pop_front() {
        if !seen.insert(predicate) {
            continue;
        }

        for rule in rules.iter().filter(|r| r.head.predicate == predicate) {
            collected.push(rule.clone());

            for body_literal in &rule.body {
                if let Literal::Positive(body_atom) = body_literal {
                    if builtins::parse_builtin(body_atom).is_none() {
                        queue.push_back(body_atom.predicate);
                    }
                }
            }
        }
    }

    collected
}

fn should_use_rule_fallback(query: &proclog_ast::Query, rules: &[Rule]) -> bool {
    let mut found_relevant_rule = false;

    for literal in &query.body {
        let predicate = match literal {
            Literal::Positive(atom) => atom.predicate,
            Literal::Negative(_) => continue,
            Literal::Aggregate(_) => continue,
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
        Literal::Aggregate(_) => false,
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
    let substituted_query_body = substitute_literals(const_env, &test_case.query.body);
    let query_atom_for_assertions = substituted_query_body
        .first()
        .and_then(|literal| literal.atom().cloned());

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
                query_atom_for_assertions
                    .as_ref()
                    .is_some_and(|query_atom| {
                        let instantiated = subst.apply_atom(query_atom);
                        atoms_match(&instantiated, &substituted)
                    })
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
            query_atom_for_assertions
                .as_ref()
                .is_some_and(|query_atom| {
                    let instantiated = subst.apply_atom(query_atom);
                    atoms_match(&instantiated, &substituted)
                })
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
fn format_query(query: &proclog_ast::Query) -> String {
    let literals: Vec<String> = query
        .body
        .iter()
        .map(|lit| match lit {
            Literal::Positive(atom) => format!("{:?}", atom.predicate),
            Literal::Negative(atom) => format!("not {:?}", atom.predicate),
            Literal::Aggregate(_) => "<aggregate>".to_string(),
        })
        .collect();
    format!("?- {}.", literals.join(", "))
}

fn format_atom_pretty(atom: &Atom) -> String {
    let predicate = atom.predicate.as_ref();
    if atom.terms.is_empty() {
        predicate.to_string()
    } else {
        let terms = atom
            .terms
            .iter()
            .map(format_term_pretty)
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}({})", predicate, terms)
    }
}

fn format_term_pretty(term: &Term) -> String {
    match term {
        Term::Variable(name) => name.as_ref().to_string(),
        Term::Constant(value) => match value {
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => {
                let mut s = f.to_string();
                if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                    s.push_str(".0");
                }
                s
            }
            Value::Boolean(b) => b.to_string(),
            Value::String(s) => format!("\"{}\"", s.as_ref()),
            Value::Atom(a) => a.as_ref().to_string(),
        },
        Term::Compound(functor, args) => {
            let rendered_args = args
                .iter()
                .map(format_term_pretty)
                .collect::<Vec<_>>()
                .join(", ");
            if args.is_empty() {
                functor.as_ref().to_string()
            } else {
                format!("{}({})", functor.as_ref(), rendered_args)
            }
        }
        Term::Range(start, end) => {
            format!("{}..{}", format_term_pretty(start), format_term_pretty(end))
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/test_runner_tests.rs"]
mod tests;
