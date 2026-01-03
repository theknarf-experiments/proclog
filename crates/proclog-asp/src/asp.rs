//! Answer Set Programming (ASP) solver
//!
//! This module implements an ASP solver that handles choice rules and finds answer sets.
//! Answer sets are stable models that satisfy all rules, constraints, and choice cardinalities.
//!
//! # Key Features
//!
//! - **Choice Rules**: `{ atom1; atom2; atom3 } min..max` with cardinality bounds
//! - **Constraints**: Filter invalid answer sets
//! - **Guess-and-Check**: Generate candidate answer sets and verify them
//!
//! # Algorithm
//!
//! 1. Extract choice rules from program
//! 2. Generate candidate selections (subsets satisfying cardinality bounds)
//! 3. For each candidate, run stratified evaluation
//! 4. Verify all constraints hold
//! 5. Return valid answer sets
//!
//! # Example
//!
//! ```ignore
//! let answer_sets = asp_evaluation(&program, &const_env)?;
//! for answer_set in answer_sets {
//!     println!("Answer set: {:?}", answer_set.atoms);
//! }
//! ```

use proclog_ast::{
    Atom, OptimizeDirection, OptimizeStatement, Program, Statement, Symbol, Term, Value,
};
use proclog_core::{ConstantEnv, FactDatabase, Substitution};
use proclog_eval::stratified_evaluation_with_constraints;
use rand::seq::index::sample;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;

/// An answer set is a stable model - a set of ground atoms
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnswerSet {
    pub atoms: HashSet<Atom>,
}

/// Sample answer sets using deterministic randomness.
#[cfg_attr(not(test), allow(dead_code))]
pub fn asp_sample(program: &Program, seed: u64, sample_count: usize) -> Vec<AnswerSet> {
    if sample_count == 0 {
        return Vec::new();
    }

    let mut all_sets = asp_evaluation(program);
    if all_sets.is_empty() {
        return Vec::new();
    }

    sort_answer_sets(&mut all_sets);

    if sample_count >= all_sets.len() {
        return all_sets;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let indices = sample(&mut rng, all_sets.len(), sample_count).into_vec();
    let mut picked: Vec<AnswerSet> = indices.into_iter().map(|i| all_sets[i].clone()).collect();

    sort_answer_sets(&mut picked);
    picked
}

fn sort_answer_sets(sets: &mut [AnswerSet]) {
    sets.sort_by_key(canonical_atoms);
}

fn canonical_atoms(answer_set: &AnswerSet) -> Vec<String> {
    let mut atoms: Vec<String> = answer_set.atoms.iter().map(format_atom).collect();
    atoms.sort();
    atoms
}

fn format_atom(atom: &Atom) -> String {
    format!("{:?}", atom)
}

/// Resolve a bound term to an integer value
///
/// Takes a Term that is either:
/// - An integer constant: Term::Constant(Value::Integer(n))
/// - An atom that refers to a constant name: Term::Constant(Value::Atom(name))
///
/// Returns Some(i64) if the term can be resolved, None otherwise
#[cfg_attr(not(test), allow(dead_code))]
pub fn resolve_bound(bound: &Term, const_env: &ConstantEnv) -> Option<i64> {
    match bound {
        Term::Constant(Value::Integer(n)) => Some(*n),
        Term::Constant(Value::Atom(name)) => const_env.get_int(name),
        _ => None,
    }
}

/// Generate all subsets of a given size range
#[cfg_attr(not(test), allow(dead_code))]
fn generate_subsets<T: Clone>(items: &[T], min_size: usize, max_size: usize) -> Vec<Vec<T>> {
    let mut result = Vec::new();
    let n = items.len();

    // For small sets, enumerate all subsets via bit patterns
    if n <= 20 {
        let total: u32 = 1 << n;
        for mask in 0..total {
            let count = mask.count_ones() as usize;
            if count >= min_size && count <= max_size {
                let mut subset = Vec::new();
                for (i, item) in items.iter().enumerate() {
                    if (mask & (1 << i)) != 0 {
                        subset.push(item.clone());
                    }
                }
                result.push(subset);
            }
        }
    } else {
        // For larger sets, use recursive backtracking (limited to avoid explosion)
        // In practice, ASP solvers use sophisticated techniques here
        generate_subsets_recursive(items, min_size, max_size, 0, &mut vec![], &mut result);
    }

    result
}

#[cfg_attr(not(test), allow(dead_code))]
fn generate_subsets_recursive<T: Clone>(
    items: &[T],
    min_size: usize,
    max_size: usize,
    start: usize,
    current: &mut Vec<T>,
    result: &mut Vec<Vec<T>>,
) {
    // Pruning: if we can't possibly reach min_size, stop
    let remaining = items.len() - start;
    if current.len() + remaining < min_size {
        return;
    }

    // If we've reached a valid size, save it
    if current.len() >= min_size && current.len() <= max_size {
        result.push(current.clone());
    }

    // If we've reached max_size, stop growing
    if current.len() >= max_size {
        return;
    }

    // Limit total results to avoid explosion (practical limitation)
    if result.len() > 10000 {
        return;
    }

    // Try including each remaining item
    for i in start..items.len() {
        current.push(items[i].clone());
        generate_subsets_recursive(items, min_size, max_size, i + 1, current, result);
        current.pop();
    }
}

/// Generate cartesian product of multiple choice rule subsets
///
/// Given: [[a, b], [c]]  and  [[x], [y, z]]
/// Returns: [[a, b, x], [a, b, y, z], [c, x], [c, y, z]]
#[cfg_attr(not(test), allow(dead_code))]
fn cartesian_product_choice_subsets(choice_subsets: &[Vec<Vec<Atom>>]) -> Vec<Vec<Atom>> {
    if choice_subsets.is_empty() {
        return vec![vec![]];
    }

    if choice_subsets.len() == 1 {
        return choice_subsets[0].clone();
    }

    // Start with first choice rule's subsets
    let mut result = choice_subsets[0].clone();

    // Iteratively combine with each subsequent choice rule
    for choice_rule_subsets in &choice_subsets[1..] {
        let mut new_result = Vec::new();

        for existing_subset in &result {
            for new_subset in choice_rule_subsets {
                // Combine the two subsets
                let mut combined = existing_subset.clone();
                combined.extend(new_subset.clone());
                new_result.push(combined);
            }
        }

        result = new_result;

        // Safety limit to prevent explosion
        if result.len() > 100000 {
            eprintln!("Warning: Cartesian product too large (>100k), truncating");
            break;
        }
    }

    result
}

/// Evaluate a program with choice rules using generate-and-test
#[cfg_attr(not(test), allow(dead_code))]
pub fn asp_evaluation(program: &Program) -> Vec<AnswerSet> {
    // Extract components
    let const_env = ConstantEnv::from_program(program);
    let mut base_facts = FactDatabase::new();
    let mut rules = Vec::new();
    let mut constraints = Vec::new();
    let mut choice_rules = Vec::new();

    for statement in &program.statements {
        match statement {
            Statement::Fact(fact) => {
                // Expand ranges in facts
                let expanded = proclog_grounding::expand_atom_ranges(&fact.atom, &const_env);
                for atom in expanded {
                    base_facts.insert(atom).unwrap();
                }
            }
            Statement::Rule(rule) => {
                rules.push(rule.clone());
            }
            Statement::Constraint(constraint) => {
                constraints.push(constraint.clone());
            }
            Statement::ChoiceRule(choice) => {
                choice_rules.push(choice.clone());
            }
            Statement::ConstDecl(_) | Statement::Test(_) | Statement::Optimize(_) => {
                // Already handled by const_env or not relevant for ASP
                // Test blocks are handled by the test runner, not ASP evaluation
                // Optimize statements will be handled separately after finding answer sets
            }
        }
    }

    // IMPORTANT: Evaluate Datalog rules FIRST to derive all facts
    // These derived facts are needed for grounding choice rule conditions
    let derived_db = if rules.is_empty() {
        base_facts.clone()
    } else {
        // Evaluate without constraints first (constraints checked per candidate)
        match stratified_evaluation_with_constraints(&rules, &[], base_facts.clone()) {
            Ok(db) => db,
            Err(_) => return vec![], // Rules themselves inconsistent
        }
    };

    // If no choice rules, just do regular evaluation with constraints
    if choice_rules.is_empty() {
        let result_db = stratified_evaluation_with_constraints(&rules, &constraints, base_facts);

        if let Ok(db) = result_db {
            let atoms: HashSet<Atom> = db.all_facts().into_iter().cloned().collect();
            return vec![AnswerSet { atoms }];
        } else {
            return vec![]; // Constraints violated
        }
    }

    // Ground each choice rule independently and generate its subsets
    // IMPORTANT: Split choice rules by body substitutions to create
    // independent choices for each body grounding
    let mut choice_rule_subsets: Vec<Vec<Vec<Atom>>> = Vec::new();

    for choice in &choice_rules {
        // Use derived_db so choice conditions can query derived facts
        // This returns Vec<Vec<Atom>> where each inner Vec is a group
        // for one body substitution
        let groups = proclog_grounding::ground_choice_rule_split(choice, &derived_db, &const_env);

        // For each group (body substitution), create an independent choice
        for group in groups {
            // Remove duplicates within this group while preserving order
            let mut unique_atoms = Vec::new();
            for atom in group {
                if !unique_atoms.contains(&atom) {
                    unique_atoms.push(atom);
                }
            }

            // Determine bounds for this specific choice rule
            // Resolve bounds from Terms to i64 values, substituting constant names
            let min_size = choice
                .lower_bound
                .as_ref()
                .and_then(|term| resolve_bound(term, &const_env))
                .unwrap_or(0) as usize;
            let max_size = choice
                .upper_bound
                .as_ref()
                .and_then(|term| resolve_bound(term, &const_env))
                .map(|u| u as usize)
                .unwrap_or(unique_atoms.len());

            // Generate all valid subsets for this group
            let subsets = generate_subsets(&unique_atoms, min_size, max_size);
            choice_rule_subsets.push(subsets);
        }
    }

    // Generate cartesian product of all choice rule subsets
    let candidates = cartesian_product_choice_subsets(&choice_rule_subsets);

    let mut answer_sets = Vec::new();

    // Test each candidate
    for candidate in candidates {
        // Create a database with base facts + chosen atoms
        let mut test_db = base_facts.clone();
        for atom in &candidate {
            test_db
                .insert(atom.clone())
                .expect("choice candidate contained non-ground atom");
        }

        // Evaluate with rules and constraints
        let result = stratified_evaluation_with_constraints(&rules, &constraints, test_db);

        if let Ok(final_db) = result {
            // This is a valid answer set
            let atoms: HashSet<Atom> = final_db.all_facts().into_iter().cloned().collect();
            answer_sets.push(AnswerSet { atoms });
        }
    }

    // Apply optimization if present
    apply_optimization(program, answer_sets, &const_env)
}

/// Apply optimization to filter answer sets to only optimal ones
/// Supports lexicographic optimization (multiple criteria applied in order)
pub fn apply_optimization(
    program: &Program,
    answer_sets: Vec<AnswerSet>,
    const_env: &ConstantEnv,
) -> Vec<AnswerSet> {
    // Extract optimization statements
    let optimize_statements: Vec<&OptimizeStatement> = program
        .statements
        .iter()
        .filter_map(|stmt| match stmt {
            Statement::Optimize(opt) => Some(opt),
            _ => None,
        })
        .collect();

    // If no optimization statements, return all answer sets
    if optimize_statements.is_empty() {
        return answer_sets;
    }

    // Apply optimizations lexicographically (in order)
    let mut current_sets = answer_sets;

    for opt_stmt in optimize_statements {
        if current_sets.is_empty() {
            return vec![];
        }

        // Evaluate each answer set for this optimization criterion
        let evaluated: Vec<(AnswerSet, i64)> = current_sets
            .into_iter()
            .map(|answer_set| {
                let value = evaluate_optimization(opt_stmt, &answer_set, const_env);
                (answer_set, value)
            })
            .collect();

        // Find optimal value based on direction
        let optimal_value = match opt_stmt.direction {
            OptimizeDirection::Minimize => evaluated.iter().map(|(_, v)| *v).min().unwrap(),
            OptimizeDirection::Maximize => evaluated.iter().map(|(_, v)| *v).max().unwrap(),
        };

        // Keep only answer sets with optimal value for this criterion
        current_sets = evaluated
            .into_iter()
            .filter(|(_, value)| *value == optimal_value)
            .map(|(answer_set, _)| answer_set)
            .collect();
    }

    current_sets
}

/// Evaluate an optimization statement for a given answer set
fn evaluate_optimization(
    opt_stmt: &OptimizeStatement,
    answer_set: &AnswerSet,
    const_env: &ConstantEnv,
) -> i64 {
    // Create a database from the answer set for querying
    let mut answer_db = FactDatabase::new();
    for atom in &answer_set.atoms {
        let _ = answer_db.insert(atom.clone());
    }

    let mut total_value = 0i64;

    // Sum up all optimization terms that match in the answer set
    for opt_term in &opt_stmt.terms {
        // Get the term value (default weight is 1)
        let term_weight = opt_term
            .weight
            .as_ref()
            .and_then(|t| extract_integer_from_term(t, const_env))
            .unwrap_or(1);

        // If there's a condition, find all substitutions that satisfy it
        if opt_term.condition.is_empty() {
            // No condition: just evaluate the term directly
            if let Some(value) = extract_integer_from_term(&opt_term.term, const_env) {
                total_value += term_weight * value;
            }
        } else {
            // With condition: find all substitutions that satisfy the condition
            let substitutions = proclog_grounding::satisfy_body(&opt_term.condition, &answer_db);

            for subst in substitutions {
                // Apply substitution to the term and extract its value
                let substituted_term = apply_substitution_to_term(&opt_term.term, &subst);
                if let Some(value) = extract_integer_from_term(&substituted_term, const_env) {
                    total_value += term_weight * value;
                }
            }
        }
    }

    total_value
}

/// Extract an integer value from a term (if possible)
fn extract_integer_from_term(term: &Term, const_env: &ConstantEnv) -> Option<i64> {
    match term {
        Term::Constant(Value::Integer(n)) => Some(*n),
        Term::Constant(Value::Atom(name)) => {
            // Try to resolve as a constant
            const_env.get(name).and_then(|v| match v {
                Value::Integer(n) => Some(*n),
                _ => None,
            })
        }
        Term::Constant(Value::Float(_))
        | Term::Constant(Value::Boolean(_))
        | Term::Constant(Value::String(_)) => None, // Can't extract integer from these
        Term::Variable(_) => None, // Unbound variable, can't extract value
        Term::Compound(functor, args) => {
            // Handle arithmetic compound terms
            evaluate_arithmetic_term(functor, args, const_env)
        }
        Term::Range(_, _) => None, // Ranges not valid for extraction
    }
}

/// Evaluate arithmetic compound terms (e.g., *(2, 3) -> 6)
fn evaluate_arithmetic_term(
    functor: &Symbol,
    args: &[Term],
    const_env: &ConstantEnv,
) -> Option<i64> {
    let op = functor.as_str();

    match op {
        "*" if args.len() == 2 => {
            let left = extract_integer_from_term(&args[0], const_env)?;
            let right = extract_integer_from_term(&args[1], const_env)?;
            Some(left * right)
        }
        "+" if args.len() == 2 => {
            let left = extract_integer_from_term(&args[0], const_env)?;
            let right = extract_integer_from_term(&args[1], const_env)?;
            Some(left + right)
        }
        "-" if args.len() == 2 => {
            let left = extract_integer_from_term(&args[0], const_env)?;
            let right = extract_integer_from_term(&args[1], const_env)?;
            Some(left - right)
        }
        "/" if args.len() == 2 => {
            let left = extract_integer_from_term(&args[0], const_env)?;
            let right = extract_integer_from_term(&args[1], const_env)?;
            if right != 0 {
                Some(left / right)
            } else {
                None // Division by zero
            }
        }
        _ => None, // Unknown operator or wrong arity
    }
}

/// Apply a substitution to a term
fn apply_substitution_to_term(term: &Term, subst: &Substitution) -> Term {
    match term {
        Term::Variable(var) => {
            // Look up variable in substitution
            if let Some(value) = subst.get(var) {
                value.clone()
            } else {
                term.clone()
            }
        }
        Term::Constant(_) => term.clone(),
        Term::Compound(functor, args) => {
            let substituted_args: Vec<Term> = args
                .iter()
                .map(|arg| apply_substitution_to_term(arg, subst))
                .collect();
            Term::Compound(*functor, substituted_args)
        }
        Term::Range(start, end) => {
            let substituted_start = Box::new(apply_substitution_to_term(start, subst));
            let substituted_end = Box::new(apply_substitution_to_term(end, subst));
            Term::Range(substituted_start, substituted_end)
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/asp_tests.rs"]
mod tests;
