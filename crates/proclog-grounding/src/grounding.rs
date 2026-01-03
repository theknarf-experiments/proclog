//! Rule grounding - generating ground instances of rules
//!
//! This module implements the grounding phase of logic programming evaluation.
//! Grounding replaces variables in rules with concrete values from the database.
//!
//! # Key Functions
//!
//! - `ground_rule`: Standard grounding for a single rule
//! - `ground_rule_semi_naive`: Optimized grounding using delta (newly derived facts)
//! - `satisfy_body`: Find all substitutions that satisfy a rule body
//! - `ground_choice_rule`: Ground choice rules with their elements
//!
//! # Example
//!
//! ```ignore
//! // Given rule: ancestor(X, Z) :- parent(X, Y), parent(Y, Z)
//! // And facts: parent(a, b), parent(b, c)
//! // Produces: ancestor(a, c)
//! let groundings = ground_rule(&rule, &db, &const_env);
//! ```

use proclog_ast::{AggregateAtom, Atom, ChoiceElement, ChoiceRule, Literal, Rule, Term, Value};
use proclog_builtins as builtins;
use proclog_core::{ConstantEnv, FactDatabase, Substitution};

#[cfg(test)]
mod allocation_tracker {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct CountingAllocator;

    static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

    pub fn reset() {
        ALLOCATIONS.store(0, Ordering::SeqCst);
    }

    pub fn allocations() -> usize {
        ALLOCATIONS.load(Ordering::SeqCst)
    }

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            System.alloc(layout)
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout)
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            System.alloc_zeroed(layout)
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            System.realloc(ptr, layout, new_size)
        }
    }
}

#[cfg(test)]
#[global_allocator]
static GLOBAL: allocation_tracker::CountingAllocator = allocation_tracker::CountingAllocator;

/// Ground a rule: generate all ground instances by substituting variables
/// For a rule like `ancestor(X, Z) :- parent(X, Y), parent(Y, Z)`
/// This finds all ways to satisfy the body and applies those substitutions to the head
pub fn ground_rule(rule: &Rule, db: &FactDatabase) -> Vec<Atom> {
    let mut results = Vec::new();

    // Get all substitutions that satisfy the entire body
    let substitutions = satisfy_body(&rule.body, db);

    // Apply each substitution to the head to get ground facts
    for subst in substitutions {
        let ground_head = subst.apply_atom(&rule.head);
        results.push(ground_head);
    }

    results
}

/// Find all substitutions that satisfy a conjunction of literals
pub fn satisfy_body(body: &[Literal], db: &FactDatabase) -> Vec<Substitution> {
    satisfy_body_with_selector(body, db, None, &|_, _| DatabaseSelection::Full)
}

enum DatabaseSelection {
    Full,
    Delta,
}

fn satisfy_body_with_selector<F>(
    body: &[Literal],
    full_db: &FactDatabase,
    delta: Option<&FactDatabase>,
    selector: &F,
) -> Vec<Substitution>
where
    F: Fn(usize, &Literal) -> DatabaseSelection,
{
    satisfy_body_with_selector_recursive(body, full_db, delta, selector, 0, &Substitution::new())
}

fn satisfy_body_with_selector_recursive<F>(
    body: &[Literal],
    full_db: &FactDatabase,
    delta: Option<&FactDatabase>,
    selector: &F,
    index: usize,
    current_subst: &Substitution,
) -> Vec<Substitution>
where
    F: Fn(usize, &Literal) -> DatabaseSelection,
{
    if index == body.len() {
        return vec![current_subst.clone()];
    }

    let literal = &body[index];

    match literal {
        Literal::Positive(atom) => {
            if let Some(builtin) = builtins::parse_builtin(atom) {
                // Built-ins act as filters - evaluate them after satisfying the rest.
                let rest_substs = satisfy_body_with_selector_recursive(
                    body,
                    full_db,
                    delta,
                    selector,
                    index + 1,
                    current_subst,
                );
                let mut result = Vec::new();

                for subst in rest_substs {
                    let applied_builtin = apply_subst_to_builtin(&subst, &builtin);
                    if let Some(true) = builtins::eval_builtin(&applied_builtin, &subst) {
                        result.push(subst);
                    }
                }

                result
            } else {
                // Apply the current substitution to the atom before querying.
                let grounded_atom = current_subst.apply_atom(atom);
                let db = match selector(index, literal) {
                    DatabaseSelection::Full => full_db,
                    DatabaseSelection::Delta => delta.unwrap_or(full_db),
                };
                let mut result = Vec::new();

                for atom_subst in db.query(&grounded_atom) {
                    if let Some(combined) = combine_substs(current_subst, &atom_subst) {
                        let mut rest_results = satisfy_body_with_selector_recursive(
                            body,
                            full_db,
                            delta,
                            selector,
                            index + 1,
                            &combined,
                        );
                        result.append(&mut rest_results);
                    }
                }

                result
            }
        }
        Literal::Negative(atom) => {
            // Negation filters substitutions produced by the rest of the body.
            let rest_substs = satisfy_body_with_selector_recursive(
                body,
                full_db,
                delta,
                selector,
                index + 1,
                current_subst,
            );
            let mut result = Vec::new();

            for subst in rest_substs {
                let grounded_atom = subst.apply_atom(atom);
                if !database_has_match(full_db, &grounded_atom) {
                    result.push(subst);
                }
            }

            result
        }
        Literal::Aggregate(_) => {
            // Aggregates are filters - evaluate after rest of body
            // For now, just pass through (evaluation happens during constraint checking)
            satisfy_body_with_selector_recursive(
                body,
                full_db,
                delta,
                selector,
                index + 1,
                current_subst,
            )
        }
    }
}

/// Apply substitution to a literal
fn apply_subst_to_literal(subst: &Substitution, literal: &Literal) -> Literal {
    match literal {
        Literal::Positive(atom) => Literal::Positive(subst.apply_atom(atom)),
        Literal::Negative(atom) => Literal::Negative(subst.apply_atom(atom)),
        Literal::Aggregate(agg) => {
            // Apply substitution to aggregate elements
            let new_elements: Vec<Literal> = agg
                .elements
                .iter()
                .map(|lit| apply_subst_to_literal(subst, lit))
                .collect();
            Literal::Aggregate(AggregateAtom {
                function: agg.function,
                variables: agg.variables.clone(),
                elements: new_elements,
                comparison: agg.comparison,
                value: subst.apply(&agg.value),
            })
        }
    }
}

/// Combine two substitutions, returning `None` if they conflict.
fn combine_substs(s1: &Substitution, s2: &Substitution) -> Option<Substitution> {
    let mut combined = s1.clone();

    for (var, term) in s2.iter() {
        // Apply bindings from both substitutions before comparing.
        let term_applied_s2 = s2.apply(term);
        let candidate = combined.apply(&term_applied_s2);

        if let Some(existing) = combined.get(var) {
            let existing_applied_s2 = s2.apply(existing);
            let existing_resolved = combined.apply(&existing_applied_s2);

            if existing_resolved != candidate {
                return None;
            }
        }

        combined.bind(*var, candidate);
    }

    Some(combined)
}

fn database_has_match(db: &FactDatabase, atom: &Atom) -> bool {
    if atom_is_ground(atom) {
        db.contains(atom)
    } else {
        !db.query(atom).is_empty()
    }
}

fn atom_is_ground(atom: &Atom) -> bool {
    atom.terms.iter().all(term_is_ground)
}

fn term_is_ground(term: &Term) -> bool {
    match term {
        Term::Variable(_) => false,
        Term::Constant(_) => true,
        Term::Range(_, _) => true,
        Term::Compound(_, args) => args.iter().all(term_is_ground),
    }
}

/// Apply substitution to a built-in predicate
fn apply_subst_to_builtin(subst: &Substitution, builtin: &builtins::BuiltIn) -> builtins::BuiltIn {
    match builtin {
        builtins::BuiltIn::Comparison(op, left, right) => {
            builtins::BuiltIn::Comparison(op.clone(), subst.apply(left), subst.apply(right))
        }
        builtins::BuiltIn::True => builtins::BuiltIn::True,
        builtins::BuiltIn::Fail => builtins::BuiltIn::Fail,
    }
}

/// Ground a rule using semi-naive evaluation
/// For multi-literal rules, this tries using delta at each position
/// and the full database for other positions
pub fn ground_rule_semi_naive(
    rule: &Rule,
    delta: &FactDatabase,
    full_db: &FactDatabase,
) -> Vec<Atom> {
    let mut results = Vec::new();

    if rule.body.is_empty() {
        // No body - just return the head
        return vec![rule.head.clone()];
    }

    // For each position i in the body, use delta for position i and full_db for others
    for delta_pos in 0..rule.body.len() {
        let substs = satisfy_body_mixed(&rule.body, delta, full_db, delta_pos);
        for subst in substs {
            let ground_head = subst.apply_atom(&rule.head);
            results.push(ground_head);
        }
    }

    results
}

/// Satisfy body literals using delta for one position and full DB for others
fn satisfy_body_mixed(
    body: &[Literal],
    delta: &FactDatabase,
    full_db: &FactDatabase,
    delta_pos: usize,
) -> Vec<Substitution> {
    satisfy_body_with_selector(body, full_db, Some(delta), &|index, literal| {
        if index == delta_pos {
            match literal {
                Literal::Positive(atom) if builtins::parse_builtin(atom).is_some() => {
                    DatabaseSelection::Full
                }
                Literal::Positive(_) => DatabaseSelection::Delta,
                Literal::Negative(_) => DatabaseSelection::Full,
                Literal::Aggregate(_) => DatabaseSelection::Full,
            }
        } else {
            DatabaseSelection::Full
        }
    })
}

/// Expand a range term into concrete integer values
/// For example, Range(1, 3) becomes [1, 2, 3]
pub fn expand_range(start: &Term, end: &Term, const_env: &ConstantEnv) -> Option<Vec<i64>> {
    // Apply constant substitution first
    let start_term = const_env.substitute_term(start);
    let end_term = const_env.substitute_term(end);

    // Extract integer values
    let start_val = match &start_term {
        Term::Constant(Value::Integer(n)) => *n,
        _ => return None, // Can't expand non-integer ranges
    };

    let end_val = match &end_term {
        Term::Constant(Value::Integer(n)) => *n,
        _ => return None,
    };

    // Generate range (inclusive)
    let range: Vec<i64> = (start_val..=end_val).collect();
    Some(range)
}

/// Expand an atom containing ranges into multiple ground atoms
/// For example, cell(1..3, 5) becomes [cell(1,5), cell(2,5), cell(3,5)]
pub fn expand_atom_ranges(atom: &Atom, const_env: &ConstantEnv) -> Vec<Atom> {
    fn expand_terms_recursive(terms: &[Term], const_env: &ConstantEnv) -> Vec<Vec<Term>> {
        if terms.is_empty() {
            return vec![vec![]];
        }

        let first = &terms[0];
        let rest = &terms[1..];

        // Expand the first term
        let first_expansions: Vec<Term> = match first {
            Term::Range(start, end) => {
                if let Some(values) = expand_range(start, end, const_env) {
                    values
                        .into_iter()
                        .map(|n| Term::Constant(Value::Integer(n)))
                        .collect()
                } else {
                    vec![first.clone()] // Can't expand, keep as-is
                }
            }
            _ => vec![first.clone()],
        };

        // Recursively expand the rest
        let rest_expansions = expand_terms_recursive(rest, const_env);

        // Combine first expansions with rest expansions (cartesian product)
        let mut result = Vec::new();
        for first_term in &first_expansions {
            for rest_terms in &rest_expansions {
                let mut combined = vec![first_term.clone()];
                combined.extend(rest_terms.clone());
                result.push(combined);
            }
        }
        result
    }

    let expanded_term_lists = expand_terms_recursive(&atom.terms, const_env);

    expanded_term_lists
        .into_iter()
        .map(|terms| Atom {
            predicate: atom.predicate,
            terms,
        })
        .collect()
}

/// Ground a choice element by finding all substitutions that satisfy its condition
pub fn ground_choice_element(
    element: &ChoiceElement,
    db: &FactDatabase,
    const_env: &ConstantEnv,
) -> Vec<Atom> {
    // First expand any ranges in the atom
    let expanded_atoms = expand_atom_ranges(&element.atom, const_env);

    let mut result = Vec::new();

    for atom in expanded_atoms {
        if element.condition.is_empty() {
            // No condition - just add the atom
            result.push(atom);
        } else {
            // Find substitutions that satisfy the condition
            let substs = satisfy_body(&element.condition, db);

            // Apply each substitution to the atom
            for subst in substs {
                let ground_atom = subst.apply_atom(&atom);
                result.push(ground_atom);
            }
        }
    }

    result
}

/// Ground a choice rule by expanding all its elements
#[allow(dead_code)]
pub fn ground_choice_rule(
    choice: &ChoiceRule,
    db: &FactDatabase,
    const_env: &ConstantEnv,
) -> Vec<Atom> {
    let mut result = Vec::new();

    // If there's a body, we need to ground it first
    let body_substs = if choice.body.is_empty() {
        vec![Substitution::new()]
    } else {
        satisfy_body(&choice.body, db)
    };

    // For each way to satisfy the body
    for body_subst in body_substs {
        // Ground each choice element
        for element in &choice.elements {
            // Apply body substitution to element
            let element_with_body_subst = ChoiceElement {
                atom: body_subst.apply_atom(&element.atom),
                condition: element
                    .condition
                    .iter()
                    .map(|lit| apply_subst_to_literal(&body_subst, lit))
                    .collect(),
            };

            // Ground this element
            let grounded = ground_choice_element(&element_with_body_subst, db, const_env);
            result.extend(grounded);
        }
    }

    result
}

/// Ground a choice rule, splitting by body substitutions
/// Returns Vec<Vec<Atom>> where each inner Vec is an independent choice group
///
/// If the choice rule has no body, returns a single group with all grounded atoms.
/// If the choice rule has a body that grounds to N substitutions,
/// returns N groups, one per body substitution.
///
/// Example:
/// ```text
/// % 1 { has_weapon(P, W) : weapon(W) } 1 :- player(P).
/// % With player(alice), player(bob), weapon(sword), weapon(bow)
/// % Returns:
/// % [ [has_weapon(alice, sword), has_weapon(alice, bow)],
/// %   [has_weapon(bob, sword), has_weapon(bob, bow)] ]
/// ```
#[cfg_attr(not(test), allow(dead_code))]
pub fn ground_choice_rule_split(
    choice: &ChoiceRule,
    db: &FactDatabase,
    const_env: &ConstantEnv,
) -> Vec<Vec<Atom>> {
    // Get all body substitutions
    let body_substs = if choice.body.is_empty() {
        vec![Substitution::new()]
    } else {
        satisfy_body(&choice.body, db)
    };

    // Create one group per body substitution
    let mut groups = Vec::new();

    for body_subst in body_substs {
        let mut group = Vec::new();

        // Ground each choice element with this body substitution
        for element in &choice.elements {
            // Apply body substitution to element
            let element_with_body_subst = ChoiceElement {
                atom: body_subst.apply_atom(&element.atom),
                condition: element
                    .condition
                    .iter()
                    .map(|lit| apply_subst_to_literal(&body_subst, lit))
                    .collect(),
            };

            // Ground this element
            let grounded = ground_choice_element(&element_with_body_subst, db, const_env);
            group.extend(grounded);
        }

        groups.push(group);
    }

    groups
}

#[cfg(test)]
#[path = "../tests/unit/grounding_tests.rs"]
mod tests;
