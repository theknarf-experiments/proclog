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

use crate::ast::{Atom, ChoiceElement, ChoiceRule, Literal, Rule, Term, Value};
use crate::builtins;
use crate::constants::ConstantEnv;
use crate::database::FactDatabase;
use crate::unification::Substitution;

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
    if body.is_empty() {
        // Empty body is trivially satisfied with empty substitution
        return vec![Substitution::new()];
    }

    // Process first literal
    let first_literal = &body[0];
    let rest = &body[1..];

    match first_literal {
        Literal::Positive(atom) => {
            // Check if this is a built-in predicate
            if let Some(builtin) = builtins::parse_builtin(atom) {
                // This is a built-in - process the rest first, then filter
                if rest.is_empty() {
                    // No more literals - evaluate built-in with empty substitution
                    let empty_subst = Substitution::new();
                    match builtins::eval_builtin(&builtin, &empty_subst) {
                        Some(true) => vec![empty_subst],
                        _ => vec![],
                    }
                } else {
                    // Process rest first, then check built-in for each substitution
                    let rest_substs = satisfy_body(rest, db);
                    let mut result = Vec::new();

                    for subst in rest_substs {
                        // Apply substitution to the built-in and evaluate
                        let applied_builtin = apply_subst_to_builtin(&subst, &builtin);
                        match builtins::eval_builtin(&applied_builtin, &subst) {
                            Some(true) => result.push(subst),
                            _ => {} // Built-in failed, skip this substitution
                        }
                    }

                    result
                }
            } else {
                // Regular atom - query the database for matches
                let initial_substs = db.query(atom);

                if rest.is_empty() {
                    // No more literals to process
                    initial_substs
                } else {
                    // For each substitution from the first literal,
                    // apply it to the rest and continue
                    let mut all_substs = Vec::new();

                    for subst in initial_substs {
                        // Apply current substitution to remaining literals
                        let applied_rest: Vec<Literal> = rest
                            .iter()
                            .map(|lit| apply_subst_to_literal(&subst, lit))
                            .collect();

                        // Recursively satisfy the rest
                        let rest_substs = satisfy_body(&applied_rest, db);

                        // Combine substitutions
                        for rest_subst in rest_substs {
                            if let Some(combined) = combine_substs(&subst, &rest_subst) {
                                all_substs.push(combined);
                            }
                        }
                    }

                    all_substs
                }
            }
        }
        Literal::Negative(atom) => {
            // Negation as failure: the atom must NOT unify with any fact in the database
            // For each substitution from the rest of the body,
            // check that the atom (with substitution applied) doesn't match any fact

            if rest.is_empty() {
                // No more literals - check if the negated atom is NOT in the database
                let matches = db.query(atom);
                if matches.is_empty() {
                    // Atom is not in the database - negation succeeds with empty substitution
                    vec![Substitution::new()]
                } else {
                    // Atom is in the database - negation fails
                    vec![]
                }
            } else {
                // Process the rest first, then filter by negation
                let rest_substs = satisfy_body(rest, db);
                let mut result = Vec::new();

                for subst in rest_substs {
                    // Apply substitution to the negated atom
                    let ground_atom = subst.apply_atom(atom);

                    // Check if this ground atom exists in the database
                    let matches = db.query(&ground_atom);

                    if matches.is_empty() {
                        // Atom doesn't exist - negation succeeds
                        result.push(subst);
                    }
                    // If atom exists, negation fails - don't include this substitution
                }

                result
            }
        }
    }
}

/// Apply substitution to a literal
fn apply_subst_to_literal(subst: &Substitution, literal: &Literal) -> Literal {
    match literal {
        Literal::Positive(atom) => Literal::Positive(subst.apply_atom(atom)),
        Literal::Negative(atom) => Literal::Negative(subst.apply_atom(atom)),
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

        combined.bind(var.clone(), candidate);
    }

    Some(combined)
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
    satisfy_body_mixed_recursive(body, delta, full_db, delta_pos, 0)
}

/// Recursive helper for satisfy_body_mixed
fn satisfy_body_mixed_recursive(
    body: &[Literal],
    delta: &FactDatabase,
    full_db: &FactDatabase,
    delta_pos: usize,
    current_pos: usize,
) -> Vec<Substitution> {
    if body.is_empty() {
        return vec![Substitution::new()];
    }

    let first_literal = &body[0];
    let rest = &body[1..];

    match first_literal {
        Literal::Positive(atom) => {
            // Check if this is a built-in predicate
            if let Some(builtin) = crate::builtins::parse_builtin(atom) {
                // This is a built-in - process the rest first, then filter
                if rest.is_empty() {
                    // No more literals - evaluate built-in with empty substitution
                    let empty_subst = Substitution::new();
                    match crate::builtins::eval_builtin(&builtin, &empty_subst) {
                        Some(true) => vec![empty_subst],
                        _ => vec![],
                    }
                } else {
                    // Process rest first, then check built-in for each substitution
                    let rest_substs = satisfy_body_mixed_recursive(
                        rest,
                        delta,
                        full_db,
                        delta_pos,
                        current_pos + 1,
                    );
                    let mut result = Vec::new();

                    for subst in rest_substs {
                        // Apply substitution to the built-in and evaluate
                        let applied_builtin = apply_subst_to_builtin(&subst, &builtin);
                        match crate::builtins::eval_builtin(&applied_builtin, &subst) {
                            Some(true) => result.push(subst),
                            _ => {} // Built-in failed, skip this substitution
                        }
                    }

                    result
                }
            } else {
                // Regular atom - use delta if current position is delta_pos, otherwise use full_db
                let db = if current_pos == delta_pos {
                    delta
                } else {
                    full_db
                };
                let initial_substs = db.query(atom);

                if rest.is_empty() {
                    initial_substs
                } else {
                    let mut all_substs = Vec::new();

                    for subst in initial_substs {
                        let applied_rest: Vec<Literal> = rest
                            .iter()
                            .map(|lit| apply_subst_to_literal(&subst, lit))
                            .collect();

                        let rest_substs = satisfy_body_mixed_recursive(
                            &applied_rest,
                            delta,
                            full_db,
                            delta_pos,
                            current_pos + 1,
                        );

                        for rest_subst in rest_substs {
                            if let Some(combined) = combine_substs(&subst, &rest_subst) {
                                all_substs.push(combined);
                            }
                        }
                    }

                    all_substs
                }
            }
        }
        Literal::Negative(atom) => {
            // Negation uses full_db (not delta) - we need the complete view
            if rest.is_empty() {
                let matches = full_db.query(atom);
                if matches.is_empty() {
                    vec![Substitution::new()]
                } else {
                    vec![]
                }
            } else {
                let rest_substs =
                    satisfy_body_mixed_recursive(rest, delta, full_db, delta_pos, current_pos + 1);
                let mut result = Vec::new();

                for subst in rest_substs {
                    let ground_atom = subst.apply_atom(atom);
                    let matches = full_db.query(&ground_atom);

                    if matches.is_empty() {
                        result.push(subst);
                    }
                }

                result
            }
        }
    }
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
            predicate: atom.predicate.clone(),
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
/// ```
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
mod tests {
    use super::*;
    use internment::Intern;

    // Helper functions
    fn atom_const(name: &str) -> Term {
        Term::Constant(Value::Atom(Intern::new(name.to_string())))
    }

    fn int(n: i64) -> Term {
        Term::Constant(Value::Integer(n))
    }

    fn var(name: &str) -> Term {
        Term::Variable(Intern::new(name.to_string()))
    }

    fn make_atom(predicate: &str, terms: Vec<Term>) -> Atom {
        Atom {
            predicate: Intern::new(predicate.to_string()),
            terms,
        }
    }

    fn make_rule(head: Atom, body: Vec<Literal>) -> Rule {
        Rule { head, body }
    }

    #[test]
    fn test_combine_substs_conflicting_bindings_are_filtered() {
        let mut s1 = Substitution::new();
        s1.bind(Intern::new("X".to_string()), atom_const("a"));

        let mut s2 = Substitution::new();
        s2.bind(Intern::new("X".to_string()), atom_const("b"));

        assert!(combine_substs(&s1, &s2).is_none());
    }

    #[test]
    fn test_combine_substs_indirect_conflict_is_filtered() {
        let mut s1 = Substitution::new();
        s1.bind(Intern::new("X".to_string()), var("Y"));

        let mut s2 = Substitution::new();
        s2.bind(Intern::new("Y".to_string()), atom_const("a"));
        s2.bind(Intern::new("X".to_string()), atom_const("b"));

        assert!(combine_substs(&s1, &s2).is_none());
    }

    #[test]
    fn test_combine_substs_indirect_resolution_succeeds() {
        let mut s1 = Substitution::new();
        s1.bind(Intern::new("X".to_string()), var("Y"));

        let mut s2 = Substitution::new();
        s2.bind(Intern::new("Y".to_string()), atom_const("a"));
        s2.bind(Intern::new("X".to_string()), atom_const("a"));

        let combined = combine_substs(&s1, &s2).expect("substitutions should merge");

        assert_eq!(combined.apply(&var("X")), atom_const("a"));
        assert_eq!(combined.apply(&var("Y")), atom_const("a"));
    }

    // Basic grounding tests
    #[test]
    fn test_ground_rule_no_variables() {
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![atom_const("john"), atom_const("mary")],
        ));

        // Rule: ancestor(john, mary) :- parent(john, mary).
        let rule = make_rule(
            make_atom("ancestor", vec![atom_const("john"), atom_const("mary")]),
            vec![Literal::Positive(make_atom(
                "parent",
                vec![atom_const("john"), atom_const("mary")],
            ))],
        );

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].predicate, Intern::new("ancestor".to_string()));
    }

    #[test]
    fn test_ground_rule_single_variable() {
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![atom_const("john"), atom_const("mary")],
        ));
        db.insert(make_atom(
            "parent",
            vec![atom_const("john"), atom_const("bob")],
        ));

        // Rule: child(X) :- parent(john, X).
        let rule = make_rule(
            make_atom("child", vec![var("X")]),
            vec![Literal::Positive(make_atom(
                "parent",
                vec![atom_const("john"), var("X")],
            ))],
        );

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 2);

        // Both child(mary) and child(bob) should be generated
        let predicates: Vec<_> = results.iter().map(|a| &a.predicate).collect();
        assert!(predicates
            .iter()
            .all(|p| **p == Intern::new("child".to_string())));
    }

    #[test]
    fn test_ground_rule_multiple_variables() {
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![atom_const("john"), atom_const("mary")],
        ));
        db.insert(make_atom(
            "parent",
            vec![atom_const("bob"), atom_const("alice")],
        ));

        // Rule: ancestor(X, Y) :- parent(X, Y).
        let rule = make_rule(
            make_atom("ancestor", vec![var("X"), var("Y")]),
            vec![Literal::Positive(make_atom(
                "parent",
                vec![var("X"), var("Y")],
            ))],
        );

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ground_rule_join_two_literals() {
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![atom_const("john"), atom_const("mary")],
        ));
        db.insert(make_atom(
            "parent",
            vec![atom_const("mary"), atom_const("alice")],
        ));
        db.insert(make_atom(
            "parent",
            vec![atom_const("bob"), atom_const("charlie")],
        ));

        // Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        let rule = make_rule(
            make_atom("grandparent", vec![var("X"), var("Z")]),
            vec![
                Literal::Positive(make_atom("parent", vec![var("X"), var("Y")])),
                Literal::Positive(make_atom("parent", vec![var("Y"), var("Z")])),
            ],
        );

        let results = ground_rule(&rule, &db);

        // Only john -> mary -> alice forms a valid chain
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_const("john"));
        assert_eq!(results[0].terms[1], atom_const("alice"));
    }

    #[test]
    fn test_ground_rule_multiple_chains() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("parent", vec![atom_const("a"), atom_const("b")]));
        db.insert(make_atom("parent", vec![atom_const("b"), atom_const("c")]));
        db.insert(make_atom("parent", vec![atom_const("b"), atom_const("d")]));

        // Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        let rule = make_rule(
            make_atom("grandparent", vec![var("X"), var("Z")]),
            vec![
                Literal::Positive(make_atom("parent", vec![var("X"), var("Y")])),
                Literal::Positive(make_atom("parent", vec![var("Y"), var("Z")])),
            ],
        );

        let results = ground_rule(&rule, &db);

        // a -> b -> c and a -> b -> d
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ground_rule_no_matches() {
        let mut db = FactDatabase::new();
        db.insert(make_atom(
            "parent",
            vec![atom_const("john"), atom_const("mary")],
        ));

        // Rule: child(X) :- parent(alice, X).
        // No facts match parent(alice, X)
        let rule = make_rule(
            make_atom("child", vec![var("X")]),
            vec![Literal::Positive(make_atom(
                "parent",
                vec![atom_const("alice"), var("X")],
            ))],
        );

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_ground_rule_empty_body() {
        let db = FactDatabase::new();

        // Rule: fact(a) :- .
        // (A rule with no body is always true)
        let rule = make_rule(make_atom("fact", vec![atom_const("a")]), vec![]);

        let results = ground_rule(&rule, &db);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], make_atom("fact", vec![atom_const("a")]));
    }

    #[test]
    fn test_ground_rule_three_literals() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("edge", vec![atom_const("a"), atom_const("b")]));
        db.insert(make_atom("edge", vec![atom_const("b"), atom_const("c")]));
        db.insert(make_atom("edge", vec![atom_const("c"), atom_const("d")]));

        // Rule: path3(X, W) :- edge(X, Y), edge(Y, Z), edge(Z, W).
        let rule = make_rule(
            make_atom("path3", vec![var("X"), var("W")]),
            vec![
                Literal::Positive(make_atom("edge", vec![var("X"), var("Y")])),
                Literal::Positive(make_atom("edge", vec![var("Y"), var("Z")])),
                Literal::Positive(make_atom("edge", vec![var("Z"), var("W")])),
            ],
        );

        let results = ground_rule(&rule, &db);

        // Only a -> b -> c -> d
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_const("a"));
        assert_eq!(results[0].terms[1], atom_const("d"));
    }

    // Integration test: Parse → Load Facts → Ground Rules → Query
    // Negation tests
    #[test]
    fn test_ground_rule_simple_negation() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("bird", vec![atom_const("tweety")]));
        db.insert(make_atom("bird", vec![atom_const("polly")]));
        db.insert(make_atom("penguin", vec![atom_const("polly")]));

        // Rule: flies(X) :- bird(X), not penguin(X).
        let rule = make_rule(
            make_atom("flies", vec![var("X")]),
            vec![
                Literal::Positive(make_atom("bird", vec![var("X")])),
                Literal::Negative(make_atom("penguin", vec![var("X")])),
            ],
        );

        let results = ground_rule(&rule, &db);

        // Only tweety should fly (polly is a penguin)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].terms[0], atom_const("tweety"));
    }

    #[test]
    fn test_ground_rule_negation_no_match() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("bird", vec![atom_const("tweety")]));

        // Rule: mammal(X) :- not bird(X).
        // Since only tweety exists and is a bird, no mammals
        let rule = make_rule(
            make_atom("mammal", vec![var("X")]),
            vec![Literal::Negative(make_atom("bird", vec![var("X")]))],
        );

        let results = ground_rule(&rule, &db);

        // No results - we can't prove something is NOT a bird
        // unless we have a closed world assumption with a finite domain
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_ground_rule_negation_with_ground_term() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("bird", vec![atom_const("tweety")]));

        // Rule: not_bird_polly :- not bird(polly).
        let rule = make_rule(
            make_atom("not_bird_polly", vec![]),
            vec![Literal::Negative(make_atom(
                "bird",
                vec![atom_const("polly")],
            ))],
        );

        let results = ground_rule(&rule, &db);

        // polly is not a bird, so this succeeds
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_ground_rule_multiple_negations() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("a", vec![atom_const("x")]));
        db.insert(make_atom("b", vec![atom_const("y")]));
        db.insert(make_atom("c", vec![atom_const("z")]));

        // Rule: result :- not a(y), not b(x), not c(w).
        let rule = make_rule(
            make_atom("result", vec![]),
            vec![
                Literal::Negative(make_atom("a", vec![atom_const("y")])),
                Literal::Negative(make_atom("b", vec![atom_const("x")])),
                Literal::Negative(make_atom("c", vec![atom_const("w")])),
            ],
        );

        let results = ground_rule(&rule, &db);

        // All negations succeed
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_integration_parse_ground_query() {
        use crate::ast::Statement;
        use crate::parser;

        // A complete ProcLog program with facts and rules
        let program_text = r#"
            % Facts about family relationships
            parent(john, mary).
            parent(mary, alice).
            parent(bob, charlie).
            parent(charlie, dave).

            % Rule: grandparent relationship
            grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        "#;

        // Step 1: Parse the program
        let program = parser::parse_program(program_text).expect("Should parse successfully");

        // Step 2: Load facts into database
        let mut db = FactDatabase::new();
        let mut rules = Vec::new();

        for statement in program.statements {
            match statement {
                Statement::Fact(fact) => {
                    db.insert(fact.atom);
                }
                Statement::Rule(rule) => {
                    rules.push(rule);
                }
                _ => {} // Ignore other statement types for this test
            }
        }

        // Verify we loaded 4 facts and 1 rule
        assert_eq!(db.len(), 4);
        assert_eq!(rules.len(), 1);

        // Step 3: Ground the rule
        let grandparent_rule = &rules[0];
        let derived_facts = ground_rule(grandparent_rule, &db);

        // Should derive two grandparent facts:
        // grandparent(john, alice) from parent(john, mary) + parent(mary, alice)
        // grandparent(bob, dave) from parent(bob, charlie) + parent(charlie, dave)
        assert_eq!(derived_facts.len(), 2);

        // Step 4: Add derived facts to database
        for fact in derived_facts {
            db.insert(fact);
        }

        // Step 5: Query for grandparents
        let grandparent_pred = Intern::new("grandparent".to_string());
        let grandparent_facts = db.get_by_predicate(&grandparent_pred);

        assert_eq!(grandparent_facts.len(), 2);

        // Verify we can query with patterns
        let query_pattern = make_atom("grandparent", vec![atom_const("john"), var("Z")]);
        let results = db.query(&query_pattern);

        assert_eq!(results.len(), 1);
        let z = Intern::new("Z".to_string());
        assert_eq!(results[0].get(&z), Some(&atom_const("alice")));

        // Query for all grandparent relationships
        let all_grandparents = make_atom("grandparent", vec![var("X"), var("Z")]);
        let all_results = db.query(&all_grandparents);

        assert_eq!(all_results.len(), 2);
    }

    // Range expansion tests
    #[test]
    fn test_expand_range_simple() {
        let const_env = ConstantEnv::new();
        let start = Term::Constant(Value::Integer(1));
        let end = Term::Constant(Value::Integer(3));

        let result = expand_range(&start, &end, &const_env);
        assert_eq!(result, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_expand_range_with_constants() {
        let mut const_env = ConstantEnv::new();
        const_env.define(Intern::new("max".to_string()), Value::Integer(5));

        let start = Term::Constant(Value::Integer(3));
        let end = Term::Constant(Value::Atom(Intern::new("max".to_string())));

        let result = expand_range(&start, &end, &const_env);
        assert_eq!(result, Some(vec![3, 4, 5]));
    }

    #[test]
    fn test_expand_atom_ranges_single_range() {
        let const_env = ConstantEnv::new();
        let atom = make_atom(
            "cell",
            vec![
                Term::Range(
                    Box::new(Term::Constant(Value::Integer(1))),
                    Box::new(Term::Constant(Value::Integer(3))),
                ),
                atom_const("a"),
            ],
        );

        let result = expand_atom_ranges(&atom, &const_env);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], make_atom("cell", vec![int(1), atom_const("a")]));
        assert_eq!(result[1], make_atom("cell", vec![int(2), atom_const("a")]));
        assert_eq!(result[2], make_atom("cell", vec![int(3), atom_const("a")]));
    }

    #[test]
    fn test_expand_atom_ranges_multiple_ranges() {
        let const_env = ConstantEnv::new();
        let atom = make_atom(
            "cell",
            vec![
                Term::Range(
                    Box::new(Term::Constant(Value::Integer(1))),
                    Box::new(Term::Constant(Value::Integer(2))),
                ),
                Term::Range(
                    Box::new(Term::Constant(Value::Integer(10))),
                    Box::new(Term::Constant(Value::Integer(11))),
                ),
            ],
        );

        let result = expand_atom_ranges(&atom, &const_env);
        assert_eq!(result.len(), 4); // 2 * 2 = 4 combinations
        assert_eq!(result[0], make_atom("cell", vec![int(1), int(10)]));
        assert_eq!(result[1], make_atom("cell", vec![int(1), int(11)]));
        assert_eq!(result[2], make_atom("cell", vec![int(2), int(10)]));
        assert_eq!(result[3], make_atom("cell", vec![int(2), int(11)]));
    }

    #[test]
    fn test_ground_choice_element_no_condition() {
        let db = FactDatabase::new();
        let const_env = ConstantEnv::new();

        let element = ChoiceElement {
            atom: make_atom("selected", vec![atom_const("a")]),
            condition: vec![],
        };

        let result = ground_choice_element(&element, &db, &const_env);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], make_atom("selected", vec![atom_const("a")]));
    }

    #[test]
    fn test_ground_choice_element_with_condition() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("item", vec![atom_const("a")]));
        db.insert(make_atom("item", vec![atom_const("b")]));

        let const_env = ConstantEnv::new();

        let element = ChoiceElement {
            atom: make_atom("selected", vec![var("X")]),
            condition: vec![Literal::Positive(make_atom("item", vec![var("X")]))],
        };

        let result = ground_choice_element(&element, &db, &const_env);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&make_atom("selected", vec![atom_const("a")])));
        assert!(result.contains(&make_atom("selected", vec![atom_const("b")])));
    }

    #[test]
    fn test_ground_choice_element_with_range() {
        let db = FactDatabase::new();
        let const_env = ConstantEnv::new();

        let element = ChoiceElement {
            atom: make_atom(
                "cell",
                vec![
                    Term::Range(
                        Box::new(Term::Constant(Value::Integer(1))),
                        Box::new(Term::Constant(Value::Integer(2))),
                    ),
                    atom_const("solid"),
                ],
            ),
            condition: vec![],
        };

        let result = ground_choice_element(&element, &db, &const_env);
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0],
            make_atom("cell", vec![int(1), atom_const("solid")])
        );
        assert_eq!(
            result[1],
            make_atom("cell", vec![int(2), atom_const("solid")])
        );
    }

    // Edge case tests for range expansion

    #[test]
    fn test_expand_range_backwards() {
        // Backwards range should produce empty result
        let const_env = ConstantEnv::new();
        let start = Term::Constant(Value::Integer(5));
        let end = Term::Constant(Value::Integer(1));

        let result = expand_range(&start, &end, &const_env);
        // Backwards range produces empty vector
        assert_eq!(result, Some(vec![]));
    }

    #[test]
    fn test_expand_range_single_element() {
        let const_env = ConstantEnv::new();
        let start = Term::Constant(Value::Integer(5));
        let end = Term::Constant(Value::Integer(5));

        let result = expand_range(&start, &end, &const_env);
        assert_eq!(result, Some(vec![5]));
    }

    #[test]
    fn test_expand_range_negative_values() {
        let const_env = ConstantEnv::new();
        let start = Term::Constant(Value::Integer(-3));
        let end = Term::Constant(Value::Integer(-1));

        let result = expand_range(&start, &end, &const_env);
        assert_eq!(result, Some(vec![-3, -2, -1]));
    }

    #[test]
    fn test_expand_range_crossing_zero() {
        let const_env = ConstantEnv::new();
        let start = Term::Constant(Value::Integer(-2));
        let end = Term::Constant(Value::Integer(2));

        let result = expand_range(&start, &end, &const_env);
        assert_eq!(result, Some(vec![-2, -1, 0, 1, 2]));
    }

    #[test]
    fn test_expand_range_with_undefined_constant() {
        let const_env = ConstantEnv::new();
        let start = Term::Constant(Value::Integer(1));
        let end = Term::Constant(Value::Atom(Intern::new("undefined".to_string())));

        let result = expand_range(&start, &end, &const_env);
        // Undefined constant can't be expanded
        assert_eq!(result, None);
    }

    #[test]
    fn test_expand_range_with_non_integer_constant() {
        let const_env = ConstantEnv::new();
        let start = Term::Constant(Value::Atom(Intern::new("not_a_number".to_string())));
        let end = Term::Constant(Value::Integer(5));

        let result = expand_range(&start, &end, &const_env);
        assert_eq!(result, None);
    }

    #[test]
    fn test_expand_atom_ranges_no_ranges() {
        let const_env = ConstantEnv::new();
        let atom = make_atom("cell", vec![atom_const("a"), atom_const("b")]);

        let result = expand_atom_ranges(&atom, &const_env);
        // No ranges, should return single atom unchanged
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], atom);
    }

    #[test]
    fn test_expand_atom_ranges_empty_range() {
        let const_env = ConstantEnv::new();
        // 5..1 is a backwards range (empty)
        let atom = make_atom(
            "cell",
            vec![
                Term::Range(
                    Box::new(Term::Constant(Value::Integer(5))),
                    Box::new(Term::Constant(Value::Integer(1))),
                ),
                atom_const("x"),
            ],
        );

        let result = expand_atom_ranges(&atom, &const_env);
        // Empty range produces no atoms
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_expand_atom_ranges_mixed_terms() {
        let const_env = ConstantEnv::new();
        let atom = make_atom(
            "cell",
            vec![
                atom_const("prefix"),
                Term::Range(
                    Box::new(Term::Constant(Value::Integer(1))),
                    Box::new(Term::Constant(Value::Integer(2))),
                ),
                atom_const("suffix"),
            ],
        );

        let result = expand_atom_ranges(&atom, &const_env);
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0],
            make_atom(
                "cell",
                vec![atom_const("prefix"), int(1), atom_const("suffix")]
            )
        );
        assert_eq!(
            result[1],
            make_atom(
                "cell",
                vec![atom_const("prefix"), int(2), atom_const("suffix")]
            )
        );
    }

    #[test]
    fn test_ground_choice_element_empty_condition_result() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("item", vec![atom_const("a")]));

        let const_env = ConstantEnv::new();

        // Condition that won't match anything
        let element = ChoiceElement {
            atom: make_atom("selected", vec![var("X")]),
            condition: vec![Literal::Positive(make_atom("nonexistent", vec![var("X")]))],
        };

        let result = ground_choice_element(&element, &db, &const_env);
        // No items match the condition
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_ground_choice_element_multiple_conditions() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("item", vec![atom_const("a")]));
        db.insert(make_atom("item", vec![atom_const("b")]));
        db.insert(make_atom("valid", vec![atom_const("a")]));
        db.insert(make_atom("safe", vec![atom_const("a")]));

        let const_env = ConstantEnv::new();

        // Multiple conditions: both must be satisfied
        let element = ChoiceElement {
            atom: make_atom("selected", vec![var("X")]),
            condition: vec![
                Literal::Positive(make_atom("item", vec![var("X")])),
                Literal::Positive(make_atom("valid", vec![var("X")])),
                Literal::Positive(make_atom("safe", vec![var("X")])),
            ],
        };

        let result = ground_choice_element(&element, &db, &const_env);
        // Only 'a' satisfies all three conditions
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], make_atom("selected", vec![atom_const("a")]));
    }

    #[test]
    fn test_ground_choice_rule_with_body_variables() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("room", vec![atom_const("r1")]));
        db.insert(make_atom("item", vec![atom_const("sword")]));
        db.insert(make_atom("item", vec![atom_const("shield")]));

        let const_env = ConstantEnv::new();

        // Choice rule with body
        let choice = ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: make_atom("place", vec![var("I"), var("R")]),
                condition: vec![Literal::Positive(make_atom("item", vec![var("I")]))],
            }],
            body: vec![Literal::Positive(make_atom("room", vec![var("R")]))],
        };

        let result = ground_choice_rule(&choice, &db, &const_env);
        // 1 room * 2 items = 2 possible placements
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_ground_choice_rule_empty_body() {
        let mut db = FactDatabase::new();
        db.insert(make_atom("item", vec![atom_const("a")]));

        let const_env = ConstantEnv::new();

        let choice = ChoiceRule {
            lower_bound: Some(Term::Constant(Value::Integer(1))),
            upper_bound: Some(Term::Constant(Value::Integer(1))),
            elements: vec![ChoiceElement {
                atom: make_atom("selected", vec![var("X")]),
                condition: vec![Literal::Positive(make_atom("item", vec![var("X")]))],
            }],
            body: vec![],
        };

        let result = ground_choice_rule(&choice, &db, &const_env);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], make_atom("selected", vec![atom_const("a")]));
    }

    #[test]
    fn test_satisfy_body_with_builtin_comparison() {
        // Test: number(X), X > 5 with number(7) in the database
        let mut db = FactDatabase::new();
        db.insert(make_atom("number", vec![Term::Constant(Value::Integer(7))]));
        db.insert(make_atom("number", vec![Term::Constant(Value::Integer(3))]));

        let body = vec![
            Literal::Positive(make_atom("number", vec![var("X")])),
            Literal::Positive(Atom {
                predicate: Intern::new(">".to_string()),
                terms: vec![var("X"), Term::Constant(Value::Integer(5))],
            }),
        ];

        let result = satisfy_body(&body, &db);

        // Should have exactly one substitution: {X -> 7}
        assert_eq!(
            result.len(),
            1,
            "Expected 1 substitution, got {}: {:#?}",
            result.len(),
            result
        );

        let x_var = Intern::new("X".to_string());
        let bound_value = result[0].get(&x_var).expect("X should be bound");
        match bound_value {
            Term::Constant(Value::Integer(7)) => {}
            _ => panic!("Expected X to be bound to 7, got {:?}", bound_value),
        }
    }

    #[test]
    fn test_ground_rule_with_builtin_comparison() {
        // Test: large(X) :- number(X), X > 5
        let mut db = FactDatabase::new();
        db.insert(make_atom("number", vec![Term::Constant(Value::Integer(7))]));
        db.insert(make_atom("number", vec![Term::Constant(Value::Integer(3))]));

        let rule = Rule {
            head: make_atom("large", vec![var("X")]),
            body: vec![
                Literal::Positive(make_atom("number", vec![var("X")])),
                Literal::Positive(Atom {
                    predicate: Intern::new(">".to_string()),
                    terms: vec![var("X"), Term::Constant(Value::Integer(5))],
                }),
            ],
        };

        let result = ground_rule(&rule, &db);

        // Should derive large(7)
        assert_eq!(
            result.len(),
            1,
            "Expected 1 derived fact, got {}: {:#?}",
            result.len(),
            result
        );

        match &result[0].terms[0] {
            Term::Constant(Value::Integer(7)) => {}
            other => panic!("Expected large(7), got large({:?})", other),
        }
    }

    // Tests for choice rule body splitting
    #[test]
    fn test_ground_choice_rule_no_body_returns_single_group() {
        // Choice rule with no body: { selected(X) : item(X) }.
        // Should return ONE group containing all grounded atoms
        let mut db = FactDatabase::new();
        db.insert(make_atom("item", vec![atom_const("a")]));
        db.insert(make_atom("item", vec![atom_const("b")]));

        let const_env = ConstantEnv::new();

        let choice = ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: make_atom("selected", vec![var("X")]),
                condition: vec![Literal::Positive(make_atom("item", vec![var("X")]))],
            }],
            body: vec![], // NO BODY
        };

        // Current function returns Vec<Atom>, but we need Vec<Vec<Atom>>
        // For now, test what we have
        let result = ground_choice_rule(&choice, &db, &const_env);

        // Should have 2 atoms: selected(a), selected(b)
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_ground_choice_rule_with_body_should_split() {
        // Choice rule with body: 1 { has_weapon(P, W) : weapon(W) } 1 :- player(P).
        // Should return TWO groups (one per player)
        let mut db = FactDatabase::new();
        db.insert(make_atom("player", vec![atom_const("alice")]));
        db.insert(make_atom("player", vec![atom_const("bob")]));
        db.insert(make_atom("weapon", vec![atom_const("sword")]));
        db.insert(make_atom("weapon", vec![atom_const("bow")]));

        let const_env = ConstantEnv::new();

        let choice = ChoiceRule {
            lower_bound: Some(Term::Constant(Value::Integer(1))),
            upper_bound: Some(Term::Constant(Value::Integer(1))),
            elements: vec![ChoiceElement {
                atom: make_atom("has_weapon", vec![var("P"), var("W")]),
                condition: vec![Literal::Positive(make_atom("weapon", vec![var("W")]))],
            }],
            body: vec![Literal::Positive(make_atom("player", vec![var("P")]))],
        };

        // Current implementation - returns flat list of 4 atoms
        let result = ground_choice_rule(&choice, &db, &const_env);

        // Currently returns 4 atoms (all combined)
        assert_eq!(result.len(), 4);

        // But we WANT 2 groups of 2 atoms each
        // This test documents the CURRENT (incorrect) behavior
        // We'll add a new function that returns Vec<Vec<Atom>>
    }

    #[test]
    fn test_ground_choice_rule_split_no_body() {
        // NEW FUNCTION: ground_choice_rule_split
        // With no body, should return single group
        let mut db = FactDatabase::new();
        db.insert(make_atom("item", vec![atom_const("a")]));
        db.insert(make_atom("item", vec![atom_const("b")]));

        let const_env = ConstantEnv::new();

        let choice = ChoiceRule {
            lower_bound: None,
            upper_bound: None,
            elements: vec![ChoiceElement {
                atom: make_atom("selected", vec![var("X")]),
                condition: vec![Literal::Positive(make_atom("item", vec![var("X")]))],
            }],
            body: vec![],
        };

        let result = ground_choice_rule_split(&choice, &db, &const_env);

        // Should return 1 group with 2 atoms
        assert_eq!(result.len(), 1, "Expected 1 group");
        assert_eq!(result[0].len(), 2, "Expected 2 atoms in the group");

        // Verify atoms
        let atoms: Vec<String> = result[0]
            .iter()
            .map(|a| format!("{:?}", a.predicate))
            .collect();
        assert!(atoms.iter().all(|s| s.contains("selected")));
    }

    #[test]
    fn test_ground_choice_rule_split_with_body() {
        // NEW FUNCTION: ground_choice_rule_split
        // With body that grounds to 2 substitutions, should return 2 groups
        let mut db = FactDatabase::new();
        db.insert(make_atom("player", vec![atom_const("alice")]));
        db.insert(make_atom("player", vec![atom_const("bob")]));
        db.insert(make_atom("weapon", vec![atom_const("sword")]));
        db.insert(make_atom("weapon", vec![atom_const("bow")]));

        let const_env = ConstantEnv::new();

        let choice = ChoiceRule {
            lower_bound: Some(Term::Constant(Value::Integer(1))),
            upper_bound: Some(Term::Constant(Value::Integer(1))),
            elements: vec![ChoiceElement {
                atom: make_atom("has_weapon", vec![var("P"), var("W")]),
                condition: vec![Literal::Positive(make_atom("weapon", vec![var("W")]))],
            }],
            body: vec![Literal::Positive(make_atom("player", vec![var("P")]))],
        };

        let result = ground_choice_rule_split(&choice, &db, &const_env);

        // Should return 2 groups (one per player)
        assert_eq!(result.len(), 2, "Expected 2 groups (one per player)");

        // Each group should have 2 atoms (one per weapon)
        assert_eq!(result[0].len(), 2, "Expected 2 atoms in first group");
        assert_eq!(result[1].len(), 2, "Expected 2 atoms in second group");

        // Verify first group is for one player (either alice or bob)
        let first_group_player = &result[0][0].terms[0];
        assert!(
            result[0]
                .iter()
                .all(|atom| &atom.terms[0] == first_group_player),
            "All atoms in first group should be for same player"
        );

        // Verify second group is for the other player
        let second_group_player = &result[1][0].terms[0];
        assert!(
            result[1]
                .iter()
                .all(|atom| &atom.terms[0] == second_group_player),
            "All atoms in second group should be for same player"
        );

        // Verify different players
        assert_ne!(
            first_group_player, second_group_player,
            "Each group should be for a different player"
        );
    }
}
