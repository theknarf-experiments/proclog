//! Datalog evaluation strategies
//!
//! This module implements multiple evaluation algorithms for Datalog programs:
//!
//! # Evaluation Strategies
//!
//! - **Naive Evaluation**: Simple fixed-point iteration (re-evaluates all facts)
//! - **Semi-Naive Evaluation**: Optimized evaluation using deltas (only new facts)
//! - **Stratified Evaluation**: Handles negation safely by evaluating in strata
//!
//! # Constraint Checking
//!
//! Constraints are integrity constraints that must not be violated. They filter
//! out invalid models during evaluation.
//!
//! # Example
//!
//! ```ignore
//! let result = semi_naive_evaluation(&rules, initial_facts)?;
//! let result = stratified_evaluation_with_constraints(&rules, &constraints, initial_facts)?;
//! ```

use proclog_ast::{ComparisonOp, Constraint, Rule, Term, Value};
use proclog_core::{FactDatabase, InsertError};
use proclog_grounding::{ground_rule, ground_rule_semi_naive, satisfy_body};
use proclog_safety::{check_program_safety, stratify, SafetyError, StratificationError};

/// Errors that can occur during evaluation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluationError {
    /// Program violates safety rules
    Safety(SafetyError),
    /// Program is not stratifiable (cycle through negation)
    Stratification(StratificationError),
    /// Derived a non-ground fact during evaluation
    Derivation(InsertError),
    /// Constraint violation (integrity constraint failed)
    ConstraintViolation {
        constraint: String,
        violation_count: usize,
    },
}

impl std::fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvaluationError::Safety(e) => write!(f, "Safety error: {}", e),
            EvaluationError::Stratification(e) => {
                write!(f, "Stratification error: {}", e)
            }
            EvaluationError::Derivation(e) => write!(f, "Derivation error: {}", e),
            EvaluationError::ConstraintViolation {
                constraint,
                violation_count,
            } => {
                write!(
                    f,
                    "Constraint '{}' violated {} time(s)",
                    constraint, violation_count
                )
            }
        }
    }
}

impl std::error::Error for EvaluationError {}

impl From<SafetyError> for EvaluationError {
    fn from(e: SafetyError) -> Self {
        EvaluationError::Safety(e)
    }
}

impl From<StratificationError> for EvaluationError {
    fn from(e: StratificationError) -> Self {
        EvaluationError::Stratification(e)
    }
}

impl From<InsertError> for EvaluationError {
    fn from(e: InsertError) -> Self {
        EvaluationError::Derivation(e)
    }
}

/// Naive evaluation: repeatedly apply all rules until fixed point
/// This is simple but inefficient - it re-evaluates all facts every iteration
#[allow(dead_code)]
pub fn naive_evaluation(
    rules: &[Rule],
    initial_facts: FactDatabase,
) -> Result<FactDatabase, InsertError> {
    let mut db = initial_facts;
    let mut changed = true;

    while changed {
        changed = false;
        let old_size = db.len();

        // Apply each rule and add derived facts
        for rule in rules {
            let derived = ground_rule(rule, &db);
            for fact in derived {
                if db.insert(fact)? {
                    changed = true;
                }
            }
        }

        // If no new facts were added, we've reached fixed point
        if db.len() == old_size {
            changed = false;
        }
    }

    Ok(db)
}

/// Semi-naive evaluation: only process new facts (delta) each iteration
/// This is much more efficient for recursive rules
///
/// The key insight: for each rule, we need to ensure at least one literal
/// uses the delta (new facts), while others can use the full database.
/// This prevents re-deriving facts from old information.
pub fn semi_naive_evaluation(
    rules: &[Rule],
    initial_facts: FactDatabase,
) -> Result<FactDatabase, InsertError> {
    let mut db = initial_facts.clone();
    let mut delta = initial_facts;

    loop {
        let mut new_delta = FactDatabase::new();

        for rule in rules {
            if rule.body.is_empty() {
                // No body - always evaluate
                let derived = ground_rule(rule, &db);
                for fact in derived {
                    if db.contains(&fact) || new_delta.contains(&fact) {
                        continue;
                    }
                    new_delta.insert(fact)?;
                }
            } else if rule.body.len() == 1 {
                // Single literal - just use delta
                let derived = ground_rule(rule, &delta);
                for fact in derived {
                    if db.contains(&fact) || new_delta.contains(&fact) {
                        continue;
                    }
                    new_delta.insert(fact)?;
                }
            } else {
                // Multi-literal: use semi-naive grounding
                // This tries delta at each position
                let derived = ground_rule_semi_naive(rule, &delta, &db);
                for fact in derived {
                    if db.contains(&fact) || new_delta.contains(&fact) {
                        continue;
                    }
                    new_delta.insert(fact)?;
                }
            }
        }

        // If no new facts, we've reached fixed point
        if new_delta.is_empty() {
            break;
        }

        let delta_next = new_delta.clone();
        db.absorb(new_delta);
        delta = delta_next;
    }

    Ok(db)
}

/// Statistics about evaluation performance
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationStats {
    pub iterations: usize,
    pub rule_applications: usize,
    pub facts_derived: usize,
}

/// Instrumented naive evaluation that tracks statistics
#[allow(dead_code)]
pub fn naive_evaluation_instrumented(
    rules: &[Rule],
    initial_facts: FactDatabase,
) -> (FactDatabase, EvaluationStats) {
    let mut db = initial_facts;
    let mut changed = true;
    let mut stats = EvaluationStats {
        iterations: 0,
        rule_applications: 0,
        facts_derived: 0,
    };

    while changed {
        changed = false;
        let old_size = db.len();
        stats.iterations += 1;

        for rule in rules {
            stats.rule_applications += 1;
            let derived = ground_rule(rule, &db);
            for fact in derived {
                if db
                    .insert(fact)
                    .expect("derived non-ground fact during semi-naive evaluation")
                {
                    changed = true;
                    stats.facts_derived += 1;
                }
            }
        }

        if db.len() == old_size {
            changed = false;
        }
    }

    (db, stats)
}

/// Instrumented semi-naive evaluation that tracks statistics
#[allow(dead_code)]
pub fn semi_naive_evaluation_instrumented(
    rules: &[Rule],
    initial_facts: FactDatabase,
) -> Result<(FactDatabase, EvaluationStats), InsertError> {
    let mut db = initial_facts.clone();
    let mut delta = initial_facts;
    let mut stats = EvaluationStats {
        iterations: 0,
        rule_applications: 0,
        facts_derived: 0,
    };

    loop {
        let mut new_delta = FactDatabase::new();

        for rule in rules {
            stats.rule_applications += 1;

            if rule.body.is_empty() {
                let derived = ground_rule(rule, &db);
                for fact in derived {
                    if db.contains(&fact) || new_delta.contains(&fact) {
                        continue;
                    }
                    new_delta.insert(fact)?;
                    stats.facts_derived += 1;
                }
            } else if rule.body.len() == 1 {
                let derived = ground_rule(rule, &delta);
                for fact in derived {
                    if db.contains(&fact) || new_delta.contains(&fact) {
                        continue;
                    }
                    new_delta.insert(fact)?;
                    stats.facts_derived += 1;
                }
            } else {
                let derived = ground_rule_semi_naive(rule, &delta, &db);
                for fact in derived {
                    if db.contains(&fact) || new_delta.contains(&fact) {
                        continue;
                    }
                    new_delta.insert(fact)?;
                    stats.facts_derived += 1;
                }
            }
        }

        if new_delta.is_empty() {
            break;
        }

        stats.iterations += 1;
        let delta_next = new_delta.clone();
        db.absorb(new_delta);
        delta = delta_next;
    }

    Ok((db, stats))
}

/// Check constraints against the database
/// Returns an error if any constraint is violated
/// A constraint is violated if its body is satisfied (i.e., there exist substitutions)
pub fn check_constraints(
    constraints: &[Constraint],
    db: &FactDatabase,
) -> Result<(), EvaluationError> {
    use proclog_ast::Literal;

    for constraint in constraints {
        // Separate aggregates from other literals
        let non_agg_literals: Vec<_> = constraint
            .body
            .iter()
            .filter(|lit| !lit.is_aggregate())
            .cloned()
            .collect();

        let agg_literals: Vec<_> = constraint
            .body
            .iter()
            .filter_map(|lit| {
                if let Literal::Aggregate(agg) = lit {
                    Some(agg)
                } else {
                    None
                }
            })
            .collect();

        // Get all substitutions that satisfy non-aggregate literals
        let violations = satisfy_body(&non_agg_literals, db);

        if violations.is_empty() {
            // No violations from non-aggregate part
            continue;
        }

        // If there are aggregates, check if any aggregate is satisfied
        let mut actual_violations = Vec::new();
        for subst in &violations {
            // Check if all aggregates are satisfied with this substitution
            let all_aggregates_satisfied = agg_literals.iter().all(|agg| {
                // Apply substitution to aggregate value if needed
                let threshold = match &agg.value {
                    Term::Constant(Value::Integer(n)) => *n,
                    term => {
                        let applied_term = subst.apply(term);
                        match applied_term {
                            Term::Constant(Value::Integer(n)) => n,
                            _ => return false,
                        }
                    }
                };

                let substitutions = satisfy_body(&agg.elements, db);
                let count = substitutions.len() as i64;

                match agg.comparison {
                    ComparisonOp::Equal => count == threshold,
                    ComparisonOp::NotEqual => count != threshold,
                    ComparisonOp::LessThan => count < threshold,
                    ComparisonOp::LessOrEqual => count <= threshold,
                    ComparisonOp::GreaterThan => count > threshold,
                    ComparisonOp::GreaterOrEqual => count >= threshold,
                }
            });

            if all_aggregates_satisfied {
                actual_violations.push(subst.clone());
            }
        }

        let final_violations = if agg_literals.is_empty() {
            violations
        } else {
            actual_violations
        };

        if !final_violations.is_empty() {
            return Err(EvaluationError::ConstraintViolation {
                constraint: format!("{:?}", constraint.body),
                violation_count: final_violations.len(),
            });
        }
    }
    Ok(())
}

/// Stratified evaluation with constraints
/// Evaluates rules stratum by stratum, then checks constraints
pub fn stratified_evaluation_with_constraints(
    rules: &[Rule],
    constraints: &[Constraint],
    initial_facts: FactDatabase,
) -> Result<FactDatabase, EvaluationError> {
    // Check safety first (variables in negation, etc.)
    check_program_safety(rules)?;

    // Stratify the program
    let stratification = stratify(rules)?;

    let mut db = initial_facts;

    // Evaluate each stratum to completion before moving to next
    for stratum_rules in &stratification.rules_by_stratum {
        // Evaluate this stratum to fixed point using semi-naive
        db = semi_naive_evaluation(stratum_rules, db).map_err(EvaluationError::from)?;
    }

    // Check constraints after evaluation
    check_constraints(constraints, &db)?;

    Ok(db)
}

/// Stratified evaluation: evaluates program stratum by stratum
/// This allows safe handling of negation by ensuring negated predicates
/// are fully computed before being used
///
/// Note: This version doesn't check constraints. Use stratified_evaluation_with_constraints
/// if you need constraint checking.
#[allow(dead_code)]
pub fn stratified_evaluation(
    rules: &[Rule],
    initial_facts: FactDatabase,
) -> Result<FactDatabase, EvaluationError> {
    // Check safety first (variables in negation, etc.)
    check_program_safety(rules)?;

    // Stratify the program
    let stratification = stratify(rules)?;

    let mut db = initial_facts;

    // Evaluate each stratum to completion before moving to next
    for stratum_rules in &stratification.rules_by_stratum {
        // Evaluate this stratum to fixed point using semi-naive
        db = semi_naive_evaluation(stratum_rules, db).map_err(EvaluationError::from)?;
    }

    Ok(db)
}

#[cfg(test)]
#[path = "../tests/unit/evaluation_tests.rs"]
mod tests;
