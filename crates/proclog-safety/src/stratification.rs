//! Stratification analysis for programs with negation
//!
//! This module analyzes programs to determine if they can be safely evaluated
//! with negation. A program is stratifiable if there are no cycles through negation.
//!
//! # Stratification
//!
//! Stratification assigns each predicate to a stratum (layer). Predicates in higher
//! strata can depend on negated predicates from lower strata, but not vice versa.
//!
//! # Algorithm
//!
//! 1. Build dependency graph of predicates
//! 2. Detect negative cycles (cycles through negated edges)
//! 3. Assign strata using topological sort
//!
//! # Example
//!
//! ```ignore
//! let stratification = stratify(&rules)?;
//! for (stratum, rules_in_stratum) in stratification.iter() {
//!     // Evaluate rules stratum by stratum
//! }
//! ```

use proclog_ast::{Literal, Rule, Symbol};
use std::collections::{HashMap, HashSet};

/// Result of stratification analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stratification {
    /// Map from predicate name to stratum number (0 = bottom stratum)
    pub predicate_strata: HashMap<Symbol, usize>,
    /// Rules organized by stratum
    pub rules_by_stratum: Vec<Vec<Rule>>,
    /// Total number of strata
    pub num_strata: usize,
}

/// Error during stratification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StratificationError {
    /// Program has a cycle through negation (not stratifiable)
    CycleThroughNegation(Vec<Symbol>),
}

impl std::fmt::Display for StratificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StratificationError::CycleThroughNegation(cycle) => {
                if cycle.is_empty() {
                    write!(
                        f,
                        "Cycle through negation detected (cycle could not be reconstructed)"
                    )
                } else {
                    let cycle_sequence: Vec<String> = cycle
                        .iter()
                        .map(|symbol| symbol.as_ref().to_string())
                        .collect();
                    let mut display_cycle = cycle_sequence.clone();
                    if let Some(first) = cycle_sequence.first() {
                        display_cycle.push(first.clone());
                    }

                    write!(
                        f,
                        "Cycle through negation detected: {}",
                        display_cycle.join(" -> ")
                    )
                }
            }
        }
    }
}

/// Dependency between predicates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DependencyType {
    Positive, // p depends positively on q
    Negative, // p depends negatively on q (through negation)
}

/// Dependency graph for stratification analysis
#[derive(Debug, Clone)]
struct DependencyGraph {
    /// Map from predicate to its dependencies
    dependencies: HashMap<Symbol, Vec<(Symbol, DependencyType)>>,
    /// All predicates in the program
    predicates: HashSet<Symbol>,
}

impl DependencyGraph {
    fn new() -> Self {
        DependencyGraph {
            dependencies: HashMap::new(),
            predicates: HashSet::new(),
        }
    }

    /// Add a dependency edge
    fn add_dependency(&mut self, from: Symbol, to: Symbol, dep_type: DependencyType) {
        self.predicates.insert(from);
        self.predicates.insert(to);

        self.dependencies
            .entry(from)
            .or_default()
            .push((to, dep_type));
    }

    /// Get all dependencies of a predicate
    fn get_dependencies(&self, pred: &Symbol) -> Vec<(Symbol, DependencyType)> {
        self.dependencies.get(pred).cloned().unwrap_or_default()
    }
}

/// Build dependency graph from rules
fn build_dependency_graph(rules: &[Rule]) -> DependencyGraph {
    let mut graph = DependencyGraph::new();

    for rule in rules {
        let head_pred = rule.head.predicate;

        for literal in &rule.body {
            match literal {
                Literal::Positive(atom) => {
                    graph.add_dependency(head_pred, atom.predicate, DependencyType::Positive);
                }
                Literal::Negative(atom) => {
                    graph.add_dependency(head_pred, atom.predicate, DependencyType::Negative);
                }
                Literal::Aggregate(agg) => {
                    // Aggregates create negative dependencies (like negation)
                    // They depend on all predicates in their elements
                    for elem_literal in &agg.elements {
                        if let Some(atom) = elem_literal.atom() {
                            graph.add_dependency(
                                head_pred,
                                atom.predicate,
                                DependencyType::Negative,
                            );
                        }
                    }
                }
            }
        }
    }

    graph
}

/// Check if there's a path from 'from' to 'to' through a negative edge
fn has_cycle_through_negation(
    graph: &DependencyGraph,
    from: &Symbol,
    to: &Symbol,
    visited: &mut HashSet<Symbol>,
    has_negative: bool,
) -> bool {
    if from == to && has_negative {
        return true;
    }

    if visited.contains(from) {
        return false;
    }

    visited.insert(*from);

    for (dep, dep_type) in graph.get_dependencies(from) {
        let is_negative = has_negative || matches!(dep_type, DependencyType::Negative);

        if has_cycle_through_negation(graph, &dep, to, visited, is_negative) {
            return true;
        }
    }

    visited.remove(from);
    false
}

/// Detect cycles through negation in the dependency graph
fn detect_negative_cycles(graph: &DependencyGraph) -> Option<Vec<Symbol>> {
    for pred in &graph.predicates {
        let mut visited = HashSet::new();
        if has_cycle_through_negation(graph, pred, pred, &mut visited, false) {
            return Some(vec![*pred]);
        }
    }
    None
}

/// Compute stratum for each predicate using iterative algorithm
/// A predicate's stratum is the maximum stratum of its negated dependencies + 1
fn compute_strata(graph: &DependencyGraph) -> HashMap<Symbol, usize> {
    let mut strata: HashMap<Symbol, usize> = HashMap::new();

    // Initialize all predicates to stratum 0
    for pred in &graph.predicates {
        strata.insert(*pred, 0);
    }

    // Iterate until fixed point
    let mut changed = true;
    while changed {
        changed = false;

        for pred in &graph.predicates {
            let mut max_stratum = 0;

            for (dep, dep_type) in graph.get_dependencies(pred) {
                let dep_stratum = *strata.get(&dep).unwrap_or(&0);

                let required_stratum = match dep_type {
                    DependencyType::Positive => dep_stratum,
                    DependencyType::Negative => dep_stratum + 1, // Must be after negated predicate
                };

                max_stratum = max_stratum.max(required_stratum);
            }

            if max_stratum > *strata.get(pred).unwrap() {
                strata.insert(*pred, max_stratum);
                changed = true;
            }
        }
    }

    strata
}

/// Stratify a program
pub fn stratify(rules: &[Rule]) -> Result<Stratification, StratificationError> {
    if rules.is_empty() {
        return Ok(Stratification {
            predicate_strata: HashMap::new(),
            rules_by_stratum: vec![],
            num_strata: 0,
        });
    }

    // Build dependency graph
    let graph = build_dependency_graph(rules);

    // Check for cycles through negation
    if let Some(cycle) = detect_negative_cycles(&graph) {
        return Err(StratificationError::CycleThroughNegation(cycle));
    }

    // Compute strata
    let predicate_strata = compute_strata(&graph);

    // Find maximum stratum
    let num_strata = predicate_strata.values().max().copied().unwrap_or(0) + 1;

    // Organize rules by stratum
    let mut rules_by_stratum: Vec<Vec<Rule>> = vec![Vec::new(); num_strata];

    for rule in rules {
        let stratum = *predicate_strata.get(&rule.head.predicate).unwrap_or(&0);
        rules_by_stratum[stratum].push(rule.clone());
    }

    Ok(Stratification {
        predicate_strata,
        rules_by_stratum,
        num_strata,
    })
}

#[cfg(test)]
#[path = "../tests/unit/stratification_tests.rs"]
mod tests;
