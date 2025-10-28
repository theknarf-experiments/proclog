use crate::ast::{Rule, Literal, Symbol, Atom};
use std::collections::{HashMap, HashSet};
use internment::Intern;

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

/// Dependency between predicates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DependencyType {
    Positive,  // p depends positively on q
    Negative,  // p depends negatively on q (through negation)
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
        self.predicates.insert(from.clone());
        self.predicates.insert(to.clone());

        self.dependencies
            .entry(from)
            .or_insert_with(Vec::new)
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
        let head_pred = rule.head.predicate.clone();

        for literal in &rule.body {
            match literal {
                Literal::Positive(atom) => {
                    graph.add_dependency(head_pred.clone(), atom.predicate.clone(), DependencyType::Positive);
                }
                Literal::Negative(atom) => {
                    graph.add_dependency(head_pred.clone(), atom.predicate.clone(), DependencyType::Negative);
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

    visited.insert(from.clone());

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
            return Some(vec![pred.clone()]);
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
        strata.insert(pred.clone(), 0);
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
                strata.insert(pred.clone(), max_stratum);
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
mod tests {
    use super::*;
    use crate::ast::{Term, Value};

    // Helper functions
    fn atom_const(name: &str) -> Term {
        Term::Constant(Value::Atom(Intern::new(name.to_string())))
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

    fn pred(name: &str) -> Symbol {
        Intern::new(name.to_string())
    }

    // Basic stratification tests
    #[test]
    fn test_stratify_empty_program() {
        let rules = vec![];
        let result = stratify(&rules).unwrap();

        assert_eq!(result.num_strata, 0);
        assert_eq!(result.rules_by_stratum.len(), 0);
    }

    #[test]
    fn test_stratify_single_rule_no_negation() {
        // p(X) :- q(X).
        let rules = vec![make_rule(
            make_atom("p", vec![var("X")]),
            vec![Literal::Positive(make_atom("q", vec![var("X")]))],
        )];

        let result = stratify(&rules).unwrap();

        assert_eq!(result.num_strata, 1);
        assert_eq!(result.predicate_strata.get(&pred("p")), Some(&0));
        assert_eq!(result.predicate_strata.get(&pred("q")), Some(&0));
    }

    #[test]
    fn test_stratify_negation_simple() {
        // p(X) :- q(X), not r(X).
        let rules = vec![make_rule(
            make_atom("p", vec![var("X")]),
            vec![
                Literal::Positive(make_atom("q", vec![var("X")])),
                Literal::Negative(make_atom("r", vec![var("X")])),
            ],
        )];

        let result = stratify(&rules).unwrap();

        // q and r should be in stratum 0, p in stratum 1 (after r)
        assert_eq!(result.predicate_strata.get(&pred("q")), Some(&0));
        assert_eq!(result.predicate_strata.get(&pred("r")), Some(&0));
        assert_eq!(result.predicate_strata.get(&pred("p")), Some(&1));
        assert_eq!(result.num_strata, 2);
    }

    #[test]
    fn test_stratify_chain_of_negations() {
        // p(X) :- not q(X).
        // q(X) :- not r(X).
        // r(X) :- s(X).
        let rules = vec![
            make_rule(
                make_atom("p", vec![var("X")]),
                vec![Literal::Negative(make_atom("q", vec![var("X")]))],
            ),
            make_rule(
                make_atom("q", vec![var("X")]),
                vec![Literal::Negative(make_atom("r", vec![var("X")]))],
            ),
            make_rule(
                make_atom("r", vec![var("X")]),
                vec![Literal::Positive(make_atom("s", vec![var("X")]))],
            ),
        ];

        let result = stratify(&rules).unwrap();

        // s and r should be in stratum 0
        // q should be in stratum 1 (after r due to negation)
        // p should be in stratum 2 (after q due to negation)
        assert_eq!(result.predicate_strata.get(&pred("s")), Some(&0));
        assert_eq!(result.predicate_strata.get(&pred("r")), Some(&0));
        assert_eq!(result.predicate_strata.get(&pred("q")), Some(&1));
        assert_eq!(result.predicate_strata.get(&pred("p")), Some(&2));
        assert_eq!(result.num_strata, 3);
    }

    #[test]
    fn test_stratify_multiple_rules_same_predicate() {
        // p(X) :- q(X).
        // p(X) :- not r(X).
        let rules = vec![
            make_rule(
                make_atom("p", vec![var("X")]),
                vec![Literal::Positive(make_atom("q", vec![var("X")]))],
            ),
            make_rule(
                make_atom("p", vec![var("X")]),
                vec![Literal::Negative(make_atom("r", vec![var("X")]))],
            ),
        ];

        let result = stratify(&rules).unwrap();

        // q and r in stratum 0, p in stratum 1
        assert_eq!(result.predicate_strata.get(&pred("q")), Some(&0));
        assert_eq!(result.predicate_strata.get(&pred("r")), Some(&0));
        assert_eq!(result.predicate_strata.get(&pred("p")), Some(&1));
    }

    // Cycle detection tests
    #[test]
    fn test_detect_direct_negative_cycle() {
        // p(X) :- not p(X).  [Illegal!]
        let rules = vec![make_rule(
            make_atom("p", vec![var("X")]),
            vec![Literal::Negative(make_atom("p", vec![var("X")]))],
        )];

        let result = stratify(&rules);
        assert!(result.is_err());

        if let Err(StratificationError::CycleThroughNegation(cycle)) = result {
            assert_eq!(cycle, vec![pred("p")]);
        }
    }

    #[test]
    fn test_detect_indirect_negative_cycle() {
        // p(X) :- not q(X).
        // q(X) :- p(X).  [Creates cycle through negation!]
        let rules = vec![
            make_rule(
                make_atom("p", vec![var("X")]),
                vec![Literal::Negative(make_atom("q", vec![var("X")]))],
            ),
            make_rule(
                make_atom("q", vec![var("X")]),
                vec![Literal::Positive(make_atom("p", vec![var("X")]))],
            ),
        ];

        let result = stratify(&rules);
        assert!(result.is_err());
    }

    #[test]
    fn test_positive_cycle_is_ok() {
        // p(X) :- p(X).  [Recursive, but no negation - OK!]
        let rules = vec![make_rule(
            make_atom("p", vec![var("X")]),
            vec![Literal::Positive(make_atom("p", vec![var("X")]))],
        )];

        let result = stratify(&rules);
        assert!(result.is_ok());

        // Should all be in stratum 0
        assert_eq!(result.unwrap().num_strata, 1);
    }

    #[test]
    fn test_transitive_closure_is_ok() {
        // path(X, Y) :- edge(X, Y).
        // path(X, Z) :- path(X, Y), edge(Y, Z).
        let rules = vec![
            make_rule(
                make_atom("path", vec![var("X"), var("Y")]),
                vec![Literal::Positive(make_atom("edge", vec![var("X"), var("Y")]))],
            ),
            make_rule(
                make_atom("path", vec![var("X"), var("Z")]),
                vec![
                    Literal::Positive(make_atom("path", vec![var("X"), var("Y")])),
                    Literal::Positive(make_atom("edge", vec![var("Y"), var("Z")])),
                ],
            ),
        ];

        let result = stratify(&rules).unwrap();

        // All in same stratum (positive recursion is fine)
        assert_eq!(result.num_strata, 1);
    }

    #[test]
    fn test_rules_organized_by_stratum() {
        // r0(X) :- base(X).
        // r1(X) :- not r0(X).
        // r2(X) :- not r1(X).
        let rules = vec![
            make_rule(
                make_atom("r0", vec![var("X")]),
                vec![Literal::Positive(make_atom("base", vec![var("X")]))],
            ),
            make_rule(
                make_atom("r1", vec![var("X")]),
                vec![Literal::Negative(make_atom("r0", vec![var("X")]))],
            ),
            make_rule(
                make_atom("r2", vec![var("X")]),
                vec![Literal::Negative(make_atom("r1", vec![var("X")]))],
            ),
        ];

        let result = stratify(&rules).unwrap();

        assert_eq!(result.num_strata, 3);
        assert_eq!(result.rules_by_stratum[0].len(), 1); // r0 rule
        assert_eq!(result.rules_by_stratum[1].len(), 1); // r1 rule
        assert_eq!(result.rules_by_stratum[2].len(), 1); // r2 rule
    }
}
