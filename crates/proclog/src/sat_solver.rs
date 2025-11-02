//! SAT solver integration for ASP evaluation
//!
//! This module provides a wrapper around splr (pure Rust CDCL SAT solver)
//! for use in ASP-to-SAT translation and stable model computation.
//!
//! Note: splr requires all variables to be declared upfront in the initial clauses.
//! Therefore, this solver builds up clauses first, then creates the splr solver.

use crate::ast::*;
use internment::Intern;
use splr::*;
use std::collections::HashMap;
use std::convert::TryFrom;

/// Wrapper around splr SAT solver for ASP evaluation
pub struct AspSatSolver {
    /// Maps ASP atoms to SAT variables (positive integers)
    atom_to_var: HashMap<Atom, i32>,
    /// Maps SAT variables back to ASP atoms
    var_to_atom: HashMap<i32, Atom>,
    /// Clauses to be added to the solver
    clauses: Vec<Vec<i32>>,
    /// Next available variable ID
    next_var: i32,
}

impl AspSatSolver {
    /// Create a new ASP SAT solver (collecting clauses)
    pub fn new() -> Self {
        Self {
            atom_to_var: HashMap::new(),
            var_to_atom: HashMap::new(),
            clauses: Vec::new(),
            next_var: 1, // SAT variables start at 1
        }
    }

    /// Get or create a SAT variable for an ASP atom
    fn get_or_create_var(&mut self, atom: &Atom) -> i32 {
        if let Some(&var) = self.atom_to_var.get(atom) {
            return var;
        }

        let var = self.next_var;
        self.next_var += 1;

        self.atom_to_var.insert(atom.clone(), var);
        self.var_to_atom.insert(var, atom.clone());
        var
    }

    /// Add a fact (unit clause: atom must be true)
    pub fn add_fact(&mut self, atom: &Atom) {
        let var = self.get_or_create_var(atom);
        self.clauses.push(vec![var]);
    }

    /// Add a constraint (body implies false, i.e., NOT body)
    /// For constraint ":- a, b, c", add clause "NOT a OR NOT b OR NOT c"
    pub fn add_constraint(&mut self, atoms: &[Atom]) {
        let clause: Vec<i32> = atoms
            .iter()
            .map(|a| -self.get_or_create_var(a))
            .collect();
        self.clauses.push(clause);
    }

    /// Add an implication: head -> body (if head then body must be true)
    /// Translated to: NOT head OR body
    pub fn add_implication(&mut self, head: &Atom, body: &Atom) {
        let head_var = self.get_or_create_var(head);
        let body_var = self.get_or_create_var(body);
        // NOT head OR body
        self.clauses.push(vec![-head_var, body_var]);
    }

    /// Add a rule (completion formula)
    /// For rule "head :- body1, body2", add clauses for Clark completion:
    /// - head -> (body1 AND body2)    =>  NOT head OR (body1 AND body2)
    /// - (body1 AND body2) -> head    =>  head OR NOT body1 OR NOT body2
    pub fn add_rule(&mut self, head: &Atom, body: &[Atom]) {
        let head_var = self.get_or_create_var(head);

        if body.is_empty() {
            // Fact: head is true
            self.clauses.push(vec![head_var]);
            return;
        }

        // head OR NOT body1 OR NOT body2 OR ...
        // (this is: (body1 AND body2 AND ...) -> head)
        let mut clause = vec![head_var];
        for body_atom in body {
            let body_var = self.get_or_create_var(body_atom);
            clause.push(-body_var);
        }
        self.clauses.push(clause);

        // For completion, we'd also need: head -> (body1 AND body2)
        // Which is: NOT head OR body1, NOT head OR body2, ...
        // But this is only needed for full Clark completion
        // For now, implementing basic rule translation
    }

    /// Solve and return a model if SAT
    /// This creates the splr solver with all collected clauses and solves
    pub fn solve(&self) -> Result<Option<HashMap<Atom, bool>>, SolverError> {
        // Create solver with all clauses
        let config = Config::default();
        let mut solver = match Solver::try_from((config, self.clauses.as_ref())) {
            Ok(s) => s,
            Err(Ok(Certificate::UNSAT)) => {
                // splr detected UNSAT during preprocessing
                return Ok(None);
            }
            Err(Ok(Certificate::SAT(_))) => {
                // This shouldn't happen - SAT is returned by solve(), not TryFrom
                // But handle it anyway to satisfy exhaustiveness
                return Err(SolverError::Inconsistent);
            }
            Err(Err(SolverError::EmptyClause)) => {
                // Empty clause detected - this means UNSAT
                return Ok(None);
            }
            Err(Err(e)) => {
                // Other actual errors
                return Err(e);
            }
        };

        match solver.solve() {
            Ok(Certificate::SAT(model)) => {
                let mut result = HashMap::new();
                for (&var, atom) in &self.var_to_atom {
                    // VarIds start from 1, but Vec is 0-indexed
                    // So we need to subtract 1 to get the correct index
                    let index = (var - 1) as usize;
                    if let Some(&lit) = model.get(index) {
                        result.insert(atom.clone(), lit > 0);
                    }
                }
                Ok(Some(result))
            }
            Ok(Certificate::UNSAT) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Get the number of variables
    pub fn num_vars(&self) -> usize {
        self.atom_to_var.len()
    }

    /// Get the number of clauses
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }
}

impl Default for AspSatSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_atom(pred: &str, args: Vec<&str>) -> Atom {
        Atom {
            predicate: Intern::new(pred.to_string()),
            terms: args
                .into_iter()
                .map(|a| Term::Constant(Value::Atom(Intern::new(a.to_string()))))
                .collect(),
        }
    }

    #[test]
    fn test_sat_solver_basic() {
        let mut solver = AspSatSolver::new();

        // Add fact: a
        let a = make_atom("a", vec![]);
        solver.add_fact(&a);

        // Solve
        let model = solver.solve().unwrap().expect("Should be SAT");

        // Check that a is true
        assert_eq!(model.get(&a), Some(&true));
    }

    #[test]
    fn test_sat_solver_constraint() {
        let mut solver = AspSatSolver::new();

        // Add facts: a, b
        let a = make_atom("a", vec![]);
        let b = make_atom("b", vec![]);
        solver.add_fact(&a);
        solver.add_fact(&b);

        // Add constraint: :- a, b (cannot have both a and b)
        solver.add_constraint(&[a.clone(), b.clone()]);

        // Solve - should be UNSAT
        let result = solver.solve().unwrap();
        assert!(result.is_none(), "Should be UNSAT due to constraint");
    }

    #[test]
    fn test_sat_solver_rule() {
        let mut solver = AspSatSolver::new();

        // Rule: c :- a, b
        let a = make_atom("a", vec![]);
        let b = make_atom("b", vec![]);
        let c = make_atom("c", vec![]);

        solver.add_fact(&a);
        solver.add_fact(&b);
        solver.add_rule(&c, &[a.clone(), b.clone()]);

        // Solve
        let model = solver.solve().unwrap().expect("Should be SAT");

        // All should be true
        assert_eq!(model.get(&a), Some(&true));
        assert_eq!(model.get(&b), Some(&true));
        assert_eq!(model.get(&c), Some(&true));
    }
}
