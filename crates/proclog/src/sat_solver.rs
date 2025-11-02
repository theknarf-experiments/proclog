//! SAT solver integration for ASP evaluation
//!
//! This module provides a wrapper around splr (pure Rust CDCL SAT solver)
//! for use in ASP-to-SAT translation and stable model computation.
//!
//! Note: splr requires all variables to be declared upfront in the initial clauses.
//! Therefore, this solver builds up clauses first, then creates the splr solver.

use crate::ast::*;
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
    pub fn get_or_create_var(&mut self, atom: &Atom) -> i32 {
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

    /// Add a constraint that an atom must be false (for Closed World Assumption)
    pub fn add_must_be_false(&mut self, atom: &Atom) {
        let var = self.get_or_create_var(atom);
        self.clauses.push(vec![-var]);
    }

    /// Add a tautology clause for a choice atom to register it with the SAT solver
    /// This adds "atom OR NOT atom" which is always true but ensures the variable appears in the model
    pub fn add_tautology_for_choice(&mut self, atom: &Atom) {
        let var = self.get_or_create_var(atom);
        // Add clause: var OR NOT var (always true, but registers the variable)
        self.clauses.push(vec![var, -var]);
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

    /// Add a rule with negation and Clark completion
    /// For rule "head :- pos1, pos2, not neg1, not neg2", add:
    /// 1. head OR NOT pos1 OR NOT pos2 OR neg1 OR neg2
    ///    (if body is satisfied, head must be true)
    /// 2. NOT head OR pos1, NOT head OR pos2, NOT head OR NOT neg1, NOT head OR NOT neg2
    ///    (if head is true, body must be satisfied - Clark completion)
    pub fn add_rule_with_negation(
        &mut self,
        head: &Atom,
        positive_body: &[Atom],
        negative_body: &[Atom],
    ) {
        let head_var = self.get_or_create_var(head);

        if positive_body.is_empty() && negative_body.is_empty() {
            // Fact: head is true
            self.clauses.push(vec![head_var]);
            return;
        }

        // Clause 1: head OR NOT pos1 OR NOT pos2 OR ... OR neg1 OR neg2 OR ...
        // This says: if all positive literals are true AND all negative literals are false, then head must be true
        let mut implication_clause = vec![head_var];

        // Add negative literals for positive body atoms
        for pos_atom in positive_body {
            let pos_var = self.get_or_create_var(pos_atom);
            implication_clause.push(-pos_var);
        }

        // Add positive literals for negative body atoms (double negation)
        for neg_atom in negative_body {
            let neg_var = self.get_or_create_var(neg_atom);
            implication_clause.push(neg_var);
        }

        self.clauses.push(implication_clause);

        // Clause 2: Clark completion - if head is true, body must be satisfied
        // For positive body literals: NOT head OR pos_atom
        for pos_atom in positive_body {
            let pos_var = self.get_or_create_var(pos_atom);
            self.clauses.push(vec![-head_var, pos_var]);
        }

        // For negative body literals: NOT head OR NOT neg_atom
        for neg_atom in negative_body {
            let neg_var = self.get_or_create_var(neg_atom);
            self.clauses.push(vec![-head_var, -neg_var]);
        }
    }

    /// Add cardinality constraint: at most k of the given atoms can be true
    /// Uses sequential counter encoding for efficiency
    pub fn add_at_most_k(&mut self, atoms: &[Atom], k: usize) {
        if k >= atoms.len() {
            return; // Constraint is trivially satisfied
        }

        let vars: Vec<i32> = atoms.iter().map(|a| self.get_or_create_var(a)).collect();

        // For small k, use pairwise encoding
        if k == 0 {
            // None of the atoms can be true
            for &var in &vars {
                self.clauses.push(vec![-var]);
            }
        } else if k == 1 {
            // At most 1: for each pair, at least one must be false
            for i in 0..vars.len() {
                for j in (i + 1)..vars.len() {
                    self.clauses.push(vec![-vars[i], -vars[j]]);
                }
            }
        } else {
            // For larger k, use sequential counter encoding
            // This is more efficient for general k
            self.add_sequential_counter(&vars, k);
        }
    }

    /// Add cardinality constraint: at least k of the given atoms must be true
    pub fn add_at_least_k(&mut self, atoms: &[Atom], k: usize) {
        if k == 0 || k > atoms.len() {
            return; // Constraint is trivially satisfied or unsatisfiable
        }

        let vars: Vec<i32> = atoms.iter().map(|a| self.get_or_create_var(a)).collect();

        // For small k, use simple encodings
        if k == 1 {
            // At least 1: simple OR clause
            self.clauses.push(vars.clone());
        } else if k == atoms.len() {
            // All must be true
            for &var in &vars {
                self.clauses.push(vec![var]);
            }
        } else if k == atoms.len() - 1 {
            // At least n-1: for each pair, at least one must be true
            for i in 0..vars.len() {
                for j in (i + 1)..vars.len() {
                    self.clauses.push(vec![vars[i], vars[j]]);
                }
            }
        } else {
            // Use sequential counter for negation: at-most (n-k) are false
            let neg_vars: Vec<i32> = vars.iter().map(|&v| -v).collect();
            self.add_sequential_counter(&neg_vars, atoms.len() - k);
        }
    }

    /// Sequential counter encoding for at-most-k constraint
    /// This introduces auxiliary variables to count true literals
    fn add_sequential_counter(&mut self, vars: &[i32], k: usize) {
        let n = vars.len();
        if n <= k {
            return;
        }

        // Create auxiliary variables s[i][j] meaning "at least j+1 of first i vars are true"
        let mut aux_vars: Vec<Vec<i32>> = Vec::new();

        for _i in 0..n {
            let mut row = Vec::new();
            for _j in 0..k {
                let aux_var = self.next_var;
                self.next_var += 1;
                row.push(aux_var);
            }
            aux_vars.push(row);
        }

        // Add clauses for the encoding
        for i in 0..n {
            for j in 0..k {
                // If var[i] is true, then s[i][0] must be true
                if j == 0 {
                    self.clauses.push(vec![-vars[i], aux_vars[i][j]]);
                }

                // If s[i-1][j] and var[i] are true, then s[i][j+1] must be true
                if i > 0 && j < k - 1 {
                    self.clauses.push(vec![-aux_vars[i - 1][j], -vars[i], aux_vars[i][j + 1]]);
                }

                // If s[i-1][j] is true, then s[i][j] must be true
                if i > 0 {
                    self.clauses.push(vec![-aux_vars[i - 1][j], aux_vars[i][j]]);
                }

                // Forbid s[i][k] (can't have more than k)
                if j == k - 1 && i == n - 1 {
                    self.clauses.push(vec![-aux_vars[i][j]]);
                }
            }
        }
    }

    /// Block a specific model by adding a clause that excludes it
    /// This is used for incremental solving to find all models
    pub fn block_model(&mut self, model: &HashMap<Atom, bool>) {
        let blocking_clause: Vec<i32> = model
            .iter()
            .filter_map(|(atom, &value)| {
                self.atom_to_var.get(atom).map(|&var| {
                    if value {
                        -var // If atom was true in model, add NOT atom to blocking clause
                    } else {
                        var // If atom was false in model, add atom to blocking clause
                    }
                })
            })
            .collect();

        if !blocking_clause.is_empty() {
            self.clauses.push(blocking_clause);
        }
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
    use internment::Intern;

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
