//! ASP evaluation using SAT solver backend
//!
//! This module implements Answer Set Programming using SAT-based techniques:
//! 1. Clark completion for positive dependencies
//! 2. Loop formulas for circular dependencies
//! 3. Guess-and-check for choice rules and negation
//!
//! The implementation follows the approach described in:
//! "Answer Set Programming via SAT" by Fangzhen Lin and Yuting Zhao

use crate::ast::*;
use crate::constants::ConstantEnv;
use crate::database::FactDatabase;
use crate::evaluation::stratified_evaluation_with_constraints;
use crate::grounding;
use crate::sat_solver::AspSatSolver;
use internment::Intern;
use std::collections::HashSet;

/// Check if a rule is ground (contains no variables)
fn rule_is_ground(rule: &Rule) -> bool {
    fn term_is_ground(term: &Term) -> bool {
        match term {
            Term::Variable(_) => false,
            Term::Constant(_) => true,
            Term::Compound(_, terms) => terms.iter().all(term_is_ground),
            Term::Range(start, end) => term_is_ground(start) && term_is_ground(end),
        }
    }

    fn atom_is_ground(atom: &Atom) -> bool {
        atom.terms.iter().all(term_is_ground)
    }

    fn literal_is_ground(lit: &Literal) -> bool {
        match lit {
            Literal::Positive(atom) | Literal::Negative(atom) => atom_is_ground(atom),
            Literal::Aggregate(_) => true, // Aggregates don't affect groundness for our purposes
        }
    }

    atom_is_ground(&rule.head) && rule.body.iter().all(literal_is_ground)
}

/// Check if a rule contains any anonymous variables (_)
fn rule_has_anonymous_variables(rule: &Rule) -> bool {
    fn term_has_anon(term: &Term) -> bool {
        match term {
            Term::Variable(var) if var.as_ref() == "_" => true,
            Term::Compound(_, args) => args.iter().any(term_has_anon),
            _ => false,
        }
    }

    fn atom_has_anon(atom: &Atom) -> bool {
        atom.terms.iter().any(term_has_anon)
    }

    fn literal_has_anon(lit: &Literal) -> bool {
        match lit {
            Literal::Positive(atom) | Literal::Negative(atom) => atom_has_anon(atom),
            Literal::Aggregate(_) => false,
        }
    }

    atom_has_anon(&rule.head) || rule.body.iter().any(literal_has_anon)
}

/// Replace anonymous variables (_) with unique named variables so they get bound during grounding
fn replace_anonymous_variables(rule: &Rule, counter: &mut usize) -> Rule {
    fn replace_term(term: &Term, counter: &mut usize) -> Term {
        match term {
            Term::Variable(var) if var.as_ref() == "_" => {
                *counter += 1;
                Term::Variable(Intern::new(format!("_anon{}", counter)))
            }
            Term::Compound(f, args) => {
                Term::Compound(f.clone(), args.iter().map(|t| replace_term(t, counter)).collect())
            }
            _ => term.clone(),
        }
    }

    fn replace_atom(atom: &Atom, counter: &mut usize) -> Atom {
        Atom {
            predicate: atom.predicate.clone(),
            terms: atom.terms.iter().map(|t| replace_term(t, counter)).collect(),
        }
    }

    fn replace_literal(lit: &Literal, counter: &mut usize) -> Literal {
        match lit {
            Literal::Positive(atom) => Literal::Positive(replace_atom(atom, counter)),
            Literal::Negative(atom) => Literal::Negative(replace_atom(atom, counter)),
            Literal::Aggregate(agg) => Literal::Aggregate(agg.clone()), // Don't modify aggregates for now
        }
    }

    Rule {
        head: replace_atom(&rule.head, counter),
        body: rule.body.iter().map(|lit| replace_literal(lit, counter)).collect(),
    }
}

/// An answer set is a stable model - a set of ground atoms
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnswerSet {
    pub atoms: HashSet<Atom>,
}

/// Evaluate an ASP program using SAT solver backend
/// This function assumes the program is already ground (no variables).
/// For non-ground programs, use `asp_sat_evaluation_with_grounding`.
pub fn asp_sat_evaluation(program: &Program) -> Vec<AnswerSet> {
    // Step 1: Check if program has choice rules or negation
    let has_choices = has_choice_rules(program);
    let has_negation = has_negation_in_rules(program);

    // Get answer sets (before optimization)
    let answer_sets = if !has_choices && !has_negation {
        // For programs without choices or negation, use direct SAT translation
        evaluate_definite_program(program)
    } else {
        // For programs with choices/negation, use proper SAT encoding
        evaluate_with_sat_encoding(program)
    };

    // Apply optimization if present
    let const_env = ConstantEnv::from_program(program);

    // Convert to asp::AnswerSet for optimization
    let asp_answer_sets: Vec<crate::asp::AnswerSet> = answer_sets
        .into_iter()
        .map(|as_set| crate::asp::AnswerSet {
            atoms: as_set.atoms,
        })
        .collect();

    // Apply optimization
    let optimized = crate::asp::apply_optimization(program, asp_answer_sets, &const_env);

    // Convert back to asp_sat::AnswerSet
    optimized
        .into_iter()
        .map(|as_set| AnswerSet {
            atoms: as_set.atoms,
        })
        .collect()
}

/// Evaluate an ASP program with variables using grounding + SAT solver
///
/// This is the main entry point for ASP-SAT evaluation with non-ground programs.
/// It follows this process:
/// 1. Extract facts and build initial database
/// 2. Evaluate Datalog rules to derive facts (needed for grounding choice conditions)
/// 3. Ground all choice rules and regular rules
/// 4. Evaluate the ground program using SAT solver
pub fn asp_sat_evaluation_with_grounding(program: &Program) -> Vec<AnswerSet> {
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
                let expanded = grounding::expand_atom_ranges(&fact.atom, &const_env);
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
                // Skip - handled elsewhere or not relevant for grounding
            }
        }
    }

    // Separate rules into:
    // 1. Definite rules (no negation) - can be pre-evaluated to derive facts
    // 2. Non-definite rules (with negation) - must be kept as rules for guess-and-check
    let (definite_rules, non_definite_rules): (Vec<Rule>, Vec<Rule>) = rules
        .into_iter()
        .partition(|rule| {
            // A rule is definite if it has no negation in its body
            !rule.body.iter().any(|lit| matches!(lit, Literal::Negative(_)))
        });

    // Evaluate ONLY definite rules to derive facts
    // These derived facts are needed for grounding choice rule conditions
    let derived_db = if definite_rules.is_empty() {
        base_facts.clone()
    } else {
        // Evaluate without constraints first (constraints checked later in SAT)
        match stratified_evaluation_with_constraints(&definite_rules, &[], base_facts.clone()) {
            Ok(db) => db,
            Err(_) => return vec![], // Rules themselves inconsistent
        }
    };

    // Now ground everything using the derived database
    let mut ground_program = Program::new();

    // Add all derived facts to the ground program
    for fact in derived_db.all_facts() {
        ground_program.add_statement(Statement::Fact(Fact {
            atom: fact.clone(),
        }));
    }

    // Preserve optimization statements (they don't need grounding)
    for statement in &program.statements {
        if let Statement::Optimize(opt) = statement {
            ground_program.add_statement(Statement::Optimize(opt.clone()));
        }
    }

    // Ground all choice rules and collect choice atoms
    let mut all_choice_atoms = Vec::new();
    for choice in &choice_rules {
        let grounded_atoms = grounding::ground_choice_rule(choice, &derived_db, &const_env);

        // Create a single choice rule with all grounded atoms, preserving the bounds
        if !grounded_atoms.is_empty() {
            ground_program.add_statement(Statement::ChoiceRule(ChoiceRule {
                lower_bound: choice.lower_bound.clone(),
                upper_bound: choice.upper_bound.clone(),
                elements: grounded_atoms
                    .iter()
                    .map(|atom| ChoiceElement {
                        atom: atom.clone(),
                        condition: vec![],
                    })
                    .collect(),
                body: vec![],
            }));
        }
        all_choice_atoms.extend(grounded_atoms);
    }

    // Create an extended database that includes choice atoms as potential facts
    // This allows rules that reference choice atoms to be grounded
    let mut extended_db = derived_db.clone();
    for atom in &all_choice_atoms {
        // Ignore errors - atom might already exist
        let _ = extended_db.insert(atom.clone());
    }

    // Ground all DEFINITE rules using the extended database
    // These rules have no negation, so they can be evaluated straightforwardly by the SAT solver
    for rule in &definite_rules {
        // Check if the rule is propositional (has no variables)
        let is_propositional = is_rule_propositional(rule);

        if is_propositional {
            // For propositional rules, just add them as-is without checking the database
            ground_program.add_statement(Statement::Rule(rule.clone()));
        } else {
            // For rules with variables, find all substitutions from the database
            let body_substs = grounding::satisfy_body(&rule.body, &extended_db);

            for subst in body_substs {
                let ground_head = subst.apply_atom(&rule.head);
                let ground_body: Vec<Literal> = rule
                    .body
                    .iter()
                    .map(|lit| match lit {
                        Literal::Positive(atom) => Literal::Positive(subst.apply_atom(atom)),
                        Literal::Negative(atom) => Literal::Negative(subst.apply_atom(atom)),
                        Literal::Aggregate(_) => lit.clone(),
                    })
                    .collect();

                ground_program.add_statement(Statement::Rule(Rule {
                    head: ground_head,
                    body: ground_body,
                }));
            }
        }
    }

    // Ground all NON-DEFINITE rules using the extended database
    // IMPORTANT: We need to ground rules WITHOUT filtering on negation of choice atoms.
    // The negation semantics will be handled during guess-and-check.
    for rule in &non_definite_rules {
        // Check if the rule is propositional (has no variables)
        let is_propositional = is_rule_propositional(rule);

        if is_propositional {
            // For propositional rules, just add them as-is without checking the database
            ground_program.add_statement(Statement::Rule(rule.clone()));
        } else {
            // Find all substitutions that satisfy ONLY the positive literals in the body
            // We extract positive literals for finding substitutions
            let positive_lits: Vec<Literal> = rule
                .body
                .iter()
                .filter(|lit| !matches!(lit, Literal::Negative(_)))
                .cloned()
                .collect();

            let body_substs = grounding::satisfy_body(&positive_lits, &extended_db);

            // Create a ground rule for each substitution
            for subst in body_substs {
                let ground_head = subst.apply_atom(&rule.head);
                let ground_body: Vec<Literal> = rule
                    .body
                    .iter()
                    .map(|lit| match lit {
                        Literal::Positive(atom) => Literal::Positive(subst.apply_atom(atom)),
                        Literal::Negative(atom) => Literal::Negative(subst.apply_atom(atom)),
                        Literal::Aggregate(_) => lit.clone(), // TODO: Handle aggregates
                    })
                    .collect();

                ground_program.add_statement(Statement::Rule(Rule {
                    head: ground_head,
                    body: ground_body,
                }));
            }
        }
    }

    // Ground all constraints using the extended database
    // Skip constraints with aggregates - they are handled separately after SAT evaluation
    for constraint in &constraints {
        // Check if constraint contains aggregates
        let has_aggregates = constraint
            .body
            .iter()
            .any(|lit| matches!(lit, Literal::Aggregate(_)));

        if has_aggregates {
            // Aggregate constraints are handled separately after SAT evaluation
            continue;
        }

        // Ground non-aggregate constraint body
        let body_substs = grounding::satisfy_body(&constraint.body, &extended_db);
        for subst in body_substs {
            let ground_body: Vec<Literal> = constraint
                .body
                .iter()
                .map(|lit| match lit {
                    Literal::Positive(atom) => Literal::Positive(subst.apply_atom(atom)),
                    Literal::Negative(atom) => Literal::Negative(subst.apply_atom(atom)),
                    Literal::Aggregate(_) => unreachable!("Should have been filtered out"),
                })
                .collect();

            ground_program.add_statement(Statement::Constraint(Constraint {
                body: ground_body,
            }));
        }
    }

    // Evaluate the ground program using SAT solver
    let mut answer_sets = asp_sat_evaluation(&ground_program);

    // Filter answer sets based on aggregate constraints
    filter_by_aggregate_constraints(&mut answer_sets, &constraints, &extended_db);

    answer_sets
}

/// Filter answer sets by checking aggregate constraints
fn filter_by_aggregate_constraints(
    answer_sets: &mut Vec<AnswerSet>,
    constraints: &[Constraint],
    db: &FactDatabase,
) {
    answer_sets.retain(|answer_set| {
        // Check if this answer set satisfies all constraints with aggregates
        for constraint in constraints {
            // Check if constraint has aggregates
            let has_aggregates = constraint
                .body
                .iter()
                .any(|lit| matches!(lit, Literal::Aggregate(_)));

            if !has_aggregates {
                continue; // Non-aggregate constraints already handled by SAT
            }

            // Evaluate the constraint for this answer set
            if violates_aggregate_constraint(constraint, answer_set, db) {
                return false; // Filter out this answer set
            }
        }
        true // Keep this answer set
    });
}

/// Check if an answer set violates an aggregate constraint
fn violates_aggregate_constraint(
    constraint: &Constraint,
    answer_set: &AnswerSet,
    _db: &FactDatabase,
) -> bool {
    use crate::unification::Substitution;

    // Separate aggregate and non-aggregate literals
    let agg_literals: Vec<&AggregateAtom> = constraint
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

    let non_agg_literals: Vec<Literal> = constraint
        .body
        .iter()
        .filter(|lit| !matches!(lit, Literal::Aggregate(_)))
        .cloned()
        .collect();

    // Create a database from ONLY the answer set atoms
    // Don't use the extended database as it contains all possible choice atoms
    let mut answer_db = FactDatabase::new();
    for atom in &answer_set.atoms {
        let _ = answer_db.insert(atom.clone());
    }

    // Get all substitutions that satisfy non-aggregate literals
    let violations = if non_agg_literals.is_empty() {
        vec![Substitution::new()]
    } else {
        grounding::satisfy_body(&non_agg_literals, &answer_db)
    };

    if violations.is_empty() {
        // No violations from non-aggregate part
        return false;
    }

    // Check if any aggregate is satisfied with these substitutions
    let mut actual_violations = Vec::new();
    for subst in &violations {
        // Check if all aggregates are satisfied with this substitution
        let all_aggregates_satisfied = agg_literals.iter().all(|agg| {
            // Get threshold value
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

            // Count substitutions that satisfy the aggregate condition
            let substitutions = grounding::satisfy_body(&agg.elements, &answer_db);
            let count = substitutions.len() as i64;

            // Compare count against threshold
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

    !final_violations.is_empty()
}

/// Evaluate program with choice rules and/or negation using proper SAT encoding
/// This encodes choice rules directly in SAT instead of enumerating all combinations
fn evaluate_with_sat_encoding(program: &Program) -> Vec<AnswerSet> {
    let mut result = Vec::new();
    let mut solver = AspSatSolver::new();

    // Get constant environment for resolving bounds
    let const_env = ConstantEnv::from_program(program);

    // Step 1: Build initial fact database (needed for grounding choice rules)
    let mut base_facts = FactDatabase::new();
    let mut rules = Vec::new();

    for statement in &program.statements {
        match statement {
            Statement::Fact(fact) => {
                // Expand ranges in facts
                let expanded = crate::grounding::expand_atom_ranges(&fact.atom, &const_env);
                for atom in expanded {
                    base_facts.insert(atom).unwrap();
                }
            }
            Statement::Rule(rule) => {
                rules.push(rule.clone());
            }
            _ => {}
        }
    }

    // Evaluate definite rules to get derived facts (needed for choice conditions)
    // This is needed because choice element conditions may depend on derived facts
    let derived_db = if rules.is_empty() {
        base_facts.clone()
    } else {
        // Evaluate without constraints first (constraints checked during SAT solving)
        match crate::evaluation::stratified_evaluation_with_constraints(&rules, &[], base_facts.clone()) {
            Ok(db) => db,
            Err(_) => return vec![], // Rules themselves inconsistent
        }
    };

    // Step 1a: Ground choice rules properly using the grounding functions
    let mut choice_rules_data: Vec<(Vec<Atom>, usize, usize)> = Vec::new(); // (atoms, min, max) for each choice rule

    for statement in &program.statements {
        if let Statement::ChoiceRule(choice_rule) = statement {
            // Use ground_choice_rule_split to properly ground the choice rule
            // This handles choice element conditions and body substitutions
            let grounded_groups = crate::grounding::ground_choice_rule_split(
                choice_rule,
                &derived_db,
                &const_env,
            );

            // Each group is an independent choice (one per body substitution)
            for grounded_atoms in grounded_groups {
                if grounded_atoms.is_empty() {
                    continue;
                }

                let min = choice_rule
                    .lower_bound
                    .as_ref()
                    .and_then(|t| crate::asp::resolve_bound(t, &const_env))
                    .unwrap_or(0) as usize;

                let max = choice_rule
                    .upper_bound
                    .as_ref()
                    .and_then(|t| crate::asp::resolve_bound(t, &const_env))
                    .map(|u| u as usize)
                    .unwrap_or(grounded_atoms.len());

                choice_rules_data.push((grounded_atoms, min, max));
            }
        }
    }

    // Step 2: Add all facts to the SAT solver
    for statement in &program.statements {
        if let Statement::Fact(fact) = statement {
            solver.add_fact(&fact.atom);
        }
    }

    // Step 2a: Create extended database with facts + all choice atoms
    // This is needed to properly ground rules that reference choice atoms
    let mut extended_db = derived_db.clone();
    for (choice_atoms, _, _) in &choice_rules_data {
        for atom in choice_atoms {
            let _ = extended_db.insert(atom.clone());
        }
    }

    // Step 3: Ground and add all rules (with negation support)
    // Rules must be grounded against extended_db to include choice atoms
    let mut anon_var_counter = 0;
    for statement in &program.statements {
        if let Statement::Rule(rule) = statement {
            // Check if rule is already ground
            if rule_is_ground(rule) {
                // Already ground - add directly without grounding
                let positive_body: Vec<Atom> = rule
                    .body
                    .iter()
                    .filter_map(|lit| match lit {
                        Literal::Positive(atom) => Some(atom.clone()),
                        _ => None,
                    })
                    .collect();

                let negative_body: Vec<Atom> = rule
                    .body
                    .iter()
                    .filter_map(|lit| match lit {
                        Literal::Negative(atom) => Some(atom.clone()),
                        _ => None,
                    })
                    .collect();

                solver.add_rule_with_negation(&rule.head, &positive_body, &negative_body);
                continue;
            }

            // Replace anonymous variables with unique named variables so they get bound
            // Only do this if the rule actually has anonymous variables
            let has_anon_vars = rule_has_anonymous_variables(rule);
            let rule_with_named_vars = if has_anon_vars {
                replace_anonymous_variables(rule, &mut anon_var_counter)
            } else {
                rule.clone()
            };

            // Ground the rule against the extended database
            let body_substs = crate::grounding::satisfy_body(&rule_with_named_vars.body, &extended_db);

            for subst in body_substs {
                let ground_head = subst.apply_atom(&rule_with_named_vars.head);

                let mut positive_body = Vec::new();
                let mut negative_body = Vec::new();

                for lit in &rule_with_named_vars.body {
                    match lit {
                        Literal::Positive(atom) => {
                            positive_body.push(subst.apply_atom(atom));
                        }
                        Literal::Negative(atom) => {
                            negative_body.push(subst.apply_atom(atom));
                        }
                        Literal::Aggregate(_) => {
                            // TODO: Handle aggregates properly
                        }
                    }
                }

                solver.add_rule_with_negation(&ground_head, &positive_body, &negative_body);
            }
        }
    }

    // Step 4: Add all constraints
    for statement in &program.statements {
        if let Statement::Constraint(constraint) = statement {
            let positive_atoms: Vec<Atom> = constraint
                .body
                .iter()
                .filter_map(|lit| {
                    if let Literal::Positive(atom) = lit {
                        Some(atom.clone())
                    } else {
                        None
                    }
                })
                .collect();

            solver.add_constraint(&positive_atoms);
        }
    }

    // Step 5: Apply Closed World Assumption (CWA)
    // Collect all atoms that can be derived (have rules/facts defining them)
    let mut derivable_atoms: HashSet<Atom> = HashSet::new();

    // Add all fact atoms
    for statement in &program.statements {
        if let Statement::Fact(fact) = statement {
            derivable_atoms.insert(fact.atom.clone());
        }
    }

    // Add all rule head atoms
    for statement in &program.statements {
        if let Statement::Rule(rule) = statement {
            derivable_atoms.insert(rule.head.clone());
        }
    }

    // Add all choice atoms (they can be true or false)
    for (choice_atoms, _, _) in &choice_rules_data {
        for atom in choice_atoms {
            derivable_atoms.insert(atom.clone());
        }
    }

    // For atoms that appear in the program but are not derivable,
    // apply CWA: they must be false
    // We need to get all atoms that appear anywhere in the program
    let mut all_atoms: HashSet<Atom> = HashSet::new();
    for statement in &program.statements {
        match statement {
            Statement::Fact(fact) => {
                all_atoms.insert(fact.atom.clone());
            }
            Statement::Rule(rule) => {
                all_atoms.insert(rule.head.clone());
                for lit in &rule.body {
                    match lit {
                        Literal::Positive(atom) | Literal::Negative(atom) => {
                            all_atoms.insert(atom.clone());
                        }
                        _ => {}
                    }
                }
            }
            Statement::Constraint(constraint) => {
                for lit in &constraint.body {
                    match lit {
                        Literal::Positive(atom) | Literal::Negative(atom) => {
                            all_atoms.insert(atom.clone());
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    // Atoms that are not derivable must be false (CWA)
    for atom in &all_atoms {
        if !derivable_atoms.contains(atom) {
            solver.add_must_be_false(atom);
        }
    }

    // Step 6: Ensure all choice atoms are registered and add cardinality constraints
    for (choice_atoms, min, max) in &choice_rules_data {
        // First, ensure all choice atoms are registered as SAT variables
        // This is important because even without constraints, we need to track them for blocking
        for atom in choice_atoms {
            solver.get_or_create_var(atom);
        }

        // Add cardinality constraints if needed
        if *min > 0 {
            solver.add_at_least_k(choice_atoms, *min);
        }
        if *max < choice_atoms.len() {
            solver.add_at_most_k(choice_atoms, *max);
        }

        // Important: If no constraints were added for this choice rule,
        // we need to add tautology clauses to register variables with splr
        // For each atom, add "atom OR NOT atom" which is always true
        if *min == 0 && *max >= choice_atoms.len() {
            for atom in choice_atoms {
                solver.add_tautology_for_choice(atom);
            }
        }
    }


    // Step 7: Find all models using incremental SAT solving
    const MAX_MODELS: usize = 100000; // Safety limit

    loop {
        match solver.solve() {
            Ok(Some(model)) => {
                // eprintln!("Debug: SAT model has {} atoms ({} true)",
                //     model.len(),
                //     model.values().filter(|&&v| v).count());

                // Convert SAT model to answer set (only include true atoms)
                let atoms: HashSet<Atom> = model
                    .iter()
                    .filter_map(|(atom, &value)| if value { Some(atom.clone()) } else { None })
                    .collect();

                result.push(AnswerSet { atoms: atoms.clone() });

                if result.len() >= MAX_MODELS {
                    eprintln!("Warning: Reached maximum of {} answer sets, stopping enumeration", MAX_MODELS);
                    break;
                }

                // Block this model to find the next one
                // Need to block on ALL atoms (including false ones) to distinguish models
                solver.block_model(&model);
            }
            Ok(None) => {
                // No more models
                break;
            }
            Err(e) => {
                eprintln!("SAT solver error: {:?}", e);
                break;
            }
        }
    }

    result
}

/// Evaluate program with choice rules and/or negation using guess-and-check (OLD METHOD - DEPRECATED)
/// This is the old implementation that enumerates all combinations
/// Kept for reference but should not be used for large programs
#[allow(dead_code)]
fn evaluate_with_guess_and_check(program: &Program) -> Vec<AnswerSet> {
    let mut result = Vec::new();

    // Extract choice rules to get atoms AND bounds
    let choice_rules: Vec<&ChoiceRule> = program
        .statements
        .iter()
        .filter_map(|stmt| match stmt {
            Statement::ChoiceRule(cr) => Some(cr),
            _ => None,
        })
        .collect();

    // Extract choice atoms (atoms that can be freely chosen)
    let choice_atoms = extract_choice_atoms(program);

    // Get constant environment for resolving bounds
    let const_env = ConstantEnv::from_program(program);

    // Get bounds from first choice rule (for now, assume single choice rule)
    let (min_size, max_size) = if let Some(first_choice) = choice_rules.first() {
        let min = first_choice
            .lower_bound
            .as_ref()
            .and_then(|t| crate::asp::resolve_bound(t, &const_env))
            .unwrap_or(0) as usize;
        let max = first_choice
            .upper_bound
            .as_ref()
            .and_then(|t| crate::asp::resolve_bound(t, &const_env))
            .map(|u| u as usize)
            .unwrap_or(choice_atoms.len());
        (min, max)
    } else {
        (0, choice_atoms.len())
    };

    // Generate subsets that respect bounds
    let candidates = generate_bounded_choice_subsets(&choice_atoms, min_size, max_size);

    // For each candidate, check if it leads to a valid answer set
    for guess in candidates {
        if let Some(answer_set) = evaluate_with_guess(program, &guess) {
            result.push(answer_set);
        }
    }

    result
}

/// Extract all atoms that appear in choice rules
fn extract_choice_atoms(program: &Program) -> Vec<Atom> {
    let mut atoms = Vec::new();

    for statement in &program.statements {
        if let Statement::ChoiceRule(choice_rule) = statement {
            // For now, handle simple choice rules without conditions
            for element in &choice_rule.elements {
                atoms.push(element.atom.clone());
            }
        }
    }

    atoms
}

/// Generate subsets of choice atoms that respect size bounds
fn generate_bounded_choice_subsets(
    atoms: &[Atom],
    min_size: usize,
    max_size: usize,
) -> Vec<HashSet<Atom>> {
    let n = atoms.len();
    let mut result = Vec::new();

    // Generate all 2^n subsets, but only keep those within bounds
    for mask in 0..(1 << n) {
        let mut subset = HashSet::new();
        for (i, atom) in atoms.iter().enumerate() {
            if (mask & (1 << i)) != 0 {
                subset.insert(atom.clone());
            }
        }

        // Only include subsets that satisfy the bounds
        let size = subset.len();
        if size >= min_size && size <= max_size {
            result.push(subset);
        }
    }

    result
}

/// Evaluate program with a specific guess for choice atoms
fn evaluate_with_guess(program: &Program, guess: &HashSet<Atom>) -> Option<AnswerSet> {
    // Create a modified program with the guess as facts
    let mut solver = AspSatSolver::new();

    // Get all choice atoms
    let all_choice_atoms = extract_choice_atoms(program);

    // Add the guessed atoms as facts
    for atom in guess {
        solver.add_fact(atom);
    }

    // Add constraints for atoms NOT in the guess
    // This ensures that choice atoms not guessed are false
    for atom in &all_choice_atoms {
        if !guess.contains(atom) {
            // Add constraint: :- atom (atom must be false)
            solver.add_constraint(&[atom.clone()]);
        }
    }

    // Collect all atoms that are rule heads (to enforce closed world assumption)
    let mut rule_heads = HashSet::new();
    // Track which rules are active (will be added to SAT solver)
    let mut active_rules: HashSet<Atom> = HashSet::new();

    // Translate the rest of the program (facts, rules, constraints)
    for statement in &program.statements {
        match statement {
            Statement::Fact(fact) => {
                solver.add_fact(&fact.atom);
            }
            Statement::Rule(rule) => {
                // Track rule head for closed world assumption
                rule_heads.insert(rule.head.clone());

                // Handle negation in rules
                let (positive_atoms, negative_atoms) = extract_rule_body(&rule.body);

                // Always add the rule with proper negation handling
                // The SAT solver will determine if the rule fires based on the guess
                solver.add_rule_with_negation(&rule.head, &positive_atoms, &negative_atoms);
                active_rules.insert(rule.head.clone());
            }
            Statement::Constraint(constraint) => {
                let (positive_atoms, _) = extract_constraint_body(&constraint.body);
                solver.add_constraint(&positive_atoms);
            }
            Statement::ChoiceRule(_) => {
                // Already handled via guess
            }
            Statement::ConstDecl(_) | Statement::Test(_) | Statement::Optimize(_) => {
                // Skip - handled elsewhere or not relevant for SAT translation
            }
        }
    }

    // Closed world assumption: atoms that are rule heads but have no active rules
    // must be false (they have no support)
    for head in &rule_heads {
        if !active_rules.contains(head) && !guess.contains(head) {
            // This head has no active rules and is not in the guess
            // So it must be false
            solver.add_constraint(&[head.clone()]);
        }
    }

    // Solve the SAT problem
    match solver.solve() {
        Ok(Some(model)) => {
            // Convert to answer set
            let mut atoms = HashSet::new();
            for (atom, value) in model {
                if value {
                    atoms.insert(atom);
                }
            }
            Some(AnswerSet { atoms })
        }
        Ok(None) => None, // UNSAT
        Err(_) => None,   // Error
    }
}

/// Extract positive and negative atoms from rule body
fn extract_rule_body(body: &[Literal]) -> (Vec<Atom>, Vec<Atom>) {
    let mut positive = Vec::new();
    let mut negative = Vec::new();

    for lit in body {
        match lit {
            Literal::Positive(atom) => positive.push(atom.clone()),
            Literal::Negative(atom) => negative.push(atom.clone()),
            Literal::Aggregate(_) => {
                // TODO: Handle aggregates
            }
        }
    }

    (positive, negative)
}

/// Extract positive and negative atoms from constraint body
fn extract_constraint_body(body: &[Literal]) -> (Vec<Atom>, Vec<Atom>) {
    extract_rule_body(body)
}

/// Evaluate a definite program (no choices, no negation) using SAT
fn evaluate_definite_program(program: &Program) -> Vec<AnswerSet> {
    let solver = translate_to_sat(program);

    // Solve
    match solver.solve() {
        Ok(Some(model)) => {
            // Convert SAT model to answer set
            let mut atoms = HashSet::new();
            for (atom, value) in model {
                if value {
                    atoms.insert(atom);
                }
            }
            vec![AnswerSet { atoms }]
        }
        Ok(None) => {
            // UNSAT - no answer sets
            vec![]
        }
        Err(_) => {
            // Error
            vec![]
        }
    }
}

/// Check if program has choice rules
fn has_choice_rules(program: &Program) -> bool {
    program.statements.iter().any(|stmt| matches!(stmt, Statement::ChoiceRule(_)))
}

/// Check if program has negation in rule bodies
fn has_negation_in_rules(program: &Program) -> bool {
    program.statements.iter().any(|stmt| {
        if let Statement::Rule(rule) = stmt {
            rule.body.iter().any(|lit| matches!(lit, Literal::Negative(_)))
        } else {
            false
        }
    })
}

/// Check if a rule is propositional (has no variables)
fn is_rule_propositional(rule: &Rule) -> bool {
    fn term_is_ground(term: &Term) -> bool {
        match term {
            Term::Variable(_) => false,
            Term::Constant(_) => true,
            Term::Compound(_, terms) => terms.iter().all(term_is_ground),
            Term::Range(start, end) => term_is_ground(start) && term_is_ground(end),
        }
    }

    fn atom_is_ground(atom: &Atom) -> bool {
        atom.terms.iter().all(term_is_ground)
    }

    fn literal_is_ground(lit: &Literal) -> bool {
        match lit {
            Literal::Positive(atom) | Literal::Negative(atom) => atom_is_ground(atom),
            Literal::Aggregate(_) => true, // Aggregates are handled separately
        }
    }

    atom_is_ground(&rule.head) && rule.body.iter().all(literal_is_ground)
}

/// Translate ASP program to SAT clauses using Clark completion
fn translate_to_sat(program: &Program) -> AspSatSolver {
    let mut solver = AspSatSolver::new();

    // Collect all rules by head predicate for Clark completion
    let mut rules_by_head: std::collections::HashMap<Atom, Vec<Vec<Atom>>> =
        std::collections::HashMap::new();

    for statement in &program.statements {
        match statement {
            Statement::Fact(fact) => {
                // Facts are unit clauses: atom must be true
                solver.add_fact(&fact.atom);
            }
            Statement::Rule(rule) => {
                // Extract positive and negative body literals
                let mut positive_body: Vec<Atom> = Vec::new();
                let mut negative_body: Vec<Atom> = Vec::new();

                for lit in &rule.body {
                    match lit {
                        Literal::Positive(atom) => positive_body.push(atom.clone()),
                        Literal::Negative(atom) => negative_body.push(atom.clone()),
                        Literal::Aggregate(_) => {
                            // TODO: Handle aggregates in rule bodies
                            // For now, skip them
                        }
                    }
                }

                // Add rule with negation support
                solver.add_rule_with_negation(&rule.head, &positive_body, &negative_body);

                // Also collect for Clark completion (only positive body for now)
                rules_by_head
                    .entry(rule.head.clone())
                    .or_insert_with(Vec::new)
                    .push(positive_body);
            }
            Statement::Constraint(constraint) => {
                let body_atoms: Vec<Atom> = constraint
                    .body
                    .iter()
                    .filter_map(|lit| {
                        if let Literal::Positive(atom) = lit {
                            Some(atom.clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                solver.add_constraint(&body_atoms);
            }
            Statement::ChoiceRule(_) => {
                // Skip choice rules in definite program translation
                // They should be handled by guess-and-check
            }
            Statement::ConstDecl(_) => {
                // Constants are handled during parsing/grounding
                // No SAT clauses needed
            }
            Statement::Test(_) => {
                // Tests are not part of the logical program
                // Skip them
            }
            Statement::Optimize(_) => {
                // Optimization is handled after finding answer sets
                // No SAT clauses needed
            }
        }
    }

    // Now add Clark completion formulas for each head
    // For head h with bodies b1, b2, ..., bn:
    // h <-> (b1 OR b2 OR ... OR bn)
    // Which in CNF becomes:
    // 1. (NOT h OR b1 OR b2 OR ... OR bn) - if h then at least one body
    // 2. For each body bi = (bi1 AND bi2 AND ...):
    //    (h OR NOT bi1 OR NOT bi2 OR ...) - if body bi then h
    //    This is already added by add_rule above
    //
    // Actually, full Clark completion needs:
    // h <-> (b1 OR b2 OR ...)
    // = (h -> (b1 OR ...)) AND ((b1 OR ...) -> h)
    // = (NOT h OR b1 OR ...) AND (NOT b1 OR h) AND (NOT b2 OR h) AND ...
    //
    // Wait, that's not right either. Let me think about this more carefully.
    //
    // For a rule h :- b1, b2:
    // Clark completion says: h <-> (b1 AND b2)
    // = (NOT h OR (b1 AND b2)) AND ((b1 AND b2) -> h)
    // = (NOT h OR b1) AND (NOT h OR b2) AND (NOT b1 OR NOT b2 OR h)
    //
    // The last clause is what we already have from add_rule.
    // We need to add: (NOT h OR b1) AND (NOT h OR b2)
    // These say "if h is true, then b1 must be true, and b2 must be true"
    //
    // For multiple rules with same head:
    // h :- b1.
    // h :- c1, c2.
    // Clark completion: h <-> (b1 OR (c1 AND c2))
    // This is more complex to translate to CNF.
    //
    // Actually, for simplicity, let me just ensure that atoms without
    // any supporting rules are false by default.

    // Rules are already added with add_rule_with_negation above
    // which handles both the implication and Clark completion
    // The rules_by_head collection is no longer needed
    let _ = rules_by_head;  // Suppress unused variable warning

    solver
}

/// Generate candidate guesses for choice rules and negation
#[allow(dead_code)]
fn generate_candidates(_program: &Program) -> Vec<HashSet<Atom>> {
    // TODO: Implement candidate generation
    vec![]
}

/// Check if a candidate is a stable model
#[allow(dead_code)]
fn is_stable_model(_program: &Program, _candidate: &HashSet<Atom>) -> bool {
    // TODO: Implement stability check
    false
}
