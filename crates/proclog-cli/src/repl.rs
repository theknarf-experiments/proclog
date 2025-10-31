use chumsky::error::Simple;
use proclog::asp::{asp_evaluation, AnswerSet};
use proclog::ast::{Constraint, Literal, Query, Rule, Statement, Symbol, Term, Value};
use proclog::constants::ConstantEnv;
use proclog::database::FactDatabase;
use proclog::evaluation::stratified_evaluation_with_constraints;
use proclog::parser::{parse_program, parse_query};
use proclog::query::{evaluate_query, query_variables};
use proclog::unification::Substitution;

/// Kind of response produced by the REPL engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResponseKind {
    Info,
    Output,
    Error,
}

/// Response produced after processing a line of input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineResponse {
    pub kind: ResponseKind,
    pub lines: Vec<String>,
}

impl EngineResponse {
    fn info(lines: Vec<String>) -> Self {
        Self {
            kind: ResponseKind::Info,
            lines,
        }
    }

    fn output(lines: Vec<String>) -> Self {
        Self {
            kind: ResponseKind::Output,
            lines,
        }
    }

    fn error(lines: Vec<String>) -> Self {
        Self {
            kind: ResponseKind::Error,
            lines,
        }
    }

    fn error_message(message: impl Into<String>) -> Self {
        Self::error(vec![message.into()])
    }
}

/// Core ProcLog REPL engine.
pub struct ReplEngine {
    statements: Vec<Statement>,
    compiled: Option<CompiledProgram>,
    dirty: bool,
}

impl Default for ReplEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplEngine {
    /// Create a new REPL engine instance.
    pub fn new() -> Self {
        Self {
            statements: Vec::new(),
            compiled: None,
            dirty: false,
        }
    }

    /// Process a single line of input and return the resulting response.
    pub fn process_line(&mut self, input: &str) -> EngineResponse {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return EngineResponse::info(Vec::new());
        }

        if trimmed.starts_with("?-") {
            self.handle_query(trimmed)
        } else {
            self.handle_statements(trimmed)
        }
    }

    fn handle_statements(&mut self, input: &str) -> EngineResponse {
        match parse_program(input) {
            Ok(program) => {
                let mut to_add = Vec::new();
                let mut summary = AdditionSummary::default();

                for statement in program.statements {
                    match statement {
                        Statement::Fact(fact) => {
                            summary.facts += 1;
                            to_add.push(Statement::Fact(fact));
                        }
                        Statement::Rule(rule) => {
                            summary.rules += 1;
                            to_add.push(Statement::Rule(rule));
                        }
                        Statement::Constraint(constraint) => {
                            summary.constraints += 1;
                            to_add.push(Statement::Constraint(constraint));
                        }
                        Statement::ConstDecl(decl) => {
                            summary.constants += 1;
                            to_add.push(Statement::ConstDecl(decl));
                        }
                        Statement::ChoiceRule(choice) => {
                            summary.choice_rules += 1;
                            to_add.push(Statement::ChoiceRule(choice));
                        }
                        Statement::ProbFact(_) => {
                            return EngineResponse::error_message(
                                "Probabilistic facts are not supported in the REPL yet.",
                            );
                        }
                        Statement::Test(_) => {
                            summary.skipped_tests += 1;
                        }
                    }
                }

                if to_add.is_empty() && summary.skipped_tests == 0 {
                    return EngineResponse::info(vec!["No statements recognized.".to_string()]);
                }

                if !to_add.is_empty() {
                    self.statements.extend(to_add);
                    self.dirty = true;
                }

                let lines = summary.summary_lines();
                if lines.is_empty() {
                    EngineResponse::info(vec!["No statements recognized.".to_string()])
                } else {
                    EngineResponse::info(lines)
                }
            }
            Err(errors) => EngineResponse::error(format_parse_errors(errors)),
        }
    }

    fn handle_query(&mut self, input: &str) -> EngineResponse {
        match parse_query(input) {
            Ok(query) => self.evaluate_query(query),
            Err(errors) => EngineResponse::error(format_parse_errors(errors)),
        }
    }

    fn evaluate_query(&mut self, query: Query) -> EngineResponse {
        match self.ensure_compiled() {
            Ok(compiled) => {
                let substituted_query = Query {
                    body: query
                        .body
                        .iter()
                        .map(|lit| substitute_literal(&compiled.const_env, lit))
                        .collect(),
                };

                let lines = match &compiled.result {
                    CompiledResult::Datalog(db) => {
                        let results = evaluate_query(&substituted_query, db);
                        format_query_results(&substituted_query, &results)
                    }
                    CompiledResult::Asp(answer_sets) => {
                        let mut all_results = Vec::new();
                        let mut answer_set_count = 0;

                        for answer_set in answer_sets {
                            answer_set_count += 1;
                            // Convert answer set to fact database for query evaluation
                            let mut db = FactDatabase::new();
                            for atom in &answer_set.atoms {
                                db.insert(atom.clone()).unwrap();
                            }

                            let results = evaluate_query(&substituted_query, &db);
                            if !results.is_empty() {
                                all_results.extend(results);
                            }
                        }

                        if answer_sets.is_empty() {
                            vec![format!("No answer sets found.")]
                        } else if all_results.is_empty() {
                            vec![format!(
                                "Query failed in all {} answer sets.",
                                answer_set_count
                            )]
                        } else {
                            let mut lines = format_query_results(&substituted_query, &all_results);
                            lines.insert(
                                0,
                                format!("Query succeeded in {} answer sets:", answer_set_count),
                            );
                            lines
                        }
                    }
                };

                EngineResponse::output(lines)
            }
            Err(err_lines) => EngineResponse::error(err_lines),
        }
    }

    fn ensure_compiled(&mut self) -> Result<&CompiledProgram, Vec<String>> {
        if self.dirty || self.compiled.is_none() {
            let compiled = self.rebuild_program().map_err(|msg| vec![msg])?;
            self.compiled = Some(compiled);
            self.dirty = false;
        }
        Ok(self.compiled.as_ref().expect("compiled program must exist"))
    }

    fn rebuild_program(&self) -> Result<CompiledProgram, String> {
        let const_env = ConstantEnv::from_statements(&self.statements);
        let mut base_facts = FactDatabase::new();
        let mut rules = Vec::new();
        let mut constraints = Vec::new();
        let mut choice_rules = Vec::new();

        for statement in &self.statements {
            match statement {
                Statement::Fact(fact) => {
                    let substituted = const_env.substitute_atom(&fact.atom);
                    base_facts.insert(substituted).unwrap();
                }
                Statement::Rule(rule) => {
                    let head = const_env.substitute_atom(&rule.head);
                    let body = rule
                        .body
                        .iter()
                        .map(|lit| substitute_literal(&const_env, lit))
                        .collect();
                    rules.push(Rule { head, body });
                }
                Statement::Constraint(constraint) => {
                    let body = constraint
                        .body
                        .iter()
                        .map(|lit| substitute_literal(&const_env, lit))
                        .collect();
                    constraints.push(Constraint { body });
                }
                Statement::ChoiceRule(choice) => {
                    // Substitute constants in choice rule bounds and elements
                    let substituted_lower = choice
                        .lower_bound
                        .as_ref()
                        .map(|term| const_env.substitute_term(term));
                    let substituted_upper = choice
                        .upper_bound
                        .as_ref()
                        .map(|term| const_env.substitute_term(term));

                    let substituted_elements: Vec<_> = choice
                        .elements
                        .iter()
                        .map(|elem| {
                            let substituted_atom = const_env.substitute_atom(&elem.atom);
                            let substituted_condition: Vec<_> = elem
                                .condition
                                .iter()
                                .map(|lit| match lit {
                                    proclog::ast::Literal::Positive(atom) => {
                                        proclog::ast::Literal::Positive(
                                            const_env.substitute_atom(atom),
                                        )
                                    }
                                    proclog::ast::Literal::Negative(atom) => {
                                        proclog::ast::Literal::Negative(
                                            const_env.substitute_atom(atom),
                                        )
                                    }
                                })
                                .collect();

                            proclog::ast::ChoiceElement {
                                atom: substituted_atom,
                                condition: substituted_condition,
                            }
                        })
                        .collect();

                    choice_rules.push(proclog::ast::ChoiceRule {
                        lower_bound: substituted_lower,
                        upper_bound: substituted_upper,
                        elements: substituted_elements,
                        body: choice.body.clone(), // Body conditions are not substituted as they reference variables
                    });
                }
                Statement::ConstDecl(_) => {
                    // Already captured in const_env
                }
                Statement::ProbFact(_) => {
                    return Err(
                        "Probabilistic facts are not supported in the REPL yet.".to_string()
                    );
                }
                Statement::Test(_) => {
                    return Err("Test blocks are not supported in the REPL.".to_string());
                }
            }
        }

        let result = if !choice_rules.is_empty() {
            // Use ASP evaluation
            let mut statements = Vec::new();

            // Add facts
            for atom in base_facts.all_facts() {
                statements.push(Statement::Fact(proclog::ast::Fact { atom: atom.clone() }));
            }

            // Add rules
            for rule in &rules {
                statements.push(Statement::Rule(rule.clone()));
            }

            // Add constraints
            for constraint in &constraints {
                statements.push(Statement::Constraint(constraint.clone()));
            }

            // Add choice rules
            for choice in &choice_rules {
                statements.push(Statement::ChoiceRule(choice.clone()));
            }

            let program = proclog::ast::Program { statements };
            CompiledResult::Asp(asp_evaluation(&program))
        } else {
            // Use regular Datalog evaluation
            let derived_db =
                stratified_evaluation_with_constraints(&rules, &constraints, base_facts)
                    .map_err(|err| format!("Evaluation error: {}", err))?;
            CompiledResult::Datalog(derived_db)
        };

        Ok(CompiledProgram { const_env, result })
    }
}

enum CompiledResult {
    Datalog(FactDatabase),
    Asp(Vec<AnswerSet>),
}

struct CompiledProgram {
    const_env: ConstantEnv,
    result: CompiledResult,
}

#[derive(Default)]
struct AdditionSummary {
    facts: usize,
    rules: usize,
    constraints: usize,
    choice_rules: usize,
    constants: usize,
    skipped_tests: usize,
}

impl AdditionSummary {
    fn summary_lines(&self) -> Vec<String> {
        let mut parts = Vec::new();
        if self.facts > 0 {
            parts.push(format_count("fact", self.facts));
        }
        if self.rules > 0 {
            parts.push(format_count("rule", self.rules));
        }
        if self.constraints > 0 {
            parts.push(format_count("constraint", self.constraints));
        }
        if self.choice_rules > 0 {
            parts.push(format_count("choice rule", self.choice_rules));
        }
        if self.constants > 0 {
            parts.push(format_count("constant", self.constants));
        }

        let mut lines = Vec::new();
        if !parts.is_empty() {
            let message = if parts.len() == 1 {
                format!("Added {}.", parts[0])
            } else if parts.len() == 2 {
                format!("Added {} and {}.", parts[0], parts[1])
            } else {
                let last = parts.pop().unwrap();
                format!("Added {}, and {}.", parts.join(", "), last)
            };
            lines.push(message);
        }

        if self.skipped_tests > 0 {
            let msg = if self.skipped_tests == 1 {
                "Skipped 1 test block (tests are not loaded into the REPL).".to_string()
            } else {
                format!(
                    "Skipped {} test blocks (tests are not loaded into the REPL).",
                    self.skipped_tests
                )
            };
            lines.push(msg);
        }

        lines
    }
}

fn format_count(label: &str, count: usize) -> String {
    if count == 1 {
        format!("1 {}", label)
    } else {
        format!("{} {}s", count, label)
    }
}

fn substitute_literal(env: &ConstantEnv, literal: &Literal) -> Literal {
    match literal {
        Literal::Positive(atom) => Literal::Positive(env.substitute_atom(atom)),
        Literal::Negative(atom) => Literal::Negative(env.substitute_atom(atom)),
    }
}

fn format_query_results(query: &Query, results: &[Substitution]) -> Vec<String> {
    let mut vars: Vec<Symbol> = query_variables(query).into_iter().collect();
    vars.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));

    if vars.is_empty() {
        if results.is_empty() {
            vec!["false.".to_string()]
        } else {
            vec!["true.".to_string()]
        }
    } else if results.is_empty() {
        vec!["No results.".to_string()]
    } else {
        let mut lines = Vec::with_capacity(results.len() + 1);
        let summary = if results.len() == 1 {
            "Found 1 result.".to_string()
        } else {
            format!("Found {} results.", results.len())
        };
        lines.push(summary);

        let mut assignments: Vec<String> = results
            .iter()
            .map(|subst| format_assignment(subst, &vars))
            .collect();
        assignments.sort();
        lines.extend(assignments);
        lines
    }
}

fn format_assignment(subst: &Substitution, vars: &[Symbol]) -> String {
    let mut parts = Vec::with_capacity(vars.len());
    for var in vars {
        let value = subst
            .get(var)
            .map(|term| subst.apply(term))
            .map(|term| format_term(&term))
            .unwrap_or_else(|| "_".to_string());
        parts.push(format!("{} = {}", var.as_ref(), value));
    }
    parts.join(", ")
}

fn format_term(term: &Term) -> String {
    match term {
        Term::Variable(sym) => sym.as_ref().to_string(),
        Term::Constant(value) => format_value(value),
        Term::Compound(functor, args) => {
            let rendered_args: Vec<String> = args.iter().map(format_term).collect();
            if rendered_args.is_empty() {
                format!("{}()", functor.as_ref())
            } else {
                format!("{}({})", functor.as_ref(), rendered_args.join(", "))
            }
        }
        Term::Range(start, end) => format!("{}..{}", format_term(start), format_term(end)),
    }
}

fn format_value(value: &Value) -> String {
    match value {
        Value::Integer(n) => n.to_string(),
        Value::Float(f) => format_float(*f),
        Value::Boolean(b) => b.to_string(),
        Value::String(sym) => format!("\"{}\"", sym.as_ref()),
        Value::Atom(sym) => sym.as_ref().to_string(),
    }
}

fn format_float(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{:.1}", value)
    } else {
        value.to_string()
    }
}

fn format_parse_errors(errors: Vec<Simple<char>>) -> Vec<String> {
    errors
        .into_iter()
        .map(|err| format!("Parse error: {}", err))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_fact_and_query_variable() {
        let mut engine = ReplEngine::new();
        let response = engine.process_line("parent(john, mary).");
        assert_eq!(response.kind, ResponseKind::Info);
        assert_eq!(response.lines, vec!["Added 1 fact.".to_string()]);

        let response = engine.process_line("parent(alice, mary).");
        assert_eq!(response.kind, ResponseKind::Info);
        assert_eq!(response.lines, vec!["Added 1 fact.".to_string()]);

        let response = engine.process_line("?- parent(X, mary).");
        assert_eq!(response.kind, ResponseKind::Output);
        assert_eq!(response.lines[0], "Found 2 results.".to_string());
        assert_eq!(
            &response.lines[1..],
            &["X = alice".to_string(), "X = john".to_string()]
        );
    }

    #[test]
    fn test_repl_choice_rules() {
        let mut engine = ReplEngine::new();

        // Add facts
        let response = engine.process_line("race(human).");
        assert_eq!(response.kind, ResponseKind::Info);
        assert!(response.lines[0].contains("Added 1 fact"));

        let response = engine.process_line("race(elf).");
        assert_eq!(response.kind, ResponseKind::Info);

        // Add choice rule
        let response = engine.process_line("1 { character_race(R) : race(R) } 1.");
        assert_eq!(response.kind, ResponseKind::Info);
        assert!(response.lines[0].contains("Added 1 choice rule"));

        // Query should work and show results from answer sets
        let response = engine.process_line("?- character_race(X).");
        assert_eq!(response.kind, ResponseKind::Output);
        // Should contain results from ASP evaluation
        assert!(response.lines.len() > 0);
    }

    #[test]
    fn test_ground_query_true_false() {
        let mut engine = ReplEngine::new();
        engine.process_line("parent(john, mary).");

        let response = engine.process_line("?- parent(john, mary).");
        assert_eq!(response.kind, ResponseKind::Output);
        assert_eq!(response.lines, vec!["true.".to_string()]);

        let response = engine.process_line("?- parent(alice, mary).");
        assert_eq!(response.kind, ResponseKind::Output);
        assert_eq!(response.lines, vec!["false.".to_string()]);
    }

    #[test]
    fn test_rule_derivation() {
        let mut engine = ReplEngine::new();
        engine.process_line("parent(john, mary).");
        engine.process_line("parent(mary, alice).");
        engine.process_line("ancestor(X, Y) :- parent(X, Y).");
        engine.process_line("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).");

        let response = engine.process_line("?- ancestor(john, alice).");
        assert_eq!(response.kind, ResponseKind::Output);
        assert_eq!(response.lines, vec!["true.".to_string()]);

        let response = engine.process_line("?- ancestor(X, alice).");
        assert_eq!(response.kind, ResponseKind::Output);
        assert_eq!(response.lines[0], "Found 2 results.".to_string());
        assert_eq!(
            &response.lines[1..],
            &["X = john".to_string(), "X = mary".to_string()]
        );
    }

    #[test]
    fn test_constants_substitution() {
        let mut engine = ReplEngine::new();
        engine.process_line("#const width = 10.");
        engine.process_line("dimension(width).");

        let response = engine.process_line("?- dimension(10).");
        assert_eq!(response.kind, ResponseKind::Output);
        assert_eq!(response.lines, vec!["true.".to_string()]);
    }

    #[test]
    fn test_parse_error_reported() {
        let mut engine = ReplEngine::new();
        let response = engine.process_line("parent(john, mary)");
        assert_eq!(response.kind, ResponseKind::Error);
        assert!(!response.lines.is_empty());
        assert!(
            response.lines[0].contains("Parse error"),
            "expected parse error message, got {:?}",
            response.lines
        );
    }
}
