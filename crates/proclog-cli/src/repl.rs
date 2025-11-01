use chumsky::error::{Simple, SimpleReason};
use proclog::asp::{asp_evaluation, asp_sample, AnswerSet};
use proclog::ast::{
    Atom, Constraint, Literal, Program, Query, Rule, Statement, Symbol, Term, Value,
};
use proclog::constants::ConstantEnv;
use proclog::database::FactDatabase;
use proclog::evaluation::stratified_evaluation_with_constraints;
use proclog::parser::{parse_program, parse_query};
use proclog::query::{evaluate_query, query_variables};
use proclog::unification::Substitution;
use std::collections::HashSet;

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
}

/// Core ProcLog REPL engine.
pub struct ReplEngine {
    statements: Vec<Statement>,
    compiled: Option<CompiledProgram>,
    dirty: bool,
    sample_count: usize,
    sample_seed: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct ReplStats {
    pub fact_count: usize,
    pub rule_count: usize,
    pub constraint_count: usize,
    pub choice_rule_count: usize,
    pub constant_count: usize,
    pub sample_count: usize,
    pub sample_seed: u64,
    pub cached_answer_sets: Option<usize>,
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
            sample_count: 5,
            sample_seed: 0,
        }
    }

    /// Process a single line of input and return the resulting response.
    pub fn process_line(&mut self, input: &str) -> EngineResponse {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return EngineResponse::info(Vec::new());
        }

        if trimmed.starts_with(":sample") {
            return self.handle_sample_command(trimmed);
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
            Err(errors) => EngineResponse::error(format_parse_errors(input, errors)),
        }
    }

    fn handle_query(&mut self, input: &str) -> EngineResponse {
        match parse_query(input) {
            Ok(query) => self.evaluate_query(query),
            Err(errors) => EngineResponse::error(format_parse_errors(input, errors)),
        }
    }

    fn handle_sample_command(&mut self, input: &str) -> EngineResponse {
        let mut parts = input.split_whitespace();
        let _ = parts.next(); // skip ":sample"

        let mut count_arg = None;
        let mut seed_arg = None;

        if let Some(part) = parts.next() {
            match part.parse::<usize>() {
                Ok(value) => count_arg = Some(value),
                Err(_) => {
                    return EngineResponse::error(vec![format!(
                        "Invalid sample count '{}'. Expected a positive integer.",
                        part
                    )])
                }
            }
        }

        if let Some(part) = parts.next() {
            match part.parse::<u64>() {
                Ok(value) => seed_arg = Some(value),
                Err(_) => {
                    return EngineResponse::error(vec![format!(
                        "Invalid sample seed '{}'. Expected a non-negative integer.",
                        part
                    )])
                }
            }
        }

        if parts.next().is_some() {
            return EngineResponse::error(vec!["Usage: :sample [count] [seed]".to_string()]);
        }

        let count = count_arg.unwrap_or(self.sample_count);
        if count == 0 {
            return EngineResponse::error(vec![
                "Sample count must be greater than zero.".to_string()
            ]);
        }

        let seed = seed_arg.unwrap_or(self.sample_seed);

        self.sample_count = count;
        self.sample_seed = seed;

        match self.ensure_compiled() {
            Ok(compiled) => match &compiled.result {
                CompiledResult::Datalog(db) => {
                    let atoms: HashSet<Atom> = db.all_facts().into_iter().cloned().collect();
                    let answer_set = AnswerSet { atoms };
                    let mut lines = Vec::new();
                    lines.push(
                        "Program has no choice rules; deterministic result shown.".to_string(),
                    );
                    lines.push(format_answer_set_line(1, &answer_set));
                    EngineResponse::output(lines)
                }
                CompiledResult::Asp { program, .. } => {
                    let samples = asp_sample(program, seed, count);
                    let mut lines = Vec::new();
                    lines.push(format!(
                        "Sampled {} answer set(s) with seed {} (requested {}).",
                        samples.len(),
                        seed,
                        count
                    ));

                    if samples.is_empty() {
                        lines.push("No answer sets found.".to_string());
                    } else {
                        for (idx, answer_set) in samples.iter().enumerate() {
                            lines.push(format_answer_set_line(idx + 1, answer_set));
                        }
                    }

                    EngineResponse::output(lines)
                }
            },
            Err(errors) => EngineResponse::error(errors),
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
                    CompiledResult::Asp { answer_sets, .. } => {
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

            let program = Program { statements };
            let answer_sets = asp_evaluation(&program);
            CompiledResult::Asp {
                program,
                answer_sets,
            }
        } else {
            // Use regular Datalog evaluation
            let derived_db =
                stratified_evaluation_with_constraints(&rules, &constraints, base_facts)
                    .map_err(|err| format!("Evaluation error: {}", err))?;
            CompiledResult::Datalog(derived_db)
        };

        Ok(CompiledProgram { const_env, result })
    }

    pub fn stats(&self) -> ReplStats {
        let mut fact_count = 0usize;
        let mut rule_count = 0usize;
        let mut constraint_count = 0usize;
        let mut choice_rule_count = 0usize;
        let mut constant_count = 0usize;

        for statement in &self.statements {
            match statement {
                Statement::Fact(_) => fact_count += 1,
                Statement::Rule(_) => rule_count += 1,
                Statement::Constraint(_) => constraint_count += 1,
                Statement::ChoiceRule(_) => choice_rule_count += 1,
                Statement::ConstDecl(_) => constant_count += 1,
                Statement::Test(_) => {}
            }
        }

        let cached_answer_sets =
            self.compiled
                .as_ref()
                .and_then(|compiled| match &compiled.result {
                    CompiledResult::Asp { answer_sets, .. } => Some(answer_sets.len()),
                    CompiledResult::Datalog(_) => None,
                });

        ReplStats {
            fact_count,
            rule_count,
            constraint_count,
            choice_rule_count,
            constant_count,
            sample_count: self.sample_count,
            sample_seed: self.sample_seed,
            cached_answer_sets,
        }
    }
}

enum CompiledResult {
    Datalog(FactDatabase),
    Asp {
        program: Program,
        answer_sets: Vec<AnswerSet>,
    },
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

fn format_atom(atom: &Atom) -> String {
    if atom.terms.is_empty() {
        atom.predicate.as_ref().to_string()
    } else {
        let rendered_terms: Vec<String> = atom.terms.iter().map(format_term).collect();
        format!("{}({})", atom.predicate.as_ref(), rendered_terms.join(", "))
    }
}

fn format_answer_set_line(index: usize, answer_set: &AnswerSet) -> String {
    let mut atoms: Vec<String> = answer_set.atoms.iter().map(format_atom).collect();
    atoms.sort();

    if atoms.is_empty() {
        format!("{}: {{}}", index)
    } else {
        format!("{}: {{{}}}", index, atoms.join(", "))
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

fn format_parse_errors(input: &str, errors: Vec<Simple<char>>) -> Vec<String> {
    #[derive(Clone)]
    struct LineInfo {
        number: usize,
        start_char: usize,
        end_char: usize,
        content: String,
    }

    fn build_line_info(input: &str) -> Vec<LineInfo> {
        let mut lines = Vec::new();
        let mut char_offset = 0usize;

        for (idx, chunk) in input.split_inclusive('\n').enumerate() {
            let has_newline = chunk.ends_with('\n');
            let content = if has_newline {
                &chunk[..chunk.len() - 1]
            } else {
                chunk
            };
            let content_chars = content.chars().count();
            let start_char = char_offset;
            let end_char = start_char + content_chars;
            lines.push(LineInfo {
                number: idx + 1,
                start_char,
                end_char,
                content: content.to_string(),
            });

            char_offset = end_char;
            if has_newline {
                char_offset += 1; // account for newline character
            }
        }

        if lines.is_empty() {
            lines.push(LineInfo {
                number: 1,
                start_char: 0,
                end_char: 0,
                content: String::new(),
            });
        }

        lines
    }

    fn locate_line<'a>(lines: &'a [LineInfo], char_index: usize) -> &'a LineInfo {
        lines
            .iter()
            .find(|line| char_index >= line.start_char && char_index <= line.end_char)
            .unwrap_or_else(|| {
                lines
                    .iter()
                    .rev()
                    .find(|line| char_index >= line.start_char)
                    .unwrap_or(&lines[0])
            })
    }

    fn char_index_to_column(line: &LineInfo, char_index: usize) -> usize {
        let relative = if char_index < line.start_char {
            0
        } else {
            char_index.saturating_sub(line.start_char)
        };
        relative.min(line.content.chars().count()) + 1
    }

    fn pointer_line(line: &LineInfo, span: &std::ops::Range<usize>) -> (usize, usize) {
        let line_len = line.content.chars().count();
        let start_offset = if span.start < line.start_char {
            0
        } else {
            span.start.saturating_sub(line.start_char)
        };
        let start_offset = start_offset.min(line_len);
        let span_len = if span.end > span.start {
            span.end - span.start
        } else {
            1
        };
        let available = line_len.saturating_sub(start_offset);
        let pointer_width = if available == 0 {
            1
        } else {
            span_len.min(available).max(1)
        };

        (start_offset, pointer_width)
    }

    fn format_expected(err: &Simple<char>) -> Vec<String> {
        let mut tokens: Vec<String> = err
            .expected()
            .into_iter()
            .map(|expected| match expected {
                Some(ch) => format!("'{}'", ch),
                None => "end of input".to_string(),
            })
            .collect();
        tokens.sort();
        tokens.dedup();
        tokens
    }

    let lines = build_line_info(input);
    let mut formatted = Vec::new();

    for error in errors {
        let span = error.span();
        let line = locate_line(&lines, span.start);
        let column = char_index_to_column(line, span.start);
        let (pointer_offset, pointer_width) = pointer_line(line, &span);

        let expected_tokens = format_expected(&error);
        let reason = match error.reason() {
            SimpleReason::Unexpected => {
                let expected = if expected_tokens.is_empty() {
                    None
                } else {
                    Some(expected_tokens.join(", "))
                };
                match (expected, error.found()) {
                    (Some(expected), Some(found)) => {
                        format!("expected {}, found '{}'", expected, found)
                    }
                    (Some(expected), None) => {
                        format!("expected {}, found end of input", expected)
                    }
                    (None, Some(found)) => format!("unexpected '{}'", found),
                    (None, None) => "unexpected end of input".to_string(),
                }
            }
            SimpleReason::Unclosed {
                span: unclosed_span,
                delimiter,
            } => {
                let delimiter = format!("'{}'", delimiter);
                let expected = if expected_tokens.is_empty() {
                    delimiter.clone()
                } else {
                    expected_tokens.join(", ")
                };
                format!(
                    "unclosed {}, expected {} to close span {:?}",
                    delimiter, expected, unclosed_span
                )
            }
            SimpleReason::Custom(msg) => msg.to_string(),
        };

        formatted.push(format!(
            "Parse error at line {}, column {}: {}",
            line.number, column, reason
        ));
        formatted.push(format!("  {}", line.content));
        formatted.push(format!(
            "  {}{}",
            " ".repeat(pointer_offset),
            "^".repeat(pointer_width)
        ));
    }

    formatted
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
    fn test_sample_command_outputs_samples() {
        let mut engine = ReplEngine::new();
        engine.process_line("item(a).");
        engine.process_line("item(b).");
        engine.process_line("{ selected(X) : item(X) }.");

        let response = engine.process_line(":sample 2 42");
        assert_eq!(response.kind, ResponseKind::Output);
        assert!(
            response.lines[0].contains("Sampled 2 answer set(s)")
                || response.lines[0].contains("Sampled 1 answer set(s)")
        );
        assert!(
            response
                .lines
                .iter()
                .skip(1)
                .any(|line| line.contains("selected(")),
            "Expected sampled answer sets to be listed"
        );
    }

    #[test]
    fn test_sample_command_invalid_arguments() {
        let mut engine = ReplEngine::new();
        let response = engine.process_line(":sample abc");
        assert_eq!(response.kind, ResponseKind::Error);
        assert!(response
            .lines
            .iter()
            .any(|line| line.contains("Invalid sample count")));
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
        assert_eq!(
            response.lines,
            vec![
                "Parse error at line 1, column 19: expected '%', '.', '/', ':', found end of input"
                    .to_string(),
                "  parent(john, mary)".to_string(),
                "                    ^".to_string(),
            ]
        );
    }
}
