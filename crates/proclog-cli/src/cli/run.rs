use crate::repl::ReplEngine;
use proclog::asp::{asp_evaluation, asp_sample};
use proclog::parser::parse_program;
use std::fs;
use std::path::Path;

pub fn run(path: &Path, sample: Option<usize>) -> Result<(), String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("failed to read '{}': {}", path.display(), e))?;

    let program = parse_program(&content)
        .map_err(|errs| format!("Parse failed with {} error(s)", errs.len()))?;

    match sample {
        Some(count) => run_with_sampling(&program, count),
        None => run_full(&program, &content),
    }
}

fn run_with_sampling(program: &proclog::ast::Program, count: usize) -> Result<(), String> {
    if count == 0 {
        return Err("Sample count must be greater than zero".into());
    }

    let samples = asp_sample(program, 0, count);
    println!("Sampled {} answer set(s).", samples.len());
    if samples.is_empty() {
        println!("No answer sets found.");
    } else {
        print_answer_sets(&samples);
    }
    Ok(())
}

fn run_full(program: &proclog::ast::Program, content: &str) -> Result<(), String> {
    let answer_sets = asp_evaluation(program);
    if !answer_sets.is_empty() {
        print_answer_sets(&answer_sets);
        return Ok(());
    }

    // Fall back to using the REPL engine to report parsing summaries etc.
    let mut engine = ReplEngine::new();
    let response = engine.process_line(content);
    for line in response.lines {
        println!("{}", line);
    }
    Ok(())
}

fn print_answer_sets(answer_sets: &[proclog::asp::AnswerSet]) {
    for (index, answer_set) in answer_sets.iter().enumerate() {
        println!("Answer set {}:", index + 1);
        let mut atoms: Vec<String> = answer_set
            .atoms
            .iter()
            .map(|atom| format_atom(atom))
            .collect();
        atoms.sort();
        for atom in atoms {
            println!("  {}", atom);
        }
    }
}

fn format_atom(atom: &proclog::ast::Atom) -> String {
    if atom.terms.is_empty() {
        atom.predicate.as_ref().to_string()
    } else {
        let formatted_terms: Vec<String> = atom
            .terms
            .iter()
            .map(|term| format!("{}", term_display(term)))
            .collect();
        format!(
            "{}({})",
            atom.predicate.as_ref(),
            formatted_terms.join(", ")
        )
    }
}

fn term_display(term: &proclog::ast::Term) -> String {
    use proclog::ast::Term;
    match term {
        Term::Variable(sym) => sym.as_ref().to_string(),
        Term::Constant(value) => format!("{}", value_display(value)),
        Term::Compound(functor, args) => {
            let rendered: Vec<String> = args.iter().map(term_display).collect();
            format!("{}({})", functor.as_ref(), rendered.join(", "))
        }
        Term::Range(start, end) => format!("{}..{}", term_display(start), term_display(end)),
    }
}

fn value_display(value: &proclog::ast::Value) -> String {
    use proclog::ast::Value;
    match value {
        Value::Integer(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::String(s) => format!("\"{}\"", s.as_ref()),
        Value::Atom(a) => a.as_ref().to_string(),
    }
}
