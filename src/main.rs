mod asp;
mod ast;
mod builtins;
mod cli;
mod constants;
mod database;
mod evaluation;
mod grounding;
mod parser;
mod query;
mod repl;
mod safety;
mod stratification;
mod test_runner;
mod unification;

use clap::{Parser, Subcommand};

#[cfg(test)]
mod asp_multiple_choice_tests;

#[cfg(test)]
mod arithmetic_integration_tests;

#[cfg(test)]
mod choice_constant_bounds_tests;

#[cfg(test)]
mod query_integration_tests;

pub(crate) const COLOR_RESET: &str = "\x1b[0m";
pub(crate) const COLOR_GREEN: &str = "\x1b[32m";
pub(crate) const COLOR_RED: &str = "\x1b[31m";
pub(crate) const COLOR_YELLOW: &str = "\x1b[33m";
pub(crate) const COLOR_CYAN: &str = "\x1b[36m";

#[derive(Parser)]
#[command(
    name = "ProcLog",
    version,
    about = "Datalog for procedural generation",
    long_about = None,
    subcommand_required = true,
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run tests from a ProcLog file
    Test {
        /// Path to the ProcLog file containing test blocks
        filename: String,
    },
    /// Start the interactive ProcLog REPL
    Repl {
        /// Optional ProcLog file to preload into the REPL
        #[arg(value_name = "FILE")]
        input: Option<std::path::PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Test { filename } => {
            cli::test::run(&filename);
        }
        Commands::Repl { input } => {
            if let Err(e) = cli::repl::run(input.as_deref()) {
                eprintln!("Failed to start REPL: {}", e);
                std::process::exit(1);
            }
        }
    }
}
