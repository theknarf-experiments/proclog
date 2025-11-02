mod cli;
mod repl;
mod test_runner;

use clap::{Parser, Subcommand};

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
        /// One or more ProcLog files containing test blocks
        #[arg(value_name = "FILE", required = true)]
        files: Vec<std::path::PathBuf>,
        /// Watch the file(s) for changes and re-run tests automatically
        #[arg(long)]
        watch: bool,
    },
    /// Start the interactive ProcLog REPL
    Repl {
        /// Optional ProcLog file to preload into the REPL
        #[arg(value_name = "FILE")]
        input: Option<std::path::PathBuf>,
    },
    /// Execute a ProcLog program once (optionally sampling answer sets)
    Run {
        /// Optional sample count; when omitted, run normally without sampling
        #[arg(long = "sample", value_name = "COUNT")]
        sample: Option<usize>,
        /// Optional seed for sampling (defaults to 0)
        #[arg(long = "seed", value_name = "SEED")]
        seed: Option<u64>,
        /// Use SAT solver backend (splr) for ASP evaluation
        #[arg(long = "sat-solver")]
        sat_solver: bool,
        /// Program file to execute
        #[arg(value_name = "FILE", required = true)]
        file: std::path::PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Test { files, watch } => {
            cli::test::run(&files, watch);
        }
        Commands::Repl { input } => {
            if let Err(e) = cli::repl::run(input.as_deref()) {
                eprintln!("Failed to start REPL: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Run {
            sample,
            seed,
            sat_solver,
            file,
        } => {
            if let Err(e) = cli::run::run(&file, sample, seed.unwrap_or(0), sat_solver) {
                eprintln!("Run failed: {}", e);
                std::process::exit(1);
            }
        }
    }
}
