use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

fn cli() -> Command {
    Command::cargo_bin("proclog-cli").expect("binary exists")
}

#[test]
fn run_subcommand_requires_file() {
    let mut cmd = cli();
    cmd.arg("run");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Usage: proclog-cli run"));
}

#[test]
fn run_subcommand_executes_program() {
    let mut cmd = cli();
    cmd.args(["run", "tests/fixtures/run_sample.proclog"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Answer set 1"));
}

#[test]
fn run_subcommand_with_sample_flag() {
    let mut cmd = cli();
    cmd.args(["run", "--sample", "2", "tests/fixtures/run_sample.proclog"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Sampled 2 answer set"));
}
