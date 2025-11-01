use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::path::PathBuf;
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

#[test]
fn run_subcommand_rejects_zero_sample_count() {
    let mut cmd = cli();
    cmd.args(["run", "--sample", "0", "tests/fixtures/run_sample.proclog"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Sample count must be greater than zero"));
}

#[test]
fn run_subcommand_supports_seed() {
    let output_seed_1 = cli()
        .args([
            "run",
            "--sample",
            "2",
            "--seed",
            "1",
            "tests/fixtures/run_sample.proclog",
        ])
        .output()
        .expect("run command should execute");
    assert!(output_seed_1.status.success());
    let stdout_1 = String::from_utf8(output_seed_1.stdout).expect("stdout utf8");

    let output_seed_2 = cli()
        .args([
            "run",
            "--sample",
            "2",
            "--seed",
            "2",
            "tests/fixtures/run_sample.proclog",
        ])
        .output()
        .expect("run command should execute");
    assert!(output_seed_2.status.success());
    let stdout_2 = String::from_utf8(output_seed_2.stdout).expect("stdout utf8");

    assert_ne!(
        stdout_1, stdout_2,
        "different seeds should affect sampling output"
    );
}

#[test]
fn run_subcommand_same_seed_is_deterministic() {
    let output_first = cli()
        .args([
            "run",
            "--sample",
            "2",
            "--seed",
            "9",
            "tests/fixtures/run_sample.proclog",
        ])
        .output()
        .expect("run command should execute");
    assert!(output_first.status.success());
    let stdout_first = String::from_utf8(output_first.stdout).expect("stdout utf8");

    let output_second = cli()
        .args([
            "run",
            "--sample",
            "2",
            "--seed",
            "9",
            "tests/fixtures/run_sample.proclog",
        ])
        .output()
        .expect("run command should execute");
    assert!(output_second.status.success());
    let stdout_second = String::from_utf8(output_second.stdout).expect("stdout utf8");

    assert_eq!(
        stdout_first, stdout_second,
        "same seed should yield the same sampled answer sets"
    );
}

#[test]
fn run_subcommand_limits_sample_count_to_available_sets() {
    let mut cmd = cli();
    cmd.args([
        "run",
        "--sample",
        "99",
        "tests/fixtures/run_sample.proclog",
    ])
    .assert()
    .success()
    .stdout(
        predicate::str::contains("Sampled 4 answer set(s).")
            .and(predicate::str::contains("Answer set 4")),
    );
}

#[test]
fn run_subcommand_samples_loot_example() {
    let mut cmd = cli();
    let example = workspace_path("examples/11_loot_loadouts_asp.pl");
    cmd.arg("run")
        .args(["--sample", "3", "--seed", "7"])
        .arg(example)
    .assert()
    .success()
    .stdout(
        predicate::str::contains("Sampled 3 answer set(s).")
            .and(predicate::str::contains("summary_weapon")),
    );
}

#[test]
fn run_subcommand_samples_party_example() {
    let mut cmd = cli();
    let example = workspace_path("examples/12_party_formation_asp.pl");
    cmd.arg("run")
        .args(["--sample", "3", "--seed", "21"])
        .arg(example)
    .assert()
    .success()
    .stdout(
        predicate::str::contains("Sampled 3 answer set(s).")
            .and(predicate::str::contains("summary_member")),
    );
}
fn workspace_path(relative: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(relative)
        .to_string_lossy()
        .into_owned()
}
