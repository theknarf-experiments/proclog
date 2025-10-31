use std::path::Path;

use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;

#[test]
fn reports_parse_error_with_context() {
    let mut cmd = cargo_bin_cmd!("proclog-cli");

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let file = Path::new(manifest_dir).join("tests/fixtures/invalid_program.pro");
    let file_display = file.display().to_string();

    cmd.arg("test").arg(file);

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains(format!(
            "\u{1b}[31merror:\u{1b}[0m Failed to parse '{}'",
            file_display
        )))
        .stderr(predicate::str::contains(format!(" --> {}:2:1", file_display)))
        .stderr(predicate::str::contains("^"))
        .stderr(predicate::str::contains("Unexpected character '#'")
            .and(predicate::str::contains("term")));
}
