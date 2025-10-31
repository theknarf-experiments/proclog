@default: test-all

test-all:
    cargo run -q -p proclog-cli -- test examples/*
