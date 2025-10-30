@default: test-all

test-all:
    cargo run -q -- test examples/*
