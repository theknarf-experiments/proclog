@default: test-all

test-all:
    for file in examples/*; do \
        cargo run -q -- test "$file"; \
    done
