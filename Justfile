@default: test-native test-sat

# Test with native ASP solver (all passing examples)
test-native:
    cargo run -q -p proclog-cli -- test \
        examples/01_basic_facts.pl \
        examples/02_rules_and_inference.pl \
        examples/03_transitive_closure.pl \
        examples/04_constants_and_arithmetic.pl \
        examples/05_dungeon_generation.pl \
        examples/06_quest_generation.pl \
        examples/07_item_generation.pl \
        examples/08_character_creation_asp.pl \
        examples/09_dungeon_generation_asp.pl \
        examples/10_item_crafting_asp.pl \
        examples/13_count_aggregates.pl \
        examples/datatypes.proclog \
        examples/dungeon.proclog

# Test with SAT solver backend (currently passing examples)
test-sat:
    cargo run -q -p proclog-cli -- test --sat-solver \
        examples/01_basic_facts.pl \
        examples/02_rules_and_inference.pl \
        examples/03_transitive_closure.pl \
        examples/04_constants_and_arithmetic.pl \
        examples/05_dungeon_generation.pl \
        examples/06_quest_generation.pl \
        examples/07_item_generation.pl \
        examples/08_character_creation_asp.pl \
        examples/10_item_crafting_asp.pl \
        examples/datatypes.proclog \
        examples/dungeon.proclog

# Test everything (may hang on some examples)
test-all:
    cargo run -q -p proclog-cli -- test examples/*

# Test everything with SAT solver (may hang on some examples)
test-all-sat:
    cargo run -q -p proclog-cli -- test --sat-solver examples/*
