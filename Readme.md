# ProcLog

A Datalog variant for procedural generation in games, combining Answer Set Programming (ASP) and declarative design.

### Syntax Support

- **Facts**: `parent(john, mary).`
- **Rules**: `ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).`
- **Constraints**: `:- unsafe(X).`
- **Negation**: `not reachable(X, Y)`
- **Comments**:
  - Line comments: `% This is a comment`
  - Block comments: `/* This is a block comment */`

### Datatypes

ProcLog supports all common datatypes:

**1. Integers**
- Positive: `42`, `100`
- Negative: `-25`, `-10`
- Zero: `0`

**2. Floats**
- Positive: `3.14`, `10.0`
- Negative: `-2.5`

**3. Booleans**
- True: `true`
- False: `false`

**4. Strings**
- Quoted text: `"Hello, World!"`, `"Alice"`
- Empty string: `""`

**5. Atoms**
- Lowercase identifiers: `john`, `warrior`, `treasure_room`
- Used for symbolic constants

**6. Variables**
- Uppercase: `X`, `Y`, `Player`
- Underscore prefix: `_tmp`, `_x`, `_y`

**7. Compound Terms**
- Nested structures: `f(a, b, c)`
- Example: `inventory(player1, item(sword, 10, 5.5))`

## Running Tests

```bash
cargo test
```

## Running the app

You can start it in REPL mode:

```bash
cargo run -q -p proclog-cli -- repl ./examples/01_basic_facts.pl
```

Or use the built inn test runner:

```bash
cargo run -q -p proclog-cli -- test ./examples/01_basic_facts.pl
```

## Example Programs

You'll find examples in the `examples/` directory.
To run all examples you can run:

```bash
cargo run -q -p proclog-cli -- test examples/*
```

## Development Approach

This project follows **Test-Driven Development (TDD)**:
- Write tests first
- Implement to pass tests
- Refactor as needed

All major components will have comprehensive test coverage before implementation.
