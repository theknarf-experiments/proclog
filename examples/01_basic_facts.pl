% Example 1: Basic Facts and Queries
%
% This example demonstrates the most basic features of ProcLog:
% - Defining facts
% - Querying facts
% - Using variables in queries

% Facts about people
person(alice).
person(bob).
person(charlie).
person(diana).

% Facts about ages
age(alice, 30).
age(bob, 25).
age(charlie, 35).
age(diana, 28).

% Facts about cities
lives_in(alice, london).
lives_in(bob, paris).
lives_in(charlie, london).
lives_in(diana, berlin).

#test "query specific person" {
    person(alice).
    person(bob).

    ?- person(alice).
    + true.

    ?- person(eve).
    - true.
}

#test "query with variables" {
    person(alice).
    person(bob).
    person(charlie).

    age(alice, 30).
    age(bob, 25).
    age(charlie, 35).

    ?- age(alice, 30).
    + true.

    ?- age(alice, 25).
    - true.
}

#test "find all people in london" {
    lives_in(alice, london).
    lives_in(bob, paris).
    lives_in(charlie, london).

    ?- lives_in(alice, london).
    + true.

    ?- lives_in(charlie, london).
    + true.

    ?- lives_in(bob, london).
    - true.
}

#test "query with multiple facts" {
    person(alice).
    age(alice, 30).
    lives_in(alice, london).

    ?- person(alice).
    + true.

    ?- age(alice, 30).
    + true.

    ?- lives_in(alice, london).
    + true.
}
