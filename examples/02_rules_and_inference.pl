% Example 2: Rules and Inference
%
% This example demonstrates:
% - Defining rules with :- (if)
% - Logical inference
% - Combining multiple conditions

% Base facts
parent(john, mary).
parent(john, tom).
parent(mary, alice).
parent(mary, bob).
parent(tom, charlie).

male(john).
male(tom).
male(bob).
male(charlie).

female(mary).
female(alice).

% Rules for family relationships
father(X, Y) :- parent(X, Y), male(X).
mother(X, Y) :- parent(X, Y), female(X).

% Grandparent rule
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

% Sibling rule (same parent, different people)
sibling(X, Y) :- parent(P, X), parent(P, Y), not X = Y.

#test "father relationships" {
    parent(john, mary).
    parent(mary, alice).
    male(john).
    female(mary).

    father(X, Y) :- parent(X, Y), male(X).

    ?- father(john, mary).
    + true.

    ?- father(mary, alice).
    - true.
}

#test "mother relationships" {
    parent(john, mary).
    parent(mary, alice).
    male(john).
    female(mary).

    mother(X, Y) :- parent(X, Y), female(X).

    ?- mother(mary, alice).
    + true.

    ?- mother(john, mary).
    - true.
}

#test "grandparent inference" {
    parent(john, mary).
    parent(mary, alice).
    parent(mary, bob).

    grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

    ?- grandparent(john, alice).
    + true.

    ?- grandparent(john, bob).
    + true.

    ?- grandparent(mary, john).
    - true.
}

#test "multi-generation family" {
    parent(john, mary).
    parent(john, tom).
    parent(mary, alice).
    parent(mary, bob).
    parent(tom, charlie).

    grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

    % John is grandparent to alice, bob, and charlie
    ?- grandparent(john, alice).
    + true.

    ?- grandparent(john, bob).
    + true.

    ?- grandparent(john, charlie).
    + true.

    % Mary is only parent, not grandparent in this tree
    ?- grandparent(mary, john).
    - true.
}

#test "complex family queries" {
    parent(john, mary).
    parent(john, tom).
    parent(mary, alice).
    male(john).
    male(tom).
    female(mary).
    female(alice).

    father(X, Y) :- parent(X, Y), male(X).
    mother(X, Y) :- parent(X, Y), female(X).
    grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

    ?- father(john, mary).
    + true.

    ?- mother(mary, alice).
    + true.

    ?- grandparent(john, alice).
    + true.
}
