% ASP-SAT Solver Examples
%
% This file demonstrates the ASP evaluation using SAT solver backend.
% The SAT-based solver handles choice rules, negation, and constraints
% using guess-and-check with splr (pure Rust CDCL SAT solver).

% Example 1: Simple Choice Rule
% =============================
% Demonstrates basic non-determinism

item(sword).
item(shield).
item(potion).

% Choice: select items
{ selected(sword) }.
{ selected(shield) }.
{ selected(potion) }.

#test "choice_basics" {
    % Multiple answer sets possible
    ?- selected(sword).
    + selected(sword).
    - selected(sword).
}

% Example 2: Choice with Derivation
% ==================================
% Shows how choices propagate through rules

{ weapon(sword) }.
{ weapon(axe) }.

% If weapon is selected, mark as equipped
equipped(X) :- weapon(X).

% Count equipped weapons
ready :- equipped(_).

#test "choice_derivation" {
    % Can have no weapons
    ?- not ready.
    + true.

    % Can have weapons
    ?- ready.
    + true.
}

% Example 3: Negation as Failure
% ===============================
% Classic ASP negation

{ a }.
b :- not a.
c :- b.

#test "negation" {
    % If a is chosen, b and c are not derived
    ?- a, not b, not c.
    + true.

    % If a is not chosen, b and c are derived
    ?- not a, b, c.
    + true.
}

% Example 4: Constraints
% ======================
% Filter invalid answer sets

{ red }.
{ blue }.
{ green }.

% Cannot choose both red and blue
:- red, blue.

% Must choose at least one color
:- not red, not blue, not green.

#test "constraints" {
    % Red and blue together is invalid
    ?- red, blue.
    - true.

    % Must have at least one
    ?- not red, not blue, not green.
    - true.

    % Valid combinations exist
    ?- red, not blue.
    + true.
}

% Example 5: Graph Coloring
% ==========================
% Classic ASP problem

node(a).
node(b).
node(c).

edge(a, b).
edge(b, c).
edge(a, c).

color(red).
color(blue).
color(green).

% Choose one color per node
{ colored(N, C) : color(C) } :- node(N).

% Each node must have exactly one color
:- node(N), not colored(N, _).
:- node(N), colored(N, C1), colored(N, C2), C1 != C2.

% Adjacent nodes cannot have the same color
:- edge(N1, N2), colored(N1, C), colored(N2, C).

#test "graph_coloring" {
    % Valid coloring exists
    ?- colored(a, _), colored(b, _), colored(c, _).
    + true.

    % Adjacent nodes have different colors
    ?- colored(a, C1), colored(b, C2), C1 != C2.
    + true.
}

% Example 6: Planning Problem
% ============================
% Simple action planning

location(home).
location(store).
location(work).

% Choose actions
{ go(home, store) }.
{ go(store, work) }.
{ go(home, work) }.

% At work if we go there
at_work :- go(_, work).

% At store if we go there
at_store :- go(_, store).

% Cannot be in two places
:- at_work, at_store.

% If at store, we went from home
from_home :- at_store, go(home, store).

#test "planning" {
    % Can go directly to work
    ?- at_work, not at_store.
    + true.

    % Can go to store from home
    ?- at_store, from_home.
    + true.
}

% Example 7: Preference with Constraints
% =======================================
% Selecting with restrictions

meal(pasta).
meal(salad).
meal(soup).

drink(water).
drink(juice).
drink(soda).

% Choose meal and drink
{ chosen_meal(M) : meal(M) }.
{ chosen_drink(D) : drink(D) }.

% Must choose exactly one meal
:- not chosen_meal(_).
:- chosen_meal(M1), chosen_meal(M2), M1 != M2.

% Must choose exactly one drink
:- not chosen_drink(_).
:- chosen_drink(D1), chosen_drink(D2), D1 != D2.

% Pasta requires water (health constraint)
:- chosen_meal(pasta), not chosen_drink(water).

#test "preferences" {
    % Pasta forces water
    ?- chosen_meal(pasta), chosen_drink(water).
    + true.

    % Pasta with soda is invalid
    ?- chosen_meal(pasta), chosen_drink(soda).
    - true.

    % Salad can go with anything
    ?- chosen_meal(salad), chosen_drink(juice).
    + true.
}

% Example 8: Stable Model Semantics
% ==================================
% Demonstrates proper stable model computation

{ p }.
q :- not p.
r :- q.
s :- p, not q.

#test "stable_models" {
    % Two stable models

    % Model 1: p is chosen
    ?- p, not q, not r, s.
    + true.

    % Model 2: p is not chosen
    ?- not p, q, r, not s.
    + true.
}

% Example 9: Complex Derivation Chain
% ====================================
% Multiple levels of derivation

{ base }.
level1 :- base.
level2 :- level1.
level3 :- level2.
complete :- level3.

#test "derivation_chain" {
    % Full chain when base is chosen
    ?- base, level1, level2, level3, complete.
    + true.

    % Nothing derived when base not chosen
    ?- not base, not level1, not level2, not level3, not complete.
    + true.
}

% Example 10: Mutual Exclusion
% =============================
% Cannot have both

{ left }.
{ right }.

% Exactly one direction
:- left, right.
:- not left, not right.

forward :- left.
forward :- right.

#test "mutual_exclusion" {
    % Must choose one
    ?- forward.
    + true.

    % Cannot choose both
    ?- left, right.
    - true.

    % Must choose something
    ?- not left, not right.
    - true.
}
