% Count Aggregates Example
%
% This example demonstrates the use of count aggregates in ASP.
% Count aggregates allow you to count how many instantiations satisfy
% a condition and use that count in constraints or rules.
%
% Syntax: count { Variables : Conditions } Operator Value
%
% Where:
%   - Variables: one or more variables (comma-separated)
%   - Conditions: literals that the variables must satisfy
%   - Operator: =, !=, <, <=, >, >=
%   - Value: integer to compare against

% Example 1: Simple Count Constraint
% ===================================
% Count the number of items and ensure we don't exceed a maximum.

item(sword).
item(shield).
item(potion).

% Choice rule: select items
{ selected(X) : item(X) }.

% Constraint: cannot select more than 2 items
:- count { X : selected(X) } > 2.

#test "simple_count_constraint" {
    % Query should succeed - there exist answer sets
    ?- selected(X).
    + selected(sword).
    + selected(shield).
    + selected(potion).
}

% Example 2: Exact Count Requirement
% ===================================
% Require exactly a specific number of selections.

color(red).
color(blue).
color(green).

{ picked(C) : color(C) }.

% Must pick exactly 2 colors
:- count { C : picked(C) } != 2.

#test "exact_count" {
    % Any query for picked colors should work
    ?- picked(C).
    + picked(red).
    + picked(blue).
    + picked(green).
}

% Example 3: Count with Multiple Conditions
% ==========================================
% Count only items that satisfy multiple criteria.

weapon(sword).
weapon(axe).
weapon(bow).

heavy(sword).
heavy(axe).

{ carry(W) : weapon(W) }.

% Cannot carry more than 1 heavy weapon
:- count { W : carry(W), heavy(W) } > 1.

#test "count_with_conditions" {
    % Can carry a heavy weapon and a light weapon
    ?- carry(sword), carry(bow).
    + true.

    % Cannot carry two heavy weapons
    ?- carry(sword), carry(axe).
    - true.
}

% Example 4: Minimum Count Requirement
% =====================================
% Ensure at least a certain number of items are selected.

member(alice).
member(bob).
member(charlie).

role(alice, healer).
role(bob, warrior).
role(charlie, mage).

{ in_team(M) : member(M) }.

% Team must have at least 2 members
:- count { M : in_team(M) } < 2.

% Team must have at most 3 members
:- count { M : in_team(M) } > 3.

#test "team_size_bounds" {
    % Valid 2-member team
    ?- in_team(alice), in_team(bob), not in_team(charlie).
    + true.

    % Single member team is invalid
    ?- in_team(alice), not in_team(bob), not in_team(charlie).
    - true.
}

% Example 5: Count in Rule Bodies
% ================================
% Use count aggregates to derive new facts based on counts.

student(alice).
student(bob).
student(charlie).

passed(alice).
passed(charlie).

% A class is successful if at least 2 students passed
class_successful :- count { S : student(S), passed(S) } >= 2.

#test "count_in_rules" {
    % The class should be successful
    ?- class_successful.
    + true.
}

% Example 6: Multiple Count Constraints
% ======================================
% Use multiple count constraints together.

skill(coding).
skill(design).
skill(management).

% Choose skills
{ has_skill(S) : skill(S) }.

% Must have at least 1 skill
:- count { S : has_skill(S) } = 0.

% Cannot have all 3 skills
:- count { S : has_skill(S) } = 3.

#test "multiple_constraints" {
    % Having 1 skill is valid
    ?- has_skill(coding), not has_skill(design), not has_skill(management).
    + true.

    % Having 2 skills is valid
    ?- has_skill(coding), has_skill(design), not has_skill(management).
    + true.

    % Having all 3 skills violates constraint
    ?- has_skill(coding), has_skill(design), has_skill(management).
    - true.

    % Having 0 skills violates constraint
    ?- not has_skill(coding), not has_skill(design), not has_skill(management).
    - true.
}
