% Example 8: Character Creation with ASP
%
% This example demonstrates procedural character creation using ASP:
% - Choice rules for selecting character attributes
% - Cardinality constraints on attribute combinations

% Available races
race(human).
race(elf).

% Available classes
class(warrior).
class(mage).

#test "basic character creation" {
    % Choose human warrior
    1 { character_race(human) } 1.
    1 { character_class(warrior) } 1.

    ?- character_race(human).
    + true.

    ?- character_class(warrior).
    + true.
}

#test "constraint violation - elf warrior" {
    % This should produce no valid answer sets due to constraint
    1 { character_race(elf) } 1.
    1 { character_class(warrior) } 1.

    % Constraint: elves cannot be warriors
    :- character_race(elf), character_class(warrior).

    % Since the constraint eliminates all answer sets, these queries should fail
    ?- character_race(elf).
    - true.

    ?- character_class(warrior).
    - true.
}