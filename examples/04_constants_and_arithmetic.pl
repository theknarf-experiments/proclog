% Example 4: Constants and Arithmetic
%
% This example demonstrates:
% - Defining constants with #const
% - Using constants in facts and rules
% - Arithmetic comparisons (>, <, >=, <=, =)
% - Combining constants with rules

#const max_health = 100.
#const min_health = 0.
#const critical_health = 20.
#const healthy_threshold = 50.

#const max_level = 50.
#const beginner_level = 10.
#const expert_level = 40.

#const max_gold = 99999.
#const wealthy_threshold = 1000.

% Character stats
character(warrior, 85, 15, 500).
character(mage, 40, 45, 1200).
character(rogue, 60, 20, 800).
character(healer, 70, 50, 200).

% character(Name, Health, Level, Gold)

% Status rules
healthy(Name) :- character(Name, Health, _, _), Health >= healthy_threshold.
critical(Name) :- character(Name, Health, _, _), Health <= critical_health.
wealthy(Name) :- character(Name, _, _, Gold), Gold >= wealthy_threshold.
beginner(Name) :- character(Name, _, Level, _), Level <= beginner_level.
expert(Name) :- character(Name, _, Level, _), Level >= expert_level.

#test "constant substitution" {
    #const max_value = 100.

    value(max_value).
    limit(max_value).

    ?- value(100).
    + true.

    ?- limit(100).
    + true.

    ?- value(50).
    - true.
}

#test "health status checks" {
    #const healthy_threshold = 50.
    #const critical_health = 20.

    character(warrior, 85, 15, 500).
    character(mage, 15, 45, 1200).
    character(rogue, 60, 20, 800).

    healthy(Name) :- character(Name, Health, _, _), Health >= healthy_threshold.
    critical(Name) :- character(Name, Health, _, _), Health <= critical_health.

    % Warrior is healthy
    ?- healthy(warrior).
    + true.

    % Mage is critical
    ?- critical(mage).
    + true.

    % Rogue is healthy
    ?- healthy(rogue).
    + true.

    % Warrior is not critical
    ?- critical(warrior).
    - true.
}

#test "level-based classifications" {
    #const beginner_level = 10.
    #const expert_level = 40.

    character(alice, 100, 5, 100).
    character(bob, 100, 25, 500).
    character(charlie, 100, 45, 1000).

    beginner(Name) :- character(Name, _, Level, _), Level <= beginner_level.
    expert(Name) :- character(Name, _, Level, _), Level >= expert_level.

    % Alice is beginner
    ?- beginner(alice).
    + true.

    % Charlie is expert
    ?- expert(charlie).
    + true.

    % Bob is neither
    ?- beginner(bob).
    - true.

    ?- expert(bob).
    - true.
}

#test "wealth calculations" {
    #const wealthy_threshold = 1000.
    #const poor_threshold = 100.

    character(rich_guy, 100, 20, 5000).
    character(middle_class, 100, 20, 500).
    character(poor_guy, 100, 20, 50).

    wealthy(Name) :- character(Name, _, _, Gold), Gold >= wealthy_threshold.
    poor(Name) :- character(Name, _, _, Gold), Gold <= poor_threshold.

    ?- wealthy(rich_guy).
    + true.

    ?- poor(poor_guy).
    + true.

    ?- wealthy(middle_class).
    - true.

    ?- poor(middle_class).
    - true.
}

#test "complex character queries" {
    #const healthy_threshold = 50.
    #const wealthy_threshold = 1000.
    #const expert_level = 40.

    character(warrior, 85, 45, 500).
    character(mage, 40, 48, 1200).
    character(rogue, 60, 35, 800).

    healthy(Name) :- character(Name, Health, _, _), Health >= healthy_threshold.
    wealthy(Name) :- character(Name, _, _, Gold), Gold >= wealthy_threshold.
    expert(Name) :- character(Name, _, Level, _), Level >= expert_level.

    % Warrior: healthy and expert, but not wealthy
    ?- healthy(warrior).
    + true.

    ?- expert(warrior).
    + true.

    ?- wealthy(warrior).
    - true.

    % Mage: wealthy and expert, but not healthy
    ?- wealthy(mage).
    + true.

    ?- expert(mage).
    + true.

    ?- healthy(mage).
    - true.

    % Rogue: healthy but not expert or wealthy
    ?- healthy(rogue).
    + true.

    ?- expert(rogue).
    - true.

    ?- wealthy(rogue).
    - true.
}

#test "range comparisons" {
    #const min_value = 10.
    #const max_value = 90.

    value(a, 5).
    value(b, 50).
    value(c, 95).

    in_range(Name) :- value(Name, V), V >= min_value, V <= max_value.
    below_range(Name) :- value(Name, V), V < min_value.
    above_range(Name) :- value(Name, V), V > max_value.

    % b is in range
    ?- in_range(b).
    + true.

    % a is below
    ?- below_range(a).
    + true.

    % c is above
    ?- above_range(c).
    + true.

    % Negative checks
    ?- in_range(a).
    - true.

    ?- above_range(b).
    - true.
}
