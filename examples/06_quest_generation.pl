% Example 6: Quest Generation
%
% This example demonstrates quest generation with:
% - Quest prerequisites and chains
% - Reward systems
% - Quest difficulty based on level requirements
% - NPC and location relationships

% NPCs in the game world
npc(village_elder).
npc(blacksmith).
npc(merchant).
npc(wizard).
npc(guard_captain).

% Locations
location(village).
location(forest).
location(cave).
location(mountain).
location(castle).

% NPC locations
npc_at(village_elder, village).
npc_at(blacksmith, village).
npc_at(merchant, village).
npc_at(wizard, castle).
npc_at(guard_captain, castle).

% Quest definitions
quest(gather_herbs).
quest(defeat_wolves).
quest(find_artifact).
quest(rescue_merchant).
quest(defeat_dragon).

% Quest givers
quest_giver(gather_herbs, village_elder).
quest_giver(defeat_wolves, guard_captain).
quest_giver(find_artifact, wizard).
quest_giver(rescue_merchant, blacksmith).
quest_giver(defeat_dragon, guard_captain).

% Quest requirements (level)
quest_level(gather_herbs, 1).
quest_level(defeat_wolves, 5).
quest_level(find_artifact, 10).
quest_level(rescue_merchant, 8).
quest_level(defeat_dragon, 20).

% Player levels used to evaluate quest availability
player_level(1).
player_level(5).
player_level(8).
player_level(10).
player_level(20).

% Quest prerequisites
requires_quest(defeat_wolves, gather_herbs).
requires_quest(find_artifact, defeat_wolves).
requires_quest(defeat_dragon, find_artifact).

% Quest locations
quest_location(gather_herbs, forest).
quest_location(defeat_wolves, forest).
quest_location(find_artifact, cave).
quest_location(rescue_merchant, mountain).
quest_location(defeat_dragon, mountain).

% Quest rewards (gold)
quest_reward(gather_herbs, 50).
quest_reward(defeat_wolves, 150).
quest_reward(find_artifact, 500).
quest_reward(rescue_merchant, 300).
quest_reward(defeat_dragon, 2000).

% Rules for quest availability
available_at_level(Quest, Level) :-
    quest(Quest),
    quest_level(Quest, Required),
    player_level(Level),
    Level >= Required.

% Transitive quest requirements
requires_completed(Quest, Prereq) :- requires_quest(Quest, Prereq).
requires_completed(Quest, Prereq) :-
    requires_quest(Quest, Intermediate),
    requires_completed(Intermediate, Prereq).

% Quest chain (all quests that lead to a final quest)
quest_chain_for(FinalQuest, Quest) :-
    requires_completed(FinalQuest, Quest).

#test "npc locations" {
    npc(village_elder).
    npc(wizard).
    location(village).
    location(castle).
    npc_at(village_elder, village).
    npc_at(wizard, castle).

    ?- npc_at(village_elder, village).
    + true.

    ?- npc_at(wizard, castle).
    + true.

    ?- npc_at(village_elder, castle).
    - true.
}

#test "quest level requirements" {
    quest(gather_herbs).
    quest(defeat_dragon).
    quest_level(gather_herbs, 1).
    quest_level(defeat_dragon, 20).

    available_at_level(Quest, Level) :-
        quest(Quest),
        quest_level(Quest, Req),
        player_level(Level),
        Level >= Req.

    % Level 1 can do gather_herbs
    ?- available_at_level(gather_herbs, 1).
    + true.

    % Level 1 cannot do defeat_dragon
    ?- available_at_level(defeat_dragon, 1).
    - true.

    % Level 20 can do both
    ?- available_at_level(gather_herbs, 20).
    + true.

    ?- available_at_level(defeat_dragon, 20).
    + true.
}

#test "direct quest prerequisites" {
    quest(gather_herbs).
    quest(defeat_wolves).
    quest(find_artifact).

    requires_quest(defeat_wolves, gather_herbs).
    requires_quest(find_artifact, defeat_wolves).

    ?- requires_quest(defeat_wolves, gather_herbs).
    + true.

    ?- requires_quest(find_artifact, defeat_wolves).
    + true.

    ?- requires_quest(find_artifact, gather_herbs).
    - true.
}

#test "transitive quest prerequisites" {
    quest(gather_herbs).
    quest(defeat_wolves).
    quest(find_artifact).

    requires_quest(defeat_wolves, gather_herbs).
    requires_quest(find_artifact, defeat_wolves).

    requires_completed(Quest, Prereq) :- requires_quest(Quest, Prereq).
    requires_completed(Quest, Prereq) :-
        requires_quest(Quest, Intermediate),
        requires_completed(Intermediate, Prereq).

    % Direct requirements
    ?- requires_completed(defeat_wolves, gather_herbs).
    + true.

    % Transitive requirements
    ?- requires_completed(find_artifact, gather_herbs).
    + true.

    ?- requires_completed(find_artifact, defeat_wolves).
    + true.
}

#test "quest chain to dragon" {
    quest(gather_herbs).
    quest(defeat_wolves).
    quest(find_artifact).
    quest(defeat_dragon).

    requires_quest(defeat_wolves, gather_herbs).
    requires_quest(find_artifact, defeat_wolves).
    requires_quest(defeat_dragon, find_artifact).

    requires_completed(Quest, Prereq) :- requires_quest(Quest, Prereq).
    requires_completed(Quest, Prereq) :-
        requires_quest(Quest, Intermediate),
        requires_completed(Intermediate, Prereq).

    % Dragon requires the entire chain
    ?- requires_completed(defeat_dragon, find_artifact).
    + true.

    ?- requires_completed(defeat_dragon, defeat_wolves).
    + true.

    ?- requires_completed(defeat_dragon, gather_herbs).
    + true.
}

#test "quest rewards" {
    quest(gather_herbs).
    quest(defeat_dragon).
    quest_reward(gather_herbs, 50).
    quest_reward(defeat_dragon, 2000).

    high_reward(Quest) :- quest_reward(Quest, Gold), Gold >= 1000.

    ?- quest_reward(gather_herbs, 50).
    + true.

    ?- quest_reward(defeat_dragon, 2000).
    + true.

    % Dragon quest has high reward
    ?- high_reward(defeat_dragon).
    + true.

    % Gather herbs does not
    ?- high_reward(gather_herbs).
    - true.
}

#test "quest giver locations" {
    npc(village_elder).
    npc(wizard).
    location(village).
    location(castle).
    npc_at(village_elder, village).
    npc_at(wizard, castle).

    quest(gather_herbs).
    quest(find_artifact).
    quest_giver(gather_herbs, village_elder).
    quest_giver(find_artifact, wizard).

    quest_available_at(Quest, Location) :-
        quest_giver(Quest, NPC),
        npc_at(NPC, Location).

    % Gather herbs available in village
    ?- quest_available_at(gather_herbs, village).
    + true.

    % Find artifact available in castle
    ?- quest_available_at(find_artifact, castle).
    + true.

    % Not at wrong locations
    ?- quest_available_at(gather_herbs, castle).
    - true.
}

#test "complex quest system" {
    quest(q1).
    quest(q2).
    quest(q3).
    quest_level(q1, 1).
    quest_level(q2, 5).
    quest_level(q3, 10).
    quest_reward(q1, 100).
    quest_reward(q2, 500).
    quest_reward(q3, 1000).

    reward_amount(100).
    reward_amount(500).
    reward_amount(600).
    reward_amount(1000).
    reward_amount(1100).
    reward_amount(1500).

    requires_quest(q2, q1).
    requires_quest(q3, q2).

    available_at_level(Quest, Level) :-
        quest(Quest),
        quest_level(Quest, Req),
        player_level(Level),
        Level >= Req.
    requires_completed(Quest, Prereq) :- requires_quest(Quest, Prereq).
    requires_completed(Quest, Prereq) :-
        requires_quest(Quest, Int),
        requires_completed(Int, Prereq).

    total_rewards_for_chain(Quest, Total) :-
        quest_reward(Quest, R1),
        requires_completed(Quest, Prereq1),
        quest_reward(Prereq1, R2),
        reward_amount(Total),
        Total = R1 + R2.

    % Level checks
    ?- available_at_level(q1, 1).
    + true.

    ?- available_at_level(q3, 5).
    - true.

    % Chain checks
    ?- requires_completed(q3, q1).
    + true.
}
