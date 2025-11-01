% Example 12: Party formation sampling with ASP
%
% Highlights:
% - Select 3 heroes subject to role balance rules.
% - Enforces one front-liner, at least one support role, and budget limits.
% - Shows how small arithmetic constraints interact with choice rules.
%
% Try it out:
%   cargo run -q -- run --sample 3 examples/12_party_formation_asp.pl

% hero(Name, Role, Position, Cost)
hero(alric, striker, front, 4).
hero(seren, mage, back, 5).
hero(bran, rogue, flex, 3).
hero(daria, healer, back, 4).
hero(vyra, ranger, flex, 4).
hero(korin, guardian, front, 5).

hero_index(alric, 1).
hero_index(seren, 2).
hero_index(bran, 3).
hero_index(daria, 4).
hero_index(vyra, 5).
hero_index(korin, 6).

% Pick exactly three heroes
3 { member(H) : hero(H, _, _, _) } 3.

% Derived facts
role_taken(R) :- member(H), hero(H, R, _, _).
position_taken(P) :- member(H), hero(H, _, P, _).
frontliner(H) :- hero(H, _, front, _).
frontliner(H) :- hero(H, _, flex, _).
backliner(H) :- hero(H, _, back, _).

% At least one front-capable hero
:- not has_frontliner.
has_frontliner :- member(H), frontliner(H).

% At least one backliner for support
:- not has_backliner.
has_backliner :- member(H), backliner(H).

% Require a support role (healer or ranger)
support_role(healer).
support_role(ranger).
:- not support_present.
support_present :- member(H), hero(H, Role, _, _), support_role(Role).

% Avoid bringing both rogue and mage together (stealth vs burst conflict)
:- member(bran), member(seren).

% Budget: total cost must be <= 12
budget_sum(Total) :-
    member(H1), member(H2), member(H3),
    hero_index(H1, I1), hero_index(H2, I2), hero_index(H3, I3),
    I1 < I2, I2 < I3,
    hero(H1, _, _, C1), hero(H2, _, _, C2), hero(H3, _, _, C3),
    Temp = C1 + C2,
    Total = Temp + C3.
:- budget_sum(Total), Total > 12.

% Summaries for easy inspection
summary_member(H, Role, Position, Cost) :-
    member(H), hero(H, Role, Position, Cost).
summary_roles(R) :- role_taken(R).
summary_positions(P) :- position_taken(P).
