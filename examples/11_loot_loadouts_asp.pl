% Example 11: Loot loadout sampling with ASP
%
% Highlights:
% - Choice rules for weapon, armor, and optional trinket slots.
% - Compatibility enforced via allowed pairings.
% - Value threshold filters out weak combinations.
%
% Try it out:
%   cargo run -q -- run --sample 3 examples/11_loot_loadouts_asp.pl

% weapon(Name, Value)
weapon(sword, 3).
weapon(axe, 4).
weapon(wand, 5).
weapon(spear, 3).

% armor(Name, Value)
armor(leather, 2).
armor(chain, 3).
armor(plate, 4).
armor(robe, 2).

% allowed_pair(Weapon, Armor)
allowed_pair(sword, chain).
allowed_pair(sword, leather).
allowed_pair(axe, chain).
allowed_pair(axe, plate).
allowed_pair(wand, robe).
allowed_pair(spear, leather).

% optional trinkets
trinket(ruby_ring, power).
trinket(feather_charm, agile).
trinket(moonstone, focus).

1 { weapon_choice(W) : weapon(W, _) } 1.
1 { armor_choice(A) : armor(A, _) } 1.
0 { trinket_choice(T) : trinket(T, _) } 1.

% Valid combinations only
:- weapon_choice(W), armor_choice(A), not allowed_pair(W, A).

% Simple synergy rules
:- weapon_choice(spear), armor_choice(plate).
:- trinket_choice(ruby_ring), weapon_choice(wand).

% Value requirement
loadout_value(V) :-
    weapon_choice(W), armor_choice(A),
    weapon(W, WV), armor(A, AV),
    V = WV + AV.
:- loadout_value(V), V < 6.

% Summaries for easy inspection
summary_weapon(W, Value) :- weapon_choice(W), weapon(W, Value).
summary_armor(A, Value) :- armor_choice(A), armor(A, Value).
summary_trinket(T, Theme) :- trinket_choice(T), trinket(T, Theme).
trinket_selected :- trinket_choice(_).
summary_trinket(none, none) :- not trinket_selected.
