% Procedural Dungeon Generator (21 layouts from 64 possibilities!)
room(entrance). room(armory). room(library). room(treasury).

% ASP Choice: 6 doors × 2 states = 2^6 = 64 combinations
{ door(entrance, armory) }.
{ door(entrance, library) }.
{ door(entrance, treasury) }.
{ door(armory, library) }.
{ door(armory, treasury) }.
{ door(library, treasury) }.

% Transitive reachability
reachable(entrance).
reachable(Y) :- reachable(X), door(X, Y).

% Constraint: all rooms must be reachable
:- room(R), not reachable(R).

#test "every layout is valid" {
  ?- reachable(treasury).
  + true.
}
