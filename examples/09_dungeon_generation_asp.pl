% Example 9: Dungeon Generation with ASP
%
% This example demonstrates procedural dungeon generation using ASP:
% - Choice rules for selecting rooms
% - Connectivity constraints

#const grid_size = 3.

% Define grid cells
cell(1, 1). cell(1, 2). cell(1, 3).
cell(2, 1). cell(2, 2). cell(2, 3).
cell(3, 1). cell(3, 2). cell(3, 3).

% Choose 2-3 rooms
2 { room(X, Y) : cell(X, Y) } 3.

% Adjacent cells (4-directional)
adjacent(X1, Y, X2, Y) :- cell(X1, Y), cell(X2, Y), X2 = X1 + 1.
adjacent(X1, Y, X2, Y) :- cell(X1, Y), cell(X2, Y), X1 = X2 + 1.
adjacent(X, Y1, X, Y2) :- cell(X, Y1), cell(X, Y2), Y2 = Y1 + 1.
adjacent(X, Y1, X, Y2) :- cell(X, Y1), cell(X, Y2), Y1 = Y2 + 1.

% Connected rooms
connected(X1, Y1, X2, Y2) :- room(X1, Y1), room(X2, Y2), adjacent(X1, Y1, X2, Y2).

% Choose exactly 1 entrance
1 { entrance(X, Y) : room(X, Y) } 1.

% Choose exactly 1 exit
1 { exit_room(X, Y) : room(X, Y) } 1.

% Reachability from entrance
reachable(X, Y) :- entrance(X, Y).
reachable(X2, Y2) :- reachable(X1, Y1), connected(X1, Y1, X2, Y2).

% Constraints: all rooms must be reachable from entrance
:- room(X, Y), not reachable(X, Y).

% Constraints: exit must be reachable
:- exit_room(X, Y), not reachable(X, Y).

#test "simple connected dungeon" {
    % Force a simple connected dungeon
    room(1, 1).
    room(2, 1).
    room(3, 1).

    1 { entrance(1, 1) } 1.
    1 { exit_room(3, 1) } 1.

    % Adjacent cells
    adjacent(X1, Y, X2, Y) :- X2 = X1 + 1, Y = 1.
    adjacent(X1, Y, X2, Y) :- X1 = X2 + 1, Y = 1.

    connected(X1, Y1, X2, Y2) :- room(X1, Y1), room(X2, Y2), adjacent(X1, Y1, X2, Y2).

    reachable(X, Y) :- entrance(X, Y).
    reachable(X2, Y2) :- reachable(X1, Y1), connected(X1, Y1, X2, Y2).

    % All rooms reachable
    :- room(X, Y), not reachable(X, Y).

    % Exit reachable
    :- exit_room(X, Y), not reachable(X, Y).

    ?- room(1, 1).
    + true.

    ?- room(2, 1).
    + true.

    ?- room(3, 1).
    + true.

    ?- entrance(1, 1).
    + true.

    ?- exit_room(3, 1).
    + true.

    ?- reachable(1, 1).
    + true.

    ?- reachable(2, 1).
    + true.

    ?- reachable(3, 1).
    + true.
}