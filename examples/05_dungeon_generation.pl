% Example 5: Dungeon Generation
%
% This example demonstrates procedural dungeon generation using:
% - Grid-based room placement
% - Connectivity rules
% - Room type assignment
% - Treasure and enemy placement

#const grid_size = 5.
#const min_rooms = 4.
#const max_rooms = 8.

% Define grid cells
cell(1, 1). cell(1, 2). cell(1, 3). cell(1, 4). cell(1, 5).
cell(2, 1). cell(2, 2). cell(2, 3). cell(2, 4). cell(2, 5).
cell(3, 1). cell(3, 2). cell(3, 3). cell(3, 4). cell(3, 5).
cell(4, 1). cell(4, 2). cell(4, 3). cell(4, 4). cell(4, 5).
cell(5, 1). cell(5, 2). cell(5, 3). cell(5, 4). cell(5, 5).

% Example dungeon layout
room(1, 1).
room(2, 1).
room(3, 1).
room(3, 2).
room(3, 3).
room(4, 3).

% Room types
room_type(1, 1, entrance).
room_type(2, 1, corridor).
room_type(3, 1, treasure_room).
room_type(3, 2, corridor).
room_type(3, 3, boss_room).
room_type(4, 3, exit).

% Adjacent cells (4-directional)
adjacent(X1, Y, X2, Y) :- cell(X1, Y), cell(X2, Y), X2 = X1 + 1.
adjacent(X1, Y, X2, Y) :- cell(X1, Y), cell(X2, Y), X1 = X2 + 1.
adjacent(X, Y1, X, Y2) :- cell(X, Y1), cell(X, Y2), Y2 = Y1 + 1.
adjacent(X, Y1, X, Y2) :- cell(X, Y1), cell(X, Y2), Y1 = Y2 + 1.

% Connected rooms (rooms that are adjacent)
connected(X1, Y1, X2, Y2) :- room(X1, Y1), room(X2, Y2), adjacent(X1, Y1, X2, Y2).

% Reachability from entrance
reachable(X, Y) :- room_type(X, Y, entrance).
reachable(X2, Y2) :- reachable(X1, Y1), connected(X1, Y1, X2, Y2).

% Special room rules
has_entrance(X, Y) :- room_type(X, Y, entrance).
has_exit(X, Y) :- room_type(X, Y, exit).
has_treasure(X, Y) :- room_type(X, Y, treasure_room).
has_boss(X, Y) :- room_type(X, Y, boss_room).

#test "basic room placement" {
    room(1, 1).
    room(2, 1).
    room(2, 2).

    ?- room(1, 1).
    + true.

    ?- room(2, 1).
    + true.

    ?- room(2, 2).
    + true.

    ?- room(3, 3).
    - true.
}

#test "room connectivity" {
    room(1, 1).
    room(2, 1).
    room(3, 1).

    connected(X1, Y1, X2, Y2) :- room(X1, Y1), room(X2, Y2), X2 = X1 + 1, Y1 = Y2.
    connected(X1, Y1, X2, Y2) :- room(X1, Y1), room(X2, Y2), X1 = X2 + 1, Y1 = Y2.

    % Rooms 1,1 and 2,1 are connected
    ?- connected(1, 1, 2, 1).
    + true.

    % Rooms 2,1 and 3,1 are connected
    ?- connected(2, 1, 3, 1).
    + true.

    % Rooms 1,1 and 3,1 are not directly connected
    ?- connected(1, 1, 3, 1).
    - true.
}

#test "room type assignment" {
    room(1, 1).
    room(2, 1).
    room(3, 1).

    room_type(1, 1, entrance).
    room_type(2, 1, corridor).
    room_type(3, 1, treasure_room).

    has_entrance(X, Y) :- room_type(X, Y, entrance).
    has_treasure(X, Y) :- room_type(X, Y, treasure_room).

    ?- has_entrance(1, 1).
    + true.

    ?- has_treasure(3, 1).
    + true.

    ?- has_entrance(2, 1).
    - true.
}

#test "linear dungeon reachability" {
    room(1, 1).
    room(2, 1).
    room(3, 1).
    room(4, 1).

    room_type(1, 1, entrance).
    room_type(2, 1, corridor).
    room_type(3, 1, treasure_room).
    room_type(4, 1, exit).

    connected(X1, Y1, X2, Y2) :- room(X1, Y1), room(X2, Y2), X2 = X1 + 1, Y1 = Y2.
    connected(X1, Y1, X2, Y2) :- room(X1, Y1), room(X2, Y2), X1 = X2 + 1, Y1 = Y2.

    reachable(X, Y) :- room_type(X, Y, entrance).
    reachable(X2, Y2) :- reachable(X1, Y1), connected(X1, Y1, X2, Y2).

    % All rooms should be reachable
    ?- reachable(1, 1).
    + true.

    ?- reachable(2, 1).
    + true.

    ?- reachable(3, 1).
    + true.

    ?- reachable(4, 1).
    + true.
}

#test "branching dungeon" {
    room(2, 2).
    room(2, 3).
    room(1, 3).
    room(3, 3).
    room(2, 4).

    room_type(2, 2, entrance).
    room_type(2, 3, corridor).
    room_type(1, 3, treasure_room).
    room_type(3, 3, treasure_room).
    room_type(2, 4, boss_room).

    % Horizontal connections
    connected(X1, Y, X2, Y) :- room(X1, Y), room(X2, Y), X2 = X1 + 1.
    connected(X1, Y, X2, Y) :- room(X1, Y), room(X2, Y), X1 = X2 + 1.
    % Vertical connections
    connected(X, Y1, X, Y2) :- room(X, Y1), room(X, Y2), Y2 = Y1 + 1.
    connected(X, Y1, X, Y2) :- room(X, Y1), room(X, Y2), Y1 = Y2 + 1.

    reachable(X, Y) :- room_type(X, Y, entrance).
    reachable(X2, Y2) :- reachable(X1, Y1), connected(X1, Y1, X2, Y2).

    % All rooms reachable
    ?- reachable(2, 2).
    + true.

    ?- reachable(2, 3).
    + true.

    ?- reachable(1, 3).
    + true.

    ?- reachable(3, 3).
    + true.

    ?- reachable(2, 4).
    + true.
}

#test "disconnected rooms" {
    room(1, 1).
    room(2, 1).
    room(4, 4).
    room(5, 4).

    room_type(1, 1, entrance).
    room_type(2, 1, corridor).
    room_type(4, 4, treasure_room).
    room_type(5, 4, exit).

    connected(X1, Y1, X2, Y2) :- room(X1, Y1), room(X2, Y2), X2 = X1 + 1, Y1 = Y2.
    connected(X1, Y1, X2, Y2) :- room(X1, Y1), room(X2, Y2), X1 = X2 + 1, Y1 = Y2.
    connected(X, Y1, X, Y2) :- room(X, Y1), room(X, Y2), Y2 = Y1 + 1.
    connected(X, Y1, X, Y2) :- room(X, Y1), room(X, Y2), Y1 = Y2 + 1.

    reachable(X, Y) :- room_type(X, Y, entrance).
    reachable(X2, Y2) :- reachable(X1, Y1), connected(X1, Y1, X2, Y2).

    % First group reachable
    ?- reachable(1, 1).
    + true.

    ?- reachable(2, 1).
    + true.

    % Second group not reachable from entrance
    ?- reachable(4, 4).
    - true.

    ?- reachable(5, 4).
    - true.
}

#test "l-shaped dungeon" {
    room(1, 1).
    room(2, 1).
    room(3, 1).
    room(3, 2).
    room(3, 3).

    room_type(1, 1, entrance).
    room_type(3, 3, exit).

    connected(X1, Y1, X2, Y2) :- room(X1, Y1), room(X2, Y2), X2 = X1 + 1, Y1 = Y2.
    connected(X1, Y1, X2, Y2) :- room(X1, Y1), room(X2, Y2), X1 = X2 + 1, Y1 = Y2.
    connected(X, Y1, X, Y2) :- room(X, Y1), room(X, Y2), Y2 = Y1 + 1.
    connected(X, Y1, X, Y2) :- room(X, Y1), room(X, Y2), Y1 = Y2 + 1.

    reachable(X, Y) :- room_type(X, Y, entrance).
    reachable(X2, Y2) :- reachable(X1, Y1), connected(X1, Y1, X2, Y2).

    has_exit(X, Y) :- room_type(X, Y, exit).
    exit_reachable :- has_exit(X, Y), reachable(X, Y).

    % All rooms reachable
    ?- reachable(1, 1).
    + true.

    ?- reachable(2, 1).
    + true.

    ?- reachable(3, 1).
    + true.

    ?- reachable(3, 2).
    + true.

    ?- reachable(3, 3).
    + true.

    % Exit is reachable
    ?- exit_reachable.
    + true.
}
