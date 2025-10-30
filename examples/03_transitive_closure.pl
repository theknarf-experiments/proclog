% Example 3: Transitive Closure
%
% This example demonstrates:
% - Recursive rules
% - Computing reachability in graphs
% - Path finding

% Direct edges in a graph
edge(a, b).
edge(b, c).
edge(c, d).
edge(d, e).
edge(b, f).
edge(f, g).

% Transitive closure: path exists if there's an edge or a path through intermediate nodes
path(X, Y) :- edge(X, Y).
path(X, Z) :- path(X, Y), edge(Y, Z).

% Alternative: both directions use path
path_alt(X, Y) :- edge(X, Y).
path_alt(X, Z) :- path_alt(X, Y), path_alt(Y, Z).

#test "direct edges are paths" {
    edge(a, b).
    edge(b, c).

    path(X, Y) :- edge(X, Y).
    path(X, Z) :- path(X, Y), edge(Y, Z).

    ?- path(a, b).
    + true.

    ?- path(b, c).
    + true.
}

#test "transitive paths" {
    edge(a, b).
    edge(b, c).
    edge(c, d).

    path(X, Y) :- edge(X, Y).
    path(X, Z) :- path(X, Y), edge(Y, Z).

    % Direct paths
    ?- path(a, b).
    + true.

    % Two-hop paths
    ?- path(a, c).
    + true.

    % Three-hop paths
    ?- path(a, d).
    + true.

    % Non-existent paths
    ?- path(d, a).
    - true.

    ?- path(b, a).
    - true.
}

#test "branching paths" {
    edge(a, b).
    edge(b, c).
    edge(b, d).
    edge(c, e).
    edge(d, e).

    path(X, Y) :- edge(X, Y).
    path(X, Z) :- path(X, Y), edge(Y, Z).

    % Multiple paths from a to e (through c and through d)
    ?- path(a, e).
    + true.

    % Path through branch
    ?- path(a, d).
    + true.

    ?- path(b, e).
    + true.
}

#test "disconnected components" {
    edge(a, b).
    edge(b, c).
    edge(x, y).
    edge(y, z).

    path(X, Y) :- edge(X, Y).
    path(X, Z) :- path(X, Y), edge(Y, Z).

    % Paths within first component
    ?- path(a, c).
    + true.

    % Paths within second component
    ?- path(x, z).
    + true.

    % No paths between components
    ?- path(a, x).
    - true.

    ?- path(x, a).
    - true.

    ?- path(b, y).
    - true.
}

#test "complex graph" {
    edge(start, a).
    edge(start, b).
    edge(a, c).
    edge(b, c).
    edge(c, d).
    edge(d, end).
    edge(a, end).

    path(X, Y) :- edge(X, Y).
    path(X, Z) :- path(X, Y), edge(Y, Z).

    % Direct path
    ?- path(start, a).
    + true.

    % Short path
    ?- path(a, end).
    + true.

    % Long path
    ?- path(start, end).
    + true.

    % Through multiple nodes
    ?- path(start, d).
    + true.

    % Convergent paths (start -> a -> c and start -> b -> c)
    ?- path(start, c).
    + true.
}

#test "linear chain" {
    edge(n1, n2).
    edge(n2, n3).
    edge(n3, n4).
    edge(n4, n5).
    edge(n5, n6).

    path(X, Y) :- edge(X, Y).
    path(X, Z) :- path(X, Y), edge(Y, Z).

    % Start to each node
    ?- path(n1, n2).
    + true.

    ?- path(n1, n3).
    + true.

    ?- path(n1, n4).
    + true.

    ?- path(n1, n5).
    + true.

    ?- path(n1, n6).
    + true.

    % No reverse paths
    ?- path(n6, n1).
    - true.

    ?- path(n3, n1).
    - true.
}
