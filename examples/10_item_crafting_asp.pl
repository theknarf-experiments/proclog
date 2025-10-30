% Example 10: Item Crafting with ASP
%
% This example demonstrates procedural item crafting using ASP:
% - Choice rules for selecting crafting materials and recipes
% - Compatibility constraints

% Available base materials
material(iron_ore).
material(wood).
material(crystal).

% Possible item types
item_type(sword).
item_type(staff).

% Recipe requirements (what materials work for what items)
recipe_compatible(sword, iron_ore).
recipe_compatible(sword, wood).
recipe_compatible(staff, wood).
recipe_compatible(staff, crystal).

% Choose exactly 1 item type to craft
1 { crafting_item(T) : item_type(T) } 1.

% Choose 2 materials for crafting
2 { used_material(M) : material(M) } 2.

% Constraint: all chosen materials must be compatible with the item type
:- crafting_item(T), used_material(M), not recipe_compatible(T, M).

% Constraint: magical items need crystal
:- crafting_item(staff), not used_material(crystal).

#test "basic sword crafting" {
    % Craft a sword with iron and wood
    1 { crafting_item(sword) } 1.
    2 { used_material(iron_ore); used_material(wood) } 2.

    recipe_compatible(sword, iron_ore).
    recipe_compatible(sword, wood).

    % Constraint: materials compatible with sword
    :- crafting_item(T), used_material(M), not recipe_compatible(T, M).

    ?- crafting_item(sword).
    + true.

    ?- used_material(iron_ore).
    + true.

    ?- used_material(wood).
    + true.
}

#test "invalid material combination" {
    % Try to craft sword with incompatible material (crystal)
    1 { crafting_item(sword) } 1.
    2 { used_material(iron_ore); used_material(crystal) } 2.

    recipe_compatible(sword, iron_ore).
    recipe_compatible(sword, wood).

    % Constraint: all materials must be compatible
    :- crafting_item(T), used_material(M), not recipe_compatible(T, M).

    % Since the constraint eliminates all answer sets, these queries should fail
    ?- crafting_item(sword).
    - true.

    ?- used_material(iron_ore).
    - true.

    ?- used_material(crystal).
    - true.
}

#test "valid staff crafting" {
    % Staff with wood and crystal
    1 { crafting_item(staff) } 1.
    2 { used_material(wood); used_material(crystal) } 2.

    recipe_compatible(staff, wood).
    recipe_compatible(staff, crystal).

    % Constraint: materials compatible
    :- crafting_item(T), used_material(M), not recipe_compatible(T, M).

    % Constraint: magical items need crystal
    :- crafting_item(staff), not used_material(crystal).

    ?- crafting_item(staff).
    + true.

    ?- used_material(wood).
    + true.

    ?- used_material(crystal).
    + true.
}

#test "invalid staff without crystal" {
    % Staff without crystal should fail
    1 { crafting_item(staff) } 1.
    2 { used_material(wood); used_material(iron_ore) } 2.

    recipe_compatible(staff, wood).
    recipe_compatible(staff, crystal).

    % Constraint: materials compatible
    :- crafting_item(T), used_material(M), not recipe_compatible(T, M).

    % Constraint: magical items need crystal
    :- crafting_item(staff), not used_material(crystal).

    % Since the constraint eliminates all answer sets, these queries should fail
    ?- crafting_item(staff).
    - true.

    ?- used_material(wood).
    - true.

    ?- used_material(iron_ore).
    - true.
}