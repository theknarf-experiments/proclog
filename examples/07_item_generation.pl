% Example 7: Item Generation
%
% This example demonstrates item generation with:
% - Item types and rarity
% - Stat calculations based on item level
% - Compatible equipment slots
% - Item requirements (level, class)

#const common_value_mult = 10.
#const rare_value_mult = 50.
#const epic_value_mult = 200.
#const legendary_value_mult = 1000.

% Item rarities
rarity(common).
rarity(rare).
rarity(epic).
rarity(legendary).

% Equipment slots
slot(weapon).
slot(armor).
slot(accessory).
slot(consumable).

% Item base types
item_type(sword, weapon).
item_type(axe, weapon).
item_type(bow, weapon).
item_type(plate_armor, armor).
item_type(leather_armor, armor).
item_type(robe, armor).
item_type(ring, accessory).
item_type(amulet, accessory).
item_type(potion, consumable).

% Generated items with stats
% item(Name, Type, Rarity, Level, Power)
item(iron_sword, sword, common, 1, 10).
item(steel_sword, sword, rare, 5, 50).
item(flame_sword, sword, epic, 10, 150).
item(excalibur, sword, legendary, 20, 500).

item(basic_bow, bow, common, 1, 8).
item(hunters_bow, bow, rare, 5, 45).

item(cloth_robe, robe, common, 1, 5).
item(archmage_robe, robe, legendary, 15, 300).

item(health_potion, potion, common, 1, 50).
item(mana_potion, potion, common, 1, 40).

% Class restrictions
class(warrior).
class(mage).
class(rogue).

can_use(warrior, sword).
can_use(warrior, axe).
can_use(warrior, plate_armor).

can_use(mage, bow).
can_use(mage, robe).

can_use(rogue, sword).
can_use(rogue, bow).
can_use(rogue, leather_armor).

% All classes can use accessories and consumables
can_use(Class, Type) :- class(Class), item_type(Type, accessory).
can_use(Class, Type) :- class(Class), item_type(Type, consumable).

% Item rules
is_weapon(Item) :- item(Item, Type, _, _, _), item_type(Type, weapon).
is_armor(Item) :- item(Item, Type, _, _, _), item_type(Type, armor).
is_accessory(Item) :- item(Item, Type, _, _, _), item_type(Type, accessory).

% Usability
usable_by(Item, Class) :-
    item(Item, Type, _, _, _),
    can_use(Class, Type).

level_appropriate(Item, CharLevel) :-
    item(Item, _, _, ItemLevel, _),
    CharLevel >= ItemLevel.

can_equip(Class, Item, CharLevel) :-
    usable_by(Item, Class),
    level_appropriate(Item, CharLevel).

% Power tiers
low_power(Item) :- item(Item, _, _, _, Power), Power < 50.
medium_power(Item) :- item(Item, _, _, _, Power), Power >= 50, Power < 200.
high_power(Item) :- item(Item, _, _, _, Power), Power >= 200.

#test "item types and slots" {
    item_type(sword, weapon).
    item_type(plate_armor, armor).
    item_type(ring, accessory).

    slot(weapon).
    slot(armor).
    slot(accessory).

    ?- item_type(sword, weapon).
    + true.

    ?- item_type(plate_armor, armor).
    + true.

    ?- item_type(sword, armor).
    - true.
}

#test "item rarity" {
    item(iron_sword, sword, common, 1, 10).
    item(excalibur, sword, legendary, 20, 500).

    ?- item(iron_sword, sword, common, 1, 10).
    + true.

    ?- item(excalibur, sword, legendary, 20, 500).
    + true.
}

#test "weapon identification" {
    item(iron_sword, sword, common, 1, 10).
    item(cloth_robe, robe, common, 1, 5).

    item_type(sword, weapon).
    item_type(robe, armor).

    is_weapon(Item) :- item(Item, Type, _, _, _), item_type(Type, weapon).
    is_armor(Item) :- item(Item, Type, _, _, _), item_type(Type, armor).

    ?- is_weapon(iron_sword).
    + true.

    ?- is_armor(cloth_robe).
    + true.

    ?- is_weapon(cloth_robe).
    - true.
}

#test "class item restrictions" {
    class(warrior).
    class(mage).

    item_type(sword, weapon).
    item_type(robe, armor).

    can_use(warrior, sword).
    can_use(mage, robe).

    ?- can_use(warrior, sword).
    + true.

    ?- can_use(mage, robe).
    + true.

    ?- can_use(warrior, robe).
    - true.
}

#test "item level requirements" {
    item(iron_sword, sword, common, 1, 10).
    item(steel_sword, sword, rare, 5, 50).
    item(excalibur, sword, legendary, 20, 500).

    level_appropriate(Item, CharLevel) :-
        item(Item, _, _, ItemLevel, _),
        CharLevel >= ItemLevel.

    % Level 1 character
    ?- level_appropriate(iron_sword, 1).
    + true.

    ?- level_appropriate(steel_sword, 1).
    - true.

    % Level 20 character
    ?- level_appropriate(iron_sword, 20).
    + true.

    ?- level_appropriate(steel_sword, 20).
    + true.

    ?- level_appropriate(excalibur, 20).
    + true.
}

#test "complete equipment check" {
    class(warrior).
    item(iron_sword, sword, common, 1, 10).
    item(archmage_robe, robe, legendary, 15, 300).

    item_type(sword, weapon).
    item_type(robe, armor).

    can_use(warrior, sword).
    can_use(mage, robe).

    usable_by(Item, Class) :-
        item(Item, Type, _, _, _),
        can_use(Class, Type).

    level_appropriate(Item, CharLevel) :-
        item(Item, _, _, ItemLevel, _),
        CharLevel >= ItemLevel.

    can_equip(Class, Item, CharLevel) :-
        usable_by(Item, Class),
        level_appropriate(Item, CharLevel).

    % Warrior level 5 can equip iron sword
    ?- can_equip(warrior, iron_sword, 5).
    + true.

    % Warrior level 5 cannot equip archmage robe (wrong class)
    ?- can_equip(warrior, archmage_robe, 5).
    - true.

    % Warrior level 1 can equip iron sword (meets level req)
    ?- can_equip(warrior, iron_sword, 1).
    + true.
}

#test "power tiers" {
    item(weak_item, sword, common, 1, 10).
    item(medium_item, sword, rare, 5, 100).
    item(strong_item, sword, legendary, 15, 500).

    low_power(Item) :- item(Item, _, _, _, Power), Power < 50.
    medium_power(Item) :- item(Item, _, _, _, Power), Power >= 50, Power < 200.
    high_power(Item) :- item(Item, _, _, _, Power), Power >= 200.

    ?- low_power(weak_item).
    + true.

    ?- medium_power(medium_item).
    + true.

    ?- high_power(strong_item).
    + true.

    ?- high_power(weak_item).
    - true.
}

#test "consumables usable by all" {
    class(warrior).
    class(mage).
    class(rogue).

    item_type(potion, consumable).

    can_use(Class, Type) :- class(Class), item_type(Type, consumable).

    ?- can_use(warrior, potion).
    + true.

    ?- can_use(mage, potion).
    + true.

    ?- can_use(rogue, potion).
    + true.
}

#test "warrior equipment options" {
    class(warrior).
    item(iron_sword, sword, common, 1, 10).
    item(steel_axe, axe, rare, 5, 60).
    item(archmage_robe, robe, legendary, 15, 300).

    item_type(sword, weapon).
    item_type(axe, weapon).
    item_type(robe, armor).

    can_use(warrior, sword).
    can_use(warrior, axe).
    can_use(mage, robe).

    usable_by(Item, Class) :-
        item(Item, Type, _, _, _),
        can_use(Class, Type).

    % Warrior can use swords and axes
    ?- usable_by(iron_sword, warrior).
    + true.

    ?- usable_by(steel_axe, warrior).
    + true.

    % Warrior cannot use mage robe
    ?- usable_by(archmage_robe, warrior).
    - true.
}

#test "item rarity progression" {
    rarity(common).
    rarity(rare).
    rarity(epic).
    rarity(legendary).

    item(item1, sword, common, 1, 10).
    item(item2, sword, rare, 5, 50).
    item(item3, sword, epic, 10, 150).
    item(item4, sword, legendary, 20, 500).

    better_than_common(Item) :- item(Item, _, Rarity, _, _), Rarity = rare.
    better_than_common(Item) :- item(Item, _, Rarity, _, _), Rarity = epic.
    better_than_common(Item) :- item(Item, _, Rarity, _, _), Rarity = legendary.

    ?- better_than_common(item2).
    + true.

    ?- better_than_common(item3).
    + true.

    ?- better_than_common(item4).
    + true.

    ?- better_than_common(item1).
    - true.
}
