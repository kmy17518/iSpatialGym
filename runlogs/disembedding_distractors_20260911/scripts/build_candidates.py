#!/usr/bin/env python3
"""Candidate distractors per disembedding target category (no simulator).

For every category that has disembedding test instances (the 39 categories under ``disembedding-0 / target`` in
``task_object_selection.csv``) this lists

* ``same``: every *ready* BEHAVIOR model of the same category (the future same-category pool), annotated with the
  grasp-library / selection data of the disembedding round (AG-valid grasps, selection score, pick rank) and the
  geometry-duplicate verdicts (exact duplicates, texture variants), and
* ``diff``: models of semantically related categories (hand-curated ``RELATED`` below, tier 1 = closest), ready
  models only, exact duplicates folded, at most ``PER_CATEGORY`` models per related category, interleaved
  round-robin inside a tier so that the list stays diverse.

Both lists are size-filtered for the source tray of the test spec (``tray/gsxbym``, 0.31 x 0.44 m): the canonical
footprint (+ 2 cm margin) must stay below ``MAX_FOOTPRINT`` (the sampler's footprint budget is 0.45 x tray area =
0.060 m^2 shared with the target; the largest target, the oatmeal pouch, uses 0.023), the longest side below
``MAX_EXTENT`` (fits the tray's inner floor next to a target) and above ``MIN_EXTENT`` (a 2 cm die is a few pixels
in the head camera).

Output: ``candidates.json`` (consumed by validate_distractors.py, compose_grids.py and assemble_pools.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
SEL = os.path.join(REPO, "runlogs", "grasp_library_20260905", "select")

MAX_FOOTPRINT = 0.045  # m^2, (x + 0.02) * (y + 0.02) of the canonical bbox (0.75 of the 0.060 budget: one big distractor + target)
MAX_FOOTPRINT_SAME = 0.051  # same-category candidates: up to the budget minus the smallest target of the round
MAX_EXTENT = 0.30  # m, longest horizontal bbox side (the tray's inner floor is ~0.27 x 0.40 m)
MAX_HEIGHT = 0.36  # m
MIN_EXTENT = 0.03  # m, longest bbox side
PER_CATEGORY = 3  # models per related category in the diff list
N_DIFF_VALIDATE = 26  # diff candidates handed to the sampler validation per target category (pool needs <= 18)

#: semantically related categories per target category, closest first. Tier 3 = generic placeholder-looking assets
#: (categories such as ``whiskey_bottle`` / ``milk_carton`` whose models share one synthetic bbox), last resort only.
RELATED = {
    "bottle_of_wine": [
        ["bottle_of_champagne", "bottle_of_beer", "bottle_of_sake", "bottle_of_rum", "bottle_of_whiskey", "bottle_of_vodka", "bottle_of_gin", "bottle_of_tequila"],
        ["bottle_of_pop", "bottle_of_water", "lemonade_bottle", "bottle_of_olive_oil", "bottle_of_soy_sauce", "bottle_of_cranberry_juice", "bottle_of_papaya_juice",
         "bottle_of_fruit_punch", "bottle_of_lime_juice", "bottle_of_coconut_water", "bottle_of_tea", "soda_water_bottle", "bottle_of_soda"],
        ["wine_bottle", "whiskey_bottle", "vodka_bottle", "water_bottle"],
    ],
    "video_game": [
        ["board_game", "hardback", "pack_of_cigarettes", "match_box", "wallet", "case_of_eyeshadow", "calculator", "chocolate_bar", "pack_of_chocolate_bar"],
        ["notebook", "notepad", "postcard", "envelope", "coaster", "box_of_cookies", "box_of_granola_bars", "tablet", "lighter", "sticker", "thermostat"],
    ],
    "box_of_cookies": [
        ["box_of_crackers", "box_of_candy", "box_of_chocolates", "bag_of_cookies", "box_of_cereal", "box_of_granola_bars", "box_of_baking_mix", "box_of_wagashi", "box_of_ice_cream", "box_of_coffee"],
        ["box_of_cane_sugar", "box_of_rice", "box_of_tissues", "board_game", "box_of_butter", "bag_of_ice_cream", "bag_of_jerky", "box_of_takeout", "pack_of_pasta", "box_of_baking_powder"],
    ],
    "can_of_soda": [
        ["can_of_icetea", "can_of_beans", "can_of_coffee", "canned_food", "can_of_tomatoes", "can_of_tomato_paste", "can_of_sardines", "can_of_corn", "can_of_oatmeal", "can_of_baking_mix", "spray_can"],
        ["bottle_of_pop", "bottle_of_water", "water_glass", "soda_cup", "paper_cup", "bottle_of_beer", "beer_glass", "jar_of_spaghetti_sauce", "salt_shaker", "pepper_shaker", "shaker"],
        ["tomato_paste_can"],
    ],
    "mug": [
        ["coffee_cup", "teacup", "beaker", "water_glass", "soda_cup", "paper_cup", "beer_glass", "goblet"],
        ["wineglass", "pencil_holder", "cream_pitcher", "jar_of_jam", "can_of_soda", "soap_dispenser", "candle_holder", "cup_of_yogurt", "box_of_yogurt", "canteen"],
    ],
    "bottle_of_beer": [
        ["bottle_of_pop", "bottle_of_wine", "bottle_of_water", "bottle_of_champagne", "lemonade_bottle", "bottle_of_sake", "bottle_of_rum", "bottle_of_whiskey", "bottle_of_ginger_beer", "bottle_of_soda", "soda_water_bottle"],
        ["can_of_soda", "bottle_of_olive_oil", "bottle_of_catsup", "bottle_of_hot_sauce", "bottle_of_orange_juice", "bottle_of_apple_juice", "bottle_of_tea", "bottle_of_coconut_water", "beer_glass", "bottle_of_strawberry_juice"],
        ["water_bottle", "wine_bottle", "whiskey_bottle"],
    ],
    "teaspoon": [
        ["tablespoon", "tablefork", "table_knife", "chopstick", "toothbrush", "pen", "pencil", "marker"],
        ["crayon", "chalk", "lipstick", "razor", "comb", "clipper", "tube_of_toothpaste", "granola_bar", "candy_cane", "cigar", "syringe", "meat_thermometer", "corkscrew", "tweezers"],
    ],
    "mouse": [
        ["webcam", "calculator", "stapler", "bar_soap", "potato", "wallet", "sunglasses", "watch", "charger", "keys", "lock", "soap_dish"],
        ["cellulose_tape_dispenser", "mousetrap", "lighter", "match_box", "pack_of_cigarettes", "video_game", "egg", "avocado", "mango", "pear", "sponge", "case_of_eyeshadow", "thermostat"],
    ],
    "notebook": [
        ["hardback", "paperback_book", "magazine", "folder", "folderal", "notepad", "board_game", "video_game", "cardstock", "paper_sheet", "tablet", "letter"],
        ["box_of_cookies", "box_of_chocolates", "calculator", "pennant", "box_of_wagashi", "postcard", "envelope", "bread_slice", "clipboard", "saucer", "coaster"],
    ],
    "soap_dispenser": [
        ["conditioner_dispenser", "shampoo_dispenser", "bottle_of_shampoo", "bottle_of_lotion", "tube_of_lotion", "bottle_of_dish_soap", "spray_bottle", "conditioner_atomizer", "water_atomizer",
         "detergent_atomizer", "deodorant_atomizer", "bleaching_agent_atomizer"],
        ["bottle_of_perfume", "bottle_of_cologne", "bottle_of_glass_cleaner", "bottle_of_face_cream", "bottle_of_sunscreen", "deodorant_stick", "pepper_grinder", "salt_shaker", "pepper_shaker",
         "bottle_of_medicine", "cream_pitcher", "bottle_of_mustard", "bottle_of_catsup", "bottle_of_mayonnaise", "spray_can", "bottle_of_disinfectant"],
        ["soap_bottle", "shampoo_bottle", "detergent_bottle", "disinfectant_bottle", "sugar_syrup_bottle"],
    ],
    "box_of_cereal": [
        ["box_of_rice", "box_of_cane_sugar", "box_of_oatmeal", "box_of_baking_mix", "box_of_crackers", "box_of_granola_bars", "box_of_cookies", "box_of_barley", "box_of_flour", "box_of_brown_sugar",
         "box_of_lasagna", "box_of_baking_powder", "box_of_coffee", "box_of_candy", "box_of_corn_flakes"],
        ["box_of_shampoo", "carton_of_milk", "carton_of_orange_juice", "carton_of_soy_milk", "box_of_almond_milk", "box_of_apple_juice", "box_of_vegetable_juice", "pack_of_pasta", "box_of_tissues",
         "bag_of_flour", "box_of_baking_soda", "box_of_sanitary_napkins"],
    ],
    "beaker": [
        ["water_glass", "mug", "coffee_cup", "teacup", "soda_cup", "paper_cup", "beer_glass", "wineglass", "goblet"],
        ["pencil_holder", "jar_of_jam", "can_of_soda", "cup_of_yogurt", "box_of_yogurt", "candle_holder", "cream_pitcher", "jar_of_mayonnaise", "canteen", "shaker"],
    ],
    "bottle_of_medicine": [
        ["bottle_of_supplements", "bottle_of_antihistamines", "bottle_of_black_pepper", "bottle_of_paprika", "bottle_of_sage", "bottle_of_seasoning", "bottle_of_ground_nutmeg", "bottle_of_ground_cloves",
         "bottle_of_ground_mace", "bottle_of_mustard_seeds", "bottle_of_sesame_seeds", "bottle_of_fennel", "bottle_of_ginger", "bottle_of_chili_pepper", "bottle_of_tea_leaves", "bottle_of_poppy_seeds",
         "bottle_of_pumpkin_pie_spice", "bottle_of_onion_powder", "jar_of_cumin", "jar_of_curry_powder", "jar_of_dill_seed", "salt_bottle", "salt_shaker", "pepper_shaker", "shaker", "nutmeg_shaker"],
        ["bottle_of_lotion", "tube_of_lotion", "deodorant_stick", "bottle_of_perfume", "bottle_of_cologne", "bottle_of_face_cream", "jar_of_jam", "jar_of_mayonnaise", "jar_of_puree", "bottle_of_peanut_butter",
         "can_of_bay_leaves", "bottle_of_pesto", "bottle_of_vinegar", "bottle_of_sunscreen", "bottle_of_glue", "spray_can", "bottle_of_essential_oil"],
    ],
    "bottle_of_milk": [
        ["carton_of_milk", "bottle_of_water", "bottle_of_pop", "bottle_of_soy_milk", "bottle_of_coconut_milk", "carton_of_soy_milk", "box_of_almond_milk", "bottle_of_milkshake", "bottle_of_orange_juice",
         "bottle_of_apple_juice", "bottle_of_cranberry_juice", "bottle_of_tea", "bottle_of_coconut_water", "bottle_of_soda", "cream_pitcher"],
        ["bottle_of_beer", "bottle_of_wine", "bottle_of_catsup", "bottle_of_mayonnaise", "bottle_of_shampoo", "bottle_of_lotion", "bottle_of_dish_soap", "bottle_of_strawberry_juice", "bottle_of_carrot_juice",
         "bottle_of_tomato_juice", "canteen"],
        ["water_bottle", "jug_of_milk"],
    ],
    "tablefork": [
        ["teaspoon", "tablespoon", "table_knife", "chopstick", "tongs", "toothbrush", "pen", "pencil", "marker"],
        ["crayon", "chalk", "ruler", "razor", "comb", "corkscrew", "tube_of_toothpaste", "candy_cane", "cigar", "scrub_brush", "lipstick", "hairbrush", "meat_thermometer", "flashlight", "syringe", "clipper"],
    ],
    "box_of_sanitary_napkins": [
        ["box_of_tissues", "bag_of_cookies", "bag_of_ice_cream", "bag_of_jerky", "bag_of_snacks", "bag_of_tea", "bag_of_breadcrumbs", "bag_of_cream_cheese", "box_of_shampoo", "box_of_cereal",
         "box_of_cane_sugar", "box_of_cookies", "paper_liners"],
        ["box_of_baking_mix", "box_of_rice", "carton_of_milk", "pack_of_cigarettes", "box_of_candy", "box_of_coffee", "pack_of_protein_powder", "paper_bag", "box_of_crackers", "box_of_baking_powder",
         "tissue_dispenser", "sponge"],
    ],
    "bar_soap": [
        ["sponge", "soap_dish", "mouse", "deodorant_stick", "sugar_cookie", "butter_cookie", "chocolate_bar", "pack_of_chocolate_bar", "biscuit", "cheesecake", "granola_bar", "egg", "potato", "lemon",
         "match_box", "lighter", "coaster", "brownie"],
        ["scrub_brush", "comb", "wallet", "case_of_eyeshadow", "video_game", "pack_of_cigarettes", "watch", "sunglasses", "cellulose_tape", "cookie_cutter", "keys", "lock", "muffin", "plum", "walnut",
         "hamburger_bun", "bread_slice"],
    ],
    "box_of_cane_sugar": [
        ["box_of_brown_sugar", "box_of_flour", "bag_of_flour", "box_of_rice", "box_of_cereal", "box_of_oatmeal", "box_of_baking_mix", "box_of_baking_powder", "box_of_baking_soda", "box_of_barley",
         "box_of_crackers", "box_of_coffee", "box_of_lasagna", "jar_of_sugar", "box_of_cookies"],
        ["box_of_shampoo", "carton_of_milk", "box_of_granola_bars", "box_of_candy", "pack_of_pasta", "box_of_tissues", "box_of_sanitary_napkins", "can_of_baking_mix", "bag_of_breadcrumbs",
         "box_of_almond_milk", "carton_of_soy_milk", "bottle_of_protein_powder"],
    ],
    "carton_of_milk": [
        ["carton_of_orange_juice", "carton_of_soy_milk", "carton_of_pineapple_juice", "box_of_almond_milk", "box_of_apple_juice", "box_of_vegetable_juice", "bottle_of_milk", "bottle_of_soy_milk",
         "bottle_of_coconut_milk", "jug_of_milk"],
        ["box_of_cereal", "box_of_rice", "box_of_cane_sugar", "box_of_shampoo", "box_of_baking_mix", "box_of_crackers", "bottle_of_orange_juice", "bottle_of_water", "bottle_of_pop", "box_of_coffee",
         "box_of_flour", "box_of_oatmeal", "cream_pitcher"],
        ["milk_carton", "orange_juice_carton", "yogurt_carton"],
    ],
    "chili": [
        ["carrot", "cucumber", "pickle", "green_bean", "hotdog_frank", "asparagus", "candy_cane", "cigar", "radish", "date", "banana", "zucchini", "celery", "bacon"],
        ["pen", "pencil", "marker", "crayon", "chalk", "lipstick", "razor", "comb", "teaspoon", "chopstick", "granola_bar", "chocolate_bar", "strawberry", "plum", "meatball", "peppermint", "clipper", "syringe"],
    ],
    "box_of_shampoo": [
        ["box_of_cereal", "box_of_rice", "box_of_cane_sugar", "box_of_baking_mix", "box_of_crackers", "box_of_coffee", "box_of_granola_bars", "box_of_tissues", "box_of_sanitary_napkins",
         "bottle_of_shampoo", "shampoo_dispenser", "conditioner_dispenser", "soap_dispenser", "box_of_whiskey", "box_of_rum", "box_of_champagne"],
        ["carton_of_milk", "carton_of_orange_juice", "box_of_oatmeal", "box_of_flour", "box_of_brown_sugar", "box_of_lasagna", "box_of_barley", "bottle_of_lotion", "tube_of_lotion", "pack_of_pasta",
         "box_of_baking_powder", "box_of_almond_milk", "box_of_sake"],
    ],
    "box_of_rice": [
        ["box_of_cereal", "box_of_cane_sugar", "box_of_brown_sugar", "box_of_flour", "box_of_oatmeal", "box_of_baking_mix", "box_of_barley", "box_of_crackers", "box_of_lasagna", "box_of_baking_powder",
         "box_of_coffee", "bag_of_flour", "box_of_granola_bars", "pack_of_pasta"],
        ["box_of_shampoo", "carton_of_milk", "box_of_cookies", "box_of_candy", "box_of_tissues", "box_of_sanitary_napkins", "box_of_baking_soda", "carton_of_soy_milk", "box_of_almond_milk",
         "box_of_vegetable_juice", "bag_of_breadcrumbs", "box_of_apple_juice"],
    ],
    "cellulose_tape": [
        ["coaster", "cookie_cutter", "cap", "cork", "watch", "bracelet", "sugar_cookie", "butter_cookie", "chocolate_chip_cookie", "biscuit", "cellulose_tape_dispenser", "peppermint", "hamburger_bun"],
        ["bar_soap", "match_box", "lighter", "case_of_eyeshadow", "video_game", "egg", "lemon", "apple", "toy_dice", "battery", "charger", "webcam", "mouse", "sponge", "soap_dish", "keys", "lock", "lid"],
    ],
    "conditioner_dispenser": [
        ["soap_dispenser", "shampoo_dispenser", "bottle_of_shampoo", "bottle_of_conditioner", "conditioner_atomizer", "bottle_of_lotion", "tube_of_lotion", "bottle_of_dish_soap", "spray_bottle",
         "water_atomizer", "detergent_atomizer", "deodorant_atomizer", "bottle_of_perfume", "bottle_of_cologne", "deodorant_stick"],
        ["bottle_of_water", "bottle_of_pop", "bottle_of_medicine", "bottle_of_sunscreen", "bottle_of_face_cream", "bottle_of_disinfectant", "pepper_grinder", "salt_shaker", "bottle_of_mustard",
         "bottle_of_catsup", "bottle_of_glass_cleaner", "spray_can", "bottle_of_mayonnaise"],
        ["shampoo_bottle", "soap_bottle", "disinfectant_bottle", "sugar_syrup_bottle"],
    ],
    "cookie_cutter": [
        ["cellulose_tape", "coaster", "cap", "cork", "bracelet", "sugar_cookie", "butter_cookie", "chocolate_chip_cookie", "biscuit", "watch", "cupcake", "toy_dice", "hamburger_bun", "peppermint"],
        ["bar_soap", "match_box", "lighter", "case_of_eyeshadow", "egg", "lemon", "apple", "battery", "charger", "webcam", "mouse", "soap_dish", "sponge", "tea_bag", "corkscrew", "keys", "lock", "lid"],
    ],
    "pencil_case": [
        ["pencil_holder", "wallet", "sunglasses", "tube_of_toothpaste", "pack_of_kielbasa", "granola_bar", "chocolate_bar", "pack_of_chocolate_bar", "cucumber", "zucchini", "burrito", "taco", "hotdog_bun",
         "flashlight", "hairbrush"],
        ["box_of_cookies", "bar_soap", "video_game", "pack_of_cigarettes", "match_box", "calculator", "stapler", "cellulose_tape_dispenser", "mouse", "sponge", "hardback", "bottle_of_milk", "pen",
         "marker", "ruler", "eggplant", "banana", "croissant"],
    ],
    "tortilla_chips": [
        ["chip", "biscuit", "sugar_cookie", "butter_cookie", "chocolate_chip_cookie", "cheese_danish", "pack_of_chocolate_bar", "chocolate_bar", "granola_bar", "peppermint", "peppermint_candy",
         "candy_cane", "bread_slice", "hamburger_bun", "bacon", "scone", "brownie"],
        ["bar_soap", "coaster", "cookie_cutter", "cellulose_tape", "cap", "video_game", "bandage", "tea_bag", "sponge", "lemon", "potato", "envelope", "postcard", "soap_dish", "muffin", "cupcake", "lid"],
    ],
    "pack_of_cigarettes": [
        ["match_box", "lighter", "pack_of_chocolate_bar", "chocolate_bar", "granola_bar", "video_game", "wallet", "case_of_eyeshadow", "cigar", "bar_soap", "box_of_butter", "tube_of_toothpaste",
         "bandage", "box_of_cream", "hardback"],
        ["box_of_cookies", "pencil_case", "calculator", "stapler", "cellulose_tape", "coaster", "watch", "sunglasses", "mouse", "sponge", "soap_dish", "box_of_baking_mix", "box_of_granola_bars",
         "board_game", "keys", "lock", "envelope", "postcard"],
    ],
    "bottle_of_pop": [
        ["bottle_of_water", "bottle_of_beer", "soda_water_bottle", "bottle_of_soda", "lemonade_bottle", "bottle_of_ginger_beer", "bottle_of_tea", "bottle_of_orange_juice", "bottle_of_apple_juice",
         "bottle_of_cranberry_juice", "bottle_of_fruit_punch", "bottle_of_papaya_juice", "bottle_of_strawberry_juice", "bottle_of_carrot_juice", "bottle_of_tomato_juice", "bottle_of_coconut_water",
         "bottle_of_milkshake", "can_of_soda", "can_of_icetea"],
        ["bottle_of_wine", "bottle_of_champagne", "bottle_of_catsup", "bottle_of_olive_oil", "bottle_of_mustard", "bottle_of_hot_sauce", "bottle_of_dish_soap", "bottle_of_shampoo", "bottle_of_lotion",
         "bottle_of_sake", "bottle_of_rum", "bottle_of_whiskey", "bottle_of_barbecue_sauce", "bottle_of_garlic_sauce", "bottle_of_lime_juice"],
        ["water_bottle"],
    ],
    "bottle_of_shampoo": [
        ["shampoo_dispenser", "soap_dispenser", "conditioner_dispenser", "bottle_of_conditioner", "conditioner_atomizer", "bottle_of_lotion", "tube_of_lotion", "bottle_of_dish_soap", "bottle_of_sunscreen",
         "deodorant_atomizer", "deodorant_stick", "detergent_atomizer", "water_atomizer", "bottle_of_perfume", "bottle_of_cologne", "bottle_of_face_cream", "box_of_shampoo", "spray_bottle"],
        ["bottle_of_water", "bottle_of_pop", "bottle_of_medicine", "bottle_of_disinfectant", "bottle_of_glass_cleaner", "bottle_of_mustard", "bottle_of_catsup", "bottle_of_mayonnaise", "pepper_grinder",
         "salt_shaker", "bottle_of_detergent", "bleaching_agent_atomizer", "spray_can", "bottle_of_supplements"],
        ["shampoo_bottle", "soap_bottle", "detergent_bottle", "disinfectant_bottle", "sugar_syrup_bottle"],
    ],
    "box_of_butter": [
        ["butter", "bag_of_cream_cheese", "box_of_cream", "box_of_cookies", "box_of_granola_bars", "box_of_baking_mix", "pack_of_chocolate_bar", "chocolate_bar", "box_of_candy", "box_of_crackers",
         "box_of_coffee", "bag_of_cookies", "cheesecake", "bag_of_ice_cream", "box_of_ice_cream"],
        ["pack_of_cigarettes", "video_game", "box_of_cereal", "box_of_rice", "box_of_cane_sugar", "box_of_tissues", "box_of_sanitary_napkins", "box_of_yogurt", "cup_of_yogurt", "pack_of_pasta",
         "pencil_case", "bar_soap", "soap_dish", "match_box", "board_game", "hardback"],
    ],
    "box_of_oatmeal": [
        ["box_of_cereal", "box_of_rice", "box_of_cane_sugar", "bag_of_flour", "box_of_flour", "box_of_brown_sugar", "box_of_barley", "box_of_baking_mix", "box_of_crackers", "can_of_oatmeal",
         "bag_of_snacks", "bag_of_tea", "bag_of_breadcrumbs", "box_of_baking_powder", "box_of_coffee", "box_of_lasagna", "pack_of_protein_powder", "bottle_of_protein_powder", "paper_bag"],
        ["box_of_shampoo", "carton_of_milk", "box_of_cookies", "box_of_granola_bars", "box_of_tissues", "box_of_sanitary_napkins", "box_of_candy", "carton_of_orange_juice", "carton_of_soy_milk",
         "box_of_almond_milk", "pack_of_pasta", "box_of_baking_soda"],
    ],
    "pencil_holder": [
        ["mug", "beaker", "coffee_cup", "water_glass", "soda_cup", "paper_cup", "pencil_case", "vase", "jar_of_jam", "can_of_soda", "can_of_coffee", "canned_food", "canister", "candle_holder", "teacup",
         "toy_box", "goblet", "beer_glass"],
        ["pencil", "pen", "marker", "box_of_yogurt", "cup_of_yogurt", "cream_pitcher", "jar_of_mayonnaise", "jar_of_puree", "jar_of_spaghetti_sauce", "soap_dispenser", "box_of_tissues",
         "box_of_cane_sugar", "can_of_baking_mix", "wineglass", "canteen", "shaker"],
    ],
    "toy_dice": [
        ["toy_box", "box_of_cream", "match_box", "box_of_yogurt", "jar_of_bath_salt", "can_of_baking_mix", "bar_soap", "cellulose_tape", "cookie_cutter", "video_game", "box_of_tissues", "box_of_butter",
         "keys", "lock", "webcam", "charger", "battery", "toy_figure"],
        ["apple", "lemon", "orange", "potato", "egg", "mug", "beaker", "teacup", "coffee_cup", "cap", "coaster", "garlic", "jar_of_jam", "jar_of_mayonnaise", "jar_of_puree", "bottle_of_black_pepper",
         "pillar_candle", "cupcake", "muffin", "brownie", "walnut", "plum", "sushi", "dumpling", "tofu", "cup_of_yogurt", "sponge"],
    ],
    "box_of_baking_mix": [
        ["box_of_cereal", "box_of_rice", "box_of_cane_sugar", "box_of_brown_sugar", "box_of_flour", "box_of_baking_powder", "box_of_baking_soda", "box_of_crackers", "box_of_barley", "box_of_lasagna",
         "box_of_coffee", "box_of_granola_bars", "box_of_cookies", "box_of_candy", "can_of_baking_mix", "pack_of_pasta"],
        ["box_of_shampoo", "box_of_oatmeal", "carton_of_milk", "box_of_tissues", "box_of_sanitary_napkins", "box_of_almond_milk", "box_of_apple_juice", "box_of_vegetable_juice", "carton_of_soy_milk",
         "bag_of_flour", "box_of_butter", "bag_of_breadcrumbs", "box_of_coffee"],
    ],
    "box_of_tissues": [
        ["box_of_sanitary_napkins", "tissue_dispenser", "paper_liners", "bag_of_cookies", "bag_of_jerky", "bag_of_ice_cream", "bag_of_cream_cheese", "box_of_cookies", "box_of_candy", "box_of_coffee",
         "sponge", "box_of_butter", "box_of_baking_mix", "paper_bag", "bag_of_snacks"],
        ["box_of_cereal", "box_of_rice", "box_of_cane_sugar", "box_of_shampoo", "carton_of_milk", "pack_of_cigarettes", "box_of_granola_bars", "box_of_yogurt", "can_of_baking_mix", "toy_box",
         "box_of_takeout", "board_game", "bag_of_tea", "box_of_crackers"],
    ],
    "box_of_granola_bars": [
        ["box_of_baking_mix", "box_of_cookies", "box_of_crackers", "box_of_cereal", "box_of_candy", "box_of_chocolates", "chocolate_bar", "pack_of_chocolate_bar", "granola_bar", "box_of_butter",
         "box_of_coffee", "box_of_rice", "box_of_cane_sugar", "bag_of_cookies", "box_of_baking_powder", "pack_of_pasta"],
        ["box_of_tissues", "pack_of_cigarettes", "video_game", "board_game", "box_of_shampoo", "carton_of_milk", "box_of_sanitary_napkins", "hardback", "notebook", "box_of_barley", "box_of_lasagna",
         "match_box", "box_of_baking_soda", "box_of_brown_sugar"],
    ],
    "box_of_yogurt": [
        ["cup_of_yogurt", "box_of_cream", "cupcake", "teacup", "coffee_cup", "paper_cup", "soda_cup", "beaker", "mug", "jar_of_jam", "jar_of_mayonnaise", "jar_of_puree", "jar_of_bath_salt",
         "bottle_of_face_cream", "bottle_of_sour_cream", "can_of_corn", "can_of_baking_mix", "canned_food", "can_of_beans", "jar_of_orange_sauce", "muffin"],
        ["can_of_soda", "bottle_of_black_pepper", "box_of_tissues", "box_of_butter", "bar_soap", "apple", "lemon", "orange", "potato", "egg", "toy_dice", "pillar_candle", "cap", "cellulose_tape",
         "brownie", "can_of_coffee", "can_of_tomatoes", "lid", "bowl"],
    ],
    "jar_of_bath_salt": [
        ["jar_of_jam", "jar_of_mayonnaise", "jar_of_puree", "jar_of_cumin", "jar_of_curry_powder", "jar_of_dill_seed", "jar_of_orange_sauce", "jar_of_spaghetti_sauce", "jar_of_sugar", "jar_of_clove",
         "jar_of_tumeric", "bottle_of_black_pepper", "bottle_of_paprika", "bottle_of_sage", "bottle_of_seasoning", "bottle_of_ground_nutmeg", "bottle_of_ground_cloves", "bottle_of_ground_mace",
         "bottle_of_mustard_seeds", "bottle_of_sesame_seeds", "bottle_of_fennel", "bottle_of_ginger", "bottle_of_chili_pepper", "bottle_of_tea_leaves", "bottle_of_poppy_seeds", "can_of_bay_leaves",
         "bottle_of_face_cream", "bottle_of_cocoa", "bottle_of_peanut_butter", "bottle_of_lemon_sauce", "bottle_of_mushroom_sauce", "bottle_of_medicine", "bottle_of_supplements", "pillar_candle"],
        ["box_of_yogurt", "cup_of_yogurt", "can_of_baking_mix", "can_of_corn", "canned_food", "can_of_beans", "can_of_coffee", "mug", "teacup", "beaker", "salt_shaker", "pepper_shaker", "bar_soap",
         "toy_dice", "bottle_of_sour_cream", "jar_of_jelly", "jar_of_strawberry_jam", "candle_holder", "shaker", "nutmeg_shaker"],
        ["tomato_sauce_jar", "peanut_butter_jar", "noodle_jar"],
    ],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "candidates.json"))
    args = ap.parse_args()

    from ispatialgym.sampling import knowledge as K

    tests = [t for t in json.load(open(os.path.join(SEL, "test_instances.json"))) if t["task"] == "disembedding"]
    cats = [t["category"] for t in tests]
    missing = sorted(set(cats) - set(RELATED))
    assert not missing, f"no RELATED entry for {missing}"

    # grasp-library data of the disembedding round
    lib = {}  # (category, model) -> dict
    for t in tests:
        gid = f"disembedding__{t['category']}__r{t['round']}"
        grp = json.load(open(os.path.join(SEL, "groups", gid + ".json")))
        out = json.load(open(os.path.join(SEL, "out", gid + ".json")))
        scores = {c["model"]: c for c in out["candidates"]}
        rank = {t["pick1"]: "pick 1", t["pick2"]: "pick 2", t["alternate"]: "alternate"}
        for c in grp["candidates"]:
            sc = scores.get(c["model"], {})
            lib[(t["category"], c["model"])] = {"grasp_key": c["key"], "ag_valid": c["n_valid"], "clean": c["n_clean"], "score": sc.get("score"),
                                                 "V": sc.get("V"), "selection": rank.get(c["model"], "candidate"), "reviewer": sc.get("note")}
        for m in grp.get("excluded", []) or []:
            mid = m["model"] if isinstance(m, dict) else m
            lib.setdefault((t["category"], mid), {"grasp_key": None, "ag_valid": 0, "clean": 0, "score": None, "V": None, "selection": "excluded", "reviewer": None})

    # geometry-duplicate verdicts
    exact = {}  # (cat, model) -> group (sorted list incl. itself)
    same_geom = {}  # (cat, model) -> cluster id (texture variants / exact)
    for c in json.load(open(os.path.join(SEL, "duplicates", "verdicts.json"))):
        for g in c["exact_groups"]:
            for m in g:
                exact[(c["category"], m)] = sorted(g)
        if c["verdict"] in ("exact_same", "texture_variants"):
            for m in c["models"]:
                same_geom[(c["category"], m)] = c["id"]

    def size_check(bb, max_footprint=MAX_FOOTPRINT):
        if bb is None:
            return False, "no bbox"
        x, y, z = bb
        ext = max(x, y)
        fp = (x + 0.02) * (y + 0.02)
        if max(bb) < MIN_EXTENT:
            return False, f"too small ({max(bb) * 100:.1f} cm)"
        if ext > MAX_EXTENT:
            return False, f"too long ({ext * 100:.0f} cm > {MAX_EXTENT * 100:.0f})"
        if z > MAX_HEIGHT:
            return False, f"too tall ({z * 100:.0f} cm)"
        if fp > max_footprint:
            return False, f"footprint {fp:.3f} m^2 > {max_footprint}"
        return True, ""

    def model_entry(cat, m, same_category=False):
        bb = K.canonical_bbox(m)
        # same-category candidates are the hard distractors: anything the sampler's own footprint budget (0.060 m^2
        # with the target) can still draw is kept and left to the sampler validation
        ok, why = size_check(bb, MAX_FOOTPRINT_SAME if same_category else MAX_FOOTPRINT)
        e = {"category": cat, "model": m, "bbox": [round(v, 4) for v in bb] if bb else None, "size_ok": ok, "size_note": why,
             "ready": K.is_ready(m), "exact_group": exact.get((cat, m)), "geom_cluster": same_geom.get((cat, m))}
        e.update(lib.get((cat, m), {"grasp_key": None, "ag_valid": None, "clean": None, "score": None, "V": None, "selection": "untested", "reviewer": None}))
        return e

    def rank_key(e):
        order = {"pick 1": 0, "pick 2": 1, "alternate": 2, "candidate": 3, "untested": 4, "excluded": 5}
        return (order.get(e["selection"], 6), -(e["score"] or 0), -(e["ag_valid"] or 0), e["model"])

    targets = OrderedDict()
    same = OrderedDict()
    diff = OrderedDict()
    for t in tests:
        cat = t["category"]
        targets[cat] = {"pick1": t["pick1"], "pick1_key": t["pick1_key"], "pick2": t["pick2"], "alternate": t["alternate"], "round": t["round"],
                        "pick1_bbox": K.canonical_bbox(t["pick1"])}
        # ---- same category: every ready model (+ tested non-ready ones flagged)
        models = set(K.models_of_category(cat, ready_only=True)) | {m for (c, m) in lib if c == cat}
        entries = sorted((model_entry(cat, m, same_category=True) for m in models), key=rank_key)
        same[cat] = entries
        # ---- related categories, by tier, round-robin, exact duplicates folded, PER_CATEGORY per category
        picked = []
        seen_exact = set()
        for tier, rel_cats in enumerate(RELATED[cat], start=1):
            per_cat = []
            for rc in rel_cats:
                if rc == cat:
                    continue
                lst = []
                for m in K.models_of_category(rc, ready_only=True):
                    e = model_entry(rc, m)
                    if not e["size_ok"]:
                        continue
                    g = tuple(e["exact_group"]) if e["exact_group"] else (m,)
                    if (rc, g) in seen_exact:
                        continue
                    seen_exact.add((rc, g))
                    e["tier"] = tier
                    lst.append(e)
                lst.sort(key=rank_key)
                if lst:
                    per_cat.append(lst[:PER_CATEGORY])
            # interleave: first model of every category, then the second ones, ...
            k = 0
            while True:
                added = False
                for lst in per_cat:
                    if k < len(lst):
                        picked.append(lst[k])
                        added = True
                if not added:
                    break
                k += 1
        diff[cat] = picked
    n_same = sum(len(v) for v in same.values())
    n_diff = sum(len(v) for v in diff.values())
    uniq = sorted({(e["category"], e["model"]) for v in list(same.values()) + list(diff.values()) for e in v if e["ready"] and e["size_ok"]})
    out = {"tray_source": {"model": "gsxbym", "bbox": K.canonical_bbox("gsxbym")}, "limits": {"max_footprint_m2": MAX_FOOTPRINT, "max_extent_m": MAX_EXTENT,
           "max_height_m": MAX_HEIGHT, "min_extent_m": MIN_EXTENT, "per_related_category": PER_CATEGORY, "n_diff_validate": N_DIFF_VALIDATE},
           "targets": targets, "same": same, "diff": diff, "related": RELATED, "unique_models": [{"category": c, "model": m} for c, m in uniq]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{len(targets)} target categories; same-category candidates {n_same} (size ok {sum(1 for v in same.values() for e in v if e['size_ok'])}); "
          f"related-category candidates {n_diff}; unique renderable models {len(uniq)} -> {args.out}")
    for cat in targets:
        s_ok = [e for e in same[cat] if e["size_ok"] and e["ready"]]
        d = diff[cat]
        print(f"  {cat:26s} same {len(s_ok):2d}/{len(same[cat]):2d}  diff {len(d):3d} from {len({e['category'] for e in d}):2d} categories"
              + (f"  (tier1 {sum(1 for e in d if e['tier'] == 1)})"))


if __name__ == "__main__":
    main()
