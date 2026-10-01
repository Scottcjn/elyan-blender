# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Person recipes: the complete description of a character as plain data.

No ``bpy`` here. A recipe is checked and filled with defaults before anything is
built, so a bad one fails with a message instead of a half-made character.
"""

import copy
import json

SCHEMA = "elyan.person/1"

# 0..1 sliders of the MakeHuman body model.
BODY_DEFAULTS = {
    "gender": 0.5,       # 0 female, 1 male
    "age": 0.5,          # 0 baby, 0.1875 child, 0.5 young adult, 1 old
    "muscle": 0.5,
    "weight": 0.5,
    "height": 0.5,
    "proportions": 0.5,  # 0 uncommon, 1 idealised
    "cupsize": 0.5,
    "firmness": 0.5,
}

DEFAULTS = {
    "schema": SCHEMA,
    "name": "Person",
    "body": BODY_DEFAULTS,
    # Weights of MakeHuman's three base shape families; they are normalized to sum to 1.
    "mix": [1.0, 1.0, 1.0],
    # Extra modelling targets by path under MakeHuman's targets folder, e.g.
    # {"nose/nose-volume-incr": 0.5}. Values outside 0..1 exaggerate.
    "targets": {},
    "skin": "young_caucasian_female",
    "eyes": "low-poly",
    "eyebrows": "eyebrow001",
    "eyelashes": "eyelashes01",
    "teeth": "teeth_base",
    "tongue": "tongue01",
    "hair": None,
    "clothes": [],
    "rig": "game_engine",
    # "none", "visemes" (15 mouth shapes) or "full" (plus 52 ARKit expression shapes).
    "face": "full",
}

FACES = ("none", "visemes", "full")
RIGS = ("game_engine", "game_engine_with_breast", "default", "default_no_toes", "mixamo", "mixamo_unity", "cmu_mb")


class RecipeError(ValueError):
    pass


def _number(value, where, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeError("{:s} must be a number, got {!r}".format(where, value))
    if (low is not None and value < low) or (high is not None and value > high):
        raise RecipeError("{:s} must be between {:g} and {:g}, got {:g}".format(where, low, high, value))
    return float(value)


def _name(value, where, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise RecipeError("{:s} must be an asset name{:s}, got {!r}".format(
            where, " or null" if optional else "", value))
    return value


def normalize(recipe):
    """Return a complete, checked copy of ``recipe`` (a dict or a JSON string)."""
    if isinstance(recipe, str):
        try:
            recipe = json.loads(recipe)
        except ValueError as ex:
            raise RecipeError("recipe is not valid JSON: {!s}".format(ex)) from None
    if not isinstance(recipe, dict):
        raise RecipeError("recipe must be an object")
    unknown = sorted(set(recipe) - set(DEFAULTS))
    if unknown:
        raise RecipeError("unknown recipe keys: {:s}; known: {:s}".format(
            ", ".join(unknown), ", ".join(sorted(DEFAULTS))))
    if recipe.get("schema", SCHEMA) != SCHEMA:
        raise RecipeError("unsupported schema {!r}, this build reads {:s}".format(recipe["schema"], SCHEMA))

    result = copy.deepcopy(DEFAULTS)
    result["name"] = _name(recipe.get("name", result["name"]), "name")

    body = recipe.get("body", {})
    if not isinstance(body, dict):
        raise RecipeError("body must be an object")
    unknown = sorted(set(body) - set(BODY_DEFAULTS))
    if unknown:
        raise RecipeError("unknown body keys: {:s}; known: {:s}".format(
            ", ".join(unknown), ", ".join(sorted(BODY_DEFAULTS))))
    for key, value in body.items():
        result["body"][key] = _number(value, "body." + key, 0.0, 1.0)

    mix = recipe.get("mix", result["mix"])
    if not isinstance(mix, (list, tuple)) or len(mix) != 3:
        raise RecipeError("mix must be three numbers")
    mix = [_number(v, "mix", 0.0) for v in mix]
    if sum(mix) <= 0.0:
        raise RecipeError("mix must not be all zero")
    result["mix"] = [v / sum(mix) for v in mix]

    targets = recipe.get("targets", {})
    if not isinstance(targets, dict):
        raise RecipeError("targets must be an object of path: weight")
    result["targets"] = {
        _name(path, "targets key"): _number(weight, "targets." + str(path), -2.0, 2.0)
        for path, weight in targets.items()
    }

    for key in ("skin", "eyes", "teeth", "tongue"):
        result[key] = _name(recipe.get(key, result[key]), key)
    for key in ("eyebrows", "eyelashes", "hair"):
        result[key] = _name(recipe.get(key, result[key]), key, optional=True)

    clothes = recipe.get("clothes", [])
    if not isinstance(clothes, (list, tuple)):
        raise RecipeError("clothes must be a list of asset names")
    result["clothes"] = [_name(item, "clothes item") for item in clothes]

    result["rig"] = recipe.get("rig", result["rig"])
    if result["rig"] not in RIGS:
        raise RecipeError("unknown rig {!r}; known: {:s}".format(result["rig"], ", ".join(RIGS)))
    result["face"] = recipe.get("face", result["face"])
    if result["face"] not in FACES:
        raise RecipeError("unknown face {!r}; known: {:s}".format(result["face"], ", ".join(FACES)))
    return result
