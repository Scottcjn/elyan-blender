# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Build a person from a recipe, using the MPFB extension (MakeHuman for Blender).

MPFB is a separate GPL-3 extension that must be installed and enabled; this
module only calls it. The result is MPFB's own character, so its tools keep
working on it, plus a measured "contract" of landmarks stored on the rig.
"""

import importlib
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

from . import recipe as recipe_module

PROPERTY = "elyan_person"

# (recipe key, asset folder, MPFB asset type)
_PARTS = (
    ("eyes", "eyes", "Eyes"),
    ("eyebrows", "eyebrows", "Eyebrows"),
    ("eyelashes", "eyelashes", "Eyelashes"),
    ("teeth", "teeth", "Teeth"),
    ("tongue", "tongue", "Tongue"),
    ("hair", "hair", "Hair"),
)

# Bones whose positions other tools fit against, by the game_engine rig's names.
_LANDMARK_BONES = (
    "pelvis", "spine_03", "neck_01", "head",
    "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
    "thigh_l", "calf_l", "foot_l",
)


class MPFBMissing(RuntimeError):
    pass


def mpfb(module, name):
    """A class from the MPFB extension, wherever Blender mounted it."""
    for loaded in list(sys.modules):
        if loaded == "mpfb" or loaded.endswith(".mpfb"):
            return getattr(importlib.import_module(loaded + "." + module), name)
    raise MPFBMissing(
        "The MPFB extension is not enabled. Install it from extensions.blender.org "
        "(or makehumancommunity/mpfb2) and enable it, then try again."
    )


def _asset(folder, name, extension):
    asset_service = mpfb("services.assetservice", "AssetService")
    path = asset_service.find_asset_absolute_path(name + extension, asset_subdir=folder)
    if path is None:
        raise recipe_module.RecipeError(
            "no {:s} asset named {!r}; install the MakeHuman system assets pack "
            "or check the name".format(folder, name))
    return path


def _mixed_coordinates(basemesh):
    """Vertex positions with every shape key at its current value, in world space."""
    key = basemesh.shape_key_add(name="_elyan_measure", from_mix=True)
    co = np.empty(len(basemesh.data.vertices) * 3, dtype=np.float32)
    key.data.foreach_get("co", co)
    basemesh.shape_key_remove(key)
    co = co.reshape(-1, 3)
    matrix = np.array(basemesh.matrix_world, dtype=np.float32)
    return co @ matrix[:3, :3].T + matrix[:3, 3]


def _hull_perimeter(points):
    """Perimeter of the convex hull of 2D points: the length of a tape drawn around them."""
    points = sorted(set(map(tuple, np.round(points, 5).tolist())))
    if len(points) < 3:
        return 0.0

    def half(sequence):
        hull = []
        for p in sequence:
            while len(hull) >= 2 and (
                (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
                - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])
            ) <= 0:
                hull.pop()
            hull.append(p)
        return hull[:-1]

    hull = half(points) + half(reversed(points))
    return float(sum(
        np.hypot(hull[i][0] - hull[i - 1][0], hull[i][1] - hull[i - 1][1]) for i in range(len(hull))
    ))


def measure(basemesh, rig):
    """
    Landmarks and girths of the body as built: what garments and props are fitted to.

    Metres, world space, rest pose. Girths are tape measurements around the torso only.
    """
    co = _mixed_coordinates(basemesh)
    groups = {group.name: group.index for group in basemesh.vertex_groups}
    body = groups.get("body")
    torso_groups = {groups[name] for name in ("pelvis", "spine_01", "spine_02", "spine_03") if name in groups}
    is_body = np.zeros(len(co), dtype=bool)
    is_torso = np.zeros(len(co), dtype=bool)
    for vert in basemesh.data.vertices:
        torso_weight = 0.0
        for element in vert.groups:
            if element.group == body:
                is_body[vert.index] = True
            elif element.group in torso_groups:
                torso_weight += element.weight
        is_torso[vert.index] = torso_weight > 0.6

    bones = {}
    for name in _LANDMARK_BONES:
        bone = rig.data.bones.get(name)
        if bone is not None:
            bones[name] = [round(v, 4) for v in rig.matrix_world @ bone.head_local]

    contract = {
        "height": round(float(co[is_body, 2].max() - co[is_body, 2].min()), 4),
        "bones": bones,
    }
    if "thigh_l" in bones and "clavicle_l" in bones:
        low, high = bones["thigh_l"][2], bones["clavicle_l"][2]
        torso = co[is_torso]
        levels = np.linspace(low, high, 41)
        band = (high - low) / 40.0
        girths = np.array([
            _hull_perimeter(torso[np.abs(torso[:, 2] - z) < band][:, :2]) for z in levels
        ])

        def pick(start, stop, chooser):
            index = start + int(chooser(girths[start:stop]))
            return {"z": round(float(levels[index]), 4), "girth": round(float(girths[index]), 4)}

        # Hips are the widest part low down, the waist the narrowest in the middle,
        # the chest the widest high up.
        contract["hip"] = pick(1, 14, np.argmax)
        contract["waist"] = pick(12, 27, np.argmin)
        contract["chest"] = pick(24, 39, np.argmax)
    return contract


def build(recipe):
    """
    Create the person a recipe describes. Returns its rig and a report.

    The normalized recipe and the measured contract are stored on the rig as JSON
    in the ``elyan_person`` custom property.
    """
    recipe = recipe_module.normalize(recipe)
    human_service = mpfb("services.humanservice", "HumanService")
    target_service = mpfb("services.targetservice", "TargetService")
    location_service = mpfb("services.locationservice", "LocationService")
    properties = mpfb("entities.objectproperties", "HumanObjectProperties")

    # Resolve every asset first, so a wrong name fails before anything is created.
    skin = _asset("skins", recipe["skin"], ".mhmat")
    parts = [
        (_asset(folder, recipe[key], ".mhclo"), asset_type)
        for key, folder, asset_type in _PARTS if recipe[key]
    ]
    parts += [(_asset("clothes", name, ".mhclo"), "Clothes") for name in recipe["clothes"]]
    targets_root = location_service.get_mpfb_data("targets")
    targets = []
    for path, weight in recipe["targets"].items():
        full = os.path.join(targets_root, path + ".target.gz")
        if not os.path.exists(full):
            raise recipe_module.RecipeError("no modelling target {!r}".format(path))
        targets.append((full, weight))

    basemesh = human_service.create_human()
    basemesh.name = recipe["name"]
    for key, value in recipe["body"].items():
        properties.set_value(key, value, entity_reference=basemesh)
    for key, value in zip(("african", "asian", "caucasian"), recipe["mix"]):
        properties.set_value(key, value, entity_reference=basemesh)
    target_service.reapply_macro_details(basemesh)
    for full, weight in targets:
        target_service.load_target(basemesh, full, weight=weight)

    human_service.set_character_skin(skin, basemesh, skin_type="GAMEENGINE")
    rig = human_service.add_builtin_rig(basemesh, recipe["rig"])
    rig.name = recipe["name"] + ".rig"
    for path, asset_type in parts:
        human_service.add_mhclo_asset(path, basemesh, asset_type=asset_type, material_type="GAMEENGINE")

    report = {}
    if recipe["face"] != "none":
        from . import face
        face.add(basemesh, recipe["face"])
        report["eye_bones"] = face.add_eye_bones(basemesh, rig)
        report["face"] = face.check(basemesh)

    bpy.context.view_layer.update()
    contract = measure(basemesh, rig)
    rig[PROPERTY] = json.dumps({"recipe": recipe, "contract": contract})
    report.update(
        rig=rig.name,
        basemesh=basemesh.name,
        objects=sorted(child.name for child in rig.children_recursive),
        contract=contract,
    )
    return rig, report


def find_rig(ob):
    """The person rig an object belongs to, or None."""
    while ob is not None:
        if ob.type == 'ARMATURE' and PROPERTY in ob:
            return ob
        ob = ob.parent
    return None


def stored(rig):
    return json.loads(rig[PROPERTY])


def eye_centers(basemesh):
    """World positions of the left and right eye, from MakeHuman's eye helper geometry."""
    co = _mixed_coordinates(basemesh)
    result = {}
    for side, group_name in (("l", "helper-l-eye"), ("r", "helper-r-eye")):
        group = basemesh.vertex_groups.get(group_name)
        if group is None:
            continue
        members = [v.index for v in basemesh.data.vertices if any(g.group == group.index for g in v.groups)]
        if members:
            result[side] = Vector(co[members].mean(axis=0).tolist())
    return result
