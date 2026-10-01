# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Talking faces: mouth shapes for speech, expression shapes, eye bones, and
numeric checks that the shapes do what their names say.

The shapes come from MakeHuman's "visemes02" and "faceunits01" packs through
MPFB. They are shape keys left at zero; the engine that plays the character
drives them.
"""

import bpy
import numpy as np
from mathutils import Vector, kdtree

from . import build

# The fifteen mouth shapes of the Oculus/Meta lip-sync set.
VISEMES = (
    "viseme_sil", "viseme_PP", "viseme_FF", "viseme_TH", "viseme_DD", "viseme_kk", "viseme_CH",
    "viseme_SS", "viseme_nn", "viseme_RR", "viseme_aa", "viseme_E", "viseme_I", "viseme_O", "viseme_U",
)
# VRChat looks for these names.
VRCHAT_NAMES = {
    "viseme_sil": "vrc.v_sil", "viseme_PP": "vrc.v_pp", "viseme_FF": "vrc.v_ff", "viseme_TH": "vrc.v_th",
    "viseme_DD": "vrc.v_dd", "viseme_kk": "vrc.v_kk", "viseme_CH": "vrc.v_ch", "viseme_SS": "vrc.v_ss",
    "viseme_nn": "vrc.v_nn", "viseme_RR": "vrc.v_rr", "viseme_aa": "vrc.v_aa", "viseme_E": "vrc.v_e",
    "viseme_I": "vrc.v_ih", "viseme_O": "vrc.v_oh", "viseme_U": "vrc.v_ou",
}
# Enough expression for a small budget: blink, gaze-adjacent lids, brows, smile, frown.
COMPACT_EXPRESSIONS = (
    "eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight", "eyeSquintLeft", "eyeSquintRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight",
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
)
EYE_BONES = {"l": "eye_l", "r": "eye_r"}


def is_face_key(name):
    """Shape keys this module manages, as opposed to the ones that shape the body."""
    return name != "Basis" and not name.startswith("$") and not name.startswith("_")


def keys_for(profile, available):
    """Which face shape keys an export for ``profile`` should carry."""
    if profile == "quest":
        wanted = VISEMES + COMPACT_EXPRESSIONS
        return [name for name in available if name in wanted]
    return list(available)


def add(basemesh, level):
    """Load the face shapes onto a person. ``level`` is "visemes" or "full"."""
    face_service = build.mpfb("services.faceservice", "FaceService")
    full = level == "full"
    if full and not face_service.is_faceunits01_installed(force_recheck=True):
        raise build.recipe_module.RecipeError(
            "face \"full\" needs MakeHuman's faceunits01 asset pack; install it or use face \"visemes\"")
    try:
        face_service.load_targets(
            basemesh, load_microsoft_visemes=False, load_meta_visemes=True, load_arkit_faceunits=full)
    except Exception as ex:
        raise build.recipe_module.RecipeError(
            "could not load the face shapes; is MakeHuman's visemes02 asset pack installed? ({!s})".format(ex))
    # Teeth, tongue, lashes and brows must move with the face.
    face_service.interpolate_targets(basemesh)


def add_eye_bones(basemesh, rig):
    """
    One bone per eye, so gaze is a rotation and not a shape. Returns the bone names added.

    The eyes point down -Y in the rest pose, so the bones do too.
    """
    centers = build.eye_centers(basemesh)
    head = rig.data.bones.get("head")
    if len(centers) < 2 or head is None:
        return []
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    to_rig = rig.matrix_world.inverted()
    for side, center in centers.items():
        bone = rig.data.edit_bones.new(EYE_BONES[side])
        bone.head = to_rig @ center
        bone.tail = to_rig @ (center + Vector((0.0, -0.03, 0.0)))
        bone.parent = rig.data.edit_bones["head"]
        bone.use_deform = True
    bpy.ops.object.mode_set(mode='OBJECT')

    object_service = build.mpfb("services.objectservice", "ObjectService")
    eyes = object_service.find_object_of_type_amongst_nearest_relatives(rig, "Eyes")
    if eyes is not None:
        for group in list(eyes.vertex_groups):
            eyes.vertex_groups.remove(group)
        world = eyes.matrix_world
        middle = (centers["l"].x + centers["r"].x) / 2.0
        sides = {"l": [], "r": []}
        for vert in eyes.data.vertices:
            # The character's left is +X.
            sides["l" if (world @ vert.co).x > middle else "r"].append(vert.index)
        for side, indices in sides.items():
            eyes.vertex_groups.new(name=EYE_BONES[side]).add(indices, 1.0, 'REPLACE')
    return sorted(EYE_BONES.values())


# -----------------------------------------------------------------------------
# Checks

def _offsets(basemesh):
    """``{name: (N, 3) movement of every vertex}`` for each face shape key, in metres."""
    blocks = basemesh.data.shape_keys.key_blocks
    count = len(basemesh.data.vertices)

    def read(block):
        co = np.empty(count * 3, dtype=np.float32)
        block.data.foreach_get("co", co)
        return co.reshape(-1, 3)

    basis = read(blocks[0])
    scale = basemesh.matrix_world.to_scale().x
    return basis * scale, {
        block.name: (read(block) - basis) * scale for block in blocks[1:] if is_face_key(block.name)
    }


def _mirror_map(basis):
    """For every vertex, the index of the vertex at its mirrored position."""
    tree = kdtree.KDTree(len(basis))
    for index, co in enumerate(basis):
        tree.insert(co, index)
    tree.balance()
    return np.array([tree.find((-co[0], co[1], co[2]))[1] for co in basis], dtype=np.int64)


def _lip_halves(basemesh, basis, opener):
    """
    Vertices of the upper and the lower lip at the middle of the mouth.

    ``opener`` is the movement of a shape that opens the mouth: the lower lip is
    what travels with the jaw, the upper lip is what stays.
    """
    group = basemesh.vertex_groups.get("lips")
    if group is None:
        return None
    lips = np.array([
        v.index for v in basemesh.data.vertices if any(g.group == group.index for g in v.groups)
    ], dtype=np.int64)
    lips = lips[np.abs(basis[lips, 0]) < 0.006]
    if len(lips) < 4:
        return None
    travel = np.linalg.norm(opener[lips], axis=1)
    upper, lower = lips[travel < 0.2 * travel.max()], lips[travel > 0.5 * travel.max()]
    return (upper, lower) if len(upper) and len(lower) else None


def _lip_opening(basis, halves, offset):
    """How far apart the lips are compared with the rest face, in metres. Negative is pressed."""
    upper, lower = halves
    moved = basis + offset
    opening = moved[upper, 2].mean() - moved[lower, 2].mean()
    return float(opening - (basis[upper, 2].mean() - basis[lower, 2].mean()))


def check(basemesh):
    """
    Measure every face shape. ``failures`` lists what a render of the neutral face would not show.

    Checked: each shape moves something and nothing flies off; left and right shapes mirror
    each other; lips meet for P/B/M and part for "aa"; no two shapes are the same shape.
    """
    if not basemesh.data.shape_keys:
        return {"keys": {}, "failures": ["the mesh has no shape keys"], "notes": [], "passed": False}
    basis, offsets = _offsets(basemesh)
    failures, notes = [], []
    keys = {}
    for name, offset in offsets.items():
        reach = float(np.linalg.norm(offset, axis=1).max())
        keys[name] = {"max_mm": round(reach * 1000.0, 2)}
        # Silence is the rest face by definition.
        if reach < 0.0002 and name != "viseme_sil":
            failures.append("{:s} moves nothing".format(name))
        elif reach > 0.06:
            failures.append("{:s} moves a vertex {:.0f} mm, which is not a face shape".format(name, reach * 1000))

    missing = [name for name in VISEMES if name not in offsets]
    if missing:
        failures.append("missing mouth shapes: " + ", ".join(missing))

    mirror = _mirror_map(basis)
    flip = np.array((-1.0, 1.0, 1.0), dtype=np.float32)
    for name, offset in offsets.items():
        if not name.endswith("Left"):
            continue
        other = offsets.get(name[:-4] + "Right")
        if other is None:
            failures.append("{:s} has no Right partner".format(name))
            continue
        size = float(np.abs(offset).mean())
        if size > 0.0:
            error = float(np.abs(offset - other[mirror] * flip).mean()) / size
            keys[name]["mirror_error"] = round(error, 3)
            if error > 0.25:
                failures.append("{:s} and its Right partner differ by {:.0f}%".format(name, error * 100))

    halves = _lip_halves(basemesh, basis, offsets["viseme_aa"]) if "viseme_aa" in offsets else None
    if halves is not None:
        openings = {name: _lip_opening(basis, halves, offsets[name])
                    for name in ("viseme_PP", "viseme_aa", "viseme_O", "jawOpen") if name in offsets}
        for name, opening in openings.items():
            keys[name]["lip_opening_mm"] = round(opening * 1000.0, 2)
        if openings["viseme_PP"] > 0.0005:
            failures.append("lips do not close for P/B/M: {:.1f} mm apart".format(openings["viseme_PP"] * 1000))
        if openings["viseme_aa"] < 0.003:
            failures.append("mouth opens only {:.1f} mm for \"aa\"".format(openings["viseme_aa"] * 1000))

    names = [name for name in offsets if keys[name]["max_mm"] > 0.2]
    flat = np.array([offsets[name].ravel() for name in names])
    if len(flat):
        flat /= np.linalg.norm(flat, axis=1, keepdims=True)
        alike = flat @ flat.T
        for i, j in zip(*np.where(np.triu(alike, 1) > 0.98)):
            pair = "{:s} and {:s} are the same shape".format(names[i], names[j])
            # A mouth shape may well equal an expression unit ("oo" is a pucker); two of a kind may not.
            if names[i].startswith("viseme_") != names[j].startswith("viseme_"):
                notes.append(pair)
            else:
                failures.append(pair)
    return {"keys": keys, "failures": failures, "notes": notes, "passed": not failures}
