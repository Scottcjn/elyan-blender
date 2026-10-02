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
_CANONICAL = {renamed: name for name, renamed in VRCHAT_NAMES.items()}

# Unity drops a blend shape that moves nothing when it imports an FBX, and VRChat needs all
# fifteen in its list. Silence gets this much movement inside the mouth, where it cannot be seen.
SILENCE_NUDGE = 0.0001

# Gates fail only what cannot be a talking face; everything else is reported as a number.
_SILENCE_MAX = 0.0002         # more than this and "sil" is no longer the rest face
_DISTINCT_FROM_SIL = 0.0003   # RMS over the mouth; the faintest shape measured is 0.8 mm
_DISTINCT_FROM_OTHER = 0.00025
_ENERGY_ABOVE_NOSE = 0.25     # share of a mouth shape's movement allowed above the nose
_BLINK_REMAINING = 0.5        # share of the eye's opening a blink may leave
_POKE_AHEAD = 0.001           # teeth or tongue further forward than the lips


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
    # An export for VRChat carries the mouth shapes under VRChat's names; they are the same shapes.
    return basis * scale, {
        _CANONICAL.get(block.name, block.name): (read(block) - basis) * scale
        for block in blocks[1:] if is_face_key(block.name)
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


def _members(ob, group_name):
    """Indices of the vertices in a vertex group, or None when the mesh has no such group."""
    group = ob.vertex_groups.get(group_name)
    if group is None:
        return None
    return np.array([
        v.index for v in ob.data.vertices if any(g.group == group.index for g in v.groups)
    ], dtype=np.int64)


def _distinct(offsets, skin, keys, failures):
    """
    How far each mouth shape is from the rest face and from its nearest neighbour.

    RMS in metres over the skin vertices any mouth shape moves, so a large quiet
    head does not dilute a small mouth.
    """
    names = [name for name in VISEMES if name in offsets and name != "viseme_sil"]
    if not names:
        return
    active = np.zeros(len(skin), dtype=bool)
    for name in names:
        active |= np.linalg.norm(offsets[name], axis=1) > 0.0001
    active &= skin
    if not active.any():
        return
    stack = np.array([offsets[name][active] for name in names])
    for i, name in enumerate(names):
        from_sil = float(np.sqrt((stack[i] ** 2).sum(axis=1).mean()))
        keys[name]["rms_from_sil_mm"] = round(from_sil * 1000.0, 3)
        if from_sil < _DISTINCT_FROM_SIL:
            failures.append("{:s} is {:.2f} mm RMS from silence, which nobody would see".format(
                name, from_sil * 1000.0))
        others = [
            (float(np.sqrt(((stack[i] - stack[j]) ** 2).sum(axis=1).mean())), names[j])
            for j in range(len(names)) if j != i
        ]
        if others:
            distance, nearest = min(others)
            keys[name]["nearest"] = nearest
            keys[name]["rms_from_nearest_mm"] = round(distance * 1000.0, 3)
            # Each close pair is reported once.
            if distance < _DISTINCT_FROM_OTHER and name < nearest:
                failures.append("{:s} and {:s} differ by only {:.2f} mm RMS".format(
                    name, nearest, distance * 1000.0))


def _energy_above(offsets, co, skin, nose, keys, failures):
    """Share of each mouth shape's squared movement that is above the nose."""
    upper = skin & (co[:, 2] > nose)
    for name in VISEMES:
        if name not in offsets or name == "viseme_sil":
            continue
        energy = (offsets[name] ** 2).sum(axis=1)
        total = float(energy[skin].sum())
        if total <= 0.0:
            continue
        share = float(energy[upper].sum()) / total
        keys[name]["above_nose"] = round(share, 4)
        if share > _ENERGY_ABOVE_NOSE:
            failures.append("{:s} puts {:.0f}% of its movement above the nose".format(name, share * 100.0))


def _blink(offsets, co, skin, centers, keys, failures):
    """
    Opening left between the eyelids by a blink, against the rest face.

    Taken in a narrow strip in front of the eye's centre: the lowest skin above
    it and the highest skin below it are the lid edges.
    """
    for side, name in (("l", "eyeBlinkLeft"), ("r", "eyeBlinkRight")):
        if name not in offsets or side not in centers:
            continue
        center = np.array(centers[side], dtype=np.float32)
        strip = np.where(
            skin & (np.abs(co[:, 0] - center[0]) < 0.006) & (np.abs(co[:, 2] - center[2]) < 0.02)
            & (co[:, 1] < center[1]))[0]
        upper, lower = strip[co[strip, 2] > center[2]], strip[co[strip, 2] <= center[2]]
        if not len(upper) or not len(lower):
            continue
        rest = float(co[upper, 2].min() - co[lower, 2].max())
        moved = co + offsets[name]
        # Lids that overlap are closed; the overlap itself is not an opening.
        closed = max(0.0, float(moved[upper, 2].min() - moved[lower, 2].max()))
        keys[name]["eye_opening_rest_mm"] = round(rest * 1000.0, 2)
        keys[name]["eye_opening_blink_mm"] = round(closed * 1000.0, 2)
        if rest > 0.0 and closed > _BLINK_REMAINING * rest:
            failures.append("{:s} leaves the eye {:.1f} mm open of {:.1f} mm".format(
                name, closed * 1000.0, rest * 1000.0))


def _inner_mouth(basemesh):
    """``{kind: (positions, offsets)}`` for the teeth and tongue that belong to a person."""
    rig = basemesh.parent
    if rig is None:
        return {}
    properties = build.mpfb("entities.objectproperties", "GeneralObjectProperties")
    found = {}
    for ob in rig.children_recursive:
        if ob.type != 'MESH' or not ob.data.shape_keys:
            continue
        kind = properties.get_value("object_type", entity_reference=ob)
        if kind in {"Teeth", "Tongue"}:
            found[kind.lower()] = (build._mixed_coordinates(ob), _offsets(ob)[1])
    return found


def _poke(basemesh, offsets, co, lips, keys, failures):
    """
    How far the teeth and tongue reach in front of the lips at each mouth shape.

    The character faces -Y. Negative is behind the frontmost point of the lips,
    where they belong; this does not catch a tooth through a cheek.
    """
    for kind, (part, part_offsets) in _inner_mouth(basemesh).items():
        for name in VISEMES:
            if name not in offsets:
                continue
            front = float((co[lips, 1] + offsets[name][lips, 1]).min())
            moved = part[:, 1] + part_offsets[name][:, 1] if name in part_offsets else part[:, 1]
            ahead = front - float(moved.min())
            keys[name][kind + "_ahead_mm"] = round(ahead * 1000.0, 2)
            if ahead > _POKE_AHEAD:
                failures.append("{:s}: {:s} {:.1f} mm in front of the lips".format(name, kind, ahead * 1000.0))


def check(basemesh):
    """
    Measure every face shape. ``failures`` lists what a render of the neutral face would not show.

    Checked: each shape moves something and nothing flies off; left and right shapes mirror
    each other; lips meet for P/B/M and part for "aa"; no two shapes are the same shape;
    every mouth shape differs from silence and from the others; mouth shapes stay below the
    nose; a blink closes the eye; teeth and tongue stay behind the lips.
    """
    if not basemesh.data.shape_keys:
        return {"keys": {}, "failures": ["the mesh has no shape keys"], "notes": [], "passed": False}
    basis, offsets = _offsets(basemesh)
    failures, notes = [], []
    keys = {}
    for name, offset in offsets.items():
        reach = float(np.linalg.norm(offset, axis=1).max())
        keys[name] = {"max_mm": round(reach * 1000.0, 2)}
        # Silence is the rest face by definition; an export may nudge it so that Unity keeps it.
        if name == "viseme_sil":
            if reach > _SILENCE_MAX:
                failures.append("viseme_sil moves a vertex {:.2f} mm; silence is the rest face".format(
                    reach * 1000.0))
        elif reach < 0.0002:
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

    # The gates below need the person as built: body shape applied, landmarks still on the mesh.
    # An exported mesh has lost its helper groups and skips them.
    body = _members(basemesh, "body")
    lips = _members(basemesh, "lips")
    centers = build.eye_centers(basemesh)
    if body is not None and len(body):
        co = build._mixed_coordinates(basemesh)
        skin = np.zeros(len(co), dtype=bool)
        skin[body] = True
        _distinct(offsets, skin, keys, failures)
        if len(centers) == 2 and lips is not None and len(lips):
            nose = (centers["l"].z + centers["r"].z + 2.0 * float(co[lips, 2].mean())) / 4.0
            _energy_above(offsets, co, skin, nose, keys, failures)
        _blink(offsets, co, skin, centers, keys, failures)
        if lips is not None and len(lips):
            _poke(basemesh, offsets, co, lips, keys, failures)
    else:
        notes.append("no landmark groups on this mesh; distinctness, blink and teeth gates skipped")

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
