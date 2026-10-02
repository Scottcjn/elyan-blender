# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Give an existing head the face shapes of a MakeHuman head.

For characters that were not built here: a hand-made or purchased head with a
few shape keys of its own gets the full expression set, without touching the
shapes it already has.

How: both heads have some shapes in common (a blink, an open mouth, brows).
Where those shapes move the most marks the same anatomical spot on both, which
gives landmarks without anyone clicking on a face. The donor is bent through
those landmarks onto the target, settled onto its surface, and each target
vertex then takes the movement of the donor surface nearest to it. The match is
made with both mouths open, so a lower lip never follows an upper one.
"""

import math

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# Shapes a target must already have, by role. Values are its own names for them.
ROLES = ("blink_l", "blink_r", "open", "brow_in", "brow_out_l", "brow_out_r")
DONOR_ROLES = {
    "blink_l": "eyeBlinkLeft", "blink_r": "eyeBlinkRight", "open": "viseme_aa",
    "brow_in": "browInnerUp", "brow_out_l": "browOuterUpLeft", "brow_out_r": "browOuterUpRight",
}

EYE_RADIUS = 0.022
REACH = 0.012   # beyond this distance from the donor surface a vertex takes no movement


def _length(vectors):
    return np.linalg.norm(vectors, axis=-1)


def landmarks(co, offsets, skin, eyes):
    """
    Anatomical points of a head, found from where its own shapes move.

    ``co`` is (N, 3) world positions; ``offsets`` maps each role in ``ROLES`` to an
    (N, 3) movement; ``skin`` is a mask of face-skin vertices; ``eyes`` maps "l"/"r"
    to eye centres. The character faces -Y and its left is +X.
    """
    index = np.flatnonzero(skin)
    points = co[index]

    def strongest(role, where=None):
        travel = _length(offsets[role][index])
        if where is not None:
            travel = np.where(where, travel, -1.0)
        return points[int(travel.argmax())]

    result = {"eye_l": np.array(eyes["l"]), "eye_r": np.array(eyes["r"])}
    for side in "lr":
        travel = _length(offsets["blink_" + side][index])
        near = _length(points - result["eye_" + side]) < EYE_RADIUS
        lid = points[near & (travel > 0.4 * travel[near].max())]
        result["lid_" + side] = strongest("blink_" + side, near)
        # Ends of the eye opening, toward the nose and toward the ear.
        low, high = lid[lid[:, 0].argmin()], lid[lid[:, 0].argmax()]
        inner, outer = (low, high) if side == "l" else (high, low)
        result["eye_in_" + side], result["eye_out_" + side] = inner, outer
    result["brow_in"] = strongest("brow_in", np.abs(points[:, 0]) < 0.012)
    result["brow_l"], result["brow_r"] = strongest("brow_out_l"), strongest("brow_out_r")

    eye_z = (result["eye_l"][2] + result["eye_r"][2]) / 2.0
    middle = np.abs(points[:, 0]) < 0.004
    opening = _length(offsets["open"][index])
    result["chin"] = strongest("open", middle)
    # The front of the face along the centre line, between chin and eyes.
    band = middle & (points[:, 2] > result["chin"][2]) & (points[:, 2] < eye_z)
    front = band & (points[:, 1] < points[band][:, 1].min() + 0.03)
    result["nose"] = points[np.where(band, -points[:, 1], -np.inf).argmax()]
    lips = front & (points[:, 2] < result["nose"][2] - 0.005)
    moving = lips & (opening > 0.5 * opening.max())
    still = lips & (opening < 0.2 * opening.max())
    result["lip_low"] = points[np.where(moving, points[:, 2], -np.inf).argmax()]
    result["lip_up"] = points[np.where(still, points[:, 2], np.inf).argmin()]
    mouth_z = (result["lip_low"][2] + result["lip_up"][2]) / 2.0
    # Mouth corners are the ends of the slit: the only place where skin that follows the
    # jaw lies right beside skin that does not. On the cheeks that change is gradual.
    wide = (
        (points[:, 2] > result["chin"][2]) & (points[:, 2] < result["nose"][2])
        & (points[:, 1] < result["lip_up"][1] + 0.035)
    )
    follows = points[wide & (opening > 0.5 * opening.max())]
    stays = points[wide & (opening < 0.2 * opening.max())]
    gap = _length(stays[:, None, :] - follows[None, :, :]).min(axis=1)
    slit = stays[gap < 0.004]
    result["mouth_l"], result["mouth_r"] = slit[slit[:, 0].argmax()], slit[slit[:, 0].argmin()]

    # The skull itself, so the fit does not hinge on the face alone.
    upper = points[points[:, 2] > mouth_z]
    result["top"] = upper[upper[:, 2].argmax()]
    result["back"] = upper[upper[:, 1].argmax()]
    level = upper[np.abs(upper[:, 2] - eye_z) < 0.01]
    result["side_l"], result["side_r"] = level[level[:, 0].argmax()], level[level[:, 0].argmin()]
    return result


class Warp:
    """Smooth bend of space that carries one set of points exactly onto another."""

    def __init__(self, source, target):
        source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
        count = len(source)
        kernel = _length(source[:, None, :] - source[None, :, :])
        affine = np.hstack((np.ones((count, 1)), source))
        system = np.zeros((count + 4, count + 4))
        system[:count, :count] = kernel + np.eye(count) * 1e-9
        system[:count, count:] = affine
        system[count:, :count] = affine.T
        right = np.zeros((count + 4, 3))
        right[:count] = target
        solution = np.linalg.solve(system, right)
        self.source, self.weights, self.affine = source, solution[:count], solution[count:]

    def __call__(self, points):
        points = np.asarray(points, dtype=np.float64)
        out = np.empty_like(points)
        for start in range(0, len(points), 4096):
            chunk = points[start:start + 4096]
            kernel = _length(chunk[:, None, :] - self.source[None, :, :])
            out[start:start + 4096] = (
                kernel @ self.weights + self.affine[0] + chunk @ self.affine[1:]
            )
        return out


def _tree(co, triangles):
    return BVHTree.FromPolygons([Vector(p) for p in co.tolist()], triangles.tolist(), all_triangles=True)


def _settle(co, triangles, target_tree, rounds=3):
    """Pull a fitted donor onto the target's surface, keeping the correction smooth."""
    neighbours = [set() for _ in range(len(co))]
    for a, b, c in triangles.tolist():
        neighbours[a].update((b, c))
        neighbours[b].update((a, c))
        neighbours[c].update((a, b))
    used = np.unique(triangles)
    co = co.copy()
    for _ in range(rounds):
        pull = np.zeros_like(co)
        for vertex in used.tolist():
            hit = target_tree.find_nearest(Vector(co[vertex]), 0.02)
            if hit[0] is not None:
                pull[vertex] = np.array(hit[0]) - co[vertex]
        # One vertex snapping to the wrong lip would fold the mesh; neighbours vote first.
        smooth = pull.copy()
        for vertex in used.tolist():
            ring = list(neighbours[vertex])
            if ring:
                smooth[vertex] = 0.5 * pull[vertex] + 0.5 * pull[ring].mean(axis=0)
        co[used] += 0.7 * smooth[used]
    return co


def _barycentric(point, a, b, c):
    v0, v1, v2 = b - a, c - a, point - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) < 1e-20:
        return np.array((1.0, 0.0, 0.0))
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    return np.clip(np.array((1.0 - v - w, v, w)), 0.0, 1.0)


def correspondence(target_co, donor_co, donor_triangles):
    """
    For each target vertex: the donor triangle it rides on, where on it, and how firmly.

    Call this with both heads in an open-mouthed pose. At rest the lips touch, and
    "nearest surface" would then let a lower lip follow the upper one; with the
    mouth open each lip is nearest only to itself.
    """
    tree = _tree(donor_co, donor_triangles)
    triangle = np.zeros((len(target_co), 3), dtype=np.int64)
    weight = np.zeros((len(target_co), 3))
    firmness = np.zeros(len(target_co))
    for vertex, point in enumerate(target_co):
        location, _normal, face, distance = tree.find_nearest(Vector(point))
        if location is None:
            continue
        corners = donor_triangles[face]
        triangle[vertex] = corners
        weight[vertex] = _barycentric(np.array(location), *donor_co[corners])
        weight[vertex] /= weight[vertex].sum()
        # Full strength on the surface, fading out by REACH.
        firmness[vertex] = 0.5 + 0.5 * math.cos(math.pi * min(distance / REACH, 1.0))
    return triangle, weight, firmness


def relax(movement, edges, rounds=1):
    """Even out vertex-to-vertex noise in a movement field along the mesh's edges."""
    a, b = edges[:, 0], edges[:, 1]
    count = np.zeros(len(movement))
    np.add.at(count, a, 1.0)
    np.add.at(count, b, 1.0)
    count = np.maximum(count, 1.0)[:, None]
    for _ in range(rounds):
        total = np.zeros_like(movement)
        np.add.at(total, a, movement[b])
        np.add.at(total, b, movement[a])
        movement = 0.5 * movement + 0.5 * total / count
    return movement


def carry(donor_offsets, triangle, weight, firmness):
    """Movement of every target vertex for one donor shape."""
    moved = (donor_offsets[triangle] * weight[:, :, None]).sum(axis=1)
    return moved * firmness[:, None]


def jaw_turn(chin, movement, hinge):
    """Angle (radians) the jaw swings about a side-to-side axis through ``hinge`` to move the chin so."""
    before, after = chin - hinge, chin + movement - hinge
    return math.atan2(after[2], -after[1]) - math.atan2(before[2], -before[1])


def swing(points, hinge, angle):
    """Rotate points about the X axis through ``hinge``."""
    cos, sin = math.cos(angle), math.sin(angle)
    relative = points - hinge
    out = relative.copy()
    out[:, 1] = relative[:, 1] * cos + relative[:, 2] * sin
    out[:, 2] = -relative[:, 1] * sin + relative[:, 2] * cos
    return out + hinge - points


# -----------------------------------------------------------------------------
# Whole transfer

def _world(ob):
    """World positions, world movement of every shape key (names stripped), and the mesh's own frame."""
    count = len(ob.data.vertices)
    matrix = np.array(ob.matrix_world, dtype=np.float64)
    rotation, origin = matrix[:3, :3], matrix[:3, 3]

    def read(block):
        co = np.empty(count * 3, dtype=np.float32)
        block.data.foreach_get("co", co)
        return co.reshape(-1, 3).astype(np.float64)

    blocks = ob.data.shape_keys.key_blocks
    basis = read(blocks[0])
    offsets = {block.name.strip(): (read(block) - basis) @ rotation.T for block in blocks[1:]}
    return basis @ rotation.T + origin, offsets, basis, rotation


def _material_masks(ob):
    """Vertex masks by material name, ignoring Blender's ".001" suffixes."""
    masks = {}
    for face in ob.data.polygons:
        name = ob.data.materials[face.material_index].name.split(".")[0]
        masks.setdefault(name, np.zeros(len(ob.data.vertices), dtype=bool))[list(face.vertices)] = True
    return masks


def _triangles(ob, mask):
    result = []
    for face in ob.data.polygons:
        corners = list(face.vertices)
        if all(mask[index] for index in corners):
            result.extend((corners[0], corners[i], corners[i + 1]) for i in range(1, len(corners) - 1))
    return np.array(result, dtype=np.int64)


def _write(ob, name, basis, rotation, movement):
    blocks = ob.data.shape_keys.key_blocks
    block = blocks.get(name) or ob.shape_key_add(name=name)
    block.data.foreach_set("co", (basis + movement @ np.linalg.inv(rotation).T).astype(np.float32).ravel())
    block.value = 0.0


def apply(head, rig, donor, donor_eyes, roles, eye_bones, jaw_bone,
          skin=("Body",), soft=("Mouth",), hard=("Teeth", "Std_Tongue"), followers=(), aliases=None):
    """
    Add the donor's expression shapes to ``head``, keeping every shape it already has.

    - ``donor``: an MPFB basemesh with the full face loaded; ``donor_eyes`` its eye centres.
    - ``roles``: the head's own names for the shapes in ``ROLES``.
    - ``eye_bones`` {"l", "r"} and ``jaw_bone``: bone names on ``rig``.
    - ``skin`` / ``soft`` / ``hard``: material names of face skin, of the mouth lining, and
      of teeth and tongue, which are carried rigidly with the jaw instead of stretched.
    - ``followers``: other meshes that sit on the face (lashes, brows) and must move with it.
    - ``aliases``: extra names to write for a shape, {new name: donor shape}.

    Returns a report with the movement each new shape produces.
    """
    from . import build, face

    co, offsets, basis, rotation = _world(head)
    masks = _material_masks(head)

    def union(names):
        out = np.zeros(len(co), dtype=bool)
        for name in names:
            out |= masks.get(name, False)
        return out

    skin_mask, soft_mask, hard_mask = union(skin), union(skin) | union(soft), union(hard)
    eyes = {side: np.array(rig.matrix_world @ rig.data.bones[eye_bones[side]].head_local) for side in "lr"}
    marks = landmarks(co, {role: offsets[name] for role, name in roles.items()}, skin_mask, eyes)

    donor_co = build._mixed_coordinates(donor).astype(np.float64)
    _basis, donor_offsets = face._offsets(donor)
    donor_offsets = {name: value.astype(np.float64) for name, value in donor_offsets.items()}
    body = donor.vertex_groups["body"].index
    donor_skin = np.array([any(g.group == body for g in v.groups) for v in donor.data.vertices])
    donor_eyes = {side: np.array(center) for side, center in donor_eyes.items()}
    donor_skin &= donor_co[:, 2] > donor_eyes["l"][2] - 0.17
    donor_marks = landmarks(
        donor_co, {role: donor_offsets[name] for role, name in DONOR_ROLES.items()}, donor_skin, donor_eyes)

    names = sorted(marks)
    region = np.flatnonzero(donor_skin)
    donor_triangles = _triangles(donor, donor_skin)
    skin_triangles = _triangles(head, skin_mask)
    warp = Warp([donor_marks[n] for n in names], [marks[n] for n in names])
    fit = donor_co.copy()
    fit[region] = warp(donor_co[region])
    fit = _settle(fit, donor_triangles, _tree(co, skin_triangles))
    carried = {}
    for name, movement in donor_offsets.items():
        moved = np.zeros_like(donor_co)
        moved[region] = warp(donor_co[region] + movement[region]) - warp(donor_co[region])
        carried[name] = moved

    # Open both mouths by the same amount at the chin and line them up again in that pose.
    skin_index = np.flatnonzero(skin_mask)
    at = {n: int(skin_index[_length(co[skin_index] - marks[n]).argmin()]) for n in names}
    donor_at = {n: int(region[_length(donor_co[region] - donor_marks[n]).argmin()]) for n in names}
    opener, donor_opener = offsets[roles["open"]], carried[DONOR_ROLES["open"]]
    amount = float(_length(opener[at["chin"]]) / _length(donor_opener[donor_at["chin"]]))
    opened, donor_opened = co + opener, fit + amount * donor_opener
    floating = {"eye_l", "eye_r"}
    warp_open = Warp(
        [marks[n] if n in floating else donor_opened[donor_at[n]] for n in names],
        [marks[n] if n in floating else opened[at[n]] for n in names],
    )
    donor_opened[region] = warp_open(donor_opened[region])
    donor_opened = _settle(donor_opened, donor_triangles, _tree(opened, skin_triangles))
    triangle, weight, firmness = correspondence(opened, donor_opened, donor_triangles)
    firmness = np.where(soft_mask, firmness, 0.0)

    edges = np.array([edge.vertices[:] for edge in head.data.edges], dtype=np.int64)
    edges = edges[soft_mask[edges].all(axis=1)]
    travel = _length(opener)
    lower = hard_mask & (travel > 0.3 * travel[hard_mask].max()) if hard_mask.any() else hard_mask
    hinge = np.array(rig.matrix_world @ rig.data.bones[jaw_bone].head_local)
    chin = at["chin"]

    wanted = {name: name for name in donor_offsets if not name.startswith("viseme_") and name != "tongueOut"}
    wanted.update(aliases or {})
    report = {}
    for name, source in sorted(wanted.items()):
        if name in offsets:
            continue
        movement = relax(carry(carried[source], triangle, weight, firmness), edges)
        # Teeth and tongue follow only shapes that really move the jaw.
        if source == "jawOpen":
            movement[lower] = swing(co[lower], hinge, jaw_turn(co[chin], movement[chin], hinge))
        elif source in {"jawLeft", "jawRight", "jawForward"}:
            movement[lower] = movement[chin]
        _write(head, name, basis, rotation, movement)
        report[name] = {"max_mm": round(float(_length(movement).max()) * 1000.0, 2)}

    for follower in followers:
        f_co, f_offsets, f_basis, f_rotation = _world(follower)
        f_triangle, f_weight, f_firmness = correspondence(f_co, donor_opened, donor_triangles)
        for name, source in sorted(wanted.items()):
            if name in f_offsets:
                continue
            movement = carry(carried[source], f_triangle, f_weight, f_firmness)
            # Most shapes do not reach the lashes; an empty shape key is only weight.
            if _length(movement).max() > 0.0003:
                _write(follower, name, f_basis, f_rotation, movement)
                report[name][follower.name] = round(float(_length(movement).max()) * 1000.0, 2)
    return {
        "shapes": report,
        "landmarks": {name: [round(float(v), 4) for v in point] for name, point in marks.items()},
        "open_scale": round(amount, 3),
        "unreached_vertices": int((firmness[soft_mask] < 0.05).sum()),
    }
