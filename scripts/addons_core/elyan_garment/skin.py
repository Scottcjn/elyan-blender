# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
``weights``: give a garment the body's skinning.

Each garment vertex takes the weights of the nearest point on the body's
surface, interpolated across that triangle. The Data Transfer modifier does the
same thing but needs the right context to apply; this needs nothing but the two
meshes, so it runs the same headless and from a button.
"""

import numpy as np

from . import surface

# A weight this small moves a vertex by less than a float can hold; it only uses up one of the four slots.
NEGLIGIBLE = 1e-6
# Closer than this to the body's centre plane a vertex has no side to be wrong about.
SIDE_EPSILON = 1e-4
# Over this change of weights along a single edge the two ends mostly follow different bones.
STEP_WARNING = 0.5
AUTO_PASSES = 8


def _rig_of(body, rig):
    import bpy
    if isinstance(rig, str):
        found = bpy.data.objects.get(rig)
        if found is None:
            raise ValueError("rig: no object named {!r}".format(rig))
        rig = found
    if rig is None:
        for modifier in body.modifiers:
            if modifier.type == 'ARMATURE' and modifier.object is not None:
                rig = modifier.object
                break
        else:
            raise ValueError("{!r} has no Armature modifier; say which rig with rig=".format(body.name))
    if rig.type != 'ARMATURE':
        raise ValueError("rig: {!r} is not an armature".format(rig.name))
    return rig


def _body_weights(body, mesh, bones):
    """Weights of ``mesh`` (the body as evaluated) as a (vertices x bones) array."""
    columns = {body.vertex_groups[name].index: column for column, name in enumerate(bones)}
    table = np.zeros((len(mesh.vertices), len(bones)), dtype=np.float64)
    for vert in mesh.vertices:
        for element in vert.groups:
            column = columns.get(element.group)
            if column is not None:
                table[vert.index, column] = element.weight
    return table


def _read_back(garment, bones):
    """Influence count and weight sum per vertex, as stored: what was written is not what is trusted."""
    wanted = {garment.vertex_groups[name].index for name in bones if name in garment.vertex_groups}
    count = np.zeros(len(garment.data.vertices), dtype=np.int64)
    total = np.zeros(len(garment.data.vertices), dtype=np.float64)
    for vert in garment.data.vertices:
        for element in vert.groups:
            if element.group in wanted and element.weight > 0.0:
                count[vert.index] += 1
                total[vert.index] += element.weight
    return count, total


def _smooth(table, pairs, amount, passes):
    """
    Blend each vertex's weights toward the mean of its neighbours, ``amount`` (0..1 per vertex) at a time.

    Cloth lying on the body has ``amount`` near zero and keeps the body's
    weights. Cloth standing off it has no good nearest point, and without this
    a skirt's two halves each follow their own leg and shear apart down the middle.
    """
    degree = np.bincount(pairs.ravel(), minlength=len(table)).astype(np.float64)
    degree[degree == 0.0] = 1.0
    for _pass in range(passes):
        total = np.zeros_like(table)
        np.add.at(total, pairs[:, 0], table[pairs[:, 1]])
        np.add.at(total, pairs[:, 1], table[pairs[:, 0]])
        table = table + amount[:, None] * (total / degree[:, None] - table)
    return table


def weights(garment, body, rig=None, limit=4, region=None, same_side=False, max_distance=None,
            region_min=0.05, smooth=None, smooth_distance=0.05):
    """
    Skin ``garment`` like ``body``: an Armature modifier on the body's rig and the body's bone weights.

    Each garment vertex takes the weights at the nearest point of the body's
    surface (as shown, unposed; interpolated across the triangle), keeps the ``limit``
    largest and is normalised to sum to one. Only deform bones are transferred.
    The garment's other vertex groups (pins, masks) are kept.

    ``region``: one body vertex group, or a list of them (bone names are groups),
    to take weights only from that part of the body: a sleeve from the arm bones
    so that it cannot stick to the ribs. A body vertex belongs to the region when
    those groups hold at least ``region_min`` of weight there.
    ``same_side``: take weights only from the body's own half (by the body's X),
    so a skirt panel cannot stick to the far leg.
    ``max_distance``: refuse if any vertex is farther than this from its source.
    ``smooth``: passes that blend weights between neighbouring garment vertices,
    in proportion to how far the cloth stands off the body (fully from
    ``smooth_distance``); for skirts and anything else that hangs free. Left
    out, 8 passes are used when the plain result would shear (``weight_step``
    over 0.5) and none otherwise; 0 switches it off.

    Refuses, writing nothing, when a vertex would get no weight at all.
    Returns the report: ``unweighted``, ``clamped`` (vertices that had more than
    ``limit`` influences), ``max_distance``, ``wrong_side`` (source point on the
    other side of the body's centre plane: a warning), ``weight_step`` (the
    largest change of weights along one garment edge, 0..1: near 1 means two
    neighbours follow different bones and the cloth will tear there when posed),
    ``bones``.
    """
    garment = surface.writable(surface.mesh_object(garment, "garment"))
    body = surface.mesh_object(body, "body")
    rig = _rig_of(body, rig)
    limit = int(limit)
    if limit < 1:
        raise ValueError("limit must be at least 1")

    deform = [bone.name for bone in rig.data.bones if bone.use_deform]
    bones = [name for name in deform if name in body.vertex_groups]
    if not bones:
        raise ValueError("{!r} has no vertex groups named after deform bones of {!r}".format(body.name, rig.name))
    region = surface.names(region)
    # The body as shown but unposed: what a Mask modifier hides is not skin to take weights from.
    with surface.at_rest(body):
        verts, mesh = surface.evaluated(body)
        tris = surface.triangles(mesh)
        table = _body_weights(body, mesh, bones)
        inside = surface.group_weight(body, region, mesh) >= float(region_min) if region else None
    # A triangle with an unweighted corner would hand out weights that sum to less than one, or nothing.
    weighted = table.sum(axis=1) > 0.0
    usable = weighted[tris].all(axis=1)
    if region:
        usable &= inside[tris].any(axis=1)
    if not usable.any():
        raise ValueError("no weighted body faces to take weights from{:s}".format(
            " in region {:s}".format(", ".join(region)) if region else ""))

    points = surface.rest_world(garment)
    # Sides are judged in the body's own space, where its centre plane is x = 0.
    inverse = np.linalg.inv(surface.matrix(body))

    def side_of(world):
        return world @ inverse[0, :3] + inverse[0, 3]

    point_side = side_of(points)

    location = np.empty_like(points)
    source = np.empty((len(points), len(bones)), dtype=np.float64)
    if same_side:
        centre = side_of(verts)[tris]
        halves = (
            (point_side > SIDE_EPSILON, usable & (centre.max(axis=1) >= 0.0)),
            (point_side < -SIDE_EPSILON, usable & (centre.min(axis=1) <= 0.0)),
            (np.abs(point_side) <= SIDE_EPSILON, usable),
        )
    else:
        halves = ((np.ones(len(points), dtype=bool), usable),)
    for chosen, faces in halves:
        if not chosen.any():
            continue
        if not faces.any():
            raise ValueError("same_side: the body has no weighted faces on one side")
        part = surface.Surface(verts, tris[faces])
        found, _normal, tri, _distance = part.nearest(points[chosen])
        location[chosen] = found
        source[chosen] = part.interpolate(table, tri, part.barycentric(found, tri))

    distance = np.linalg.norm(points - location, axis=1)
    source_side = side_of(location)
    wrong = (np.abs(point_side) > SIDE_EPSILON) & (np.abs(source_side) > SIDE_EPSILON) & (
        np.sign(point_side) != np.sign(source_side))

    pairs = surface.edges(garment.data)
    amount = np.clip(distance / max(float(smooth_distance), surface.TINY), 0.0, 1.0)

    def finish(table):
        """Limit and normalise; also the largest change of weights along one edge."""
        table = table.copy()
        table[table < NEGLIGIBLE] = 0.0
        over = (table > 0.0).sum(axis=1) > limit
        if over.any():
            # Zero everything below each row's ``limit``-th largest weight.
            np.put_along_axis(table, np.argsort(-table, axis=1)[:, limit:], 0.0, axis=1)
        total = table.sum(axis=1)
        table /= np.where(total <= 0.0, 1.0, total)[:, None]
        # Half the summed difference: 0 for equal weights, 1 for two vertices that share no bone.
        change = 0.5 * np.abs(table[pairs[:, 0]] - table[pairs[:, 1]]).sum(axis=1) if len(pairs) else np.zeros(1)
        return table, over, total <= 0.0, change

    final, clamped, unweighted, step = finish(source)
    plain_step = float(step.max())
    if smooth is None:
        # Left to itself: smooth only a garment that would shear, so cloth on the body keeps the body's weights.
        smooth = AUTO_PASSES if plain_step > STEP_WARNING else 0
    smooth = max(0, int(smooth))
    if smooth:
        final, clamped, unweighted, step = finish(_smooth(source, pairs, amount, smooth))
    used = [name for column, name in enumerate(bones) if (final[:, column] > 0.0).any()]

    report = {
        "tool": "weights",
        "garment": garment.name,
        "body": body.name,
        "rig": rig.name,
        "verts": int(len(points)),
        "limit": limit,
        "region": region,
        "same_side": bool(same_side),
        "source_faces": int(usable.sum()),
        "unweighted": int(unweighted.sum()),
        "clamped": int(clamped.sum()),
        "max_distance": float(distance.max()),
        "max_distance_vertex": int(distance.argmax()),
        "mean_distance": float(distance.mean()),
        "wrong_side": int(wrong.sum()),
        "wrong_side_vertices": np.flatnonzero(wrong)[:20].tolist(),
        "smooth": smooth,
        "weight_step": float(step.max()),
        "weight_step_unsmoothed": plain_step,
        "weight_step_edge": pairs[int(step.argmax())].tolist() if len(pairs) else [],
        "bones": used,
        "bone_count": len(used),
    }
    if unweighted.any():
        return surface.refusal(report, "{:d} vertices would get no weight".format(report["unweighted"]))
    if max_distance is not None and report["max_distance"] > float(max_distance):
        return surface.refusal(report, "vertex {:d} is {!r} from the body, over max_distance {!r}".format(
            report["max_distance_vertex"], report["max_distance"], float(max_distance)))

    # Old skinning goes, also for bones no longer used: a stale group would still pull.
    for name in deform:
        group = garment.vertex_groups.get(name)
        if group is not None:
            garment.vertex_groups.remove(group)
    for column, name in enumerate(bones):
        if name not in used:
            continue
        group = garment.vertex_groups.new(name=name)
        values = final[:, column]
        for index in np.flatnonzero(values > 0.0).tolist():
            group.add((index,), values[index], 'REPLACE')

    modifier = next((m for m in garment.modifiers if m.type == 'ARMATURE' and (m.object is None or m.object == rig)), None)
    report["modifier_added"] = modifier is None
    if modifier is None:
        modifier = garment.modifiers.new("Armature", 'ARMATURE')
        # Skinning comes before anything that adds geometry, as on the body.
        garment.modifiers.move(len(garment.modifiers) - 1, 0)
    modifier.object = rig
    modifier.use_vertex_groups = True
    report["modifier"] = modifier.name
    garment.data.update()

    count, total = _read_back(garment, used)
    report["max_influences"] = int(count.max())
    report["max_sum_error"] = float(np.abs(total - 1.0).max())
    report["ok"] = report["max_influences"] <= limit and report["max_sum_error"] < 1e-4
    report["applied"] = True
    warnings = []
    if report["wrong_side"]:
        warnings.append("{:d} vertices take weights from the other side of the body; try same_side or region".format(
            report["wrong_side"]))
    if report["weight_step"] > STEP_WARNING:
        warnings.append("neighbouring vertices {:d} and {:d} follow different bones (step {:.2f}); "
                        "the cloth will shear there when posed; try more smooth passes or region".format(
                            *report["weight_step_edge"], report["weight_step"]))
    if warnings:
        report["warning"] = "; ".join(warnings)
    return report
