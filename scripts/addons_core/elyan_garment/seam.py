# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
``weld``: move an open boundary of a garment exactly onto a ring.

``check`` has measured seam rings against the contract for a while; nothing
repaired one. This moves the boundary vertices onto the ring's points and
eases the rows behind them, so a sleeve meets its armhole within the 1e-5 the
contract asks for without a crease one row in.
"""

import math

import bpy
import numpy as np

from . import surface


def _nearest_points(points, candidates):
    """For each point the distance to, and index of, the nearest candidate."""
    delta = points[:, None, :] - candidates[None, :, :]
    squared = np.einsum("ijk,ijk->ij", delta, delta)
    index = squared.argmin(axis=1)
    return np.sqrt(squared[np.arange(len(points)), index]), index


def _on_polyline(points, ring):
    """The nearest point on the closed polyline through ``ring`` for each point."""
    start, end = ring, np.roll(ring, -1, axis=0)
    along = end - start
    length = np.einsum("ij,ij->i", along, along)
    length[length < surface.TINY] = 1.0
    # points x segments
    t = np.einsum("psk,sk->ps", points[:, None, :] - start[None, :, :], along) / length
    foot = start[None, :, :] + np.clip(t, 0.0, 1.0)[:, :, None] * along[None, :, :]
    delta = points[:, None, :] - foot
    best = np.einsum("psk,psk->ps", delta, delta).argmin(axis=1)
    return foot[np.arange(len(points)), best]


def _match(loop_points, ring):
    """
    A target on the ring for every loop vertex, and how it was found.

    Equal counts: the ring is taken to be in order around the seam, like the
    loop, so only the starting index and the direction are unknown; all of
    them are tried. If the ring's points are in no such order, nearest
    neighbours are used when they pair off one to one.
    """
    count = len(loop_points)
    if count != len(ring):
        return _on_polyline(loop_points, ring), {"matched": "polyline", "note": (
            "counts differ, so vertices went to the nearest point on the ring's outline; "
            "ring points between them have no vertex")}

    best = None
    indices = np.arange(count)
    for reverse in (False, True):
        ordered = ring[::-1] if reverse else ring
        for shift in range(count):
            target = ordered[(indices + shift) % count]
            cost = float(((loop_points - target) ** 2).sum())
            if best is None or cost < best[0]:
                best = (cost, target, {"matched": "cyclic", "shift": shift, "reversed": reverse})
    _distance, index = _nearest_points(loop_points, ring)
    if len(set(index.tolist())) == count:
        target = ring[index]
        cost = float(((loop_points - target) ** 2).sum())
        # Only when it is clearly better: on an ordered ring both give the same pairing.
        if cost < best[0] * 0.999:
            best = (cost, target, {"matched": "nearest"})
    return best[1], best[2]


def _spread(points, ring):
    """The seam measure ``check`` uses: from each ring point to the nearest boundary vertex."""
    distance = _nearest_points(ring, points)[0]
    return {"max": float(distance.max()), "rms": float(math.sqrt(float(np.mean(distance * distance))))}


def _gap(points, target):
    distance = np.linalg.norm(points - target, axis=1)
    return {"max": float(distance.max()), "rms": float(math.sqrt(float(np.mean(distance * distance))))}


def _pick(ob, loops, which, role):
    """Narrow ``loops`` by an index or a vertex group; None keeps them all."""
    if which is None:
        return list(range(len(loops)))
    if isinstance(which, bool):
        raise ValueError("{:s}: expected a loop index or a vertex group name".format(role))
    if isinstance(which, int):
        if not 0 <= which < len(loops):
            raise ValueError("{:s}: {!r} has loops 0..{:d}, not {:d}".format(role, ob.name, len(loops) - 1, which))
        return [which]
    mask = surface.group_mask(ob, [str(which)])
    counts = [int(mask[loop].sum()) for loop in loops]
    if max(counts) == 0:
        raise ValueError("{:s}: no boundary loop of {!r} has vertices in group {!r}".format(role, ob.name, which))
    return [int(np.argmax(counts))]


def _loop_summary(points, loops):
    return [
        {"index": index, "verts": len(loop), "centre": points[loop].mean(axis=0).tolist()}
        for index, loop in enumerate(loops)
    ]


def weld(garment, ring, tolerance=1e-5, loop=None, ring_loop=None, rings=2, max_move=0.05, pin=None):
    """
    Move one open boundary loop of ``garment`` onto ``ring``, to within ``tolerance``.

    ``ring``: a list of world-space points in order around the seam (the
    contract's armhole ring), or another object (or its name), whose open
    boundary is the ring. ``loop`` / ``ring_loop`` choose the boundary on each
    side: a loop index as listed in the report's ``loops``, or a vertex group
    the loop's vertices are in; left out, the pair lying closest together is used.

    Equal counts: each vertex goes to its own ring point (every start index and
    both directions are tried). Different counts: each vertex goes to the
    nearest point on the ring's outline. ``rings`` rows of vertices behind the
    boundary follow with a falloff so no crease is left; 0 moves the boundary
    only. Vertices in ``pin`` groups and other open boundaries never move.
    Every shape key receives the same displacement.

    Refuses, writing nothing, when a vertex would have to move more than
    ``max_move`` (the wrong loop, or the wrong ring) or the result is not
    within ``tolerance``. Returns the report: ``before`` and ``after`` with
    ``max`` and ``rms`` vertex-to-ring distance, unrounded, and ``loop``.
    """
    garment = surface.writable(surface.mesh_object(garment, "garment"))
    mesh = garment.data
    tolerance = float(tolerance)
    rings = max(0, int(rings))
    points = surface.rest_world(garment)
    loops, tangled = surface.boundary_loops(mesh)
    if not loops:
        raise ValueError("{!r} has no open boundary loop to weld".format(garment.name))
    candidates = _pick(garment, loops, loop, "loop")

    report = {
        "tool": "weld",
        "garment": garment.name,
        "tolerance": tolerance,
        "loops": _loop_summary(points, loops),
        "tangled_boundaries": tangled,
    }

    source = bpy.data.objects.get(ring) if isinstance(ring, str) else ring
    if isinstance(ring, str) and source is None:
        raise ValueError("ring: no object named {!r}".format(ring))
    if getattr(source, "type", None) == 'MESH':
        other = surface.mesh_object(source, "ring")
        other_points = surface.rest_world(other)
        other_loops, _tangled = surface.boundary_loops(other.data)
        if not other_loops:
            raise ValueError("ring: {!r} has no open boundary loop".format(other.name))
        targets = [(index, other_points[other_loops[index]])
                   for index in _pick(other, other_loops, ring_loop, "ring_loop")]
        report["ring"] = {"object": other.name, "loops": _loop_summary(other_points, other_loops)}
    else:
        try:
            ring_points = np.array(ring, dtype=np.float64).reshape(-1, 3)
        except (TypeError, ValueError):
            raise ValueError("ring: expected a list of [x, y, z] points or a mesh object") from None
        if len(ring_points) < 3 or not np.isfinite(ring_points).all():
            raise ValueError("ring: needs at least 3 finite points")
        targets = [(None, ring_points)]
        report["ring"] = {"object": None}

    # The pair that already lies closest together is the seam that was meant.
    best = None
    for index in candidates:
        for ring_index, ring_points in targets:
            score = float(_nearest_points(points[loops[index]], ring_points)[0].mean())
            if best is None or score < best[0]:
                best = (score, index, ring_index, ring_points)
    _score, index, ring_index, ring_points = best
    chosen = np.array(loops[index], dtype=np.int64)
    target, how = _match(points[chosen], ring_points)

    report["ring"].update({"loop": ring_index, "points": int(len(ring_points))})
    report["loop"] = {"index": index, "verts": int(len(chosen)), "centre": points[chosen].mean(axis=0).tolist(),
                      "counts_match": len(chosen) == len(ring_points)}
    report["loop"].update(how)
    report["before"] = _gap(points[chosen], target)
    report["ring_to_boundary_before"] = _spread(points[chosen], ring_points)

    delta = np.zeros_like(points)
    delta[chosen] = target - points[chosen]
    fixed = np.zeros(len(points), dtype=bool)
    for other_loop in loops:
        fixed[other_loop] = True
    pin = surface.names(pin)
    if pin:
        pinned = surface.group_mask(garment, pin)
        report["pinned_on_loop"] = int(pinned[chosen].sum())
        if report["pinned_on_loop"]:
            return surface.refusal(report, "{:d} vertices of the loop are pinned".format(report["pinned_on_loop"]))
        fixed |= pinned

    eased = 0
    if rings:
        # Row by row away from the boundary: each vertex takes the mean move of the row before it,
        # then the whole row is scaled down. The cosine starts and ends flat, which is what hides the join.
        links = surface.neighbours(mesh)
        level = np.full(len(points), -1, dtype=np.int64)
        level[chosen] = 0
        raw = delta.copy()
        front = chosen.tolist()
        for row in range(1, rings + 1):
            following = []
            for vert in front:
                for other_vert in links[vert]:
                    if level[other_vert] == -1 and not fixed[other_vert]:
                        level[other_vert] = row
                        following.append(other_vert)
            scale = 0.5 * (1.0 + math.cos(math.pi * row / (rings + 1)))
            for vert in following:
                behind = [other_vert for other_vert in links[vert] if level[other_vert] == row - 1]
                raw[vert] = raw[behind].mean(axis=0)
                delta[vert] = raw[vert] * scale
            eased += len(following)
            front = following
    report["eased_verts"] = eased
    report["rings"] = rings
    report["max_move"] = float(max_move)
    report["largest_move"] = report["before"]["max"]

    if report["before"]["max"] > float(max_move):
        return surface.refusal(report, "a vertex would move {!r}, over max_move {!r}".format(
            report["before"]["max"], float(max_move)))
    # Judged on what the mesh can actually store, which is float32.
    after = surface.stored(garment, delta)
    report["after"] = _gap(after[chosen], target)
    report["ring_to_boundary_after"] = _spread(after[chosen], ring_points)
    if report["after"]["max"] > tolerance:
        return surface.refusal(report, "would still be {!r} from the ring, over the tolerance {!r}".format(
            report["after"]["max"], tolerance))

    report["shape_keys"] = surface.displace(garment, delta)
    report["ok"] = True
    report["applied"] = True
    # With equal counts this is the number ``check`` will report for the seam.
    report["seam_within_tolerance"] = report["ring_to_boundary_after"]["max"] <= tolerance
    return report
