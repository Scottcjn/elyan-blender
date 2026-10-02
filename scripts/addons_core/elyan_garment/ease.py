# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
``fit``: push the parts of a garment that touch or enter the body back out to a set ease.

Not a shrinkwrap. A shrinkwrap moves every vertex to one distance and loses
the cut; this moves only the vertices that are too close, by as little as
clears them, and leaves the rest of the garment exactly as it was made.
"""

import numpy as np

from . import surface

# Vertices are stored as float32; aiming this far past the ease keeps the stored result from landing under it.
MARGIN = 1e-5
PASSES = 16


def _smooth(delta, offenders, links, passes):
    """
    Even the push out among the vertices being pushed.

    Each takes half of the mean of its pushed neighbours. Vertices that were
    clear are no part of it: they stay where they are, which is the point of
    the tool, and the push already fades to nothing where the garment reaches
    the ease by itself.
    """
    chosen = np.flatnonzero(offenders).tolist()
    for _pass in range(passes):
        result = delta.copy()
        for vert in chosen:
            near = [other for other in links[vert] if offenders[other]]
            if near:
                result[vert] = 0.5 * delta[vert] + 0.5 * delta[near].mean(axis=0)
        delta = result
    return delta


def _through(skin, cloth_points, cloth_tris, reach):
    """
    Body vertices that show through the garment: the cloth's faces pass beneath the skin there.

    Clearance is measured at the garment's vertices, and a coarse garment can
    have every vertex clear while the body pokes out through the middle of a
    face. Looking from the body's side catches that. A body vertex counts
    when the nearest point of the cloth is within ``reach``, lies inside a face
    (not on the open edge, where skin simply continues past the hem) and is
    under the skin as the body's own normal sees it.
    Returns the body vertex indices, the cloth point, triangle and depth for each, and the cloth surface.
    """
    cloth = surface.Surface(cloth_points, cloth_tris)
    low, high = cloth_points.min(axis=0) - reach, cloth_points.max(axis=0) + reach
    candidates = np.flatnonzero(((skin.verts >= low) & (skin.verts <= high)).all(axis=1))
    if len(candidates) == 0:
        return candidates, np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros(0), cloth
    location, face_normal, tri, _signed = cloth.nearest(skin.verts[candidates])
    away = location - skin.verts[candidates]
    depth = np.linalg.norm(away, axis=1)
    square_on = np.abs(np.einsum("ij,ij->i", away, face_normal)) >= 0.9 * depth
    beneath = np.einsum("ij,ij->i", away, skin.vertex_normals()[candidates]) < 0.0
    found = (depth < reach) & square_on & beneath
    return candidates[found], location[found], tri[found], depth[found], cloth


def _through_stats(found):
    depth = found[3]
    return {"body_verts": int(len(depth)), "max_depth": float(depth.max()) if len(depth) else 0.0}


def fit(garment, body, ease=0.006, groups=None, max_move=0.03, pin=None, max_fraction=0.0, smooth=2, faces=False):
    """
    Push garment vertices closer to ``body`` than ``ease`` (or inside it) out until they clear by ``ease``.

    Vertices already clear are not moved. Each offending vertex moves away from
    its nearest point on the body's surface, along the surface normal there.
    ``groups``: garment vertex groups to limit the fit to. ``pin``: garment
    vertex groups that never move (necklines, hems, welded seams).
    ``max_move``: the farthest any vertex may travel. ``smooth``: passes that
    even the push out among neighbouring pushed vertices so it does not dimple.
    ``faces``: also lift faces the body still shows through although their
    corners are clear; this does move vertices that were clear.
    Every shape key receives the same displacement. Measured unposed: the body
    as shown (its shape keys and modifiers), the garment as stored.

    Refuses, writing nothing, when more than ``max_fraction`` of the offending
    vertices would still be closer than ``ease`` after moving ``max_move``
    (0.0: all of them must clear), or when the result would be worse than the
    start. Returns the report: ``before`` and ``after`` clearance (``min``,
    ``percentile_5``, ``below_threshold``, ``inside``: the definitions of the
    bridge's ``check``), ``moved``, ``largest_move``, ``unresolved``, and
    ``shows_through`` before and after: body vertices poking out through the
    garment's faces, which the clearance numbers cannot see.
    """
    garment = surface.writable(surface.mesh_object(garment, "garment"))
    body = surface.mesh_object(body, "body")
    ease, max_move, max_fraction = float(ease), float(max_move), float(max_fraction)
    if ease < 0.0 or max_move <= 0.0:
        raise ValueError("ease must not be negative and max_move must be positive")

    skin = surface.Surface.from_object(body)
    start = surface.rest_world(garment)
    cloth_tris = surface.triangles(garment.data)
    groups, pin = surface.names(groups), surface.names(pin)
    free = np.ones(len(start), dtype=bool)
    if groups:
        free &= surface.group_mask(garment, groups)
    pinned = surface.group_mask(garment, pin) if pin else np.zeros(len(start), dtype=bool)
    free &= ~pinned

    distance = skin.distance(start)
    offenders = free & (distance < ease)
    through = _through(skin, start, cloth_tris, max_move) if len(cloth_tris) else None
    report = {
        "tool": "fit",
        "garment": garment.name,
        "body": body.name,
        "ease": ease,
        "max_move": max_move,
        "max_fraction": max_fraction,
        "groups": groups,
        "pin": pin,
        "faces": bool(faces),
        "before": surface.clearance_stats(distance, ease),
        "offending": int(offenders.sum()),
        # Too close, but not this call's to move.
        "pinned_below": int((pinned & (distance < ease)).sum()),
        "outside_groups_below": int((~free & ~pinned & (distance < ease)).sum()),
        "moved": 0,
        "largest_move": 0.0,
        "unresolved": 0,
        "shape_keys": 0,
    }
    if through is not None:
        report["shows_through"] = {"before": _through_stats(through)}
    lift = bool(faces) and through is not None and len(through[0]) > 0
    if not offenders.any() and not lift:
        report.update({"after": report["before"], "ok": True, "applied": False,
                       "note": "nothing to do: no free vertex is closer than the ease"})
        if through is not None:
            report["shows_through"]["after"] = report["shows_through"]["before"]
            if len(through[0]):
                report["warning"] = _warning(report)
        return report

    links = surface.neighbours(garment.data)
    goal = ease + MARGIN

    def clamp(delta):
        # Never farther than allowed, whatever the later passes ask for.
        travelled = np.linalg.norm(delta, axis=1)
        over = travelled > max_move
        delta[over] *= (max_move / travelled[over])[:, None]
        return delta

    def settle(points, active, even):
        """Push the ``active`` vertices of ``points`` out until they clear, or the passes run out."""
        for step in range(PASSES):
            location, face_normal, distance = skin.clearance(points)
            short = active & (distance < goal)
            if not short.any():
                break
            away = points - location
            length = np.linalg.norm(away, axis=1)
            # Outside, the way out is straight away from the nearest point; inside, straight through it.
            # That line is the surface normal wherever the nearest point is inside a face.
            outward = away / np.where(length < 1e-9, 1.0, length)[:, None] * np.where(distance < 0.0, -1.0, 1.0)[:, None]
            # On the surface itself there is no such line; the face's normal stands in.
            on_surface = length < 1e-9
            outward[on_surface] = face_normal[on_surface]
            push = np.zeros_like(points)
            push[short] = outward[short] * (goal - distance[short])[:, None]
            delta = points + push - start
            if step == 0 and even > 0:
                delta = _smooth(delta, active, links, even)
            points = start + clamp(delta)
        return points

    points = settle(start.copy(), offenders, int(smooth))
    active = offenders.copy()
    if faces:
        for _pass in range(PASSES):
            verts, location, tri, _depth, cloth = _through(skin, points, cloth_tris, max_move)
            if len(verts) == 0:
                break
            # The cloth point over each such body vertex has to end up ``goal`` above it. Moving a
            # triangle's corners by their weights' share does that with the least movement.
            wanted = skin.verts[verts] + skin.vertex_normals()[verts] * goal - location
            corner = cloth.barycentric(location, tri)
            share = corner / np.einsum("ij,ij->i", corner, corner)[:, None]
            lifted = np.zeros_like(points)
            size = np.zeros(len(points))
            for row in range(len(verts)):
                for slot in range(3):
                    vert = cloth_tris[tri[row], slot]
                    move = wanted[row] * share[row, slot]
                    length = float(np.linalg.norm(move))
                    # Several body vertices may ask for the same corner; the largest request covers the others.
                    if free[vert] and length > size[vert]:
                        size[vert], lifted[vert] = length, move
            if not size.any():
                break
            active |= size > 0.0
            points = settle(start + clamp(points + lifted - start), active, 0)

    delta = points - start
    # Measured on what will be stored, so the report cannot be better than the mesh.
    final = surface.stored(garment, delta)
    after = skin.distance(final)
    travelled = np.linalg.norm(delta, axis=1)
    unresolved = active & (after < ease)
    report.update({
        "after": surface.clearance_stats(after, ease),
        "moved": int((travelled > 0.0).sum()),
        "largest_move": float(travelled.max()),
        "at_max_move": int((travelled >= max_move * (1.0 - 1e-9)).sum()),
        "unresolved": int(unresolved.sum()),
        "unresolved_vertices": np.flatnonzero(unresolved)[:20].tolist(),
    })
    if through is not None:
        report["shows_through"]["after"] = _through_stats(_through(skin, final, cloth_tris, max_move))
    allowed = max_fraction * max(report["offending"], int(active.sum()))
    if report["unresolved"] > allowed:
        return surface.refusal(report, (
            "{:d} of {:d} offending vertices cannot reach {:g} of clearance within max_move {:g} "
            "(allowed: {:g}); the garment is too far inside the body for a fit").format(
                report["unresolved"], int(active.sum()), ease, max_move, allowed))
    if report["after"]["inside"] > report["before"]["inside"] or report["after"]["min"] < report["before"]["min"]:
        return surface.refusal(report, "the fit would leave the garment worse than it is")

    report["shape_keys"] = surface.displace(garment, delta)
    report["ok"] = True
    report["applied"] = True
    if through is not None and report["shows_through"]["after"]["body_verts"]:
        report["warning"] = _warning(report)
    displayed = surface.displayed_clearance(garment, body, ease)
    if displayed is not None:
        report["check"] = displayed
    return report


def _warning(report):
    after = report["shows_through"]["after"]
    return ("{:d} body vertices still show through the garment's faces (up to {:.1f} mm) although its "
            "vertices are clear{:s}").format(
                after["body_verts"], after["max_depth"] * 1000.0,
                "" if report["faces"] else "; faces=true lifts those faces")
