# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
``shapekeys``: make a garment follow the body's shape keys.

VRChat drives body sliders and mouth shapes as shape keys on the body. A
garment without keys of the same names stays put while the body grows through
it. Each garment vertex is tied to a point on the body's surface at rest, the
point is found again with the key on, and the difference becomes a garment
key of the same name.
"""

import numpy as np

from . import surface

# A key that moves nothing by more than this is not worth a slot on the garment.
MIN_MOVE = 1e-5


def _fade(distance, limit):
    """Full influence up to half of ``limit``, none beyond it, smooth in between."""
    t = np.clip((np.abs(distance) - 0.5 * limit) / (0.5 * limit), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def _body_moves(body, wanted, base):
    """
    How each wanted key moves the body as shown: world displacement per evaluated vertex.

    Read by switching the key on and evaluating, not from the key's stored
    points: only that sees a Mask modifier, a subdivision or a key that is
    relative to another one.
    """
    blocks = body.data.shape_keys.key_blocks
    pinned = body.show_only_shape_key
    moves = {}
    try:
        body.show_only_shape_key = False
        for name in wanted:
            block = blocks[name]
            value = block.value
            off = base
            if value != 0.0:
                block.value = 0.0
                off = surface.evaluated(body)[0]
            block.value = 1.0
            moves[name] = surface.evaluated(body)[0] - off
            block.value = value
    finally:
        body.show_only_shape_key = pinned
    return moves


def shapekeys(garment, body, keys=None, distance=0.03, max_error=0.004):
    """
    Give ``garment`` a shape key for each of ``body``'s, so it follows body sliders and visemes.

    ``keys``: body shape key names; left out, every key that is at 0 (a key
    that is switched on is part of the body's shape, not a slider).
    Each garment vertex is bound to the nearest triangle of the body as shown,
    unposed (corner weights, offset along the surface normal), and placed again
    with the key on. The influence fades to nothing between half of
    ``distance`` and ``distance`` from the body, so cloth standing off the
    body is left alone.

    Per key, in ``keys``: ``status`` is ``written``, ``skipped`` (the key moves
    nothing near the garment) or ``refused``; ``max_displacement``; and
    ``clearance`` of the garment over the body with the key on (``min``,
    ``percentile_5``, ``inside``), next to ``rest_clearance``. A key is refused,
    and not written, when its minimum clearance is more than ``max_error`` below
    the minimum at rest: the garment would be poked through with that slider on.
    A key that exists on the garment is replaced. ``ok`` is false if any key was refused.
    """
    garment = surface.writable(surface.mesh_object(garment, "garment"))
    body = surface.mesh_object(body, "body")
    distance, max_error = float(distance), float(max_error)
    if distance <= 0.0:
        raise ValueError("distance must be positive")
    if not body.data.shape_keys:
        raise ValueError("{!r} has no shape keys".format(body.name))
    blocks = body.data.shape_keys.key_blocks
    reference = body.data.shape_keys.reference_key
    wanted = surface.names(keys)
    missing = [name for name in wanted if name not in blocks]
    if missing:
        raise ValueError("{!r} has no shape key {:s}".format(body.name, ", ".join(repr(n) for n in missing)))
    switched_on = []
    if not wanted:
        wanted = [block.name for block in blocks if block != reference and block.value == 0.0]
        switched_on = [block.name for block in blocks if block != reference and block.value != 0.0]

    with surface.at_rest(body):
        verts, mesh = surface.evaluated(body)
        skin = surface.Surface(verts, surface.triangles(mesh), surface.polygons(mesh))
        moves = _body_moves(body, wanted, verts)

    points = surface.rest_world(garment)
    location, _face_normal, tri, _signed = skin.nearest(points)
    corner = skin.barycentric(location, tri)
    # The offset is split into a part along the body's smooth normal, which turns
    # with the surface, and what is left over, which is carried as it is.
    normal = surface.unit(skin.interpolate(skin.vertex_normals(), tri, corner))
    offset = points - location
    height = np.einsum("ij,ij->i", offset, normal)
    leftover = offset - height[:, None] * normal
    signed = skin.distance(points)
    influence = _fade(signed, distance)
    near = influence > 0.0

    report = {
        "tool": "shapekeys",
        "garment": garment.name,
        "body": body.name,
        "distance": distance,
        "max_error": max_error,
        "verts": int(len(points)),
        "bound_verts": int(near.sum()),
        "keys": {},
        "written": [],
        "skipped": [],
        "refused": [],
        # Left alone without being asked for: on, so part of the body's shape rather than a slider.
        "switched_on": switched_on,
    }
    if not near.any():
        report.update({"ok": True, "applied": False, "skipped": wanted,
                       "note": "no garment vertex is within {:g} of the body".format(distance)})
        return report
    # Judged where the garment follows the body; cloth out of reach is not this tool's to answer for.
    rest = surface.clearance_stats(signed[near], 0.0)
    report["rest_clearance"] = {key: rest[key] for key in ("min", "percentile_5", "inside")}
    touched = np.unique(skin.tris[tri[near]])

    results = {}
    for name in wanted:
        entry = report["keys"][name] = {}
        moved = moves[name]
        entry["body_move_near_garment"] = float(np.linalg.norm(moved[touched], axis=1).max())
        if entry["body_move_near_garment"] < MIN_MOVE:
            entry.update({"status": "skipped", "max_displacement": 0.0,
                          "reason": "the body does not move within reach of the garment"})
            report["skipped"].append(name)
            continue

        shaped = skin.moved(moved)
        placed = (shaped.interpolate(shaped.verts, tri, corner) + leftover
                  + height[:, None] * surface.unit(shaped.interpolate(shaped.vertex_normals(), tri, corner)))
        delta = (placed - points) * influence[:, None]
        entry["max_displacement"] = float(np.linalg.norm(delta, axis=1).max())
        if entry["max_displacement"] < MIN_MOVE:
            entry.update({"status": "skipped", "reason": "the garment would not move"})
            report["skipped"].append(name)
            continue

        local = surface.rest_local(garment) + surface.vectors_to_local(garment, delta)
        # Measured on what the key will store, against the body with its key fully on.
        stored = surface.to_world(garment, local.astype(np.float32).astype(np.float64))
        stats = surface.clearance_stats(shaped.distance(stored[near]), 0.0)
        entry["clearance"] = {key: stats[key] for key in ("min", "percentile_5", "inside")}
        entry["clearance_change"] = stats["min"] - rest["min"]
        if stats["min"] < rest["min"] - max_error:
            entry.update({"status": "refused", "reason": (
                "minimum clearance {!r} with the key on is more than {:g} below {!r} at rest").format(
                    stats["min"], max_error, rest["min"])})
            report["refused"].append(name)
            continue
        entry["status"] = "written"
        results[name] = local.astype(np.float32).ravel()

    if results and not garment.data.shape_keys:
        garment.shape_key_add(name="Basis", from_mix=False)
    for name, coords in results.items():
        block = blocks[name]
        own = garment.data.shape_keys.key_blocks.get(name)
        report["keys"][name]["replaced"] = own is not None
        if own is None:
            own = garment.shape_key_add(name=name, from_mix=False)
        own.relative_key = garment.data.shape_keys.reference_key
        own.data.foreach_set("co", coords)
        # Same range and setting as the body's, so both look alike until something drives them together.
        own.slider_min, own.slider_max = block.slider_min, block.slider_max
        own.value = block.value
        report["written"].append(name)
    if results:
        garment.data.update()
    report["ok"] = not report["refused"]
    report["applied"] = bool(results)
    return report
