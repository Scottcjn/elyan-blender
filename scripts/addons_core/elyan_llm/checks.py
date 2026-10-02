# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Geometry checks and scene-state diffs: numbers, so nobody has to judge a render.

``check()`` fails closed: anything it cannot measure is listed as a failure
rather than skipped. Distances are computed in double precision and returned
unrounded, since seam rings are expected to match within 1e-5.
"""

import collections
import hashlib
import json

import bpy
import numpy as np

SEAM_TOLERANCE = 1e-5
AREA_EPSILON = 1e-12
SCALE_EPSILON = 1e-6
MAX_SNAPSHOTS = 20
# Distances are computed in blocks of this many target points to bound memory.
_CHUNK = 2048

# Objects kept in these collections are stored copies, not part of the scene's content.
CHECKPOINT_PREFIX = "_checkpoint_"

# Named scene states, oldest first.
_snapshots = collections.OrderedDict()


# The artist's pins (see marks.py) are notes, not part of the work being compared.
MARKS_COLLECTION = "Elyan Marks"


def is_checkpoint_object(ob):
    """Objects that scene comparisons leave out: checkpoint copies and mark pins."""
    return any(
        c.name.startswith(CHECKPOINT_PREFIX) or c.name == MARKS_COLLECTION for c in ob.users_collection)


# -----------------------------------------------------------------------------
# Measuring

def _triangles(mesh):
    count = len(mesh.polygons)
    if count == 0:
        return 0
    totals = np.empty(count, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", totals)
    return int(totals.sum()) - 2 * count


def _world_coords(mesh, matrix):
    """Vertex positions in world space as float64, so later differences keep their precision."""
    coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(-1, 3).astype(np.float64)
    matrix = np.array(matrix, dtype=np.float64)
    return coords @ matrix[:3, :3].T + matrix[:3, 3]


def _group_mask(ob, mesh, names):
    """Which vertices of ``mesh`` carry weight in any of the named groups of ``ob``."""
    indices = {ob.vertex_groups[name].index for name in names}
    mask = np.zeros(len(mesh.vertices), dtype=bool)
    for vert in mesh.vertices:
        for element in vert.groups:
            if element.group in indices and element.weight > 0.0:
                mask[vert.index] = True
                break
    return mask


def _topology(mesh, area_epsilon):
    """Defect counts, plus the indices of the vertices that lie on an open boundary."""
    import bmesh
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        boundary = [e for e in bm.edges if len(e.link_faces) == 1]
        # Boundary edges sharing a vertex belong to the same loop: count the connected groups.
        parent = {}

        def find(index):
            while parent.setdefault(index, index) != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for edge in boundary:
            parent[find(edge.verts[0].index)] = find(edge.verts[1].index)
        return {
            "non_manifold_edges": sum(1 for e in bm.edges if len(e.link_faces) > 2),
            "wire_edges": sum(1 for e in bm.edges if not e.link_faces),
            "boundary_edges": len(boundary),
            "boundary_loops": len({find(index) for index in parent}),
            "loose_verts": sum(1 for v in bm.verts if not v.link_edges),
            "zero_area_faces": sum(1 for f in bm.faces if f.calc_area() < area_epsilon),
        }, sorted(parent)
    finally:
        bm.free()


def _mesh_object(name, failures, role):
    ob = bpy.data.objects.get(name) if isinstance(name, str) else None
    if ob is None:
        failures.append("{:s}: no object named {!r}".format(role, name))
    elif ob.type != 'MESH':
        failures.append("{:s}: {!r} is a {:s}, not a mesh".format(role, name, ob.type))
    else:
        return ob
    return None


def _nearest(points, candidates):
    """For each point the distance to, and index of, the nearest candidate."""
    distances = np.empty(len(points), dtype=np.float64)
    indices = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), _CHUNK):
        block = points[start:start + _CHUNK]
        delta = block[:, None, :] - candidates[None, :, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        nearest = squared.argmin(axis=1)
        indices[start:start + _CHUNK] = nearest
        distances[start:start + _CHUNK] = np.sqrt(squared[np.arange(len(block)), nearest])
    return distances, indices


def _object_report(ob, depsgraph, evaluated, area_epsilon):
    ob_eval = ob.evaluated_get(depsgraph)
    mesh = ob_eval.data if evaluated else ob.data
    report = {
        "name": ob.name,
        # What ships is the mesh after modifiers, so budgets always use that.
        "triangles": _triangles(ob_eval.data),
        "triangles_before_modifiers": _triangles(ob.data),
        "verts": len(mesh.vertices),
        "material_slots": len(ob.material_slots),
        "empty_material_slots": sum(1 for slot in ob.material_slots if slot.material is None),
        "scale": list(ob.scale),
        "scale_applied": all(abs(s - 1.0) <= SCALE_EPSILON for s in ob.scale),
    }
    report.update(_topology(mesh, area_epsilon)[0])
    return report


def _seam(spec, depsgraph, failures):
    """How far each target point is from the nearest open-boundary vertex of the object."""
    label = "seam {!s}".format(spec.get("object"))
    result = {"object": spec.get("object")}
    try:
        tolerance = float(spec.get("tolerance", SEAM_TOLERANCE))
    except (TypeError, ValueError):
        failures.append("{:s}: tolerance must be a number".format(label))
        return result
    result["tolerance"] = tolerance
    before = len(failures)
    ob = _mesh_object(spec.get("object"), failures, label)

    points = None
    if spec.get("points") is not None:
        try:
            points = np.array(spec["points"], dtype=np.float64).reshape(-1, 3)
        except (TypeError, ValueError):
            failures.append("{:s}: 'points' must be a list of [x, y, z]".format(label))
        result["target"] = "points"
    elif spec.get("other") is not None:
        other = _mesh_object(spec["other"], failures, label)
        group = spec.get("other_group")
        if other is not None and (not group or group not in other.vertex_groups):
            failures.append("{:s}: {!r} has no vertex group {!r}".format(label, other.name, group))
        elif other is not None:
            other_eval = other.evaluated_get(depsgraph)
            mask = _group_mask(other, other_eval.data, [group])
            points = _world_coords(other_eval.data, other_eval.matrix_world)[mask]
        result["target"] = "{!s}[{!s}]".format(spec["other"], group)
    else:
        failures.append("{:s}: needs 'points', or 'other' with 'other_group'".format(label))

    candidates = None
    if ob is not None:
        ob_eval = ob.evaluated_get(depsgraph)
        mesh = ob_eval.data
        ring = np.array(_topology(mesh, AREA_EPSILON)[1], dtype=np.int64)
        group = spec.get("group")
        if group is not None and group not in ob.vertex_groups:
            failures.append("{:s}: {!r} has no vertex group {!r}".format(label, ob.name, group))
        else:
            if group is not None:
                ring = ring[_group_mask(ob, mesh, [group])[ring]] if len(ring) else ring
                result["group"] = group
            candidates = _world_coords(mesh, ob_eval.matrix_world)[ring]
            result["boundary_verts"] = int(len(ring))
    if len(failures) > before or points is None or candidates is None:
        return result

    result["points"] = int(len(points))
    if len(points) == 0 or len(candidates) == 0:
        failures.append("{:s}: nothing to compare ({:d} points, {:d} boundary vertices)".format(
            label, len(points), len(candidates)))
        return result
    if not (np.isfinite(points).all() and np.isfinite(candidates).all()):
        failures.append("{:s}: coordinates are not finite".format(label))
        return result

    distances, nearest = _nearest(points, candidates)
    worst = int(distances.argmax())
    result.update({
        "max": float(distances.max()),
        "rms": float(np.sqrt(np.mean(distances * distances))),
        "worst_point": points[worst].tolist(),
        "worst_point_index": worst,
        "worst_vertex": int(ring[nearest[worst]]),
        "worst_vertex_position": candidates[nearest[worst]].tolist(),
    })
    if spec.get("group") is not None:
        # With the ring named on this side too, it must not have vertices the target lacks.
        reverse = _nearest(candidates, points)[0]
        result["reverse_max"] = float(reverse.max())
        result["boundary_verts_unmatched"] = int((reverse > tolerance).sum())
        if result["reverse_max"] > tolerance:
            failures.append("{:s}: {:d} ring vertices have no target point within {:g} (max {!r})".format(
                label, result["boundary_verts_unmatched"], tolerance, result["reverse_max"]))
    if result["max"] > tolerance:
        failures.append("{:s}: max deviation {!r} is over {:g} (rms {!r})".format(
            label, result["max"], tolerance, result["rms"]))
    return result


def _clearance(spec, depsgraph, failures):
    """Distance from garment vertices to the body surface, negative where a vertex is inside."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    label = "clearance {!s} over {!s}".format(spec.get("garment"), spec.get("body"))
    result = {"garment": spec.get("garment"), "body": spec.get("body")}
    before = len(failures)
    try:
        threshold = float(spec.get("threshold", 0.0))
        max_below = int(spec.get("max_below", 0))
        samples = int(spec.get("samples") or 0)
    except (TypeError, ValueError):
        failures.append("{:s}: threshold, max_below and samples must be numbers".format(label))
        return result
    result["threshold"] = threshold
    garment = _mesh_object(spec.get("garment"), failures, label)
    body = _mesh_object(spec.get("body"), failures, label)
    ignore = spec.get("ignore_groups") or []
    if garment is not None:
        # A misspelt group would quietly change what is measured, so it is an error.
        for name in ignore:
            if name not in garment.vertex_groups:
                failures.append("{:s}: {!r} has no vertex group {!r} to ignore".format(label, garment.name, name))
    if len(failures) > before:
        return result

    garment_eval = garment.evaluated_get(depsgraph)
    body_eval = body.evaluated_get(depsgraph)
    body_mesh = body_eval.data
    if len(body_mesh.polygons) == 0:
        failures.append("{:s}: the body has no faces".format(label))
        return result
    # World-space tree: the object-space one would distort distances under a scaled body.
    tree = BVHTree.FromPolygons(
        _world_coords(body_mesh, body_eval.matrix_world).tolist(),
        [tuple(polygon.vertices) for polygon in body_mesh.polygons],
    )

    coords = _world_coords(garment_eval.data, garment_eval.matrix_world)
    keep = np.ones(len(coords), dtype=bool)
    if ignore:
        keep &= ~_group_mask(garment, garment_eval.data, ignore)
    indices = np.flatnonzero(keep)
    result["verts"] = int(len(coords))
    result["ignored"] = int(len(coords) - len(indices))
    if samples and len(indices) > samples:
        # Evenly spaced and repeatable, so two runs measure the same vertices.
        indices = indices[np.linspace(0, len(indices) - 1, samples).astype(np.int64)]
    result["sampled"] = int(len(indices))
    if len(indices) == 0:
        failures.append("{:s}: no garment vertices left to measure".format(label))
        return result

    distances = np.empty(len(indices), dtype=np.float64)
    for slot, index in enumerate(indices):
        point = Vector(coords[index])
        location, normal, _face, distance = tree.find_nearest(point)
        if location is None:
            failures.append("{:s}: no body surface found near vertex {:d}".format(label, int(index)))
            return result
        # Behind the nearest face means inside the body: poke-through, however small the distance.
        distances[slot] = -distance if (point - location).dot(normal) < 0.0 else distance

    worst = int(distances.argmin())
    below = int((distances < threshold).sum())
    result.update({
        "min": float(distances.min()),
        "percentile_5": float(np.percentile(distances, 5)),
        "median": float(np.median(distances)),
        "below_threshold": below,
        "inside": int((distances < 0.0).sum()),
        "worst_vertex": int(indices[worst]),
        "worst_position": coords[indices[worst]].tolist(),
    })
    if below > max_below:
        failures.append("{:s}: {:d} vertices closer than {:g} (min {!r} at vertex {:d})".format(
            label, below, threshold, result["min"], result["worst_vertex"]))
    return result


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def check(args):
    """
    Numeric checks, see ``commands.cmd_check`` for the arguments.

    ``failures`` is empty and ``passed`` true only when everything asked for was
    measured and within limits.
    """
    failures = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    seams = _as_list(args.get("seam"))
    clearances = _as_list(args.get("clearance"))
    evaluated = args.get("evaluated", True) is not False
    area_epsilon = float(args.get("area_epsilon", AREA_EPSILON))

    names = args.get("names")
    if names:
        objects = [ob for ob in (_mesh_object(name, failures, "check") for name in names) if ob is not None]
    elif seams or clearances:
        objects = []
    else:
        objects = [ob for ob in bpy.context.selected_objects if ob.type == 'MESH'] or [
            ob for ob in bpy.context.scene.objects if ob.type == 'MESH' and not is_checkpoint_object(ob)]
        if not objects:
            failures.append("check: no mesh objects to check")

    parts = [_object_report(ob, depsgraph, evaluated, area_epsilon) for ob in objects]
    totals = {
        key: sum(part[key] for part in parts)
        for key in ("triangles", "material_slots", "empty_material_slots", "non_manifold_edges", "wire_edges",
                    "boundary_edges", "boundary_loops", "loose_verts", "zero_area_faces")
    }
    totals["materials"] = len({
        slot.material.name for ob in objects for slot in ob.material_slots if slot.material is not None})
    totals["unapplied_scale"] = sum(1 for part in parts if not part["scale_applied"])

    budgets = args.get("budgets") or {}
    for key, limit in budgets.items():
        if key not in totals:
            failures.append("budget {!r} is not a measured total; known: {:s}".format(key, ", ".join(sorted(totals))))
        elif not isinstance(limit, (int, float)) or isinstance(limit, bool):
            failures.append("budget {:s}: {!r} is not a number".format(key, limit))
        elif totals[key] > limit:
            failures.append("{:s}: {:d} over the budget of {:g}".format(key, totals[key], limit))
    for part in parts:
        for key in ("non_manifold_edges", "loose_verts", "zero_area_faces"):
            if part[key]:
                failures.append("{:s}: {:d} {:s}".format(part["name"], part[key], key.replace("_", " ")))
        if not part["scale_applied"]:
            failures.append("{:s}: unapplied scale {!r}".format(part["name"], part["scale"]))
    # Open edges are normal for garments, so their count only fails against a stated expectation.
    measured = {part["name"]: part["boundary_loops"] for part in parts}
    for name, expected in (args.get("boundary_loops") or {}).items():
        if name not in measured:
            failures.append("boundary_loops: {!r} was not among the checked objects".format(name))
        elif measured[name] != expected:
            failures.append("{:s}: {:d} boundary loops, expected {!r}".format(name, measured[name], expected))

    report = {
        "evaluated": evaluated,
        "totals": totals,
        "budgets": budgets,
        "parts": parts,
    }
    if seams:
        report["seam"] = [
            _seam(spec, depsgraph, failures) if isinstance(spec, dict)
            else failures.append("seam: expected an object, got {!r}".format(spec))
            for spec in seams
        ]
    if clearances:
        report["clearance"] = [
            _clearance(spec, depsgraph, failures) if isinstance(spec, dict)
            else failures.append("clearance: expected an object, got {!r}".format(spec))
            for spec in clearances
        ]
    report["failures"] = failures
    report["passed"] = not failures
    return report


# -----------------------------------------------------------------------------
# Scene state

def _rounded(values, digits=6):
    # Adding zero turns -0.0 into 0.0, which would otherwise read as a change.
    return [round(v, digits) + 0.0 for v in values]


def scene_state():
    """One record per object of the active scene, each with a hash of its fields."""
    from mathutils import Vector
    depsgraph = bpy.context.evaluated_depsgraph_get()
    state = {}
    for ob in bpy.context.scene.objects:
        if is_checkpoint_object(ob):
            continue
        ob_eval = ob.evaluated_get(depsgraph)
        matrix = ob_eval.matrix_world
        corners = [matrix @ Vector(corner) for corner in ob_eval.bound_box]
        record = {
            "type": ob.type,
            "verts": 0,
            "tris": 0,
            "bbox_min": _rounded(min(c[i] for c in corners) for i in range(3)),
            "bbox_max": _rounded(max(c[i] for c in corners) for i in range(3)),
            "materials": [slot.material.name if slot.material else None for slot in ob.material_slots],
            "modifiers": [modifier.type for modifier in ob.modifiers],
            "transform": _rounded(v for row in matrix for v in row),
        }
        if ob.type == 'MESH':
            mesh = ob_eval.data
            record["verts"] = len(mesh.vertices)
            record["tris"] = _triangles(mesh)
            # The bounding box misses edits inside it; the positions themselves do not.
            coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", coords)
            record["geometry"] = hashlib.sha1(coords.tobytes()).hexdigest()[:16]
        record["hash"] = hashlib.sha1(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        state[ob.name] = record
    return state


def diff_states(before, after):
    changed = {}
    for name in sorted(before.keys() & after.keys()):
        if before[name]["hash"] == after[name]["hash"]:
            continue
        changed[name] = {
            key: {"before": before[name].get(key), "after": after[name].get(key)}
            for key in sorted(before[name].keys() | after[name].keys())
            if key != "hash" and before[name].get(key) != after[name].get(key)
        }
    return {
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
        "changed": changed,
        "unchanged": len(before.keys() & after.keys()) - len(changed),
    }


def snapshot(name):
    state = scene_state()
    _snapshots.pop(name, None)
    _snapshots[name] = state
    while len(_snapshots) > MAX_SNAPSHOTS:
        _snapshots.popitem(last=False)
    return state


def stored(name):
    if name not in _snapshots:
        raise ValueError("no snapshot named {!r}; known: {:s}".format(name, ", ".join(_snapshots) or "none"))
    return _snapshots[name]


def snapshot_names():
    return list(_snapshots)
