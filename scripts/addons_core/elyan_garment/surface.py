# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Shared measuring for the garment tools: rest positions, the body as a surface, clearance.

Everything is measured unposed and in world space: a garment is cut on the
body at rest, and a correction read off a bent arm would only hold in that
pose. The garment is read as stored (its basis shape, no modifiers), because
that is what the tools write. The body is read as shown, with its armatures
held at rest: a generated person's shape lives in shape keys that are switched
on, and its mesh carries helper geometry that a Mask modifier hides, so the
stored basis is not the body anyone sees. Distances are float64 and unrounded.
"""

import contextlib

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# Below this length a direction is noise, not a direction.
TINY = 1e-12


# -----------------------------------------------------------------------------
# Arguments

def mesh_object(value, role):
    """The mesh object ``value`` names (or is). ``role`` words the error."""
    ob = bpy.data.objects.get(value) if isinstance(value, str) else value
    if ob is None:
        raise ValueError("{:s}: no object named {!r}".format(role, value))
    if getattr(ob, "type", None) != 'MESH':
        raise ValueError("{:s}: {!r} is not a mesh object".format(role, getattr(ob, "name", ob)))
    if len(ob.data.vertices) == 0:
        raise ValueError("{:s}: {!r} has no vertices".format(role, ob.name))
    return ob


def writable(ob):
    """Refuse a mesh that is open in Edit Mode: leaving the mode would overwrite what is written here."""
    if ob.data.is_editmode:
        raise ValueError("{!r} is in Edit Mode; leave Edit Mode first".format(ob.name))
    return ob


def names(value):
    """``None``, one name or a list of names, as a list."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


# -----------------------------------------------------------------------------
# Positions

def _coords(collection, count):
    co = np.empty(count * 3, dtype=np.float32)
    collection.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def rest_local(ob):
    """Rest positions in the object's own space: the basis shape key when there are keys."""
    mesh = ob.data
    if mesh.shape_keys:
        return _coords(mesh.shape_keys.reference_key.data, len(mesh.vertices))
    return _coords(mesh.vertices, len(mesh.vertices))


def key_local(ob, block):
    return _coords(block.data, len(ob.data.vertices))


def matrix(ob):
    return np.array(ob.matrix_world, dtype=np.float64)


def to_world(ob, local):
    m = matrix(ob)
    return local @ m[:3, :3].T + m[:3, 3]


def vectors_to_world(ob, local):
    return local @ matrix(ob)[:3, :3].T


def vectors_to_local(ob, world):
    return world @ np.linalg.inv(matrix(ob)[:3, :3]).T


def rest_world(ob):
    return to_world(ob, rest_local(ob))


def displace(ob, delta_world):
    """
    Add a world-space displacement to the mesh and to every shape key.

    Shape keys store whole positions, so a move that reached only the basis
    would be undone, vertex by vertex, as soon as another key is turned on.
    Returns the number of shape keys written.
    """
    mesh = ob.data
    count = len(mesh.vertices)
    delta = vectors_to_local(ob, delta_world)
    blocks = mesh.shape_keys.key_blocks if mesh.shape_keys else ()
    for block in blocks:
        block.data.foreach_set("co", (_coords(block.data, count) + delta).astype(np.float32).ravel())
    if blocks:
        # The mesh's own vertices mirror the basis key.
        base = _coords(mesh.shape_keys.reference_key.data, count)
    else:
        base = _coords(mesh.vertices, count) + delta
    mesh.vertices.foreach_set("co", base.astype(np.float32).ravel())
    mesh.update()
    return len(blocks)


def stored(ob, delta_world):
    """Where ``displace`` would leave the rest positions, in world space, after float32 storage."""
    local = rest_local(ob) + vectors_to_local(ob, delta_world)
    return to_world(ob, local.astype(np.float32).astype(np.float64))


@contextlib.contextmanager
def at_rest(*objects):
    """Hold the armatures that deform ``objects`` in their rest position for the duration."""
    armatures = {
        modifier.object.data for ob in objects for modifier in ob.modifiers
        if modifier.type == 'ARMATURE' and modifier.object is not None}
    saved = [(armature, armature.pose_position) for armature in armatures]
    try:
        for armature in armatures:
            armature.pose_position = 'REST'
        yield
    finally:
        for armature, position in saved:
            armature.pose_position = position
        bpy.context.view_layer.update()


def evaluated(ob):
    """
    ``ob`` as its shape keys and modifiers leave it: world positions and the mesh they came from.

    The mesh belongs to the dependency graph and is only good until the next
    change, so anything needed from it has to be read straight away.
    """
    bpy.context.view_layer.update()
    ob_eval = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = ob_eval.data
    world = np.array(ob_eval.matrix_world, dtype=np.float64)
    return _coords(mesh.vertices, len(mesh.vertices)) @ world[:3, :3].T + world[:3, 3], mesh


# -----------------------------------------------------------------------------
# Topology

def triangles(mesh):
    mesh.calc_loop_triangles()
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tris)
    return tris.reshape(-1, 3)


def polygons(mesh):
    return [tuple(polygon.vertices) for polygon in mesh.polygons]


def edges(mesh):
    pairs = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", pairs)
    return pairs.reshape(-1, 2)


def neighbours(mesh):
    """For each vertex, the vertices one edge away."""
    result = [[] for _ in range(len(mesh.vertices))]
    for a, b in edges(mesh).tolist():
        result[a].append(b)
        result[b].append(a)
    return result


def boundary_loops(mesh):
    """
    Open boundaries as ordered, closed runs of vertex indices.

    Returns ``(loops, tangled)``: ``tangled`` counts boundaries that are not a
    simple ring (a vertex with more than two open edges), which are left out
    because they have no single order to match against a ring.
    """
    uses = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("edge_index", uses)
    open_edges = edges(mesh)[np.bincount(uses, minlength=len(mesh.edges)) == 1]
    links = {}
    for a, b in open_edges.tolist():
        links.setdefault(a, []).append(b)
        links.setdefault(b, []).append(a)

    loops, tangled, seen = [], 0, set()
    for start in sorted(links):
        if start in seen:
            continue
        # Everything connected to ``start`` along open edges.
        group, stack = {start}, [start]
        while stack:
            for other in links[stack.pop()]:
                if other not in group:
                    group.add(other)
                    stack.append(other)
        seen |= group
        if any(len(links[index]) != 2 for index in group):
            tangled += 1
            continue
        loop, previous, current = [start], None, start
        while True:
            a, b = links[current]
            following = b if a == previous else a
            if following == start:
                break
            loop.append(following)
            previous, current = current, following
        loops.append(loop)
    return loops, tangled


# -----------------------------------------------------------------------------
# Vertex groups

def group_weight(ob, group_names, mesh=None):
    """
    Per vertex, the summed weight in the named vertex groups. A missing group is an error.

    ``mesh``: read the weights from this (evaluated) mesh of ``ob`` instead of the stored one.
    """
    missing = [name for name in group_names if name not in ob.vertex_groups]
    if missing:
        raise ValueError("{!r} has no vertex group {:s}".format(ob.name, ", ".join(repr(n) for n in missing)))
    wanted = {ob.vertex_groups[name].index for name in group_names}
    mesh = ob.data if mesh is None else mesh
    total = np.zeros(len(mesh.vertices), dtype=np.float64)
    for vert in mesh.vertices:
        for element in vert.groups:
            if element.group in wanted:
                total[vert.index] += element.weight
    return total


def group_mask(ob, group_names):
    """Which vertices carry any weight in the named groups (the rule ``check`` uses for ``ignore_groups``)."""
    return group_weight(ob, group_names) > 0.0


# -----------------------------------------------------------------------------
# The body as a surface

class Surface:
    """A triangle mesh in world space that can be asked for its nearest point."""

    def __init__(self, verts, tris, polygons=None):
        if len(tris) == 0:
            raise ValueError("the body has no faces")
        self.verts = verts
        self.tris = tris
        self.polygons = polygons
        # World-space tree: the object-space one would distort distances under a scaled body.
        self.tree = BVHTree.FromPolygons(verts.tolist(), tris.tolist())
        self._faces = None
        self._normals = None

    @classmethod
    def from_object(cls, ob):
        """The body as shown, unposed."""
        with at_rest(ob):
            verts, mesh = evaluated(ob)
            return cls(verts, triangles(mesh), polygons(mesh))

    def moved(self, delta):
        """The same surface with every vertex displaced: the body with a shape key on."""
        return Surface(self.verts + delta, self.tris, self.polygons)

    @staticmethod
    def _query(tree, points):
        count = len(points)
        location = np.empty((count, 3), dtype=np.float64)
        normal = np.empty((count, 3), dtype=np.float64)
        face = np.empty(count, dtype=np.int64)
        distance = np.empty(count, dtype=np.float64)
        find = tree.find_nearest
        for index, point in enumerate(points.tolist()):
            point = Vector(point)
            found, face_normal, found_face, length = find(point)
            if found is None:
                raise RuntimeError("no body surface found near garment vertex {:d}".format(index))
            location[index] = found
            normal[index] = face_normal
            face[index] = found_face
            # Behind the nearest face means inside the body: poke-through, however small the distance.
            distance[index] = -length if (point - found).dot(face_normal) < 0.0 else length
        return location, normal, face, distance

    def nearest(self, points):
        """For each point: the nearest surface point, that triangle's normal, its index, and the signed distance."""
        return self._query(self.tree, points)

    def clearance(self, points):
        """
        For each point: the nearest surface point, the face normal there and the signed distance.

        This is the measurement of ``elyan_llm.checks``, built the same way (a tree
        of the body's own faces, the sign from the nearest face's normal), so
        the two agree to the last digit also where a four-sided face is not flat.
        """
        if self.polygons is None:
            location, normal, _face, distance = self._query(self.tree, points)
            return location, normal, distance
        if self._faces is None:
            self._faces = BVHTree.FromPolygons(self.verts.tolist(), self.polygons)
        location, normal, _face, distance = self._query(self._faces, points)
        return location, normal, distance

    def distance(self, points):
        return self.clearance(points)[2]

    def barycentric(self, location, tri):
        """Weights of each triangle's three corners that give ``location``."""
        a, b, c = (self.verts[self.tris[tri, corner]] for corner in range(3))
        v0, v1, v2 = b - a, c - a, location - a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)
        denominator = d00 * d11 - d01 * d01
        flat = np.abs(denominator) < TINY
        denominator[flat] = 1.0
        v = (d11 * d20 - d01 * d21) / denominator
        w = (d00 * d21 - d01 * d20) / denominator
        weights = np.stack((1.0 - v - w, v, w), axis=1)
        # A zero-area triangle has no inside; its first corner stands for it.
        weights[flat] = (1.0, 0.0, 0.0)
        # The nearest point is on the triangle, so anything outside 0..1 is rounding.
        weights = np.clip(weights, 0.0, 1.0)
        return weights / weights.sum(axis=1, keepdims=True)

    def interpolate(self, values, tri, weights):
        """Per-vertex ``values`` carried to points given by triangle and corner weights."""
        corners = values[self.tris[tri]]
        return np.einsum("ij,ij...->i...", weights, corners)

    def vertex_normals(self):
        """Area-weighted vertex normals, computed here so they exist for a shape that is not on screen."""
        if self._normals is None:
            self._normals = vertex_normals(self.verts, self.tris)
        return self._normals


def vertex_normals(verts, tris):
    a, b, c = (verts[tris[:, corner]] for corner in range(3))
    # The cross product's length is twice the area, which is the weighting wanted.
    face = np.cross(b - a, c - a)
    normals = np.zeros_like(verts)
    for corner in range(3):
        np.add.at(normals, tris[:, corner], face)
    return unit(normals)


def unit(vectors):
    length = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(length < TINY, 1.0, length)


# -----------------------------------------------------------------------------
# Reports

def clearance_stats(distances, threshold):
    """The numbers ``check`` reports for clearance, under the same names."""
    worst = int(distances.argmin())
    return {
        "threshold": float(threshold),
        "verts": int(len(distances)),
        "min": float(distances.min()),
        "percentile_5": float(np.percentile(distances, 5)),
        "median": float(np.median(distances)),
        "below_threshold": int((distances < threshold).sum()),
        "inside": int((distances < 0.0).sum()),
        "worst_vertex": worst,
    }


def displayed_clearance(garment, body, threshold):
    """
    The bridge's own ``check`` clearance, as a second opinion on what is on screen.

    It measures both meshes after modifiers, pose and shape key values, so it
    agrees with the numbers here while the rig is at rest and the garment's own
    keys are off. None when the bridge add-on is not installed.
    """
    try:
        from elyan_llm import checks
    except ImportError:
        return None
    report = checks.check({"clearance": {
        "garment": garment.name, "body": body.name, "threshold": threshold, "max_below": 10 ** 9}})
    entry = report["clearance"][0]
    if report["failures"]:
        return {"failures": report["failures"]}
    return {key: entry[key] for key in ("min", "percentile_5", "median", "below_threshold", "inside")}


def refusal(report, reason):
    """Mark ``report`` as a refusal: nothing was written."""
    report["ok"] = False
    report["applied"] = False
    report["refused"] = reason
    return report
