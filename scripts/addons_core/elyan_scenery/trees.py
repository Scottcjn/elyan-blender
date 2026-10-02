# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Trees and plants: small seeded meshes meant to be scattered by the hundred.

A plant is a list of parts (trunk, branches, foliage), each a few dozen
triangles. Wood is tapered tubes swept along wandering paths; foliage is
solid low-polygon shapes (lumps, skirts, fronds) with no alpha, which keeps
ray tracing fast and the look soft at scenery distances.
"""

import math
import random

import bpy
from bpy.props import EnumProperty, FloatProperty, IntProperty
from bpy.types import Operator
from mathutils import Quaternion, Vector, noise

from . import materials

# Natural height is in metres.
SPECIES = {
    'CONIFER': {"label": "Conifer", "description": "Fir: a straight trunk under skirts of needles", "height": 12.0},
    'BROADLEAF': {"label": "Broadleaf", "description": "A forking trunk under a rounded crown", "height": 9.0},
    'PALM': {"label": "Palm", "description": "A leaning trunk with a crown of arching fronds", "height": 8.0},
    'DEAD': {"label": "Dead Tree", "description": "Bare, crooked branches", "height": 7.0},
    'BUSH': {"label": "Bush", "description": "A low mound of leaves", "height": 1.6},
}

SPECIES_ITEMS = tuple((key, spec["label"], spec["description"]) for key, spec in SPECIES.items())

# Detail levels tried in turn, finest first, until a plant fits its triangle budget.
_QUALITIES = (1.0, 0.8, 0.62, 0.46, 0.32, 0.2, 0.1, 0.0)

_WOOD, _LEAF = 0, 1


def _count(faces):
    return sum(len(face) - 2 for face in faces)


def _wobble(rng):
    return Vector((rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)))


def _path(rng, start, direction, length, segments, wander, lift):
    """Points along a branch that strays by ``wander`` each step and bends up (or down) by ``lift``."""
    points = [start.copy()]
    heading = direction.normalized()
    step = length / segments
    for _ in range(segments):
        heading = (heading + _wobble(rng) * wander + Vector((0.0, 0.0, lift))).normalized()
        points.append(points[-1] + heading * step)
    return points


def _along(points, t):
    """Position and direction at ``t`` (0..1) of a path."""
    x = min(max(t, 0.0), 0.999) * (len(points) - 1)
    i = int(x)
    return points[i].lerp(points[i + 1], x - i), (points[i + 1] - points[i]).normalized()


def _aside(direction, angle, turn):
    """``direction`` tipped over by ``angle`` and swung around itself by ``turn``."""
    tipped = Quaternion(direction.orthogonal().normalized(), angle) @ direction
    return Quaternion(direction, turn) @ tipped


def _tube(points, radii, sides):
    """A tapered tube closed to a point at its tip, open at its base (which is always buried)."""
    verts, faces = [], []
    normal = (points[1] - points[0]).orthogonal().normalized()
    rings = len(points) - 1
    for i in range(rings):
        tangent = (points[i + 1] - points[max(i - 1, 0)]).normalized()
        # Carry the previous ring's orientation along, so the tube does not twist.
        normal = (normal - tangent * normal.dot(tangent)).normalized()
        binormal = tangent.cross(normal)
        for s in range(sides):
            angle = math.tau * s / sides
            verts.append(points[i] + (normal * math.cos(angle) + binormal * math.sin(angle)) * radii[i])
    verts.append(points[-1])
    tip = len(verts) - 1
    for i in range(rings - 1):
        for s in range(sides):
            a, b = i * sides + s, i * sides + (s + 1) % sides
            faces.append((a, b, b + sides, a + sides))
    last = (rings - 1) * sides
    for s in range(sides):
        faces.append((last + s, last + (s + 1) % sides, tip))
    return verts, faces


def _icosphere(level):
    """Unit icosphere as (points, triangles): 20 triangles at level 0, 80 at level 1."""
    g = (1.0 + math.sqrt(5.0)) / 2.0
    points = [
        Vector(p).normalized() for p in (
            (-1, g, 0), (1, g, 0), (-1, -g, 0), (1, -g, 0), (0, -1, g), (0, 1, g),
            (0, -1, -g), (0, 1, -g), (g, 0, -1), (g, 0, 1), (-g, 0, -1), (-g, 0, 1),
        )
    ]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
        (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
        (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    for _ in range(level):
        middle = {}
        finer = []

        def between(a, b):
            key = (min(a, b), max(a, b))
            if key not in middle:
                points.append((points[a] + points[b]).normalized())
                middle[key] = len(points) - 1
            return middle[key]

        for a, b, c in faces:
            ab, bc, ca = between(a, b), between(b, c), between(c, a)
            finer.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = finer
    return points, faces


_ICOSPHERES = {}


def _lump(rng, center, radius, level, squash=0.75):
    """A clump of leaves: a ball made lumpy by noise, a little flattened."""
    if level not in _ICOSPHERES:
        _ICOSPHERES[level] = _icosphere(level)
    points, faces = _ICOSPHERES[level]
    offset = _wobble(rng) * 50.0
    stretch = Vector((rng.uniform(0.8, 1.25), rng.uniform(0.8, 1.25), squash * rng.uniform(0.85, 1.15)))
    verts = [
        center + point * stretch * (radius * (1.0 + 0.45 * noise.noise(point * 1.3 + offset)))
        for point in points
    ]
    return verts, list(faces)


def _skirt(rng, center, radius, height, points):
    """One tier of a fir: a cone whose hem hangs in uneven points."""
    verts = [center + Vector((0.0, 0.0, height))]
    turn = rng.uniform(0.0, math.tau)
    for i in range(points * 2):
        angle = turn + math.tau * i / (points * 2)
        # Odd points are the notches between boughs; boughs droop below the notches.
        reach = radius * (rng.uniform(0.85, 1.15) if i % 2 == 0 else rng.uniform(0.5, 0.65))
        drop = -radius * (rng.uniform(0.15, 0.4) if i % 2 == 0 else 0.0)
        verts.append(center + Vector((math.cos(angle) * reach, math.sin(angle) * reach, drop)))
    count = points * 2
    faces = [(0, 1 + i, 1 + (i + 1) % count) for i in range(count)]
    return verts, faces


def _frond(rng, start, direction, length, width, segments):
    """A palm leaf: a strip folded along its midrib that arches out and hangs under its own weight."""
    spine = [start.copy()]
    heading = direction.normalized()
    for i in range(segments):
        # Weight pulls harder the further out the leaf has got.
        heading = (heading + Vector((0.0, 0.0, -0.55 * (i + 1) / segments)) + _wobble(rng) * 0.05).normalized()
        spine.append(spine[-1] + heading * (length / segments))
    verts, faces = [], []
    for i, point in enumerate(spine[:-1]):
        tangent = (spine[i + 1] - point).normalized()
        side = tangent.cross(Vector((0.0, 0.0, 1.0)))
        side = side.normalized() if side.length > 1e-4 else Vector((1.0, 0.0, 0.0))
        up = side.cross(tangent)
        t = i / segments
        half = width * (0.25 + 0.75 * math.sin(math.pi * min(1.0, 0.15 + t))) * 0.5
        # The blades hang down from the midrib, so the fold is an upside-down V.
        verts.extend((point - side * half - up * half * 0.7, point, point + side * half - up * half * 0.7))
    verts.append(spine[-1])
    tip = len(verts) - 1
    for i in range(segments - 1):
        a = i * 3
        faces.append((a, a + 1, a + 4, a + 3))
        faces.append((a + 1, a + 2, a + 5, a + 4))
    a = (segments - 1) * 3
    faces.extend(((a, a + 1, tip), (a + 1, a + 2, tip)))
    return verts, faces


def _trunk(rng, height, radius, sides, segments, lean, wander, flare=1.5):
    """Trunk path and radii. The base is sunk below the ground so it does not float on a slope."""
    start = Vector((0.0, 0.0, -0.06 * height))
    direction = Vector((lean * math.cos(rng.uniform(0.0, math.tau)), lean * math.sin(rng.uniform(0.0, math.tau)), 1.0))
    points = _path(rng, start, direction, height * 1.06, segments, wander, 0.25 * lean + 0.05)
    radii = [radius * (1.0 - 0.8 * i / segments) for i in range(segments + 1)]
    radii[0] *= flare
    return points, radii, _tube(points, radii, sides)


def _branches(rng, parts, parent, parent_radius, count, first, length, angle, wander, lift, sides, segments):
    """Grow ``count`` side branches from a path, spiralling around it. Returns their paths."""
    grown = []
    turn = rng.uniform(0.0, math.tau)
    for i in range(count):
        t = first + (1.0 - first) * (i + rng.uniform(0.2, 0.8)) / count
        start, tangent = _along(parent, t)
        # The golden angle spreads branches so that none sits right above another.
        turn += 2.4 + rng.uniform(-0.5, 0.5)
        direction = _aside(tangent, angle * rng.uniform(0.75, 1.25), turn)
        reach = length * rng.uniform(0.7, 1.15) * (1.0 - 0.45 * t)
        points = _path(rng, start, direction, reach, segments, wander, lift)
        radius = parent_radius * (1.0 - 0.8 * t) * 0.62
        radii = [radius * (1.0 - 0.85 * k / segments) for k in range(segments + 1)]
        parts.append((_WOOD, *_tube(points, radii, sides)))
        grown.append((points, radius))
    return grown


# -----------------------------------------------------------------------------
# Species. Each returns parts in order of importance: when even the coarsest
# level is over budget, parts are dropped from the end.

def _conifer(rng, height, q):
    parts = []
    radius = height * 0.028
    points, _radii, tube = _trunk(rng, height, radius, 3 + round(3 * q), 2 + round(2 * q), 0.03, 0.02)
    parts.append((_WOOD, *tube))
    tiers = 4 + round(9 * q)
    boughs = 4 + round(5 * q)
    bottom = rng.uniform(0.14, 0.24)
    spread = height * rng.uniform(0.20, 0.27)
    # The top tiers come first: a fir that loses parts to its budget keeps its point.
    for i in reversed(range(tiers)):
        t = bottom + (1.0 - bottom) * i / tiers
        center, _tangent = _along(points, t)
        tier_radius = spread * (1.0 - 0.9 * i / tiers) * rng.uniform(0.9, 1.1)
        tier_height = height * (1.0 - bottom) / tiers * 2.4
        if i == tiers - 1:
            tier_height = (points[-1] - center).z * 1.05
        parts.append((_LEAF, *_skirt(rng, center, tier_radius, tier_height, boughs)))
    return parts


def _broadleaf(rng, height, q):
    parts = []
    radius = height * 0.04
    sides = 3 + round(4 * q)
    trunk_height = height * rng.uniform(0.55, 0.68)
    points, _radii, tube = _trunk(rng, trunk_height, radius, sides, 3 + round(2 * q), 0.12, 0.10)
    parts.append((_WOOD, *tube))
    wood = []
    boughs = _branches(
        rng, wood, points, radius, 3 + round(3 * q), 0.42, height * 0.42, 0.95, 0.16, 0.22,
        max(3, sides - 2), 2 + round(q),
    )
    # A few rounded lumps look better than many angular ones, so roundness is given up last.
    level = 1 if q >= 0.3 else 0
    crown = height * rng.uniform(0.17, 0.21)
    leaves = [_lump(rng, points[-1] + Vector((0.0, 0.0, crown * 0.3)), crown * 1.25, level)]
    twigs = []
    for path, bough_radius in boughs:
        leaves.append(_lump(rng, path[-1], crown * rng.uniform(0.85, 1.2), level))
        if q >= 0.8:
            twigs.extend(_branches(
                rng, wood, path, bough_radius, 1 + round(q), 0.45, height * 0.2, 0.8, 0.2, 0.15, 3, 2,
            ))
    for path, _radius in twigs:
        leaves.append(_lump(rng, path[-1], crown * rng.uniform(0.6, 0.9), level))
    # Leaves and the wood that carries them alternate, so trimming never leaves a bare stick first.
    for i in range(max(len(leaves), len(wood))):
        if i < len(leaves):
            parts.append((_LEAF, *leaves[i]))
        if i < len(wood):
            parts.append(wood[i])
    return parts


def _palm(rng, height, q):
    parts = []
    radius = height * 0.022
    trunk_height = height * 0.82
    points, _radii, _tube_unused = _trunk(rng, trunk_height, radius, 3, 3 + round(4 * q), 0.28, 0.03)
    # Palms barely taper: sweep the tube again with a fat top for the crown to sit on.
    radii = [radius * (1.0 - 0.3 * i / (len(points) - 1)) for i in range(len(points))]
    radii[0] *= 1.8
    tube = _tube(points + [points[-1] + Vector((0.0, 0.0, radius))], radii + [0.0], 3 + round(4 * q))
    parts.append((_WOOD, *tube))
    top = points[-1]
    count = 7 + round(9 * q)
    segments = 3 + round(3 * q)
    turn = rng.uniform(0.0, math.tau)
    for i in range(count):
        turn += 2.4 + rng.uniform(-0.3, 0.3)
        # Young leaves stand up in the middle, old ones hang at the outside.
        tilt = rng.uniform(0.25, 1.35)
        direction = Vector((math.cos(turn) * math.sin(tilt), math.sin(turn) * math.sin(tilt), math.cos(tilt)))
        length = height * rng.uniform(0.34, 0.46)
        parts.append((_LEAF, *_frond(rng, top, direction, length, length * 0.3, segments)))
    return parts


def _dead(rng, height, q):
    parts = []
    radius = height * 0.045
    sides = 3 + round(3 * q)
    points, _radii, tube = _trunk(rng, height * 0.85, radius, sides, 4 + round(3 * q), 0.2, 0.16)
    parts.append((_WOOD, *tube))
    boughs = _branches(
        rng, parts, points, radius, 3 + round(4 * q), 0.3, height * 0.5, 1.0, 0.28, 0.12,
        max(3, sides - 1), 3 + round(2 * q),
    )
    if q >= 0.2:
        for path, bough_radius in boughs:
            _branches(rng, parts, path, bough_radius, 1 + round(2 * q), 0.3, height * 0.22, 0.9, 0.3, 0.1, 3, 2 + round(q))
    return parts


def _bush(rng, height, q):
    parts = []
    level = 1 if q >= 0.3 else 0
    count = 3 + round(5 * q)
    width = height * rng.uniform(0.5, 0.8)
    parts.append((_LEAF, *_lump(rng, Vector((0.0, 0.0, height * 0.45)), height * 0.5, level, 0.9)))
    for _ in range(count):
        angle, reach = rng.uniform(0.0, math.tau), width * rng.uniform(0.35, 1.0)
        size = height * rng.uniform(0.28, 0.45)
        center = Vector((math.cos(angle) * reach, math.sin(angle) * reach, size * rng.uniform(0.5, 0.9)))
        parts.append((_LEAF, *_lump(rng, center, size, level, 0.85)))
    if q >= 0.3:
        for _ in range(2 + round(2 * q)):
            direction = Vector((rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), 1.0))
            points = _path(rng, Vector((0.0, 0.0, -0.05 * height)), direction, height * 0.6, 2, 0.1, 0.0)
            parts.append((_WOOD, *_tube(points, [height * 0.035, height * 0.025, 0.0], 3)))
    return parts


_BUILDERS = {
    'CONIFER': _conifer,
    'BROADLEAF': _broadleaf,
    'PALM': _palm,
    'DEAD': _dead,
    'BUSH': _bush,
}


def build_parts(species, seed, height, max_triangles):
    """
    Parts of a plant that together stay within ``max_triangles``.

    The same species, seed, height and budget always give the same plant.
    Raises ``ValueError`` when not even a trunk (or one lump) fits.
    """
    for quality in _QUALITIES:
        # The random stream restarts for every attempt, a coarser try is not a different tree.
        parts = _BUILDERS[species](random.Random(seed), height, quality)
        if sum(_count(faces) for _kind, _verts, faces in parts) <= max_triangles:
            return parts
    kept, total = [], 0
    for part in parts:
        triangles = _count(part[2])
        if total + triangles > max_triangles:
            break
        kept.append(part)
        total += triangles
    if not kept:
        raise ValueError("{:d} triangles is too few for any {:s}".format(max_triangles, SPECIES[species]["label"]))
    return kept


def build_mesh(species, seed=1, height=0.0, max_triangles=600):
    """Mesh of a plant standing on the origin. ``height`` 0 is the species' natural height."""
    height = height if height > 0.0 else SPECIES[species]["height"]
    verts, faces, kinds = [], [], []
    for kind, part_verts, part_faces in build_parts(species, seed, height, max_triangles):
        base = len(verts)
        verts.extend(part_verts)
        faces.extend(tuple(index + base for index in face) for face in part_faces)
        kinds.extend([kind] * len(part_faces))
    mesh = bpy.data.meshes.new(SPECIES[species]["label"])
    mesh.from_pydata(verts, [], faces)
    mesh.polygons.foreach_set("material_index", kinds)
    mesh.shade_smooth()
    # Slot order matches _WOOD, _LEAF.
    mesh.materials.append(materials.bark(species))
    if _LEAF in kinds:
        mesh.materials.append(materials.foliage(species))
    mesh.update()
    return mesh


def triangle_count(mesh):
    return sum(len(polygon.vertices) - 2 for polygon in mesh.polygons)


class ELYAN_OT_tree_add(Operator):
    """Add a tree or plant at the 3D cursor"""
    bl_idname = "elyan_scenery.tree_add"
    bl_label = "Tree"
    bl_options = {'REGISTER', 'UNDO'}

    species: EnumProperty(name="Species", items=SPECIES_ITEMS)
    seed: IntProperty(name="Seed", description="Same seed, same plant", default=1, min=0)
    height: FloatProperty(
        name="Height", description="Zero uses the natural height of the species",
        default=0.0, min=0.0, soft_max=40.0, unit='LENGTH',
    )
    triangles: IntProperty(
        name="Triangles", description="The plant never has more triangles than this",
        default=600, min=8, soft_max=5000, max=50000,
    )

    def execute(self, context):
        try:
            mesh = build_mesh(self.species, self.seed, self.height, self.triangles)
        except ValueError as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        ob = bpy.data.objects.new(mesh.name, mesh)
        context.collection.objects.link(ob)
        ob.location = context.scene.cursor.location
        for other in context.selected_objects:
            other.select_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
        self.report({'INFO'}, "{:s}, {:d} triangles".format(mesh.name, triangle_count(mesh)))
        return {'FINISHED'}


classes = (
    ELYAN_OT_tree_add,
)
