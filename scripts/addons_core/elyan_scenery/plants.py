# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Small plants for places seen close up: wildflowers, lily pads and grass.

A drift of flowers is one mesh, not hundreds of objects, so a meadow stays
light in the outliner and in the file.
"""

import math

import bpy
import numpy as np

from . import materials

# petals: colour and count; length and stem in metres; heart is the centre's colour.
FLOWERS = {
    'DAISY': {"label": "Daisy", "petal": (0.95, 0.95, 0.92), "heart": (0.95, 0.70, 0.05),
              "petals": 12, "length": 0.055, "stem": 0.24},
    'POPPY': {"label": "Poppy", "petal": (0.85, 0.08, 0.04), "heart": (0.05, 0.04, 0.03),
              "petals": 5, "length": 0.070, "stem": 0.40},
    'BLUEBELL': {"label": "Bluebell", "petal": (0.35, 0.30, 0.85), "heart": (0.90, 0.90, 0.60),
                 "petals": 6, "length": 0.045, "stem": 0.30},
    'BUTTERCUP': {"label": "Buttercup", "petal": (1.00, 0.80, 0.05), "heart": (0.90, 0.60, 0.05),
                  "petals": 5, "length": 0.038, "stem": 0.20},
    'LAVENDER': {"label": "Lavender", "petal": (0.48, 0.32, 0.72), "heart": (0.40, 0.28, 0.60),
                 "petals": 8, "length": 0.030, "stem": 0.45},
}
FLOWER_ITEMS = tuple((key, spec["label"], "") for key, spec in FLOWERS.items())

_STEM, _PETAL, _HEART = 0, 1, 2
_LEAF_GREEN = (0.05, 0.20, 0.03)


def _flower(spec):
    """One flower standing on the origin: (vertices, faces, material of each face)."""
    verts, faces, kinds = [], [], []
    height, length = spec["stem"], spec["length"]

    def quad(a, b, c, d, kind):
        base = len(verts)
        verts.extend((a, b, c, d))
        faces.append((base, base + 1, base + 2, base + 3))
        kinds.append(kind)

    # Stem: three thin sides.
    radius = 0.004
    ring = [(math.cos(i * math.tau / 3) * radius, math.sin(i * math.tau / 3) * radius) for i in range(3)]
    for i in range(3):
        (x0, y0), (x1, y1) = ring[i], ring[(i + 1) % 3]
        quad((x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, height), (x0, y0, height), _STEM)
    # Two leaves low on the stem.
    for side in (1.0, -1.0):
        z = height * 0.3
        quad((0.0, 0.0, z), (side * 0.03, -0.012, z + 0.03), (side * 0.07, 0.0, z + 0.035),
             (side * 0.03, 0.012, z + 0.03), _STEM)
    # Petals fan out from the top, lifted a little so the flower is a shallow cup.
    count = spec["petals"]
    half = math.pi / count * 0.85
    for i in range(count):
        angle = i * math.tau / count
        inner, outer = length * 0.25, length
        points = []
        for reach, spread, lift in ((inner, -half, 0.0), (outer, -half * 0.6, length * 0.3),
                                    (outer, half * 0.6, length * 0.3), (inner, half, 0.0)):
            points.append((math.cos(angle + spread) * reach, math.sin(angle + spread) * reach, height + lift))
        quad(*points, _PETAL)
    # The heart: a low six-sided cone.
    base = len(verts)
    verts.append((0.0, 0.0, height + length * 0.22))
    for i in range(6):
        verts.append((math.cos(i * math.tau / 6) * length * 0.28, math.sin(i * math.tau / 6) * length * 0.28, height))
    for i in range(6):
        faces.append((base, base + 1 + i, base + 1 + (i + 1) % 6))
        kinds.append(_HEART)
    return np.array(verts, dtype=np.float32), faces, kinds


def _merged(name, single, places, seed, scale):
    """One mesh holding a turned and scaled copy of ``single`` at every place."""
    verts, faces, kinds = single
    rng = np.random.default_rng(seed)
    all_verts = np.empty((len(places) * len(verts), 3), dtype=np.float32)
    all_faces, all_kinds = [], []
    for index, place in enumerate(places):
        turn = rng.uniform(0.0, math.tau)
        size = rng.uniform(scale[0], scale[1])
        cos, sin = math.cos(turn), math.sin(turn)
        block = verts * size
        x = block[:, 0] * cos - block[:, 1] * sin
        y = block[:, 0] * sin + block[:, 1] * cos
        start = index * len(verts)
        all_verts[start:start + len(verts), 0] = x + place[0]
        all_verts[start:start + len(verts), 1] = y + place[1]
        all_verts[start:start + len(verts), 2] = block[:, 2] + place[2]
        all_faces.extend(tuple(i + start for i in face) for face in faces)
        all_kinds.extend(kinds)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(all_verts.tolist(), [], all_faces)
    mesh.polygons.foreach_set("material_index", all_kinds)
    mesh.update()
    return mesh


def flower_mesh(species, places, seed):
    """All the flowers of one species as a single mesh. ``places`` are (x, y, z) rows."""
    spec = FLOWERS[species]
    mesh = _merged(spec["label"], _flower(spec), places, seed, (0.75, 1.25))
    # Slot order matches _STEM, _PETAL, _HEART.
    mesh.materials.append(materials.plain("Stem", _LEAF_GREEN, 0.7))
    mesh.materials.append(materials.plain("Petal " + spec["label"], spec["petal"], 0.5))
    mesh.materials.append(materials.plain("Heart " + spec["label"], spec["heart"], 0.7))
    return mesh


def _pad():
    """A lily pad: a disc with the wedge cut out, one unit across."""
    verts = [(0.0, 0.0, 0.0)]
    steps = 14
    for i in range(steps + 1):
        angle = 0.22 + i * (math.tau - 0.44) / steps
        verts.append((math.cos(angle) * 0.5, math.sin(angle) * 0.5, 0.0))
    faces = [(0, i, i + 1) for i in range(1, steps + 1)]
    return np.array(verts, dtype=np.float32), faces, [0] * len(faces)


def _water_lily():
    """A water lily: two rings of upturned petals, one unit across."""
    verts, faces, kinds = [], [], []
    for ring, (reach, lift, count) in enumerate(((0.5, 0.18, 8), (0.32, 0.3, 6))):
        for i in range(count):
            angle = i * math.tau / count + ring * 0.4
            half = math.pi / count * 0.8
            base = len(verts)
            verts.extend((
                (0.0, 0.0, 0.02),
                (math.cos(angle - half) * reach * 0.6, math.sin(angle - half) * reach * 0.6, lift * 0.5),
                (math.cos(angle) * reach, math.sin(angle) * reach, lift),
                (math.cos(angle + half) * reach * 0.6, math.sin(angle + half) * reach * 0.6, lift * 0.5),
            ))
            faces.append((base, base + 1, base + 2, base + 3))
            kinds.append(0)
    return np.array(verts, dtype=np.float32), faces, kinds


def lily_meshes(pads, blooms, seed):
    """Lily pads at ``pads`` and water lilies at ``blooms`` ((x, y, z) rows), as two meshes."""
    pad_mesh = _merged("Lily Pads", _pad(), pads, seed, (0.28, 0.5))
    pad_mesh.materials.append(materials.plain("Lily Pad", (0.04, 0.17, 0.04), 0.35))
    bloom_mesh = _merged("Water Lilies", _water_lily(), blooms, seed + 1, (0.16, 0.24))
    bloom_mesh.materials.append(materials.plain("Water Lily", (0.92, 0.55, 0.68), 0.5))
    return pad_mesh, bloom_mesh


def add_grass(ground, keep, per_square_metre, length, color, seed):
    """
    Grow grass on a terrain as render-time strands. ``keep`` is a per-vertex 0..1 weight
    of where it grows (none in the pond). Strands are not geometry: they render in
    Blender but do not export.
    """
    group = ground.vertex_groups.new(name="Grass")
    for weight in np.unique(np.round(keep, 1)):
        if weight > 0.0:
            group.add(np.flatnonzero(np.round(keep, 1) == weight).tolist(), float(weight), 'REPLACE')
    ground.data.materials.append(materials.grass(color))

    modifier = ground.modifiers.new("Grass", 'PARTICLE_SYSTEM')
    settings = modifier.particle_system.settings
    settings.name = "Scenery Grass"
    settings.type = 'HAIR'
    size = ground.elyan_terrain.size
    children = 12
    settings.count = int(min(per_square_metre * size * size / children, 60000))
    settings.hair_length = length
    settings.use_advanced_hair = True
    settings.brownian_factor = length * 0.25
    settings.child_type = 'INTERPOLATED'
    settings.rendered_child_count = children
    # A sparse preview keeps the viewport responsive; the render grows the rest.
    settings.child_percent = 1
    settings.display_percentage = 20
    settings.roughness_endpoint = length * 0.4
    settings.root_radius = 0.6
    settings.tip_radius = 0.0
    settings.radius_scale = 0.01
    settings.material = len(ground.data.materials)
    modifier.particle_system.seed = seed
    modifier.particle_system.vertex_group_density = group.name
    return modifier
