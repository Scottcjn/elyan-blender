# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Places: a patch of ground a few tens of metres across, seen from standing height.

Where a vista is a mountain range from a distance, a place is a meadow with a
pond, a clearing in a wood, an oasis. It is built from gently rolling ground,
an optional pond with its stones and lilies, grass, drifts of flowers, a few
trees, and a far treeline so the ground runs out to the horizon.

Heights in the scene: the ground's border sits at world Z 0.
"""

import math
import random

import bpy
import numpy as np
from mathutils import Vector

from . import heightfield, materials, objects, plants, scatter, sky, terrain

# The bank stays clear of flowers and grass this far beyond the water's edge, in metres.
_BANK = 0.5
# How far the water lies below the level ground around the pond.
_WATER_BELOW = 0.08


def _ground(context, spec, seed, rng):
    """The terrain, and what was dug into it. Returns (object, grid, pond or None)."""
    size, n = spec["size"], spec.get("resolution", 257)
    relief = spec["relief"]
    height = heightfield.rolls(n, seed) * relief
    base = relief * 0.5
    # The border is level, so the far ground can meet it without a step.
    axis = np.linspace(0.0, 1.0, n, dtype=np.float32)
    edge = np.minimum(axis, 1.0 - axis)
    ramp = np.clip(edge / 0.12, 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    height = base + (height - base) * ramp[:, None] * ramp[None, :]

    pond = None
    if spec.get("pond"):
        rx, ry = (r * rng.uniform(0.85, 1.15) for r in spec["pond"]["radii"])
        cx, cy = rng.uniform(-0.08, 0.12) * size, rng.uniform(-0.05, 0.10) * size
        shore = spec["pond"].get("shore", 0.9)
        depth = spec["pond"]["depth"]
        # The radii asked for are those of the water's edge. The hollow's own rim lies part-way
        # down the bank, so find how far out the bank rises to the waterline and dig that much smaller.
        bank = 1.0 + shore / min(rx, ry)
        reach = np.linspace(0.55, bank, 400)
        t = (bank - reach) / (bank - 0.55)
        waterline = float(reach[np.argmax(depth * t * t * (3.0 - 2.0 * t) < _WATER_BELOW)])
        hollow = heightfield.basin(
            n, (cx / size + 0.5, cy / size + 0.5), (rx / waterline / size, ry / waterline / size),
            shore / waterline / size)
        # Level the ground the pond sits in, well past the bank, then dig.
        apron = heightfield.basin(
            n, (cx / size + 0.5, cy / size + 0.5), (rx / size, ry / size), (shore + 0.3 * max(rx, ry)) / size)
        height = height + (base - height) * np.clip(apron * 4.0, 0.0, 1.0)
        height = height - depth * hollow
        pond = {"center": (cx, cy), "radii": (rx, ry), "shore": shore, "level": base - _WATER_BELOW}

    low = float(height.min())
    span = float(height.max()) - low
    grid = ((height - low) / span).astype(np.float32)
    ground = terrain.create(context, "Ground", grid, size, span)
    ground.location.z = low - base

    lines = None
    if pond:
        pond["local_level"] = pond["level"] - low
        # The bare, wet band reaches a little above the water; the meadow itself is hardly higher.
        wet = (pond["local_level"] + 0.045) / span
        lines = (wet, wet + (1.0 - wet) * 0.5, wet + (1.0 - wet) * 0.9)
    ground.data.materials.append(materials.terrain(spec["ground"], lines))
    return ground, grid, pond


def _in_pond(pond, x, y, margin=0.0):
    """Whether local (x, y) points lie within the pond's rim widened by ``margin`` metres."""
    (cx, cy), (rx, ry) = pond["center"], pond["radii"]
    return ((x - cx) / (rx + margin)) ** 2 + ((y - cy) / (ry + margin)) ** 2 < 1.0


def _dress_pond(context, ground, grid, pond, spec, seed, rng):
    settings = ground.elyan_terrain
    size, span = settings.size, settings.height_applied
    (cx, cy), (rx, ry) = pond["center"], pond["radii"]
    # An oval reaching just under the bank. A square sheet would show wherever
    # the ground beyond the pond dips below the water level.
    tuck = 0.35
    ring = [(math.cos(i * math.tau / 48) * (rx + tuck), math.sin(i * math.tau / 48) * (ry + tuck), 0.0)
            for i in range(48)]
    mesh = bpy.data.meshes.new("Water")
    mesh.from_pydata(ring, [], [tuple(range(48))])
    mesh.materials.append(materials.water())
    water = objects.link(context, bpy.data.objects.new("Water", mesh))
    water.parent = ground
    water.location = (cx, cy, pond["local_level"])
    counts = {}

    stones = spec["pond"].get("stones", 24)
    if stones:
        places = []
        for _ in range(stones):
            angle, out = rng.uniform(0.0, math.tau), rng.uniform(0.99, 1.10)
            x, y = cx + math.cos(angle) * rx * out, cy + math.sin(angle) * ry * out
            places.append((x, y, heightfield.sample(grid, x / size + 0.5, y / size + 0.5) * span))
        meshes = [objects.rock_mesh(seed * 100 + index) for index in range(4)]
        scatter.instance(context, ground, meshes, places, seed, (0.09, 0.22), 0.25, 0.25, "Pond Stones")
        counts["stones"] = stones

    pads = spec["pond"].get("lilies", 0)
    if pads:
        def afloat(count):
            rows = []
            while len(rows) < count:
                angle, out = rng.uniform(0.0, math.tau), math.sqrt(rng.uniform(0.02, 0.5))
                rows.append((cx + math.cos(angle) * rx * out, cy + math.sin(angle) * ry * out,
                             pond["local_level"] + 0.006))
            return rows
        pad_places = afloat(pads)
        bloom_places = [(x, y, z + 0.004) for x, y, z in pad_places[:max(1, pads // 3)]]
        for mesh in plants.lily_meshes(pad_places, bloom_places, seed):
            ob = objects.link(context, bpy.data.objects.new(mesh.name, mesh))
            ob.parent = ground
        counts["lily_pads"], counts["water_lilies"] = pads, len(bloom_places)
    return counts


def _keep_dry(pond, places):
    """Drop places that are in the pond or on its bank."""
    if pond is None or not len(places):
        return places
    return places[~_in_pond(pond, places[:, 0], places[:, 1], _BANK)]


def _camera(context, ground, grid, pond, rng):
    """Stand near one side at eye height and look across at the pond, or the middle."""
    settings = ground.elyan_terrain
    size, span = settings.size, settings.height_applied
    focus = Vector((*pond["center"], pond["local_level"] + 0.3)) if pond else Vector((0.0, 0.0, span * 0.5))
    # From the south-west to south-east: the far treeline is then behind the subject.
    angle = math.radians(rng.uniform(-150.0, -60.0))
    distance = size * 0.36
    x = min(max(focus.x + math.cos(angle) * distance, -size * 0.45), size * 0.45)
    y = min(max(focus.y + math.sin(angle) * distance, -size * 0.45), size * 0.45)
    eye = Vector((x, y, heightfield.sample(grid, x / size + 0.5, y / size + 0.5) * span + 1.6))

    data = bpy.data.cameras.new("Scenery Camera")
    data.lens = 32.0
    data.clip_end = max(size * 60.0, 2000.0)
    camera = objects.link(context, bpy.data.objects.new("Scenery Camera", data))
    camera.location = ground.matrix_world @ eye
    aim = ground.matrix_world @ focus - camera.location
    camera.rotation_euler = aim.to_track_quat('-Z', 'Y').to_euler()
    context.scene.camera = camera
    return eye, focus, aim


def _backdrop(context, ground, spec, seed, rng):
    """Far ground with a hole where the place sits, and a loose ring of distant trees."""
    size = ground.elyan_terrain.size
    inner, outer = size * 0.5 - 0.02, size * 25.0
    verts = [(sx * r, sy * r, 0.0) for r in (inner, outer) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    faces = [(i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i) for i in range(4)]
    mesh = bpy.data.meshes.new("Far Ground")
    mesh.from_pydata(verts, [], faces)
    colors = materials.TERRAIN_PRESETS[spec["ground"]]["colors"]
    mesh.materials.append(materials.plain("Far " + materials.TERRAIN_PRESETS[spec["ground"]]["label"], colors[2], 0.9))
    far = objects.link(context, bpy.data.objects.new("Far Ground", mesh))
    # A whisker below the border, so the two never fight over the same pixels.
    far.location.z = -0.01

    count = 0
    level = -ground.location.z
    for index, (species, number) in enumerate(spec.get("treeline", ())):
        places = []
        for _ in range(number):
            angle, radius = rng.uniform(0.0, math.tau), rng.uniform(1.3, 3.2) * size
            places.append((math.cos(angle) * radius, math.sin(angle) * radius, level))
        meshes = scatter.make_meshes(species, seed + 50 + index, 3, 400)
        scatter.instance(context, ground, meshes, places, seed + index, (0.8, 1.4), 0.03, 0.0, "Treeline " + species.title())
        count += number
    return count


def build(context, spec, seed=1, sky_key=None, plant_factor=1.0):
    """
    Build the place ``spec`` describes (see ``world.ENVIRONMENTS``). Returns a report.

    ``sky_key`` overrides the sky the seed would pick; ``plant_factor`` scales how
    many trees, bushes and flowers grow.
    """
    rng = random.Random(seed)
    ground, grid, pond = _ground(context, spec, seed, rng)
    settings = ground.elyan_terrain
    size, span = settings.size, settings.height_applied
    report = {"ground": ground.name, "size": size, "pond": None, "counts": {}}
    counts = report["counts"]

    if pond:
        counts.update(_dress_pond(context, ground, grid, pond, spec, seed, rng))
        report["pond"] = {"center": pond["center"], "radii": pond["radii"], "water_level": pond["level"]}
    water = pond["local_level"] if pond else None

    eye, focus, aim = _camera(context, ground, grid, pond, rng)
    # Nothing tall right in front of the lens, nor between it and the subject.
    toward = Vector((focus.x - eye.x, focus.y - eye.y)).normalized()
    clear = [(eye.x + toward.x * step, eye.y + toward.y * step, radius)
             for step, radius in ((0.0, 4.5), (4.0, 3.0), (7.5, 2.0))]
    if pond:
        clear.append((*pond["center"], max(pond["radii"]) + pond["shore"] + 0.6))

    for index, (species, number, tall) in enumerate(spec.get("trees", ())):
        number = int(round(number * plant_factor))
        if number <= 0:
            continue
        made = scatter.scatter(
            context, ground, species, seed + index + 1, density=number * 400.0 / (size * size / 10000.0),
            altitude=(0.0, 1.0), max_slope=40.0, water=water, clearance=0.0,
            spacing=2.0 if species == 'BUSH' else 3.5, clumping=0.2, scale=tall, variants=3,
            triangles=spec.get("tree_triangles", 3000), limit=number, avoid=clear,
        )
        counts[species.lower()] = len(made.objects) if made else 0

    rocks = spec.get("rocks", 0)
    if rocks:
        made = scatter.scatter(
            context, ground, 'ROCK', seed + 30, density=rocks * 200.0 / (size * size / 10000.0), altitude=(0.0, 1.0),
            water=water, spacing=1.0, clumping=0.4, scale=(0.08, 0.35), limit=rocks, avoid=clear[:1],
        )
        counts["rocks"] = len(made.objects) if made else 0

    for index, (species, number) in enumerate(spec.get("flowers", ())):
        number = int(round(number * plant_factor))
        if number <= 0:
            continue
        # Flowers grow in drifts, not evenly: a strongly clumped draw, then keep them off the bank.
        places = scatter.find_places(
            grid, size, span, seed + 70 + index, density=number * 300.0 / (size * size / 10000.0),
            water=water, clearance=0.02, clumping=0.92, limit=number * 2, margin=0.06, avoid=clear[:1],
        )
        places = _keep_dry(pond, places)[:number]
        if len(places):
            mesh = plants.flower_mesh(species, places, seed + index)
            ob = objects.link(context, bpy.data.objects.new(mesh.name, mesh))
            ob.parent = ground
            counts[species.lower()] = len(places)

    grass = spec.get("grass")
    if grass:
        co = terrain._read_co(ground.data)
        keep = np.ones(len(co), dtype=np.float32)
        if pond:
            keep[_in_pond(pond, co[:, 0], co[:, 1], _BANK * 0.6)] = 0.0
        plants.add_grass(ground, keep, grass["density"], grass["length"], grass["color"], seed)
        counts["grass_strands"] = ground.modifiers["Grass"].particle_system.settings.count * 12

    counts["treeline"] = _backdrop(context, ground, spec, seed, rng)

    preset = sky.SKY_PRESETS[sky_key or rng.choice(spec["skies"])]
    facing = math.degrees(math.atan2(aim.x, aim.y))
    sky.build(
        context.scene, preset["elevation"], facing + rng.uniform(100.0, 150.0) * rng.choice((-1, 1)),
        preset["haze"], preset["clouds"], preset.get("stars", 0.0),
        preset.get("mist", 0.0) * spec.get("mist", 0.12), preset.get("mist_color", (1.0, 1.0, 1.0)),
    )
    for other in context.selected_objects:
        other.select_set(False)
    ground.select_set(True)
    context.view_layer.objects.active = ground
    report["camera"] = context.scene.camera.name
    return report
