# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Scenery objects: water, rocks, organic blobs, and the one-click landscape.
"""

import math
import random

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
from bpy.types import Operator
from mathutils import Vector, noise

from . import heightfield, materials, sky, terrain

# Which ground and sky suit each landscape when the choice is left to chance.
_PAIRINGS = {
    'MOUNTAINS': (('ALPINE', 'ARCTIC'), ('NOON', 'FAIR', 'GOLDEN')),
    'HILLS': (('ALPINE', 'TROPICAL'), ('FAIR', 'GOLDEN', 'HAZE')),
    'DUNES': (('DESERT',), ('NOON', 'SUNSET', 'GOLDEN')),
    'CANYON': (('DESERT',), ('NOON', 'GOLDEN', 'SUNSET')),
    'ISLAND': (('TROPICAL', 'ALPINE'), ('NOON', 'FAIR', 'SUNSET')),
    'VOLCANO': (('VOLCANIC',), ('SUNSET', 'HAZE', 'OVERCAST')),
    'PLATEAU': (('DESERT', 'ALPINE'), ('FAIR', 'GOLDEN')),
    'CRATERS': (('MOON', 'ALIEN'), ('NIGHT', 'NOON')),
}
# Landscapes that look right with a sea around or between them.
_WET = {'ISLAND', 'MOUNTAINS', 'HILLS', 'VOLCANO'}


def _link(context, ob):
    context.collection.objects.link(ob)
    return ob


def add_water(context, level, size):
    mesh = bpy.data.meshes.new("Water")
    half = size / 2.0
    mesh.from_pydata(
        [(-half, -half, 0.0), (half, -half, 0.0), (half, half, 0.0), (-half, half, 0.0)], [], [(0, 1, 2, 3)],
    )
    mesh.materials.append(materials.water())
    ob = _link(context, bpy.data.objects.new("Water", mesh))
    ob.location.z = level
    return ob


def add_rock(context, seed, size):
    """A boulder: a sphere pushed around by noise, squashed and flat-bottomed."""
    rng = random.Random(seed)
    bm_mesh = bpy.data.meshes.new("Rock")
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
    offset = Vector((rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(-100, 100)))
    squash = Vector((rng.uniform(0.7, 1.3), rng.uniform(0.7, 1.3), rng.uniform(0.45, 0.85)))
    for vert in bm.verts:
        direction = vert.co.normalized()
        lumps = noise.fractal(direction * 1.1 + offset, 1.0, 2.0, 4)
        facets = noise.cell(direction * 2.3 + offset)
        vert.co = direction * (1.0 + 0.28 * lumps + 0.10 * facets)
        vert.co *= squash
        vert.co.z = max(vert.co.z, -0.35 * squash.z)
    bm.to_mesh(bm_mesh)
    bm.free()
    bm_mesh.materials.append(materials.rock())
    ob = _link(context, bpy.data.objects.new("Rock", bm_mesh))
    ob.scale = (size, size, size)
    ob.rotation_euler.z = rng.uniform(0.0, math.tau)
    return ob


def add_blob(context, seed, count, size):
    """Metaballs strung along a wandering path melt into one organic form."""
    rng = random.Random(seed)
    data = bpy.data.metaballs.new("Blob")
    data.resolution = size * 0.12
    data.render_resolution = size * 0.06
    position = Vector((0.0, 0.0, 0.0))
    heading = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.2, 1))).normalized()
    for _ in range(count):
        element = data.elements.new()
        element.co = position
        element.radius = size * rng.uniform(0.55, 1.0)
        turn = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.6, 1.0)))
        heading = (heading + turn * 0.7).normalized()
        position = position + heading * element.radius * 0.75
    data.materials.append(materials.clay())
    return _link(context, bpy.data.objects.new("Blob", data))


def _frame_camera(context, ground, water_level, rng, outside, high):
    """
    Find a place to stand with a clear view of the highest peak, and look at it.

    ``outside`` stands back beyond the terrain's edge, for islands and lone mountains.
    ``high`` prefers a perch over a valley floor, for land that is mostly cliffs.
    """
    settings = ground.elyan_terrain
    size, height = settings.size, settings.height_applied
    grid = terrain.read_grid(ground)
    n = grid.shape[0]

    def surface(x, y):
        u, v = x / size + 0.5, y / size + 0.5
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return max(0.0, water_level)
        return max(heightfield.sample(grid, u, v) * height, water_level)

    row, column = divmod(int(grid.argmax()), n)
    peak = Vector(((column / (n - 1) - 0.5) * size, (row / (n - 1) - 0.5) * size, float(grid.max()) * height * 0.6))

    best, best_score = None, -1e9
    start = rng.uniform(0.0, math.tau)
    for step in range(24):
        angle = start + step * math.tau / 24.0
        for radius in ((0.62, 0.78) if outside else (0.30, 0.40, 0.48)):
            x, y = math.cos(angle) * size * radius, math.sin(angle) * size * radius
            if math.hypot(peak.x - x, peak.y - y) < size * 0.3:
                continue
            eye = Vector((x, y, surface(x, y) + height * 0.06 + 1.7))
            # How far the lines of sight stay above the ground: straight to the peak, and
            # to either side of it so a nearby wall does not fill half the picture.
            sideways = Vector((peak.y - y, x - peak.x, 0.0)) * 0.4
            clearance = min(
                eye.lerp(aim, t).z - surface(*eye.lerp(aim, t).xy)
                for aim, reach in ((peak, 16), (peak + sideways, 10), (peak - sideways, 10))
                for t in (i / 20.0 for i in range(1, reach + 1))
            )
            # Usually low ground looking up is the better picture; a little chance keeps seeds varied.
            perch = eye.z / height * (0.25 if high else -0.25)
            score = min(clearance, height * 0.12) / height + perch + rng.uniform(0.0, 0.04)
            if score > best_score:
                best, best_score = eye, score

    data = bpy.data.cameras.new("Scenery Camera")
    data.lens = 28.0
    data.clip_end = max(size * 20.0, 1000.0)
    camera = _link(context, bpy.data.objects.new("Scenery Camera", data))
    camera.location = ground.matrix_world @ best
    aim = ground.matrix_world @ peak - camera.location
    camera.rotation_euler = aim.to_track_quat('-Z', 'Y').to_euler()
    context.scene.camera = camera
    return camera, aim


class ELYAN_OT_water_add(Operator):
    """Add a sheet of water"""
    bl_idname = "elyan_scenery.water_add"
    bl_label = "Water"
    bl_options = {'REGISTER', 'UNDO'}

    level: FloatProperty(name="Level", default=0.0, unit='LENGTH')
    size: FloatProperty(name="Size", default=2000.0, min=1.0, unit='LENGTH')

    def execute(self, context):
        add_water(context, self.level, self.size)
        return {'FINISHED'}


class ELYAN_OT_rock_add(Operator):
    """Add a boulder at the 3D cursor"""
    bl_idname = "elyan_scenery.rock_add"
    bl_label = "Rock"
    bl_options = {'REGISTER', 'UNDO'}

    seed: IntProperty(name="Seed", default=1, min=0)
    size: FloatProperty(name="Size", default=2.0, min=0.01, unit='LENGTH')

    def execute(self, context):
        add_rock(context, self.seed, self.size).location = context.scene.cursor.location
        return {'FINISHED'}


class ELYAN_OT_blob_add(Operator):
    """Add an organic form made of melted-together spheres, at the 3D cursor"""
    bl_idname = "elyan_scenery.blob_add"
    bl_label = "Organic Blob"
    bl_options = {'REGISTER', 'UNDO'}

    seed: IntProperty(name="Seed", default=1, min=0)
    count: IntProperty(name="Parts", default=7, min=1, max=64)
    size: FloatProperty(name="Size", default=1.0, min=0.01, unit='LENGTH')

    def execute(self, context):
        add_blob(context, self.seed, self.count, self.size).location = context.scene.cursor.location
        return {'FINISHED'}


class ELYAN_OT_instant(Operator):
    """Build a whole landscape: terrain, water, rocks, sky and a camera looking at it"""
    bl_idname = "elyan_scenery.instant"
    bl_label = "Instant Scenery"
    bl_options = {'REGISTER', 'UNDO'}

    landscape: EnumProperty(
        name="Landscape",
        items=(('RANDOM', "Surprise Me", "Chosen by the seed"),) + tuple(
            item for item in terrain.LANDSCAPE_ITEMS if item[0] in _PAIRINGS
        ),
    )
    seed: IntProperty(name="Seed", description="Same seed, same scenery", default=1, min=0)
    size: FloatProperty(name="Size", default=300.0, min=10.0, unit='LENGTH')
    resolution: IntProperty(name="Resolution", default=385, min=65, max=2049)
    erosion: FloatProperty(name="Erosion", default=0.3, min=0.0, max=1.0, subtype='FACTOR')
    use_water: BoolProperty(name="Water", description="Flood the lowlands where it suits the landscape", default=True)
    rocks: IntProperty(name="Rocks", default=12, min=0, max=200)

    def execute(self, context):
        rng = random.Random(self.seed)
        kind = self.landscape if self.landscape != 'RANDOM' else rng.choice(sorted(_PAIRINGS))
        grounds, skies = _PAIRINGS[kind]
        size = self.size
        height = size * rng.uniform(0.14, 0.22) * (0.5 if kind in {'DUNES', 'HILLS', 'CRATERS'} else 1.0)

        grid = heightfield.generate(kind, self.resolution, self.seed)
        if self.erosion > 0.0:
            grid = heightfield.hydraulic_erosion(grid, int(20 + 120 * self.erosion), self.seed)
            grid = heightfield.thermal_erosion(grid, int(4 + 20 * self.erosion))
        ground = terrain.create(context, "Terrain", grid.astype("float32"), size, height)
        ground.data.materials.append(materials.terrain(rng.choice(grounds)))

        water_level = -1.0
        if self.use_water and kind in _WET:
            water_level = height * (0.06 if kind == 'ISLAND' else rng.uniform(0.10, 0.20))
            add_water(context, water_level, size * 12.0)

        for index in range(self.rocks):
            # Rocks gather on dry ground, try a few spots for each.
            for _attempt in range(8):
                u, v = rng.uniform(0.08, 0.92), rng.uniform(0.08, 0.92)
                z = heightfield.sample(grid, u, v) * height
                if z > water_level + 0.2:
                    break
            scale = size * 0.004 * rng.uniform(0.5, 2.5)
            rock = add_rock(context, self.seed * 1000 + index, scale)
            rock.location = ground.matrix_world @ Vector(((u - 0.5) * size, (v - 0.5) * size, z))

        preset = sky.SKY_PRESETS[rng.choice(skies)]
        # Light from behind and beside the camera reads better than a fixed compass direction.
        _camera, aim = _frame_camera(
            context, ground, water_level, rng, kind in {'ISLAND', 'VOLCANO'}, kind in {'CANYON', 'PLATEAU'},
        )
        facing = math.degrees(math.atan2(aim.x, aim.y))
        sky.build(
            context.scene, preset["elevation"], facing + rng.uniform(100.0, 150.0) * rng.choice((-1, 1)),
            preset["haze"], preset["clouds"], preset.get("stars", 0.0),
        )
        self.report({'INFO'}, "{:s}, seed {:d}".format(kind.title(), self.seed))
        return {'FINISHED'}


classes = (
    ELYAN_OT_water_add,
    ELYAN_OT_rock_add,
    ELYAN_OT_blob_add,
    ELYAN_OT_instant,
)
