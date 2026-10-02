# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Instant Scenery: one operator that builds terrain, water, rocks, plants, sky
and a camera, with choices that suit each other.
"""

import math
import random

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
from bpy.types import Operator
from mathutils import Vector

from . import heightfield, materials, objects, scatter, sky, terrain

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

# What grows on each kind of ground: (kind, rules). Altitudes are fractions of the terrain's
# relief and match the colour bands of the ground material, so firs stop below the alpine snow
# line. "shore" instead keeps a plant within that much relief of the waterline (or the lowest
# ground), wherever that is. Bare ground (lava, moon dust) gets rocks only.
# Plants at their natural height look like toys against a 300 m mountain range.
_PLANT_SCALE = 0.6

_PLANTS = {
    'ALPINE': (
        ('CONIFER', {"density": 160.0, "altitude": (0.0, 0.60), "max_slope": 40.0}),
        ('BROADLEAF', {"density": 40.0, "altitude": (0.0, 0.30)}),
        ('BUSH', {"density": 50.0, "altitude": (0.0, 0.50)}),
    ),
    'TROPICAL': (
        ('PALM', {"density": 120.0, "shore": 0.12}),
        ('BROADLEAF', {"density": 110.0, "altitude": (0.08, 0.80), "max_slope": 36.0}),
        ('BUSH', {"density": 60.0, "altitude": (0.0, 0.80)}),
    ),
    'DESERT': (
        ('DEAD', {"density": 5.0, "altitude": (0.0, 0.60)}),
        ('BUSH', {"density": 25.0, "altitude": (0.0, 0.50), "clumping": 0.8}),
    ),
    'ARCTIC': (
        ('CONIFER', {"density": 40.0, "altitude": (0.0, 0.25), "clumping": 0.85}),
    ),
    'VOLCANIC': (),
    'MOON': (),
    'ALIEN': (),
}


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
            # The first steps are short: a hump a few metres ahead hides more than a ridge far off.
            sideways = Vector((peak.y - y, x - peak.x, 0.0))
            clearance = min(
                eye.lerp(aim, t).z - surface(*eye.lerp(aim, t).xy)
                for aim, reach in (
                    (peak, 16), (peak + sideways * 0.35, 10), (peak - sideways * 0.35, 10),
                    (peak + sideways * 0.65, 6), (peak - sideways * 0.65, 6),
                )
                for t in (0.01, 0.02, 0.035, *(i / 20.0 for i in range(1, reach + 1)))
            )
            # Usually low ground looking up is the better picture; a little chance keeps seeds varied.
            perch = eye.z / height * (0.25 if high else -0.25)
            score = min(clearance, height * 0.12) / height + perch + rng.uniform(0.0, 0.04)
            if score > best_score:
                best, best_score = eye, score

    data = bpy.data.cameras.new("Scenery Camera")
    data.lens = 28.0
    data.clip_end = max(size * 60.0, 1000.0)
    camera = objects.link(context, bpy.data.objects.new("Scenery Camera", data))
    camera.location = ground.matrix_world @ best
    aim = ground.matrix_world @ peak - camera.location
    camera.rotation_euler = aim.to_track_quat('-Z', 'Y').to_euler()
    context.scene.camera = camera
    return camera, aim


class ELYAN_OT_instant(Operator):
    """Build a whole landscape: terrain, water, rocks, plants, sky and a camera looking at it"""
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
    ground: EnumProperty(
        name="Ground", items=(('AUTO', "Suit the Landscape", "Chosen by the seed"),) + materials.TERRAIN_ITEMS,
    )
    sky: EnumProperty(
        name="Sky", items=(('AUTO', "Suit the Landscape", "Chosen by the seed"),) + sky.SKY_ITEMS,
    )
    size: FloatProperty(name="Size", default=300.0, min=10.0, unit='LENGTH')
    resolution: IntProperty(name="Resolution", default=385, min=65, max=2049)
    erosion: FloatProperty(name="Erosion", default=0.3, min=0.0, max=1.0, subtype='FACTOR')
    use_water: BoolProperty(name="Water", description="Flood the lowlands where it suits the landscape", default=True)
    rocks: IntProperty(name="Rocks", default=12, min=0, max=200)
    plants: FloatProperty(
        name="Plants", description="How thickly trees and bushes grow, where the ground suits any",
        default=1.0, min=0.0, soft_max=2.0, max=5.0,
    )

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
        # The seed's choice is drawn even when overridden, so the rest of the scene stays the same.
        ground_material = rng.choice(grounds)
        if self.ground != 'AUTO':
            ground_material = self.ground
        ground.data.materials.append(materials.terrain(ground_material))

        low, high = float(grid.min()) * height, float(grid.max()) * height
        water = None
        if self.use_water and kind in _WET:
            water = height * (0.06 if kind == 'ISLAND' else rng.uniform(0.10, 0.20))
            # Wide enough that its far edge is lost in the haze instead of drawing a line under the sky.
            objects.add_water(context, water, size * 40.0)
        else:
            # Dry land carries on to the horizon; without it the sky below the horizon shows, and that is black.
            plain = objects.add_water(context, low - 0.01 * height, size * 40.0)
            plain.name = plain.data.name = "Ground"
            plain.data.materials[0] = materials.ground(ground_material)

        sky_key = rng.choice(skies)
        preset = sky.SKY_PRESETS[sky_key if self.sky == 'AUTO' else self.sky]
        camera, aim = _frame_camera(
            context, ground, -1.0 if water is None else water, rng,
            kind in {'ISLAND', 'VOLCANO'}, kind in {'CANYON', 'PLATEAU'},
        )

        # Everything is sized for a 300 m terrain and scales with it.
        factor = size / 300.0
        eye = ground.matrix_world.inverted() @ camera.location
        count = 0
        if self.rocks:
            # Many more tries than rocks wanted, so that the count is met even on a flooded terrain.
            rocks = scatter.scatter(
                context, ground, 'ROCK', self.seed, density=self.rocks * 8.0 / (size * size / 10000.0),
                scale=(0.6, 3.0), size_factor=factor, variants=4, limit=self.rocks, water=water,
                avoid=((eye.x, eye.y, 4.0 * factor),),
            )
            count += len(rocks.objects) if rocks else 0
        shore = (max(low, water or low) - low) / max(high - low, 1e-9)
        # Nothing right in front of the lens, and less and less of a gap further along the view.
        ahead = Vector((aim.x, aim.y)).normalized()
        clear = tuple(
            (eye.x + ahead.x * reach * factor, eye.y + ahead.y * reach * factor, radius * factor)
            for reach, radius in ((0.0, 14.0), (22.0, 12.0), (44.0, 9.0))
        )
        for index, (what, rules) in enumerate(_PLANTS[ground_material] if self.plants > 0.0 else ()):
            rules = dict(rules)
            if "shore" in rules:
                rules["altitude"] = (0.0, shore + rules.pop("shore"))
            # Densities are per hectare of a 300 m terrain: a bigger terrain has bigger trees, not more of them.
            rules["density"] *= self.plants / (factor * factor)
            plants = scatter.scatter(
                context, ground, what, self.seed + index + 1, size_factor=factor * _PLANT_SCALE, triangles=600,
                water=water, avoid=clear, **rules,
            )
            count += len(plants.objects) if plants else 0

        # Light from behind and beside the camera reads better than a fixed compass direction.
        facing = math.degrees(math.atan2(aim.x, aim.y))
        sky.build(
            context.scene, preset["elevation"], facing + rng.uniform(100.0, 150.0) * rng.choice((-1, 1)),
            preset["haze"], preset["clouds"], preset.get("stars", 0.0),
            preset.get("mist", 0.0), preset.get("mist_color", (1.0, 1.0, 1.0)),
        )
        for other in context.selected_objects:
            other.select_set(False)
        ground.select_set(True)
        context.view_layer.objects.active = ground
        self.report({'INFO'}, "{:s}, seed {:d}, {:d} rocks and plants".format(kind.title(), self.seed, count))
        return {'FINISHED'}


classes = (
    ELYAN_OT_instant,
)
