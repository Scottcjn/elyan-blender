# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Scatter: strew rocks and plants over a terrain by rule.

Places are chosen from the terrain's height field, so nothing is ray cast and a
few hundred take milliseconds. What is placed are linked duplicates: every
object shares one of a handful of meshes, so memory and render time stay small.
"""

import math

import bpy
import numpy as np
from bpy.props import EnumProperty, FloatProperty, IntProperty
from bpy.types import Operator

from . import heightfield, objects, terrain, trees

WHAT_ITEMS = (
    ('ROCK', "Rocks", "Boulders"),
    *trees.SPECIES_ITEMS,
    ('SELECTED', "Selected Objects", "Copies of the other selected mesh objects"),
)

# Rules that suit each kind of thing; choosing a kind in the operator loads them.
# density is per hectare; altitude is a fraction of the terrain's relief; slope in degrees.
DEFAULTS = {
    'ROCK': {"density": 6.0, "altitude": (0.0, 1.0), "slope": 40.0, "spacing": 2.0, "clumping": 0.3, "scale": (0.4, 2.2)},
    'CONIFER': {"density": 40.0, "altitude": (0.08, 0.62), "slope": 34.0, "spacing": 4.0, "clumping": 0.6, "scale": (0.6, 1.25)},
    'BROADLEAF': {"density": 20.0, "altitude": (0.04, 0.40), "slope": 26.0, "spacing": 6.0, "clumping": 0.7, "scale": (0.7, 1.3)},
    'PALM': {"density": 14.0, "altitude": (0.0, 0.25), "slope": 20.0, "spacing": 4.0, "clumping": 0.5, "scale": (0.7, 1.3)},
    'DEAD': {"density": 3.0, "altitude": (0.0, 0.7), "slope": 28.0, "spacing": 8.0, "clumping": 0.3, "scale": (0.6, 1.3)},
    'BUSH': {"density": 30.0, "altitude": (0.03, 0.60), "slope": 32.0, "spacing": 2.0, "clumping": 0.6, "scale": (0.6, 1.5)},
    'SELECTED': {"density": 10.0, "altitude": (0.0, 1.0), "slope": 30.0, "spacing": 3.0, "clumping": 0.0, "scale": (0.8, 1.2)},
}


def find_places(
        grid, size, height, seed, density, altitude=(0.0, 1.0), max_slope=90.0, water=None, clearance=0.0,
        spacing=0.0, clumping=0.0, limit=500, margin=0.03, avoid=(),
):
    """
    Choose places on a height field. Returns an array of local ``(x, y, z)`` rows.

    ``density`` is tried places per hectare: as many as that land on ground that
    passes the rules, fewer where it does not, so suitable ground is evenly
    covered whatever share of the terrain it is. ``altitude`` is a (low, high)
    fraction between the lowest and highest point, the same measure the terrain
    materials use for their snow and tree lines. ``water`` is the local height
    of the water surface, or None; places lie at least ``clearance`` above it.
    ``avoid`` holds ``(x, y, radius)`` circles to keep clear, such as the camera.
    """
    rng = np.random.default_rng(seed)
    tries = int(min(density * size * size / 10000.0, limit * 40))
    if tries <= 0 or limit <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    u = rng.uniform(margin, 1.0 - margin, tries)
    v = rng.uniform(margin, 1.0 - margin, tries)
    level = heightfield.sample_many(grid, u, v)
    low, high = float(grid.min()), float(grid.max())
    relief = (level - low) / max(high - low, 1e-9)
    keep = (relief >= altitude[0]) & (relief <= altitude[1])

    n = grid.shape[0]
    steep = heightfield.slope(grid * height, size / (n - 1))
    row = np.rint(v * (n - 1)).astype(np.int32)
    column = np.rint(u * (n - 1)).astype(np.int32)
    keep &= steep[row, column] <= math.tan(math.radians(min(max_slope, 89.9)))

    if water is not None:
        keep &= level * height >= water + clearance

    if clumping > 0.0:
        # Plants gather where a broad noise field is high and leave the rest bare.
        m = 65
        groves = heightfield.value_noise(m, 5, rng)
        chance = groves[np.rint(v * (m - 1)).astype(np.int32), np.rint(u * (m - 1)).astype(np.int32)]
        chance = np.clip((chance - 0.38) / 0.24, 0.0, 1.0)
        keep &= rng.random(tries) < (1.0 - clumping) + clumping * chance

    x, y = (u - 0.5) * size, (v - 0.5) * size
    for avoid_x, avoid_y, radius in avoid:
        keep &= np.hypot(x - avoid_x, y - avoid_y) > radius

    places = np.stack((x, y, level * height), axis=-1)[keep]
    if spacing > 0.0 and len(places):
        # Take places in their random order, turning down any too near one already taken.
        # A grid of buckets as wide as the spacing means only the neighbouring buckets need checking.
        taken = {}
        chosen = []
        for index, (px, py, _pz) in enumerate(places):
            cell = (int(px // spacing), int(py // spacing))
            crowded = any(
                (px - qx) ** 2 + (py - qy) ** 2 < spacing * spacing
                for cx in (cell[0] - 1, cell[0], cell[0] + 1)
                for cy in (cell[1] - 1, cell[1], cell[1] + 1)
                for qx, qy in taken.get((cx, cy), ())
            )
            if not crowded:
                taken.setdefault(cell, []).append((px, py))
                chosen.append(index)
                if len(chosen) >= limit:
                    break
        places = places[chosen]
    return places[:limit].astype(np.float32)


def water_level(scene, ground):
    """Height of the scene's water in the terrain's own space, or None if there is none."""
    for ob in scene.objects:
        if ob.type == 'MESH' and ob.name.startswith("Water"):
            return (ground.matrix_world.inverted() @ ob.matrix_world.translation).z
    return None


def instance(context, ground, meshes, places, seed, scale=(1.0, 1.0), tilt=0.0, sink=0.0, name="Scatter"):
    """
    Put one of ``meshes`` at each place, as children of the terrain in a collection of their own.

    ``tilt`` is the most a thing leans, in radians (rocks lie as they fell, trees stand up).
    ``sink`` buries the base by that fraction of the object's scale. Returns the collection.
    """
    rng = np.random.default_rng(seed)
    collection = bpy.data.collections.new(name)
    context.scene.collection.children.link(collection)
    for x, y, z in places:
        mesh = meshes[int(rng.integers(len(meshes)))]
        ob = bpy.data.objects.new(mesh.name, mesh)
        factor = float(rng.uniform(scale[0], scale[1]))
        ob.scale = (factor, factor, factor)
        ob.rotation_euler = (
            float(rng.uniform(-tilt, tilt)), float(rng.uniform(-tilt, tilt)), float(rng.uniform(0.0, math.tau)),
        )
        ob.location = (float(x), float(y), float(z) - sink * factor)
        # Parented with no inverse: locations are in the terrain's space, and follow it when it moves.
        ob.parent = ground
        collection.objects.link(ob)
    return collection


def make_meshes(what, seed, variants, triangles=600, height=0.0):
    """A few different meshes of one kind for the scattered objects to share."""
    if what == 'ROCK':
        return [objects.rock_mesh(seed * 100 + index) for index in range(variants)]
    return [trees.build_mesh(what, seed * 100 + index, height, triangles) for index in range(variants)]


def scatter(
        context, ground, what, seed=1, density=None, altitude=None, max_slope=None, clearance=0.3,
        spacing=None, clumping=None, scale=None, size_factor=1.0, variants=3, triangles=600, limit=500,
        avoid=(), water=None,
):
    """
    Scatter rocks or one species of plant over ``ground``; rules left as None use the defaults for the kind.

    ``size_factor`` scales the objects and their spacing together. Returns the new collection, or None if
    nowhere on the terrain passed the rules.
    """
    rules = DEFAULTS[what]
    settings = ground.elyan_terrain
    scale = scale or rules["scale"]
    places = find_places(
        terrain.read_grid(ground), settings.size, settings.height_applied, seed,
        rules["density"] if density is None else density,
        altitude or rules["altitude"],
        rules["slope"] if max_slope is None else max_slope,
        water, clearance,
        (rules["spacing"] if spacing is None else spacing) * size_factor,
        rules["clumping"] if clumping is None else clumping,
        limit, avoid=avoid,
    )
    if not len(places):
        return None
    meshes = make_meshes(what, seed, min(variants, len(places)), triangles)
    rock = what == 'ROCK'
    return instance(
        context, ground, meshes, places, seed, (scale[0] * size_factor, scale[1] * size_factor),
        0.25 if rock else 0.04, 0.25 if rock else 0.0, "Scatter " + (meshes[0].name.split(".")[0]),
    )


class ELYAN_OT_scatter(Operator):
    """Scatter rocks or plants over the active terrain, by altitude, slope and spacing"""
    bl_idname = "elyan_scenery.scatter"
    bl_label = "Scatter"
    bl_options = {'REGISTER', 'UNDO'}

    def _load_defaults(self, _context):
        rules = DEFAULTS[self.what]
        self.density = rules["density"]
        self.altitude_min, self.altitude_max = rules["altitude"]
        self.slope_max = math.radians(rules["slope"])
        self.spacing = rules["spacing"]
        self.clumping = rules["clumping"]
        self.scale_min, self.scale_max = rules["scale"]

    what: EnumProperty(name="Scatter", items=WHAT_ITEMS, update=_load_defaults)
    seed: IntProperty(name="Seed", description="Same seed, same places", default=1, min=0)
    density: FloatProperty(
        name="Density", description="How many per hectare (100 m square) of ground that passes the rules",
        default=6.0, min=0.0, soft_max=200.0,
    )
    limit: IntProperty(name="At Most", description="Never place more than this many", default=500, min=1, max=20000)
    altitude_min: FloatProperty(
        name="Lowest", description="As a fraction of the way from the terrain's lowest point to its highest",
        default=0.0, min=0.0, max=1.0, subtype='FACTOR',
    )
    altitude_max: FloatProperty(
        name="Highest", description="As a fraction of the way from the terrain's lowest point to its highest",
        default=1.0, min=0.0, max=1.0, subtype='FACTOR',
    )
    slope_max: FloatProperty(
        name="Steepest Slope", default=math.radians(40.0), min=0.0, max=math.radians(90.0), subtype='ANGLE',
    )
    water_clearance: FloatProperty(
        name="Above Water", description="Least height above the water, where the scene has an object named Water",
        default=0.3, min=0.0, unit='LENGTH',
    )
    spacing: FloatProperty(name="Spacing", description="Least distance between two", default=2.0, min=0.0, unit='LENGTH')
    clumping: FloatProperty(
        name="Clumping", description="Gather in groves and patches instead of spreading evenly",
        default=0.3, min=0.0, max=1.0, subtype='FACTOR',
    )
    scale_min: FloatProperty(name="Smallest", default=0.4, min=0.01, soft_max=5.0)
    scale_max: FloatProperty(name="Largest", default=2.2, min=0.01, soft_max=5.0)
    variants: IntProperty(
        name="Variants", description="Different meshes made; every scattered object shares one of them",
        default=3, min=1, max=16,
    )
    triangles: IntProperty(name="Triangles", description="Budget for each plant", default=600, min=8, max=50000)

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'MESH' and ob.elyan_terrain.is_terrain and ob.mode == 'OBJECT'

    def draw(self, _context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "what")
        layout.prop(self, "seed")
        col = layout.column(align=True)
        col.prop(self, "density")
        col.prop(self, "limit")
        col.prop(self, "spacing")
        col.prop(self, "clumping")
        col = layout.column(align=True)
        col.prop(self, "altitude_min", text="Altitude Lowest")
        col.prop(self, "altitude_max", text="Highest")
        col.prop(self, "slope_max")
        col.prop(self, "water_clearance")
        col = layout.column(align=True)
        col.prop(self, "scale_min", text="Scale Smallest")
        col.prop(self, "scale_max", text="Largest")
        if self.what != 'SELECTED':
            col = layout.column(align=True)
            col.prop(self, "variants")
            if self.what != 'ROCK':
                col.prop(self, "triangles")

    def execute(self, context):
        ground = context.object
        settings = ground.elyan_terrain
        places = find_places(
            terrain.read_grid(ground), settings.size, settings.height_applied, self.seed, self.density,
            (self.altitude_min, max(self.altitude_min, self.altitude_max)), math.degrees(self.slope_max),
            water_level(context.scene, ground), self.water_clearance, self.spacing, self.clumping, self.limit,
        )
        if not len(places):
            self.report({'WARNING'}, "Nowhere on the terrain passes these rules")
            return {'CANCELLED'}

        scale = (min(self.scale_min, self.scale_max), max(self.scale_min, self.scale_max))
        if self.what == 'SELECTED':
            meshes = [ob.data for ob in context.selected_objects if ob.type == 'MESH' and ob is not ground]
            if not meshes:
                self.report({'ERROR'}, "Select the mesh objects to scatter, then the terrain last")
                return {'CANCELLED'}
            name, tilt, sink = "Scatter", 0.0, 0.0
        else:
            try:
                meshes = make_meshes(self.what, self.seed, min(self.variants, len(places)), self.triangles)
            except ValueError as ex:
                self.report({'ERROR'}, str(ex))
                return {'CANCELLED'}
            rock = self.what == 'ROCK'
            name, tilt, sink = "Scatter " + meshes[0].name.split(".")[0], (0.25 if rock else 0.04), (0.25 if rock else 0.0)
        instance(context, ground, meshes, places, self.seed, scale, tilt, sink, name)
        self.report({'INFO'}, "Scattered {:d} sharing {:d} meshes".format(len(places), len(meshes)))
        return {'FINISHED'}


classes = (
    ELYAN_OT_scatter,
)
