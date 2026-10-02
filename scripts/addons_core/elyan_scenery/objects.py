# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Scenery objects: water, rocks and organic blobs.
"""

import math
import random

import bmesh
import bpy
from bpy.props import FloatProperty, IntProperty
from bpy.types import Operator
from mathutils import Vector, noise

from . import materials


def link(context, ob):
    context.collection.objects.link(ob)
    return ob


def add_water(context, level, size):
    mesh = bpy.data.meshes.new("Water")
    half = size / 2.0
    mesh.from_pydata(
        [(-half, -half, 0.0), (half, -half, 0.0), (half, half, 0.0), (-half, half, 0.0)], [], [(0, 1, 2, 3)],
    )
    mesh.materials.append(materials.water())
    ob = link(context, bpy.data.objects.new("Water", mesh))
    ob.location.z = level
    return ob


def rock_mesh(seed):
    """A boulder about two units across: a sphere pushed around by noise, squashed and flat-bottomed."""
    rng = random.Random(seed)
    mesh = bpy.data.meshes.new("Rock")
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
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(materials.rock())
    return mesh


def add_rock(context, seed, size):
    ob = link(context, bpy.data.objects.new("Rock", rock_mesh(seed)))
    ob.scale = (size, size, size)
    ob.rotation_euler.z = random.Random(seed).uniform(0.0, math.tau)
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
    return link(context, bpy.data.objects.new("Blob", data))


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


classes = (
    ELYAN_OT_water_add,
    ELYAN_OT_rock_add,
    ELYAN_OT_blob_add,
)
