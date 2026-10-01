# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Terrain objects: a square grid mesh whose vertex heights *are* the height map.

Nothing is stored beside the mesh, so terrains survive undo, saving and being
opened in a Blender without this add-on.
"""

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import Operator, PropertyGroup
from mathutils import Vector

from . import heightfield, materials

LANDSCAPE_ITEMS = (
    ('MOUNTAINS', "Mountains", "Sharp ridged peaks"),
    ('HILLS', "Rolling Hills", "Soft rounded hills"),
    ('DUNES', "Dunes", "Wind-blown sand ridges"),
    ('CANYON', "Canyon", "Flat mesas cut by a gorge"),
    ('ISLAND', "Island", "Peaks falling away to a shoreline on every side"),
    ('VOLCANO', "Volcano", "A single cone with a crater"),
    ('PLATEAU', "Plateau", "High flat ground with broken edges"),
    ('CRATERS', "Craters", "Impact-scarred ground"),
    ('PLAINS', "Plains", "Nearly flat, gently uneven ground"),
)


def _height_update(self, _context):
    ob = self.id_data
    if not self.is_terrain or self.height_applied <= 0.0 or self.height == self.height_applied:
        return
    co = _read_co(ob.data)
    co[:, 2] *= self.height / self.height_applied
    ob.data.vertices.foreach_set("co", co.ravel())
    ob.data.update()
    self.height_applied = self.height


class TerrainSettings(PropertyGroup):
    is_terrain: BoolProperty(options={'HIDDEN'})
    resolution: IntProperty(options={'HIDDEN'})
    size: FloatProperty(options={'HIDDEN'})
    height_applied: FloatProperty(options={'HIDDEN'})
    height: FloatProperty(
        name="Height",
        description="Height of the tallest possible point",
        min=0.01, soft_max=500.0, unit='LENGTH',
        update=_height_update,
    )


# -----------------------------------------------------------------------------
# Mesh <-> height field

def _read_co(mesh):
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def read_grid(ob):
    """Height field of a terrain, values 0..1 of its height."""
    settings = ob.elyan_terrain
    n = settings.resolution
    return (_read_co(ob.data)[:, 2] / settings.height_applied).reshape(n, n)


def write_grid(ob, grid):
    settings = ob.elyan_terrain
    n = grid.shape[0]
    if n != settings.resolution:
        # Vertex count changes, so the mesh is rebuilt; materials are kept.
        old = ob.data
        ob.data = build_mesh(old.name, grid, settings.size, settings.height_applied)
        for material in old.materials:
            ob.data.materials.append(material)
        bpy.data.meshes.remove(old)
        settings.resolution = n
        return
    co = _read_co(ob.data)
    co[:, 2] = grid.ravel() * settings.height_applied
    ob.data.vertices.foreach_set("co", co.ravel())
    ob.data.update()


def build_mesh(name, grid, size, height):
    n = grid.shape[0]
    axis = np.linspace(-size / 2.0, size / 2.0, n, dtype=np.float32)
    co = np.empty((n, n, 3), dtype=np.float32)
    co[..., 0] = axis[None, :]
    co[..., 1] = axis[:, None]
    co[..., 2] = grid * height

    index = np.arange(n * n, dtype=np.int32).reshape(n, n)
    quads = np.stack(
        (index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]), axis=-1,
    ).reshape(-1, 4)

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(n * n)
    mesh.vertices.foreach_set("co", co.ravel())
    mesh.loops.add(quads.size)
    mesh.loops.foreach_set("vertex_index", quads.ravel())
    mesh.polygons.add(len(quads))
    mesh.polygons.foreach_set("loop_start", np.arange(len(quads), dtype=np.int32) * 4)
    mesh.update(calc_edges=True)

    uv = (co[..., :2] / size + 0.5).reshape(-1, 2)[quads.ravel()]
    mesh.uv_layers.new(name="UVMap").data.foreach_set("uv", uv.ravel())
    mesh.shade_smooth()
    return mesh


def create(context, name, grid, size, height):
    ob = bpy.data.objects.new(name, build_mesh(name, grid, size, height))
    settings = ob.elyan_terrain
    settings.is_terrain = True
    settings.resolution = grid.shape[0]
    settings.size = size
    settings.height_applied = height
    settings.height = height
    context.collection.objects.link(ob)
    for other in context.selected_objects:
        other.select_set(False)
    ob.select_set(True)
    context.view_layer.objects.active = ob
    return ob


def height_at(ob, x, y):
    """World height of the terrain surface under a world-space ``x``, ``y``."""
    settings = ob.elyan_terrain
    local = ob.matrix_world.inverted() @ Vector((x, y, 0.0))
    u = local.x / settings.size + 0.5
    v = local.y / settings.size + 0.5
    local.z = heightfield.sample(read_grid(ob), u, v) * settings.height_applied
    return (ob.matrix_world @ local).z


class _TerrainOperator:
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'MESH' and ob.elyan_terrain.is_terrain and ob.mode == 'OBJECT'


# -----------------------------------------------------------------------------
# Operators

class ELYAN_OT_terrain_add(Operator):
    """Add a fractal terrain"""
    bl_idname = "elyan_scenery.terrain_add"
    bl_label = "Terrain"
    bl_options = {'REGISTER', 'UNDO'}

    landscape: EnumProperty(name="Landscape", items=LANDSCAPE_ITEMS)
    seed: IntProperty(name="Seed", description="Same seed, same terrain", default=1, min=0)
    resolution: IntProperty(
        name="Resolution", description="Points along each side", default=257, min=17, max=2049,
    )
    size: FloatProperty(name="Size", default=200.0, min=1.0, unit='LENGTH')
    height: FloatProperty(name="Height", default=45.0, min=0.01, unit='LENGTH')
    edge_falloff: FloatProperty(
        name="Edge Falloff", description="Lower the borders so the terrain meets the ground",
        default=0.0, min=0.0, max=0.5, subtype='FACTOR',
    )
    material: EnumProperty(name="Material", items=materials.TERRAIN_ITEMS, default='ALPINE')

    def execute(self, context):
        grid = heightfield.generate(self.landscape, self.resolution, self.seed)
        if self.edge_falloff > 0.0:
            grid = heightfield.edge_falloff(grid, self.edge_falloff)
        ob = create(context, "Terrain", grid, self.size, self.height)
        ob.location = context.scene.cursor.location
        ob.data.materials.append(materials.terrain(self.material))
        return {'FINISHED'}


class ELYAN_OT_terrain_generate(_TerrainOperator, Operator):
    """Replace or blend the active terrain's shape with a fractal landscape"""
    bl_idname = "elyan_scenery.terrain_generate"
    bl_label = "Generate Landscape"

    landscape: EnumProperty(name="Landscape", items=LANDSCAPE_ITEMS)
    seed: IntProperty(name="Seed", default=1, min=0)
    blend: EnumProperty(
        name="Blend",
        items=(
            ('REPLACE', "Replace", ""),
            ('ADD', "Add", "Pile the new landscape on top"),
            ('MAX', "Maximum", "Keep whichever is higher"),
            ('MULTIPLY', "Multiply", "New landscape only where there was height"),
        ),
    )
    factor: FloatProperty(name="Factor", default=1.0, min=0.0, max=1.0, subtype='FACTOR')

    def execute(self, context):
        ob = context.object
        old = read_grid(ob)
        new = heightfield.generate(self.landscape, old.shape[0], self.seed)
        if self.blend == 'ADD':
            new = old + new
        elif self.blend == 'MAX':
            new = np.maximum(old, new)
        elif self.blend == 'MULTIPLY':
            new = old * new
        write_grid(ob, old + (new - old) * self.factor)
        return {'FINISHED'}


class ELYAN_OT_terrain_filter(_TerrainOperator, Operator):
    """Reshape the active terrain"""
    bl_idname = "elyan_scenery.terrain_filter"
    bl_label = "Terrain Filter"

    filter: EnumProperty(
        name="Filter",
        items=(
            ('ERODE_RAIN', "Rain Erosion", "Water carves gullies and silts the valleys"),
            ('ERODE_SCREE', "Scree Erosion", "Steep faces crumble into slopes"),
            ('SMOOTH', "Smooth", ""),
            ('TERRACE', "Terrace", "Stepped strata"),
            ('EDGE_FALLOFF', "Edge Falloff", "Lower the borders to zero"),
            ('RAISE', "Raise", "Lift everything, lowlands most"),
            ('LOWER', "Lower", "Push everything down, flooding the lowlands flat"),
            ('SHARPEN', "Sharpen Peaks", "Exaggerate the high ground"),
            ('INVERT', "Invert", "Peaks become pits"),
            ('NORMALIZE', "Use Full Height", "Stretch to use the whole height range"),
        ),
    )
    amount: FloatProperty(name="Amount", default=0.5, min=0.0, max=1.0, subtype='FACTOR')
    seed: IntProperty(name="Seed", default=0, min=0)

    def execute(self, context):
        ob = context.object
        grid = read_grid(ob)
        amount = self.amount
        if self.filter == 'ERODE_RAIN':
            grid = heightfield.hydraulic_erosion(grid, int(10 + 150 * amount), self.seed)
        elif self.filter == 'ERODE_SCREE':
            grid = heightfield.thermal_erosion(grid, int(5 + 95 * amount))
        elif self.filter == 'SMOOTH':
            grid = heightfield.smooth(grid, int(1 + 12 * amount))
        elif self.filter == 'TERRACE':
            grid = heightfield.terrace(grid, 3 + int(9 * (1.0 - amount)), 0.75)
        elif self.filter == 'EDGE_FALLOFF':
            grid = heightfield.edge_falloff(grid, 0.02 + 0.45 * amount)
        elif self.filter == 'RAISE':
            grid = grid + (1.0 - grid) * amount * 0.5
        elif self.filter == 'LOWER':
            grid = np.clip(grid - amount * 0.5, 0.0, None)
        elif self.filter == 'SHARPEN':
            grid = np.clip(grid, 0.0, None) ** (1.0 + 2.0 * amount)
        elif self.filter == 'INVERT':
            grid = float(grid.max()) - grid
        elif self.filter == 'NORMALIZE':
            grid = heightfield.normalize(grid)
        write_grid(ob, grid.astype(np.float32))
        return {'FINISHED'}


class ELYAN_OT_terrain_resample(_TerrainOperator, Operator):
    """Change how many points the active terrain has, keeping its shape"""
    bl_idname = "elyan_scenery.terrain_resample"
    bl_label = "Change Resolution"

    resolution: IntProperty(name="Resolution", default=513, min=17, max=2049)

    def execute(self, context):
        ob = context.object
        write_grid(ob, heightfield.resample(read_grid(ob), self.resolution))
        return {'FINISHED'}


class ELYAN_OT_terrain_to_image(_TerrainOperator, Operator):
    """Copy the active terrain's heights into an image that can be painted on"""
    bl_idname = "elyan_scenery.terrain_to_image"
    bl_label = "Heights to Image"

    def execute(self, context):
        ob = context.object
        grid = np.clip(read_grid(ob), 0.0, 1.0)
        n = grid.shape[0]
        name = ob.name + " Heights"
        image = bpy.data.images.get(name)
        if image is None or tuple(image.size) != (n, n):
            if image is not None:
                bpy.data.images.remove(image)
            image = bpy.data.images.new(name, n, n, alpha=False, float_buffer=True, is_data=True)
        pixels = np.ones((n, n, 4), dtype=np.float32)
        pixels[..., :3] = grid[..., None]
        image.pixels.foreach_set(pixels.ravel())
        image.update()
        self.report({'INFO'}, "Paint on image \"{:s}\", then use Image to Heights".format(name))
        return {'FINISHED'}


class ELYAN_OT_terrain_from_image(_TerrainOperator, Operator):
    """Shape the active terrain from an image: white is high, black is low"""
    bl_idname = "elyan_scenery.terrain_from_image"
    bl_label = "Image to Heights"

    image: StringProperty(name="Image", description="Image name, empty uses \"<terrain> Heights\"")

    def execute(self, context):
        ob = context.object
        image = bpy.data.images.get(self.image or ob.name + " Heights")
        if image is None or image.size[0] == 0:
            self.report({'ERROR'}, "No such image")
            return {'CANCELLED'}
        width, height = image.size
        pixels = np.empty(width * height * image.channels, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        grey = pixels.reshape(height, width, image.channels)[..., :min(3, image.channels)].mean(axis=-1)
        # Nearest-sample to a square, then let the bilinear resize do the rest.
        side = max(width, height)
        rows = np.linspace(0, height - 1, side).round().astype(np.int32)
        cols = np.linspace(0, width - 1, side).round().astype(np.int32)
        square = grey[rows[:, None], cols[None, :]]
        write_grid(ob, heightfield.resample(square, ob.elyan_terrain.resolution))
        return {'FINISHED'}


class ELYAN_OT_terrain_material(_TerrainOperator, Operator):
    """Give the active terrain an altitude-and-slope material"""
    bl_idname = "elyan_scenery.terrain_material"
    bl_label = "Terrain Material"

    material: EnumProperty(name="Material", items=materials.TERRAIN_ITEMS)

    def execute(self, context):
        mesh = context.object.data
        material = materials.terrain(self.material)
        if mesh.materials:
            mesh.materials[0] = material
        else:
            mesh.materials.append(material)
        return {'FINISHED'}


classes = (
    TerrainSettings,
    ELYAN_OT_terrain_add,
    ELYAN_OT_terrain_generate,
    ELYAN_OT_terrain_filter,
    ELYAN_OT_terrain_resample,
    ELYAN_OT_terrain_to_image,
    ELYAN_OT_terrain_from_image,
    ELYAN_OT_terrain_material,
)
