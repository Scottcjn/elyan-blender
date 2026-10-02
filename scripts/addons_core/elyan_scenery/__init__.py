# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Scenery: fractal terrain with erosion, one-click skies with distance haze,
water, rocks, trees, scattering and organic blobs, in the spirit of the
landscape programs of the 1990s.
"""

bl_info = {
    "name": "Elyan Scenery",
    "description": "Fractal terrain, erosion, skies, haze, water, trees and scattering, with one-click landscapes",
    "author": "Elyan Labs",
    "version": (0, 2),
    "blender": (4, 2, 0),
    "location": "3D Viewport -> Sidebar -> Scenery, and Add -> Scenery",
    "support": "OFFICIAL",
    "category": "Add Mesh",
}

import bpy
from bpy.props import PointerProperty
from bpy.types import Menu, Panel

from . import instant, objects, scatter, sky, terrain, trees


class _SceneryPanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Scenery"


class ELYAN_PT_scenery_create(_SceneryPanel, Panel):
    bl_label = "Create"

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.scale_y = 1.5
        col.operator("elyan_scenery.instant", icon='WORLD')
        col = layout.column(align=True)
        col.operator("elyan_scenery.terrain_add", icon='RNDCURVE')
        col.operator("elyan_scenery.water_add", icon='MOD_OCEAN')
        col.operator("elyan_scenery.rock_add", icon='MESH_ICOSPHERE')
        col.operator_menu_enum("elyan_scenery.tree_add", "species", icon='STRANDS')
        col.operator("elyan_scenery.blob_add", icon='META_BALL')


class ELYAN_PT_scenery_terrain(_SceneryPanel, Panel):
    bl_label = "Terrain Editor"

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'MESH' and ob.elyan_terrain.is_terrain

    def draw(self, context):
        layout = self.layout
        settings = context.object.elyan_terrain
        layout.prop(settings, "height")
        layout.label(text="{:d} x {:d} points".format(settings.resolution, settings.resolution))

        col = layout.column(align=True)
        col.label(text="Erode")
        col.operator("elyan_scenery.terrain_filter", text="Rain").filter = 'ERODE_RAIN'
        col.operator("elyan_scenery.terrain_filter", text="Scree").filter = 'ERODE_SCREE'

        col = layout.column(align=True)
        col.label(text="Shape")
        grid = col.grid_flow(columns=2, align=True)
        for label, identifier in (
                ("Smooth", 'SMOOTH'), ("Terrace", 'TERRACE'), ("Raise", 'RAISE'), ("Lower", 'LOWER'),
                ("Sharpen", 'SHARPEN'), ("Invert", 'INVERT'), ("Edge Falloff", 'EDGE_FALLOFF'),
                ("Full Height", 'NORMALIZE'),
        ):
            grid.operator("elyan_scenery.terrain_filter", text=label).filter = identifier

        col = layout.column(align=True)
        col.label(text="Fractal")
        col.operator("elyan_scenery.terrain_generate", text="New Landscape").blend = 'REPLACE'
        col.operator("elyan_scenery.terrain_generate", text="Add Landscape On Top").blend = 'ADD'

        col = layout.column(align=True)
        col.label(text="Paint")
        col.operator("elyan_scenery.terrain_to_image", icon='IMAGE_DATA')
        col.operator("elyan_scenery.terrain_from_image", icon='IMPORT')

        col = layout.column(align=True)
        col.label(text="Cover")
        # Choosing what to scatter loads rules that suit it; they can be changed in the redo panel.
        col.operator_menu_enum("elyan_scenery.scatter", "what", icon='PARTICLES')
        col.operator_menu_enum("elyan_scenery.terrain_material", "material", icon='MATERIAL')
        col.operator("elyan_scenery.terrain_resample", icon='MOD_MULTIRES')


class ELYAN_PT_scenery_sky(_SceneryPanel, Panel):
    bl_label = "Sky Lab"

    def draw(self, context):
        grid = self.layout.grid_flow(columns=2, align=True)
        for identifier, label, _description in sky.SKY_ITEMS:
            props = grid.operator("elyan_scenery.sky_set", text=label)
            # Assigning the preset runs its update, which fills in the sun, haze and clouds.
            props.preset = identifier


class ELYAN_MT_scenery_add(Menu):
    bl_idname = "ELYAN_MT_scenery_add"
    bl_label = "Scenery"

    def draw(self, context):
        layout = self.layout
        layout.operator_context = 'INVOKE_REGION_WIN'
        layout.operator("elyan_scenery.instant", icon='WORLD')
        layout.separator()
        layout.operator("elyan_scenery.terrain_add", icon='RNDCURVE')
        layout.operator("elyan_scenery.water_add", icon='MOD_OCEAN')
        layout.operator("elyan_scenery.rock_add", icon='MESH_ICOSPHERE')
        layout.operator_menu_enum("elyan_scenery.tree_add", "species", icon='STRANDS')
        layout.operator("elyan_scenery.blob_add", icon='META_BALL')
        layout.separator()
        # Greyed out unless a terrain is active.
        layout.operator_menu_enum("elyan_scenery.scatter", "what", icon='PARTICLES')
        layout.separator()
        layout.operator_menu_enum("elyan_scenery.sky_set", "preset", icon='LIGHT_SUN')


def _add_menu(self, _context):
    self.layout.menu(ELYAN_MT_scenery_add.bl_idname, icon='WORLD')


classes = (
    *terrain.classes,
    *sky.classes,
    *objects.classes,
    *trees.classes,
    *scatter.classes,
    *instant.classes,
    ELYAN_PT_scenery_create,
    ELYAN_PT_scenery_terrain,
    ELYAN_PT_scenery_sky,
    ELYAN_MT_scenery_add,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.elyan_terrain = PointerProperty(type=terrain.TerrainSettings)
    bpy.types.VIEW3D_MT_add.append(_add_menu)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(_add_menu)
    del bpy.types.Object.elyan_terrain
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
