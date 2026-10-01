# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
People: characters from a recipe, measured, and exported within budget.

Bodies, rigs and clothing come from the MPFB extension (MakeHuman for Blender),
which must be installed separately. This add-on adds the recipe, the landmark
contract and the game-ready export.
"""

bl_info = {
    "name": "Elyan People",
    "description": "Characters from a recipe (via MPFB), with landmark contract and budget export",
    "author": "Elyan Labs",
    "version": (0, 1),
    "blender": (4, 2, 0),
    "location": "3D Viewport -> Sidebar -> People",
    "support": "OFFICIAL",
    "category": "Add Mesh",
}

import json

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import Operator, Panel

from . import build, export, recipe

PROFILE_ITEMS = (
    ('web', "Web / WebXR", "GLB for the browser"),
    ('quest', "VRChat Quest", "Tight triangle and texture budget"),
    ('vrchat_pc', "VRChat PC", "Roomier budget"),
)


class ELYAN_OT_person_create(Operator):
    """Create a person from a recipe"""
    bl_idname = "elyan_people.create"
    bl_label = "Create Person"
    bl_options = {'REGISTER', 'UNDO'}

    recipe: StringProperty(
        name="Recipe", description="Recipe as JSON; empty gives the default person", default="",
    )
    filepath: StringProperty(name="Recipe File", description="Read the recipe from this JSON file", subtype='FILE_PATH')

    def execute(self, context):
        try:
            text = self.recipe
            if self.filepath:
                with open(bpy.path.abspath(self.filepath), encoding="utf-8") as fh:
                    text = fh.read()
            rig, _report = build.build(text or {})
        except (recipe.RecipeError, build.MPFBMissing, OSError) as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        rig.location = context.scene.cursor.location
        return {'FINISHED'}


class ELYAN_OT_person_export(Operator):
    """Export the active person as one mesh with one material, within the profile's budget"""
    bl_idname = "elyan_people.export"
    bl_label = "Export Person"

    filepath: StringProperty(name="File", subtype='FILE_PATH')
    profile: EnumProperty(name="Profile", items=PROFILE_ITEMS)
    use_fbx: BoolProperty(name="Also FBX", description="Write an FBX for Unity beside the GLB", default=False)

    @classmethod
    def poll(cls, context):
        return build.find_rig(context.object) is not None

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        rig = build.find_rig(context.object)
        formats = ("glb", "fbx") if self.use_fbx else ("glb",)
        manifest = export.export(rig, bpy.path.abspath(self.filepath), self.profile, formats)
        failures = manifest.get("validation", {}).get("failures", [])
        if failures:
            self.report({'WARNING'}, "Exported, but: " + "; ".join(failures))
        else:
            self.report({'INFO'}, "Exported {:d} triangles".format(manifest["round_trip"]["triangles"]))
        return {'FINISHED'}


class ELYAN_PT_people(Panel):
    bl_label = "People"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "People"

    def draw(self, context):
        layout = self.layout
        layout.operator("elyan_people.create", icon='OUTLINER_OB_ARMATURE')
        rig = build.find_rig(context.object)
        if rig is None:
            return
        contract = json.loads(rig[build.PROPERTY])["contract"]
        col = layout.column(align=True)
        col.label(text="Height {:.2f} m".format(contract["height"]))
        for key in ("chest", "waist", "hip"):
            if key in contract:
                col.label(text="{:s} {:.0f} cm".format(key.title(), contract[key]["girth"] * 100.0))
        layout.operator("elyan_people.export", icon='EXPORT')


classes = (
    ELYAN_OT_person_create,
    ELYAN_OT_person_export,
    ELYAN_PT_people,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
