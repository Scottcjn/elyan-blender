# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
People: characters from a recipe, measured, able to talk, and exported within budget.

Bodies, rigs, clothing and face shapes come from the MPFB extension (MakeHuman
for Blender) and its asset packs, which must be installed separately. This
add-on adds the recipe, the landmark contract, face checks, body motion, lip-sync
tracks (``speech.py``), a runtime face player (``runtime/``) and the game-ready export.
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
import os

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import Operator, Panel

from . import build, export, motion, recipe

list_assets = build.list_assets


def make(recipe, path, profile="web", clips=None, formats=("glb",), keep=False, sheet=False, split_head=None):
    """
    Recipe to files in one call: build the person, check the face, export, and return the manifest.

    ``recipe`` is a dict or JSON text; ``clips`` names motions from ``motion.CLIPS`` (default: all).
    The manifest gains ``build`` (seconds and the face check's verdict) and is saved again with it.
    With ``keep`` false the person is removed from the scene afterwards, so calls do not pile up.
    Raises ``recipe.RecipeError`` or ``build.MPFBMissing`` before anything is written.
    """
    rig, report = build.build(recipe)
    clips = tuple(motion.CLIPS) if clips is None else tuple(clips)
    manifest = export.export(
        rig, path, profile, tuple(formats), clips=clips, sheet=sheet, split_head=split_head)
    checked = report.get("face", {})
    manifest["build"] = {
        "seconds": report["seconds"],
        "face_passed": checked.get("passed"),
        "face_failures": checked.get("failures", []),
        "face_notes": checked.get("notes", []),
    }
    manifest["timings"] = dict({"build": report["seconds"]}, **manifest["timings"])
    stem = os.path.splitext(os.path.abspath(path))[0]
    with open(stem + ".manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    if not keep:
        for ob in list(rig.children_recursive) + [rig]:
            bpy.data.objects.remove(ob)
    return manifest


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
    """Export the active person with one material, within the profile's budget"""
    bl_idname = "elyan_people.export"
    bl_label = "Export Person"

    filepath: StringProperty(name="File", subtype='FILE_PATH')
    profile: EnumProperty(name="Profile", items=PROFILE_ITEMS)
    use_fbx: BoolProperty(name="Also FBX", description="Write an FBX for Unity beside the GLB", default=False)
    use_motion: BoolProperty(
        name="Body Motion", description="Include every generated body motion as an animation", default=True,
    )

    @classmethod
    def poll(cls, context):
        return build.find_rig(context.object) is not None

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        rig = build.find_rig(context.object)
        formats = ("glb", "fbx") if self.use_fbx else ("glb",)
        clips = tuple(motion.CLIPS) if self.use_motion else ()
        manifest = export.export(rig, bpy.path.abspath(self.filepath), self.profile, formats, clips=clips)
        failures = manifest.get("validation", {}).get("failures", [])
        if failures:
            self.report({'WARNING'}, "Exported, but: " + "; ".join(failures))
        else:
            self.report({'INFO'}, "Exported {:d} triangles".format(manifest["round_trip"]["triangles"]))
        return {'FINISHED'}


class ELYAN_OT_person_build_and_export(Operator):
    """Create a person from a recipe and export it, in one step"""
    bl_idname = "elyan_people.build_and_export"
    bl_label = "Build and Export Person"

    recipe: StringProperty(
        name="Recipe", description="Recipe as JSON; empty gives the default person", default="",
    )
    recipe_file: StringProperty(
        name="Recipe File", description="Read the recipe from this JSON file", subtype='FILE_PATH',
    )
    filepath: StringProperty(name="File", description="Where to write; the extension follows the format",
                             subtype='FILE_PATH')
    profile: EnumProperty(name="Profile", items=PROFILE_ITEMS)
    clips: StringProperty(
        name="Clips", description="Comma-separated body motions; \"all\" for every one, empty for none",
        default="all",
    )
    use_fbx: BoolProperty(name="Also FBX", description="Write an FBX for Unity beside the GLB", default=False)
    use_sheet: BoolProperty(
        name="Contact Sheet", description="Render every clip's pose to one image (slow)", default=False,
    )
    keep: BoolProperty(name="Keep Person", description="Leave the person in the scene", default=False)

    def execute(self, _context):
        if not self.filepath:
            self.report({'ERROR'}, "No file to write to")
            return {'CANCELLED'}
        try:
            text = self.recipe
            if self.recipe_file:
                with open(bpy.path.abspath(self.recipe_file), encoding="utf-8") as fh:
                    text = fh.read()
            wanted = self.clips.strip()
            clips = None if wanted == "all" else tuple(name.strip() for name in wanted.split(",") if name.strip())
            manifest = make(
                text or {}, bpy.path.abspath(self.filepath), self.profile, clips,
                formats=("glb", "fbx") if self.use_fbx else ("glb",), keep=self.keep, sheet=self.use_sheet)
        except (recipe.RecipeError, build.MPFBMissing, OSError, ValueError) as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        failures = manifest.get("validation", {}).get("failures", []) + manifest["build"]["face_failures"]
        if failures:
            self.report({'WARNING'}, "Exported, but: " + "; ".join(failures))
        else:
            self.report({'INFO'}, "Exported {:d} triangles in {:.0f} s".format(
                manifest["round_trip"]["triangles"], sum(manifest["timings"].values())))
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
    ELYAN_OT_person_build_and_export,
    ELYAN_PT_people,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
