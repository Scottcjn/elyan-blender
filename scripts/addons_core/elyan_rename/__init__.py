# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Rename many things at once, from one small panel with a preview.

Blender's own batch rename (Ctrl F2) does more, but it is a stack of chained
steps in a popup. This is the common cases with nothing to learn: pick what
to rename, pick one change, see the result before applying it.
"""

bl_info = {
    "name": "Elyan Quick Rename",
    "description": "Rename selected objects, bones, materials, vertex groups or shape keys, with a preview",
    "author": "Elyan Labs",
    "version": (0, 1),
    "blender": (4, 2, 0),
    "location": "3D Viewport -> Sidebar -> Elyan -> Quick Rename",
    "support": "OFFICIAL",
    "category": "Object",
}

import re

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup

PREVIEW_ROWS = 6


def new_names(names, mode, find="", replace="", text="", start=1, digits=2, match_case=True):
    """The names ``names`` would get. Pure: no Blender data is touched."""
    result = []
    for index, name in enumerate(names):
        if mode == 'REPLACE':
            if not find:
                new = name
            elif match_case:
                new = name.replace(find, replace)
            else:
                new = re.sub(re.escape(find), lambda _m: replace, name, flags=re.IGNORECASE)
        elif mode == 'PREFIX':
            new = text + name
        elif mode == 'SUFFIX':
            new = name + text
        elif mode == 'NUMBER':
            new = "{:s}{:0{:d}d}".format(text, start + index, digits)
        elif mode == 'STRIP':
            # The ".001" Blender adds to duplicates.
            new = re.sub(r"\.\d{3,}$", "", name)
        else:
            raise ValueError("unknown mode {!r}".format(mode))
        result.append(new)
    return result


def _targets(context, what):
    """The things to rename, in a stable order: a list of objects with a ``name``."""
    ob = context.object
    if what == 'OBJECTS':
        return sorted(context.selected_objects, key=lambda o: o.name)
    if what == 'DATA':
        seen, items = set(), []
        for item in sorted(context.selected_objects, key=lambda o: o.name):
            if item.data is not None and item.data.name not in seen:
                seen.add(item.data.name)
                items.append(item.data)
        return items
    if what == 'MATERIALS':
        seen, items = set(), []
        for item in context.selected_objects:
            for slot in item.material_slots:
                if slot.material and slot.material.name not in seen:
                    seen.add(slot.material.name)
                    items.append(slot.material)
        return sorted(items, key=lambda m: m.name)
    if ob is None:
        return []
    if what == 'BONES':
        if ob.type != 'ARMATURE':
            return []
        if context.mode == 'POSE':
            return [bone.bone for bone in context.selected_pose_bones or ()]
        if context.mode == 'EDIT_ARMATURE':
            return list(context.selected_editable_bones or ())
        return list(ob.data.bones)
    if what == 'VERTEX_GROUPS':
        return list(ob.vertex_groups)
    if what == 'SHAPE_KEYS':
        keys = getattr(ob.data, "shape_keys", None)
        # The first key is the basis; renaming it by accident breaks other tools.
        return list(keys.key_blocks[1:]) if keys else []
    return []


def _planned(context):
    settings = context.window_manager.elyan_rename
    targets = _targets(context, settings.what)
    names = new_names(
        [item.name for item in targets], settings.mode, settings.find, settings.replace,
        settings.text, settings.start, settings.digits, settings.match_case)
    return targets, names


class ElyanRenameSettings(PropertyGroup):
    what: EnumProperty(
        name="Rename",
        items=(
            ('OBJECTS', "Selected Objects", ""),
            ('DATA', "Their Mesh/Data", "The mesh, armature or curve data of the selected objects"),
            ('MATERIALS', "Their Materials", "Materials used by the selected objects"),
            ('BONES', "Bones", "Selected bones in Pose or Edit Mode, otherwise every bone of the active armature"),
            ('VERTEX_GROUPS', "Vertex Groups", "Of the active object"),
            ('SHAPE_KEYS', "Shape Keys", "Of the active object, leaving the basis alone"),
        ),
    )
    mode: EnumProperty(
        name="Change",
        items=(
            ('REPLACE', "Find and Replace", ""),
            ('PREFIX', "Add to the Start", ""),
            ('SUFFIX', "Add to the End", ""),
            ('NUMBER', "Name and Number", "Give them all one name followed by a running number"),
            ('STRIP', "Remove .001", "Remove the number Blender adds to duplicates"),
        ),
    )
    find: StringProperty(name="Find")
    replace: StringProperty(name="Replace With")
    text: StringProperty(name="Text")
    start: IntProperty(name="Start At", default=1, min=0)
    digits: IntProperty(name="Digits", default=2, min=1, max=6)
    match_case: BoolProperty(name="Match Case", default=True)


class ELYAN_OT_rename_apply(Operator):
    """Rename as shown in the preview"""
    bl_idname = "elyan_rename.apply"
    bl_label = "Rename"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        targets, names = _planned(context)
        changed = 0
        for item, name in zip(targets, names):
            if name and name != item.name:
                item.name = name
                changed += 1
        self.report({'INFO'}, "Renamed {:d} of {:d}".format(changed, len(targets)))
        return {'FINISHED'}


class ELYAN_PT_rename(Panel):
    bl_label = "Quick Rename"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Elyan"

    def draw(self, context):
        layout = self.layout
        settings = context.window_manager.elyan_rename
        layout.prop(settings, "what", text="")
        layout.prop(settings, "mode", text="")
        col = layout.column(align=True)
        if settings.mode == 'REPLACE':
            col.prop(settings, "find")
            col.prop(settings, "replace")
            col.prop(settings, "match_case")
        elif settings.mode in {'PREFIX', 'SUFFIX'}:
            col.prop(settings, "text")
        elif settings.mode == 'NUMBER':
            col.prop(settings, "text", text="Name")
            row = col.row(align=True)
            row.prop(settings, "start")
            row.prop(settings, "digits")

        targets, names = _planned(context)
        changes = [(item.name, name) for item, name in zip(targets, names) if name != item.name]
        box = layout.box()
        if not targets:
            box.label(text="Nothing to rename here", icon='INFO')
        elif not changes:
            box.label(text="{:d} found, no names would change".format(len(targets)), icon='INFO')
        else:
            for old, new in changes[:PREVIEW_ROWS]:
                box.label(text="{:s}  ->  {:s}".format(old, new))
            if len(changes) > PREVIEW_ROWS:
                box.label(text="... and {:d} more".format(len(changes) - PREVIEW_ROWS))
        row = layout.row()
        row.enabled = bool(changes)
        row.operator("elyan_rename.apply", text="Rename {:d}".format(len(changes)) if changes else "Rename")


classes = (
    ElyanRenameSettings,
    ELYAN_OT_rename_apply,
    ELYAN_PT_rename,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.elyan_rename = PointerProperty(type=ElyanRenameSettings)


def unregister():
    del bpy.types.WindowManager.elyan_rename
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
