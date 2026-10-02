# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Apply Pose as Rest Pose, for the meshes too.

Blender's own "Apply Pose as Rest Pose" changes only the armature: every mesh
rigged to it then jumps, because it is deformed from a new rest position. This
moves the meshes with it, shape keys included, so the character looks exactly
as posed and that pose is the new rest.

Each shape key is read as the armature deforms it and written back, then the
armature's rest pose is replaced. Skinning is linear in the vertex positions,
so every shape key keeps working on the re-posed mesh.
"""

bl_info = {
    "name": "Apply Pose as Rest Pose (with Meshes)",
    "description": "Make the current pose the rest pose and keep rigged meshes and their shape keys in shape",
    "author": "Elyan Labs",
    "version": (0, 1),
    "blender": (4, 2, 0),
    "location": "Pose Mode: Pose -> Apply, and 3D Viewport -> Sidebar -> Elyan",
    "support": "OFFICIAL",
    "category": "Rigging",
}

import bpy
import numpy as np
from bpy.props import BoolProperty
from bpy.types import Operator, Panel


def rigged_meshes(rig, only_selected=False):
    """Mesh objects deformed by ``rig`` through an Armature modifier."""
    found = []
    for ob in bpy.data.objects:
        if ob.type != 'MESH' or (only_selected and not ob.select_get()):
            continue
        if any(m.type == 'ARMATURE' and m.object == rig for m in ob.modifiers):
            found.append(ob)
    return found


def _deformed(context, ob):
    """Vertex positions of ``ob`` as its enabled modifiers leave them, in its own space."""
    context.view_layer.update()
    evaluated = ob.evaluated_get(context.evaluated_depsgraph_get())
    co = np.empty(len(evaluated.data.vertices) * 3, dtype=np.float32)
    evaluated.data.vertices.foreach_get("co", co)
    return co


def bake_pose(context, ob, rig):
    """
    Give ``ob`` the shape the armature currently deforms it into, in every shape key.

    Returns the number of shape keys rewritten (0 for a mesh without any).
    """
    mesh = ob.data
    if mesh.users > 1:
        # Another object shares this mesh and would be moved a second time by its own modifier.
        ob.data = mesh = mesh.copy()
    count = len(mesh.vertices)

    # Only this armature's deformation is wanted, not a mirror or subdivision on top.
    modifiers = [(m, m.show_viewport) for m in ob.modifiers]
    for modifier, _shown in modifiers:
        modifier.show_viewport = modifier.type == 'ARMATURE' and modifier.object == rig
    pinned, active = ob.show_only_shape_key, ob.active_shape_key_index
    try:
        blocks = mesh.shape_keys.key_blocks if mesh.shape_keys else None
        if not blocks:
            co = _deformed(context, ob)
            if len(co) != count * 3:
                raise RuntimeError("{:s}: the armature modifier changes the vertex count".format(ob.name))
            mesh.vertices.foreach_set("co", co)
            mesh.update()
            return 0

        results = []
        ob.show_only_shape_key = True
        for index, block in enumerate(blocks):
            # A pinned key shows at full strength; a mask or a mute would hide part of it from the reading.
            mask, muted = block.vertex_group, block.mute
            block.vertex_group, block.mute = "", False
            ob.active_shape_key_index = index
            results.append(_deformed(context, ob))
            block.vertex_group, block.mute = mask, muted
        for block, co in zip(blocks, results):
            block.data.foreach_set("co", co)
        # The mesh's own vertices mirror the first key.
        mesh.vertices.foreach_set("co", results[0])
        mesh.update()
        return len(blocks)
    finally:
        ob.show_only_shape_key, ob.active_shape_key_index = pinned, active
        for modifier, shown in modifiers:
            modifier.show_viewport = shown


class ELYAN_OT_apply_rest_pose(Operator):
    """Make the current pose the rest pose, moving rigged meshes and their shape keys with it"""
    bl_idname = "elyan_restpose.apply"
    bl_label = "Apply Pose as Rest Pose (with Meshes)"
    bl_options = {'REGISTER', 'UNDO'}

    only_selected: BoolProperty(
        name="Selected Meshes Only",
        description="Leave rigged meshes that are not selected as they are; they will jump to the new rest pose",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'ARMATURE' and context.mode == 'POSE'

    def execute(self, context):
        rig = context.object
        meshes = rigged_meshes(rig, self.only_selected)
        keys = 0
        try:
            for ob in meshes:
                keys += bake_pose(context, ob, rig)
        except RuntimeError as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        bpy.ops.pose.armature_apply(selected=False)
        self.report({'INFO'}, "New rest pose; {:d} meshes and {:d} shape keys kept in shape".format(len(meshes), keys))
        return {'FINISHED'}


class ELYAN_PT_rest_pose(Panel):
    bl_label = "Rest Pose"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Elyan"

    @classmethod
    def poll(cls, context):
        return ELYAN_OT_apply_rest_pose.poll(context)

    def draw(self, context):
        layout = self.layout
        layout.operator("elyan_restpose.apply", text="Apply Pose as Rest Pose", icon='ARMATURE_DATA')
        layout.label(text="{:d} rigged meshes will follow".format(len(rigged_meshes(context.object))))


def _apply_menu(self, _context):
    self.layout.operator("elyan_restpose.apply")


classes = (
    ELYAN_OT_apply_rest_pose,
    ELYAN_PT_rest_pose,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_pose_apply.append(_apply_menu)


def unregister():
    bpy.types.VIEW3D_MT_pose_apply.remove(_apply_menu)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
