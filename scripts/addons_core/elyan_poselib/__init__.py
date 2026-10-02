# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
The classic Pose Library: a named list of poses kept with the armature.

Blender had this until 3.5, when the asset-based pose library replaced it. The
old one was a list in the armature's properties: pose the rig, press "+", give
it a name; pick a name, press apply. This brings that back, with the same
operator names and shortcuts, and it reads pose libraries made in old files.

A pose library is an Action. Each pose is one frame of keys in it, named by a
pose marker on that frame, exactly as before.
"""

bl_info = {
    "name": "Pose Library (Classic)",
    "description": "The pose list from before Blender 3.5: add, apply, rename and browse named poses",
    "author": "Elyan Labs",
    "version": (0, 1),
    "blender": (4, 2, 0),
    "location": "Properties -> Armature -> Pose Library, and 3D Viewport -> Sidebar -> Elyan (Pose Mode)",
    "support": "OFFICIAL",
    "category": "Animation",
}

import re

import bpy
from bpy.props import EnumProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Menu, Operator, Panel

_PATH = re.compile(r'^pose\.bones\["(.+)"\]\.(\w+)$')
_ROTATION = {'QUATERNION': "rotation_quaternion", 'AXIS_ANGLE': "rotation_axis_angle"}


# -----------------------------------------------------------------------------
# Data

def _fcurves(action, create=False):
    """The action's F-curves, wherever this Blender keeps them."""
    if hasattr(action, "fcurves"):
        return action.fcurves
    # Layered actions (4.4 and later): curves live in a channelbag for a slot.
    from bpy_extras import anim_utils
    slot = action.slots[0] if len(action.slots) else None
    if slot is None:
        if not create:
            return None
        slot = action.slots.new(id_type='OBJECT', name="Poses")
    if create:
        return anim_utils.action_ensure_channelbag_for_slot(action, slot).fcurves
    bag = anim_utils.action_get_channelbag_for_slot(action, slot)
    return bag.fcurves if bag else None


def _new_fcurve(fcurves, data_path, index, bone_name):
    try:
        return fcurves.new(data_path, index=index, group_name=bone_name)
    except TypeError:
        return fcurves.new(data_path, index=index, action_group=bone_name)


def _selected(bone):
    # Selection moved from the bone to the pose bone in Blender 5.0.
    return bone.select if hasattr(bone, "select") else bone.bone.select


def _bones(ob):
    """The bones a pose is taken from or applied to: the selection, or all if none is selected."""
    chosen = [bone for bone in ob.pose.bones if _selected(bone)]
    return chosen or list(ob.pose.bones)


def _channels(bone):
    yield "location", 3
    path = _ROTATION.get(bone.rotation_mode, "rotation_euler")
    yield path, 3 if path == "rotation_euler" else 4
    yield "scale", 3


def _library_object(context):
    ob = context.object
    if ob is not None and ob.type == 'ARMATURE':
        return ob
    return None


def _marker(action, name):
    """A pose by name, or the active one when ``name`` is empty."""
    if name:
        return action.pose_markers.get(name)
    return action.pose_markers.active


def _free_frame(action):
    used = {marker.frame for marker in action.pose_markers}
    frame = 1
    while frame in used:
        frame += 1
    return frame


def _clear_frame(action, frame):
    fcurves = _fcurves(action)
    for fcurve in list(fcurves or ()):
        for point in [p for p in fcurve.keyframe_points if round(p.co.x) == frame]:
            fcurve.keyframe_points.remove(point)
        if not len(fcurve.keyframe_points):
            fcurves.remove(fcurve)


def store_pose(ob, action, frame, name):
    """Key the pose of the selected bones (or all) on ``frame`` and name it."""
    fcurves = _fcurves(action, create=True)
    _clear_frame(action, frame)
    for bone in _bones(ob):
        for prop, size in _channels(bone):
            data_path = 'pose.bones["{:s}"].{:s}'.format(bone.name, prop)
            values = getattr(bone, prop)
            for index in range(size):
                fcurve = fcurves.find(data_path, index=index) or _new_fcurve(fcurves, data_path, index, bone.name)
                fcurve.keyframe_points.insert(frame, values[index], options={'FAST'})
                fcurve.update()
    marker = next((m for m in action.pose_markers if m.frame == frame), None)
    if marker is None:
        marker = action.pose_markers.new(name)
        marker.frame = frame
    marker.name = name
    action.pose_markers.active = marker
    return marker


def apply_pose(ob, action, frame):
    """Set bones to the values keyed on ``frame``. Returns how many bones changed."""
    targets = {bone.name for bone in _bones(ob)}
    touched = set()
    for fcurve in _fcurves(action) or ():
        match = _PATH.match(fcurve.data_path)
        if not match or match.group(1) not in targets:
            continue
        bone = ob.pose.bones.get(match.group(1))
        if bone is None or not hasattr(bone, match.group(2)):
            continue
        # Only channels keyed on this very frame belong to the pose.
        point = next((p for p in fcurve.keyframe_points if round(p.co.x) == frame), None)
        if point is None:
            continue
        getattr(bone, match.group(2))[fcurve.array_index] = point.co.y
        touched.add(bone.name)
    return len(touched)


# -----------------------------------------------------------------------------
# Operators

class _PoseLibOperator:
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = _library_object(context)
        return ob is not None and ob.pose_library is not None


class POSELIB_OT_new(Operator):
    """Add New Pose Library to active Object"""
    bl_idname = "poselib.new"
    bl_label = "New Pose Library"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _library_object(context) is not None

    def execute(self, context):
        action = bpy.data.actions.new("PoseLib")
        # Nothing else uses the action, so without this it would be dropped on save.
        action.use_fake_user = True
        context.object.pose_library = action
        return {'FINISHED'}


class POSELIB_OT_unlink(_PoseLibOperator, Operator):
    """Remove Pose Library from active Object"""
    bl_idname = "poselib.unlink"
    bl_label = "Unlink Pose Library"

    def execute(self, context):
        context.object.pose_library = None
        return {'FINISHED'}


class POSELIB_OT_pose_add(_PoseLibOperator, Operator):
    """Add the current Pose to the active Pose Library"""
    bl_idname = "poselib.pose_add"
    bl_label = "PoseLib Add Pose"

    frame: IntProperty(name="Frame", description="Frame to store pose on, 0 picks a free one", default=0, min=0)
    name: StringProperty(name="Pose Name", default="Pose")

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        ob = context.object
        action = ob.pose_library
        store_pose(ob, action, self.frame or _free_frame(action), self.name or "Pose")
        return {'FINISHED'}


class POSELIB_OT_pose_replace(_PoseLibOperator, Operator):
    """Store the current pose over the active pose in the library"""
    bl_idname = "poselib.pose_replace"
    bl_label = "PoseLib Replace Pose"

    def execute(self, context):
        ob = context.object
        marker = ob.pose_library.pose_markers.active
        if marker is None:
            self.report({'ERROR'}, "No pose is chosen in the list")
            return {'CANCELLED'}
        store_pose(ob, ob.pose_library, marker.frame, marker.name)
        return {'FINISHED'}


class POSELIB_OT_pose_remove(_PoseLibOperator, Operator):
    """Remove a pose from the active Pose Library"""
    bl_idname = "poselib.pose_remove"
    bl_label = "PoseLib Remove Pose"

    pose: StringProperty(name="Pose", description="Name of the pose, empty for the one chosen in the list")

    def execute(self, context):
        action = context.object.pose_library
        marker = _marker(action, self.pose)
        if marker is None:
            self.report({'ERROR'}, "No such pose")
            return {'CANCELLED'}
        index = action.pose_markers.active_index
        _clear_frame(action, marker.frame)
        action.pose_markers.remove(marker)
        action.pose_markers.active_index = max(0, min(index, len(action.pose_markers) - 1))
        return {'FINISHED'}


class POSELIB_OT_pose_rename(_PoseLibOperator, Operator):
    """Rename specified pose from the active Pose Library"""
    bl_idname = "poselib.pose_rename"
    bl_label = "PoseLib Rename Pose"

    name: StringProperty(name="New Pose Name", default="RenamedPose")
    pose: StringProperty(name="Pose", description="Name of the pose, empty for the one chosen in the list")

    def invoke(self, context, _event):
        marker = _marker(context.object.pose_library, self.pose)
        if marker is not None:
            self.name = marker.name
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        marker = _marker(context.object.pose_library, self.pose)
        if marker is None:
            self.report({'ERROR'}, "No such pose")
            return {'CANCELLED'}
        marker.name = self.name
        return {'FINISHED'}


class POSELIB_OT_pose_move(_PoseLibOperator, Operator):
    """Move the pose up or down in the active Pose Library"""
    bl_idname = "poselib.pose_move"
    bl_label = "PoseLib Move Pose"

    direction: EnumProperty(name="Direction", items=(('UP', "Up", ""), ('DOWN', "Down", "")))

    def execute(self, context):
        markers = context.object.pose_library.pose_markers
        index = markers.active_index
        other = index + (-1 if self.direction == 'UP' else 1)
        if not (0 <= index < len(markers) and 0 <= other < len(markers)):
            return {'CANCELLED'}
        # The list itself cannot be reordered from Python; trading what two rows hold does the same.
        a, b = markers[index], markers[other]
        a.name, b.name = b.name, a.name
        a.frame, b.frame = b.frame, a.frame
        markers.active_index = other
        return {'FINISHED'}


class POSELIB_OT_apply_pose(_PoseLibOperator, Operator):
    """Apply specified Pose Library pose to the rig"""
    bl_idname = "poselib.apply_pose"
    bl_label = "Apply Pose Library Pose"

    pose_index: IntProperty(
        name="Pose", description="Index of the pose to apply, -1 for the one chosen in the list", default=-1, min=-1,
    )

    def execute(self, context):
        ob = context.object
        markers = ob.pose_library.pose_markers
        index = markers.active_index if self.pose_index < 0 else self.pose_index
        if not 0 <= index < len(markers):
            self.report({'ERROR'}, "No such pose")
            return {'CANCELLED'}
        apply_pose(ob, ob.pose_library, markers[index].frame)
        markers.active_index = index
        return {'FINISHED'}


class POSELIB_OT_browse_interactive(_PoseLibOperator, Operator):
    """Interactively browse poses in 3D-View"""
    bl_idname = "poselib.browse_interactive"
    bl_label = "PoseLib Browse Poses"

    def invoke(self, context, _event):
        ob = context.object
        if not len(ob.pose_library.pose_markers):
            self.report({'WARNING'}, "The pose library is empty")
            return {'CANCELLED'}
        self._before = {bone.name: bone.matrix_basis.copy() for bone in ob.pose.bones}
        self._index = max(0, ob.pose_library.pose_markers.active_index)
        self._show(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _restore(self, context):
        for bone in context.object.pose.bones:
            bone.matrix_basis = self._before[bone.name]

    def _show(self, context):
        ob = context.object
        markers = ob.pose_library.pose_markers
        # Start from the pose as it was, so browsing never piles one pose on another.
        self._restore(context)
        apply_pose(ob, ob.pose_library, markers[self._index].frame)
        context.area.header_text_set(
            "Pose {:d}/{:d}: {:s}   |   Wheel or arrows: browse   Enter or click: keep   Esc: cancel".format(
                self._index + 1, len(markers), markers[self._index].name))
        context.area.tag_redraw()

    def modal(self, context, event):
        count = len(context.object.pose_library.pose_markers)
        if event.value == 'PRESS' and event.type in {'WHEELDOWNMOUSE', 'DOWN_ARROW', 'RIGHT_ARROW'}:
            self._index = (self._index + 1) % count
            self._show(context)
        elif event.value == 'PRESS' and event.type in {'WHEELUPMOUSE', 'UP_ARROW', 'LEFT_ARROW'}:
            self._index = (self._index - 1) % count
            self._show(context)
        elif event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            context.object.pose_library.pose_markers.active_index = self._index
            context.area.header_text_set(None)
            return {'FINISHED'}
        elif event.type in {'ESC', 'RIGHTMOUSE'}:
            self._restore(context)
            context.area.header_text_set(None)
            context.area.tag_redraw()
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}


class POSELIB_OT_action_sanitize(_PoseLibOperator, Operator):
    """Make action suitable for use as a Pose Library"""
    bl_idname = "poselib.action_sanitize"
    bl_label = "Sanitize Pose Library Action"

    def execute(self, context):
        action = context.object.pose_library
        keyed = set()
        for fcurve in _fcurves(action) or ():
            keyed.update(round(point.co.x) for point in fcurve.keyframe_points)
        named = {marker.frame for marker in action.pose_markers}
        # Every keyed frame becomes a pose; a name with no keys behind it goes.
        for frame in sorted(keyed - named):
            action.pose_markers.new("Pose").frame = frame
        for marker in [m for m in action.pose_markers if m.frame not in keyed]:
            action.pose_markers.remove(marker)
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Interface

class POSELIB_MT_classic_add(Menu):
    bl_label = "Add Pose"

    def draw(self, _context):
        layout = self.layout
        layout.operator_context = 'INVOKE_DEFAULT'
        layout.operator("poselib.pose_add", text="Add New")
        layout.operator("poselib.pose_replace", text="Replace Chosen Pose")


def _draw(layout, ob):
    row = layout.row(align=True)
    row.prop(ob, "pose_library", text="")
    row.operator("poselib.new", text="", icon='ADD')
    if ob.pose_library is None:
        return
    row.operator("poselib.unlink", text="", icon='X')
    action = ob.pose_library

    row = layout.row()
    row.template_list("UI_UL_list", "pose_markers", action, "pose_markers", action.pose_markers, "active_index", rows=4)
    col = row.column(align=True)
    col.menu("POSELIB_MT_classic_add", icon='ADD', text="")
    col.operator("poselib.pose_remove", icon='REMOVE', text="")
    col.separator()
    col.operator("poselib.apply_pose", icon='ZOOM_SELECTED', text="")
    col.separator()
    col.operator("poselib.pose_move", icon='TRIA_UP', text="").direction = 'UP'
    col.operator("poselib.pose_move", icon='TRIA_DOWN', text="").direction = 'DOWN'
    col.separator()
    col.operator("poselib.action_sanitize", icon='HELP', text="")

    row = layout.row(align=True)
    row.operator("poselib.apply_pose", text="Apply Pose")
    row.operator("poselib.browse_interactive", text="Browse")
    row.operator("poselib.pose_rename", text="Rename")


class DATA_PT_pose_library_classic(Panel):
    bl_label = "Pose Library"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "data"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return _library_object(context) is not None

    def draw(self, context):
        _draw(self.layout, context.object)


class VIEW3D_PT_pose_library_classic(Panel):
    bl_label = "Pose Library"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Elyan"

    @classmethod
    def poll(cls, context):
        return context.mode == 'POSE' and _library_object(context) is not None

    def draw(self, context):
        _draw(self.layout, context.object)


classes = (
    POSELIB_OT_new,
    POSELIB_OT_unlink,
    POSELIB_OT_pose_add,
    POSELIB_OT_pose_replace,
    POSELIB_OT_pose_remove,
    POSELIB_OT_pose_rename,
    POSELIB_OT_pose_move,
    POSELIB_OT_apply_pose,
    POSELIB_OT_browse_interactive,
    POSELIB_OT_action_sanitize,
    POSELIB_MT_classic_add,
    DATA_PT_pose_library_classic,
    VIEW3D_PT_pose_library_classic,
)

# The shortcuts the pose library had in Pose Mode.
_KEYS = (
    ("poselib.browse_interactive", {"alt": True}),
    ("poselib.pose_add", {"shift": True}),
    ("poselib.pose_remove", {"alt": True, "shift": True}),
    ("poselib.pose_rename", {"ctrl": True, "shift": True}),
)
_keymap_items = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.pose_library = PointerProperty(
        name="Pose Library", description="Action holding this armature's named poses", type=bpy.types.Action,
    )
    # No add-on key configuration when running without a window.
    config = bpy.context.window_manager.keyconfigs.addon
    if config:
        keymap = config.keymaps.new(name="Pose")
        for idname, modifiers in _KEYS:
            _keymap_items.append((keymap, keymap.keymap_items.new(idname, 'L', 'PRESS', **modifiers)))


def unregister():
    for keymap, item in _keymap_items:
        keymap.keymap_items.remove(item)
    _keymap_items.clear()
    del bpy.types.Object.pose_library
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
