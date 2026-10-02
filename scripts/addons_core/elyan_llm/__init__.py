# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
LLM bridge: lets a local assistant drive this Blender directly.

A token-protected HTTP server on localhost runs Python on the main thread and
answers with scene summaries, API look-ups and renders. See ``client.py``.
"""

bl_info = {
    "name": "Elyan LLM Bridge",
    "description": "Let a local LLM assistant inspect, script and render this Blender session",
    "author": "Elyan Labs",
    "version": (0, 1),
    "blender": (4, 2, 0),
    "location": "3D Viewport -> Sidebar -> Elyan",
    "support": "OFFICIAL",
    "category": "Development",
}

import os
import textwrap

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, IntProperty, StringProperty
from bpy.types import AddonPreferences, Operator, Panel

from . import commands, marks, selftest, server, staging

# Sidebar labels do not wrap; notes are cut into lines of about this many characters.
WRAP = 38

# Starts the bridge whatever the preference says, e.g. ``ELYAN_LLM_BRIDGE=1 blender``.
ENV_AUTOSTART = "ELYAN_LLM_BRIDGE"


def serve(port=0):
    """Blocking loop for headless sessions, see ``server.serve``."""
    server.serve(port)


def _pref(context, name, default):
    # Absent when enabled for one session only (``--addons``), fall back to the default.
    addon = context.preferences.addons.get(__package__)
    return getattr(addon.preferences, name) if addon else default


class ElyanLLMPreferences(AddonPreferences):
    bl_idname = __package__

    autostart: BoolProperty(
        name="Start With Blender",
        description="Start listening as soon as Blender opens",
        default=False,
    )
    port: IntProperty(
        name="Port",
        description="Localhost port to listen on, 0 picks a free one",
        default=0, min=0, max=65535,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "autostart")
        layout.prop(self, "port")
        layout.label(text="Anything that can read your session file can run Python here.", icon='ERROR')


class ELYAN_OT_llm_start(Operator):
    bl_idname = "elyan.llm_start"
    bl_label = "Start LLM Bridge"
    bl_description = "Let a local assistant work in this Blender session"

    def execute(self, context):
        try:
            port = server.start(_pref(context, "port", 0))
        except OSError as ex:
            self.report({'ERROR'}, "Could not listen: {!s}".format(ex))
            return {'CANCELLED'}
        self.report({'INFO'}, "LLM bridge listening on 127.0.0.1:{:d}".format(port))
        return {'FINISHED'}


class ELYAN_OT_llm_stop(Operator):
    bl_idname = "elyan.llm_stop"
    bl_label = "Stop LLM Bridge"
    bl_description = "Stop listening; the assistant can no longer reach this session"

    def execute(self, context):
        server.stop()
        return {'FINISHED'}


def _object_mode():
    # Checkpoints copy objects, which misses edits still held by Edit Mode.
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def _wrapped(layout, text, icon='NONE'):
    for index, line in enumerate(textwrap.wrap(text, WRAP) or [""]):
        layout.label(text=line, icon=icon if index == 0 else 'NONE')


class ELYAN_OT_llm_selftest(Operator):
    bl_idname = "elyan.llm_selftest"
    bl_label = "Test LLM Bridge"
    bl_description = (
        "Check that the assistant can reach this window, that its changes can be undone "
        "and that it can take pictures. Takes a few seconds and leaves the scene as it is"
    )

    def execute(self, context):
        try:
            selftest.start(_pref(context, "port", 0))
        except (RuntimeError, OSError) as ex:
            self.report({'WARNING'}, str(ex))
            return {'CANCELLED'}
        return {'FINISHED'}


class ELYAN_OT_llm_mark(Operator):
    bl_idname = "elyan.llm_mark"
    bl_label = "Mark This"
    bl_description = (
        "Leave a pin with your note for the assistant: at the selection in Edit or Pose Mode, "
        "at the 3D cursor otherwise"
    )
    bl_options = {'REGISTER', 'UNDO'}

    note: StringProperty(
        name="Note",
        description="What is wrong here, in your own words",
    )

    def execute(self, context):
        wm = context.window_manager
        # The panel's text field is used unless a script passes the note itself.
        note = self.note.strip() or wm.elyan_llm_mark_note.strip()
        mark = marks.add(marks.capture(note))
        wm.elyan_llm_mark_note = ""
        self.report({'INFO'}, "Marked: {:s}".format(mark["pin"]))
        return {'FINISHED'}


class ELYAN_OT_llm_mark_remove(Operator):
    bl_idname = "elyan.llm_mark_remove"
    bl_label = "Remove Mark"
    bl_description = "Remove this mark and its pin"
    bl_options = {'REGISTER', 'UNDO'}

    id: IntProperty(name="Mark", min=0)

    def execute(self, context):
        if not marks.remove({self.id}):
            return {'CANCELLED'}
        return {'FINISHED'}


class ELYAN_OT_llm_keep(Operator):
    bl_idname = "elyan.llm_keep"
    bl_label = "Keep"
    bl_description = "Keep what the assistant changed"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return staging.pending(context.scene) is not None

    def execute(self, context):
        try:
            _object_mode()
            staging.keep(context.scene)
        except (RuntimeError, ValueError) as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        return {'FINISHED'}


class ELYAN_OT_llm_undo_change(Operator):
    bl_idname = "elyan.llm_undo_change"
    bl_label = "Undo"
    bl_description = "Put everything back as it was before the assistant's change"
    # Rolling back records its own undo step, see ``commands.cmd_rollback``.
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return staging.pending(context.scene) is not None

    def execute(self, context):
        try:
            _object_mode()
            staging.undo(context.scene)
        except (RuntimeError, ValueError) as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        return {'FINISHED'}


class ELYAN_PT_llm(Panel):
    bl_label = "LLM Bridge"
    bl_idname = "ELYAN_PT_llm"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Elyan"

    def draw(self, context):
        layout = self.layout
        if server.is_running():
            layout.label(text="Listening on port {:d}".format(server.port()), icon='LINKED')
            layout.operator("elyan.llm_stop", icon='CANCEL')
        else:
            layout.label(text="Not listening", icon='UNLINKED')
            layout.operator("elyan.llm_start", icon='PLAY')
        self.draw_pending(context)
        if server.log:
            col = layout.column(align=True)
            col.label(text="Recent requests:")
            for clock, cmd, summary, ok in list(server.log)[-6:]:
                col.label(
                    text="{:s}  {:s}  {:s}".format(clock, cmd, summary),
                    icon='CHECKMARK' if ok else 'ERROR',
                )
        self.draw_selftest(context)

    def draw_pending(self, context):
        record = staging.pending(context.scene)
        if record is None:
            return
        box = self.layout.box()
        col = box.column(align=True)
        col.label(text="The assistant changed something:", icon='INFO')
        _wrapped(col, record.get("note", ""))
        col = box.column(align=True)
        for line in record.get("summary", ()):
            _wrapped(col, line, icon='DOT')
        if record.get("outside"):
            col = box.column(align=True)
            _wrapped(col, "Undo will not put these back: {:s}".format(", ".join(record["outside"])), icon='ERROR')
        if record.get("sheet"):
            box.label(text="Pictures: {:s}".format(os.path.basename(record["sheet"])), icon='IMAGE_DATA')
        row = box.row(align=True)
        row.scale_y = 1.4
        row.operator("elyan.llm_keep", icon='CHECKMARK')
        undo = row.row(align=True)
        undo.enabled = bool(record.get("can_undo"))
        undo.operator("elyan.llm_undo_change", icon='LOOP_BACK')

    def draw_selftest(self, context):
        layout = self.layout
        active = selftest.running()
        if active is not None:
            label, steps = active
            col = layout.column(align=True)
            col.label(text="Testing: {:s}".format(label or "finishing"), icon='TIME')
        else:
            layout.operator("elyan.llm_selftest", icon='CHECKBOX_HLT')
            result = selftest.last()
            if result is None:
                layout.label(text="Not tested in a window yet")
                return
            steps = result.get("steps", ())
            col = layout.column(align=True)
            col.label(
                text="{:s} on {:s}".format(
                    "Test passed" if result.get("passed") else "Test failed", str(result.get("date", ""))[:10]),
                icon='CHECKMARK' if result.get("passed") else 'ERROR',
            )
        for step in steps:
            col.label(text=str(step.get("label", "")), icon='CHECKMARK' if step.get("ok") else 'ERROR')
            if not step.get("ok") and step.get("error"):
                _wrapped(col, str(step["error"]), icon='BLANK1')


class ELYAN_PT_llm_marks(Panel):
    bl_label = "Marks"
    bl_parent_id = "ELYAN_PT_llm"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Elyan"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.prop(context.window_manager, "elyan_llm_mark_note", text="")
        col.operator("elyan.llm_mark", icon='PINNED')
        col = layout.column(align=True)
        for mark in marks.read(context.scene):
            row = col.row(align=True)
            text = "{:d}  {:s}".format(mark.get("id", 0), mark.get("note") or mark.get("object") or "")
            row.label(text=text[:WRAP], icon='PINNED' if mark.get("author") == "artist" else 'DOT')
            row.operator("elyan.llm_mark_remove", text="", icon='X').id = mark.get("id", 0)


@persistent
def _on_load_post(_filepath):
    commands.invalidate()
    # The session file names the open .blend, keep it current.
    server.write_session()


@persistent
def _on_undo_redo(_scene):
    commands.invalidate()


_HANDLERS = (
    ("load_post", _on_load_post),
    ("undo_post", _on_undo_redo),
    ("redo_post", _on_undo_redo),
)


classes = (
    ElyanLLMPreferences,
    ELYAN_OT_llm_start,
    ELYAN_OT_llm_stop,
    ELYAN_OT_llm_selftest,
    ELYAN_OT_llm_mark,
    ELYAN_OT_llm_mark_remove,
    ELYAN_OT_llm_keep,
    ELYAN_OT_llm_undo_change,
    ELYAN_PT_llm,
    ELYAN_PT_llm_marks,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    # On the window manager, not the scene: a half-typed note is not part of the file.
    bpy.types.WindowManager.elyan_llm_mark_note = StringProperty(
        name="Note",
        description="What is wrong at the spot you are marking, in your own words",
    )
    for name, handler in _HANDLERS:
        getattr(bpy.app.handlers, name).append(handler)

    wanted = _pref(bpy.context, "autostart", False) or os.environ.get(ENV_AUTOSTART, "") not in {"", "0"}
    # Headless sessions have no event loop for the timer, they call ``serve()`` instead.
    if wanted and not bpy.app.background:
        try:
            server.start(_pref(bpy.context, "port", 0))
        except OSError as ex:
            print("elyan_llm: could not listen:", ex)


def unregister():
    selftest.cancel()
    server.stop()
    for name, handler in _HANDLERS:
        getattr(bpy.app.handlers, name).remove(handler)
    del bpy.types.WindowManager.elyan_llm_mark_note
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
