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

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, IntProperty
from bpy.types import AddonPreferences, Operator, Panel

from . import commands, server

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


class ELYAN_PT_llm(Panel):
    bl_label = "LLM Bridge"
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
        if server.log:
            col = layout.column(align=True)
            col.label(text="Recent requests:")
            for clock, cmd, summary, ok in list(server.log)[-6:]:
                col.label(
                    text="{:s}  {:s}  {:s}".format(clock, cmd, summary),
                    icon='CHECKMARK' if ok else 'ERROR',
                )


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
    ELYAN_PT_llm,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
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
    server.stop()
    for name, handler in _HANDLERS:
        getattr(bpy.app.handlers, name).remove(handler)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
