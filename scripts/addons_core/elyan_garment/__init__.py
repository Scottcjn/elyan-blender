# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
One-call tools for clothes on an avatar body: weights, weld, fit, shape keys.

A finished VRChat outfit needs the same four things every time: the body's
skinning on the garment, seams that meet within the contract's tolerance,
nothing poking through, and the body's sliders carried onto the cloth. Blender
can do each, by hand, in several steps that depend on selection and mode.
Here each is one function on named objects that answers with numbers and
refuses, leaving the mesh alone, when its own measure says the result is bad.

    import elyan_garment
    elyan_garment.weights("Bodice", "Body")
    elyan_garment.weld("Sleeve", ARMHOLE_RING)
    elyan_garment.fit("Bodice", "Body", ease=0.006, pin="neckline")
    elyan_garment.shapekeys("Bodice", "Body")

The buttons use the active object as the garment and the other selected mesh
as the body. With the Elyan LLM Bridge present the same four are the commands
``garment_weights``, ``garment_weld``, ``garment_fit`` and ``garment_shapekeys``.
Everything is measured unposed: the body as shown, the garment as stored.
"""

bl_info = {
    "name": "Elyan Garment",
    "description": "Skin, weld, fit and shape-key a garment to an avatar body, each in one call with a numeric report",
    "author": "Elyan Labs",
    "version": (0, 1),
    "blender": (4, 2, 0),
    "location": "3D Viewport -> Sidebar -> Elyan -> Garment",
    "support": "OFFICIAL",
    "category": "Object",
}

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import Operator, Panel

from . import bridge
from .ease import fit
from .follow import shapekeys
from .seam import weld
from .skin import weights

__all__ = ("weights", "weld", "fit", "shapekeys")

# What the last button press measured, for the panel.
_last = []


def _pair(context):
    """The garment (active) and the one other selected mesh, or None for either."""
    garment = context.object
    if garment is None or garment.type != 'MESH' or not garment.select_get():
        return None, None
    others = [ob for ob in context.selected_objects if ob != garment and ob.type == 'MESH']
    return garment, others[0] if len(others) == 1 else None


def _lists(text):
    """A comma-separated text field as a list of names, or None when empty."""
    found = [part.strip() for part in text.split(",") if part.strip()]
    return found or None


class _GarmentOperator:
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        garment, other = _pair(context)
        return garment is not None and other is not None and context.mode == 'OBJECT'

    def run(self, garment, other):
        raise NotImplementedError

    def lines(self, report):
        raise NotImplementedError

    def execute(self, context):
        garment, other = _pair(context)
        try:
            report = self.run(garment, other)
        except (ValueError, RuntimeError) as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        _last[:] = ["{:s}: {:s}".format(self.bl_label, garment.name)]
        if report.get("refused"):
            _last.append("Refused, nothing changed:")
            _last.append(report["refused"])
            self.report({'ERROR'}, "Refused, nothing changed: {:s}".format(report["refused"]))
            return {'CANCELLED'}
        _last.extend(self.lines(report))
        self.report({'INFO'}, "; ".join(_last[1:]))
        return {'FINISHED'}


class ELYAN_OT_garment_weights(_GarmentOperator, Operator):
    """Skin the active garment like the other selected mesh: its armature and its bone weights"""
    bl_idname = "elyan_garment.weights"
    bl_label = "Weights from Body"

    limit: IntProperty(name="Influences", description="Bones per vertex", default=4, min=1, max=8)
    same_side: BoolProperty(
        name="Same Side Only",
        description="Never take weights from the far half of the body: for skirts and anything between the legs",
        default=False,
    )
    smooth: IntProperty(
        name="Smooth",
        description="Passes blending weights between neighbours where the cloth stands off the body. "
                    "-1: only when the garment would otherwise shear",
        default=-1, min=-1, max=64)
    region: StringProperty(
        name="From Bones",
        description="Body vertex groups to take weights from, separated by commas; empty for the whole body",
    )

    def run(self, garment, other):
        return weights(garment, other, limit=self.limit, region=_lists(self.region), same_side=self.same_side,
                       smooth=None if self.smooth < 0 else self.smooth)

    def lines(self, report):
        found = [
            "{:d} bones, {:d} vertices clamped to {:d}".format(report["bone_count"], report["clamped"], report["limit"]),
            "smoothed {:d} passes, largest step {:.2f}".format(report["smooth"], report["weight_step"]),
            "farthest from the body: {:.1f} mm".format(report["max_distance"] * 1000.0),
        ]
        if report["wrong_side"]:
            found.append("{:d} take weights from the other side".format(report["wrong_side"]))
        if report["weight_step"] > 0.5:
            found.append("weights still jump between neighbours: more Smooth")
        return found


class ELYAN_OT_garment_weld(_GarmentOperator, Operator):
    """Move the open edge of the active garment onto the nearest open edge of the other selected mesh"""
    bl_idname = "elyan_garment.weld"
    bl_label = "Weld Edge to Edge"

    tolerance: FloatProperty(name="Tolerance", default=1e-5, min=0.0, precision=6, unit='LENGTH')
    rings: IntProperty(
        name="Ease Rows", description="Rows of vertices behind the edge that follow, to avoid a crease",
        default=2, min=0, max=12)
    max_move: FloatProperty(name="Farthest Move", default=0.05, min=0.0, unit='LENGTH')

    def run(self, garment, other):
        return weld(garment, other, tolerance=self.tolerance, rings=self.rings, max_move=self.max_move)

    def lines(self, report):
        return [
            "loop {:d} ({:d} vertices), {:s}".format(
                report["loop"]["index"], report["loop"]["verts"], report["loop"]["matched"]),
            "gap {:.3g} -> {:.3g} m".format(report["before"]["max"], report["after"]["max"]),
        ]


class ELYAN_OT_garment_fit(_GarmentOperator, Operator):
    """Push the parts of the active garment that touch or enter the other selected mesh out to the ease"""
    bl_idname = "elyan_garment.fit"
    bl_label = "Fit to Body"

    ease: FloatProperty(
        name="Ease", description="Clearance to leave between the body and the garment",
        default=0.006, min=0.0, precision=4, unit='LENGTH')
    max_move: FloatProperty(
        name="Farthest Move", description="No vertex moves farther than this", default=0.03, min=0.0001,
        precision=4, unit='LENGTH')
    pin: StringProperty(
        name="Pin", description="Garment vertex groups that must not move, separated by commas")
    groups: StringProperty(
        name="Only", description="Garment vertex groups to fit, separated by commas; empty for the whole garment")
    max_fraction: FloatProperty(
        name="Allow Unresolved", description="Share of offending vertices that may stay too close",
        default=0.0, min=0.0, max=1.0, subtype='FACTOR')

    faces: BoolProperty(
        name="Lift Faces",
        description="Also lift faces the body shows through although their corners are clear",
        default=False,
    )

    def run(self, garment, other):
        return fit(garment, other, ease=self.ease, max_move=self.max_move, pin=_lists(self.pin),
                   groups=_lists(self.groups), max_fraction=self.max_fraction, faces=self.faces)

    def lines(self, report):
        return [
            "{:d} vertices moved, at most {:.1f} mm".format(report["moved"], report["largest_move"] * 1000.0),
            "closest {:.1f} -> {:.1f} mm".format(report["before"]["min"] * 1000.0, report["after"]["min"] * 1000.0),
            "inside the body {:d} -> {:d}".format(report["before"]["inside"], report["after"]["inside"]),
            "body showing through faces: {:d} -> {:d} vertices".format(
                report["shows_through"]["before"]["body_verts"], report["shows_through"]["after"]["body_verts"]),
        ]


class ELYAN_OT_garment_shapekeys(_GarmentOperator, Operator):
    """Give the active garment shape keys that follow those of the other selected mesh"""
    bl_idname = "elyan_garment.shapekeys"
    bl_label = "Shape Keys from Body"

    distance: FloatProperty(
        name="Reach", description="Cloth farther than this from the body does not follow it",
        default=0.03, min=0.0001, precision=4, unit='LENGTH')
    max_error: FloatProperty(
        name="Allowed Loss", description="A key is not written if it brings the body this much closer than at rest",
        default=0.004, min=0.0, precision=4, unit='LENGTH')

    def run(self, garment, other):
        return shapekeys(garment, other, distance=self.distance, max_error=self.max_error)

    def lines(self, report):
        found = ["{:d} written, {:d} skipped, {:d} refused".format(
            len(report["written"]), len(report["skipped"]), len(report["refused"]))]
        if report["refused"]:
            found.append("refused: {:s}".format(", ".join(report["refused"][:6])))
        return found


class ELYAN_PT_garment(Panel):
    bl_label = "Garment"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Elyan"

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def draw(self, context):
        layout = self.layout
        garment, other = _pair(context)
        col = layout.column(align=True)
        if garment is None or other is None:
            col.label(text="Select the body, then the garment", icon='INFO')
        else:
            col.label(text="Garment: {:s}".format(garment.name), icon='MOD_CLOTH')
            col.label(text="Body: {:s}".format(other.name), icon='OUTLINER_OB_MESH')
        col = layout.column(align=True)
        col.operator("elyan_garment.weights", icon='MOD_VERTEX_WEIGHT')
        col.operator("elyan_garment.weld", icon='AUTOMERGE_ON')
        col.operator("elyan_garment.fit", icon='MOD_SHRINKWRAP')
        col.operator("elyan_garment.shapekeys", icon='SHAPEKEY_DATA')
        if _last:
            box = layout.box()
            for line in _last[:6]:
                box.label(text=line)


classes = (
    ELYAN_OT_garment_weights,
    ELYAN_OT_garment_weld,
    ELYAN_OT_garment_fit,
    ELYAN_OT_garment_shapekeys,
    ELYAN_PT_garment,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bridge.install()


def unregister():
    bridge.remove()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
