# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Marks: the artist points at what is wrong, the assistant reads where that is.

"The hem is too long here" means nothing to a program without the "here". A
mark records it: the object, the mode, the 3D cursor, what was selected (as
indices, a centre and a box in world space), the nearest bone, which way the
view was looking, and her note.

Marks are kept as JSON in a custom property of the scene, so they are saved
with the file, and each one also gets a small Empty in the collection
``Elyan Marks`` so she sees a pin where she put it. The pins never render.
"""

import json
import time

import bpy

from . import commands

COLLECTION = "Elyan Marks"

# Scene custom property holding the list of marks as JSON.
_KEY_MARKS = "elyan_marks"
# Custom property on a pin, tying it to its mark even after the pin is renamed.
_KEY_ID = "elyan_mark_id"

# A whole-mesh selection must not turn the file's custom property into megabytes.
MAX_INDICES = 200
# Blender cuts ID names at 63 bytes; leave room for "Mark.001 " and a ".001" suffix.
_MAX_NOTE_IN_NAME = 40
# Radius of a pin in scene units: readable next to a person-sized model.
_PIN_SIZE = 0.03


# -----------------------------------------------------------------------------
# Storage

def read(scene=None):
    """The marks of ``scene`` (default: the active one), oldest first."""
    scene = scene or bpy.context.scene
    try:
        marks = json.loads(scene.get(_KEY_MARKS) or "[]")
    except (TypeError, ValueError):
        # Hand-edited or damaged; an unreadable list must not break ``state``.
        return []
    return marks if isinstance(marks, list) else []


def _write(scene, marks):
    if marks:
        scene[_KEY_MARKS] = json.dumps(marks)
    elif _KEY_MARKS in scene:
        del scene[_KEY_MARKS]


def count():
    return len(read())


def _pins(scene):
    """``{mark id: pin}`` for the pins that still exist."""
    collection = bpy.data.collections.get(COLLECTION)
    if collection is None:
        return {}
    return {ob[_KEY_ID]: ob for ob in collection.objects if _KEY_ID in ob}


def _collection(scene):
    collection = bpy.data.collections.get(COLLECTION)
    if collection is None:
        collection = bpy.data.collections.new(COLLECTION)
        collection.hide_render = True
    if collection.name not in scene.collection.children:
        scene.collection.children.link(collection)
    return collection


def _add_pin(scene, mark):
    note = " ".join(mark["note"].split())[:_MAX_NOTE_IN_NAME]
    pin = bpy.data.objects.new("Mark.{:03d} {:s}".format(mark["id"], note).strip(), None)
    pin.empty_display_type = 'SPHERE'
    pin.empty_display_size = _PIN_SIZE
    pin.location = mark["point"]
    # A pin inside the body would be invisible, and its name is the note.
    pin.show_in_front = True
    pin.show_name = True
    pin.hide_render = True
    pin[_KEY_ID] = mark["id"]
    _collection(scene).objects.link(pin)
    return pin


# -----------------------------------------------------------------------------
# Reading the context

def _vec(value, digits=5):
    return [round(float(v), digits) for v in value]


def _selected_mesh(ob):
    """What is selected in Edit Mode on a mesh, in world space, or None when nothing is."""
    import numpy as np
    # The mesh datablock lags behind the edit session until this is called.
    ob.update_from_editmode()
    mesh = ob.data
    total = len(mesh.vertices)
    if total == 0:
        return None
    chosen = np.zeros(total, dtype=bool)
    mesh.vertices.foreach_get("select", chosen)
    indices = np.flatnonzero(chosen)
    if len(indices) == 0:
        return None
    coords = np.empty(total * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    matrix = np.array(ob.matrix_world, dtype=np.float64)
    world = coords.reshape(-1, 3)[indices] @ matrix[:3, :3].T + matrix[:3, 3]

    faces = np.zeros(len(mesh.polygons), dtype=bool)
    mesh.polygons.foreach_get("select", faces)
    face_indices = np.flatnonzero(faces)
    return {
        "kind": "faces" if len(face_indices) else "vertices",
        "vertex_count": int(len(indices)),
        "vertices": [int(i) for i in indices[:MAX_INDICES]],
        "face_count": int(len(face_indices)),
        "faces": [int(i) for i in face_indices[:MAX_INDICES]],
        "indices_cut": bool(len(indices) > MAX_INDICES or len(face_indices) > MAX_INDICES),
        "centroid": _vec(world.mean(axis=0)),
        "bbox_min": _vec(world.min(axis=0)),
        "bbox_max": _vec(world.max(axis=0)),
    }


def _bone_selected(pose_bone):
    # Selection moved from the bone to the pose bone in Blender 5.0.
    if hasattr(pose_bone, "select"):
        return bool(pose_bone.select)
    return bool(pose_bone.bone.select)


def _selected_bones(ob):
    """The bones selected in Pose Mode, or None when none are."""
    from mathutils import Vector
    chosen = [pb for pb in ob.pose.bones if _bone_selected(pb)]
    if not chosen:
        return None
    matrix = ob.matrix_world
    middles = [matrix @ ((pb.head + pb.tail) * 0.5) for pb in chosen]
    ends = [matrix @ p for pb in chosen for p in (pb.head, pb.tail)]
    return {
        "kind": "bones",
        "bone_count": len(chosen),
        "bones": [pb.name for pb in chosen[:MAX_INDICES]],
        "indices_cut": len(chosen) > MAX_INDICES,
        "centroid": _vec(sum(middles, Vector()) / len(middles)),
        "bbox_min": _vec([min(p[i] for p in ends) for i in range(3)]),
        "bbox_max": _vec([max(p[i] for p in ends) for i in range(3)]),
    }


def _armature_of(ob):
    if ob is None:
        return None
    if ob.type == 'ARMATURE':
        return ob
    rig = ob.find_armature()
    if rig is None and ob.parent is not None and ob.parent.type == 'ARMATURE':
        rig = ob.parent
    return rig


def nearest_bone(rig, point):
    """The bone of ``rig`` whose posed head-to-tail segment passes closest to a world point."""
    from mathutils import Vector
    from mathutils.geometry import intersect_point_line
    point = Vector(point)
    matrix = rig.matrix_world
    best = None
    for pb in rig.pose.bones:
        head, tail = matrix @ pb.head, matrix @ pb.tail
        if (tail - head).length_squared == 0.0:
            closest = head
        else:
            closest, factor = intersect_point_line(point, head, tail)
            # The nearest point on the endless line can lie beyond either end of the bone.
            closest = head if factor <= 0.0 else tail if factor >= 1.0 else closest
        distance = (point - closest).length
        if best is None or distance < best[1]:
            best = (pb.name, distance)
    if best is None:
        return None
    return {"armature": rig.name, "bone": best[0], "distance": round(best[1], 5)}


def _view():
    """Which way the first 3D viewport is looking, or None without a window."""
    from mathutils import Vector
    if bpy.app.background:
        # A background session still has a default screen; nobody is looking through it.
        return None
    override = commands.view3d_context()
    if override is None:
        return None
    region_3d = override["area"].spaces.active.region_3d
    if region_3d is None:
        return None
    return {
        # Views look down their own -Z axis.
        "direction": _vec(region_3d.view_rotation @ Vector((0.0, 0.0, -1.0))),
        "target": _vec(region_3d.view_location),
        "distance": round(region_3d.view_distance, 5),
        "perspective": region_3d.view_perspective,
    }


def capture(note, author="artist"):
    """
    A mark (not yet stored) describing what the current context points at.

    The point is the middle of the selection in Edit and Pose Mode, and the 3D
    cursor otherwise: placing the cursor is how one points in Object Mode.
    """
    context = bpy.context
    scene = context.scene
    ob = context.view_layer.objects.active
    mode = context.mode
    cursor = _vec(scene.cursor.location)
    mark = {
        "note": note,
        "author": author,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "object": ob.name if ob else None,
        "mode": mode,
        "cursor": cursor,
        "point": cursor,
        "point_from": "cursor",
    }
    selection = None
    if ob is not None and mode == 'EDIT_MESH' and ob.type == 'MESH':
        selection = _selected_mesh(ob)
    elif ob is not None and mode == 'POSE' and ob.type == 'ARMATURE':
        selection = _selected_bones(ob)
    if selection is not None:
        mark["selection"] = selection
        mark["point"] = selection["centroid"]
        mark["point_from"] = "selection"
    view = _view()
    if view is not None:
        mark["view"] = view
    return mark


def add(mark, scene=None):
    """Store a mark, give it an id and a pin. Returns the stored mark."""
    scene = scene or bpy.context.scene
    marks = read(scene)
    mark = dict(mark)
    mark["id"] = max((m.get("id", 0) for m in marks), default=0) + 1
    if "nearest_bone" not in mark:
        rig = _armature_of(bpy.data.objects.get(mark.get("object") or ""))
        near = nearest_bone(rig, mark["point"]) if rig is not None else None
        if near is not None:
            mark["nearest_bone"] = near
    mark["pin"] = _add_pin(scene, mark).name
    marks.append(mark)
    _write(scene, marks)
    return mark


def remove(ids=None, scene=None):
    """Delete the marks with these ids (all of them when None) and their pins. Returns the ids removed."""
    scene = scene or bpy.context.scene
    marks = read(scene)
    gone = [m for m in marks if ids is None or m.get("id") in ids]
    gone_ids = {m.get("id") for m in gone}
    for mark_id, pin in _pins(scene).items():
        # With everything going, pins whose mark was lost go too.
        if ids is None or mark_id in gone_ids:
            bpy.data.objects.remove(pin, do_unlink=True)
    _write(scene, [m for m in marks if m.get("id") not in gone_ids])
    collection = bpy.data.collections.get(COLLECTION)
    if collection is not None and not collection.objects and not read(scene):
        # An empty collection left in her outliner is only clutter.
        bpy.data.collections.remove(collection)
    return sorted(gone_ids)


def listing(scene=None):
    """The marks, each with where its pin is now: she may have dragged it to a better spot."""
    scene = scene or bpy.context.scene
    pins = _pins(scene)
    result = []
    for mark in read(scene):
        mark = dict(mark)
        pin = pins.get(mark.get("id"))
        if pin is None:
            mark["pin"] = None
        else:
            mark["pin"] = pin.name
            # ``matrix_world`` is only current after an update; pins have no parent unless she gave one.
            where = _vec(pin.location if pin.parent is None else pin.matrix_world.translation)
            if where != mark.get("point"):
                mark["pin_moved_to"] = where
        result.append(mark)
    return result


# -----------------------------------------------------------------------------
# Command

def _point(value, what):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)):
        raise ValueError("{:s} must be three numbers [x, y, z] in world space, got {!r}".format(what, value))
    return _vec(value)


def cmd_marks(args):
    """
    The spots the artist has pointed at, each with her note. Read these before changing anything.

    Without arguments: ``{"marks": [...]}``, oldest first. Each mark has ``id``,
    ``note``, ``author`` ("artist" or "agent"), ``time``, ``object`` (the active
    object), ``mode``, ``cursor`` and ``point`` (world space; ``point_from`` says
    whether the point is the middle of the selection or the 3D cursor),
    ``nearest_bone`` ``{armature, bone, distance}`` when the object is rigged,
    ``view`` ``{direction, target, distance, perspective}`` when a viewport was
    open, and ``selection``: in Edit Mode ``{kind, vertex_count, vertices,
    face_count, faces, centroid, bbox_min, bbox_max}`` (indices of the base
    mesh, at most 200 of each, ``indices_cut`` says when there were more), in
    Pose Mode ``{kind: "bones", bone_count, bones, centroid, bbox_min, bbox_max}``.
    ``pin`` is the Empty that shows the mark in the viewport (collection
    "Elyan Marks", never rendered); ``pin_moved_to`` appears when she has
    dragged it since.

    ``remove=ID`` (or a list of ids) deletes those marks and their pins;
    ``clear=true`` deletes all. Remove a mark once what it asked for is done
    and she has kept the change, not before.

    ``add=true note="..."`` leaves a mark of your own, e.g. to show her where
    you changed something. With ``point=[x, y, z]`` it goes there, and
    ``object`` (name) and ``bone`` may be given; without ``point`` it is taken
    from the current selection or 3D cursor exactly as her button does.
    """
    scene = bpy.context.scene
    response = {}
    if args.get("clear"):
        response["removed"] = remove(None, scene)
    elif args.get("remove") is not None:
        wanted = args["remove"] if isinstance(args["remove"], (list, tuple)) else [args["remove"]]
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in wanted):
            raise ValueError("remove takes a mark id (a whole number) or a list of them, got {!r}".format(
                args["remove"]))
        known = {m.get("id") for m in read(scene)}
        missing = sorted(set(wanted) - known)
        if missing:
            raise ValueError("no mark with id {:s}; there are: {:s}".format(
                ", ".join(str(v) for v in missing), ", ".join(str(v) for v in sorted(known)) or "none"))
        response["removed"] = remove(set(wanted), scene)
    if args.get("add"):
        note = args.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError("add needs 'note', the text shown beside the pin")
        mark = capture(note.strip(), author=str(args.get("author") or "agent"))
        if args.get("object") is not None:
            if args["object"] not in bpy.data.objects:
                raise ValueError("no object named {!r}".format(args["object"]))
            mark["object"] = args["object"]
        if args.get("point") is not None:
            mark["point"] = _point(args["point"], "point")
            mark["point_from"] = "given"
            # The selection belongs to wherever the context was, not to this point.
            mark.pop("selection", None)
        if args.get("bone") is not None:
            rig = _armature_of(bpy.data.objects.get(mark.get("object") or ""))
            if rig is None or args["bone"] not in rig.pose.bones:
                raise ValueError("no bone named {!r} on the rig of {!r}".format(args["bone"], mark.get("object")))
            mark["nearest_bone"] = {"armature": rig.name, "bone": args["bone"], "distance": None}
        response["added"] = add(mark, scene)
    response["marks"] = listing(scene)
    return response


COMMANDS = {"marks": cmd_marks}

STATE = {"marks": count}
