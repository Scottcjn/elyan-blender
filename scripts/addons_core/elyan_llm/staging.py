# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Proposals: a change the assistant makes is held for the artist to keep or undo.

An assistant that edits a model while its owner watches must be refusable.
``propose`` takes a checkpoint of the collection it is about to change, makes
the change, and leaves a record saying what it did in plain words. The panel
then shows that note with two buttons. Keep throws the checkpoint away; Undo
rolls the collection back to it.

Only the artist keeps a change: there is deliberately no bridge command for
it. The assistant may withdraw its own proposal with ``discard``.

The checkpoint machinery is the one in ``commands``; nothing is copied here.
The record lives in a custom property of the scene, so Blender's own undo
takes the record and the checkpoint back together.
"""

import json
import time

import bpy

from . import commands

# The one checkpoint a proposal owns. A leading underscore keeps it apart from names an agent picks.
CHECKPOINT = "_pending"

# Scene custom property holding the pending record as JSON.
_KEY_PENDING = "elyan_pending"

# The panel is narrow; a rebuild touching fifty objects must not fill it.
MAX_LINES = 12


# -----------------------------------------------------------------------------
# Record

def pending(scene=None):
    """The pending record, or None."""
    scene = scene or bpy.context.scene
    try:
        record = json.loads(scene.get(_KEY_PENDING) or "null")
    except (TypeError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    # A file saved elsewhere, or a checkpoint deleted by hand: say so rather than fail on Undo.
    record["can_undo"] = _store() is not None
    return record


def _clear(scene):
    if _KEY_PENDING in scene:
        del scene[_KEY_PENDING]


def _store():
    from . import checks
    return bpy.data.collections.get(checks.CHECKPOINT_PREFIX + CHECKPOINT)


# -----------------------------------------------------------------------------
# Saying what changed

def _is_pin(name):
    from . import marks
    ob = bpy.data.objects.get(name)
    return ob is not None and any(c.name == marks.COLLECTION for c in ob.users_collection)


def _moved_vertices(name):
    """How many base-mesh vertices of ``name`` differ from its checkpoint copy; None if not comparable."""
    import numpy as np
    store = _store()
    ob = bpy.data.objects.get(name)
    if store is None or ob is None or ob.type != 'MESH':
        return None
    kept = next((copy for copy in store.objects if copy.get(commands._KEY_SOURCE) == name), None)
    if kept is None or kept.type != 'MESH' or len(kept.data.vertices) != len(ob.data.vertices):
        return None
    count = len(ob.data.vertices)
    now, then = np.empty(count * 3, dtype=np.float32), np.empty(count * 3, dtype=np.float32)
    ob.data.vertices.foreach_get("co", now)
    kept.data.vertices.foreach_get("co", then)
    return int(np.any(now.reshape(-1, 3) != then.reshape(-1, 3), axis=1).sum())


def _describe(name, fields):
    """One object's changes in words an artist uses."""
    parts = []
    if "verts" in fields:
        parts.append("{:d} vertices became {:d}".format(fields["verts"]["before"], fields["verts"]["after"]))
    elif "geometry" in fields:
        moved = _moved_vertices(name)
        # Zero moved with a different result means a modifier or a shape key did it.
        parts.append("{:d} vertices moved".format(moved) if moved else "shape changed")
    elif "tris" in fields:
        parts.append("{:d} triangles became {:d}".format(fields["tris"]["before"], fields["tris"]["after"]))
    if "transform" in fields:
        parts.append("moved, turned or resized")
    if "materials" in fields:
        parts.append("materials changed")
    if "modifiers" in fields:
        parts.append("modifiers changed")
    return "{:s}: {:s}".format(name, ", ".join(parts) or "changed")


def summarize(diff):
    """A scene diff (see ``checks.diff_states``) as short lines for the panel."""
    lines = [_describe(name, fields) for name, fields in diff.get("changed", {}).items()]
    lines += ["{:s}: new".format(name) for name in diff.get("added", ())]
    lines += ["{:s}: deleted".format(name) for name in diff.get("removed", ())]
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES] + ["and {:d} more".format(len(lines) - MAX_LINES)]
    return lines


def _touched(diff):
    return [*diff.get("changed", {}), *diff.get("added", ()), *diff.get("removed", ())]


# -----------------------------------------------------------------------------
# Keep and Undo. Called by the panel's operators, and by tests.

def keep(scene=None):
    """The artist accepts the change: the checkpoint goes, the change stays."""
    scene = scene or bpy.context.scene
    record = pending(scene)
    if record is None:
        raise RuntimeError("nothing is waiting to be kept")
    if _store() is not None:
        commands.cmd_checkpoints({"delete": CHECKPOINT})
    _clear(scene)
    return record


def undo(scene=None):
    """The change is refused: the collection goes back to how it was, the record goes."""
    scene = scene or bpy.context.scene
    record = pending(scene)
    if record is None:
        raise RuntimeError("nothing is waiting to be undone")
    if _store() is None:
        raise RuntimeError(
            "the copy kept for undoing {!r} is gone from this file, so it cannot be undone here".format(
                record.get("note")))
    rolled = commands.cmd_rollback({"name": CHECKPOINT})
    commands.cmd_checkpoints({"delete": CHECKPOINT})
    _clear(scene)
    return {"proposal": record, "restored": rolled["restored"], "removed": rolled["removed"]}


# -----------------------------------------------------------------------------
# Commands

def cmd_propose(args):
    """
    Make a change the artist can look at and then Keep or Undo. Use this instead
    of ``exec`` or ``rebuild`` whenever a person has the file open in a window.

    ``collection``: the collection the change is confined to (its objects,
    child collections included, are copied first). ``note``: one or two plain
    sentences for the artist saying what you changed and why; she reads this in
    the panel, so no code and no jargon. Then either ``script`` (path of a
    builder, run as ``rebuild`` runs it; ``argv`` and ``root`` are passed on) or
    ``code`` (Python, run as ``exec`` runs it). ``sheet``: path of a contact
    sheet you rendered of the result, shown to her as a file name.

    Returns ``pending`` (the record: note, time, collection, ``summary`` lines,
    ``outside`` = objects changed that are NOT in the collection and that Undo
    will therefore not restore) and ``diff``. Only one proposal can be open:
    while one is waiting, ``propose`` is refused until she presses Keep or
    Undo, or you ``discard`` it. If the change raises an error the collection
    is rolled back at once and nothing is left pending. A change that alters
    nothing leaves nothing pending either.

    There is no command to keep a proposal. That is hers alone to press.
    Needs Object Mode.
    """
    scene = bpy.context.scene
    waiting = pending(scene)
    if waiting is not None:
        raise RuntimeError(
            "a proposal is already waiting for the artist ({!r}, {:s}); she presses Keep or Undo in the "
            "panel, or you withdraw it with 'discard'".format(waiting.get("note"), waiting.get("time", "")))
    note = args.get("note")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("propose needs 'note': plain words for the artist about what is changing")
    script, code = args.get("script"), args.get("code")
    if bool(script) == bool(code):
        raise ValueError("propose needs exactly one of 'script' (a builder's path) or 'code' (Python)")
    name = args.get("collection")
    if not isinstance(name, str) or name not in bpy.data.collections:
        raise ValueError("propose needs 'collection', the name of an existing collection; there are: {:s}".format(
            ", ".join(c.name for c in scene.collection.children) or "none"))

    # A checkpoint without a record is left over from a crash or a hand edit; it is nobody's.
    commands.cmd_checkpoint({"name": CHECKPOINT, "collection": name, "replace": True})
    members = {ob.name for ob in bpy.data.collections[name].all_objects}
    before = commands._state_before({"diff": True})
    label = "proposal: " + note.strip()[:48]
    if script:
        ran = commands.cmd_rebuild({"path": script, "argv": args.get("argv"), "root": args.get("root")})
    else:
        ran = commands.cmd_exec({"code": code, "label": label})
    response = {key: ran[key] for key in ("result", "stdout", "error", "generation", "mode") if key in ran}

    if ran.get("ok") is False:
        # Half a change is not something to ask her about.
        response["ok"] = False
        response["pending"] = None
        try:
            commands.cmd_rollback({"name": CHECKPOINT})
            commands.cmd_checkpoints({"delete": CHECKPOINT})
            response["rolled_back"] = True
        except Exception as ex:
            response["rolled_back"] = False
            response["error"] += "\nand rolling back failed: {!s}".format(ex)
        return response

    commands._state_after(before, response)
    if "diff" not in response:
        # Without a diff she would be asked to keep something nobody can describe.
        commands.cmd_rollback({"name": CHECKPOINT})
        commands.cmd_checkpoints({"delete": CHECKPOINT})
        raise RuntimeError("the change ran but could not be described, so it was rolled back:\n{:s}".format(
            response.get("diff_error", "")))
    diff = response["diff"]
    # Her pins are not part of the model.
    for key in ("added", "removed"):
        diff[key] = [ob for ob in diff[key] if not _is_pin(ob)]
    if not _touched(diff):
        commands.cmd_checkpoints({"delete": CHECKPOINT})
        response["pending"] = None
        response["nothing_changed"] = True
        return response

    now = {ob.name for ob in bpy.data.collections[name].all_objects} if name in bpy.data.collections else set()
    record = {
        "note": note.strip(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "collection": name,
        "summary": summarize(diff),
        "outside": sorted(ob for ob in _touched(diff) if ob not in members and ob not in now),
    }
    if args.get("sheet"):
        record["sheet"] = str(args["sheet"])
    scene[_KEY_PENDING] = json.dumps(record)
    # The record was written after the change's own undo step; without a step of
    # its own, her first Ctrl+Z would seem to do nothing.
    commands._undo_push("LLM " + label)
    response["pending"] = pending(scene)
    return response


def cmd_pending(args):
    """
    The proposal waiting for the artist, as ``{"pending": record}``, or
    ``{"pending": null}`` once she has pressed Keep or Undo (which one is not
    recorded: look at the scene). ``can_undo`` false means the copy kept for
    Undo is no longer in the file.
    """
    return {"pending": pending()}


def cmd_discard(args):
    """
    Withdraw your own pending proposal: the collection is rolled back to how
    it was before ``propose`` and the record is cleared, as if she had pressed
    Undo. Needs Object Mode.
    """
    return {"discarded": undo(), "pending": None}


COMMANDS = {
    "propose": cmd_propose,
    "pending": cmd_pending,
    "discard": cmd_discard,
}

STATE = {"pending": pending}
