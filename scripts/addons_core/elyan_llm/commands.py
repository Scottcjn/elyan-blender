# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Bridge commands. Everything here runs on Blender's main thread.
"""

import ast
import contextlib
import importlib
import importlib.util
import inspect
import io
import json
import os
import runpy
import sys
import time
import traceback

import bpy

MAX_TEXT = 200_000
MAX_ITEMS = 2000

# Answered by the server itself, off the main thread, so they work while a job runs.
JOB_COMMANDS = ("submit", "status", "result", "cancel", "jobs")

# Custom properties that tie a checkpoint copy to what it was copied from.
_KEY_SOURCE = "elyan_checkpoint_source"
_KEY_DATA = "elyan_checkpoint_data"
_KEY_COLLECTIONS = "elyan_checkpoint_collections"
_KEY_TIME = "elyan_checkpoint_time"

# Set by ``server.serve()`` so the ``quit`` command can end a headless session.
quit_callback = None

# Names survive between ``exec`` calls so work can be built up step by step.
_namespace = {}

# Counts undo, redo and file loads. Each one frees Blender data that names in the
# namespace may still point at, so the namespace is emptied and the caller is told.
generation = 0


def invalidate():
    global generation
    generation += 1
    _namespace.clear()


# -----------------------------------------------------------------------------
# Helpers

def view3d_context():
    """Keyword arguments for ``bpy.context.temp_override`` targeting a 3D viewport, or None."""
    wm = bpy.context.window_manager
    for window in (wm.windows if wm else ()):
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for region in area.regions:
                if region.type == 'WINDOW':
                    return {"window": window, "screen": window.screen, "area": area, "region": region}
    return None


def _jsonable(value, depth=0):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth > 6:
        return repr(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(v, depth + 1) for v, _ in zip(value, range(MAX_ITEMS))]
        if len(value) > MAX_ITEMS:
            items.append("... [{:d} more items cut]".format(len(value) - MAX_ITEMS))
        return items
    if type(value).__module__ == "mathutils":
        try:
            return [_jsonable(v, depth + 1) for v in value]
        except TypeError:
            pass
    return repr(value)


def _clip(text):
    if len(text) <= MAX_TEXT:
        return text
    return text[:MAX_TEXT] + "\n... [{:d} characters cut]".format(len(text) - MAX_TEXT)


def _round(values, digits=4):
    return [round(v, digits) for v in values]


def _tri_count(mesh):
    import numpy as np
    count = len(mesh.polygons)
    if count == 0:
        return 0
    totals = np.empty(count, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", totals)
    return int(totals.sum()) - 2 * count


def _undo_push(message):
    """Let the artist undo what a request did. Returns whether a step was recorded."""
    if bpy.app.background:
        return False
    override = view3d_context()
    try:
        if override:
            with bpy.context.temp_override(**override):
                bpy.ops.ed.undo_push(message=message)
        else:
            bpy.ops.ed.undo_push(message=message)
    except RuntimeError:
        return False
    return True


# -----------------------------------------------------------------------------
# Commands

def cmd_ping(args):
    return {
        "blender": bpy.app.version_string,
        "blend": bpy.data.filepath,
        "dirty": bpy.data.is_dirty,
        "background": bpy.app.background,
        "pid": os.getpid(),
        "generation": generation,
        "commands": sorted((*COMMANDS, *JOB_COMMANDS)),
    }


def cmd_exec(args):
    """
    Run Python. The value of a trailing expression (or a variable named ``result``) is returned.

    The namespace persists between calls; pass ``reset`` to clear it. It is also
    cleared by undo, redo and loading a file: ``generation`` in the reply changes.
    ``diff`` adds what the code changed in the scene, see ``cmd_diff``.
    """
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("exec needs 'code'")
    before = _state_before(args)
    if args.get("reset"):
        _namespace.clear()
    _namespace.setdefault("bpy", bpy)
    _namespace.setdefault("view3d_context", view3d_context)
    _namespace.pop("result", None)

    tree = ast.parse(code, "<llm>")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)

    out = io.StringIO()
    response = {}
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            exec(compile(tree, "<llm>", "exec"), _namespace)
            value = eval(compile(last, "<llm>", "eval"), _namespace) if last else _namespace.get("result")
        response["result"] = _jsonable(value)
    except (Exception, SystemExit):
        # Report the failure together with whatever was printed before it.
        response["ok"] = False
        response["error"] = traceback.format_exc()
    response["stdout"] = _clip(out.getvalue())
    response["undo_pushed"] = _undo_push("LLM: " + (args.get("label") or code.strip().splitlines()[0][:60]))
    response["generation"] = generation
    response["mode"] = bpy.context.mode
    _state_after(before, response)
    return response


def _state_before(args):
    if not args.get("diff"):
        return None
    from . import checks
    return checks.scene_state()


def _state_after(before, response):
    if before is None:
        return
    from . import checks
    try:
        response["diff"] = checks.diff_states(before, checks.scene_state())
    except Exception:
        # The code ran; failing to describe its effect must not hide its own outcome.
        response["diff_error"] = traceback.format_exc()


def _under(path, root):
    try:
        return os.path.commonpath([os.path.realpath(path), root]) == root
    except ValueError:
        return False


def _purge_modules(root):
    """Forget every imported module whose source lives under ``root``. Returns their names."""
    purged = []
    for name, module in list(sys.modules.items()):
        if name == "__main__" or name.split(".")[0] == __package__:
            continue
        source = getattr(module, "__file__", None)
        if not source or not _under(source, root):
            continue
        # Compiled files are matched to their source by whole seconds and size, which
        # misses a quick small edit. Dropping them makes the source the only truth.
        try:
            os.remove(importlib.util.cache_from_source(source))
        except (OSError, ValueError, NotImplementedError):
            pass
        del sys.modules[name]
        purged.append(name)
    importlib.invalidate_caches()
    return sorted(purged)


def cmd_rebuild(args):
    """
    Run a builder script as ``blender --python`` would, but in this session.

    The script gets a fresh ``__main__`` namespace, its own ``__file__``, and its
    folder first on ``sys.path``. Modules already imported from that folder tree
    (``root`` widens or narrows it) are forgotten first, so an edited
    ``contract.py`` beside the script takes effect instead of the cached import.
    ``argv`` is passed on as ``sys.argv[1:]``; ``diff`` reports what changed.
    """
    path = args.get("path")
    if not path:
        raise ValueError("rebuild needs 'path'")
    path = os.path.realpath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise ValueError("no script at {:s}".format(path))
    folder = os.path.dirname(path)
    root = os.path.realpath(os.path.expanduser(args.get("root") or folder))
    # Forgetting the standard library or this bridge would break the session.
    for keep in (os.__file__, __file__, bpy.__file__ or __file__):
        if _under(keep, root):
            raise ValueError("refusing to reload everything under {:s}: it contains {:s}".format(root, keep))
    before = _state_before(args)

    purged = _purge_modules(root)
    known = set(sys.modules)
    saved_argv, saved_path, saved_bytecode = sys.argv, list(sys.path), sys.dont_write_bytecode
    out = io.StringIO()
    response = {"path": path, "root": root}
    started = time.time()
    try:
        sys.argv = [path] + [str(arg) for arg in args.get("argv") or ()]
        sys.path.insert(0, folder)
        sys.dont_write_bytecode = True
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            namespace = runpy.run_path(path, run_name="__main__")
        response["result"] = _jsonable(namespace.get("result"))
    except SystemExit as ex:
        # Builders written for ``blender -b --python`` often end with ``sys.exit(0)``.
        if ex.code not in {None, 0}:
            response["ok"] = False
            response["error"] = "the script exited with {!r}\n{:s}".format(ex.code, traceback.format_exc())
    except Exception:
        response["ok"] = False
        response["error"] = traceback.format_exc()
    finally:
        sys.argv = saved_argv
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_bytecode
    loaded = {
        name for name, module in sys.modules.items()
        if name not in known and _under(getattr(module, "__file__", None) or os.sep, root)
    }
    response["seconds"] = round(time.time() - started, 2)
    response["stdout"] = _clip(out.getvalue())
    response["reloaded"] = sorted(loaded.intersection(purged))
    response["imported"] = sorted(loaded.difference(purged))
    response["purged_not_reimported"] = sorted(set(purged).difference(loaded))
    response["undo_pushed"] = _undo_push("LLM rebuild: " + os.path.basename(path))
    response["generation"] = generation
    response["mode"] = bpy.context.mode
    _state_after(before, response)
    return response


def _require_object_mode(what):
    # Edit-mode changes live outside the mesh until the mode is left, a copy would miss them.
    if bpy.context.mode != 'OBJECT':
        raise RuntimeError("{:s} needs Object Mode, Blender is in {:s}".format(what, bpy.context.mode))


def _checkpoint_collection(name):
    from . import checks
    if not isinstance(name, str) or not name:
        raise ValueError("a checkpoint needs 'name'")
    return checks.CHECKPOINT_PREFIX + name


def _copy_objects(objects):
    """
    Copy objects together with their data. Returns ``{source name: copy}``.

    Parents, modifier targets and constraint targets that point at another
    copied object are moved to its copy, so the set stays self-contained.
    Actions and materials are shared, not copied.
    """
    data_copies, copies = {}, {}
    for ob in objects:
        new = ob.copy()
        if ob.data is not None:
            # Objects sharing one mesh keep sharing one.
            key = (type(ob.data).__name__, ob.data.name)
            if key not in data_copies:
                data_copies[key] = ob.data.copy()
            new.data = data_copies[key]
        copies[ob.name] = new
    for new in copies.values():
        if new.parent is not None and new.parent.name in copies:
            new.parent = copies[new.parent.name]
        for owner in (*new.modifiers, *new.constraints):
            for prop in owner.bl_rna.properties:
                if prop.type != 'POINTER' or prop.is_readonly:
                    continue
                target = getattr(owner, prop.identifier)
                if isinstance(target, bpy.types.Object) and target.name in copies:
                    setattr(owner, prop.identifier, copies[target.name])
    return copies


def _remove_objects(objects):
    """Delete objects, and their data when nothing else uses it."""
    data = {ob.data for ob in objects if ob.data is not None}
    for ob in objects:
        bpy.data.objects.remove(ob, do_unlink=True)
    orphans = [block for block in data if block.users == 0]
    if orphans:
        bpy.data.batch_remove(orphans)


def _drop_checkpoint(collection):
    _remove_objects(list(collection.objects))
    bpy.data.collections.remove(collection)


def _checkpoint_info(collection):
    from . import checks
    return {
        "name": collection.name[len(checks.CHECKPOINT_PREFIX):],
        "collection": collection.get(_KEY_SOURCE),
        "objects": sorted(ob.get(_KEY_SOURCE, ob.name) for ob in collection.objects),
        "time": collection.get(_KEY_TIME),
    }


def cmd_checkpoint(args):
    """
    Keep a copy of a collection's objects under ``name``, to return to with ``rollback``.

    The copies live in the hidden collection ``_checkpoint_<name>``, excluded from
    every view layer, so nothing is saved to disk and nothing renders or exports.
    Objects in child collections are included. ``replace`` overwrites the name.
    """
    from . import checks
    _require_object_mode("checkpoint")
    store_name = _checkpoint_collection(args.get("name"))
    source = bpy.data.collections.get(args.get("collection") or "")
    if source is None:
        raise ValueError("checkpoint needs 'collection', the name of an existing collection")
    if source.name.startswith(checks.CHECKPOINT_PREFIX):
        raise ValueError("a checkpoint cannot be checkpointed")
    existing = bpy.data.collections.get(store_name)
    if existing is not None:
        if not args.get("replace"):
            raise ValueError("checkpoint {!r} exists; pass 'replace' to overwrite it".format(args["name"]))
        _drop_checkpoint(existing)

    objects = list(source.all_objects)
    store = bpy.data.collections.new(store_name)
    store[_KEY_SOURCE] = source.name
    store[_KEY_TIME] = time.strftime("%Y-%m-%d %H:%M:%S")
    for name, copy in _copy_objects(objects).items():
        original = bpy.data.objects[name]
        copy[_KEY_SOURCE] = name
        if original.data is not None:
            copy[_KEY_DATA] = original.data.name
        copy[_KEY_COLLECTIONS] = json.dumps([c.name for c in original.users_collection])
        store.objects.link(copy)
    # Linked into the scene so it survives as long as the file is open, then switched off everywhere.
    scene = bpy.context.scene
    scene.collection.children.link(store)
    store.hide_viewport = True
    store.hide_render = True
    for view_layer in scene.view_layers:
        view_layer.layer_collection.children[store.name].exclude = True
    return _checkpoint_info(store)


def cmd_rollback(args):
    """
    Put a collection back as it was at ``checkpoint`` time.

    Objects made since are deleted, objects deleted since return, the others are
    replaced by fresh copies of the stored ones under their old names. Whatever
    pointed at a replaced object (parents, modifiers, other collections) now
    points at its replacement. The checkpoint stays, so this can be repeated.
    """
    _require_object_mode("rollback")
    store = bpy.data.collections.get(_checkpoint_collection(args.get("name")))
    if store is None:
        raise ValueError("no checkpoint named {!r}".format(args.get("name")))
    source = bpy.data.collections.get(store[_KEY_SOURCE])
    if source is None:
        # The whole collection was deleted since; bring it back too.
        source = bpy.data.collections.new(store[_KEY_SOURCE])
        bpy.context.scene.collection.children.link(source)

    current = {ob.name: ob for ob in source.all_objects}
    stored = list(store.objects)
    copies = _copy_objects(stored)
    restored = []
    for kept in stored:
        new = copies[kept.name]
        name = kept[_KEY_SOURCE]
        data_name = kept.get(_KEY_DATA)
        homes = json.loads(kept.get(_KEY_COLLECTIONS, "[]"))
        for key in (_KEY_SOURCE, _KEY_DATA, _KEY_COLLECTIONS):
            if key in new:
                del new[key]
        old = current.pop(name, None)
        if old is not None:
            # Everything that used the old object uses the restored one, collections included.
            old.user_remap(new)
            _remove_objects([old])
        if not new.users_collection:
            homes = [bpy.data.collections[home] for home in homes if home in bpy.data.collections]
            for home in homes or [source]:
                home.objects.link(new)
        new.name = name
        if new.data is not None and data_name:
            new.data.name = data_name
        restored.append(name)
    removed = sorted(current)
    _remove_objects(list(current.values()))
    bpy.context.view_layer.update()
    return {
        "name": args["name"],
        "collection": source.name,
        "restored": sorted(restored),
        "removed": removed,
        "undo_pushed": _undo_push("LLM rollback: " + args["name"]),
    }


def cmd_checkpoints(args):
    """List checkpoints; ``delete`` removes the named one first."""
    from . import checks
    if args.get("delete"):
        _require_object_mode("deleting a checkpoint")
        store = bpy.data.collections.get(_checkpoint_collection(args["delete"]))
        if store is None:
            raise ValueError("no checkpoint named {!r}".format(args["delete"]))
        _drop_checkpoint(store)
    return {"checkpoints": [
        _checkpoint_info(collection) for collection in bpy.data.collections
        if collection.name.startswith(checks.CHECKPOINT_PREFIX)
    ]}


def cmd_scene(args):
    """Overview of the scene: one compact record per object."""
    from . import checks
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    limit = int(args.get("limit") or 500)
    objects = []
    total_tris = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    # Checkpoint copies are storage, not scene content.
    listed = [ob for ob in scene.objects if not checks.is_checkpoint_object(ob)]
    for ob in listed[:limit]:
        item = {
            "name": ob.name,
            "type": ob.type,
            "location": _round(ob.location),
            "dimensions": _round(ob.dimensions),
            "visible": ob.visible_get(view_layer=view_layer),
        }
        if ob.parent:
            item["parent"] = ob.parent.name
        collections = [c.name for c in ob.users_collection]
        if collections:
            item["collections"] = collections
        if ob.modifiers:
            item["modifiers"] = [m.type for m in ob.modifiers]
        if ob.type == 'MESH':
            mesh = ob.data
            # What ships is the mesh after modifiers, so that is what budgets are checked against.
            tris = _tri_count(ob.evaluated_get(depsgraph).data)
            total_tris += tris
            item["verts"] = len(mesh.vertices)
            item["tris"] = tris
            item["tris_before_modifiers"] = _tri_count(mesh)
            item["materials"] = [m.name if m else None for m in mesh.materials]
            if mesh.shape_keys:
                item["shape_keys"] = len(mesh.shape_keys.key_blocks)
            if ob.vertex_groups:
                item["vertex_groups"] = len(ob.vertex_groups)
            item["uv_layers"] = [uv.name for uv in mesh.uv_layers]
        elif ob.type == 'ARMATURE':
            item["bones"] = len(ob.data.bones)
        objects.append(item)
    active = view_layer.objects.active
    return {
        "blend": bpy.data.filepath,
        "scene": scene.name,
        "mode": bpy.context.mode,
        "active": active.name if active else None,
        "selected": [ob.name for ob in scene.objects if ob.select_get(view_layer=view_layer)],
        "frame": [scene.frame_start, scene.frame_current, scene.frame_end],
        "unit_scale": scene.unit_settings.scale_length,
        "render_engine": scene.render.engine,
        "camera": scene.camera.name if scene.camera else None,
        "object_count": len(listed),
        "objects_truncated": len(listed) > limit,
        "mesh_tris_listed": total_tris,
        "objects": objects,
    }


def _material_info(material):
    info = {"name": material.name, "users": material.users}
    if material.node_tree:
        info["nodes"] = [
            {"name": node.name, "type": node.bl_idname,
             **({"image": node.image.filepath or node.image.name}
                if getattr(node, "image", None) is not None else {})}
            for node in material.node_tree.nodes
        ]
    return info


def cmd_object(args):
    """Everything worth knowing about one object."""
    name = args.get("name")
    ob = bpy.data.objects.get(name) if name else bpy.context.view_layer.objects.active
    if ob is None:
        raise ValueError("no object named {!r}".format(name))
    from mathutils import Vector
    corners = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    info = {
        "name": ob.name,
        "type": ob.type,
        "parent": ob.parent.name if ob.parent else None,
        "parent_bone": ob.parent_bone or None,
        "children": [c.name for c in ob.children],
        "location": _round(ob.location),
        "rotation_euler": _round(ob.rotation_euler),
        "scale": _round(ob.scale),
        "world_bounds_min": _round([min(c[i] for c in corners) for i in range(3)]),
        "world_bounds_max": _round([max(c[i] for c in corners) for i in range(3)]),
        "modifiers": [
            {"name": m.name, "type": m.type, "show_viewport": m.show_viewport, "show_render": m.show_render}
            for m in ob.modifiers
        ],
        "custom_properties": {k: _jsonable(ob[k]) for k in ob.keys()},
    }
    if ob.type == 'MESH':
        mesh = ob.data
        info["mesh"] = {
            "name": mesh.name,
            "verts": len(mesh.vertices),
            "edges": len(mesh.edges),
            "faces": len(mesh.polygons),
            "tris": _tri_count(mesh),
            "uv_layers": [uv.name for uv in mesh.uv_layers],
            "color_attributes": [a.name for a in mesh.color_attributes],
            "shape_keys": [k.name for k in mesh.shape_keys.key_blocks] if mesh.shape_keys else [],
        }
        info["vertex_groups"] = [g.name for g in ob.vertex_groups]
        info["materials"] = [_material_info(m) if m else None for m in mesh.materials]
    elif ob.type == 'ARMATURE':
        info["bones"] = [
            {"name": b.name, "parent": b.parent.name if b.parent else None,
             "head": _round(b.head_local), "tail": _round(b.tail_local)}
            for b in ob.data.bones
        ]
    return info


def cmd_render(args):
    """
    Write a PNG so the result can be looked at.

    ``mode`` "camera" renders through the scene camera; "viewport" captures the 3D
    viewport as the artist sees it (needs the UI). Scene settings are restored.
    """
    path = args.get("path")
    if not path:
        raise ValueError("render needs 'path'")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scene = bpy.context.scene
    render = scene.render
    mode = args.get("mode") or "camera"
    override = view3d_context() if mode == "viewport" else None
    if mode == "viewport" and override is None:
        raise RuntimeError("viewport mode needs a 3D viewport; use mode 'camera' when headless")

    saved = {
        "filepath": render.filepath,
        "file_format": render.image_settings.file_format,
        "resolution_x": render.resolution_x,
        "resolution_y": render.resolution_y,
        "resolution_percentage": render.resolution_percentage,
        "engine": render.engine,
        "camera": scene.camera,
    }
    try:
        render.filepath = path
        render.image_settings.file_format = 'PNG'
        if args.get("resolution"):
            render.resolution_x, render.resolution_y = (int(v) for v in args["resolution"])
            render.resolution_percentage = 100
        if args.get("engine"):
            render.engine = args["engine"]
        if args.get("camera"):
            scene.camera = bpy.data.objects[args["camera"]]
        started = time.time()
        if mode == "viewport":
            with bpy.context.temp_override(**override):
                bpy.ops.render.opengl(write_still=True, view_context=True)
        else:
            if scene.camera is None:
                raise RuntimeError("the scene has no camera")
            bpy.ops.render.render(write_still=True)
        return {"path": path, "seconds": round(time.time() - started, 2), "mode": mode, "engine": render.engine}
    finally:
        render.filepath = saved["filepath"]
        render.image_settings.file_format = saved["file_format"]
        render.resolution_x = saved["resolution_x"]
        render.resolution_y = saved["resolution_y"]
        render.resolution_percentage = saved["resolution_percentage"]
        render.engine = saved["engine"]
        scene.camera = saved["camera"]


def cmd_screenshot(args):
    """Capture the whole Blender window, panels included."""
    path = args.get("path")
    if not path:
        raise ValueError("screenshot needs 'path'")
    if bpy.app.background:
        raise RuntimeError("screenshot needs the UI")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    window = bpy.context.window_manager.windows[0]
    with bpy.context.temp_override(window=window, screen=window.screen):
        bpy.ops.screen.screenshot(filepath=path)
    return {"path": path}


def _operator_info(module_name, func_name):
    rna = getattr(getattr(bpy.ops, module_name), func_name).get_rna_type()
    return {
        "call": "bpy.ops.{:s}.{:s}".format(module_name, func_name),
        "label": rna.name,
        "description": rna.description,
        "properties": _properties(rna),
    }


def _properties(rna):
    result = []
    for prop in rna.properties:
        if prop.identifier == "rna_type":
            continue
        item = {"name": prop.identifier, "type": prop.type, "description": prop.description}
        if prop.type == 'ENUM':
            item["items"] = [e.identifier for e in prop.enum_items]
        elif prop.type == 'POINTER' and prop.fixed_type:
            item["points_to"] = prop.fixed_type.identifier
        if prop.is_readonly:
            item["readonly"] = True
        result.append(item)
    return result


def cmd_api(args):
    """
    Look up this build's Python API instead of guessing it.

    ``query`` "mesh.subdivide" or "subdivide" finds operators;
    a type name such as "SubsurfModifier" lists that type's properties.
    """
    query = (args.get("query") or "").strip()
    if not query:
        raise ValueError("api needs 'query'")
    limit = int(args.get("limit") or 25)

    struct = getattr(bpy.types, query, None)
    if struct is not None and hasattr(struct, "bl_rna"):
        rna = struct.bl_rna
        return {
            "type": query,
            "description": rna.description,
            "base": rna.base.identifier if rna.base else None,
            "properties": _properties(rna),
            "functions": [f.identifier for f in rna.functions],
        }

    needle = query.lower().removeprefix("bpy.ops.")
    operators = []
    truncated = False
    for module_name in dir(bpy.ops):
        for func_name in dir(getattr(bpy.ops, module_name)):
            full = "{:s}.{:s}".format(module_name, func_name)
            if needle not in full:
                continue
            if len(operators) >= limit:
                truncated = True
                break
            operators.append(_operator_info(module_name, func_name))
        if truncated:
            break
    types = [name for name in dir(bpy.types) if needle in name.lower()][:limit]
    return {"operators": operators, "operators_truncated": truncated, "types_matching": types}


def cmd_backup(args):
    """Save a timestamped copy next to the file, in ``backups/``, without changing the open file."""
    source = bpy.data.filepath
    if not source:
        raise RuntimeError("the file has never been saved, so there is nowhere to put a backup")
    folder = os.path.join(os.path.dirname(source), "backups")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(source))[0]
    label = "".join(c if c.isalnum() or c in "-_" else "-" for c in (args.get("label") or "")).strip("-")
    name = "{:s}_{:s}{:s}.blend".format(stem, time.strftime("%Y%m%d-%H%M%S"), "_" + label if label else "")
    path = os.path.join(folder, name)
    bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
    return {"path": path, "bytes": os.path.getsize(path)}


def cmd_validate(args):
    """
    Check objects against a delivery profile (see ``validate.PROFILES``).

    ``names`` lists the objects; without it the selection is used, or failing that
    every mesh in the scene. A rig given by name brings its skinned meshes along.
    """
    from . import checks, validate
    profile = args.get("profile") or "web"
    if profile not in validate.PROFILES:
        raise ValueError("unknown profile {!r}; known: {:s}".format(profile, ", ".join(sorted(validate.PROFILES))))
    names = args.get("names")
    if names:
        objects = [bpy.data.objects[name] for name in names]
        objects += [child for ob in objects if ob.type == 'ARMATURE' for child in ob.children_recursive]
    else:
        objects = list(bpy.context.selected_objects) or [
            ob for ob in bpy.context.scene.objects if ob.type == 'MESH' and not checks.is_checkpoint_object(ob)]
    return validate.validate(objects, profile)


def cmd_check(args):
    """
    Numeric geometry checks that fail closed; ``passed`` is true only if ``failures`` is empty.

    ``names`` lists mesh objects (default: the selection, else every mesh). For
    each: triangles after modifiers, material slots, non-manifold and wire edges,
    boundary edges and loops, loose vertices, zero-area faces, unapplied scale.
    Defects are counted after modifiers unless ``evaluated`` is false.

    ``budgets``: ``{"triangles": N, "material_slots": N, ...}`` against the totals.
    ``boundary_loops``: ``{object: expected count}``.
    ``seam``: ``{"object": A, "points": [[x, y, z], ...]}`` or ``{"object": A, "group": G,
    "other": B, "other_group": H}``, optional ``tolerance`` (1e-5): the distance from each
    target point to A's nearest open-boundary vertex, as ``max`` and ``rms``, unrounded.
    ``clearance``: ``{"garment": G, "body": B, "threshold": T, "ignore_groups": [...],
    "samples": N, "max_below": 0}``: signed distance of garment vertices to the body
    surface (negative inside), as ``min``, ``percentile_5`` and the count below ``threshold``.
    ``seam`` and ``clearance`` also take lists.
    """
    from . import checks
    return checks.check(args)


def cmd_snapshot(args):
    """Remember the scene's state under ``name`` for a later ``diff``."""
    from . import checks
    name = args.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("snapshot needs 'name'")
    state = checks.snapshot(name)
    return {"name": name, "objects": len(state), "snapshots": checks.snapshot_names()}


def cmd_diff(args):
    """
    What changed since snapshot ``name``, or between it and snapshot ``to``.

    Objects are compared by evaluated vertex and triangle counts, vertex positions,
    world bounding box, material names, modifier types and world transform.
    """
    from . import checks
    before = checks.stored(args.get("name"))
    after = checks.stored(args["to"]) if args.get("to") else checks.scene_state()
    response = checks.diff_states(before, after)
    response["from"] = args["name"]
    response["to"] = args.get("to") or "now"
    return response


def cmd_contact_sheet(args):
    """
    Render fixed views of ``names`` (objects, with their children) into one labelled PNG at ``path``.

    ``views`` defaults to front, back, left, right and three_quarter ("top" also exists).
    ``closeup``: ``{"armature": RIG, "bone": BONE}`` or ``{"object": NAME}``, with optional
    ``size`` (width shown, in scene units) and ``view``. ``tile`` is the size of one view in
    pixels (at most 400), ``samples`` at most 24, ``columns`` the grid width.
    Rendered with Cycles on the CPU in a temporary scene; the artist's scene is not touched.
    """
    from . import sheet
    return sheet.contact_sheet(args)


def cmd_help(args):
    """
    What the bridge can do. With ``command``, that command's full description;
    otherwise one line for each.
    """
    name = args.get("command")
    if name:
        func = COMMANDS.get(name)
        if func is None:
            raise ValueError("unknown command {!r}; known: {:s}".format(name, ", ".join(sorted(COMMANDS))))
        return {"command": name, "help": inspect.cleandoc(func.__doc__ or "No description.")}
    lines = {}
    for key, func in sorted(COMMANDS.items()):
        doc = inspect.cleandoc(func.__doc__ or "").strip()
        lines[key] = doc.split("\n\n")[0].replace("\n", " ") if doc else ""
    return {"commands": lines, "job_commands": list(JOB_COMMANDS)}


def cmd_state(args):
    """
    Where things stand: the file, whether a person has it open in a window, what is
    selected, open checkpoints, and the latest requests. The first call of a session.
    """
    from . import server, session
    view_layer = bpy.context.view_layer
    active = view_layer.objects.active
    state = {
        "blender": bpy.app.version_string,
        "blend": bpy.data.filepath,
        "dirty": bpy.data.is_dirty,
        "window": not bpy.app.background,
        "mode": bpy.context.mode,
        "active": active.name if active else None,
        "selected": [ob.name for ob in bpy.context.scene.objects if ob.select_get(view_layer=view_layer)][:50],
        "collections": [c.name for c in bpy.context.scene.collection.children if not c.name.startswith("_checkpoint_")],
        "checkpoints": [c.name[len("_checkpoint_"):] for c in bpy.data.collections if c.name.startswith("_checkpoint_")],
        "generation": generation,
        "recent": [
            {"time": clock, "command": cmd, "what": what, "ok": ok} for clock, cmd, what, ok in list(server.log)[-8:]
        ],
        "window_selftest": session.read_selftest(),
    }
    # Parts that other modules add when they are present.
    for key, func in STATE_EXTRAS.items():
        state[key] = func()
    return state


def cmd_quit(args):
    """End a headless ``serve()`` session. Refused in the UI, where the artist owns the window."""
    if quit_callback is None:
        raise RuntimeError("quit only applies to a headless serve() session")
    quit_callback()
    return {"quitting": True}


COMMANDS = {
    "ping": cmd_ping,
    "exec": cmd_exec,
    "rebuild": cmd_rebuild,
    "checkpoint": cmd_checkpoint,
    "rollback": cmd_rollback,
    "checkpoints": cmd_checkpoints,
    "check": cmd_check,
    "snapshot": cmd_snapshot,
    "diff": cmd_diff,
    "contact_sheet": cmd_contact_sheet,
    "scene": cmd_scene,
    "object": cmd_object,
    "render": cmd_render,
    "screenshot": cmd_screenshot,
    "api": cmd_api,
    "backup": cmd_backup,
    "validate": cmd_validate,
    "help": cmd_help,
    "state": cmd_state,
    "quit": cmd_quit,
}

# ``state`` asks each of these for its part: name -> function returning something JSON can hold.
STATE_EXTRAS = {}

# Command groups that live in files of their own. Each defines ``COMMANDS`` and
# may define ``STATE``; one that is missing is simply not offered.
_GROUPS = ("contract", "compare", "marks", "staging")


def _load_groups():
    for name in _GROUPS:
        try:
            module = importlib.import_module("." + name, __package__)
        except ImportError:
            continue
        COMMANDS.update(module.COMMANDS)
        STATE_EXTRAS.update(getattr(module, "STATE", {}))


_load_groups()


def dispatch(cmd, args):
    func = COMMANDS.get(cmd)
    if func is None:
        return {"ok": False, "error": "unknown command {!r}; known: {:s}".format(cmd, ", ".join(sorted(COMMANDS)))}
    try:
        response = func(args) or {}
    except (Exception, SystemExit):
        # ``sys.exit()`` in a script must not take the artist's Blender down with it.
        return {"ok": False, "error": traceback.format_exc()}
    response.setdefault("ok", True)
    return response


def summarize(cmd, args):
    """One short line for the panel log."""
    if cmd == "exec":
        code = (args.get("code") or "").strip()
        return args.get("label") or (code.splitlines()[0][:48] if code else "")
    if cmd == "submit":
        return "{!s} {:s}".format(args.get("cmd"), summarize(str(args.get("cmd")), args.get("args") or {})).strip()
    for key in ("name", "path", "query", "label", "job"):
        if args.get(key):
            return str(args[key])[-48:]
    return ""
