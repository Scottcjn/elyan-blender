# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Bridge commands. Everything here runs on Blender's main thread.
"""

import ast
import contextlib
import io
import os
import time
import traceback

import bpy

MAX_TEXT = 200_000
MAX_ITEMS = 2000

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
        "commands": sorted(COMMANDS),
    }


def cmd_exec(args):
    """
    Run Python. The value of a trailing expression (or a variable named ``result``) is returned.

    The namespace persists between calls; pass ``reset`` to clear it. It is also
    cleared by undo, redo and loading a file: ``generation`` in the reply changes.
    """
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("exec needs 'code'")
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
    return response


def cmd_scene(args):
    """Overview of the scene: one compact record per object."""
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    limit = int(args.get("limit") or 500)
    objects = []
    total_tris = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for ob in scene.objects[:limit]:
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
        "object_count": len(scene.objects),
        "objects_truncated": len(scene.objects) > limit,
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
    from . import validate
    profile = args.get("profile") or "web"
    if profile not in validate.PROFILES:
        raise ValueError("unknown profile {!r}; known: {:s}".format(profile, ", ".join(sorted(validate.PROFILES))))
    names = args.get("names")
    if names:
        objects = [bpy.data.objects[name] for name in names]
        objects += [child for ob in objects if ob.type == 'ARMATURE' for child in ob.children_recursive]
    else:
        objects = list(bpy.context.selected_objects) or [ob for ob in bpy.context.scene.objects if ob.type == 'MESH']
    return validate.validate(objects, profile)


def cmd_quit(args):
    """End a headless ``serve()`` session. Refused in the UI, where the artist owns the window."""
    if quit_callback is None:
        raise RuntimeError("quit only applies to a headless serve() session")
    quit_callback()
    return {"quitting": True}


COMMANDS = {
    "ping": cmd_ping,
    "exec": cmd_exec,
    "scene": cmd_scene,
    "object": cmd_object,
    "render": cmd_render,
    "screenshot": cmd_screenshot,
    "api": cmd_api,
    "backup": cmd_backup,
    "validate": cmd_validate,
    "quit": cmd_quit,
}


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
    for key in ("name", "path", "query", "label"):
        if args.get(key):
            return str(args[key])[-48:]
    return ""
