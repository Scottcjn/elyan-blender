# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Export a person for a delivery target: one material, one atlas, within budget, as one
mesh or as a head that carries the face shapes plus a body that does not.

Works on a copy, the character in the scene is left as it was. Shape keys that
are at zero (visemes, expressions) survive; those that are in use (the body's
shape) are baked in.
"""

import json
import math
import os
import time

import bpy
import numpy as np

from mathutils import Vector, kdtree

from . import build, face, motion

try:
    from elyan_llm import validate as _validate
except ImportError:
    _validate = None

# Share of the atlas side each kind of part gets. Powers of two, so they pack exactly.
_TILE = {
    "Basemesh": 0.5, "Clothes": 0.5, "Hair": 0.25,
    "Teeth": 0.125, "Tongue": 0.125, "Eyes": 0.125, "Eyebrows": 0.125, "Eyelashes": 0.125,
}
# Parts small enough on screen to be cut to a fixed size whatever the budget.
_TRIANGLE_CAP = {"Teeth": 1500, "Tongue": 250}
# Parts whose triangles are traded against the budget.
_FLEXIBLE = ("Basemesh", "Hair", "Clothes")
_FACE_BONES = ("head", "neck_01", "neck", "jaw")
_FLOOR = "_elyan_reduction_floor"
# Parts that belong to the face whatever they weigh, and the bones that mark the head's skin.
_HEAD_KINDS = ("Teeth", "Tongue", "Eyes", "Eyebrows", "Eyelashes")
_HEAD_BONES = ("head", "eye_l", "eye_r", "jaw")
_HEAD_FACE = "_elyan_head"
_VERTEX_ID = "_elyan_vertex"
_VERTEX_NORMAL = "_elyan_normal"
_KEY_OFFSET = "_elyan_key_"
_VRCHAT = ("quest", "vrchat_pc")
# Where a contact sheet's camera stands, as seen from the person, who faces -Y.
_VIEWS = {"body": Vector((0.45, -1.0, 0.12)), "front": Vector((0.0, -1.0, 0.05)), "side": Vector((1.0, -0.05, 0.05))}
_FALLBACK_BUDGET = {"web": 45000, "vrchat_pc": 70000, "quest": 15000}


def _object_type(ob):
    return build.mpfb("entities.objectproperties", "GeneralObjectProperties").get_value(
        "object_type", entity_reference=ob)


def _triangles(mesh):
    count = len(mesh.polygons)
    if count == 0:
        return 0
    totals = np.empty(count, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", totals)
    return int(totals.sum()) - 2 * count


def _coordinates(source, count):
    co = np.empty(count * 3, dtype=np.float32)
    source.foreach_get("co", co)
    return co.reshape(-1, 3)


# -----------------------------------------------------------------------------
# Shape keys

def _settle_shape_keys(ob, wanted):
    """
    Bake the shape keys that are in use and lift out the ones at rest.

    Returns the rest ones named in ``wanted`` as ``(positions before, {name: offsets})``
    so they can be put back after the mesh has been reduced; other rest ones are dropped.
    Their movement also rides along on the mesh as attributes, which reduction blends
    the way it blends weights.
    """
    mesh = ob.data
    if not mesh.shape_keys:
        return None
    blocks = mesh.shape_keys.key_blocks
    count = len(mesh.vertices)
    basis = _coordinates(blocks[0].data, count)
    resting = {
        block.name: _coordinates(block.data, count) - basis
        for block in blocks[1:] if block.value == 0.0 and block.name in wanted
    }
    mix = ob.shape_key_add(name="_elyan_mix", from_mix=True)
    mixed = _coordinates(mix.data, count)
    ob.shape_key_clear()
    mesh.vertices.foreach_set("co", mixed.ravel())
    for number, offsets in enumerate(resting.values()):
        attribute = mesh.attributes.new(_KEY_OFFSET + str(number), 'FLOAT_VECTOR', 'POINT')
        attribute.data.foreach_set("vector", offsets.ravel())
    mesh.update()
    return (mixed, resting) if resting else None


def _restore_shape_keys(ob, settled, rename):
    """
    Put resting shape keys back, from the movement that rode along as attributes.

    Taking each vertex's movement from the nearest vertex of the mesh as it was is only
    the fallback: where two surfaces touch, as closed teeth do, the nearest vertex can
    be on the wrong one, and a lower tooth then stays up when the jaw drops.
    """
    if settled is None:
        return
    before, resting = settled
    mesh = ob.data
    now = _coordinates(mesh.vertices, len(mesh.vertices))
    nearest = None
    ob.shape_key_add(name="Basis")
    for number, (name, offsets) in enumerate(resting.items()):
        attribute = mesh.attributes.get(_KEY_OFFSET + str(number))
        if attribute is not None and attribute.domain == 'POINT' and len(attribute.data) == len(now):
            carried = np.empty(len(now) * 3, dtype=np.float32)
            attribute.data.foreach_get("vector", carried)
            carried = carried.reshape(-1, 3)
        else:
            if nearest is None:
                tree = kdtree.KDTree(len(before))
                for index, co in enumerate(before):
                    tree.insert(co, index)
                tree.balance()
                nearest = np.array([tree.find(co)[1] for co in now], dtype=np.int64)
            carried = offsets[nearest]
        block = ob.shape_key_add(name=rename.get(name, name))
        block.data.foreach_set("co", (now + carried).ravel())
    for attribute in [a for a in mesh.attributes if a.name.startswith(_KEY_OFFSET)]:
        mesh.attributes.remove(attribute)


def _nudge_silence(parts, name):
    """
    Give the silence shape a movement too small to see, on a few vertices at the back of
    the tongue (or the teeth), so importers that discard empty shapes keep it.

    Returns the kind of part that was moved, or None when there was nothing inside the mouth.
    """
    for wanted in ("Tongue", "Teeth"):
        ob = next((ob for ob, kind in parts if kind == wanted), None)
        if ob is None:
            continue
        mesh = ob.data
        if not mesh.shape_keys:
            ob.shape_key_add(name="Basis")
        block = mesh.shape_keys.key_blocks.get(name) or ob.shape_key_add(name=name)
        block.value = 0.0
        # From the rest shape, not from the new key: a key added to a mesh that already
        # has keys does not reliably start as a copy of it.
        co = _coordinates(mesh.shape_keys.key_blocks[0].data, len(mesh.vertices))
        # The character faces -Y, so the largest Y is the deepest in the mouth.
        for index in np.argsort(co[:, 1])[-3:]:
            co[index, 2] += face.SILENCE_NUDGE
        block.data.foreach_set("co", co.ravel())
        return wanted
    return None


def _mark_head(parts):
    """
    Flag, per face, what goes into the head mesh: the face parts whole, and of the skin
    whatever a shape key moves or the head bones hold, with two rings to spare.

    The spare rings put the seam where nothing moves, so the two meshes cannot part there
    and the lighting across it does not change when the face does.
    """
    for ob, kind in parts:
        mesh = ob.data
        flags = np.zeros(len(mesh.polygons), dtype=np.int32)
        if kind in _HEAD_KINDS:
            flags[:] = 1
        elif kind == "Basemesh":
            count = len(mesh.vertices)
            core = np.zeros(count, dtype=bool)
            if mesh.shape_keys:
                blocks = mesh.shape_keys.key_blocks
                basis = _coordinates(blocks[0].data, count)
                for block in blocks[1:]:
                    core |= np.abs(_coordinates(block.data, count) - basis).max(axis=1) > 1e-7
            indices = {ob.vertex_groups[name].index for name in _HEAD_BONES if name in ob.vertex_groups}
            for vert in mesh.vertices:
                if sum(g.weight for g in vert.groups if g.group in indices) >= 0.5:
                    core[vert.index] = True
            corners = np.empty(len(mesh.loops), dtype=np.int32)
            mesh.loops.foreach_get("vertex_index", corners)
            owner = np.repeat(np.arange(len(mesh.polygons)), [poly.loop_total for poly in mesh.polygons])
            for _ring in range(2):
                flags[:] = 0
                flags[owner[core[corners]]] = 1
                core[corners[flags[owner] == 1]] = True
            flags[:] = 0
            flags[owner[core[corners]]] = 1
        attribute = mesh.attributes.get(_HEAD_FACE) or mesh.attributes.new(_HEAD_FACE, 'INT', 'FACE')
        attribute.data.foreach_set("value", flags)


def _delete_faces(mesh, flagged, value):
    """Remove the faces whose head flag is ``value``, and the vertices only they used."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    doomed = [bm.faces[index] for index in np.where(flagged == value)[0]]
    bmesh.ops.delete(bm, geom=doomed, context='FACES')
    bm.to_mesh(mesh)
    bm.free()


def _split_head(joined, name):
    """
    Cut the joined mesh in two: ``joined`` keeps the body and loses its shape keys, the
    returned object is the head with them.

    The ring of vertices where they meet exists in both, with the same positions and
    weights because both are copies, and with the normals the uncut mesh had there.
    """
    mesh = joined.data
    count = len(mesh.vertices)
    flagged = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.attributes[_HEAD_FACE].data.foreach_get("value", flagged)
    mesh.attributes.remove(mesh.attributes[_HEAD_FACE])

    normals = np.empty(count * 3, dtype=np.float32)
    mesh.vertex_normals.foreach_get("vector", normals)
    mesh.attributes.new(_VERTEX_NORMAL, 'FLOAT_VECTOR', 'POINT').data.foreach_set("vector", normals)
    mesh.attributes.new(_VERTEX_ID, 'INT', 'POINT').data.foreach_set("value", np.arange(count, dtype=np.int32))

    head = joined.copy()
    head.data = mesh.copy()
    head.name = head.data.name = name + ".head"
    for collection in joined.users_collection:
        collection.objects.link(head)
    _delete_faces(head.data, flagged, 0)
    joined.shape_key_clear()
    _delete_faces(mesh, flagged, 1)

    def identities(ob):
        ids = np.empty(len(ob.data.vertices), dtype=np.int32)
        ob.data.attributes[_VERTEX_ID].data.foreach_get("value", ids)
        return ids

    seam = np.intersect1d(identities(head), identities(joined))
    for ob in (head, joined):
        part = ob.data
        on_seam = np.isin(identities(ob), seam)
        stored = np.empty(len(part.vertices) * 3, dtype=np.float32)
        part.attributes[_VERTEX_NORMAL].data.foreach_get("vector", stored)
        corners = np.empty(len(part.loops), dtype=np.int32)
        part.loops.foreach_get("vertex_index", corners)
        # A zero normal leaves a corner as it is; only the seam is told what to be.
        custom = np.zeros((len(part.loops), 3), dtype=np.float32)
        custom[on_seam[corners]] = stored.reshape(-1, 3)[corners[on_seam[corners]]]
        part.normals_split_custom_set(custom.tolist())
        for attribute in (_VERTEX_ID, _VERTEX_NORMAL):
            part.attributes.remove(part.attributes[attribute])
    return head, len(seam)


# -----------------------------------------------------------------------------
# Reduction

def _protect_face(ob):
    """A vertex group of everything that may be simplified: all but the head and neck."""
    indices = {ob.vertex_groups[name].index for name in _FACE_BONES if name in ob.vertex_groups}
    free = [
        vert.index for vert in ob.data.vertices
        if sum(g.weight for g in vert.groups if g.group in indices) < 0.05
    ]
    group = ob.vertex_groups.new(name="_elyan_reducible")
    group.add(free, 1.0, 'REPLACE')
    return group


def _decimate(ob, ratio, protect_face=False):
    """Collapse to ``ratio`` of the triangles. The armature is not applied."""
    if ratio >= 0.999:
        return
    group = _protect_face(ob) if protect_face else None
    if group is not None:
        # The ratio counts the whole mesh but only the unprotected part can give, so asking
        # too much would eat the arms and hands entirely. They keep at least a quarter of
        # what they started with, however many rounds of reduction there are.
        total = len(ob.data.polygons)
        if _FLOOR not in ob:
            reducible = {v.index for v in ob.data.vertices if any(g.group == group.index for g in v.groups)}
            giving = sum(1 for face in ob.data.polygons if all(index in reducible for index in face.vertices))
            ob[_FLOOR] = total - giving + 0.25 * giving
        ratio = max(ratio, ob[_FLOOR] / max(1, total))
        if ratio >= 0.999:
            ob.vertex_groups.remove(group)
            return
    armatures = [m for m in ob.modifiers if m.type == 'ARMATURE']
    for modifier in armatures:
        modifier.show_viewport = False
    decimate = ob.modifiers.new("_elyan_decimate", 'DECIMATE')
    decimate.ratio = max(ratio, 0.02)
    if group is not None:
        decimate.vertex_group = group.name
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    reduced = bpy.data.meshes.new_from_object(ob.evaluated_get(depsgraph), preserve_all_data_layers=True,
                                              depsgraph=depsgraph)
    old = ob.data
    ob.modifiers.remove(decimate)
    reduced.name = old.name
    ob.data = reduced
    bpy.data.meshes.remove(old)
    for modifier in armatures:
        modifier.show_viewport = True
    if group is not None:
        ob.vertex_groups.remove(ob.vertex_groups[group.name])


def _fit_budget(parts, budget):
    """Reduce parts until their triangles fit ``budget``. ``parts`` is [(object, type)]."""
    for ob, kind in parts:
        cap = _TRIANGLE_CAP.get(kind)
        if cap and _triangles(ob.data) > cap:
            _decimate(ob, cap / _triangles(ob.data))
    # The face is protected, so the body gives up less than asked; a few rounds settle it.
    for _ in range(8):
        total = sum(_triangles(ob.data) for ob, _kind in parts)
        if total <= budget:
            break
        flexible = [(ob, kind) for ob, kind in parts if kind in _FLEXIBLE]
        fixed = total - sum(_triangles(ob.data) for ob, _kind in flexible)
        ratio = max(0.1, (budget * 0.95 - fixed) / max(1, total - fixed))
        for ob, kind in flexible:
            _decimate(ob, ratio, protect_face=kind == "Basemesh")


def _clean_weights(ob, rig):
    """Only bone groups, at most four per vertex, summing to one."""
    bones = {bone.name for bone in rig.data.bones}
    for group in list(ob.vertex_groups):
        if group.name not in bones:
            ob.vertex_groups.remove(group)
    if not ob.vertex_groups:
        return
    for vert in ob.data.vertices:
        weights = sorted(((g.weight, g.group) for g in vert.groups), reverse=True)
        keep = weights[:4]
        total = sum(weight for weight, _index in keep)
        for _weight, index in weights[4:]:
            ob.vertex_groups[index].remove([vert.index])
        if total > 0.0:
            for weight, index in keep:
                ob.vertex_groups[index].add([vert.index], weight / total, 'REPLACE')


# -----------------------------------------------------------------------------
# Atlas

def _pack(sizes):
    """
    Place squares (side as a fraction of the atlas, powers of two) without overlap.

    Returns ``[(x, y, side)]`` in the order given, or None when they do not fit.
    """
    free = [(0.0, 0.0, 1.0)]
    placed = {}
    for index in sorted(range(len(sizes)), key=lambda i: -sizes[i]):
        side = sizes[index]
        candidates = [square for square in free if square[2] >= side]
        if not candidates:
            return None
        x, y, size = min(candidates, key=lambda square: square[2])
        free.remove((x, y, size))
        while size > side:
            size /= 2.0
            free += [(x + size, y, size), (x, y + size, size), (x + size, y + size, size)]
        placed[index] = (x, y, side)
    return [placed[index] for index in range(len(sizes))]


def _base_color(material):
    """The image feeding a material's base colour, or its flat colour."""
    flat = (0.8, 0.8, 0.8, 1.0)
    if material is None or not material.node_tree:
        return None, flat
    shader = next((n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if shader is None:
        return None, flat
    socket = shader.inputs["Base Color"]
    flat = tuple(socket.default_value)
    pending, seen = [socket], set()
    while pending:
        socket = pending.pop()
        for link in socket.links:
            node = link.from_node
            if node in seen:
                continue
            seen.add(node)
            if node.type == 'TEX_IMAGE' and node.image:
                return node.image, flat
            pending += list(node.inputs)
    return None, flat


def _tile_pixels(image, flat, side):
    """``side`` x ``side`` RGBA pixels of an image, or of a flat colour."""
    if image is None or image.size[0] == 0:
        return np.tile(np.array(flat, dtype=np.float32), (side, side, 1))
    scaled = image.copy()
    scaled.scale(side, side)
    pixels = np.empty(side * side * 4, dtype=np.float32)
    scaled.pixels.foreach_get(pixels)
    bpy.data.images.remove(scaled)
    return pixels.reshape(side, side, 4)


def _build_atlas(parts, size, name):
    """One image holding every part's texture, and each part's UVs moved onto its tile."""
    sizes = [_TILE.get(kind, 0.125) for _ob, kind in parts]
    # Only the first garment gets a large tile.
    seen_clothes = False
    for index, (_ob, kind) in enumerate(parts):
        if kind == "Clothes":
            if seen_clothes:
                sizes[index] = 0.25
            seen_clothes = True
    tiles = _pack(sizes)
    while tiles is None:
        sizes = [side / 2.0 for side in sizes]
        tiles = _pack(sizes)

    atlas = np.zeros((size, size, 4), dtype=np.float32)
    for (ob, _kind), (x, y, side) in zip(parts, tiles):
        mesh = ob.data
        image, flat = _base_color(mesh.materials[0] if mesh.materials else None)
        pixel_side = int(round(side * size))
        px, py = int(round(x * size)), int(round(y * size))
        atlas[py:py + pixel_side, px:px + pixel_side] = _tile_pixels(image, flat, pixel_side)

        layer = mesh.uv_layers.active or mesh.uv_layers.new(name="UVMap")
        uv = np.empty(len(mesh.loops) * 2, dtype=np.float32)
        layer.data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2)
        # Tiling coordinates fold back into the tile; half a pixel in keeps neighbours from bleeding.
        inset = 0.5 / pixel_side
        uv = np.clip(uv - np.floor(uv), inset, 1.0 - inset) * side + np.array((x, y), dtype=np.float32)
        layer.data.foreach_set("uv", uv.ravel())
        layer.name = "UVMap"
        for extra in [other for other in mesh.uv_layers if other != layer]:
            mesh.uv_layers.remove(extra)

    image = bpy.data.images.new(name, size, size, alpha=True)
    image.pixels.foreach_set(atlas.ravel())
    return image


def _atlas_material(name, image):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    # Rounding the alpha makes exporters write a cut-out, which sorts correctly, not a blend.
    cutout = nodes.new("ShaderNodeMath")
    cutout.operation = 'ROUND'
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Roughness"].default_value = 0.7
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(texture.outputs["Color"], shader.inputs["Base Color"])
    links.new(texture.outputs["Alpha"], cutout.inputs[0])
    links.new(cutout.outputs[0], shader.inputs["Alpha"])
    links.new(shader.outputs[0], output.inputs["Surface"])
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = 'DITHERED'
    return material


# -----------------------------------------------------------------------------
# Export

def _select_only(objects, active):
    for ob in bpy.context.view_layer.objects:
        ob.select_set(False)
    for ob in objects:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = active


def _round_trip(path):
    """Read an exported GLB back and count what arrived."""
    before = set(bpy.data.objects)
    actions = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=path)
    arrived = [ob for ob in bpy.data.objects if ob not in before]
    arrived_actions = [action for action in bpy.data.actions if action not in actions]
    # The importer adds a small unskinned mesh to draw bones with; it is not part of the asset.
    meshes = [ob for ob in arrived if ob.type == 'MESH' and ob.vertex_groups]
    result = {
        "triangles": sum(_triangles(ob.data) for ob in meshes),
        "meshes": len(meshes),
        "materials": len({m.name for ob in meshes for m in ob.data.materials if m}),
        "bones": sum(len(ob.data.bones) for ob in arrived if ob.type == 'ARMATURE'),
        "shape_keys": max(
            (len(ob.data.shape_keys.key_blocks) - 1 for ob in meshes if ob.data.shape_keys), default=0),
        "animations": sorted(action.name for action in arrived_actions),
    }
    for ob in arrived:
        bpy.data.objects.remove(ob)
    for action in arrived_actions:
        bpy.data.actions.remove(action)
    return result


def contact_sheet(rig, path, shots=None, view="body", tile=(270, 400), samples=12, columns=6):
    """
    Render a person in several poses side by side and save one image: what to look at
    before believing the numbers.

    A shot is ``(clip, seconds)`` or ``(clip, seconds, {shape key: value})``; the default is
    every clip of ``motion.CLIPS`` at its telling moment. ``view`` is "body" (three-quarter),
    "front", "side" or "face", or ``{"target": point, "toward": direction, "scale": metres}``.
    Rendered on the processor with Cycles in a scene of its own, so nothing else shows.
    Returns the shots in the order drawn, left to right, top to bottom.
    """
    shots = [tuple(shot) for shot in (shots or [(name, motion.SHOWN_AT[name]) for name in motion.CLIPS])]
    meshes = [ob for ob in rig.children_recursive if ob.type == 'MESH']
    head = rig.data.bones.get("head")
    top = max((rig.matrix_world @ Vector(corner)).z for ob in meshes for corner in ob.bound_box) if meshes else 1.7

    scene = bpy.data.scenes.new("_elyan_sheet")
    made = []
    try:
        for ob in [rig] + meshes:
            scene.collection.objects.link(ob)
        camera = bpy.data.objects.new("_elyan_sheet_camera", bpy.data.cameras.new("_elyan_sheet_camera"))
        sun = bpy.data.objects.new("_elyan_sheet_sun", bpy.data.lights.new("_elyan_sheet_sun", 'SUN'))
        made += [camera, sun]
        for ob in made:
            scene.collection.objects.link(ob)
        camera.data.type = 'ORTHO'
        if isinstance(view, dict):
            # A closer look at anything: where to aim, from which side, how many metres tall.
            target = Vector(view["target"])
            camera.data.ortho_scale = view.get("scale", 0.5)
            toward = Vector(view.get("toward", _VIEWS["body"]))
        elif view == "face" and head is not None:
            target = rig.matrix_world @ head.head_local + Vector((0.0, 0.0, -0.01))
            camera.data.ortho_scale = 0.36
            toward = Vector((0.3, -1.0, 0.05))
        else:
            target = Vector((rig.matrix_world.translation.x, rig.matrix_world.translation.y, top * 0.52))
            camera.data.ortho_scale = top * 1.22
            toward = _VIEWS.get(view, _VIEWS["body"])
        camera.location = target + toward.normalized() * 6.0
        camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
        sun.data.energy = 3.0
        sun.rotation_euler = (math.radians(55.0), 0.0, math.radians(25.0))
        world = bpy.data.worlds.new("_elyan_sheet")
        world.use_nodes = True
        background = world.node_tree.nodes.get("Background")
        if background is not None:
            background.inputs[0].default_value = (0.72, 0.72, 0.75, 1.0)
            background.inputs[1].default_value = 1.0
        scene.world = world
        scene.camera = camera
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = min(samples, 24)
        scene.cycles.use_denoising = False
        scene.render.resolution_x, scene.render.resolution_y = min(tile[0], 540), min(tile[1], 540)
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'

        width, height = scene.render.resolution_x, scene.render.resolution_y
        columns = min(columns, len(shots))
        rows = -(-len(shots) // columns)
        sheet = np.ones((rows * height, columns * width, 4), dtype=np.float32)
        stem = os.path.splitext(os.path.abspath(path))[0]
        for number, shot in enumerate(shots):
            motion.show(rig, shot[0], shot[1])
            values = shot[2] if len(shot) > 2 else {}
            for ob in meshes:
                if ob.data.shape_keys:
                    for block in ob.data.shape_keys.key_blocks[1:]:
                        if face.is_face_key(block.name):
                            block.value = values.get(block.name, 0.0)
            scene.render.filepath = "{:s}.tile{:02d}.png".format(stem, number)
            bpy.ops.render.render(write_still=True, scene=scene.name)
            image = bpy.data.images.load(scene.render.filepath)
            pixels = np.empty(width * height * 4, dtype=np.float32)
            image.pixels.foreach_get(pixels)
            bpy.data.images.remove(image)
            os.remove(scene.render.filepath)
            # Image rows run bottom to top, the sheet reads top to bottom.
            row, column = rows - 1 - number // columns, number % columns
            sheet[row * height:(row + 1) * height, column * width:(column + 1) * width] = pixels.reshape(
                height, width, 4)
        result = bpy.data.images.new("_elyan_sheet", columns * width, rows * height, alpha=True)
        result.pixels.foreach_set(sheet.ravel())
        result.filepath_raw = path
        result.file_format = 'PNG'
        result.save()
        bpy.data.images.remove(result)
    finally:
        motion._pose(rig, [])
        for ob in meshes:
            if ob.data.shape_keys:
                for block in ob.data.shape_keys.key_blocks[1:]:
                    if face.is_face_key(block.name):
                        block.value = 0.0
        for ob in made:
            data = ob.data
            bpy.data.objects.remove(ob)
            (bpy.data.cameras if isinstance(data, bpy.types.Camera) else bpy.data.lights).remove(data)
        world = scene.world
        bpy.data.scenes.remove(scene)
        if world is not None:
            bpy.data.worlds.remove(world)
    return [list(shot[:2]) for shot in shots]


def export(rig, path, profile="web", formats=("glb",), keep=False, clips=(), split_head=None, sheet=False):
    """
    Write ``path`` (extension is replaced per format) for a delivery profile.

    ``clips`` names body motions from ``motion.CLIPS`` to include as animations.
    ``split_head`` keeps the head, with the face shapes, apart from the body, so shapes
    are stored for the head's vertices only; by default the web profile does. ``sheet``
    also renders a contact sheet of the clips (slow: Cycles on the processor).
    Returns a manifest, also saved beside the files as ``<name>.manifest.json``.
    """
    timings = {}
    started = [time.perf_counter()]

    def lap(stage):
        now = time.perf_counter()
        timings[stage] = round(timings.get(stage, 0.0) + now - started[0], 3)
        started[0] = now

    if split_head is None:
        split_head = profile == "web"
    limits = _validate.PROFILES[profile] if _validate else {"triangles": _FALLBACK_BUDGET[profile]}
    export_service = build.mpfb("services.exportservice", "ExportService")
    object_service = build.mpfb("services.objectservice", "ObjectService")

    source = object_service.find_object_of_type_amongst_nearest_relatives(rig, "Basemesh")
    root = export_service.create_character_copy(source, name_suffix="_" + profile)
    basemesh = object_service.find_object_of_type_amongst_nearest_relatives(root, "Basemesh")
    export_rig = object_service.find_object_of_type_amongst_nearest_relatives(root, "Skeleton")
    export_service.bake_modifiers_remove_helpers(
        basemesh, bake_masks=True, bake_subdiv=False, remove_helpers=True, also_proxy=True)

    parts = [(basemesh, "Basemesh")] + [
        (ob, _object_type(ob)) for ob in export_rig.children_recursive if ob.type == 'MESH' and ob != basemesh
    ]
    for ob, _kind in parts:
        for modifier in [m for m in ob.modifiers if m.type != 'ARMATURE']:
            ob.modifiers.remove(modifier)
    lap("copy")

    available = [
        block.name for block in (basemesh.data.shape_keys.key_blocks if basemesh.data.shape_keys else ())
        if face.is_face_key(block.name) and block.value == 0.0
    ]
    wanted = set(face.keys_for(profile, available))
    rename = face.VRCHAT_NAMES if profile in _VRCHAT else {}
    settled = [(ob, _settle_shape_keys(ob, wanted)) for ob, _kind in parts]
    before = sum(_triangles(ob.data) for ob, _kind in parts)
    _fit_budget(parts, limits["triangles"])
    lap("reduce")
    for ob, state in settled:
        _restore_shape_keys(ob, state, rename)
    nudged = None
    if profile in _VRCHAT and "viseme_sil" in wanted:
        nudged = _nudge_silence(parts, rename["viseme_sil"])
    lap("shape_keys")

    size = min(limits.get("texture_size", 2048), 2048)
    name = build.stored(rig)["recipe"]["name"] + "_" + profile
    atlas = _build_atlas(parts, size, name + "_atlas")
    material = _atlas_material(name, atlas)
    for ob, _kind in parts:
        ob.data.materials.clear()
        ob.data.materials.append(material)
        _clean_weights(ob, export_rig)
    lap("atlas")

    if split_head:
        _mark_head(parts)
    _select_only([ob for ob, _kind in parts], basemesh)
    bpy.ops.object.join()
    # Joining leaves merged shape keys switched on; a rest face has them all off.
    if basemesh.data.shape_keys:
        for block in basemesh.data.shape_keys.key_blocks[1:]:
            block.value = 0.0
    basemesh.name = basemesh.data.name = name
    export_rig.name = name + ".rig"
    meshes = [basemesh]
    seam = 0
    if split_head:
        head, seam = _split_head(basemesh, name)
        meshes = [head, basemesh]
    lap("join")

    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, os.path.splitext(os.path.basename(path))[0])
    atlas.filepath_raw = stem + "_atlas.png"
    atlas.file_format = 'PNG'
    atlas.save()

    animations = motion.add_clips(export_rig, clips) if clips else []
    lap("clips")
    _select_only(meshes + [export_rig], export_rig)
    files = {}
    if "glb" in formats:
        files["glb"] = stem + ".glb"
        bpy.ops.export_scene.gltf(
            filepath=files["glb"], export_format='GLB', use_selection=True,
            export_skins=True, export_morph=True, export_apply=False,
            export_animations=bool(animations), export_animation_mode='NLA_TRACKS',
            export_morph_animation=False, export_force_sampling=True,
        )
        lap("write_glb")
    if "fbx" in formats:
        files["fbx"] = stem + ".fbx"
        # The FBX exporter bakes one take per strip but passes over muted tracks, which is
        # how the clips are parked; they are unmuted just for it.
        tracks = list(export_rig.animation_data.nla_tracks) if export_rig.animation_data else []
        for track in tracks:
            track.mute = False
        bpy.ops.export_scene.fbx(
            filepath=files["fbx"], use_selection=True, add_leaf_bones=False,
            object_types={'ARMATURE', 'MESH'}, bake_anim=bool(animations), bake_anim_use_all_actions=False,
            bake_anim_use_nla_strips=True, path_mode='COPY', embed_textures=True,
        )
        for track in tracks:
            track.mute = True
        motion._pose(export_rig, [])
        lap("write_fbx")

    manifest = {
        "schema": "elyan.person.export/1",
        "profile": profile,
        "recipe": build.stored(rig)["recipe"],
        "contract": build.stored(rig)["contract"],
        "triangles_before": before,
        "meshes": [
            {
                "name": ob.name, "triangles": _triangles(ob.data), "vertices": len(ob.data.vertices),
                "shape_keys": len(ob.data.shape_keys.key_blocks) - 1 if ob.data.shape_keys else 0,
            } for ob in meshes
        ],
        "split_head": bool(split_head),
        "seam_vertices": seam,
        "shape_keys": sorted(rename.get(name, name) for name in wanted),
        "animations": animations,
        "atlas": {"file": atlas.filepath_raw, "size": size},
        "files": {key: {"path": value, "bytes": os.path.getsize(value)} for key, value in files.items()},
    }
    if nudged:
        manifest["silence_nudge"] = {"part": nudged, "metres": face.SILENCE_NUDGE}
    if _validate:
        report = _validate.validate(meshes + [export_rig], profile)
        manifest["validation"] = {key: report[key] for key in ("totals", "limits", "failures", "passed")}
        lap("validate")
    if "glb" in files:
        manifest["round_trip"] = _round_trip(files["glb"])
        lap("round_trip")
    if sheet:
        manifest["contact_sheet"] = {
            "path": stem + ".sheet.png",
            "shots": contact_sheet(
                export_rig, stem + ".sheet.png",
                [(clip, motion.SHOWN_AT[clip]) for clip in (animations or motion.CLIPS)]),
        }
        lap("contact_sheet")
    manifest["timings"] = timings
    with open(stem + ".manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    if not keep:
        tracks = export_rig.animation_data.nla_tracks if export_rig.animation_data else ()
        for action in {strip.action for track in tracks for strip in track.strips if strip.action}:
            bpy.data.actions.remove(action)
        for ob in meshes + [export_rig]:
            bpy.data.objects.remove(ob)
    return manifest
