# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Export a person for a delivery target: one mesh, one material, one atlas, within budget.

Works on a copy, the character in the scene is left as it was. Shape keys that
are at zero (visemes, expressions) survive; those that are in use (the body's
shape) are baked in.
"""

import json
import os

import bpy
import numpy as np

from mathutils import kdtree

from . import build, face

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
    mesh.update()
    return (mixed, resting) if resting else None


def _restore_shape_keys(ob, settled, rename):
    """
    Put resting shape keys back. Each vertex takes the movement of the nearest
    vertex of the mesh as it was, which is itself wherever nothing was reduced.
    """
    if settled is None:
        return
    before, resting = settled
    mesh = ob.data
    tree = kdtree.KDTree(len(before))
    for index, co in enumerate(before):
        tree.insert(co, index)
    tree.balance()
    now = _coordinates(mesh.vertices, len(mesh.vertices))
    nearest = np.array([tree.find(co)[1] for co in now], dtype=np.int64)
    ob.shape_key_add(name="Basis")
    for name, offsets in resting.items():
        block = ob.shape_key_add(name=rename.get(name, name))
        block.data.foreach_set("co", (now + offsets[nearest]).ravel())


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
    bpy.ops.import_scene.gltf(filepath=path)
    arrived = [ob for ob in bpy.data.objects if ob not in before]
    # The importer adds a small unskinned mesh to draw bones with; it is not part of the asset.
    meshes = [ob for ob in arrived if ob.type == 'MESH' and ob.vertex_groups]
    result = {
        "triangles": sum(_triangles(ob.data) for ob in meshes),
        "meshes": len(meshes),
        "materials": len({m.name for ob in meshes for m in ob.data.materials if m}),
        "bones": sum(len(ob.data.bones) for ob in arrived if ob.type == 'ARMATURE'),
        "shape_keys": max(
            (len(ob.data.shape_keys.key_blocks) - 1 for ob in meshes if ob.data.shape_keys), default=0),
    }
    for ob in arrived:
        bpy.data.objects.remove(ob)
    return result


def export(rig, path, profile="web", formats=("glb",), keep=False):
    """
    Write ``path`` (extension is replaced per format) for a delivery profile.

    Returns a manifest, also saved beside the files as ``<name>.manifest.json``.
    """
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

    available = [
        block.name for block in (basemesh.data.shape_keys.key_blocks if basemesh.data.shape_keys else ())
        if face.is_face_key(block.name) and block.value == 0.0
    ]
    wanted = set(face.keys_for(profile, available))
    rename = face.VRCHAT_NAMES if profile in {"quest", "vrchat_pc"} else {}
    settled = [(ob, _settle_shape_keys(ob, wanted)) for ob, _kind in parts]
    before = sum(_triangles(ob.data) for ob, _kind in parts)
    _fit_budget(parts, limits["triangles"])
    for ob, state in settled:
        _restore_shape_keys(ob, state, rename)

    size = min(limits.get("texture_size", 2048), 2048)
    name = build.stored(rig)["recipe"]["name"] + "_" + profile
    atlas = _build_atlas(parts, size, name + "_atlas")
    material = _atlas_material(name, atlas)
    for ob, _kind in parts:
        ob.data.materials.clear()
        ob.data.materials.append(material)
        _clean_weights(ob, export_rig)

    _select_only([ob for ob, _kind in parts], basemesh)
    bpy.ops.object.join()
    basemesh.name = name
    export_rig.name = name + ".rig"

    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, os.path.splitext(os.path.basename(path))[0])
    atlas.filepath_raw = stem + "_atlas.png"
    atlas.file_format = 'PNG'
    atlas.save()

    _select_only([basemesh, export_rig], export_rig)
    files = {}
    if "glb" in formats:
        files["glb"] = stem + ".glb"
        bpy.ops.export_scene.gltf(
            filepath=files["glb"], export_format='GLB', use_selection=True,
            export_skins=True, export_morph=True, export_animations=False, export_apply=False,
        )
    if "fbx" in formats:
        files["fbx"] = stem + ".fbx"
        bpy.ops.export_scene.fbx(
            filepath=files["fbx"], use_selection=True, add_leaf_bones=False,
            object_types={'ARMATURE', 'MESH'}, bake_anim=False, path_mode='COPY', embed_textures=True,
        )

    manifest = {
        "schema": "elyan.person.export/1",
        "profile": profile,
        "recipe": build.stored(rig)["recipe"],
        "contract": build.stored(rig)["contract"],
        "triangles_before": before,
        "shape_keys": sorted(rename.get(name, name) for name in wanted),
        "atlas": {"file": atlas.filepath_raw, "size": size},
        "files": {key: {"path": value, "bytes": os.path.getsize(value)} for key, value in files.items()},
    }
    if _validate:
        report = _validate.validate([basemesh, export_rig], profile)
        manifest["validation"] = {key: report[key] for key in ("totals", "limits", "failures", "passed")}
    if "glb" in files:
        manifest["round_trip"] = _round_trip(files["glb"])
    with open(stem + ".manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    if not keep:
        for ob in (basemesh, export_rig):
            bpy.data.objects.remove(ob)
    return manifest
