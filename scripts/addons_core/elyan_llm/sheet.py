# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Contact sheet: fixed views of some objects in one labelled PNG.

Everything is rendered in a temporary scene with its own camera, light and
world, all removed afterwards, so the artist's scene is left as it was. Only
Cycles on the CPU is used: it needs no window and no graphics driver.
"""

import os
import shutil
import tempfile
import time

import bpy
import numpy as np

MAX_TILE = 400
MAX_SAMPLES = 24
DEFAULT_VIEWS = ("front", "back", "left", "right", "three_quarter")

# Where the camera stands, as seen from the subject. Blender's front view looks along +Y.
VIEWS = {
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "three_quarter": (1.0, -1.0, 0.5),
}

# 5x7 dot glyphs, one hex byte per row. Blender's own text drawing needs a GPU context.
_GLYPHS = {
    "A": "0E11111F111111", "B": "1E11111E11111E", "C": "0E11101010110E", "D": "1E11111111111E",
    "E": "1F10101E10101F", "F": "1F10101E101010", "G": "0E11101711110F", "H": "1111111F111111",
    "I": "0E04040404040E", "J": "0702020202120C", "K": "11121418141211", "L": "1010101010101F",
    "M": "111B1515111111", "N": "11191513111111", "O": "0E11111111110E", "P": "1E11111E101010",
    "Q": "0E11111115120D", "R": "1E11111E141211", "S": "0F10100E01011E", "T": "1F040404040404",
    "U": "1111111111110E", "V": "11111111110A04", "W": "11111115151B11",
    "X": "11110A040A1111", "Y": "11110A04040404", "Z": "1F01020408101F",
    "0": "0E11131519110E", "1": "040C040404040E", "2": "0E11010204081F", "3": "1F02040201110E",
    "4": "02060A121F0202", "5": "1F101E0101110E", "6": "0608101E11110E", "7": "1F010204080808",
    "8": "0E11110E11110E", "9": "0E11110F01020C",
    "-": "0000001F000000", "_": "0000000000001F", ".": "00000000000C0C", "/": "01010204081010",
    ":": "000C0C000C0C00", " ": "00000000000000",
}
_UNKNOWN = "1F11111111111F"
_GLYPH_SCALE = 2
_LABEL_HEIGHT = 7 * _GLYPH_SCALE + 8
_BACKGROUND = (0.11, 0.11, 0.12)
_INK = (0.92, 0.92, 0.92)


def _draw_text(canvas, text, left, top):
    """Write ``text`` into ``canvas`` (rows top to bottom), clipped at its right edge."""
    # Long labels on small tiles are drawn at half size rather than cut short.
    scale = _GLYPH_SCALE if left + len(text) * 6 * _GLYPH_SCALE <= canvas.shape[1] else 1
    step = 6 * scale
    for position, char in enumerate(text.upper()):
        x = left + position * step
        if x + step > canvas.shape[1]:
            break
        rows = bytes.fromhex(_GLYPHS.get(char, _UNKNOWN))
        for row, bits in enumerate(rows):
            for column in range(5):
                if bits & (0x10 >> column):
                    y0 = top + row * scale
                    x0 = x + column * scale
                    canvas[y0:y0 + scale, x0:x0 + scale, :3] = _INK


def _load_pixels(path):
    """A saved render as a float array, rows top to bottom."""
    image = bpy.data.images.load(path)
    try:
        width, height = image.size
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
    finally:
        bpy.data.images.remove(image)
    # Blender stores the bottom row first.
    return pixels.reshape(height, width, 4)[::-1]


def _save_pixels(canvas, path):
    height, width = canvas.shape[:2]
    image = bpy.data.images.new("_elyan_contact_sheet", width, height, alpha=False)
    try:
        image.pixels.foreach_set(np.ascontiguousarray(canvas[::-1]).reshape(-1))
        image.filepath_raw = path
        image.file_format = 'PNG'
        image.save()
    finally:
        bpy.data.images.remove(image)


def _world_corners(objects, depsgraph):
    from mathutils import Vector
    corners = []
    for ob in objects:
        ob_eval = ob.evaluated_get(depsgraph)
        corners += [ob_eval.matrix_world @ Vector(corner) for corner in ob_eval.bound_box]
    return corners


def _frame(camera, light, direction, corners, center=None, size=None):
    """Aim the orthographic camera along ``-direction`` so that ``corners`` (or ``size``) fill the tile."""
    from mathutils import Matrix, Vector
    direction = Vector(direction).normalized()
    # World Z stays up in the picture; looking straight down, +Y does.
    upward = Vector((0.0, 1.0, 0.0)) if abs(direction.z) > 0.999 else Vector((0.0, 0.0, 1.0))
    right = upward.cross(direction).normalized()
    up = direction.cross(right)
    # A camera looks down its own -Z, so its +Z is the way back to where it stands.
    rotation = Matrix((right, up, direction)).transposed().to_quaternion()
    if center is None:
        low = Vector(min(c[i] for c in corners) for i in range(3))
        high = Vector(max(c[i] for c in corners) for i in range(3))
        center = (low + high) * 0.5
    if size is None:
        size = 2.0 * max(max(abs((c - center).dot(axis)) for c in corners) for axis in (right, up))
    reach = max((c - center).length for c in corners) if corners else size
    distance = reach * 2.0 + size + 1.0
    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = rotation
    camera.location = center + direction * distance
    camera.data.ortho_scale = max(size * 1.08, 1e-4)
    camera.data.clip_start = 0.01
    camera.data.clip_end = distance * 2.0 + reach + 10.0
    # The light comes from over the camera's shoulder, so every view is lit alike.
    light.rotation_mode = 'QUATERNION'
    light.rotation_quaternion = (-(direction + up * 0.45 + right * 0.35)).to_track_quat('-Z', 'Y')


def _closeup_target(spec, depsgraph):
    """Centre and default size of the close-up, from a bone or an object."""
    if spec.get("bone"):
        rig = bpy.data.objects.get(spec.get("armature") or "")
        if rig is None or rig.type != 'ARMATURE':
            raise ValueError("close-up of a bone needs 'armature', the name of its rig")
        bone = rig.pose.bones.get(spec["bone"])
        if bone is None:
            raise ValueError("{!r} has no bone named {!r}".format(rig.name, spec["bone"]))
        rig_eval = rig.evaluated_get(depsgraph)
        head = rig_eval.matrix_world @ bone.head
        tail = rig_eval.matrix_world @ bone.tail
        return (head + tail) * 0.5, (tail - head).length * 3.0, spec["bone"]
    ob = bpy.data.objects.get(spec.get("object") or "")
    if ob is None:
        raise ValueError("close-up needs 'bone' with 'armature', or 'object'")
    corners = _world_corners([ob], depsgraph)
    center = sum(corners, corners[0] * 0.0) / len(corners)
    return center, 2.0 * max((c - center).length for c in corners), ob.name


def contact_sheet(args):
    """See ``commands.cmd_contact_sheet`` for the arguments."""
    path = args.get("path")
    if not path:
        raise ValueError("contact_sheet needs 'path'")
    path = os.path.abspath(os.path.expanduser(path))
    names = args.get("names") or args.get("name")
    names = [names] if isinstance(names, str) else list(names or ())
    if not names:
        raise ValueError("contact_sheet needs 'names'")
    subjects = []
    for name in names:
        ob = bpy.data.objects.get(name)
        if ob is None:
            raise ValueError("no object named {!r}".format(name))
        # A rig or an empty stands for whatever hangs under it.
        for item in (ob, *ob.children_recursive):
            if item not in subjects:
                subjects.append(item)
    views = list(args.get("views") or DEFAULT_VIEWS)
    for view in views:
        if view not in VIEWS:
            raise ValueError("unknown view {!r}; known: {:s}".format(view, ", ".join(sorted(VIEWS))))
    tile = max(32, min(int(args.get("tile") or 256), MAX_TILE))
    samples = max(1, min(int(args.get("samples") or 8), MAX_SAMPLES))
    columns = max(1, int(args.get("columns") or 3))
    closeup = args.get("closeup")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    started = time.time()
    source = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    corners = _world_corners(subjects, depsgraph)
    shots = [(view, VIEWS[view], None, None) for view in views]
    if closeup:
        center, size, label = _closeup_target(closeup, depsgraph)
        view = closeup.get("view") or "front"
        if view not in VIEWS:
            raise ValueError("unknown close-up view {!r}".format(view))
        shots.append(("close " + label, VIEWS[view], center, float(closeup.get("size") or size)))

    scene = camera_data = camera = light_data = light = world = folder = None
    # Rendering creates this image if the file has none yet; then it is ours to remove.
    had_result = "Render Result" in bpy.data.images
    try:
        scene = bpy.data.scenes.new("_elyan_contact_sheet")
        for ob in subjects:
            scene.collection.objects.link(ob)
        camera_data = bpy.data.cameras.new("_elyan_contact_sheet")
        camera_data.type = 'ORTHO'
        camera = bpy.data.objects.new("_elyan_contact_sheet_camera", camera_data)
        light_data = bpy.data.lights.new("_elyan_contact_sheet", 'SUN')
        light_data.energy = 2.0
        light_data.angle = 0.3
        light = bpy.data.objects.new("_elyan_contact_sheet_light", light_data)
        scene.collection.objects.link(camera)
        scene.collection.objects.link(light)
        scene.camera = camera

        # Flat grey surroundings fill the shadows, so no side of the subject goes black.
        world = bpy.data.worlds.new("_elyan_contact_sheet")
        if world.node_tree is None:
            world.use_nodes = True
        for node in world.node_tree.nodes:
            if node.type == 'BACKGROUND':
                node.inputs["Color"].default_value = (0.5, 0.51, 0.53, 1.0)
                node.inputs["Strength"].default_value = 0.6
        scene.world = world

        render = scene.render
        render.engine = 'CYCLES'
        # Never the GPU: the sheet must be safe to make on any machine, window or not.
        scene.cycles.device = 'CPU'
        scene.cycles.samples = samples
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 3
        render.resolution_x = render.resolution_y = tile
        render.resolution_percentage = 100
        render.image_settings.file_format = 'PNG'
        render.image_settings.color_mode = 'RGBA'
        render.use_file_extension = False
        scene.view_settings.view_transform = 'Standard'
        scene.display_settings.display_device = source.display_settings.display_device
        # Same frame as the artist's scene, so animated subjects are shown as they stand now.
        # Assigned rather than ``frame_set``: the render evaluates it, and add-ons' frame
        # handlers are not run an extra time on a scene they know nothing about.
        scene.frame_current = source.frame_current
        scene.frame_subframe = source.frame_subframe

        folder = tempfile.mkdtemp(prefix=".contact_sheet_", dir=os.path.dirname(path))
        tiles = []
        for index, (label, direction, center, size) in enumerate(shots):
            _frame(camera, light, direction, corners, center, size)
            render.filepath = os.path.join(folder, "{:02d}.png".format(index))
            bpy.ops.render.render(write_still=True, scene=scene.name)
            tiles.append((label, _load_pixels(render.filepath)))
    finally:
        # Removing the scene only unlinks the subjects from it; they stay in the artist's scene.
        if scene is not None:
            bpy.data.scenes.remove(scene)
        for block, owner in ((camera, bpy.data.objects), (light, bpy.data.objects), (camera_data, bpy.data.cameras),
                             (light_data, bpy.data.lights), (world, bpy.data.worlds)):
            if block is not None:
                owner.remove(block)
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)
        if not had_result and "Render Result" in bpy.data.images:
            bpy.data.images.remove(bpy.data.images["Render Result"])

    rows = -(-len(tiles) // columns)
    columns = min(columns, len(tiles))
    cell_height = tile + _LABEL_HEIGHT
    canvas = np.empty((rows * cell_height, columns * tile, 4), dtype=np.float32)
    canvas[:, :, :3] = _BACKGROUND
    canvas[:, :, 3] = 1.0
    for index, (label, pixels) in enumerate(tiles):
        top = (index // columns) * cell_height
        left = (index % columns) * tile
        _draw_text(canvas[top:top + _LABEL_HEIGHT, left:left + tile], label.replace("_", " "), 4, 4)
        canvas[top + _LABEL_HEIGHT:top + cell_height, left:left + tile, :3] = pixels[:, :, :3]
    _save_pixels(canvas, path)

    return {
        "path": path,
        "size": [columns * tile, rows * cell_height],
        "views": [label for label, _pixels in tiles],
        "objects": [ob.name for ob in subjects],
        "hidden_in_render": [ob.name for ob in subjects if ob.hide_render],
        "tile": tile,
        "samples": samples,
        "engine": "CYCLES",
        "device": "CPU",
        "seconds": round(time.time() - started, 2),
    }
