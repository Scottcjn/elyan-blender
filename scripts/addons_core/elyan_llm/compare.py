# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Silhouettes as numbers: what was built against a reference picture, without a render.

A garment is built from a contract of numbers and judged against a photo. Reading
widths off the photo by hand and rendering once per guess is slow, so this
projects the evaluated triangles of the objects straight into a mask, pulls a
mask out of the reference image, fits one onto the other and reports how wide
each is at a set of heights.

The masks are filled with numpy on the CPU. No render engine and no GPU are
touched: a silhouette needs neither, and EEVEE and Workbench cannot run on a
machine without a usable graphics driver.
"""

import json
import os
import time

import bpy
import numpy as np

DEFAULT_SIZE = 512
MAX_SIZE = 4096
DEFAULT_ROWS = 17
MAX_ROWS = 512
DEFAULT_MARGIN = 0.05
DEFAULT_TOLERANCE = 0.1
# Row spans are built for this many (triangle, row) pairs at a time to bound memory.
_CHUNK_PAIRS = 2_000_000
# A flood that has not settled after this many sweeps is a maze, not a backdrop.
_MAX_SWEEPS = 400

# Where the viewer stands, as seen from the subject, and which way is up in the
# picture. The same convention as the contact sheet: the front view looks along +Y.
VIEWS = {
    "front": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "back": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "left": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "right": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "top": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
}

_MESH_LIKE = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}
_ANCHORS = ("center", "top", "bottom", "left", "right")
_SCALE_BY = ("height", "width", "area")

_BACKGROUND = (0.11, 0.11, 0.12)
_ONLY_A = (1.0, 0.45, 0.10)
_ONLY_B = (0.15, 0.50, 1.0)
_BOTH = (0.85, 0.85, 0.85)
_WORST = (1.0, 0.95, 0.2)

PHOTO_NOTE = (
    "Approximate: a photo has perspective, a lens, a pose and a background that was guessed; "
    "the scene mask is orthographic. Read the numbers as a guide to which rows to change and "
    "by roughly how much, look at the overlay, and do not treat them as a pass or a fail."
)


# -----------------------------------------------------------------------------
# Images

def _load_pixels(path):
    """An image file as a float array of RGBA, rows top to bottom."""
    if not os.path.isfile(path):
        raise ValueError("no image at {:s}".format(path))
    image = bpy.data.images.load(path, check_existing=False)
    try:
        width, height = image.size
        if width == 0 or height == 0:
            raise ValueError("{:s} could not be read as an image".format(path))
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
    finally:
        bpy.data.images.remove(image)
    # Blender stores the bottom row first.
    return pixels.reshape(height, width, 4)[::-1]


def _save_pixels(canvas, path):
    """Write an RGB float array, rows top to bottom, as a PNG."""
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    height, width = canvas.shape[:2]
    rgba = np.ones((height, width, 4), dtype=np.float32)
    rgba[:, :, :3] = canvas[::-1]
    image = bpy.data.images.new("_elyan_compare", width, height, alpha=False)
    try:
        image.pixels.foreach_set(rgba.reshape(-1))
        image.filepath_raw = path
        image.file_format = 'PNG'
        image.save()
    finally:
        bpy.data.images.remove(image)
    return path


def _save_mask(mask, path):
    canvas = np.zeros((*mask.shape, 3), dtype=np.float32)
    canvas[mask] = 1.0
    return _save_pixels(canvas, path)


def _crop_box(crop, width, height):
    if not isinstance(crop, (list, tuple)) or len(crop) != 4:
        raise ValueError("crop is [x0, y0, x1, y1] from the top left corner, in pixels or as fractions of the image")
    box = [float(v) for v in crop]
    if max(box) <= 1.0:
        box = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("crop {!r} leaves nothing of a {:d}x{:d} image".format(crop, width, height))
    return x0, y0, x1, y1


def _sweep(candidate, reached):
    """Spread ``reached`` along every unbroken run of ``candidate`` in each row."""
    height, width = candidate.shape
    breaks = ~candidate
    # A run must not continue from the end of one row into the start of the next.
    breaks[:, 0] = True
    labels = np.cumsum(breaks.reshape(-1))
    hit = np.bincount(labels, weights=reached.reshape(-1), minlength=int(labels[-1]) + 1) > 0
    return (hit[labels] & candidate.reshape(-1)).reshape(height, width)


def _connected_to_border(candidate):
    """The part of ``candidate`` that can be walked to from the image border."""
    reached = np.zeros_like(candidate)
    reached[0], reached[-1] = candidate[0], candidate[-1]
    reached[:, 0], reached[:, -1] = candidate[:, 0], candidate[:, -1]
    count = -1
    # Whole runs are filled per sweep, so a plain backdrop settles in a few passes
    # where a pixel-by-pixel flood would need one pass per pixel of distance.
    for _ in range(_MAX_SWEEPS):
        reached = _sweep(candidate, reached)
        reached = _sweep(np.ascontiguousarray(candidate.T), np.ascontiguousarray(reached.T)).T
        total = int(reached.sum())
        if total == count:
            break
        count = total
    return reached


def _vertical_median(column, window):
    """A border column with anything short-lived (a hand, a hem touching the edge) voted out."""
    pad = window // 2
    padded = np.pad(column, ((pad, pad), (0, 0)), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window, axis=0), axis=-1)


def _background_distance(rgb, background):
    """How far each pixel's colour is from the backdrop, and a description of the backdrop."""
    height, width = rgb.shape[:2]
    if isinstance(background, (list, tuple)):
        if len(background) != 3:
            raise ValueError("background as a colour is [r, g, b], each 0..1")
        colour = np.array(background, dtype=np.float32)
        return np.abs(rgb - colour).max(axis=2), [round(float(v), 3) for v in colour]
    if background == "rows":
        # A studio backdrop shades from top to bottom: model each row from its two ends.
        window = max(3, (height // 16) | 1)
        left = _vertical_median(rgb[:, 0], window)[:, None, :]
        right = _vertical_median(rgb[:, -1], window)[:, None, :]
        blend = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
        return np.abs(rgb - (left * (1.0 - blend) + right * blend)).max(axis=2), "rows"
    if background in (None, "border"):
        border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
        colour = np.median(border, axis=0)
        return np.abs(rgb - colour).max(axis=2), [round(float(v), 3) for v in colour]
    raise ValueError("background is 'border', 'rows' or a colour [r, g, b]")


def _image_mask(path, crop=None, tolerance=DEFAULT_TOLERANCE, background=None, limit=1024):
    """
    The subject of an image as a mask, and how it was found.

    Transparency is used when the image has any; a pure black and white image is
    taken as a mask already; otherwise whatever is the backdrop's colour and can
    be reached from the border is removed.
    """
    path = os.path.abspath(os.path.expanduser(path))
    pixels = _load_pixels(path)
    full_height, full_width = pixels.shape[:2]
    info = {"path": path, "image_size": [full_width, full_height]}
    if crop is not None:
        x0, y0, x1, y1 = _crop_box(crop, full_width, full_height)
        pixels = pixels[y0:y1, x0:x1]
        info["crop"] = [x0, y0, x1, y1]
    # A phone photo has far more pixels than a silhouette comparison can use.
    step = max(1, -(-max(pixels.shape[:2]) // max(64, limit)))
    pixels = pixels[::step, ::step]
    info["step"] = step
    rgb, alpha = pixels[:, :, :3], pixels[:, :, 3]
    grey = rgb.mean(axis=2)
    if float(alpha.min()) < 0.5:
        mask = alpha > 0.5
        info["method"] = "alpha"
    elif bool(np.all((rgb < 0.02) | (rgb > 0.98))) and float(grey.min()) < 0.5:
        mask = grey > 0.5
        info["method"] = "mask"
    else:
        distance, described = _background_distance(rgb, background)
        mask = ~_connected_to_border(distance <= float(tolerance))
        info.update({"method": "background", "background": described, "tolerance": float(tolerance)})
    covered = float(mask.mean())
    info["coverage"] = round(covered, 4)
    if not mask.any():
        raise ValueError("{:s}: no subject found ({:s}); try a smaller 'tolerance' or a 'crop'".format(
            os.path.basename(path), info["method"]))
    if info["method"] == "background" and covered > 0.95:
        raise ValueError("{:s}: the whole image came out as subject; raise 'tolerance', set 'background' "
                         "to 'rows' or a colour, or crop to the subject".format(os.path.basename(path)))
    return np.ascontiguousarray(mask), info


# -----------------------------------------------------------------------------
# Scene

def _subjects(names):
    names = [names] if isinstance(names, str) else list(names or ())
    subjects = []
    for name in names:
        ob = bpy.data.objects.get(name)
        if ob is None:
            raise ValueError("no object named {!r}".format(name))
        # A rig or an empty stands for whatever hangs under it.
        for item in (ob, *ob.children_recursive):
            if item.type in _MESH_LIKE and item not in subjects:
                subjects.append(item)
    if not subjects:
        raise ValueError("nothing with a surface among {!r}".format(names))
    return subjects


def _world_triangles(subjects):
    """Triangles of the objects as shown: after modifiers, shape keys and pose. ``(count, 3, 3)`` in world space."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    blocks = []
    for ob in subjects:
        ob_eval = ob.evaluated_get(depsgraph)
        try:
            mesh = ob_eval.to_mesh()
        except RuntimeError:
            mesh = None
        if mesh is None:
            continue
        try:
            mesh.calc_loop_triangles()
            count = len(mesh.loop_triangles)
            if count == 0:
                continue
            coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", coords)
            corners = np.empty(count * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", corners)
        finally:
            ob_eval.to_mesh_clear()
        matrix = np.array(ob_eval.matrix_world, dtype=np.float64)
        coords = coords.reshape(-1, 3).astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
        blocks.append(coords[corners].reshape(count, 3, 3))
    if not blocks:
        raise ValueError("{:s}: no faces to project".format(", ".join(ob.name for ob in subjects)))
    return np.concatenate(blocks)


def _basis(view, direction, up):
    """Picture axes in world space: ``right``, ``up`` and the way back to the viewer."""
    if direction is None:
        name = view or "front"
        if name not in VIEWS:
            raise ValueError("unknown view {!r}; known: {:s}, or give 'direction'".format(name, ", ".join(VIEWS)))
        direction, default_up = VIEWS[name]
    else:
        if isinstance(direction, str):
            if direction not in VIEWS:
                raise ValueError("unknown direction {!r}; known: {:s}".format(direction, ", ".join(VIEWS)))
            direction, default_up = VIEWS[direction]
        else:
            default_up = (0.0, 0.0, 1.0)
    toward = np.array(direction, dtype=np.float64)
    if toward.shape != (3,) or not np.linalg.norm(toward) > 1e-9:
        raise ValueError("direction is three numbers, where the viewer stands as seen from the subject")
    toward /= np.linalg.norm(toward)
    upward = np.array(up if up is not None else default_up, dtype=np.float64)
    if upward.shape != (3,):
        raise ValueError("up is three numbers")
    if abs(float(upward @ toward)) > 0.999 * np.linalg.norm(upward):
        # Looking straight along the requested up: fall back as the contact sheet does.
        upward = np.array((0.0, 1.0, 0.0) if abs(toward[2]) > 0.999 else (0.0, 0.0, 1.0))
    right = np.cross(upward, toward)
    right /= np.linalg.norm(right)
    return right, np.cross(toward, right), toward


def _frame(points, basis, window, margin, size):
    """Which part of the picture plane the mask covers, and at how many metres per pixel."""
    right, up, _toward = basis
    if window is not None:
        if not isinstance(window, dict) or not {"center", "width", "height"} <= set(window):
            raise ValueError("window is {\"center\": [x, y, z], \"width\": metres, \"height\": metres}")
        center = np.array(window["center"], dtype=np.float64)
        if center.shape != (3,):
            raise ValueError("window center is a world position [x, y, z]")
        mid_u, mid_v = float(center @ right), float(center @ up)
        width, height = float(window["width"]), float(window["height"])
    else:
        u, v = points @ right, points @ up
        mid_u, mid_v = float(u.min() + u.max()) * 0.5, float(v.min() + v.max()) * 0.5
        width, height = float(np.ptp(u)), float(np.ptp(v))
        pad = 2.0 * float(margin) * max(width, height)
        width, height = width + pad, height + pad
    if not (width > 0.0 and height > 0.0):
        raise ValueError("the frame has no size: width {:g}, height {:g}".format(width, height))
    pixel = max(width, height) / size
    columns, rows = max(1, int(round(width / pixel))), max(1, int(round(height / pixel)))
    return {
        "left": mid_u - width * 0.5,
        "top": mid_v + height * 0.5,
        "pixel": pixel,
        "shape": (rows, columns),
        "window": {
            "center": [round(float(c), 6) for c in right * mid_u + up * mid_v],
            "width": round(width, 6),
            "height": round(height, 6),
        },
    }


def _rasterise(triangles, shape):
    """
    Fill triangles given in pixel coordinates (x right, y down) into a mask.

    A pixel is set when its centre lies inside a triangle. Each triangle becomes
    one span per row it crosses, and the spans are painted with a running sum,
    so the cost follows the triangle count and not triangles times pixels.
    """
    height, width = shape
    x, y = triangles[:, :, 0], triangles[:, :, 1]
    first = np.clip(np.ceil(y.min(axis=1) - 0.5), 0, height).astype(np.int64)
    last = np.clip(np.floor(y.max(axis=1) - 0.5), -1, height - 1).astype(np.int64)
    counts = np.maximum(last - first + 1, 0)
    edges = np.zeros(height * (width + 1), dtype=np.int64)
    ends = np.cumsum(counts)
    start = 0
    while start < len(counts):
        # As many triangles as fit the pair budget, and always at least one.
        stop = int(np.searchsorted(ends, (ends[start - 1] if start else 0) + _CHUNK_PAIRS, side="right"))
        stop = max(stop, start + 1)
        count = counts[start:stop]
        total = int(count.sum())
        if total:
            index = np.repeat(np.arange(start, stop), count)
            offset = np.cumsum(count) - count
            row = first[index] + np.arange(total) - np.repeat(offset, count)
            centre = row + 0.5
            low = np.full(total, np.inf)
            high = np.full(total, -np.inf)
            for a, b in ((0, 1), (1, 2), (2, 0)):
                xa, ya, xb, yb = x[index, a], y[index, a], x[index, b], y[index, b]
                # Walk every edge from its upper end, so two triangles sharing an
                # edge compute the very same crossing and leave no gap between them.
                swap = ya > yb
                xa, xb = np.where(swap, xb, xa), np.where(swap, xa, xb)
                ya, yb = np.where(swap, yb, ya), np.where(swap, ya, yb)
                rise = yb - ya
                crossing = (rise > 0.0) & (centre >= ya) & (centre <= yb)
                at = xa + (centre - ya) * (xb - xa) / np.where(rise > 0.0, rise, 1.0)
                low = np.where(crossing, np.minimum(low, at), low)
                high = np.where(crossing, np.maximum(high, at), high)
            keep = np.isfinite(low) & np.isfinite(high)
            row, low, high = row[keep], low[keep], high[keep]
            left = np.clip(np.ceil(low - 0.5), 0, width).astype(np.int64)
            beyond = np.clip(np.floor(high - 0.5) + 1, 0, width).astype(np.int64)
            keep = beyond > left
            base = row[keep] * (width + 1)
            edges += np.bincount(base + left[keep], minlength=len(edges))
            edges -= np.bincount(base + beyond[keep], minlength=len(edges))
        start = stop
    return np.cumsum(edges.reshape(height, width + 1), axis=1)[:, :width] > 0


def _scene_mask(triangles, basis, frame):
    right, up, _toward = basis
    pixels = np.empty((len(triangles), 3, 2), dtype=np.float64)
    pixels[:, :, 0] = (triangles @ right - frame["left"]) / frame["pixel"]
    pixels[:, :, 1] = (frame["top"] - triangles @ up) / frame["pixel"]
    return _rasterise(pixels, frame["shape"])


# -----------------------------------------------------------------------------
# Fitting

def _bbox(mask):
    """``(x0, y0, x1, y1)`` of the filled pixels, the far edges exclusive."""
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    return float(columns[0]), float(rows[0]), float(columns[-1] + 1), float(rows[-1] + 1)


def _resample(mask, shape, scale, shift):
    """``mask`` drawn into a picture of ``shape`` after ``x' = scale * x + shift``."""
    height, width = shape
    columns = np.floor((np.arange(width) + 0.5 - shift[0]) / scale).astype(np.int64)
    rows = np.floor((np.arange(height) + 0.5 - shift[1]) / scale).astype(np.int64)
    inside_x = (columns >= 0) & (columns < mask.shape[1])
    inside_y = (rows >= 0) & (rows < mask.shape[0])
    out = np.zeros(shape, dtype=bool)
    out[np.ix_(inside_y, inside_x)] = mask[np.ix_(rows[inside_y], columns[inside_x])]
    return out


def _iou(a, b):
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def _placement(box_a, box_b, scale, anchor):
    """The shift that puts ``box_b``, scaled, onto ``box_a`` at the anchored edge."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    shift_x = (ax0 + ax1) * 0.5 - scale * (bx0 + bx1) * 0.5
    shift_y = (ay0 + ay1) * 0.5 - scale * (by0 + by1) * 0.5
    if anchor == "top":
        shift_y = ay0 - scale * by0
    elif anchor == "bottom":
        shift_y = ay1 - scale * by1
    elif anchor == "left":
        shift_x = ax0 - scale * bx0
    elif anchor == "right":
        shift_x = ax1 - scale * bx1
    return shift_x, shift_y


def _fit(mask_a, mask_b, mode, scale_by, anchor, refine):
    """``mask_b`` moved into the picture of ``mask_a``, and how it was moved."""
    if anchor not in _ANCHORS:
        raise ValueError("anchor is one of: {:s}".format(", ".join(_ANCHORS)))
    if scale_by not in _SCALE_BY:
        raise ValueError("scale_by is one of: {:s}".format(", ".join(_SCALE_BY)))
    if mode == "none":
        if mask_a.shape != mask_b.shape:
            raise ValueError("fit=none needs both pictures the same size, got {:d}x{:d} and {:d}x{:d}; "
                             "use fit=bbox".format(*mask_a.shape[::-1], *mask_b.shape[::-1]))
        return mask_b, {"mode": "none", "scale": 1.0, "offset": [0.0, 0.0]}
    if mode != "bbox":
        raise ValueError("fit is 'bbox' or 'none'")
    box_a, box_b = _bbox(mask_a), _bbox(mask_b)
    wide_a, high_a = box_a[2] - box_a[0], box_a[3] - box_a[1]
    wide_b, high_b = box_b[2] - box_b[0], box_b[3] - box_b[1]
    scale = {
        "height": high_a / high_b,
        "width": wide_a / wide_b,
        "area": ((wide_a * high_a) / (wide_b * high_b)) ** 0.5,
    }[scale_by]
    shift = _placement(box_a, box_b, scale, anchor)

    def place(factor, dx, dy):
        scaled = scale * factor
        sx, sy = _placement(box_a, box_b, scaled, anchor)
        return scaled, (sx + dx, sy + dy)

    best = (1.0, 0.0, 0.0)
    if refine:
        # Bounding boxes are thrown by one stray hand or shadow. From that start,
        # nudge scale and position for as long as the overlap keeps growing.
        score = _iou(mask_a, _resample(mask_b, mask_a.shape, scale, shift))
        steps = [0.02, 0.02 * high_a, 0.02 * high_a]
        for _ in range(7):
            improved = True
            while improved:
                improved = False
                for axis in range(3):
                    for sign in (1.0, -1.0):
                        trial = list(best)
                        trial[axis] += sign * steps[axis]
                        trial_score = _iou(mask_a, _resample(mask_b, mask_a.shape, *place(*trial)))
                        if trial_score > score + 1e-9:
                            best, score, improved = tuple(trial), trial_score, True
            steps = [step * 0.5 for step in steps]
        scale, shift = place(*best)
    fitted = _resample(mask_b, mask_a.shape, scale, shift)
    return fitted, {
        "mode": "bbox",
        "scale_by": scale_by,
        "anchor": anchor,
        "refined": bool(refine),
        "scale": round(float(scale), 6),
        "offset": [round(float(shift[0]), 3), round(float(shift[1]), 3)],
    }


# -----------------------------------------------------------------------------
# Measuring

def _spans(mask):
    """For every row: whether it has any pixel, its leftmost edge and its rightmost edge (exclusive)."""
    filled = mask.any(axis=1)
    left = np.argmax(mask, axis=1).astype(np.float64)
    right = (mask.shape[1] - np.argmax(mask[:, ::-1], axis=1)).astype(np.float64)
    return filled, left, right


def _sample_rows(box, band, count):
    """``count`` row indices evenly spaced over the band of the silhouette's height."""
    top, bottom = box[1], box[3] - 1.0
    fractions = np.linspace(0.0, 1.0, count) if count > 1 else np.array([0.5])
    inside = band[0] + fractions * (band[1] - band[0])
    return fractions, np.clip(np.rint(top + inside * (bottom - top)), top, bottom).astype(np.int64)


def _asymmetry(mask, band_rows, unit, axis=None):
    """How lopsided a silhouette is about a vertical axis, over the rows in the band."""
    filled, left, right = _spans(mask)
    rows = np.arange(band_rows[0], band_rows[1] + 1)
    rows = rows[filled[rows]]
    if len(rows) == 0:
        return None
    middle = (left[rows] + right[rows]) * 0.5
    if axis is None:
        # The median row centre: an arm held out on one side must not move the axis.
        axis = float(np.median(middle))
    lean = 2.0 * (middle - axis)
    worst = int(np.argmax(np.abs(lean)))
    # Mirror about the axis; a column c holds [c, c + 1), so it lands on 2 * axis - 1 - c.
    columns = np.rint(2.0 * axis - 1.0 - np.arange(mask.shape[1])).astype(np.int64)
    inside = (columns >= 0) & (columns < mask.shape[1])
    mirrored = np.zeros_like(mask)
    mirrored[:, inside] = mask[:, columns[inside]]
    span = slice(band_rows[0], band_rows[1] + 1)
    return {
        "axis_px": round(axis, 2),
        "mean_right_minus_left": round(float(lean.mean()) * unit, 4),
        "max_right_minus_left": round(float(lean[worst]) * unit, 4),
        "max_at_s": round(float(rows[worst] - band_rows[0]) / max(band_rows[1] - band_rows[0], 1), 4),
        "mirror_iou": round(_iou(mask[span], mirrored[span]), 4),
    }


def _normalised(profile):
    """A profile as her contract writes it: 0 at the first row's width, 1 at the last row's."""
    usable = [(s, w) for s, w in profile if w is not None]
    if len(usable) < 2 or abs(usable[-1][1] - usable[0][1]) < 1e-9:
        return None
    first, last = usable[0][1], usable[-1][1]
    return [[s, round((w - first) / (last - first), 4)] for s, w in usable]


def _overlay(mask_a, mask_b, worst_row):
    canvas = np.empty((*mask_a.shape, 3), dtype=np.float32)
    canvas[:] = _BACKGROUND
    canvas[mask_a & ~mask_b] = _ONLY_A
    canvas[mask_b & ~mask_a] = _ONLY_B
    canvas[mask_a & mask_b] = _BOTH
    if worst_row is not None:
        canvas[worst_row, ::2] = _WORST
    return canvas


# -----------------------------------------------------------------------------
# Command

def _with_view(args):
    """The arguments with a named view from the ``views`` file filled in underneath them."""
    path = args.get("views")
    if not path:
        return args
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, encoding="utf-8") as fh:
        table = json.load(fh)
    if not isinstance(table, dict):
        raise ValueError("{:s} must hold one JSON object: view name -> settings".format(path))
    name = args.get("view")
    if name is None:
        raise ValueError("'views' needs 'view', one of: {:s}".format(", ".join(sorted(table))))
    if name not in table:
        if name in VIEWS:
            return args
        raise ValueError("no view {!r} in {:s}; it has: {:s}".format(name, path, ", ".join(sorted(table))))
    entry = dict(table[name])
    if "objects" in entry:
        entry.setdefault("names", entry.pop("objects"))
    folder = os.path.dirname(path)
    for key in ("reference", "other", "image"):
        # Paths in the file are relative to it, so the folder can be moved as a whole.
        if isinstance(entry.get(key), str) and not os.path.isabs(os.path.expanduser(entry[key])):
            entry[key] = os.path.join(folder, entry[key])
    # A view that names no direction of its own is one of the built-in ones.
    entry.setdefault("direction", name if name in VIEWS else "front")
    merged = dict(entry)
    merged.update({key: value for key, value in args.items() if key not in ("view", "views")})
    return merged


def _band(args, box, frame):
    """The part of the silhouette's height to measure, as two fractions from its top."""
    band = args.get("band")
    if args.get("band_z") is not None:
        if frame is None:
            raise ValueError("band_z is in scene units and needs a scene silhouette; use 'band' for images")
        high, low = sorted((float(v) for v in args["band_z"]), reverse=True)
        top, bottom = box[1], box[3] - 1.0
        rows = [(frame["top"] - z) / frame["pixel"] - 0.5 for z in (high, low)]
        band = [(row - top) / max(bottom - top, 1.0) for row in rows]
    if band is None:
        return 0.0, 1.0
    if not isinstance(band, (list, tuple)) or len(band) != 2:
        raise ValueError("band is [from, to] as fractions of the silhouette's height, 0 = top, 1 = bottom")
    first, last = (min(1.0, max(0.0, float(v))) for v in band)
    if not last > first:
        raise ValueError("band {!r} is empty".format(band))
    return first, last


def cmd_compare(args):
    """
    Silhouette of objects as numbers, alone or against a reference picture. No render, CPU only.

    A = what you built. Give one of:
      ``names``   objects (a rig or empty stands for its children). Their triangles
                  after modifiers, shape keys and pose are projected orthographically.
      ``image``   an image file instead (``image_crop`` as ``crop`` below).
    B = what to compare with. Optional; give one of:
      ``reference``  a photo or drawing. Fitted onto A by scale + shift. APPROXIMATE.
      ``other``      an image with the same framing as A (a mask written by ``out``,
                     a before/after render). Compared pixel for pixel unless ``fit=bbox``.
      ``against``    other objects, drawn in the same frame as ``names``.
    Without B you get A's width profile only.

    View (scene silhouettes): ``view`` = front (default, viewer at -Y), back, left,
    right, top; or ``direction`` [x, y, z] = where the viewer stands as seen from the
    subject, with ``up`` [x, y, z] (default +Z). Picture right is ``up x direction``:
    in the front view +X (the character's left) is on the right.
    Frame: the objects' bounds plus ``margin`` (0.05 of the longer side), or
    ``window`` = {"center": [x, y, z], "width": m, "height": m}. The reply holds the
    window used; pass it back so two runs cover exactly the same area.
    ``size`` = pixels on the longer side (512, at most 4096).

    Reading an image (``reference``, ``other``, ``image``): transparency is used if
    there is any; a pure black/white image is a mask (white = subject); otherwise
    everything within ``tolerance`` (0.1, per channel, 0..1) of the backdrop and
    reachable from the border is removed. ``background`` = "border" (one colour,
    the median of the border; default), "rows" (each row shaded between its left
    and right ends: studio backdrops that darken from top to bottom) or [r, g, b].
    ``crop`` = [x0, y0, x1, y1] from the top left, in pixels, or fractions if all
    are <= 1. The reply's ``b.coverage`` and ``b_mask`` (written when ``b_out`` is
    given) show whether the subject was found: LOOK at it before trusting numbers.

    Fit (B onto A): ``fit`` = "bbox" (default for ``reference``) scales uniformly and
    shifts so the bounding boxes meet; "none" (default for ``other``, ``against``).
    ``scale_by`` = height (default), width or area picks which dimension must match.
    ``anchor`` = center (default), top, bottom, left, right: that edge of the two
    boxes coincides, the other axis is centred (matters when scale_by is not the
    anchored dimension). ``refine=true`` then nudges scale and position to the best
    overlap: use it when a hand or shadow throws the bounding box.

    Rows: ``rows`` (17) heights evenly spaced over A's silhouette, ``s`` = 0 at its
    top to 1 at its bottom. ``band`` = [from, to] in those fractions, or ``band_z`` =
    [z_top, z_bottom] in scene units, measures only that part (waist to hem) and
    ``s`` then runs over the band. A half-width is half the distance between the
    outermost filled pixels of the row, so a gap between two legs is not seen.
    ``axis`` = picture-right coordinate (world X in the front view) to measure
    asymmetry about; default is the median row centre.

    Reply: lengths are in metres (``units``: "m") when A comes from the scene,
    else in pixels of A.
      ``profile``             [[s, half_width], ...] of A: paste into a contract.
      ``profile_normalised``  [[s, g], ...] with g = 0 at the first row's width and 1
                              at the last, the form of the dress contract's PROFILE.
      ``profile_b``, ``profile_b_normalised``  the same for B after the fit.
      ``rows``   per row: s, height (scene up coordinate), a, b (half-widths),
                 diff = b - a (positive: B is wider, A must grow there),
                 diff_fraction = diff / A's height, shift = B's centre - A's centre.
      ``worst``  the row with the largest |diff|; ``mean_abs_diff``.
      ``iou``    overlap / union of the two masks, 1 = identical. ``only_a`` and
                 ``only_b`` are the shares of the union covered by just one.
      ``asymmetry`` for a and b: right minus left extent about the axis (mean, and
                 the largest with its s), and ``mirror_iou`` (1 = symmetric).
      ``a`` / ``b``  size, area, where it came from; ``fit``; ``window``; ``seconds``.
      ``note``   present for a ``reference``: the comparison is approximate
                 (perspective, pose, background removal). There is never a verdict.
    Files: ``out`` = A's mask PNG, ``b_out`` = B's mask as fitted, ``overlay`` = both:
    orange = only A, blue = only B, light grey = both, dashed yellow = worst row.

    Views file: ``views`` = path to a JSON file kept beside the contract, ``view`` =
    a name in it. Each entry may hold any argument above (``objects`` is accepted
    for ``names``); image paths are relative to the file. Arguments given in the
    call win over the file.
        {"front": {"direction": [0, -1, 0], "up": [0, 0, 1], "objects": ["Skirt1850"],
                   "window": {"center": [0, 0, 0.6], "width": 1.8, "height": 1.3},
                   "reference": "refs/ref.jpg", "crop": [40, 610, 1150, 1310],
                   "background": "rows", "anchor": "top"}}

    Examples:
        compare names='["Skirt1850"]' rows=17 out=/work/skirt.png
        compare names='["Skirt1850"]' reference=/refs/ref.jpg crop='[40,610,1150,1310]' overlay=/work/ov.png
        compare views=/dress/views.json view=front overlay=/work/ov.png b_out=/work/ref_mask.png
        compare image=/work/before.png other=/work/after.png overlay=/work/change.png

    Not measured: instances made by geometry nodes or collection instances (only
    each object's own evaluated mesh), wire or loose-edge geometry, and anything
    thinner than a pixel.
    """
    started = time.time()
    args = _with_view(dict(args))
    size = max(16, min(int(args.get("size") or DEFAULT_SIZE), MAX_SIZE))
    row_count = max(1, min(int(args.get("rows") or DEFAULT_ROWS), MAX_ROWS))
    tolerance = float(args.get("tolerance", DEFAULT_TOLERANCE))
    limit = max(1024, 2 * size)
    sources = [key for key in ("reference", "other", "against") if args.get(key)]
    if len(sources) > 1:
        raise ValueError("give one of 'reference', 'other', 'against', not {:s}".format(" and ".join(sources)))
    kind_b = sources[0] if sources else None

    frame = None
    basis = None
    mask_b = None
    info_b = None
    if args.get("names"):
        if args.get("image"):
            raise ValueError("give 'names' or 'image', not both")
        subjects = _subjects(args["names"])
        basis = _basis(args.get("view"), args.get("direction"), args.get("up"))
        triangles = _world_triangles(subjects)
        points = triangles.reshape(-1, 3)
        if kind_b == "against":
            subjects_b = _subjects(args["against"])
            triangles_b = _world_triangles(subjects_b)
            # Both must be in view, or the comparison would be cut off by the frame.
            points = np.concatenate((points, triangles_b.reshape(-1, 3)))
        frame = _frame(points, basis, args.get("window"), args.get("margin", DEFAULT_MARGIN), size)
        mask_a = _scene_mask(triangles, basis, frame)
        info_a = {"kind": "scene", "objects": [ob.name for ob in subjects], "triangles": int(len(triangles))}
        if kind_b == "against":
            mask_b = _scene_mask(triangles_b, basis, frame)
            info_b = {"kind": "scene", "objects": [ob.name for ob in subjects_b], "triangles": int(len(triangles_b))}
            if not mask_b.any():
                raise ValueError("'against' objects are outside the window")
        hidden = [ob.name for ob in subjects if not ob.visible_get()]
        if hidden:
            info_a["hidden"] = hidden
        if not mask_a.any():
            raise ValueError("the objects are outside the window, or too thin to fill a pixel")
    elif args.get("image"):
        if kind_b == "against":
            raise ValueError("'against' compares objects and needs 'names'")
        mask_a, info_a = _image_mask(args["image"], args.get("image_crop"), tolerance, args.get("background"), limit)
        info_a["kind"] = "image"
    else:
        raise ValueError("compare needs 'names' (objects) or 'image' (a file)")

    unit = frame["pixel"] if frame else 1.0
    box_a = _bbox(mask_a)
    high_a = box_a[3] - box_a[1]
    result = {"units": "m" if frame else "px"}

    def describe(mask, info):
        box = _bbox(mask)
        info.update({
            "width": round((box[2] - box[0]) * unit, 4),
            "height": round((box[3] - box[1]) * unit, 4),
            "area": round(float(mask.sum()) * unit * unit, 6),
        })
        return info

    fit = None
    if kind_b in ("reference", "other"):
        loaded, info_b = _image_mask(args[kind_b], args.get("crop"), tolerance, args.get("background"), limit)
        info_b["kind"] = kind_b
        mode = args.get("fit") or ("bbox" if kind_b == "reference" else "none")
        mask_b, fit = _fit(mask_a, loaded, mode, args.get("scale_by") or "height",
                           args.get("anchor") or "center", bool(args.get("refine")))
        if not mask_b.any():
            raise ValueError("after the fit nothing of B is inside A's picture")
        if frame and mode == "bbox":
            # What one pixel of the file is worth in the scene, for measuring off the photo.
            fit["image_pixel_m"] = round(frame["pixel"] * fit["scale"] / info_b["step"], 6)
    elif kind_b == "against":
        fit = {"mode": "none", "scale": 1.0, "offset": [0.0, 0.0]}

    band = _band(args, box_a, frame)
    fractions, rows = _sample_rows(box_a, band, row_count)
    band_rows = (int(rows[0]), int(rows[-1]))
    filled_a, left_a, right_a = _spans(mask_a)
    if mask_b is not None:
        filled_b, left_b, right_b = _spans(mask_b)

    table = []
    profile, profile_b = [], []
    for fraction, row in zip(fractions, rows):
        s = round(float(fraction), 4)
        entry = {"s": s}
        if frame:
            entry["height"] = round(frame["top"] - (row + 0.5) * frame["pixel"], 4)
        half_a = (right_a[row] - left_a[row]) * 0.5 * unit if filled_a[row] else None
        entry["a"] = None if half_a is None else round(half_a, 4)
        profile.append([s, entry["a"]])
        if mask_b is not None:
            half_b = (right_b[row] - left_b[row]) * 0.5 * unit if filled_b[row] else None
            entry["b"] = None if half_b is None else round(half_b, 4)
            profile_b.append([s, entry["b"]])
            if half_a is not None and half_b is not None:
                diff = half_b - half_a
                entry["diff"] = round(diff, 4)
                entry["diff_fraction"] = round(diff / (high_a * unit), 4)
                entry["shift"] = round(
                    ((left_b[row] + right_b[row]) - (left_a[row] + right_a[row])) * 0.5 * unit, 4)
        table.append(entry)

    result["a"] = describe(mask_a, info_a)
    result["profile"] = profile
    normalised = _normalised(profile)
    if normalised:
        result["profile_normalised"] = normalised
    if band != (0.0, 1.0):
        result["band"] = [round(band[0], 4), round(band[1], 4)]

    axis = args.get("axis")
    if axis is not None:
        if frame is None:
            raise ValueError("axis is in scene units and needs a scene silhouette")
        axis = (float(axis) - frame["left"]) / frame["pixel"]
    result["asymmetry"] = {"a": _asymmetry(mask_a, band_rows, unit, axis)}

    worst_row = None
    if mask_b is not None:
        result["b"] = describe(mask_b, info_b)
        result["fit"] = fit
        union = float((mask_a | mask_b).sum())
        result["iou"] = round(_iou(mask_a, mask_b), 4)
        result["only_a"] = round(float((mask_a & ~mask_b).sum()) / union, 4)
        result["only_b"] = round(float((mask_b & ~mask_a).sum()) / union, 4)
        result["profile_b"] = profile_b
        normalised = _normalised(profile_b)
        if normalised:
            result["profile_b_normalised"] = normalised
        result["rows"] = table
        measured = [(abs(entry["diff"]), index) for index, entry in enumerate(table) if "diff" in entry]
        if measured:
            _largest, index = max(measured)
            worst_row = int(rows[index])
            result["worst"] = dict(table[index], wider="b" if table[index]["diff"] > 0 else "a")
            result["mean_abs_diff"] = round(sum(gap for gap, _ in measured) / len(measured), 4)
        # B is measured about A's axis, so a sideways offset of the whole shape shows up.
        result["asymmetry"]["b"] = _asymmetry(mask_b, band_rows, unit, axis)
        if kind_b == "reference":
            result["note"] = PHOTO_NOTE
    else:
        result["rows"] = table

    if frame:
        right, up, toward = basis
        result["window"] = frame["window"]
        result["view"] = {
            "direction": [round(float(v), 6) for v in toward],
            "up": [round(float(v), 6) for v in up],
            "right": [round(float(v), 6) for v in right],
            "pixel_size": round(frame["pixel"], 8),
        }
    result["size"] = [int(mask_a.shape[1]), int(mask_a.shape[0])]

    if args.get("out"):
        result["mask"] = _save_mask(mask_a, args["out"])
    if args.get("b_out"):
        if mask_b is None:
            raise ValueError("b_out needs something to compare with")
        result["b_mask"] = _save_mask(mask_b, args["b_out"])
    if args.get("overlay"):
        if mask_b is None:
            raise ValueError("overlay needs something to compare with: 'reference', 'other' or 'against'")
        result["overlay"] = _save_pixels(_overlay(mask_a, mask_b, worst_row), args["overlay"])
        result["overlay_legend"] = "orange: only A; blue: only B; light grey: both; dashed yellow: worst row"
    result["seconds"] = round(time.time() - started, 3)
    return result


COMMANDS = {"compare": cmd_compare}
