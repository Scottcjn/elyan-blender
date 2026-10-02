# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Height-field maths: fractal landscapes, filters and erosion.

Pure NumPy, no ``bpy``. A height field is a square ``float32`` array indexed
``[y, x]`` with values nominally in 0..1.
"""

import numpy as np

_NEIGHBORS_8 = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


# -----------------------------------------------------------------------------
# Building blocks

def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def normalize(grid):
    low, high = float(grid.min()), float(grid.max())
    if high - low < 1e-9:
        return np.zeros_like(grid)
    return ((grid - low) / (high - low)).astype(np.float32)


def value_noise(n, cells, rng):
    """
    Smooth random field of ``n`` x ``n`` samples with about ``cells`` bumps across it.

    Made by keeping one band of frequencies of white noise, so it has no grain
    direction and no lattice to show through.
    """
    # Work on a larger tile and crop, the transform would otherwise wrap edge to edge.
    m = 1 << int(np.ceil(np.log2(n * 1.25)))
    spectrum = np.fft.rfft2(rng.standard_normal((m, m), dtype=np.float32))
    fy = np.fft.fftfreq(m)[:, None] * m
    fx = np.fft.rfftfreq(m)[None, :] * m
    center = cells * m / n
    band = np.exp(-0.5 * ((np.hypot(fy, fx) - center) / (0.35 * center)) ** 2)
    field = np.fft.irfft2(spectrum * band, s=(m, m))[:n, :n]
    # About three standard deviations either side fills 0..1.
    return np.clip(0.5 + field / (6.0 * field.std() + 1e-12), 0.0, 1.0).astype(np.float32)


def fbm(n, rng, octaves=7, cells=3, gain=0.5, ridged=False):
    """Fractal sum of value noise, normalized to 0..1. ``ridged`` gives sharp crests."""
    total = np.zeros((n, n), dtype=np.float32)
    amplitude = 1.0
    weight = np.ones((n, n), dtype=np.float32)
    for _ in range(octaves):
        if cells > n // 2:
            break
        layer = value_noise(n, cells, rng)
        if ridged:
            layer = 1.0 - np.abs(2.0 * layer - 1.0)
            layer *= layer
            # Detail gathers on the ridges and leaves the valleys smooth.
            total += layer * amplitude * weight
            weight = np.clip(layer * 2.0, 0.0, 1.0)
        else:
            total += layer * amplitude
        amplitude *= gain
        cells *= 2
    return normalize(total)


def _radial(n):
    """Distance from the centre, 0 in the middle and 1 at the middle of an edge."""
    axis = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    return np.sqrt(axis[:, None] ** 2 + axis[None, :] ** 2)


# -----------------------------------------------------------------------------
# Landscapes

def _mountains(n, rng):
    grid = 0.7 * fbm(n, rng, octaves=8, ridged=True) + 0.3 * fbm(n, rng, octaves=6)
    return normalize(grid) ** 1.6


def _hills(n, rng):
    return smooth(fbm(n, rng, octaves=5, cells=2, gain=0.42) ** 1.2, 2)


def _dunes(n, rng):
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)[None, :]
    warp = fbm(n, rng, octaves=3, cells=2)
    crest = 0.5 + 0.5 * np.sin(2.0 * np.pi * (x * 7.0 + warp * 1.6))
    return normalize(crest ** 1.5 * (0.35 + 0.65 * fbm(n, rng, octaves=3, cells=2)))


def _canyon(n, rng):
    base = fbm(n, rng, octaves=6)
    mesa = _smoothstep(np.clip((base - 0.42) / 0.14, 0.0, 1.0))
    river = fbm(n, rng, octaves=4, cells=2, ridged=True)
    cut = _smoothstep(np.clip((river - 0.80) / 0.15, 0.0, 1.0))
    return normalize(0.75 * mesa + 0.25 * terrace(base, 6, 0.8) - 0.5 * cut * mesa)


def _island(n, rng):
    shore = np.clip(1.0 - _radial(n) + 0.25 * (fbm(n, rng, octaves=4, cells=2) - 0.5), 0.0, 1.0)
    return normalize(_mountains(n, rng) * 0.8 + 0.2) * _smoothstep(np.clip(shore * 1.4, 0.0, 1.0))


def _volcano(n, rng):
    r = _radial(n)
    cone = np.clip(1.0 - r / 0.85, 0.0, 1.0) ** 1.4
    crater = 0.45 * np.exp(-(r / 0.11) ** 2)
    return normalize(cone - crater + 0.12 * fbm(n, rng, octaves=7, ridged=True) * (0.3 + cone))


def _plateau(n, rng):
    base = fbm(n, rng, octaves=6)
    return normalize(np.clip(base * 1.7 - 0.35, 0.0, 0.7) + 0.05 * fbm(n, rng, octaves=7))


def _craters(n, rng):
    grid = 0.5 + 0.25 * (fbm(n, rng, octaves=6) - 0.5)
    axis = np.linspace(0.0, 1.0, n, dtype=np.float32)
    for _ in range(int(rng.integers(14, 26))):
        cx, cy = rng.random(2)
        radius = 0.03 + 0.12 * rng.random() ** 2
        d = np.sqrt((axis[None, :] - cx) ** 2 + (axis[:, None] - cy) ** 2) / radius
        bowl = np.where(d < 1.0, (d * d - 1.0) * 0.6, 0.0)
        rim = 0.35 * np.exp(-((d - 1.0) / 0.18) ** 2)
        grid += (bowl + rim) * radius * 1.8
    return normalize(grid)


def _plains(n, rng):
    return 0.18 * fbm(n, rng, octaves=5, cells=2, gain=0.4)


def rolls(n, seed):
    """Gentle rolling ground, 0..1, for places seen from a few metres away."""
    rng = np.random.default_rng(seed)
    return normalize(0.75 * fbm(n, rng, octaves=3, cells=2, gain=0.45) + 0.25 * fbm(n, rng, octaves=5, cells=5))


def basin(n, center, radii, shore):
    """
    Oval hollow as a 0..1 depth profile: 1 across the middle, easing to 0 over the bank.

    ``center`` and ``radii`` are (u, v) in 0..1 of the field; ``shore`` is the width
    of the bank beyond the rim, in the same unit.
    """
    axis = np.linspace(0.0, 1.0, n, dtype=np.float32)
    # Distance in "rim units": 1 on the rim.
    reach = np.hypot((axis[None, :] - center[0]) / radii[0], (axis[:, None] - center[1]) / radii[1])
    bank = 1.0 + shore / min(radii)
    return _smoothstep(np.clip((bank - reach) / (bank - 0.55), 0.0, 1.0)).astype(np.float32)


LANDSCAPES = {
    'MOUNTAINS': _mountains,
    'HILLS': _hills,
    'DUNES': _dunes,
    'CANYON': _canyon,
    'ISLAND': _island,
    'VOLCANO': _volcano,
    'PLATEAU': _plateau,
    'CRATERS': _craters,
    'PLAINS': _plains,
}


def generate(kind, n, seed):
    rng = np.random.default_rng(seed)
    return LANDSCAPES[kind](n, rng).astype(np.float32)


# -----------------------------------------------------------------------------
# Filters

def _neighbor(grid, dy, dx):
    """Value of each cell's neighbour, edge cells see themselves."""
    padded = np.pad(grid, 1, mode='edge')
    n = grid.shape[0]
    return padded[1 + dy:1 + dy + n, 1 + dx:1 + dx + n]


def _push(target, amount, dy, dx):
    """Add ``amount`` leaving each cell to its neighbour (nothing leaves the grid)."""
    n = target.shape[0]
    src_y = slice(max(0, -dy), n - max(0, dy))
    src_x = slice(max(0, -dx), n - max(0, dx))
    dst_y = slice(max(0, dy), n - max(0, -dy))
    dst_x = slice(max(0, dx), n - max(0, -dx))
    target[dst_y, dst_x] += amount[src_y, src_x]


def smooth(grid, iterations=1):
    grid = grid.astype(np.float32)
    for _ in range(iterations):
        padded = np.pad(grid, 1, mode='edge')
        n = grid.shape[0]
        total = np.zeros_like(grid)
        for dy in range(3):
            for dx in range(3):
                total += padded[dy:dy + n, dx:dx + n]
        grid = total / 9.0
    return grid


def terrace(grid, levels=6, sharpness=0.7):
    """Stepped strata. ``sharpness`` 0 leaves the slope, 1 gives hard cliffs."""
    scaled = grid * levels
    step = np.floor(scaled)
    frac = scaled - step
    power = 1.0 + sharpness * 12.0
    return ((step + frac ** power) / levels).astype(np.float32)


def edge_falloff(grid, width=0.2):
    """Ease the borders down to zero so the terrain meets the ground."""
    n = grid.shape[0]
    axis = np.linspace(0.0, 1.0, n, dtype=np.float32)
    border = np.minimum(axis, 1.0 - axis)
    ramp = _smoothstep(np.clip(border / max(width, 1e-6), 0.0, 1.0))
    return (grid * ramp[:, None] * ramp[None, :]).astype(np.float32)


def thermal_erosion(grid, iterations=30, talus=3.0):
    """
    Slopes steeper than the talus angle slide downhill, leaving scree fans.

    ``talus`` is the steepest slope left standing, as height over distance where
    the terrain is as tall as it is wide (3 is about 30 degrees on typical scenery).
    """
    grid = grid.astype(np.float32).copy()
    limit = talus / grid.shape[0]
    for _ in range(iterations):
        delta = np.zeros_like(grid)
        for dy, dx in _NEIGHBORS_8:
            distance = float(np.hypot(dy, dx))
            excess = np.clip(grid - _neighbor(grid, dy, dx) - limit * distance, 0.0, None) * (0.06 / distance)
            delta -= excess
            _push(delta, excess, dy, dx)
        grid += delta
    return grid


def _fill_pits(grid, rounds):
    """Raise hollows toward their rims so water finds a way out instead of pooling."""
    filled = grid.copy()
    step = 1e-5
    for _ in range(rounds):
        lowest = _neighbor(filled, *_NEIGHBORS_8[0])
        for dy, dx in _NEIGHBORS_8[1:]:
            lowest = np.minimum(lowest, _neighbor(filled, dy, dx))
        raised = np.maximum(filled, lowest + step)
        # The border is the sea, water may always leave there.
        raised[0, :], raised[-1, :], raised[:, 0], raised[:, -1] = filled[0, :], filled[-1, :], filled[:, 0], filled[:, -1]
        if np.array_equal(raised, filled):
            break
        filled = raised
    return filled


def _drainage(grid):
    """
    For every cell: how steeply it drains, and how much upstream land drains through it.

    Each cell hands its water to its steepest lower neighbour. Returns the slope to
    that neighbour (height per cell) and the number of cells upstream, itself included.
    """
    n = grid.shape[0]
    index = np.arange(n * n, dtype=np.int64).reshape(n, n)
    best_slope = np.zeros_like(grid)
    receiver = index.copy()
    for dy, dx in _NEIGHBORS_8:
        slope = (grid - _neighbor(grid, dy, dx)) / np.hypot(dy, dx)
        steeper = slope > best_slope
        best_slope = np.where(steeper, slope, best_slope)
        receiver = np.where(steeper, _neighbor(index, dy, dx), receiver)

    receiver = receiver.ravel()
    drains = receiver != index.ravel()
    targets = receiver[drains]
    area = np.ones(n * n, dtype=np.float64)
    # Water moves one cell per round, so this follows rivers up to 2n cells long.
    for _ in range(2 * n):
        updated = 1.0 + np.bincount(targets, weights=area[drains], minlength=n * n)
        if np.array_equal(updated, area):
            break
        area = updated
    return best_slope, area.reshape(n, n).astype(np.float32)


def hydraulic_erosion(grid, iterations=60, seed=0, strength=0.5):
    """
    Rain gathers into streams that cut branching valleys: the more land drains
    through a spot and the steeper it is, the deeper the cut.
    """
    rng = np.random.default_rng(seed)
    grid = grid.astype(np.float32).copy()
    n = grid.shape[0]
    for _ in range(max(1, iterations // 12)):
        # A little roughness keeps water from running in dead-straight lines.
        grid += (rng.random(grid.shape, dtype=np.float32) - 0.5) * (0.4 / n)
        slope, area = _drainage(_fill_pits(grid, 40))
        # Hillsides barely change; a stream draining a couple of rows' worth of land cuts hard.
        stream = np.minimum(1.0, np.sqrt(area / (2.0 * n)))
        # ``slope`` is the drop to the next cell, a stream never cuts below that cell.
        grid -= slope * np.minimum(0.9, strength * (0.02 + stream))
        grid = thermal_erosion(grid, 1, talus=5.0)
    return grid


def resample(grid, n):
    """Bilinear resize to ``n`` x ``n``."""
    old = grid.shape[0]
    if old == n:
        return grid.astype(np.float32)
    t = np.linspace(0.0, old - 1, n, dtype=np.float32)
    i = np.clip(t.astype(np.int32), 0, old - 2)
    f = t - i
    fy, fx = f[:, None], f[None, :]
    iy, ix = i[:, None], i[None, :]
    top = grid[iy, ix] * (1 - fx) + grid[iy, ix + 1] * fx
    bottom = grid[iy + 1, ix] * (1 - fx) + grid[iy + 1, ix + 1] * fx
    return (top * (1 - fy) + bottom * fy).astype(np.float32)


def sample(grid, u, v):
    """Height at ``u``, ``v`` in 0..1 (x, y)."""
    n = grid.shape[0]
    x = min(max(u, 0.0), 1.0) * (n - 1)
    y = min(max(v, 0.0), 1.0) * (n - 1)
    ix, iy = min(int(x), n - 2), min(int(y), n - 2)
    fx, fy = x - ix, y - iy
    top = grid[iy, ix] * (1 - fx) + grid[iy, ix + 1] * fx
    bottom = grid[iy + 1, ix] * (1 - fx) + grid[iy + 1, ix + 1] * fx
    return float(top * (1 - fy) + bottom * fy)


def sample_many(grid, u, v):
    """Heights at arrays of ``u``, ``v`` in 0..1 (x, y)."""
    n = grid.shape[0]
    x = np.clip(u, 0.0, 1.0) * (n - 1)
    y = np.clip(v, 0.0, 1.0) * (n - 1)
    ix = np.minimum(x.astype(np.int32), n - 2)
    iy = np.minimum(y.astype(np.int32), n - 2)
    fx, fy = x - ix, y - iy
    top = grid[iy, ix] * (1 - fx) + grid[iy, ix + 1] * fx
    bottom = grid[iy + 1, ix] * (1 - fx) + grid[iy + 1, ix + 1] * fx
    return top * (1 - fy) + bottom * fy


def slope(grid, run):
    """
    Steepness of every cell as rise over run (1 is 45 degrees).

    ``run`` is the distance between neighbouring points, in the unit of the heights.
    """
    dy, dx = np.gradient(grid.astype(np.float32), run)
    return np.hypot(dx, dy)
