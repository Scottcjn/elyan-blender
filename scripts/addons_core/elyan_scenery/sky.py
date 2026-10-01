# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Sky Lab: one-click skies built as world shader nodes.

A physical sky lights the scene, a noise layer projected onto a flat ceiling
gives clouds, and thresholded noise gives stars.
"""

import math

import bpy
from bpy.props import EnumProperty, FloatProperty
from bpy.types import Operator

WORLD_NAME = "Scenery Sky"

# elevation and rotation in degrees; haze 0..10; clouds is coverage 0..1.
SKY_PRESETS = {
    'NOON': {"label": "Clear Noon", "elevation": 62.0, "rotation": 40.0, "haze": 1.0, "clouds": 0.15},
    'FAIR': {"label": "Fair Weather", "elevation": 38.0, "rotation": 130.0, "haze": 1.5, "clouds": 0.45},
    'GOLDEN': {"label": "Golden Hour", "elevation": 9.0, "rotation": 200.0, "haze": 3.0, "clouds": 0.35},
    'SUNSET': {"label": "Sunset", "elevation": 1.5, "rotation": 185.0, "haze": 4.5, "clouds": 0.5},
    'HAZE': {"label": "Dawn Haze", "elevation": 5.0, "rotation": 330.0, "haze": 8.0, "clouds": 0.1},
    'OVERCAST': {"label": "Overcast", "elevation": 30.0, "rotation": 90.0, "haze": 5.0, "clouds": 0.95},
    'NIGHT': {"label": "Starry Night", "elevation": -12.0, "rotation": 0.0, "haze": 1.0, "clouds": 0.1, "stars": 1.0},
}

SKY_ITEMS = tuple((key, preset["label"], "") for key, preset in SKY_PRESETS.items())

# The physical sky is far brighter than display white; this keeps a noon scene well exposed.
SKY_STRENGTH = 0.08
NIGHT_COLOR = (0.012, 0.020, 0.050)


def _set_enum(node, attr, *candidates):
    """The sky model was renamed in 5.0, try the names newest first."""
    allowed = {item.identifier for item in node.bl_rna.properties[attr].enum_items}
    for value in candidates:
        if value in allowed:
            setattr(node, attr, value)
            return


def _set_haze(node, value):
    for attr in ("aerosol_density", "dust_density"):
        if hasattr(node, attr):
            setattr(node, attr, value)
            return


def build(scene, elevation, rotation, haze, clouds, stars=0.0):
    """Create (or rebuild) the scenery world and assign it to ``scene``."""
    world = bpy.data.worlds.get(WORLD_NAME) or bpy.data.worlds.new(WORLD_NAME)
    world.use_nodes = True
    tree = world.node_tree
    nodes, links = tree.nodes, tree.links
    nodes.clear()

    sky = nodes.new("ShaderNodeTexSky")
    _set_enum(sky, "sky_type", 'MULTIPLE_SCATTERING', 'NISHITA')
    sky.sun_elevation = math.radians(elevation)
    sky.sun_rotation = math.radians(rotation)
    _set_haze(sky, haze)
    color = sky.outputs[0]
    # A low sun gives little light; open the exposure as it sinks, the way a camera would.
    strength = SKY_STRENGTH * (1.0 + 3.0 * max(0.0, min(1.0, (20.0 - elevation) / 20.0))) if elevation >= 0.0 else SKY_STRENGTH

    coords = nodes.new("ShaderNodeTexCoord")
    direction = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coords.outputs["Generated"], direction.inputs[0])
    up = direction.outputs["Z"]

    if elevation < 0.0:
        # The physical sky is black once the sun has set, give the night some colour.
        glow = nodes.new("ShaderNodeMix")
        glow.data_type = 'RGBA'
        glow.blend_type = 'ADD'
        glow.inputs[0].default_value = 1.0
        glow.inputs[7].default_value = (*(c / strength for c in NIGHT_COLOR), 1.0)
        links.new(color, glow.inputs[6])
        color = glow.outputs[2]

    if stars > 0.0:
        specks = nodes.new("ShaderNodeTexNoise")
        specks.inputs["Scale"].default_value = 600.0
        specks.inputs["Detail"].default_value = 0.0
        links.new(coords.outputs["Generated"], specks.inputs["Vector"])
        bright = nodes.new("ShaderNodeMapRange")
        for socket, value in zip(list(bright.inputs)[1:5], (0.80, 0.84, 0.0, 2.0 * stars / strength)):
            socket.default_value = value
        links.new(specks.outputs[0], bright.inputs[0])
        add = nodes.new("ShaderNodeMix")
        add.data_type = 'RGBA'
        add.blend_type = 'ADD'
        add.inputs[0].default_value = 1.0
        links.new(color, add.inputs[6])
        links.new(bright.outputs[0], add.inputs[7])
        color = add.outputs[2]

    if clouds > 0.0:
        # Project the view direction onto a flat ceiling so clouds shrink toward the horizon.
        height = nodes.new("ShaderNodeMath")
        height.operation = 'MAXIMUM'
        height.inputs[1].default_value = 0.04
        links.new(up, height.inputs[0])
        ceiling = nodes.new("ShaderNodeVectorMath")
        ceiling.operation = 'DIVIDE'
        links.new(coords.outputs["Generated"], ceiling.inputs[0])
        spread = nodes.new("ShaderNodeCombineXYZ")
        for socket in spread.inputs:
            links.new(height.outputs[0], socket)
        links.new(spread.outputs[0], ceiling.inputs[1])

        puffs = nodes.new("ShaderNodeTexNoise")
        puffs.inputs["Scale"].default_value = 1.3
        puffs.inputs["Detail"].default_value = 8.0
        puffs.inputs["Roughness"].default_value = 0.6
        links.new(ceiling.outputs[0], puffs.inputs["Vector"])

        # Noise averages 0.5: full coverage starts the ramp low, clear sky starts it high.
        start = 0.72 - 0.42 * clouds
        cover = nodes.new("ShaderNodeMapRange")
        for socket, value in zip(list(cover.inputs)[1:5], (start, start + 0.22, 0.0, 1.0)):
            socket.default_value = value
        links.new(puffs.outputs[0], cover.inputs[0])
        horizon = nodes.new("ShaderNodeMapRange")
        for socket, value in zip(list(horizon.inputs)[1:5], (0.0, 0.12, 0.0, 1.0)):
            socket.default_value = value
        links.new(up, horizon.inputs[0])
        mask = nodes.new("ShaderNodeMath")
        mask.operation = 'MULTIPLY'
        links.new(cover.outputs[0], mask.inputs[0])
        links.new(horizon.outputs[0], mask.inputs[1])

        # Clouds are lit by the sun: bright by day, warm and dim near the horizon, dark at night.
        day = max(0.0, min(1.0, elevation / 25.0))
        low = max(0.0, min(1.0, (elevation + 6.0) / 12.0))
        tint = (1.0, 0.55 + 0.45 * day, 0.35 + 0.65 * day)
        level = (0.004 + 0.9 * low * (0.3 + 0.7 * day)) * (1.0 - 0.45 * clouds) / strength
        layer = nodes.new("ShaderNodeMix")
        layer.data_type = 'RGBA'
        links.new(mask.outputs[0], layer.inputs[0])
        links.new(color, layer.inputs[6])
        layer.inputs[7].default_value = (*(c * level for c in tint), 1.0)
        color = layer.outputs[2]

    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = strength
    links.new(color, background.inputs["Color"])
    output = nodes.new("ShaderNodeOutputWorld")
    links.new(background.outputs[0], output.inputs["Surface"])

    scene.world = world
    return world


class ELYAN_OT_sky_set(Operator):
    """Replace the world with a sky: sun, haze, clouds and stars"""
    bl_idname = "elyan_scenery.sky_set"
    bl_label = "Sky"
    bl_options = {'REGISTER', 'UNDO'}

    def _load_preset(self, _context):
        preset = SKY_PRESETS[self.preset]
        self.elevation = preset["elevation"]
        self.rotation = preset["rotation"]
        self.haze = preset["haze"]
        self.clouds = preset["clouds"]
        self.stars = preset.get("stars", 0.0)

    preset: EnumProperty(name="Preset", items=SKY_ITEMS, update=_load_preset)
    elevation: FloatProperty(
        name="Sun Height", description="Degrees above the horizon, negative is night",
        default=62.0, min=-30.0, max=90.0,
    )
    rotation: FloatProperty(name="Sun Direction", description="Degrees around the horizon", default=40.0)
    haze: FloatProperty(name="Haze", default=1.0, min=0.0, max=10.0)
    clouds: FloatProperty(name="Clouds", description="Cloud cover", default=0.15, min=0.0, max=1.0, subtype='FACTOR')
    stars: FloatProperty(name="Stars", default=0.0, min=0.0, max=2.0)

    def execute(self, context):
        build(context.scene, self.elevation, self.rotation, self.haze, self.clouds, self.stars)
        return {'FINISHED'}


classes = (
    ELYAN_OT_sky_set,
)
