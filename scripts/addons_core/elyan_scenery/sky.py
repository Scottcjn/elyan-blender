# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Sky Lab: one-click skies built as world shader nodes.

A physical sky lights the scene, a noise layer projected onto a flat ceiling
gives clouds, and thresholded noise gives stars. Distance haze is not part of
the world (a world volume would swallow the sky): it lives in a node group that
the scenery materials share, and fades them toward the colour of the horizon.
"""

import math

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty
from bpy.types import Operator
from mathutils import Vector

from . import materials

WORLD_NAME = "Scenery Sky"
MOON_NAME = "Scenery Moon"

# elevation and rotation in degrees; haze 0..10 is dust in the sky itself; clouds is
# coverage 0..1; mist 0..1 is how much the distance fades the land into the horizon.
SKY_PRESETS = {
    'NOON': {"label": "Clear Noon", "elevation": 62.0, "rotation": 40.0, "haze": 1.0, "clouds": 0.15, "mist": 0.25},
    'FAIR': {"label": "Fair Weather", "elevation": 38.0, "rotation": 130.0, "haze": 1.5, "clouds": 0.45, "mist": 0.35},
    'GOLDEN': {"label": "Golden Hour", "elevation": 9.0, "rotation": 200.0, "haze": 3.0, "clouds": 0.35, "mist": 0.5},
    'SUNSET': {"label": "Sunset", "elevation": 1.5, "rotation": 185.0, "haze": 4.5, "clouds": 0.5, "mist": 0.55},
    'HAZE': {"label": "Dawn Haze", "elevation": 5.0, "rotation": 330.0, "haze": 8.0, "clouds": 0.1, "mist": 1.0},
    'OVERCAST': {
        "label": "Overcast", "elevation": 30.0, "rotation": 90.0, "haze": 5.0, "clouds": 0.95, "mist": 0.6,
        "mist_color": (0.85, 0.88, 0.95),
    },
    'NIGHT': {
        "label": "Starry Night", "elevation": -12.0, "rotation": 0.0, "haze": 1.0, "clouds": 0.1, "stars": 1.0,
        "mist": 0.3,
    },
}

SKY_ITEMS = tuple((key, preset["label"], "") for key, preset in SKY_PRESETS.items())

# The physical sky is far brighter than display white; this keeps a noon scene well exposed.
SKY_STRENGTH = 0.03
NIGHT_COLOR = (0.012, 0.020, 0.050)
# Where the haze takes its colour from, as the Z of a level view direction. On Blender 5's
# sky it is the very horizon: at low sun the last degree is a band darker than what is above
# it, and hazed water that took its colour from higher up would draw a pale line under it.
# The older sky is murky brown for several degrees above the horizon, and haze that colour
# would dirty the whole distance, so there the haze looks above the murk and the line is
# accepted. (The world's sky cannot be shifted to match, Cycles ignores its vector input.)
HORIZON_LOOK = 0.003
HORIZON_LOOK_LEGACY = 0.08
MOON_COLOR = (0.55, 0.68, 1.0)
MOON_STRENGTH = 1.6
STAR_BRIGHTNESS = 10.0
MOON_ELEVATION = 38.0


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


def _scene_span(scene):
    """Width of the largest terrain in the scene: the yardstick for how far away "far" is."""
    spans = [
        ob.elyan_terrain.size * max(ob.scale.x, ob.scale.y)
        for ob in scene.objects if ob.type == 'MESH' and ob.elyan_terrain.is_terrain
    ]
    return max(spans, default=300.0)


def _build_haze(scene, elevation, rotation, haze, strength, amount, tint, look):
    """
    Fill in the haze group that the scenery materials end in.

    Each surface is mixed toward a glow the colour of the sky on the horizon behind
    it, by how far it is from the camera. The colour comes from a second copy of
    the sky, so haze is warm toward a low sun and blue away from it, and changes
    with the sky instead of needing to be matched by hand.
    """
    group = materials.haze_group()
    nodes, links = group.nodes, group.links
    nodes.clear()
    enter = nodes.new("NodeGroupInput")
    leave = nodes.new("NodeGroupOutput")
    if amount <= 0.0:
        links.new(enter.outputs[0], leave.inputs[0])
        return group

    camera = nodes.new("ShaderNodeCameraData")
    # At full strength something one terrain-width away keeps about a third of its own colour.
    thin = nodes.new("ShaderNodeMath")
    thin.operation = 'MULTIPLY'
    thin.inputs[1].default_value = -amount * 1.1 / _scene_span(scene)
    links.new(camera.outputs["View Distance"], thin.inputs[0])
    clear = nodes.new("ShaderNodeMath")
    clear.operation = 'EXPONENT'
    links.new(thin.outputs[0], clear.inputs[0])
    factor = nodes.new("ShaderNodeMath")
    factor.operation = 'SUBTRACT'
    factor.inputs[0].default_value = 1.0
    links.new(clear.outputs[0], factor.inputs[1])

    # Look up the sky just above the horizon in the direction the camera sees this point.
    # The camera's own vector is used, not the incoming ray, so reflections agree with it.
    world_space = nodes.new("ShaderNodeVectorTransform")
    world_space.vector_type = 'VECTOR'
    world_space.convert_from = 'CAMERA'
    world_space.convert_to = 'WORLD'
    links.new(camera.outputs["View Vector"], world_space.inputs[0])
    flat = nodes.new("ShaderNodeVectorMath")
    flat.operation = 'MULTIPLY'
    flat.inputs[1].default_value = (1.0, 1.0, 0.0)
    links.new(world_space.outputs[0], flat.inputs[0])
    level = nodes.new("ShaderNodeVectorMath")
    level.operation = 'NORMALIZE'
    links.new(flat.outputs[0], level.inputs[0])
    lift = nodes.new("ShaderNodeVectorMath")
    lift.operation = 'ADD'
    lift.inputs[1].default_value = (0.0, 0.0, look)
    links.new(level.outputs[0], lift.inputs[0])

    horizon = nodes.new("ShaderNodeTexSky")
    _set_enum(horizon, "sky_type", 'MULTIPLE_SCATTERING', 'NISHITA')
    horizon.sun_elevation = math.radians(elevation)
    horizon.sun_rotation = math.radians(rotation)
    # The sun's disc would put a blinding spot in the haze wherever the sun sits low.
    horizon.sun_disc = False
    _set_haze(horizon, haze)
    links.new(lift.outputs[0], horizon.inputs["Vector"])

    color = nodes.new("ShaderNodeMix")
    color.data_type = 'RGBA'
    color.blend_type = 'MULTIPLY'
    color.inputs[0].default_value = 1.0
    links.new(horizon.outputs[0], color.inputs[6])
    color.inputs[7].default_value = (*tint, 1.0)
    glow_color = color.outputs[2]
    if elevation < 0.0:
        night = nodes.new("ShaderNodeMix")
        night.data_type = 'RGBA'
        night.blend_type = 'ADD'
        night.inputs[0].default_value = 1.0
        links.new(glow_color, night.inputs[6])
        night.inputs[7].default_value = (*(c * t / strength for c, t in zip(NIGHT_COLOR, tint)), 1.0)
        glow_color = night.outputs[2]

    glow = nodes.new("ShaderNodeEmission")
    glow.inputs["Strength"].default_value = strength
    links.new(glow_color, glow.inputs["Color"])
    mix = nodes.new("ShaderNodeMixShader")
    links.new(factor.outputs[0], mix.inputs[0])
    links.new(enter.outputs[0], mix.inputs[1])
    links.new(glow.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], leave.inputs[0])
    return group


def _set_moon(scene, elevation, rotation):
    """After sunset the sky gives no light to speak of; a dim blue moon keeps the land readable."""
    moon = bpy.data.objects.get(MOON_NAME)
    if elevation >= 0.0:
        if moon is not None:
            bpy.data.objects.remove(moon)
        return None
    if moon is None:
        data = bpy.data.lights.new(MOON_NAME, 'SUN')
        moon = bpy.data.objects.new(MOON_NAME, data)
    if moon.name not in scene.collection.all_objects:
        scene.collection.objects.link(moon)
    moon.data.color = MOON_COLOR
    moon.data.energy = MOON_STRENGTH
    # A wide disc gives the soft-edged shadows of a night exposure.
    moon.data.angle = math.radians(3.0)
    turn, lift = math.radians(rotation), math.radians(MOON_ELEVATION)
    toward = Vector((math.sin(turn) * math.cos(lift), math.cos(turn) * math.cos(lift), math.sin(lift)))
    moon.rotation_euler = toward.to_track_quat('Z', 'Y').to_euler()
    return moon


def haze_everything(scene):
    """Give the haze to materials that did not come from this add-on. Returns how many changed."""
    changed = 0
    seen = set()
    for ob in scene.objects:
        for slot in ob.material_slots:
            material = slot.material
            if material is not None and material.name not in seen:
                seen.add(material.name)
                changed += materials.add_haze(material)
    return changed


def build(scene, elevation, rotation, haze, clouds, stars=0.0, mist=0.0, mist_color=(1.0, 1.0, 1.0)):
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
    low_sun = max(0.0, min(1.0, (20.0 - elevation) / 20.0))
    strength = SKY_STRENGTH * (1.0 + 9.0 * low_sun * low_sun) if elevation >= 0.0 else SKY_STRENGTH

    coords = nodes.new("ShaderNodeTexCoord")
    direction = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coords.outputs["Generated"], direction.inputs[0])
    up = direction.outputs["Z"]

    look = HORIZON_LOOK if sky.sky_type == 'MULTIPLE_SCATTERING' else HORIZON_LOOK_LEGACY

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
        # One star at most in each cell of a cellular pattern: a dot at the cell's centre,
        # its brightness the cell's random number raised to a power so that most are faint.
        cells = nodes.new("ShaderNodeTexVoronoi")
        cells.inputs["Scale"].default_value = 130.0
        links.new(coords.outputs["Generated"], cells.inputs["Vector"])
        dot = nodes.new("ShaderNodeMapRange")
        for socket, value in zip(list(dot.inputs)[1:5], (0.04, 0.14, 1.0, 0.0)):
            socket.default_value = value
        links.new(cells.outputs["Distance"], dot.inputs[0])
        random = nodes.new("ShaderNodeSeparateColor")
        links.new(cells.outputs["Color"], random.inputs[0])
        faint = nodes.new("ShaderNodeMath")
        faint.operation = 'POWER'
        faint.inputs[1].default_value = 7.0
        links.new(random.outputs[0], faint.inputs[0])
        bright = nodes.new("ShaderNodeMath")
        bright.operation = 'MULTIPLY'
        links.new(dot.outputs[0], bright.inputs[0])
        links.new(faint.outputs[0], bright.inputs[1])
        level = nodes.new("ShaderNodeMath")
        level.operation = 'MULTIPLY'
        level.inputs[1].default_value = STAR_BRIGHTNESS * stars / strength
        links.new(bright.outputs[0], level.inputs[0])
        add = nodes.new("ShaderNodeMix")
        add.data_type = 'RGBA'
        add.blend_type = 'ADD'
        add.inputs[0].default_value = 1.0
        links.new(color, add.inputs[6])
        links.new(level.outputs[0], add.inputs[7])
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
    _build_haze(scene, elevation, rotation, haze, strength, mist, mist_color, look)
    _set_moon(scene, elevation, rotation)
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
        self.mist = preset.get("mist", 0.0)
        self.mist_color = preset.get("mist_color", (1.0, 1.0, 1.0))

    preset: EnumProperty(name="Preset", items=SKY_ITEMS, update=_load_preset)
    elevation: FloatProperty(
        name="Sun Height", description="Degrees above the horizon, negative is night",
        default=62.0, min=-30.0, max=90.0,
    )
    rotation: FloatProperty(name="Sun Direction", description="Degrees around the horizon", default=40.0)
    haze: FloatProperty(name="Sky Dust", description="Dust in the air: a whiter, warmer sky", default=1.0, min=0.0, max=10.0)
    clouds: FloatProperty(name="Clouds", description="Cloud cover", default=0.15, min=0.0, max=1.0, subtype='FACTOR')
    stars: FloatProperty(name="Stars", default=0.0, min=0.0, max=2.0)
    mist: FloatProperty(
        name="Distance Haze", description="How strongly far-away land fades into the horizon",
        default=0.25, min=0.0, soft_max=1.0, max=4.0, subtype='FACTOR',
    )
    mist_color: FloatVectorProperty(
        name="Haze Tint", description="Colours the haze; white leaves it the colour of the sky on the horizon",
        subtype='COLOR', size=3, default=(1.0, 1.0, 1.0), min=0.0, soft_max=1.0,
    )
    haze_all: BoolProperty(
        name="Haze Every Material",
        description="Also fade objects whose materials were not made by Scenery",
        default=False,
    )

    def execute(self, context):
        build(
            context.scene, self.elevation, self.rotation, self.haze, self.clouds, self.stars,
            self.mist, tuple(self.mist_color),
        )
        if self.haze_all:
            haze_everything(context.scene)
        return {'FINISHED'}


classes = (
    ELYAN_OT_sky_set,
)
