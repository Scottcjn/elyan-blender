# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Procedural materials: terrain coloured by altitude and slope, water, rock, clay,
bark and leaves.

Materials are shared by name, asking twice for the same preset returns the same one.
Every material ends in the shared haze node group, which the sky fills in so that
distance fades all of them toward the horizon together.
"""

import bpy

HAZE_GROUP = "Scenery Haze"

# Colours are (shore, lowland, upland, peak, cliff); lines are the altitudes (0..1)
# where lowland, upland and peak colours take over.
# Snow is kept well short of white: in full sun anything brighter clips, and with it goes
# the shading that shows the shape of the land.
TERRAIN_PRESETS = {
    'ALPINE': {
        "label": "Alpine",
        "colors": ((0.16, 0.13, 0.08), (0.04, 0.13, 0.02), (0.10, 0.085, 0.065), (0.80, 0.83, 0.88), (0.06, 0.055, 0.05)),
        "lines": (0.06, 0.46, 0.70), "cliff": 0.9, "roughness": 0.85,
    },
    'DESERT': {
        "label": "Desert",
        "colors": ((0.55, 0.38, 0.20), (0.62, 0.42, 0.22), (0.50, 0.28, 0.14), (0.66, 0.47, 0.30), (0.36, 0.19, 0.10)),
        "lines": (0.10, 0.50, 0.85), "cliff": 0.7, "roughness": 0.9,
    },
    'VOLCANIC': {
        "label": "Volcanic",
        "colors": ((0.03, 0.03, 0.03), (0.06, 0.05, 0.05), (0.10, 0.06, 0.05), (0.25, 0.08, 0.03), (0.02, 0.02, 0.02)),
        "lines": (0.10, 0.50, 0.88), "cliff": 0.8, "roughness": 0.75,
    },
    'ARCTIC': {
        "label": "Arctic",
        "colors": ((0.07, 0.10, 0.13), (0.26, 0.33, 0.40), (0.46, 0.52, 0.60), (0.60, 0.64, 0.70), (0.035, 0.045, 0.06)),
        "lines": (0.06, 0.34, 0.62), "cliff": 0.95, "roughness": 0.6,
    },
    'TROPICAL': {
        "label": "Tropical",
        "colors": ((0.72, 0.62, 0.42), (0.10, 0.28, 0.05), (0.05, 0.16, 0.04), (0.24, 0.22, 0.17), (0.16, 0.13, 0.10)),
        "lines": (0.10, 0.40, 0.85), "cliff": 0.75, "roughness": 0.8,
    },
    'MOON': {
        "label": "Moon",
        "colors": ((0.16, 0.16, 0.16), (0.24, 0.24, 0.23), (0.32, 0.32, 0.31), (0.44, 0.44, 0.43), (0.12, 0.12, 0.12)),
        "lines": (0.10, 0.50, 0.85), "cliff": 0.5, "roughness": 0.95,
    },
    'ALIEN': {
        "label": "Alien",
        "colors": ((0.10, 0.02, 0.14), (0.30, 0.04, 0.24), (0.05, 0.22, 0.25), (0.75, 0.80, 0.25), (0.05, 0.02, 0.09)),
        "lines": (0.10, 0.45, 0.80), "cliff": 0.8, "roughness": 0.6,
    },
    # Ground for places seen close up. The lowest band is the wet edge of a pond.
    'MEADOW': {
        "label": "Meadow",
        "colors": ((0.09, 0.065, 0.04), (0.05, 0.15, 0.025), (0.07, 0.18, 0.03), (0.09, 0.19, 0.04), (0.10, 0.08, 0.05)),
        "lines": (0.10, 0.55, 0.85), "cliff": 0.5, "roughness": 0.9,
    },
    'FOREST': {
        "label": "Forest Floor",
        "colors": ((0.06, 0.045, 0.03), (0.07, 0.075, 0.03), (0.10, 0.075, 0.04), (0.06, 0.09, 0.03), (0.07, 0.055, 0.04)),
        "lines": (0.10, 0.55, 0.85), "cliff": 0.5, "roughness": 0.95,
    },
    'SAND': {
        "label": "Sand",
        "colors": ((0.30, 0.22, 0.13), (0.62, 0.47, 0.28), (0.68, 0.52, 0.31), (0.72, 0.57, 0.36), (0.42, 0.30, 0.18)),
        "lines": (0.10, 0.55, 0.85), "cliff": 0.4, "roughness": 0.95,
    },
    'SNOW': {
        "label": "Snow",
        "colors": ((0.20, 0.28, 0.36), (0.55, 0.60, 0.67), (0.60, 0.64, 0.70), (0.63, 0.67, 0.72), (0.12, 0.14, 0.17)),
        "lines": (0.10, 0.55, 0.85), "cliff": 0.6, "roughness": 0.55,
    },
}

# Altitude over which one band's colour gives way to the next.
BAND_BLEND = 0.07

TERRAIN_ITEMS = tuple((key, preset["label"], "") for key, preset in TERRAIN_PRESETS.items())


def _new(name):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    return material, tree.nodes, tree.links


def _math(nodes, operation, a=None, b=None):
    node = nodes.new("ShaderNodeMath")
    node.operation = operation
    for socket, value in zip(node.inputs, (a, b)):
        if value is not None:
            socket.default_value = value
    return node


def _map_range(nodes, from_min, from_max, to_min=0.0, to_max=1.0):
    node = nodes.new("ShaderNodeMapRange")
    for socket, value in zip(list(node.inputs)[1:5], (from_min, from_max, to_min, to_max)):
        socket.default_value = value
    return node


def _bump(nodes, links, height_socket, strength):
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    links.new(height_socket, bump.inputs["Height"])
    return bump


def haze_group():
    """
    The node group every scenery material passes through on its way to the output.

    It starts out doing nothing; ``sky.build`` rewrites its inside. Being one shared
    group, a change of sky reaches every material without touching any of them.
    """
    group = bpy.data.node_groups.get(HAZE_GROUP)
    if group is None:
        group = bpy.data.node_groups.new(HAZE_GROUP, "ShaderNodeTree")
        group.interface.new_socket("Shader", in_out='INPUT', socket_type="NodeSocketShader")
        group.interface.new_socket("Shader", in_out='OUTPUT', socket_type="NodeSocketShader")
        enter = group.nodes.new("NodeGroupInput")
        leave = group.nodes.new("NodeGroupOutput")
        group.links.new(enter.outputs[0], leave.inputs[0])
    return group


def add_haze(material):
    """Route a material's surface through the haze group. Returns False if it already was, or cannot be."""
    tree = material.node_tree
    if tree is None:
        return False
    group = haze_group()
    output = next((node for node in tree.nodes if node.type == 'OUTPUT_MATERIAL' and node.is_active_output), None)
    if output is None or not output.inputs["Surface"].is_linked:
        return False
    source = output.inputs["Surface"].links[0].from_socket
    if source.node.type == 'GROUP' and source.node.node_tree == group:
        return False
    node = tree.nodes.new("ShaderNodeGroup")
    node.node_tree = group
    node.label = "Distance Haze"
    tree.links.new(source, node.inputs[0])
    tree.links.new(node.outputs[0], output.inputs["Surface"])
    return True


def _finish(nodes, links, bsdf):
    output = nodes.new("ShaderNodeOutputMaterial")
    haze = nodes.new("ShaderNodeGroup")
    haze.node_tree = haze_group()
    haze.label = "Distance Haze"
    links.new(bsdf.outputs[0], haze.inputs[0])
    links.new(haze.outputs[0], output.inputs["Surface"])


def terrain(key, lines=None):
    """
    Lowlands, uplands and peaks by altitude, bare rock wherever it is steep.

    ``lines`` replaces the preset's band altitudes, for ground whose bands must sit
    at measured heights (the wet edge of a pond). Such a material is made afresh,
    not shared.
    """
    preset = TERRAIN_PRESETS[key]
    name = "Scenery Terrain " + preset["label"]
    if lines is None and name in bpy.data.materials:
        return bpy.data.materials[name]
    material, nodes, links = _new(name)
    shore, low, mid, peak, cliff = preset["colors"]
    low_line, mid_line, peak_line = lines or preset["lines"]

    coords = nodes.new("ShaderNodeTexCoord")
    # Generated Z runs 0..1 from the lowest to the highest point of the terrain.
    altitude = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coords.outputs["Generated"], altitude.inputs[0])

    breakup = nodes.new("ShaderNodeTexNoise")
    breakup.inputs["Scale"].default_value = 0.06
    breakup.inputs["Detail"].default_value = 6.0
    links.new(coords.outputs["Object"], breakup.inputs["Vector"])
    wobble = _math(nodes, 'MULTIPLY_ADD', b=0.24)
    wobble.inputs[2].default_value = -0.12
    links.new(breakup.outputs[0], wobble.inputs[0])
    ragged = _math(nodes, 'ADD')
    links.new(altitude.outputs["Z"], ragged.inputs[0])
    links.new(wobble.outputs[0], ragged.inputs[1])

    ramp = nodes.new("ShaderNodeValToRGB")
    stops = ramp.color_ramp.elements
    stops[0].position, stops[0].color = 0.0, (*shore, 1.0)
    stops[1].position, stops[1].color = low_line, (*low, 1.0)
    # Each band holds its colour up to a short blend below the next line; a ramp that
    # blended all the way between lines would leave most of the land a muddy average.
    for position, color in (
            (mid_line - BAND_BLEND, low), (mid_line, mid), (peak_line - BAND_BLEND, mid), (peak_line, peak),
    ):
        stop = stops.new(position)
        stop.color = (*color, 1.0)
    links.new(ragged.outputs[0], ramp.inputs[0])

    geometry = nodes.new("ShaderNodeNewGeometry")
    normal = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geometry.outputs["Normal"], normal.inputs[0])
    # Normal Z is 1 on flat ground; below about 0.6 the ground is too steep to hold soil or snow.
    steep = _map_range(nodes, 0.78, 0.55, 0.0, preset["cliff"])
    links.new(normal.outputs["Z"], steep.inputs[0])

    mix = nodes.new("ShaderNodeMix")
    mix.data_type = 'RGBA'
    links.new(steep.outputs[0], mix.inputs[0])
    links.new(ramp.outputs["Color"], mix.inputs[6])
    mix.inputs[7].default_value = (*cliff, 1.0)

    grain = nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 0.9
    grain.inputs["Detail"].default_value = 8.0
    links.new(coords.outputs["Object"], grain.inputs["Vector"])

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = preset["roughness"]
    links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    links.new(_bump(nodes, links, grain.outputs[0], 0.35).outputs[0], bsdf.inputs["Normal"])
    _finish(nodes, links, bsdf)
    return material


def water():
    name = "Scenery Water"
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    material, nodes, links = _new(name)
    coords = nodes.new("ShaderNodeTexCoord")
    waves = nodes.new("ShaderNodeTexNoise")
    waves.inputs["Scale"].default_value = 0.7
    waves.inputs["Detail"].default_value = 4.0
    links.new(coords.outputs["Object"], waves.inputs["Vector"])
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.02, 0.10, 0.13, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.03
    bsdf.inputs["IOR"].default_value = 1.333
    bsdf.inputs["Transmission Weight"].default_value = 0.85
    links.new(_bump(nodes, links, waves.outputs[0], 0.08).outputs[0], bsdf.inputs["Normal"])
    _finish(nodes, links, bsdf)
    return material


def rock():
    name = "Scenery Rock"
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    material, nodes, links = _new(name)
    coords = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 2.5
    noise.inputs["Detail"].default_value = 10.0
    links.new(coords.outputs["Object"], noise.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.05, 0.05, 0.045, 1.0)
    ramp.color_ramp.elements[1].color = (0.30, 0.28, 0.25, 1.0)
    links.new(noise.outputs[0], ramp.inputs[0])
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.9
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(_bump(nodes, links, noise.outputs[0], 0.6).outputs[0], bsdf.inputs["Normal"])
    _finish(nodes, links, bsdf)
    return material


def clay(color=(0.55, 0.20, 0.12)):
    name = "Scenery Clay"
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    material, nodes, links = _new(name)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Coat Weight"].default_value = 0.4
    _finish(nodes, links, bsdf)
    return material


# (dark, light) leaf colours and (dark, light) bark colours for each plant species.
PLANT_COLORS = {
    'CONIFER': (((0.010, 0.045, 0.025), (0.035, 0.11, 0.045)), ((0.045, 0.028, 0.018), (0.12, 0.075, 0.05))),
    'BROADLEAF': (((0.025, 0.085, 0.015), (0.10, 0.23, 0.04)), ((0.05, 0.04, 0.03), (0.16, 0.13, 0.10))),
    'PALM': (((0.03, 0.10, 0.015), (0.15, 0.27, 0.05)), ((0.10, 0.08, 0.06), (0.26, 0.21, 0.15))),
    'DEAD': (None, ((0.09, 0.08, 0.07), (0.30, 0.28, 0.25))),
    'BUSH': (((0.03, 0.075, 0.015), (0.12, 0.19, 0.05)), ((0.05, 0.035, 0.02), (0.14, 0.10, 0.07))),
}


def _mottled(nodes, links, dark, light, scale, stretch=1.0):
    """Noise between two colours, in object space so a plant keeps its pattern wherever it stands."""
    coords = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale / stretch)
    links.new(coords.outputs["Object"], mapping.inputs["Vector"])
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = 5.0
    links.new(mapping.outputs[0], noise.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[0].color = 0.3, (*dark, 1.0)
    ramp.color_ramp.elements[1].position, ramp.color_ramp.elements[1].color = 0.7, (*light, 1.0)
    links.new(noise.outputs[0], ramp.inputs[0])
    return noise, ramp


def foliage(species):
    """Leaves: mottled green, each plant a slightly different shade so a forest is not a flat mass."""
    name = "Scenery Leaves " + species.title()
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    material, nodes, links = _new(name)
    dark, light = PLANT_COLORS[species][0]
    noise, ramp = _mottled(nodes, links, dark, light, 1.6)

    # Linked copies share this material; the object's random number is all that tells them apart.
    info = nodes.new("ShaderNodeObjectInfo")
    hue = _math(nodes, 'MULTIPLY_ADD', b=0.05)
    hue.inputs[2].default_value = 0.475
    links.new(info.outputs["Random"], hue.inputs[0])
    value = _math(nodes, 'MULTIPLY_ADD', b=0.6)
    value.inputs[2].default_value = 0.7
    links.new(info.outputs["Random"], value.inputs[0])
    shade = nodes.new("ShaderNodeHueSaturation")
    links.new(hue.outputs[0], shade.inputs["Hue"])
    links.new(value.outputs[0], shade.inputs["Value"])
    links.new(ramp.outputs["Color"], shade.inputs["Color"])

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.65
    links.new(shade.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(_bump(nodes, links, noise.outputs[0], 0.5).outputs[0], bsdf.inputs["Normal"])
    _finish(nodes, links, bsdf)
    return material


def bark(species):
    name = "Scenery Bark " + species.title()
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    material, nodes, links = _new(name)
    dark, light = PLANT_COLORS[species][1]
    # Stretched along the trunk, the noise reads as furrows.
    noise, ramp = _mottled(nodes, links, dark, light, 9.0, 8.0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.9
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(_bump(nodes, links, noise.outputs[0], 0.7).outputs[0], bsdf.inputs["Normal"])
    _finish(nodes, links, bsdf)
    return material


def ground(key):
    """Flat land around a terrain, in the colour of that terrain's lowest ground."""
    preset = TERRAIN_PRESETS[key]
    name = "Scenery Ground " + preset["label"]
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    material, nodes, links = _new(name)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*preset["colors"][0], 1.0)
    bsdf.inputs["Roughness"].default_value = preset["roughness"]
    _finish(nodes, links, bsdf)
    return material


def plain(name, color, roughness=0.6):
    """A flat colour, shared by name."""
    name = "Scenery " + name
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    material, nodes, links = _new(name)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    _finish(nodes, links, bsdf)
    return material


def grass(color):
    """Blades: darker at the root, each strand a slightly different shade."""
    name = "Scenery Grass"
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    material, nodes, links = _new(name)
    strand = nodes.new("ShaderNodeHairInfo")
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*(c * 0.35 for c in color), 1.0)
    ramp.color_ramp.elements[1].color = (*color, 1.0)
    links.new(strand.outputs["Intercept"], ramp.inputs[0])
    shade = nodes.new("ShaderNodeHueSaturation")
    value = _math(nodes, 'MULTIPLY_ADD', b=0.7)
    value.inputs[2].default_value = 0.65
    links.new(strand.outputs["Random"], value.inputs[0])
    links.new(value.outputs[0], shade.inputs["Value"])
    links.new(ramp.outputs["Color"], shade.inputs["Color"])
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.6
    links.new(shade.outputs["Color"], bsdf.inputs["Base Color"])
    _finish(nodes, links, bsdf)
    return material
