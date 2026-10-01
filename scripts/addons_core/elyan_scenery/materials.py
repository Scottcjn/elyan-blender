# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Procedural materials: terrain coloured by altitude and slope, water, rock, clay.

Materials are shared by name, asking twice for the same preset returns the same one.
"""

import bpy

# Colours are (shore, lowland, upland, peak, cliff); lines are the altitudes (0..1)
# where lowland, upland and peak colours take over.
TERRAIN_PRESETS = {
    'ALPINE': {
        "label": "Alpine",
        "colors": ((0.20, 0.17, 0.12), (0.07, 0.16, 0.04), (0.19, 0.17, 0.14), (0.85, 0.87, 0.90), (0.13, 0.12, 0.11)),
        "lines": (0.08, 0.45, 0.72), "cliff": 0.9, "roughness": 0.85,
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
        "colors": ((0.30, 0.40, 0.48), (0.55, 0.62, 0.70), (0.66, 0.71, 0.77), (0.78, 0.80, 0.83), (0.16, 0.20, 0.25)),
        "lines": (0.05, 0.30, 0.60), "cliff": 0.6, "roughness": 0.55,
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
}

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


def _finish(nodes, links, bsdf):
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs[0], output.inputs["Surface"])


def terrain(key):
    """Lowlands, uplands and peaks by altitude, bare rock wherever it is steep."""
    preset = TERRAIN_PRESETS[key]
    name = "Scenery Terrain " + preset["label"]
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    material, nodes, links = _new(name)
    shore, low, mid, peak, cliff = preset["colors"]
    low_line, mid_line, peak_line = preset["lines"]

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
    for position, color in ((mid_line, mid), (peak_line, peak)):
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
