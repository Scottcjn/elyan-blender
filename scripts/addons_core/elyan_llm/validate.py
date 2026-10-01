# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Numeric checks on assets, so problems are caught as numbers before anyone renders.

``measure()`` gathers metrics for a set of objects, ``validate()`` compares them
with a delivery profile and lists what is over budget or broken.
"""

import bpy
import numpy as np

# Limits per delivery target. The VRChat figures follow its published avatar
# performance ranks ("Good" on PC, "Medium" on Quest) and must be re-checked
# against the current SDK before relying on them; the web figures are our own.
PROFILES = {
    "web": {
        "triangles": 45000, "materials": 2, "skinned_meshes": 2, "bones": 90,
        "shape_keys": 70, "texture_size": 2048, "influences": 4,
    },
    "vrchat_pc": {
        "triangles": 70000, "materials": 8, "skinned_meshes": 2, "bones": 150,
        "shape_keys": 100, "texture_size": 4096, "influences": 4,
    },
    "quest": {
        "triangles": 15000, "materials": 2, "skinned_meshes": 2, "bones": 150,
        "shape_keys": 60, "texture_size": 1024, "influences": 4,
    },
    # Static props: no rig expected.
    "prop": {"triangles": 6000, "materials": 1, "texture_size": 1024},
}


def _evaluated_triangles(ob, depsgraph):
    mesh = ob.evaluated_get(depsgraph).data
    count = len(mesh.polygons)
    if count == 0:
        return 0
    totals = np.empty(count, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", totals)
    return int(totals.sum()) - 2 * count


def _mesh_defects(mesh):
    """Counts of the usual export breakers, from the unmodified mesh."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    defects = {
        "non_manifold_edges": sum(1 for e in bm.edges if len(e.link_faces) > 2),
        "boundary_edges": sum(1 for e in bm.edges if len(e.link_faces) == 1),
        "loose_vertices": sum(1 for v in bm.verts if not v.link_edges),
        "degenerate_faces": sum(1 for f in bm.faces if f.calc_area() < 1e-12),
    }
    bm.free()
    return defects


def _skin_weights(ob, bone_names):
    """How many vertices are unweighted, over-influenced, or not normalized."""
    group_is_bone = [group.name in bone_names for group in ob.vertex_groups]
    unweighted = over = unnormalized = 0
    most = 0
    for vert in ob.data.vertices:
        weights = [g.weight for g in vert.groups if group_is_bone[g.group] and g.weight > 0.0]
        if not weights:
            unweighted += 1
            continue
        most = max(most, len(weights))
        if len(weights) > 4:
            over += 1
        if abs(sum(weights) - 1.0) > 0.01:
            unnormalized += 1
    return {
        "unweighted_vertices": unweighted,
        "vertices_over_4_influences": over,
        "vertices_not_normalized": unnormalized,
        "max_influences": most,
    }


def measure(objects):
    """Metrics for ``objects`` taken together, plus one record per mesh."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    meshes = [ob for ob in objects if ob.type == 'MESH']
    armatures = {ob for ob in objects if ob.type == 'ARMATURE'}
    for ob in meshes:
        armatures.update(m.object for m in ob.modifiers if m.type == 'ARMATURE' and m.object)

    materials, images, parts = set(), {}, []
    totals = {
        "triangles": 0, "skinned_meshes": 0, "shape_keys": 0, "influences": 0,
        "non_manifold_edges": 0, "loose_vertices": 0, "degenerate_faces": 0,
        "unweighted_vertices": 0, "vertices_not_normalized": 0,
        "meshes_without_uv": 0, "objects_with_unapplied_scale": 0,
    }
    for ob in meshes:
        mesh = ob.data
        part = {"name": ob.name, "triangles": _evaluated_triangles(ob, depsgraph)}
        part.update(_mesh_defects(mesh))
        part["shape_keys"] = len(mesh.shape_keys.key_blocks) - 1 if mesh.shape_keys else 0
        part["uv_layers"] = len(mesh.uv_layers)
        part["scale_applied"] = all(abs(s - 1.0) < 1e-4 for s in ob.scale)
        rig = next((m.object for m in ob.modifiers if m.type == 'ARMATURE' and m.object), None)
        if rig is not None:
            part.update(_skin_weights(ob, {bone.name for bone in rig.data.bones}))
            totals["skinned_meshes"] += 1
            totals["influences"] = max(totals["influences"], part["max_influences"])
            totals["unweighted_vertices"] += part["unweighted_vertices"]
            totals["vertices_not_normalized"] += part["vertices_not_normalized"]
        for key in ("triangles", "non_manifold_edges", "loose_vertices", "degenerate_faces"):
            totals[key] += part[key]
        totals["shape_keys"] = max(totals["shape_keys"], part["shape_keys"])
        totals["meshes_without_uv"] += part["uv_layers"] == 0
        totals["objects_with_unapplied_scale"] += not part["scale_applied"]
        for material in mesh.materials:
            if material is None:
                continue
            materials.add(material.name)
            if material.node_tree:
                for node in material.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        images[node.image.name] = max(node.image.size)
        parts.append(part)

    totals["materials"] = len(materials)
    totals["bones"] = sum(len(rig.data.bones) for rig in armatures)
    totals["texture_size"] = max(images.values(), default=0)
    return {"totals": totals, "parts": parts, "textures": images}


# These must be zero whatever the profile.
_ALWAYS_ZERO = (
    "loose_vertices", "degenerate_faces", "unweighted_vertices", "vertices_not_normalized",
    "meshes_without_uv", "objects_with_unapplied_scale",
)


def validate(objects, profile):
    """Measure and compare with a profile. ``failures`` is empty when the asset passes."""
    limits = PROFILES[profile]
    report = measure(objects)
    totals = report["totals"]
    failures = [
        "{:s}: {:d} over the {:s} limit of {:d}".format(key, totals[key], profile, limit)
        for key, limit in limits.items() if totals[key] > limit
    ]
    failures += ["{:s}: {:d}, must be 0".format(key, totals[key]) for key in _ALWAYS_ZERO if totals[key]]
    report.update(profile=profile, limits=limits, failures=failures, passed=not failures)
    return report
