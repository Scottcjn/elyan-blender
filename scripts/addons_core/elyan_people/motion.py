# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Body motion for people who stand and talk: generated, looping, and small.

Clips are sums of slow sine waves with whole numbers of cycles per loop, so
they repeat without a seam and need no motion capture. They are made for the
"game_engine" rig; bones a rig lacks are skipped.
"""

import math

import bpy
from mathutils import Matrix, Vector

FPS = 30
# Keys this far apart are enough for motion this slow; the curves between are smooth.
KEY_EVERY = 3

_X, _Y, _Z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))


def _wave(time, length, cycles, phase=0.0):
    """Sine with ``cycles`` whole cycles over a clip of ``length`` seconds."""
    return math.sin(math.tau * (cycles * time / length + phase))


def _pulse(time, start, length):
    """One smooth bump from 0 to 1 and back, starting at ``start``."""
    if not start <= time <= start + length:
        return 0.0
    return 0.5 - 0.5 * math.cos(math.tau * (time - start) / length)


def _arms_down():
    """
    The rig rests with arms out; people stand with them at their sides.

    Aiming at a direction, not turning by an angle, works whatever the rest pose is.
    The character faces -Y, its left is +X.
    """
    return [
        ("upperarm_l", "aim", Vector((0.16, 0.03, -1.0))), ("upperarm_r", "aim", Vector((-0.16, 0.03, -1.0))),
        ("lowerarm_l", "aim", Vector((0.10, -0.22, -1.0))), ("lowerarm_r", "aim", Vector((-0.10, -0.22, -1.0))),
    ]


def _idle(time, length):
    breath = _wave(time, length, 1)
    sway = _wave(time, length, 0.5 if length >= 6 else 1, 0.2)
    return _arms_down() + [
        ("pelvis", _Y, 1.4 * sway),
        ("spine_01", _Y, -1.0 * sway),
        ("spine_02", _X, -0.9 * breath),
        ("spine_03", _X, -1.1 * breath),
        ("clavicle_l", _Y, -0.8 * breath), ("clavicle_r", _Y, 0.8 * breath),
        ("neck_01", _X, 0.6 * breath),
        ("head", _Z, 1.6 * _wave(time, length, 1, 0.35)),
        ("head", _X, 0.8 * _wave(time, length, 2, 0.1)),
    ]


def _listen(time, length):
    return _idle(time, length) + [
        ("spine_02", _X, 2.0),
        ("head", _Y, 4.0 + 1.0 * _wave(time, length, 1, 0.6)),
        ("head", _X, 2.5 * max(0.0, _wave(time, length, 2, 0.75)) ** 2),
    ]


def _talk(time, length):
    beat_l = sum(_pulse(time, start, 0.7) for start in (0.3, 2.2))
    beat_r = sum(_pulse(time, start, 0.8) for start in (1.2, 3.0))
    return _idle(time, length) + [
        ("lowerarm_l", _X, -34.0 * beat_l), ("upperarm_l", _X, -9.0 * beat_l),
        ("lowerarm_r", _X, -30.0 * beat_r), ("upperarm_r", _X, -8.0 * beat_r),
        ("head", _X, 2.2 * _wave(time, length, 3)),
        ("head", _Z, 2.5 * _wave(time, length, 2, 0.15)),
        ("spine_03", _Z, 1.5 * _wave(time, length, 1, 0.4)),
    ]


def _nod(time, length):
    return _idle(time, 4.0) + [("head", _X, 10.0 * (_pulse(time, 0.1, 0.5) + 0.7 * _pulse(time, 0.6, 0.45)))]


def _shake(time, length):
    envelope = _pulse(time, 0.0, length)
    return _idle(time, 4.0) + [("head", _Z, 14.0 * envelope * math.sin(math.tau * 2.0 * time / length))]


# name: (pose function, seconds, loops)
CLIPS = {
    "idle": (_idle, 4.0, True),
    "listen": (_listen, 4.0, True),
    "talk": (_talk, 4.0, True),
    "nod": (_nod, 1.2, False),
    "shake": (_shake, 1.4, False),
}


def _pose(rig, turns):
    """
    Apply turns on top of the rest pose, parents first.

    A turn is ``(bone, axis, degrees)`` about a world axis, or ``(bone, "aim", direction)``
    to point the bone along a world direction.
    """
    for bone in rig.pose.bones:
        bone.rotation_mode = 'QUATERNION'
        bone.matrix_basis = Matrix.Identity(4)
    by_bone = {}
    for name, axis, amount in turns:
        by_bone.setdefault(name, []).append((axis, amount))
    touched = []
    # ``pose.bones`` lists parents before children.
    for bone in rig.pose.bones:
        if bone.name not in by_bone:
            continue
        bpy.context.view_layer.update()
        matrix = bone.matrix.copy()
        pivot = Matrix.Translation(matrix.translation)
        for axis, amount in by_bone[bone.name]:
            if isinstance(axis, str):
                # Bones point along their own Y axis.
                turn = matrix.to_3x3().col[1].rotation_difference(amount).to_matrix().to_4x4()
            else:
                turn = Matrix.Rotation(math.radians(amount), 4, axis)
            matrix = pivot @ turn @ pivot.inverted() @ matrix
        bone.matrix = matrix
        touched.append(bone)
    bpy.context.view_layer.update()
    return touched


def add_clips(rig, names=None):
    """
    Create the named clips (default: all) as actions on ``rig``, parked in NLA tracks
    so exporters write each one. Returns the action names.
    """
    names = list(names or CLIPS)
    unknown = [name for name in names if name not in CLIPS]
    if unknown:
        raise ValueError("unknown clips: {:s}; known: {:s}".format(", ".join(unknown), ", ".join(CLIPS)))
    scene = bpy.context.scene
    scene.render.fps = FPS
    animation = rig.animation_data or rig.animation_data_create()
    present = {bone.name for bone in rig.pose.bones}
    made = []
    for name in names:
        function, length, loops = CLIPS[name]
        action = bpy.data.actions.new(name)
        animation.action = action
        last = int(round(length * FPS))
        frames = sorted(set(range(0, last, KEY_EVERY)) | {last})
        for frame in frames:
            # A loop's last frame is its first, so it closes exactly.
            time = 0.0 if loops and frame == last else frame / FPS
            turns = [turn for turn in function(time, length) if turn[0] in present]
            for bone in _pose(rig, turns):
                bone.keyframe_insert("rotation_quaternion", frame=frame)
        animation.action = None
        track = animation.nla_tracks.new()
        track.name = name
        track.strips.new(name, 0, action)
        track.mute = True
        made.append(action.name)
    _pose(rig, [])
    return made


def show(rig, name, time):
    """Put the rig in a clip's pose at ``time`` seconds, without creating any animation."""
    function, length, _loops = CLIPS[name]
    present = {bone.name for bone in rig.pose.bones}
    _pose(rig, [turn for turn in function(time % length, length) if turn[0] in present])


def show_speech(meshes, track, time, rename=None):
    """Set the mouth of ``meshes`` to a lip-sync track at ``time`` seconds, for previews."""
    position = min(max(time, 0.0), track["duration"]) * track["fps"]
    index = min(int(position), len(track["frames"]) - 2)
    mix = position - index
    for slot, name in enumerate(track["names"]):
        value = track["frames"][index][slot] * (1.0 - mix) + track["frames"][index + 1][slot] * mix
        name = (rename or {}).get(name, name)
        for ob in meshes:
            keys = ob.data.shape_keys
            if keys and name in keys.key_blocks:
                keys.key_blocks[name].value = value
