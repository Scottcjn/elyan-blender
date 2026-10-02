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


def _ease(value):
    """Smooth step from 0 to 1."""
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _hold(time, length, rise=0.45, fall=0.55, start=0.0):
    """Rise from 0 to 1, stay, and come back to 0 by the end of a one-shot clip."""
    return _ease((time - start) / rise) * _ease((length - time) / fall)


# Where the arms hang. The character faces -Y, its left is +X.
_DOWN = {
    "l": (Vector((0.16, 0.03, -1.0)), Vector((0.10, -0.22, -1.0))),
    "r": (Vector((-0.16, 0.03, -1.0)), Vector((-0.10, -0.22, -1.0))),
}
# How much more each finger curls than the index: a hand at rest closes toward the little finger.
_FINGERS = (("index", 0.8), ("middle", 1.0), ("ring", 1.15), ("pinky", 1.3))
_RELAXED = (16.0, 22.0, 14.0)
_FIST = (72.0, 88.0, 58.0)


def _mix(low, high, weight):
    return low + (high - low) * weight


def _arm(side, weight=0.0, upper=None, lower=None):
    """
    One arm, ``weight`` of the way from hanging at the side to pointing along ``upper``/``lower``.

    The rig rests with arms out; people stand with them at their sides. Aiming at a
    direction, not turning by an angle, works whatever the rest pose is.
    """
    down_upper, down_lower = _DOWN[side]
    upper = down_upper if upper is None else down_upper.lerp(Vector(upper).normalized() * 1.02, weight)
    lower = down_lower if lower is None else down_lower.lerp(Vector(lower).normalized() * 1.05, weight)
    return [("upperarm_" + side, "aim", upper), ("lowerarm_" + side, "aim", lower)]


def _hand(side, fist=0.0, index=None, spread=0.0):
    """
    Finger curls: relaxed at 0, a fist at 1. ``index`` sets the index finger apart
    (0 for a pointing hand); ``spread`` straightens every finger for an open palm.
    """
    turns = []
    for finger, share in _FINGERS:
        amount = fist if index is None or finger != "index" else index
        for joint, (relaxed, closed) in enumerate(zip(_RELAXED, _FIST), 1):
            degrees = _mix(relaxed * share * (1.0 - spread), closed, amount)
            turns.append(("{:s}_{:02d}_{:s}".format(finger, joint, side), "curl", degrees))
    # The thumb rests splayed wide in the rig. It folds in the plane of the palm, toward the fingers.
    for joint, relaxed, closed in ((1, 24.0, 38.0), (2, 8.0, 30.0), (3, 10.0, 40.0)):
        turns.append(("thumb_{:02d}_{:s}".format(joint, side), "tuck", _mix(relaxed, closed, fist)))
    return turns


def _body(time, length):
    """Breathing and a slow sway, with the feet kept where they stand."""
    breath = _wave(time, length, 1)
    # Whole cycles only: half a cycle ends where it began but heading the other way.
    sway = _wave(time, length, 1, 0.2)
    return [
        ("pelvis", _Y, 1.4 * sway),
        ("spine_01", _Y, -1.0 * sway),
        ("spine_02", _X, -0.9 * breath),
        ("spine_03", _X, -1.1 * breath),
        ("clavicle_l", _Y, -0.8 * breath), ("clavicle_r", _Y, 0.8 * breath),
        ("neck_01", _X, 0.6 * breath),
        ("head", _Z, 1.6 * _wave(time, length, 1, 0.35)),
        ("head", _X, 0.8 * _wave(time, length, 2, 0.1)),
    ] + _FEET


# Whatever the hips do, the legs follow so the feet neither slide nor tilt.
_FEET = [
    ("thigh_l", "pin", "foot_l"), ("thigh_r", "pin", "foot_r"),
    ("foot_l", "level", None), ("foot_r", "level", None),
]


def _idle(time, length):
    return _body(time, length) + _arm("l") + _arm("r") + _hand("l") + _hand("r")


def _listen(time, length):
    return _idle(time, length) + [
        ("spine_02", _X, 2.0),
        ("head", _Y, 4.0 + 1.0 * _wave(time, length, 1, 0.6)),
        ("head", _X, 2.5 * max(0.0, _wave(time, length, 2, 0.75)) ** 2),
    ]


def _talk(time, length):
    # Beats: the left hand opens outward twice, the right answers, then both together.
    beat_l = _pulse(time, 0.25, 0.9) + 0.8 * _pulse(time, 2.9, 0.9)
    beat_r = _pulse(time, 1.2, 1.0) + 0.8 * _pulse(time, 2.9, 0.9)
    return _body(time, length) + [
        ("head", _X, 2.2 * _wave(time, length, 3)),
        ("head", _Z, 2.5 * _wave(time, length, 2, 0.15)),
        ("spine_03", _Z, 1.5 * _wave(time, length, 1, 0.4)),
    ] + (
        _arm("l", 0.8 * beat_l, (0.22, -0.12, -1.0), (0.42, -0.9, 0.12))
        + _arm("r", 0.8 * beat_r, (-0.22, -0.12, -1.0), (-0.42, -0.9, 0.12))
        + [("lowerarm_l", "twist", -55.0 * beat_l), ("lowerarm_r", "twist", 55.0 * beat_r)]
        + _hand("l", spread=0.7 * min(1.0, beat_l)) + _hand("r", spread=0.7 * min(1.0, beat_r))
    )


def _nod(time, length):
    return _idle(time, 4.0) + [("head", _X, 10.0 * (_pulse(time, 0.1, 0.5) + 0.7 * _pulse(time, 0.6, 0.45)))]


def _shake(time, length):
    envelope = _pulse(time, 0.0, length)
    return _idle(time, 4.0) + [("head", _Z, 14.0 * envelope * math.sin(math.tau * 2.0 * time / length))]


def _agree(time, length):
    nods = _pulse(time, 0.15, 0.5) + 0.6 * _pulse(time, 0.7, 0.45)
    lean = _hold(time, length, 0.4, 0.6)
    return _idle(time, 4.0) + [
        ("head", _X, 6.0 * nods), ("neck_01", _X, 2.0 * nods),
        ("spine_02", _X, 2.5 * lean), ("spine_03", _X, 1.5 * lean),
    ]


def _think(time, length):
    held = _hold(time, length, 0.7, 0.8)
    return _body(time, 4.0) + [
        ("head", _Y, -7.0 * held), ("head", _Z, -9.0 * held), ("head", _X, 4.0 * held),
        ("spine_03", _Z, -3.0 * held),
    ] + (
        _arm("l") + _arm("r", held, (-0.10, -0.75, -0.75), (0.36, 0.02, 1.0))
        + [("lowerarm_r", "twist", 40.0 * held)]
        + _hand("l") + _hand("r", fist=0.55 * held, index=0.15 * held)
    )


def _greet(time, length):
    held = _hold(time, length, 0.5, 0.6)
    wave = held * math.sin(math.tau * 3.0 * (time - 0.5) / (length - 1.1)) if 0.5 < time < length - 0.6 else 0.0
    return _body(time, 4.0) + [
        ("head", _Y, -3.0 * held), ("spine_03", _Y, -2.0 * held),
        ("lowerarm_r", _Y, 16.0 * wave),
    ] + (
        _arm("l") + _arm("r", held, (-1.0, -0.25, -0.45), (-0.30, -0.25, 1.0))
        + [("lowerarm_r", "twist", 70.0 * held)]
        + _hand("l") + _hand("r", spread=held)
    )


def _shrug(time, length):
    held = _hold(time, length, 0.4, 0.6)
    return _body(time, 4.0) + [
        ("clavicle_l", _Y, -11.0 * held), ("clavicle_r", _Y, 11.0 * held),
        ("head", _Y, 5.0 * held), ("head", _X, -3.0 * held),
    ] + (
        _arm("l", 0.75 * held, (0.30, 0.0, -1.0), (0.75, -0.75, -0.25))
        + _arm("r", 0.75 * held, (-0.30, 0.0, -1.0), (-0.75, -0.75, -0.25))
        + [("lowerarm_l", "twist", -80.0 * held), ("lowerarm_r", "twist", 80.0 * held)]
        + _hand("l", spread=held) + _hand("r", spread=held)
    )


def _point(time, length):
    held = _hold(time, length, 0.5, 0.6)
    return _body(time, 4.0) + [
        ("spine_03", _Z, 4.0 * held), ("head", _X, 2.0 * held),
    ] + (
        _arm("l") + _arm("r", held, (-0.22, -1.0, -0.45), (-0.08, -1.0, -0.12))
        + _hand("l") + _hand("r", fist=held, index=0.0)
    )


def _weight_shift(time, length):
    side = _wave(time, length, 1)
    return [turn for turn in _idle(time, length) if turn[0] != "pelvis"] + [
        # The hips travel over one foot and drop on the other side; the spine bends back to balance.
        ("pelvis", "move", Vector((0.03 * side, 0.0, -0.003 * side * side))),
        ("pelvis", _Y, 1.6 * side),
        ("spine_01", _Y, -2.0 * side), ("spine_03", _Y, -1.4 * side),
        ("head", _Y, 1.2 * side),
    ]


# name: (pose function, seconds, loops)
CLIPS = {
    "idle": (_idle, 4.0, True),
    "listen": (_listen, 4.0, True),
    "talk": (_talk, 4.0, True),
    "weight_shift": (_weight_shift, 6.0, True),
    "nod": (_nod, 1.2, False),
    "shake": (_shake, 1.4, False),
    "agree": (_agree, 1.6, False),
    "think": (_think, 3.6, False),
    "greet": (_greet, 2.6, False),
    "shrug": (_shrug, 1.8, False),
    "point": (_point, 2.2, False),
}
# The moment of each clip that shows what it does, for contact sheets.
SHOWN_AT = {
    "idle": 1.0, "listen": 1.4, "talk": 0.7, "weight_shift": 1.5, "nod": 0.35, "shake": 0.5,
    "agree": 0.4, "think": 1.8, "greet": 1.2, "shrug": 0.9, "point": 1.1,
}


def _pose(rig, turns):
    """
    Apply turns on top of the rest pose, parents first. Returns the bones posed.

    A turn is ``(bone, axis, degrees)`` about an axis of the rig, or one of
    ``(bone, "aim", direction)``   point the bone along a direction,
    ``(bone, "twist", degrees)``   turn it about its own length,
    ``(bone, "curl", degrees)``    bend a finger about its hand's knuckle line,
    ``(bone, "tuck", degrees)``    fold a thumb in the plane of its palm,
    ``(bone, "pin", other)``       turn it so a bone further down stays where it rests,
    ``(bone, "level", None)``      give it back the orientation it rests in,
    ``(bone, "move", offset)``     shift it.

    Matrices are worked out here and not read back from Blender, which would need the
    whole character re-evaluated after every bone. That holds for plain rigs: no
    constraints, every bone inheriting its parent's rotation.
    """
    by_bone = {}
    for name, axis, amount in turns:
        by_bone.setdefault(name, []).append((axis, amount))
    rest = {bone.name: bone.bone.matrix_local for bone in rig.pose.bones}
    posed = {}
    touched = []

    def carried(name, by):
        """Rest head of bone ``name`` as carried along by the posed bone ``by``."""
        return posed[by] @ rest[by].inverted() @ rest[name].translation

    # ``pose.bones`` lists parents before children.
    for bone in rig.pose.bones:
        bone.rotation_mode = 'QUATERNION'
        name = bone.name
        if bone.parent is None:
            start = rest[name].copy()
        else:
            start = posed[bone.parent.name] @ rest[bone.parent.name].inverted() @ rest[name]
        matrix = start.copy()
        for axis, amount in by_bone.get(name, ()):
            here = matrix.to_3x3()
            if axis == "move":
                matrix = Matrix.Translation(amount) @ matrix
                continue
            if axis == "level":
                matrix = Matrix.Translation(matrix.translation) @ rest[name].to_3x3().to_4x4()
                continue
            if axis == "aim":
                # Bones point along their own Y axis.
                turn = here.col[1].rotation_difference(amount).to_matrix()
            elif axis == "twist":
                turn = Matrix.Rotation(math.radians(amount), 3, here.col[1])
            elif axis == "curl":
                side = name[-1]
                hand = "hand_" + side
                across = carried("pinky_01_" + side, hand) - carried("index_01_" + side, hand)
                # Mirrored hands curl about mirrored lines.
                turn = Matrix.Rotation(math.radians(amount if side == "l" else -amount), 3, across.normalized())
            elif axis == "tuck":
                side = name[-1]
                hand = "hand_" + side
                across = carried("pinky_01_" + side, hand) - carried("index_01_" + side, hand)
                # The back of the hand: along the hand, crossed with the knuckle line.
                turn = Matrix.Rotation(math.radians(amount), 3, posed[hand].to_3x3().col[1].cross(across).normalized())
            elif axis == "pin":
                posed[name] = matrix
                now = carried(amount, name) - matrix.translation
                turn = now.rotation_difference(rest[amount].translation - matrix.translation).to_matrix()
            else:
                turn = Matrix.Rotation(math.radians(amount), 3, axis)
            pivot = Matrix.Translation(matrix.translation)
            matrix = pivot @ turn.to_4x4() @ pivot.inverted() @ matrix
        posed[name] = matrix
        if name in by_bone:
            bone.matrix_basis = start.inverted() @ matrix
            touched.append(bone)
        else:
            bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    return touched


def _turns(rig, name, time):
    """A clip's turns at ``time``, without those for bones the rig lacks."""
    function, length, _loops = CLIPS[name]
    present = {bone.name for bone in rig.pose.bones}
    return [
        turn for turn in function(time, length)
        if turn[0] in present and _has_all(present, turn)
    ]


def _has_all(present, turn):
    """Whether the other bones a turn refers to exist."""
    name, kind, amount = turn
    # ``kind`` may be a vector, which compares but does not hash.
    if kind == "pin":
        return amount in present
    if kind in ("curl", "tuck"):
        side = name[-1]
        return {"hand_" + side, "index_01_" + side, "pinky_01_" + side} <= present
    return True


def add_clips(rig, names=None):
    """
    Create the named clips (default: all) as actions on ``rig``, parked in NLA tracks
    so exporters write each one. Returns the names the animations are exported under.
    """
    names = list(names or CLIPS)
    unknown = [name for name in names if name not in CLIPS]
    if unknown:
        raise ValueError("unknown clips: {:s}; known: {:s}".format(", ".join(unknown), ", ".join(CLIPS)))
    scene = bpy.context.scene
    scene.render.fps = FPS
    animation = rig.animation_data or rig.animation_data_create()
    made = []
    for name in names:
        _function, length, loops = CLIPS[name]
        action = bpy.data.actions.new(name)
        animation.action = action
        last = int(round(length * FPS))
        frames = sorted(set(range(0, last, KEY_EVERY)) | {last})
        for frame in frames:
            # A loop's last frame is its first, so it closes exactly.
            time = 0.0 if loops and frame == last else frame / FPS
            turns = _turns(rig, name, time)
            moved = {turn[0] for turn in turns if turn[1] == "move"}
            for bone in _pose(rig, turns):
                bone.keyframe_insert("rotation_quaternion", frame=frame)
                if bone.name in moved:
                    bone.keyframe_insert("location", frame=frame)
        animation.action = None
        track = animation.nla_tracks.new()
        track.name = name
        track.strips.new(name, 0, action)
        track.mute = True
        # Exporters name an animation after its track, which no earlier action can have taken.
        made.append(track.name)
    _pose(rig, [])
    return made


def show(rig, name, time):
    """Put the rig in a clip's pose at ``time`` seconds, without creating any animation."""
    _pose(rig, _turns(rig, name, time % CLIPS[name][1]))


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
