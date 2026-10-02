# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
The garment tools as commands of the Elyan LLM Bridge, so an agent needs no ``exec``.

Each takes the bridge's ``args`` (object names and numbers, as ``key=value``
from the command line) and returns the tool's report. The docstrings are what
``help garment_fit`` prints, so they are written as the manual.
"""

from . import ease, follow, seam, skin


def _call(func, args, required, optional):
    unknown = sorted(set(args) - set(required) - set(optional))
    if unknown:
        # A misspelt option would otherwise run with its default and look like success.
        raise ValueError("unknown argument {:s}; known: {:s}".format(
            ", ".join(unknown), ", ".join((*required, *optional))))
    missing = [name for name in required if args.get(name) in (None, "")]
    if missing:
        raise ValueError("needs {:s}".format(", ".join(missing)))
    report = func(**{key: value for key, value in args.items() if value is not None})
    if report.get("refused"):
        # The bridge client prints ``error`` for a reply that is not ok.
        report["error"] = "refused, nothing was changed: {:s}".format(report["refused"])
    return report


def cmd_garment_weights(args):
    """
    Skin a garment like the body: Armature modifier plus the body's bone weights.

    ``garment=NAME body=NAME``; optional ``rig=NAME`` (default: the body's
    armature), ``limit=4`` influences per vertex, ``region=BONE`` or
    ``region=["upperarm_l","lowerarm_l"]`` (body vertex groups to take weights
    from, so a sleeve cannot stick to the ribs), ``same_side=true`` (never take
    weights from the far half of the body: skirts between two legs),
    ``max_distance=0.1`` (refuse if a vertex is farther from its source),
    ``smooth=8`` (passes blending weights between neighbouring vertices where
    the cloth stands off the body, fully from ``smooth_distance=0.05``; left
    out, 8 passes when the plain result would shear, else none; 0 for never).

    Weights come from the nearest point of the body's surface, unposed. Only
    deform bones are transferred; other garment groups are kept.
    Reply: ``unweighted``, ``clamped``, ``max_distance``, ``wrong_side`` (a
    warning: source on the other side of the body), ``weight_step`` (largest
    change of weights along one edge, 0..1; near 1 the cloth shears there when
    posed, and ``warning`` says so), ``bones``,
    ``max_influences``, ``max_sum_error``. Refused, with nothing changed, if any
    vertex would be unweighted. Check a bent pose afterwards with ``check``
    ``clearance`` and a ``contact_sheet``.
    """
    return _call(skin.weights, args, ("garment", "body"),
                 ("rig", "limit", "region", "same_side", "max_distance", "region_min", "smooth", "smooth_distance"))


def cmd_garment_weld(args):
    """
    Move an open boundary loop of a garment exactly onto a ring.

    ``garment=NAME ring=[[x,y,z],...]`` (world-space points in order around the
    seam) or ``ring=OTHER_OBJECT`` (its open boundary); optional
    ``tolerance=1e-5``, ``loop=0`` or ``loop=GROUP`` (which garment boundary:
    an index from the reply's ``loops``, or a vertex group on it),
    ``ring_loop=`` likewise for the other object, ``rings=2`` (rows behind the
    boundary that follow with a falloff; 0 for none), ``max_move=0.05``,
    ``pin=GROUP``. Left out, the loop nearest the ring is used.

    Equal counts: vertex to ring point. Different counts: vertex to the
    nearest point on the ring's outline, and ``ring_to_boundary_after`` says how
    far the ring's own points are from a vertex. All shape keys move alike.
    Reply: ``before`` / ``after`` ``max`` and ``rms`` distance (unrounded),
    ``loop`` (which, how matched). Refused, with nothing changed, if a vertex
    would move more than ``max_move`` or the result is not within ``tolerance``.
    """
    return _call(seam.weld, args, ("garment", "ring"),
                 ("tolerance", "loop", "ring_loop", "rings", "max_move", "pin"))


def cmd_garment_fit(args):
    """
    Push the parts of a garment that touch or enter the body out to an ease.

    ``garment=NAME body=NAME``; optional ``ease=0.006`` (metres of clearance),
    ``max_move=0.03``, ``groups=GROUP`` or a list (only these garment vertices),
    ``pin=GROUP`` or a list (never moved: necklines, hems, welded seams),
    ``max_fraction=0`` (share of offending vertices allowed to stay too close),
    ``smooth=2``, ``faces=true`` (also lift faces the body shows through
    although their corners are clear; moves vertices that were clear).

    Not a shrinkwrap: vertices already clear stay exactly where they are. All
    shape keys move alike. Measured unposed: the body as shown, the garment as stored.
    Reply: ``before`` / ``after`` clearance (``min``, ``percentile_5``,
    ``below_threshold``, ``inside``, as ``check`` defines them), ``moved``,
    ``largest_move``, ``unresolved``, ``pinned_below``, and ``shows_through``
    before / after: body vertices poking out through the garment's faces, which
    clearance (measured at garment vertices) cannot see. Refused, with nothing
    changed, if too many vertices cannot clear within ``max_move``: for a
    garment built from a contract, change its ease constant and rebuild instead.
    """
    return _call(ease.fit, args, ("garment", "body"),
                 ("ease", "groups", "max_move", "pin", "max_fraction", "smooth", "faces"))


def cmd_garment_shapekeys(args):
    """
    Give a garment shape keys that follow the body's (body sliders, visemes).

    ``garment=NAME body=NAME``; optional ``keys=NAME`` or a list (default: all
    of the body's that are at 0), ``distance=0.03`` (cloth farther than this from the body
    does not follow; the influence fades out over the outer half),
    ``max_error=0.004``.

    Reply, per key in ``keys``: ``status`` (``written``, ``skipped`` when the
    key moves nothing near the garment, ``refused``), ``max_displacement``,
    ``clearance`` with the key on next to ``rest_clearance``. A key is refused
    and not written when its minimum clearance falls more than ``max_error``
    below the minimum at rest. ``ok`` is false if any key was refused.
    """
    return _call(follow.shapekeys, args, ("garment", "body"), ("keys", "distance", "max_error"))


COMMANDS = {
    "garment_weights": cmd_garment_weights,
    "garment_weld": cmd_garment_weld,
    "garment_fit": cmd_garment_fit,
    "garment_shapekeys": cmd_garment_shapekeys,
}


def install():
    """Add the commands to the bridge. False when the bridge add-on is not there."""
    try:
        from elyan_llm import commands as bridge
    except ImportError:
        return False
    bridge.COMMANDS.update(COMMANDS)
    return True


def remove():
    try:
        from elyan_llm import commands as bridge
    except ImportError:
        return
    for name, func in COMMANDS.items():
        # Only what was put there: another add-on may have replaced a name since.
        if bridge.COMMANDS.get(name) is func:
            del bridge.COMMANDS[name]
