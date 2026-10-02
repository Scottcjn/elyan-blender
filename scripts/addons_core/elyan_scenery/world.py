# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Worlds: choose a kind of environment and get the whole scene.

Two scales share one list. A *vista* is land seen from afar (a mountain range,
an island); a *place* is ground seen from standing height (a meadow with a
pond, a clearing). ``list_environments()`` describes them for a script or an
assistant; ``build()`` makes one.
"""

from bpy.props import EnumProperty, FloatProperty, IntProperty
from bpy.types import Operator

from . import instant, place, sky

# Vistas name a landscape and ground for ``instant.build``. Places carry what
# ``place.build`` reads: sizes in metres, trees as (species, count, (scale low, high)),
# flowers and treeline as (species, count).
ENVIRONMENTS = {
    'ALPINE_RANGE': {
        "label": "Alpine Range", "scale": 'VISTA',
        "description": "Snow-capped peaks above fir forest and a lake",
        "landscape": 'MOUNTAINS', "ground": 'ALPINE', "skies": ('FAIR', 'GOLDEN', 'NOON'),
    },
    'ROLLING_HILLS': {
        "label": "Rolling Hills", "scale": 'VISTA',
        "description": "Soft green hills with woods and water in the valleys",
        "landscape": 'HILLS', "ground": 'ALPINE', "skies": ('FAIR', 'GOLDEN', 'HAZE'),
    },
    'TROPICAL_ISLAND': {
        "label": "Tropical Island", "scale": 'VISTA',
        "description": "A green island with palms on the shore, alone in the sea",
        "landscape": 'ISLAND', "ground": 'TROPICAL', "skies": ('FAIR', 'NOON', 'SUNSET'),
    },
    'DESERT_DUNES': {
        "label": "Desert Dunes", "scale": 'VISTA',
        "description": "Wind-blown sand ridges to the horizon",
        "landscape": 'DUNES', "ground": 'DESERT', "skies": ('NOON', 'GOLDEN', 'SUNSET'),
    },
    'CANYONLANDS': {
        "label": "Canyonlands", "scale": 'VISTA',
        "description": "Flat red mesas cut by a gorge",
        "landscape": 'CANYON', "ground": 'DESERT', "skies": ('GOLDEN', 'NOON', 'SUNSET'),
    },
    'VOLCANO': {
        "label": "Volcano", "scale": 'VISTA',
        "description": "A lone dark cone rising from the water",
        "landscape": 'VOLCANO', "ground": 'VOLCANIC', "skies": ('SUNSET', 'HAZE', 'OVERCAST'),
    },
    'ARCTIC_PEAKS': {
        "label": "Arctic Peaks", "scale": 'VISTA',
        "description": "Ice and snow mountains over cold water",
        "landscape": 'MOUNTAINS', "ground": 'ARCTIC', "skies": ('FAIR', 'HAZE', 'OVERCAST'),
    },
    'MOONSCAPE': {
        "label": "Moonscape", "scale": 'VISTA',
        "description": "Grey cratered ground under stars",
        "landscape": 'CRATERS', "ground": 'MOON', "skies": ('NIGHT',), "water": False,
    },
    'ALIEN_WORLD': {
        "label": "Alien World", "scale": 'VISTA',
        "description": "Cratered ground in colours no earthly rock has",
        "landscape": 'CRATERS', "ground": 'ALIEN', "skies": ('SUNSET', 'NIGHT', 'HAZE'), "water": False,
    },
    'MEADOW_POND': {
        "label": "Meadow with Pond", "scale": 'PLACE',
        "description": "A sunny meadow: a lily pond ringed with stones, wildflowers, two or three trees",
        "size": 34.0, "relief": 0.4, "ground": 'MEADOW', "skies": ('FAIR', 'NOON', 'GOLDEN'),
        "pond": {"radii": (3.6, 2.5), "depth": 0.7, "shore": 0.9, "stones": 26, "lilies": 9},
        "trees": (('BROADLEAF', 3, (0.6, 0.9)), ('BUSH', 4, (0.7, 1.0))),
        "flowers": (('DAISY', 320), ('POPPY', 160), ('BLUEBELL', 200), ('BUTTERCUP', 240)),
        "grass": {"density": 900.0, "length": 0.14, "color": (0.10, 0.30, 0.04)},
        "treeline": (('BROADLEAF', 60), ('CONIFER', 30)),
    },
    'WILDFLOWER_FIELD': {
        "label": "Wildflower Field", "scale": 'PLACE',
        "description": "Open grass thick with flowers and a single tree",
        "size": 40.0, "relief": 0.7, "ground": 'MEADOW', "skies": ('GOLDEN', 'FAIR', 'HAZE'),
        "trees": (('BROADLEAF', 1, (0.8, 1.0)), ('BUSH', 3, (0.7, 1.0))),
        "flowers": (('POPPY', 500), ('DAISY', 500), ('LAVENDER', 400), ('BUTTERCUP', 400), ('BLUEBELL', 200)),
        "grass": {"density": 900.0, "length": 0.18, "color": (0.12, 0.30, 0.05)},
        "treeline": (('BROADLEAF', 50), ('CONIFER', 20)),
    },
    'FOREST_CLEARING': {
        "label": "Forest Clearing", "scale": 'PLACE',
        "description": "A glade among firs and oaks, with a small dark pool and bluebells",
        "size": 36.0, "relief": 0.9, "ground": 'FOREST', "skies": ('HAZE', 'FAIR', 'OVERCAST'),
        "pond": {"radii": (2.2, 1.6), "depth": 0.5, "shore": 0.7, "stones": 14, "lilies": 4},
        "trees": (('CONIFER', 14, (0.6, 1.1)), ('BROADLEAF', 8, (0.6, 1.0)), ('BUSH', 10, (0.6, 1.1))),
        "rocks": 12, "flowers": (('BLUEBELL', 450), ('DAISY', 80)),
        "grass": {"density": 500.0, "length": 0.10, "color": (0.07, 0.20, 0.04)},
        "treeline": (('CONIFER', 110), ('BROADLEAF', 60)), "mist": 0.35,
    },
    'DESERT_OASIS': {
        "label": "Desert Oasis", "scale": 'PLACE',
        "description": "A pool among palms in bare sand",
        "size": 40.0, "relief": 1.2, "ground": 'SAND', "skies": ('NOON', 'GOLDEN', 'SUNSET'),
        "pond": {"radii": (4.2, 3.0), "depth": 0.8, "shore": 1.2, "stones": 10, "lilies": 0},
        "trees": (('PALM', 7, (0.7, 1.1)), ('BUSH', 5, (0.5, 0.9))),
        "rocks": 10, "treeline": (('PALM', 8), ('DEAD', 6)),
    },
    'WINTER_CLEARING': {
        "label": "Winter Clearing", "scale": 'PLACE',
        "description": "Snow among firs and bare trees, with a frozen-blue pool",
        "size": 36.0, "relief": 0.8, "ground": 'SNOW', "skies": ('OVERCAST', 'HAZE', 'FAIR'),
        "pond": {"radii": (2.6, 1.9), "depth": 0.4, "shore": 0.8, "stones": 12, "lilies": 0},
        "trees": (('CONIFER', 12, (0.6, 1.1)), ('DEAD', 4, (0.7, 1.0))),
        "rocks": 8, "treeline": (('CONIFER', 120), ('DEAD', 20)), "mist": 0.35,
    },
}

ENVIRONMENT_ITEMS = tuple((key, spec["label"], spec["description"]) for key, spec in ENVIRONMENTS.items())
TIME_ITEMS = (('AUTO', "Suit the Environment", "Chosen by the seed from skies that suit it"),) + sky.SKY_ITEMS


def list_environments():
    """What can be built: ``[{key, label, scale, description, skies}]``."""
    return [
        {"key": key, "label": spec["label"], "scale": spec["scale"].lower(),
         "description": spec["description"], "skies": list(spec["skies"])}
        for key, spec in ENVIRONMENTS.items()
    ]


def build(context, environment, seed=1, time='AUTO', plants=1.0, size=0.0):
    """
    Build an environment by key. ``time`` is a sky preset or 'AUTO'; ``plants`` scales
    how much grows; ``size`` 0 keeps the environment's own size. Returns a report.
    """
    if environment not in ENVIRONMENTS:
        raise ValueError("unknown environment {!r}; known: {:s}".format(environment, ", ".join(ENVIRONMENTS)))
    if time != 'AUTO' and time not in sky.SKY_PRESETS:
        raise ValueError("unknown time {!r}; known: AUTO, {:s}".format(time, ", ".join(sky.SKY_PRESETS)))
    spec = ENVIRONMENTS[environment]
    if spec["scale"] == 'VISTA':
        # The seed picks among the skies that suit this environment, not the landscape's wider set.
        chosen = time if time != 'AUTO' else spec["skies"][seed % len(spec["skies"])]
        report = instant.build(
            context, spec["landscape"], seed, spec["ground"], chosen, size or 300.0,
            use_water=spec.get("water", True), plant_factor=plants,
        )
        report["sky"] = chosen
    else:
        if size:
            spec = dict(spec, size=size)
        report = place.build(context, spec, seed, None if time == 'AUTO' else time, plants)
    report.update(environment=environment, scale=spec["scale"].lower(), seed=seed)
    return report


class ELYAN_OT_world(Operator):
    """Build a whole environment of the chosen kind"""
    bl_idname = "elyan_scenery.world"
    bl_label = "World"
    bl_options = {'REGISTER', 'UNDO'}

    environment: EnumProperty(name="Environment", items=ENVIRONMENT_ITEMS)
    seed: IntProperty(name="Seed", description="Same seed, same world", default=1, min=0)
    time: EnumProperty(name="Sky", items=TIME_ITEMS)
    plants: FloatProperty(
        name="Plants", description="How thickly trees, bushes and flowers grow",
        default=1.0, min=0.0, soft_max=2.0, max=5.0,
    )
    size: FloatProperty(
        name="Size", description="Width of the ground, 0 keeps the environment's own", default=0.0, min=0.0,
        unit='LENGTH',
    )

    def execute(self, context):
        report = build(context, self.environment, self.seed, self.time, self.plants, self.size)
        self.report({'INFO'}, "{:s}, seed {:d}".format(ENVIRONMENTS[report["environment"]]["label"], self.seed))
        return {'FINISHED'}


classes = (
    ELYAN_OT_world,
)
