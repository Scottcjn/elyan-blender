# Elyan Blender

A fork of [Blender](https://projects.blender.org/blender/blender) by Elyan Labs.
Branch `elyan/main` follows upstream `main` and adds bundled add-ons. Everything
added so far is Python under `scripts/addons_core/`, so it also runs on a stock
Blender 4.2 or newer by copying the add-on folder into your add-ons directory.

Status: early. The fork has not been compiled yet; the add-ons were tested
headless on stock Blender 5.2.2 and 4.3.2.

## Elyan LLM Bridge (`scripts/addons_core/elyan_llm`)

Lets an LLM coding agent on the same machine work inside a running Blender
instead of starting a new process for every step.

- Token-protected HTTP server on localhost only; off until started
  (3D Viewport > Sidebar > Elyan, or `ELYAN_LLM_BRIDGE=1`).
- Commands: `exec` (Python, names persist between calls), `scene`, `object`,
  `render` (camera or viewport), `screenshot`, `api` (looks up this build's
  operators and types), `validate` (budgets and mesh defects per delivery
  profile), `backup`, `ping`, `quit`.
- `client.py` is a standard-library command-line client:

  ```sh
  python3 scripts/addons_core/elyan_llm/client.py launch scene.blend   # headless
  python3 scripts/addons_core/elyan_llm/client.py scene
  python3 scripts/addons_core/elyan_llm/client.py exec -c "len(bpy.data.objects)"
  python3 scripts/addons_core/elyan_llm/client.py render out.png --res 900 1200
  ```

Anything that can read the session file (owner-only, in the user config
directory) can run Python as you. Same-user processes are trusted; nothing
else is.

Not yet tested with a real window: the UI timer path, viewport capture,
screenshot and undo steps.

## Elyan Scenery (`scripts/addons_core/elyan_scenery`)

Landscape generation in the spirit of 1990s scenery programs.
3D Viewport > Sidebar > Scenery, or Add > Scenery.

- **Instant Scenery**: one click builds terrain, water, rocks, sky and a camera
  with a clear view. Same seed, same scene.
- **Terrain**: nine fractal landscapes; rain and scree erosion; smooth, terrace,
  raise, lower, sharpen, invert, edge falloff; paint heights as an image and
  read them back; change resolution without losing shape.
- **Sky Lab**: seven presets (noon to starry night) with sun height, direction,
  haze, clouds and stars.
- **Materials**: terrain coloured by altitude and slope (alpine, desert,
  volcanic, arctic, tropical, moon, alien), water, rock.
- **Organic forms**: boulders and metaball blobs.

Every tool is an operator, so the LLM bridge can drive all of it.

Known gaps: no distance haze, no trees or scattering beyond rocks, no terrain
sculpt brushes (use the height image), and no thumbnails for presets.

## Elyan People (`scripts/addons_core/elyan_people`)

Characters from a recipe, able to talk, exported within budget.
3D Viewport > Sidebar > People.

Needs the separate [MPFB](https://github.com/makehumancommunity/mpfb2) extension
(GPL-3) and MakeHuman's CC0 asset packs: `makehuman_system_assets` (skins, hair,
clothes, eyes, teeth), `visemes02` (15 mouth shapes) and `faceunits01` (52 ARKit
shapes). None of that is in this repository.

- **Recipe**: one JSON object (`elyan.person/1`) with body sliders, skin, hair,
  clothes, rig and face level. Unknown keys and out-of-range values are refused
  before anything is built.
- **Contract**: height, bone landmarks and chest/waist/hip girths are measured
  and stored on the rig, for fitting garments and props.
- **Face**: mouth shapes and expressions are loaded and spread to teeth, tongue,
  lashes and brows; eye bones are added; numeric checks confirm the lips close
  for P/B/M, open for "aa", and that left and right shapes mirror.
- **Motion**: generated idle, listen, talk, nod and shake clips.
- **Export**: one mesh, one atlas material, reduced to the profile's triangle
  budget with the face left intact and its shape keys carried across; GLB and
  optional FBX, a manifest, validation and a read-back check.
- **`speech.py`**: timed phonemes (or text plus clip length, via eSpeak NG, or
  Rhubarb cues) to a per-frame mouth-shape track. Standard library only.
- **`runtime/elyan_face.js`**: engine-independent player for those tracks with
  blinks, eye darts, gaze with head follow and emotion presets; includes a
  three.js adapter. `node --test runtime/elyan_face.test.mjs`.

```python
from elyan_people import build, export
rig, report = build.build({"name": "Ada", "body": {"gender": 0.0}, "hair": "bob01",
                           "clothes": ["female_casualsuit01", "shoes01"]})
export.export(rig, "ada.glb", profile="quest", clips=("idle", "talk"))
```

Measured on stock Blender 5.2.2 with MPFB 2.0.17: a clothed person builds in
about 14 s; the Quest export is 14,843 triangles, 1 material, 38 shape keys,
5 animations, 4.5 MB.

Not yet done: the three.js adapter and the FBX animation path have not been
tried in an engine; VRChat limits in `elyan_llm/validate.py` are from memory
and need checking against the current SDK; hand gestures are minimal; lip-sync
from text alone is approximate (use the speech engine's phoneme times).

## Building

Large binary assets are stored with Git LFS on Blender's own server. After
cloning, follow the upstream build instructions (`make update`, then `make`).

## License

GPL-2.0-or-later, as Blender. See `COPYING`.
