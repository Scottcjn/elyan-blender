# Elyan Blender

A fork of [Blender](https://projects.blender.org/blender/blender) by Elyan Labs.
Branch `elyan/main` follows upstream `main` and adds bundled add-ons. Everything
added so far is Python under `scripts/addons_core/`, so it also runs on a stock
Blender 4.2 or newer by copying the add-on folder into your add-ons directory.

Status: early. The add-ons are tested headless on stock Blender 5.2.2 (and
4.3.2 where they do not depend on MPFB).

## Elyan LLM Bridge (`scripts/addons_core/elyan_llm`)

Lets an LLM coding agent on the same machine work inside a running Blender
instead of starting a new process for every step.

- Token-protected HTTP server on localhost only; off until started
  (3D Viewport > Sidebar > Elyan, or `ELYAN_LLM_BRIDGE=1`).
- Looking: `scene`, `object`, `api` (this build's operators and types), `render`
  (camera or viewport), `screenshot`, `contact_sheet` (labelled multi-view grid).
- Doing: `exec` (Python; names persist between calls; `--diff` reports what changed),
  `rebuild` (runs a builder script in a fresh namespace and re-imports the modules
  beside it, so an edited contract file is never stale), `backup`.
- Checking: `validate` (budgets and defects per delivery profile), `check`
  (triangles after modifiers, non-manifold and loose geometry, seam-ring distance
  at full precision, garment-to-body clearance), `snapshot` / `diff`.
- Safety: `checkpoint` / `rollback` of a collection without saving the file.
- Long work: `submit` / `status` / `result` / `cancel`; a request that times out
  keeps running and returns a job id instead of inviting a blind resend.
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

- **World**: choose a kind of environment and get the whole scene, at one of
  two scales. *Vistas* are land seen from afar: alpine range, rolling hills,
  tropical island, desert dunes, canyonlands, volcano, arctic peaks, moonscape,
  alien world. *Places* are ground seen from standing height: meadow with pond,
  wildflower field, forest clearing, desert oasis, winter clearing, built from
  rolling ground, a pond with rim stones and lilies, grass, drifts of flowers,
  nearby trees and a far treeline. Seed, sky and plant density are options.
  From Python: `from elyan_scenery import world; world.list_environments();
  world.build(bpy.context, "MEADOW_POND", seed=7, time="GOLDEN")`.
- **Instant Scenery**: the vista builder on its own, with landscape, ground and
  sky chosen separately. Same seed, same scene.
- **Terrain**: nine fractal landscapes; rain and scree erosion; smooth, terrace,
  raise, lower, sharpen, invert, edge falloff; paint heights as an image and
  read them back; change resolution without losing shape.
- **Sky Lab**: seven presets (noon to starry night) with sun height, direction,
  haze, clouds and stars.
- **Materials**: terrain coloured by altitude and slope (alpine, desert,
  volcanic, arctic, tropical, moon, alien), water, rock.
- **Organic forms**: boulders and metaball blobs.

Every tool is an operator, so the LLM bridge can drive all of it.

- **Atmosphere**: distance haze coloured from the horizon sky.
- **Plants**: conifer, broadleaf, palm, dead tree and bush, built to a triangle
  budget; rule-based scatter by density, altitude, slope, water and spacing.

Known gaps: grass is render-time strands and does not export; flowers and
trees are simple low-poly shapes; haze is a render effect and does not export; on Blender 4.3 the old
sky model leaves a brown band at the horizon; trees have no UVs; no terrain
sculpt brushes (use the height image); the panels have not been seen in a window.

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
- **Motion**: generated clips: idle, listen, talk, weight shift (loops) and nod,
  shake, think, greet, shrug, point, agree (one-shots), with relaxed hands and
  planted feet.
- **Transfer** (`transfer.py`): gives a head made elsewhere the full expression
  set while keeping the shapes it has, finding landmarks from shapes both heads
  share (a blink, an open mouth, brows).
- **Export**: one mesh, one atlas material, reduced to the profile's triangle
  budget with the face left intact and its shape keys carried across; GLB and
  optional FBX, a manifest, validation and a read-back check.
- **`speech.py`**: timed phonemes (or text plus clip length, via eSpeak NG, or
  Rhubarb cues) to a per-frame mouth-shape track. Standard library only.
- **`runtime/elyan_face.js`**: engine-independent player for those tracks with
  blinks, eye darts, gaze with head follow and emotion presets; includes a
  three.js adapter and an `ElyanCharacter` helper; see `runtime/README.md`.

```python
import elyan_people
manifest = elyan_people.make(
    {"name": "Ada", "body": {"gender": 0.0}, "hair": "bob01",
     "clothes": ["female_casualsuit01", "shoes01"]},
    "ada.glb", "quest", clips=("idle", "talk"))
elyan_people.list_assets()   # installed skins, hair, clothes...
```

Measured on stock Blender 5.2.2 with MPFB 2.0.17: a clothed person builds in
about 14 s; the Quest export is about 14,900 triangles, 1 material, 38 shape
keys, 4.9 MB with eleven clips. The web export keeps 67 shape keys on a
head-only mesh (9.6 MB).

Not yet verified: anything on screen in a browser, headset, Unity or VRChat;
VRChat limits in `elyan_llm/validate.py` are from memory; lip-sync from text
alone is approximate (use the speech engine's phoneme times).

## Building

Large binary assets are stored with Git LFS on Blender's own server. After
cloning, follow the upstream build instructions (`make update`, then `make`).

## License

GPL-2.0-or-later, as Blender. See `COPYING`.
