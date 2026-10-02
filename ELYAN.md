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
- Starting out: `state` (file, window or headless, selection, what is waiting),
  `help`, and a one-screen card when the client is run bare. `AGENT.md` in the
  add-on folder is the one-page working loop.
- Numbers, not meshes: `contract` lists the constants of a contract file and
  changes one, keeping comments, copying the old file and adding a changelog
  line. `compare` measures a silhouette against a reference photo or another
  image without rendering: overlap, half-width at each height, worst row.
- With a person at the window: "Mark This" pins a note to a spot (`marks`
  lists them); `propose` makes a change the artist then keeps or undoes from
  the panel (there is no command to approve); "Test LLM Bridge" checks the
  window paths once and `state` reports the result.
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

Not yet run in a real window by its developers: the timer path, viewport
capture, screenshot, undo steps and the panel. "Test LLM Bridge" exists to
find out on the artist's machine.

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

## Elyan Garment (`scripts/addons_core/elyan_garment`)

One-call tools for fitting clothes to an avatar body, each returning numbers
and refusing (mesh untouched) when its own check fails. Functions on named
objects, buttons in the sidebar's Elyan tab, and bridge commands
`garment_weights`, `garment_weld`, `garment_fit`, `garment_shapekeys`.

- **weights**: give a garment the body's skinning from the nearest surface,
  deform bones only, at most four influences, normalised; smooths
  automatically where a skirt would shear between two legs.
- **weld**: move an open edge exactly onto a ring of points or another
  object's edge (within 1e-7 m in tests), easing the rows beside it.
- **fit**: push only what is closer to the body than the ease outward, never
  past a limit and never the pinned groups; reports where the body still
  shows through faces.
- **shapekeys**: make the garment follow the body's sliders, refusing any key
  that makes clearance worse.

105 assertions pass on Blender 5.2.2 and 4.3.2, and it was run on a generated
person. Known weak spots: skirts standing off the body, sleeves near the
torso, tight ease in hard bends, high collars (untested). Not seen in a window.

## Elyan Quick Rename (`scripts/addons_core/elyan_rename`)

Rename selected objects, their data, their materials, bones, vertex groups or
shape keys from one small panel with a live preview: find and replace, add to
the start or end, one name plus a running number, or remove the `.001` Blender
adds to duplicates. 3D Viewport > Sidebar > Elyan > Quick Rename. On by default.

## Pose Library (Classic) (`scripts/addons_core/elyan_poselib`)

The pose list Blender had until 3.5: pose the rig, press "+", name it; pick a
name, press Apply. Properties > Armature > Pose Library, and in Pose Mode the
sidebar's Elyan tab. On by default.

- Same operators as before (`poselib.new`, `pose_add`, `pose_remove`,
  `pose_rename`, `pose_move`, `apply_pose`, `browse_interactive`,
  `action_sanitize`), the same Pose Mode shortcuts (Shift L add, Alt L browse,
  Shift Alt L remove, Ctrl Shift L rename), and `Object.pose_library` again.
- Same storage: an Action with one keyed frame per pose and a pose marker
  naming it. A library made in Blender 3.4 was opened in 5.2.2 and 4.3.2 and
  its poses applied with the right values. Old files do not remember which
  armature a library belonged to: choose the action in the panel once.
- Poses store location, rotation and scale of the selected bones (all bones if
  none are selected) and apply to the selection in the same way. Custom
  properties and bendy-bone settings are not stored.
- Tested headless on both versions, including save and reload. The interactive
  browser and the shortcuts need a window and have not been tried.

## Apply Pose as Rest Pose, with Meshes (`scripts/addons_core/elyan_restpose`)

Blender's "Apply Pose as Rest Pose" changes only the armature, so rigged meshes
jump. This makes the current pose the rest pose and keeps every mesh rigged to
the armature, and every one of its shape keys, exactly as posed. Pose Mode:
Pose > Apply, or the sidebar's Elyan tab. On by default.

Measured on Blender 5.2.2 and 4.3.2 with a two-bone rig (rotated, one bone
scaled), a mesh with a plain and a masked shape key under a subdivision
modifier, and a second mesh without keys: vertex positions before and after
differ by 0, at rest, with each key on and with a key half on. Masks and
modifier visibility are put back. Objects parented directly to bones are not
handled. Not tried in a window.

## Tools that used to ship with Blender

Blender 4.2 moved its bundled add-ons out to an online extensions site. These
are back in the box, off until enabled in Preferences > Add-ons, taken from
the last bundled versions (May 2024, GPL-2.0-or-later):

LoopTools, F2, Bool Tool, Carver, Auto Mirror, BSurfaces, Edit Mesh Tools,
Snap Utilities Line, Tissue, Align Tools, Copy Attributes, Modifier Tools,
3D Navigation, Collection Manager, Edit Linked Library, Bone Selection Sets,
Material Utilities, MeasureIt, Node Arrange, Cell Fracture, Skinify, Curve
Tools, Extra Curve Objects, Extra Mesh Objects, BoltFactory, Sapling Tree Gen
and A.N.T. Landscape.

All 27 enable on Blender 5.2.2. Six operators were run as a spot check
(LoopTools flatten, Sapling, BoltFactory, Bool Tool union and an extra mesh
object finished; A.N.T. Landscape did not complete headless). The rest are
untested beyond loading, and none has been tried in a window. 3D Viewport Pie
Menus did not load and is left out.

## Building

Large binary assets are stored with Git LFS on Blender's own server. After
cloning, follow the upstream build instructions (`make update`, then `make`).

## License

GPL-2.0-or-later, as Blender. See `COPYING`.
