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
  operators and types), `backup`, `ping`, `quit`.
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

## Building

Large binary assets are stored with Git LFS on Blender's own server. After
cloning, follow the upstream build instructions (`make update`, then `make`).

## License

GPL-2.0-or-later, as Blender. See `COPYING`.
