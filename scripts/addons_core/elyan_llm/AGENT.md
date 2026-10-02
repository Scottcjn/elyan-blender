# Working in Blender through the bridge: a quick-start for agents

`eb` below is `python3 <this folder>/client.py`. Run it bare for a one-screen card.

## The first two minutes

```sh
eb sessions                      # is a Blender already serving?
eb launch /path/file.blend       # if not: start a headless one; prints "serving, pid N"
eb --pid N state                 # the file, window or headless, selection, checkpoints, recent requests
eb --pid N help                  # one line per command;  eb --pid N help COMMAND  for the details
```

`state` says `"window": true` when a person has that Blender open. Then it is their scene:
ask before changing it, and never work while they are in Edit Mode or sculpting.

## The loop that works

1. **Change a number, not a mesh.** Corrections such as "neckline higher" belong in the project's
   contract file: `eb contract build/contract.py` lists its constants,
   `eb contract build/contract.py NECKLINE_FRONT_Z 1.44 --note "neckline higher"` changes one,
   keeps a copy of the old file and adds a line to the contract's changelog.
2. **Rebuild.** `eb --pid N rebuild build/run.py --diff` runs the builder in the open session and
   re-imports the contract every time. Use `rebuild` for builders, not `exec`: a running Blender
   remembers imports, and `exec` would keep using the old numbers.
3. **Read numbers before pictures.** `eb --pid N check NAMES... --max-tris 16000`, with `--seam` and
   `--clearance` for seams and garment-to-body distance; `eb --pid N validate --profile quest` for
   delivery limits. `check` exits 2 when something fails.
4. **Compare with the reference as numbers.** `eb --pid N compare names='["Skirt1850"]'
   reference=refs/ref.jpg overlay=out/ov.png b_out=out/ref_mask.png` gives silhouette overlap, the
   half-width at each of 17 heights for both, and the worst row, with no rendering. Always open
   `b_out`: if the photo's background was not removed cleanly the numbers mean nothing. A photo
   comparison is approximate (perspective, pose) and never passes a build on its own.
   `eb --pid N help compare` has the options (crop, band, views file).
5. **Then look.** `eb --pid N contact_sheet out.png NAMES...` gives labelled views in one image. Open
   it and look. Numbers passing has not been the same as it looking right: an arm erased by mesh
   reduction, a face stuck mid-blink and a pond drawn as a square sheet all passed every number.

## When a person has the file open

- **Marks.** The artist can point at a spot and type a note ("Mark This" in the Elyan tab).
  `eb --pid N marks` lists them with the object, world position, selected vertices or bones and the
  nearest bone; `eb --pid N marks remove=ID` once dealt with. `state` shows how many are waiting.
- **Proposals.** In a window session make changes with
  `eb --pid N propose collection=Dress1850 note="neckline 8 mm higher" script=build/run.py`.
  The artist sees the note and what changed, and presses Keep or Undo. There is no command to
  approve: that button is theirs. `pending` shows what is waiting; `discard` withdraws your own.
- **First time on a machine:** ask them to press "Test LLM Bridge" once. `state` then reports the
  result under `window_selftest`. Until it has passed, treat undo and viewport capture as unproven.

## Safety

- `eb --pid N checkpoint NAME COLLECTION` before a risky step; `rollback NAME` puts it back.
  A checkpoint lives in the session only. Keep saving timestamped `.blend` backups (`eb backup`).
- A request that times out keeps running and returns a job id. Fetch it with `result JOB`;
  do not send the work again.
- `exec -c CODE` is for looking and probing. Names persist between calls and are cleared by undo,
  redo and file load (`generation` in the reply changes).

## Other toolkits reachable from `exec` or a builder script

- People: `import elyan_people; elyan_people.make(recipe, path, profile, clips)`; `list_assets()`.
- Worlds: `from elyan_scenery import world; world.list_environments(); world.build(bpy.context, KEY, seed)`.
- Rename: `bpy.context.window_manager.elyan_rename` + `bpy.ops.elyan_rename.apply()`.
- Rest pose with meshes: `bpy.ops.elyan_restpose.apply()` in Pose Mode.
- Classic pose library: `ob.pose_library`, `bpy.ops.poselib.pose_add(name=...)`, `apply_pose(pose_index=...)`.

## Say what you checked

Report what was run and what was looked at, and what was not. Rendering on a machine without a
safe GPU means Cycles on CPU, small and few samples.
