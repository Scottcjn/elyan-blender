# Elyan People runtime

Plays the faces of characters exported by the Elyan People add-on: lip sync,
blinks, eye darts, gaze with a head that follows, and an emotion layer.
Plain ES modules, no build step.

| File | What it is |
| --- | --- |
| `elyan_face.js` | `FacePlayer` (engine-neutral) and `threeAdapter` (three.js) |
| `elyan_character.js` | `ElyanCharacter`: mixer, face, speech and gaze wired together for three.js |
| `elyan_face.test.mjs` | `FacePlayer` tests, no dependencies |
| `elyan_three.test.mjs` | adapter and `ElyanCharacter` tests against real exported GLBs |

## Use in a three.js / WebXR app

```js
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { ElyanCharacter } from './elyan_character.js';

const listener = new THREE.AudioListener();
camera.add(listener);
const audio = listener.context;

const gltf = await new GLTFLoader().loadAsync('ada_web.glb');
scene.add(gltf.scene);
// Build it before anything animates the skeleton: the rest pose tells it which way she faces.
const ada = new ElyanCharacter(THREE, gltf, { clock: () => audio.currentTime });
ada.playClip('idle');
ada.setEmotion('happy');
ada.lookAtWorld(camera.position);          // the Vector3 is kept and followed every frame

async function speak(wavUrl, trackUrl) {
  const [buffer, track] = await Promise.all([
    fetch(wavUrl).then((r) => r.arrayBuffer()).then((b) => audio.decodeAudioData(b)),
    fetch(trackUrl).then((r) => r.json()),
  ]);
  const source = new AudioBufferSourceNode(audio, { buffer });
  source.connect(audio.destination);
  const startAt = audio.currentTime + 0.05;
  source.start(startAt);
  ada.say(track, startAt);                 // the mouth follows the audio clock from here on
}

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {          // setAnimationLoop also drives WebXR sessions
  ada.update(clock.getDelta());
  renderer.render(scene, camera);
});
```

`playClip(name, fade, { once: true })` plays a nod or a shake once and fades
back to the clip that was looping. `say` fades to the `talk` clip if there is
one and back when the sentence ends. Web and Quest/VRChat exports are driven by
the same code; the adapter finds `vrc.v_*` shapes by itself.

Without `ElyanCharacter`, keep this order every frame. The body clips key
`head`, `eye_l` and `eye_r`; the face adds its turn on top of that pose:

```js
adapter.release();   // take last frame's face turn off the bones
mixer.update(dt);    // body animation
player.update(dt);   // face: morphs, then head and eyes on top of the animated pose
```

## Track format: `elyan.visemes/1`

Made by `../speech.py` (`python3 speech.py "Hello." --duration 0.6`, or from
timed phonemes or Rhubarb cues).

```json
{ "schema": "elyan.visemes/1", "fps": 30, "duration": 2.4,
  "names": ["viseme_sil", "viseme_PP", "..."],
  "frames": [[0.75, 0.0, "..."], "..."] }
```

`frames[i][j]` is the weight (0..1) of shape `names[j]` at time `i / fps`;
there are `ceil(duration * fps) + 1` rows and each row sums to 1 with
`viseme_sil` as the remainder. The player interpolates between rows and does
not drive `viseme_sil`.

## Adapter contract

`FacePlayer` talks to the engine through three calls, once per `update`:

- `setMorph(name, weight)`: weight 0..1. Names are the web profile's
  (`viseme_aa`, `eyeBlinkLeft`, ...). Unknown names must be ignored.
- `setHead(yaw, pitch)`: radians, relative to the rest pose. Positive yaw turns
  to the character's left, positive pitch up. Called before `setEye`.
- `setEye('l' | 'r', yaw, pitch)`: what is left of the gaze after the head's
  share. The line of sight must end up at azimuth `head yaw + eye yaw` and
  elevation `head pitch + eye pitch`.

`threeAdapter(THREE, root, { morphNames, bones })` adds `release()`,
`headFrame(quaternion)`, `hasMorph(name)`, `missing`, `meshes` and `bones`.
It assumes the glTF convention the exporter follows: Y up, the character facing
+Z with its left at +X in the frame of `root`. The bones' own axes are not
used, only their rest pose.

## Tests

```sh
mkdir -p ../../../../../work/runtime && (cd ../../../../../work/runtime && npm init -y && npm install three)
node --test
```

three.js is deliberately not in the repository. `elyan_three.test.mjs` looks
for it in `$ELYAN_THREE_DIR/node_modules/three` (default: `work/runtime` next
to the blender checkout) and for `ada_web.glb`, `tom_web.glb` and
`ada_quest.glb` in `$ELYAN_SAMPLES_DIR` (default: `people-tests` next to the
checkout). Tests that miss either are skipped, not failed. The sentence test
also needs `python3` and `espeak-ng`. Node has no image decoder, so the tests
load GLBs with a GLTFLoader plugin that returns empty textures; see the top of
the test file.

## Verified, and not

Run in Node 22 with three.js 0.186.1 against the three sample files: turn
directions and angles of head and eyes in world space, Quest name mapping,
several meshes sharing a shape, layering on the idle clip, clip cross-fades,
`lookAtWorld`, and a spoken sentence changing `morphTargetInfluences`.

Not verified, because it needs a browser, a GPU or a headset:

- Anything rendered. That the morph targets and the eye skinning look right on
  screen was only checked in Blender, never in WebGL or WebGPU.
- Sound. `say` was tested against a simulated clock, not a real `AudioContext`,
  and audio output latency (`audioContext.outputLatency`) is not compensated.
- WebXR: frame timing in a session, and looking at a headset camera.
- three.js versions other than 0.186.1, and loaders other than GLTFLoader.
- The FBX export, Unity and VRChat. This runtime does not touch them.

Known limits: both eyes get the same angles (no convergence on near targets);
`lookAtWorld` leaves the clip's own head sway in, so with a clip playing the
line of sight wanders around the target by about that sway (measured on
`tom_web.glb`: up to 1.7 degrees during `idle`, 4.4 during `talk`); targets
beyond about 83 degrees to the side or 40 degrees up or down are out of reach
and the gaze stops at the limit.
