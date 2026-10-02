// SPDX-FileCopyrightText: 2026 Elyan Labs
//
// SPDX-License-Identifier: GPL-2.0-or-later

/**
 * The three.js adapter and ElyanCharacter against real exported characters.
 *
 * Neither three.js nor the sample characters are in the repository, so every
 * test here skips (it does not fail) when it cannot find what it needs:
 *
 *   ELYAN_THREE_DIR    directory that has node_modules/three (made with
 *                      `npm install three`). Default: work/runtime next to the
 *                      blender checkout, ../../../../../work/runtime from here.
 *   ELYAN_SAMPLES_DIR  directory with ada_web.glb, tom_web.glb, ada_quest.glb.
 *                      Default: people-tests next to the blender checkout.
 *
 * The sentence test also needs python3 and espeak-ng, for ../speech.py.
 *
 * Node has no image decoder and no WebGL. `loadGlb` below therefore registers
 * a GLTFLoader plugin whose `loadTexture` hands back an empty texture, which
 * stops the loader before it reaches for `Image`, `createImageBitmap` or
 * object URLs. Geometry, skins, morph targets and animations load as in a
 * browser; materials come out untextured and nothing is ever rendered.
 */

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { ElyanCharacter } from './elyan_character.js';
import { VISEMES, VRCHAT_NAMES, threeAdapter } from './elyan_face.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const checkout = path.resolve(here, '../../../../..');
const threeDir = process.env.ELYAN_THREE_DIR || path.join(checkout, 'work/runtime');
const samplesDir = process.env.ELYAN_SAMPLES_DIR || path.join(checkout, 'people-tests');

let THREE = null;
let GLTFLoader = null;
try {
  // By file path, so the loader and this file share one copy of three.js (its ES module build).
  const base = path.join(threeDir, 'node_modules/three');
  THREE = await import(pathToFileURL(path.join(base, 'build/three.module.js')).href);
  ({ GLTFLoader } = await import(pathToFileURL(path.join(base, 'examples/jsm/loaders/GLTFLoader.js')).href));
} catch {
  THREE = null;
}

const DEG = 180 / Math.PI;

/** Skip reason for a test that needs these sample files, or false when everything is there. */
function unless(...files) {
  if (!THREE) return `three.js not found under ${threeDir} (set ELYAN_THREE_DIR)`;
  const absent = files.filter((file) => !fs.existsSync(path.join(samplesDir, file)));
  return absent.length ? `${absent.join(', ')} not found in ${samplesDir} (set ELYAN_SAMPLES_DIR)` : false;
}

async function loadGlb(file) {
  const loader = new GLTFLoader();
  loader.register(() => ({ name: 'ELYAN_node_no_textures', loadTexture: () => Promise.resolve(new THREE.Texture()) }));
  const data = fs.readFileSync(path.join(samplesDir, file));
  const buffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  return new Promise((resolve, reject) => loader.parse(buffer, '', resolve, reject));
}

const worldQuat = (node) => {
  node.updateWorldMatrix(true, false);
  return node.getWorldQuaternion(new THREE.Quaternion());
};

/**
 * Tells where a bone points, as { yaw, pitch } in degrees in the frame `frame`
 * (a world rotation; identity for a character standing as exported).
 *
 * "Where it points" is the model's forward axis carried along by the bone from
 * its rest pose, which sidesteps the bones' own leaning axes: at rest every
 * bone reads yaw 0, pitch 0.
 */
function pointer(bone, frame = new THREE.Quaternion()) {
  const carried = frame.clone().invert().multiply(worldQuat(bone)).invert();
  return (current = frame) => {
    const v = new THREE.Vector3(0, 0, 1)
      .applyQuaternion(carried)
      .applyQuaternion(worldQuat(bone))
      .applyQuaternion(current.clone().invert());
    return { yaw: Math.atan2(v.x, v.z) * DEG, pitch: Math.asin(v.y) * DEG, vector: v };
  };
}

/**
 * Angle between two rotations in radians. Quaternion.angleTo goes through acos, which
 * cannot tell angles below about 1e-7 apart; this goes through asin and can.
 */
function angleBetween(a, b) {
  const d = a.clone().normalize().invert().multiply(b.clone().normalize());
  return 2 * Math.asin(Math.min(1, Math.hypot(d.x, d.y, d.z)));
}

const near = (actual, expected, tolerance, what) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${what}: ${actual.toFixed(3)}, expected ${expected.toFixed(3)} +- ${tolerance}`);
};

function mouthTrack() {
  const frames = [];
  for (let i = 0; i <= 30; i++) {
    const aa = Math.max(0, 1 - Math.abs(i - 15) / 5);
    const row = VISEMES.map(() => 0);
    row[0] = 1 - aa;
    row[VISEMES.indexOf('viseme_aa')] = aa;
    frames.push(row);
  }
  return { schema: 'elyan.visemes/1', fps: 30, duration: 1, names: VISEMES, frames };
}

const quiet = { blink: false, saccades: false };

test('sample characters have the meshes, bones and clips the runtime assumes', { skip: unless('ada_web.glb', 'tom_web.glb', 'ada_quest.glb') }, async () => {
  const expected = {
    'ada_web.glb': { clips: {}, mouth: 'viseme_aa', morphs: 67 },
    'tom_web.glb': { clips: { idle: 4, talk: 4 }, mouth: 'viseme_aa', morphs: 67 },
    'ada_quest.glb': { clips: { idle: 4, listen: 4, talk: 4, nod: 1.2, shake: 1.4 }, mouth: 'vrc.v_aa', morphs: 38 },
  };
  for (const [file, want] of Object.entries(expected)) {
    const gltf = await loadGlb(file);
    const meshes = [];
    gltf.scene.traverse((node) => { if (node.isMesh) meshes.push(node); });
    assert.equal(meshes.length, 1, `${file} mesh count`);
    assert.ok(meshes[0].isSkinnedMesh);
    assert.equal(Object.keys(meshes[0].morphTargetDictionary).length, want.morphs, `${file} morph count`);
    assert.ok(want.mouth in meshes[0].morphTargetDictionary);
    assert.ok('eyeBlinkLeft' in meshes[0].morphTargetDictionary);

    const head = gltf.scene.getObjectByName('head');
    assert.equal(head.parent.name, 'neck_01');
    for (const name of ['eye_l', 'eye_r']) {
      const eye = gltf.scene.getObjectByName(name);
      assert.ok(eye.isBone);
      assert.equal(eye.parent, head);
      // Rest orientation: local +Y is the line of sight (+Z), local +Z is up, local +X is the model's -X.
      const q = worldQuat(eye);
      const axis = (x, y, z) => new THREE.Vector3(x, y, z).applyQuaternion(q);
      assert.ok(axis(0, 1, 0).distanceTo(new THREE.Vector3(0, 0, 1)) < 1e-3, `${file} ${name} looks along +Z`);
      assert.ok(axis(0, 0, 1).distanceTo(new THREE.Vector3(0, 1, 0)) < 1e-3);
      assert.ok(axis(1, 0, 0).distanceTo(new THREE.Vector3(-1, 0, 0)) < 1e-3);
    }
    // The character's left eye is on +X, and both are in front of the head bone.
    const at = (name) => new THREE.Vector3().setFromMatrixPosition(gltf.scene.getObjectByName(name).matrixWorld);
    assert.ok(at('eye_l').x > 0.02 && at('eye_r').x < -0.02);
    assert.ok(at('eye_l').z > at('head').z);

    const clips = Object.fromEntries(gltf.animations.map((clip) => [clip.name, clip.duration]));
    assert.deepEqual(Object.keys(clips), Object.keys(want.clips), `${file} clips`);
    for (const [name, duration] of Object.entries(want.clips)) near(clips[name], duration, 1e-3, `${file} ${name} duration`);
    // Why layering is needed at all: the clips key the very bones the face turns.
    for (const clip of gltf.animations) {
      const keyed = new Set(clip.tracks.map((track) => track.name));
      for (const name of ['head', 'eye_l', 'eye_r']) assert.ok(keyed.has(`${name}.quaternion`), `${clip.name} keys ${name}`);
    }
  }
});

for (const file of ['ada_web.glb', 'tom_web.glb', 'ada_quest.glb']) {
  test(`adapter turns head and eyes the right way by the right angle (${file})`, { skip: unless(file) }, async () => {
    const { scene } = await loadGlb(file);
    const adapter = threeAdapter(THREE, scene);
    const head = pointer(adapter.bones.head);
    const eyes = { l: pointer(adapter.bones.eye_l), r: pointer(adapter.bones.eye_r) };
    near(head().yaw, 0, 1e-3, 'head at rest');

    adapter.setHead(0.5, 0);
    assert.ok(head().vector.x > 0.4, 'positive yaw turns the head to the character\'s left, +X');
    near(head().yaw, 0.5 * DEG, 0.05, 'head yaw');
    near(head().pitch, 0, 0.05, 'head yaw leaves pitch alone');
    near(eyes.l().yaw, 0.5 * DEG, 0.05, 'the eyes ride on the head');

    adapter.setHead(0, 0.3);
    assert.ok(head().vector.y > 0.25, 'positive pitch lifts the head');
    near(head().pitch, 0.3 * DEG, 0.05, 'head pitch');
    near(head().yaw, 0, 0.05, 'head pitch leaves yaw alone');

    adapter.setHead(-0.4, -0.2);
    near(head().yaw, -0.4 * DEG, 0.05, 'head yaw right');
    near(head().pitch, -0.2 * DEG, 0.05, 'head pitch down');

    adapter.setHead(0, 0);
    for (const which of ['l', 'r']) {
      adapter.setEye(which, 0.3, 0);
      assert.ok(eyes[which]().vector.x > 0.25, `eye_${which} looks left`);
      near(eyes[which]().yaw, 0.3 * DEG, 0.05, `eye_${which} yaw`);
      near(eyes[which]().pitch, 0, 0.05, `eye_${which} yaw leaves pitch alone`);
      adapter.setEye(which, -0.1, 0.2);
      assert.ok(eyes[which]().vector.y > 0.15, `eye_${which} looks up`);
      near(eyes[which]().yaw, -0.1 * DEG, 0.05, `eye_${which} yaw right`);
      near(eyes[which]().pitch, 0.2 * DEG, 0.05, `eye_${which} pitch`);
    }
    near(head().yaw, 0, 1e-3, 'turning the eyes does not turn the head');

    // Head and eyes together: the angles add, which is what FacePlayer counts on.
    adapter.setHead(0.56, 0.18);
    adapter.setEye('l', 0.24, 0.12);
    near(eyes.l().yaw, 0.8 * DEG, 0.05, 'gaze azimuth is head yaw + eye yaw');
    near(eyes.l().pitch, 0.3 * DEG, 0.05, 'gaze elevation is head pitch + eye pitch');

    // The same call twice is the same pose, and release gives the rest pose back.
    const once = adapter.bones.head.quaternion.clone();
    adapter.setHead(0.56, 0.18);
    assert.ok(angleBetween(adapter.bones.head.quaternion, once) < 1e-12, 'turns do not pile up');
    adapter.release();
    near(head().yaw, 0, 1e-3, 'head released');
    near(eyes.l().yaw, 0, 1e-3, 'eye released');
    near(eyes.l().pitch, 0, 1e-3, 'eye released');
  });
}

test('adapter angles are relative to the character, wherever it stands and faces', { skip: unless('ada_web.glb') }, async () => {
  const { scene } = await loadGlb('ada_web.glb');
  const stage = new THREE.Group();
  stage.add(scene);
  scene.position.set(3, 0, -2);
  scene.rotation.set(0.2, 2.1, -0.1);
  const facing = worldQuat(scene);
  const adapter = threeAdapter(THREE, scene);
  const head = pointer(adapter.bones.head, facing);
  const eye = pointer(adapter.bones.eye_r, facing);
  adapter.setHead(0.5, -0.2);
  adapter.setEye('r', 0.2, 0.3);
  near(head().yaw, 0.5 * DEG, 0.05, 'head yaw');
  near(head().pitch, -0.2 * DEG, 0.05, 'head pitch');
  near(eye().yaw, 0.7 * DEG, 0.05, 'gaze yaw');
  near(eye().pitch, 0.1 * DEG, 0.05, 'gaze pitch');
  // Turned around afterwards, as an application would: still relative to the character.
  scene.rotation.y = -1;
  const turned = worldQuat(scene);
  near(head(turned).yaw, 0.5 * DEG, 0.05, 'head yaw after the character turned');
});

test('adapter finds Quest/VRChat mouth shapes under their own names', { skip: unless('ada_quest.glb', 'ada_web.glb') }, async () => {
  const quest = await loadGlb('ada_quest.glb');
  const adapter = threeAdapter(THREE, quest.scene);
  const [mesh] = adapter.meshes;
  for (const [index, name] of VISEMES.entries()) {
    assert.equal(mesh.morphTargetDictionary[name], undefined, `${name} is not in the Quest file`);
    assert.ok(adapter.hasMorph(name), `${name} resolves`);
    mesh.morphTargetInfluences.fill(0);
    adapter.setMorph(name, 0.25 + index / 100);
    const slot = mesh.morphTargetDictionary[VRCHAT_NAMES[name]];
    assert.equal(mesh.morphTargetInfluences[slot], 0.25 + index / 100);
    assert.equal(mesh.morphTargetInfluences.filter((value) => value !== 0).length, 1, `${name} drives one shape only`);
  }
  // Expression shapes keep their names in both profiles.
  adapter.setMorph('eyeBlinkLeft', 1);
  assert.equal(mesh.morphTargetInfluences[mesh.morphTargetDictionary.eyeBlinkLeft], 1);
  // The compact Quest set has no tongue; asking for it is harmless and is reported.
  adapter.setMorph('tongueOut', 1);
  assert.deepEqual([...adapter.missing], ['tongueOut']);

  // The web file is driven by the same calls, and an explicit map wins over nothing.
  const web = await loadGlb('ada_web.glb');
  const mapped = threeAdapter(THREE, web.scene, { morphNames: { smile: ['mouthSmileLeft_missing', 'mouthSmileLeft'] } });
  const face = mapped.meshes[0];
  mapped.setMorph('viseme_aa', 0.5);
  mapped.setMorph('smile', 0.75);
  mapped.setMorph('tongueOut', 0.5);
  assert.equal(face.morphTargetInfluences[face.morphTargetDictionary.viseme_aa], 0.5);
  assert.equal(face.morphTargetInfluences[face.morphTargetDictionary.mouthSmileLeft], 0.75);
  assert.equal(face.morphTargetInfluences[face.morphTargetDictionary.tongueOut], 0.5);
  assert.equal(mapped.missing.size, 0);
});

test('adapter sets a shape on every mesh that has it', { skip: unless('ada_quest.glb') }, async () => {
  const { scene } = await loadGlb('ada_quest.glb');
  // A second, unskinned mesh with two of the shapes under web names and one of its own, as
  // separately exported lashes or a beard would be.
  const extra = new THREE.Mesh(new THREE.BufferGeometry());
  extra.morphTargetDictionary = { viseme_aa: 0, eyeBlinkLeft: 1, beardOnly: 2 };
  extra.morphTargetInfluences = [0, 0, 0];
  scene.getObjectByName('head').add(extra);
  const adapter = threeAdapter(THREE, scene);
  assert.equal(adapter.meshes.length, 2);
  const body = adapter.meshes.find((mesh) => mesh.isSkinnedMesh);
  adapter.setMorph('viseme_aa', 0.6);
  adapter.setMorph('eyeBlinkLeft', 0.9);
  adapter.setMorph('viseme_O', 0.3);
  adapter.setMorph('beardOnly', 0.2);
  assert.deepEqual(extra.morphTargetInfluences, [0.6, 0.9, 0.2]);
  assert.equal(body.morphTargetInfluences[body.morphTargetDictionary['vrc.v_aa']], 0.6);
  assert.equal(body.morphTargetInfluences[body.morphTargetDictionary.eyeBlinkLeft], 0.9);
  assert.equal(body.morphTargetInfluences[body.morphTargetDictionary['vrc.v_oh']], 0.3);
  assert.equal(adapter.missing.size, 0);
});

test('head and eye turns ride on top of a playing clip', { skip: unless('ada_quest.glb') }, async () => {
  // Two copies of one character play the same clip; one also turns its head and eyes.
  const plain = await loadGlb('ada_quest.glb');
  const turned = await loadGlb('ada_quest.glb');
  const names = ['head', 'eye_l', 'neck_01'];
  const bare = Object.fromEntries(names.map((name) => [name, plain.scene.getObjectByName(name)]));
  const adapter = threeAdapter(THREE, turned.scene);
  const neck = turned.scene.getObjectByName('neck_01');
  const bareHead = pointer(bare.head);
  const mixers = [plain, turned].map((gltf) => {
    const mixer = new THREE.AnimationMixer(gltf.scene);
    mixer.clipAction(gltf.animations.find((clip) => clip.name === 'idle')).play();
    return mixer;
  });

  const yaw = 0.4;
  const pitch = 0.15;
  const turn = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw)
    .multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -pitch));
  const neckRest = worldQuat(neck);
  let swing = 0;
  let worst = 0;
  const rest = worldQuat(bare.head);
  for (let frame = 0; frame < 240; frame++) {
    adapter.release();
    for (const mixer of mixers) mixer.update(1 / 60);
    adapter.setHead(yaw, pitch);
    adapter.setEye('l', 0.2, -0.1);
    adapter.setEye('r', 0.2, -0.1);

    // The clip did not touch the neck differently in the two copies.
    assert.ok(angleBetween(worldQuat(neck), worldQuat(bare.neck_01)) < 1e-6, 'bodies agree');
    swing = Math.max(swing, angleBetween(worldQuat(bare.head), rest) * DEG);

    // Expected: the animated head, turned about the model's axes as the animated neck carries them.
    const carried = worldQuat(neck).multiply(neckRest.clone().invert());
    const expected = carried.clone().multiply(turn).multiply(carried.clone().invert()).multiply(worldQuat(bare.head));
    const error = angleBetween(worldQuat(adapter.bones.head), expected) * DEG;
    worst = Math.max(worst, error);
    assert.ok(error < 0.01, `frame ${frame}: head is ${error} degrees from clip pose plus turn`);

    // And in plain terms: against the clip's own head it has moved left and up by the angles asked.
    const animated = bareHead();
    const now = new THREE.Vector3(0, 0, 1)
      .applyQuaternion(rest.clone().invert())
      .applyQuaternion(worldQuat(adapter.bones.head));
    near(Math.atan2(now.x, now.z) * DEG - animated.yaw, yaw * DEG, 1, `frame ${frame}: yaw over the clip`);
    near(Math.asin(now.y) * DEG - animated.pitch, pitch * DEG, 1, `frame ${frame}: pitch over the clip`);

    // The eye is turned over its own animated pose and the head's.
    const eyeTurn = angleBetween(worldQuat(adapter.bones.eye_l), worldQuat(adapter.bones.head).multiply(bare.eye_l.quaternion));
    near(eyeTurn * DEG, Math.acos(Math.cos(0.2) * Math.cos(0.1)) * DEG, 0.5, `frame ${frame}: eye turn over the clip`);
  }
  // The test means nothing unless the clip really moves the head.
  assert.ok(swing > 0.5, `idle moves the head by only ${swing} degrees`);
  assert.ok(worst < 0.01);

  // Without release() the mixer does not rewrite bones whose keys hold still (the eyes),
  // and the adapter must not stack turn on turn.
  const still = angleBetween(worldQuat(adapter.bones.eye_l), worldQuat(adapter.bones.head).multiply(bare.eye_l.quaternion)) * DEG;
  for (let frame = 0; frame < 30; frame++) {
    mixers[1].update(0);
    adapter.setHead(yaw, pitch);
    adapter.setEye('l', 0.2, -0.1);
  }
  const after = angleBetween(worldQuat(adapter.bones.eye_l), worldQuat(adapter.bones.head).multiply(bare.eye_l.quaternion)) * DEG;
  near(after, still, 1e-3, 'eye turn after 30 frames without release');
});

test('ElyanCharacter cross-fades clips and returns from a one-shot', { skip: unless('ada_quest.glb') }, async () => {
  const character = new ElyanCharacter(THREE, await loadGlb('ada_quest.glb'), { face: quiet });
  assert.deepEqual(Object.keys(character.clips), ['idle', 'listen', 'talk', 'nod', 'shake']);
  assert.equal(character.playClip('missing'), false);
  assert.equal(character.playClip('idle', 0), true);
  character.update(0.5);
  const weight = (name) => character.mixer.clipAction(character.clips[name]).getEffectiveWeight();
  assert.equal(weight('idle'), 1);

  character.playClip('listen', 0.4);
  character.update(0.2);
  near(weight('idle'), 0.5, 0.01, 'idle half faded out');
  near(weight('listen'), 0.5, 0.01, 'listen half faded in');
  character.update(0.3);
  assert.equal(weight('idle'), 0);
  assert.equal(weight('listen'), 1);

  // A nod plays once (1.2 s) and hands back to what was looping.
  const head = pointer(character.adapter.bones.head);
  character.playClip('nod', 0.2, { once: true });
  assert.equal(character.current, 'nod');
  let dip = 0;
  for (let i = 0; i < 60; i++) {
    character.update(1 / 60);
    dip = Math.min(dip, head().pitch);
  }
  assert.ok(dip < -3, `the nod lowers the head, lowest ${dip} degrees`);
  for (let i = 0; i < 60; i++) character.update(1 / 60);
  assert.equal(character.current, 'listen');
  near(weight('listen'), 1, 1e-6, 'listen is back');
  near(weight('nod'), 0, 1e-6, 'nod is gone');
  character.dispose();
});

test('lookAtWorld aims the eyes at a point in the world', { skip: unless('ada_web.glb') }, async () => {
  const character = new ElyanCharacter(THREE, await loadGlb('ada_web.glb'), { face: quiet });
  const stage = new THREE.Group();
  stage.add(character.root);
  const { eye_l: left, eye_r: right, head } = character.adapter.bones;
  const middle = () => {
    character.root.updateWorldMatrix(true, true);
    return new THREE.Vector3().setFromMatrixPosition(left.matrixWorld)
      .add(new THREE.Vector3().setFromMatrixPosition(right.matrixWorld)).multiplyScalar(0.5);
  };
  const sight = (eye) => new THREE.Vector3(0, 1, 0).applyQuaternion(worldQuat(eye));
  const miss = (target) => sight(left).angleTo(target.clone().sub(middle())) * DEG;
  const settle = () => { for (let i = 0; i < 240; i++) character.update(1 / 60); };

  // Straight ahead at eye height: nothing to do.
  const eyeLevel = middle();
  const ahead = character.lookAtWorld(new THREE.Vector3(0, eyeLevel.y, 5));
  near(ahead.yaw, 0, 1e-6, 'yaw to a point straight ahead');
  near(ahead.pitch * DEG, 0, 0.5, 'pitch to a point straight ahead');

  // To the character's left (+X) and above, by known angles.
  const leftUp = character.lookAtWorld(new THREE.Vector3(eyeLevel.x + 3, eyeLevel.y + Math.tan(0.3) * Math.hypot(3, 3), eyeLevel.z + 3));
  near(leftUp.yaw * DEG, 45, 0.01, 'yaw to a point front-left');
  near(leftUp.pitch * DEG, 0.3 * DEG, 0.01, 'pitch to a point above');
  assert.ok(character.lookAtWorld(new THREE.Vector3(-3, 0.2, 3)).yaw < 0, 'a point on the right has negative yaw');
  assert.ok(character.lookAtWorld(new THREE.Vector3(-3, 0.2, 3)).pitch < 0, 'a point near the floor has negative pitch');

  // Once the head has settled the line of sight passes through the point.
  for (const point of [[1.5, 1.9, 2.5], [-2, 1.2, 3], [0.4, 0.9, 1.2], [-0.2, 2.2, 4]]) {
    const target = new THREE.Vector3(...point);
    character.lookAtWorld(target);
    settle();
    near(miss(target), 0, 0.2, `line of sight to ${point}`);
    near(sight(left).angleTo(sight(right)) * DEG, 0, 1e-3, 'both eyes are parallel');
  }
  // The head has taken its share.
  assert.ok(angleBetween(worldQuat(head), new THREE.Quaternion(-0.0258, 0, 0, 0.9997)) * DEG > 1, 'the head turned too');

  // The character moved and turned in the world, as it will be in an application.
  character.root.position.set(4, 0.5, -3);
  character.root.rotation.set(0, 2.4, 0);
  const moved = new THREE.Vector3(4 + 2 * Math.sin(2.4 + 0.5), 2.2, -3 + 2 * Math.cos(2.4 + 0.5));
  const aim = character.lookAtWorld(moved);
  near(aim.yaw, 0.5, 0.03, 'yaw is measured from where the character faces');
  settle();
  near(miss(moved), 0, 0.2, 'line of sight after the character moved');

  // A tracked object is followed without another call.
  moved.set(4 + 2 * Math.sin(2.4 - 0.6), 1.3, -3 + 2 * Math.cos(2.4 - 0.6));
  settle();
  near(miss(moved), 0, 0.2, 'line of sight follows the moving target');

  // Behind the character is out of reach: it turns as far as head and eyes go, toward that side.
  const behind = new THREE.Vector3(4 + 2 * Math.sin(2.4 + 3), 2, -3 + 2 * Math.cos(2.4 + 3));
  character.lookAtWorld(behind);
  settle();
  assert.ok(miss(behind) > 60, 'does not twist its neck off');
  const frame = worldQuat(character.root);
  const gaze = sight(left).applyQuaternion(frame.invert());
  near(Math.atan2(gaze.x, gaze.z), 1.45, 0.02, 'stops at head limit plus eye limit');

  character.lookAtWorld(null);
  settle();
  near(Math.atan2(sight(left).applyQuaternion(worldQuat(character.root).invert()).x, 1) * DEG, 0, 0.1, 'back to straight ahead');
});

test('lookAtWorld holds its target while a clip moves the body', { skip: unless('tom_web.glb') }, async () => {
  const character = new ElyanCharacter(THREE, await loadGlb('tom_web.glb'), { face: quiet });
  character.playClip('idle', 0);
  const { eye_l: left, eye_r: right } = character.adapter.bones;
  const target = new THREE.Vector3(1.2, 1.5, 2.5);
  character.lookAtWorld(target);
  for (let i = 0; i < 180; i++) character.update(1 / 60);
  let worst = 0;
  for (let i = 0; i < 240; i++) {
    character.update(1 / 60);
    const middle = new THREE.Vector3().setFromMatrixPosition(left.matrixWorld)
      .add(new THREE.Vector3().setFromMatrixPosition(right.matrixWorld)).multiplyScalar(0.5);
    const sight = new THREE.Vector3(0, 1, 0).applyQuaternion(worldQuat(left));
    worst = Math.max(worst, sight.angleTo(target.clone().sub(middle)) * DEG);
  }
  // The clip's own head sway is deliberately left in, so the aim is close, not exact.
  assert.ok(worst < 3, `line of sight strays ${worst.toFixed(2)} degrees from the target during idle`);
});

test('say follows the audio clock, not the frame clock', { skip: unless('ada_web.glb') }, async () => {
  let audio = 10;
  const character = new ElyanCharacter(THREE, await loadGlb('ada_web.glb'), { face: quiet, clock: () => audio });
  const [mesh] = character.adapter.meshes;
  const aa = () => mesh.morphTargetInfluences[mesh.morphTargetDictionary.viseme_aa];
  assert.equal(character.say(mouthTrack(), 10.25), 1);
  // Frames arrive unevenly and the frame clock is wrong by a wide margin; only `audio` counts.
  character.update(0.5);
  assert.equal(aa(), 0, 'sound has not started');
  assert.equal(character.face.speaking, true);
  audio = 10.25 + 0.5;
  character.update(0.001);
  near(aa(), 1, 1e-6, '0.5 s into the sound the mouth is fully open');
  audio = 10.25 + 11.5 / 30;
  character.update(2);
  near(aa(), 0.3, 1e-6, 'the clock may even run backwards (a seek)');
  audio = 10.25 + 1.01;
  character.update(0.016);
  assert.equal(aa(), 0);
  assert.equal(character.face.speaking, false);
});

test('a spoken sentence moves the real mesh and leaves it at rest', { skip: unless('tom_web.glb', 'ada_quest.glb') }, async (t) => {
  let track;
  try {
    const out = execFileSync('python3', [path.join(here, '../speech.py'), 'Hello there, my baby whale.', '--duration', '2.4'], { encoding: 'utf8' });
    track = JSON.parse(out);
  } catch (error) {
    t.skip(`speech.py could not make a track (python3 and espeak-ng are needed): ${String(error.message).split('\n')[0]}`);
    return;
  }
  assert.equal(track.schema, 'elyan.visemes/1');
  assert.equal(track.duration, 2.4);
  assert.equal(track.frames.length, 73);
  assert.deepEqual(track.names, VISEMES);

  for (const file of ['tom_web.glb', 'ada_quest.glb']) {
    let audio = 0;
    const character = new ElyanCharacter(THREE, await loadGlb(file), { face: { seed: 5 }, clock: () => audio });
    const [mesh] = character.adapter.meshes;
    const slot = (name) => mesh.morphTargetDictionary[name] ?? mesh.morphTargetDictionary[VRCHAT_NAMES[name]];
    const read = (name) => mesh.morphTargetInfluences[slot(name)];
    const mouth = VISEMES.slice(1);
    character.playClip('idle', 0);
    character.setEmotion('happy');

    const step = (seconds) => { audio += seconds; return character.update(seconds); };
    for (let i = 0; i < 30; i++) step(1 / 60);
    assert.ok(mouth.every((name) => read(name) === 0), `${file}: mouth at rest before speaking`);
    near(read('mouthSmileLeft'), 0.7, 0.05, `${file}: smiling`);

    const start = audio + 0.1;
    assert.equal(character.say(track, start), 2.4);
    assert.equal(character.current, 'talk', `${file}: talk clip while speaking`);
    const peak = Object.fromEntries(mouth.map((name) => [name, 0]));
    const samples = [];
    let previous = null;
    let changes = 0;
    let lipsClosedAt = null;
    for (let i = 0; i < 60 * 2.6; i++) {
      step(1 / 60);
      const now = mouth.map(read);
      for (const [index, name] of mouth.entries()) peak[name] = Math.max(peak[name], now[index]);
      if (previous && now.some((value, index) => Math.abs(value - previous[index]) > 1e-4)) changes++;
      if (lipsClosedAt === null && read('viseme_PP') > 0.8) lipsClosedAt = audio - start;
      for (const value of mesh.morphTargetInfluences) assert.ok(value >= 0 && value <= 1);
      // What is on the mesh is the track at the audio time, frame for frame.
      const position = (audio - start) * track.fps;
      const index = Math.floor(position);
      if (index >= 0 && index < track.frames.length - 1) {
        const column = track.names.indexOf('viseme_aa');
        const want = track.frames[index][column] + (track.frames[index + 1][column] - track.frames[index][column]) * (position - index);
        near(read('viseme_aa'), want, 1e-6, `${file}: viseme_aa at ${(audio - start).toFixed(3)} s`);
      }
      samples.push(now);
      previous = now;
    }
    const used = mouth.filter((name) => peak[name] > 0.3);
    assert.ok(used.length >= 5, `${file}: only ${used} were used`);
    assert.ok(changes > 100, `${file}: the mouth moved on ${changes} frames`);
    // "my baby": the lips must close for M and B, somewhere in the middle of the sentence.
    assert.ok(lipsClosedAt > 0.6 && lipsClosedAt < 2.0, `${file}: lips closed at ${lipsClosedAt}`);
    assert.ok(peak.viseme_aa > 0.5 || peak.viseme_E > 0.5, `${file}: a vowel opens the mouth`);

    assert.equal(character.face.speaking, false, `${file}: finished`);
    assert.equal(character.current, 'idle', `${file}: back to the idle clip`);
    assert.ok(mouth.every((name) => read(name) === 0), `${file}: mouth back at rest`);
    for (let i = 0; i < 30; i++) step(1 / 60);
    near(read('mouthSmileLeft'), 0.7, 0.05, `${file}: still smiling afterwards`);
    character.setEmotion('neutral');
    character.face.blinkEnabled = false;
    for (let i = 0; i < 180; i++) step(1 / 60);
    assert.ok(mesh.morphTargetInfluences.every((value) => value < 1e-3), `${file}: whole face back at rest`);
    character.dispose();
  }
});
