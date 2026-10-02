// SPDX-FileCopyrightText: 2026 Elyan Labs
//
// SPDX-License-Identifier: GPL-2.0-or-later

/**
 * Runtime face for characters exported by the Elyan People add-on.
 *
 * Plays lip-sync tracks ("elyan.visemes/1", from speech.py), and adds what
 * makes a face look alive between the words: blinks, small darting eye
 * movements, gaze that the head follows, and an emotion layer.
 *
 * It knows nothing about any engine. Give it an adapter:
 *
 *   { setMorph(name, weight), setHead(yaw, pitch), setEye(side, yaw, pitch) }
 *
 * Angles are radians; positive yaw looks to the character's left, positive
 * pitch looks up. Each frame `setHead` is called before `setEye`, and the eye
 * angles are what is left of the gaze after the head's share: the gaze
 * direction has azimuth `head yaw + eye yaw` and elevation
 * `head pitch + eye pitch`. `threeAdapter` below builds one for three.js.
 */

export const VISEMES = [
  'viseme_sil', 'viseme_PP', 'viseme_FF', 'viseme_TH', 'viseme_DD', 'viseme_kk', 'viseme_CH',
  'viseme_SS', 'viseme_nn', 'viseme_RR', 'viseme_aa', 'viseme_E', 'viseme_I', 'viseme_O', 'viseme_U',
];

/** Expression presets as ARKit shape weights. */
export const EMOTIONS = {
  neutral: {},
  happy: { mouthSmileLeft: 0.7, mouthSmileRight: 0.7, cheekSquintLeft: 0.4, cheekSquintRight: 0.4 },
  sad: { mouthFrownLeft: 0.6, mouthFrownRight: 0.6, browInnerUp: 0.7, eyeSquintLeft: 0.2, eyeSquintRight: 0.2 },
  angry: { browDownLeft: 0.8, browDownRight: 0.8, noseSneerLeft: 0.4, noseSneerRight: 0.4, mouthFrownLeft: 0.3, mouthFrownRight: 0.3 },
  surprised: { browInnerUp: 0.8, browOuterUpLeft: 0.7, browOuterUpRight: 0.7, eyeWideLeft: 0.7, eyeWideRight: 0.7 },
  thinking: { browDownLeft: 0.3, browOuterUpRight: 0.5, eyeSquintLeft: 0.3, mouthPucker: 0.2 },
};

const EYE_LIMIT = 0.45;   // about 26 degrees, past this the head must turn
const HEAD_LIMIT = 1.0;
const BLINK_CLOSE = 0.07;
const BLINK_OPEN = 0.13;

/** Small seedable random source, so a character's fidgeting can be reproduced. */
function randomSource(seed) {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
/** Fraction of the way to a target covered in `dt` seconds, for a given settling time. */
const approach = (dt, time) => 1 - Math.exp(-dt / Math.max(time, 1e-4));

export class FacePlayer {
  /**
   * @param adapter see the file comment
   * @param options { seed, blink: true, saccades: true }
   */
  constructor(adapter, options = {}) {
    this.adapter = adapter;
    this.random = randomSource(options.seed ?? 1);
    this.blinkEnabled = options.blink ?? true;
    this.saccadesEnabled = options.saccades ?? true;

    this.track = null;
    this.trackTime = 0;
    this.trackClock = null;
    this.emotion = {};
    this.emotionTarget = {};
    this.gaze = { yaw: 0, pitch: 0 };
    this.head = { yaw: 0, pitch: 0 };
    this.dart = { yaw: 0, pitch: 0 };
    this.nextDart = 1;
    this.nextBlink = 2;
    this.blinkClock = -1;
    this.clock = 0;
    this.speaking = false;
  }

  /**
   * Start a lip-sync track. Returns its duration in seconds.
   *
   * Without `clock` the track starts now and advances with `update(dt)`. Frame
   * time drifts away from audio time within seconds, so when there is sound pass
   * `clock`: a function returning the seconds of audio already played, negative
   * while the sound has not started. The mouth then follows the sound.
   */
  play(track, clock = null) {
    if (track.schema !== 'elyan.visemes/1') throw new Error(`unknown track schema ${track.schema}`);
    this.track = track;
    this.trackTime = 0;
    this.trackClock = clock;
    this.speaking = true;
    return track.duration;
  }

  stop() {
    this.track = null;
    this.trackClock = null;
    this.speaking = false;
  }

  /** Blend toward an expression: a name from EMOTIONS or an object of shape weights. */
  setEmotion(emotion, strength = 1) {
    const preset = typeof emotion === 'string' ? EMOTIONS[emotion] : emotion;
    if (!preset) throw new Error(`unknown emotion ${emotion}`);
    this.emotionTarget = Object.fromEntries(Object.entries(preset).map(([k, v]) => [k, v * strength]));
  }

  /** Look in a direction relative to straight ahead. The eyes go first, the head follows. */
  lookAt(yaw, pitch) {
    this.gaze = { yaw: clamp(yaw, -HEAD_LIMIT - EYE_LIMIT, HEAD_LIMIT + EYE_LIMIT), pitch: clamp(pitch, -0.7, 0.7) };
  }

  /** Blink now, whatever the schedule says. */
  blink() {
    if (this.blinkClock < 0) this.blinkClock = 0;
  }

  _visemes(weights) {
    const track = this.track;
    if (!track) return;
    const position = this.trackTime * track.fps;
    // The sound has been scheduled but is not playing yet: hold the rest face.
    if (position < 0) return;
    const index = Math.floor(position);
    if (index >= track.frames.length - 1) {
      this.stop();
      return;
    }
    const mix = position - index;
    const a = track.frames[index];
    const b = track.frames[index + 1];
    track.names.forEach((name, i) => {
      // Silence is the absence of the others; driving it would fight the emotion layer.
      if (name !== 'viseme_sil') weights[name] = a[i] + (b[i] - a[i]) * mix;
    });
  }

  _blink(dt) {
    if (this.blinkClock < 0) {
      if (this.blinkEnabled && this.clock >= this.nextBlink) this.blinkClock = 0;
      else return 0;
    }
    this.blinkClock += dt;
    const t = this.blinkClock;
    if (t < BLINK_CLOSE) return t / BLINK_CLOSE;
    if (t < BLINK_CLOSE + BLINK_OPEN) return 1 - (t - BLINK_CLOSE) / BLINK_OPEN;
    this.blinkClock = -1;
    // People blink every few seconds, now and then twice in a row.
    this.nextBlink = this.clock + (this.random() < 0.15 ? 0.25 : 2 + this.random() * 4);
    return 0;
  }

  _darts() {
    if (!this.saccadesEnabled || this.clock < this.nextDart) return;
    // Eyes never hold still: small jumps a few times a second while listening, fewer while speaking.
    const reach = 0.035 + this.random() * 0.035;
    const angle = this.random() * Math.PI * 2;
    this.dart = { yaw: Math.cos(angle) * reach, pitch: Math.sin(angle) * reach * 0.5 };
    this.nextDart = this.clock + (this.speaking ? 0.8 : 0.4) + this.random() * 1.6;
  }

  /** Advance by `dt` seconds and push the result to the adapter. Call once per frame. */
  update(dt) {
    this.clock += dt;
    const weights = Object.fromEntries(VISEMES.slice(1).map((name) => [name, 0]));

    if (this.track) {
      this.trackTime = this.trackClock ? this.trackClock() : this.trackTime + dt;
      this._visemes(weights);
    }

    const ease = approach(dt, 0.18);
    for (const name of new Set([...Object.keys(this.emotion), ...Object.keys(this.emotionTarget)])) {
      const current = this.emotion[name] ?? 0;
      const next = current + ((this.emotionTarget[name] ?? 0) - current) * ease;
      if (next < 1e-4 && !(name in this.emotionTarget)) delete this.emotion[name];
      else this.emotion[name] = next;
      weights[name] = (weights[name] ?? 0) + next;
    }
    // A wide-open mouth and a full smile fight over the same lips; speech wins.
    const open = Math.max(weights.viseme_aa, weights.viseme_O, weights.viseme_U, weights.viseme_PP);
    for (const name of ['mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthPucker']) {
      if (name in weights) weights[name] *= 1 - 0.6 * open;
    }

    const blink = this._blink(dt);
    weights.eyeBlinkLeft = Math.max(weights.eyeBlinkLeft ?? 0, blink);
    weights.eyeBlinkRight = Math.max(weights.eyeBlinkRight ?? 0, blink);

    this._darts();
    // The head takes up most of a large turn, slowly; the eyes cover the rest at once.
    const headEase = approach(dt, 0.35);
    this.head.yaw += (clamp(this.gaze.yaw * 0.7, -HEAD_LIMIT, HEAD_LIMIT) - this.head.yaw) * headEase;
    this.head.pitch += (clamp(this.gaze.pitch * 0.6, -0.5, 0.5) - this.head.pitch) * headEase;
    const eyeYaw = clamp(this.gaze.yaw - this.head.yaw + this.dart.yaw, -EYE_LIMIT, EYE_LIMIT);
    const eyePitch = clamp(this.gaze.pitch - this.head.pitch + this.dart.pitch, -0.35, 0.35);
    // Lids follow the eyes: looking down lowers them.
    if (eyePitch < 0) {
      const lower = Math.min(0.4, -eyePitch);
      weights.eyeBlinkLeft = Math.max(weights.eyeBlinkLeft, lower);
      weights.eyeBlinkRight = Math.max(weights.eyeBlinkRight, lower);
    }

    for (const [name, value] of Object.entries(weights)) this.adapter.setMorph(name, clamp(value, 0, 1));
    // Head first: an adapter needs the head's pitch to place the eyes (see the file comment).
    this.adapter.setHead(this.head.yaw, this.head.pitch);
    this.adapter.setEye('l', eyeYaw, eyePitch);
    this.adapter.setEye('r', eyeYaw, eyePitch);
    return weights;
  }
}

/** Names the Quest/VRChat export profile gives the mouth shapes (face.py, VRCHAT_NAMES). */
export const VRCHAT_NAMES = {
  viseme_sil: 'vrc.v_sil', viseme_PP: 'vrc.v_pp', viseme_FF: 'vrc.v_ff', viseme_TH: 'vrc.v_th',
  viseme_DD: 'vrc.v_dd', viseme_kk: 'vrc.v_kk', viseme_CH: 'vrc.v_ch', viseme_SS: 'vrc.v_ss',
  viseme_nn: 'vrc.v_nn', viseme_RR: 'vrc.v_rr', viseme_aa: 'vrc.v_aa', viseme_E: 'vrc.v_e',
  viseme_I: 'vrc.v_ih', viseme_O: 'vrc.v_oh', viseme_U: 'vrc.v_ou',
};

/**
 * Adapter for three.js. `root` is the loaded glTF scene (or the character's
 * node in it). Build it while the skeleton is still in its rest pose, that is
 * before an AnimationMixer has run: the rest pose is where it learns which way
 * the character faces.
 *
 * Frame. glTF is Y-up and the exporter turns the Blender character (facing -Y,
 * left hand at +X) to face +Z with its left still at +X. So in the model's
 * frame a positive yaw is a positive rotation about +Y and a positive pitch a
 * negative rotation about +X. The bones' own axes are no help: an eye bone
 * points along its local +Y with local +X at the model's -X, the head bone
 * points up and leans a few degrees, and its parent the neck leans some twenty.
 * So every turn is made about the model's axes, expressed in the frame of the
 * bone's parent as it stood at rest.
 *
 * Layering. A body clip keys `head`, `eye_l` and `eye_r` too. The turn is put
 * on top of whatever pose the bone has when `setHead`/`setEye` is called
 * (`parent-frame turn * animated rotation`), never in place of it. The order
 * each frame is:
 *
 *   adapter.release();   // take last frame's turn off again
 *   mixer.update(dt);    // body animation writes the bones
 *   player.update(dt);   // the face adds its turn and sets the morphs
 *
 * `release()` matters because the mixer skips bones whose animated value did
 * not change since its last write, and because it snapshots bones when an
 * action starts. If it is never called the adapter still does not pile turn
 * upon turn: a bone that still holds exactly what the adapter wrote is taken
 * as untouched and the earlier pose is reused.
 *
 * Morph names. A shape is looked up under the name asked for, then under
 * `options.morphNames[name]` (a name or a list), then under its Quest/VRChat
 * name, on every mesh that has morph targets, so one player drives both export
 * profiles and characters split over several meshes. Shapes no mesh has are
 * ignored and listed in `missing`.
 *
 * @param THREE the three.js module
 * @param root Object3D
 * @param options { morphNames: {}, bones: { head, eye_l, eye_r } }
 */
export function threeAdapter(THREE, root, options = {}) {
  const boneNames = { head: 'head', eye_l: 'eye_l', eye_r: 'eye_r', ...options.bones };
  const meshes = [];
  const found = {};
  root.traverse((node) => {
    // Rigid parts (hair cards, glasses) can carry morph targets without a skin.
    if (node.isMesh && node.morphTargetDictionary && node.morphTargetInfluences) meshes.push(node);
    for (const [key, name] of Object.entries(boneNames)) {
      if (node.name === name && (node.isBone || !found[key])) found[key] = node;
    }
  });

  const modelSpin = root.getWorldQuaternion(new THREE.Quaternion()).invert();
  /** Rotation of `node` in the model's frame, in the pose it has right now. */
  const inModel = (node) => modelSpin.clone().multiply(node.getWorldQuaternion(new THREE.Quaternion()));

  const joints = {};
  for (const [key, bone] of Object.entries(found)) {
    // The model's axes as the bone's parent sees them. Constant: it rides along with the parent.
    const frame = bone.parent ? inModel(bone.parent).invert() : new THREE.Quaternion();
    joints[key] = {
      bone,
      frame,
      frameInverse: frame.clone().invert(),
      base: bone.quaternion.clone(),
      written: null,
    };
  }

  const up = new THREE.Vector3(0, 1, 0);
  const side = new THREE.Vector3(1, 0, 0);
  const turn = new THREE.Quaternion();
  const tilt = new THREE.Quaternion();
  const delta = new THREE.Quaternion();
  let headPitch = 0;

  /** Put the model-frame rotation in `delta` on top of the joint's current pose. */
  const layer = (joint) => {
    const { bone } = joint;
    // Anything other than our own last write is a new pose from the animation.
    if (!joint.written || !bone.quaternion.equals(joint.written)) joint.base.copy(bone.quaternion);
    bone.quaternion.copy(joint.frame).multiply(delta).multiply(joint.frameInverse).multiply(joint.base);
    joint.written = (joint.written || new THREE.Quaternion()).copy(bone.quaternion);
  };

  const lookups = new Map();
  const missing = new Set();
  const lookup = (name) => {
    let targets = lookups.get(name);
    if (targets) return targets;
    const candidates = [name, ...[options.morphNames?.[name] ?? []].flat()];
    if (VRCHAT_NAMES[name]) candidates.push(VRCHAT_NAMES[name]);
    targets = [];
    for (const mesh of meshes) {
      const hit = candidates.find((candidate) => mesh.morphTargetDictionary[candidate] !== undefined);
      if (hit !== undefined) targets.push([mesh, mesh.morphTargetDictionary[hit]]);
    }
    if (!targets.length) missing.add(name);
    lookups.set(name, targets);
    return targets;
  };

  return {
    meshes,
    bones: Object.fromEntries(Object.entries(joints).map(([key, joint]) => [key, joint.bone])),
    /** Shape names that were asked for and that no mesh has. */
    missing,
    hasMorph: (name) => lookup(name).length > 0,
    setMorph(name, weight) {
      // The array is read from the mesh each time: cloning a mesh replaces it.
      for (const [mesh, index] of lookup(name)) mesh.morphTargetInfluences[index] = weight;
    },
    setHead(yaw, pitch) {
      headPitch = pitch;
      if (!joints.head) return;
      turn.setFromAxisAngle(up, yaw);
      tilt.setFromAxisAngle(side, -pitch);
      delta.copy(turn).multiply(tilt);
      layer(joints.head);
    },
    setEye(which, yaw, pitch) {
      const joint = joints[`eye_${which}`];
      if (!joint) return;
      // The eye rides on the head, which has already pitched. Undoing that pitch around the
      // eye's own turn makes the two add up: azimuth = head yaw + eye yaw, elevation likewise.
      tilt.setFromAxisAngle(side, -headPitch);
      delta.copy(tilt).invert();
      delta.multiply(turn.setFromAxisAngle(up, yaw));
      delta.multiply(turn.setFromAxisAngle(side, -pitch)).multiply(tilt);
      layer(joint);
    },
    /**
     * World rotation of the frame head angles are measured in: the model's axes as the
     * head's parent carries them. Written to `target`; null if there is no head bone.
     */
    headFrame(target) {
      const joint = joints.head;
      if (!joint) return null;
      return joint.bone.parent ? joint.bone.parent.getWorldQuaternion(target).multiply(joint.frame) : target.identity();
    },
    /** Take the face's turn off the bones again. Call before the AnimationMixer updates. */
    release() {
      for (const joint of Object.values(joints)) {
        if (joint.written && joint.bone.quaternion.equals(joint.written)) joint.bone.quaternion.copy(joint.base);
        joint.written = null;
      }
    },
  };
}
