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
 *   { setMorph(name, weight), setEye(side, yaw, pitch), setHead(yaw, pitch) }
 *
 * Angles are radians; positive yaw looks to the character's left, positive
 * pitch looks up. `threeAdapter` below builds one for three.js.
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

  /** Start a lip-sync track now. Returns its duration in seconds. */
  play(track) {
    if (track.schema !== 'elyan.visemes/1') throw new Error(`unknown track schema ${track.schema}`);
    this.track = track;
    this.trackTime = 0;
    this.speaking = true;
    return track.duration;
  }

  stop() {
    this.track = null;
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
      this.trackTime += dt;
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
    this.adapter.setEye('l', eyeYaw, eyePitch);
    this.adapter.setEye('r', eyeYaw, eyePitch);
    this.adapter.setHead(this.head.yaw, this.head.pitch);
    return weights;
  }
}

/**
 * Adapter for three.js. `root` is the loaded glTF scene.
 *
 * Bones exported from Blender point along their local Y; yaw turns about the
 * model's up axis and pitch about its side axis, composed on the rest rotation.
 * Not yet tried against a live three.js scene.
 */
export function threeAdapter(THREE, root) {
  const meshes = [];
  const bones = {};
  root.traverse((node) => {
    if (node.isSkinnedMesh && node.morphTargetDictionary) meshes.push(node);
    if (node.isBone) bones[node.name] = node;
  });
  const rest = Object.fromEntries(Object.entries(bones).map(([name, bone]) => [name, bone.quaternion.clone()]));
  const up = new THREE.Vector3(0, 1, 0);
  const side = new THREE.Vector3(1, 0, 0);
  const turn = new THREE.Quaternion();
  const tilt = new THREE.Quaternion();
  const aim = (name, yaw, pitch) => {
    const bone = bones[name];
    if (!bone) return;
    turn.setFromAxisAngle(up, yaw);
    tilt.setFromAxisAngle(side, -pitch);
    // Rotate in the parent's frame, then apply the rest rotation.
    bone.quaternion.copy(turn).multiply(tilt).multiply(rest[name]);
  };
  return {
    setMorph(name, weight) {
      for (const mesh of meshes) {
        const index = mesh.morphTargetDictionary[name];
        if (index !== undefined) mesh.morphTargetInfluences[index] = weight;
      }
    },
    setEye: (which, yaw, pitch) => aim(`eye_${which}`, yaw, pitch),
    setHead: (yaw, pitch) => aim('head', yaw, pitch),
  };
}
