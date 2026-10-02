// SPDX-FileCopyrightText: 2026 Elyan Labs
//
// SPDX-License-Identifier: GPL-2.0-or-later

/**
 * One exported character in a three.js scene: body clips, face, speech, gaze.
 *
 * It only ties together what three.js and elyan_face.js already do, and owns
 * the one thing that is easy to get wrong: the order in which the body
 * animation and the face write the head and eye bones (see `update`).
 *
 * three.js is passed in and never imported, so this file loads wherever the
 * application's own copy of three.js lives.
 */

import { FacePlayer, threeAdapter } from './elyan_face.js';

export class ElyanCharacter {
  /**
   * @param THREE the three.js module
   * @param gltf what GLTFLoader gives: { scene, animations }
   * @param options {
   *   clock:    function returning seconds on the clock the audio is scheduled on, for
   *             example () => audioContext.currentTime. Without it speech runs on frame time.
   *   talkClip: clip to cross-fade to while speaking, 'talk' by default, null for none
   *   face:     options for FacePlayer (seed, blink, saccades)
   *   morphNames, bones: passed on to threeAdapter
   * }
   */
  constructor(THREE, gltf, options = {}) {
    this.THREE = THREE;
    this.root = gltf.scene;
    // Before the mixer exists, so the adapter sees the rest pose.
    this.adapter = threeAdapter(THREE, this.root, { morphNames: options.morphNames, bones: options.bones });
    this.face = new FacePlayer(this.adapter, options.face);
    this.mixer = new THREE.AnimationMixer(this.root);
    this.clips = Object.fromEntries((gltf.animations ?? []).map((clip) => [clip.name, clip]));
    this.talkClip = options.talkClip === undefined ? 'talk' : options.talkClip;

    this.time = 0;
    this.clock = options.clock ?? (() => this.time);
    this.current = null;      // name of the clip that is playing
    this.resting = null;      // looping clip to come back to after a one-shot or after speech
    this._action = null;
    this._wasSpeaking = false;
    this._afterSpeech = null; // clip to return to when the sentence is over
    this._returnFade = 0.3;
    this._target = null;

    this._onFinished = (event) => {
      // A nod or a shake has ended: ease back into what was looping before it.
      if (event.action === this._action && this.resting && this.resting !== this.current) {
        this.playClip(this.resting, this._returnFade);
      }
    };
    this.mixer.addEventListener('finished', this._onFinished);

    this._eyeMiddle = new THREE.Vector3();
    this._scratch = new THREE.Vector3();
    this._from = new THREE.Vector3();
    this._spin = new THREE.Quaternion();
  }

  /**
   * Cross-fade to a body clip. Returns false if the character has no such clip.
   *
   * @param fade seconds of cross-fade
   * @param options { once: play it one time, then return to the clip that was looping }
   */
  playClip(name, fade = 0.3, options = {}) {
    const clip = this.clips[name];
    if (!clip) return false;
    if (name === this.current && !options.once) return true;
    // A starting action snapshots the bones as "the pose without animation". Keep the face out of it.
    this.adapter.release();
    const action = this.mixer.clipAction(clip);
    action.reset();
    action.setLoop(options.once ? this.THREE.LoopOnce : this.THREE.LoopRepeat, Infinity);
    // Hold the last frame while fading back, or the body would snap to rest for that long.
    action.clampWhenFinished = Boolean(options.once);
    action.play();
    if (this._action && this._action !== action) this._action.crossFadeTo(action, fade, false);
    else if (!this._action) action.fadeIn(fade);
    if (!options.once) this.resting = name;
    this._returnFade = fade;
    this._action = action;
    this.current = name;
    return true;
  }

  /**
   * Speak a lip-sync track in time with its sound. Returns the track's duration.
   *
   * @param track "elyan.visemes/1"
   * @param audioStartTime when the sound starts, on the clock given to the constructor
   *   (the value passed to AudioBufferSourceNode.start). Now, if left out.
   */
  say(track, audioStartTime = this.clock()) {
    const duration = this.face.play(track, () => this.clock() - audioStartTime);
    if (this.talkClip && this.clips[this.talkClip]) {
      const before = this.resting;
      this.playClip(this.talkClip);
      // `playClip` made the talk clip the resting one; remember what to return to instead.
      // A second sentence begun during the first must not forget it.
      if (!this._afterSpeech && before !== this.talkClip) this._afterSpeech = before;
    }
    this._wasSpeaking = true;
    return duration;
  }

  /** Stop speaking at once. */
  hush() {
    this.face.stop();
  }

  /** Blend toward an expression: a name from EMOTIONS or an object of shape weights. */
  setEmotion(emotion, strength = 1) {
    this.face.setEmotion(emotion, strength);
  }

  /**
   * Look at a point given in world space. It is followed from frame to frame, so the gaze
   * holds while the body moves. Pass null to look straight ahead again.
   *
   * @param point Vector3-like { x, y, z } or null. The object is kept, so a camera's
   *   or another character's `position` can be handed over once and is tracked.
   */
  lookAtWorld(point) {
    this._target = point;
    if (!point) {
      this.face.lookAt(0, 0);
      return null;
    }
    this._findEyes();
    return this._aim();
  }

  /**
   * Note where the point between the eyes is, in the root's frame, with the head as it is
   * turned now. Turning the head carries the eyes some centimetres sideways; an aim taken
   * from the unturned head would miss a near target by about a degree.
   */
  _findEyes() {
    const { head, eye_l: left, eye_r: right } = this.adapter.bones;
    if (!head) return;
    this.root.updateWorldMatrix(true, true);
    if (left && right) {
      this._eyeMiddle.setFromMatrixPosition(left.matrixWorld).add(this._scratch.setFromMatrixPosition(right.matrixWorld)).multiplyScalar(0.5);
    } else {
      this._eyeMiddle.setFromMatrixPosition(head.matrixWorld);
    }
    this.root.worldToLocal(this._eyeMiddle);
  }

  /**
   * Yaw and pitch of the target as the face player means them: relative to straight
   * ahead for a head that has not been turned. "Straight ahead" is the model's forward
   * axis carried along by the head's parent, so a body that leans or turns takes it with it.
   */
  _aim() {
    // World rotation of the model's axes as the neck carries them: parent * (parent at rest)^-1 * model.
    const frame = this.adapter.headFrame(this._spin);
    if (!frame) return null;
    // The eyes as last seen (see _findEyes), wherever the character has moved to since.
    const eyes = this.root.localToWorld(this._from.copy(this._eyeMiddle));
    const d = this._scratch.copy(this._target).sub(eyes).applyQuaternion(frame.invert());
    const yaw = Math.atan2(d.x, d.z);
    const pitch = Math.atan2(d.y, Math.hypot(d.x, d.z));
    this.face.lookAt(yaw, pitch);
    return { yaw, pitch };
  }

  /**
   * Advance by `dt` seconds. Call once per frame, before rendering.
   *
   * The order is the point of this class. The body clip and the face both write the head
   * and eye bones; the face must be the last writer and must add to the animated pose:
   *
   *   1. take last frame's face turn off the bones
   *   2. the mixer writes the body pose
   *   3. the gaze target is measured against that pose
   *   4. the face adds its turn and sets the morphs
   *
   * Afterwards the bones hold the final pose; render, or read them, before the next update.
   */
  update(dt) {
    this.time += dt;
    this.adapter.release();
    this.mixer.update(dt);
    if (this._target) this._aim();
    const weights = this.face.update(dt);
    if (this._target) this._findEyes();
    if (this._wasSpeaking && !this.face.speaking) {
      this._wasSpeaking = false;
      if (this._afterSpeech && this.current === this.talkClip) this.playClip(this._afterSpeech);
      this._afterSpeech = null;
    }
    return weights;
  }

  /** Free what the mixer cached for this character. */
  dispose() {
    this.mixer.removeEventListener('finished', this._onFinished);
    this.mixer.stopAllAction();
    this.mixer.uncacheRoot(this.root);
    this.adapter.release();
  }
}
