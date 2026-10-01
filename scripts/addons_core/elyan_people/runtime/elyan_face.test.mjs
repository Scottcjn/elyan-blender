// SPDX-FileCopyrightText: 2026 Elyan Labs
//
// SPDX-License-Identifier: GPL-2.0-or-later

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { FacePlayer, VISEMES } from './elyan_face.js';

function recorder() {
  const state = { morphs: {}, eyes: {}, head: null };
  return {
    state,
    setMorph: (name, weight) => { state.morphs[name] = weight; },
    setEye: (side, yaw, pitch) => { state.eyes[side] = { yaw, pitch }; },
    setHead: (yaw, pitch) => { state.head = { yaw, pitch }; },
  };
}

/** A one-second track: rest, then "aa" fully open at 0.5 s, then rest. */
function track() {
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

test('plays a track in time and stops at its end', () => {
  const adapter = recorder();
  const player = new FacePlayer(adapter, { blink: false, saccades: false });
  assert.equal(player.play(track()), 1);
  for (let i = 0; i < 15; i++) player.update(1 / 30);
  assert.ok(adapter.state.morphs.viseme_aa > 0.95, `aa at 0.5 s was ${adapter.state.morphs.viseme_aa}`);
  for (let i = 0; i < 20; i++) player.update(1 / 30);
  assert.equal(player.speaking, false);
  assert.equal(adapter.state.morphs.viseme_aa, 0);
});

test('interpolates between frames', () => {
  const adapter = recorder();
  const player = new FacePlayer(adapter, { blink: false, saccades: false });
  player.play(track());
  player.update(11.5 / 30);
  assert.ok(Math.abs(adapter.state.morphs.viseme_aa - 0.3) < 0.02);
});

test('rejects tracks it does not understand', () => {
  assert.throws(() => new FacePlayer(recorder()).play({ schema: 'other/1' }));
});

test('blinks on its own, fully and briefly', () => {
  const adapter = recorder();
  const player = new FacePlayer(adapter, { seed: 7, saccades: false });
  let peak = 0;
  let closedFrames = 0;
  for (let i = 0; i < 60 * 20; i++) {
    player.update(1 / 60);
    peak = Math.max(peak, adapter.state.morphs.eyeBlinkLeft);
    if (adapter.state.morphs.eyeBlinkLeft > 0.5) closedFrames++;
  }
  assert.ok(peak > 0.9, `deepest blink ${peak}`);
  // 20 s should hold a handful of blinks, each a fraction of a second.
  assert.ok(closedFrames > 10 && closedFrames < 200, `${closedFrames} closed frames`);
  assert.equal(adapter.state.morphs.eyeBlinkLeft, adapter.state.morphs.eyeBlinkRight);
});

test('eyes lead and the head follows a look', () => {
  const adapter = recorder();
  const player = new FacePlayer(adapter, { blink: false, saccades: false });
  player.lookAt(0.8, 0);
  player.update(1 / 60);
  assert.ok(adapter.state.eyes.l.yaw > 0.4, 'eyes jump first');
  assert.ok(adapter.state.head.yaw < 0.05, 'head has barely moved');
  for (let i = 0; i < 180; i++) player.update(1 / 60);
  assert.ok(Math.abs(adapter.state.head.yaw - 0.56) < 0.02, `head settled at ${adapter.state.head.yaw}`);
  assert.ok(Math.abs(adapter.state.head.yaw + adapter.state.eyes.l.yaw - 0.8) < 0.02, 'together they reach the target');
});

test('emotion eases in and yields to an open mouth', () => {
  const adapter = recorder();
  const player = new FacePlayer(adapter, { blink: false, saccades: false });
  player.setEmotion('happy');
  player.update(1 / 60);
  assert.ok(adapter.state.morphs.mouthSmileLeft < 0.2, 'does not snap');
  for (let i = 0; i < 120; i++) player.update(1 / 60);
  const smiling = adapter.state.morphs.mouthSmileLeft;
  assert.ok(Math.abs(smiling - 0.7) < 0.01);
  player.play(track());
  for (let i = 0; i < 30; i++) player.update(1 / 60);
  assert.ok(adapter.state.morphs.mouthSmileLeft < smiling * 0.5, 'smile gives way at full "aa"');
  assert.throws(() => player.setEmotion('nonsense'));
});

test('every weight stays between 0 and 1 and the same seed repeats', () => {
  const run = () => {
    const adapter = recorder();
    const player = new FacePlayer(adapter, { seed: 3 });
    player.setEmotion('surprised');
    player.play(track());
    const log = [];
    for (let i = 0; i < 300; i++) {
      const weights = player.update(1 / 60);
      for (const [name, value] of Object.entries(adapter.state.morphs)) {
        assert.ok(value >= 0 && value <= 1, `${name} = ${value}`);
      }
      log.push(weights.eyeBlinkLeft, adapter.state.eyes.l.yaw);
    }
    return log;
  };
  assert.deepEqual(run(), run());
});
