#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Lip-sync tracks: timed phonemes in, mouth-shape weights per frame out.

No ``bpy``, standard library only, so a speech server can run it beside the
text-to-speech engine. Best input is the engine's own phoneme timestamps
(``from_phonemes``). Without them, ``from_text`` asks eSpeak NG for the phonemes
and spreads them over the clip's duration, which keeps mouths plausible but is
not frame-accurate.

   speech.py "Hello there, traveller." --duration 1.8 -o hello.visemes.json
"""

import argparse
import json
import math
import subprocess
import sys

SCHEMA = "elyan.visemes/1"

VISEMES = (
    "viseme_sil", "viseme_PP", "viseme_FF", "viseme_TH", "viseme_DD", "viseme_kk", "viseme_CH",
    "viseme_SS", "viseme_nn", "viseme_RR", "viseme_aa", "viseme_E", "viseme_I", "viseme_O", "viseme_U",
)

# IPA symbol -> [(mouth shape, strength)]. Diphthongs glide through two shapes.
IPA = {
    "p": [("PP", 1.0)], "b": [("PP", 1.0)], "m": [("PP", 1.0)],
    "f": [("FF", 1.0)], "v": [("FF", 1.0)],
    "θ": [("TH", 1.0)], "ð": [("TH", 1.0)],
    "t": [("DD", 0.9)], "d": [("DD", 0.9)], "ɾ": [("DD", 0.6)],
    "k": [("kk", 0.8)], "ɡ": [("kk", 0.8)], "g": [("kk", 0.8)], "ŋ": [("kk", 0.7)], "x": [("kk", 0.7)],
    "tʃ": [("CH", 1.0)], "dʒ": [("CH", 1.0)], "ʃ": [("CH", 1.0)], "ʒ": [("CH", 1.0)],
    "s": [("SS", 0.9)], "z": [("SS", 0.9)],
    "n": [("nn", 0.8)], "l": [("nn", 0.8)],
    "ɹ": [("RR", 0.9)], "r": [("RR", 0.9)], "ɚ": [("RR", 0.8)], "ɝ": [("RR", 0.9)],
    "a": [("aa", 1.0)], "ɑ": [("aa", 1.0)], "æ": [("aa", 0.9)], "ʌ": [("aa", 0.7)], "ɐ": [("aa", 0.6)],
    "e": [("E", 0.9)], "ɛ": [("E", 0.9)], "ə": [("E", 0.45)], "ɜ": [("E", 0.7)],
    "i": [("I", 0.9)], "ɪ": [("I", 0.8)], "j": [("I", 0.6)],
    "o": [("O", 1.0)], "ɔ": [("O", 1.0)], "ɒ": [("O", 0.9)],
    "u": [("U", 1.0)], "ʊ": [("U", 0.8)], "w": [("U", 0.9)],
    "h": [("aa", 0.25)],
    "aɪ": [("aa", 1.0), ("I", 0.8)], "aʊ": [("aa", 1.0), ("U", 0.8)], "ɔɪ": [("O", 1.0), ("I", 0.8)],
    "eɪ": [("E", 0.9), ("I", 0.7)], "oʊ": [("O", 1.0), ("U", 0.8)],
}
# ARPAbet (CMU dictionary, most English text-to-speech engines) -> IPA.
ARPABET = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AX": "ə", "AY": "aɪ", "B": "b", "CH": "tʃ",
    "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ",
    "IY": "i", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ",
    "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}
# Rhubarb Lip Sync's mouth cues -> mouth shape.
RHUBARB = {
    "A": ("PP", 1.0), "B": ("SS", 0.8), "C": ("E", 0.9), "D": ("aa", 1.0), "E": ("O", 0.9),
    "F": ("U", 1.0), "G": ("FF", 1.0), "H": ("nn", 0.9), "X": ("sil", 1.0),
}
_VOWELS = set("aɑæʌɐeɛəɜiɪoɔɒuʊ")
_PAUSE = "_"

# Mouths start forming a sound before it is heard and relax after it.
ANTICIPATION = 0.045
RELEASE = 0.07


def tokenize_ipa(text):
    """Split an IPA string into known symbols; spaces and punctuation become pauses."""
    tokens = []
    index = 0
    while index < len(text):
        pair = text[index:index + 2]
        if pair in IPA:
            tokens.append(pair)
            index += 2
            continue
        char = text[index]
        index += 1
        if char in IPA:
            tokens.append(char)
        elif char in ",.;:!?\n" and tokens and tokens[-1] != _PAUSE:
            tokens.append(_PAUSE)
        # Stress marks, length marks and anything unknown shape no mouth.
    return tokens


def espeak_ipa(text, voice="en-us"):
    """Phonemes of ``text`` from eSpeak NG, as an IPA string."""
    try:
        done = subprocess.run(
            ["espeak-ng", "-q", "--ipa", "-v", voice, text], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise RuntimeError("espeak-ng is not installed; pass timed phonemes instead") from None
    return done.stdout


def spread(tokens, duration):
    """Give untimed phonemes plausible times across ``duration`` seconds."""
    weights = [2.2 if t == _PAUSE else 1.5 if t[0] in _VOWELS else (1.9 if len(t) == 2 else 1.0) for t in tokens]
    scale = duration / (sum(weights) or 1.0)
    timed, clock = [], 0.0
    for token, weight in zip(tokens, weights):
        timed.append((token, clock, clock + weight * scale))
        clock += weight * scale
    return timed


def _window(time, start, end):
    """0..1 envelope of one sound: eases in before it starts, holds, eases out after."""
    if time < start - ANTICIPATION or time > end + RELEASE:
        return 0.0
    if time < start:
        return 0.5 - 0.5 * math.cos(math.pi * (time - start + ANTICIPATION) / ANTICIPATION)
    if time > end:
        return 0.5 + 0.5 * math.cos(math.pi * (time - end) / RELEASE)
    return 1.0


def track(segments, duration, fps=30):
    """
    Build a track from ``[(shape, strength, start, end)]``; shape is a name without "viseme_".

    Shapes overlap as neighbouring sounds do. Lip closures win over whatever
    surrounds them, because a P that never closes is what the eye catches first.
    """
    index = {name[7:]: i for i, name in enumerate(VISEMES)}
    frames = []
    for number in range(int(math.ceil(duration * fps)) + 1):
        time = number / fps
        row = [0.0] * len(VISEMES)
        for shape, strength, start, end in segments:
            if shape != "sil":
                slot = index[shape]
                row[slot] = max(row[slot], strength * _window(time, start, end))
        closure = max(row[index["PP"]], row[index["FF"]])
        for slot in range(1, len(row)):
            if slot not in (index["PP"], index["FF"]):
                row[slot] *= 1.0 - 0.85 * closure
        total = sum(row)
        if total > 1.0:
            row = [value / total for value in row]
            total = 1.0
        row[0] = 1.0 - total
        frames.append([round(value, 3) for value in row])
    return {"schema": SCHEMA, "fps": fps, "duration": round(duration, 4), "names": list(VISEMES), "frames": frames}


def from_phonemes(phonemes, duration=None, fps=30, alphabet="ipa"):
    """
    Track from timed phonemes: ``[(symbol, start, end)]`` in seconds.

    ``alphabet`` is "ipa" or "arpabet" (stress digits are ignored).
    """
    segments = []
    for symbol, start, end in phonemes:
        if alphabet == "arpabet":
            symbol = ARPABET.get(symbol.rstrip("012").upper(), "")
        shapes = IPA.get(symbol)
        if not shapes:
            continue
        step = (end - start) / len(shapes)
        for part, (shape, strength) in enumerate(shapes):
            segments.append((shape, strength, start + part * step, start + (part + 1) * step))
    if duration is None:
        duration = max((end for _symbol, _start, end in phonemes), default=0.0) + RELEASE
    return track(segments, duration, fps)


def from_text(text, duration, fps=30, voice="en-us"):
    """Track for a sentence when only the clip's length is known. Approximate timing."""
    return from_phonemes(spread(tokenize_ipa(espeak_ipa(text, voice)), duration), duration, fps)


def from_rhubarb(cues, fps=30):
    """Track from Rhubarb Lip Sync's JSON ``mouthCues`` list."""
    segments = [(*RHUBARB[cue["value"]], cue["start"], cue["end"]) for cue in cues if cue["value"] in RHUBARB]
    return track(segments, max((cue["end"] for cue in cues), default=0.0), fps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", help="sentence to speak (needs --duration and espeak-ng)")
    parser.add_argument("--duration", type=float, help="length of the audio clip in seconds")
    parser.add_argument("--phonemes", help="JSON file of [[symbol, start, end], ...]")
    parser.add_argument("--alphabet", choices=("ipa", "arpabet"), default="ipa")
    parser.add_argument("--rhubarb", help="Rhubarb Lip Sync JSON output file")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("-o", "--output", help="write here instead of standard output")
    opts = parser.parse_args()

    if opts.phonemes:
        with open(opts.phonemes, encoding="utf-8") as fh:
            result = from_phonemes([tuple(item) for item in json.load(fh)], opts.duration, opts.fps, opts.alphabet)
    elif opts.rhubarb:
        with open(opts.rhubarb, encoding="utf-8") as fh:
            result = from_rhubarb(json.load(fh)["mouthCues"], opts.fps)
    elif opts.text and opts.duration:
        result = from_text(opts.text, opts.duration, opts.fps)
    else:
        parser.error("give a sentence with --duration, or --phonemes, or --rhubarb")
    text = json.dumps(result, separators=(",", ":"))
    if opts.output:
        with open(opts.output, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
