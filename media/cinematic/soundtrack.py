"""Synthesize the scores for the Kaggriculture films (numpy only).

Each film has a ``Cue`` sheet: cross-faded string-like pad chords, low impacts
on scene cuts, optional thunder/rain, noise risers into reveals, a quiet
machine pulse, and a high shimmer arpeggio.

* ``OPTION1`` — "Cinematic & Visionary": D minor (storm, climate crisis)
  resolving to D major (sunrise over a fed world).
* ``EVERYONE_WINS`` — warm G major throughout, plucked arpeggios for the
  harvest and the uplift, a thinking pulse under the Prisoner's Dilemma.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 44_100


def hz(note: str) -> float:
    names = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
             "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
    name, octave = note[:-1], int(note[-1])
    return 440.0 * 2 ** ((names[name] + 12 * (octave + 1) - 69) / 12)


@dataclass
class Cue:
    duration: float
    chords: list  # (start, end, notes, level) — cross-faded pad chords
    impacts: list = field(default_factory=list)
    impact_gain: float = .28
    thunder: list = field(default_factory=list)
    rain: tuple | None = None  # (start, end)
    risers: list = field(default_factory=list)  # (start, end)
    tremolo: tuple | None = None  # (start, end) nervous tremolo on the pads
    pulse: tuple | None = None  # (start, end, note) 120 bpm eighths
    arps: list = field(default_factory=list)  # (start, end, notes, step, gain)
    seed: int = 2033


OPTION1 = Cue(
    duration=96.0,
    chords=[
        (0.0, 7.5, ["D2", "A2", "D3"], .7),
        (6.0, 17.5, ["D2", "A2", "D3", "F3", "A3"], .8),
        (16.5, 25.5, ["Bb1", "F2", "Bb2", "D3", "F3", "Bb3"], 1.0),
        (24.0, 32.0, ["F2", "C3", "F3", "A3", "C4"], .8),
        (31.0, 39.5, ["C2", "G2", "C3", "E3", "G3"], .8),
        (38.0, 51.5, ["G1", "D2", "G2", "Bb2", "D3", "Eb3"], .85),
        (50.5, 58.5, ["Bb1", "F2", "Bb2", "D3", "F3"], .85),
        (57.5, 66.5, ["C2", "G2", "C3", "E3", "G3", "C4"], .95),
        (65.5, 77.5, ["F2", "C3", "F3", "A3", "C4", "F4"], 1.0),
        (76.5, 88.0, ["D2", "A2", "D3", "F#3", "A3", "D4", "F#4"], 1.1),
        (87.0, 96.0, ["D2", "A2", "D3", "E3", "A3", "D4"], .8),
    ],
    impacts=[7.0, 16.9, 24.8, 38.9, 50.9, 65.9, 87.0],
    thunder=[9.3, 12.7, 15.1],
    rain=(6.5, 17.2),
    risers=[(13.5, 16.9), (62.0, 65.9)],
    tremolo=(38.5, 51.0),
    pulse=(52.0, 65.9, "D3"),
    arps=[(77.0, 88.0, ["D5", "F#5", "A5", "D6", "A5", "F#5"], .5, .05)],
)

_G = ["G2", "D3", "G3", "B3"]
_C = ["C2", "G2", "C3", "E3", "G3"]
_CADD9 = ["C2", "G2", "C3", "D4", "E4"]
_D = ["D2", "A2", "D3", "F#3", "A3"]
_EM = ["E2", "B2", "E3", "G3", "B3"]
_PENTA = ["G5", "B5", "D6", "E6", "D6", "B5"]

EVERYONE_WINS = Cue(
    duration=128.0,
    chords=[
        (0.0, 9.0, _G, .7),
        (8.0, 15.0, _G + ["D4"], .8), (14.0, 21.0, _CADD9, .8), (20.0, 27.5, _EM, .8),
        (26.0, 33.0, _C, .85), (32.0, 39.0, _D, .85), (38.0, 42.5, ["E2", "B2", "E3"], .55),
        (42.0, 48.5, _G + ["D4", "G4"], 1.05),
        (47.5, 55.0, _C, .9), (54.5, 61.0, ["B1", "G2", "D3", "G3", "B3"], .9),
        (60.5, 67.0, ["A2", "E3", "G3", "C4"], .9), (66.5, 72.5, _D + ["D4"], 1.0),
        (71.5, 80.0, _EM, .8), (79.5, 87.5, _C, .85), (87.0, 93.0, _D, .95), (92.5, 99.5, _G + ["D4", "G4"], 1.05),
        (98.5, 105.0, _CADD9, .9), (104.5, 110.0, _G + ["D4"], 1.0), (109.5, 115.0, _EM, .9),
        (114.5, 121.5, _D + ["D4"], 1.0),
        (120.0, 128.0, _G + ["D4", "G4"], 1.1),
    ],
    impacts=[8.0, 42.0, 48.5, 72.0, 81.3, 120.5],
    impact_gain=.2,
    risers=[(39.5, 42.0), (79.0, 81.3), (118.0, 120.5)],
    pulse=(72.5, 99.0, "G3"),
    arps=[(8.5, 27.0, _PENTA, .25, .035), (42.0, 72.0, _PENTA, .5, .03), (99.0, 126.0, _PENTA, .25, .04)],
    seed=80,
)


def synthesize(path: Path, cue: Cue = OPTION1) -> Path:
    n = int(SR * cue.duration)
    T = np.arange(n) / SR
    rng = np.random.default_rng(cue.seed)

    def window(a: float, b: float, fade: float) -> np.ndarray:
        return np.clip((T - a) / fade, 0, 1) * np.clip((b - T) / fade, 0, 1)

    def bandpass_noise(lo: float, hi: float) -> np.ndarray:
        spec = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / SR)
        spec[(f < lo) | (f > hi)] = 0
        out = np.fft.irfft(spec, n)
        return out / (np.abs(out).max() + 1e-9)

    def pad(freq: float, sl: slice) -> np.ndarray:
        t = T[sl]
        out = np.zeros(len(t))
        for det in (-.12, 0.0, .12):  # three slightly detuned voices -> chorus
            f = freq * 2 ** (det / 12 / 4)
            vib = 1 + .002 * np.sin(2 * np.pi * (4.6 + det) * t)
            ph = 2 * np.pi * f * np.cumsum(vib) / SR
            for h, amp in ((1, 1.0), (2, .45), (3, .22), (4, .1), (5, .05)):
                out += amp * np.sin(h * ph + rng.uniform(0, 6.28))
        return out / 5

    def impact(at: float) -> np.ndarray:
        tt = np.clip(T - at, 0, None)
        on = T >= at
        f = 30 + 70 * np.exp(-tt * 6)
        boom = np.sin(2 * np.pi * np.cumsum(f * on) / SR) * np.exp(-tt * 1.6) * on
        hit = bandpass_noise(40, 900) * np.exp(-tt * 9) * on
        return 1.4 * boom + .35 * hit

    def reverb(x: np.ndarray, seconds: float = 3.2, mix: float = .35) -> np.ndarray:
        m = int(SR * seconds)
        ir = rng.standard_normal(m) * np.exp(-np.arange(m) / SR * 2.4)
        ir[: int(SR * .02)] = 0
        size = 1 << int(np.ceil(np.log2(len(x) + m)))
        wet = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]
        wet /= np.abs(wet).max() + 1e-9
        return (1 - mix) * x / (np.abs(x).max() + 1e-9) + mix * wet

    left = np.zeros(n)
    right = np.zeros(n)

    # pads, each voice panned across the stereo field
    for a, b, notes, lvl in cue.chords:
        sl = slice(int(a * SR), min(n, int(b * SR)))
        w = window(a, b, 1.8)[sl] * lvl
        for i, nt in enumerate(notes):
            v = pad(hz(nt), sl) * w
            p = .5 + .35 * np.sin(i * 1.7)
            left[sl] += v * (1 - p)
            right[sl] += v * p
    gain = .16
    if cue.tremolo:
        gain = gain * (1 - .35 * window(*cue.tremolo, 1.5) * (.5 + .5 * np.sin(2 * np.pi * 6 * T)))
    left *= gain
    right *= gain

    if cue.impacts:
        hits = sum(impact(t) for t in cue.impacts) * cue.impact_gain
        left += hits
        right += hits

    if cue.thunder:
        rumble = bandpass_noise(20, 220)
        for s in cue.thunder:
            tt = np.clip(T - s - .15, 0, None)
            env = (T >= s + .15) * np.minimum(tt * 8, 1) * np.exp(-tt * 1.2)
            left += .5 * rumble * env
            right += .5 * np.roll(rumble, 900) * env
    if cue.rain:
        rain = bandpass_noise(1500, 9000) * window(*cue.rain, 1.5) * .03
        left += rain
        right += np.roll(rain, 3000)

    if cue.risers:
        hiss = bandpass_noise(800, 12000)
        for a, b in cue.risers:
            k = np.clip((T - a) / (b - a), 0, 1) * (T <= b)
            left += .22 * hiss * k ** 3
            right += .22 * np.roll(hiss, 1500) * k ** 3

    if cue.pulse:
        a, b, note = cue.pulse
        tt = (T - a) % .25
        pulse = np.sin(2 * np.pi * hz(note) * T) * np.exp(-tt * 28) * window(a, b, 1.5) * .09
        left += pulse
        right += pulse

    for a, b, notes, step, g in cue.arps:
        for i in range(int((b - a) / step)):
            s = a + i * step
            sl = slice(int(s * SR), min(n, int((s + 2.5) * SR)))
            tt = T[sl] - s
            tone = np.sin(2 * np.pi * hz(notes[i % len(notes)]) * T[sl]) * np.exp(-tt * 2.2) * g
            left[sl] += tone * (.7 if i % 2 else .3)
            right[sl] += tone * (.3 if i % 2 else .7)

    mix = np.stack([reverb(left), reverb(right)], axis=1)
    mix *= window(0, cue.duration, 1.5)[:, None] * np.clip(T / 2.5, 0, 1)[:, None]
    mix = np.tanh(mix / (np.abs(mix).max() + 1e-9) * 1.4) * .89

    pcm = (mix * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    return path


CUES = {"cinematic": OPTION1, "everyone_wins": EVERYONE_WINS}

if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "cinematic"
    print(synthesize(Path(__file__).resolve().parent / f"{name}_soundtrack.wav", CUES[name]))
