"""Synthesize the score for the Kaggriculture cinematic (numpy only).

Slow string-like pads that move from D minor (the storm, the climate crisis)
to D major (sunrise over a fed world), with low impacts on each scene cut,
thunder under the lightning, noise risers into the two reveals, and a quiet
machine pulse while the latent search wakes the fields.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 44_100
DURATION = 96.0
N = int(SR * DURATION)
T = np.arange(N) / SR
RNG = np.random.default_rng(2033)


def hz(note: str) -> float:
    names = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
             "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
    name, octave = note[:-1], int(note[-1])
    return 440.0 * 2 ** ((names[name] + 12 * (octave + 1) - 69) / 12)


def window(a: float, b: float, fade: float) -> np.ndarray:
    return np.clip((T - a) / fade, 0, 1) * np.clip((b - T) / fade, 0, 1)


def bandpass_noise(lo: float, hi: float, n: int = N) -> np.ndarray:
    spec = np.fft.rfft(RNG.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    out = np.fft.irfft(spec, n)
    return out / (np.abs(out).max() + 1e-9)


# (start, end, notes, level) — cross-faded pad chords
CHORDS = [
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
]
IMPACTS = [7.0, 16.9, 24.8, 38.9, 50.9, 65.9, 87.0]
THUNDER = [9.3, 12.7, 15.1]
RISERS = [(13.5, 16.9), (62.0, 65.9)]


def pad(freq: float, sl: slice) -> np.ndarray:
    t = T[sl]
    out = np.zeros(len(t))
    for det in (-.12, 0.0, .12):  # three slightly detuned voices -> chorus
        f = freq * 2 ** (det / 12 / 4)
        vib = 1 + .002 * np.sin(2 * np.pi * (4.6 + det) * t)
        ph = 2 * np.pi * f * np.cumsum(vib) / SR
        for h, amp in ((1, 1.0), (2, .45), (3, .22), (4, .1), (5, .05)):
            out += amp * np.sin(h * ph + RNG.uniform(0, 6.28))
    return out / 5


def impact(at: float) -> np.ndarray:
    tt = np.clip(T - at, 0, None)
    on = T >= at
    f = 30 + 70 * np.exp(-tt * 6)
    boom = np.sin(2 * np.pi * np.cumsum(f * on) / SR) * np.exp(-tt * 1.6) * on
    hit = bandpass_noise(40, 900) * np.exp(-tt * 9) * on
    return 1.4 * boom + .35 * hit


def reverb(x: np.ndarray, seconds: float = 3.2, mix: float = .35) -> np.ndarray:
    n = int(SR * seconds)
    ir = RNG.standard_normal(n) * np.exp(-np.arange(n) / SR * 2.4)
    ir[: int(SR * .02)] = 0
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    wet = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]
    wet /= np.abs(wet).max() + 1e-9
    return (1 - mix) * x / (np.abs(x).max() + 1e-9) + mix * wet


def synthesize(path: Path) -> Path:
    left = np.zeros(N)
    right = np.zeros(N)

    # pads, each voice panned across the stereo field
    for a, b, notes, lvl in CHORDS:
        sl = slice(int(a * SR), min(N, int(b * SR)))
        w = window(a, b, 1.8)[sl] * lvl
        for i, nt in enumerate(notes):
            v = pad(hz(nt), sl) * w
            p = .5 + .35 * np.sin(i * 1.7)
            left[sl] += v * (1 - p)
            right[sl] += v * p
    pads = .16

    # climate section: a nervous tremolo on the dissonant chord
    trem = 1 - .35 * window(38.5, 51, 1.5) * (.5 + .5 * np.sin(2 * np.pi * 6 * T))
    left *= trem * pads
    right *= trem * pads

    # impacts
    hits = sum(impact(t) for t in IMPACTS) * .28
    left += hits
    right += hits

    # thunder rumble under the lightning
    rumble = bandpass_noise(20, 220)
    for s in THUNDER:
        tt = np.clip(T - s - .15, 0, None)
        env = (T >= s + .15) * np.minimum(tt * 8, 1) * np.exp(-tt * 1.2)
        left += .5 * rumble * env
        right += .5 * np.roll(rumble, 900) * env
    rain = bandpass_noise(1500, 9000) * window(6.5, 17.2, 1.5) * .03
    left += rain
    right += np.roll(rain, 3000)

    # risers into the two reveals
    hiss = bandpass_noise(800, 12000)
    for a, b in RISERS:
        k = np.clip((T - a) / (b - a), 0, 1) * (T <= b)
        left += .22 * hiss * k ** 3
        right += .22 * np.roll(hiss, 1500) * k ** 3

    # machine pulse while the fields wake (120 bpm eighths)
    tt = (T - 50.5) % .25
    pulse = np.sin(2 * np.pi * hz("D3") * T) * np.exp(-tt * 28) * window(52, 65.9, 1.5) * .09
    left += pulse
    right += pulse

    # sunrise shimmer: a slow high arpeggio over the D major swell
    arp = ["D5", "F#5", "A5", "D6", "A5", "F#5"]
    for i in range(int((88 - 77) / .5)):
        s = 77 + i * .5
        tt = np.clip(T - s, 0, None)
        env = (T >= s) * np.exp(-tt * 2.2) * .05
        tone = np.sin(2 * np.pi * hz(arp[i % len(arp)]) * T) * env
        left += tone * (.7 if i % 2 else .3)
        right += tone * (.3 if i % 2 else .7)

    mix = np.stack([reverb(left), reverb(right)], axis=1)
    master = window(0, DURATION, 1.5)[:, None] * np.clip(T / 2.5, 0, 1)[:, None]
    mix *= master
    mix = np.tanh(mix / (np.abs(mix).max() + 1e-9) * 1.4) * .89

    pcm = (mix * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    return path


if __name__ == "__main__":
    print(synthesize(Path(__file__).resolve().parent / "soundtrack.wav"))
