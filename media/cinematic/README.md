# Kaggriculture CloudFronts — Films

Two procedural films. Every visual and every sound is generated in code; there is no stock footage or third-party audio. Both are 1920×1080 @ 30 fps, 2.39:1 letterbox, stereo score.

## 1. Cinematic & Visionary — `kaggriculture_cinematic.mp4` (1:36)

| Time | Scene | Line |
| :--- | :--- | :--- |
| 0:00 | Dust in a shaft of light — studio credit | *Kaggriculture CloudFronts presents* |
| 0:07 | Storm over parched fields, lightning | "When we set out to compete in this challenge, we didn't build for second place." |
| 0:17 | Golden burst, god rays | "We aimed for **absolute victory**." |
| 0:25 | A network of intelligence wraps the planet | "Unprecedented success for every innovator involved. Not just to top a leaderboard — but to redefine what is possible at the intersection of high-performance intelligence and global survival." |
| 0:39 | Overheating globe, climate volatility index | "In an era defined by extreme climate volatility, incremental progress is no longer enough." |
| 0:51 | Latent MCTS tree descends; dead fields turn green (hθ · gθ · fθ · Sampled MuZero) | "We confronted a massive agricultural hurdle and transformed it into an intelligence-driven farming revolution." |
| 1:06 | Laurel, "1ST", best Kaggle score counting up to 2033 | "Today, as first-place winners, we aren't just presenting a model." |
| 1:17 | Sunrise over a green planet | "We are presenting **the future of world food security**." |
| 1:27 | End card | "This is how we did it." |

## 2. Everyone Wins — `kaggriculture_everyone_wins.mp4` (2:08)

| Time | Scene | Line |
| :--- | :--- | :--- |
| 0:00 | Pollen in warm light — studio credit, title | *Everyone Wins* |
| 0:08 | Dawn over farmland; produce flows from every farm into a town whose windows light up | "Imagine a world where every farmer succeeds…" → "…feeding communities overnight." |
| 0:27 | Nine neighboring farms joined by gold threads; they dim for the question, then burst into light | "Now imagine a supply chain where your neighbor's prosperity is your primary concern…" → "…everyone wins — and they do so spectacularly." |
| 0:48 | Head to head: both seats' cash curves rise together; both harvests sell out | "We set out to prove this vision…" → "Because that is the only future worth building." |
| 1:12 | Two players, a Prisoner's Dilemma payoff matrix; mutual cooperation lights up | "With zero direct communication, we cracked the Prisoner's Dilemma…" |
| 1:21 | Step 0: the real NeML opening orders from `dist/main_mz.py` type out; value counter | "On Step 0… the NeML move." → "…redefines financial possibilities." |
| 1:39 | Teal and amber strands rise as a double helix; storm clouds clear to dawn | "Beyond the numbers…" → "“I learned this; now let's win together.”" → "…shared success is the only way forward." |
| 2:01 | End card | "Everyone wins." |

The Step 0 on-screen value comes from `NEML_VALUE` at the top of `everyone_wins.html`.

## Rebuild

```bash
pip install playwright imageio-ffmpeg numpy
python media/cinematic/render_movie.py                                 # film 1 (~8 min)
python media/cinematic/render_movie.py --film everyone_wins            # film 2 (~11 min)
python media/cinematic/render_movie.py --film everyone_wins --stills 20 60   # preview frames
```

- `engine.js` — shared engine: timing/easing, globe, grain, letterbox, word-by-word subtitles, and `film(DURATION, SCENES)`, which exposes `window.renderAt(t)`.
- `cinematic.html`, `everyone_wins.html` — each film as a canvas scene graph. Open one in a browser to watch it live (silent).
- `soundtrack.py` — numpy-synthesized scores, one `Cue` sheet per film (`OPTION1`, `EVERYONE_WINS`).
- `render_movie.py` — steps a film page frame-by-frame in headless Chromium and muxes its score with ffmpeg.
