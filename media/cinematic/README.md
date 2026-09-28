# Kaggriculture CloudFronts — Cinematic (Option 1: Cinematic & Visionary)

`kaggriculture_cinematic.mp4` — 1:36, 1920×1080 @ 30 fps, 2.39:1 letterbox, stereo score.

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

## Rebuild

Everything is procedural — no stock footage or third-party audio.

```bash
pip install playwright imageio-ffmpeg numpy
python media/cinematic/render_movie.py                    # full film (~8 min)
python media/cinematic/render_movie.py --stills 20 60 80  # preview frames
```

- `cinematic.html` — the film as a canvas scene graph; `window.renderAt(t)` draws any frame. Open it in a browser to watch it live (silent).
- `soundtrack.py` — numpy-synthesized score: D-minor pads resolving to D major, impacts on cuts, thunder, risers, machine pulse, sunrise shimmer.
- `render_movie.py` — steps the page frame-by-frame in headless Chromium and muxes the score with ffmpeg.
