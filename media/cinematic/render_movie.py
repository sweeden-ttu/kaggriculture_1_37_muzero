#!/usr/bin/env python3
"""Render the Kaggriculture CloudFronts cinematic to MP4.

Steps through ``cinematic.html`` frame by frame in headless Chromium
(``window.renderAt(t)`` is a pure function of time), pipes the frames into
ffmpeg, and muxes the synthesized score from ``soundtrack.py``.

    pip install playwright imageio-ffmpeg numpy
    python media/cinematic/render_movie.py                 # full 1080p film
    python media/cinematic/render_movie.py --stills 10 20  # preview PNGs
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
PAGE = (HERE / "cinematic.html").as_uri() + "?capture"


def _chromium_path() -> str | None:
    base = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")) / "chromium"
    for cand in (base / "chrome-linux" / "chrome", base):
        if cand.is_file():
            return str(cand)
    return None


def _open_page(pw):
    browser = pw.chromium.launch(executable_path=_chromium_path())
    page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    page.goto(PAGE)
    page.wait_for_function("typeof window.renderAt === 'function'")
    page.evaluate("document.fonts.ready")
    return browser, page


def _frame(page, t: float) -> bytes:
    page.evaluate(f"window.renderAt({t:.5f})")
    return page.locator("#c").screenshot(type="jpeg", quality=95)


def stills(times: list[float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser, page = _open_page(pw)
        for t in times:
            (out_dir / f"still_{t:06.2f}.jpg").write_bytes(_frame(page, t))
        browser.close()


def render(out: Path, fps: int, audio: Path | None) -> None:
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    with sync_playwright() as pw:
        browser, page = _open_page(pw)
        duration = float(page.evaluate("window.DURATION"))
        n = int(round(duration * fps))
        cmd = [ff, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-i", "-"]
        if audio:
            cmd += ["-i", str(audio)]
        cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "23", "-tune", "grain", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart"]
        if audio:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd.append(str(out))
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in range(n):
            proc.stdin.write(_frame(page, i / fps))
            if i % (fps * 5) == 0:
                print(f"  frame {i}/{n}", flush=True)
        proc.stdin.close()
        proc.wait()
        browser.close()
    if proc.returncode:
        raise SystemExit(f"ffmpeg failed ({proc.returncode})")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "kaggriculture_cinematic.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--stills", type=float, nargs="*", help="render preview JPEGs at these times instead")
    ap.add_argument("--stills-dir", type=Path, default=HERE / "stills")
    args = ap.parse_args()
    if args.stills is not None:
        stills(args.stills, args.stills_dir)
        return
    audio = None
    if not args.no_audio:
        from soundtrack import synthesize

        audio = HERE / "soundtrack.wav"
        synthesize(audio)
    render(args.out, args.fps, audio)


if __name__ == "__main__":
    main()
