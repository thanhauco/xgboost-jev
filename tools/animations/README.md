# Explainer animations

A small motion-graphics renderer (Pillow frames piped into ffmpeg) that produces the videos in [`docs/media/`](../../docs/media/). The gallery is in [`docs/algorithms.md`](../../docs/algorithms.md).

- `motion.py`: drawing primitives (cards, arrows, gauges, bars, chips), easing, and the ffmpeg encoder. Frames are drawn at 2× and downsampled for antialiasing. GIFs use a per-video 96-colour palette.
- `scenes.py`: one pure function `draw(canvas, t)` per algorithm. Numbers are read from `docs/benchmarks.json`.
- `render.py`: the CLI.

Requirements: `ffmpeg` on `PATH` and Pillow (`pip install pillow`). Fonts: Inter if installed, otherwise DejaVu Sans, otherwise Pillow's default. Rendering all seven scenes takes about a minute on four cores.
