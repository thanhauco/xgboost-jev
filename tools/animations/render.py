"""Render the explainer animations into docs/media/.

python tools/animations/render.py            # all scenes, MP4 + GIF
python tools/animations/render.py --only 03  # one scene
python tools/animations/render.py --still 7.5 --only 01   # one PNG frame for review
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from motion import BG, SS, Canvas, H, W, render  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scenes import SCENES  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "media"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--still", type=float, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args()
    for name, (scene, duration) in SCENES.items():
        if a.only and not name.startswith(a.only):
            continue
        if a.still is not None:
            img = Image.new("RGB", (W * SS, H * SS), BG)
            scene(Canvas(img, ImageDraw.Draw(img)), a.still)
            path = a.out / f"{name}_t{a.still:g}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            img.resize((W, H), Image.LANCZOS).save(path)
            print(path)
            continue
        mp4 = a.out / f"{name}.mp4"
        render(scene, duration, mp4, None if a.no_gif else a.out / f"{name}.gif")
        print(mp4)


if __name__ == "__main__":
    main()
