"""Tiny motion-graphics engine: Pillow frames piped into ffmpeg.

Scenes are pure functions ``draw(c: Canvas, t: float)`` of time, so any frame
can be rendered independently. Frames are drawn at 2x and downsampled for
antialiasing.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, SS, FPS = 960, 540, 2, 30

BG = (13, 17, 28)
PANEL = (24, 30, 46)
GRID = (32, 40, 60)
TEXT = (222, 228, 238)
MUTED = (132, 144, 166)
JEV = (90, 150, 255)
XGB = (255, 160, 67)
META = (167, 139, 250)
EMB = (45, 212, 191)
OK = (52, 211, 120)
BAD = (255, 92, 92)
WARN = (250, 204, 21)

_FONT_CANDIDATES = {
    "regular": [
        "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "bold": [
        "/usr/share/fonts/opentype/inter/Inter-Bold.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "mono": ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
}


@cache
def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES[weight]:
        if Path(path).exists():
            return ImageFont.truetype(path, size * SS)
    return ImageFont.load_default(size * SS)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def ease(x: float) -> float:
    x = clamp(x)
    return x * x * (3 - 2 * x)


def seg(t: float, a: float, b: float) -> float:
    """Eased progress of ``t`` through [a, b]."""
    return ease((t - a) / (b - a)) if b > a else float(t >= a)


def lerp(a, b, k: float):
    if isinstance(a, tuple):
        return tuple(lerp(x, y, k) for x, y in zip(a, b, strict=True))
    return a + (b - a) * k


def mix(color, k: float, base=BG):
    """Fade ``color`` toward ``base``; k=1 is the full color."""
    return tuple(int(round(b + (c - b) * clamp(k))) for c, b in zip(color, base, strict=True))


def window(t: float, a: float, b: float, fade: float = 0.35) -> float:
    """1 inside [a, b] with fades at both ends; 0 outside."""
    return min(seg(t, a, a + fade), 1 - seg(t, b - fade, b))


@dataclass
class Canvas:
    img: Image.Image
    d: ImageDraw.ImageDraw

    @staticmethod
    def _s(v):
        return v * SS

    def rect(self, x, y, w, h, fill=None, outline=None, width=2, r=10, k=1.0):
        if k <= 0:
            return
        self.d.rounded_rectangle(
            [x * SS, y * SS, (x + w) * SS, (y + h) * SS],
            radius=r * SS,
            fill=mix(fill, k) if fill else None,
            outline=mix(outline, k) if outline else None,
            width=int(width * SS),
        )

    def circle(self, x, y, r, fill=None, outline=None, width=2, k=1.0):
        if k <= 0 or r <= 0:
            return
        self.d.ellipse(
            [(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS],
            fill=mix(fill, k) if fill else None,
            outline=mix(outline, k) if outline else None,
            width=int(width * SS),
        )

    def line(self, pts, color, width=2, k=1.0):
        if k <= 0:
            return
        self.d.line(
            [(x * SS, y * SS) for x, y in pts],
            fill=mix(color, k),
            width=int(width * SS),
            joint="curve",
        )

    def arrow(self, p0, p1, color, progress=1.0, width=2.5, k=1.0, head=9):
        if progress <= 0 or k <= 0:
            return
        x = lerp(p0[0], p1[0], progress)
        y = lerp(p0[1], p1[1], progress)
        self.line([p0, (x, y)], color, width, k)
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        if progress > 0.95:
            pts = [
                (x, y),
                (x - head * math.cos(ang - 0.45), y - head * math.sin(ang - 0.45)),
                (x - head * math.cos(ang + 0.45), y - head * math.sin(ang + 0.45)),
            ]
            self.d.polygon([(a * SS, b * SS) for a, b in pts], fill=mix(color, k))

    def path_point(self, pts, k):
        """Point at fraction k along a polyline."""
        lens = [math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False)]
        total = sum(lens) or 1
        dist = clamp(k) * total
        for (a, b), L in zip(zip(pts, pts[1:], strict=False), lens, strict=True):
            if dist <= L:
                return lerp(a, b, dist / L if L else 0)
            dist -= L
        return pts[-1]

    def text(self, x, y, s, size=18, color=TEXT, weight="regular", anchor="la", k=1.0):
        if k <= 0:
            return
        self.d.text((x * SS, y * SS), s, font=font(size, weight), fill=mix(color, k), anchor=anchor)

    def text_width(self, s, size=18, weight="regular") -> float:
        return self.d.textlength(s, font=font(size, weight)) / SS

    def bar(self, x, y, w, h, value, color, k=1.0, label=None, track=GRID):
        self.rect(x, y, w, h, fill=track, r=h / 2, k=k)
        if value > 0.001:
            self.rect(x, y, max(h, w * clamp(value)), h, fill=color, r=h / 2, k=k)
        if label:
            self.text(x - 10, y + h / 2, label, 14, MUTED, anchor="rm", k=k)

    def gauge(self, cx, cy, r, value, color, label, k=1.0):
        self.d.arc(
            [(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS],
            180,
            360,
            fill=mix(GRID, k),
            width=int(12 * SS),
        )
        if value > 0.002:
            self.d.arc(
                [(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS],
                180,
                180 + 180 * clamp(value),
                fill=mix(color, k),
                width=int(12 * SS),
            )
        self.text(cx, cy - 8, f"{value:.2f}", 26, TEXT, "bold", "mb", k)
        self.text(cx, cy + 14, label, 14, MUTED, anchor="mt", k=k)

    def chip(self, x, y, s, color, k=1.0, size=15, anchor="mm"):
        w = self.text_width(s, size, "bold") + 22
        h = size + 14
        x0 = x - w / 2 if anchor == "mm" else x
        self.rect(
            x0, y - h / 2, w, h, fill=mix(color, 0.22), outline=color, width=1.5, r=h / 2, k=k
        )
        self.text(x0 + w / 2, y, s, size, color, "bold", "mm", k)
        return w


def header(c: Canvas, t: float, title: str, subtitle: str, step: str):
    k = seg(t, 0, 0.6)
    c.text(36, 30, step.upper(), 13, MUTED, "bold", k=k)
    c.text(36, 50, title, 28, TEXT, "bold", k=k)
    c.text(36, 88, subtitle, 16, MUTED, k=k)


def caption(c: Canvas, t: float, items: list[tuple[float, float, str]]):
    """Bottom caption track: (start, end, text)."""
    for a, b, s in items:
        k = window(t, a, b)
        if k > 0:
            c.text(W / 2, H - 34, s, 17, TEXT, anchor="mm", k=k)


def render(
    scene,
    duration: float,
    out_mp4: Path,
    gif: Path | None = None,
    gif_width: int = 720,
    gif_fps: int = 15,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to render animations")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ],
        stdin=subprocess.PIPE,
    )
    n = int(duration * FPS)
    for i in range(n):
        img = Image.new("RGB", (W * SS, H * SS), BG)
        scene(Canvas(img, ImageDraw.Draw(img)), i / FPS)
        proc.stdin.write(img.resize((W, H), Image.LANCZOS).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_mp4}")
    if gif:
        vf = (
            f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];"
            "[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(out_mp4),
                "-vf",
                vf,
                "-loop",
                "0",
                str(gif),
            ],
            check=True,
        )
