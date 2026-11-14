#!/usr/bin/env python3
# Конвертация floor_texture.png (на самом деле JPEG) в floor.data:
# формат ЛР2 — 8 байт little-endian (ширина, высота) + пиксели RGBA.
# Использование: python3 make_floor.py [вход] [выход]
# Требуется Pillow (pip install pillow); вход открывается по содержимому.
import struct
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "floor_texture.png"
dst = sys.argv[2] if len(sys.argv) > 2 else "floor.data"

from PIL import Image

img = Image.open(src).convert("RGBA")
w, h = img.size
if w < 1 or h < 1 or w > 16384 or h > 16384:
    raise SystemExit(f"ERROR: invalid texture size {w}x{h}")

with open(dst, "wb") as f:
    f.write(struct.pack("<ii", w, h))
    f.write(img.tobytes())
print(f"ok: {dst} {w}x{h}")
