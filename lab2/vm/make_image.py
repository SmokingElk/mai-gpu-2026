import struct
import zlib
import math

W = H = 512

BG = (35, 35, 55)
ELLIPSE = (210, 70, 60)
RECT = (70, 190, 90)
TRI = (230, 200, 70)
LINE = (240, 240, 240)


def in_ellipse(x, y):
    ex = (x - 180) / 90.0
    ey = (y - 280) / 90.0
    return ex * ex + ey * ey <= 1.0


def in_rect(x, y):
    return 300 <= x <= 470 and 150 <= y <= 330


def in_triangle(x, y):
    ax, ay = 160.0, 460.0
    bx, by = 360.0, 460.0
    cx, cy = 260.0, 330.0
    d1 = (x - bx) * (ay - by) - (ax - bx) * (y - by)
    d2 = (x - cx) * (by - cy) - (bx - cx) * (y - cy)
    d3 = (x - ax) * (cy - ay) - (cx - ax) * (y - ay)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)


def on_line(x, y):
    if x < 18 or x > 502:
        return False
    return abs(x + y - 520) <= 2.5 * math.sqrt(2.0)


rows = []
for y in range(H):
    row = bytearray(3 * W)
    for x in range(W):
        if y < 120:
            t = y / 119.0
            r, g, b = int(40 + 180 * t), int(40 + 60 * t), int(60 + 20 * t)
        else:
            r, g, b = BG
        if in_ellipse(x, y):
            r, g, b = ELLIPSE
        if in_rect(x, y):
            r, g, b = RECT
        if in_triangle(x, y):
            r, g, b = TRI
        if on_line(x, y):
            r, g, b = LINE
        row[3 * x] = r
        row[3 * x + 1] = g
        row[3 * x + 2] = b
    rows.append(bytes(row))


def write_png(path, rows):
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

    raw = bytearray()
    for row in rows:
        raw.append(0)
        raw.extend(row)
    ihdr = struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(chunk(b'IHDR', ihdr))
        f.write(chunk(b'IDAT', zlib.compress(bytes(raw), 9)))
        f.write(chunk(b'IEND', b''))


write_png('original.png', rows)

with open('in.data', 'wb') as f:
    f.write(struct.pack('<ii', W, H))
    for row in rows:
        for i in range(0, len(row), 3):
            f.write(bytes((row[i], row[i + 1], row[i + 2], 255)))

print('original.png и in.data готовы:', W, 'x', H)
