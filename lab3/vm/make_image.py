import struct
import zlib
import math

W = H = 512
NOISE = 3
SAMPLES = 64

BG = (35, 35, 55)
ELLIPSE = (210, 70, 60)
RECT = (70, 190, 90)
TRI = (230, 200, 70)
LINE = (240, 240, 240)

COLORS = [BG, ELLIPSE, RECT, TRI, LINE]


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


def shape_at(x, y):
    if in_ellipse(x, y):
        return 1
    if in_rect(x, y):
        return 2
    if in_triangle(x, y):
        return 3
    if on_line(x, y):
        return 4
    return 0


def noise(x, y, ch):
    h = (x * 374761393 + y * 668265263 + ch * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h >> 16) % (2 * NOISE + 1)) - NOISE


def clamp(v):
    return 0 if v < 0 else (255 if v > 255 else v)


rows = []
for y in range(H):
    row = bytearray(3 * W)
    for x in range(W):
        r, g, b = COLORS[shape_at(x, y)]
        row[3 * x] = clamp(r + noise(x, y, 0))
        row[3 * x + 1] = clamp(g + noise(x, y, 1))
        row[3 * x + 2] = clamp(b + noise(x, y, 2))
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


def class_samples(k, count):
    pts = []
    st = (987654321 + k * 111111) & 0xFFFFFFFF
    while len(pts) < count:
        st = (st * 1103515245 + 12345) & 0xFFFFFFFF
        x = (st >> 16) % W
        st = (st * 1103515245 + 12345) & 0xFFFFFFFF
        y = (st >> 16) % H
        if shape_at(x, y) == k:
            pts.append((x, y))
    return pts


write_png('original.png', rows)

with open('in.data', 'wb') as f:
    f.write(struct.pack('<ii', W, H))
    for row in rows:
        for i in range(0, len(row), 3):
            f.write(bytes((row[i], row[i + 1], row[i + 2], 0)))

with open('classes.txt', 'w') as f:
    f.write('in.data\nout.data\n%d\n' % len(COLORS))
    for k in range(len(COLORS)):
        pts = class_samples(k, SAMPLES)
        f.write('%d %s\n' % (len(pts), ' '.join('%d %d' % p for p in pts)))

print('original.png, in.data и classes.txt готовы:', W, 'x', H, ', классов:', len(COLORS),
      ', точек в выборке каждого класса:', SAMPLES)
