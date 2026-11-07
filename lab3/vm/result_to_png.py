import struct
import zlib
import colorsys

PALETTE = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / 32.0, 0.85, 0.95))
           for i in range(32)]


def read_img(path):
    data = open(path, 'rb').read()
    w, h = struct.unpack('<ii', data[:8])
    px = data[8:8 + w * h * 4]
    if len(px) != w * h * 4:
        raise SystemExit('ERROR: out.data has wrong size')
    return w, h, px


def write_png(path, w, h, rows):
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

    raw = bytearray()
    for row in rows:
        raw.append(0)
        raw.extend(row)
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(chunk(b'IHDR', ihdr))
        f.write(chunk(b'IDAT', zlib.compress(bytes(raw), 9)))
        f.write(chunk(b'IEND', b''))


w, h, px = read_img('out.data')
rows = []
for y in range(h):
    row = bytearray(3 * w)
    for x in range(w):
        cls = px[(y * w + x) * 4 + 3]
        r, g, b = PALETTE[cls % len(PALETTE)]
        row[3 * x] = r
        row[3 * x + 1] = g
        row[3 * x + 2] = b
    rows.append(bytes(row))
write_png('result.png', w, h, rows)
print('result.png готов:', w, 'x', h, '(альфа-канал = номер класса, показан цветами палитры)')
