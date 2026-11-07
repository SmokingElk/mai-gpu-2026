import struct
import zlib


def read_img(path):
    data = open(path, 'rb').read()
    w, h = struct.unpack('<ii', data[:8])
    px = data[8:8 + w * h * 4]
    if len(px) != w * h * 4:
        raise SystemExit('ERROR: out.data has wrong size')
    return w, h, px


def write_png(path, w, h, px):
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

    raw = bytearray()
    for y in range(h):
        raw.append(0)
        base = y * w * 4
        row = bytearray(3 * w)
        for x in range(w):
            i = base + 4 * x
            row[3 * x] = px[i]
            row[3 * x + 1] = px[i + 1]
            row[3 * x + 2] = px[i + 2]
        raw.extend(row)
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(chunk(b'IHDR', ihdr))
        f.write(chunk(b'IDAT', zlib.compress(bytes(raw), 9)))
        f.write(chunk(b'IEND', b''))


w, h, px = read_img('out.data')
write_png('result.png', w, h, px)
print('result.png готов:', w, 'x', h)
