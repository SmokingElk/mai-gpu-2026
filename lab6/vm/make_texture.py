#!/usr/bin/env python3
# Генерирует текстуру сферы (шахматная доска 64x64 RGBA) и печатает ее в виде
# C-массива. Результат встраивается в lab6.cu вместо маркера:
#     /* __SPHERE_TEX_DATA__ */

TEX = 64
CELL = 8
C1 = (60, 120, 210)
C2 = (235, 235, 235)


def pixel(x, y):
    return C1 if ((x // CELL) + (y // CELL)) % 2 == 0 else C2


lines = []
lines.append('static const unsigned char sphereTexData[%d * %d * 4] = {' % (TEX, TEX))
for y in range(TEX):
    row = []
    for x in range(TEX):
        r, g, b = pixel(x, y)
        row.extend((r, g, b, 255))
    lines.append('    ' + ','.join(str(v) for v in row) + ',')
lines.append('};')
print('\n'.join(lines))
