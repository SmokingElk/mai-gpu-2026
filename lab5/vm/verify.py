#!/usr/bin/env python3
"""Проверка выхода lab5: длина совпадает с n, выход отсортирован и равен sorted(вход)."""
import array
import struct
import sys


def read_pair(path):
    with open(path, "rb") as f:
        raw_n = f.read(4)
        if len(raw_n) != 4:
            raise EOFError("no 4-byte header in %s" % path)
        n = struct.unpack("<i", raw_n)[0]
        if n < 0:
            raise ValueError("negative n in %s" % path)
        a = array.array("i")
        if n > 0:
            a.fromfile(f, n)
    return n, a


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: verify.py input.bin output.bin")
    try:
        n, inp = read_pair(sys.argv[1])
        out = array.array("i")
        with open(sys.argv[2], "rb") as f:
            out.fromfile(f, n)
    except (EOFError, OSError, ValueError) as e:
        print("FAIL: %s" % e)
        return 1
    if len(inp) != n or len(out) != n:
        print("FAIL: expected %d numbers, got input=%d output=%d" % (n, len(inp), len(out)))
        return 1
    if n == 0:
        print("OK: n=0, пустой выход")
        return 0
    ref = sorted(inp)
    for i in range(n):
        if out[i] != ref[i]:
            print("FAIL: первое расхождение в индексе %d: получено %d, ожидалось %d" % (i, out[i], ref[i]))
            return 1
    print("OK: n=%d, выход отсортирован и совпадает с sorted(вход)" % n)
    return 0


sys.exit(main())
