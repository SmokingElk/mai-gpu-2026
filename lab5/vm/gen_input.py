#!/usr/bin/env python3
"""Генерация бинарного входа для lab5: 4 байта LE (int32 n), затем n чисел int32."""
import array
import os
import struct
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: gen_input.py N [output.bin]")
    n = int(sys.argv[1])
    if n < 0:
        raise SystemExit("N must be >= 0")
    out = sys.argv[2] if len(sys.argv) > 2 else ("input_n%d.bin" % n)

    a = array.array("i")
    if a.itemsize != 4:
        raise SystemExit("unexpected array itemsize: %d" % a.itemsize)
    with open(out, "wb") as f:
        f.write(struct.pack("<i", n))
        if n > 0:
            a.frombytes(os.urandom(4 * n))
            a.tofile(f)
    print("written:", out, "n =", n)


main()
