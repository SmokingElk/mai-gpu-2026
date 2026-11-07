#!/usr/bin/env python3
import math
import sys

MASK = (1 << 64) - 1
MUL = 6364136223846793005
ADD = 1442695040888963407
SEED = 42
INV53 = 1.0 / 9007199254740992.0


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: gen_input.py N [output.txt]")
    n = int(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else ("input_n%d.txt" % n)

    state = SEED
    S = math.sqrt(12.0 * 2.718281828459045 / n)
    with open(out, "w") as f:
        f.write("%d\n" % n)
        for i in range(n):
            row = []
            ap = row.append
            for j in range(n):
                state = (state * MUL + ADD) & MASK
                u = (state >> 11) * INV53
                ap(repr((u - 0.5) * S))
            f.write(" ".join(row))
            f.write("\n")
    print("written:", out, "n =", n)


main()
