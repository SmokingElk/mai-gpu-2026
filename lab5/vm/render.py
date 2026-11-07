#!/usr/bin/env python3
import os
import re

RES = "results"
SWEEP_N = 1 << 22
WORK = (512, 256, 512)  # stageGrid, stageBlock, mergeGrid
MAIN_SIZES = [1 << 10, 8192, 1 << 16, 1 << 18, 1 << 20, SWEEP_N, 1 << 24, 1 << 26]
NVPROF = [(4096, "n=4096 (одноблочный smem-путь)"), (1 << 20, "n=2^20"), (1 << 24, "n=2^24")]


def ru(v, dec=2):
    return ("%.*f" % (dec, v)).replace(".", ",")


def fmt_n(n):
    v, p = n, 0
    while v > 1 and v % 2 == 0:
        v //= 2
        p += 1
    if v == 1 and p > 0:
        return "%d (2^%d)" % (n, p)
    return str(n)


bench = []
cpu = {}
verify = []
copyms = {}

path = os.path.join(RES, "bench5_out.txt")
if os.path.exists(path):
    for line in open(path, errors="replace"):
        p = line.split()
        if not p or p[0].startswith("#"):
            continue
        if p[0] == "BENCH" and len(p) == 8:
            bench.append((p[1], (int(p[2]), int(p[3]), int(p[4])), int(p[5]),
                          float(p[6]), float(p[7])))
        elif p[0] == "CPU":
            cpu[int(p[1])] = float(p[2])
        elif p[0] == "VERIFY":
            verify.append((int(p[1]), int(p[2])))
        elif p[0] == "COPY":
            copyms[int(p[1])] = (float(p[2]), float(p[3]))

main = {}
for tag, key, n, ev, wall in bench:
    if tag == "MAIN":
        main[n] = (ev, wall)

print("## 1. Время битонической сортировки от размера (рабочая конфигурация)")
print()
if main:
    print("| Размер n | m | Путь | GPU, мс (event) | GPU, мс (wall) | CPU std::sort, мс | "
          "Ускорение | H2D+D2H, мс |")
    print("|---|---|---|---|---|---|---|---|")
    for n in MAIN_SIZES:
        if n not in main:
            continue
        ev, wall = main[n]
        m = 1
        while m < n:
            m <<= 1
        p = "smem, одно ядро" if m <= 8192 else "адаптивный (global + smem)"
        cells = [fmt_n(n), fmt_n(m), p, ru(ev, 3), ru(wall, 3)]
        cells.append(ru(cpu[n], 3) if n in cpu else "—")
        cells.append(ru(cpu[n] / ev, 1) if n in cpu else "—")
        cells.append(ru(copyms[n][0] + copyms[n][1], 3) if n in copyms else "—")
        print("| " + " | ".join(cells) + " |")
    print()
    print("Примечание: при m ≤ 8192 весь массив помещается в разделяемую память и "
          "сортируется одним блоком (ядро bitonicSortSmem); при больших m работают "
          "стадии с чанком > 8192 через глобальную память (bitonicStageGlobal) и "
          "адаптивное слияние в разделяемой памяти (bitonicMergeSmem).")
else:
    print("Данные бенчмарка отсутствуют (бенч прерван или не дал результатов).")
print()

stage_rows = [(key, ev) for tag, key, n, ev, wall in bench
              if tag == "SWEEP" and n == SWEEP_N and key[2] == WORK[2]]
merge_rows = [(key, ev) for tag, key, n, ev, wall in bench
              if tag == "SWEEP" and n == SWEEP_N
              and key[0] == WORK[0] and key[1] == WORK[1] and key[2] != WORK[2]]

print("## 2. Свип конфигураций stage-ядра (глобальные стадии), n=%s" % fmt_n(SWEEP_N))
print()
if stage_rows:
    stage_rows.sort(key=lambda r: r[1])
    print("| stage: grid × block | Всего нитей | GPU, мс (event) |")
    print("|---|---|---|")
    for (sg, sb, _), ev in stage_rows:
        print("| %d × %d | %d | %s |" % (sg, sb, sg * sb, ru(ev, 3)))
else:
    print("Данные свипа отсутствуют.")
print()

print("## 3. Свип конфигураций merge-ядра (разделяемая память), n=%s" % fmt_n(SWEEP_N))
print()
if merge_rows:
    merge_rows.sort(key=lambda r: r[1])
    print("| merge: grid × block | Всего нитей | GPU, мс (event) |")
    print("|---|---|---|")
    for (_, _, mg), ev in merge_rows:
        print("| %d × 1024 | %d | %s |" % (mg, mg * 1024, ru(ev, 3)))
else:
    print("Данные свипа отсутствуют.")
print()

print("## 4. Верификация")
print()
if verify:
    total = len(verify)
    bad = [n for n, ok in verify if not ok]
    if bad:
        print("- Бенчмарк: %d проверок, ПРОВАЛЫ на размерах: %s" % (total, bad))
    else:
        print("- Бенчмарк: %d проверок (memcmp с CPU-эталоном std::sort) — все OK." % total)
for n, label in NVPROF:
    f = os.path.join(RES, "verify_n%d.txt" % n)
    if os.path.exists(f):
        txt = open(f, errors="replace").read().strip()
        if txt.startswith("OK"):
            print("- Боевой lab5 под nvprof, %s: OK" % label)
        else:
            first = txt.splitlines()[0][:80] if txt else "(пусто)"
            print("- Боевой lab5 под nvprof, %s: FAIL — %s" % (label, first))
print()

TIME_RE = re.compile(r"^[\d.]+(?:s|ms|us|ns)$")
UNIT = {"s": 1000.0, "ms": 1.0, "us": 0.001, "ns": 0.000001}


def to_ms(tok):
    for u in ("ms", "us", "ns", "s"):
        if tok.endswith(u):
            return float(tok[:-len(u)]) * UNIT[u]
    return None


def parse_nvprof(path):
    if not os.path.exists(path):
        return None
    rows = []
    in_gpu = False
    for line in open(path, errors="replace"):
        if "GPU activities" in line or "GPU activity" in line:
            in_gpu = True
            idx = line.find(":")
            if idx >= 0:
                line = line[idx + 1:]
            else:
                continue
        if not in_gpu:
            continue
        p = line.split()
        if len(p) < 7 or not p[0].endswith("%"):
            if rows:
                break
            continue
        if not TIME_RE.match(p[1]):
            continue
        try:
            calls = int(p[2])
        except ValueError:
            continue
        if not (TIME_RE.match(p[3]) and TIME_RE.match(p[4]) and TIME_RE.match(p[5])):
            continue
        pct = float(p[0][:-1])
        total = to_ms(p[1])
        name = " ".join(p[6:])
        rows.append((pct, total, calls, name))
    return rows


print("## 5. nvprof: разбивка по операциям GPU (боевой lab5)")
print()
for n, label in NVPROF:
    rows = parse_nvprof(os.path.join(RES, "nvprof_n%d.txt" % n))
    print("### %s" % label)
    print()
    if not rows:
        print("Не удалось разобрать nvprof-лог (или nvprof недоступен) — см. nvprof_n%d.txt." % n)
        print()
        continue
    print("| Операция | Время, мс | Вызовов | Доля, % |")
    print("|---|---|---|---|")
    for pct, ms, calls, name in rows[:10]:
        if len(name) > 48:
            name = name[:45] + "..."
        print("| %s | %s | %d | %s |" % (name, ru(ms, 3), calls, ru(pct, 1)))
    print()

print("## 6. Параметры GPU (query.cu)")
print()
qf = os.path.join(RES, "query_out.txt")
if os.path.exists(qf):
    print("```")
    print(open(qf, errors="replace").read().rstrip())
    print("```")
else:
    print("query_out.txt не найден.")
