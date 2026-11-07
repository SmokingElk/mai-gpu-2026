#!/usr/bin/env python3
import os
import re

RES = "results"
SWEEP = [256, 1024, 3000]
LIMIT = 4096
ALL_SIZES = SWEEP + [LIMIT]
WORK = (16, 16, 32, 8)


def ru(v, dec=2):
    return ("%.*f" % (dec, v)).replace(".", ",")


def fmt_threads(v):
    if v < 1000:
        return str(v)
    if v < 1000000:
        return "%d %03d" % (v // 1000, v % 1000)
    return "%d %03d %03d" % (v // 1000000, (v // 1000) % 1000, v % 1000)


order = []
ev = {}
wall = {}
cpu = {}
cpu_threads = 1
verify = []
verifyu = []
uncoal = {}
dets = {}

for line in open(os.path.join(RES, "bench4_out.txt")):
    p = line.split()
    if not p or p[0].startswith("#"):
        continue
    if p[0] == "BENCH":
        key = (int(p[3]), int(p[4]), int(p[5]), int(p[6]))
        if key not in ev:
            order.append(key)
        ev.setdefault(key, {})[int(p[7])] = float(p[8])
        wall.setdefault(key, {})[int(p[7])] = float(p[9])
    elif p[0] == "CPU":
        cpu[int(p[1])] = float(p[2])
        cpu_threads = int(p[3])
    elif p[0] == "VERIFY":
        verify.append((int(p[1]), float(p[2])))
    elif p[0] == "VERIFYU":
        verifyu.append((int(p[1]), float(p[2])))
    elif p[0] == "UNCOAL":
        uncoal[int(p[1])] = float(p[2])
    elif p[0] == "DET":
        dets[int(p[1])] = (float(p[2]), float(p[3]))

bench_sizes = [n for n in ALL_SIZES
               if n in cpu or any(n in ev[k] for k in order)]

if not order:
    print("## Данные бенчмарка отсутствуют (бенч прерван или не дал результатов)")
    print()
else:
    missing = [n for n in ALL_SIZES if n not in bench_sizes]
    if missing:
        print("ВНИМАНИЕ: данные бенча неполны, отсутствуют размеры: %s" % missing)
        print()

    print("## 1. Время исключения Гаусса по конфигурациям (cudaEvent, мс)")
    print()
    header = "| Конфигурация (сетка; блок) | Всего нитей |"
    sep = "|---|---|"
    for n in bench_sizes:
        header += " n=%d, мс |" % n
        sep += "---|"
    print(header)
    print(sep)
    for key in order:
        gx, gy, bx, by = key
        cells = ["<<<%d, %d>>> (%d×%d; %d×%d)" % (gx * gy, bx * by, gx, gy, bx, by),
                 fmt_threads(gx * gy * bx * by)]
        for n in bench_sizes:
            if n in ev[key]:
                cells.append(ru(ev[key][n], 3))
            else:
                cells.append("—")
        print("| " + " | ".join(cells) + " |")
    cpu_cells = ["CPU (OpenMP, %d потоков)" % cpu_threads, fmt_threads(cpu_threads)]
    for n in bench_sizes:
        cpu_cells.append(ru(cpu[n], 3) if n in cpu else "—")
    print("| " + " | ".join(cpu_cells) + " |")

    print()
    print("## 2. Ускорения относительно CPU")
    print()
    for n in bench_sizes:
        if n not in cpu:
            continue
        avail = [k for k in order if n in ev[k]]
        if not avail:
            continue
        best = min(avail, key=lambda k: ev[k][n])
        print("- n=%d: лучшая <<<%d, %d>>> (%d×%d; %d×%d) — %s мс (ускорение %s×); "
              "рабочая <<<%d, %d>>> — %s мс (ускорение %s×); CPU — %s мс" % (
                  n, best[0] * best[1], best[2] * best[3], best[0], best[1], best[2], best[3],
                  ru(ev[best][n], 3), ru(cpu[n] / ev[best][n], 1),
                  WORK[0] * WORK[1], WORK[2] * WORK[3],
                  ru(ev[WORK][n], 3) if n in ev[WORK] else "—",
                  ru(cpu[n] / ev[WORK][n], 1) if n in ev[WORK] else "—",
                  ru(cpu[n], 3)))

    coal_sizes = [n for n in SWEEP if n in uncoal and n in ev.get(WORK, {})]
    if coal_sizes:
        print()
        print("## 3. Влияние объединения запросов к глобальной памяти (рабочая конфигурация)")
        print()
        print("| Размер | Coalesced, мс | Uncoalesced (x→j), мс | Замедление |")
        print("|---|---|---|---|")
        for n in coal_sizes:
            c = ev[WORK][n]
            u = uncoal[n]
            print("| n=%d | %s | %s | %s× |" % (n, ru(c, 3), ru(u, 3), ru(u / c, 1)))

    if verify or verifyu:
        print()
        print("## 4. Верификация")
        print()
        allv = verify + verifyu
        mx = max(d for _, d in allv)
        print("Проверок GPU против CPU-эталона: %d (включая uncoalesced-вариант: %d). "
              "Максимальное относительное расхождение: %s (алгоритм один и тот же, ожидается ~0)." % (
                  len(allv), len(verifyu), ("%.1e" % mx).replace(".", ",")))

print()
print("## 5. Сквозная проверка боевого lab4 (запуски под nvprof)")
print()
for n in (1024, 3000):
    f = os.path.join(RES, "det_n%d.txt" % n)
    if not os.path.exists(f):
        print("- n=%d: файл det_n%d.txt не найден" % (n, n))
        continue
    txt = open(f).read().split()
    if not txt or txt[0] == "ERROR:":
        print("- n=%d: lab4 завершился с ошибкой, см. err_n%d.txt" % (n, n))
        continue
    val = float(txt[0])
    if n in dets:
        ref = dets[n][0]
        rd = abs(val - ref) / max(abs(val), abs(ref), 1e-300)
        print("- n=%d: det(lab4) = %.15e, det(CPU-эталон бенча) = %.15e, относительное расхождение %s" % (
            n, val, ref, ("%.1e" % rd).replace(".", ",")))
    else:
        print("- n=%d: det(lab4) = %.15e (CPU-эталон из бенча недоступен)" % (n, val))

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


print()
print("## 6. nvprof: разбивка по операциям GPU (боевой lab4)")
print()
for n in (1024, 3000):
    rows = parse_nvprof(os.path.join(RES, "nvprof_n%d.txt" % n))
    print("### n=%d" % n)
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
