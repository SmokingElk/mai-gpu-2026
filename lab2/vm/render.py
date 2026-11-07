N1, N2, N3 = 262144, 16777216, 100000000
order = [(1, 32), (1, 64), (1, 128), (1, 256), (1, 512), (1, 1024),
         (4, 256), (16, 256), (64, 256), (256, 256), (1024, 256), (1024, 1024)]

gpu = {}
geom = {}
cpu = {}
for line in open('bench2_out.txt'):
    p = line.split()
    if not p:
        continue
    if p[0] == 'BENCH':
        b, t = int(p[1]), int(p[2])
        geom[(b, t)] = (int(p[3]), int(p[4]), int(p[5]), int(p[6]))
        gpu.setdefault((b, t), {})[int(p[7])] = float(p[8])
    elif p[0] == 'CPU':
        cpu[int(p[1])] = float(p[2])

verifies = [l.split() for l in open('bench2_out.txt') if l.startswith('VERIFY')]
print('проверок с эталоном:', len(verifies),
      '| все совпадения в допуске (maxdiff <= 1):', all(int(v[5]) <= 1 for v in verifies))

assert set(gpu.keys()) == set(order), 'bench output incomplete'
assert set(cpu.keys()) == {N1, N2, N3}, 'cpu output incomplete'


def fmt_us(ms):
    return ('%.1f' % (ms * 1000.0)).replace('.', ',')


def fmt_ms(ms):
    return ('%.2f' % ms).replace('.', ',')


def fmt_threads(v):
    return str(v) if v < 1000 else ('%d %03d' % (v // 1000, v % 1000) if v < 1000000
                                    else '%d %03d %03d' % (v // 1000000, (v // 1000) % 1000, v % 1000))


print()
print('| Конфигурация (блоки × нити) | Всего нитей | Малый 512×512, мкс | Средний 4096×4096, мс | Предельный 10000×10000, мс |')
print('|---|---|---|---|---|')
for (b, t) in order:
    gx, gy, bx, by = geom[(b, t)]
    d = gpu[(b, t)]
    print('| <<<%d, %d>>> (%d×%d; %d×%d) | %s | %s | %s | %s |' % (
        b, t, gx, gy, bx, by, fmt_threads(b * t),
        fmt_us(d[N1]), fmt_ms(d[N2]), fmt_ms(d[N3])))
print('| CPU (1 поток) | 1 поток | %s | %s | %s |' % (fmt_us(cpu[N1]), fmt_ms(cpu[N2]), fmt_ms(cpu[N3])))

print()
bytes_eff = 8 * N3
best_cfg = min(order, key=lambda k: gpu[k][N3])
best_ms = gpu[best_cfg][N3]
work_ms = gpu[(1024, 256)][N3]
print('Пропускная способность на предельном наборе (эффективно 4 байта чтения + 4 байта записи на пиксель, 8*n = %d МБ;' % (bytes_eff // 1000000))
print('четырёхкратное перекрытие чтений соседей поглощается кэшами L1/L2):')
print('  лучшая конфигурация <<<%d, %d>>>: %.2f мс -> %.1f ГБ/с' % (best_cfg[0], best_cfg[1], best_ms, bytes_eff / (best_ms * 1e-3) / 1e9))
print('  рабочая конфигурация  <<<1024, 256>>> (32×32; 16×16): %.2f мс -> %.1f ГБ/с' % (work_ms, bytes_eff / (work_ms * 1e-3) / 1e9))
for name, nn in [('512x512', N1), ('4096x4096', N2), ('10000x10000', N3)]:
    print('CPU на %s: %.2f мс; ускорение лучшей конфигурации: %.1fx, рабочей: %.1fx' % (
        name, cpu[nn], cpu[nn] / gpu[best_cfg][nn], cpu[nn] / gpu[(1024, 256)][nn]))
best_small = min(order, key=lambda k: gpu[k][N1])
print('Лучшее время на малом наборе: %.1f мкс (<<<%d, %d>>>)' % (gpu[best_small][N1] * 1000.0,
                                                                best_small[0], best_small[1]))
