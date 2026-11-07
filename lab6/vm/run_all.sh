#!/usr/bin/env bash
# Оркестратор: сборка lab6 на ВМ с GPU и прогон 4 сценариев для скриншотов отчета.
# Использование (на ВМ): bash run_all.sh
set -e
cd "$(dirname "$0")"

echo '== компиляция =='
nvcc -Werror cross-execution-space-call -lm -lGL -o lab6 lab6.cu

mkdir -p results

run() {
    name="$1"
    script="$2"
    echo "== $name : $script"
    printf '%s\n' "$script" | ./lab6 > "results/$name.data"
    python3 result_to_png.py "results/$name.data" "results/$name.png"
}

# 1. общий вид: система стабилизировалась, частицы лежат на полу
run shot1_view   'time,10'

# 2. движение камеры внутрь куба: частицы "убегают" от одноименного заряда игрока
run shot2_move   'time,10;forward,35;time,5'

# 3. поворот камеры (yaw параллельно полу, pitch вверх)
run shot3_rotate 'time,10;yaw,25;pitch,8;time,1'

# 4. выстрел: прицел в слой частиц у пола (pitch до -25), пуля на 4-й секунде
#    проходит через центр куба на z~1 и расталкивает частицы
run shot4_shoot  'time,10;pitch,-16;shoot,1;time,4'

echo '== готово =='
ls -l results
