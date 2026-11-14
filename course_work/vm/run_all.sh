#!/usr/bin/env bash
# Оркестратор: сборка курсовой (kp) на ВМ с GPU и прогон всех серий
# для материалов отчёта: анимация, wireframe, hi-res, 4 ракурса,
# сравнение GPU vs CPU. Использование (на ВМ): bash run_all.sh
set -e
cd "$(dirname "$0")"

ANIM_FRAMES=240   # должно совпадать с anim.txt
WIRE_FRAMES=120   # должно совпадать с wire.txt

echo '== компиляция =='
nvcc -Werror cross-execution-space-call -lm -o kp kp.cu

mkdir -p results/anim results/anim_png results/wire results/wire_png \
         results/hires results/still1 results/still2 results/still3 \
         results/still4 results/bench results/logs

echo '== конфигурация по умолчанию (--default) =='
./kp --default > results/default_config.txt 2> results/logs/default.txt

echo '== анимация (GPU, 240 кадров 960x540 ssaa=2) =='
./kp --gpu < anim.txt > results/anim_stats.txt 2> results/logs/anim.txt
for i in $(seq 0 $((ANIM_FRAMES - 1))); do
    printf -v p "results/anim_png/img_%04d.png" "$i"
    python3 result_to_png.py "results/anim/img_$i.data" "$p"
done

echo '== wireframe-анимация (GPU --wireframe, 120 кадров) =='
./kp --gpu --wireframe < wire.txt > results/wire_stats.txt 2> results/logs/wire.txt
for i in $(seq 0 $((WIRE_FRAMES - 1))); do
    printf -v p "results/wire_png/img_%04d.png" "$i"
    python3 result_to_png.py "results/wire/img_$i.data" "$p"
done

echo '== hi-res рендер (3840x2160 ssaa=2) =='
./kp --gpu < hires.txt > results/hires_stats.txt 2> results/logs/hires.txt
python3 result_to_png.py results/hires/img_0.data results/hires.png

echo '== ракурсы (4 шт, 1280x720 ssaa=3) =='
for n in 1 2 3 4; do
    ./kp --gpu < still$n.txt > "results/still${n}_stats.txt" 2> "results/logs/still$n.txt"
    python3 result_to_png.py "results/still$n/img_0.data" "results/still$n.png"
done

echo '== бенчмарк GPU vs CPU (12 кадров 640x360) =='
./kp --cpu < bench.txt > results/bench_cpu.txt 2> results/logs/bench_cpu.txt
./kp --gpu < bench.txt > results/bench_gpu.txt 2> results/logs/bench_gpu.txt

echo '== сборка видео =='
if command -v ffmpeg > /dev/null 2>&1; then
    ffmpeg -y -loglevel error -framerate 24 -i results/anim_png/img_%04d.png \
        -c:v libx264 -pix_fmt yuv420p -crf 18 results/animation.mp4 || \
    ffmpeg -y -loglevel error -framerate 24 -i results/anim_png/img_%04d.png \
        -c:v mpeg4 -qscale:v 4 results/animation.mp4
    ffmpeg -y -loglevel error -framerate 24 -i results/wire_png/img_%04d.png \
        -c:v libx264 -pix_fmt yuv420p -crf 18 results/wire.mp4 || \
    ffmpeg -y -loglevel error -framerate 24 -i results/wire_png/img_%04d.png \
        -c:v mpeg4 -qscale:v 4 results/wire.mp4
    echo 'видео: results/animation.mp4, results/wire.mp4'
else
    echo 'WARN: ffmpeg не найден — видео собрать локально (см. README.md)'
fi

echo '== чистка сырых кадров (PNG уже построены) =='
# .data-кадры занимают сотни МБ и после конвертации в PNG не нужны;
# статистика и логи остаются в results/*.txt и results/logs/
find results -name 'img_*.data' -delete

echo '== упаковка результатов =='
tar -czf results.tar.gz results
echo "архив: $(pwd)/results.tar.gz ($(du -h results.tar.gz | cut -f1))"

echo '== готово =='
ls -l results
