#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

mkdir -p results

echo "=== 1/5: компиляция ==="
nvcc query.cu -o query
nvcc -Werror cross-execution-space-call -lm lab3.cu -o lab3 2>/dev/null \
  || nvcc --Werror cross-execution-space-call -lm lab3.cu -o lab3 \
  || nvcc lab3.cu -lm -o lab3
nvcc bench3.cu -lm -o bench3

echo "=== 2/5: параметры GPU ==="
./query | tee results/query_out.txt

echo "=== 3/5: бенчмарк (несколько минут, CPU-эталон на 10000x10000 самый долгий) ==="
./bench3 | tee results/bench3_out.txt

echo "=== 4/5: тестовое изображение и классификация ==="
python3 make_image.py
mv original.png results/original.png
./lab3 < classes.txt
python3 result_to_png.py
mv result.png results/result.png

echo "=== 5/5: таблица для отчета ==="
python3 render.py | tee results/table.md

echo
echo "Готово. Результаты в ./results:"
ls -la results
