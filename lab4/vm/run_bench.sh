#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

NVCC="${NVCC:-nvcc}"
if ! command -v "$NVCC" >/dev/null 2>&1 && [ -x /usr/bin/nvcc ]; then
    NVCC=/usr/bin/nvcc
fi

mkdir -p results

echo "=== 1/6: компиляция ==="
"$NVCC" query.cu -o query
"$NVCC" -Werror cross-execution-space-call -lm lab4.cu -o lab4 2>/dev/null \
  || "$NVCC" --Werror cross-execution-space-call -lm lab4.cu -o lab4 \
  || "$NVCC" lab4.cu -lm -o lab4
"$NVCC" -O2 -Xcompiler -fopenmp -lm bench4.cu -o bench4 2>/dev/null \
  || "$NVCC" -O2 -lm bench4.cu -o bench4 \
  || "$NVCC" -lm bench4.cu -o bench4

echo "=== 2/6: параметры GPU и версии ==="
./query | tee results/query_out.txt
"$NVCC" --version | tee results/nvcc_version.txt
nvidia-smi | tee results/nvidia_smi.txt || true

echo "=== 3/6: генерация входов для nvprof (n=1024, n=3000) ==="
python3 gen_input.py 1024 input_n1024.txt
python3 gen_input.py 3000 input_n3000.txt

echo "=== 4/6: nvprof на боевом lab4 ==="
if command -v nvprof >/dev/null 2>&1; then
  nvprof --log-file results/nvprof_n1024.txt ./lab4 < input_n1024.txt \
    > results/det_n1024.txt 2> results/err_n1024.txt || true
  nvprof --log-file results/nvprof_n3000.txt ./lab4 < input_n3000.txt \
    > results/det_n3000.txt 2> results/err_n3000.txt || true
elif command -v nsys >/dev/null 2>&1; then
  echo "nvprof не найден, использую nsys" | tee results/nvprof_note.txt
  nsys profile --stats=true -o nsys_n1024 \
    sh -c './lab4 < input_n1024.txt > results/det_n1024.txt' \
    > results/nvprof_n1024.txt 2>&1 || true
  nsys profile --stats=true -o nsys_n3000 \
    sh -c './lab4 < input_n3000.txt > results/det_n3000.txt' \
    > results/nvprof_n3000.txt 2>&1 || true
else
  echo "nvprof и nsys не найдены" | tee results/nvprof_note.txt
fi

echo "=== 5/6: бенчмарк (~2-3 минуты; прогресс пишется построчно) ==="
timeout 900 ./bench4 | tee results/bench4_out.txt || true

echo "=== 6/6: таблицы для отчета ==="
python3 render.py | tee results/report_tables.md

rm -f input_n1024.txt input_n3000.txt
tar czf results.tar.gz results
echo
echo "Готово. Результаты: results.tar.gz и папка results/"
ls -la results
