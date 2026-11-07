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
"$NVCC" -Werror cross-execution-space-call -lm lab5.cu -o lab5 2>/dev/null \
  || "$NVCC" --Werror cross-execution-space-call -lm lab5.cu -o lab5 \
  || "$NVCC" lab5.cu -lm -o lab5
"$NVCC" -O2 bench5.cu -o bench5

echo "=== 2/6: параметры GPU и версии ==="
./query | tee results/query_out.txt
"$NVCC" --version | tee results/nvcc_version.txt
nvidia-smi | tee results/nvidia_smi.txt || true

echo "=== 3/6: генерация входов для nvprof (n=4096, 2^20, 2^24) ==="
python3 gen_input.py 4096 input_n4096.bin
python3 gen_input.py 1048576 input_n1048576.bin
python3 gen_input.py 16777216 input_n16777216.bin

echo "=== 4/6: nvprof на боевом lab5 + проверка результатов ==="
if command -v nvprof >/dev/null 2>&1; then
  for n in 4096 1048576 16777216; do
    nvprof --log-file results/nvprof_n${n}.txt ./lab5 < input_n${n}.bin \
      > sorted_n${n}.bin 2> results/err_n${n}.txt || true
    python3 verify.py input_n${n}.bin sorted_n${n}.bin | tee results/verify_n${n}.txt
  done
elif command -v nsys >/dev/null 2>&1; then
  echo "nvprof не найден, использую nsys" | tee results/nvprof_note.txt
  for n in 4096 1048576 16777216; do
    nsys profile --stats=true -o nsys_n${n} \
      sh -c "./lab5 < input_n${n}.bin > sorted_n${n}.bin" \
      > results/nvprof_n${n}.txt 2>&1 || true
    python3 verify.py input_n${n}.bin sorted_n${n}.bin | tee results/verify_n${n}.txt
  done
else
  echo "nvprof и nsys не найдены" | tee results/nvprof_note.txt
fi

echo "=== 5/6: бенчмарк bench5 (~1-2 минуты; прогресс пишется построчно) ==="
timeout 1200 ./bench5 | tee results/bench5_out.txt || true

echo "=== 6/6: таблицы для отчета ==="
python3 render.py | tee results/report_tables.md

rm -f input_n4096.bin input_n1048576.bin input_n16777216.bin
rm -f sorted_n4096.bin sorted_n1048576.bin sorted_n16777216.bin
tar czf results.tar.gz results
echo
echo "Готово. Результаты: results.tar.gz и папка results/"
ls -la results
