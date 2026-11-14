# Запуск курсовой (kp, вариант 9) на ВМ с GPU

Содержимое:
- `kp.cu` — копия основного `../kp.cu` (при изменении основного файла перекопируй!)
- `floor.data` — текстура пола (формат ЛР2), сгенерирована из `floor_texture.png`
  скриптом `make_floor.py`
- `anim.txt` — анимация: 240 кадров, 960x540, ssaa=2 (видео 10 с при 24 fps)
- `wire.txt` — wireframe-анимация: 120 кадров, 960x540
- `hires.txt` — hi-res рендер: 3840x2160, ssaa=2, 1 кадр
- `still1..4.txt` — 4 ракурса: 1280x720, ssaa=3, по 1 кадру
- `bench.txt` — бенчмарк GPU vs CPU: 12 кадров, 640x360
- `run_all.sh` — сборка + все прогоны + конвертация в PNG + видео + упаковка
  всего в `results.tar.gz` (сырые .data-кадры после конвертации удаляются)
- `result_to_png.py` — конвертация raw (формат ЛР2) в PNG

## Команды (выполнять на маке, из каталога course_work/)

```bash
# 1. Очистить прошлый прогон на ВМ и скопировать заново
#    (без rm scp -r положит всё в ~/kp/vm/ и запустится старый код):
ssh -p 1858 -i ../golev-andrei-ssh andrey-golev@infway.ru 'rm -rf kp'
scp -P 1858 -i ../golev-andrei-ssh -r vm andrey-golev@infway.ru:kp

# 2. Собрать и прогнать всё (результаты появятся в ~/kp/results на ВМ,
#    упакованы в один архив ~/kp/results.tar.gz):
ssh -p 1858 -i ../golev-andrei-ssh andrey-golev@infway.ru 'cd kp && bash run_all.sh'

# 3. Забрать результаты (один сжатый архив вместо сотен файлов):
scp -P 1858 -i ../golev-andrei-ssh andrey-golev@infway.ru:kp/results.tar.gz ./vm_results.tar.gz
rm -rf vm_results && mkdir vm_results
tar -xzf vm_results.tar.gz -C vm_results --strip-components=1
```

## Материалы для отчёта (в results/)

- `animation.mp4` — видео анимации (10 с, 24 fps, зациклено)
- `wire.mp4` — wireframe-анимация сцены
- `hires.png` — рендер в высоком разрешении (3840x2160)
- `still1..4.png` — рендеры с разных ракурсов
- `bench_cpu.txt` / `bench_gpu.txt` — статистика для сравнения GPU и CPU
  (формат: кадр, время кадра в мс, общее количество лучей)
- `anim_stats.txt`, `wire_stats.txt` — статистика основных серий
- `default_config.txt` — вывод `./kp --default`
- `logs/` — stderr всех прогонов (INFO о GPU, потоках, конфигурации)

## Если на ВМ нет ffmpeg

Видео собирается локально (после `brew install ffmpeg`), кадры уже забраны:

```bash
cd vm_results
ffmpeg -framerate 24 -i anim_png/img_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 animation.mp4
ffmpeg -framerate 24 -i wire_png/img_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 wire.mp4
```

## Ручной прогон с другим конфигом

```bash
./kp < anim.txt              # GPU (по умолчанию)
./kp --cpu < bench.txt       # только CPU
./kp --wireframe < wire.txt  # каркасный рендер
./kp --default               # напечатать красивый конфиг
```

## Замечания

- Компиляция (как у чекера): `nvcc -Werror cross-execution-space-call -lm`
- Кадры нумеруются с нуля, t = 2*pi*i/N (анимация зацикливается)
- Упрощения на «4»: пол квадратный и параллелен XY; преломлённый луч далее
  не отражается (кроме полного внутреннего отражения внутри тела — эффект
  бесконечности); отражения итеративные, без рекурсии
- В stderr пишется отладка (устройство, число потоков, конфигурация) —
  полезно для отчёта, в чекер не мешает
