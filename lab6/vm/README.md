# Запуск лабы 6 на ВМ с GPU

Содержимое:
- `lab6.cu` — копия основного `../lab6.cu` (при изменении основного файла перекопируй!)
- `make_texture.py` — генератор текстуры сферы (массив уже встроен в lab6.cu)
- `result_to_png.py` — конвертация raw-вывода программы в PNG
- `run_all.sh` — сборка + 4 прогона для скриншотов отчета

## Команды (выполнять на маке, из каталога lab6/)

```bash
# 1. Скопировать на ВМ:
scp -P 1858 -i ../golev-andrei-ssh -r vm andrey-golev@infway.ru:lab6

# 2. Собрать и прогнать (4 скриншота появятся в ~/lab6/results на ВМ):
ssh -p 1858 -i ../golev-andrei-ssh andrey-golev@infway.ru 'cd lab6 && bash run_all.sh'

# 3. Забрать результаты:
scp -P 1858 -i ../golev-andrei-ssh -r andrey-golev@infway.ru:lab6/results ./vm_results
```

## Ручной прогон с другим сценарием

```bash
printf 'time,10;pitch,-16;shoot,1;time,4\n' | ./lab6 > out.data
python3 result_to_png.py out.data out.png
```

## Формат входной строки

`left,right,forward,back,up,down,D` — сдвиг камеры на D единиц; `yaw,deg` —
поворот параллельно полу; `pitch,deg` — наклон вверх; `shoot,1` — выстрел;
`time,T` — моделирование T секунд. Рендер выполняется один раз в конце.

## Замечания

- Компиляция на ВМ: `nvcc -Werror cross-execution-space-call -lm -lGL`
  (команда чекера дополнительно линкует `-lcublas -lcurand -lGLU -lglut -lGLEW` —
  этих библиотек на ВМ нет, программой они не используются).
- GL-контекст создается через EGL (NVIDIA, без X-сервера); заголовков EGL на ВМ
  нет, поэтому функции EGL загружаются через dlopen.
- В stderr программа пишет отладочные строки (версия GL, статус interop) —
  их полезно прислать вместе со скриншотами.
