# Parallelization in `src/dataset.py`

The spectrogram generation runs across 4 worker processes via
`ProcessPoolExecutor`. This document explains how tasks are dispatched and
what to expect when you run `make test`.

## TL;DR

- A single shared FIFO queue holds all 1000 songs.
- Workers race through the queue; whichever finishes first pulls the next task.
- At any moment, all 4 workers are typically processing songs from the **same genre** (the one at the front of the queue).
- The printed output is alphabetical by genre/song regardless of which worker actually ran which song.

## How the work list is built

`main()` walks the dataset sequentially and appends one tuple per song to a
flat list:

```python
work_items: list[tuple[str, str]] = []
for genre in sorted(os.listdir(INPUT_DIR)):              # blues, classical, ...
    for wav_file in sorted(os.listdir(genre_input_dir)): # 00000, 00001, ...
        work_items.append((wav_path, song_output_dir))
```

The result is a single list of 1000 `(wav_path, song_output_dir)` tuples,
ordered alphabetically by genre, then by song name.

## How tasks are dispatched

`executor.map(process_song, *zip(*work_items))` does three things:

1. Unpacks the list of tuples into two parallel iterables.
2. Calls `process_song(wav_path, song_output_dir)` once per item.
3. Hands each call to whichever worker is **next available**.

Tasks are pulled from a single FIFO queue — there is no per-genre
partitioning. The 4 workers share the queue and race through it.

## Concrete timeline

Tasks are submitted in alphabetical order. At any given moment, the queue
head is whichever songs haven't been started yet. Early on, the head is the
blues entries, so all 4 workers end up on blues at the same time:

| Time    | Worker 0           | Worker 1           | Worker 2           | Worker 3           |
|---------|--------------------|--------------------|--------------------|--------------------|
| t=0     | blues/00000        | blues/00001        | blues/00002        | blues/00003        |
| t=2s    | blues/00004        | blues/00001        | blues/00002        | blues/00003        |
| t=2.5s  | blues/00004        | blues/00005        | blues/00002        | blues/00003        |
| ...     | (workers racing)   |                    |                    |                    |
| t=25s   | classical/00007    | blues/00098        | classical/00003    | blues/00099        |

If workers run at different speeds, the faster ones will eventually pull
ahead and start working on later genres while slower ones are still
finishing the earlier ones. Workers do **not** move genre-by-genre in
lockstep — they pull whatever is next in the queue.

## Output ordering

`executor.map` yields results **in input order**, not completion order. Even
though Worker 2 might finish `classical/00050` before Worker 1 finishes
`blues/00050`, the `print(f"Processed: ...")` lines come out in
alphabetical order:

```
Processed: blues/blues.00000
Processed: blues/blues.00001
...
Processed: blues/blues.00099
Processed: classical/classical.00000
...
```

The log is identical to a sequential run; the parallelism is invisible
from the output.

## Why this approach

A per-genre partitioning (e.g. 10 batches of 100 songs each, dispatched one
per genre) might sound more "balanced" but is worse in practice:

- Smaller per-worker batches (25 songs per worker with 4 workers and 100 per
  genre) increase scheduling overhead.
- Worker startup cost is amortized over fewer songs.
- There's no performance benefit — the bottleneck (matplotlib PNG
  rendering) is the same per song regardless of which genre it's in.

The FIFO dispatch is the standard pattern for this kind of embarrassingly
parallel workload.