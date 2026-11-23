# blend/data: frames, records and the oracle for the learned-blend experiment

Python 3 + numpy only (no Pillow). `BLEND` (env, default `/home/mo/DieGanzeSau/AI_Video_Training_Data/blend`) is the data root.
`fgtest` comes from `cd gpu && meson setup build-blend && ninja -C build-blend fgtest`.

| script | what |
|---|---|
| `common.py` | paths, clip lists, `group(clip)` |
| `extract.py CLIP` | one ffmpeg pass per clip (`select=between(..)`, `-fps_mode passthrough`, rgba), drops anchors with repeated frames (mean abs diff < 0.05), writes `frames/<clip>/<idx>.rgba` and `frames/<clip>.anchors.json` |
| `build_manifest.py` | merges the anchor files into `frames/manifest.json` (+ `skipped.json`), writes `frames/DONE` |
| `dump.py` | the weight-stage dump: runs `fgtest --window 1 --set algo=v3 --t T --v3weights .. --v3wmax N` for chunks of pairs (one `flock gpu.lock ccorral --` call per chunk, same t), appends prev / curr / target / x / y / id columns, writes `records/<id>.npy` ([n, 40] float32) and `records/index.json`; failures go to `records/failed.json` |
| `oracle.py` | exact per-pixel simplex-constrained least squares over the 4 taps (and over taps + prev(x) + curr(x)); `--selftest` checks it against a brute-force grid; CLI writes `records/<id>.oracle.npy` ([n, 10]) |
| `measure.py` | the D8 tables (rule / oracle 4 / oracle 6 / time blend / nearest, per split x group x case; Forza by unsure and vector length) -> `records/measure.md/.json` |
| `make_images.py` | comparison PNGs for six Forza validation samples (full-resolution dump, deleted after use) |
| `check_record.py` | the record sanity check (tap range, sum of weights, agreement with fgtest's output) |

Record columns (0-based): 0-11 taps CAp CAc CBp CBc, 12-17 oA holeA uA oB holeB uB, 18 t, 19 lenA, 20 lenB, 21 sigma, 22-25 w0..w3, 26 s, 27 pad,
28-30 prev(x), 31-33 curr(x), 34-36 target, 37 x, 38 y, 39 manifest sample index (float).

## Dump for samples with teacher targets
The teacher frames live in `$BLEND/teacher/<clip>/mid_<i>.rgba` (`i` = the target index of the manifest row). Samples whose file is missing are skipped
(printed to stderr).

    cd blend/data
    python3 dump.py --kind teacher --ids ids.txt        # ids.txt: one sample id per line; without --ids: every teacher sample
    python3 oracle.py --kind teacher                     # oracle weights for them (the CLI takes --kind real|teacher)

Options: `--split train|val`, `--chunk N` (pairs per lock hold, default 30, about 30 s), `--vmax` (8192) / `--vmax-forza` (32768), `--force`, `--outdir`.
