# blend/train: the learned-blend nets A, B, C

Trains the three tiny per-pixel nets of `v3Blend` (SPEC section 4, `gpu/ml/blend/ref.py` is the definition) on the dumped records
(`$BLEND/records`, `$BLEND/records_t`; `$BLEND` = `/home/mo/DieGanzeSau/AI_Video_Training_Data/blend`), writes `$BLEND/train/out/{A,B,C}.npz`
(the input of `gpu/ml/blend/export_glsl.py`) and scores them. Interpreter: `/home/mo/Schinken/Programming/Local/ml-fg/.venv/bin/python`
(PyTorch, CPU only); start everything with `ccorral -- ...`, run from this folder.

| script | what |
|---|---|
| `feat.py` | the 34-value input vector (`features`), paths, group list |
| `build_cache.py real\|teacher [--cap N]` | fixed-seed subsample per sample (default 4096 rows; the real set was built with `--cap 8192`), features, rule weights, target and oracle colours -> `$BLEND/train/cache/*.npy`. Teacher: only train samples whose `records_t/<id>.npy` exists; the oracle is computed here (4 / 6 choices, `blend/data/oracle.py`) on the subsample only |
| `train.py OUTDIR [--teacher] [--steps N]` | trains A, B, C together on the same batches (65,536 rows: 40 % Forza, 60 % evenly over the other groups; teacher rows weight 0.5), Adam 2e-3 -> 1e-4 cosine, `W2 = b2 = 0` start, validation on the real validation rows every 500 steps, best step kept; asserts no validation sample is in the training set |
| `check_forward.py` | the torch forward pass against `ref.py` (random weights) |
| `score.py` | streams every sample (all rows) and writes `$BLEND/train/scores.md/.json`: PSNR / MAE / share of errors > 30 for rule, nets, oracles per split, group, case; Forza by unsure bin; teacher-target validation; mean w4 + w5 and mean change of the weights |

Rerun: `build_cache.py real --cap 8192`, `build_cache.py teacher --cap 2048`, `train.py $BLEND/train/out/real_only`, `train.py $BLEND/train/out/with_teacher --teacher`, `score.py`.
Run the export with `python3 gpu/ml/blend/export_glsl.py A.npz B.npz C.npz -o somewhere.glsl` (never to `gpu/shaders/` unless intended).
