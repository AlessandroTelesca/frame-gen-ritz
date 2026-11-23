# Fine-tuning the blend nets on your own footage (handoff note for an AI coding agent)

You are working for someone who records games. This note tells you how to run a short, fixed pipeline on his machine and send back a few small files. You need no knowledge of the project beyond this note; do the steps in order, check each result, and stop and report if a check fails.

## Purpose

The repository `Ritze03/frame-gen-ritz` contains a Vulkan frame-generation library. One of its modes mixes four colour samples per output pixel with a hand-written rule; three tiny per-pixel neural nets (called A, B, C; 244 to 334 numbers each) can replace that rule, and they were trained on the project owner's own recordings. This pipeline fine-tunes those nets on **your** recordings (Forza, 1920x1080, 120 fps) so that they also fit your footage, **without any large file leaving your machine**.
What comes back: the fine-tuned weights (`A.npz`, `B.npz`, `C.npz`, a few KB each), a score table (`scores.md`) and a run description (`run.json`), as one **pull request from your fork against the branch `dev`**.

## What you need

| thing | amount |
|---|---|
| OS | Nobara Linux (Fedora based); the commands use `dnf` |
| CPU / RAM | 8 cores / 16 threads; 16 GB RAM is enough (RAM per step: see the steps) |
| GPU | AMD Radeon RX 9070 XT with the RADV driver (the default Mesa Vulkan driver); the GPU is used by the dump step only |
| free SSD space | **90 GB** for `--every 2` (50 min of footage), table below. The recordings themselves need not be on that SSD |
| footage | one or more recordings, 1920x1080, **120 fps constant**, about 50 minutes in total (77 GB) |
| file `owner_rows.npz` (about 165 MB) | the project owner sends it to you separately; it is NOT in the repository |
| GitHub account | to fork the repository and open the pull request |

Disk space (estimates computed from sizes measured on the owner's machine; peak is at the end of the dump step, the frames are the largest part and can be deleted once dump and oracle are done):

| `--every` (seconds between anchors) | anchors per 50 min | frames (5 x 8.3 MB per anchor) | records + oracle | cache (cap 2048) | clone, build, python | **peak** | free space to have |
|---|---|---|---|---|---|---|---|
| 1 | 3,000 | 124 GB | 20 GB | 5 GB | 3 GB | about 150 GB | 170 GB |
| **2 (use this)** | 1,500 | 62 GB | 10 GB | 2.5 GB | 3 GB | about 78 GB | **90 GB** |
| 4 | 750 | 31 GB | 5 GB | 1.2 GB | 3 GB | about 40 GB | 50 GB |

Records + oracle = 1.31 MB + 0.33 MB per sample (measured), 4 samples per anchor. Frames, records and the cache scale linearly with the footage length too.

## 1. Setup

Install the packages. **This list is not tested on Nobara; if a package name is wrong, find the right one with `dnf provides <file-or-command>`** (for example `dnf provides '*/glslangValidator'`).

```
sudo dnf install -y git gh meson ninja-build gcc-c++ pkgconf-pkg-config \
    vulkan-loader-devel vulkan-headers vulkan-tools mesa-vulkan-drivers glslang \
    python3 python3-pip python3-numpy ffmpeg util-linux
```

What each is for: `meson ninja-build gcc-c++ pkgconf-pkg-config` build the C++ test program; `vulkan-loader-devel vulkan-headers` (Vulkan loader and headers); `glslang` provides `glslangValidator` (shader compiler); `vulkan-tools` provides `vulkaninfo`; `ffmpeg` provides `ffmpeg` and `ffprobe` (if Nobara's ffmpeg conflicts with `ffmpeg-free`, keep the one that is installed, it must have a decoder for your recordings); `util-linux` provides `flock`; `gh` is the GitHub CLI (optional, the web page works too). Check:

```
for c in meson ninja g++ glslangValidator ffmpeg ffprobe python3 git flock vulkaninfo; do command -v $c >/dev/null && echo "ok $c" || echo "MISSING $c"; done
vulkaninfo --summary | grep -iE "deviceName|driverName"
```

`vulkaninfo` must list the Radeon (`deviceName` containing `RX 9070 XT`) with `driverName = radv`. If it lists only `llvmpipe` (a software renderer), the GPU driver is not set up: stop and report.

Fork and clone. On the web page https://github.com/Ritze03/frame-gen-ritz click "Fork" (or `gh repo fork Ritze03/frame-gen-ritz --clone=false` after `gh auth login`). Then (replace `YOURNAME`):

```
cd ~
git clone https://github.com/YOURNAME/frame-gen-ritz.git
cd frame-gen-ritz
git checkout dev
git log --oneline -1
export REPO=$PWD
```

The scripts were tested at commit `4fa2f47` of `dev` (and everything after it that only touches other files). Use the tip of `dev` of your fork after syncing it with `Ritze03/frame-gen-ritz` (`git fetch https://github.com/Ritze03/frame-gen-ritz.git dev && git merge --ff-only FETCH_HEAD` if your fork is behind).

Build the test program `fgtest` (plain setup is enough; about 3 to 6 minutes; it compiles many shaders):

```
cd $REPO/gpu
meson setup build-blend
ninja -C build-blend fgtest
ls -la build-blend/fgtest
```

The folder name `build-blend` matters: `dump.py` looks for `gpu/build-blend/fgtest`.

Python for training (PyTorch, CPU build). The data scripts need only the system `python3` + numpy; only training needs PyTorch:

```
python3 -m venv ~/venv-blend
~/venv-blend/bin/pip install numpy
~/venv-blend/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/venv-blend/bin/python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
```

Why the CPU build: the nets have 244 to 334 numbers and train in a few minutes on a CPU; a GPU build of PyTorch for AMD is a multi-gigabyte install that would save no time. (If pip finds no wheel for your Python version, install an older one, e.g. `sudo dnf install python3.12`, and create the venv with `python3.12 -m venv`. Not tested.)

## 2. The data folder and `clips.json`

All big files go into one folder on the SSD, called `$BLEND` (the scripts read the variable `BLEND`; **always export it in every new shell**):

```
export BLEND=/path/on/ssd/blend
mkdir -p $BLEND
```

Create `$BLEND/clips.json`, one entry per recording. With this file present the scripts know only the clips named in it.

```
{
  "forza_1": {"path": "/abs/path/recording1.mkv", "fps": 120, "group": "forza", "skip": [[0, 35.5], [1810, 1900]]},
  "forza_2": {"path": "/abs/path/recording2.mp4", "fps": 120, "group": "forza"}
}
```

- name (`forza_1`): any short name with letters, digits, `_`; it is used for folders and file names.
- `path`: absolute path of the file. `fps`: 120. `group`: always `"forza"` for your footage.
- `skip` (optional): time ranges in **seconds from the start of that file** that must produce no training samples: menus, replays, pause screens, loading screens, photo mode. To find them, scrub through each recording in a video player and write down the times (seconds = minutes x 60 + seconds). If you are unsure, leave a range out: pictures that stand still are removed anyway by the repeated-frame filter in the extract step.
- `"split": "val"` or `"train"` (optional): put the whole recording on the validation side or the training side. Without it, **the last 10 % of every recording is the held-out validation part** (that gives whole held-out stretches).

Check each recording (the scripts also check this and stop with a message):

```
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames -of default=nw=1 /abs/path/recording1.mkv
```

Needed: `width=1920`, `height=1080`, `r_frame_rate=120/1` (119.88 is accepted too, anything between 119.5 and 120.5), `avg_frame_rate` the same. `nb_frames=N/A` is normal for `.mkv`: the extract script then counts the packets (fast) and prints "from packet count".

## 3. The steps, in order

Run everything in the same shell (`REPO` and `BLEND` exported). **If a step stops (reboot, Ctrl-C, error), run the same command again**; what to expect on a rerun is stated per step. Times marked "estimate" were not measured on your machine.

### 3.1 extract: frames around sampled positions

Per recording (about 1,500 anchors in total for 50 minutes at `--every 2`; a 120 fps anchor is 5 consecutive frames, 41 MB raw):

```
cd $REPO/blend/data
python3 extract.py forza_1 --every 2
python3 extract.py forza_2 --every 2
```

Writes `$BLEND/frames/<clip>/<frameindex>.rgba` (raw 1920x1080 RGBA, 8.3 MB each) and `$BLEND/frames/<clip>.anchors.json`. Prints `<clip> <frames> 120 kept N skipped M` at the end (`skipped` = anchors dropped because a frame was repeated). Load: all CPU cores for the decoding, sequential disk writes (62 GB in total), about 1 GB RAM. Time: estimate 10 to 30 minutes per 50 minutes of footage (one pass over the file; decoding speed decides). Worked: the `.anchors.json` exists and `kept` is roughly (length in seconds / 2) minus the skipped ranges. Resume: run the same command again; it starts that recording again and overwrites its frames; recordings whose `.anchors.json` exists are finished and need not be redone. If frames of a recording are in the folder but its `.anchors.json` is missing, the run was interrupted: rerun that recording.

### 3.2 manifest

```
python3 build_manifest.py
```

Writes `$BLEND/frames/manifest.json` (every sample: which frames, what target; `split` train / val) and prints `N samples` (11 per anchor: 4 with a real target frame, 7 with a "teacher" target that is not used here). Seconds, no load. Rerun any time (it is rebuilt from the `.anchors.json` files).

### 3.3 dump: run the library on every sample and record per-pixel data

Close games and every other GPU program first. `fgtest` uses the GPU through Vulkan; do not use the machine for anything GPU-heavy while this runs.

```
python3 dump.py --kind real --jobs 8 --no-gpu-lock --vmax-forza 8192
```

`--kind real` dumps only samples with a real target frame (the teacher samples in the manifest are skipped on purpose). `--jobs 8` runs 8 `fgtest` processes at once (use `--jobs 4` if the GPU driver complains or the screen stutters). `--no-gpu-lock` leaves out the owner's machine-specific lock (`flock`) and `ccorral`, which you do not have. `--vmax-forza 8192` = 8,192 pixels per sample (the owner's default of 32,768 would write 4 times more records than the next steps use). `--chunk` stays at its default 30 pairs per `fgtest` call. Writes `$BLEND/records/<sample id>.npy` (1.3 MB each; 6,000 samples = about 8 GB) and `records/index.json`; prints `done/total samples` lines, and at the end `done N records, K failed`. Load: GPU + CPU for fgtest, reads three 8.3 MB frames per sample from disk (about 150 GB of reads in total: if the frames are not in the page cache this step is limited by the SSD), RAM about 4 GB with 8 jobs (estimate). Time: measured on the owner's machine with frames in the page cache about 3,600 samples/min for one job and about 10,000 samples/min for 4 to 8 jobs, so 6,000 samples take 1 to 5 minutes there; with 62 GB of frames (much more than the RAM) expect the SSD speed to decide: estimate 5 to 20 minutes. Worked: `K failed` is 0 (otherwise see `$BLEND/records/failed.json`) and `ls $BLEND/records/*.npy | wc -l` is about 4 x the number of anchors (plus `.oracle.npy` files later). Resume: run the same command again; samples that already have a record are skipped and failed ones are retried.

After this step the frames are no longer needed by the next steps. To save 62 GB you may delete them now (`rm -r $BLEND/frames/<clip>/` for every clip, **not** the `.anchors.json`, `manifest.json`). Only do it after the dump finished with 0 failed.

### 3.4 oracle: the best possible mix per pixel (the training target)

```
python3 oracle.py --procs 14
```

Writes `$BLEND/records/<id>.oracle.npy` next to each record (0.33 MB each). Load: 14 CPU processes, little RAM (about 0.3 GB each). Time: measured 0.33 s per sample on one core, so 6,000 samples = about 33 minutes on one core, about 3 minutes with 14 processes (estimate for your CPU). Worked: it prints `oracle done N`. Resume: rerun; existing oracle files are skipped.

### 3.5 cache: a random subsample of rows for training

```
cd $REPO/blend/train
python3 build_cache.py real --cap 2048 --procs 14
```

Writes `$BLEND/train/cache/real_train_*.npy` and `real_val_*.npy` (and prints `real_train N rows ...`, `real_val N rows ...`; with 6,000 samples about 11 million training rows and 1.2 million validation rows, about 2.5 GB). `--cap 2048` = at most 2,048 rows per sample (the owner used 8,192; with 32 GB of RAM or more you may use `--cap 4096`, it doubles the cache and the RAM use). Load: 14 processes, RAM peak about 5 GB (estimate). Time: estimate 3 to 10 minutes. Worked: `real_val` has rows of `forza` only; the val rows come from the last 10 % of each recording. Resume: rerun (it rebuilds the whole cache).

### 3.6 train: fine-tune A, B and C

Put the file the owner sent you at a known place, for example `export OWNER_ROWS=$HOME/owner_rows.npz`. Then:

```
~/venv-blend/bin/python train.py $BLEND/train/out_ft \
    --init $REPO/blend/weights/base --mix $OWNER_ROWS \
    --steps 4000 --lr 2e-4 --lr-end 1e-5 --threads 8
```

What it does: starts from the owner's current weights (`blend/weights/base`) instead of from scratch; every batch of 65,536 pixels is half your rows (from the cache) and half the owner's rows from other games (`owner_rows.npz`, balanced over its 8 groups, 40 % Forza), so the nets learn your footage without forgetting the rest. Every 500 steps it validates on your held-out rows and on the owner's validation rows, and **keeps the step with the lowest mean of both** (step 0 = the unchanged base weights, so the result is never worse than the start on that measure). Learning rate: `2e-4` falling to `1e-5` (cosine), a tenth of the rate of the from-scratch run (`2e-3`), because the start is already trained and a high rate only throws it away; 4,000 steps because the from-scratch run needed 6,000 to 15,000 and a fine-tune needs fewer. **These two numbers are a reasoned choice, not tuned on real foreign footage**; if the validation numbers printed every 500 steps keep falling to the last step, a run with `--steps 8000` is fine. Load: CPU only (`--threads 8`), RAM about 4 GB. Time: measured on the owner's machine 300 steps in 8 s with 4 threads, so 4,000 steps take about 2 to 4 minutes (estimate; plus about 1 minute to load the files and score at the end). Writes into `$BLEND/train/out_ft`: `A.npz B.npz C.npz log.json scores.md run.json`. Worked: the last lines are `best {...}` and `wrote scores.md and run.json`. Resume: just run it again (a run is short and has no checkpoints); to try other settings write to a different folder, e.g. `out_ft2`.

## 4. Checks before sending

Open `$BLEND/train/out_ft/scores.md`. For each of the nets A, B, C it shows PSNR (dB, higher is better), MAE (average error in colour levels, lower is better) and the share of pixels with an error above 30 levels (lower is better), each for **rule** (the hand-written weights), **base** (the owner's weights) and **tuned** (your result), (i) on your held-out rows and (ii) on the owner's validation rows per game group.

A sane result:
- on **your validation rows** (table i): `tuned` is not worse than `base`, and ideally better (a gain of 0.1 dB or more is a good result; the point of the exercise);
- on **the owner's validation rows** (table ii, group `all`): `tuned` is not worse than `base` by more than about 0.05 dB;
- `base` and `rule` are about equal or `base` is better on your footage; if `tuned` equals `base` exactly, the best step was step 0: the fine-tune found nothing better, which is a valid result.

If it is worse than that anyway, **report it anyway**: send the pull request as described below and say in its text what you saw. Do not change the numbers or retry until the table looks good; you may try one more run with a lower rate (`--lr 1e-4 --lr-end 5e-6`) in another folder and send both, in two folders.

Also check `run.json`: it lists no absolute paths and no user names (the scripts do not write any; if you see one, remove it before committing).

## 5. Sending it back

Choose a short name for the folder, e.g. `forza-rx9070xt` (letters, digits, `-`). In your fork:

```
cd $REPO
git checkout -b weights/forza-rx9070xt dev
mkdir -p blend/weights/forza-rx9070xt
cp $BLEND/train/out_ft/{A,B,C}.npz $BLEND/train/out_ft/scores.md $BLEND/train/out_ft/run.json blend/weights/forza-rx9070xt/
git add blend/weights/forza-rx9070xt/A.npz blend/weights/forza-rx9070xt/B.npz blend/weights/forza-rx9070xt/C.npz blend/weights/forza-rx9070xt/scores.md blend/weights/forza-rx9070xt/run.json
git status --short
git commit -m "Weights fine-tuned on Forza footage (forza-rx9070xt)"
git push -u origin weights/forza-rx9070xt
gh pr create --repo Ritze03/frame-gen-ritz --base dev --head YOURNAME:weights/forza-rx9070xt \
    --title "Weights: Forza fine-tune (forza-rx9070xt)" --body-file ~/pr.txt
```

(Write the PR text first into the file `~/pr.txt`, outside the repository; or open the pull request on the web page, base branch `dev`.) `git status --short` before the commit must list exactly those five files and nothing else.

The PR text should say: how much footage (number of recordings, total minutes, 1920x1080, 120 fps), the sampling (`--every 2`, number of anchors kept, number of samples: see `run.json`), the training options (steps, learning rate), the machine (CPU, GPU, "Nobara"), whether anything failed or was changed, and the **score table**: paste the rows of table (i) and the `all` rows of table (ii) from `scores.md` (the full file is in the PR as well), plus one sentence on whether the result is sane by the checks above.

**Do not** commit footage, frames, records, the cache or `owner_rows.npz`. **Do not** change any other file of the repository. **Do not** regenerate the shader header (`gpu/shaders/v3_blend_weights.glsl`): the owner does that himself after scoring your weights on his side.

## 6. Keep afterwards

Keep the recordings, `$BLEND/clips.json` and `$BLEND/frames/*.anchors.json` (`.anchors.json` are a few KB): a later round with changed nets needs the records dumped again, and the exact anchors must be reproduced. The frames, the records and the cache can be deleted (`rm -r $BLEND/frames/*/ $BLEND/records $BLEND/train/cache`) and are re-created by steps 3.1 to 3.5 when needed (3.1 can be skipped for a clip whose frames are still there). The result folder `$BLEND/train/out_ft` is small; keep it too.

## 7. If something fails

| symptom | cause / fix |
|---|---|
| `the recording must be 1920x1080, it is ...` | the recording has another resolution; this pipeline works at 1920x1080 only; stop and report |
| `a 120 fps clip is expected, ffprobe says r_frame_rate = 60/1 ...` | not 120 fps, or a variable-frame-rate recording (OBS "VFR"); the frame index would not be constant in time; re-record at constant 120 fps or stop and report |
| `nb_frames` missing / `N/A` | normal for `.mkv`; the script counts packets and prints `(from packet count)`; if even that fails it uses duration x rate |
| `select expression ... too long` | more than about 3,500 anchors in one recording: use a larger `--every` or cut the recording into two files |
| `ffmpeg` reads fewer frames than expected (`AssertionError` in extract) | corrupt or truncated file; test with `ffmpeg -v error -i FILE -f null -` |
| `fgtest` not found / `dump.py` all failed | the build is missing: `ls $REPO/gpu/build-blend/fgtest`; read `records/failed.json` for the message |
| Vulkan device not found / `llvmpipe` only | `vulkaninfo --summary` must show the Radeon with `radv`; install `mesa-vulkan-drivers`, check the driver |
| dump is very slow or the system stutters | `--jobs 4`; close GPU programs |
| `No space left on device` | see the table of free space; delete the frames after a finished dump (step 3.3), use a larger `--every` |
| `MemoryError` / killed in cache or training | lower `--cap` (e.g. 1024), close other programs |
| `ModuleNotFoundError: torch` | train with `~/venv-blend/bin/python`, not `python3` |
| `AssertionError: validation row in the training set` | `build_manifest.py` was run, then clips changed: delete `$BLEND/train/cache`, rerun 3.2 to 3.5 |

If a script is wrong or missing something, fixes are welcome as a **separate pull request** (not mixed with the weights).
