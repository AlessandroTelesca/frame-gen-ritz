"""Dump the V3 weight-stage records of manifest samples with fgtest and append prev/curr/target columns (SPEC section 3).

usage: dump.py [--kind real|teacher] [--ids FILE] [--split train|val] [--chunk N] [--fgtest PATH]
               [--vmax N] [--vmax-forza N] [--force] [--limit N]
               [--jobs N] [--no-gpu-lock]
  --kind real     samples whose target is a real frame  (frames/<clip>/<target>.rgba)
  --kind teacher  samples whose target is the teacher's  (teacher/<clip>/mid_<i>.rgba); samples whose file is missing are skipped
  --ids FILE      only the sample ids listed in FILE (one per line); without it, every sample of the kind
Output: records/<id>.npy float32 [n, 40]  and records/index.json ({id: {n, stride, vmax, kind}}).
Existing records are kept unless --force. Every fgtest run is one `flock gpu.lock ccorral --` call with CHUNK pairs (same t, same vmax),
so the lock is held for a short time only.
--jobs N runs up to N chunks at once; with the lock (default) the fgtest parts still run one at a time, --no-gpu-lock leaves `flock` out
(only for a machine whose GPU is otherwise idle; N about the number of cores). `ccorral --` is used only if ccorral is installed.
 A failed sample is listed in records/failed.json with the reason.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from common import *

LOCK = os.path.expanduser("~/tmp/fg-handoff/gpu.lock")
FGTEST = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../gpu/build-blend/fgtest"))


def frame_path(s, idx):
    return f"{BLEND}/frames/{s['clip']}/{idx:06d}.rgba"


def target_path(s):
    if s["target_kind"] == "real":
        return frame_path(s, s["target"])
    return f"{BLEND}/teacher/{s['clip']}/mid_{s['target']:06d}.rgba"


def pix(path, idx):
    return np.fromfile(path, np.uint8).reshape(-1, 4)[idx, :3].astype(np.float32)


def build_record(s, sidx, binpath, stride, count):
    r = np.fromfile(binpath, np.float32)[:count * 28].reshape(count, 28)
    idx = np.arange(count, dtype=np.int64) * stride
    out = np.empty((count, 40), np.float32)
    out[:, :28] = r
    out[:, 28:31] = pix(frame_path(s, s["prev"]), idx)
    out[:, 31:34] = pix(frame_path(s, s["curr"]), idx)
    out[:, 34:37] = pix(target_path(s), idx)
    out[:, 37] = idx % W
    out[:, 38] = idx // W
    out[:, 39] = sidx
    return out


def run_chunk(chunk, t, vmax, fgtest, outdir, index, failed, lock, wrap):
    scratch = tempfile.mkdtemp(prefix=".scratch_", dir=outdir)
    try:
        return run_chunk_in(chunk, t, vmax, scratch, fgtest, outdir, index, failed, lock, wrap)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run_chunk_in(chunk, t, vmax, scratch, fgtest, outdir, index, failed, lock, wrap):
    prefix = f"{scratch}/w"
    cmd = [fgtest, "--window", "1", "--set", "algo=v3", "--t", repr(t), "--v3weights", prefix, "--v3wmax", str(vmax), str(W), str(H)]
    for _, s in chunk:
        cmd += [frame_path(s, s["prev"]), frame_path(s, s["curr"]), "/dev/null"]
    t0 = time.time()
    r = subprocess.run(wrap + cmd, capture_output=True, text=True)
    dt = time.time() - t0
    meta = {}
    for line in r.stdout.splitlines():
        if line.startswith("V3WEIGHTS"):
            _, p, cnt, stride, _, _ = line.split()
            meta[int(p)] = (int(cnt), int(stride))
    for p, (sidx, s) in enumerate(chunk):
        if p not in meta or not os.path.exists(f"{prefix}_{p:04d}.bin"):
            with lock:
                failed[s["id"]] = f"no V3WEIGHTS for pair {p}; rc={r.returncode}; stderr: {r.stderr[-300:]}"
            continue
        cnt, stride = meta[p]
        try:
            rec = build_record(s, sidx, f"{prefix}_{p:04d}.bin", stride, cnt)
        except Exception as e:  # noqa: BLE001
            with lock:
                failed[s["id"]] = f"record build: {e!r}"
            continue
        np.save(f"{outdir}/{s['id']}.npy.part.npy", rec)
        os.replace(f"{outdir}/{s['id']}.npy.part.npy", f"{outdir}/{s['id']}.npy")
        with lock:
            index[s["id"]] = {"n": cnt, "stride": stride, "vmax": vmax, "kind": s["target_kind"]}
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="real", choices=["real", "teacher"])
    ap.add_argument("--ids")
    ap.add_argument("--split")
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--fgtest", default=FGTEST)
    ap.add_argument("--vmax", type=int, default=8192)
    ap.add_argument("--vmax-forza", type=int, default=32768)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-gpu-lock", action="store_true")
    ap.add_argument("--outdir", default=f"{BLEND}/records")
    a = ap.parse_args()
    man = json.load(open(f"{BLEND}/frames/manifest.json"))
    outdir = a.outdir
    os.makedirs(outdir, exist_ok=True)
    want = set(open(a.ids).read().split()) if a.ids else None
    ipath = f"{outdir}/index.json"
    index = json.load(open(ipath)) if os.path.exists(ipath) else {}
    fpath = f"{outdir}/failed.json"
    failed = {}
    todo = []
    for i, s in enumerate(man["samples"]):
        if s["target_kind"] != a.kind or (want is not None and s["id"] not in want) or (a.split and s["split"] != a.split):
            continue
        if s["id"] in index and not a.force and os.path.exists(f"{outdir}/{s['id']}.npy"):
            continue
        if not os.path.exists(target_path(s)):
            print("skip (target missing):", s["id"], file=sys.stderr)
            continue
        todo.append((i, s))
    if a.limit:
        todo = todo[:a.limit]
    groups = {}
    for i, s in todo:
        vm = a.vmax_forza if group(s["clip"]) == "forza" else a.vmax
        groups.setdefault((s["t"], vm), []).append((i, s))
    wrap = ([] if a.no_gpu_lock else ["flock", LOCK]) + (["ccorral", "--"] if shutil.which("ccorral") else [])
    if a.jobs > 1 and not a.no_gpu_lock:
        print("note: the gpu lock serialises the fgtest part of the chunks; --no-gpu-lock runs them at once", file=sys.stderr)
    lock = threading.Lock()
    total, used = len(todo), 0.0
    done = 0
    chunks = [(t, vm, lst[k:k + a.chunk]) for (t, vm), lst in groups.items() for k in range(0, len(lst), a.chunk)]
    ex = ThreadPoolExecutor(max(1, a.jobs))
    try:
        futs = {ex.submit(run_chunk, ch, t, vm, a.fgtest, outdir, index, failed, lock, wrap): ch for t, vm, ch in chunks}
        for f in as_completed(futs):
            used += f.result()
            done += len(futs[f])
            with lock:
                json.dump(index, open(ipath + ".part", "w"))
                os.replace(ipath + ".part", ipath)
                print(f"{done}/{total} samples, lock+run time so far {used:.0f}s, failed {len(failed)}", flush=True)
    except BaseException:
        ex.shutdown(cancel_futures=True)
        raise
    ex.shutdown()
    old = json.load(open(fpath)) if os.path.exists(fpath) else {}
    old = {k: v for k, v in old.items() if k not in index}
    old.update(failed)
    json.dump(old, open(fpath, "w"), indent=1)
    print("done", len(index), "records,", len(failed), "failed, gpu-run seconds", round(used))


if __name__ == "__main__":
    main()
