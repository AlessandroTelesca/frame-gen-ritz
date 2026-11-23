"""Writes ONE compressed file with a fixed-seed sample of the cached real rows, for fine-tuning on other footage (train.py --mix FILE).
usage: export_sample.py OUT.npz [--train-rows N] [--val-per-group M] [--seed S]      (plain python3 + numpy, loads the cache: about 3 GB of RAM)
Training rows: N rows (default 2,000,000) balanced the way train.py balances its batches (40 % group forza, the rest evenly over the other groups; a group with
fewer rows than its share gives all it has). Validation rows: up to M (default 100,000) of every group.
File content (float32 unless noted): names (group names, the index space of the G arrays), and for sp in train, val: <sp>_X [n,34], <sp>_R [n,4], <sp>_Y [n,3],
<sp>_O4 [n,3], <sp>_O6 [n,3], <sp>_G [n] int8 (index into names). Same meaning as the cache files of build_cache.py. No sample ids, no paths, no manifest needed."""
import sys

import numpy as np

from feat import *

args = sys.argv[1:]
out = args[0]


def opt(name, default):
    return int(args[args.index(name) + 1]) if name in args else default


N, M, seed = opt("--train-rows", 2_000_000), opt("--val-per-group", 100_000), opt("--seed", 0)


def counts(groups, B):   # = train.py mix_counts
    nf = int(round(0.4 * B)) if 0 in groups else 0
    others = [g for g in groups if g != 0]
    c = {g: (B - nf) // len(others) for g in others}
    if nf:
        c[0] = B - sum(c.values())
    return c


def load(name):
    return {k: np.load(f"{CACHE}/{name}_{k}.npy", mmap_mode="r") for k in ("X", "R", "Y", "O4", "O6", "G", "T")}


rng = np.random.default_rng(seed)
tr, va = load("real_train"), load("real_val")
assert not np.asarray(tr["T"]).any() and not np.asarray(va["T"]).any()
groups = [g for g in range(len(GROUPS)) if (np.asarray(tr["G"]) == g).any()]
sel_tr = {g: np.nonzero(np.asarray(tr["G"]) == g)[0] for g in groups}
pick = np.sort(np.concatenate([rng.choice(sel_tr[g], min(c, len(sel_tr[g])), replace=False) for g, c in counts(groups, N).items()]))
vpick = np.sort(np.concatenate([rng.choice(ix, min(M, len(ix)), replace=False)
                                for ix in (np.nonzero(np.asarray(va["G"]) == g)[0] for g in range(len(GROUPS))) if len(ix)]))
res = {"names": np.array(GROUPS)}
for sp, d, p in (("train", tr, pick), ("val", va, vpick)):
    for k in ("X", "R", "Y", "O4", "O6"):
        res[f"{sp}_{k}"] = np.asarray(d[k][p], np.float32)
    res[f"{sp}_G"] = np.asarray(d["G"][p], np.int8)
    print(sp, len(p), "rows", {GROUPS[g]: int((res[f"{sp}_G"] == g).sum()) for g in range(len(GROUPS)) if (res[f"{sp}_G"] == g).any()}, flush=True)
np.savez_compressed(out, **res)
import os
print("wrote", out, "%.1f MB" % (os.path.getsize(out) / 1e6))
