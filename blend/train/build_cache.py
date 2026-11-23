"""Builds the training caches under $BLEND/train/cache/.   build_cache.py real | teacher [--cap N] [--procs P]
Per sample: a fixed-seed random subsample (<= cap rows, default 4096), then features (34), the rule's weights, the target, the oracle colours.
Files per set NAME (real_train, real_val, teacher_train): NAME_X.npy [N,34], NAME_R [N,4], NAME_Y [N,3] (target/255), NAME_O4 / NAME_O6 [N,3]
(oracle colours /255), NAME_G [N] int8 (group index), NAME_S [N] int32 (manifest sample index, column 39), NAME_T [N] (1 = teacher target),
NAME_ids.json (sample ids included). The split of every sample is asserted through column 39 and the manifest. Teacher val samples are not cached
(they are only scored, streaming, by score.py).
"""
import sys, os, json, zlib
import numpy as np
from multiprocessing import Pool
from feat import *
import oracle as OR

kind = sys.argv[1]
cap = int(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else 4096
procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 3
man = manifest()
idxd = f"{BLEND}/records" if kind == "real" else f"{BLEND}/records_t"
ids = sorted(s for s, v in man.items() if v["target_kind"] == kind and os.path.exists(f"{idxd}/{s}.npy")
             and (kind == "real" or v["split"] == "train"))
if kind == "real":
    ids = [s for s in ids if os.path.exists(f"{idxd}/{s}.oracle.npy")]


def job(sid):
    s = man[sid]
    try:
        rec = np.load(f"{idxd}/{sid}.npy", mmap_mode="r")
        n = len(rec)
        rng = np.random.default_rng(zlib.crc32(sid.encode()))
        sel = np.sort(rng.choice(n, min(cap, n), replace=False))
        rec = np.array(rec[sel])
    except Exception:  # file still being written
        return sid, None
    assert rec.shape[1] == 40 and (rec[:, 39] == rec[0, 39]).all()
    assert man_by_idx[int(rec[0, 39])] == sid, "column 39 does not match the file"
    taps, six, y = OR.points(rec)
    orc = np.load(f"{idxd}/{sid}.oracle.npy", mmap_mode="r")[sel] if kind == "real" else OR.oracle_for(rec)
    c4 = np.einsum("nk,nkc->nc", orc[:, :4], taps)
    c6 = np.einsum("nk,nkc->nc", orc[:, 4:], six)
    return sid, dict(X=features(rec), R=rec[:, 22:26], Y=rec[:, 34:37] / 255, O4=c4 / 255, O6=c6 / 255,
                     G=np.full(len(rec), GROUPS.index(group(s["clip"])), np.int8), S=rec[:, 39].astype(np.int32))


man_by_idx = {i: s["id"] for i, s in enumerate(json.load(open(f"{BLEND}/frames/manifest.json"))["samples"])}

if __name__ == "__main__":
    out = {}
    with Pool(procs) as p:
        for i, (sid, d) in enumerate(p.imap(job, ids, chunksize=4)):
            if d is not None:
                out.setdefault(man[sid]["split"], []).append((sid, d))
            if i % 200 == 0:
                print(i, len(ids), flush=True)
    os.makedirs(CACHE, exist_ok=True)
    for split, lst in out.items():
        name = f"{kind}_{split}"
        d = {k: np.concatenate([x[1][k] for x in lst]) for k in lst[0][1]}
        for k in ("X", "R", "Y", "O4", "O6"):
            d[k] = d[k].astype(np.float32)
        d["T"] = np.full(len(d["G"]), kind == "teacher", np.int8)
        for sid, _ in lst:
            assert man[sid]["split"] == split
        for k, v in d.items():
            np.save(f"{CACHE}/{name}_{k}.npy", v)
        json.dump([sid for sid, _ in lst], open(f"{CACHE}/{name}_ids.json", "w"))
        print(name, len(d["G"]), "rows", len(lst), "samples", {GROUPS[g]: int((d["G"] == g).sum()) for g in range(len(GROUPS))})
