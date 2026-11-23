"""Extra per-row columns for diag.py: sigma (col 21) and s (col 26) of the records, in the row order of the cache (same fixed-seed selection as build_cache.py).
usage: extras.py  -> $BLEND/train/diag/extra_{train,val}.npy [N, 2]; asserts the rows line up (features and S of the cache)."""
import zlib, json
from feat import *
cap = 8192
man_idx = {s["id"]: i for i, s in enumerate(json.load(open(f"{BLEND}/frames/manifest.json"))["samples"])}
os.makedirs(f"{TRAIN}/diag", exist_ok=True)
for split in ("train", "val"):
    ids = json.load(open(f"{CACHE}/real_{split}_ids.json"))
    X = np.load(f"{CACHE}/real_{split}_X.npy", mmap_mode="r")
    S = np.load(f"{CACHE}/real_{split}_S.npy", mmap_mode="r")
    out, pos = [], 0
    for sid in ids:
        rec = np.load(f"{BLEND}/records/{sid}.npy", mmap_mode="r")
        n = len(rec)
        sel = np.sort(np.random.default_rng(zlib.crc32(sid.encode())).choice(n, min(cap, n), replace=False))
        r = np.array(rec[sel])
        k = len(sel)
        assert (S[pos:pos + k] == man_idx[sid]).all()
        assert np.array_equal(features(r), X[pos:pos + k]), sid
        out.append(r[:, [21, 26]])
        pos += k
    out = np.concatenate(out).astype(np.float32)
    assert len(out) == len(X)
    np.save(f"{TRAIN}/diag/extra_{split}.npy", out)
    print(split, out.shape, out.min(0), out.max(0), out.mean(0), flush=True)
