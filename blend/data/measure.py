"""Aggregate the decision-D8 measurement over the real-target records and their oracle files.
usage: measure.py [--out records/measure.md]
Per (split, group, case): rule, oracle 4, oracle 6, time blend, nearest real frame (t < 0.5 -> prev, else curr; at exactly 0.5 prev):
PSNR from the MSE over the 3 channels (levels), mean abs error, share of pixels with a channel error > 30.  Forza additionally by
the rule's unsure value max(uA,uB) and by vector length max(lenA,lenB).
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

from common import *

VARS = ["rule", "oracle4", "oracle6", "time", "nearest"]
UBINS = [(0.0, 0.2, "u<0.2"), (0.2, 0.6, "0.2<=u<=0.6"), (0.6, 9.0, "u>0.6")]
LBINS = [(0.0, 8.0, "len<8"), (8.0, 32.0, "8<=len<=32"), (32.0, 1e9, "len>32")]


def colours(rec, orc):
    c = rec[:, 0:12].reshape(-1, 4, 3)
    e0, e1 = rec[:, 28:31], rec[:, 31:34]
    w = rec[:, 22:26]
    t = rec[:, 18:19]
    return {"rule": np.einsum("nk,nkc->nc", w, c),
            "oracle4": np.einsum("nk,nkc->nc", orc[:, :4], c),
            "oracle6": np.einsum("nk,nkc->nc", orc[:, 4:8], c) + orc[:, 8:9] * e0 + orc[:, 9:10] * e1,
            "time": (1 - t) * e0 + t * e1,
            "nearest": np.where(t < 0.5, e0, e1)}


class Acc:
    def __init__(self):
        self.n = 0
        self.v = {k: np.zeros(3) for k in VARS}   # sum of squared err (3 ch), sum abs err (3 ch), count big

    def add(self, cols, y, mask=None):
        if mask is not None:
            if not mask.any():
                return
            cols = {k: c[mask] for k, c in cols.items()}
            y = y[mask]
        self.n += len(y)
        for k in VARS:
            d = cols[k].astype(np.float64) - y
            self.v[k] += [(d * d).sum(), np.abs(d).sum(), (np.abs(d).max(axis=1) > 30).sum()]

    def fig(self, k):
        if self.n == 0:
            return None
        se, ae, big = self.v[k]
        mse = se / (3 * self.n)
        return {"psnr": 10 * np.log10(255 ** 2 / max(mse, 1e-12)), "mae": ae / (3 * self.n), "big": big / self.n}


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"{BLEND}/records/measure.md"
    man = {s["id"]: s for s in json.load(open(f"{BLEND}/frames/manifest.json"))["samples"]}
    idx = json.load(open(f"{BLEND}/records/index.json"))
    acc = defaultdict(Acc)
    sumdev = 0.0
    nrec = 0
    for sid, meta in idx.items():
        if meta["kind"] != "real":
            continue
        op = f"{BLEND}/records/{sid}.oracle.npy"
        if not os.path.exists(op):
            continue
        s = man[sid]
        rec = np.load(f"{BLEND}/records/{sid}.npy")
        orc = np.load(op)
        g = group(s["clip"])
        cols = colours(rec, orc)
        y = rec[:, 34:37].astype(np.float64)
        sumdev = max(sumdev, float(np.abs(rec[:, 22:26].sum(1) - 1).max()))
        nrec += 1
        for case in (s["case"], "all"):
            acc[(s["split"], g, case, "")].add(cols, y)
            if g == "forza":
                u = np.maximum(rec[:, 14], rec[:, 17])
                L = np.maximum(rec[:, 19], rec[:, 20])
                for lo, hi, nm in UBINS:
                    acc[(s["split"], g, case, nm)].add(cols, y, (u >= lo) & (u < hi if hi < 9 else u >= lo))
                for lo, hi, nm in LBINS:
                    acc[(s["split"], g, case, nm)].add(cols, y, (L >= lo) & (L <= hi))
    lines = [f"records measured: {nrec}; max |sum(w)-1| of the rule's weights: {sumdev:.2e}", ""]
    res = {}
    groups = ["forza", "cs2", "schedule1", "minecraft_rtx", "minecraft_vanilla", "portland", "minecraft120", "battlefield"]
    cases = ["30to60", "30to120", "60to120", "all"]
    for split in ("val", "train"):
        lines += [f"## {split}", "", "| group | case | px | " + " | ".join(f"{k} PSNR / MAE / >30" for k in VARS) + " | rule->o4 dB | o4->o6 dB |", "|" + "---|" * (5 + len(VARS))]
        for g in groups:
            for case in cases:
                a = acc.get((split, g, case, ""))
                if not a or a.n == 0:
                    continue
                f = {k: a.fig(k) for k in VARS}
                res[f"{split}/{g}/{case}"] = f
                lines.append(f"| {g} | {case} | {a.n} | " + " | ".join(f"{f[k]['psnr']:.2f} / {f[k]['mae']:.2f} / {100 * f[k]['big']:.2f}%" for k in VARS)
                             + f" | {f['oracle4']['psnr'] - f['rule']['psnr']:.2f} | {f['oracle6']['psnr'] - f['oracle4']['psnr']:.2f} |")
        lines.append("")
    lines += ["## forza by bin", ""]
    for split in ("val", "train"):
        for case in cases:
            tot = acc.get((split, "forza", case, ""))
            if not tot or tot.n == 0:
                continue
            lines += [f"### {split} {case}", "", "| bin | px share | " + " | ".join(f"{k} PSNR / MAE / >30" for k in VARS) + " |", "|" + "---|" * (2 + len(VARS))]
            for nm in [b[2] for b in UBINS + LBINS]:
                a = acc.get((split, "forza", case, nm))
                if not a or a.n == 0:
                    continue
                f = {k: a.fig(k) for k in VARS}
                res[f"{split}/forza/{case}/{nm}"] = dict(f, share=a.n / tot.n)
                lines.append(f"| {nm} | {100 * a.n / tot.n:.1f}% | " + " | ".join(f"{f[k]['psnr']:.2f} / {f[k]['mae']:.2f} / {100 * f[k]['big']:.2f}%" for k in VARS) + " |")
            lines.append("")
    open(out, "w").write("\n".join(lines))
    json.dump(res, open(out[:-3] + ".json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
