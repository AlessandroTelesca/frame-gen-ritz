"""Scores the rule, the nets and the oracles on ALL rows of every sample (streaming, no subsample).
usage: score.py [--out DIR]   (default $BLEND/train)   -> scores.md / scores.json
Weight sets: real_only = out/real_only/{A,B,C}.npz, final = out/{A,B,C}.npz and out/with_teacher/ if present; zero_* = untrained W2 = b2 = 0 nets (check 2).
Real-target samples (records/ + .oracle.npy): rule, nets, oracle 4, oracle 6. Teacher-target validation samples (records_t/, 60 fps cuts): rule and nets.
Per (split, group, case): PSNR (pooled MSE over RGB, levels), MAE, share of pixels with any channel error > 30. Forza val also by u = max(uA, uB).
Also per net: mean w4 + w5 and mean |w - rule| over the four weights.
"""
import sys, os, json
from collections import defaultdict
import numpy as np, torch
from feat import *
import oracle as OR
import train as TR

out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else TRAIN
torch.set_num_threads(5)
man = manifest()


def loadnet(path):
    z = np.load(path)
    return [torch.from_numpy(np.asarray(z[k], np.float32)) for k in ("W1", "b1", "W2", "b2")]


rng0 = np.random.default_rng(0)
NETS = {}  # name -> (params, NI, K)
for setname, d in (("real_only", f"{TRAIN}/out/real_only"), ("final", f"{TRAIN}/out"), ("with_teacher", f"{TRAIN}/out/with_teacher")):
    for v, (NI, K, _) in TR.VARS.items():
        if os.path.exists(f"{d}/{v}.npz"):
            NETS[f"{setname}/{v}"] = (loadnet(f"{d}/{v}.npz"), NI, K)
for v, (NI, K, _) in TR.VARS.items():
    NETS[f"zero/{v}"] = ([torch.from_numpy(rng0.normal(0, .1, (8, NI)).astype(np.float32)), torch.zeros(8), torch.zeros(K, 8), torch.zeros(K)], NI, K)
UBINS = [(0.0, 0.2, "u<0.2"), (0.2, 0.6, "0.2<=u<=0.6"), (0.6, 9.0, "u>0.6")]


class Acc:
    def __init__(self):
        self.n = 0
        self.v = defaultdict(lambda: np.zeros(3))

    def add(self, cols, y, mask=None):
        if mask is not None:
            if not mask.any():
                return
            cols = {k: c[mask] for k, c in cols.items()}
            y = y[mask]
        self.n += len(y)
        for k, c in cols.items():
            d = c.astype(np.float64) - y
            self.v[k] += [(d * d).sum(), np.abs(d).sum(), (np.abs(d).max(axis=1) > 30).sum()]

    def fig(self, k):
        se, ae, big = self.v[k]
        return {"psnr": float(10 * np.log10(255 ** 2 / max(se / (3 * self.n), 1e-12))), "mae": float(ae / (3 * self.n)), "big": float(big / self.n)}


def net_colours(rec, X):
    cols, stats = {}, {}
    Xt, Rt = torch.from_numpy(X), torch.from_numpy(np.ascontiguousarray(rec[:, 22:26]))
    with torch.no_grad():
        for name, (p, NI, K) in NETS.items():
            C, w = TR.forward(p, Xt, Rt, K, NI)
            cols[name] = C.numpy().astype(np.float64) * 255
            stats[name] = np.array([(w[:, 4:].sum(1)).sum().item() if K == 6 else 0.0, (w[:, :4] - Rt).abs().mean(1).sum().item()])
    return cols, stats


def main():
    acc = defaultdict(Acc)
    st = defaultdict(lambda: np.zeros(3))   # (split, group, net) -> sum w45, sum |dw|, n
    tasks = [(sid, "real") for sid, s in man.items() if s["target_kind"] == "real"]
    tasks += [(sid, "teacher") for sid, s in man.items() if s["target_kind"] == "teacher" and s["split"] == "val" and s["class"] == 60
              and os.path.exists(f"{BLEND}/records_t/{sid}.npy")]
    for i, (sid, kind) in enumerate(sorted(tasks)):
        s = man[sid]
        d = f"{BLEND}/records" if kind == "real" else f"{BLEND}/records_t"
        if kind == "real" and not os.path.exists(f"{d}/{sid}.oracle.npy"):
            continue
        rec = np.load(f"{d}/{sid}.npy")
        g = group(s["clip"])
        y = rec[:, 34:37].astype(np.float64)
        taps = rec[:, 0:12].reshape(-1, 4, 3)
        cols = {"rule": np.einsum("nk,nkc->nc", rec[:, 22:26], taps)}
        if kind == "real":
            orc = np.load(f"{d}/{sid}.oracle.npy")
            cols["oracle4"] = np.einsum("nk,nkc->nc", orc[:, :4], taps)
            cols["oracle6"] = np.einsum("nk,nkc->nc", orc[:, 4:8], taps) + orc[:, 8:9] * rec[:, 28:31] + orc[:, 9:10] * rec[:, 31:34]
        nc, ns = net_colours(rec, features(rec))
        cols.update(nc)
        split = s["split"] if kind == "real" else "val-teacher"
        for case in (s["case"],):
            acc[(split, g, case, "")].add(cols, y)
            if g == "forza" and split == "val":
                u = np.maximum(rec[:, 14], rec[:, 17])
                for lo, hi, nm in UBINS:
                    acc[(split, g, case, nm)].add(cols, y, (u >= lo) & (u < hi))
        for k, v in ns.items():
            st[(split, g, k)] += [v[0], v[1], len(rec)]
            st[(split, "all", k)] += [v[0], v[1], len(rec)]
        if i % 300 == 0:
            print(i, len(tasks), flush=True)
    res = {"/".join(k).rstrip("/"): {c: a.fig(c) for c in a.v} | {"px": a.n} for k, a in acc.items()}
    stj = {"/".join(k): {"mean_w45": float(v[0] / v[2]), "mean_abs_dw": float(v[1] / v[2]), "n": int(v[2])} for k, v in st.items()}
    json.dump({"scores": res, "weight_stats": stj}, open(f"{out}/scores.json", "w"), indent=1)

    def f(r, c):
        return f"{r[c]['psnr']:.2f} / {r[c]['mae']:.2f} / {100 * r[c]['big']:.2f}%" if c in r else "-"
    groups = GROUPS
    cases = ["30to60", "30to120", "60to120"]
    L = ["# Learned-blend scores (all rows of every sample, real targets unless marked)", "",
         "Cell: PSNR dB (pooled MSE over RGB, levels) / MAE / share of pixels with any channel error > 30.", ""]
    for setname in ("real_only", "final", "with_teacher"):
        if f"{setname}/A" not in NETS:
            continue
        names = ["rule", f"{setname}/A", f"{setname}/B", f"{setname}/C", "oracle4", "oracle6"]
        for split in ("val", "train"):
            L += [f"## {setname}: {split}", "", "| group | case | px | rule | A | B | C | oracle 4 | oracle 6 |", "|" + "---|" * 9]
            for g in groups:
                for case in cases:
                    r = res.get(f"{split}/{g}/{case}")
                    if r:
                        L.append(f"| {g} | {case} | {r['px']} | " + " | ".join(f(r, c) for c in names) + " |")
            L.append("")
        L += [f"## {setname}: Forza val 30to60 by u = max(uA, uB)", "", "| bin | px | rule | A | B | C | oracle 4 | oracle 6 |", "|" + "---|" * 8]
        for nm in [b[2] for b in UBINS]:
            r = res.get(f"val/forza/30to60/{nm}")
            if r:
                L.append(f"| {nm} | {r['px']} | " + " | ".join(f(r, c) for c in names) + " |")
        L.append("")
        L += [f"## {setname}: teacher-target validation (60 fps cuts), against the teacher's frame", "", "| group | case | px | rule | A | B | C |", "|" + "---|" * 7]
        for g in groups:
            for case in cases:
                r = res.get(f"val-teacher/{g}/{case}")
                if r:
                    L.append(f"| {g} | {case} | {r['px']} | " + " | ".join(f(r, c) for c in names[:4]) + " |")
        L.append("")
    L += ["## Weight statistics (mean w4 + w5 / mean |w - rule| over the four tap weights)", "",
          "| net | val forza | val all groups | train all |", "|---|---|---|---|"]
    for k in NETS:
        row = [stj.get(f"{a}/{k}") for a in ("val/forza", "val/all", "train/all")]
        L.append(f"| {k} | " + " | ".join(f"{r['mean_w45']:.4f} / {r['mean_abs_dw']:.4f}" if r else "-" for r in row) + " |")
    L.append("")
    open(f"{out}/scores.md", "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
