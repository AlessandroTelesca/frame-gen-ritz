"""Diagnosis: why do the learned-blend nets stop at the rule?  Size, loss or inputs.
usage: diag.py RUN [--steps N] [--bench] [--threads T]      RUN in base mse h32 h64x2 h64x2+ h64x2mse
Trains the four-choice (NI 25) and six-choice (NI 34) form of one configuration on the real-target cache rows only (same batches for both forms),
validation on the cached real validation rows (group balanced 40 % Forza as train.py), best validation step kept, then scores the best weights on all
cached validation rows and on a fixed 3 M-row subsample of the cached training rows. Writes $BLEND/train/diag/RUN.json.
diag.py report  -> $BLEND/train/diag/diag.md / diag.json from the RUN.json files.
Everything else as train.py: prior log(max(rule, 1e-3)) (+ -5, -5 for the two extra choices), last layer 0, batches of 65536, Adam, cosine.
Hidden units: max(z, 0.1 z).  '+' inputs: the rule's four weights, sigma * 20, s (+ prev(x), curr(x) and their three differences for the four-choice form).
Needs extras.py first (sigma, s).
"""
import sys, os, json, time, math
import numpy as np
import torch
from feat import *
import train as TR

CFG = {  # run: (hidden, loss, plus, lr0, lr1)
    "base": ([8], "charb", False, 2e-3, 1e-4), "mse": ([8], "mse", False, 2e-3, 1e-4), "h32": ([32], "charb", False, 2e-3, 1e-4),
    "h64x2": ([64, 64], "charb", False, 1e-3, 5e-5), "h64x2+": ([64, 64], "charb", True, 1e-3, 5e-5), "h64x2mse": ([64, 64], "mse", False, 1e-3, 5e-5)}
FORMS = {"4": (25, 4), "6": (34, 6)}
DIAG = f"{TRAIN}/diag"
BATCH = int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 65536
CASES = ["30to60", "30to120", "60to120"]
UBINS = [(0.0, 0.2, "u<0.2"), (0.2, 0.6, "0.2<=u<=0.6"), (0.6, 9.0, "u>0.6")]


def load(split):
    d = {k: np.load(f"{CACHE}/real_{split}_{k}.npy") for k in ("X", "R", "Y", "O4", "O6", "G", "S")}
    d["E"] = np.load(f"{DIAG}/extra_{split}.npy")
    return d


def extra_in(form, X, R, E):
    """The additional inputs of the '+' runs."""
    parts = [R, E[:, :1] * 20, E[:, 1:2]]
    if form == "4":
        parts.append(X[:, 25:34])
    return torch.cat(parts, 1)


def inputs(form, plus, X, R, E):
    NI = FORMS[form][0]
    return torch.cat([X[:, :NI], extra_in(form, X, R, E)], 1) if plus else X[:, :NI]


def init(nin, hidden, K, rng):
    P, a = [], nin
    for h in hidden:
        P += [torch.tensor(rng.normal(0, 0.3 / math.sqrt(a) if a == nin else math.sqrt(2 / a), (h, a)), dtype=torch.float32, requires_grad=True),
              torch.zeros(h, requires_grad=True)]
        a = h
    P += [torch.zeros(K, a, requires_grad=True), torch.zeros(K, requires_grad=True)]
    return P


def forward(P, inp, X, R, K):
    h = inp
    for i in range(0, len(P) - 2, 2):
        z = h @ P[i].T + P[i + 1]
        h = torch.where(z > 0, z, 0.1 * z)
    z2 = h @ P[-2].T + P[-1]
    lg = torch.log(torch.clamp(R, min=1e-3))
    if K == 6:
        lg = torch.cat([lg, torch.full((len(lg), 2), -5.0), ], 1)
    w = torch.softmax(lg + z2, 1)
    C = (w[:, :4, None] * X[:, :12].view(-1, 4, 3)).sum(1)
    if K == 6:
        C = C + w[:, 4:5] * X[:, 25:28] + w[:, 5:6] * X[:, 28:31]
    return C


def lossf(kind, C, Y):
    return torch.sqrt((C - Y) ** 2 + 1e-6).mean(1) if kind == "charb" else ((C - Y) ** 2).mean(1)


def colours(P, form, plus, d, ix, K):
    out = []
    with torch.no_grad():
        for a in range(0, len(ix), 262144):
            j = torch.from_numpy(ix[a:a + 262144])
            X, R, E = d["X"][j], d["R"][j], d["E"][j]
            out.append(forward(P, inputs(form, plus, X, R, E), X, R, K))
    return torch.cat(out).numpy().astype(np.float64) * 255


class Acc:
    def __init__(self):
        self.n, self.v = 0, {}

    def add(self, cols, y, mask):
        if not mask.any():
            return
        y = y[mask]
        self.n += len(y)
        for k, c in cols.items():
            dd = c[mask] - y
            a = self.v.setdefault(k, np.zeros(3))
            a += [(dd * dd).sum(), np.abs(dd).sum(), (np.abs(dd).max(axis=1) > 30).sum()]

    def fig(self):
        return {k: {"psnr": float(10 * np.log10(255 ** 2 / max(s / (3 * self.n), 1e-12))), "mae": float(a / (3 * self.n)), "big": float(b / self.n)}
                for k, (s, a, b) in self.v.items()} | {"px": self.n}


def score(d, ix, cols, case_of, split):
    """cols: name -> [len(ix), 3] levels. Returns {selector: {name: fig}}."""
    G, S = d["G"][ix], d["S"][ix]
    y = d["Y"][ix].astype(np.float64) * 255
    case = case_of[S]
    sel = {}
    groups = {g: (G == GROUPS.index(g)) for g in GROUPS}
    for g, m in groups.items():
        sel[f"{split}/{g}"] = m
    sel[f"{split}/forza/30to60"] = groups["forza"] & (case == "30to60")
    sel[f"{split}/all"] = np.ones(len(ix), bool)
    if split == "val":
        u = np.maximum(d["X"][ix, 14], d["X"][ix, 17])
        for lo, hi, nm in UBINS:
            sel[f"val/forza/30to60/{nm}"] = groups["forza"] & (case == "30to60") & (u >= lo) & (u < hi)
    out = {}
    for k, m in sel.items():
        a = Acc()
        a.add(cols, y, m)
        if a.n:
            out[k] = a.fig()
    return out


def main():
    run = sys.argv[1]
    hidden, loss, plus, lr0, lr1 = CFG[run]
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else (12000 if len(hidden) == 2 else 6000)
    bench = "--bench" in sys.argv
    torch.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 3)
    tr, va = load("train"), load("val")
    man = json.load(open(f"{BLEND}/frames/manifest.json"))["samples"]
    is_val = np.array([s["split"] == "val" for s in man])
    assert not is_val[tr["S"]].any(), "validation row in the training set"
    assert is_val[va["S"]].all()
    case_of = np.array([s["case"] for s in man])
    pools = {g: np.nonzero(tr["G"] == g)[0] for g in range(len(GROUPS)) if (tr["G"] == g).any()}
    vpools = {g: np.nonzero(va["G"] == g)[0] for g in range(len(GROUPS)) if (va["G"] == g).any()}
    cnt = TR.mix_counts(list(pools), BATCH)
    rng = np.random.default_rng(0)
    # validation subsample for the every-500-steps check (fixed, 60k per group at most)
    vsub = {g: np.sort(np.random.default_rng(1).choice(ix, min(len(ix), 60000), replace=False)) for g, ix in vpools.items()}
    T = {k: torch.from_numpy(v) for k, v in tr.items()}
    V = {k: torch.from_numpy(v) for k, v in va.items()}
    P = {}
    for f, (NI, K) in FORMS.items():
        nin = NI + (len(extra_in(f, T["X"][:2], T["R"][:2], T["E"][:2])[0]) if plus else 0)
        P[f] = init(nin, hidden, K, rng)
    opt = torch.optim.Adam([t for v in P.values() for t in v], lr=lr0)
    best = {f: (1e9, None, 0) for f in FORMS}
    log = []

    def validate():
        res = {}
        with torch.no_grad():
            for f, (NI, K) in FORMS.items():
                per = {}
                for g, ix in vsub.items():
                    j = torch.from_numpy(ix)
                    X, R, E = V["X"][j], V["R"][j], V["E"][j]
                    per[g] = lossf(loss, forward(P[f], inputs(f, plus, X, R, E), X, R, K), V["Y"][j]).mean().item()
                res[f] = 0.4 * per[0] + 0.6 * np.mean([x for g, x in per.items() if g != 0])
        return res

    t0 = time.time()
    for step in range(steps + 1):
        if (step % 500 == 0 or step == steps) and not bench:
            vl = validate()
            for f in FORMS:
                if vl[f] < best[f][0]:
                    best[f] = (vl[f], [t.detach().clone() for t in P[f]], step)
            log.append(dict(step=step, val=vl, sec=time.time() - t0))
            print(step, {k: round(x, 6) for k, x in vl.items()}, "%.0fs" % (time.time() - t0), flush=True)
        if step == steps:
            break
        if bench and step == 20:
            print("sec/step %.3f" % ((time.time() - t0) / 20))
            return
        lr = lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * step / steps))
        for gp in opt.param_groups:
            gp["lr"] = lr
        idx = torch.from_numpy(np.concatenate([rng.choice(pools[g], c) for g, c in cnt.items()]))
        X, R, E, Y = T["X"][idx], T["R"][idx], T["E"][idx], T["Y"][idx]
        L = 0.0
        for f, (NI, K) in FORMS.items():
            L = L + lossf(loss, forward(P[f], inputs(f, plus, X, R, E), X, R, K), Y).mean()
        opt.zero_grad()
        L.backward()
        opt.step()
    train_sec = time.time() - t0
    # scoring with the best weights
    res = {"run": run, "hidden": hidden, "loss": loss, "plus": plus, "steps": steps, "train_sec": train_sec, "log": log,
           "best": {f: [best[f][0], best[f][2]] for f in FORMS}, "scores": {}}
    trsub = np.sort(np.random.default_rng(2).choice(len(tr["G"]), 3_000_000, replace=False))
    for split, d, ix in (("val", va, np.arange(len(va["G"]))), ("train", tr, trsub)):
        dt = {k: torch.from_numpy(v) for k, v in d.items()}
        taps = d["X"][ix, :12].reshape(-1, 4, 3).astype(np.float64) * 255
        for f, (NI, K) in FORMS.items():
            cols = {"rule": np.einsum("nk,nkc->nc", d["R"][ix].astype(np.float64), taps), "oracle": d["O4" if f == "4" else "O6"][ix].astype(np.float64) * 255,
                    "net": colours(best[f][1], f, plus, dt, ix, K)}
            res["scores"][f] = res["scores"].get(f, {}) | score(d, ix, cols, case_of, split)
    json.dump(res, open(f"{DIAG}/{run}.json", "w"), indent=1)
    print("done", run, {f: round(res["scores"][f]["val/forza/30to60"]["net"]["psnr"], 3) for f in FORMS},
          "rule", round(res["scores"]["4"]["val/forza/30to60"]["rule"]["psnr"], 3), "train %.0fs" % train_sec)


if __name__ == "__main__":
    main()
