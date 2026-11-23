"""Trains the three learned-blend nets A, B, C together (same batches, one gather per step).
usage: train.py OUTDIR [--teacher] [--steps N] [--seed S] [--threads T]
  OUTDIR gets A.npz B.npz C.npz (+ log.json). --teacher adds the cached teacher training rows (weight 0.5 per row).
Net = gpu/ml/blend/ref.py: 8 hidden units (max(z, 0.1 z)), softmax(log(max(rule, 1e-3)) [+ -5, -5] + z2), W2 = b2 = 0 at the start.
A: Charbonnier distance of the colour to the four-choice oracle colour; B: to the target; C: six choices, to the six-choice oracle colour.
Batches: 65536 rows, 40 % forza, 60 % evenly over the other groups; Adam 2e-3 -> 1e-4 cosine; validation (real rows, group balanced the same way) every 500 steps, best kept.
"""
import sys, os, json, time, math
import numpy as np
import torch
from feat import *

args = sys.argv[1:]
out = args[0] if args else ""
use_teacher = "--teacher" in args
steps = int(args[args.index("--steps") + 1]) if "--steps" in args else 15000
seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 0
torch.set_num_threads(int(args[args.index("--threads") + 1]) if "--threads" in args else 4)
BATCH, H = 65536, 8
LR0 = float(args[args.index("--lr") + 1]) if "--lr" in args else 2e-3
VARS = {"A": (25, 4, "O4"), "B": (25, 4, "Y"), "C": (34, 6, "O6")}


def load(name):
    return {k: np.load(f"{CACHE}/{name}_{k}.npy") for k in ("X", "R", "Y", "O4", "O6", "G", "T", "S")}


def concat(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def forward(p, X, R, K, NI):
    W1, b1, W2, b2 = p
    z1 = X[:, :NI] @ W1.T + b1
    z2 = torch.where(z1 > 0, z1, 0.1 * z1) @ W2.T + b2
    lg = torch.log(torch.clamp(R, min=1e-3))
    if K == 6:
        lg = torch.cat([lg, torch.full((len(lg), 2), -5.0)], 1)
    w = torch.softmax(lg + z2, 1)
    C = (w[:, :4, None] * X[:, :12].view(-1, 4, 3)).sum(1)
    if K == 6:
        C = C + w[:, 4:5] * X[:, 25:28] + w[:, 5:6] * X[:, 28:31]
    return C, w


def charb(C, Y):
    return torch.sqrt((C - Y) ** 2 + 1e-6).mean(1)


def init(NI, K, rng):
    return [torch.tensor(rng.normal(0, 0.3 / math.sqrt(NI), (H, NI)), dtype=torch.float32, requires_grad=True),
            torch.zeros(H, requires_grad=True), torch.zeros(K, H, requires_grad=True), torch.zeros(K, requires_grad=True)]


def mix_counts(groups_present, B):
    nf = int(round(0.4 * B)) if 0 in groups_present else 0
    others = [g for g in groups_present if g != 0]
    c = {g: (B - nf) // len(others) for g in others}
    if nf:
        c[0] = B - sum(c.values())
    return c


def main():
    tr = load("real_train")
    if use_teacher:
        tr = concat(tr, load("teacher_train"))
    va = load("real_val")
    # leakage guard: every cached row's sample must be a train sample (cache builder asserted per file too)
    man = json.load(open(f"{BLEND}/frames/manifest.json"))["samples"]
    split_of = np.array([s["split"] == "val" for s in man])
    assert not split_of[tr["S"]].any(), "validation row in the training set"
    assert split_of[va["S"]].all()
    rows = {}
    for nm, d in (("train", tr), ("val", va)):
        for t in (0, 1):
            rows[(nm, t)] = {GROUPS[g]: int(((d["G"] == g) & (d["T"] == t)).sum()) for g in range(len(GROUPS))}
    print("rows per group", json.dumps({f"{a}/{'teacher' if t else 'real'}": v for (a, t), v in rows.items() if sum(v.values())}), flush=True)
    pools = {g: np.nonzero(tr["G"] == g)[0] for g in range(len(GROUPS)) if (tr["G"] == g).any()}
    vpools = {g: np.nonzero(va["G"] == g)[0] for g in range(len(GROUPS)) if (va["G"] == g).any()}
    cnt = mix_counts(list(pools), BATCH)
    T = {k: torch.from_numpy(v) for k, v in tr.items() if k in ("X", "R", "Y", "O4", "O6", "T")}
    V = {k: torch.from_numpy(v) for k, v in va.items() if k in ("X", "R", "Y", "O4", "O6")}
    rng = np.random.default_rng(seed)
    P = {v: init(NI, K, rng) for v, (NI, K, _) in VARS.items()}
    opt = torch.optim.Adam([t for v in P.values() for t in v], lr=LR0)
    best = {v: (1e9, None) for v in VARS}
    log = []

    def validate():
        res = {}
        with torch.no_grad():
            for v, (NI, K, tgt) in VARS.items():
                per = {}
                for g, ix in vpools.items():
                    s = 0.0
                    for a in range(0, len(ix), 262144):
                        j = torch.from_numpy(ix[a:a + 262144])
                        C, _ = forward(P[v], V["X"][j], V["R"][j], K, NI)
                        s += charb(C, V[tgt][j]).sum().item()
                    per[g] = s / len(ix)
                # same 40 / 60 balance as the batches
                res[v] = 0.4 * per[0] + 0.6 * np.mean([x for g, x in per.items() if g != 0]) if 0 in per else np.mean(list(per.values()))
        return res

    t0 = time.time()
    for step in range(steps + 1):
        if step % 500 == 0 or step == steps:
            vl = validate()
            for v in VARS:
                if vl[v] < best[v][0]:
                    best[v] = (vl[v], [t.detach().clone() for t in P[v]], step)
            log.append(dict(step=step, val=vl, sec=time.time() - t0))
            print(step, {k: round(x, 5) for k, x in vl.items()}, "%.0fs" % (time.time() - t0), flush=True)
        if step == steps:
            break
        lr = 1e-4 + 0.5 * (LR0 - 1e-4) * (1 + math.cos(math.pi * step / steps))
        for gp in opt.param_groups:
            gp["lr"] = lr
        idx = torch.from_numpy(np.concatenate([rng.choice(pools[g], c) for g, c in cnt.items()]))
        X, R, wt = T["X"][idx], T["R"][idx], 1.0 - 0.5 * T["T"][idx].float()
        loss = 0.0
        for v, (NI, K, tgt) in VARS.items():
            C, _ = forward(P[v], X, R, K, NI)
            loss = loss + (charb(C, T[tgt][idx]) * wt).sum() / wt.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    os.makedirs(out, exist_ok=True)
    for v in VARS:
        W1, b1, W2, b2 = (t.numpy().astype(np.float32) for t in best[v][1])
        np.savez(f"{out}/{v}.npz", W1=W1, b1=b1, W2=W2, b2=b2)
    json.dump(dict(log=log, best={v: [best[v][0], best[v][2]] for v in VARS}, rows={f"{a}/{t}": r for (a, t), r in rows.items()},
                   steps=steps, seconds=time.time() - t0, teacher=use_teacher, batch_counts={GROUPS[g]: c for g, c in cnt.items()}),
              open(f"{out}/log.json", "w"), indent=1)
    print("best", {v: (round(best[v][0], 5), best[v][2]) for v in VARS})


if __name__ == "__main__":
    main()
