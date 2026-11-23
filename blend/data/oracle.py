"""Exact per-pixel oracle blends.  For K colour points P_k (levels) and a target y: the weights on the simplex
(w >= 0, sum w = 1) minimising |sum_k w_k P_k - y|^2.

Method: the optimum lies in the relative interior of one face of the simplex, where it is the least-squares solution on the affine hull
of that face's points. So: for every non-empty subset S (2^K - 1: 15 for K=4, 63 for K=6) solve the equality-constrained least squares
(sum w = 1) by its KKT system on the 3x3 Gram data, keep the feasible solutions (all w >= 0), take the lowest error.  A tiny ridge
makes dependent subsets (collinear / equal points) solvable; singletons are always feasible so a candidate always exists.

usage: oracle.py [--kind real|teacher] [--procs N] [--force] | oracle.py --selftest
Writes records/<id>.oracle.npy float32 [n, 10]: 4 weights of oracle 4 (taps c0..c3), then 6 weights of oracle 6 (c0..c3, prev(x), curr(x)).
"""
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

SUBSETS = {K: [S for r in range(1, K + 1) for S in itertools.combinations(range(K), r)] for K in (4, 6)}


def solve(P, y):
    """P: [n, K, 3], y: [n, 3] (float64). Returns (w [n, K], err [n])."""
    n, K, _ = P.shape
    G = np.einsum("nic,njc->nij", P, P)
    b = np.einsum("nic,nc->ni", P, y)
    yy = np.einsum("nc,nc->n", y, y)
    best = np.full(n, np.inf)
    bestw = np.zeros((n, K))
    scale = np.trace(G, axis1=1, axis2=2) / K + 1.0
    for S in SUBSETS[K]:
        s = len(S)
        ix = np.array(S)
        if s == 1:
            w = np.ones((n, 1))
        else:
            M = np.zeros((n, s + 1, s + 1))
            M[:, :s, :s] = G[:, ix[:, None], ix[None, :]]
            M[:, np.arange(s), np.arange(s)] += (1e-9 * scale)[:, None]
            M[:, :s, s] = 1.0
            M[:, s, :s] = 1.0
            rhs = np.zeros((n, s + 1))
            rhs[:, :s] = b[:, ix]
            rhs[:, s] = 1.0
            w = np.linalg.solve(M, rhs[:, :, None])[:, :s, 0]
        ok = (w >= -1e-9).all(axis=1)
        w = np.where(ok[:, None], np.maximum(w, 0.0), 0.0)
        sw = w.sum(axis=1)
        ok &= sw > 0.5
        w = w / np.maximum(sw, 1e-12)[:, None]
        err = np.einsum("ni,nij,nj->n", w, G[:, ix[:, None], ix[None, :]], w) - 2 * np.einsum("ni,ni->n", w, b[:, ix]) + yy
        err = np.where(ok, err, np.inf)
        upd = err < best
        best = np.where(upd, err, best)
        bestw[upd] = 0.0
        full = np.zeros((n, K))
        full[:, ix] = w
        bestw[upd] = full[upd]
    return bestw, best


def points(rec):
    """rec [n, 40] -> taps [n, 4, 3], six points [n, 6, 3], target [n, 3] (float64)."""
    taps = rec[:, 0:12].reshape(-1, 4, 3).astype(np.float64)
    e = np.stack([rec[:, 28:31], rec[:, 31:34]], axis=1).astype(np.float64)
    return taps, np.concatenate([taps, e], axis=1), rec[:, 34:37].astype(np.float64)


def oracle_for(rec, step=65536):
    out = np.empty((len(rec), 10), np.float32)
    for a in range(0, len(rec), step):
        taps, six, y = points(rec[a:a + step])
        out[a:a + step, :4] = solve(taps, y)[0]
        out[a:a + step, 4:] = solve(six, y)[0]
    return out


def _job(args):
    path, force = args
    op = path[:-4] + ".oracle.npy"
    if os.path.exists(op) and not force:
        return path, 0
    rec = np.load(path)
    o = oracle_for(rec)
    np.save(op + ".part.npy", o)
    os.replace(op + ".part.npy", op)
    return path, len(rec)


def selftest():
    rng = np.random.default_rng(1)
    for K in (4, 6):
        n = 300
        P = rng.uniform(0, 255, (n, K, 3))
        P[:60, 1] = P[:60, 0]                       # equal points
        P[60:100] = P[60:100, :1] * 0 + P[60:100, :1] + rng.uniform(-3, 3, (100 - 60, K, 3))  # clustered
        y = rng.uniform(0, 255, (n, 3))
        y[100:200] = (P[100:200] * rng.dirichlet(np.ones(K), 100)[:, :, None]).sum(1)   # inside the hull
        y[200:250] = P[200:250, 0] * 1.5 - 0.5 * P[200:250, 1]                            # outside
        w, err = solve(P, y)
        assert np.allclose(w.sum(1), 1) and (w >= -1e-12).all()
        # brute force: coarse simplex grid (step 1/20 for K=4, 1/8 for K=6) then local refinement by random simplex steps
        res = 20 if K == 4 else 8
        grid = np.array([c for c in itertools.product(range(res + 1), repeat=K) if sum(c) == res], float) / res
        worst = 0.0
        for i in range(n):
            e = ((grid @ P[i] - y[i]) ** 2).sum(1)
            gb = e.min()
            assert err[i] <= gb + 1e-6 * (1 + gb), (K, i, err[i], gb)
            worst = max(worst, (gb - err[i]))
            # the claimed weights reproduce the claimed error
            assert abs(((w[i] @ P[i] - y[i]) ** 2).sum() - err[i]) < 1e-5 * (1 + err[i]), (K, i)
        # hull-inside cases reach (near) zero
        assert (err[100:200] < 1e-3).all(), err[100:200].max()
        # random local perturbation never improves on the oracle
        for _ in range(200):
            d = rng.dirichlet(np.ones(K), n) * 0.3
            w2 = (1 - 0.3) * w + d
            e2 = ((np.einsum("nk,nkc->nc", w2, P) - y) ** 2).sum(1)
            assert (e2 >= err - 1e-6 * (1 + err)).all()
        print(f"K={K}: ok over {n} random cases (largest grid-minus-oracle gap {worst:.2f})")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    from common import BLEND
    rd = f"{BLEND}/records"
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 4
    force = "--force" in sys.argv
    idx = json.load(open(f"{rd}/index.json"))
    kind = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else "real"
    paths = [f"{rd}/{k}.npy" for k, v in idx.items() if v["kind"] == kind]
    with Pool(procs) as p:
        done = 0
        for path, n in p.imap_unordered(_job, [(q, force) for q in paths]):
            done += 1
            if done % 100 == 0:
                print(done, "/", len(paths), flush=True)
    print("oracle done", len(paths))


if __name__ == "__main__":
    main()
