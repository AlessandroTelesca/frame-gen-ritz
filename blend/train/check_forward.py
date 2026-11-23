"""Check 1: the torch forward pass of train.py equals gpu/ml/blend/ref.py (random weights, rows of one real record file). Also checks feat.features vs ref's inputs."""
import sys, os, glob
import numpy as np, torch
from feat import *
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gpu", "ml", "blend"))
sys.argv = sys.argv[:1]
import ref
import train

files = sorted(glob.glob(f"{BLEND}/records/racing_forza6_c_*.npy"))
files = [f for f in files if ".oracle" not in f][:1] + [f for f in sorted(glob.glob(f"{BLEND}/records/minecraft120_01_*.npy")) if ".oracle" not in f][:1]
rng = np.random.default_rng(0)
worst = 0.0
for f in files:
    rec = np.load(f)
    X = torch.from_numpy(features(rec))
    for NI, K in ((25, 4), (34, 6)):
        z = dict(W1=rng.uniform(-.5, .5, (8, NI)).astype(np.float32), b1=rng.uniform(-.5, .5, 8).astype(np.float32),
                 W2=rng.uniform(-.5, .5, (K, 8)).astype(np.float32), b2=rng.uniform(-.5, .5, K).astype(np.float32))
        w, w45 = ref.blend_weights(rec, z, rec[:, 28:31], rec[:, 31:34])
        p = [torch.from_numpy(z[k]) for k in ("W1", "b1", "W2", "b2")]
        C, wt = train.forward(p, X, torch.from_numpy(rec[:, 22:26]), K, NI)
        wt = wt.numpy()
        d = max(np.abs(wt[:, :4] - w).max(), np.abs((wt[:, 4] + wt[:, 5] if K == 6 else 0 * w45) - w45).max())
        print(os.path.basename(f), "n", len(rec), "NI", NI, "K", K, "max|dw| %.3g" % d)
        worst = max(worst, d)
print("worst", worst, "OK" if worst < 1e-5 else "FAIL")
