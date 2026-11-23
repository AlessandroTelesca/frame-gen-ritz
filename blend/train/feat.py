"""Shared pieces of the learned-blend training: features, paths. The network definition is gpu/ml/blend/ref.py."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from common import BLEND, group  # noqa

TRAIN = f"{BLEND}/train"
CACHE = f"{TRAIN}/cache"
GROUPS = ["forza", "cs2", "schedule1", "minecraft_rtx", "minecraft_vanilla", "portland", "minecraft120", "battlefield"]


def manifest():
    return {s["id"]: s for s in json.load(open(f"{BLEND}/frames/manifest.json"))["samples"]}


def _m(v):
    return 4.0 * np.abs(v).mean(axis=1)


def features(rec):
    """rec [n, >=34] float32 -> X [n, 34] float32 (the 34-input layout of SPEC section 4; the 25-input nets use X[:, :25])."""
    rec = np.asarray(rec, np.float32)
    c = [rec[:, 3 * i:3 * i + 3] / np.float32(255) for i in range(4)]
    f0, f1 = rec[:, 28:31] / np.float32(255), rec[:, 31:34] / np.float32(255)
    X = np.concatenate([np.concatenate(c, axis=1), rec[:, 12:19], np.minimum(rec[:, 19:21], 256) / 64,
                        np.stack([_m(c[0] - c[1]), _m(c[2] - c[3]), _m(c[0] - c[2]), _m(c[1] - c[3])], axis=1),
                        f0, f1, np.stack([_m(f0 - f1), _m(c[0] - f0), _m(c[3] - f1)], axis=1)], axis=1)
    return X.astype(np.float32)
