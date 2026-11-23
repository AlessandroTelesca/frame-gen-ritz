#!/usr/bin/env python3
"""Aggregate check_state.json (written by `teacher.py check`) into tables and write 4 triptychs
(worst + median case of each check, by teacher MSE) as PNG via PPM + ffmpeg.  GPU use: 4 model runs.
Run:  flock ... ccorral -- <python> report.py"""
import json, math, os, subprocess
import numpy as np
import teacher as T

st = json.load(open(f"{T.OUT}/check_state.json"))
grp = lambda k: "battlefield" if k.startswith("battlefield") else "minecraft120"
psnr = lambda mse: 10 * math.log10(255 ** 2 / max(mse, 1e-12))
CHK = {"120to60": (0, 2, 1), "30to60": (0, 4, 2)}
for chk in CHK:
    print(f"\n== {chk}: {'model(a, a+2) vs real a+1' if chk == '120to60' else 'model(a, a+4) vs real a+2'}")
    print(f"{'group':14s}{'n':>5s} | {'method':8s}{'PSNR dB':>9s}{'MAE':>8s}{'>30 share':>11s}")
    for g in ("minecraft120", "battlefield", "all"):
        ks = [k for k in st if g == "all" or grp(k) == g]
        for m in ("teacher", "avg"):
            v = np.array([st[k][chk][m] for k in ks])
            print(f"{g:14s}{len(ks):5d} | {m:8s}{psnr(v[:, 0].mean()):9.2f}{v[:, 1].mean():8.3f}{100 * v[:, 2].mean():10.2f}%")
    # per-anchor spread
    v = np.array([psnr(st[k][chk]["teacher"][0]) for k in st])
    print(f"per-anchor teacher PSNR: min {v.min():.2f}  median {np.median(v):.2f}  max {v.max():.2f}")

os.makedirs(f"{T.OUT}/img", exist_ok=True)
net, _ = T.load_net()
for chk, (i, j, k) in CHK.items():
    order = sorted(st, key=lambda q: st[q][chk]["teacher"][0])
    for tag, key in (("worst", order[-1]), ("median", order[len(order) // 2])):
        c, a0 = key.rsplit(":", 1); a0 = int(a0)
        fa, fb, real = T.read(c, a0 + i), T.read(c, a0 + j), T.read(c, a0 + k)
        tch = T.interp(net, fa, fb)
        tri = np.concatenate([fa, tch, real], 1)[..., :3]
        ppm = f"{T.OUT}/img/tmp.ppm"
        with open(ppm, "wb") as f:
            f.write(b"P6\n%d %d\n255\n" % (tri.shape[1], tri.shape[0])); f.write(tri.tobytes())
        out = f"{T.OUT}/img/{chk}_{tag}_{c}_{a0}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", ppm, "-vf", "scale=iw/2:ih/2", out], check=True)
        os.remove(ppm)
        print("wrote", out, f"(MSE {st[key][chk]['teacher'][0]:.1f})")
