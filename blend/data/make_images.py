"""Comparison PNGs for 6 Forza validation 30->60 samples: the 2 with the largest rule-vs-oracle4 gap (from the sparse records) and 4 spread
over the clip. For each, a full-resolution weight dump (--v3wmax 2073600, deleted after use) gives the rule's colour and the exact oracles.
Writes records/img/<id>_{target,rule,oracle4,oracle6,err_rule,err_oracle4}.png (errors x4, clipped).   usage: make_images.py"""
import json
import os
import struct
import subprocess
import sys
import zlib

import numpy as np

from common import *
from dump import FGTEST, LOCK, frame_path, target_path
from oracle import solve, points


def png(path, a):
    a = np.ascontiguousarray(a, np.uint8)
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b""))


def main():
    man = json.load(open(f"{BLEND}/frames/manifest.json"))["samples"]
    idx = json.load(open(f"{BLEND}/records/index.json"))
    cand = [s for s in man if s["clip"] == "racing_forza6_c" and s["case"] == "30to60" and s["id"] in idx]
    gap = {}
    for s in cand:
        rec = np.load(f"{BLEND}/records/{s['id']}.npy")
        orc = np.load(f"{BLEND}/records/{s['id']}.oracle.npy")
        c = np.einsum("nk,nkc->nc", rec[:, 22:26], rec[:, 0:12].reshape(-1, 4, 3))
        o = np.einsum("nk,nkc->nc", orc[:, :4], rec[:, 0:12].reshape(-1, 4, 3))
        y = rec[:, 34:37]
        gap[s["id"]] = float(((c - y) ** 2).mean() - ((o - y) ** 2).mean())
    top = sorted(cand, key=lambda s: -gap[s["id"]])[:2]
    rest = [s for s in cand if s not in top]
    spread = [rest[int(i)] for i in np.linspace(0, len(rest) - 1, 4)]
    os.makedirs(f"{BLEND}/records/img", exist_ok=True)
    scratch = f"{BLEND}/records/img/.scratch"
    os.makedirs(scratch, exist_ok=True)
    info = {}
    for s in top + spread:
        sid = s["id"]
        pre = f"{scratch}/full"
        cmd = [FGTEST, "--window", "1", "--set", "algo=v3", "--t", repr(s["t"]), "--v3weights", pre, "--v3wmax", str(W * H), str(W), str(H),
               frame_path(s, s["prev"]), frame_path(s, s["curr"]), "/dev/null"]
        r = subprocess.run(["flock", LOCK, "ccorral", "--"] + cmd, capture_output=True, text=True)
        rec = np.fromfile(f"{pre}_0000.bin", np.float32).reshape(-1, 28)
        assert len(rec) == W * H, (len(rec), r.stdout, r.stderr)
        tgt = np.fromfile(target_path(s), np.uint8).reshape(-1, 4)[:, :3].astype(np.float64)
        taps = rec[:, 0:12].reshape(-1, 4, 3).astype(np.float64)
        e = np.stack([np.fromfile(frame_path(s, s["prev"]), np.uint8).reshape(-1, 4)[:, :3], np.fromfile(frame_path(s, s["curr"]), np.uint8).reshape(-1, 4)[:, :3]], 1).astype(np.float64)
        six = np.concatenate([taps, e], 1)
        o4 = np.empty((W * H, 3)); o6 = np.empty((W * H, 3))
        for a in range(0, W * H, 1 << 17):
            sl = slice(a, a + (1 << 17))
            w4 = solve(taps[sl], tgt[sl])[0]; w6 = solve(six[sl], tgt[sl])[0]
            o4[sl] = np.einsum("nk,nkc->nc", w4, taps[sl]); o6[sl] = np.einsum("nk,nkc->nc", w6, six[sl])
        rule = np.einsum("nk,nkc->nc", rec[:, 22:26].astype(np.float64), taps)
        # the pictures show the weight-stage colour (before clamp / smear / static mask), same as the numbers
        sh = (H, W, 3)
        clip8 = lambda a: np.clip(np.rint(a), 0, 255).reshape(sh)
        png(f"{BLEND}/records/img/{sid}_target.png", clip8(tgt))
        png(f"{BLEND}/records/img/{sid}_rule.png", clip8(rule))
        png(f"{BLEND}/records/img/{sid}_oracle4.png", clip8(o4))
        png(f"{BLEND}/records/img/{sid}_oracle6.png", clip8(o6))
        png(f"{BLEND}/records/img/{sid}_err_rule.png", clip8(4 * np.abs(rule - tgt)))
        png(f"{BLEND}/records/img/{sid}_err_oracle4.png", clip8(4 * np.abs(o4 - tgt)))
        mse = lambda c: float(((c - tgt) ** 2).mean() / 3)
        info[sid] = {"mse_rule": mse(rule), "mse_oracle4": mse(o4), "mse_oracle6": mse(o6), "sparse_gap_mse": gap[sid],
                     "mae_rule": float(np.abs(rule - tgt).mean()), "mae_oracle4": float(np.abs(o4 - tgt).mean()), "mae_oracle6": float(np.abs(o6 - tgt).mean()),
                     "psnr_rule": 10 * np.log10(255 ** 2 / mse(rule)), "psnr_oracle4": 10 * np.log10(255 ** 2 / mse(o4)), "psnr_oracle6": 10 * np.log10(255 ** 2 / mse(o6)),
                     "why": "top-gap" if s in top else "spread"}
        os.remove(f"{pre}_0000.bin")
        print(sid, info[sid], flush=True)
    json.dump(info, open(f"{BLEND}/records/img/info.json", "w"), indent=1)
    os.rmdir(scratch)


main()
