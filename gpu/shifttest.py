#!/usr/bin/env python3
"""Known-shift test of the long-vector paths (Settings::searchRadius / searchRadiusTop / pyramidMinSide at their largest).

  shifttest.py FGTEST [--validate] [--keep DIR]        exit status 0 = all cases pass

Builds a 1/f-noise picture (broad-band structure, no repetition), curr = prev shifted by a known whole-pixel amount (the content that
leaves the frame is replaced by the picture's next columns / rows, as a real pan), runs fgtest on it (V1, V2 Quality, V2 Performance)
and requires, in the region visible in both frames:
  * the estimated half-vector (--dump: V1 / V2 Quality per-pixel field, V2 Performance block field) within 1 flow pixel of the truth
    shift / (2 * flowScale) on at least 95 % of the pixels / blocks;
  * the generated frame at t = 0.5 equal to the true half-shifted picture (PSNR >= 35 dB).
Cases: (1) flowScale 2, shift (400, 100) at searchRadius 12 / searchRadiusTop 12 / pyramidMinSide 16: beyond the default reach
(~250 px), below the two 127-px clip limits of the histogram and the V2 Quality claims; (2) flowScale 1, shift (600, 300): beyond both,
so the wide median histogram and the wide claims decide; (3) shift (1) with the default settings must NOT be found.
--validate runs fgtest under the Khronos validation layer (GPU-assisted and synchronization validation) and fails on any message
other than the layer's start-up notes.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fgmaps  # noqa: E402

W, H = 1920, 1080
WIDE = "searchRadius=12,searchRadiusTop=12,pyramidMinSide=16"


def noise(shape, rng):
    """1/f noise: white noise filtered with an amplitude spectrum 1 / f, plus a little white noise (fine detail)."""
    f = np.fft.fft2(rng.standard_normal(shape))
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    r = np.sqrt(fx * fx + fy * fy)
    r[0, 0] = 1.0
    n = np.fft.ifft2(f / r).real
    n = (n - n.mean()) / n.std()
    return n + 0.15 * rng.standard_normal(shape)


def scene(dx, dy):
    """(prev, curr, truth_t05) uint8 RGBA H x W x 4; curr(x) = prev(x - d), truth = prev shifted by d / 2."""
    rng = np.random.default_rng(12345)
    cw, ch = W + abs(dx) + 16, H + abs(dy) + 16
    can = np.stack([noise((ch, cw), rng) for _ in range(3)], -1)
    can = np.clip(128 + 42 * can, 0, 255).astype(np.uint8)
    ox, oy = max(dx, 0) + 8, max(dy, 0) + 8

    def crop(sx, sy):
        out = np.full((H, W, 4), 255, np.uint8)
        out[..., :3] = can[oy - sy:oy - sy + H, ox - sx:ox - sx + W]
        return out
    return crop(0, 0), crop(dx, dy), crop(dx // 2, dy // 2)


def run(fgtest, tmp, name, settings, prev, curr, validate):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    pf, cf, of = (os.path.join(d, n) for n in ("prev.rgba", "curr.rgba", "out.rgba"))
    prev.tofile(pf)
    curr.tofile(cf)
    env = dict(os.environ)
    if validate:
        env["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
        env["VK_LAYER_ENABLES"] = "VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT:VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT"
    p = subprocess.run([fgtest, "--set", settings, "--t", "0.5", "--dump", os.path.join(d, "maps"), str(W), str(H), pf, cf, of],
                       capture_output=True, text=True, env=env)
    # (the layer's start-up notes - VALIDATION-SETTINGS, Setting-Limit-Adjusted, GPU-AV + core check - are warnings of the settings, not errors)
    msgs = [l for l in (p.stdout + p.stderr).splitlines()
            if any(k in l for k in ("Validation Error", "Validation Warning", "VUID", "SYNC-HAZARD", "UNASSIGNED"))
            and not any(k in l for k in ("VALIDATION-SETTINGS", "Setting-Limit-Adjusted"))]
    out = np.fromfile(of, np.uint8).reshape(H, W, 4) if p.returncode == 0 and os.path.exists(of) else None
    maps = fgmaps.load(os.path.join(d, "maps", "pair_0000.fgmaps")) if p.returncode == 0 else None
    return p, msgs, out, maps


def check(label, maps, out, truth, dx, dy, perf):
    """-> (fraction of valid pixels / blocks within 1 flow px, psnr of the t = 0.5 frame in the visible interior)"""
    fs = maps["fscale"]
    hx, hy = dx / (2.0 * fs), dy / (2.0 * fs)     # true half-vector in flow pixels
    if perf:
        f, step, off = maps["block"], 8, 4        # block (bx, by) centre at flow (8 bx + 4, 8 by + 4)
    else:
        f, step, off = maps["pf"], 1, 0
    fh_, fw_ = f.shape[:2]
    ys, xs = np.mgrid[0:fh_, 0:fw_]
    cx, cy = xs * step + off, ys * step + off
    m = 8   # margin (flow px) from the strip that leaves the frame
    ok = (cx - abs(hx) >= m) & (cx + abs(hx) <= maps["fw"] - 1 - m) & (cy - abs(hy) >= m) & (cy + abs(hy) <= maps["fh"] - 1 - m)
    good = (np.abs(f[..., 0] - hx) <= 1.0) & (np.abs(f[..., 1] - hy) <= 1.0)
    frac = float(good[ok].mean())
    # t = 0.5 frame: visible where both prev(x - d/2) and curr(x + d/2) are inside the frame
    ys, xs = np.mgrid[0:H, 0:W]
    mm = 16
    vis = (xs - abs(dx) // 2 >= mm) & (xs + abs(dx) // 2 <= W - 1 - mm) & (ys - abs(dy) // 2 >= mm) & (ys + abs(dy) // 2 <= H - 1 - mm)
    e = (out[..., :3].astype(np.float64) - truth[..., :3].astype(np.float64))[vis]
    mse = float((e * e).mean())
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    print(f"  {label}: {100 * frac:5.1f}% of {int(ok.sum())} {'blocks' if perf else 'pixels'} within 1 flow px of ({hx:g}, {hy:g}); "
          f"t=0.5 interior PSNR {psnr:5.1f} dB   (median field {np.median(f[ok][:, 0]):g}, {np.median(f[ok][:, 1]):g})")
    return frac, psnr


def main():
    a = sys.argv[1:]
    validate = "--validate" in a
    keep = a[a.index("--keep") + 1] if "--keep" in a else None
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] != "--keep")]
    if not pos:
        print(__doc__)
        return 2
    fgtest = os.path.abspath(pos[0])
    fails = 0
    tmpctx = tempfile.TemporaryDirectory(dir=keep) if not keep else None
    tmp = keep or tmpctx.name
    # (name, flowScale, shift, extra settings, must be found)
    cases = [("1 fs2 (400, 100), radius 12/12/16", 2, (400, 100), WIDE, True),
             ("2 fs1 (600, 300), radius 12/12/16", 1, (600, 300), WIDE, True),
             ("3 fs2 (400, 100), default settings", 2, (400, 100), "", False)]
    scenes = {}
    for cname, fs, (dx, dy), extra, found in cases:
        if (dx, dy) not in scenes:
            scenes[(dx, dy)] = scene(dx, dy)
        prev, curr, truth = scenes[(dx, dy)]
        print(f"case {cname}" + ("" if found else "  (must NOT be found)"))
        for label, algo in (("V1", "algo=v1"), ("V2 Quality", "algo=v2,v2Mode=quality"), ("V2 Performance", "algo=v2,v2Mode=performance")):
            # (the strip a pan reveals is more than the default 15 % of the frame: the whole-frame fallbacks are off, they are not under test)
            s = f"{algo},flowScale={fs},globalFallback=1,sceneCutSad=255" + ("," + extra if extra else "")
            p, msgs, out, maps = run(fgtest, tmp, f"c{cname[0]}_{label[:2]}{label[-1]}", s, prev, curr, validate)
            if p.returncode != 0:
                print(f"  {label}: fgtest failed ({p.returncode}): {p.stderr[-300:]}")
                fails += 1
                continue
            frac, psnr = check(label, maps, out, truth, dx, dy, label == "V2 Performance")
            if msgs:
                fails += 1
                print(f"  {label}: {len(msgs)} validation message(s):")
                for l in msgs[:6]:
                    print("    " + l)
            if found:
                # (V2 Performance's frame at t = 0.5: only the field is required, its synthesis is not the plain blend of the quality path)
                good = frac >= 0.95 and (psnr >= 35.0)
                if not good:
                    fails += 1
                    print(f"  {label}: FAIL (need >= 95% and PSNR >= 35 dB)")
            elif frac >= 0.5:
                fails += 1
                print(f"  {label}: FAIL: the default settings DO find this shift ({100 * frac:.0f}%): the test premise is wrong")
    if tmpctx:
        tmpctx.cleanup()
    print("shifttest:", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
