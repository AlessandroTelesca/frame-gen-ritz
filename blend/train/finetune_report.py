"""The result of a fine-tune (train.py --init): OUTDIR/scores.md and OUTDIR/run.json (called at the end of train.py, nothing to run by hand).
scores.md: for A, B and C, PSNR / MAE / share of pixels with an error above 30 levels against the real target frame (the definitions of score.py / diag.py: pooled
over RGB, levels 0..255; diag.Acc is used), for the rule, the base weights (--base, default blend/weights/base) and the fine-tuned weights, on
(i) the local validation rows (the held-out part of the local footage) and (ii) the validation rows of the --mix file, per group.
run.json: repository commit, options, row counts per group, steps, best step, training time, clips (names, frame counts, anchor counts). No paths."""
import json
import os
import subprocess
import sys

import numpy as np
import torch

from feat import *
import diag

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))


def colours(TR, P, v, d, ix):
    NI, K, _ = TR.VARS[v]
    out = []
    with torch.no_grad():
        for a in range(0, len(ix), 262144):
            j = torch.from_numpy(ix[a:a + 262144])
            C, _ = TR.forward(P[v], torch.from_numpy(d["X"][j]), torch.from_numpy(d["R"][j]), K, NI)
            out.append(C)
    return torch.cat(out).numpy().astype(np.float64) * 255


def table(TR, title, d, nets):
    """d: dict of numpy arrays (X R Y G); nets: {'rule': None, 'base': P, 'tuned': P}. Returns markdown lines."""
    lines = [f"### {title}", "", "| group | net | rows | PSNR rule | PSNR base | PSNR tuned | MAE rule | MAE base | MAE tuned | >30 rule % | >30 base % | >30 tuned % |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    gs = sorted(set(int(g) for g in d["G"]))
    for sel in [("all", None)] + [(GROUPS[g], g) for g in gs] if len(gs) > 1 else [(GROUPS[gs[0]], gs[0])]:
        name, g = sel
        ix = np.arange(len(d["G"])) if g is None else np.nonzero(d["G"] == g)[0]
        y = d["Y"][ix].astype(np.float64) * 255
        taps = d["X"][ix, :12].reshape(-1, 4, 3).astype(np.float64) * 255
        rule = np.einsum("nk,nkc->nc", d["R"][ix].astype(np.float64), taps)
        for v in TR.VARS:
            cols = {"rule": rule, "base": colours(TR, nets["base"], v, d, ix), "tuned": colours(TR, nets["tuned"], v, d, ix)}
            a = diag.Acc()
            a.add(cols, y, np.ones(len(ix), bool))
            f = a.fig()
            lines.append(f"| {name} | {v} | {len(ix)} | " + " | ".join(f"{f[k]['psnr']:.3f}" for k in cols) + " | " + " | ".join(f"{f[k]['mae']:.4f}" for k in cols)
                         + " | " + " | ".join(f"{100 * f[k]['big']:.3f}" for k in cols) + " |")
    return lines + [""]


def nz(d):
    return {k: v for k, v in d.items() if v}


def commit():
    try:
        h = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no", "blend", "gpu/ml"], capture_output=True, text=True).stdout.strip())
        return h, dirty
    except Exception:  # noqa: BLE001
        return None, None


def write(out, init_dir, base_dir, va, mva, rows, mrows, cnt_local, cnt_mix, best_step, steps, seconds, options):
    TR = sys.modules["train"]
    base_dir = base_dir or os.path.join(REPO, "blend/weights/base")
    nets = {"base": TR.load_net(base_dir), "tuned": TR.load_net(out)}
    # up to 3 M local validation rows (a fixed subsample) to keep the scoring short
    ix = np.arange(len(va["G"]))
    if len(ix) > 3_000_000:
        ix = np.sort(np.random.default_rng(0).choice(ix, 3_000_000, replace=False))
    lv = {k: va[k][ix] for k in ("X", "R", "Y", "G")}
    md = ["# Fine-tune scores", "",
          "PSNR (dB, higher is better), MAE (levels 0..255) and the share of pixels with any channel error above 30 levels, against the real target frame.",
          "`rule` = the hand-written weights, `base` = the weights this run started from (or `blend/weights/base`), `tuned` = the result of this run. Nets A, B, C.", ""]
    md += table(TR, f"Local validation rows (held-out part of the local footage; {len(lv['G'])} rows)", lv, nets)
    if mva is not None:
        md += table(TR, f"Validation rows of the --mix file ({len(mva['G'])} rows; other footage, per group)", mva, nets)
    open(f"{out}/scores.md", "w").write("\n".join(md))
    man = json.load(open(f"{BLEND}/frames/manifest.json"))["clips"]
    clips = {}
    for name, c in man.items():
        try:
            a = json.load(open(f"{BLEND}/frames/{name}.anchors.json"))
            clips[name] = {"fps": a["fps"], "frames": a["frames"], "frames_from": a.get("frames_from"), "anchors_every_s": a.get("every"),
                           "anchors_kept": len(a["kept"]), "anchors_dropped_repeated": len(a["skipped"])}
        except Exception:  # noqa: BLE001
            clips[name] = {"fps": c.get("fps"), "frames": c.get("frames")}
    h, dirty = commit()
    json.dump({"commit": h, "tracked_files_changed": dirty, "options": options, "steps": steps, "best_step": best_step, "train_seconds": round(seconds, 1),
               "rows_local_train": nz(rows[("train", 0)]), "rows_local_val": nz(rows[("val", 0)]), "rows_mix": {k: nz(v) for k, v in mrows.items()},
               "batch_local": {GROUPS[g]: c for g, c in cnt_local.items()}, "batch_mix": {GROUPS[g]: c for g, c in cnt_mix.items()} if cnt_mix else None,
               "clips": clips}, open(f"{out}/run.json", "w"), indent=1)
    print("wrote scores.md and run.json")
