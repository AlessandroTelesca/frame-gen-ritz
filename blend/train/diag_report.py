"""diag_report.py: $BLEND/train/diag/{RUN}.json (the runs that exist) -> diag.md / diag.json. Reads only the small JSON files."""
import json, os, glob
BLEND = "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend"
D = f"{BLEND}/train/diag"
GROUPS = ["forza", "cs2", "schedule1", "minecraft_rtx", "minecraft_vanilla", "portland", "minecraft120", "battlefield"]
ORDER = ["base", "mse", "h32", "h64x2", "h64x2+", "h64x2mse"]
runs = {r: json.load(open(f"{D}/{r}.json")) for r in ORDER if os.path.exists(f"{D}/{r}.json")}
missing = [r for r in ORDER if r not in runs]


def f(r, k):
    return f"{r[k]['psnr']:.2f} / {r[k]['mae']:.2f} / {100 * r[k]['big']:.2f}%"


def row(sel, form, split_key=None):
    out = {}
    for n, r in runs.items():
        s = r["scores"][form].get(sel)
        if s:
            out[n] = s
    return out


L = ["# Why the learned-blend nets stop at the rule: size, loss, inputs", "",
     "Cell: PSNR dB / MAE / share of pixels with any channel error > 30. All scores on the CACHED (subsampled, <= 8192 rows per sample) real-target rows.",
     "Forms: 4 = four choices (NI 25), 6 = six choices (NI 34). Rule and oracle are on the same rows (rule on Forza val 30to60 here: "
     f"{runs['base']['scores']['4']['val/forza/30to60']['rule']['psnr']:.2f} dB, full-row score.py value 29.90).", "",
     "Runs: " + ", ".join(f"{n} (hidden {r['hidden']}, {r['loss']}, plus={r['plus']}, {r['steps']} steps, {r['train_sec']:.0f}s)" for n, r in runs.items()), "",
     f"NOT RUN: {', '.join(missing)} (the session was killed by the out-of-memory daemon while h64x2+ ran; no further runs were started).", ""]
for form in ("4", "6"):
    names = list(runs)
    for sel_title, sels in (("Validation", [f"val/{g}" for g in GROUPS] + ["val/forza/30to60", "val/all"]),
                            ("Training rows (3 M-row subsample)", ["train/forza", "train/all"]),
                            ("Forza validation 30to60 by u = max(uA, uB)", [f"val/forza/30to60/{b}" for b in ("u<0.2", "0.2<=u<=0.6", "u>0.6")])):
        L += [f"## Form {form}: {sel_title}", "", "| rows | px | rule | oracle | " + " | ".join(names) + " |", "|" + "---|" * (4 + len(names))]
        for sel in sels:
            r0 = runs["base"]["scores"][form].get(sel)
            if not r0:
                continue
            L.append(f"| {sel} | {r0['px']} | {f(r0, 'rule')} | {f(r0, 'oracle')} | " + " | ".join(f(runs[n]["scores"][form][sel], "net") for n in names) + " |")
        L.append("")
    # mean over the other groups
    L += [f"Form {form}: mean over the 7 groups other than Forza (all cases pooled per group), PSNR dB: rule {sum(runs['base']['scores'][form][f'val/{g}']['rule']['psnr'] for g in GROUPS[1:]) / 7:.3f}, "
          f"oracle {sum(runs['base']['scores'][form][f'val/{g}']['oracle']['psnr'] for g in GROUPS[1:]) / 7:.3f}, " +
          ", ".join(f"{n} {sum(runs[n]['scores'][form][f'val/{g}']['net']['psnr'] for g in GROUPS[1:]) / 7:.3f}" for n in runs), ""]
L += ["## Validation loss log (group-balanced, on a fixed 60k-per-group validation subsample; the run's own loss)", ""]
for n, r in runs.items():
    L.append(f"- {n}: best step form 4 {r['best']['4'][1]}, form 6 {r['best']['6'][1]}; last logged step {r['log'][-1]['step']}; "
             + "form 4 " + " ".join(f"{e['val']['4']:.5f}" for e in r["log"][::2]) + "; form 6 " + " ".join(f"{e['val']['6']:.5f}" for e in r["log"][::2]))
open(f"{D}/diag.md", "w").write("\n".join(L) + "\n")
json.dump({"runs": {n: {k: v for k, v in r.items() if k != "log"} for n, r in runs.items()}, "not_run": missing}, open(f"{D}/diag.json", "w"), indent=1)
print("\n".join(L))
