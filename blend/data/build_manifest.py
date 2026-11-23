"""Merge frames/<clip>.anchors.json into frames/manifest.json (SPEC section 2) and write frames/DONE."""
import json
import os

from common import *

# (case, prev offset, curr offset, [(t, kind, target offset)]) per class
PLAN120 = [("30to60", 0, 4, [(0.5, "real", 2)]),
           ("30to120", 0, 4, [(0.25, "real", 1), (0.75, "real", 3)]),
           ("60to120", 0, 2, [(0.5, "real", 1)]),
           ("120to240", 0, 1, [(0.5, "teacher", 0)]),
           ("60to240", 0, 2, [(0.25, "teacher", 0), (0.75, "teacher", 1)]),
           ("30to240", 0, 4, [(0.125, "teacher", 0), (0.375, "teacher", 1), (0.625, "teacher", 2), (0.875, "teacher", 3)])]
PLAN60 = [("30to60", 0, 2, [(0.5, "real", 1)]),
          ("60to120", 0, 1, [(0.5, "teacher", 0)]),
          ("30to120", 0, 2, [(0.25, "teacher", 0), (0.75, "teacher", 1)])]

clips, samples = {}, {}
skipped = {}
out = []
for name in all_clips():
    a = json.load(open(f"{BLEND}/frames/{name}.anchors.json"))
    clips[name] = {"path": a["path"], "fps": a["fps"], "frames": a["frames"]}
    skipped[name] = a["skipped"]
    cls, plan = (120, PLAN120) if is120(name) else (60, PLAN60)
    for anc in a["kept"]:
        if foreign(name) and foreign(name).get("split"):   # clips.json: "split" "val" / "train" = the whole clip
            split = foreign(name)["split"]
        else:   # the last 10 % of the clip (the owner's own clips: and racing_forza6_c as a whole)
            split = "val" if ((name == "racing_forza6_c" and not FOREIGN) or anc >= 0.9 * a["frames"]) else "train"
        for case, po, co, ts in plan:
            for ti, (t, kind, to) in enumerate(ts):
                out.append({"id": f"{name}_{anc:06d}_{case}_{ti}", "clip": name, "class": cls, "case": case,
                            "prev": anc + po, "curr": anc + co, "t": t, "target_kind": kind,
                            "target": anc + to, "split": split})
json.dump({"clips": clips, "samples": out}, open(f"{BLEND}/frames/manifest.json.part", "w"))
os.replace(f"{BLEND}/frames/manifest.json.part", f"{BLEND}/frames/manifest.json")
json.dump(skipped, open(f"{BLEND}/frames/skipped.json", "w"))
open(f"{BLEND}/frames/DONE", "w").close()
print(len(out), "samples")
