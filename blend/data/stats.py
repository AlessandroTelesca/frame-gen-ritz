"""Counts of the manifest: samples per class / case / kind / split, skipped anchors, disk use."""
import collections
import json
import subprocess

from common import *

m = json.load(open(BLEND + "/frames/manifest.json"))
c = collections.Counter((s["class"], s["case"], s["target_kind"], s["split"]) for s in m["samples"])
for k, v in sorted(c.items()):
    print(k, v)
sk = json.load(open(BLEND + "/frames/skipped.json"))
print("skipped anchors", sum(len(v) for v in sk.values()), {k: len(v) for k, v in sk.items() if v})
anc = {(s["clip"], s["prev"]) for s in m["samples"] if s["case"] in ("30to60", "60to120") and s["prev"] == s["prev"]}
per = collections.Counter(s["clip"] for s in m["samples"] if s["case"] == "30to60")
print("30to60 samples (= kept anchors) per class:", sum(v for k, v in per.items() if is120(k)), sum(v for k, v in per.items() if not is120(k)))
print(subprocess.run(["du", "-sh", BLEND + "/frames"], capture_output=True, text=True).stdout)
n = sum(len(__import__("os").listdir(f"{BLEND}/frames/{k}")) for k in m["clips"])
print("frames written", n)
