"""Compare N random extracted frames with the single-frame ffmpeg command of the SPEC (-vsync 0 spelled -fps_mode passthrough).
usage: spotcheck.py [N]"""
import filecmp
import json
import os
import random
import subprocess
import sys

from common import *

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
man = json.load(open(f"{BLEND}/frames/manifest.json"))
files = [(c, f) for c in man["clips"] for f in os.listdir(f"{BLEND}/frames/{c}") if f.endswith(".rgba")]
os.makedirs(f"{BLEND}/frames/.spot", exist_ok=True)
for c, f in random.sample(files, n):
    i = int(f[:6])
    out = f"{BLEND}/frames/.spot/{c}_{f}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", man["clips"][c]["path"], "-vf", f"select='eq(n\\,{i})'", "-fps_mode", "passthrough",
                    "-frames:v", "1", "-pix_fmt", "rgba", "-f", "rawvideo", out], check=True)
    print(c, f, "identical" if filecmp.cmp(out, f"{BLEND}/frames/{c}/{f}", shallow=False) else "DIFFERENT", flush=True)
    os.remove(out)
os.rmdir(f"{BLEND}/frames/.spot")
