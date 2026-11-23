"""Extract the needed frames of one clip in a single ffmpeg pass, drop anchors with repeated frames,
write frames/<clip>/<idx>.rgba and frames/<clip>.anchors.json.   usage: extract.py CLIP"""
import json
import os
import subprocess
import sys

import numpy as np

from common import *


def anchors_for(name, nframes):
    if is120(name):
        step, last = 120, 8
    else:
        step, last = (20 if name.startswith("racing_forza6") else 60), 2
    return [a for a in range(step, nframes, step) if a + last < nframes]


def main(name):
    path = clip_path(name)
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_frames,r_frame_rate,width,height", "-of", "json", path],
                       capture_output=True, text=True, check=True)
    st = json.loads(r.stdout)["streams"][0]
    assert (st["width"], st["height"]) == (W, H), st
    nfr = int(st["nb_frames"])
    num, den = map(int, st["r_frame_rate"].split("/"))
    fps = round(num / den)
    need = 5 if is120(name) else 3
    anchors = anchors_for(name, nfr)
    expr = "+".join(f"between(n\\,{a}\\,{a + need - 1})" for a in anchors)
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"select='{expr}'", "-fps_mode", "passthrough",
           "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=FRAME_BYTES * 2)
    d = f"{BLEND}/frames/{name}"
    os.makedirs(d, exist_ok=True)
    kept, skipped = [], []
    for a in anchors:
        fr = []
        for k in range(need):
            b = p.stdout.read(FRAME_BYTES)
            while len(b) < FRAME_BYTES:
                c = p.stdout.read(FRAME_BYTES - len(b))
                assert c, (name, a, k)
                b += c
            fr.append(np.frombuffer(b, np.uint8))
        rep = [k for k in range(need - 1) if np.abs(fr[k].astype(np.int16) - fr[k + 1]).mean() < 0.05]
        if rep:
            skipped.append({"anchor": a, "repeated": [a + k for k in rep]})
            continue
        for k in range(need):
            with open(f"{d}/{a + k:06d}.rgba.part", "wb") as f:
                f.write(fr[k].tobytes())
            os.replace(f"{d}/{a + k:06d}.rgba.part", f"{d}/{a + k:06d}.rgba")
        kept.append(a)
    assert p.stdout.read(1) == b""
    p.wait()
    json.dump({"clip": name, "path": path, "fps": fps, "frames": nfr, "kept": kept, "skipped": skipped},
              open(f"{BLEND}/frames/{name}.anchors.json", "w"))
    print(name, nfr, fps, "kept", len(kept), "skipped", len(skipped), flush=True)


main(sys.argv[1])
