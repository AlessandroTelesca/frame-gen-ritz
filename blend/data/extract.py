"""Extract the needed frames of one clip in a single ffmpeg pass, drop anchors with repeated frames,
write frames/<clip>/<idx>.rgba and frames/<clip>.anchors.json.   usage: extract.py CLIP [--every SECONDS]
clips.json (see common.py) adds recordings from elsewhere; its "skip" ranges give no anchors."""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

from common import *


def anchors_for(name, nframes, every=None, skip=()):
    """Anchor frame indices. every = step in seconds (default: today's steps: 120 frames for the 120 fps class, 20 for Forza, 60 otherwise);
    skip = [[from_sec, to_sec], ..]: an anchor whose frames a .. a+last touch a range gives none (seconds count at the class rate)."""
    rate = 120 if is120(name) else 60
    if is120(name):
        step, last = 120, 8
    else:
        step, last = (20 if name.startswith("racing_forza6") else 60), 2
    if every is not None:
        step = max(1, round(every * rate))
        assert step >= (5 if is120(name) else 3), f"--every {every} gives a step of {step} frames, below the frames one anchor needs"
    rng = [(s0 * rate, s1 * rate) for s0, s1 in skip]
    return [a for a in range(step, nframes, step) if a + last < nframes and not any(a <= r1 and a + last >= r0 for r0, r1 in rng)]


def frame_count(path, st):
    """Number of video frames: nb_frames of the stream when the container has it, else the packet count (ffprobe -count_packets, demuxing only,
    equals the frame count for ordinary video), else duration x rate. Returns (n, how)."""
    if str(st.get("nb_frames", "N/A")).isdigit():
        return int(st["nb_frames"]), "nb_frames"
    r = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets",
                        "-of", "json", path], capture_output=True, text=True)
    try:
        n = int(json.loads(r.stdout)["streams"][0]["nb_read_packets"])
        if n > 0:
            return n, "packet count"
    except (ValueError, KeyError, IndexError):
        pass
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path], capture_output=True, text=True)
    dur = float(json.loads(r.stdout)["format"]["duration"])
    return int(dur * rate_of(st["r_frame_rate"])), "duration x rate"


def rate_of(s):
    num, den = map(int, s.split("/"))
    return num / den if den else 0.0


def build_select(anchors, need):
    """select expression for the frames a .. a+need-1 of every anchor. ffmpeg's expression parser refuses a long a+b+c+... chain
    (about 100 terms: 'Cannot allocate memory'), so the terms are summed in a tree of groups of 32 (1,500 anchors: 40 kB, one argument)."""
    t = [f"between(n\\,{a}\\,{a + need - 1})" for a in anchors]
    while len(t) > 1:
        t = ["(" + "+".join(t[i:i + 32]) + ")" for i in range(0, len(t), 32)]
    return t[0]


def main(name, every=None):
    path = clip_path(name)
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_frames,r_frame_rate,avg_frame_rate,width,height", "-of", "json", path],
                       capture_output=True, text=True, check=True)
    st = json.loads(r.stdout)["streams"][0]
    assert (st["width"], st["height"]) == (W, H), f"{name}: the recording must be {W}x{H}, it is {st['width']}x{st['height']}"
    nfr, how = frame_count(path, st)
    fps = round(rate_of(st["r_frame_rate"]))
    if is120(name):
        for k in ("r_frame_rate", "avg_frame_rate"):
            v = rate_of(st[k])
            assert v == 0 or abs(v - 120) < 0.5, f"{name}: a 120 fps clip is expected, ffprobe says {k} = {st[k]} (variable frame rate, or the wrong fps in clips.json)"
    print(name, "frames:", nfr, f"(from {how})", "fps", fps, flush=True)
    need = 5 if is120(name) else 3
    fe = foreign(name)
    anchors = anchors_for(name, nfr, every, fe.get("skip", []) if fe else [])
    assert anchors, f"{name}: no anchors"
    expr = build_select(anchors, need)
    assert len(expr) < 100000, f"{name}: select expression of {len(expr)} characters is too long for one command line; use a larger --every or split the recording"
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
    json.dump({"clip": name, "path": path, "fps": fps, "frames": nfr, "frames_from": how, "every": every, "kept": kept, "skipped": skipped},
              open(f"{BLEND}/frames/{name}.anchors.json", "w"))
    print(name, nfr, fps, "kept", len(kept), "skipped", len(skipped), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--every", type=float, default=None, help="anchor step in seconds (default: 1 s for 120 fps clips, 1 s for 60 fps clips except Forza 1/3 s)")
    a = ap.parse_args()
    main(a.clip, a.every)
