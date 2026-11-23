"""Shared constants for the blend data tools."""
import glob
import json
import os

BLEND = os.environ.get("BLEND", "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend")
FOOT = "/home/mo/Schinken/Programming/GitHubProjects/frame-gen-ritz/footage"
W, H = 1920, 1080
FRAME_BYTES = W * H * 4
CLIPS120 = [f"minecraft120_{i:02d}" for i in range(1, 15)] + ["battlefield"]
CUTS60 = ["racing_forza6_a", "racing_forza6_b", "racing_forza6_c", "fps_cs2_a", "fps_cs2_b", "fps_cs2_c",
          "fps_cs2_d", "indie_schedule1_a", "indie_schedule1_b", "minecraft_rtx_a", "minecraft_rtx_b",
          "minecraft_vanilla_a", "minecraft_vanilla_b", "realworld_portland_walk_a"]


# Foreign footage: when $BLEND/clips.json exists, only the clips named there exist, e.g.
#   {"forza_x1": {"path": "/abs/file.mkv", "fps": 120, "group": "forza", "skip": [[0, 35.5], [1810, 1900]], "split": "val"}}
# fps 120 or 60 (the class), group one of KNOWN_GROUPS (default: "forza"), skip (optional) = time ranges in seconds that give no anchors,
# split (optional) "val" / "train" = the whole clip; otherwise the last 10 % of the clip is validation. Without the file everything below is as always.
KNOWN_GROUPS = ["forza", "cs2", "schedule1", "minecraft_rtx", "minecraft_vanilla", "portland", "minecraft120", "battlefield"]  # = train/feat.py GROUPS


def _foreign():
    p = f"{BLEND}/clips.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    assert isinstance(d, dict) and d, f"{p}: expected an object with one entry per clip"
    for n, e in d.items():
        assert os.path.isabs(e["path"]) and os.path.exists(e["path"]), f"{p}: {n}: path must be an absolute path of an existing file"
        assert e["fps"] in (60, 120), f"{p}: {n}: fps must be 60 or 120"
        assert e.get("group", "forza") in KNOWN_GROUPS, f"{p}: {n}: group must be one of {KNOWN_GROUPS}"
        assert e.get("split") in (None, "val", "train"), f"{p}: {n}: split must be \"val\" or \"train\" (or left out)"
        for r in e.get("skip", []):
            assert len(r) == 2 and 0 <= r[0] < r[1], f"{p}: {n}: skip ranges are [from_sec, to_sec]"
    return d


FOREIGN = _foreign()


def foreign(name):
    """The clips.json entry of NAME, or None (no clips.json: the owner's own clips)."""
    return FOREIGN[name] if FOREIGN is not None else None


def clip_path(name):
    if FOREIGN is not None:
        return FOREIGN[name]["path"]
    if name == "battlefield":
        return FOOT + "/battlefield.mp4"
    return sorted(glob.glob(f"{FOOT}/cuts/{name}*.mp4"))[0]


def all_clips():
    if FOREIGN is not None:
        return list(FOREIGN)
    return CLIPS120 + CUTS60


def is120(name):
    if FOREIGN is not None:
        return FOREIGN[name]["fps"] == 120
    return name.startswith("minecraft120_") or name == "battlefield"


def group(name):
    if FOREIGN is not None:
        return FOREIGN[name].get("group", "forza")
    for g, pre in [("forza", "racing_forza6"), ("cs2", "fps_cs2"), ("schedule1", "indie_schedule1"),
                   ("minecraft_rtx", "minecraft_rtx"), ("minecraft_vanilla", "minecraft_vanilla"),
                   ("portland", "realworld_portland"), ("minecraft120", "minecraft120"), ("battlefield", "battlefield")]:
        if name.startswith(pre):
            return g
    return name
