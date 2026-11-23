"""Shared constants for the blend data tools."""
import glob
import os

BLEND = os.environ.get("BLEND", "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend")
FOOT = "/home/mo/Schinken/Programming/GitHubProjects/frame-gen-ritz/footage"
W, H = 1920, 1080
FRAME_BYTES = W * H * 4
CLIPS120 = [f"minecraft120_{i:02d}" for i in range(1, 15)] + ["battlefield"]
CUTS60 = ["racing_forza6_a", "racing_forza6_b", "racing_forza6_c", "fps_cs2_a", "fps_cs2_b", "fps_cs2_c",
          "fps_cs2_d", "indie_schedule1_a", "indie_schedule1_b", "minecraft_rtx_a", "minecraft_rtx_b",
          "minecraft_vanilla_a", "minecraft_vanilla_b", "realworld_portland_walk_a"]


def clip_path(name):
    if name == "battlefield":
        return FOOT + "/battlefield.mp4"
    return sorted(glob.glob(f"{FOOT}/cuts/{name}*.mp4"))[0]


def all_clips():
    return CLIPS120 + CUTS60


def is120(name):
    return name.startswith("minecraft120_") or name == "battlefield"


def group(name):
    for g, pre in [("forza", "racing_forza6"), ("cs2", "fps_cs2"), ("schedule1", "indie_schedule1"),
                   ("minecraft_rtx", "minecraft_rtx"), ("minecraft_vanilla", "minecraft_vanilla"),
                   ("portland", "realworld_portland"), ("minecraft120", "minecraft120"), ("battlefield", "battlefield")]:
        if name.startswith(pre):
            return g
    return name
