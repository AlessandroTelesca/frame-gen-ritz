#!/usr/bin/env python3
"""Teacher for the learned-blend experiment: Practical-RIFE 4.25 (full), float32, 1080p.

  teacher.py mids  [--list FILE] [--max N] [--budget-sec S]   write missing mid(i) files
  teacher.py check [--budget-sec S]                           sanity measurement vs real frames (resumable)
  teacher.py bench                                            time + VRAM at 1080p

Run on the GPU only as:  flock $HOME/tmp/fg-handoff/gpu.lock ccorral -- <python> teacher.py ...
List file: lines "clip index" (mid(index) of clip). Without --list: every teacher target of manifest.json.
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

BLEND = os.environ.get("BLEND", "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend")
RIFE = os.path.join(BLEND, "rife")
REPO = os.path.join(RIFE, "Practical-RIFE")           # only model/warplayer.py is used from it
PKG = os.path.join(RIFE, "pkg425", "train_log")        # IFNet_HDv3.py + flownet.pkl (4.25)
W, H = 1920, 1080
PAD = 128                                              # scale 16 plus two stride-2 convs need a multiple of 64; upstream pads to 128 at scale 1
FR = os.path.join(BLEND, "frames")
OUT = os.path.join(BLEND, "teacher")
dev = torch.device("cuda")


def load_net():
    sys.path.insert(0, REPO)
    spec = importlib.util.spec_from_file_location("IFNet_HDv3", os.path.join(PKG, "IFNet_HDv3.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    net = mod.IFNet()
    sd = torch.load(os.path.join(PKG, "flownet.pkl"), map_location="cpu", weights_only=True)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    r = net.load_state_dict(sd, strict=False)           # upstream also loads with strict=False
    assert not r.missing_keys, r.missing_keys
    return net.to(dev).eval(), r.unexpected_keys


def read(clip, i):
    a = np.fromfile(f"{FR}/{clip}/{i:06d}.rgba", dtype=np.uint8)
    assert a.size == W * H * 4, (clip, i, a.size)
    return a.reshape(H, W, 4)


def to_t(a):
    t = torch.from_numpy(a[..., :3].copy()).to(dev).permute(2, 0, 1).float().div_(255).unsqueeze(0)
    return F.pad(t, (0, (-W) % PAD, 0, (-H) % PAD))


@torch.no_grad()
def interp(net, a0, a1, t=0.5):
    """a0, a1: uint8 [H,W,4] -> uint8 [H,W,4] frame at time t (RGB 0..1, NCHW, padded, cropped back)."""
    x = torch.cat((to_t(a0), to_t(a1)), 1)
    _, _, merged = net(x, t, [16, 8, 4, 2, 1])
    y = merged[-1][0, :, :H, :W].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    out = np.empty((H, W, 4), np.uint8); out[..., :3] = y; out[..., 3] = 255
    return out


def manifest():
    return json.load(open(f"{FR}/manifest.json"))


def wanted(list_file):
    if list_file:
        return [(c, int(i)) for c, i in (l.split() for l in open(list_file) if l.strip())]
    s = {(x["clip"], x["target"]) for x in manifest()["samples"] if x["target_kind"] == "teacher"}
    return sorted(s)


def have(c, i):
    p = f"{OUT}/{c}/mid_{i:06d}.rgba"
    return os.path.exists(p) and os.path.getsize(p) == W * H * 4


def cmd_mids(a):
    net, _ = load_net()
    todo = [(c, i) for c, i in wanted(a.list) if not have(c, i)]
    print(f"missing {len(todo)}", flush=True)
    t0 = time.time(); n = 0; failed = []
    for c, i in todo:
        if n >= a.max or time.time() - t0 > a.budget_sec:
            break
        try:
            y = interp(net, read(c, i), read(c, i + 1))
            os.makedirs(f"{OUT}/{c}", exist_ok=True)
            p = f"{OUT}/{c}/mid_{i:06d}.rgba"
            y.tofile(p + ".part"); os.replace(p + ".part", p)
            n += 1
        except Exception as e:
            failed.append((c, i)); print("FAILED", c, i, repr(e), flush=True)
    dt = time.time() - t0
    print(f"wrote {n} in {dt:.1f}s ({n / max(dt, 1e-9):.2f} fps), remaining {len(todo) - n - len(failed)}, failed {len(failed)}, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


def cmd_bench(a):
    net, un = load_net()
    print("unexpected keys:", un)
    x = np.random.randint(0, 256, (H, W, 4), dtype=np.uint8); y = np.roll(x, 7, axis=1)
    interp(net, x, y); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(5):
        interp(net, x, y)
    torch.cuda.synchronize()
    print(f"{5 / (time.time() - t0):.2f} fps, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


def stats(p, r):
    d = p[..., :3].astype(np.float32) - r[..., :3].astype(np.float32)
    return float((d ** 2).mean()), float(np.abs(d).mean()), float((np.abs(d).max(-1) > 30).mean())


def cmd_check(a):
    """Anchors: every (clip, a) of the 120 fps clips used by the plan (prev of a 30->120 sample)."""
    anchors = sorted({(x["clip"], x["prev"]) for x in manifest()["samples"]
                      if x["curr"] - x["prev"] == 4 and x["case"].startswith("30") and str(x["class"]).startswith("120")})
    sp = f"{OUT}/check_state.json"
    st = json.load(open(sp)) if os.path.exists(sp) else {}
    net, _ = load_net(); t0 = time.time()
    for c, a0 in anchors:
        key = f"{c}:{a0}"
        if key in st:
            continue
        if time.time() - t0 > a.budget_sec:
            break
        f = [read(c, a0 + k) for k in range(5)]
        r = {}
        for name, (i, j, k) in {"120to60": (0, 2, 1), "30to60": (0, 4, 2)}.items():
            tch = interp(net, f[i], f[j])
            avg = ((f[i][..., :3].astype(np.uint16) + f[j][..., :3] + 1) // 2).astype(np.uint8)
            r[name] = {"teacher": stats(tch, f[k]), "avg": stats(avg, f[k])}
        st[key] = r
    os.makedirs(OUT, exist_ok=True)
    json.dump(st, open(sp, "w"))
    print(f"check state: {len(st)}/{len(anchors)} anchors")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mids", "check", "bench"])
    ap.add_argument("--list"); ap.add_argument("--max", type=int, default=10**9); ap.add_argument("--budget-sec", type=float, default=100)
    a = ap.parse_args()
    {"mids": cmd_mids, "check": cmd_check, "bench": cmd_bench}[a.cmd](a)
