#!/usr/bin/env python3
"""replay_capture.py CAPTURE_DIR OUT_DIR [--fgtest PATH]: replays a host's frame capture (capture.txt v=1, gamescope-ritz) offline with fgtest
and compares every generated frame with the one the host captured.

What it does (task 109):
- reads capture.txt (one record per line: the kind, then key=value tokens; unknown kinds and keys are ignored), applies every
  `setting` line by field name (fgtest --set; enums as their numbers), the frame size and format of the `capture` line;
- groups consecutive `frame` records with the same (prevId, currId) into one pair: fgtest runs recordEstimate once and recordSynth per
  captured t1 (= recordSynthBlur(t, t) when t0 == t1 or blurSamples < 2), in the captured order. With UI protection on fgtest
  recordObserve's every real frame once, in pair order, consecutive pairs sharing it (fgtest's ring of 3, as a host);
- the images are raw and tight: B8G8R8A8 (vk_format 44 / 50) runs fgtest with --bgra (the host's image format), R8G8B8A8 (37 / 43)
  without; other formats are refused. Converted copies and the replayed outputs go to OUT_DIR (replay_NNN.rgba, RGBA8).
- prints per frame: seq, pair, t, estimate, cached, mean / max abs difference replay vs captured (max over R, G, B), and for both images
  the dark measures: `dark` = share of pixels with luma < 16, `darker` = share of pixels more than 40 levels darker than the darker of
  the two real frames at that pixel (what a black flicker shows; normal motion: up to ~1.5 %, at occlusions). A frame whose captured output is dark (darker > 5 %)
  while the replay is not (or the reverse) is flagged DARK-MISMATCH; a frame within 2 levels everywhere is reproduced offline.
Run it under the GPU lock of the machine, e.g.  flock ~/tmp/fg-handoff/gpu.lock ccorral -- python3 gpu/replay_capture.py DIR OUT
Limits: the frames the host observed between captured pairs are unknown (a gap is bridged as fgtest does); a blurred output
(t0 < t1 with blurSamples >= 2) is replayed at t1 only and flagged BLUR."""
import os, subprocess, sys
import numpy as np


def parse(path):
    recs = []
    for line in open(path, encoding='utf-8', errors='replace'):
        w = line.split()
        if w: recs.append((w[0], dict(t.split('=', 1) for t in w[1:] if '=' in t)))
    return recs


def luma(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def main():
    args = sys.argv[1:]
    exe = 'gpu/build/fgtest'
    if '--fgtest' in args:
        i = args.index('--fgtest'); exe = args[i + 1]; del args[i:i + 2]
    if len(args) != 2: sys.exit(__doc__)
    cap, out = args
    os.makedirs(out, exist_ok=True)
    recs = parse(os.path.join(cap, 'capture.txt'))
    head = next((r for k, r in recs if k == 'capture'), {})
    W, H, vkf = int(head.get('width', 0)), int(head.get('height', 0)), int(head.get('vk_format', 44))
    if head.get('bytes_per_pixel', '4') != '4' or vkf not in (37, 43, 44, 50): sys.exit('format %s (%s): only 8-bit RGBA / BGRA' % (vkf, head.get('format_name')))
    bgra = vkf in (44, 50)
    sets = [r for k, r in recs if k == 'setting']
    setstr = ','.join('%s=%s' % kv for r in sets for kv in r.items() if kv[0] != 'text')
    blurN = next((float(r['blurSamples']) for r in sets if 'blurSamples' in r), 0)
    frames = [r for k, r in recs if k == 'frame']
    print('capture %dx%d %s, %d settings, %d frames, library %s, motion %s' % (W, H, head.get('format_name'), len(sets), len(frames), head.get('library_commit'), head.get('motion_effective')))

    def img(name):   # a captured image as RGBA8 (H, W, 4)
        a = np.fromfile(os.path.join(cap, name), np.uint8)
        if a.size != W * H * 4: sys.exit('%s: %d bytes, not %d' % (name, a.size, W * H * 4))
        a = a.reshape(H, W, 4)
        return a[..., [2, 1, 0, 3]] if bgra else a

    groups = []   # [prev_file, curr_file, [frame records]]
    for f in frames:
        if groups and (groups[-1][0], groups[-1][1]) == (f['prev_file'], f['curr_file']): groups[-1][2].append(f)
        else: groups.append([f['prev_file'], f['curr_file'], [f]])
    written, pos = set(), []
    for g, (pf, cf, fs) in enumerate(groups):
        for n in (pf, cf):
            if n not in written: img(n).tofile(os.path.join(out, n + '.rgba')); written.add(n)
        pos += [os.path.join(out, pf + '.rgba'), os.path.join(out, cf + '.rgba'), os.path.join(out, 'g%03d.rgba' % g)]
    tseq = ';'.join(','.join(f['t1'] for f in fs) for _, _, fs in groups)
    cmd = [exe] + (['--bgra'] if bgra else []) + ['--split', '--set', setstr, '--tseq', tseq, str(W), str(H)] + pos
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode: sys.exit('fgtest failed (%d):\n%s' % (r.returncode, r.stdout[-2000:]))

    print('seq  pair         t       estimate cached | replay vs captured: mean  max  >2 lv | captured dark darker | replay dark darker')
    bad = 0
    for g, (pf, cf, fs) in enumerate(groups):
        lp, lc = luma(img(pf).astype(np.float32)), luma(img(cf).astype(np.float32))
        lo = np.minimum(lp, lc) - 40
        for k, f in enumerate(fs):
            src = os.path.join(out, 'g%03d.rgba' % g) + ('.%d' % k if len(fs) > 1 else '')
            mine = np.fromfile(src, np.uint8).reshape(H, W, 4)
            os.replace(src, os.path.join(out, 'replay_%03d.rgba' % int(f['seq'])))
            got = img(f['out_file'])
            d = np.abs(mine[..., :3].astype(np.int16) - got[..., :3].astype(np.int16)).max(-1)
            lm, lg = luma(mine.astype(np.float32)), luma(got.astype(np.float32))
            dm, dg = (lm < 16).mean(), (lg < 16).mean()
            km, kg = (lm < lo).mean(), (lg < lo).mean()
            flag = ''
            if (kg > 0.05) != (km > 0.05): flag += " DARK-MISMATCH"; bad += 1   # ponytail: 5 % of the pixels darker = "dark" (normal motion reaches ~1.5 %); tune on a real capture
            if d.max() <= 2: flag += ' REPRODUCED'
            if float(f['t0']) < float(f['t1']) and blurN >= 2: flag += ' BLUR'
            print('%3s  %-12s %.4f  %-8s %-6s | %6.2f %4d %.5f | %.4f %.4f | %.4f %.4f%s' % (
                f['seq'], '%s-%s' % (f['prevId'], f['currId']), float(f['t1']), f.get('estimate', '?'), f.get('cached', '?'),
                d.mean(), d.max(), (d > 2).mean(), dg, kg, dm, km, flag))
    print('frames %d, DARK-MISMATCH %d' % (len(frames), bad))


if __name__ == '__main__':
    main()
