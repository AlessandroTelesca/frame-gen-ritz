"""Record check: python3 check_record.py BIN PREV.rgba CURR.rgba OUT.rgba STRIDE
(a) taps are levels 0..255, (b) w0+w1+w2+w3 = 1, (c) share of pixels where sum w_k tap_k is within 2 levels of fgtest's output pixel."""
import sys

import numpy as np

W, H = 1920, 1080
binp, prev, curr, out, stride = sys.argv[1:6]
r = np.fromfile(binp, np.float32).reshape(-1, 28)
idx = np.arange(len(r), dtype=np.int64) * int(stride)
o = np.fromfile(out, np.uint8).reshape(-1, 4)[idx, :3].astype(np.float64)
tap = r[:, 0:12].reshape(-1, 4, 3)
print("(a) tap range", tap.min(), tap.max(), "| prev(x) vs CAp/CBp mean abs diff:",
      np.abs(np.fromfile(prev, np.uint8).reshape(-1, 4)[idx, :3] - tap[:, 0]).mean(), np.abs(np.fromfile(curr, np.uint8).reshape(-1, 4)[idx, :3] - tap[:, 3]).mean())
w = r[:, 22:26]
print("(b) sum w: min %.6f max %.6f; w min %.4f; columns 18 (t) %.3f, 21 sigma mean %.3f, 26 (s) mean %.3f" % (w.sum(1).min(), w.sum(1).max(), w.min(), r[0, 18], r[:, 21].mean(), r[:, 26].mean()))
col = np.einsum("nk,nkc->nc", w, tap)
d = np.abs(col - o).max(axis=1)
print("(c) within 2 levels: %.2f%%  within 5: %.2f%%  mean abs %.3f  n=%d" % (100 * (d <= 2).mean(), 100 * (d <= 5).mean(), np.abs(col - o).mean(), len(r)))
