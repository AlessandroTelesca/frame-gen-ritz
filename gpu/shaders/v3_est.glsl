// MotionVersion::V3's estimate passes (gpu/V3.md section 3; build step E1): level geometry computed from the frame size (the rule of
// section 2), texel / record addressing of V3B_PYR and V3B_FLOW, feature decoding and the matching cost of section 2. Included after
// common.glsl and v3_common.glsl by v3_pyr, v3_search and v3_dense only (kept out of v3_common.glsl so that the SPIR-V of the passes that
// do not need it, v3_synth among them, is untouched).

int v3Half(int w) { int k = w >> 1; return (w & 1) != 0 ? k + (k & 1) : k; }   // rint(w / 2), half to even (V1's Level rule)

// Geometry of pyramid level l: size, block grid (8x8 blocks at level 0, 4x4 above), the texel offset of the level's raw plane of frame 0
// (sum of w h of the levels below: texel(l, f, x, y) = 2 off + f w h + y w + x) and the word offset of its block records in V3B_FLOW
// (16 words per block, sum over the levels below).
struct V3Lv { int w, h, gw, gh, off, blk; };
V3Lv v3Lv(int l) {
    V3Lv r;
    r.w = pc.fw; r.h = pc.fh; r.off = 0; r.blk = 0;
    for (int i = 0;; i++) {
        const int bs = i == 0 ? 8 : 4;
        r.gw = (r.w + bs - 1) / bs;
        r.gh = (r.h + bs - 1) / bs;
        if (i == l) break;
        r.off += r.w * r.h;
        r.blk += 16 * r.gw * r.gh;
        r.w = v3Half(r.w);
        r.h = v3Half(r.h);
    }
    return r;
}
// Totals over the pyramid: P (texels of one frame's raw planes, all levels; the feature region starts at texel 2 P) and the word offset of FD
// (16 B: BLK is 16 words per block).
void v3Totals(out int P, out int Bw, out int nLev) {
    int w = pc.fw, h = pc.fh;
    P = 0; Bw = 0; nLev = 0;
    for (;;) {
        const int bs = nLev == 0 ? 8 : 4;
        P += w * h;
        Bw += 16 * ((w + bs - 1) / bs) * ((h + bs - 1) / bs);
        nLev++;
        if (min(w, h) <= 32 || nLev == 16) break;
        w = v3Half(w);
        h = v3Half(h);
    }
}

// feature texel (Yh, Cbh, Crh, Y) of frame f at level L, pixel p (inside the level); fOff = 2 P
vec4 v3FeatAt(const V3Lv L, int fOff, int f, ivec2 p) {
    const uvec2 t = pyr[fOff + 2 * L.off + f * L.w * L.h + p.y * L.w + p.x];
    return vec4(unpackHalf2x16(t.x), unpackHalf2x16(t.y));
}
vec4 v3FeatC(const V3Lv L, int fOff, int f, ivec2 p) { return v3FeatAt(L, fOff, f, clamp(p, ivec2(0), ivec2(L.w - 1, L.h - 1))); }
// bilinear (clamp to edge) at pixel-index coordinates q (pixel k's centre at k)
vec4 v3FeatBil(const V3Lv L, int fOff, int f, vec2 q) {
    q = clamp(q, vec2(0.0), vec2(float(L.w - 1), float(L.h - 1)));
    const ivec2 i0 = ivec2(floor(q));
    const vec2 a = q - vec2(i0);
    const ivec2 i1 = min(i0 + 1, ivec2(L.w - 1, L.h - 1));
    const vec4 t00 = v3FeatAt(L, fOff, f, i0), t10 = v3FeatAt(L, fOff, f, ivec2(i1.x, i0.y));
    const vec4 t01 = v3FeatAt(L, fOff, f, ivec2(i0.x, i1.y)), t11 = v3FeatAt(L, fOff, f, i1);
    return mix(mix(t00, t10, a.x), mix(t01, t11, a.x), a.y);
}

// the cost weights of section 2: Yh / Cbh / Crh high-pass, and the raw Y with the pair's mean-luma difference removed
const vec4 V3_CW = vec4(0.45, 0.20, 0.20, 0.15);
const float V3_LAMBDA = 0.3;   // levels per level px, the prior of every candidate but the first
// |a - b| per channel, weighted; b is the other frame's feature, dm = (K, gm): the pair's lighting change as seen from this direction. The role-1 frame is
// scaled to role 0's contrast and mean: raw term Y0 - (gm (Y1 - mu1) + mu0) = Y0 - gm Y1 + K, K = gm mu1 - mu0, and the high-pass channels Yh0 - gm Yh1 (gm = 1:
// section 2's dMu form, K = mu1 - mu0). Step E2: the gain gm (a fade) was added because the sub-pixel refinement of a faded pair's F10 was biased by the
// amplitude mismatch (level-0 block vectors 48% within 0.25 flow px against 98% for F01).
float v3Term(vec4 a, vec4 b, vec2 dm) { return dot(abs(vec4(a.xyz - dm.y * b.xyz, a.w - dm.y * b.w + dm.x)), V3_CW); }

// dm of this block's direction (dir 0 = F01: role 0 prev, role 1 curr): K as above and the gain gm = sigma0 / sigma1 of the luma (from G.sum, G.sumSq of v3_hist),
// clamped to [0.5, 2] and 1 inside a dead band of +-0.03 (shifts and layers move a little content in and out of the frame: their sigma ratio is not a lighting change)
vec2 v3DMu(int dir) {
    const int g = gOff();
    const float n = float(pc.fw * pc.fh);
    const float m0 = float(ax[g + G_SUM]) / n, m1 = float(ax[g + G_SUM + 1]) / n;
    const float v0 = max(float(ax[g + G_SUMSQ]) * 64.0 / n - m0 * m0, 1e-3), v1 = max(float(ax[g + G_SUMSQ + 1]) * 64.0 / n - m1 * m1, 1e-3);
    float r = sqrt(v0 / v1);   // sigma_prev / sigma_curr: the gain of role 1 = curr (dir 0)
    if (dir == 1) r = 1.0 / r;
    r = clamp(r, 0.5, 2.0);
    const float gm = abs(r - 1.0) < 0.03 ? 1.0 : r - sign(r - 1.0) * 0.03;
    const float mu0 = dir == 0 ? m0 : m1, mu1 = dir == 0 ? m1 : m0;
    return vec2(gm * mu1 - mu0, gm);
}

// V3B_FLOW addressing (float words)
int v3BlkIdx(const V3Lv L, int dir, int slot, ivec2 b) {   // the vec4 record BLK[l][dir][slot](b), clamped to the grid
    b = clamp(b, ivec2(0), ivec2(L.gw - 1, L.gh - 1));
    return L.blk + (dir * 2 + slot) * L.gw * L.gh * 4 + (b.y * L.gw + b.x) * 4;
}

#ifndef V3J_SCALE
#define V3J_SCALE 24.0   // the colour distance (levels, L1 over Y Cb Cr) at which the weight is 1/e
#endif
#ifndef V3J_FLOOR
#define V3J_FLOOR 0.0    // the page's formula has no floor; 0.3 was measured as a milder variant (V3.md A.21)
#endif
#if V3J
// hook (j) (gpu/V3.md section 3.5): the raw (Y, Cb, Cr) texel of frame f at level L, pixel p (inside the level; v3RawC clamps), and the colour weight
// FLOOR + (1 - FLOOR) exp(-|a - c|_1 / SCALE) (page: 24, no floor) of a window pixel of colour a around a centre of colour c
vec3 v3RawAt(const V3Lv L, int f, ivec2 p) {
    const uvec2 t = pyr[2 * L.off + f * L.w * L.h + p.y * L.w + p.x];
    return vec3(unpackHalf2x16(t.x), unpackHalf2x16(t.y).x);
}
vec3 v3RawC(const V3Lv L, int f, ivec2 p) { return v3RawAt(L, f, clamp(p, ivec2(0), ivec2(L.w - 1, L.h - 1))); }
float v3ColourWgt(vec3 a, vec3 c) {
    const vec3 d = abs(a - c);
    return V3J_FLOOR + (1.0 - V3J_FLOOR) * exp(-(d.x + d.y + d.z) * (1.0 / V3J_SCALE));
}
#endif

// Task 105 (gpu/V3.md "As built (HUD bonus, task 105)"): Settings::hudBonus for V3. pc.hudB = hudBonus x K (V3's cost is in other units than V1's) is a reduction of the
// zero vector's cost where there is evidence that the content did not move: the zero vector matches (c0 small: the two frames show the same there, gain-compensated) and
// the window has detail (a flat window matches at every vector: it proves nothing, so a flat sky never freezes). Both are soft ramps. hudB = 0 skips every use.
// detail of a feature texel: the cost's own channel weights over the high-pass channels
float v3Detail(vec4 a) { return dot(abs(a.xyz), V3_CW.xyz); }
float v3HudEv(float det) { return smoothstep(pc.hp[2], pc.hp[2] + 1.1, det); } // the window has detail
// the motion the bonus would undo is worth undoing: the (best) moving vector, in frame pixels (vpx), is not a slow drift (slow real motion must not freeze)
float v3HudGate(float vpx) { return smoothstep(pc.hp[3], pc.hp[4], vpx); }
const float V3H_D0 = -0.5;   // the margin (levels) at which the zero vector starts to take over in the dense pass; it takes over fully 1.5 above
float v3HudAbs(float c0) { return 1.0 - smoothstep(pc.hp[0], pc.hp[1], c0); }  // the zero vector's window cost is small
