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
// |a - b| per channel, weighted; b is the other frame's feature, dMu = mean(role 1) - mean(role 0) (section 2: the raw term is Y0 - (Y1 - dMu))
float v3Term(vec4 a, vec4 b, float dMu) { return dot(abs(vec4(a.xyz - b.xyz, a.w - b.w + dMu)), V3_CW); }

// dMu of this block's direction: dir 0 (F01: roles prev, curr) mean(curr) - mean(prev); dir 1 the negative
float v3DMu(int dir) {
    const int g = gOff() + G_SUM;
    const float s0 = float(ax[g]), s1 = float(ax[g + 1]);   // sums of luma, per frame (v3_hist.comp)
    const float d = (s1 - s0) / float(pc.fw * pc.fh);
    return dir == 0 ? d : -d;
}

// V3B_FLOW addressing (float words)
int v3BlkIdx(const V3Lv L, int dir, int slot, ivec2 b) {   // the vec4 record BLK[l][dir][slot](b), clamped to the grid
    b = clamp(b, ivec2(0), ivec2(L.gw - 1, L.gh - 1));
    return L.blk + (dir * 2 + slot) * L.gw * L.gh * 4 + (b.y * L.gw + b.x) * 4;
}
