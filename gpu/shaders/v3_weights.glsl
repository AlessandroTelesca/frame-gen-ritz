// MotionVersion::V3's blend-weight stage (gpu/V3.md section 3.16): the ONE replaceable place that decides how the four taps are mixed. A small net may later
// replace this include (or a pass fed from the same record) and nothing else. One function, no side effects, no buffer access; everything it
// needs comes in through V3WIn. Included by v3_synth.comp after v3_format.glsl (V3_D: the per-channel distance of two working values, in levels).
//
// Taps: hypothesis A (the prev side, F01): CAp = prev(x - t gA), CAc = curr(x + (1 - t) gA); hypothesis B (the curr side, F10): CBc = curr(x - (1 - t) gB),
// CBp = prev(x + t gB). oA: the partner of A's prev pixel is hidden in curr; oB: B's curr pixel is hidden in prev. holeX: nothing lands at this output
// pixel via that hypothesis. uX: unsure. Every output is continuous in every input: no branch, no hard choice.
struct V3WIn {
    vec3 CAp, CAc, CBp, CBc;   // working values (levels; F16: the value itself; YUV: Y, Cb, Cr levels)
    float oA, oB;              // occlusion [0,1]
    float holeA, holeB;        // [0,1]
    float uA, uB;              // unsure [0,1]
    float t;                   // sample time (0,1)
    float lenA, lenB;          // hypothesis vector length, px
};
struct V3WOut {
    float w[4];                // weights of CAp, CAc, CBp, CBc: >= 0, sum 1
    float s;                   // smear amount [0,1] (Y1: not used by the synth yet)
};

void v3Weights(in V3WIn i, out V3WOut o) {
    // one-sided where the partner is hidden: A owns prev (hidden in curr -> prev only), B owns curr (hidden in prev -> curr only)
    const float wAc = i.t * (1.0 - i.oA), wAp = 1.0 - wAc;
    const float wBp = (1.0 - i.t) * (1.0 - i.oB), wBc = 1.0 - wBp;
    // pair agreement: how well each hypothesis' own two samples match (ignored where the partner is hidden: they cannot match then)
    const vec3 dA3 = V3_D(i.CAp, i.CAc), dB3 = V3_D(i.CBp, i.CBc);
    const float dA = max(dA3.r, max(dA3.g, dA3.b)), dB = max(dB3.r, max(dB3.g, dB3.b));
    const float sA = mix(1.0 - smoothstep(12.0, 40.0, dA), 1.0, i.oA), sB = mix(1.0 - smoothstep(12.0, 40.0, dB), 1.0, i.oB);
    const float HA = (1.0 - i.holeA) * (1.0 - 0.5 * i.uA) * (0.25 + 0.75 * sA);
    const float HB = (1.0 - i.holeB) * (1.0 - 0.5 * i.uB) * (0.25 + 0.75 * sB);
    // the hypothesis whose own samples agree wins softly, equal -> average; both holes (HA + HB -> 0): the ratio tends to 1/2 (a time blend of the
    // taps) and the smear takes over (s >= hole). The page's `HA + HB < 1e-3` special case is this continuous limit.
    const float wA = (HA + 1e-4) / (HA + HB + 2e-4), wB = 1.0 - wA;
    o.w[0] = wA * wAp;
    o.w[1] = wA * wAc;
    o.w[2] = wB * wBp;
    o.w[3] = wB * wBc;
    const float hole = 1.0 - smoothstep(0.0, 0.02, HA + HB);
    o.s = max(wA * i.uA + wB * i.uB, hole);
}
