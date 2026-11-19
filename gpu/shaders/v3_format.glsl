// V3's copy of the V2_* format macros of v2_motion.glsl (kept apart so that V2's SPIR-V stays untouched; if V2's change, change these):
// the synthesis blends "working values" in the source's own space (V3_LEVELS of a raw sample: Rgba8 / Rgb10 8-bit levels, RgbaF16 the
// value itself, YUV (Y, Cb, Cr) levels) and every threshold is in 8-bit levels, compared on V3_LV of working values. V3_D: the per-channel
// distance of two working values, V3_G their difference, V3_TAP a bilinear tap (position rounded to 1/32 px), V3_PX a texel (YUV: Y at p,
// CbCr of p's 2x2 block), V3_UIP the position whose CrosshairV2 footprint covers both, V3_QUANT a sub-frame exactly as the store gives
// it. Built once per format (v3_planes.comp, v3_synth.comp; gpu/meson.build v3_colour_srcs).
vec4 tapRaw(sampler2D t, vec2 p) {
    p = roundEven(p * 32.0) * (1.0 / 32.0);
    return textureLod(t, (p + 0.5) / vec2(pc.W, pc.H), 0.0);
}
#if defined(FG_YUV)
vec4 tapYuv(sampler2D ty, sampler2D tc, vec2 p) {
    p = roundEven(p * 32.0) * (1.0 / 32.0);
    vec2 cs = 2.0 * vec2((pc.W + 1) >> 1, (pc.H + 1) >> 1);
    return vec4(textureLod(ty, (p + 0.5) / vec2(pc.W, pc.H), 0.0).r, textureLod(tc, (p + 1.0) / cs, 0.0).rg, 1.0);
}
vec3 v3QL(vec3 l) { return clamp(floor(l / YU_QL + 0.5), 0.0, YU_CODES) * YU_QL; }
#define V3_TAP(t, tc, p) tapYuv(t, tc, p)
#define V3_PX(t, tc, p) vec4(texelFetch(t, p, 0).r, texelFetch(tc, (p) >> 1, 0).rg, 1.0)
#define V3_LEVELS(x) ((x) * YU_SCALE)
#define V3_LV(x) fgYuvDiff(x)
#define V3_D(a, b) abs(fgYuvDiff((a) - (b)))
#define V3_G(a, b) fgYuvDiff((a) - (b))
#define V3_UIP(p) ((p) + 0.5)
#define V3_QUANT(x) v3QL(x)
#else
#define V3_TAP(t, tc, p) tapRaw(t, p)
#define V3_PX(t, tc, p) texelFetch(t, p, 0)
#define V3_LEVELS(x) FG_FETCH(x)
#define V3_UIP(p) (p)
#if defined(FG_FORMAT_F16)
#define V3_LV(x) fgMetric(x)
#define V3_D(a, b) abs(fgMetric(a) - fgMetric(b))
#define V3_G(a, b) (fgMetric(a) - fgMetric(b))
vec3 v3Half(vec3 x) {   // the nearest half float, round to nearest even (the store's own conversion may truncate on AMD)
    x = clamp(x, -65504.0, 65504.0);
    uvec3 b = floatBitsToUint(x);
    vec3 n = uintBitsToFloat((b + 0xFFFu + ((b >> 13) & 1u)) & ~0x1FFFu);
    return mix(n, roundEven(x * 16777216.0) * (1.0 / 16777216.0), lessThan(abs(x), vec3(6.103515625e-05)));
}
#define V3_QUANT(x) v3Half(x)
#elif defined(FG_FORMAT_RGB10)
#define V3_LV(x) (x)
#define V3_D(a, b) abs((a) - (b))
#define V3_G(a, b) ((a) - (b))
#define V3_QUANT(x) (floor(clamp(x, 0.0, 255.0) * (1023.0 / 255.0) + 0.5) * (255.0 / 1023.0))
#else
#define V3_LV(x) (x)
#define V3_D(a, b) abs(a - b)
#define V3_G(a, b) a - b
#define V3_QUANT(x) clamp(floor(x + 0.5), 0.0, 255.0)
#endif
#endif
