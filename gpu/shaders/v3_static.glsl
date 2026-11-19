// MotionVersion::V3 (gpu/V3.md sections 3.13, 3.15 as built in step Y2): the synthesis's view of the static-structure mask. V3 reuses V2's
// v2_static.comp (S, the cell maxima s8) and its own v3_enclose.comp (E, the "overlay near" bits C); the layout is V2B_QUAL's, so these are
// v2_motion.glsl's accessors (sF, eF, sCell, sCellBits, sAt, sOverAt) and v2_morph.glsl's tap rejection (sReject, tapsStatic) over V3's binding 14
// (`ax`) and V3's format macros. Included by v3_synth.comp only (after v3_format.glsl). The tap rejection lets a tap that lands on a static overlay
// take its partner (no dragged copy of a HUD stroke); the pixel's own static weight (v3_synth.comp) puts the overlay back at its place.
const int V3_BLOCK = 8;

float sF(ivec2 m) {
    m = clamp(m, ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    return uintBitsToFloat(ax[sOff() + m.y * pc.fw + m.x]);
}
float eF(ivec2 c) {
    c = clamp(c, ivec2(0), ivec2(s8W() - 1, s8H() - 1));
    return uintBitsToFloat(ax[eOff() + c.y * s8W() + c.x]);
}
// the cell of full-resolution position x and its C bits (bit 0: an overlay is near; bit 1: a warp tap of a pixel of the cell can land near one)
ivec2 sCell(vec2 x) { return clamp(ivec2(floor((x + 0.5) * (1.0 / float(V3_BLOCK * pc.fscale)))), ivec2(0), ivec2(s8W() - 1, s8H() - 1)); }
uint sCellBits(ivec2 c) { return ax[cOff() + c.y * s8W() + c.x]; }
bool sNear(ivec2 c) { return (sCellBits(c) & 1u) != 0u; }
ivec2 sTexel(vec2 x) { return ivec2(floor((x + 0.5) * (1.0 / float(pc.fscale)))); }
// S (the nearest flow px), its enclosure E (bilinear on the cell grid) and S dilated by about 1 flow px at full-resolution position x
vec3 sAt(vec2 x, uint bits) {   // bits: sCellBits of x's cell
    if ((bits & 1u) == 0u) return vec3(0.0);
    vec2 s = (x + 0.5) * (1.0 / float(V3_BLOCK * pc.fscale)) - 0.5;
    vec2 fi = floor(s);
    vec2 f = s - fi;
    ivec2 i = ivec2(fi);
    float E = mix(mix(eF(i), eF(i + ivec2(1, 0)), f.x), mix(eF(i + ivec2(0, 1)), eF(i + ivec2(1, 1)), f.x), f.y);
    ivec2 m = ivec2(floor((x + 0.5) * (1.0 / float(pc.fscale)) - 0.5)), n = sTexel(x);
    float s00 = sF(m), s10 = sF(m + ivec2(1, 0)), s01 = sF(m + ivec2(0, 1)), s11 = sF(m + ivec2(1, 1));
    float sn = n.y == m.y ? (n.x == m.x ? s00 : s10) : (n.x == m.x ? s01 : s11);
    return vec3(sn, E, max(max(s00, s10), max(s01, s11)));
}
// the overlay at a tap: the largest of the 4 S texels around x, and E of its cell (c = sCell(x))
vec2 sOverAt(vec2 x, ivec2 c) {
    if (!sNear(c)) return vec2(0.0);
    ivec2 i = ivec2(floor((x + 0.5) * (1.0 / float(pc.fscale)) - 0.5));
    return vec2(max(max(sF(i), sF(i + ivec2(1, 0))), max(sF(i + ivec2(0, 1)), sF(i + ivec2(1, 1)))), eF(c));
}

// A tap on a static overlay picks the overlay up at the wrong place (a dragged copy of HUD text) while the field moves: the tap takes the other
// one, softly by how much more of the overlay it hits (v2_morph.glsl's sReject). Returns x: how much of the overlay the taps leave (1 - the overlay
// at the tap that hits it least; 0 at the zero vector); y: both taps on the overlay; z: the same for S alone. za2, zc2: sOverAt of the taps.
vec3 sReject(vec2 pa, vec2 pc_, inout vec3 wa, inout vec3 wc, vec2 za2, vec2 zc2) {
    if (all(lessThan(abs(pc_ - pa), vec2(1.0)))) return vec3(0.0);   // (nearly) the zero vector: both taps are the overlay's own place
    float za = max(za2.x, za2.y), zc = max(zc2.x, zc2.y);
    vec3 a = wa;
    wa = mix(wa, wc, clamp(2.0 * (za - zc), 0.0, 1.0));
    wc = mix(wc, a, clamp(2.0 * (zc - za), 0.0, 1.0));
    return vec3(1.0 - min(za, zc), min(za, zc), min(za2.x, zc2.x));
}

// Both taps on S: how surely the structure they hit is static (1: both frames show the same colours at each tap); ra, rc: the taps' raw texels
const float V3_SZMAX = 12.0;
float tapsStatic(vec2 pa, vec2 pc_, vec3 ra, vec3 rc) {
    vec3 d = max(V3_D(V3_LEVELS(ra), V3_LEVELS(V3_TAP(texCurr, texCurrC, pa).rgb)), V3_D(V3_LEVELS(rc), V3_LEVELS(V3_TAP(texPrev, texPrevC, pc_).rgb)));
    return clamp(1.0 - max(d.r, max(d.g, d.b)) * (1.0 / V3_SZMAX), 0.0, 1.0);
}

// the zero-vector decision (v2_morph.glsl's constants)
const float V3_Z0 = 8.0, V3_Z1 = 12.0, V3_ZMAX = 6.0;
const float V3_SEPS = 8.0, V3_SR0 = 0.3, V3_SR1 = 0.6;
const float V3_SD0 = 16.0, V3_SD1 = 24.0;
