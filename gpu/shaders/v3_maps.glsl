// MotionVersion::V3 (gpu/V3.md sections 3.9, 3.11, step Y1): the MAP word (cost8 | e8 << 8 | amb8 << 16 | g8 << 24), the occlusion ramp and the
// unsure measure u, shared by v3_occl (the estimate's defaults), v3_fill and v3_project (the runtime knobs v3OcclSoft, v3Unsure). Included after
// v3_common.glsl. Everything here is a SOFT ramp: no pass takes a hard decision from these values.

// the stored raw measurements of one flow px of one direction
struct V3Map { float cost, e, amb, g; };
V3Map v3MapUnpack(uint w) {
    V3Map m;
    m.cost = float(w & 255u) * (1.0 / 8.0);
    m.e = float((w >> 8) & 255u) * (1.0 / 8.0);
    m.amb = float((w >> 16) & 255u) * (1.0 / 255.0);
    m.g = float(w >> 24) * (1.0 / 255.0);
    return m;
}

// occlusion from the forward-backward error e (flow px) of a vector of length F (flow px): tol = 0.6 + 0.02 F; a ramp from 1 tol to (1 + 1.5 soft) tol
float v3Occ(float e, float F, float soft) {
    return smoothstep(1.0, 1.0 + 1.5 * soft, e / (0.6 + 0.02 * F));
}

// section 3.11. o = v3Occ (raw), g = the stored garbage measure (a broad inconsistent region is not an occlusion strip: o_eff = o (1 - g))
float v3OccEff(float o, float g) { return o * (1.0 - g); }
float v3UnsureOf(float cost, float oEff, float amb, float g, float kappa) {
    const float ucost = smoothstep(6.0 * kappa, 18.0 * kappa, cost);
    return max(max(ucost * (1.0 - oEff), 0.8 * amb), max(0.25 * oEff, g));
}

// the camera model (G.aff when valid, else the first valid dominant motion, else zero) at flow px m, as the displacement prev -> curr (the fallback of
// v3_project, and v3_occl's test of which pixels could have found a partner at all)
vec2 camModel(vec2 m) {
    const int g = gOff();
    const uint valid = ax[g + G_VALID];
    if ((valid & 256u) != 0u) {
        const vec2 XY = (m - vec2(pc.cx, pc.cy)) / vec2(pc.cx, pc.cy);
        const vec3 a = vec3(uintBitsToFloat(ax[g + G_AFF]), uintBitsToFloat(ax[g + G_AFF + 1]), uintBitsToFloat(ax[g + G_AFF + 2]));
        const vec3 b = vec3(uintBitsToFloat(ax[g + G_AFF + 3]), uintBitsToFloat(ax[g + G_AFF + 4]), uintBitsToFloat(ax[g + G_AFF + 5]));
        return vec2(dot(a, vec3(1.0, XY)), dot(b, vec3(1.0, XY)));
    }
    for (int k = 0; k < 4; k++)
        if ((valid & (1u << k)) != 0u) return vec2(uintBitsToFloat(ax[g + G_DOM + 2 * k]), uintBitsToFloat(ax[g + G_DOM + 2 * k + 1]));
    return vec2(0.0);
}
