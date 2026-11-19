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
