`base/` is the net weights the library currently uses (`gpu/shaders/v3_blend_weights.glsl` is generated from them with `gpu/ml/blend/export_glsl.py`).
Fine-tuned weights from other people go to `blend/weights/<name>/` (A.npz B.npz C.npz, scores.md, run.json); see `blend/FINETUNE.md`.
