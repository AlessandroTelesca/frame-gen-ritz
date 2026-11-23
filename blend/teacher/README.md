# Teacher: Practical-RIFE 4.25 (full)

Offline frame-interpolation teacher for the learned-blend experiment. Nothing of the model is vendored here; only `teacher.py`.

- Code: https://github.com/hzwer/Practical-RIFE, commit `bbfd2ea90910789a860ea3e2b32a240cd577b75e`, cloned to `$BLEND/rife/Practical-RIFE`. Licence: MIT ("MIT License, Copyright (c) 2021 hzwer"). Only `model/warplayer.py` is imported from it.
- Weights package 4.25 (not lite, heavy or 4.26): the official Google Drive link from the repository README (file id `1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg`), fetched with `gdown`; unpacked to `$BLEND/rife/pkg425/train_log/`.
  - zip SHA-256: `e63d481b7ae5d4a4e6ad7ac5b410ff78f3bf7be3b51b2e38ca8152747abde5b4`
  - `flownet.pkl` SHA-256: `6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2`
  - imported from the package: `IFNet_HDv3.py` only (network definition). `RIFE_HDv3.py` and `refine.py` are not imported.
- Loading: `torch.load(flownet.pkl, map_location="cpu", weights_only=True)`, `module.` prefix stripped, `load_state_dict(strict=False)` (the file holds unused training heads `teacher.*`, `caltime.*`).
- Inference: float32, RGB 0..1, bottom/right zero padding to a multiple of 128, `timestep` t, `scale_list = [16, 8, 4, 2, 1]` (scale 1.0), output rounded to 8 bit, alpha 255.

Usage (GPU, in chunks of about 100 s):

    flock $HOME/tmp/fg-handoff/gpu.lock ccorral -- /home/mo/Schinken/Programming/Local/ml-fg/.venv/bin/python teacher.py mids --budget-sec 100
