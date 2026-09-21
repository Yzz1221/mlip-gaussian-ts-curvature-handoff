# EquiformerV2 checkpoint

`eqv2.ckpt` is the Hessian-supervised EquiformerV2 checkpoint used for both
MLIP--OneShot--ReadFC and External--CalcAll calculations in the accompanying
study. The exact file is distributed through the GitHub release
[`eqv2-model-v1`](https://github.com/Yzz1221/mlip-gaussian-ts-curvature-handoff/releases/tag/eqv2-model-v1).
It is split into numbered parts to make uploads reliable through restricted
network proxies.

From the repository root, run `python examples/setup.py` to download,
reconstruct, and verify the checkpoint automatically. To do the same steps
manually, download every `eqv2.ckpt.NN.part` asset and
`eqv2.ckpt.sha256` into one directory, then run:

```bash
cat eqv2.ckpt.[0-9][0-9].part > eqv2.ckpt
sha256sum -c eqv2.ckpt.sha256
```

## Provenance and integrity

- upstream project: HORM
- upstream repository: <https://github.com/deepprinciple/HORM>
- pinned HORM commit: `b4c2a35a28985c72ca47261bad0a96b2bc2ba084`
- upstream checkpoint name: `eqv2.ckpt`
- file size: 224,512,698 bytes
- SHA256:
  `6b5adb66776041a45ab85e5e496c3b1e37f2b102be31247384a8ee69ee56016a`
- release parts: 14, ordered `00` through `13`

The checkpoint is distributed under the upstream HORM terms. Cite the HORM
dataset/model paper using the entry in `../CITATIONS.bib`.
The exact local EquiformerV2 network source used to construct this checkpoint
during inference is in `../model_architecture/`. `examples/setup.py` also
installs that file into the pinned HORM submodule.
