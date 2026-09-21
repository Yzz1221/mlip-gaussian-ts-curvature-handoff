# EquiformerV2 checkpoint

`eqv2.ckpt` is the Hessian-supervised EquiformerV2 checkpoint used for both
MLIP--OneShot--ReadFC and External--CalcAll. Its exact bytes are included in
this directory as 14 numbered parts. GitHub blocks a single ordinary Git
file over 100 MiB, so the 224.5 MB checkpoint is reconstructed after cloning.

From the repository root, run `python examples/setup.py` to reconstruct and
verify `ckpt/eqv2.ckpt` automatically. This uses the bundled parts and needs
no extra download. To reconstruct manually from this directory, run:

```bash
cat eqv2.ckpt.[0-9][0-9].part > eqv2.ckpt
sha256sum eqv2.ckpt
```

## Provenance and integrity

- upstream project: HORM
- upstream repository: <https://github.com/deepprinciple/HORM>
- pinned HORM commit: `b4c2a35a28985c72ca47261bad0a96b2bc2ba084`
- upstream checkpoint name: `eqv2.ckpt`
- file size: 224,512,698 bytes
- SHA256:
  `6b5adb66776041a45ab85e5e496c3b1e37f2b102be31247384a8ee69ee56016a`
- repository parts: 14, ordered `00` through `13`

The checkpoint is distributed under the upstream HORM terms. Cite the HORM
dataset/model paper using the entry in `../CITATIONS.bib`.
The exact local EquiformerV2 network source used during inference is included
at `../horm/nets/equiformer_v2/equiformer_v2_oc20.py`.
`examples/setup.py` verifies its checksum.
